# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input, adj):
        '''
        GCN - ChebNet's first-order approximation
        :param input: (B, N, in_F)
        :param adj: (N， N)
        :return: (B, N, out_F)
        '''

        support = torch.matmul(input, self.weight)

        output = torch.matmul(adj, support)

        return output + self.bias if self.bias is not None else output


class GCN_layer(nn.Module):
    def __init__(self, gcn1_in_feature, gcn1_out_feature, gcn2_out_feature):
        super(GCN_layer, self).__init__()
        self.gc1 = GraphConvolution(gcn1_in_feature, gcn1_out_feature)
        self.gc2 = GraphConvolution(gcn1_out_feature, gcn2_out_feature)

    def forward(self, x, adj):
        '''
        GCN for each timestep.
        :param x: (B, N, in_F, T)
        :param adj: (N， N)
        :return: (B, N, gcn_F, T)
        '''

        batch_size, node, in_channels, timesteps = x.shape

        gcn_outputs = []

        for time_step in range(timesteps):
            gcn_1 = self.gc1(x[:, :, :, time_step], adj)  # (B, N, in_F) - (B, N, gcn1_F)

            gcn_2 = self.gc2(gcn_1, adj)  # (B, N, gcn1_F) - (B, N, gcn_F)

            gcn_outputs.append(gcn_2.unsqueeze(-1))  # (B, N, gcn_F) - (B, N, gcn_F, 1)

        return F.relu(torch.cat(gcn_outputs, dim=-1))  # (B, N, gcn_F, T)


class Temporal_Attention(nn.Module):
    def __init__(self, DEVICE, in_channels, nodes, timesteps):
        super(Temporal_Attention, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(nodes).to(DEVICE))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, nodes).to(DEVICE))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.be = nn.Parameter(torch.FloatTensor(1, timesteps, timesteps).to(DEVICE))
        self.Ve = nn.Parameter(torch.FloatTensor(timesteps, timesteps).to(DEVICE))

    def forward(self, x):
        '''
        ASTGCN - temporal attention
        :param x: (B, N, in_F, T)
        :return: (B, T, T)
        '''

        # x:(B, N, in_F, T) -> (B, T, in_F, N)
        # (B, T, in_F, N)(N) -> (B, T, in_F)
        # (B, T, in_F)(in_F,N) -> (B, T, N)
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)

        rhs = torch.matmul(self.U3, x)  # (F)(B, N, F, T) -> (B, N, T)

        product = torch.matmul(lhs, rhs)  # (B, T, N)(B, N, T) -> (B, T, T)

        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)

        E_normalized = F.softmax(E, dim=1)

        return E_normalized


class Spatial_Attention(nn.Module):
    '''
    compute spatial attention scores
    '''

    def __init__(self, DEVICE, in_channels, nodes, timesteps):
        super(Spatial_Attention, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(timesteps).to(DEVICE))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, timesteps).to(DEVICE))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.bs = nn.Parameter(torch.FloatTensor(1, nodes, nodes).to(DEVICE))
        self.Vs = nn.Parameter(torch.FloatTensor(nodes, nodes).to(DEVICE))

    def forward(self, x):
        '''
        ASTGCN - spatial attention
        :param x: (B, N, in_F, T)
        :return: (B, N, N)
        '''

        # (B, N, F, T)(T) -> (B, N, F)(F, T) -> (B, N, T)
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)

        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(B, N, F, T) -> (B, N, T) -> (B, T, N)

        product = torch.matmul(lhs, rhs)  # (B, N, T)(B, T, N) -> (B, N, N)

        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N, N)(B, N, N) -> (B, N, N)

        S_normalized = F.softmax(S, dim=1)

        return S_normalized


class Encoder(nn.Module):
    def __init__(self, gcn1_in_feature, gcn1_out_feature, gcn2_out_feature, dropout, node_length, nb_time_filter, device):
        super(Encoder, self).__init__()
        self.spatial_gcn = GCN_layer(gcn1_in_feature, gcn1_out_feature, gcn2_out_feature)
        self.dropout = dropout
        self.time_convlstm = ConvLSTM(node_length, node_length, nb_time_filter, kernel_size=3, stride=1, padding=1, DEVICE=device)

    def forward(self, x, adj):
        '''
        Encoder for spatial-temporal correlations
        :param x: (B, N, in_F, T)
        :param adj: (N, N)
        :return: (T, B, N, convlstm_F)
        '''

        spatial_gcn = self.spatial_gcn(x, adj) # (B, N, in_F, T) - (B, N, gcn_F, T)

        spatial_gcn = F.dropout(spatial_gcn, self.dropout, training=self.training)

        # (B, N, gcn_F, T) - (T, B, N, gcn_F) - (T, B, N, convlstm1_F)
        output, h_state = self.time_convlstm(spatial_gcn.permute(3, 0, 1, 2), None, x.shape[-1])

        return h_state


class Predictor(nn.Module):
    def __init__(self, time_step, node_length, nb_time_filter, pre_len, device):
        super(Predictor, self).__init__()
        self.time_convlstm = ConvLSTM(node_length, node_length, nb_time_filter, kernel_size=3, stride=1, padding=1, DEVICE=device)
        self.upconv = nn.Conv2d(pre_len, time_step, kernel_size=(1, 1), stride=(1, 1))
        self.bn = nn.BatchNorm2d(nb_time_filter)
        self.pre_len = pre_len

    def forward(self, hidden_state):
        '''
        Predictor for multi-step prediction
        :param hidden_state: (T, B, N, convlstm_F)
        :return: (B, T, N, convlstm_F)
        '''

        # (T, B, N, convlstm_F) - (pre_T, B, N, convlstm_F)
        conv_feature, h_state = self.time_convlstm(None, hidden_state, self.pre_len)

        # (pre_T, B, N, convlstm_F) - (T, B, N, convlstm_F) - (B, T, N, convlstm_F)
        output = self.upconv(conv_feature.permute(1, 0, 2, 3))

        output = self.bn(output.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)  # (B, T, N, convlstm_F)

        return output


class AutoEncoder(nn.Module):
    def __init__(self, gcn1_in_feature, gcn1_out_feature, gcn2_out_feature, dropout, node_length, nb_time_filter,
                 time_step, pre_len, device):
        super(AutoEncoder, self).__init__()
        self.encoder_layer = Encoder(gcn1_in_feature, gcn1_out_feature, gcn2_out_feature, dropout, node_length, nb_time_filter, device)
        self.decoder_layer = Predictor(time_step, node_length, nb_time_filter, pre_len, device)

    def forward(self, x, adj):
        '''
        Encoder-Predictor architecture
        :param x: (B, N, in_F, T)
        :param adj: (N, N)
        :return: (B, T, N, convlstm_F)
        '''

        encoder = self.encoder_layer(x, adj)  # (B, N, in_F, T) - (T, B, N, convlstm_F)
        decoder = self.decoder_layer(encoder) # (T, B, N, convlstm_F) - (B, T, N, convlstm_F)

        return decoder


class ConvLSTM(nn.Module):
    def __init__(self, input_channel, num_filter, embedding, kernel_size, stride, padding, DEVICE):
        super(ConvLSTM, self).__init__()
        self._conv = nn.Conv1d(in_channels=input_channel + num_filter,
                               out_channels=num_filter * 4,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        self._state = embedding
        self.DEVICE = DEVICE
        self.Wci = nn.Parameter(torch.zeros(1, num_filter, self._state)).to(DEVICE)
        self.Wcf = nn.Parameter(torch.zeros(1, num_filter, self._state)).to(DEVICE)
        self.Wco = nn.Parameter(torch.zeros(1, num_filter, self._state)).to(DEVICE)
        self._input_channel = input_channel
        self._num_filter = num_filter


    def forward(self, inputs=None, states=None, seq_len=None):
        # inputs and states should not be all none
        if states is None:
            c = torch.zeros((inputs.size(1), self._num_filter, self._state), dtype=torch.float).to(self.DEVICE)
            h = torch.zeros((inputs.size(1), self._num_filter, self._state), dtype=torch.float).to(self.DEVICE)
        else:
            h, c = states

        outputs = []
        for index in range(seq_len):
            # initial inputs
            if inputs is None:
                x = torch.zeros((h.size(0), self._input_channel, self._state), dtype=torch.float).to(self.DEVICE)
            else:
                x = inputs[index, ...]
            cat_x = torch.cat([x, h], dim=1)

            # Conv-1D for dimension matching
            conv_x = self._conv(cat_x)

            i, f, tmp_c, o = torch.chunk(conv_x, 4, dim=1)

            # Input gate, Forget fate, Cell memory state, Output gate
            i = torch.sigmoid(i + self.Wci * c)
            f = torch.sigmoid(f + self.Wcf * c)
            c = f * c + i * torch.tanh(tmp_c)
            o = torch.sigmoid(o + self.Wco * c)
            h = o * torch.tanh(c)

            outputs.append(h)
        return torch.stack(outputs), (h, c)
