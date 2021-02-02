# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GCN_layer, Spatial_Attention, Temporal_Attention, AutoEncoder


class MCSTGCN(nn.Module):
    def __init__(self, time_step, gcn1_in_feature, gcn1_out_feature, gcn2_out_feature, nb_time_filter,
                 pre_len, time_strides):
        '''
        Parameter of Multi-component stgcn.
        :param time_step: int, length of input sequence
        :param gcn1_in_feature: int, in_feature for GCN layer 1
        :param gcn1_out_feature: int, out_feature for GCN layer 1
        :param gcn2_out_feature: int, out_feature for GCN layer 2
        :param nb_time_filter: int, out_feature for time CNN
        :param pre_len: int, length of prediction
        :param time_strides: int, length of time CNN kernel stride
        '''

        super(MCSTGCN, self).__init__()
        self.spatial_gcn = GCN_layer(gcn1_in_feature, gcn1_out_feature, gcn2_out_feature)
        self.time_conv = nn.Conv2d(gcn2_out_feature, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides),
                                   padding=(0, 1))
        self.residual_conv = nn.Conv2d(gcn1_in_feature, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)
        self.final_conv = nn.Conv2d(int(time_step / time_strides), pre_len, kernel_size=(1, nb_time_filter))

    def forward(self, x, adj):
        '''
        :param x: (batch_size, node, timestep, in_feature)——(B, N, T, in_F)
        :param adj: (N, N)
        :return: (B, N, pre_T, in_F)
        '''

        x = x.permute(0, 1, 3, 2)  # (B, N, in_F, T)

        # GCN for spatial feature
        spatial_gcn = self.spatial_gcn(x, adj)  # (B, N, gcn_F, T)

        # convolution along time axis
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))  # (B, gcn_F, N, T) - (B, cnn_F, N, stride_T)

        residual_conv = self.residual_conv(x.permute(0, 2, 1, 3))  # (B, in_F, N, T) - (B, cnn_F, N, stride_T)

        # (B, cnn_F, N, stride_T) - (B, stride_T, N, cnn_F) - (B, N, cnn_F, stride_T)
        x_residual = self.ln(F.relu(residual_conv + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)

        # (B, N, cnn_F, stride_T) - (B, stride_T, N, cnn_F) - (B, pre_T, N, in_F) - (B, N, pre_T, in_F)
        pred = self.final_conv(x_residual.permute(0, 3, 1, 2)).permute(0, 2, 1, 3)

        return pred


class AMRGCN(nn.Module):
    def __init__(self, node_length, time_step, gcn1_in_feature, gcn1_out_feature, gcn2_out_feature, nb_time_filter,
                 pre_len, dropout, device):
        '''
        Parameter of Augmented Multi-component Recurrent Gcn.
        :param node_length: int, num of nodes
        :param time_step: int, length of input sequence
        :param gcn1_in_feature: int, in_feature for GCN layer 1
        :param gcn1_out_feature: int, out_feature for GCN layer 1
        :param gcn2_out_feature: int, out_feature for GCN layer 2
        :param nb_time_filter: int, out_feature for time CNN
        :param pre_len: int, length of prediction
        :param dropout: double, dropout for preventing overfitting
        :param device: cuda:0 or cpu
        '''

        super(AMRGCN, self).__init__()
        self.autoencoder = AutoEncoder(gcn1_in_feature, gcn1_out_feature, gcn2_out_feature, dropout, node_length,
                                       nb_time_filter, time_step, pre_len, device)
        self.residual_conv = nn.Conv2d(gcn1_in_feature, nb_time_filter, kernel_size=(1, 1), stride=(1, 1))
        self.ln = nn.LayerNorm(nb_time_filter)
        self.final_conv = nn.Conv2d(time_step, pre_len, kernel_size=(1, nb_time_filter))

    def forward(self, x, adj):
        '''
        :param x: (batch_size, node, timestep, in_feature)—(B, N, T, in_F)
        :param adj: (N, N)
        :return: (B, N, pre_T, in_F)
        '''

        x = x.permute(0, 1, 3, 2)  # (B, N, in_F, T)

        # Encoder-Predictor architecture: (B, N, in_F, T) - (B, T, N, convlstm_F) - (B, convlstm_F, N, T)
        time_conv_output = self.autoencoder(x, adj).permute(0, 3, 2, 1)

        # (B, in_F, N, T) - (B, convlstm_F, N, T)
        residual_conv = self.residual_conv(x.permute(0, 2, 1, 3))

        # residual connection
        x_residual = self.ln(F.relu(residual_conv + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)

        # (B, convlstm_F, N, T) - (B, T, convlstm_F, N) - (B, pre_T, N, in_F) - (B, N, pre_T, in_F)
        pred = self.final_conv(x_residual.permute(0, 3, 1, 2)).permute(0, 2, 1, 3)

        return pred


class ASTGCN(nn.Module):
    def __init__(self, node_length, time_step, gcn1_in_feature, gcn1_out_feature, gcn2_out_feature, nb_time_filter,
                 pre_len, time_strides, DEVICE):
        '''
        Parameter of Attention Multi-component stgcn.
        :param node_length: int, num of nodes
        :param time_step: int, length of input sequence
        :param gcn1_in_feature: int, in_feature for GCN layer 1
        :param gcn1_out_feature: int, out_feature for GCN layer 1
        :param gcn2_out_feature: int, out_feature for GCN layer 2
        :param nb_time_filter: int, out_feature for time CNN
        :param pre_len: int, length of prediction
        :param time_strides: int, length of time CNN kernel stride
        :param DEVICE: cuda:0 or cpu
        '''

        super(ASTGCN, self).__init__()
        self.TAt = Temporal_Attention(DEVICE, gcn1_in_feature, node_length, time_step)
        self.SAt = Spatial_Attention(DEVICE, gcn1_in_feature, node_length, time_step)
        self.spatial_gcn = GCN_layer(gcn1_in_feature, gcn1_out_feature, gcn2_out_feature)
        # ((time_step - kernel_size) + padding * 2) / stride + 1)
        self.time_conv = nn.Conv2d(gcn2_out_feature, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides),
                                   padding=(0, 1))
        self.residual_conv = nn.Conv2d(gcn1_in_feature, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)
        self.final_conv = nn.Conv2d(int(time_step / time_strides), pre_len, kernel_size=(1, nb_time_filter))

    def forward(self, x, adj):
        '''
        :param x: (batch_size, node, timestep, in_feature)—(B, N, T, in_F)
        :param adj: (N, N)
        :return: (B, N, pre_T, in_F)
        '''

        x = x.permute(0, 1, 3, 2)  # (B, N, in_F, T)

        batch_size, node_length, num_of_features, time_step = x.shape

        # Temporal Attention
        temporal_At = self.TAt(x)  # (B, T, T)

        # Get temporal score
        x_TAt = torch.matmul(x.reshape(batch_size, -1, time_step), temporal_At).reshape(batch_size, node_length,
                                                                                        num_of_features,
                                                                                        time_step)

        # Spatial Attention
        spatial_At = self.SAt(x_TAt)  # (B, N, N)

        # Get spatial score
        x_SAT = torch.matmul(x.reshape(batch_size, -1, node_length), spatial_At).reshape(batch_size, node_length,
                                                                                         num_of_features,
                                                                                         time_step)

        # GCN for spatial feature
        spatial_gcn = self.spatial_gcn(x_SAT, adj)  # (B, N, in_F, T) - (B, N, gcn_F, T)

        # convolution along time axis
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))  # (B, gcn_F, N, T) - (B, cnn_F, N, stride_T)

        residual_conv = self.residual_conv(x.permute(0, 2, 1, 3))  # (B, in_F, N, T) - (B, cnn_F, N, stride_T)

        # (B, cnn_F, N, stride_T) - (B, stride_T, N, cnn_F) - (B, N, cnn_F, stride_T)
        x_residual = self.ln(F.relu(residual_conv + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)

        # (B, N, cnn_F, stride_T) - (B, stride_T, N, cnn_F) - (B, pre_T, N, in_F) - (B, N, pre_T, in_F)
        pred = self.final_conv(x_residual.permute(0, 3, 1, 2)).permute(0, 2, 1, 3)

        return pred


class Baseline_LSTM(nn.Module):
    def __init__(self, node_length, input_size, hidden_size, pre_len):
        '''
        Parameter of LSTM
        :param node_length: int, num of nodes
        :param input_size: int, in_feature for LSTM
        :param hidden_size: int
        :param pre_len: int, length of prediction
        '''

        super(Baseline_LSTM, self).__init__()
        self.pre_len = pre_len
        self.lstm = nn.LSTM(batch_first=True, input_size=input_size, hidden_size=hidden_size)

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, node_length)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1, pre_len)
        )

    def forward(self, x):
        '''
        LSTM Baseline
        :param x: (batch_size, node, timestep, in_feature)—(B, N, T, in_F)
        :return: (B, N, pre_T, in_F)
        '''

        x = x.squeeze()  # (B, N, T)

        batch, node_length, timestep = x.shape

        x = x.permute(0, 2, 1)  # (B, T, N)

        x, _ = self.lstm(x)  # x: (B, T, hidden)

        x = x[:, -1, :].view(batch, -1)  # (B, hidden)

        x = self.fc1(x).view(batch, node_length, 1)  # (B, N, 1)

        x = self.fc2(x)  # (B, N, pre_T)

        x = x.view(batch, node_length, -1, 1)  # (B, N, pre_T, in_F)

        return x


class Baseline_GRU(nn.Module):
    def __init__(self, node_length, input_size, hidden_size, pre_len):
        '''
        Parameter of GRU
        :param node_length: int, num of nodes
        :param input_size: int, in_feature for GRU
        :param hidden_size: int
        :param pre_len: int, length of prediction
        '''
        super(Baseline_GRU, self).__init__()
        self.pre_len = pre_len
        self.gru = nn.GRU(batch_first=True, input_size=input_size, hidden_size=hidden_size)

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, node_length)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1, pre_len)
        )

    def forward(self, x):
        '''
        GRU Baseline
        :param x: (batch_size, node, timestep, in_feature)—(B, N, T, in_F)
        :return: (B, N, pre_T, in_F)
        '''

        x = x.squeeze()  # (B, N, T)

        batch, node_length, timestep = x.shape

        x = x.permute(0, 2, 1)  # (B, T, N)

        x, _ = self.gru(x)  # x: (B, T, hidden)

        x = x[:, -1, :].view(batch, -1)  # (B, hidden)

        x = self.fc1(x).view(batch, node_length, 1)  # (B, N, 1)

        x = self.fc2(x)  # (B, N, pre_T)

        x = x.view(batch, node_length, -1, 1)  # (B, N, pre_T, in_F)

        return x


class AM_LSTM_GCN(nn.Module):
    def __init__(self, time_step, gcn1_in_feature, gcn1_out_feature, gcn2_out_feature, nb_time_filter, pre_len):
        '''
        Comparison Model of replacing ConvLSTM-1D with LSTM
        :param time_step:int, length of input sequence
        :param gcn1_in_feature: int, in_feature for GCN layer 1
        :param gcn1_out_feature: int, out_feature for GCN layer 1
        :param gcn2_out_feature: int, out_feature for GCN layer 2
        :param nb_time_filter: int, out_feature for time CNN
        :param pre_len: int, length of prediction
        '''

        super(AM_LSTM_GCN, self).__init__()
        self.spatial_gcn = GCN_layer(gcn1_in_feature, gcn1_out_feature, gcn2_out_feature)
        self.gru = nn.LSTM(gcn2_out_feature, nb_time_filter, batch_first=True)
        self.residual_conv = nn.Conv2d(gcn1_in_feature, gcn2_out_feature, kernel_size=(1, 1),
                                       stride=(1, time_step / pre_len))
        self.ln = nn.LayerNorm(gcn2_out_feature)
        self.final_conv = nn.Conv2d(pre_len, pre_len, kernel_size=(1, nb_time_filter))
        self.pre_len = pre_len
        self.fc = nn.Linear(1, pre_len)

    def forward(self, x, adj):
        '''
        :param x: (batch_size, node, timestep, in_feature)——(B, N, T, in_F)
        :param adj: (N, N)
        :return: (B, N, pre_T, in_F)
        '''

        x = x.permute(0, 1, 3, 2) # (B, N, in_F, T)

        batch_size, node_length, num_of_features, time_step = x.shape

        # GCN for spatial feature
        spatial_gcn = self.spatial_gcn(x, adj)  # (B, N, gcn_F, T)

        spatial_gcn = spatial_gcn.permute(0, 3, 2, 1)  # (B, T, gcn_F, N)

        gru_cat = []

        for i in range(node_length):

            conv_gcn = spatial_gcn[:, :, :, i]  # (B, T, gcn_F)

            gru_out, _ = self.gru(conv_gcn)  # gru_out: (B, T, hidden)

            gru_cat.append(gru_out[:, -1, :].unsqueeze(-1))  # (B, hidden, 1)

        gru_output = torch.cat(gru_cat, dim=-1).unsqueeze(-1)  # (B, hidden, N, 1)

        time_output = self.fc(gru_output)  # (B, hidden, N, pre_T)

        # (B, N, in_F, T) - (B, in_F, N, T) - (B, hidden, N, pre_T)
        residual_conv = self.residual_conv(x.permute(0, 2, 1, 3))

        # (B, hidden, N, pre_T) - (B, N, pre_T, hidden) - (B, pre_T, N, hidden)
        x_residual = self.ln(F.relu(residual_conv + time_output).permute(0, 2, 3, 1)).permute(0, 2, 1, 3)

        # (B, pre_T, N, hidden) - (B, pre_T, N, in_F) - (B, N, pre_T, in_F)
        pred = self.final_conv(x_residual).permute(0, 2, 1, 3)

        return pred
