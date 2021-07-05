# -*- coding:utf-8 -*-

import time
import random
import torch.optim as optim

from torch.autograd import Variable
from utils import *
from models import MCSTGCN, ASTGCN, AM_LSTM_GCN, AMRGCN
from opts import TrainOptions
from torch.utils.data import DataLoader
from PreprocessData import datasets

# Parameter option
set_seed(6666)

opt = TrainOptions().parse()

# Intinial datasets class
OpenDataset = datasets(opt.dataset)

# Important Parameter
node_length = 170 if "pems08" in opt.dataset else 307
predict_length = int(opt.hdwps.split(',')[-2])
periodic_shift = int(opt.hdwps.split(',')[-1])
process_method = opt.process_method
train_thread = opt.train_thread

# Get Dynamic Multi-component sequence
adj, features, labels, test_features, test_labels, max_data = load_data(OpenDataset, opt.dataset, process_method,
                                                                        opt.adj, node_length, opt.Multidataset,
                                                                        opt.hdwps, is_training=True)

print(adj.size(), features.size(), labels.size(), test_features.size(), test_labels.size(), max_data,
      opt.model + '_' + process_method)

# get training dataset
train_Dataset = PEMS_dataset(features, labels)
train_loader = DataLoader(dataset=train_Dataset,
                          batch_size=opt.batch_size,
                          shuffle=True)

# get validation dataset
test_Dataset = PEMS_dataset(test_features, test_labels)
test_loader = DataLoader(dataset=test_Dataset,
                         batch_size=opt.batch_size,
                         shuffle=False)

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if use_gpu:
    torch.cuda.manual_seed(42)

model, optimizer = None, None

# Define the model
if (opt.model == 'MCSTGCN'):
    print("| Constructing MCSTGCN model...")
    model = MCSTGCN(
        time_step=features.shape[2],
        gcn1_in_feature=features.shape[-1],
        gcn1_out_feature=opt.gcn1_out_feature,
        gcn2_out_feature=opt.gcn2_out_feature,
        nb_time_filter=opt.nb_time_filter,
        pre_len=predict_length,
        time_strides=2,
    )
elif (opt.model == 'AMRGCN'):
    print("| Constructing AMRGCN model...")
    model = AMRGCN(
        node_length=node_length,
        time_step=features.shape[2],
        gcn1_in_feature=features.shape[-1],
        gcn1_out_feature=opt.gcn1_out_feature,
        gcn2_out_feature=opt.gcn2_out_feature,
        nb_time_filter=opt.nb_time_filter,
        pre_len=predict_length,
        dropout=opt.dropout,
        device=device
    )
elif (opt.model == 'ASTGCN'):
    print("| Constructing ASTGCN model...")
    model = ASTGCN(
        node_length=node_length,
        time_step=features.shape[2],
        gcn1_in_feature=features.shape[-1],
        gcn1_out_feature=opt.gcn1_out_feature,
        gcn2_out_feature=opt.gcn2_out_feature,
        nb_time_filter=opt.nb_time_filter,
        pre_len=predict_length,
        time_strides=2,
        DEVICE=device
    )
elif (opt.model == 'AM_LSTM_GCN'):
    print("| Constructing AM_LSTM_GCN model...")
    model = AM_LSTM_GCN(
        time_step=features.shape[2],
        gcn1_in_feature=features.shape[-1],
        gcn1_out_feature=opt.gcn1_out_feature,
        gcn2_out_feature=opt.gcn2_out_feature,
        nb_time_filter=opt.nb_time_filter,
        pre_len=predict_length,
    )
else:
    raise NotImplementedError

# Initialization method
for p in model.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)
    else:
        torch.nn.init.uniform_(p)

if use_gpu:
    model.cuda()

# Define the optimizer
if (opt.optimizer == 'rmsprop'):
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=opt.lr,
        weight_decay=opt.weight_decay,
        momentum=0.9
    )
elif (opt.optimizer == 'adam'):
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt.lr,
        weight_decay=opt.weight_decay
    )
else:
    raise NotImplementedError

# Parameter information
print('Net\'s state_dict:')
total_param = 0
for param_tensor in model.state_dict():
    print(param_tensor, '\t', model.state_dict()[param_tensor].size())
    total_param += np.prod(model.state_dict()[param_tensor].size())
print('Net\'s total params:', total_param)

# save path
save_point = opt.save_path
if not os.path.isdir(save_point):
    os.mkdir(save_point)


def train(inputs, labels, adj, epoch, is_training):
    # Train
    if is_training:
        model.train()
        optimizer.lr = opt.lr
        optimizer.zero_grad()

        output = model(inputs, adj)
        criterion = torch.nn.MSELoss()

        # renormalization
        if epoch >= train_thread:
            loss_train = criterion(output * max_data, labels * max_data)
        else:
            loss_train = criterion(output, labels)

        loss_train.backward()
        optimizer.step()

        return loss_train.item()

    else:
        # Validation
        with torch.no_grad():
            output = model(inputs, adj)
            output = output * max_data
            test_y = labels * max_data

            MSE = torch.mean(torch.pow((output - test_y), 2))
            RMSE = torch.sqrt(torch.mean(torch.pow((output - test_y), 2)))
            MAE = torch.mean(torch.abs(output - test_y))

            return MSE, RMSE, MAE

def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Main code for training
if __name__ == "__main__":
    print("\n[STEP 2] : Obtain (adjacency, feature, label) matrix")
    print("| Adjacency matrix : {}".format(adj.shape))
    print("| Feature matrix   : {}".format(features.shape))
    print("| Label matrix     : {}".format(labels.shape))
    print("| test_features matrix   : {}".format(test_features.shape))
    print("| test_labels matrix     : {}".format(test_labels.shape))

    best_acc = 1000000000
    best_epoch = 0
    for epoch in range(1, opt.epoch + 1):
        total_loss = 0.0
        ave_loss = 0.0
        MSE_loss = 0.0
        MAE_loss = 0.0
        RMSE_loss = 0.0
        start_time = time.time()
        for index, data in enumerate(train_loader):
            inputs, labels = data
            if use_gpu:
                inputs, labels, adj = list(map(lambda x: x.cuda(), [inputs, labels, adj]))
                inputs, labels, adj = list(map(lambda x: Variable(x), [inputs, labels, adj]))

            # batch loss
            total_loss += train(inputs, labels, adj, epoch, True)

            # average loss
            ave_loss = total_loss * max_data / (index + 1) if epoch < train_thread else total_loss / (index + 1)

        for index, data in enumerate(test_loader):
            test_x, test_y = data

            if use_gpu:
                test_x, test_y, adj = list(map(lambda x: x.cuda(), [test_x, test_y, adj]))

            # batch loss
            MSE, RMSE, MAE = train(test_x, test_y, adj, epoch, False)
            MSE_loss += MSE
            MAE_loss += MAE
            RMSE_loss += RMSE

            # average loss
            mse_loss = MSE_loss / (index + 1)
            mae_loss = MAE_loss / (index + 1)
            rmse_loss = RMSE_loss / (index + 1)

        # Best epoch
        if mse_loss < best_acc:
            best_acc = mse_loss
            best_MAE = mae_loss
            best_RMSE = rmse_loss
            state = {
                'model': model,
                'acc': best_acc,
                'epoch': epoch,
                'MAE': best_MAE,
                'RMSE': best_RMSE,
            }
            best_epoch = epoch
            torch.save(state, os.path.join(save_point,
                                           '%s.t7' % (opt.model + '_' + process_method + '_' + str(
                                               predict_length) + '_' + str(periodic_shift))))

        end_time = time.time()
        print("=>{}".format(epoch),
              "|Training:{:5.2f}".format(ave_loss),
              "|Test:{:5.2f}".format(mse_loss.data.cpu().numpy()),
              "|MAE:{:.2f}".format(mae_loss.data.cpu().numpy()),
              "|RMSE:{:.2f}".format(rmse_loss.data.cpu().numpy()),
              "|Best:: {:.2f}".format(best_acc),
              "|epoch: {}".format(best_epoch),
              "|time:{}".format(end_time - start_time),
              )

    print("\n=> Training finished!")
