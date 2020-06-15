# -*- coding:utf-8 -*-
import time
import random
import torch.optim as optim

from torch.autograd import Variable
from utils import *
from models import Baseline_LSTM, Baseline_GRU
from opts import TrainOptions
from torch.utils.data import DataLoader
from PreprocessData import datasets

# Parameter option
opt = TrainOptions().parse()

# Intinial datasets class
OpenDataset = datasets(opt.dataset)

#
node_length = 170 if "pems08" in opt.dataset else 307
predict_length = int(opt.hdwps.split(',')[-2])
periodic_shift = int(opt.hdwps.split(',')[-1])
process_method = opt.process_method

# Get SlideWindow sequence
features, labels, test_features, test_labels, max_data = load_data(OpenDataset, opt.dataset, process_method, None,
                                                                   node_length, opt.Multidataset, opt.hdwps,
                                                                   is_training=True)
print(features.size(), labels.size(), test_features.size(), test_labels.size())

# get training dataset
train_Dataset = PEMS_dataset(features, labels)
train_loader = DataLoader(dataset=train_Dataset,
                          batch_size=32,
                          shuffle=True)

# get training dataset
test_Dataset = PEMS_dataset(test_features, test_labels)
test_loader = DataLoader(dataset=test_Dataset,
                         batch_size=32,
                         shuffle=False)

use_gpu = torch.cuda.is_available()

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if use_gpu:
    torch.cuda.manual_seed(42)

model, optimizer = None, None

# Define the model and optimizer
if (opt.model == 'Baseline_LSTM'):
    print("| Constructing Baseline_LSTM model...")
    model = Baseline_LSTM(
        node_length=node_length,
        input_size=node_length,
        hidden_size=opt.lstm_hidden,
        pre_len=predict_length,
    )
elif (opt.model == 'Baseline_GRU'):
    print("| Constructing Baseline_GRU model...")
    model = Baseline_GRU(
        node_length=node_length,
        input_size=node_length,
        hidden_size=opt.lstm_hidden,
        pre_len=predict_length,
    )
else:
    raise NotImplementedError

if (opt.optimizer == 'sgd'):
    optimizer = optim.SGD(
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

# Initialization method
for p in model.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)
    else:
        torch.nn.init.uniform_(p)

if use_gpu:
    model.cuda()

# Define the optimizer
print('Net\'s state_dict:')
total_param = 0
for param_tensor in model.state_dict():
    print(param_tensor, '\t', model.state_dict()[param_tensor].size())
    total_param += np.prod(model.state_dict()[param_tensor].size())

# Parameter information
print('Net\'s total params:', total_param)

print('Optimizer\'s state_dict:')
for var_name in optimizer.state_dict():
    print(var_name, '\t', optimizer.state_dict()[var_name])

if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')

# save path
save_point = opt.save_path
if not os.path.isdir(save_point):
    os.mkdir(save_point)



def train(inputs, labels, is_training):
    # Train
    if is_training:
        model.train()
        optimizer.lr = opt.lr
        optimizer.zero_grad()

        output = model(inputs)
        criterion = torch.nn.MSELoss()
        loss_train = criterion(output * max_data, labels * max_data)

        loss_train.backward()
        optimizer.step()

        return loss_train

    # Validation
    else:
        with torch.no_grad():
            output = model(test_x)
            output = output * max_data
            test_y = labels * max_data

            MSE = torch.mean(torch.pow((output - test_y), 2))
            RMSE = torch.sqrt(torch.mean(torch.pow((output - test_y), 2)))
            MAE = torch.mean(torch.abs(output - test_y))
            return MSE, RMSE, MAE


# Main code for training
if __name__ == "__main__":
    print("\n[STEP 2] : Obtain (feature, label) matrix")
    print("| Feature matrix   : {}".format(features.shape))
    print("| Label matrix     : {}".format(labels.shape))

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
                inputs, labels = list(map(lambda x: x.cuda(), [inputs, labels]))
                inputs, labels = list(map(lambda x: Variable(x), [inputs, labels]))

            # batch loss
            total_loss += train(inputs, labels, True)

            # average loss
            ave_loss = total_loss / (index + 1)

        for index, data in enumerate(test_loader):
            test_x, test_y = data

            if use_gpu:
                test_x, test_y = list(map(lambda x: x.cuda(), [test_x, test_y]))

            # batch loss
            MSE, RMSE, MAE = train(test_x, test_y, False)
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
            torch.save(state, os.path.join(save_point, '%s.t7' % (opt.model + '_' + str(predict_length))))

        end_time = time.time()
        print("=>{}".format(epoch),
              "|Training:{:5.2f}".format(ave_loss.data.cpu().numpy()),
              "|Test:{:5.2f}".format(mse_loss.data.cpu().numpy()),
              "|Best:: {:.2f}".format(best_acc),
              "|epoch: {}".format(best_epoch),
              "|MAE:{:.2f}".format(mae_loss.data.cpu().numpy()),
              "|RMSE:{:.2f}".format(rmse_loss.data.cpu().numpy()),
              "|time:{}".format(end_time - start_time),
              )

    print("\n=> Training finished!")
