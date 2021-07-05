# -*- coding:utf-8 -*-
import time
from utils import *
from opts import TestOptions
from torch.utils.data import DataLoader
from PreprocessData import datasets

# Parameter option
opt = TestOptions().parse()

# Intinial datasets class
OpenDataset = datasets(opt.dataset)

# Important Parameter
node_length = 170 if "pems08" in opt.dataset else 307
predict_length = int(opt.hdwps.split(',')[-2])
periodic_shift = int(opt.hdwps.split(',')[-1])
process_method = opt.process_method

adj, features, labels, max_data = load_data(OpenDataset, opt.dataset, process_method, opt.adj, node_length,
                                            opt.Multidataset, opt.hdwps, is_training=False)
print(adj.size(), features.size(), labels.size(), max_data)
use_gpu = torch.cuda.is_available()

print("\n[STEP 2] : Obtain (adjacency, feature, label) matrix")
print("| Adjacency matrix : {}".format(adj.shape))
print("| Feature matrix   : {}".format(features.shape))
print("| Label matrix     : {}".format(labels.shape))

# get best model
load_model = torch.load(
    os.path.join(opt.save_path,
                 '%s.t7' % (opt.model + '_' + process_method + '_' + str(predict_length) + '_' + str(periodic_shift))))
model = load_model['model']
epoch = load_model['epoch']
acc_val = load_model['acc']
MAE_val = load_model['MAE']
RMSE_val = load_model['RMSE']

if use_gpu:
    _, features, adj, labels = list(map(lambda x: x.cuda(), [model, features, adj, labels]))

# get test dataset
test_Dataset = PEMS_dataset(features, labels)
test_loader = DataLoader(dataset=test_Dataset,
                         batch_size=opt.batch_size,
                         shuffle=False)


def test(features, labels, adj):
    with torch.no_grad():
        output = model(features, adj)
        output = output * max_data
        labels = labels * max_data

        # Set to 0 if the predicted result is less than 0
        output = torch.where(output > 0, output, torch.full_like(output, 0))

        MSE = torch.mean(torch.pow((output - labels), 2))
        RMSE = torch.sqrt(torch.mean(torch.pow((output - labels), 2)))
        MAE = torch.mean(torch.abs(output - labels))

        return MSE, RMSE, MAE


if __name__ == "__main__":

    print("\n[STEP 4] : Testing")
    print(opt.model + '_' + process_method)
    print(load_model)
    MSE_loss = 0.0
    MAE_loss = 0.0
    RMSE_loss = 0.0

    start_time = time.time()
    for index, data in enumerate(test_loader):
        inputs, labels = data

        # batch loss
        MSE, RMSE, MAE = test(inputs, labels, adj)

        MSE_loss += MSE
        MAE_loss += MAE
        RMSE_loss += RMSE

        # average loss
        mse_loss = MSE_loss / (index + 1)
        mae_loss = MAE_loss / (index + 1)
        rmse_loss = RMSE_loss / (index + 1)

    end_time = time.time()
    print(
        "|Validation acc : {:.2f}".format(acc_val.data.cpu().numpy()),
        "|MAE_loss: {:.2f}".format(MAE_val.data.cpu().numpy()),
        "|RMSE_loss: {:.2f}".format(RMSE_val.data.cpu().numpy()),
        "|epoch {} ".format(epoch),
        "|time:{}".format(end_time - start_time),
    )
    print("| Test acc : {:.2f}".format(mse_loss.data.cpu().numpy()),
          "|MAE_loss: {:.2f}".format(mae_loss.data.cpu().numpy()),
          "|RMSE_loss: {:.2f}".format(rmse_loss.data.cpu().numpy()), )
