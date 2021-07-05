# -*- coding:utf-8 -*-
import numpy as np
import scipy.sparse as sp
import torch
import os
import random
from torch.utils.data import Dataset

def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)

    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt).tocoo()


def load_data(OpenDataset, filename, process_method, adj_name, node_length, multi_filename, hdwps, is_training):
    '''
    Get SlideWindow data or Dynamic MultiComponent data
    :param OpenDataset: a class for preparing data
    :param filename: string, original dataset
    :param process_method: string, 'SlideWindow' or 'MultiComponent'
    :param adj_name: string, adjacency filename
    :param node_length: int, num of nodes
    :param multi_filename: string, Dynamic MultiComponent filename
    :param hdwps: string, length of hour,day,week,prediction,temporal shift, respectively
    :param is_training: bool, True of False
    :return: max_data is float type, others are np.ndarray
    '''

    pre_len = int(hdwps.split(',')[-2])

    # SlideWindow means the nearest time slice
    if process_method == 'SlideWindow':

        # if there is not adj, meaning to use LSTM or GRU, else GCN
        win_len = 8 if adj_name is not None else 1

        # get slidewindow sequence
        sample, label, max_data = OpenDataset.PEMS_SlideWindow(filename, WindowSize=win_len, PredictSize=1)

        sample = np.expand_dims(sample.transpose(0, 2, 1), -1)
        label = np.expand_dims(label.transpose(0, 2, 1)[:, :, :pre_len], -1)
        print(sample.shape, label.shape)

        # PEMSD8 or PEMSD4
        pre_day = 12 if "8" in filename else 9

        # split training, validation, test dataset
        split_line1 = int(len(sample) * 0.6)
        split_line2 = int(len(sample) - 12 * 24 * pre_day)

        # if adjacency matrix is not none, get Laplacian matrix
        if adj_name is not None:
            adj, distance_adj = OpenDataset.PEMS_adjacency_matrix(adj_name, node_length)
            print(adj.shape)
            adj = normalize_adj(sp.coo_matrix(adj) + sp.eye(adj.shape[0]))
            adj = torch.FloatTensor(np.array(adj.todense()))

        # training process
        if is_training:
            print("SlideWindow Training Processing")
            train_sample = sample[:split_line1]
            train_label = label[:split_line1]
            val_sample = sample[split_line1:split_line2]
            val_label = label[split_line1:split_line2]

            features = torch.FloatTensor(train_sample)
            labels = torch.FloatTensor(train_label)
            val_features = torch.FloatTensor(val_sample)
            val_labels = torch.FloatTensor(val_label)

            # if adjacency matrix is not none, return GCN, else return LSTM or GRU
            if adj_name is not None:
                return adj, features, labels, val_features, val_labels, max_data

            return features, labels, val_features, val_labels, max_data

        # test process
        else:
            print("SlideWindow Testing Processing")
            test_sample = sample[split_line2:]
            test_label = label[split_line2:]

            features = torch.FloatTensor(test_sample)
            labels = torch.FloatTensor(test_label)

            if adj_name is not None:
                return adj, features, labels, max_data

            return features, labels, max_data

    # Dynamic Multi-component
    elif "MultiComponent" in process_method:

        # get Laplacian matrix
        adj, distance_adj = OpenDataset.PEMS_adjacency_matrix(adj_name, node_length)
        print(adj.shape)
        adj = normalize_adj(sp.coo_matrix(adj) + sp.eye(adj.shape[0]))
        adj = torch.FloatTensor(np.array(adj.todense()))

        # Dynamic Multi-Component parameter
        num_hdwps = hdwps.split(',')
        num_of_shift = int(num_hdwps[4])
        num_for_predict = int(num_hdwps[3])
        num_of_weeks = int(num_hdwps[2])
        num_of_days = int(num_hdwps[1])
        num_of_hours = int(num_hdwps[0])

        print("Dynamic MultiComponent parameter: hour {}, day {}, week {}, predict {}, temporal shift {}".format(
            num_of_hours, num_of_days,
            num_of_weeks,
            num_for_predict,
            num_of_shift))

        # whether existing Dynamic Multi-Component Dataset
        if not os.path.exists(multi_filename):
            print("Generate Dynamic Multi-Component Dataset")
            _ = OpenDataset.PEMS_MultiComponent(filename, num_of_weeks, num_of_days, num_of_hours, num_for_predict,
                                                num_of_shift, True)

        file_data = np.load(multi_filename)
        max_data = file_data['max_data'].tolist()
        if is_training:
            print(process_method + " Training Processing")
            train_x = file_data['train_x']
            train_target = file_data['train_target'][:, :, :pre_len]

            val_x = file_data['val_x']
            val_target = file_data['val_target'][:, :, :pre_len]

            features = torch.FloatTensor(np.expand_dims(train_x, -1))
            labels = torch.FloatTensor(np.expand_dims(train_target, -1))
            val_x = torch.FloatTensor(np.expand_dims(val_x, -1))
            val_target = torch.FloatTensor(np.expand_dims(val_target, -1))

            return adj, features, labels, val_x, val_target, max_data

        else:
            print(process_method + " Testing Processing")
            test_x = file_data['test_x']
            test_target = file_data['test_target'][:, :, :pre_len]

            test_features = torch.FloatTensor(np.expand_dims(test_x, -1))
            test_labels = torch.FloatTensor(np.expand_dims(test_target, -1))

            return adj, test_features, test_labels, max_data


class PEMS_dataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        return (self.X[index], self.Y[index])

    def __len__(self):
        return len(self.X)
