# -*- coding:utf-8 -*-
import argparse


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # filename
        self.parser.add_argument('--dataset', type=str, default='./dataset/pems08.npz', help='path for dataset')
        self.parser.add_argument('--save_path', type=str, default='./checkpoint/PEMS08', help='path for saving model')
        self.parser.add_argument('--adj', type=str, default='./dataset/distance08.csv', help='filename for adjacency matrix')
        self.parser.add_argument('--Multidataset', type=str, default='./dataset/pems08_h2_d1_w1_p12_s1_MultiComponent.npz',
                                 help='whether there exists Multidataset, create one if not.')

        '''
        if '--Multidataset' is not existing, create one based on '--hdwps'.
        Note that we use the same Multidataset as p=12(1 hour) when predicting p = 6(30min) or 3(15min) in '--hdwps'
        e.g We predict different time slices p all based on pems08_h2_d1_w1_p12_s1_MultiComponent.npz  
        '''
        self.parser.add_argument('--hdwps', type=str, default='2,1,1,12,1',
                                 help='hour(h), day(d), week(w), and shift(s) are multiples of prediction(p) ')
        '''
        if 'process_method' is MultiComponent, there requires existing Multidataset
        if model is LSTM or GRU, 'process_method' chooses SlideWindow.
        '''
        self.parser.add_argument('--process_method', type=str, default='MultiComponent', help='MultiComponent |SlideWindow')

        # GCN-based model network parameter
        self.parser.add_argument('--gcn1_out_feature', type=int, default=128, help='out_feature of GCN layer1')
        self.parser.add_argument('--gcn2_out_feature', type=int, default=64, help='out_feature of GCN layer2')
        self.parser.add_argument('--nb_time_filter', type=int, default=64, help='out_feature of CNN')
        self.parser.add_argument('--dropout', type=float, default=0.8, help='only for DM-RGCN')
        self.parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')

        # hidden_size for lstm and gru
        self.parser.add_argument('--lstm_hidden', type=int, default=64, help='hidden size of lstm or gru')

        self.parser.add_argument('--model', type=str, default='DMRGCN',
                                 help='DMRGCN |Baseline_LSTM |Baseline_GRU |MCSTGCN |ASTGCN |DM_LSTM_GCN')

    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()

        self.opt.isTrain = self.isTrain
        args = vars(self.opt)

        return self.opt


class TrainOptions(BaseOptions):
    # Override
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
        self.parser.add_argument('--optimizer', type=str, default='adam', help='[sgd | adam]')
        self.parser.add_argument('--epoch', type=int, default=150, help='number of training epochs')
        self.isTrain = True


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.isTrain = False
