# AM-RGCN
Augmented Multi-component Recurrent Graph Convolustional Network for traffic flow forecasting([AM-RGCN])(https://www.mdpi.com/2220-9964/11/2/88)

# Dataset
The public traffic datasets, PEMSD4 and PEMSD8, are the real highway traffic datasets in California released by Guo ([ASTGCN](https://github.com/wanhuaiyu/ASTGCN/blob/master/papers/2019%20AAAI_Attention%20Based%20Spatial-Temporal%20Graph%20Convolutional%20Networks%20for%20Traffic%20Flow%20Forecasting.pdf)). 
The observations of the sensors are aggregated into 5-minute windows, including three dimensions of time-stamped total traffic flow, average speed, and average occupancy. 
Moreover, the geographic information of the sensors is also contained. 

- PeMSD4 records two months of statistics on traffic flow, ranging from Jan 1st 2018 to Feb 28th 2018, including 307 sensors on the highways of San Francisco Bay.
We choose data on the first 50 days as the training set and valid set, and the remaining 9 days as the test set. 
- PeMSD8 contains two months of statistics on traffic flow, ranging from July 1st 2016 to Aug 31st 2016, including 170 sensors on the highways of San Bernardino. 
We select data on the first 50 days as the training set and valid set, and the remaining 12 days as the test set. 

- visual/showdata.npy. Didi’s real-world traffic flow data ranging from 31 October 2019 to 30 November 2019 in Beijing (a small district which has been masked).

# Parameter Setting
The detail setting of our experiment refers to our paper. 


CUDA memory-usage: >7GB for PEMSD8; >12GB for PEMSD4. You can reduce the batch_size if necessary.


# Usage
You need edit the options in opt.py:
 - dataset: str, choose *pems04* or *pems08*
 - save_path: str, checkpoint path for model
 - adj: str, the path of adjacency matrix, *distance08.csv* or *distance04.csv*
 - Multidataset: str,  whether there exists Multidataset, create one automaticly if not
 - process_method: str, *MultiComponent* or *SlideWindow*
 - hdwps: str, hour(h), day(d), week(w), and shift(s) are multiples of prediction(p) length
 - model:  **AM-RGCN, Baseline_LSTM, Baseline_GRU, MCSTGCN, ASTGCN, DM_LSTM_GCN**. We provide the realizations of six models. More details of these models can be found in the paper.
 
 if model is in (**AM-RGCN, MCSTGCN, ASTGCN, DM_LSTM_GCN**)
 
```
 python Multi_train.py
 
 python Multi_test.py
```

 if model is in (**Baseline_LSTM, Baseline_GRU**) the process method is *SlideWindow*
 ```
 python lstm_gru_train.py
 
 python lstm_gru_test.py
 ```
 
 # Citation
 @Article{ijgi11020088,
AUTHOR = {Zhang, Chi and Zhou, Hong-Yu and Qiu, Qiang and Jian, Zhichun and Zhu, Daoye and Cheng, Chengqi and He, Liesong and Liu, Guoping and Wen, Xiang and Hu, Runbo},
TITLE = {Augmented Multi-Component Recurrent Graph Convolutional Network for Traffic Flow Forecasting},
JOURNAL = {ISPRS International Journal of Geo-Information},
VOLUME = {11},
YEAR = {2022},
NUMBER = {2},
ARTICLE-NUMBER = {88},
URL = {https://www.mdpi.com/2220-9964/11/2/88},
ISSN = {2220-9964},
ABSTRACT = {Due to the periodic and dynamic changes of traffic flow and the spatial&ndash;temporal coupling interaction of complex road networks, traffic flow forecasting is highly challenging and rarely yields satisfactory prediction results. In this paper, we propose a novel methodology named the Augmented Multi-component Recurrent Graph Convolutional Network (AM-RGCN) for traffic flow forecasting by addressing the problems above. We first introduce the augmented multi-component module to the traffic forecasting model to tackle the problem of periodic temporal shift emerging in traffic series. Then, we propose an encoder&ndash;decoder architecture for spatial&ndash;temporal prediction. Specifically, we propose the Temporal Correlation Learner (TCL) which incorporates one-dimensional convolution into LSTM to utilize the intrinsic temporal characteristics of traffic flow. Moreover, we combine TCL with the graph convolutional network to handle the spatial&ndash;temporal coupling interaction of the road network. Similarly, the decoder consists of TCL and convolutional neural networks to obtain high-dimensional representations from multi-step predictions based on spatial&ndash;temporal sequences. Extensive experiments on two real-world road traffic datasets, PEMSD4 and PEMSD8, demonstrate that our AM-RGCN achieves the best results.},
DOI = {10.3390/ijgi11020088}
}
