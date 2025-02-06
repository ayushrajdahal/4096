import os
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
import pandas as pd
from utils.timefeatures import time_features
from utils.custom_timefeatures import custom_time_features # alternate time features that use sine-cosine.

class Dataset_Energy_hour(Dataset):
    def __init__(self, root_path="./datasets/Energy/", flag='train', size=None, 
                 features='S', data_path='load_forecasting.csv',
                 target='nat_demand', scale=True, timeenc=0, freq='h', scaler_name='standard'): # NOTE: scaler_name isn't passed in the referenced variation
        if size == None:
            self.seq_len = 24 * 4 * 4   # length considered
            self.label_len = 24 * 4     # ground truth
            self.pred_len = 24 * 4      # prediction
        else:
            (self.seq_len, self.label_len, self.pred_len) = size
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'test':1, 'val':2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.scaler_name = scaler_name

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):

        scaler_dict = {
            'standard': StandardScaler,         # often a good choice for time series data, especially when dealing with outliers or when the ranges in training and prediction data are significantly different
            'minmax': MinMaxScaler,             # may not be the best choice for time series data with significant outliers or volatility, could be useful with neural networks that have output activation functions constrained to specific ranges
            'quantile': QuantileTransformer,    # transforms the features to follow a uniform or normal distribution, which can be beneficial for algorithms that assume normally distributed data.
        }

        # check if the scaler specified is supported
        assert self.scaler_name in scaler_dict.keys(), f"scaler_name:{self.scaler_name} not supported. here are the possible values: {scaler_dict.keys()}"
        
        # use the specified scaler by fetching from scaler_dict, () added for creating an object
        self.scaler = scaler_dict[self.scaler_name]()

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS': # multivariate data
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S': # univariate data
            df_data = df_raw[[self.target]] # two square brackets because we want to fetch a dataframe (not series)
        
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['datetime']][border1:border2]
        df_data['datetime'] = pd.to_datetime(df_stamp.datetime)

        # encode time features
        if self.timeenc == 0: # integer values of day, month, etc.
            df_stamp['month'] = df_data['datetime'].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_data['datetime'].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_data['datetime'].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_data['datetime'].apply(lambda row: row.day, 1)
            df_stamp = df_stamp.drop(['datetime']).values
        elif self.timeenc == 1: # [-0.5, 0.5]
            data_stamp = time_features(pd.to_datetime(df_stamp['datetime'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1,0)
        elif self.timeenc == 2: # sines/cosines
            data_stamp = custom_time_features(pd.to_datetime(df_stamp['datetime'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1,0)
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # EXTRA STUFF FOR TESTING BEGINS
        self.border1 = border1
        self.border2 = border2
        self.data = data
        # EXTRA STUFF FOR TESTING ENDS

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        # actual data
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        # corresponding time embedding dataframe
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1 # ???

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path="./datasets/ETT-small/", flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

