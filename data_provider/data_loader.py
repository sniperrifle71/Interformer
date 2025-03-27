import os, sys
import warnings
import re

sys.path.append("..")

current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import datasets
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from utils.timefeatures import time_features


warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
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


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
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

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
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
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
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


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
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

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
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


class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
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
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)): int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values

        self.data_x = df
        self.data_y = df

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 4))
        seq_y_mark = torch.zeros((seq_x.shape[0], 4))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class UCRAnomalyloader(Dataset):
    def __init__(self, root_path, data_path, seq_len, patch_len, flag="train"):
        self.root_path = root_path
        self.data_path = data_path
        self.seq_len = seq_len
        self.input_len = seq_len - patch_len
        self.patch_len = patch_len
        self.flag = flag
        self.stride = 1 if self.flag == "train" else self.seq_len - 2 * self.patch_len
        self.dataset_file_path = os.path.join(self.root_path, self.data_path)
        data_list = []
        assert self.dataset_file_path.endswith('.txt')
        try:
            with open(self.dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    data_list.append(data_line)
            self.data = np.stack(data_list, 0)
        except ValueError:
            with open(self.dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line[0].split()])
            self.data = data_line
            self.data = np.expand_dims(self.data, axis=1)

        self.border = self.find_border_number(self.data_path)
        self.scaler = StandardScaler()
        self.scaler.fit(self.data[:self.border])
        self.data = self.scaler.transform(self.data)
        if self.flag == "train":
            self.data = self.data[:self.border]
        else:
            self.data = self.data[self.border - self.patch_len:]

    def find_border_number(self, input_string):
        parts = input_string.split('_')

        if len(parts) < 3:
            return None

        border_str = parts[-3]

        try:
            border = int(border_str)
            return border
        except ValueError:
            return None

    def __len__(self):
        return (self.data.shape[0] - self.seq_len) // self.stride + 1

    def __getitem__(self, index):
        index = index * self.stride
        return self.data[index:index + self.seq_len, :]



"""
All single-variate series in UTSD are divided into (input-output) windows with a uniform length based on S3.
"""
class UTSDataset(Dataset):
    def __init__(self, subset_name=r'UTSD-4G', flag='train', split=0.9,
                 input_len=None, output_len=None, scale=True, stride=1):
        self.input_len = input_len
        self.output_len = output_len
        self.seq_len = input_len + output_len
        assert flag in ['train', 'val', 'test','pretrain']
        assert split >= 0 and split <=1.0
        flag = 'train' if flag=='pretrain' else flag
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag
        
        self.scale = scale
        self.split = split
        self.stride = stride

        self.data_list = []
        self.n_window_list = []
        self.backup_list = []

        self.subset_name = subset_name
        self.__read_data__()

    def __read_data__(self):
        print(self.subset_name)
        dataset = datasets.load_dataset("thuml/utsd", self.subset_name, split='train')
        # split='train' contains all the time series, which have not been divided into splits, 
        # you can split them by yourself, or use our default split as train:val = 9:1
        # train:val:test = 8:1:1
        print('Indexing dataset...')

        print(len(dataset))
        for item in tqdm(dataset):
            self.scaler = StandardScaler()
            data = item['target']
            data = np.array(data).reshape(-1, 1)
            num_train = int(len(data) * self.split)
            num_vali = int(len(data) * 0.1)
            num_test = len(data) - num_train - num_vali
            border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(data)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.scale:
                train_data = data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data)

                data = self.scaler.transform(data)

            data = data[border1:border2]
            n_window = (len(data) - self.seq_len) // self.stride + 1
            if n_window < 1:
                # print("Data {}: {} length too small".format(item['item_id'], len(item['target'])))
                self.backup_list+=item['target']
                continue

            self.data_list.append(data)
            self.n_window_list.append(n_window if len(self.n_window_list) == 0 else self.n_window_list[-1] + n_window)
            if len(self.backup_list)>2880:
                data = np.array(self.backup_list).reshape(-1, 1)
                num_train = int(len(data) * self.split)
                num_vali = int(len(data) * 0.1)
                num_test = len(data) - num_train - num_vali
                border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
                border2s = [num_train, num_train + num_vali, len(data)]
                border1 = border1s[self.set_type]
                border2 = border2s[self.set_type]

                if self.scale:
                    train_data = data[border1s[0]:border2s[0]]
                    self.scaler.fit(train_data)

                    data = self.scaler.transform(data)

                data = data[border1:border2]
                n_window = (len(data) - self.seq_len) // self.stride + 1
                self.data_list.append(data)
                self.n_window_list.append(n_window if len(self.n_window_list) == 0 else self.n_window_list[-1] + n_window)
                self.backup_list = []

    def __getitem__(self, index):
        # you can wirte your own processing code here
        dataset_index = 0
        while index >= self.n_window_list[dataset_index]:
            dataset_index += 1

        index = index - self.n_window_list[dataset_index - 1] if dataset_index > 0 else index
        n_timepoint = (len(self.data_list[dataset_index]) - self.seq_len) // self.stride + 1

        s_begin = index % n_timepoint
        s_begin = self.stride * s_begin
        s_end = s_begin + self.seq_len
        p_begin = s_end
        p_end = p_begin + self.output_len
        seq_x = self.data_list[dataset_index][s_begin:s_end, :]
        seq_y = self.data_list[dataset_index][p_begin:p_end, :]
        
        if seq_y.shape[0] <self.output_len:
            padding_size = self.output_len - seq_y.shape[0]
            seq_y = np.concatenate([seq_y, np.zeros((padding_size,1))], axis=0)
        seq_x_mark = np.zeros((seq_x.shape))
        seq_y_mark = np.zeros((seq_y.shape))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.n_window_list[-1]


# See ```download_dataset.py``` to download the dataset first
if __name__ == '__main__':
    current_file_path = os.path.abspath(__file__)
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
    raw_path = os.path.join(root_path,'data', 'raw_data',"huggingface")
    # dataset = UTSDataset(subset_name=r'UTSD-1G', input_len=672, output_len=0, flag='train')
    dataset = datasets.load_dataset("thuml/utsd", 'UTSD-1G', split='train')
    # dataset = dataset.select(range(5000))
    # filtered_data = dataset.filter(lambda x: re.match(r'^Health_SelfRegulationSCP', x['item_id']))
    # print(len(filtered_data))
    length_list = []
    for item in dataset:
        if len(item['target']) not in length_list:
            length_list.append(len(item['target']))
            print(item['item_id'])
            print(len(item['target']))
    
