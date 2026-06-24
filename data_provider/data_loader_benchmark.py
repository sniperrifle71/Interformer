import warnings

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.timefeatures import time_features

warnings.filterwarnings('ignore')


DEEP_MIMO_KEYS = (
    'channel',
    'channels',
    'channel_data',
    'H',
    'h',
    'DeepMIMO',
    'data',
)


def _to_2d_time_series(data):
    data = np.asarray(data)
    data = np.squeeze(data)
    if data.size == 0:
        raise ValueError('DeepMIMO data is empty')
    if np.iscomplexobj(data):
        if data.ndim == 1:
            data = np.stack([data.real, data.imag], axis=-1)
        else:
            data = np.concatenate([data.real, data.imag], axis=-1)
    data = data.astype(np.float32, copy=False)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    elif data.ndim > 2:
        data = data.reshape(data.shape[0], -1)
    return data


def _select_numeric_array(data_dict, dataset_file_path):
    for key in DEEP_MIMO_KEYS:
        if key in data_dict:
            try:
                return _to_2d_time_series(data_dict[key])
            except (TypeError, ValueError):
                pass

    candidates = []
    items = data_dict.items() if hasattr(data_dict, 'items') else (
        (key, data_dict[key]) for key in data_dict.files
    )
    for key, value in items:
        if key.startswith('__'):
            continue
        array = np.asarray(value)
        if array.size == 0:
            continue
        if np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
            candidates.append((key, array))
    if not candidates:
        raise ValueError(
            'No numeric DeepMIMO array found in {}. Expected one of {} or a numeric array.'.format(
                dataset_file_path,
                ', '.join(DEEP_MIMO_KEYS),
            )
        )
    candidates.sort(key=lambda item: item[1].size, reverse=True)
    return _to_2d_time_series(candidates[0][1])


def read_deepmimo_data(dataset_file_path):
    if dataset_file_path.endswith('.csv'):
        df_raw = pd.read_csv(dataset_file_path)
        if 'date' in df_raw.columns:
            df_raw = df_raw.drop(columns=['date'])
        return pd.DataFrame(df_raw.values)
    if dataset_file_path.endswith('.txt'):
        return pd.read_csv(dataset_file_path, header=None)
    if dataset_file_path.endswith('.npy'):
        data = np.load(dataset_file_path, allow_pickle=True)
        return pd.DataFrame(_to_2d_time_series(data))
    if dataset_file_path.endswith('.npz'):
        data = np.load(dataset_file_path, allow_pickle=True)
        return pd.DataFrame(_select_numeric_array(data, dataset_file_path))
    if dataset_file_path.endswith('.mat'):
        data = loadmat(dataset_file_path)
        return pd.DataFrame(_select_numeric_array(data, dataset_file_path))
    raise ValueError('Unknown DeepMIMO data format: {}'.format(dataset_file_path))


class CIDatasetBenchmark(Dataset):
    def __init__(self, root_path='dataset', flag='train', input_len=None, pred_len=None,
                 data_type='custom', scale=True, timeenc=1, freq='h', stride=1, subset_rand_ratio=1.0):
        self.subset_rand_ratio = subset_rand_ratio
        # size [seq_len, label_len, pred_len]
        # info
        self.input_len = input_len
        self.pred_len = pred_len
        self.seq_len = input_len + pred_len
        self.timeenc = timeenc
        self.scale = scale
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.data_type = data_type
        if self.set_type == 0:
            self.internal = int(1 // self.subset_rand_ratio)
        else:
            self.internal = 1

        self.root_path = root_path
        self.dataset_name = self.root_path.split('/')[-1].split('.')[0]

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        dataset_file_path = self.root_path
        if self.data_type == 'DeepMIMO':
            df_raw = read_deepmimo_data(dataset_file_path)
        elif dataset_file_path.endswith('.csv'):
            df_raw = pd.read_csv(dataset_file_path)
        elif dataset_file_path.endswith('.txt'):
            df_raw = []
            with open(dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    df_raw.append(data_line)
            df_raw = np.stack(df_raw, 0)
            df_raw = pd.DataFrame(df_raw)

        elif dataset_file_path.endswith('.npz'):
            data = np.load(dataset_file_path, allow_pickle=True)
            data = data['data'][:, :, 0]
            df_raw = pd.DataFrame(data)
        else:
            raise ValueError('Unknown data format: {}'.format(dataset_file_path))

        if self.data_type == 'DeepMIMO':
            data_len = len(df_raw)
            num_train = int(data_len * 0.7)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.input_len, data_len - num_test - self.input_len]
            border2s = [num_train, num_train + num_vali, data_len]
        elif self.data_type == 'custom':
            data_len = len(df_raw)
            num_train = int(data_len * 0.3)
            num_test = int(data_len * 0.7)
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.input_len, data_len - num_test - self.input_len]
            border2s = [num_train, num_train + num_vali, data_len]
        elif self.data_type == 'ETTh' or self.data_type == 'ETTh1' or self.data_type == 'ETTh2':
            border1s = [0, 12 * 30 * 24 - self.input_len, 12 * 30 * 24 + 4 * 30 * 24 - self.input_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif self.data_type == 'ETTm' or self.data_type == 'ETTm1' or self.data_type == 'ETTm2':
            border1s = [0, 12 * 30 * 24 * 4 - self.input_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.input_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        elif self.data_type == 'PEMS':
            data_len = len(df_raw)
            num_train = int(data_len * 0.6)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.input_len, data_len - num_test - self.input_len]
            border2s = [num_train, num_train + num_vali, data_len]
        else:
            data_len = len(df_raw)
            num_train = int(data_len * 0.7)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.input_len, data_len - num_test - self.input_len]
            border2s = [num_train, num_train + num_vali, data_len]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        has_date_column = 'date' in df_raw.columns
        if has_date_column and isinstance(df_raw[df_raw.columns[0]][2], str):
            data = df_raw[df_raw.columns[1:]].values
        else:
            data = df_raw.values

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        if self.timeenc == 0:
            if not has_date_column:
                data_stamp = np.zeros((len(df_raw), 4))
            else:
                df_stamp = df_raw[['date']]
                df_stamp['date'] = pd.to_datetime(df_stamp.date)
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
                df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
                data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            if has_date_column and isinstance(df_raw[df_raw.columns[0]][2], str):
                data_stamp = time_features(pd.to_datetime(pd.to_datetime(df_raw.date).values), freq='h')
                data_stamp = data_stamp.transpose(1, 0)
            else:
                data_stamp = np.zeros((len(df_raw), 4))
        else:
            raise ValueError('Unknown timeenc: {}'.format(self.timeenc))

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp[border1:border2]

        self.n_var = self.data_x.shape[-1]
        self.n_timepoint = len(self.data_x) - self.input_len - self.pred_len + 1
        if self.n_timepoint < 1:
            raise ValueError(
                '{} split is too short for input_len={} and pred_len={}. Got {} time points.'.format(
                    self.dataset_name,
                    self.input_len,
                    self.pred_len,
                    len(self.data_x),
                )
            )

    def __getitem__(self, index):
        if self.set_type == 0:
            index = index * self.internal
        c_begin = index // self.n_timepoint  # select variable
        s_begin = index % self.n_timepoint   # select start time
        s_end = s_begin + self.input_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        seq_x = self.data_x[s_begin:s_end, c_begin:c_begin + 1]
        seq_y = self.data_y[r_begin:r_end, c_begin:c_begin + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0:
            return max(int(self.n_var * self.n_timepoint * self.subset_rand_ratio), 1)
        else:
            return int(self.n_var * self.n_timepoint)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class CIAutoRegressionDatasetBenchmark(CIDatasetBenchmark):
    def __init__(self, root_path='dataset', flag='train', input_len=None, label_len=None, pred_len=None,
                 data_type='custom', scale=True, timeenc=1, freq='h', stride=1, subset_rand_ratio=1.0):
        self.label_len = label_len
        super().__init__(root_path=root_path, flag=flag, input_len=input_len, pred_len=pred_len,
                         data_type=data_type, scale=scale, timeenc=timeenc, freq=freq, stride=stride,
                         subset_rand_ratio=subset_rand_ratio)

    def __getitem__(self, index):
        if self.set_type == 0:
            index = index * self.internal
        c_begin = index // self.n_timepoint  # select variable
        s_begin = index % self.n_timepoint   # select start time
        s_end = s_begin + self.input_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, c_begin:c_begin + 1]
        seq_y = self.data_y[r_begin:r_end, c_begin:c_begin + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark
