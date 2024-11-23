import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import random
from torch.utils.data import DistributedSampler


class TsForecastingDataset(Dataset):

    def __init__(self, data, seq_len, stride, starts=np.array([]), horizon=1):
        self.sequences = get_sub_seqs(data, seq_len=seq_len, stride=stride, start_discont=starts, return_seqs=True)
        self.horizon = horizon

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        feature = self.sequences[idx][:-self.horizon, :]
        y = self.sequences[idx][-self.horizon:, :]

        return feature, y


class TsReconstructionDataset(Dataset):

    def __init__(self, data, seq_len, stride, starts=np.array([])):
        sequences = get_sub_seqs(data, seq_len=seq_len, stride=stride, start_discont=starts, return_seqs=True)
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        feature = self.sequences[idx]
        return feature


class TsForecastingDataset_former(Dataset):

    def __init__(self, data, data_stamp, seq_len, label_len, pred_len, seq_starts):

        self.data_x = data
        self.data_y = data
        self.data_stamp = data_stamp

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.seq_starts = seq_starts

    def __len__(self):
        return len(self.seq_starts)

    def __getitem__(self, idx_):
        s_begin = self.seq_starts[idx_]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark


class TsReconstructionDataset_former(Dataset):

    def __init__(self, data, data_stamp, seq_len, seq_starts, is_train=True):

        self.data_x = data
        self.data_y = data
        self.is_train = is_train
        self.data_stamp = data_stamp

        self.seq_len = seq_len
        self.seq_starts = seq_starts

    def __len__(self):
        return len(self.seq_starts)

    def __getitem__(self, idx_):
        s_begin = self.seq_starts[idx_]
        s_end = s_begin + self.seq_len

        seq_x = self.data_x[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]

        return seq_x, seq_x_mark


def get_train_data_loaders(x_seqs: np.ndarray, batch_size: int, train_val_percentage: float, seed: int, shuffle: bool = True,
    usetorch = True, num_workers=0, distributed=False):
    """
    Splits the train data between train, val, etc. Creates and returns pytorch data loaders
    :param shuffle: boolean that determines whether samples are shuffled before splitting the data
    :param seed: seed used for the random shuffling (if shuffling there is)
    :param x_seqs: input data where each row is a sample (a sequence) and each column is a channel
    :param batch_size: number of samples per batch
    :param splits: list of split fractions, should sum up to 1.
    :param usetorch: if True returns dataloaders, otherwise return datasets
    :return: a tuple of data loaders as long as splits. If len_splits = 1, only 1 data loader is returned
    """
    # if np.sum(splits) != 1:
    #     scale_factor = np.sum(splits)
    #     splits = [fraction/scale_factor for fraction in splits]

    dataset_len = int(len(x_seqs))
    train_use_len = int(dataset_len * (1 - train_val_percentage))
    val_use_len = int(dataset_len * train_val_percentage)

    if shuffle:
        np.random.seed(seed)
        val_start_index = random.randrange(train_use_len)
    else:
        val_start_index = dataset_len - val_use_len

    indices = np.arange(dataset_len)
    val_indices = indices[val_start_index:val_start_index+val_use_len]
    train_indices = np.concatenate([indices[:val_start_index], indices[val_start_index+val_use_len:]])

    if usetorch:
        train_sub_indices = torch.from_numpy(train_indices)
        val_sub_indices = torch.from_numpy(val_indices)

        if isinstance(x_seqs, torch.utils.data.Dataset):

            if distributed:
                train_sampler = DistributedSampler(Subset(x_seqs, train_sub_indices), shuffle=True)
                val_sampler = DistributedSampler(Subset(x_seqs, val_indices), shuffle=False)
                train_loaders = torch.utils.data.DataLoader(x_seqs, batch_size=batch_size, sampler=train_sampler, drop_last=True, num_workers=num_workers)
                val_loaders = torch.utils.data.DataLoader(x_seqs, batch_size=batch_size, sampler=val_sampler, drop_last=True, num_workers=num_workers)
            else:
                train_loaders = DataLoader(Subset(x_seqs, train_sub_indices), batch_size=batch_size, shuffle=True,drop_last=True, pin_memory=True, num_workers=num_workers)
                val_loaders = DataLoader(Subset(x_seqs, val_sub_indices), batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True, num_workers=num_workers)
        else:
            if distributed:
                train_sampler = DistributedSampler(x_seqs[train_indices], shuffle=True)
                val_sampler = DistributedSampler(x_seqs[val_indices], shuffle=False)
                train_loaders = torch.utils.data.DataLoader(x_seqs, batch_size=batch_size, sampler=train_sampler, drop_last=True, num_workers=num_workers)
                val_loaders = torch.utils.data.DataLoader(x_seqs, batch_size=batch_size, sampler=val_sampler, drop_last=True, num_workers=num_workers)
            else:
                train_loaders = DataLoader(x_seqs[train_indices], batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=num_workers)
                val_loaders = DataLoader(x_seqs[val_indices], batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True, num_workers=num_workers)
        loaders = tuple([train_loaders, val_loaders])
        return loaders
    else:
        datasets = tuple([x_seqs[train_indices], x_seqs[val_indices]])
        return datasets


def get_sub_seqs(x_arr, seq_len, stride=1, start_discont=np.array([]), return_seqs=True):
    """
    :param start_discont: the start points of each sub-part in case the x_arr is just multiple parts joined together
    :param x_arr: dim 0 is time, dim 1 is channels
    :param seq_len: size of window used to create subsequences from the data
    :param stride: number of time points the window will move between two subsequences
    :return:
    """
    excluded_starts = []
    [excluded_starts.extend(range((start - seq_len + 1), start)) for start in start_discont if start > seq_len]
    seq_starts = np.setdiff1d(np.arange(0, x_arr.shape[0] - seq_len + 1, stride), excluded_starts)
    if return_seqs:
        x_seqs = np.array([x_arr[i:i + seq_len] for i in seq_starts])
        return x_seqs
    else:
        return seq_starts


