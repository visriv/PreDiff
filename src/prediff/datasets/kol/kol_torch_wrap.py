"""
Code is adapted from https://github.com/amazon-science/earth-forecasting-transformer/blob/e60ff41c7ad806277edc2a14a7a9f45585997bd7/src/earthformer/datasets/sevir/sevir_torch_wrap.py
Add data augmentation.
Only return "VIL" data in `torch.Tensor` format instead of `Dict`
"""
import os
from typing import Union, Dict, Sequence, Tuple, List
import numpy as np
import datetime
import pandas as pd
import h5py
import torch
from torch import nn
from torch.utils.data import Dataset as TorchDataset, DataLoader, random_split
from torchvision import transforms
from einops import rearrange
from lightning import LightningDataModule, seed_everything
# from .kol_dataloader import kolDataLoader
import math






class kolTorchDataset(TorchDataset):

    def __init__(self, data_path='datasets/kol/results.h5', split="train", window_length = 1,
                 train_ratio = 0.8, val_ratio = 0.1, standardize=True,
                 flatten = False, crop=0):
        super().__init__()
        # === parameters ===
        self.standardize = standardize
        self.crop = crop
        self.window_length = window_length
        # === data preprocess ===
        print(os.getcwd())
        data_file = data_path
        hdf_file = h5py.File(data_file, 'r')
        velocity = hdf_file['velocity_field'][:]

        self.data = velocity
        self.row = velocity.shape[1]
        self.col = velocity.shape[2]
        self.num_channels = velocity.shape[3]

        # lat = np.load('datasets/kol/lat-4degree.npy')
        # lon = np.load('datasets/kol/lon-4degree.npy')


        # normalize location to [0,1] use min max
        # lat_min = lat.min()
        # lat_max = lat.max()
        # lon_min = lon.min()
        # lon_max = lon.max()

        # lat = (lat - lat_min)/(lat_max - lat_min)
        # lon = (lon - lon_min)/(lon_max - lon_min)


        

        # === dataset split ===
        assert split in ("train", "val", "test"), "Unknown dataset split"
        assert train_ratio + val_ratio < 1, "train_ratio + val_ratio must be less than 1"

        # crop if required
        if (crop > 0):
            min_y = self.row//2-int(crop/2)
            max_y = self.row//2+int(crop/2)
            min_x = self.col//2-int(crop/2)
            max_x = self.col//2+int(crop/2)
            self.data = self.data[:, min_y:max_y, min_x:max_x]

        self.num_rows = self.data.shape[1]
        self.num_cols = self.data.shape[2]


        # flatten spatial dimensions
        if (flatten == True):
            self.data = self.data.reshape(self.data.shape[0], -1) # (n_t, N_grid)
        
        # self.data = self.data[...,None] # (n_t, N_grid, d_features)
        self.nt = self.data.shape[0]

        self.train_nt = int(self.nt * train_ratio)
        self.val_nt = int(self.nt * val_ratio)
        self.test_nt = self.nt - self.train_nt - self.val_nt

        # === normalization of sample ===
        # calculate mean and std using training data
        if self.standardize:
            if (flatten):
                self.data_mean = self.data[:self.train_nt].mean(axis=0)
                self.data_std = self.data[:self.train_nt].std(axis=0)
            else:
                self.data_mean = self.data[:self.train_nt].mean()
                self.data_std = self.data[:self.train_nt].std()
            self.data = (self.data - self.data_mean)/self.data_std

        if split == "train":
            self.data = self.data[:self.train_nt]
        elif split == "val":
            self.data = self.data[self.train_nt:self.train_nt+self.val_nt]
        elif split == "test":
            self.data = self.data[self.train_nt+self.val_nt:]

            
        self.nt = self.data.shape[0]


        # === create samples by sliding windows ===
        self.n_samples = self.nt-self.window_length+1 # number of samples in dataset

    def __len__(self):
        # return 1000
        return self.n_samples

    def __getitem__(self, idx):
        # === Sample ===

        sample = self.data[idx:idx+self.window_length]


        sample = sample.reshape(-1,
                                self.window_length, 
                                self.num_rows,
                                self.num_cols,
                                self.num_channels) # num_channels
        
 
        sample = torch.tensor(sample, dtype = torch.float32)
        sample = torch.squeeze(sample, 0)
        return sample
    


class kolLightningDataModule(LightningDataModule):

    def __init__(self,
                 data_path = 'datasets/kol/results.h5',
                 seq_len: int = 5,
                 crop: int = 0,
                 layout: str = "NTHWC",
                 output_type = np.float32,
                 rescale_method: str = "01",
                 verbose: bool = False,
                #  aug_mode: str = "0",
                 ret_contiguous: bool = True,
                 # datamodule_only
                 dataset_name: str = "era5_4deg",
                 kol_dir: str = None,
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 batch_size: int = 1,
                 num_workers: int = 1,
                 seed: int = 0,
                 ):
        super(kolLightningDataModule, self).__init__()
        self.data_path = data_path
        self.seq_len = seq_len
        self.crop = crop
        # self.stride = stride
        assert layout[0] == "N"
        self.layout = layout.replace("N", "")
        self.output_type = output_type
        # self.preprocess = preprocess
        self.rescale_method = rescale_method
        self.verbose = verbose
        # self.aug_mode = aug_mode
        self.ret_contiguous = ret_contiguous
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        if kol_dir is not None:
            kol_dir = os.path.abspath(kol_dir)
        if dataset_name == "kol_30":
            # if kol_dir is None:
            #     kol_dir = default_dataset_sevir_dir
            # catalog_path = os.path.join(kol_dir, "CATALOG.csv")
            # raw_data_dir = os.path.join(kol_dir, "data")
            raw_seq_len = 49
            interval_real_time = 5
            img_height = 90
            img_width = 46
        elif dataset_name == "kol_40":
            # if kol_dir is None:
            #     kol_dir = default_dataset_sevirlr_dir
            # catalog_path = os.path.join(kol_dir, "CATALOG.csv")
            # raw_data_dir = os.path.join(kol_dir, "data")
            raw_seq_len = 25
            interval_real_time = 5
            img_height = 360
            img_width = 181
        else:
            raise ValueError(f"Wrong dataset name {dataset_name}. Must be 'era5_4deg' or 'era5_1deg'.")
        self.dataset_name = dataset_name
        self.kol_dir = kol_dir
        # self.catalog_path = catalog_path
        # self.raw_data_dir = raw_data_dir
        self.raw_seq_len = raw_seq_len
        self.interval_real_time = interval_real_time
        self.img_height = img_height
        self.img_width = img_width
        # train val test split
        # self.start_date = datetime.datetime(*start_date) \
        #     if start_date is not None else None
        # self.train_test_split_date = datetime.datetime(*train_test_split_date) \
        #     if train_test_split_date is not None else None
        # self.end_date = datetime.datetime(*end_date) \
        #     if end_date is not None else None
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        

    def prepare_data(self) -> None:
        # if os.path.exists(self.kol_dir):
        #     # Further check
        #     assert os.path.exists(self.catalog_path), f"CATALOG.csv not found! Should be located at {self.catalog_path}"
        #     assert os.path.exists(self.raw_data_dir), f"SEVIR data not found! Should be located at {self.raw_data_dir}"
        # else:
        #     if self.dataset_name == "era5_4deg":
        #         download_SEVIR(save_dir=os.path.dirname(self.kol_dir))
        #     elif self.dataset_name == "era5_1deg":
        #         download_SEVIRLR(save_dir=os.path.dirname(self.kol_dir))
        #     else:
        #         raise NotImplementedError
        pass
    def setup(self, stage = None) -> None:
        seed_everything(seed=self.seed)
        if stage in (None, "fit"):
            print(os.getcwd())

            self.kol_train = kolTorchDataset(
                split= "train", 
                data_path = self.data_path,
                window_length=self.seq_len,
                train_ratio = self.train_ratio,
                val_ratio = self.val_ratio,
                standardize=True,
                crop=self.crop)
            
            self.kol_val = kolTorchDataset(
                split= "val", 
                data_path = self.data_path,
                window_length=self.seq_len,
                train_ratio = self.train_ratio,
                val_ratio = self.val_ratio,
                standardize=True,
                crop=self.crop)

            self.std = self.kol_train.data_std
            self.mean = self.kol_train.data_mean


            
        if stage in (None, "test"):
            self.kol_test = kolTorchDataset(
                split= "test",                 
                data_path = self.data_path,
                window_length=self.raw_seq_len,
                train_ratio = self.train_ratio,
                val_ratio = self.val_ratio,
                standardize=True,
                crop=self.crop)
            
            self.std = self.kol_test.data_std
            self.mean = self.kol_test.data_mean

    def train_dataloader(self):
        return DataLoader(self.kol_train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.kol_val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.kol_test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    @property
    def num_train_samples(self):
        return len(self.kol_train)

    @property
    def num_val_samples(self):
        return len(self.kol_val)

    @property
    def num_test_samples(self):
        return len(self.kol_test)
