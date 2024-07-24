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
import torch
from torch import nn
from torch.utils.data import Dataset as TorchDataset, DataLoader, random_split
from torchvision import transforms
from einops import rearrange
from lightning import LightningDataModule, seed_everything
from .e4deg_dataloader import e4degDataLoader
import math





# class e4degTorchDataset(TorchDataset):

#     orig_dataloader_layout = "NHWT"
#     orig_dataloader_squeeze_layout = orig_dataloader_layout.replace("N", "")
#     aug_layout = "THW"

#     def __init__(self,
#                  seq_len: int = 5,
#                  raw_seq_len: int = 10,
#                  sample_mode: str = "sequent",
#                  stride: int = 12,
#                  layout: str = "THWC",
#                  split_mode: str = "uneven",
#                  sevir_catalog: Union[str, pd.DataFrame] = None,
#                  sevir_data_dir: str = None,
#                  start_date: datetime.datetime = None,
#                  end_date: datetime.datetime = None,
#                  datetime_filter = None,
#                  catalog_filter = "default",
#                  shuffle: bool = False,
#                  shuffle_seed: int = 1,
#                  output_type = np.float32,
#                  preprocess: bool = True,
#                  rescale_method: str = "01",
#                  verbose: bool = False,
#                  aug_mode: str = "0",
#                  ret_contiguous: bool = True):
#         super(e4degTorchDataset, self).__init__()
#         self.layout = layout.replace("C", "1")
#         self.ret_contiguous = ret_contiguous
#         self.e4deg_dataloader = e4degDataLoader(
#             data_types=["vil", ],
#             seq_len=seq_len,
#             raw_seq_len=raw_seq_len,
#             sample_mode=sample_mode,
#             stride=stride,
#             batch_size=1,
#             layout=self.orig_dataloader_layout,
#             num_shard=1,
#             rank=0,
#             split_mode=split_mode,
#             sevir_catalog=sevir_catalog,
#             sevir_data_dir=sevir_data_dir,
#             start_date=start_date,
#             end_date=end_date,
#             datetime_filter=datetime_filter,
#             catalog_filter=catalog_filter,
#             shuffle=shuffle,
#             shuffle_seed=shuffle_seed,
#             output_type=output_type,
#             preprocess=preprocess,
#             rescale_method=rescale_method,
#             downsample_dict=None,
#             verbose=verbose)
#         self.aug_mode = aug_mode
#         if aug_mode == "0":
#             self.aug = lambda x:x
#         elif aug_mode == "1":
#             self.aug = nn.Sequential(
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomVerticalFlip(),
#                 transforms.RandomRotation(degrees=180),
#             )
#         elif aug_mode == "2":
#             self.aug = nn.Sequential(
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomVerticalFlip(),
#                 TransformsFixRotation(angles=[0, 90, 180, 270]),
#             )
#         else:
#             raise NotImplementedError

#     def __getitem__(self, index):
#         data_dict = self.e4deg_dataloader._idx_sample(index=index)
#         data = data_dict["vil"].squeeze(0)
#         if self.aug_mode != "0":
#             data = rearrange(data, f"{' '.join(self.orig_dataloader_squeeze_layout)} -> {' '.join(self.aug_layout)}")
#             data = self.aug(data)
#             data = rearrange(data, f"{' '.join(self.aug_layout)} -> {' '.join(self.layout)}")
#         else:
#             data = rearrange(data, f"{' '.join(self.orig_dataloader_squeeze_layout)} -> {' '.join(self.layout)}")
#         if self.ret_contiguous:
#             return data.contiguous()
#         else:
#             return data

#     def __len__(self):
#         return self.e4deg_dataloader.__len__()

class e4degTorchDataset(TorchDataset):

    def __init__(self,  split="train", window_length = 1,
                 train_ratio = 0.8, val_ratio = 0.1, standardize=True,
                 flatten = False, crop=0):
        super().__init__()
        # === parameters ===
        self.standardize = standardize
        self.crop = crop
        self.window_length = window_length
        # === data preprocess ===
        data_file = 'datasets/e4deg/slp-2002-2022-4degree.npy'
        lat = np.load('datasets/e4deg/lat-4degree.npy')
        lon = np.load('datasets/e4deg/lon-4degree.npy')


        # normalize location to [0,1] use min max
        lat_min = lat.min()
        lat_max = lat.max()
        lon_min = lon.min()
        lon_max = lon.max()

        lat = (lat - lat_min)/(lat_max - lat_min)
        lon = (lon - lon_min)/(lon_max - lon_min)


        

        # === dataset split ===
        assert split in ("train", "val", "test"), "Unknown dataset split"
        assert train_ratio + val_ratio < 1, "train_ratio + val_ratio must be less than 1"
        self.data = np.load(data_file)

        # crop if required
        if (crop > 0):
            min_y = 23-int(crop/2)
            max_y = 23+int(crop/2)
            min_x = 45-int(crop/2)
            max_x = 45+int(crop/2)
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
                                1) # num_channels
        
 
        sample = torch.tensor(sample, dtype = torch.float32)
        sample = torch.squeeze(sample, 0)
        return sample
    


class e4degLightningDataModule(LightningDataModule):

    def __init__(self,
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
                 e4deg_dir: str = None,
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 batch_size: int = 1,
                 num_workers: int = 1,
                 seed: int = 0,
                 ):
        super(e4degLightningDataModule, self).__init__()
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
        if e4deg_dir is not None:
            e4deg_dir = os.path.abspath(e4deg_dir)
        if dataset_name == "era5_4deg":
            # if e4deg_dir is None:
            #     e4deg_dir = default_dataset_sevir_dir
            # catalog_path = os.path.join(e4deg_dir, "CATALOG.csv")
            # raw_data_dir = os.path.join(e4deg_dir, "data")
            raw_seq_len = 49
            interval_real_time = 5
            img_height = 90
            img_width = 46
        elif dataset_name == "era5_1deg":
            # if e4deg_dir is None:
            #     e4deg_dir = default_dataset_sevirlr_dir
            # catalog_path = os.path.join(e4deg_dir, "CATALOG.csv")
            # raw_data_dir = os.path.join(e4deg_dir, "data")
            raw_seq_len = 25
            interval_real_time = 5
            img_height = 360
            img_width = 181
        else:
            raise ValueError(f"Wrong dataset name {dataset_name}. Must be 'era5_4deg' or 'era5_1deg'.")
        self.dataset_name = dataset_name
        self.e4deg_dir = e4deg_dir
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
        # if os.path.exists(self.e4deg_dir):
        #     # Further check
        #     assert os.path.exists(self.catalog_path), f"CATALOG.csv not found! Should be located at {self.catalog_path}"
        #     assert os.path.exists(self.raw_data_dir), f"SEVIR data not found! Should be located at {self.raw_data_dir}"
        # else:
        #     if self.dataset_name == "era5_4deg":
        #         download_SEVIR(save_dir=os.path.dirname(self.e4deg_dir))
        #     elif self.dataset_name == "era5_1deg":
        #         download_SEVIRLR(save_dir=os.path.dirname(self.e4deg_dir))
        #     else:
        #         raise NotImplementedError
        pass
    def setup(self, stage = None) -> None:
        seed_everything(seed=self.seed)
        if stage in (None, "fit"):

            self.e4deg_train = e4degTorchDataset(
                split= "train", 
                window_length=self.seq_len,
                train_ratio = self.train_ratio,
                val_ratio = self.val_ratio,
                standardize=True,
                crop=self.crop)
            
            self.e4deg_val = e4degTorchDataset(
                split= "val", 
                window_length=self.seq_len,
                train_ratio = self.train_ratio,
                val_ratio = self.val_ratio,
                standardize=True,
                crop=self.crop)

            self.std = self.e4deg_train.data_std
            self.mean = self.e4deg_train.data_mean


            
        if stage in (None, "test"):
            self.e4deg_test = e4degTorchDataset(
                split= "test", 
                window_length=self.raw_seq_len,
                train_ratio = self.train_ratio,
                val_ratio = self.val_ratio,
                standardize=True,
                crop=self.crop)
            
            self.std = self.e4deg_test.data_std
            self.mean = self.e4deg_test.data_mean

    def train_dataloader(self):
        return DataLoader(self.e4deg_train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.e4deg_val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.e4deg_test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    @property
    def num_train_samples(self):
        return len(self.e4deg_train)

    @property
    def num_val_samples(self):
        return len(self.e4deg_val)

    @property
    def num_test_samples(self):
        return len(self.e4deg_test)
