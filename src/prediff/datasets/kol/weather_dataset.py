import os
import torch
import numpy as np
import sys
import lightning as L
import torch
import torch_harmonics as th



project_name = "deeponet"
current_path = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = current_path[:current_path.find(project_name) + len(project_name)]
os.chdir(ROOT_PATH)
CWD = os.getcwd()
sys.path.append(CWD)

class WeatherDataset(torch.utils.data.Dataset):

    def __init__(self, n_init_state = 5, pred_length=1, split="train",
                 train_ratio = 0.8, val_ratio = 0.1, standardize=True):
        super().__init__()
        # === parameters ===
        self.n_init_state = n_init_state
        self.standardize = standardize
        self.pred_length = pred_length 
        self.sample_length = self.pred_length + self.n_init_state

        # === data preprocess ===
        data_file = './src/data/slp/slp-2002-2022-4degree.npy'
        lat = np.load('./src/data/slp/lat-4degree.npy')
        lon = np.load('./src/data/slp/lon-4degree.npy')

        # normalize location to [0,1] use min max
        lat_min = lat.min()
        lat_max = lat.max()
        lon_min = lon.min()
        lon_max = lon.max()

        lat = (lat - lat_min)/(lat_max - lat_min)
        lon = (lon - lon_min)/(lon_max - lon_min)

        x,y = np.meshgrid(lon,lat)
        self.loc = np.stack([x.flatten(),y.flatten()],axis=0)

        # add SHT features to be computed once
        lon_grid = torch.from_numpy(x)
        lat_grid = torch.from_numpy(y)

        sht = th.RealSHT(len(lat), len(lon), grid="equiangular")
        lon_coeffs = sht(lon_grid)
        lat_coeffs = sht(lat_grid)

        z1 = lat_coeffs.abs()
        z1 = z1.repeat(lon.shape[0],1)
        z2 = lon_coeffs.abs()
        z2 = z2.repeat(lon.shape[0],1)

        self.lat_sht = z1
        self.lon_sht = z2
        



        # === dataset split ===
        assert split in ("train", "val", "test"), "Unknown dataset split"
        assert train_ratio + val_ratio < 1, "train_ratio + val_ratio must be less than 1"
        self.data = np.load(data_file)
        # flatten spatial dimensions
        self.data = self.data.reshape(self.data.shape[0], -1) # (n_t, N_grid)
        # self.data = self.data[...,None] # (n_t, N_grid, d_features)
        self.nt = self.data.shape[0]

        self.train_nt = int(self.nt * train_ratio)
        self.val_nt = int(self.nt * val_ratio)
        self.test_nt = self.nt - self.train_nt - self.val_nt

        # === normalization of sample and loc ===
        # calculate mean and std using training data
        if self.standardize:
            self.data_mean = self.data[:self.train_nt].mean(axis=0)
            self.data_std = self.data[:self.train_nt].std(axis=0)
            self.data = (self.data - self.data_mean)/self.data_std

        if split == "train":
            self.data = self.data[:self.train_nt]
        elif split == "val":
            self.data = self.data[self.train_nt:self.train_nt+self.val_nt]
        elif split == "test":
            self.data = self.data[self.train_nt+self.val_nt:]

        
        # self.loc[0] = (self.loc[0] - lon_min)/(lon_max - lon_min)
        # self.loc[1] = (self.loc[1] - lat_min)/(lat_max - lat_min)
            
        self.nt = self.data.shape[0]


        # === create samples by sliding windows ===
        self.n_samples = self.nt-self.sample_length+1 # number of samples in dataset

    def __len__(self):
        # return 1000
        return self.n_samples

    def __getitem__(self, idx):
        # === Sample ===
        sample = self.data[idx:idx+self.sample_length]
        sample_input = sample[:self.n_init_state]
        sample_target = sample[self.n_init_state:]

        sample_input = torch.tensor(sample_input, dtype = torch.float32).permute(1,0)
        sample_target = torch.tensor(sample_target,dtype=torch.float32).permute(1,0)

        return sample_input, sample_target, torch.tensor(self.loc).permute(1,0), self.lon_sht, self.lat_sht
    
if __name__ == '__main__':
    # test dataset
    dataset = WeatherDataset(split= "train", standardize=True, pred_length=1, n_init_state=5)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    sample_input, sample_target,loc = next(iter(dataloader))
    print(sample_input.shape)
    print(sample_target.shape)
    print(loc.shape)
    print(loc.min())
    print(loc.max())
    # x, y, loc = dataset[0]
    # print(dataset.data.shape)
    # print(x.shape)
    # print(y.shape)
    # print(loc.shape)
