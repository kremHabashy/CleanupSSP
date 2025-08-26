import os
import torch
import numpy as np
from torch.utils.data import Dataset, random_split
from power_spherical import HypersphericalUniform
from scipy.stats import qmc

def uniform_noise(ssp_dim):
    dist = HypersphericalUniform(dim=ssp_dim)
    return dist.rsample((1,))

def gaussian_noise(ssp_dim):
    noise = np.random.normal(0, 1, size=ssp_dim)
    return torch.tensor(noise, dtype=torch.float32)

class SSPDataset(Dataset):
    def __init__(self, data_dir, ssp_dim, target_type='coordinate', noise_type='uniform_hypersphere', signal_strength=1.0, mode='train', device="cpu"):
        """
        Args:
            data_dir (str): The directory where the data is stored.
            ssp_dim (int): The dimensionality of the SSP vectors.
            target_type (str): Type of the target distribution ('coordinate' or 'scene').
            noise_type (str): Type of the initial noise distribution ('uniform_hypersphere' or 'gaussian').
            signal_strength (float): Ratio of noise in the initial distribution.
            mode (str): Mode of the dataset, either 'train' or 'test'.
            device (str): The device to use for tensors ('cpu' or 'cuda'). (Kept for API compatibility but not used internally)
        """
        self.data_dir = data_dir
        self.ssp_dim = ssp_dim
        self.target_type = target_type
        self.noise_type = noise_type
        self.signal_strength = signal_strength
        self.device = device  # retained for compatibility but not used here

        if mode not in ['train', 'test']:
            raise ValueError(f"Unknown mode: {mode}")

        self.data_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.npy')])
        if not self.data_files:
            raise FileNotFoundError(f"No data files found in {self.data_dir}")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        target_file = os.path.join(self.data_dir, self.data_files[idx])
        target_ssp = np.load(target_file)
        target_ssp = torch.tensor(target_ssp, dtype=torch.float32)  # CPU

        if self.noise_type == 'uniform_hypersphere':
            noise_ssp = uniform_noise(self.ssp_dim).squeeze(0)  # shape (d,)
        elif self.noise_type == 'gaussian':
            noise_ssp = gaussian_noise(self.ssp_dim)
        else:
            raise ValueError(f"Unknown noise_type: {self.noise_type}")

        prior_start = self.signal_strength * target_ssp + (1 - self.signal_strength) * noise_ssp
        return prior_start, target_ssp  # both CPU

    def split_dataset(self, val_split=0.1):
        val_size = int(len(self) * val_split)
        train_size = len(self) - val_size
        train_dataset, val_dataset = random_split(self, [train_size, val_size])
        return train_dataset, val_dataset
