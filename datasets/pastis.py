import os
import torch
import numpy as np
from torch.utils.data import Dataset

class PASTISDataset(Dataset):
    def __init__(self, root, split='train', temporal_mode='mean'):
        self.root = root
        self.split = split
        self.temporal_mode = temporal_mode  # 'mean' or 'stack'
        self.sits_dir = os.path.join(root, split, 'S2')
        self.mask_dir = os.path.join(root, split, 'ANNOTATIONS')

        # List all samples
        self.samples = [f for f in os.listdir(self.sits_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Load time-series data (T×10×128×128)
        sits_path = os.path.join(self.sits_dir, self.samples[idx])
        sits = np.load(sits_path)  # Shape: [T, 10, 128, 128]

        # Load semantic mask (128×128)
        mask_path = os.path.join(self.mask_dir, self.samples[idx].replace('s2', 'mask'))
        mask = np.load(mask_path).astype(np.int64)  # Shape: [128, 128]

        # Temporal aggregation (mean or stack)
        if self.temporal_mode == 'mean':
            image = torch.tensor(sits.mean(axis=0), dtype=torch.float32)  # [10, 128, 128]
        elif self.temporal_mode == 'stack':
            image = torch.tensor(sits, dtype=torch.float32)  # [T, 10, 128, 128]

        # Normalize (example: scale to [0, 1])
        image = image / 10000.0

        return image, torch.tensor(mask)