import os
import torch
import numpy as np
from torch.utils.data import Dataset

class PASTISDataset(Dataset):
    def __init__(self, root, split='train', temporal_mode='mean'):
        self.root = os.path.join(root, "PASTIS")  # Add "PASTIS" subdirectory
        self.split = split
        self.temporal_mode = temporal_mode
        
        # Correct paths to match Kaggle structure
        self.sits_dir = os.path.join(self.root, "DATA_S2")  # Changed from S2 to DATA_S2
        self.mask_dir = os.path.join(self.root, "ANNOTATIONS")
        
        # Verify paths
        if not os.path.exists(self.sits_dir):
            raise FileNotFoundError(f"SITS dir missing: {self.sits_dir}")
        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"Mask dir missing: {self.mask_dir}")

        # List samples
        self.samples = [f for f in os.listdir(self.sits_dir) if f.endswith('.npy')]

    # Rest of the code remains the same...
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