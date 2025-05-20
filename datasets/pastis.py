import os
import torch
import numpy as np
from torch.utils.data import Dataset

class PASTISDataset(Dataset):
    def __init__(self, root, split='train', temporal_mode='mean'):
        self.root = root
        self.split = split
        self.temporal_mode = temporal_mode
        
        # Dataset paths (UPDATED MASK DIRECTORY)
        self.sits_dir = os.path.join(self.root, "DATA_S2")
        self.mask_dir = os.path.join(self.root, "ANNOTATIONS", "INSTANCE_ANNOTATIONS")  # Changed
        
        # Verify paths exist
        if not os.path.exists(self.sits_dir):
            raise FileNotFoundError(f"SITS directory missing: {self.sits_dir}")
        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"Mask directory missing: {self.mask_dir}")

        # Get all S2 samples (UPDATED FILENAME HANDLING)
        self.samples = [f for f in os.listdir(self.sits_dir) if f.endswith('.npy')]
        
        # Verify samples have corresponding masks (FIXED MASK PATH CONSTRUCTION)
        for fname in self.samples:
            mask_fname = fname.replace("S2", "mask").replace("s2", "mask")  # Case-insensitive
            mask_path = os.path.join(self.mask_dir, mask_fname)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask file missing: {mask_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Load time-series data
        sits_path = os.path.join(self.sits_dir, self.samples[idx])
        sits = np.load(sits_path)  # [T, 10, 128, 128]

        # Load corresponding mask (FIXED MASK PATH)
        mask_fname = self.samples[idx].replace("S2", "mask").replace("s2", "mask")
        mask_path = os.path.join(self.mask_dir, mask_fname)
        mask = np.load(mask_path).astype(np.int64)  # [128, 128]

        # Temporal aggregation
        if self.temporal_mode == 'mean':
            image = torch.tensor(sits.mean(axis=0), dtype=torch.float32)  # [10, 128, 128]
        elif self.temporal_mode == 'stack':
            image = torch.tensor(sits, dtype=torch.float32)  # [T, 10, 128, 128]

        # Normalization (scale to [0, 1])
        image = image / 10000.0

        return image, torch.tensor(mask)