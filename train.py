import sys
sys.path.append("/kaggle/working/CAT-Seg")  # Add project directory to Python path

from datasets import get_dataset  # Now this should work

import torch
import yaml
from torch.utils.data import DataLoader
from datasets import get_dataset
from models.catseg import CATSeg

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# Load config
config = load_yaml('configs/pastis.yaml')

# Dataset (FIXED HERE)
train_dataset = get_dataset(
    dataset_name="pastis",  # Pass dataset_name explicitly
    root="/kaggle/input/pastis-dataset-ddp/PASTIS",
    split="train",
    temporal_mode="mean"  # or "stack"
)

# Model
model = CATSeg(
    clip_model="ViT-B/16",  # Hardcode model args
    num_classes=19,
    input_channels=10
).cuda()

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(50):  # Hardcode epochs
    model.train()
    for images, masks in DataLoader(train_dataset, batch_size=8):
        images, masks = images.cuda(), masks.cuda()
        outputs = model(images)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/50 | Loss: {loss.item():.4f}")