import torch
import yaml  # Add this import
from torch.utils.data import DataLoader  # Add this import
from datasets import get_dataset
from models.catseg import CATSeg

def load_yaml(path):
    """Load YAML config file"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# Load config
config = load_yaml('configs/pastis.yaml')
# Replace this:
train_dataset = get_dataset(
    name=config['dataset']['name'],
    root=config['dataset']['root'],
    split='train',
    temporal_mode=config['dataset']['temporal_mode']
)

# With this (directly specify parameters):
train_dataset = get_dataset(
    dataset_name="pastis",  # Hardcode dataset name
    root="/kaggle/input/pastis-dataset-ddp/PASTIS",
    split="train",
    temporal_mode="mean"  # or "stack"
)

# Model
model = CATSeg(
    clip_model=config['model']['clip_model'],
    num_classes=config['model']['num_classes'],
    input_channels=config['model']['input_channels']
).cuda()

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])

# Training loop
for epoch in range(config['training']['epochs']):
    model.train()
    for images, masks in train_loader:
        images, masks = images.cuda(), masks.cuda()
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1}/{config['training']['epochs']} | Loss: {loss.item():.4f}")