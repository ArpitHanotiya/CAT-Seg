import torch
from datasets import get_dataset
from models.catseg import CATSeg

# Load config
config = load_yaml('configs/pastis.yaml')

# Dataset
train_dataset = get_dataset(
    name=config['dataset']['name'],
    root=config['dataset']['root'],
    split='train',
    temporal_mode=config['dataset']['temporal_mode']
)
train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'])

# Model
model = CATSeg(
    clip_model=config['model']['clip_model'],
    num_classes=config['model']['num_classes'],
    input_channels=config['model']['input_channels']
)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])

# Training loop
for epoch in range(config['training']['epochs']):
    for images, masks in train_loader:
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()