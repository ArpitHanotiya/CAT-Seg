import torch
import clip
from clip.model import VisionTransformer  # Import from internal module

class CustomCLIP(VisionTransformer):  # Inherit from VisionTransformer
    def __init__(self, input_channels=10, 
                 embed_dim=768, 
                 patch_size=16,
                 **kwargs):
        super().__init__(
            input_resolution=224,  # CLIP default
            patch_size=patch_size,
            width=embed_dim,
            layers=12,
            heads=12,
            **kwargs
        )
        
        # Replace first conv layer to accept 10 channels
        self.conv1 = torch.nn.Conv2d(
            in_channels=input_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )

def build_clip_adapter(input_channels=10, clip_model="ViT-B/16"):
    # Load original CLIP model
    model, _ = clip.load(clip_model)
    
    # Extract VisionTransformer parameters
    embed_dim = model.visual.conv1.out_channels
    patch_size = model.visual.conv1.kernel_size[0]
    
    # Create modified CLIP
    return CustomCLIP(
        input_channels=input_channels,
        embed_dim=embed_dim,
        patch_size=patch_size
    )