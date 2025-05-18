import clip
import torch
from clip.model import VisionTransformer

class CustomCLIP(VisionTransformer):
    def __init__(self, input_channels=10, 
                 embed_dim=768, 
                 patch_size=16, 
                 output_dim=512,
                 **kwargs):
        super().__init__(
            input_resolution=224,
            patch_size=patch_size,
            width=embed_dim,
            layers=12,
            heads=12,
            output_dim=output_dim,
            **kwargs
        )
        self.embed_dim = embed_dim  # Explicitly store embed_dim
        self.conv1 = torch.nn.Conv2d(
            input_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size, 
            bias=False
        )

def build_clip_adapter(input_channels=10, clip_model="ViT-B/16"):
    model, _ = clip.load(clip_model)
    embed_dim = model.visual.conv1.out_channels
    patch_size = model.visual.conv1.kernel_size[0]
    output_dim = model.visual.proj.shape[1]
    
    return CustomCLIP(
        input_channels=input_channels,
        embed_dim=embed_dim,  # Pass to CustomCLIP
        patch_size=patch_size,
        output_dim=output_dim
    )