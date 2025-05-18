import torch
import clip

class CustomCLIP(clip.VisionTransformer):
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
        
        # Modified first convolutional layer
        self.conv1 = torch.nn.Conv2d(
            in_channels=input_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )

    def forward(self, x: torch.Tensor):
        return super().forward(x)

def build_clip_adapter(input_channels=10, clip_model="ViT-B/16"):
    model, _ = clip.load(clip_model)
    return CustomCLIP(
        input_channels=input_channels,
        embed_dim=model.visual.embed_dim,
        patch_size=model.visual.patch_size
    )