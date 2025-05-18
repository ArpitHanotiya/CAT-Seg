import clip
import torch
import torch.nn as nn
from .clip_adapter import build_clip_adapter
from datasets import CLASS_NAMES

class CATSeg(nn.Module):
    def __init__(self, 
                 num_classes=19, 
                 clip_model="ViT-B/16",
                 input_channels=10):
        super().__init__()
        
        # CLIP Adapter for 10-channel input
        self.clip_adapter = build_clip_adapter(
            input_channels=input_channels,
            clip_model=clip_model
        )
        
        # Load CLIP model and freeze parameters
        self.clip_model, _ = clip.load(clip_model)
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Segmentation Head
        self.head = nn.Conv2d(
            in_channels=self.clip_adapter.embed_dim,
            out_channels=num_classes,
            kernel_size=1
        )
        
    def compute_cosine_similarity(self, image_embeds, text_embeds):
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return torch.einsum('bchw,nc->bnhw', image_embeds, text_embeds)
        
    def forward(self, x):
        # Image embeddings
        image_embeddings = self.clip_adapter(x)  # [B, embed_dim, h, w]
        
        # Text embeddings (tokenize CLASS_NAMES)
        text_tokens = clip.tokenize(CLASS_NAMES).to(x.device)  # Tokenize text
        text_embeddings = self.clip_model.encode_text(text_tokens)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        # Cost volume
        cost_volume = self.compute_cosine_similarity(image_embeddings, text_embeddings)
        
        # Segmentation logits
        logits = self.head(cost_volume)
        return logits