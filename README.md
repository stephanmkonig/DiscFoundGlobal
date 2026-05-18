[SSL_Model_README.md](https://github.com/user-attachments/files/27969577/SSL_Model_README.md)
# RETFound-Green SSL Backbone

A DINOv2 ViT-Small backbone further pre-trained on optic disc crops from fundus images via self-supervised token reconstruction.

Download `reconstruct_392_010eps_0.221448.pth` from [Releases](../../releases).

## Installation

```bash
pip install torch torchvision timm
```

## Loading the model

The checkpoint is a pickled `TokenReconstructor` training wrapper. You need the class defined before loading so Python can unpickle it, then extract the bare encoder from `.model`:

```python
import torch
import torch.nn as nn
from timm.models.vision_transformer import LayerScale


class TokenReconstructor(nn.Module):
    """Training wrapper — only needed to unpickle the checkpoint."""
    def __init__(self, original_encoder, model, corruption_ratio=1/3,
                 sample_corruption_ratio=True, project=True, last_sample_clean=True,
                 pixel_space_corruption=True, pixel_space_corruption_scale=0.2):
        super().__init__()
        self.original_encoder = original_encoder
        self.model = model
        self.corruption_ratio = corruption_ratio
        self.corruption_token = nn.Parameter(torch.zeros(1, 1, self.model.embed_dim))
        self.sample_corruption_ratio = sample_corruption_ratio
        self.last_sample_clean = last_sample_clean
        self.pixel_space_corruption = pixel_space_corruption
        self.pixel_space_corruption_scale = pixel_space_corruption_scale
        self.project = project
        if self.project:
            self.model.projector = nn.Sequential(
                nn.LayerNorm(self.model.embed_dim),
                nn.Linear(self.model.embed_dim, self.model.embed_dim),
                nn.GELU(),
                nn.Linear(self.model.embed_dim, self.model.embed_dim),
            )
            self.model.ls_projector = LayerScale(self.model.embed_dim, init_values=1e-5)


checkpoint = torch.load('reconstruct_392_010eps_0.221448.pth', map_location='cpu', weights_only=False)
encoder = checkpoint.model  # timm VisionTransformer, ready for inference
encoder.eval()
```

## Inference

Images must be resized to **392×392** and normalized with **mean=0.5, std=0.5**.

```python
import torch
from torchvision.transforms import v2 as T
from torchvision.io import read_image

transform = T.Compose([
    T.Resize((392, 392), antialias=True),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.5], std=[0.5]),
])

img = read_image('fundus.jpg')           # [C, H, W] uint8
img = transform(img).unsqueeze(0)        # [1, 3, 392, 392]

with torch.inference_mode():
    tokens = encoder.forward_features(img)   # [1, 789, 384]
```

`forward_features` returns 789 tokens: 1 CLS token, 4 register tokens, and 784 patch tokens (28×28 patches).

## Model details

| | |
|---|---|
| Architecture | ViT-Small (`vit_small_patch14_reg4_dinov2`) |
| Starting weights | DINOv2 pretrained on LVD-142M |
| Input size | 392 × 392 |
| Output | 789 tokens × 384 dims |
| Normalization | mean=0.5, std=0.5 |
| Pretraining task | Token reconstruction (masked patch prediction in feature space) |
