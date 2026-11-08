from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from vendor.generator import DitGen

FEATURE_BATCH = 4


def build_generator(device):
    return DitGen(cond_dim=400, num_classes=10, noise_classes=64, noise_coords=32,
                  input_size=32, in_channels=3, n_cls_tokens=16, patch_size=4,
                  hidden_size=400, depth=9, num_heads=8, mlp_ratio=4, out_channels=3,
                  use_qknorm=True, use_swiglu=True, use_rope=True, use_rmsnorm=True,
                  use_bf16=device.type == 'cuda', attn_fp32=False, use_remat=False).to(device)


class DinoFeatures(nn.Module):
    def __init__(self, weights, layers, device):
        super().__init__()
        import timm
        from safetensors.torch import load_file
        if not Path(weights).is_file():
            raise FileNotFoundError(weights)
        self.device = device
        self.layers = [layer-1 for layer in layers]
        self.backbone = timm.create_model('vit_base_patch16_dinov3', pretrained=False,
                                          num_classes=0, dynamic_img_size=True)
        self.backbone.load_state_dict(load_file(str(weights)), strict=True)
        self.backbone.eval().requires_grad_(False)
        self.backbone.set_grad_checkpointing(False)
        self.amp = device.type == 'cuda'
        if self.amp:
            self.backbone.bfloat16()
        self.to(device)
        data = timm.data.resolve_model_data_config(self.backbone)
        self.register_buffer('mean', torch.tensor(data['mean'], device=device).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(data['std'], device=device).reshape(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(False)
        self.backbone.eval()
        return self

    def forward(self, images):
        x = (images.permute(0, 3, 1, 2).float()+1)*.5
        x = F.interpolate(x, size=(256, 256), mode='bicubic', align_corners=False)
        x = (x-self.mean)/self.std
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
            layers = self.backbone.forward_intermediates(
                x, indices=self.layers, return_prefix_tokens=True, norm=True,
                stop_early=False, output_fmt='NLC', intermediates_only=True)
        result = {}
        for layer, (spatial, prefix) in zip(self.layers, layers):
            if spatial.shape[1:] != (256, 768):
                raise ValueError('DINO patch feature shape must be [batch,256,768]')
            result[f'layer_{layer+1}_patch'] = spatial
            result[f'layer_{layer+1}_cls'] = prefix[:, :1]
        return result


@torch.no_grad()
def extract(encoder, images):
    result = {}
    for first in range(0, len(images), FEATURE_BATCH):
        chunk = images[first:first+FEATURE_BATCH]
        for key, value in encoder(chunk).items():
            if key not in result:
                result[key] = value.new_empty((len(images), *value.shape[1:]))
            result[key][first:first+len(chunk)].copy_(value)
    return result


def image_gradient(encoder, images, cotangents):
    result = torch.empty_like(images, dtype=torch.float32)
    keys = sorted(cotangents)
    for first in range(0, len(images), FEATURE_BATCH):
        x = images[first:first+FEATURE_BATCH].detach().requires_grad_(True)
        output = encoder(x)
        grad = tuple(cotangents[key][first:first+len(x)].to(output[key].dtype) for key in keys)
        result[first:first+len(x)] = torch.autograd.grad(tuple(output[key] for key in keys), x,
                                                       grad_outputs=grad)[0]
    return result
