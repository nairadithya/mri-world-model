"""Lesion-centred observation primitives for P2.3.

The crop zooms the union lesion before the frozen ViT, then pools its patch
tokens separately by compartment and immediate peritumoral context.  Masks
are inputs used for localization, never future targets.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

REGIONS = ("necrotic_non_enhancing", "enhancing", "edema_flair",
           "peritumoral_ring")


def lesion_centered_crop(images: torch.Tensor, masks: torch.Tensor,
                         crop_size: int = 64, output_size: int = 96
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop around current union lesion and resize images/masks to ViT input.

    ``images`` is ``(B,M,D,H,W)`` and mutually compatible ``masks`` is
    ``(B,3,D,H,W)``. Empty masks fall back to a centre crop.
    """
    if images.ndim != 5 or masks.ndim != 5 or images.shape[0] != masks.shape[0]:
        raise ValueError("expected images (B,M,D,H,W) and masks (B,3,D,H,W)")
    if images.shape[-3:] != masks.shape[-3:]:
        raise ValueError("image and mask grids must match")
    spatial = images.shape[-3:]
    if crop_size > min(spatial):
        raise ValueError("crop_size exceeds the input grid")
    image_crops, mask_crops = [], []
    for image, mask in zip(images, masks):
        index = torch.nonzero(mask.any(0), as_tuple=False)
        centre = (index.float().mean(0) if index.numel()
                  else image.new_tensor([(value - 1) / 2 for value in spatial]))
        starts = []
        for coordinate, extent in zip(centre, spatial):
            start = int(round(float(coordinate))) - crop_size // 2
            starts.append(max(0, min(start, extent - crop_size)))
        z, y, x = starts
        image_crops.append(image[:, z:z + crop_size, y:y + crop_size,
                                 x:x + crop_size])
        mask_crops.append(mask[:, z:z + crop_size, y:y + crop_size,
                               x:x + crop_size])
    image_batch = torch.stack(image_crops)
    mask_batch = torch.stack(mask_crops).float()
    image_batch = F.interpolate(image_batch, size=[output_size] * 3,
                                mode="trilinear", align_corners=False)
    mask_batch = F.interpolate(mask_batch, size=[output_size] * 3,
                               mode="nearest") > 0
    return image_batch, mask_batch


def region_patch_weights(masks: torch.Tensor, patch_size: int = 16
                         ) -> torch.Tensor:
    """Return compartment + adjacent-ring weights ``(B,4,P)``."""
    if masks.ndim != 5 or masks.shape[1] != 3:
        raise ValueError("expected masks (B,3,D,H,W)")
    shape = masks.shape[-3:]
    if any(value % patch_size for value in shape):
        raise ValueError("mask dimensions must be divisible by patch_size")
    weights = F.avg_pool3d(masks.float(), kernel_size=patch_size)
    occupied = weights.amax(1, keepdim=True) > 0
    expanded = F.max_pool3d(occupied.float(), kernel_size=3, stride=1,
                            padding=1) > 0
    ring = (expanded & ~occupied).float()
    return torch.cat([weights, ring], dim=1).flatten(2)


def pool_region_tokens(tokens: torch.Tensor, weights: torch.Tensor
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool ``(B,M,P,F)`` tokens into region and lesion-minus-ring features."""
    if tokens.ndim != 4 or weights.ndim != 3:
        raise ValueError("expected tokens (B,M,P,F) and weights (B,R,P)")
    if tokens.shape[0] != weights.shape[0] or tokens.shape[2] != weights.shape[2]:
        raise ValueError("token and weight batches/patch counts must match")
    weight = weights[:, None, :, :, None]
    pooled = (tokens[:, :, None] * weight).sum(3) / weight.sum(3).clamp_min(1e-9)
    contrast = pooled[:, :, :3] - pooled[:, :, 3:4]
    return pooled, contrast
