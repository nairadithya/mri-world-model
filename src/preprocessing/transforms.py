"""Runtime (in-memory) transforms for DataLoader-time preprocessing."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def load_nifti_tensor(path: str) -> torch.Tensor:
    """Load a NIfTI file as a (1, D, H, W) float32 tensor."""
    import nibabel as nib

    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    # nibabel returns (X, Y, Z); add channel dim -> (1, X, Y, Z)
    return torch.from_numpy(data).unsqueeze(0)


def resize_to(tensor: torch.Tensor, size: tuple[int, int, int] = (96, 96, 96)) -> torch.Tensor:
    """Trilinear resize of a (C, D, H, W) tensor to target size."""
    if tuple(tensor.shape[1:]) == size:
        return tensor
    return F.interpolate(tensor.unsqueeze(0), size=size, mode="trilinear", align_corners=False).squeeze(0)


def zscore_nonzero(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Z-score normalize over nonzero voxels only (BRAINIAC contract)."""
    mask = tensor != 0
    if mask.sum() == 0:
        return tensor
    vals = tensor[mask]
    mean = vals.mean()
    std = vals.std().clamp_min(eps)
    out = torch.zeros_like(tensor)
    out[mask] = (tensor[mask] - mean) / std
    return out


def runtime_transform(path: str, size: tuple[int, int, int] = (96, 96, 96)) -> torch.Tensor:
    """Full lightweight path: load -> resize -> z-score. Returns (1, D, H, W)."""
    t = load_nifti_tensor(path)
    t = resize_to(t, size)
    return zscore_nonzero(t)
