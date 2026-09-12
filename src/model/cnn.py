"""Supervised 3D CNN comparator (Tikhonov/Matoso family) for cross-site
adaptability vs the JEPA encoder.

MONAI's 3D ResNet-18 (torchvision is banned by D16; MONAI ships the same
architecture). Input is a consecutive-visit pair: 4 modalities at visit t and
visit t+1 stacked as 8 channels, mirroring the field's pair framing. The
penultimate avgpool features (512-d) are the representation used for the
SAILOR few-shot adaptation comparison.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SupervisedCNN(nn.Module):
    def __init__(self, n_input_channels: int = 8, num_classes: int = 4,
                 layers: tuple[int, ...] = (2, 2, 2, 2)):
        super().__init__()
        from monai.networks.nets import ResNet

        self.net = ResNet(
            block="basic", layers=list(layers),
            block_inplanes=[64, 128, 256, 512], spatial_dims=3,
            n_input_channels=n_input_channels, num_classes=num_classes,
        )
        self.feat_dim = 512  # block_inplanes[-1] with widen_factor=1

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate pooled features (B, 512)."""
        h = self.net.conv1(x)
        h = self.net.maxpool(h)
        h = self.net.layer1(h)
        h = self.net.layer2(h)
        h = self.net.layer3(h)
        h = self.net.layer4(h)
        h = self.net.avgpool(h)
        return h.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def pair_channels(mri: torch.Tensor, t: int) -> torch.Tensor:
    """Stack modality channels at visits t and t+1 -> (B, 2*M, D, H, W).

    mri: (B, T, M, 1, D, H, W); missing modalities contribute zeros.
    """
    B, T, M = mri.shape[:3]
    v0 = mri[:, t].reshape(B, M, *mri.shape[4:])
    v1 = mri[:, t + 1].reshape(B, M, *mri.shape[4:])
    return torch.cat([v0, v1], dim=1)


def pair_present(mri_mask: torch.Tensor, t: int) -> torch.Tensor:
    """Per-batch-element bool: both visits t and t+1 have >=1 modality."""
    return mri_mask[:, t].any(dim=1) & mri_mask[:, t + 1].any(dim=1)


class SupervisedCNN2D(nn.Module):
    """2D axial-slice ResNet-18 (Matoso-style): many more samples than 3D."""

    def __init__(self, n_input_channels: int = 8, num_classes: int = 4,
                 layers: tuple[int, ...] = (2, 2, 2, 2)):
        super().__init__()
        from monai.networks.nets import ResNet

        self.net = ResNet(
            block="basic", layers=list(layers),
            block_inplanes=[64, 128, 256, 512], spatial_dims=2,
            n_input_channels=n_input_channels, num_classes=num_classes,
        )
        self.feat_dim = 512

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net.conv1(x)
        h = self.net.maxpool(h)
        h = self.net.layer1(h)
        h = self.net.layer2(h)
        h = self.net.layer3(h)
        h = self.net.layer4(h)
        return self.net.avgpool(h).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def pair_slices(mri: torch.Tensor, t: int, min_fg: int = 64) -> torch.Tensor:
    """Stacked axial slices of a visit pair -> (S, 2*M, H, W).

    mri: (1, T, M, 1, D, H, W). Slices where either visit is effectively empty
    (foreground voxels < min_fg) are dropped."""
    v0 = mri[0, t, :, 0]        # (M, D, H, W)
    v1 = mri[0, t + 1, :, 0]
    D = v0.shape[1]
    keep = []
    for z in range(D):
        a, b = v0[:, z], v1[:, z]
        if int((a != 0).sum()) < min_fg and int((b != 0).sum()) < min_fg:
            continue
        keep.append(torch.cat([a, b], dim=0))  # (2M, H, W)
    if not keep:
        return torch.empty(0, 2 * v0.shape[0], *v0.shape[-2:])
    return torch.stack(keep)


def roi_pair_slices(v0: torch.Tensor, v1: torch.Tensor, mask: torch.Tensor,
                    crop: int = 64) -> torch.Tensor:
    """Tumor-centred ROI slices for a visit pair -> (S, 2*M, crop, crop).

    v0/v1: (M, D, H, W) volumes; mask: (D, H, W) bool union of the two visits'
    tumor masks. A fixed `crop`-cube is centred on the mask centroid; only
    slices containing tumor are returned. Removes the whole-brain label noise
    (every background slice carrying the visit's label) that collapsed A30.
    """
    idx = torch.nonzero(mask, as_tuple=False)
    if idx.numel() == 0:
        return torch.empty(0)
    c = idx.float().mean(0)
    D, H, W = mask.shape
    boxes = []
    for s in range(3):
        dim = (D, H, W)[s]
        half = crop // 2
        lo = int(round(float(c[s]))) - half
        lo = max(0, min(lo, max(0, dim - crop)))
        boxes.append((lo, min(dim, lo + crop)))
    (z0, z1), (y0, y1), (x0, x1) = boxes
    mv = mask[z0:z1, y0:y1, x0:x1]
    a = v0[:, z0:z1, y0:y1, x0:x1]
    b = v1[:, z0:z1, y0:y1, x0:x1]
    sl = [torch.cat([a[:, z], b[:, z]], dim=0)
          for z in range(mv.shape[0]) if bool(mv[z].any())]
    return torch.stack(sl) if sl else torch.empty(0)
