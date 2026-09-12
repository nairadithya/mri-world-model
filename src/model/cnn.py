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
