"""Locked R37 spatial-history predictor and analytic frozen-head decoding."""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn
import torch.nn.functional as F


class SpatialPrediction(NamedTuple):
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    pooled_hidden: torch.Tensor


class SpatialDynamics(nn.Module):
    """Input maps are caller-standardized [B,T,128,D,H,W]."""
    channels = 128
    width = 16
    max_history = 4

    def __init__(self):
        super().__init__()
        self.reduce = nn.Conv3d(128, 16, 1)
        self.gates = nn.Conv3d(16 + 16 + 5, 2 * 16, 3, padding=1)
        self.candidate = nn.Conv3d(16 + 16 + 5, 16, 3, padding=1)
        self.residual = nn.Conv3d(16, 128, 1)
        self.scale = nn.Linear(16, 128)

    def forward(self, maps: torch.Tensor, availability: torch.Tensor,
                gap_log1p: torch.Tensor, history_mask: torch.Tensor, *,
                mode: str = "H") -> SpatialPrediction:
        if mode not in {"H", "L", "G"}:
            raise ValueError("mode must be H, L, or G")
        if maps.ndim != 6 or maps.shape[1] < 1 or maps.shape[1] > 4 or maps.shape[2] != 128:
            raise ValueError("maps must be [B,T<=4,128,D,H,W]")
        b, t, _, d, h, w = maps.shape
        if availability.shape != (b, t, 4) or gap_log1p.shape != (b, t) or history_mask.shape != (b, t):
            raise ValueError("history covariate shape mismatch")
        if not torch.isfinite(maps).all() or not torch.isfinite(availability).all() or not torch.isfinite(gap_log1p).all():
            raise ValueError("nonfinite history input")
        if not ((availability == 0) | (availability == 1)).all() or not ((history_mask == 0) | (history_mask == 1)).all():
            raise ValueError("availability and history mask must be binary")
        mask = history_mask.bool()
        if not mask[:, -1].all() or (mask[:, :-1] & ~mask[:, 1:]).any():
            raise ValueError("history must be right aligned with a present latest visit")
        if (gap_log1p[mask] < 0).any() or (gap_log1p[~mask] != 0).any():
            raise ValueError("gaps must be nonnegative and padded gaps zero")
        first = mask & ~F.pad(mask[:, :-1], (1, 0), value=False)
        if (gap_log1p[first] != 0).any() or (gap_log1p[mask & ~first] <= 0).any():
            raise ValueError("first retained gap must be zero and later retained gaps positive")
        current = maps[:, -1]
        xmaps = maps.clone()
        if mode != "H":
            earlier = xmaps[:, :-1]
            if mode == "L":
                xmaps[:, :-1] = 0
            else:
                pooled = earlier.mean(dim=(-3, -2, -1), keepdim=True)
                xmaps[:, :-1] = pooled.expand_as(earlier)
        hidden = maps.new_zeros((b, 16, d, h, w))
        for i in range(t):
            valid = mask[:, i].view(b, 1, 1, 1, 1)
            reduced = self.reduce(xmaps[:, i])
            cov = torch.cat((availability[:, i, :, None, None, None].expand(-1, -1, d, h, w),
                             gap_log1p[:, i, None, None, None, None].expand(-1, 1, d, h, w)), dim=1)
            reset_update = torch.sigmoid(self.gates(torch.cat((reduced, hidden, cov), dim=1)))
            reset, update = reset_update.chunk(2, dim=1)
            proposal = torch.tanh(self.candidate(torch.cat((reduced, reset * hidden, cov), dim=1)))
            next_hidden = (1 - update) * hidden + update * proposal
            hidden = torch.where(valid, next_hidden, torch.zeros_like(hidden))
        mu = current + self.residual(hidden)
        pooled = hidden.mean(dim=(-3, -2, -1))
        sigma = F.softplus(self.scale(pooled)) + 1e-6
        return SpatialPrediction(mu, sigma, pooled)


def gaussian_crps(mean: torch.Tensor, scale: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Elementwise exact Gaussian CRPS."""
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not torch.isfinite(target).all() or (scale <= 0).any():
        raise ValueError("CRPS inputs must be finite and scale positive")
    z = (target - mean) / scale
    phi = torch.exp(-0.5 * z.square()) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    return scale * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))


class FrozenAffineVolumeDecoder:
    """Frozen affine feature->log-volume decoder plus saved normalizers.

    Predicted feature means and scales are in standardized coordinates; the
    saved source feature mean/std invert that normalization. Saved target
    mean/std invert target normalization. Each channel epsilon is spatially shared.
    """
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor,
                 feature_mean: torch.Tensor, feature_std: torch.Tensor,
                 target_mean: torch.Tensor, target_std: torch.Tensor):
        if weight.ndim != 2 or bias.shape != (weight.shape[0],):
            raise ValueError("decoder weight/bias shape mismatch")
        if any(x.shape != (weight.shape[1],) for x in (feature_mean, feature_std)) or any(
                x.shape != (weight.shape[0],) for x in (target_mean, target_std)):
            raise ValueError("decoder normalization shape mismatch")
        stats = (weight, bias, feature_mean, feature_std, target_mean, target_std)
        if any(not torch.isfinite(x).all() for x in stats) or (feature_std <= 0).any() or (target_std <= 0).any():
            raise ValueError("decoder statistics must be finite with positive scales")
        self.weight = weight.detach().clone()
        self.bias = bias.detach().clone()
        self.feature_mean = feature_mean.detach().clone()
        self.feature_std = feature_std.detach().clone()
        self.target_mean = target_mean.detach().clone()
        self.target_std = target_std.detach().clone()

    def __call__(self, mu: torch.Tensor, sigma: torch.Tensor,
                 calibration: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        if calibration not in (0.5, 1.0, 2.0):
            raise ValueError("calibration must be selected from [0.5, 1, 2]")
        if mu.ndim != 5 or mu.shape[1] != self.weight.shape[1] or sigma.shape != (mu.shape[0], mu.shape[1]):
            raise ValueError("decoder input shape mismatch")
        if not torch.isfinite(mu).all() or not torch.isfinite(sigma).all() or (sigma <= 0).any():
            raise ValueError("decoder inputs must be finite with positive scales")
        feature_mean = mu.mean(dim=(-3, -2, -1)) * self.feature_std + self.feature_mean
        feature_sigma = sigma * self.feature_std
        raw_mean = F.linear(feature_mean, self.weight, self.bias)
        mean = raw_mean * self.target_std + self.target_mean
        effective = self.weight * self.target_std[:, None]
        variance = (effective.square().unsqueeze(0) * feature_sigma.square().unsqueeze(1)).sum(dim=2)
        scale = torch.sqrt(variance * calibration ** 2 + 0.05 ** 2)
        return mean, scale
