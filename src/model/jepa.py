"""JEPA head components: predictor, EMA target encoder, loss + health metrics."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Max volumes per backbone forward. A full batch can hold B*T*M volumes
# (e.g. 4*14*4=224); chunking keeps ViT-B peak memory flat. Grad flows fine
# through chunked index-assigns.
ENCODE_CHUNK = 8


def encode_chunked(backbone: nn.Module, flat: torch.Tensor,
                   mask: torch.Tensor, out_dim: int = 768,
                   chunk: int = ENCODE_CHUNK) -> torch.Tensor:
    """Run backbone over masked rows in chunks. Returns (N, out_dim)."""
    lat = torch.zeros(flat.shape[0], out_dim, device=flat.device, dtype=flat.dtype)
    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
    for s in range(0, idx.numel(), chunk):
        sub = idx[s:s + chunk]
        inp = flat[sub].float()
        # Checkpoint the grad path (online encoder): ViT activations for
        # every volume of a long-history patient otherwise sit alive until
        # backward and blow the 16 GB budget (D21). Recomputed at backward
        # with identical math (RNG preserved). The no-grad target path
        # keeps the plain call (nothing retained anyway).
        v = checkpoint(backbone, inp, use_reentrant=False) \
            if torch.is_grad_enabled() else backbone(inp)
        if isinstance(v, (tuple, list)):
            v = v[0]
            v = v[:, 0] if v.dim() == 3 else v
        lat[sub] = v.to(lat.dtype)
    return lat


class Predictor(nn.Module):
    """MLP: temporal state (d_model) -> target projection space (output_dim)."""

    def __init__(self, in_dim: int = 1152, hidden_dim: int = 1024,
                 output_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class ImageProjector(nn.Module):
    """Online/target projection head: vision latent 768 -> projection space."""

    def __init__(self, in_dim: int = 768, out_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return self.net(v)


class TargetEncoder(nn.Module):
    """EMA twin of (backbone -> image projector). Image-only, stop-grad.

    Owns a deep copy of the online backbone; call update_ema() after each step.
    """

    def __init__(self, online_backbone: nn.Module, projector: ImageProjector,
                 momentum: float = 0.996):
        super().__init__()
        self.momentum = momentum
        self.backbone = copy.deepcopy(online_backbone)
        self.projector = copy.deepcopy(projector)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single-image (or flat-batch) target encoding: backbone -> project."""
        v = self.backbone(x)
        if isinstance(v, (tuple, list)):
            v = v[0]
            v = v[:, 0] if v.dim() == 3 else v
        return self.projector(v)

    @torch.no_grad()
    def update_ema(self, online_backbone: nn.Module, online_projector: nn.Module) -> None:
        for tp, op in zip(self.backbone.parameters(), online_backbone.parameters()):
            tp.data.mul_(self.momentum).add_(op.data, alpha=1 - self.momentum)
        for tp, op in zip(self.projector.parameters(), online_projector.parameters()):
            tp.data.mul_(self.momentum).add_(op.data, alpha=1 - self.momentum)


def jepa_loss(z_hat: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity, averaged. Both (N, D)."""
    z_hat = F.normalize(z_hat, dim=-1)
    z_target = F.normalize(z_target, dim=-1)
    return (1 - (z_hat * z_target).sum(dim=-1)).mean()


@torch.no_grad()
def collapse_metrics(z: torch.Tensor) -> dict[str, float]:
    """Training-health monitors: per-dim std and effective rank of targets."""
    if z.numel() == 0:
        return {"target_std": 0.0, "target_eff_rank": 0.0}
    std = z.std(dim=0).mean().item() if z.shape[0] > 1 else 0.0
    _, s, _ = torch.linalg.svd(z.float(), full_matrices=False)
    p = s / s.sum().clamp_min(1e-12)
    eff_rank = torch.exp(-(p * (p + 1e-12).log()).sum()).item()
    return {"target_std": std, "target_eff_rank": eff_rank}
