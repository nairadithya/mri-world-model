"""Mandatory baselines the JEPA model must beat (PROPOSAL §6).

All baselines operate in the same target projection space so losses are
directly comparable:
  1. Persistence: last visit's projected latent = prediction.
  2. GRU dynamics: GRU over fused tokens -> linear head.
  3. Clinical-only: MLP on static clinical vector -> prediction.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.model.jepa import ImageProjector, encode_chunked, jepa_loss
from src.model.temporal import Fusion


class PersistenceBaseline(nn.Module):
    """Predict z_{t+1} = proj(latent of visit t). No learnable dynamics."""

    def __init__(self, projector: ImageProjector):
        super().__init__()
        self.projector = projector

    @torch.no_grad()
    def forward(self, batch: dict, encode_visits_fn) -> dict:
        v = encode_visits_fn(batch["mri"], batch["mri_mask"])  # (B,T,768)
        z = self.projector(v.reshape(-1, 768)).view(*v.shape[:2], -1)
        z_hat, z_tgt = z[:, :-1], z[:, 1:]
        has_img = batch["mri_mask"].any(dim=2)
        valid = batch["visit_mask"][:, :-1] & batch["visit_mask"][:, 1:] \
            & has_img[:, :-1] & has_img[:, 1:]
        fv = valid.reshape(-1)
        loss = jepa_loss(z_hat.reshape(-1, z_hat.shape[-1])[fv],
                         z_tgt.reshape(-1, z_tgt.shape[-1])[fv]) if fv.any() else z.sum() * 0.0
        return {"loss": loss}


def gru_baseline_loss(model: "GRUDynamics", batch: dict) -> torch.Tensor:
    out = model(batch)
    return out["loss"]


class GRUDynamics(nn.Module):
    """GRU over fused (vision-mean + clinical) tokens -> linear predictor head."""

    def __init__(self, backbone: nn.Module, clinical: nn.Module, fusion: Fusion,
                 d_model: int = 1152, proj_dim: int = 768):
        super().__init__()
        self.backbone = backbone
        self.clinical = clinical
        self.fusion = fusion
        self.gru = nn.GRU(d_model, d_model, num_layers=2, batch_first=True)
        self.head = nn.Linear(d_model, proj_dim)
        self.projector = ImageProjector(768, proj_dim)

    def encode_visits(self, mri, mri_mask):
        B, T, M = mri.shape[:3]
        flat = mri.reshape(B * T * M, *mri.shape[3:])
        mask = mri_mask.reshape(B * T * M)
        with torch.no_grad():
            lat = encode_chunked(self.backbone, flat, mask) if mask.any() \
                else torch.zeros(B * T * M, 768, device=mri.device)
        lat = lat.view(B, T, M, 768)
        w = mri_mask.float().unsqueeze(-1)
        return (lat * w).sum(2) / w.sum(2).clamp_min(1e-9)

    def forward(self, batch: dict) -> dict:
        mri, mri_mask, visit_mask = batch["mri"], batch["mri_mask"], batch["visit_mask"]
        B, T = visit_mask.shape
        v = self.encode_visits(mri, mri_mask)
        c = self.clinical(batch["clinical"])
        tokens = self.fusion(v, c.unsqueeze(1).expand(-1, T, -1))
        h, _ = self.gru(tokens)
        z_hat = self.head(h[:, :-1])
        with torch.no_grad():
            B2, T2, M = mri.shape[:3]
            tgt = self.projector(
                self.encode_visits(mri, mri_mask).reshape(-1, 768)
            ).view(B2, T2, -1)[:, 1:]
        has_img = mri_mask.any(dim=2)
        valid = visit_mask[:, :-1] & visit_mask[:, 1:] & has_img[:, :-1] & has_img[:, 1:]
        fv = valid.reshape(-1)
        loss = jepa_loss(z_hat.reshape(-1, z_hat.shape[-1])[fv],
                         tgt.reshape(-1, tgt.shape[-1])[fv]) if fv.any() else z_hat.sum() * 0.0
        return {"loss": loss}


class ClinicalOnlyForecaster(nn.Module):
    """MLP on static clinical vector -> next-visit latent. No imaging."""

    def __init__(self, in_dim: int = 6, hidden: int = 256, proj_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, proj_dim),
        )

    def forward(self, batch: dict, target_fn) -> dict:
        B, T = batch["visit_mask"].shape
        z_hat = self.net(batch["clinical"]).unsqueeze(1).expand(-1, T - 1, -1)
        with torch.no_grad():
            tgt = target_fn(batch)[:, 1:]
        has_img = batch["mri_mask"].any(dim=2)
        valid = batch["visit_mask"][:, :-1] & batch["visit_mask"][:, 1:] \
            & has_img[:, :-1] & has_img[:, 1:]
        fv = valid.reshape(-1)
        loss = jepa_loss(z_hat.reshape(-1, z_hat.shape[-1])[fv],
                         tgt.reshape(-1, tgt.shape[-1])[fv]) if fv.any() else z_hat.sum() * 0.0
        return {"loss": loss}
