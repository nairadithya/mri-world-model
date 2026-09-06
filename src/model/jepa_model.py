"""JEPAWorldModel: orchestrates backbone + clinical + fusion + temporal + predictor.

Batch layout from collate:
  mri:        (B, T, M, 1, D, H, W)   M = modalities
  mri_mask:   (B, T, M) bool
  visit_mask: (B, T) bool
  clinical:   (B, 6)
  actions:    (B, T) long (currently logged; conditioning hook for stage C+)
  time_deltas:(B, T) float days
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .brainiac import BrainiacEncoder
from .clinical import ClinicalEncoders
from .heads import RANOHeads
from .jepa import (
    HorizonPredictor,
    ImageProjector,
    Predictor,
    TargetEncoder,
    collapse_metrics,
    encode_chunked,
    jepa_loss,
)
from .temporal import Fusion, TemporalTransformer, TimeDeltaEncoding


class JEPAWorldModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        self.cfg = cfg
        self.backbone = BrainiacEncoder(
            ckpt_path=m["brainiac"].get("checkpoint"),
            lora_rank=m["brainiac"].get("lora_rank", 8),
            lora_alpha=m["brainiac"].get("lora_alpha", 16),
            lora_dropout=m["brainiac"].get("lora_dropout", 0.05),
            freeze_backbone=m["brainiac"].get("freeze_backbone", True),
        )
        self.clinical = ClinicalEncoders(
            hidden_dim=m["clinical"].get("hidden_dim", 128),
            dropout=m["clinical"].get("dropout", 0.1),
        )
        self.fusion = Fusion(
            vision_dim=m["fusion"].get("vision_dim", 768),
            clinical_dim=self.clinical.out_dim,
        )
        t = m["temporal"]
        self.temporal = TemporalTransformer(
            d_model=t.get("d_model", 1152),
            nhead=t.get("nhead", 8),
            num_layers=t.get("num_layers", 6),
            dim_feedforward=t.get("dim_feedforward", 2048),
            dropout=t.get("dropout", 0.1),
            max_days=t.get("max_time_delta_days", 3000.0),
        )
        p = m["predictor"]
        self.predictor = Predictor(
            in_dim=t.get("d_model", 1152),
            hidden_dim=p.get("hidden_dim", 1024),
            output_dim=p.get("output_dim", 768),
            dropout=p.get("dropout", 0.1),
        )
        # Multi-horizon head (frozen-probe-gated follow-up): predicts z_{t+n}
        # for every future visit n>=1 from state_t, conditioned on the day
        # gap. Built always (tiny); active only when
        # model.predictor.horizon.enabled is true, in which case the JEPA
        # loss below is the 1/n^power-weighted multi-horizon loss and the
        # plain 1-step predictor sits unused. Absent from pre-horizon
        # checkpoints, so resume uses strict=False (see run_train.py).
        hz_cfg = (p.get("horizon") or {})
        self.horizon_enabled = bool(hz_cfg.get("enabled", False))
        self.horizon_power = float(hz_cfg.get("weight_power", 1.0))
        self.horizon_gap_enc = TimeDeltaEncoding(
            t.get("d_model", 1152), t.get("max_time_delta_days", 3000.0))
        self.horizon_predictor = HorizonPredictor(
            state_dim=t.get("d_model", 1152),
            gap_dim=t.get("d_model", 1152),
            hidden_dim=p.get("hidden_dim", 1024),
            output_dim=p.get("output_dim", 768),
            dropout=p.get("dropout", 0.1),
        )
        self.projector = ImageProjector(768, m["target"].get("projection_dim", 768))
        self.target = TargetEncoder(
            self.backbone, self.projector,
            momentum=m["target"].get("ema_momentum", 0.996),
        )
        # RANO aux heads (D25): built always (tiny), active only when the
        # run enables aux loss. Random init — champion checkpoints predate
        # them, so resume uses strict=False (see run_train.py).
        self.rano_heads = RANOHeads(d_model=t.get("d_model", 1152))
        # Fail fast on inconsistent dims (fusion out must feed temporal;
        # predictor/target-projection dims must agree).
        assert self.fusion.out_dim == t.get("d_model", 1152), (
            f"fusion out {self.fusion.out_dim} != temporal d_model")
        assert p.get("output_dim", 768) == m["target"].get("projection_dim", 768), (
            "predictor output != target projection dim")
    def encode_visits(self, mri: torch.Tensor, mri_mask: torch.Tensor) -> torch.Tensor:
        """Per-visit vision latent: mean over available modalities. (B,T,768)."""
        B, T, M = mri.shape[:3]
        flat = mri.reshape(B * T * M, *mri.shape[3:])
        mask = mri_mask.reshape(B * T * M)
        lat = encode_chunked(self.backbone, flat, mask) if mask.any() \
            else torch.zeros(B * T * M, 768, device=mri.device, dtype=mri.dtype)
        lat = lat.view(B, T, M, 768)
        w = mri_mask.float().unsqueeze(-1)
        return (lat * w).sum(dim=2) / w.sum(dim=2).clamp_min(1e-9)

    @torch.no_grad()
    def encode_target_visit(self, mri: torch.Tensor, mri_mask: torch.Tensor) -> torch.Tensor:
        """EMA target latents for every visit: mean over modalities, then project."""
        B, T, M = mri.shape[:3]
        flat = mri.reshape(B * T * M, *mri.shape[3:])
        mask = mri_mask.reshape(B * T * M)
        lat = encode_chunked(self.target.backbone, flat, mask) if mask.any() \
            else torch.zeros(B * T * M, 768, device=mri.device)
        lat = lat.view(B, T, M, 768)
        w = mri_mask.float().unsqueeze(-1)
        vmean = (lat * w).sum(dim=2) / w.sum(dim=2).clamp_min(1e-9)  # (B,T,768)
        return self.target.projector(vmean.reshape(-1, 768)).view(B, T, -1)

    def _horizon_loss(self, states, time_deltas, visit_mask, has_img,
                      mri, mri_mask, z_hat_1, z_tgt_1, valid_1, loss_1):
        """Weighted multi-horizon JEPA loss: every state_t predicts every
        future visit z_{t+n}, conditioned on the day gap, weighted 1/n^power.

        Returns (loss, z_hat_1, z_tgt_1, valid_1, horizon_info) where the
        middle three are the n=1 slices (kept for aux heads and loggers) and
        horizon_info carries per-pair horizons + detached errors for
        per-horizon eval breakdowns (None when no valid pair exists).
        """
        B, Tm1, _ = states.shape
        T = Tm1 + 1
        cum = time_deltas.cumsum(dim=1)  # cum[u]-cum[t] = days t -> u
        with torch.no_grad():
            z_all = self.encode_target_visit(mri, mri_mask)  # (B,T,proj)
        b_idx, t_idx, u_idx = [], [], []
        for t in range(T - 1):
            ok = ((visit_mask[:, t] & has_img[:, t]).unsqueeze(1)
                  & visit_mask[:, t + 1:] & has_img[:, t + 1:])
            bb, uu = torch.nonzero(ok, as_tuple=True)
            if bb.numel() == 0:
                continue
            b_idx.append(bb)
            t_idx.append(torch.full_like(bb, t))
            u_idx.append(uu + t + 1)
        if not b_idx:  # degenerate batch guard (no valid pair)
            return states.sum() * 0.0, z_hat_1, z_tgt_1, valid_1, None
        b_idx = torch.cat(b_idx)
        t_idx = torch.cat(t_idx)
        u_idx = torch.cat(u_idx)
        n = (u_idx - t_idx).to(states.dtype)
        w = 1.0 / n.clamp_min(1).pow(self.horizon_power)
        w = w / w.sum().clamp_min(1e-12)
        gaps = cum[b_idx, u_idx] - cum[b_idx, t_idx]  # (P,) days
        gap_enc = self.horizon_gap_enc(gaps.unsqueeze(0)).squeeze(0)
        pred = self.horizon_predictor(states[b_idx, t_idx], gap_enc)
        tgt = z_all[b_idx, u_idx]
        err = 1 - (F.normalize(pred, dim=-1)
                   * F.normalize(tgt, dim=-1)).sum(dim=-1)
        loss = (w * err).sum()
        with torch.no_grad():
            metrics_t = collapse_metrics(tgt)
        info = {"n": (u_idx - t_idx).detach(),
                "err": err.detach(),
                "zt": z_all[b_idx, t_idx].detach(),
                "zu": tgt.detach(),
                "target_std": metrics_t["target_std"],
                "target_eff_rank": metrics_t["target_eff_rank"]}
        return loss, z_hat_1, z_tgt_1, valid_1, info

    def forward(self, batch: dict) -> dict:
        mri, mri_mask = batch["mri"], batch["mri_mask"]
        visit_mask = batch["visit_mask"]
        B, T = visit_mask.shape

        v = self.encode_visits(mri, mri_mask)                    # (B,T,768)
        c = self.clinical(batch["clinical"])                     # (B,384)
        tokens = self.fusion(v, c.unsqueeze(1).expand(-1, T, -1))  # (B,T,d)

        states, valid = self.temporal.forward_prefixes(
            tokens, batch["time_deltas"], visit_mask
        )                                                        # (B,T-1,d)
        # Belt-and-braces: both sides of a (t -> t+1) pair need real pixels.
        # (Dataset drops imageless visits; this covers load-time failures.)
        has_img = mri_mask.any(dim=2)                            # (B,T)
        valid = valid & has_img[:, :-1] & has_img[:, 1:]
        z_hat = self.predictor(states)                           # (B,T-1,proj)
        with torch.no_grad():
            z_tgt = self.encode_target_visit(mri, mri_mask)[:, 1:]  # (B,T-1,proj)

        flat_valid = valid.reshape(-1)
        loss = (
            jepa_loss(z_hat.reshape(-1, z_hat.shape[-1])[flat_valid],
                      z_tgt.reshape(-1, z_tgt.shape[-1])[flat_valid])
            if flat_valid.any()
            else z_hat.sum() * 0.0  # degenerate batch guard
        )
        horizon = None
        if self.horizon_enabled:
            loss, z_hat, z_tgt, valid, horizon = self._horizon_loss(
                states, batch["time_deltas"], visit_mask, has_img, mri, mri_mask,
                z_hat, z_tgt, valid, loss)
        # RANO aux (D25): forecast framing on clean-labelled valid pairs.
        # best-val tracks the TOTAL so best.pt follows the joint objective.
        aux_cfg = (self.cfg.get("aux") or {})
        aux = {"flat": loss * 0.0, "prog": loss * 0.0,
               "resp": loss * 0.0, "n_aux": 0}
        lam = aux_cfg.get("lambda", 0.0)
        if lam > 0 and "actions" in batch:
            cw = aux_cfg.get("class_weights")
            aux = self.rano_heads.aux_losses(
                states, batch["actions"], valid,
                {"class_weights": torch.tensor(cw) if cw else None,
                 "prog_pos_weight": aux_cfg.get("prog_pos_weight", 0.56),
                 "resp_pos_weight": aux_cfg.get("resp_pos_weight", 2.0)})
            loss = loss + lam * (aux["flat"] + aux["prog"] + aux["resp"])
        with torch.no_grad():
            metrics = collapse_metrics(z_tgt.reshape(-1, z_tgt.shape[-1])[flat_valid]) \
                if flat_valid.any() else {"target_std": 0.0, "target_eff_rank": 0.0}
        if horizon is not None:
            # Monitors over all valid horizon targets (stricter pool).
            metrics = {"target_std": horizon["target_std"],
                       "target_eff_rank": horizon["target_eff_rank"]}
        return {"loss": loss, "z_hat": z_hat, "z_target": z_tgt,
                "valid": valid, "aux": aux, "horizon": horizon, **metrics}

    def update_target(self) -> None:
        self.target.update_ema(self.backbone, self.projector)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
