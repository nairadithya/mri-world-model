"""Prediction heads used by representation-error evaluation.

- ``load_1step_predictor``: the champion's own MLP (state_t -> z_{t+1}).
- ``GapHeadPredictor`` + ``load_gap_head``: the horizon-conditioned probe from
  ``checkpoints/probe_head.pt`` (state_t + log-gap -> z_{t+n}); architecture
  matches ``scripts/horizon_probe.HorizonPredictor`` so existing checkpoints
  load unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ...model.jepa import Predictor


class GapHeadPredictor(nn.Module):
    """[state_t (1152), standardized log-gap (1)] -> z_{t+n} (768)."""

    def __init__(self, hidden: int = 1024, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        blocks = []
        in_dim = 1152 + 1
        for _ in range(layers):
            blocks += [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden),
                       nn.GELU(), nn.Dropout(dropout)]
            in_dim = hidden
        blocks.append(nn.Linear(hidden, 768))
        self.net = nn.Sequential(*blocks)

    def forward(self, s: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, g.unsqueeze(-1)], dim=-1))


def load_1step_predictor(champion_path: str) -> Predictor:
    ckpt = torch.load(champion_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    net = Predictor()
    net.load_state_dict({k.replace("predictor.", ""): v for k, v in sd.items()
                         if k.startswith("predictor.")})
    net.eval()
    return net


def load_gap_head(path: str):
    """Return ``(net, mu, sd)``; mu/sd standardize log1p-gap (train stats)."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    net = GapHeadPredictor(hidden=saved.get("hidden", 1024),
                           layers=saved.get("layers", 2))
    net.load_state_dict(saved["net"])
    net.eval()
    return net, saved["mu"], saved["sd"]
