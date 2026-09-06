"""Time-continuous dynamics: velocity field integrated across true gaps.

Instead of emitting a fixed-scale next-state delta per step, the model learns
a velocity field over frozen target-space latents and integrates it across
the actual day gap. The natural fixed point is v ~= 0 (persistence): change
must be earned with evidence, not emitted by default.

Field inputs per pair (t -> u): running latent z_k, history summary h_t,
treatment-phase embedding from the visit-t action id, static clinical vector.
A patient tempo scalar stretches/compresses time per individual.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.functional import softplus


class VelocityField(nn.Module):
    """[z (proj), h (d_model), phase (phase_dim), clinical (clin_dim)] -> v."""

    def __init__(self, proj_dim: int = 768, state_dim: int = 1152,
                 n_actions: int = 6, phase_dim: int = 32,
                 clin_dim: int = 384, hidden_dim: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.phase_emb = nn.Embedding(n_actions, phase_dim)
        in_dim = proj_dim + state_dim + phase_dim + clin_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, z: torch.Tensor, h: torch.Tensor,
                phase: torch.Tensor, clin: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, h, self.phase_emb(phase), clin], dim=-1))


class PatientTempo(nn.Module):
    """Positive per-patient time scalar from the static clinical vector."""

    def __init__(self, clin_dim: int = 384):
        super().__init__()
        self.net = nn.Linear(clin_dim, 1)

    def forward(self, clin: torch.Tensor) -> torch.Tensor:
        return softplus(self.net(clin)).squeeze(-1) + 1e-3  # (B,) > 0


def integrate(field: VelocityField, tempo: torch.Tensor, z0: torch.Tensor,
              h: torch.Tensor, phase: torch.Tensor, clin: torch.Tensor,
              gaps: torch.Tensor, steps: int = 3) -> torch.Tensor:
    """Euler-integrate dz/dt = tempo * field(z, h, phase, clin) over gaps.

    All flat (P, ...); gaps in days. Returns predicted endpoints (P, proj).
    """
    z = z0
    dt = (gaps / max(steps, 1)).unsqueeze(-1)
    for _ in range(max(steps, 1)):
        z = z + dt * tempo.unsqueeze(-1) * field(z, h, phase, clin)
    return z
