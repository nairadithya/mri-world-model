"""Clinical encoders: one MLP per annotation family -> shared latent.

Input clinical vector (from dataset): [sex, age/100, idh_id, mgmt_id, mgmt_q, surv/200]
Families:
  - demographics: sex (2-way embed) + age (scalar)
  - pathology:    idh (4-way embed) + mgmt qual (3-way embed) + mgmt quant (scalar)
  - context:      survival scalar + mgmt quant (prognosis context)

Each family -> hidden_dim (default 128); concatenated -> 3*hidden_dim = 384.
Missing values are pre-mapped to UNK ids / zeros in the dataset.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FamilyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClinicalEncoders(nn.Module):
    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        # small embeddings for categoricals
        self.sex_emb = nn.Embedding(2, 8)
        self.idh_emb = nn.Embedding(4, 8)
        self.mgmt_emb = nn.Embedding(3, 8)

        self.demo_mlp = FamilyMLP(8 + 1, hidden_dim, dropout)      # sex_emb + age
        self.path_mlp = FamilyMLP(8 + 8 + 1, hidden_dim, dropout)  # idh + mgmt + mgmt_q
        self.ctx_mlp = FamilyMLP(1 + 1, hidden_dim, dropout)       # surv + mgmt_q

    def forward(self, clinical: torch.Tensor) -> torch.Tensor:
        """clinical: (B, 6) -> (B, 3*hidden_dim)."""
        sex = self.sex_emb(clinical[:, 0].long().clamp(0, 1))
        age = clinical[:, 1:2]
        idh = self.idh_emb(clinical[:, 2].long().clamp(0, 3))
        mgmt = self.mgmt_emb(clinical[:, 3].long().clamp(0, 2))
        mgmt_q = clinical[:, 4:5]
        surv = clinical[:, 5:6]

        d = self.demo_mlp(torch.cat([sex, age], dim=-1))
        p = self.path_mlp(torch.cat([idh, mgmt, mgmt_q], dim=-1))
        c = self.ctx_mlp(torch.cat([surv, mgmt_q], dim=-1))
        return torch.cat([d, p, c], dim=-1)

    @property
    def out_dim(self) -> int:
        return 3 * self.hidden_dim
