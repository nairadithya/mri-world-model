"""Fusion (LayerNorm + concat) and temporal transformer with time-delta encodings."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class Fusion(nn.Module):
    """LayerNorm each branch, concatenate: (B,768)+(B,C) -> (B,768+C)."""

    def __init__(self, vision_dim: int = 768, clinical_dim: int = 384):
        super().__init__()
        self.ln_v = nn.LayerNorm(vision_dim)
        self.ln_c = nn.LayerNorm(clinical_dim)
        self.out_dim = vision_dim + clinical_dim

    def forward(self, vision: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.ln_v(vision), self.ln_c(clinical)], dim=-1)


class TimeDeltaEncoding(nn.Module):
    """Sinusoidal encoding of days-since-previous-visit + learned residual.

    Maps scalar delta_d (days) -> (d_model,) vector added to the token.
    """

    def __init__(self, d_model: int, max_days: float = 3000.0):
        super().__init__()
        self.d_model = d_model
        self.max_days = max_days
        self.residual = nn.Sequential(
            nn.Linear(1, d_model // 4), nn.GELU(), nn.Linear(d_model // 4, d_model)
        )

    def forward(self, deltas: torch.Tensor) -> torch.Tensor:
        """deltas: (B, T) days -> (B, T, d_model)."""
        x = deltas.clamp_min(0).unsqueeze(-1) / self.max_days  # (B,T,1) in [0,1]
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=deltas.device) / half
        )
        angles = x * freqs * self.max_days / 30.0  # scale: ~months carry the signal
        sinusoid = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if sinusoid.shape[-1] != self.d_model:  # odd d_model guard
            sinusoid = sinusoid[..., : self.d_model]
        return sinusoid + 0.1 * self.residual(x)


class TemporalTransformer(nn.Module):
    """TransformerEncoder over fused visit tokens. Returns last-real-visit state."""

    def __init__(
        self,
        d_model: int = 1152,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_days: float = 3000.0,
    ):
        super().__init__()
        self.time_enc = TimeDeltaEncoding(d_model, max_days)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,      # (B, T, d_model) fused visit tokens
        time_deltas: torch.Tensor,  # (B, T) days
        visit_mask: torch.Tensor,   # (B, T) bool, True = real visit
    ) -> torch.Tensor:
        h = tokens + self.time_enc(time_deltas)
        h = self.encoder(h, src_key_padding_mask=~visit_mask)
        h = self.norm(h)
        # gather last real visit per patient
        lengths = visit_mask.sum(dim=1).clamp_min(1) - 1  # (B,)
        idx = lengths.view(-1, 1, 1).expand(-1, 1, h.shape[-1])
        return h.gather(1, idx).squeeze(1)  # (B, d_model)

    def _encode_prefix(
        self,
        tok_slice: torch.Tensor,      # (B, t+1, d) fused tokens
        dt_slice: torch.Tensor,       # (B, t+1) days
        mask_slice: torch.Tensor,     # (B, t+1) bool
    ) -> torch.Tensor:
        """Single-prefix encode, factored out for gradient checkpointing."""
        h = tok_slice + self.time_enc(dt_slice)
        h = self.encoder(h, src_key_padding_mask=~mask_slice)
        return self.norm(h)

    def forward_prefixes(
        self,
        tokens: torch.Tensor,
        time_deltas: torch.Tensor,
        visit_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """States for every prefix t (predict visit t+1 from visits <= t).

        Returns (states, valid) where states[b, t] = state after visits 0..t
        and valid[b, t] = (visits t and t+1 both real). states: (B, T-1, d).
        """
        B, T, _ = tokens.shape
        if T < 2:  # degenerate: no prefix->target pairs exist
            d = tokens.shape[-1]
            return (torch.zeros(B, 0, d, device=tokens.device, dtype=tokens.dtype),
                    torch.zeros(B, 0, dtype=torch.bool, device=tokens.device))
        states, valids = [], []
        for t in range(T - 1):
            sub_mask = visit_mask[:, : t + 1]
            # Checkpoint each prefix: activations freed after forward,
            # recomputed at backward. Peak drops from (T-1) encodes to ~1
            # (same math; RNG state is preserved); ~30% slower temporal.
            h = checkpoint(
                self._encode_prefix,
                tokens[:, : t + 1], time_deltas[:, : t + 1], sub_mask,
                use_reentrant=False,
            )
            lengths = sub_mask.sum(dim=1).clamp_min(1) - 1
            idx = lengths.view(-1, 1, 1).expand(-1, 1, h.shape[-1])
            states.append(h.gather(1, idx).squeeze(1))
            valids.append(visit_mask[:, t] & visit_mask[:, t + 1])
        return torch.stack(states, dim=1), torch.stack(valids, dim=1)
