"""RANO readout heads for joint JEPA + aux training (path B / TRACE-style).

Heads read temporal states (history summary), forecast framing:
state_t (visits <= t) predicts the status of visit t+1.

- flat: 4-class {PD, SD, PR, CR} linear (vs 0.50 SOTA line / frozen 0.33).
- prog: binary PD-vs-rest (TRACE binary framing, cf. their 0.71).
- resp: binary (PR|CR)-vs-SD on non-PD visits (response signal; sidesteps
  the n=1-PR problem by pooling PR+CR).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# batch action id (dataset RANO_ACTION_MAP) -> flat class {PD:0, SD:1, PR:2, CR:3}
ACTION_TO_FLAT = {3: 0, 2: 1, 5: 2, 4: 3}
CLEAN_ACTIONS = (2, 3, 4, 5)  # response classes only (no operative/missing)


class RANOHeads(nn.Module):
    def __init__(self, d_model: int = 1152):
        super().__init__()
        self.flat = nn.Linear(d_model, 4)
        self.prog = nn.Linear(d_model, 1)
        self.resp = nn.Linear(d_model, 1)

    def aux_losses(
        self,
        states: torch.Tensor,      # (B, T-1, d) prefix states
        actions: torch.Tensor,     # (B, T) action ids
        valid: torch.Tensor,       # (B, T-1) JEPA pair-valid mask
        weights: dict,
    ) -> dict[str, torch.Tensor]:
        """CE/BCE aux losses on clean-labelled valid pairs (forecast framing)."""
        B, Tm1, _ = states.shape
        a_next = actions[:, 1:]                     # status of visit t+1
        clean = torch.zeros_like(a_next, dtype=torch.bool)
        for a in CLEAN_ACTIONS:
            clean |= a_next == a
        use = clean & valid
        out = {}
        if not use.any():
            z = states.sum() * 0.0
            return {"flat": z, "prog": z, "resp": z, "n_aux": 0}
        flat_tgt = torch.full_like(a_next, -1)
        for a, c in ACTION_TO_FLAT.items():
            flat_tgt[a_next == a] = c
        h = states[use]
        cw = weights.get("class_weights")
        out["flat"] = F.cross_entropy(self.flat(h), flat_tgt[use],
                                      weight=cw.to(h.device) if cw is not None else None)
        prog_tgt = (a_next[use] == 3).float()
        pw = weights.get("prog_pos_weight", 0.56)
        out["prog"] = F.binary_cross_entropy_with_logits(
            self.prog(h).squeeze(1), prog_tgt,
            pos_weight=torch.tensor(pw, device=h.device))
        nonpd = use & (a_next != 3) & ((a_next == 2) | (a_next == 4) | (a_next == 5))
        if nonpd.any():
            resp_tgt = ((a_next[nonpd] == 4) | (a_next[nonpd] == 5)).float()
            rw = weights.get("resp_pos_weight", 2.0)
            out["resp"] = F.binary_cross_entropy_with_logits(
                self.resp(states[nonpd]).squeeze(1), resp_tgt,
                pos_weight=torch.tensor(rw, device=h.device))
        else:
            out["resp"] = states.sum() * 0.0
        out["n_aux"] = int(use.sum())
        return out
