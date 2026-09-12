"""Transition classes and inverse-prevalence weights for JEPA pair loss.

R7/R13: the SSL objective currently weights every valid pair equally, but the
transitions are dominated by stable/near-stable pairs (cf. R2 atlas). Matoso's
weighted sampler is the supervised analogue. We reweight the 1-step JEPA loss
per pair by inverse prevalence of its coarse transition class.
"""
from __future__ import annotations

CLEAN = (2, 3, 4, 5)          # RANO action ids: SD, PD, CR, PR
RESPONSE = (4, 5)             # CR, PR
N_CLASSES = 4
CLASS_NAMES = ["stable->SD", "->PD", "->response", "other"]


def transition_class(a: int, b: int) -> int:
    """Coarse transition class for action id at t (a) -> t+1 (b)."""
    if a not in CLEAN or b not in CLEAN:
        return 3
    if b in RESPONSE:
        return 2
    if b == 3:
        return 1
    return 0  # b == SD


def inverse_prevalence_weights(counts) -> list[float]:
    """counts: length-N_CLASSES. Returns w_c = total/(N*count_c) (>=1 classes)."""
    total = float(sum(counts))
    out = []
    for c in counts:
        out.append(total / (N_CLASSES * max(float(c), 1.0)))
    return out
