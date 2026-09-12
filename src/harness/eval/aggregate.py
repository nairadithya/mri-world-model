"""Uncertainty: percentile CIs and patient-cluster bootstrap.

One implementation replaces the three near-copies (``probe_rano._bootstrap``,
``finetune_lora.bootstrap``, ``radiomics_probe._bootstrap``). The patient is
the resampling unit because visits within a patient are not independent.
"""
from __future__ import annotations

import random

import torch

from .metrics import macro_f1


def percentile_ci(samples, lo: float = 2.5, hi: float = 97.5):
    s = sorted(samples)
    n = len(s)
    return s[int(lo / 100 * n)], s[min(n - 1, int(hi / 100 * n))]


def pooled(oof: dict, pids, metric_fn=macro_f1):
    """Concatenate per-patient (pred, y) for ``pids`` and score pooled."""
    pred = torch.cat([oof[p][0] for p in pids])
    y = torch.cat([oof[p][1] for p in pids])
    return metric_fn(pred, y), pred, y


def patient_bootstrap(oof_a: dict, oof_b: dict | None = None, boot: int = 10000,
                      seed: int = 42, metric_fn=macro_f1):
    """Patient-cluster bootstrap of a metric (and the paired difference).

    Returns ``(a_vals, diff_vals)`` where ``diff = a - b`` on the same
    resampled patients (``None`` when ``oof_b`` is None). RNG consumption is
    identical to the legacy probe implementation so scores reproduce exactly.
    """
    pids = sorted(set(oof_a) & (set(oof_b) if oof_b else set(oof_a)))
    rng = random.Random(seed)
    a_vals, d_vals = [], []
    for _ in range(boot):
        samp = [rng.choice(pids) for _ in pids]
        fa, _, _ = pooled(oof_a, samp, metric_fn)
        a_vals.append(fa)
        if oof_b is not None:
            fb, _, _ = pooled(oof_b, samp, metric_fn)
            d_vals.append(fa - fb)
    return a_vals, (d_vals if oof_b is not None else None)


def mean_std(values) -> tuple[float, float]:
    import statistics
    if not values:
        return float("nan"), float("nan")
    return statistics.mean(values), statistics.pstdev(values)
