"""Feature views: named latent slots a task can read from a cache.

A view is either ``snapshot`` (one row per visit: ``(T, d)``) or ``prefix``
(``(T-1, d)`` history summaries from the temporal stack). Adding a latent slot
is one name here (plus wherever it is encoded); no evaluator edits.
"""
from __future__ import annotations

import torch

from ..registry import VIEWS

# Views indexed by visit (T rows). "clinical" is static and broadcast.
SNAPSHOT = {
    "vision", "vision_mod", "vision_roi", "vision_roi_mod",
    "fused", "volumes", "clinical", "roi", "roi_concat", "mod_concat",
    "ema_z", "z",
}
# Views indexed by history prefix (T-1 rows).
PREFIX = {"states", "states_roi"}


def is_prefix(name: str) -> bool:
    return name in PREFIX


def is_snapshot(name: str) -> bool:
    return name in SNAPSHOT


def has(patient: dict, name: str) -> bool:
    feats = patient.get("features")
    if feats is not None and name in feats:
        return True
    return name in patient


def get(patient: dict, name: str) -> torch.Tensor:
    feats = patient.get("features")
    if feats is not None and name in feats:
        return feats[name]
    if name in patient:
        return patient[name]
    raise KeyError(f"view {name!r} absent (have "
                   f"{sorted(set(patient.get('features', {})) | set(patient))})")


def available(patient: dict) -> set[str]:
    return set(patient.get("features", {})) | set(patient)


@VIEWS.register("snapshot")
def _snapshot() -> frozenset:
    return frozenset(SNAPSHOT)


@VIEWS.register("prefix")
def _prefix() -> frozenset:
    return frozenset(PREFIX)
