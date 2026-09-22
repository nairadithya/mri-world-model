"""Continuous anatomy rows for the v2 harness."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from src.harness.encode.anatomy import load_cache


@dataclass(frozen=True)
class AnatomyRows:
    x: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    delta_target: torch.Tensor
    patient_ids: list[str]
    row_ids: list[str]


def rows(path: str, *, cohort: str, patients: list[str] | None = None) -> AnatomyRows:
    cache = load_cache(path)
    allowed = None if patients is None else set(patients)
    selected = [(row_id, row) for row_id, row in sorted(cache["pairs"].items())
                if row["cohort"] == cohort
                and (allowed is None or row["patient_id"] in allowed)]
    if not selected:
        raise ValueError(f"no anatomy rows for cohort={cohort}")
    return AnatomyRows(
        x=torch.stack([r["x"] for _, r in selected]),
        source=torch.stack([r["source"] for _, r in selected]),
        target=torch.stack([r["target"] for _, r in selected]),
        delta_target=torch.stack([r["delta_target"] for _, r in selected]),
        patient_ids=[r["patient_id"] for _, r in selected],
        row_ids=[row_id for row_id, _ in selected])
