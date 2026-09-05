"""Collate: load NIfTIs, pad variable-length sequences to batch max."""
from __future__ import annotations

from functools import partial

import torch

from src.preprocessing.transforms import runtime_transform

from .dataset import MODALITIES


def collate_fn(batch: list[dict], size: tuple[int, int, int] = (96, 96, 96)) -> dict:
    B = len(batch)
    T = max(s["n_visits"] for s in batch)
    C, D, H, W = 1, *size
    M = len(MODALITIES)

    mri = torch.zeros(B, T, M, C, D, H, W)  # (B, visits, modality, C, D, H, W)
    mri_mask = torch.zeros(B, T, M, dtype=torch.bool)
    visit_mask = torch.zeros(B, T, dtype=torch.bool)
    actions = torch.zeros(B, T, dtype=torch.long)
    deltas = torch.zeros(B, T, dtype=torch.float32)

    clinical = torch.stack([s["clinical"] for s in batch])
    patient_ids = [s["patient_id"] for s in batch]
    n_visits = torch.tensor([s["n_visits"] for s in batch])

    for b, s in enumerate(batch):
        n = s["n_visits"]
        visit_mask[b, :n] = True
        actions[b, :n] = s["actions"]
        deltas[b, :n] = s["time_deltas"]
        for mi, mod in enumerate(MODALITIES):
            for t, path in enumerate(s["paths"][mod]):
                if path is None:
                    continue
                try:
                    mri[b, t, mi] = runtime_transform(path, size)
                    mri_mask[b, t, mi] = True
                except Exception:
                    continue  # leave zero-filled, mask False

    return {
        "mri": mri,
        "mri_mask": mri_mask,
        "visit_mask": visit_mask,
        "clinical": clinical,
        "actions": actions,
        "time_deltas": deltas,
        "n_visits": n_visits,
        "patient_id": patient_ids,
    }


def make_collate(size: tuple[int, int, int] = (96, 96, 96)):
    """Picklable collate factory (functools.partial survives num_workers>0)."""
    return partial(collate_fn, size=size)
