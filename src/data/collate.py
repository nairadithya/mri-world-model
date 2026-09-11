"""Collate: load NIfTIs, pad variable-length sequences to batch max."""
from __future__ import annotations

from functools import partial

import torch

from src.preprocessing.transforms import runtime_transform

from .dataset import MODALITIES


def _augment_volume(v: torch.Tensor) -> torch.Tensor:
    """Training-only augmentation on a z-scored (1, D, H, W) volume.

    Random axis flips + intensity scale/shift + small Gaussian noise. Cheap
    and label-preserving; the missing regularizer for the 65-patient LoRA
    finetune (Step 3).
    """
    for ax in (1, 2, 3):
        if torch.rand(1).item() < 0.5:
            v = torch.flip(v, [ax])
    a = 1.0 + (torch.rand(1).item() * 2 - 1) * 0.1
    b = (torch.rand(1).item() * 2 - 1) * 0.1
    v = v * a + b
    return v + torch.randn(v.shape) * 0.02


def collate_fn(batch: list[dict], size: tuple[int, int, int] = (96, 96, 96),
               dtype: torch.dtype = torch.float32,
               augment: bool = False) -> dict:
    B = len(batch)
    T = max(s["n_visits"] for s in batch)
    C, D, H, W = 1, *size
    M = len(MODALITIES)

    # NOTE: padded rows are storage only — encode_chunked skips unmasked rows,
    # so padding never touches the backbone. dtype may be float16 to halve the
    # input footprint (backbone casts to float on the chunk path); the default
    # float32 preserves exact legacy numerics.
    mri = torch.zeros(B, T, M, C, D, H, W, dtype=dtype)  # (B, visits, modality, C, D, H, W)
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
                    vol = runtime_transform(path, size)
                    if augment:
                        vol = _augment_volume(vol)
                    mri[b, t, mi] = vol
                    mri_mask[b, t, mi] = True
                except Exception:
                    continue  # leave zero-filled, mask False

    out = {
        "mri": mri,
        "mri_mask": mri_mask,
        "visit_mask": visit_mask,
        "clinical": clinical,
        "actions": actions,
        "time_deltas": deltas,
        "n_visits": n_visits,
        "patient_id": patient_ids,
    }
    # Treatment phase channel (SAILOR; absent for LUMIERE). Padded 3 =
    # unknown phase, matching the adapter default. Dynamics prefers this
    # over RANO actions when present (real treatment, not response proxy).
    if any("treatment" in s for s in batch):
        treatment = torch.full((B, T), 3, dtype=torch.long)
        for b, s in enumerate(batch):
            if "treatment" in s:
                n = s["n_visits"]
                treatment[b, :n] = s["treatment"][:n]
        out["treatment"] = treatment
    return out


def make_collate(size: tuple[int, int, int] = (96, 96, 96),
                 dtype: torch.dtype = torch.float32,
                 augment: bool = False):
    """Picklable collate factory (functools.partial survives num_workers>0)."""
    return partial(collate_fn, size=size, dtype=dtype, augment=augment)
