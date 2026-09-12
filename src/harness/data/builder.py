"""Dataset/split construction shared by every encode/eval entry point."""
from __future__ import annotations

from ..config import find_meta
from ...data.dataset import LUMIEREDataset
from ...data.splits import patient_splits


def _common(cfg: dict) -> dict:
    return dict(
        meta_dir=cfg["data"]["meta_dir"],
        processed_root=cfg["data"]["root"],
        raw_root=cfg["data"].get("raw_root"),
        modalities=tuple(cfg["data"].get("modalities", ["CT1", "T1", "T2", "FLAIR"])),
        min_visits=cfg["data"].get("min_visits", 2),
    )


def build_datasets(cfg: dict, patients=None):
    """Return ``(datasets_by_split, splits)`` for the hero patient split.

    Identical ordering/seed semantics to the legacy
    ``probe_rano.build_datasets``. When ``patients`` is given, a single
    ``smoke`` dataset restricted to that list is returned (used for smoke
    runs; avoids duplicating patients across the hero splits).
    """
    common = _common(cfg)
    if patients:
        patients = list(patients)
        return ({"smoke": LUMIEREDataset(patients=patients, **common)},
                {"smoke": patients})
    meta_dir = cfg["data"]["meta_dir"]
    demo_csv = find_meta(meta_dir, "demographics")
    splits = patient_splits(
        demo_csv,
        train_frac=cfg["data"].get("train_split", 0.7),
        val_frac=cfg["data"].get("val_split", 0.15),
        seed=cfg["data"].get("seed", 42),
    )
    datasets = {k: LUMIEREDataset(patients=v, **common)
                for k, v in splits.items()}
    return datasets, splits


def build_dataset(cfg: dict, patients=None) -> LUMIEREDataset:
    """Single dataset over all patients (or a given list)."""
    common = _common(cfg)
    if patients is not None:
        common["patients"] = list(patients)
    return LUMIEREDataset(**common)
