"""Locked downstream-classification evaluation protocol (Step 1).

One frozen, committed source of truth for probe evaluation:

- ``encoder_train``  — the 65 hero-split patients the champion encoder saw
  during SSL training.
- ``encoder_unseen`` — the 26 = 13 val + 13 test patients the encoder never
  trained on (K3-16). The probe cohort.
- ``folds``          — a fixed 5-fold partition of ``encoder_unseen``,
  materialized in ``info/eval_folds.json`` and never re-drawn.
- ``dev`` / ``final``— the val / test halves used for transfer reporting.

Rules (see ``info/eval_protocol.md``):

1. No probe is trained or scored on an encoder-train patient when making a
   representation claim; ``assert_disjoint`` enforces this.
2. Fold assignments come from the JSON, never a fresh ``shuffle``.
3. Selection and headline both use the locked patient-wise CV; uncertainty
   is a patient-cluster bootstrap with *paired* differences between models.
"""
from __future__ import annotations

import json
import os
import random

from src.data.splits import patient_splits

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROTOCOL = os.path.join(_HERE, "..", "..", "info", "eval_folds.json")

LABEL_MAP = {"PD": 0, "SD": 1, "PR": 2, "CR": 3}
PRIMARY_METRIC = "macro_f1"


def _demo_csv(meta_dir: str) -> str:
    return next(os.path.join(meta_dir, f) for f in os.listdir(meta_dir)
                if f.startswith("demographics"))


def hero_splits(cfg: dict) -> dict[str, list[str]]:
    """Reproduce the hero 65/13/13 split exactly from the config seed."""
    return patient_splits(
        _demo_csv(cfg["data"]["meta_dir"]),
        train_frac=cfg["data"].get("train_split", 0.7),
        val_frac=cfg["data"].get("val_split", 0.15),
        seed=cfg["data"].get("seed", 42),
    )


def build_protocol(cfg: dict, k: int = 5, fold_seed: int = 2026) -> dict:
    """Construct the frozen protocol (does not write it)."""
    sp = hero_splits(cfg)
    unseen = sorted(sp["val"] + sp["test"])
    shuffled = list(unseen)
    random.Random(fold_seed).shuffle(shuffled)
    folds = [shuffled[i::k] for i in range(k)]
    fold_of = {pid: i for i, fold in enumerate(folds) for pid in fold}
    return {
        "version": 1,
        "created": "2026-09-11",
        "k": k,
        "fold_seed": fold_seed,
        "encoder_split_seed": cfg["data"].get("seed", 42),
        "encoder_train": sorted(sp["train"]),
        "encoder_unseen": unseen,
        "dev": sorted(sp["val"]),
        "final": sorted(sp["test"]),
        "folds": fold_of,
        "label_map": LABEL_MAP,
        "primary_metric": PRIMARY_METRIC,
    }


def write_protocol(protocol: dict, path: str = DEFAULT_PROTOCOL) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(protocol, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def load_protocol(path: str = DEFAULT_PROTOCOL) -> dict:
    with open(path) as f:
        return json.load(f)


def fold_patients(protocol: dict, i: int) -> list[str]:
    return sorted(p for p, f in protocol["folds"].items() if f == i)


def assert_disjoint(protocol: dict) -> None:
    """Encoder-train and probe-fold patients must not overlap."""
    train = set(protocol["encoder_train"])
    probe = set(protocol["folds"])
    overlap = train & probe
    assert not overlap, f"encoder-train/probe leakage: {sorted(overlap)}"
    assert set(protocol["encoder_unseen"]) == probe, "folds != encoder_unseen"
    n = len(set(protocol["folds"].values()))
    assert n == protocol["k"], f"expected {protocol['k']} folds, found {n}"
