"""Frozen artifact paths — the single source of truth for harness I/O.

These are the exact defaults the pre-refactor scripts used. Do not rename any
of them: Kaggle kernels, ``info/`` records and committed provenance all point
at these files. New experiments add names here rather than inlining strings.
"""
from __future__ import annotations

import os

# --- checkpoints / caches -------------------------------------------------
CHAMPION = "checkpoints/champion_0.0081.pt"
PROBE_CACHE = "checkpoints/probe_cache.pt"
INTERFACE_CACHE = "checkpoints/interface_cache.pt"
HORIZON_CACHE = "checkpoints/horizon_cache.pt"
FIELD_CACHE = "checkpoints/field_cache.pt"
FIELD_MODELS = "checkpoints/field_models.pt"
FIELD_SCORES = "checkpoints/field_scores.pt"
PROBE_HEAD = "checkpoints/probe_head.pt"
SAILOR_CACHE = "checkpoints/sailor_cache.pt"
SAILOR_Z_CACHE = "checkpoints/sailor_z_cache.pt"
CNN_FEATURES = "checkpoints/cnn_features.pt"
CNN2D_FEATURES = "checkpoints/cnn2d_features.pt"
RADIOMICS_FEATURES = "checkpoints/radiomics_features.pt"
CHECKPOINT_DIR = "checkpoints/"

# --- evaluation protocol --------------------------------------------------
EVAL_FOLDS = "info/eval_folds.json"

# --- data roots -----------------------------------------------------------
LUMIERE_ROOT = "data/lumiere_preprocessed"
LUMIERE_META = "data/lumiere_meta"
LUMIERE_RAW = "data/lumiere_sample/Imaging"
SAILOR_ROOT = "data/sailor/sailor_ebrains_pseud/derivatives/mni2009c-n-s"
MASK_ROOT = "data/autoseg/extracted/Imaging"
VOL_ROOT = "data/autoseg/vols/Imaging"

DEFAULT_CONFIG = "config/default.yaml"


def exists(path: str) -> bool:
    return os.path.exists(path)


def resolve(path: str | None, default: str) -> str:
    return default if path is None else path
