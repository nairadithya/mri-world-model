"""Canonical representation cache schema + legacy adapters.

Canonical form::

    {"patients": {pid: {"split": str, "labels": LongTensor(T,),
                        "features": {view_name: Tensor, ...}, ...meta}},
     "provenance": {...}}

Legacy caches store features at the top level (``patient["states"]``); the
canonicalizer lifts those into ``features`` without copying or renaming any
file. This is what lets the harness read every committed cache unchanged.
"""
from __future__ import annotations

import os

import torch

from .. import provenance as _prov
from . import views

CACHE_SCHEMA_VERSION = 2

FEATURE_KEYS = (
    "vision", "vision_mod", "vision_roi", "vision_roi_mod", "fused",
    "states", "states_roi", "volumes", "clinical", "ema_z", "z",
)


def make_cache(prov: dict | None = None) -> dict:
    cache = {"schema_version": CACHE_SCHEMA_VERSION, "patients": {}}
    if prov is not None:
        cache["provenance"] = prov
    return cache


def add_patient(cache: dict, pid: str, *, split: str, features: dict | None = None,
                **meta) -> dict:
    entry = {"split": split, **meta}
    if features:
        entry["features"] = dict(features)
    cache["patients"][pid] = entry
    return entry


def save(path: str, cache: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(cache, path)


def load(path: str, *, warn_provenance: bool = True,
         require_current: bool = False) -> dict:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    _prov.assert_provenance(cache, warn=warn_provenance)
    if require_current:
        problems = validate(cache, strict=True)
        if problems:
            raise ValueError(f"invalid cache {path}: {'; '.join(problems)}")
    return cache


def canonicalize(cache: dict) -> dict:
    """Return a normalized cache: features nested, meta preserved.

    Non-destructive: feature tensors are shared by reference, so this is
    cheap and safe to call on a freshly loaded legacy cache.
    """
    out = {"schema_version": cache.get("schema_version", 1), "patients": {}}
    if "provenance" in cache:
        out["provenance"] = cache["provenance"]
    for pid, p in cache["patients"].items():
        feats = dict(p.get("features") or {})
        meta = {}
        for k, v in p.items():
            if k == "features":
                continue
            if k in FEATURE_KEYS:
                feats.setdefault(k, v)
            else:
                meta[k] = v
        out["patients"][pid] = {**meta, "features": feats}
    return out


def validate(cache: dict, require=("labels",), strict: bool = False) -> list[str]:
    """Return human-readable problems; strict mode enforces current artifacts."""
    problems = []
    if strict and cache.get("schema_version") != CACHE_SCHEMA_VERSION:
        problems.append(f"cache schema {cache.get('schema_version', 1)} != {CACHE_SCHEMA_VERSION}")
    if strict and "provenance" not in cache:
        problems.append("missing provenance")
    if "patients" not in cache:
        problems.append("cache has no 'patients'")
        return problems
    for pid, p in cache["patients"].items():
        if "split" not in p:
            problems.append(f"{pid}: missing split")
        for r in require:
            if r == "labels" and "labels" not in p and "features" not in p:
                problems.append(f"{pid}: missing labels")
    return problems


def provenance_of(cache: dict) -> dict | None:
    return cache.get("provenance")
