"""Row-exact joins between anatomy pairs and frozen representation caches."""
from __future__ import annotations

import json

import torch

from src.harness.encode.anatomy import load_cache as load_anatomy_cache

VIEWS = ("vision", "history_mean", "jepa_state")


def representation_rows(*, feature_cache: str, manifest_path: str,
                        representation_cache: str, cohort: str,
                        view: str) -> dict[str, torch.Tensor]:
    if view not in VIEWS:
        raise ValueError(f"unknown representation view {view!r}")
    anatomy = load_anatomy_cache(feature_cache)
    manifest = json.load(open(manifest_path))
    encoded = torch.load(representation_cache, map_location="cpu", weights_only=False)
    visits = {v["visit_id"]: v for v in manifest["visits"]}
    out = {}
    for row_id, row in anatomy["pairs"].items():
        if row["cohort"] != cohort:
            continue
        patient = encoded["patients"].get(row["patient_id"])
        if patient is None:
            continue
        source = visits[row["source_visit_id"]]
        if cohort == "LUMIERE":
            names = list(patient.get("visits", []))
        else:
            names = [f"ses-{i + 1:02d}" for i in range(len(patient["vision"]))]
        if source["visit"] not in names:
            continue
        t = names.index(source["visit"])
        if view == "vision":
            value = patient["vision"][t]
        elif view == "history_mean":
            value = patient["vision"][:t + 1].mean(0)
        else:
            if t >= len(patient["states"]):
                continue
            value = patient["states"][t]
        value = torch.as_tensor(value).float().flatten()
        if torch.isfinite(value).all():
            out[row_id] = value
    return out
