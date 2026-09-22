"""P2 physical lesion features keyed to ``lesion-state-v1`` row IDs.

Inputs are native segmentation masks and affines.  Pair features are built
strictly from visits through the source visit; target measurements are stored
in a separate tensor and never concatenated into ``x``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from src.harness.data.anatomy_manifest import COMPARTMENTS, TARGET_VERSION
from src.harness.provenance import git_sha

FEATURE_SCHEMA_VERSION = 1
FEATURE_VERSION = "lesion-physical-v1"
SHAPE_FIELDS = ("present", "centroid_x", "centroid_y", "centroid_z",
                "extent_x", "extent_y", "extent_z",
                "principal_sd_1", "principal_sd_2", "principal_sd_3")


def feature_names() -> list[str]:
    names = [f"log_volume:{c}" for c in COMPARTMENTS]
    names += ["log_volume:total"]
    names += [f"volume_fraction:{c}" for c in COMPARTMENTS]
    names += [f"{field}:{c}" for c in COMPARTMENTS for field in SHAPE_FIELDS]
    names += ["modality:CT1", "modality:T1", "modality:T2", "modality:FLAIR"]
    return names


def _shape(mask: np.ndarray, affine: np.ndarray) -> list[float]:
    ijk = np.column_stack(np.nonzero(mask)).astype(np.float64)
    if len(ijk) == 0:
        return [0.0] * len(SHAPE_FIELDS)
    linear, offset = affine[:3, :3], affine[:3, 3]
    center = linear @ ijk.mean(0) + offset
    lo, hi = ijk.min(0), ijk.max(0) + 1.0
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    world_corners = corners @ linear.T + offset
    extent = world_corners.max(0) - world_corners.min(0)
    if len(ijk) > 1:
        centered = ijk - ijk.mean(0)
        covariance = (centered.T @ centered) / len(ijk)
        world_covariance = linear @ covariance @ linear.T
        principal = np.sqrt(np.clip(np.linalg.eigvalsh(world_covariance), 0, None))[::-1]
    else:
        principal = np.zeros(3)
    return [1.0, *center.tolist(), *extent.tolist(), *principal.tolist()]


def _masks(visit: dict) -> tuple[dict[str, np.ndarray], np.ndarray]:
    measurement = visit["measurement"]
    if "mask" in measurement:  # LUMIERE integer atlas
        spec = measurement["mask"]
        image = nib.load(spec["path"])
        data = np.asanyarray(image.dataobj)
        labels = spec["labels"]
        return ({c: np.isfinite(data) & (data == labels[c]) for c in COMPARTMENTS},
                np.asarray(image.affine, dtype=np.float64))
    masks, affine = {}, None  # SAILOR binary compartment masks
    for compartment in COMPARTMENTS:
        image = nib.load(measurement["masks"][compartment]["path"])
        data = np.asanyarray(image.dataobj)
        current_affine = np.asarray(image.affine, dtype=np.float64)
        if affine is not None and not np.allclose(affine, current_affine, atol=1e-4):
            raise ValueError(f"{visit['visit_id']}: compartment affine mismatch")
        affine = current_affine
        masks[compartment] = np.isfinite(data) & (data > 0)
    return masks, affine


def visit_features(visit: dict) -> np.ndarray:
    values = visit["measurement"]["values_mm3"]
    volumes = np.array([float(values[c]) for c in COMPARTMENTS], dtype=np.float64)
    total = float(volumes.sum())
    masks, affine = _masks(visit)
    out = [*np.log1p(volumes), math.log1p(total)]
    out += list(volumes / max(total, 1.0))
    for compartment in COMPARTMENTS:
        out += _shape(masks[compartment], affine)
    modalities = visit.get("modalities", {})
    out += [float(bool(modalities.get(name))) for name in ("CT1", "T1", "T2", "FLAIR")]
    result = np.asarray(out, dtype=np.float32)
    if len(result) != len(feature_names()) or not np.isfinite(result).all():
        raise ValueError(f"{visit['visit_id']}: invalid physical feature vector")
    return result


def history_vector(history: list[np.ndarray]) -> np.ndarray:
    """Forecast input from source history only; target visit is never passed."""
    if not history:
        raise ValueError("history cannot be empty")
    current = history[-1]
    log_vol = current[:len(COMPARTMENTS)]
    previous = history[-2][:len(COMPARTMENTS)] if len(history) > 1 else log_vol
    nadir = np.stack([x[:len(COMPARTMENTS)] for x in history]).min(0)
    return np.concatenate([current, log_vol - previous, log_vol - nadir,
                           np.array([math.log1p(len(history))], dtype=np.float32)]).astype(np.float32)


def history_feature_names() -> list[str]:
    return (feature_names()
            + [f"previous_delta:{c}" for c in COMPARTMENTS]
            + [f"nadir_delta:{c}" for c in COMPARTMENTS]
            + ["log_history_length"])


def build_cache(manifest: dict, manifest_path: str) -> dict:
    if manifest.get("target_version") != TARGET_VERSION:
        raise ValueError("manifest target contract mismatch")
    visit_by_id = {v["visit_id"]: v for v in manifest["visits"]}
    usable_ids = {p["source_visit_id"] for p in manifest["pairs"] if p["usable"]}
    usable_ids |= {p["target_visit_id"] for p in manifest["pairs"] if p["usable"]}
    encoded = {}
    for i, visit_id in enumerate(sorted(usable_ids), 1):
        visit = visit_by_id[visit_id]
        encoded[visit_id] = visit_features(visit)
        if i == 1 or i % 50 == 0 or i == len(usable_ids):
            print(f"physical features {i}/{len(usable_ids)}", flush=True)
    visits_by_patient: dict[tuple[str, str], list[dict]] = {}
    for visit in manifest["visits"]:
        visits_by_patient.setdefault((visit["cohort"], visit["patient_id"]), []).append(visit)
    pair_rows = {}
    for pair in manifest["pairs"]:
        if not pair["usable"]:
            continue
        key = (pair["cohort"], pair["patient_id"])
        ordered = sorted(visits_by_patient[key], key=lambda v: (
            v["day"] if v["day"] is not None else float("inf"), v["visit"]))
        source_i = next(i for i, v in enumerate(ordered)
                        if v["visit_id"] == pair["source_visit_id"])
        history = [encoded[v["visit_id"]] for v in ordered[:source_i + 1]
                   if v["visit_id"] in encoded]
        source_values = visit_by_id[pair["source_visit_id"]]["measurement"]["values_mm3"]
        target_values = visit_by_id[pair["target_visit_id"]]["measurement"]["values_mm3"]
        source_y = np.log1p([source_values[c] for c in COMPARTMENTS]).astype(np.float32)
        target_y = np.log1p([target_values[c] for c in COMPARTMENTS]).astype(np.float32)
        pair_rows[pair["row_id"]] = {
            "cohort": pair["cohort"], "patient_id": pair["patient_id"],
            "source_visit_id": pair["source_visit_id"],
            "target_visit_id": pair["target_visit_id"],
            "x": torch.from_numpy(history_vector(history)),
            "source": torch.from_numpy(source_y),
            "target": torch.from_numpy(target_y),
            "delta_target": torch.from_numpy(target_y - source_y),
            "history_length": len(history), "gap_days": pair["gap_days"],
        }
    manifest_bytes = Path(manifest_path).read_bytes()
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION, "target_version": TARGET_VERSION,
        "feature_names": history_feature_names(),
        "target_names": [f"future_log_volume:{c}" for c in COMPARTMENTS],
        "pairs": pair_rows,
        "provenance": {
            "manifest": str(Path(manifest_path).resolve()),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "git_sha": git_sha(), "date": time.strftime("%Y-%m-%d"),
            "information_cutoff": "history through source visit only",
            "geometry": "native mask affine",
        },
    }


def load_cache(path: str) -> dict:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if cache.get("feature_version") != FEATURE_VERSION:
        raise ValueError(f"expected {FEATURE_VERSION}, got {cache.get('feature_version')}")
    names = cache.get("feature_names", [])
    for row_id, row in cache.get("pairs", {}).items():
        if row["x"].numel() != len(names) or not torch.isfinite(row["x"]).all():
            raise ValueError(f"{row_id}: invalid feature row")
        if row["target"].shape != (len(COMPARTMENTS),):
            raise ValueError(f"{row_id}: invalid target shape")
    return cache


def main(argv=None):
    ap = argparse.ArgumentParser(description="build P2 physical lesion features")
    ap.add_argument("--manifest", default="outputs/anatomy_harmonization.json")
    ap.add_argument("--out", default="checkpoints/anatomy_features.pt")
    args = ap.parse_args(argv)
    manifest = json.loads(Path(args.manifest).read_text())
    cache = build_cache(manifest, args.manifest)
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    print(f"wrote {path}: pairs={len(cache['pairs'])} "
          f"features={len(cache['feature_names'])}")


if __name__ == "__main__":
    main()
