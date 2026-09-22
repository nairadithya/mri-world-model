"""Audit lesion support on BRAINIAC's 6x6x6 ViT patch grid.

This is the first P2.3 gate.  It uses exactly the legacy interface alignment
(nearest-neighbour resize of atlas masks to 96 cubed) but keeps the three
lesion compartments separate.  It does not encode images or fit a predictor.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from src.harness.data.anatomy_manifest import COMPARTMENTS

SCHEMA_VERSION = 1
GRID_SIZE = 96
PATCH_SIZE = 16


def patch_occupancy(mask: np.ndarray, size: int = GRID_SIZE,
                    patch: int = PATCH_SIZE) -> torch.Tensor:
    """Return foreground fraction in each encoder patch as a 3-D tensor."""
    if mask.ndim != 3:
        raise ValueError(f"expected a 3-D mask, got shape {mask.shape}")
    if size % patch:
        raise ValueError("target size must be divisible by patch size")
    binary = torch.from_numpy(np.asarray(mask > 0, dtype=np.float32))[None, None]
    resized = F.interpolate(binary, size=(size, size, size), mode="nearest")
    return F.avg_pool3d(resized, kernel_size=patch)[0, 0]


def support_stats(occupancy: torch.Tensor) -> dict[str, float | int]:
    """Summarize how much independent token support a mask receives."""
    flat = occupancy.flatten().float()
    mass = float(flat.sum())
    sq = float((flat * flat).sum())
    return {
        "nonzero_patches": int((flat > 0).sum()),
        "patches_ge_1pct": int((flat >= 0.01).sum()),
        "patches_ge_10pct": int((flat >= 0.10).sum()),
        "max_patch_fraction": float(flat.max()) if flat.numel() else 0.0,
        "occupancy_mass": mass,
        "effective_patches": (mass * mass / sq) if sq > 0 else 0.0,
        "dominant_patch_mass_fraction": (float(flat.max()) / mass
                                          if mass > 0 else 0.0),
    }


def _visit_occupancies(visit: dict) -> dict[str, torch.Tensor] | None:
    measurement = visit.get("measurement")
    if not measurement:
        return None
    if visit["cohort"] == "LUMIERE":
        meta = measurement.get("mask")
        if not meta or not Path(meta["path"]).exists():
            return None
        data = np.asanyarray(nib.load(meta["path"]).dataobj)
        return {name: patch_occupancy(data == int(meta["labels"][name]))
                for name in COMPARTMENTS}
    masks = measurement.get("masks", {})
    if not all(name in masks and Path(masks[name]["path"]).exists()
               for name in COMPARTMENTS):
        return None
    return {name: patch_occupancy(
        np.asanyarray(nib.load(masks[name]["path"]).dataobj))
            for name in COMPARTMENTS}


def audit_visit(visit: dict) -> dict | None:
    occupancies = _visit_occupancies(visit)
    if occupancies is None:
        return None
    union = torch.stack([value > 0 for value in occupancies.values()]).any(0)
    # One-token Chebyshev shell: context available to immediately adjacent
    # patch tokens without mixing it into the lesion token itself.
    expanded = F.max_pool3d(union.float()[None, None], kernel_size=3,
                            stride=1, padding=1)[0, 0] > 0
    ring = expanded & ~union
    return {
        "visit_id": visit["visit_id"], "cohort": visit["cohort"],
        "patient_id": visit["patient_id"], "visit": visit["visit"],
        "compartments": {name: support_stats(value)
                         for name, value in occupancies.items()},
        "union": support_stats(torch.stack(list(occupancies.values())).amax(0)),
        "adjacent_ring_patches": int(ring.sum()),
    }


def _summary(rows: list[dict]) -> dict:
    result = {}
    for cohort in sorted({row["cohort"] for row in rows}):
        subset = [row for row in rows if row["cohort"] == cohort]
        compartments = {}
        for name in COMPARTMENTS:
            values = np.asarray([row["compartments"][name]["effective_patches"]
                                 for row in subset], dtype=float)
            occupied = np.asarray([row["compartments"][name]["nonzero_patches"]
                                   for row in subset], dtype=float)
            compartments[name] = {
                "visits": len(subset),
                "empty_visits": int(np.count_nonzero(occupied == 0)),
                "effective_patches_p10_p50_p90": [float(x) for x in
                    np.percentile(values, [10, 50, 90])],
                "nonzero_patches_p10_p50_p90": [float(x) for x in
                    np.percentile(occupied, [10, 50, 90])],
                "fraction_lt_2_effective_patches": float(np.mean(values < 2)),
                "fraction_lt_4_effective_patches": float(np.mean(values < 4)),
            }
        result[cohort] = {"visits": len(subset), "compartments": compartments,
                          "adjacent_ring_patches_p10_p50_p90": [float(x) for x in
                              np.percentile([row["adjacent_ring_patches"]
                                             for row in subset], [10, 50, 90])]}
    return result


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="outputs/anatomy_harmonization.json")
    ap.add_argument("--out", default="outputs/p2_lesion_patch_coverage.json")
    args = ap.parse_args(argv)
    with open(args.manifest) as handle:
        manifest = json.load(handle)
    rows = []
    for visit in manifest["visits"]:
        row = audit_visit(visit)
        if row is not None:
            rows.append(row)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "target_version": manifest["target_version"],
        "grid": {"input_size": GRID_SIZE, "patch_size": PATCH_SIZE,
                 "shape": [GRID_SIZE // PATCH_SIZE] * 3,
                 "alignment": "nearest resize matching legacy interface"},
        "summary": _summary(rows), "visits": rows,
        "provenance": {"manifest": str(Path(args.manifest).resolve()),
                       "git_sha": _git_sha(), "date": time.strftime("%Y-%m-%d")},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
    print(json.dumps(artifact["summary"], indent=2, sort_keys=True))
    print(f"wrote {len(rows)} visits to {args.out}")


if __name__ == "__main__":
    main()
