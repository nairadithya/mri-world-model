"""Build versioned physical regional targets from prepared source-mask provenance."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.preprocessing.geometry import GEOMETRY_CONTRACT, Grid
from src.preprocessing.regional_targets import (
    REGIONAL_TARGET_CONTRACT,
    REGION_ORDER,
    regional_volumes,
)


COMPARTMENTS = ("necrotic_non_enhancing", "enhancing", "edema_flair")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def build(prepared: Path, membership: Path, mask_root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"target cache already exists: {output}")
    input_hashes = {"prepared": digest(prepared), "membership": digest(membership)}
    payload = json.loads(prepared.read_text())
    members = json.loads(membership.read_text())
    if payload.get("schema") != "r38-prepared-v3":
        raise ValueError("expected r38-prepared-v3 source-mask provenance")
    groups = members["rows"]
    if set(groups) != {"fitting", "selection", "scoring"} or not groups["fitting"]:
        raise ValueError(
            "expected fitting, selection, and scoring membership with fitting visits"
        )
    all_ids = [r["visit_id"] for rows in groups.values() for r in rows]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("visit membership must be unique across and within groups")
    patients = [{r["patient_id"] for r in groups[g]} for g in groups]
    if any(a & b for i, a in enumerate(patients) for b in patients[i + 1 :]):
        raise ValueError("patient groups overlap")
    first = payload["visits"][groups["fitting"][0]["visit_id"]]["visit"]["pregrid"]
    reference = Grid(tuple(first["shape"]), first["affine"])
    root = mask_root.resolve()
    records = {}
    for group, rows in groups.items():
        for row in rows:
            vid = row["visit_id"]
            record = payload["visits"][vid]
            visit = record["visit"]
            if visit["patient_id"] != row["patient_id"]:
                raise ValueError("source provenance and membership disagree on patient")
            grid = Grid(tuple(visit["pregrid"]["shape"]), visit["pregrid"]["affine"])
            if not grid.same_as(reference):
                raise ValueError("visits must share the fixed physical regional field")
            alignment = record["alignment"]
            if (
                alignment["geometry_contract"] != GEOMETRY_CONTRACT
                or alignment["image_pregrid"] != visit["pregrid"]
            ):
                raise ValueError("source visit lacks matching physical-grid provenance")
            labels = [visit["labels"][c] for c in COMPARTMENTS]
            path = (root / visit["mask_relative"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("mask path must remain within mask root")
            records[vid] = (record, path, labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "contract": REGIONAL_TARGET_CONTRACT,
        "groups": {},
        "all_visits_conserve_source_volume": True,
        "future_values_used_to_define_regions": False,
    }
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as temporary:
        stage = Path(temporary)
        sources = {}
        for group, rows in groups.items():
            volumes, outside, totals = [], [], []
            for row in rows:
                vid = row["visit_id"]
                record, path, labels = records[vid]
                expected = record["raw_mask_sha256"]
                if digest(path) != expected:
                    raise ValueError(
                        "source segmentation hash differs from prepared provenance"
                    )
                result = regional_volumes(path, labels, reference)
                if digest(path) != expected:
                    raise ValueError("source segmentation changed while aggregating")
                declared = np.array(
                    [record["visit"]["values_mm3"][c] for c in COMPARTMENTS]
                )
                np.testing.assert_allclose(
                    result.source_mm3, declared, rtol=1e-6, atol=1
                )
                np.testing.assert_allclose(
                    result.volumes_mm3.sum(0) + result.outside_mm3,
                    result.source_mm3,
                    rtol=1e-12,
                    atol=1e-8,
                )
                volumes.append(result.volumes_mm3)
                outside.append(result.outside_mm3)
                totals.append(result.source_mm3)
                sources[vid] = {
                    "mask_sha256": expected,
                    "labels": labels,
                    "patient_id": row["patient_id"],
                    "group": group,
                }
            values = np.asarray(volumes, dtype=np.float64).reshape(-1, 8, 3)
            excluded = np.asarray(outside, dtype=np.float64).reshape(-1, 3)
            source = np.asarray(totals, dtype=np.float64).reshape(-1, 3)
            np.savez_compressed(
                stage / f"{group}.npz",
                visit_ids=np.array([r["visit_id"] for r in rows], dtype=str),
                volumes_mm3=values,
                log1p_ml=np.log1p(values / 1000),
                outside_mm3=excluded,
                source_mm3=source,
            )
            report["groups"][group] = {
                "visits": len(rows),
                "patients": len(Counter(r["patient_id"] for r in rows)),
                "visits_with_outside_foreground": int((excluded > 0).any(1).sum()),
                "outside_mm3": excluded.sum(0).tolist(),
            }
        if (
            digest(prepared) != input_hashes["prepared"]
            or digest(membership) != input_hashes["membership"]
        ):
            raise ValueError("input metadata changed while building targets")
        source_files = [
            Path(__file__),
            Path(__file__).resolve().parents[1]
            / "src/preprocessing/regional_targets.py",
            Path(__file__).resolve().parents[1] / "src/preprocessing/geometry.py",
        ]
        manifest = {
            "contract": REGIONAL_TARGET_CONTRACT,
            "units": "mm3",
            "transformed_target": "log1p(volumes_mm3 / 1000)",
            "compartments": list(COMPARTMENTS),
            "reference_grid": reference.record(),
            "region_order": [list(x) for x in REGION_ORDER],
            "boundary_rule": "Source voxel centers; half-open field and regional bounds; upper region owns internal boundaries",
            "source_voxel_intersection_fractions": False,
            "input_sha256": input_hashes,
            "implementation_sha256": {p.name: digest(p) for p in source_files},
            "sources": sources,
            "groups": report["groups"],
            "arrays_sha256": {g: digest(stage / f"{g}.npz") for g in groups},
        }
        save(stage / "manifest-private.json", manifest)
        report["manifest_sha256"] = digest(stage / "manifest-private.json")
        save(stage / "receipt.json", report)
        stage.rename(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared",
        type=Path,
        required=True,
        help="Verified prepared.json source-mask provenance",
    )
    parser.add_argument(
        "--membership",
        type=Path,
        required=True,
        help="State membership JSON with rows per group",
    )
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New versioned cache directory"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.prepared, args.membership, args.mask_root, args.output), indent=2
        )
    )


if __name__ == "__main__":
    main()
