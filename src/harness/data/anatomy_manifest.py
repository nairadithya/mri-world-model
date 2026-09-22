"""Build a RANO-free, cross-cohort lesion-state manifest.

The manifest is deliberately conservative: it records native measurement
provenance and timing uncertainty instead of treating DeepBraTumIA JSON
volumes and SAILOR ONCO masks as interchangeable ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import nibabel as nib
import numpy as np

from src.data.dataset import parse_week_to_days, sort_timepoints

SCHEMA_VERSION = 1
TARGET_VERSION = "lesion-state-v1"
COMPARTMENTS = ("necrotic_non_enhancing", "enhancing", "edema_flair")
LUMIERE_KEYS = {
    "necrotic_non_enhancing": "Necrotic_NonEnhancing",
    "enhancing": "Enhancing_Core",
    "edema_flair": "Edema_Compartment",
}
# DeepBraTumIA atlas seg_mask values, verified against all 599 shipped
# measured_volumes_in_mm3.json files (1,797/1,797 exact compartment matches).
LUMIERE_LABELS = {
    "necrotic_non_enhancing": 2,
    "enhancing": 1,
    "edema_flair": 3,
}
SAILOR_MASKS = {
    "necrotic_non_enhancing": "NecrosisMask-ONCO",
    "enhancing": "ContrastEnhancedMask-ONCO",
    "edema_flair": "EdemaMask-ONCO",
}
SAILOR_CL_MASKS = {
    "enhancing": "ContrastEnhancedMask-CL",
    "edema_flair": "EdemaMask-CL",
}


def _stable_id(*parts: object) -> str:
    raw = json.dumps([TARGET_VERSION, *parts], separators=(",", ":"))
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def _find_nii(directory: Path, stem: str) -> Path | None:
    for suffix in (".nii.gz", ".nii"):
        path = directory / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def measure_mask(path: str | Path) -> dict:
    """Measure a binary mask in physical units and retain geometry QA."""
    path = Path(path)
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    affine = np.asarray(img.affine, dtype=float)
    voxel_mm3 = float(abs(np.linalg.det(affine[:3, :3])))
    finite = bool(np.isfinite(data).all())
    geometry_valid = (
        data.ndim == 3
        and affine.shape == (4, 4)
        and np.isfinite(affine).all()
        and math.isfinite(voxel_mm3)
        and voxel_mm3 > 0
    )
    foreground = int(np.count_nonzero(np.isfinite(data) & (data > 0)))
    return {
        "path": str(path),
        "shape": list(data.shape),
        "voxel_volume_mm3": voxel_mm3,
        "foreground_voxels": foreground,
        "volume_mm3": foreground * voxel_mm3,
        "finite": finite,
        "geometry_valid": bool(geometry_valid),
    }


def compare_masks(first: str | Path, second: str | Path) -> dict:
    """Compare two masks in a shared voxel grid for target sensitivity."""
    a_img, b_img = nib.load(str(first)), nib.load(str(second))
    a = np.asanyarray(a_img.dataobj)
    b = np.asanyarray(b_img.dataobj)
    same_grid = (a.shape == b.shape and
                 np.allclose(a_img.affine, b_img.affine, rtol=0, atol=1e-4))
    if not same_grid:
        return {"same_grid": False, "dice": None, "volume_ratio_cl_to_onco": None}
    aa = np.isfinite(a) & (a > 0)
    bb = np.isfinite(b) & (b > 0)
    denom = int(aa.sum() + bb.sum())
    dice = 1.0 if denom == 0 else float(2 * np.count_nonzero(aa & bb) / denom)
    onco_n, cl_n = int(aa.sum()), int(bb.sum())
    ratio = ((cl_n + 1.0) / (onco_n + 1.0))
    return {"same_grid": True, "dice": dice,
            "volume_ratio_cl_to_onco": float(ratio),
            "onco_foreground_voxels": onco_n, "cl_foreground_voxels": cl_n}


def measure_label_map(path: str | Path, labels: dict[str, int]) -> dict:
    """Measure mutually exclusive integer compartments in physical units."""
    path = Path(path)
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    affine = np.asarray(img.affine, dtype=float)
    voxel_mm3 = float(abs(np.linalg.det(affine[:3, :3])))
    geometry_valid = (
        data.ndim == 3 and affine.shape == (4, 4)
        and np.isfinite(affine).all() and math.isfinite(voxel_mm3)
        and voxel_mm3 > 0 and np.isfinite(data).all()
    )
    values = {name: float(np.count_nonzero(data == label) * voxel_mm3)
              for name, label in labels.items()}
    return {"path": str(path), "shape": list(data.shape),
            "voxel_volume_mm3": voxel_mm3, "values_mm3": values,
            "finite": bool(np.isfinite(data).all()),
            "geometry_valid": bool(geometry_valid),
            "labels": dict(labels)}


def _modalities(directory: Path, names: dict[str, str]) -> dict[str, bool]:
    return {slot: _find_nii(directory, stem) is not None
            for slot, stem in names.items()}


def _measurement(values: dict[str, float], *, source: str,
                 geometry_verified: bool) -> dict:
    total = float(sum(values.values()))
    return {
        "values_mm3": {k: float(values[k]) for k in COMPARTMENTS},
        "total_mm3": total,
        "source": source,
        "geometry_verified": geometry_verified,
        "complete": all(math.isfinite(float(values[k])) and float(values[k]) >= 0
                        for k in COMPARTMENTS),
    }


def lumiere_visits(processed_root: str, volume_root: str,
                    mask_root: str) -> list[dict]:
    processed, volumes, masks = (Path(processed_root), Path(volume_root),
                                  Path(mask_root))
    patients = sorted({p.name for root in (processed, volumes) if root.exists()
                       for p in root.glob("Patient-*") if p.is_dir()})
    rows = []
    for patient in patients:
        names = {p.name for root in (processed / patient, volumes / patient)
                 if root.exists() for p in root.glob("week-*") if p.is_dir()}
        for visit in sort_timepoints(list(names)):
            image_dir = processed / patient / visit
            volume_json = (volumes / patient / visit / "DeepBraTumIA-segmentation" /
                           "atlas/segmentation/measured_volumes_in_mm3.json")
            mask_path = (masks / patient / visit / "DeepBraTumIA-segmentation" /
                         "atlas/segmentation/seg_mask.nii.gz")
            measurement = None
            exclusions = []
            json_values = None
            if volume_json.exists():
                try:
                    with open(volume_json) as f:
                        raw = json.load(f)
                    json_values = {name: float(raw[key])
                                   for name, key in LUMIERE_KEYS.items()}
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    exclusions.append(f"invalid_measurement_json:{type(exc).__name__}")
            if mask_path.exists():
                try:
                    mask = measure_label_map(mask_path, LUMIERE_LABELS)
                    measurement = _measurement(
                        mask["values_mm3"], source="DeepBraTumIA atlas seg_mask.nii.gz",
                        geometry_verified=mask["geometry_valid"])
                    measurement["mask"] = mask
                    measurement["reported_json_path"] = (
                        str(volume_json) if volume_json.exists() else None)
                    if json_values is not None:
                        delta = {k: mask["values_mm3"][k] - json_values[k]
                                 for k in COMPARTMENTS}
                        measurement["reported_json_delta_mm3"] = delta
                        measurement["reported_json_exact"] = all(
                            abs(x) < 1e-6 for x in delta.values())
                        if not measurement["reported_json_exact"]:
                            exclusions.append("mask_json_volume_mismatch")
                    if not mask["geometry_valid"]:
                        exclusions.append("invalid_lumiere_mask_geometry")
                except (OSError, ValueError) as exc:
                    exclusions.append(f"invalid_lumiere_mask:{type(exc).__name__}")
            elif json_values is not None:
                measurement = _measurement(
                    json_values, source="DeepBraTumIA measured_volumes_in_mm3.json",
                    geometry_verified=False)
                measurement["path"] = str(volume_json)
                exclusions.append("missing_source_mask")
            else:
                exclusions.append("missing_lesion_measurement")
            mods = _modalities(image_dir, {"CT1": "CT1", "T1": "T1",
                                            "T2": "T2", "FLAIR": "FLAIR"})
            if not any(mods.values()):
                exclusions.append("missing_all_shared_modalities")
            rows.append({
                "cohort": "LUMIERE", "patient_id": patient, "visit": visit,
                "visit_id": _stable_id("LUMIERE", patient, visit),
                "day": float(parse_week_to_days(visit)),
                "timing_source": "curated week bin relative to pre-operative scan x 7",
                "timing_verified": False,
                "ordering_verified": True,
                "gap_precision": "7-day bins; same-week suffix gives order only",
                "modalities": mods, "measurement": measurement,
                "exclusions": exclusions,
            })
    return rows


def _sailor_sessions(subject_dir: Path) -> list[Path]:
    def key(path: Path) -> int:
        try:
            return int(path.name.split("-")[1])
        except (IndexError, ValueError):
            return 0
    return sorted((p for p in subject_dir.glob("ses-*") if p.is_dir()), key=key)


def sailor_visits(root: str) -> list[dict]:
    root_path = Path(root)
    rows = []
    for subject_dir in sorted(p for p in root_path.glob("sub-*") if p.is_dir()):
        sessions = _sailor_sessions(subject_dir)
        gap_path = subject_dir / "intervals-days.txt"
        try:
            gaps = [float(x) for x in gap_path.read_text().split()]
        except (OSError, ValueError):
            gaps = []
        days: list[float | None] = [0.0]
        for i in range(1, len(sessions)):
            previous = days[-1]
            days.append(previous + gaps[i - 1]
                        if previous is not None and i - 1 < len(gaps) else None)
        for i, session_dir in enumerate(sessions):
            measurements = {}
            exclusions = []
            shapes = set()
            for compartment, stem in SAILOR_MASKS.items():
                path = _find_nii(session_dir, stem)
                if path is None:
                    exclusions.append(f"missing_mask:{compartment}")
                    continue
                try:
                    m = measure_mask(path)
                    measurements[compartment] = m
                    shapes.add(tuple(m["shape"]))
                    if not m["finite"]:
                        exclusions.append(f"nonfinite_mask:{compartment}")
                    if not m["geometry_valid"]:
                        exclusions.append(f"invalid_geometry:{compartment}")
                except (OSError, ValueError) as exc:
                    exclusions.append(f"invalid_mask:{compartment}:{type(exc).__name__}")
            measurement = None
            sensitivity = {}
            if len(measurements) == len(COMPARTMENTS):
                vals = {k: measurements[k]["volume_mm3"] for k in COMPARTMENTS}
                measurement = _measurement(
                    vals, source="SAILOR ONCO native NIfTI masks",
                    geometry_verified=(len(shapes) == 1 and all(
                        measurements[k]["geometry_valid"] for k in COMPARTMENTS)))
                measurement["masks"] = measurements
                if len(shapes) != 1:
                    exclusions.append("compartment_geometry_mismatch")
                for compartment, stem in SAILOR_CL_MASKS.items():
                    cl_path = _find_nii(session_dir, stem)
                    if cl_path is not None:
                        sensitivity[compartment] = compare_masks(
                            measurements[compartment]["path"], cl_path)
            mods = _modalities(session_dir, {"CT1": "T1c", "T1": "T1",
                                              "T2": "T2", "FLAIR": "Flair"})
            if not any(mods.values()):
                exclusions.append("missing_all_shared_modalities")
            if i > 0 and (i - 1 >= len(gaps) or days[i] is None):
                exclusions.append("missing_interval")
            rows.append({
                "cohort": "SAILOR", "patient_id": subject_dir.name,
                "visit": session_dir.name,
                "visit_id": _stable_id("SAILOR", subject_dir.name, session_dir.name),
                "day": days[i], "timing_source": "intervals-days cumulative sum",
                "timing_verified": False,
                "ordering_verified": True,
                "gap_precision": "mixed exact/manual-estimated; see SAILOR history.txt",
                "modalities": mods, "measurement": measurement,
                "measurement_sensitivity": sensitivity,
                "exclusions": exclusions,
            })
    return rows


def build_pairs(visits: list[dict]) -> list[dict]:
    by_patient: dict[tuple[str, str], list[dict]] = {}
    for visit in visits:
        by_patient.setdefault((visit["cohort"], visit["patient_id"]), []).append(visit)
    pairs = []
    for (cohort, patient), rows in sorted(by_patient.items()):
        rows.sort(key=lambda x: (x["day"] if x["day"] is not None else float("inf"),
                                 x["visit"]))
        for source, target in zip(rows[:-1], rows[1:]):
            exclusions = []
            if source["measurement"] is None:
                exclusions.append("source_measurement_unavailable")
            if target["measurement"] is None:
                exclusions.append("target_measurement_unavailable")
            gap = (target["day"] - source["day"]
                   if target["day"] is not None and source["day"] is not None
                   else None)
            if gap is None or not math.isfinite(gap) or gap <= 0:
                exclusions.append("invalid_gap")
            pairs.append({
                "row_id": _stable_id(cohort, patient, source["visit"], target["visit"]),
                "cohort": cohort, "patient_id": patient,
                "source_visit": source["visit"], "target_visit": target["visit"],
                "source_visit_id": source["visit_id"],
                "target_visit_id": target["visit_id"],
                "gap_days": gap, "timing_verified": bool(
                    source["timing_verified"] and target["timing_verified"]),
                "ordering_verified": bool(
                    source.get("ordering_verified") and target.get("ordering_verified")),
                "target_version": TARGET_VERSION,
                "usable": not exclusions,
                "exclusions": exclusions,
            })
    return pairs


def _summary(visits: list[dict], pairs: list[dict]) -> dict:
    out = {}
    for cohort in ("LUMIERE", "SAILOR"):
        vv = [v for v in visits if v["cohort"] == cohort]
        pp = [p for p in pairs if p["cohort"] == cohort]
        out[cohort] = {
            "patients": len({v["patient_id"] for v in vv}),
            "visits": len(vv),
            "visits_with_measurement": sum(v["measurement"] is not None for v in vv),
            "geometry_verified_measurements": sum(
                bool(v["measurement"] and v["measurement"]["geometry_verified"])
                for v in vv),
            "consecutive_pairs": len(pp),
            "usable_pairs": sum(p["usable"] for p in pp),
            "ordering_verified_pairs": sum(p["ordering_verified"] for p in pp),
            "verified_timing_pairs": sum(p["timing_verified"] for p in pp),
        }
        if cohort == "SAILOR":
            for compartment in SAILOR_CL_MASKS:
                comparisons = [
                    v.get("measurement_sensitivity", {}).get(compartment)
                    for v in vv
                ]
                comparisons = [c for c in comparisons if c and c["same_grid"]]
                dices = [c["dice"] for c in comparisons]
                ratios = [c["volume_ratio_cl_to_onco"] for c in comparisons]
                out[cohort][f"{compartment}_cl_onco_n"] = len(comparisons)
                out[cohort][f"{compartment}_cl_onco_median_dice"] = (
                    float(np.median(dices)) if dices else None)
                out[cohort][f"{compartment}_cl_onco_median_volume_ratio"] = (
                    float(np.median(ratios)) if ratios else None)
    return out


def build_manifest(*, lumiere_root: str, lumiere_volumes: str,
                   lumiere_masks: str, sailor_root: str) -> dict:
    visits = (lumiere_visits(lumiere_root, lumiere_volumes, lumiere_masks)
              + sailor_visits(sailor_root))
    pairs = build_pairs(visits)
    return {
        "schema_version": SCHEMA_VERSION,
        "target_version": TARGET_VERSION,
        "compartments": list(COMPARTMENTS),
        "contract": {
            "units": "mm3",
            "primary_targets": ["future_log_volume", "delta_log_volume"],
            "primary_horizon": "next observed visit",
            "pair_eligibility_frozen": True,
            "pair_eligibility": (
                "consecutive curated visits; source and target have complete "
                "lesion-state-v1 measurements; positive ordered gap"),
            "primary_metrics": ["log_volume_mae", "delta_log_volume_mae",
                                "spearman", "persistence_relative_mae"],
            "classification_semantics": "none; these are measurements, not RANO",
            "timing_policy": (
                "ordering supports next-visit evaluation; gap_days is auxiliary "
                "and cannot support fixed-horizon claims"),
        },
        "summary": _summary(visits, pairs),
        "visits": visits,
        "pairs": pairs,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="build RANO-free lesion-state manifest")
    ap.add_argument("--lumiere-root", default="data/lumiere_preprocessed")
    ap.add_argument("--lumiere-volumes", default="data/autoseg/vols/Imaging")
    ap.add_argument("--lumiere-masks", default="data/autoseg/extracted/Imaging")
    ap.add_argument("--sailor-root", default=(
        "data/sailor/sailor_ebrains_pseud/derivatives/mni2009c-n-s"))
    ap.add_argument("--out", default="outputs/anatomy_harmonization.json")
    args = ap.parse_args(argv)
    manifest = build_manifest(lumiere_root=args.lumiere_root,
                              lumiere_volumes=args.lumiere_volumes,
                              lumiere_masks=args.lumiere_masks,
                              sailor_root=args.sailor_root)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    for cohort, summary in manifest["summary"].items():
        print(cohort, " ".join(f"{k}={v}" for k, v in summary.items()))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
