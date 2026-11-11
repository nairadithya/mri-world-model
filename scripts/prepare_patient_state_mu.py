#!/usr/bin/env python3
"""Prepare MU fitting/selection observations only; never open reserved patients."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nibabel as nib
import numpy as np
from src.patient_state.labels import harmonize_labels
from src.patient_state.state import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mu", type=Path, required=True)
    parser.add_argument("--lumiere-catalogue", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    membership_path = args.mu / "membership.json"
    if digest(membership_path) != "4df898761ecdb299bc87f99749fbebc8faab294043d454c4055850fcc676e2d6":
        raise ValueError("MU membership changed")
    members = json.loads(membership_path.read_text())
    catalogue = json.loads(args.lumiere_catalogue.read_text())
    for rows in catalogue["groups"].values():
        for row in rows:
            row["dataset"] = "LUMIERE"
    catalogue["cohort"] = "LUMIERE+MU"
    catalogue["mu_membership_sha256"] = digest(membership_path)
    catalogue["mu_membership_path"] = str(membership_path)
    args.out.mkdir(parents=True, exist_ok=True)
    counts, excluded = {}, []
    for group in ("fitting", "selection"):
        count = 0
        for patient in members[group]:
            if patient in members["reserved"]:
                raise ValueError("Reserved patient cannot be prepared")
            patient_dir = args.mu / "images" / patient
            for visit in sorted(patient_dir.glob("Timepoint_*")):
                prefix = f"{patient}_{visit.name}"
                paths = [visit / f"{prefix}_{suffix}.nii.gz" for suffix in
                         ("brain_t1c", "brain_t1n", "brain_t2w", "brain_t2f", "tumorMask")]
                if not all(p.is_file() for p in paths):
                    excluded.append({"patient": patient, "visit": visit.name, "group": group,
                                     "reason": "missing image or anatomical target", "missing": [p.name for p in paths if not p.is_file()]})
                    continue
                images = [nib.load(p) for p in paths]
                first = images[0]
                if any(im.shape != first.shape or not np.allclose(im.affine, first.affine, atol=1e-5, rtol=0) for im in images):
                    raise ValueError("Noncommon within-visit geometry")
                if not np.allclose(nib.affines.voxel_sizes(first.affine), 1., atol=1e-5):
                    raise ValueError("Expected released 1mm MU grid")
                transform = nib.orientations.ornt_transform(nib.orientations.io_orientation(first.affine), nib.orientations.axcodes2ornt(("L", "A", "S")))
                affine = first.affine @ nib.orientations.inv_ornt_aff(transform, first.shape)
                directory = args.out / group
                directory.mkdir(exist_ok=True)
                row = {"patient": patient, "visit": visit.name, "stem": prefix, "dataset": "MU",
                       "visit_order": [int(visit.name.split("_")[-1]), 0],
                       "time_unit": "visit_order_only_not_elapsed_time", "affine": affine.tolist(),
                       "shape_xyz": np.asarray(first.shape)[np.argsort(transform[:, 0])].tolist(),
                       "label_source": "reviewed MU; cavity excluded from shared segmentation"}
                for key in ("images", "support", "mask"):
                    row[key] = str(directory / f"{prefix}_{key}.npy")
                if all(Path(row[k]).exists() for k in ("images", "support", "mask")):
                    shape = row["shape_xyz"]
                    expected = (shape[2], shape[0], shape[1])
                    for key in ("images", "support", "mask"):
                        if np.load(row[key], mmap_mode="r").shape != ((4, *expected) if key == "images" else expected):
                            raise ValueError("Incomplete preparation cache")
                    catalogue["groups"][group].append(row)
                    count += 1
                    continue
                arrays = []
                support = None
                for i, im in enumerate(images[:4]):
                    a = nib.orientations.apply_orientation(np.asarray(im.dataobj, dtype=np.float32), transform)
                    if not np.isfinite(a).all() or not (a != 0).any():
                        raise ValueError("Nonfinite or empty MU MRI")
                    foreground = a != 0
                    if i == 0:
                        support = foreground
                    sd = float(a[foreground].std(dtype=np.float64))
                    if sd <= 0:
                        raise ValueError("Degenerate MU normalization")
                    z = np.zeros_like(a)
                    z[foreground] = (a[foreground]-a[foreground].mean(dtype=np.float64))/sd
                    arrays.append(z.transpose(2, 0, 1).astype(np.float16))
                raw_mask = nib.orientations.apply_orientation(np.asanyarray(images[-1].dataobj), transform)
                labels = harmonize_labels(raw_mask, "MU")
                for key, array in (("images", np.stack(arrays)), ("support", support.transpose(2, 0, 1).astype(np.uint8)), ("mask", labels.transpose(2, 0, 1))):
                    path = directory / f"{prefix}_{key}.npy"
                    np.save(path, array)
                    row[key] = str(path)
                catalogue["groups"][group].append(row)
                count += 1
            print(json.dumps({"group": group, "prepared_visits": count}), flush=True)
        counts[group] = {"assigned_patients": len(members[group]), "eligible_patients": len({r["patient"] for r in catalogue["groups"][group] if r["dataset"] == "MU"}), "visits": count}
    write_json(args.out / "catalogue.json", catalogue)
    write_json(args.out / "preparation.json", {"MU": counts, "reserved_images_opened": 0,
        "MU_scoring_images_opened": 0, "cavity_target": "ignore -1, not background", "orientation": "LAS 1mm",
        "forecasting": "MU timing not restored; observation tasks only", "exclusions": excluded})


if __name__ == "__main__":
    main()
