"""Reuse the released-atlas LUMIERE preparation and patient allocation."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
import torch

AFFINE = [[-1., 0., 0., 90.], [0., 1., 0., -126.], [0., 0., 1., -72.], [0., 0., 0., 1.]]


def catalogue(path):
    value = json.loads(Path(path).read_text())
    if value["cohort"] not in ("LUMIERE", "LUMIERE+MU") or value["membership_sha256"] != "0943aa68a363c77f0f57c411545e34fe94917a727d1a3202700397a383e2aea6":
        raise ValueError("Expected fixed LUMIERE development allocation")
    if value["cohort"] == "LUMIERE+MU":
        from .state import digest
        member_path = value["mu_membership_path"]
        if digest(member_path) != "4df898761ecdb299bc87f99749fbebc8faab294043d454c4055850fcc676e2d6":
            raise ValueError("MU membership changed")
        members = json.loads(Path(member_path).read_text())
        for group, rows in value["groups"].items():
            for row in rows:
                if row.get("dataset") == "MU" and (group not in ("fitting", "selection") or row["patient"] not in members[group] or row["patient"] in members["reserved"]):
                    raise ValueError("MU patient outside authorized fitting/selection split")
    seen = set()
    for group, rows in value["groups"].items():
        patients = {r["patient"] for r in rows}
        if patients & seen:
            raise ValueError("Patient overlap across splits")
        seen |= patients
    return value


def metadata(row):
    if row.get("dataset") == "MU":
        return {"patient": row["patient"], "visit": row["visit"], "nominal_week": None,
                "visit_order": row["visit_order"], "time_unit": row.get("time_unit", "visit_order_only_not_elapsed_time"),
                "diagnosis_relative_day": row.get("diagnosis_relative_day"),
                "availability": [True]*4, "affine": row["affine"], "shape_xyz": row["shape_xyz"],
                "geometry": "released MU 1mm, reoriented LAS",
                "normalization": "per-sequence nonzero-foreground z-score; zero background",
                "label_source": row["label_source"]}
    return {"patient": row["patient"], "visit": row["visit"],
            "nominal_week": row["visit_order"][0], "visit_order": row["visit_order"],
            "time_unit": "nominal_week", "availability": [True] * 4,
            "affine": AFFINE, "shape_xyz": [182, 218, 182],
            "geometry": "released DeepBraTumIA LAS atlas, 1 mm; upstream full-panel processing",
            "normalization": "per-sequence nonzero-foreground z-score; zero background",
            "label_source": "automated DeepBraTumIA development proxy"}


def load_visit(row, labels=False):
    # Preparation uses ZXY storage. Models and NIfTI outputs use XYZ.
    raw = np.load(row["images"], mmap_mode="r", allow_pickle=False)
    shape = row.get("shape_xyz", [182, 218, 182])
    if raw.shape != (4, shape[2], shape[0], shape[1]):
        raise ValueError("Invalid prepared geometry")
    x = np.array(raw.transpose(0, 2, 3, 1), dtype=np.float32, order="C")
    support = np.array(np.load(row["support"], mmap_mode="r").transpose(1, 2, 0), dtype=bool)
    if not np.isfinite(x).all() or not support.any():
        raise ValueError("Nonfinite image or empty support")
    result = {"images": torch.from_numpy(x)[None], "support": torch.from_numpy(support)[None, None]}
    if labels:
        y = np.array(np.load(row["mask"], mmap_mode="r").transpose(1, 2, 0), dtype=np.int64)
        if not np.isin(y, (-1, 0, 1, 2, 3, 4) if row.get("supervised_classes") == 5 else (-1, 0, 1, 2, 3)).all():
            raise ValueError("Unknown labels for declared supervision schema")
        result["labels"] = torch.from_numpy(y)[None]
    return result


def horizon_inventory(value, horizons=(12, 24), tolerance=4):
    from .timing import elapsed_weeks
    result = {}
    for group, rows in value["groups"].items():
        grouped = defaultdict(list)
        for row in rows:
            if row.get("time_unit") == "visit_order_only_not_elapsed_time":
                continue
            grouped[row["patient"]].append(row)
        counts = {}
        for h in horizons:
            pairs, patients, history_pairs, exact = 0, set(), 0, 0
            for patient, visits in grouped.items():
                visits.sort(key=lambda r: r["visit_order"])
                for i, visit in enumerate(visits):
                    gaps = [elapsed_weeks(r) - elapsed_weeks(visit) for r in visits[i+1:]]
                    eligible = [g for g in gaps if g > 0 and abs(g-h) <= tolerance]
                    if eligible:
                        pairs += 1
                        patients.add(patient)
                        history_pairs += int(i > 0)
                        exact += int(h in eligible)
            counts[str(h)] = {"cutoffs": pairs, "patients": len(patients), "with_history": history_pairs,
                              "exact_nominal_horizon_cutoffs": exact, "tolerance_nominal_weeks": tolerance}
        result[group] = counts
    return result
