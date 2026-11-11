"""Shared full-volume selection/evaluation for explicitly supervised cavity states."""
from collections import defaultdict
import hashlib
import time

import numpy as np
import torch

from .anatomy_readouts import anatomy_readouts, readout_contract
from .data import load_visit, metadata
from .losses import segmentation_metrics
from .state import SEQUENCES


def bounded_rows(rows, patients_per_dataset=4):
    """Choose patients by identifier hash and their middle visit, before outcomes."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("dataset", "LUMIERE"), row["patient"])].append(row)
    selected = []
    for dataset in sorted({k[0] for k in groups}):
        keys = sorted((k for k in groups if k[0] == dataset),
                      key=lambda k: hashlib.sha256(k[1].encode()).hexdigest())
        for key in keys[:patients_per_dataset]:
            visits = sorted(groups[key], key=lambda r: r["visit_order"])
            selected.append(visits[len(visits)//2])
    return selected


@torch.inference_mode()
def evaluate_visit(model, row, nuisance=False):
    started = time.perf_counter()
    case = load_visit(row, labels=True)
    x, y, b = case["images"].cuda(), case["labels"], case["support"][:, 0]
    classes = row.get("supervised_classes", 4)
    def infer(availability, scale=1.):
        state = model.encode(x*scale, dict(metadata(row), availability=availability, dataset=row.get("dataset", "LUMIERE")))
        out = model.read(state)
        return {k: v.float().cpu() if isinstance(v, torch.Tensor) else v for k, v in out.items()}, state.payload_bytes
    out, payload = infer([True]*4)
    # LUMIERE's comparable score conditions on its annotated four categories.
    # Native five-way predictions are reported separately, never silently merged.
    logits = out["logits"]
    maps = anatomy_readouts(logits, b)
    native = maps["native"]
    conditional = native if classes == 5 else maps["conditional_legacy"]
    scored = y >= 0
    metrics = segmentation_metrics(conditional, y, scored, range(1, classes))
    record = {"patient": row["patient"], "visit": row["visit"], "dataset": row.get("dataset", "LUMIERE"),
              "supervised_classes": classes, "segmentation": metrics,
              "anatomy_readouts": readout_contract(model.num_classes, row.get("dataset", "LUMIERE")),
              "model_id": model.model_id,
              "conditional_legacy_available_label_scores": segmentation_metrics(maps["conditional_legacy"], y, scored),
              "score_readout": "exclusive five-way" if classes == 5 else "conditional four-way; cavity unannotated",
              "native_available_label_scores": segmentation_metrics(native, y, scored),
              "common_foreground_three_compartments": segmentation_metrics(conditional, y, b),
              "native_predicted_cavity_ml": float(((native == 4) & scored).sum())/1000.,
              "core_in_reference_cavity_ml": float(((native == 2) & (y == 4) & scored).sum())/1000. if classes == 5 else None,
              "cavity_in_reference_core_ml": float(((native == 4) & (y == 2) & scored).sum())/1000. if classes == 5 else None,
              "state_payload_bytes": payload}
    voxel_ml = abs(float(np.linalg.det(np.asarray(metadata(row)["affine"])[:3, :3]))) / 1000
    for name, specification in record["anatomy_readouts"].items():
        specification["volumes_ml"] = {label: float((maps[name] == index).sum()) * voxel_ml
                                        for index, label in specification["labels"].items() if index != 0}
        specification["source"] = "Same observation decoder and state as image completion"
    del out, logits
    missing = {}
    # Keep lesion-detail population identical: cavity is not counted as tumor.
    lesion = (y > 0) & (y < 4) & b
    for channel, name in enumerate(SEQUENCES):
        out, _ = infer([c != channel for c in range(4)])
        residual = out["images"][:, channel]-case["images"][:, channel]
        error = residual.abs()
        gradients = []
        for axis in (1, 2, 3):
            a, z = [slice(None)]*4, [slice(None)]*4
            a[axis], z[axis] = slice(1, None), slice(None, -1)
            a, z = tuple(a), tuple(z)
            valid = b[a] & b[z]
            gradients.append(float((residual[a]-residual[z]).abs()[valid].mean()))
        hidden_seg = out["logits"][:, :classes].argmax(1)
        if model.num_classes == 4:
            hidden_seg = hidden_seg.masked_fill(~b, 0)
        missing[name] = {"brain_mae": float(error[b].mean()),
                         "lesion_mae": float(error[lesion].mean()) if lesion.any() else None,
                         "cavity_mae": float(error[(y == 4) & b].mean()) if classes == 5 and ((y == 4) & b).any() else None,
                         "gradient_mae": float(np.mean(gradients)),
                         "segmentation": segmentation_metrics(hidden_seg, y, scored, range(1, classes))}
        del out
    record["missing"] = missing
    if nuisance:
        out, _ = infer([True]*4, 1.05)
        perturbed = out["logits"][:, :classes].argmax(1)
        if model.num_classes == 4:
            perturbed = perturbed.masked_fill(~b, 0)
        record["intensity_times_1_05"] = segmentation_metrics(perturbed, y, scored, range(1, classes))
    record["seconds"] = time.perf_counter()-started
    return record


def summarize(records):
    by_dataset = {}
    for dataset in sorted({r["dataset"] for r in records}):
        groups = defaultdict(list)
        for r in records:
            if r["dataset"] == dataset:
                groups[r["patient"]].append(r)
        def average(fn):
            values = []
            for visits in groups.values():
                vs = [v for r in visits if (v := fn(r)) is not None]
                if vs:
                    values.append(float(np.mean(vs)))
            return float(np.mean(values)) if values else None
        first = next(iter(groups.values()))[0]
        by_dataset[dataset] = {
            "patients": len(groups), "visits": sum(map(len, groups.values())),
            "dice": average(lambda r: np.mean([c["dice"] for c in r["segmentation"].values()])),
            "compartments": {c: {k: average(lambda r, c=c, k=k: abs(r["segmentation"][c][k]) if r["segmentation"][c][k] is not None else None)
                                for k in ("dice", "nonempty_dice", "volume_error_ml", "empty_reference_fp_ml")}
                             for c in first["segmentation"]},
            "missing": {name: {k: average(lambda r, name=name, k=k: r["missing"][name][k])
                                for k in ("brain_mae", "lesion_mae", "cavity_mae", "gradient_mae")} for name in SEQUENCES},
            "common_foreground_three_compartment_dice": average(lambda r: np.mean([v["dice"] for v in r["common_foreground_three_compartments"].values()])),
            "native_available_label_dice": average(lambda r: np.mean([v["dice"] for v in r["native_available_label_scores"].values()])),
            "native_cavity_ml": average(lambda r: r["native_predicted_cavity_ml"]),
            "core_in_reference_cavity_ml": average(lambda r: r["core_in_reference_cavity_ml"]),
            "cavity_in_reference_core_ml": average(lambda r: r["cavity_in_reference_core_ml"]),
            "mean_visit_seconds": average(lambda r: r["seconds"])}
        for key, image_key in (("mae", "brain_mae"), ("lesion_mae", "lesion_mae"), ("gradient_mae", "gradient_mae")):
            by_dataset[dataset][key] = average(lambda r, k=image_key: np.mean([v[k] for v in r["missing"].values() if v[k] is not None]) if any(v[k] is not None for v in r["missing"].values()) else None)
    aggregate = {key: float(np.mean([r[key] for r in by_dataset.values() if r[key] is not None]))
                 for key in ("dice", "mae", "lesion_mae", "gradient_mae")}
    aggregate["selection_score"] = 1-aggregate["dice"]+.25*aggregate["lesion_mae"]+.1*aggregate["mae"]+.1*aggregate["gradient_mae"]
    return dict(aggregate, by_dataset=by_dataset)


def select_full(model, rows):
    model.eval()
    records = [evaluate_visit(model, row) for row in bounded_rows(rows)]
    model.train()
    return dict(summarize(records), records=records,
                scope="bounded full-volume selection: four patients per dataset, middle visit; conditional LUMIERE / five-way MU")
