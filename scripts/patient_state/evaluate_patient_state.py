#!/usr/bin/env python3
"""Full-volume current anatomy and four missing-sequence evaluations, without feature caches."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from src.patient_state.data import catalogue, load_visit, metadata
from src.patient_state.losses import segmentation_metrics
from src.patient_state.observation import JointObservation
from src.patient_state.state import write_json, SEQUENCES, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--group", choices=("selection", "scoring"), default="selection")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Instrument check only, not cohort evaluation")
    parser.add_argument("--r69", type=Path, help="Frozen R69 reference run")
    parser.add_argument("--r71", type=Path, help="Frozen R71 references; requires --r69")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = JointObservation.from_checkpoint(args.checkpoint)
    reference, adapted = None, None
    if args.r69 or args.r71:
        if not args.r69 or not args.r71:
            parser.error("Supply both reference run directories")
        from src.patient_state.bridge import LegacyObservation, LegacyAdaptedAnatomy
        reference = LegacyObservation(args.r69, args.r71)
        adapted = LegacyAdaptedAnatomy(args.r71)
    rows = catalogue(args.catalogue)["groups"][args.group]
    if args.limit:
        rows = rows[:args.limit]
    args.out.mkdir(parents=True, exist_ok=False)
    records = []
    for index, row in enumerate(rows):
        started = time.perf_counter()
        case = load_visit(row, labels=True)
        x, y, b = case["images"].cuda(), case["labels"], case["support"]
        meta = metadata(row)
        state = model.encode(x, meta)
        payload = state.payload_bytes
        out = model.read(state)
        logits = out["logits"].cpu()
        seg = segmentation_metrics(logits.argmax(1), y, b[:, 0])
        del state, out, logits
        missing = {}
        for channel, name in enumerate(SEQUENCES):
            hidden_meta = dict(meta, availability=[c != channel for c in range(4)])
            state = model.encode(x, hidden_meta)
            out = model.read(state)
            predicted = out["images"][:, channel].float().cpu()
            target = case["images"][:, channel]
            error = (predicted-target).abs()
            supported, lesion = b[:, 0], b[:, 0] & (y > 0)
            grad = []
            for axis in (1, 2, 3):
                a, z = [slice(None)]*4, [slice(None)]*4
                a[axis], z[axis] = slice(1, None), slice(None, -1)
                a, z = tuple(a), tuple(z)
                valid = supported[a] & supported[z]
                residual = predicted-target
                grad.append(float((residual[a]-residual[z]).abs()[valid].mean()))
            missing[name] = {"brain_mae": float(error[supported].mean()),
                             "lesion_mae": float(error[lesion].mean()) if lesion.any() else None,
                             "gradient_mae": float(np.mean(grad)),
                             "segmentation": segmentation_metrics(out["logits"].cpu().argmax(1), y, b[:, 0])}
            del state, out
        # Fixed nuisance perturbation on all available images, after selection.
        state = model.encode(x*1.05, meta)
        perturbed = model.read(state)["logits"].cpu().argmax(1)
        nuisance = segmentation_metrics(perturbed, y, b[:, 0])
        references = {}
        if reference is not None and row.get("dataset", "LUMIERE") == "LUMIERE":
            reference_state = reference.encode(x, meta)
            reference_outputs = reference.read(reference_state)
            references["r71_learned_panel_C"] = segmentation_metrics(reference_outputs["logits"].cpu().argmax(1), y, b[:, 0])
            references["r71_adapted_panel_C"] = segmentation_metrics(adapted(x).cpu().argmax(1), y, b[:, 0])
            reference_error = (reference_outputs["images"][:, 0].float().cpu()-case["images"][:, 3]).abs()
            references["r69_missing_FLAIR"] = {"brain_mae": float(reference_error[b[:, 0]].mean()),
                "lesion_mae": float(reference_error[b[:, 0] & (y > 0)].mean()) if (b[:, 0] & (y > 0)).any() else None}
            del reference_state, reference_outputs
        record = {"patient": row["patient"], "visit": row["visit"], "dataset": row.get("dataset", "LUMIERE"),
                  "segmentation": seg, "missing": missing, "intensity_times_1_05": nuisance,
                  "matched_references": references,
                  "state_payload_bytes": payload, "seconds": time.perf_counter()-started}
        records.append(record)
        write_json(args.out / (row["stem"]+".json"), record)
        print(f"Evaluated {index+1}/{len(rows)} complete visits", flush=True)
        del x, case, state
    summaries = {}
    for dataset in sorted({r["dataset"] for r in records}):
        groups = defaultdict(list)
        for record in records:
            if record["dataset"] == dataset:
                groups[record["patient"]].append(record)
        def average(fn):
            patient_values = []
            for visits in groups.values():
                values = [v for r in visits if (v := fn(r)) is not None]
                if values:
                    patient_values.append(float(np.mean(values)))
            return float(np.mean(patient_values)) if patient_values else None
        summaries[dataset] = {"patients": len(groups), "visits": sum(map(len, groups.values())),
            "segmentation_dice": average(lambda r: np.mean([c["dice"] for c in r["segmentation"].values()])),
            "compartments": {c: {metric: average(lambda r, c=c, metric=metric: abs(r["segmentation"][c][metric]) if r["segmentation"][c][metric] is not None else None)
                                  for metric in ("dice", "nonempty_dice", "volume_error_ml", "empty_reference_fp_ml")} for c in ("1", "2", "3")},
            "missing_sequences": {name: {metric: average(lambda r, name=name, metric=metric: r["missing"][name][metric]) for metric in ("brain_mae", "lesion_mae", "gradient_mae")} for name in SEQUENCES},
            "mean_visit_seconds": average(lambda r: r["seconds"]), "mean_state_bytes": average(lambda r: r["state_payload_bytes"])}
        if dataset == "LUMIERE" and reference is not None:
            summaries[dataset]["matched_panel_C_dice"] = {
                "joint": average(lambda r: np.mean([c["dice"] for c in r["missing"]["FLAIR"]["segmentation"].values()])),
                **{name: average(lambda r, name=name: np.mean([c["dice"] for c in r["matched_references"][name].values()]))
                   for name in ("r71_learned_panel_C", "r71_adapted_panel_C")}}
            summaries[dataset]["r69_missing_FLAIR"] = {metric: average(lambda r, metric=metric: r["matched_references"]["r69_missing_FLAIR"][metric]) for metric in ("brain_mae", "lesion_mae")}
    failures = {}
    for dataset in summaries:
        available = [r for r in records if r["dataset"] == dataset]
        failures[dataset] = {
            "lowest_segmentation_dice": sorted(available, key=lambda r: np.mean([c["dice"] for c in r["segmentation"].values()]))[:3],
            "largest_missing_flair_lesion_error": sorted([r for r in available if r["missing"]["FLAIR"]["lesion_mae"] is not None], key=lambda r: -r["missing"]["FLAIR"]["lesion_mae"])[:3]}
    write_json(args.out / "failure-candidates.json", failures)
    write_json(args.out / "aggregate.json", {"scope": "instrument check" if args.limit else "full-volume development evaluation",
        "group": args.group, "model_sha256": digest(args.checkpoint), "datasets": summaries,
        "usefulness_established": False, "next": "Review full-volume accuracy, baseline comparisons, failure cases and training-boundary trend"})


if __name__ == "__main__":
    main()
