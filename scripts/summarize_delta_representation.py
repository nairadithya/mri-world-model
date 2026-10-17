"""Export aggregate-only R36 companions from saved assessments, without refitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.data.representation import digest, save_json


def average(values):
    first = values[0]
    if isinstance(first, dict):
        if any(set(v) != set(first) for v in values):
            raise ValueError("seed summary keys differ")
        return {key: average([v[key] for v in values]) for key in first
                if key not in ("per_patient", "per_patient_mae")}
    if isinstance(first, list):
        return [average(list(v)) for v in zip(*values, strict=True)]
    if first is None or isinstance(first, str):
        if any(v != first for v in values):
            raise ValueError("seed support or definition differs")
        return first
    if not all(np.isfinite(v) for v in values):
        raise ValueError("nonfinite aggregate")
    return float(np.mean(values))


def summarize(results: Path, report: Path, out: Path) -> dict:
    core = json.loads(report.read_text())
    if core["missing_companion_artifacts"]:
        raise ValueError("cannot publish an incomplete companion matrix")
    result = {"contract_sha256": core["contract_sha256"], "seeds": [42, 43, 44],
              "hypotheses": core["hypotheses"], "comparisons": core["comparisons"],
              "state_C-A": core["state_C-A"], "control_comparisons": core["control_comparisons"],
              "nuisance_comparisons": core["nuisance_comparisons"],
              "aggregation": "Mean of three seed errors, not ensemble predictions. Companion summaries are descriptive.",
              "limits": "Observed two-scan automated targets; both cohorts are developmental. No forecast or biological validity claim.",
              "cohorts": {}, "artifact_sha256": {"core_report": digest(report)},
              "generator_sha256": digest(Path(__file__))}
    for cohort in ("lumiere", "sailor"):
        result["cohorts"][cohort] = {}
        for arm in ("A", "B", "C", "D", "R35"):
            evaluations, ridges, nuisance = [], [], []
            for seed in result["seeds"]:
                directory = results / f"seed{seed}" / arm
                for filename in (f"{cohort}.json", f"ridge-{cohort}.json", "nuisance.json"):
                    path = directory / filename
                    result["artifact_sha256"][f"seed{seed}/{arm}/{filename}"] = digest(path)
                evaluations.append(json.loads((directory / f"{cohort}.json").read_text()))
                ridges.append(json.loads((directory / f"ridge-{cohort}.json").read_text()))
                nuisance.append(json.loads((directory / "nuisance.json").read_text()))
            if any(e["status"] != "complete" or e["arm"] != arm or e["seed"] != seed
                   or e["cohort"].lower() != cohort for e, seed in zip(evaluations, result["seeds"])):
                raise ValueError("assessment identity mismatch")
            if any(e["state"]["n_rows"] != len(e["states"]) or
                   any(v["n_rows"] != len(e["rows"]) for v in e["local"].values()) for e in evaluations):
                raise ValueError("assessment support mismatch")
            support = {}
            for method in evaluations[0]["local"]:
                records = []
                for evaluation in evaluations:
                    rows = evaluation["rows"]
                    records.append({
                        "changed_compartment_pairs": [sum(r[method]["changed_count"][c] > 0 for r in rows) for c in range(3)],
                        "appearance_voxels": sum(r[method]["appearance_support"] for r in rows),
                        "disappearance_voxels": sum(r[method]["disappearance_support"] for r in rows),
                        "mean_pair_changed_voxel_auprc": float(np.mean([r[method]["changed_voxel_auprc"] for r in rows
                                                                       if r[method]["changed_voxel_auprc"] is not None])),
                        "appearance_mae_nonempty_pair_mean": float(np.mean([r[method]["appearance_mae"] for r in rows
                                                                           if r[method]["appearance_mae"] is not None])),
                        "disappearance_mae_nonempty_pair_mean": float(np.mean([r[method]["disappearance_mae"] for r in rows
                                                                              if r[method]["disappearance_mae"] is not None]))})
                support[method] = average(records)
            result["cohorts"][cohort][arm] = {
                "state": average([e["state"] for e in evaluations]),
                "local": average([e["local"] for e in evaluations]),
                "local_support": support,
                "companions": average([e["descriptive_companions"] for e in evaluations]),
                "ridge": average([r["scores"] for r in ridges])}
            if arm in ("B", "C", "D"):
                result["cohorts"][cohort][arm]["reversal"] = {
                    key: float(np.mean([r[key] for e in evaluations for r in e["rows"]]))
                    for key in ("reversal_map_error", "reversal_volume_error")}
            if cohort == "lumiere":
                result["cohorts"][cohort][arm]["nuisance"] = average([n["conditions"] for n in nuisance])
    save_json(out, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.results, args.report, args.out)
    print(f"Aggregate-only companions written to {args.out}")
