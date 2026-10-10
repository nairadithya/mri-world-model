"""Audit state gains for within-patient temporal specificity and empty targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.representation import digest, save_json
from src.harness.analysis.representation import patient_errors


def audit(predictions_path: str, out: str, trials: int = 100) -> dict:
    if trials < 1:
        raise ValueError("at least one shuffle trial is required")
    with np.load(predictions_path, allow_pickle=False) as data:
        truth = data["state_truth"]
        prediction = data["state_prediction"]
        mean = data["state_mean"]
        patients = data["state_patients"]
    original = patient_errors(truth, prediction, patients)
    unique = sorted(original)
    rng = np.random.default_rng(2026)
    shuffled_errors = []
    for _ in range(trials):
        shuffled = prediction.copy()
        for patient in unique:
            indexes = np.flatnonzero(patients == patient)
            shuffled[indexes] = prediction[rng.permutation(indexes)]
        errors = patient_errors(truth, shuffled, patients)
        shuffled_errors.append([errors[p] for p in unique])
    average_shuffle = np.asarray(shuffled_errors).mean(0)
    difference = np.asarray([original[p] for p in unique]) - average_shuffle
    draws = difference[rng.integers(0, len(unique), (10000, len(unique)))].mean(1)
    strata = {}
    for index, name in enumerate(("necrotic_non_enhancing", "enhancing", "edema_flair")):
        strata[name] = {}
        for label, mask in (("empty", truth[:, index] == 0), ("nonempty", truth[:, index] > 0)):
            if not mask.any():
                strata[name][label] = {"n_rows": 0}
                continue
            groups = patients[mask]
            a = patient_errors(truth[mask, index:index + 1], prediction[mask, index:index + 1], groups)
            b = patient_errors(truth[mask, index:index + 1], mean[mask, index:index + 1], groups)
            strata[name][label] = {"n_rows": int(mask.sum()), "n_patients": len(a),
                                   "prediction_patient_uniform_mae": float(np.mean(list(a.values()))),
                                   "mean_patient_uniform_mae": float(np.mean(list(b.values())))}
    result = {"predictions_sha256": digest(predictions_path), "shuffle_trials": trials, "seed": 2026,
              "state_patient_uniform_mae": float(np.mean(list(original.values()))),
              "within_patient_shuffled_mae": float(average_shuffle.mean()),
              "original_minus_shuffle": float(difference.mean()),
              "paired_patient_95ci": np.quantile(draws, [0.025, 0.975]).tolist(),
              "per_patient": {p: {"original": original[p], "mean_shuffle": float(average_shuffle[i])}
                              for i, p in enumerate(unique)}, "target_strata": strata,
              "interpretation": "evaluation permutation diagnostic; target-defined strata are reporting only"}
    save_json(out, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = audit(args.predictions, args.out)
    print(json.dumps({k: result[k] for k in ("state_patient_uniform_mae", "within_patient_shuffled_mae",
                                           "original_minus_shuffle", "paired_patient_95ci")}, sort_keys=True))
