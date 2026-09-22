"""Support-only SAILOR calibration for the learned lesion forecaster.

Zero-shot remains the primary transfer result. Query patients are fixed before
calibration; support sets are nested within each seed and never overlap query.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge

from src.model.lesion_forecaster import LesionTransitionForecaster


def patient_rows(cache, model):
    rows = {}
    model.eval()
    with torch.no_grad():
        for patient, item in cache.items():
            source = item["volumes"].float()
            out = model(item["features"].float(), item["gaps"].float(), source)
            target = source[1:]
            gap = torch.log1p(item["gaps"][1:].float()).unsqueeze(1) / 8.0
            rows[patient] = {"source": source[:-1].numpy(),
                             "target": target.numpy(),
                             "raw": out["prediction"].numpy(),
                             "gap": gap.numpy()}
    return rows


def metric(rows, patients, predictions=None):
    learned, persistence = [], []
    for patient in patients:
        row = rows[patient]
        pred = row["raw"] if predictions is None else predictions[patient]
        learned.append(np.abs(pred - row["target"]).mean())
        persistence.append(np.abs(row["source"] - row["target"]).mean())
    mae, base = float(np.mean(learned)), float(np.mean(persistence))
    diff = np.asarray(learned) - np.asarray(persistence)
    draws = np.random.default_rng(42).choice(
        diff, (10000, len(diff)), replace=True).mean(1)
    return {"patient_uniform_log_volume_mae": mae,
            "persistence_mae": base, "persistence_relative_mae": mae / base,
            "mae_difference": float(diff.mean()),
            "mae_difference_95ci": [float(x) for x in np.quantile(draws, [0.025, 0.975])]}


def fit_bias(rows, support):
    residual = np.concatenate([rows[p]["target"] - rows[p]["raw"] for p in support])
    return residual.mean(0)


def fit_ridge(rows, support, alpha=10.0):
    x, y = [], []
    for patient in support:
        row = rows[patient]
        x.append(np.concatenate([row["raw"] - row["source"], row["source"], row["gap"]], 1))
        y.append(row["target"] - row["source"])
    return Ridge(alpha=alpha).fit(np.concatenate(x), np.concatenate(y))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="checkpoints/learned_lesion_eval_cache.pt")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="outputs/p2_learned_lesion_adaptation.json")
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args(argv)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)["sailor"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = LesionTransitionForecaster(); model.load_state_dict(checkpoint["forecaster"])
    rows = patient_rows(cache, model); patients = sorted(rows)
    query = patients[::4]
    support_pool = [patient for patient in patients if patient not in set(query)]
    sizes = [size for size in (1, 3, 5, 10, 20) if size <= len(support_pool)]
    result = {"schema_version": 1, "query_patients": query,
              "support_pool": support_pool, "nested_support_sizes": sizes,
              "zero_shot_query": metric(rows, query), "methods": {}}
    collected = {method: {size: [] for size in sizes} for method in ("bias", "ridge")}
    support_orders = {}
    for seed in range(args.seeds):
        order = list(support_pool); np.random.default_rng(seed).shuffle(order)
        support_orders[str(seed)] = order
        for size in sizes:
            support = order[:size]
            bias = fit_bias(rows, support)
            bias_predictions = {p: rows[p]["raw"] + bias for p in query}
            collected["bias"][size].append(metric(rows, query, bias_predictions))
            ridge = fit_ridge(rows, support)
            ridge_predictions = {}
            for p in query:
                row = rows[p]
                x = np.concatenate([row["raw"] - row["source"], row["source"], row["gap"]], 1)
                ridge_predictions[p] = row["source"] + ridge.predict(x)
            collected["ridge"][size].append(metric(rows, query, ridge_predictions))
    for method, by_size in collected.items():
        result["methods"][method] = {}
        for size, values in by_size.items():
            relative = np.array([value["persistence_relative_mae"] for value in values])
            mae = np.array([value["patient_uniform_log_volume_mae"] for value in values])
            result["methods"][method][str(size)] = {
                "relative_mae_median": float(np.median(relative)),
                "relative_mae_p10_p90": [float(x) for x in np.quantile(relative, [0.1, 0.9])],
                "mae_median": float(np.median(mae)), "runs": values}
    result["support_orders"] = support_orders
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("zero-shot fixed query", result["zero_shot_query"])
    for method, by_size in result["methods"].items():
        for size, value in by_size.items():
            print(f"{method:6s} K={size:>2s} rel median={value['relative_mae_median']:.3f} "
                  f"p10-p90={value['relative_mae_p10_p90']}")


if __name__ == "__main__": main()
