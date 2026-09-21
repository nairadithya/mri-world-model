"""Generate the current-schema P0 evaluation scorecard.

The scorecard is deliberately frozen to the locked unseen protocol.  It
reports pooled and patient-uniform metrics, paired patient-cluster intervals
for prespecified comparisons, and readout seed/width sensitivity.  It does
not touch the historical ``final`` slice.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.harness.config import load_config
from src.harness.data.protocols import LockedProtocol
from src.harness.data.tasks import TASKS
from src.harness.encode.cache import load as load_cache
from src.harness.eval.aggregate import patient_bootstrap, percentile_ci
from src.harness.eval.evaluator import ReadoutEvaluator
from src.harness.eval.metrics import compute_metrics


BOOT = 10_000
SEEDS = (0, 42, 2026)
WIDTHS = (0, 256, 512)

# These are the complete local classification views needed for the P0
# assessment/forecast contract.  Pairings below are kept smaller and
# prespecified so the scorecard does not become a post-hoc comparison search.
EVALUATIONS = {
    "rano4_forecast": ("states_forecast", "fused", "vision", "clinical",
                        "ema_z", "z"),
    "rano4_current": ("fused", "vision", "clinical", "states_current",
                       "ema_z", "z"),
    "pd_forecast": ("states_forecast", "fused", "vision", "clinical"),
    "response_forecast": ("states_forecast", "fused", "vision", "clinical"),
}

PAIRED = {
    "rano4_forecast": (("states_forecast", "fused"),
                        ("states_forecast", "clinical")),
    "rano4_current": (("fused", "vision"), ("fused", "clinical")),
    "pd_forecast": (("states_forecast", "fused"),),
    "response_forecast": (("states_forecast", "fused"),),
}


def _git_state() -> dict:
    def run(*args):
        return subprocess.check_output(args, text=True).strip()

    sha = run("git", "rev-parse", "HEAD")
    dirty = bool(run("git", "status", "--porcelain"))
    return {"git_sha": sha, "git_dirty": dirty}


def _metric_requires_scores(name: str) -> bool:
    from src.harness import METRICS

    metric = METRICS.get(name)
    metric = metric() if isinstance(metric, type) else metric
    return getattr(metric, "requires", "labels") == "scores"


def _paired_metric(name: str, n_cls: int):
    scores = _metric_requires_scores(name)

    def fn(pred, y):
        labels = pred.argmax(dim=-1) if scores else pred
        out = compute_metrics([name], y, labels,
                              scores=pred if scores else None, n_cls=n_cls)
        if name not in out:
            raise ValueError(f"paired metric must be scalar: {name!r}")
        return out[name]

    return fn


def _result_dict(result) -> dict:
    return {
        "name": result.name,
        "metrics": result.metrics,
        "ci": {k: list(v) for k, v in result.ci.items()},
        "patient_uniform": result.patient_metrics,
        "fold_values": result.fold_values,
        "n_pat": result.n_pat,
        "n_rows": result.n_rows,
        "majority": result.majority,
        "meta": result.meta,
    }


def _run(task_name, view, patients, protocol, *, boot, seed=42, hidden=256):
    task = TASKS.get(task_name)()
    readout = "linear" if hidden == 0 else "mlp"
    ev = ReadoutEvaluator(
        task, protocol, view=view, readout=readout, hidden=hidden,
        seed=seed, standardize=False, train_pool="unseen", cohort="unseen",
        metrics=(
            ("macro_f1", "accuracy", "per_class_recall")
            if task.frame != "binary" else
            ("macro_f1", "accuracy", "auc", "auprc", "brier", "logloss",
             "ece", "sensitivity", "specificity", "prevalence")
        ),
        boot=boot, metric_seed=42,
    )
    return ev.run(patients)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="checkpoints/p0_probe_cache.pt")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--out", default="outputs/p0_scorecard.json")
    ap.add_argument("--boot", type=int, default=BOOT)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cache = load_cache(args.cache, warn_provenance=True, require_current=True)
    patients = cache["patients"]
    protocol = LockedProtocol("info/eval_folds.json")

    results = {}
    live = {}
    for task_name, views in EVALUATIONS.items():
        for view in views:
            result = _run(task_name, view, patients, protocol,
                          boot=args.boot)
            key = f"{task_name}:{view}"
            results[key] = _result_dict(result)
            live[key] = result
            print(f"{key}: pooled macro_f1="
                  f"{result.metrics.get('macro_f1', float('nan')):.4f} "
                  f"patient_uniform="
                  f"{result.patient_metrics.get('macro_f1', float('nan')):.4f}",
                  flush=True)

    paired = []
    for task_name, pairs in PAIRED.items():
        for left_name, right_name in pairs:
            left = live[f"{task_name}:{left_name}"]
            right = live[f"{task_name}:{right_name}"]
            for metric in ("macro_f1", "accuracy"):
                if metric not in left.metrics or metric not in right.metrics:
                    continue
                score_mode = _metric_requires_scores(metric)
                a = left.score_oof if score_mode else left.oof
                b = right.score_oof if score_mode else right.oof
                _, diffs = patient_bootstrap(
                    a, b, boot=args.boot, seed=42,
                    metric_fn=_paired_metric(metric, left.meta["n_cls"]),
                )
                lo, hi = percentile_ci(diffs)
                paired.append({
                    "task": task_name,
                    "left": left_name,
                    "right": right_name,
                    "metric": metric,
                    "left_minus_right": left.metrics[metric] - right.metrics[metric],
                    "patient_uniform_left_minus_right":
                        left.patient_metrics.get(metric, float("nan")) -
                        right.patient_metrics.get(metric, float("nan")),
                    "ci": [lo, hi],
                    "significant": bool(lo > 0 or hi < 0),
                    "n_pat": left.n_pat,
                    "n_rows": left.n_rows,
                })

    sensitivity = []
    for width in WIDTHS:
        for seed in SEEDS:
            result = _run("rano4_forecast", "states_forecast", patients,
                           protocol, boot=0, seed=seed, hidden=width)
            sensitivity.append({
                "seed": seed,
                "hidden": width,
                "pooled_macro_f1": result.metrics.get("macro_f1"),
                "patient_uniform_macro_f1":
                    result.patient_metrics.get("macro_f1"),
                "fold_values": result.fold_values,
            })
            print(f"sensitivity seed={seed} hidden={width}: "
                  f"macro_f1={result.metrics.get('macro_f1', float('nan')):.4f}",
                  flush=True)

    record = {
        "schema": "p0_scorecard_v1",
        "created": dt.date.today().isoformat(),
        "provenance": {
            **_git_state(),
            "cache": os.path.abspath(args.cache),
            "cache_provenance": cache.get("provenance", {}),
            "config": os.path.abspath(args.config),
            "protocol": "info/eval_folds.json",
            "bootstrap": args.boot,
            "bootstrap_seed": 42,
            "readout_policy": {
                "readout": "mlp", "hidden": 256, "seed": 42,
                "standardize": False,
            },
        },
        "results": results,
        "paired": paired,
        "readout_sensitivity": sensitivity,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
