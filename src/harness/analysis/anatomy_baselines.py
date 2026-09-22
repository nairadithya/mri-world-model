"""Frozen P1 baselines for RANO-free next-observed-visit anatomy forecasting.

The evaluation unit is a patient, never a scan.  Hyperparameters are selected
inside each training fold and all methods predict the same three log-volume
targets from the immutable ``lesion-state-v1`` manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.data.eval_protocol import fold_patients, load_protocol
from src.harness.data.anatomy_manifest import COMPARTMENTS, TARGET_VERSION

ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
LEARNED = ("volume_ridge", "trend_ridge", "current_image", "history_mean", "jepa_state")
METHODS = ("persistence", "mean_delta", "linear_trend", *LEARNED)


def _log_values(visit: dict) -> np.ndarray:
    values = visit["measurement"]["values_mm3"]
    return np.log1p([values[name] for name in COMPARTMENTS]).astype(np.float64)


def build_rows(cache: dict, manifest: dict, cohort: str) -> dict[str, list[dict]]:
    """Join immutable manifest pairs to encoded visits without positional drift."""
    visits = {(v["cohort"], v["patient_id"], v["visit"]): v
              for v in manifest["visits"]}
    pairs = [p for p in manifest["pairs"] if p["cohort"] == cohort and p["usable"]]
    out: dict[str, list[dict]] = {}
    for pair in pairs:
        pid, source_name, target_name = (pair["patient_id"], pair["source_visit"],
                                         pair["target_visit"])
        encoded = cache["patients"].get(pid)
        if encoded is None:
            continue
        if cohort == "LUMIERE":
            names = list(encoded["visits"])
        else:
            names = [f"ses-{i + 1:02d}" for i in range(len(encoded["vision"]))]
        if source_name not in names or target_name not in names:
            continue
        t, u = names.index(source_name), names.index(target_name)
        if u != t + 1 or t >= len(encoded["states"]):
            continue
        source = _log_values(visits[(cohort, pid, source_name)])
        target = _log_values(visits[(cohort, pid, target_name)])
        history = []
        for name in names[:t + 1]:
            visit = visits.get((cohort, pid, name))
            if visit and visit.get("measurement"):
                history.append(_log_values(visit))
        if not history or not bool(encoded["has_img"][t]):
            continue
        history_arr = np.stack(history)
        previous = history_arr[-2] if len(history_arr) > 1 else source
        trend = source - previous
        out.setdefault(pid, []).append({
            "row_id": pair["row_id"], "source": source, "target": target,
            "history": history_arr, "volume_ridge": source,
            "trend_ridge": np.r_[source, trend, source - history_arr.min(0)],
            "current_image": np.asarray(encoded["vision"][t], dtype=np.float64),
            "history_mean": np.asarray(encoded["vision"][:t + 1].mean(0), dtype=np.float64),
            "jepa_state": np.asarray(encoded["states"][t], dtype=np.float64),
        })
    return out


def _flat(rows: dict, pids: list[str], key: str | None = None):
    rr = [r for p in pids for r in rows.get(p, [])]
    X = None if key is None else np.stack([r[key] for r in rr])
    y = np.stack([r["target"] for r in rr])
    groups = np.array([p for p in pids for _ in rows.get(p, [])])
    return X, y, groups, rr


def _patient_uniform_mae(y: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> float:
    return float(np.mean([np.abs(y[groups == p] - pred[groups == p]).mean()
                          for p in np.unique(groups)]))


def _fit_ridge(rows: dict, pids: list[str], key: str):
    X, y, groups, _ = _flat(rows, pids, key)
    n_splits = min(3, len(np.unique(groups)))
    scores = {alpha: [] for alpha in ALPHAS}
    if n_splits >= 2:
        for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups):
            scaler = StandardScaler().fit(X[tr])
            for alpha in ALPHAS:
                model = Ridge(alpha=alpha, solver="lsqr").fit(scaler.transform(X[tr]), y[tr])
                pred = model.predict(scaler.transform(X[va]))
                scores[alpha].append(_patient_uniform_mae(y[va], pred, groups[va]))
    alpha = min(ALPHAS, key=lambda a: np.mean(scores[a]) if scores[a] else float("inf"))
    scaler = StandardScaler().fit(X)
    return scaler, Ridge(alpha=alpha, solver="lsqr").fit(scaler.transform(X), y), alpha


def _predict(rows: dict, train: list[str], test: list[str], method: str):
    _, y, groups, rr = _flat(rows, test)
    if method == "persistence":
        pred = np.stack([r["source"] for r in rr])
        return y, pred, groups, None
    if method == "mean_delta":
        _, _, _, train_rows = _flat(rows, train)
        delta = np.mean([r["target"] - r["source"] for r in train_rows], axis=0)
        pred = np.stack([r["source"] + delta for r in rr])
        return y, pred, groups, None
    if method == "linear_trend":
        pred = np.stack([r["source"] + (r["history"][-1] - r["history"][-2]
                                       if len(r["history"]) > 1 else 0) for r in rr])
        return y, pred, groups, None
    scaler, model, alpha = _fit_ridge(rows, train, method)
    X, _, _, _ = _flat(rows, test, method)
    return y, model.predict(scaler.transform(X)), groups, alpha


def _metrics(y, pred, groups):
    abs_error = np.abs(y - pred)
    per_patient = {p: float(abs_error[groups == p].mean()) for p in np.unique(groups)}
    correlations = []
    for j in range(y.shape[1]):
        if np.ptp(y[:, j]) == 0 or np.ptp(pred[:, j]) == 0:
            value = np.nan
        else:
            value = spearmanr(y[:, j], pred[:, j]).statistic
        correlations.append(None if not np.isfinite(value) else float(value))
    return {"log_volume_mae": float(abs_error.mean()),
            "patient_uniform_log_volume_mae": float(np.mean(list(per_patient.values()))),
            "signed_bias": float((pred - y).mean()),
            "spearman": correlations, "per_patient_mae": per_patient,
            "n_rows": len(y)}


def _evaluate_fold(rows, train, test, methods):
    result = {}
    for method in methods:
        y, pred, groups, alpha = _predict(rows, train, test, method)
        metric = _metrics(y, pred, groups)
        # Delta-log-volume MAE is distinct from level MAE only in reporting:
        # both subtract the identical observed source, so their errors coincide.
        metric["delta_log_volume_mae"] = metric["log_volume_mae"]
        metric["selected_alpha"] = alpha
        result[method] = metric
    baseline = result["persistence"]["log_volume_mae"]
    for metric in result.values():
        metric["persistence_relative_mae"] = metric["log_volume_mae"] / baseline
    return result


def _merge_folds(folds: list[dict], methods: tuple[str, ...]):
    merged = {}
    for method in methods:
        weights = np.array([f[method]["n_rows"] for f in folds])
        patient_errors = {p: e for f in folds for p, e in f[method]["per_patient_mae"].items()}
        correlations = []
        for j in range(len(COMPARTMENTS)):
            values = [f[method]["spearman"][j] for f in folds
                      if f[method]["spearman"][j] is not None]
            correlations.append(float(np.mean(values)) if values else None)
        merged[method] = {
            "patient_uniform_log_volume_mae": float(np.mean(list(patient_errors.values()))),
            "log_volume_mae": float(np.average(
                [f[method]["log_volume_mae"] for f in folds], weights=weights)),
            "delta_log_volume_mae": float(np.average(
                [f[method]["delta_log_volume_mae"] for f in folds], weights=weights)),
            "signed_bias": float(np.average(
                [f[method]["signed_bias"] for f in folds], weights=weights)),
            "spearman_fold_mean": correlations,
            "n_rows": int(weights.sum()),
            "persistence_relative_mae": None,
            "per_patient_mae": patient_errors,
            "selected_alphas": [f[method]["selected_alpha"] for f in folds],
        }
    base = merged["persistence"]["patient_uniform_log_volume_mae"]
    for metric in merged.values():
        metric["persistence_relative_mae"] = metric["patient_uniform_log_volume_mae"] / base
    return merged


def _bootstrap_differences(merged, boot: int, seed: int):
    rng = np.random.default_rng(seed)
    base = merged["persistence"]["per_patient_mae"]
    pids = np.array(sorted(base))
    for method, metric in merged.items():
        diffs = np.array([metric["per_patient_mae"][p] - base[p] for p in pids])
        draws = np.mean(rng.choice(diffs, (boot, len(diffs)), replace=True), axis=1)
        metric["mae_difference_vs_persistence"] = float(diffs.mean())
        metric["mae_difference_95ci"] = [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def main(argv=None):
    ap = argparse.ArgumentParser(description="P1 continuous anatomy baselines")
    ap.add_argument("--manifest", default="outputs/anatomy_harmonization.json")
    ap.add_argument("--lumiere-cache", default="checkpoints/p0_interface_cache.pt")
    ap.add_argument("--sailor-cache", default="checkpoints/p0_sailor_cache.pt")
    ap.add_argument("--protocol", default="info/eval_folds.json")
    ap.add_argument("--out", default="outputs/p1_anatomy_baselines.json")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    manifest = json.load(open(args.manifest))
    if manifest["target_version"] != TARGET_VERSION:
        raise ValueError("manifest target contract mismatch")
    protocol = load_protocol(args.protocol)
    lum_cache = torch.load(args.lumiere_cache, map_location="cpu", weights_only=False)
    sailor_cache = torch.load(args.sailor_cache, map_location="cpu", weights_only=False)
    lum = build_rows(lum_cache, manifest, "LUMIERE")
    sailor = build_rows(sailor_cache, manifest, "SAILOR")
    unseen = sorted(protocol["encoder_unseen"])
    folds = []
    for i in range(protocol["k"]):
        test = fold_patients(protocol, i)
        train = [p for p in unseen if p not in set(test)]
        folds.append(_evaluate_fold(lum, train, test, METHODS))
    within = _merge_folds(folds, METHODS)
    _bootstrap_differences(within, args.boot, args.seed)
    # Transfer is a developmental robustness check: no SAILOR labels are fit.
    transfer = _evaluate_fold({**lum, **sailor}, unseen, sorted(sailor), METHODS)
    _bootstrap_differences(transfer, args.boot, args.seed + 1)
    output = {
        "schema_version": 1, "target_version": TARGET_VERSION,
        "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        "protocol": "five-fold patient CV on encoder_unseen; fit all unseen -> SAILOR",
        "n_lumiere_patients": len(lum), "n_lumiere_rows": sum(map(len, lum.values())),
        "n_sailor_patients": len(sailor), "n_sailor_rows": sum(map(len, sailor.values())),
        "within_lumiere": within, "sailor_transfer": transfer,
        "limitations": ["next-observed-visit horizon only", "SAILOR is a developmental transfer cohort"],
    }
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    for section in ("within_lumiere", "sailor_transfer"):
        print(section)
        for method, metric in output[section].items():
            key = "patient_uniform_log_volume_mae"
            print(f"  {method:14s} MAE={metric[key]:.4f} rel={metric['persistence_relative_mae']:.3f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
