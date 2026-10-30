"""R39 patient-held-out forecasting from saved MRI states and cutoff status."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import ndtr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


PHASES = ("CRT", "TMZ", "no", "unknown")
COMPARTMENTS = ("necrotic_non_enhancing", "enhancing", "edema_flair")
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
ARMS = ("L", "LT", "H", "HT", "time_only", "phase_time")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Rows:
    current: np.ndarray
    previous: np.ndarray
    truth: np.ndarray
    measured_current: np.ndarray
    timing: np.ndarray
    phase: np.ndarray
    groups: np.ndarray
    has_history: np.ndarray

    def subset(self, index) -> Rows:
        return Rows(**{name: value[index] for name, value in vars(self).items()})


def load_rows(path: Path) -> Rows:
    with np.load(path, allow_pickle=False) as data:
        rows = Rows(**{name: data[name].copy() for name in Rows.__dataclass_fields__})
    n = len(rows.groups)
    for name in ("current", "previous", "truth", "measured_current"):
        if getattr(rows, name).shape != (n, 3):
            raise ValueError(f"invalid {name} shape")
    if rows.timing.shape != (n, 5):
        raise ValueError("invalid cutoff timing shape")
    for name in ("phase", "groups", "has_history"):
        if getattr(rows, name).shape != (n,):
            raise ValueError(f"invalid {name} shape")
    if not all(np.isfinite(value).all() for value in vars(rows).values()):
        raise ValueError("nonfinite input")
    if rows.phase.dtype.kind not in "iu" or not np.isin(rows.phase, range(4)).all():
        raise ValueError("invalid treatment category")
    if rows.groups.dtype.kind not in "iu" or rows.has_history.dtype != bool:
        raise ValueError("invalid patient or history encoding")
    return rows


def prepare(inventory_path: Path, prediction_path: Path, root: Path, out: Path) -> dict:
    """Join immutable visit identities, without reading target-session treatment."""
    inventory = json.loads(inventory_path.read_text())
    if inventory["cohort"] != "SAILOR":
        raise ValueError("R39 requires the SAILOR inventory")
    with np.load(prediction_path, allow_pickle=False) as archive:
        ids = archive["visit_ids"].tolist()
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate prediction identity")
        predictions = dict(zip(ids, archive["state_prediction"].astype(float)))
        truths = dict(zip(ids, archive["state_truth"]))
    visits = {v["visit_id"]: v for v in inventory["visits"]}
    if len(visits) != len(inventory["visits"]) or set(ids) != set(visits):
        raise ValueError("prediction/inventory membership mismatch")
    for identifier, visit in visits.items():
        expected = np.log1p([visit["values_mm3"][c] for c in COMPARTMENTS])
        if not np.allclose(truths[identifier], expected, atol=1e-10, rtol=0):
            raise ValueError("anatomical target order or identity mismatch")
    patients = sorted({v["patient_id"] for v in visits.values()})
    chronology, sidecar_hashes = {}, {}
    for pid in patients:
        directory = root / pid
        sessions = sorted(
            (p.name for p in directory.glob("ses-*") if p.is_dir()),
            key=lambda name: int(name.split("-")[1]),
        )
        path = directory / "intervals-days.txt"
        gaps = np.asarray([float(v) for v in path.read_text().split()])
        if (
            len(gaps) != len(sessions) - 1
            or not np.isfinite(gaps).all()
            or (gaps <= 0).any()
        ):
            raise ValueError("invalid original session chronology")
        chronology[pid] = (sessions, np.r_[0.0, np.cumsum(gaps)], gaps)
        sidecar_hashes[str(path.relative_to(root))] = sha256(path)
    lookup = {(v["patient_id"], v["visit"]): key for key, v in visits.items()}
    values = {name: [] for name in Rows.__dataclass_fields__}
    excluded, private_ids = Counter(), []
    for pair in inventory["pairs"]:
        pid = pair["patient_id"]
        sessions, elapsed, gaps = chronology[pid]
        i = sessions.index(pair["source_visit"])
        if (
            not pair["usable"]
            or not pair["ordering_verified"]
            or i + 1 >= len(sessions)
            or sessions[i + 1] != pair["target_visit"]
        ):
            excluded["nonadjacent_or_unusable"] += 1
            continue
        source = lookup[(pid, sessions[i])]
        target = lookup[(pid, sessions[i + 1])]
        if source != pair["source_visit_id"] or target != pair["target_visit_id"]:
            raise ValueError("pair identity mismatch")
        previous = lookup.get((pid, sessions[i - 1])) if i else None
        path = root / pid / sessions[i] / "treatment.txt"
        phase = path.read_text().strip() if path.exists() else "unknown"
        if phase not in PHASES:
            raise ValueError("unrecognized treatment token")
        if path.exists():
            sidecar_hashes[str(path.relative_to(root))] = sha256(path)
        current = predictions[source]
        row = dict(
            current=current,
            previous=predictions[previous] if previous else current,
            truth=truths[target],
            measured_current=truths[source],
            timing=[
                np.log1p(elapsed[i]),
                np.log1p(elapsed[i]) ** 2,
                np.log1p(gaps[i - 1]) if i else 0.0,
                np.log1p(i),
                float(i > 0),
            ],
            phase=PHASES.index(phase),
            groups=patients.index(pid),
            has_history=previous is not None,
        )
        for name, value in row.items():
            values[name].append(value)
        private_ids.append(pair["row_id"])
    arrays = {name: np.asarray(value) for name, value in values.items()}
    if len(set(arrays["groups"])) < 20 or len(arrays["groups"]) < 100:
        raise ValueError("insufficient patient or adjacent-pair coverage")
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "rows.npz", **arrays)
    load_rows(out / "rows.npz")
    source = {
        "inventory_sha256": sha256(inventory_path),
        "predictions_sha256": sha256(prediction_path),
        "rows_sha256": sha256(out / "rows.npz"),
        "sidecar_sha256": sidecar_hashes,
        "patient_mapping": patients,
        "row_ids": private_ids,
    }
    (out / "private-source.json").write_text(json.dumps(source, indent=2) + "\n")
    summary = {
        "patients": len(set(arrays["groups"])),
        "pairs": len(private_ids),
        "history_pairs": int(arrays["has_history"].sum()),
        "source_phase_counts": {
            name: int((arrays["phase"] == i).sum()) for i, name in enumerate(PHASES)
        },
        "excluded": dict(excluded),
        "inventory_sha256": source["inventory_sha256"],
        "predictions_sha256": source["predictions_sha256"],
        "rows_sha256": source["rows_sha256"],
    }
    (out / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def features(rows: Rows, arm: str) -> np.ndarray:
    phase = np.eye(4)[rows.phase]
    if arm == "time_only":
        return rows.timing
    if arm == "phase_time":
        return np.c_[rows.timing, phase]
    parts = [rows.current, rows.timing, rows.has_history[:, None]]
    if arm in ("H", "HT"):
        parts.append(rows.current - rows.previous)
    if arm in ("LT", "HT"):
        parts.extend(
            [phase, (phase[:, :, None] * rows.current[:, None, :]).reshape(-1, 12)]
        )
    return np.concatenate(parts, axis=1)


def weights(groups: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def patient_means(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    return np.asarray([values[groups == group].mean() for group in np.unique(groups)])


def crps(truth: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    z = (truth - mean) / sd
    return sd * (
        z * (2 * ndtr(z) - 1)
        + 2 * np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
        - 1 / np.sqrt(np.pi)
    )


def fit_ridge(x, y, groups, alpha):
    weight = weights(groups)
    center = np.average(x, axis=0, weights=weight)
    scale = np.sqrt(np.average((x - center) ** 2, axis=0, weights=weight))
    scale[scale < 1e-12] = 1.0
    model = Ridge(alpha=alpha, solver="svd").fit(
        (x - center) / scale, y, sample_weight=weight
    )
    return {
        "center": center,
        "scale": scale,
        "coef": model.coef_,
        "intercept": model.intercept_,
    }


def predict(model, x):
    return ((x - model["center"]) / model["scale"]) @ model["coef"].T + model[
        "intercept"
    ]


def fit_transition(rows: Rows, arm: str):
    x = features(rows, arm)
    baseline = (
        rows.current if arm in ("L", "LT", "H", "HT") else np.zeros_like(rows.current)
    )
    target = rows.truth - baseline
    splits = list(GroupKFold(3).split(x, groups=rows.groups))
    scores, oof = [], []
    for alpha in ALPHAS:
        prediction = np.zeros_like(target)
        for train, valid in splits:
            model = fit_ridge(x[train], target[train], rows.groups[train], alpha)
            prediction[valid] = predict(model, x[valid])
        scores.append(
            patient_means(np.abs(target - prediction).mean(1), rows.groups).mean()
        )
        oof.append(prediction)
    chosen = int(np.argmin(scores))
    residual = target - oof[chosen]
    variance = np.average(residual**2, axis=0, weights=weights(rows.groups))
    model = fit_ridge(x, target, rows.groups, ALPHAS[chosen])
    model["sd"] = np.sqrt(variance).clip(min=0.05)
    model["alpha"] = np.asarray(ALPHAS[chosen])
    return model


def forecast(model: dict, rows: Rows, arm: str):
    baseline = (
        rows.current if arm in ("L", "LT", "H", "HT") else np.zeros_like(rows.current)
    )
    return baseline + predict(model, features(rows, arm))


def paired_summary(candidate, baseline, groups, draws=10000):
    difference = patient_means(candidate - baseline, groups)
    rng = np.random.default_rng(2026)
    means = difference[rng.integers(0, len(difference), (draws, len(difference)))].mean(
        1
    )
    return {
        "difference": float(difference.mean()),
        "ci99": np.quantile(means, [0.005, 0.995]).tolist(),
        "relative_reduction": float(
            1
            - patient_means(candidate, groups).mean()
            / patient_means(baseline, groups).mean()
        ),
    }


def run(rows_path: Path, out: Path, canary=False) -> dict:
    started = time.monotonic()
    rows = load_rows(rows_path)
    if len(set(rows.groups)) < 20 or len(rows.groups) < 100:
        raise ValueError("insufficient real-data coverage")
    out.mkdir(parents=True, exist_ok=False)
    unique = np.random.default_rng(2026).permutation(np.unique(rows.groups))
    folds = {group: i % 5 for i, group in enumerate(unique)}
    fold_ids = np.asarray([folds[group] for group in rows.groups])
    names = (*ARMS, "persistence", "measured_persistence")
    means = {name: np.full_like(rows.truth, np.nan) for name in names}
    scales = {name: np.full_like(rows.truth, np.nan) for name in names}
    selected = {name: [] for name in ARMS}
    for fold in range(1 if canary else 5):
        train, test = rows.subset(fold_ids != fold), rows.subset(fold_ids == fold)
        for arm in ARMS:
            model = fit_transition(train, arm)
            path = out / f"fold{fold}-{arm}.npz"
            np.savez_compressed(path, **model)
            with np.load(path, allow_pickle=False) as saved:
                restored = {key: saved[key] for key in saved.files}
            prediction = forecast(restored, test, arm)
            np.testing.assert_array_equal(prediction, forecast(model, test, arm))
            means[arm][fold_ids == fold] = prediction
            scales[arm][fold_ids == fold] = model["sd"]
            selected[arm].append(float(model["alpha"]))
        for arm, source in (
            ("persistence", "current"),
            ("measured_persistence", "measured_current"),
        ):
            residual = train.truth - getattr(train, source)
            sd = np.sqrt(
                np.average(residual**2, axis=0, weights=weights(train.groups))
            ).clip(min=0.05)
            means[arm][fold_ids == fold] = getattr(test, source)
            scales[arm][fold_ids == fold] = sd
            np.savez_compressed(out / f"fold{fold}-{arm}.npz", sd=sd)
        print(f"fold {fold + 1}/5 complete", flush=True)
    included = fold_ids == 0 if canary else np.ones(len(fold_ids), dtype=bool)
    loss = {
        name: crps(
            rows.truth[included], means[name][included], scales[name][included]
        ).mean(1)
        for name in names
    }
    if not all(np.isfinite(value).all() for value in loss.values()):
        raise ValueError("incomplete or nonfinite scores")
    groups = rows.groups[included]
    scores = {
        name: {
            "crps": float(patient_means(loss[name], groups).mean()),
            "mae": float(
                patient_means(
                    np.abs(rows.truth[included] - means[name][included]).mean(1), groups
                ).mean()
            ),
        }
        for name in names
    }
    comparisons = {
        f"HT_vs_{name}": paired_summary(loss["HT"], loss[name], groups)
        for name in ("H", "LT", "persistence", "phase_time")
    }
    comparisons["LT_vs_L"] = paired_summary(loss["LT"], loss["L"], groups)
    primary = comparisons["HT_vs_H"]
    treatment_supported = (
        primary["relative_reduction"] >= 0.05 and primary["ci99"][1] < 0
    )
    history_supported = treatment_supported and all(
        comparisons[f"HT_vs_{name}"]["ci99"][1] < 0
        for name in ("LT", "persistence", "phase_time")
    )
    subsets = {}
    for label, mask in (
        ("known_status", rows.phase[included] != 3),
        ("history_available", rows.has_history[included]),
    ):
        subsets[label] = {
            "patients": len(set(groups[mask])),
            "pairs": int(mask.sum()),
            "scores": {
                name: float(patient_means(loss[name][mask], groups[mask]).mean())
                for name in names
            },
        }
    result = {
        "experiment": "R39",
        "status": "ENGINEERING_CANARY" if canary else "COMPLETE",
        "treatment_feasibility_supported": bool(treatment_supported)
        if not canary
        else None,
        "history_followup_supported": bool(history_supported) if not canary else None,
        "patients": len(set(groups)),
        "pairs": int(included.sum()),
        "scores": scores,
        "comparisons": comparisons,
        "subsets": subsets,
        "selected_alphas": selected,
        "elapsed_seconds": time.monotonic() - started,
        "rows_sha256": sha256(rows_path),
        "limitations": [
            "reused SAILOR development cohort",
            "observational current status, not planned intervention",
            "prospective status availability unverified",
            "approximate recorded past intervals",
            "automated anatomical proxy targets",
            "compressed MRI states, no new representation learning",
            "constant Gaussian uncertainty",
            "bootstrap conditions on fitted folds",
            "no acquisition-robustness or causal treatment-effect evidence",
        ],
    }
    np.savez_compressed(
        out / "predictions.npz",
        truth=rows.truth,
        groups=rows.groups,
        fold=fold_ids,
        **{f"mean_{name}": value for name, value in means.items()},
        **{f"sd_{name}": value for name, value in scales.items()},
    )
    (out / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result
