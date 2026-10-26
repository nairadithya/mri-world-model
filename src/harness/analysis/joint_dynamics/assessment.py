from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

import numpy as np
from scipy.special import ndtr, ndtri

from .baselines import (ControlFits, GRID, OBSERVATION_SD, _digest, gaussian_crps,
                        mixture_crps, patient_weights)

ARMS = ("OH", "OL", "JH", "JL")
SEEDS = (42, 43, 44)
CONDITIONS = ("identity", "intensity", "resolution")
LEVELS = (0.5, 0.8, 0.95)
COHORTS = {"primary": (75, 18), "single": (26, 23), "nuisance": (18, 18)}
DRAWS = 10000
BOOTSTRAP_SEED = 2026


def _need(arrays: Mapping[str, np.ndarray], key: str, shape: tuple[int, ...],
          kind: str = "f") -> np.ndarray:
    if key not in arrays:
        raise ValueError(f"missing array {key}")
    value = np.asarray(arrays[key])
    if value.shape != shape or value.dtype.kind not in kind:
        raise ValueError(f"{key} must have shape {shape} and dtype kind {kind}")
    if value.dtype.kind in "fiu" and not np.isfinite(value).all():
        raise ValueError(f"nonfinite {key}")
    return value


def _same(actual: Any, expected: Any, name: str) -> None:
    a, e = np.asarray(actual), np.asarray(expected)
    if a.shape != e.shape or not np.allclose(a, e, rtol=1e-10, atol=1e-10):
        raise ValueError(f"{name} differs from bound receipt")


def _scalar(arrays: Mapping[str, np.ndarray], key: str, expected: float) -> float:
    value = float(_need(arrays, key, ()))
    if value != expected:
        raise ValueError(f"{key} differs from selected calibration/observation noise")
    return value


def _validate_keys(arrays: Mapping[str, np.ndarray]) -> None:
    shared = {"row_ids", "patient_ids", "truth", "cutoff_measured"}
    for key, value in arrays.items():
        if not isinstance(key, str) or np.asarray(value).dtype.kind not in "USfiub":
            raise ValueError("NPZ keys must be strings and arrays must not require pickle")
        if np.asarray(value).dtype.kind in "fiu" and not np.isfinite(value).all():
            raise ValueError(f"nonfinite saved array {key}")
        if key == "seeds":
            continue
        parts = key.split("_")
        if parts[0] not in COHORTS:
            raise ValueError(f"unrecognized prediction namespace {key}")
        cohort = parts[0]
        if key[len(cohort)+1:] in shared:
            continue
        if len(parts) < 3:
            raise ValueError(f"malformed key {key}")
        if cohort != "nuisance" and parts[1] in ("P", "E", "FP", "SSM"):
            allowed = {"centers", "weights", "observation_sd"} if parts[1] == "E" else {
                "mean", "intrinsic_scale", "calibration"}
            if "_".join(parts[2:]) not in allowed:
                raise ValueError(f"unknown control field {key}")
            continue
        if parts[1] not in ARMS:
            raise ValueError(f"unknown arm {key}")
        seed_index = 2
        if cohort == "nuisance":
            if parts[2] not in CONDITIONS:
                raise ValueError(f"unknown nuisance condition {key}")
            seed_index = 3
        if len(parts) <= seed_index+1 or parts[seed_index] not in tuple(map(str, SEEDS)):
            raise ValueError(f"unknown or missing seed label {key}")
        if not "_".join(parts[seed_index+1:]):
            raise ValueError(f"missing field {key}")


def _bound_rows(arrays: Mapping[str, np.ndarray], identity: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(identity.get("cohort_rows"), dict) or set(identity["cohort_rows"]) != set(COHORTS):
        raise ValueError("missing or extra bound cohorts")
    rows = identity["cohort_rows"]
    for cohort, (n, patients) in COHORTS.items():
        records = rows[cohort]
        if not isinstance(records, list) or len(records) != n:
            raise ValueError(f"{cohort} must contain exactly {n} rows")
        for r in records:
            for field in ("row_id", "patient_id", "cutoff_visit_id", "target_visit_id"):
                if not isinstance(r.get(field), str) or not r[field]:
                    raise ValueError(f"missing {cohort} identity {field}")
            if r["cutoff_visit_id"] == r["target_visit_id"]:
                raise ValueError("cutoff and future must be distinct visits")
        row_ids = [r["row_id"] for r in records]
        patient_ids = [r["patient_id"] for r in records]
        if len(set(row_ids)) != n or len(set(patient_ids)) != patients:
            raise ValueError(f"incorrect {cohort} row/patient membership")
        if not np.array_equal(_need(arrays, cohort+"_row_ids", (n,), "US"), row_ids):
            raise ValueError(f"{cohort} row identity/order mismatch")
        if not np.array_equal(_need(arrays, cohort+"_patient_ids", (n,), "US"), patient_ids):
            raise ValueError(f"{cohort} patient identity/order mismatch")
        for field in ("truth", "cutoff_measured"):
            actual = _need(arrays, cohort+"_"+field, (n, 3))
            expected = np.asarray([r[field] for r in records], dtype=float)
            if expected.shape != (n, 3) or not np.isfinite(expected).all() or (expected < 0).any():
                raise ValueError("measured native log-volumes must be finite and nonnegative")
            if not np.array_equal(actual, expected):
                raise ValueError(f"{cohort} measured {field} differs from bound identity")
    if {r["row_id"] for r in rows["primary"]} & {r["row_id"] for r in rows["single"]}:
        raise ValueError("primary and single row overlap")
    earliest = {}
    for row in rows["primary"]:
        earliest.setdefault(row["patient_id"], row)
    if rows["nuisance"] != list(earliest.values()):
        raise ValueError("nuisance must be the bound earliest primary row per patient")
    return rows


def _calibration(identity: dict[str, Any], fits: ControlFits) -> dict[str, Any]:
    c = identity.get("calibration")
    wanted = {f"{arm}_{seed}" for arm in ARMS for seed in SEEDS} | {"P", "E", "FP", "SSM"}
    if not isinstance(c, dict) or set(c) != wanted:
        raise ValueError("incomplete or extra fitted calibration identities")
    for name in ("P", "E", "FP", "SSM"):
        if c[name] not in GRID or c[name] != fits.state["calibration"][name]:
            raise ValueError("common-control calibration differs from portable fit")
    for arm in ARMS:
        for seed in SEEDS:
            entry = c[f"{arm}_{seed}"]
            if not isinstance(entry, dict) or not {"forecast", "persistence", "persistence_rms"} <= set(entry):
                raise ValueError("incomplete own-persistence calibration receipt")
            for field in ("forecast", "persistence"):
                if not np.isscalar(entry[field]) or entry[field] not in GRID:
                    raise ValueError("calibration must be selected from fixed grid")
            rms = np.asarray(entry["persistence_rms"], float)
            if rms.shape != (3,) or not np.isfinite(rms).all() or (rms < 0).any():
                raise ValueError("invalid own-persistence residual RMS receipt")
    return c


def _intervals_gaussian(mean: np.ndarray, scale: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    z = ndtri((1+level)/2)
    return mean-z*scale, mean+z*scale


def _intervals_mixture(centers: np.ndarray, weights: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    result = []
    for probability in ((1-level)/2, (1+level)/2):
        lo = centers.min(axis=1)-12*OBSERVATION_SD
        hi = centers.max(axis=1)+12*OBSERVATION_SD
        for _ in range(60):
            mid = (lo+hi)/2
            cdf = np.sum(weights[None, :, None]*ndtr((mid[:, None, :]-centers)/OBSERVATION_SD), axis=1)
            lo, hi = np.where(cdf < probability, mid, lo), np.where(cdf >= probability, mid, hi)
        result.append((lo+hi)/2)
    return result[0], result[1]


def _metrics(mean: np.ndarray, truth: np.ndarray, crps: np.ndarray,
             intervals: dict[float, tuple[np.ndarray, np.ndarray]]) -> dict[str, np.ndarray]:
    if not all(np.isfinite(x).all() for x in (mean, truth, crps)):
        raise ValueError("nonfinite reconstructed scores")
    result = {"crps": crps, "mae": np.abs(mean-truth)}
    for level, (lo, hi) in intervals.items():
        if not np.isfinite(lo).all() or not np.isfinite(hi).all() or (lo > hi).any():
            raise ValueError("invalid predictive interval")
        result[f"coverage_{level}"] = ((truth >= lo) & (truth <= hi)).astype(float)
        result[f"width_{level}"] = hi-lo
    if not all(np.isfinite(x).all() for x in result.values()):
        raise ValueError("nonfinite reconstructed error/width")
    return result


def _gaussian(arrays: Mapping[str, np.ndarray], prefix: str, truth: np.ndarray,
              calibration: float) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    n = len(truth)
    mean = _need(arrays, prefix+"_mean", (n, 3))
    intrinsic = _need(arrays, prefix+"_intrinsic_scale", (n, 3))
    if (intrinsic < 0).any():
        raise ValueError(f"negative intrinsic scale {prefix}")
    k = _scalar(arrays, prefix+"_calibration", calibration)
    total = np.hypot(k*intrinsic, OBSERVATION_SD)
    if not np.isfinite(total).all() or (total <= 0).any():
        raise ValueError("invalid total Gaussian scale")
    metrics = _metrics(mean, truth, gaussian_crps(mean, total, truth), {
        level: _intervals_gaussian(mean, total, level) for level in LEVELS})
    return metrics, mean, intrinsic


def patient_values(values: np.ndarray, patient_ids: Any) -> np.ndarray:
    ids = np.asarray(patient_ids)
    v = np.asarray(values, float)
    patient_weights(ids)
    if v.shape[0] != len(ids) or not np.isfinite(v).all():
        raise ValueError("invalid patient scores")
    return np.stack([v[ids == p].mean(axis=0) for p in sorted(set(ids))])


def _average_seeds(metrics: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not metrics or any(set(m) != set(metrics[0]) for m in metrics):
        raise ValueError("mismatched seed metrics")
    return {name: np.mean([m[name] for m in metrics], axis=0) for name in metrics[0]}


def _summary(metrics: dict[str, np.ndarray], truth: np.ndarray, cutoff: np.ndarray,
             patient_ids: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, values in metrics.items():
        by_compartment = patient_values(values, patient_ids).mean(axis=0)
        result[name] = {"mean": float(by_compartment.mean()), "per_compartment": by_compartment.tolist()}
    partitions = {"empty_target": truth == 0, "appearance": (cutoff == 0) & (truth > 0),
                  "disappearance": (cutoff > 0) & (truth == 0),
                  "both_empty": (cutoff == 0) & (truth == 0), "empty_cutoff": cutoff == 0}
    subsets = {}
    for name, mask in partitions.items():
        by_compartment = []
        for c in range(3):
            selected = mask[:, c]
            item: dict[str, Any] = {"rows": int(selected.sum()),
                                    "patients": len(set(patient_ids[selected]))}
            for metric in ("crps", "mae", "current_mae", "current_dice", "future_dice"):
                if metric in metrics:
                    item[metric] = (float(patient_values(metrics[metric][selected, c],
                                                         patient_ids[selected]).mean())
                                    if selected.any() else None)
            by_compartment.append(item)
        subsets[name] = by_compartment
    result["sparse_subsets"] = subsets
    return result


def paired_interval(patient_differences: np.ndarray, draws: np.ndarray) -> dict[str, Any]:
    differences = np.asarray(patient_differences, float)
    if differences.ndim != 1 or not len(differences) or not np.isfinite(differences).all():
        raise ValueError("invalid paired differences")
    if draws.shape != (DRAWS, len(differences)) or draws.dtype.kind not in "iu":
        raise ValueError("invalid shared bootstrap draws")
    if (draws < 0).any() or (draws >= len(differences)).any():
        raise ValueError("bootstrap index outside patient set")
    samples = differences[draws].mean(axis=1)
    return {"difference": float(differences.mean()),
            "ci_99_5": np.quantile(samples, [0.0025, 0.9975]).tolist(),
            "ci_95": np.quantile(samples, [0.025, 0.975]).tolist()}


def _direction(interval: dict[str, Any], positive_favors: bool = False) -> str:
    lo, hi = interval["ci_99_5"]
    favorable = lo > 0 if positive_favors else hi < 0
    harmful = hi < 0 if positive_favors else lo > 0
    if favorable:
        return "favorable"
    if harmful:
        return "harmful"
    return "unresolved"


def _contrasts(scores: dict[str, np.ndarray], draws: np.ndarray) -> dict[str, Any]:
    pairs = (("JH", "OH"), ("JL", "OL"), ("JH", "JL"), ("OH", "OL"),
             ("JH", "P"), ("JH", "E"), ("JH", "FP"), ("JH", "SSM"))
    result = {}
    for candidate, reference in pairs:
        item = paired_interval(scores[candidate]-scores[reference], draws)
        denominator = float(scores[reference].mean())
        if denominator <= 0:
            raise ValueError("nonpositive relative CRPS comparator denominator")
        item.update({"direction": "negative_favors_candidate", "candidate": candidate,
                     "reference": reference, "relative_denominator": reference,
                     "denominator_crps": denominator,
                     "relative_reduction": float((scores[reference].mean()-scores[candidate].mean())/denominator),
                     "adjusted_result": _direction(item)})
        result[f"{candidate}-{reference}"] = item
    interaction = paired_interval(scores["JH"]-scores["JL"]-scores["OH"]+scores["OL"], draws)
    interaction.update({"relative_denominator": None, "direction": "negative_more_gradient_advantage_with_history",
                        "interpretation": ("larger_gradient_advantage_with_history" if interaction["ci_99_5"][1] < 0
                                           else "larger_gradient_advantage_with_latest" if interaction["ci_99_5"][0] > 0
                                           else "unresolved")})
    result["JH-JL-OH+OL"] = interaction
    gain = paired_interval((scores["P_JH"]-scores["JH"])-(scores["P_OH"]-scores["OH"]), draws)
    gain.update({"direction": "positive_favors_greater_JH_own_persistence_gain",
                 "relative_denominator": None, "adjusted_result": _direction(gain, True)})
    result["(P_JH-JH)-(P_OH-OH)"] = gain
    return result


def _retention(metrics: dict[str, dict[str, np.ndarray]], patient_ids: np.ndarray,
               draws: np.ndarray) -> dict[str, Any]:
    result = {}
    for joint, observation in (("JH", "OH"), ("JL", "OL")):
        comparisons = {}
        for metric, margin in (("current_mae", 0.05), ("current_dice", 0.02)):
            difference = metrics[joint][metric]-metrics[observation][metric]
            if metric == "current_dice":
                difference = -difference
            pv = patient_values(difference, patient_ids)
            items = []
            for c in range(3):
                interval = paired_interval(pv[:, c], draws)
                interval.update({"margin": margin, "passes": interval["ci_95"][1] < margin})
                items.append(interval)
            comparisons[metric] = {"direction": "joint_minus_observation" if metric == "current_mae"
                                    else "observation_minus_joint", "per_compartment": items,
                                    "passes": all(item["passes"] for item in items)}
        comparisons["passes"] = all(comparisons[k]["passes"] for k in ("current_mae", "current_dice"))
        result[f"{joint}-{observation}"] = comparisons
    result["passes"] = all(result[k]["passes"] for k in ("JH-OH", "JL-OL"))
    return result


def _cohort(arrays: Mapping[str, np.ndarray], cohort: str, rows: list[dict[str, Any]],
            fits: ControlFits, calibration: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    n = len(rows)
    truth, cutoff = arrays[cohort+"_truth"], arrays[cohort+"_cutoff_measured"]
    ids = np.asarray([r["patient_id"] for r in rows])
    metrics: dict[str, dict[str, np.ndarray]] = {}
    summaries = {}
    for arm in ARMS:
        seed_metrics, persistence_metrics = [], []
        per_seed, persistence_seed = {}, {}
        for seed in SEEDS:
            prefix = f"{cohort}_{arm}_{seed}"
            receipt = calibration[f"{arm}_{seed}"]
            m, _, _ = _gaussian(arrays, prefix, truth, receipt["forecast"])
            current = _need(arrays, prefix+"_current_mean", (n, 3))
            m["current_mae"] = np.abs(current-cutoff)
            for suffix in ("current_dice", "future_dice"):
                dice = _need(arrays, prefix+"_"+suffix, (n, 3))
                if (dice < 0).any() or (dice > 1).any():
                    raise ValueError("binary Dice outside [0,1]")
                empty = (cutoff == 0) if suffix == "current_dice" else (truth == 0)
                if np.any(empty & (dice != 0) & (dice != 1)):
                    raise ValueError("empty measured target requires binary Dice zero or one")
                m[suffix] = dice
            rms = _need(arrays, prefix+"_persistence_rms", (3,))
            _same(rms, receipt["persistence_rms"], prefix+" persistence RMS")
            if (rms < 0).any():
                raise ValueError("negative own-persistence RMS")
            k = _scalar(arrays, prefix+"_persistence_calibration", receipt["persistence"])
            scale = np.broadcast_to(np.hypot(k*rms, OBSERVATION_SD), (n, 3))
            pm = _metrics(current, truth, gaussian_crps(current, scale, truth), {
                level: _intervals_gaussian(current, scale, level) for level in LEVELS})
            seed_metrics.append(m); persistence_metrics.append(pm)
            per_seed[str(seed)] = _summary(m, truth, cutoff, ids)
            persistence_seed[str(seed)] = _summary(pm, truth, cutoff, ids)
        metrics[arm] = _average_seeds(seed_metrics)
        metrics["P_"+arm] = _average_seeds(persistence_metrics)
        summaries[arm] = {"seeds": per_seed, "seed_score_average": _summary(metrics[arm], truth, cutoff, ids)}
        summaries["P_"+arm] = {"seeds": persistence_seed,
                                  "seed_score_average": _summary(metrics["P_"+arm], truth, cutoff, ids)}
    predictions = {}
    for control in ("P", "FP", "SSM"):
        prefix = f"{cohort}_{control}"
        m, mean, intrinsic = _gaussian(arrays, prefix, truth, calibration[control])
        metrics[control] = m
        predictions[control] = (mean, intrinsic)
        summaries[control] = _summary(m, truth, cutoff, ids)
    state = fits.state
    _same(predictions["P"][1], np.broadcast_to(state["p_rms"], (n, 3)), "P RMS")
    parameters = state["filter"]["parameters"]
    b = np.asarray([p["final"][0] for p in parameters])
    q = np.asarray([p["final"][1] for p in parameters])
    horizon = state["typical_outgoing_weeks"]
    _same(predictions["SSM"][0], predictions["FP"][0]+b*horizon, "SSM/FP posterior mean")
    _same(predictions["SSM"][1]**2, predictions["FP"][1]**2+q*horizon, "SSM/FP posterior variance")
    count = len(state["e_changes"])
    centers = _need(arrays, f"{cohort}_E_centers", (n, count, 3))
    weights = _need(arrays, f"{cohort}_E_weights", (count,))
    _scalar(arrays, f"{cohort}_E_observation_sd", OBSERVATION_SD)
    _same(weights, state["e_weights"], "E weights")
    changes, mean = np.asarray(state["e_changes"]), np.asarray(state["e_mean"])
    expected = predictions["P"][0][:, None, :]+mean+calibration["E"]*(changes-mean)
    _same(centers, expected, "E centers/canonical P cutoff estimates")
    e_mean = np.sum(centers*weights[None, :, None], axis=1)
    em = _metrics(e_mean, truth, mixture_crps(centers, weights, truth), {
        level: _intervals_mixture(centers, weights, level) for level in LEVELS})
    metrics["E"] = em
    summaries["E"] = _summary(em, truth, cutoff, ids)
    return {"rows": n, "patients": len(set(ids)), "scores": summaries}, metrics


def _nuisance(arrays: Mapping[str, np.ndarray], rows: dict[str, list[dict[str, Any]]],
              calibration: dict[str, Any], draws: np.ndarray) -> dict[str, Any]:
    records = rows["nuisance"]
    ids = np.asarray([r["patient_id"] for r in records])
    truth, cutoff = arrays["nuisance_truth"], arrays["nuisance_cutoff_measured"]
    primary_index = {r["row_id"]: i for i, r in enumerate(rows["primary"])}
    selected = np.asarray([primary_index[r["row_id"]] for r in records])
    n = len(records)
    result = {}
    for arm in ARMS:
        condition_metrics = {}
        condition_means = {}
        summaries = {}
        for condition in CONDITIONS:
            by_seed, means, current_means = [], [], []
            seed_summary = {}
            for seed in SEEDS:
                prefix = f"nuisance_{arm}_{condition}_{seed}"
                m, mean, _ = _gaussian(arrays, prefix, truth, calibration[f"{arm}_{seed}"]["forecast"])
                current = _need(arrays, prefix+"_current_mean", (n, 3))
                m["current_mae"] = np.abs(current-cutoff)
                if condition == "identity":
                    for suffix in ("mean", "intrinsic_scale", "current_mean"):
                        _same(arrays[prefix+"_"+suffix], arrays[f"primary_{arm}_{seed}_{suffix}"][selected],
                              "nuisance identity/primary "+suffix)
                by_seed.append(m); means.append(mean); current_means.append(current)
                seed_summary[str(seed)] = _summary(m, truth, cutoff, ids)
            condition_metrics[condition] = _average_seeds(by_seed)
            condition_means[condition] = (means, current_means)
            summaries[condition] = {"seeds": seed_summary,
                                    "seed_score_average": _summary(condition_metrics[condition], truth, cutoff, ids)}
        changes = {}
        for condition in ("intensity", "resolution"):
            per_compartment = []
            identity_crps = patient_values(condition_metrics["identity"]["crps"], ids)
            perturb_crps = patient_values(condition_metrics[condition]["crps"], ids)
            identity_mae = patient_values(condition_metrics["identity"]["mae"], ids)
            perturb_mae = patient_values(condition_metrics[condition]["mae"], ids)
            drifts = np.mean([np.abs(p-i) for p, i in zip(condition_means[condition][0],
                                                          condition_means["identity"][0])], axis=0)
            current_drifts = np.mean([np.abs(p-i) for p, i in zip(condition_means[condition][1],
                                                                  condition_means["identity"][1])], axis=0)
            drift_pv = patient_values(drifts, ids)
            current_drift_pv = patient_values(current_drifts, ids)
            seed_changes = {}
            for seed in SEEDS:
                p = f"nuisance_{arm}_{condition}_{seed}_mean"
                i = f"nuisance_{arm}_identity_{seed}_mean"
                base_scores = summaries["identity"]["seeds"][str(seed)]
                perturbed_scores = summaries[condition]["seeds"][str(seed)]
                base_crps = np.asarray(base_scores["crps"]["per_compartment"])
                perturbed_crps = np.asarray(perturbed_scores["crps"]["per_compartment"])
                seed_changes[str(seed)] = {
                    "crps_change": (perturbed_crps-base_crps).tolist(),
                    "relative_crps_worsening": ((perturbed_crps-base_crps)/base_crps).tolist(),
                    "mae_change": (np.asarray(perturbed_scores["mae"]["per_compartment"])
                                   - np.asarray(base_scores["mae"]["per_compartment"])).tolist(),
                    "mean_absolute_predictive_drift": patient_values(np.abs(arrays[p]-arrays[i]), ids).mean(axis=0).tolist()}
            for c in range(3):
                denominator = float(identity_crps[:, c].mean())
                if denominator <= 0:
                    raise ValueError("nonpositive nuisance CRPS denominator")
                relative = float((perturb_crps[:, c].mean()-identity_crps[:, c].mean())/denominator)
                mae_change = float((perturb_mae[:, c]-identity_mae[:, c]).mean())
                crps_interval = paired_interval(perturb_crps[:, c]-identity_crps[:, c], draws)
                ratios = ((perturb_crps[:, c][draws].mean(axis=1)-identity_crps[:, c][draws].mean(axis=1))
                          / identity_crps[:, c][draws].mean(axis=1))
                per_compartment.append({"relative_crps_worsening": relative,
                                        "relative_denominator": "identity_compartment_crps",
                                        "identity_crps": denominator,
                                        "relative_ci_95": np.quantile(ratios, [0.025, 0.975]).tolist(),
                                        "crps_change": crps_interval,
                                        "mae_change": paired_interval(perturb_mae[:, c]-identity_mae[:, c], draws),
                                        "mean_absolute_predictive_drift": paired_interval(drift_pv[:, c], draws),
                                        "mean_absolute_current_drift": paired_interval(current_drift_pv[:, c], draws),
                                        "crps_passes": relative <= 0.10, "mae_passes": mae_change <= 0.05,
                                        "passes": relative <= 0.10 and mae_change <= 0.05})
            changes[condition] = {"per_compartment": per_compartment, "seeds": seed_changes,
                                   "passes": all(c["passes"] for c in per_compartment)}
        result[arm] = {"conditions": summaries, "changes": changes,
                       "passes": all(changes[c]["passes"] for c in ("intensity", "resolution"))}
    result["JH_passes"] = result["JH"]["passes"]
    result["rows"] = len(records)
    result["patients"] = len(set(ids))
    return result


def _capability_flags(contrasts: dict[str, Any], retention: dict[str, Any],
                      coverage: np.ndarray, nuisance: dict[str, Any]) -> dict[str, bool]:
    flags = {
        "joint_beats_observation": contrasts["JH-OH"]["ci_99_5"][1] < 0,
        "history_beats_latest": contrasts["JH-JL"]["ci_99_5"][1] < 0,
        "joint_material_reduction": contrasts["JH-OH"]["relative_reduction"] >= 0.05,
        "history_material_reduction": contrasts["JH-JL"]["relative_reduction"] >= 0.05,
        "JH_state_retention": retention["JH-OH"]["passes"],
        "JL_state_retention": retention["JL-OL"]["passes"],
        "both_joint_arm_state_retention": retention["passes"],
        "coverage_80": bool(((coverage >= 0.70) & (coverage <= 0.90)).all()),
        "nuisance": nuisance["JH_passes"]}
    for control in ("P", "E", "FP", "SSM"):
        flags["beats_"+control] = contrasts[f"JH-{control}"]["ci_99_5"][1] < 0
    flags["beats_all_common_controls"] = all(flags["beats_"+c] for c in ("P", "E", "FP", "SSM"))
    return flags


def analyze(arrays: Mapping[str, np.ndarray], identity: dict[str, Any]) -> dict[str, Any]:
    """Validate complete private predictions, then reconstruct a JSON-only staged record.

    Budget success and publication are exclusively runner responsibilities. Independent
    result review is a later, separately hash-bound record, not a mutation here.
    """
    if not isinstance(arrays, Mapping) or not isinstance(identity, dict):
        raise ValueError("arrays and identity must be mappings")
    try:
        return _analyze(arrays, identity)
    except (KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError("incomplete or malformed prediction/identity bundle") from exc


def _analyze(arrays: Mapping[str, np.ndarray], identity: dict[str, Any]) -> dict[str, Any]:
    for key in ("contract_digest", "source_sha256", "fitted_manifest_sha256"):
        if not isinstance(identity.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", identity[key]):
            raise ValueError(f"missing or invalid bound digest {key}")
    _validate_keys(arrays)
    if not np.array_equal(_need(arrays, "seeds", (3,), "iu"), SEEDS):
        raise ValueError("global seed labels/order differ from fixed matrix")
    rows = _bound_rows(arrays, identity)
    fits = ControlFits.from_dict(identity["control_fits"])
    calibration = _calibration(identity, fits)
    source = set(fits.state["source"]["patient_ids"])
    validation = set(fits.state["validation"]["patient_ids"])
    assessed = {r["patient_id"] for cohort in rows.values() for r in cohort}
    if assessed & (source | validation):
        raise ValueError("assessment patients overlap fitting/calibration membership")
    cohorts = {}; all_metrics = {}
    for name in ("primary", "single"):
        cohorts[name], all_metrics[name] = _cohort(arrays, name, rows[name], fits, calibration)
    ids = np.asarray([r["patient_id"] for r in rows["primary"]])
    patients = len(set(ids))
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, patients, size=(DRAWS, patients))
    metrics = all_metrics["primary"]
    scores = {name: patient_values(m["crps"].mean(axis=1), ids) for name, m in metrics.items()}
    contrasts = _contrasts(scores, draws)
    retention = _retention(metrics, ids, draws)
    nuisance = _nuisance(arrays, rows, calibration, draws)
    coverage = patient_values(metrics["JH"]["coverage_0.8"], ids).mean(axis=0)
    coverage_passes = (coverage >= 0.70) & (coverage <= 0.90)
    capability = _capability_flags(contrasts, retention, coverage, nuisance)
    gains = {}
    for arm in ARMS:
        item = paired_interval(scores["P_"+arm]-scores[arm], draws)
        item["direction"] = "positive_favors_forecast_over_own_persistence"
        gains[arm] = item
    result = {
        "status": "staged_only", "independent_review": "not_performed",
        "identity": {k: identity[k] for k in ("contract_digest", "source_sha256", "fitted_manifest_sha256")},
        "control_fits_sha256": _digest(identity["control_fits"]),
        "selected_calibration": calibration,
        "common_control_calibration_grid_scores": {
            k: v["grid_scores"] for k, v in fits.state["calibration_receipts"].items()},
        "cohort_identity_sha256": {k: _digest(v) for k, v in rows.items()},
        "protocol": {"version": 3, "arms": list(ARMS), "seeds": list(SEEDS), "conditions": list(CONDITIONS),
                     "compartments": ["necrotic_non_enhancing", "enhancing", "edema_flair"],
                     "units": "native_log1p_mm3", "cohorts": {k: list(v) for k, v in COHORTS.items()},
                     "aggregation": "seed_scores_then_rows_within_patient_then_equal_patients",
                     "calibration_grid": list(GRID), "observation_sd": OBSERVATION_SD,
                     "bootstrap_draws": DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
                     "bootstrap_patient_order": "sorted_patient_id", "bootstrap_draws_sha256": _digest(draws),
                     "contrast_family_size": 10, "adjusted_confidence": 0.995, "companion_confidence": 0.95,
                     "minimum_relative_reduction": 0.05, "retention_mae_margin": 0.05,
                     "retention_dice_margin": 0.02, "coverage_80_range": [0.70, 0.90],
                     "nuisance_relative_crps_margin": 0.10, "nuisance_mae_margin": 0.05,
                     "epochs": 20, "updates_per_epoch": 262, "total_successful_updates": 62880},
        "cohorts": cohorts, "primary_contrasts": contrasts,
        "factorial_interaction": contrasts["JH-JL-OH+OL"], "own_persistence_gains": gains,
        "current_state_retention": retention,
        "JH_coverage_80": {"per_compartment": coverage.tolist(), "passes_per_compartment": coverage_passes.tolist()},
        "nuisance": nuisance, "capability_flags": capability,
        "scoped_capability_outcome": "SCOPED_CAPABILITY_SUPPORTED" if all(capability.values())
                                      else "INCONCLUSIVE_FOR_FULL_CAPABILITY",
        "interpretation_limits": ["Reused development patients; not independent validation.",
                                  "Automated target agreement is not expert-verified biological prediction.",
                                  "A co-adapted system result is not intrinsic frozen-encoder information gain.",
                                  "Unresolved comparisons are inconclusive, never equivalence or no dynamics.",
                                  "Intensity normalization may cancel the transform; resolution is synthetic.",
                                  "Common frozen control encoder used internal13 checkpoint selection.",
                                  "Staged scoring is not budget completion, publication or independent approval."]}
    return _json_value(result)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
