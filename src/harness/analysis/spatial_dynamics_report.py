"""Saved-prediction evaluation and locked patient-level R37 decisions."""
from __future__ import annotations

from pathlib import Path
from statistics import NormalDist
import time

import numpy as np
from scipy.special import ndtr

from .spatial_dynamics_baselines import patient_weights, CALIBRATION_GRID
from .spatial_dynamics_fit import gaussian_scores, patient_mean
from .spatial_dynamics_data import canonical_hash, save_json

LEVELS = (.5, .8, .95)


def interval_gaussian(mean, scale, level):
    z = NormalDist().inv_cdf((1 + level) / 2)
    return mean - z * scale, mean + z * scale


def interval_mixture(centers, weights, observation_sd, level):
    """Marginal mixture quantiles, retaining the joint centers in saved outputs."""
    centers = np.asarray(centers)
    result = []
    for p in ((1 - level) / 2, (1 + level) / 2):
        lower = centers.min(axis=1) - 12 * observation_sd
        upper = centers.max(axis=1) + 12 * observation_sd
        for _ in range(60):
            middle = (lower + upper) / 2
            cdf = (ndtr((middle[:, None, :] - centers) / observation_sd) * weights[None, :, None]).sum(axis=1)
            lower = np.where(cdf < p, middle, lower)
            upper = np.where(cdf >= p, middle, upper)
        result.append((lower + upper) / 2)
    return tuple(result)


def gaussian_bundle(prediction, truth, calibration):
    scale = np.sqrt((calibration * prediction["intrinsic_scale"]) ** 2 + .05 ** 2)
    result = {"mean": prediction["mean"], "scale": scale,
              "crps": gaussian_scores(prediction["mean"], scale, truth),
              "uncalibrated_crps": gaussian_scores(prediction["mean"], np.sqrt(prediction["intrinsic_scale"] ** 2 + .05 ** 2), truth)}
    if "latent_mse" in prediction:
        result["latent_mse"] = prediction["latent_mse"]
    for level in LEVELS:
        lower, upper = interval_gaussian(prediction["mean"], scale, level)
        result[f"lower_{level}"] = lower
        result[f"upper_{level}"] = upper
    return result


def empirical_bundle(control, latest, truth, calibration):
    weights = patient_weights(control.patient_ids)
    centers = control.distribution(latest, calibration)
    result = {"mean": (centers * weights[None, :, None]).sum(axis=1),
              "crps": control.crps(latest, truth, calibration),
              "uncalibrated_crps": control.crps(latest, truth, 1.)}
    for level in LEVELS:
        lower, upper = interval_mixture(centers, weights, control.observation_sd, level)
        result[f"lower_{level}"] = lower
        result[f"upper_{level}"] = upper
    return result


def patient_values(values, rows):
    values = np.asarray(values)
    patients = sorted({r["patient_id"] for r in rows})
    ids = np.asarray([r["patient_id"] for r in rows])
    return patients, np.stack([values[ids == patient].mean(axis=0) for patient in patients])


def contrast(candidate, reference, rows, *, seed=2026, confidence=.95):
    """Seeds must already be averaged in score space, never prediction space."""
    if np.asarray(candidate).shape != np.asarray(reference).shape:
        raise ValueError("paired candidate/reference shape mismatch")
    patients, differences = patient_values(np.asarray(candidate) - reference, rows)
    if not len(patients) or not np.isfinite(differences).all():
        raise ValueError("invalid paired patient scores")
    if differences.ndim > 1:
        differences = differences.mean(axis=tuple(range(1, differences.ndim)))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(patients), size=(10000, len(patients)))
    boots = differences[draws].mean(axis=1)
    tail = (1 - confidence) / 2
    return {"difference": float(differences.mean()), "ci": np.quantile(boots, [tail, 1-tail]).tolist(),
            "confidence": confidence, "patients": len(patients), "examples": len(rows),
            "draws": 10000, "seed": seed}


def summarize_bundle(bundle, truth, rows):
    crps = np.asarray(bundle["crps"])
    mae = np.abs(bundle["mean"] - truth)
    result = {"patients": len({r["patient_id"] for r in rows}), "examples": len(rows),
              "crps": float(patient_mean(crps, rows).mean()), "compartment_crps": patient_mean(crps, rows).tolist(),
              "uncalibrated_crps": float(patient_mean(bundle["uncalibrated_crps"], rows).mean()),
              "mae": float(patient_mean(mae, rows).mean()), "compartment_mae": patient_mean(mae, rows).tolist(),
              "coverage": {}, "width": {}, "subsets": {}}
    for level in LEVELS:
        lo, hi = bundle[f"lower_{level}"], bundle[f"upper_{level}"]
        result["coverage"][str(level)] = patient_mean(((truth >= lo) & (truth <= hi)).astype(float), rows).tolist()
        result["width"][str(level)] = patient_mean(hi - lo, rows).tolist()
    if "latent_mse" in bundle:
        result["latent_mse_diagnostic"] = float(patient_weights([r["patient_id"] for r in rows]) @ bundle["latent_mse"])
    cutoff = np.asarray([r["cutoff_measured"] for r in rows])
    partitions = {"empty_target": truth == 0, "appearance": (cutoff == 0) & (truth > 0),
                  "disappearance": (cutoff > 0) & (truth == 0), "both_empty": (cutoff == 0) & (truth == 0)}
    for name, mask in partitions.items():
        values = []
        for c in range(3):
            selected = np.flatnonzero(mask[:, c])
            if len(selected):
                subset_rows = [rows[i] for i in selected]
                weights = patient_weights([r["patient_id"] for r in subset_rows])
                values.append({"patients": len({r["patient_id"] for r in subset_rows}), "examples": len(selected),
                               "crps": float(weights @ crps[selected, c]), "mae": float(weights @ mae[selected, c])})
            else:
                values.append({"patients": 0, "examples": 0, "crps": None, "mae": None})
        result["subsets"][name] = values
    return result


def expected_cohorts(cache):
    patients = set(cache["splits"]["unseen26"])
    primary = [r for r in cache["rows"] if r["patient_id"] in patients and r["primary_history"]]
    single = [r for r in cache["rows"] if r["patient_id"] in patients and not r["primary_history"]]
    earliest = {}
    for row in primary:
        earliest.setdefault(row["patient_id"], row)
    return {"primary": primary, "single": single, "nuisance": list(earliest.values())}


def assessment_identity(cache, contract_digest, fitting_sha256, artifact_manifest_sha256):
    return {"contract_digest": contract_digest, "fitting_completion_sha256": fitting_sha256,
            "fitted_artifact_manifest_sha256": artifact_manifest_sha256,
            "cohort_row_sha256": {name: canonical_hash(rows) for name, rows in expected_cohorts(cache).items()}}


def verify_assessment(bundle, marker, cache, identity):
    """Reject stale bundles, missing rows/seeds and nonfinite/incomplete predictions."""
    for key, value in identity.items():
        if bundle.get(key) != value or marker.get(key) != value:
            raise ValueError("assessment run/fitting/cohort provenance mismatch")
    expected = expected_cohorts(cache)
    for name, key in (("primary", "primary_rows"), ("single", "single_rows"), ("nuisance", "nuisance_rows")):
        if canonical_hash(bundle.get(key)) != canonical_hash(expected[name]):
            raise ValueError("assessment omitted or substituted locked cohort rows")
    def check(values, arm, rows):
        seeds = [42, 43, 44] if arm in {"H", "L", "G"} else [None]
        if len(values) != len(seeds) or [b.get("seed", "missing") for b in values] != seeds:
            raise ValueError("missing or mismatched prediction seeds")
        for prediction in values:
            fields = ["mean", "crps", "uncalibrated_crps"] + [f"{side}_{level}" for side in ("lower", "upper") for level in LEVELS]
            for field in fields:
                value = np.asarray(prediction[field], dtype=float)
                if value.shape != (len(rows), 3) or not np.isfinite(value).all():
                    raise ValueError("incomplete/nonfinite saved prediction array")
            if any((prediction[f"lower_{level}"] > prediction[f"upper_{level}"]).any() for level in LEVELS):
                raise ValueError("reversed saved prediction intervals")
    for cohort in ("primary", "single"):
        arms = bundle["cohorts"][cohort]
        wanted = {"H", "L", "G", "P", "E", "R"} if expected[cohort] else set()
        if set(arms) != wanted:
            raise ValueError("missing assessment cohort arms")
        for arm, values in arms.items():
            check(values, arm, expected[cohort])
    if set(bundle["nuisance"]) != {"H", "L"}:
        raise ValueError("missing nuisance arm")
    for arm in ("H", "L"):
        if set(bundle["nuisance"][arm]) != {"identity", "intensity", "resolution"}:
            raise ValueError("missing nuisance condition")
        for values in bundle["nuisance"][arm].values():
            check(values, arm, expected["nuisance"])
    return True


def publish_aggregate(path, record, authority):
    """Only publish after the authoritative report reservation completed within budget."""
    with authority.lock():
        ledger, events = authority.read_verified()
        stages = ledger["stages"]
        if not stages or stages[-1]["stage"] != "report" or any(s["status"] != "completed" for s in stages):
            raise RuntimeError("report resource/completion check did not pass")
        cpu = ledger["setup_cpu_charge_seconds"] + sum(s.get("cpu_seconds", 0.) for s in stages)
        gpu = sum(s.get("gpu_seconds", 0.) for s in stages)
        if time.time() > ledger["deadline_unix"] or cpu > ledger["cpu_seconds_limit"] or gpu > ledger["gpu_seconds_limit"]:
            raise TimeoutError("final resource checks failed; aggregate remains staged and ungraded")
        record.update(status="completed_not_scientifically_reviewed", completed_budget_journal_tip=events[-1]["event_sha256"])
        save_json(path, record)
    return record


def analyze(primary, rows, nuisance, nuisance_rows, out=None):
    """primary maps arm -> list of seed prediction bundles; baselines have one."""
    if set(primary) != {"H", "L", "G", "P", "E", "R"} or len({r["patient_id"] for r in rows}) < 15:
        raise ValueError("incomplete arms or insufficient primary history patients")
    earliest = {}
    for row in rows:
        earliest.setdefault(row["patient_id"], row)
    if canonical_hash(nuisance_rows) != canonical_hash(list(earliest.values())):
        raise ValueError("nuisance patients/cutoffs are incomplete or substituted")
    if set(nuisance) != {"H", "L"} or any(set(nuisance[a]) != {"identity", "intensity", "resolution"} or
            any(len(v) != 3 for v in nuisance[a].values()) for a in ("H", "L")):
        raise ValueError("nuisance arms, conditions or fitted seeds are incomplete")
    truth = np.asarray([r["truth"] for r in rows])
    scores, errors, summaries = {}, {}, {}
    for arm, bundles in primary.items():
        if len(bundles) != (3 if arm in {"H", "L", "G"} else 1):
            raise ValueError("missing fitted seed")
        summaries[arm] = [summarize_bundle(b, truth, rows) for b in bundles]
        scores[arm] = np.mean([b["crps"] for b in bundles], axis=0)
        errors[arm] = np.mean([np.abs(b["mean"] - truth) for b in bundles], axis=0)
    comparisons = {}
    for other in ("L", "P", "E", "R", "G"):
        adjusted = contrast(scores["H"], scores[other], rows, confidence=.99)
        adjusted["unadjusted_ci"] = contrast(scores["H"], scores[other], rows)["ci"]
        adjusted["relative_crps_reduction"] = 1 - float(patient_mean(scores["H"], rows).mean() / patient_mean(scores[other], rows).mean())
        comparisons[f"H-{other}"] = adjusted
    margins = {f"H-{other}": contrast(errors["H"], errors[other], rows) for other in ("L", "P")}
    secondary = {f"L-{other}": contrast(scores["L"], scores[other], rows) for other in ("P", "E")}
    for other in ("P", "E"):
        secondary[f"L-{other}"]["relative_crps_reduction"] = 1 - float(patient_mean(scores["L"], rows).mean() / patient_mean(scores[other], rows).mean())
    coverage = np.mean([s["coverage"]["0.8"] for s in summaries["H"]], axis=0)
    nuisance_report = {}
    for arm in ("H", "L"):
        nuisance_report[arm] = {}
        nt = np.asarray([r["truth"] for r in nuisance_rows])
        identity = nuisance[arm]["identity"]
        identity_crps = np.mean([b["crps"] for b in identity], axis=0)
        identity_mae = np.mean([np.abs(b["mean"] - nt) for b in identity], axis=0)
        for condition in ("identity", "intensity", "resolution"):
            bundles = nuisance[arm][condition]
            crps = np.mean([b["crps"] for b in bundles], axis=0)
            error = np.mean([np.abs(b["mean"] - nt) for b in bundles], axis=0)
            drift = np.mean([np.abs(b["mean"] - base["mean"]) for b, base in zip(bundles, identity)], axis=0)
            nuisance_report[arm][condition] = {"seed_summaries": [summarize_bundle(b, nt, nuisance_rows) for b in bundles],
                "crps_difference": contrast(crps, identity_crps, nuisance_rows),
                "mae_difference": contrast(error, identity_mae, nuisance_rows),
                "relative_crps_worsening": float(patient_mean(crps, nuisance_rows).mean() / patient_mean(identity_crps, nuisance_rows).mean() - 1),
                "mean_absolute_volume_drift": patient_mean(drift, nuisance_rows).tolist(),
                "mean_absolute_volume_drift_ci": contrast(drift, np.zeros_like(drift), nuisance_rows),
                "coverage_difference_ci": {str(level): [contrast(
                    np.mean([((nt >= b[f'lower_{level}']) & (nt <= b[f'upper_{level}'])).astype(float) for b in bundles], axis=0)[:, c],
                    np.mean([((nt >= b[f'lower_{level}']) & (nt <= b[f'upper_{level}'])).astype(float) for b in identity], axis=0)[:, c],
                    nuisance_rows) for c in range(3)] for level in LEVELS}}
    nuisance_pass = all(nuisance_report["H"][c]["relative_crps_worsening"] <= .1 and
                        nuisance_report["H"][c]["mae_difference"]["difference"] <= .05 for c in ("intensity", "resolution"))
    primary_pass = comparisons["H-L"]["relative_crps_reduction"] >= .05 and all(comparisons[f"H-{c}"]["ci"][1] < 0 for c in ("L", "P", "E", "R"))
    mae_pass = all(value["ci"][1] < .05 for value in margins.values())
    coverage_pass = bool(((coverage >= .7) & (coverage <= .9)).all())
    history_pass = primary_pass and mae_pass and coverage_pass and nuisance_pass
    spatial_pass = history_pass and comparisons["H-G"]["ci"][1] < 0
    current_pass = all(s["ci"][1] < 0 and s["relative_crps_reduction"] >= .05 for s in secondary.values())
    # Failed capability criteria do not prove equivalence or absence of useful dynamics.
    decision = "ADDED_HISTORY_BENEFIT" if history_pass else "INCONCLUSIVE_FOR_FULL_CAPABILITY"
    record = {"schema": "r37-full-aggregate-v1", "status": "staged_not_graded", "decision": decision,
              "history_benefit": history_pass, "earlier_spatial_detail_benefit": spatial_pass,
              "secondary_current_state_prediction_benefit": current_pass,
              "criteria": {"primary_contrasts": primary_pass, "mae_noninferiority": mae_pass,
                           "coverage": coverage_pass, "nuisance": nuisance_pass},
              "cohort": {"patients": len({r["patient_id"] for r in rows}), "examples": len(rows),
                         "status": "Reused LUMIERE development assessment; automated targets"},
              "units": "log1p(mm3)", "aggregation": "average seed scores, then examples within patient, then equal patients; not ensemble predictions",
              "E_primary_condition": "gaussian_observation", "multiplicity": "five H contrasts, two-sided 99% patient bootstrap intervals",
              "arm_scores": {arm: {"crps": float(patient_mean(scores[arm], rows).mean()),
                                    "mae": float(patient_mean(errors[arm], rows).mean()),
                                    "compartment_crps": patient_mean(scores[arm], rows).tolist(),
                                    "compartment_mae": patient_mean(errors[arm], rows).tolist()} for arm in scores},
              "arm_seed_summaries": summaries, "primary_comparisons": comparisons,
              "mae_margins": margins, "secondary_latest_comparisons": secondary,
              "H_80_coverage_by_compartment": coverage.tolist(), "nuisance": nuisance_report,
              "demonstrated_harms": [name for name, value in comparisons.items() if value["ci"][0] > 0],
              "limits": ["Automated masks are not expert biological truth", "Reused development patients are not independent validation",
                         "Historical elapsed times use approximate week-bin gaps", "Unknown future scheduling and interventions",
                         "Restricted channelwise spatial covariance and frozen decoder bias", "No fixed-day, multistep or treatment-effect claim"],
              "stopping_decision": "STOP_AFTER_ONE_ANALYSIS_REGARDLESS_OF_OUTCOME",
              "automatic_followup": False}
    if out is not None:
        save_json(Path(out), record)
    return record
