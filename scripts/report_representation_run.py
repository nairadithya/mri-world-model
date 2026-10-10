"""Verify and summarize all fixed R35 arms without selecting a seed or view."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.data.representation import COMPARTMENTS, CONTRACT, digest, save_json
from scripts.audit_representation_predictions import audit

SEEDS = (42, 43, 44)
ARMS = ("state", "paired", "later_only", "frozen")


def load_predictions(path: Path, expected: dict | None = None) -> dict:
    with np.load(path, allow_pickle=False) as data:
        result = {k: data[k].copy() for k in data.files}
    for key in ("state_truth", "change_truth"):
        if not np.isfinite(result[key]).all():
            raise ValueError(f"nonfinite target in {path}")
    for key in ("visit_ids", "row_ids"):
        if len(set(result[key])) != len(result[key]):
            raise ValueError(f"duplicate row identifiers in {path}")
    if expected is not None:
        for key in ("visit_ids", "row_ids", "state_patients", "pair_patients"):
            np.testing.assert_array_equal(result[key], expected[key])
        for key in ("state_truth", "change_truth"):
            np.testing.assert_allclose(result[key], expected[key], rtol=0, atol=1e-7)
    return result


def seed_errors(records: list[dict], key: str, state: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    truth = records[0]["state_truth" if state else "change_truth"]
    groups = records[0]["state_patients" if state else "pair_patients"]
    for record in records:
        np.testing.assert_array_equal(groups, record["state_patients" if state else "pair_patients"])
        np.testing.assert_allclose(truth, record["state_truth" if state else "change_truth"], rtol=0, atol=1e-7)
        if record[key].shape != truth.shape or not np.isfinite(record[key]).all():
            raise ValueError(f"invalid prediction array {key}")
    errors = np.stack([np.abs(r[key] - truth) for r in records])
    unique = sorted(set(groups))
    per_patient = np.stack([errors[:, groups == p].mean((1, 2)) for p in unique], axis=1)
    return errors, per_patient, truth, groups


def aggregate(records: list[dict], key: str, state: bool) -> dict:
    errors, per, truth, groups = seed_errors(records, key, state)
    return {"seed_mean_patient_uniform_mae": float(per.mean()),
            "seed_mean_pooled_mae": float(errors.mean()),
            "seed_patient_uniform_mae": per.mean(1).tolist(),
            "seed_mean_pooled_compartment_mae": dict(zip(COMPARTMENTS, errors.mean((0, 1)).tolist())),
            "n_rows": len(truth), "n_patients": len(set(groups))}


def compare(records: list[dict], key: str, baselines: list[dict], baseline_key: str, state: bool) -> dict:
    _, a, truth, groups = seed_errors(records, key, state)
    _, b, baseline_truth, baseline_groups = seed_errors(baselines, baseline_key, state)
    np.testing.assert_allclose(truth, baseline_truth, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(groups, baseline_groups)
    delta = a.mean(0) - b.mean(0)
    rng = np.random.default_rng(2026)
    means = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(1)
    return {"difference": float(delta.mean()), "paired_patient_95ci": np.quantile(means, [.025, .975]).tolist(),
            "family12_adjusted_ci": np.quantile(means, [.05 / 24, 1 - .05 / 24]).tolist(),
            "n_patients": len(delta), "bootstrap_seed": 2026,
            "interval_status": "descriptive seed-mean summary, not an additional prespecified gate"}


def direct_report(matrix: dict, reference: dict) -> dict:
    arms = {a: [matrix[s, a] for s in SEEDS] for a in ARMS}
    scores = {a: {"state": aggregate(rs, "state_prediction", True),
                  "state_mean": aggregate(rs, "state_mean", True),
                  "independent_decoder_difference": aggregate(rs, "independent_decoder_difference", False),
                  "zero": aggregate(rs, "zero", False)} for a, rs in arms.items()}
    for a in ARMS[1:]:
        scores[a]["pair_head"] = aggregate(arms[a], "pair_head", False)
    comparisons = {"state_vs_frozen_source52": compare(arms["state"], "state_prediction", [reference], "state_prediction", True),
                   "state_vs_source_mean": compare(arms["state"], "state_prediction", arms["state"], "state_mean", True),
                   "state_difference_vs_zero": compare(arms["state"], "independent_decoder_difference", arms["state"], "zero", False)}
    for name, baseline, key in (("zero", "paired", "zero"), ("later_only", "later_only", "pair_head"),
                                ("independent_state", "state", "independent_decoder_difference"),
                                ("same_encoder_difference", "paired", "independent_decoder_difference"),
                                ("frozen_encoder_pair", "frozen", "pair_head")):
        comparisons[f"paired_head_vs_{name}"] = compare(arms["paired"], "pair_head", arms[baseline], key, False)
    return {"arms": scores, "comparisons": comparisons, "frozen_source52": aggregate([reference], "state_prediction", True)}


def report(results: str, transfer: str, reference_source: str, prepared: str, out: str) -> dict:
    source, query, output = Path(results), Path(transfer), Path(out)
    output.mkdir(parents=True, exist_ok=True)
    status = json.loads((query / "status.json").read_text())
    if status["status"] != "complete" or len(status["runs"]) != 12 or any(r["status"] != "complete" for r in status["runs"]):
        raise ValueError("transfer matrix is incomplete")
    metadata = json.loads((Path(prepared) / "prepared.json").read_text())
    assert digest(Path(prepared) / "prepared.json") == status["prepared_sha256"]
    lookup = {v["visit_id"]: i for i, v in enumerate(metadata["visits"])}
    source_indexes = [lookup[p["source_visit_id"]] for p in metadata["pairs"]]
    target_indexes = [lookup[p["target_visit_id"]] for p in metadata["pairs"]]
    fresh = load_predictions(Path(reference_source))
    ref_query = load_predictions(query / "frozen-reference-source52.predictions.npz")
    source_direct, query_direct, transfer_views, nuisances, hashes, audits = {}, {}, {}, {}, {}, {}
    expected_source, expected_query = fresh, ref_query
    for seed in SEEDS:
        for arm in ARMS:
            label = f"seed{seed}/{arm}"
            a, b = source / f"seed{seed}" / arm, query / f"seed{seed}" / arm
            cp_hash = digest(a / "best.pt")
            features = torch.load(b / "transfer.features.pt", map_location="cpu", weights_only=False)
            assert features["contract"] == CONTRACT
            assert features["provenance"]["checkpoint_sha256"] == cp_hash
            assert features["provenance"]["query_prepared_sha256"] == status["prepared_sha256"]
            assert len(features["features"]) == len(ref_query["visit_ids"])
            assert all(np.isfinite(v).all() for v in features["features"].values())
            if arm != "state":
                assert len(features["pair_features"]) == len(ref_query["row_ids"])
                assert all(np.isfinite(v).all() for v in features["pair_features"].values())
            source_direct[seed, arm] = load_predictions(a / "direct.predictions.npz", expected_source)
            query_direct[seed, arm] = load_predictions(b / "transfer.direct.predictions.npz", expected_query)
            for budget, filename in ((26, "transfer"), (52, "transfer.source52")):
                prediction = load_predictions(b / f"{filename}.predictions.npz", expected_query)
                scored = json.loads((b / f"{filename}.json").read_text())
                assert scored["provenance"]["checkpoint_sha256"] == cp_hash
                assert scored["provenance"]["training_metadata_sha256"] == status["training_metadata_sha256"]
                assert len(scored["provenance"]["readout_training_patients"]) == budget
                assert not set(scored["provenance"]["readout_training_patients"]) & set(prediction["state_patients"])
                saved = torch.load(b / f"{filename}.readouts.pt", map_location="cpu", weights_only=False)
                assert saved["training_patients"] == sorted(scored["provenance"]["readout_training_patients"])
                np.testing.assert_array_equal(prediction["visit_ids"], [v["visit_id"] for v in metadata["visits"]])
                np.testing.assert_array_equal(prediction["row_ids"], [p["row_id"] for p in metadata["pairs"]])
                x = np.stack([features["features"][v["visit_id"]] for v in metadata["visits"]])
                xa, xb = x[source_indexes], x[target_indexes]
                views = {"state": x, "difference": xb - xa, "concatenation": np.concatenate([xa, xb], 1), "later_only": xb}
                if arm != "state":
                    views["learned_pair"] = np.stack([features["pair_features"][p["row_id"]] for p in metadata["pairs"]])
                for name, values in views.items():
                    scaler, readout, _ = saved["readouts"][name]
                    np.testing.assert_allclose(readout.predict(scaler.transform(values)), prediction["state_prediction" if name == "state" else name], rtol=0, atol=1e-6)
                transfer_views[seed, arm, budget] = prediction
            nuisances[seed, arm] = json.loads((a / "nuisance.json").read_text())
            for cohort, path in (("LUMIERE", a / "direct.predictions.npz"), ("SAILOR", b / "transfer.direct.predictions.npz")):
                audit_path = output / "audits" / cohort / f"seed{seed}" / arm / "state.json"
                audits[f"{cohort}/{label}"] = audit(str(path), str(audit_path))
            hashes[label] = {"checkpoint": cp_hash, "source_predictions": digest(a / "direct.predictions.npz"),
                             "query_predictions": digest(b / "transfer.direct.predictions.npz"),
                             "query_features": digest(b / "transfer.features.pt")}
    nuisance_report = {}
    for arm in ARMS:
        nuisance_report[arm] = {}
        for kind in ("identity", "intensity", "noise", "resolution", "bias"):
            records = [nuisances[s, arm][kind] for s in SEEDS]
            assert all(r["n_patients"] == 26 for r in records)
            patients = sorted(records[0]["per_patient"])
            deltas = np.asarray([[r["per_patient"][p]["false_change_mae"] - nuisances[s, arm]["identity"]["per_patient"][p]["false_change_mae"]
                                  for p in patients] for s, r in zip(SEEDS, records)]).mean(0)
            rng = np.random.default_rng(2026)
            draws = deltas[rng.integers(0, len(deltas), (10000, len(deltas)))].mean(1)
            nuisance_report[arm][kind] = {"seed_mean_false_change_mae": float(np.mean([r["false_change_mae"] for r in records])),
                                         "seed_mean_state_readout_drift": float(np.mean([r["state_readout_drift"] for r in records])),
                                         "false_change_excess_over_identity": float(deltas.mean()),
                                         "paired_patient_95ci": np.quantile(draws, [.025, .975]).tolist(),
                                         "n_patients": 26}
    nuisance_report["frozen_reference"] = {}
    reference_nuisance = json.loads((source.parent / "reference/nuisance.json").read_text())
    for kind in ("identity", "intensity", "noise", "resolution", "bias"):
        value = reference_nuisance[kind]
        nuisance_report["frozen_reference"][kind] = {
            "seed_mean_false_change_mae": value["false_change_mae"],
            "seed_mean_state_readout_drift": value["state_readout_drift"],
            "false_change_excess_over_identity": value["false_change_mae"] - reference_nuisance["identity"]["false_change_mae"],
            "n_patients": value["n_patients"], "n_fits": 1}
    calibrated = {}
    for budget in (26, 52):
        calibrated[str(budget)] = {}
        for arm in ARMS:
            rs = [transfer_views[s, arm, budget] for s in SEEDS]
            keys = ["zero", "difference", "concatenation", "later_only", "independent_state_difference", "train_mean_change"]
            if arm != "state":
                keys.append("learned_pair")
            calibrated[str(budget)][arm] = {"state": aggregate(rs, "state_prediction", True),
                                            "state_mean": aggregate(rs, "state_mean", True),
                                            "observed_change": {k: aggregate(rs, k, False) for k in keys},
                                            "comparisons": {"state_vs_mean": compare(rs, "state_prediction", rs, "state_mean", True)}}
            for view in ("difference", "concatenation", "learned_pair"):
                if view not in keys:
                    continue
                for baseline in ("zero", "later_only", "independent_state_difference"):
                    calibrated[str(budget)][arm]["comparisons"][f"{view}_vs_{baseline}"] = compare(rs, view, rs, baseline, False)
    for budget in (26, 52):
        reference = load_predictions(query / f"frozen-reference-source{budget}.predictions.npz", expected_query)
        calibrated[str(budget)]["frozen_reference"] = {
            "state": aggregate([reference], "state_prediction", True),
            "state_mean": aggregate([reference], "state_mean", True),
            "observed_change": {k: aggregate([reference], k, False) for k in
                                ("zero", "difference", "concatenation", "later_only", "independent_state_difference", "train_mean_change")}}
    result = {"status": "complete", "trained_arms": 12, "seeds": list(SEEDS),
              "aggregation": "mean errors across seeds, not ensemble predictions; patients are bootstrap units",
              "cohort_status": "both cohorts are developmental; SAILOR receives no fitting or checkpoint selection",
              "LUMIERE_direct": direct_report(source_direct, fresh),
              "SAILOR_direct": direct_report(query_direct, ref_query),
              "SAILOR_source_readouts": calibrated, "LUMIERE_nuisance": nuisance_report,
              "state_audits": audits, "artifact_hashes": hashes,
              "blocked": ["biological change needs expert annotations and segmentation repeatability",
                          "independent review needs native subagent availability",
                          "untouched external validation needs a new reserved cohort"]}
    save_json(output / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--transfer", required=True)
    parser.add_argument("--reference-source", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = report(args.results, args.transfer, args.reference_source, args.prepared, args.out)
    print(json.dumps({"status": result["status"], "trained_arms": result["trained_arms"]}))
