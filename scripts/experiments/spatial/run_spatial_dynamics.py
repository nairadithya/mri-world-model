#!/usr/bin/env python3
"""Content-bound, remote-only execution of the fixed R37 full pilot."""
from __future__ import annotations

import argparse
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from src.harness.analysis.spatial_dynamics_data import (
    SpatialData, canonical_hash, configure_torch, digest, extract_cache,
    load_bound_inputs, load_encoder, save_json,
)
from src.harness.analysis.spatial_dynamics_fit import (
    SEEDS, calibrate_gaussian, fit_neural, fit_ridge, neural_predictions,
    predict_ridge,
)
from src.harness.analysis.spatial_dynamics_baselines import (
    fit_empirical_change, fit_persistence, fit_uncertainty_scale, CALIBRATION_GRID,
)
from src.harness.analysis.spatial_dynamics_report import (
    analyze, assessment_identity, empirical_bundle, gaussian_bundle, publish_aggregate, summarize_bundle, verify_assessment,
)
from src.harness.analysis.spatial_dynamics_runtime import budget_stage, production_authority

SOURCES = ["scripts/experiments/spatial/run_spatial_dynamics.py", "scripts/experiments/spatial/run_spatial_noise_ablation.py",
           "src/model/spatial_dynamics.py", "src/model/representation.py",
           "src/data/representation.py", "src/preprocessing/transforms.py", "src/preprocessing/geometry.py"] + [
    f"src/harness/analysis/spatial_dynamics_{name}.py" for name in ("data", "fit", "report", "runtime", "baselines", "protocol")]


def source_hashes():
    return {p: digest(ROOT / p) for p in SOURCES}


def verify_review(path):
    review = json.loads(Path(path).read_text())
    if review.get("verdict") != "APPROVED_FOR_R37_IMPLEMENTATION":
        raise ValueError("independent implementation approval is absent")
    if review.get("reviewed_code") != source_hashes():
        raise ValueError("implementation review covers different code bytes")
    report = ROOT / review["review_path"]
    if digest(report) != review["review_sha256"] or review["verdict"] not in report.read_text():
        raise ValueError("independent report bytes or verdict do not match gate")
    return review


def contract_for(out, review_path, ledger):
    review = verify_review(review_path)
    payload = torch.load(out / "features.pt", map_location="cpu", weights_only=False)
    authority = production_authority()
    authority.require_path(ledger)
    budget, _ = authority.read_verified()
    return {"schema": "r37-full-run-v1", "code_sha256": source_hashes(),
            "budget_identity": authority.snapshot(ledger), "geometry_receipts_sha256": payload["geometry_receipts_sha256"],
            "cache_sha256": digest(out / "features.pt"), "image_sha256": payload["image_hashes"],
            "original_input_sha256": payload["input_hashes"], "checkpoint_sha256": payload["checkpoint_sha256"],
            "row_manifest_sha256": canonical_hash(payload["rows"]), "splits": payload["splits"],
            "review": review, "review_gate_sha256": digest(review_path),
            "config": {"arms": ["H", "L", "P", "E", "R", "G"], "seeds": list(SEEDS), "epochs": 20,
                       "E_primary_condition": "gaussian_observation", "H_contrasts": 5,
                       "adjusted_confidence": .99, "observation_sd": .05,
                       "calibration_grid": list(CALIBRATION_GRID), "bootstrap_draws": 10000, "bootstrap_seed": 2026,
                       "sampling": "patient-uniform weighted replacement; matched saved orders",
                       "dtype": "float32 neural/encoder; float64 ridge moments; TF32 and AMP disabled",
                       "timing": "approximate historical week-bin difference times seven days; no future gap input",
                       "ridge_cv_noise": "fold-training RMS for CV CRPS; selected-alpha OOF RMS for final distribution"},
            "budget": {k: budget[k] for k in ("started_at", "deadline_unix", "gpu_seconds_limit", "cpu_seconds_limit", "prior_expenditure")},
            "hardware": {"host": os.uname().nodename, "gpu": torch.cuda.get_device_name(0),
                         "torch": torch.__version__, "cuda": torch.version.cuda},
            "environment": sorted(f"{p.metadata['Name']}=={p.version}" for p in distributions()),
            "protocol_sha256": digest(ROOT / "info/METHODS.md")}


def verify_contract(out, expected, ledger):
    contract = json.loads((out / "contract.json").read_text())
    if canonical_hash(contract) != expected:
        raise ValueError("run-contract digest mismatch")
    production_authority().verify_extension(ledger, contract["budget_identity"])
    if contract["code_sha256"] != source_hashes() or digest(out / "features.pt") != contract["cache_sha256"]:
        raise ValueError("code or cache changed after lock")
    if digest(ROOT / "info/METHODS.md") != contract["protocol_sha256"]:
        raise ValueError("protocol changed after lock")
    original, _, _, _ = load_bound_inputs()
    if {k: digest(p) for k, p in original["files"].items()} != contract["original_input_sha256"]:
        raise ValueError("original input metadata changed after lock")
    verify_review(out / "implementation-gate.json")
    payload = torch.load(out / "features.pt", map_location="cpu", weights_only=False)
    if canonical_hash(payload["rows"]) != contract["row_manifest_sha256"] or payload["splits"] != contract["splits"]:
        raise ValueError("row/split binding mismatch")
    if canonical_hash(payload["geometry_receipts"]) != contract["geometry_receipts_sha256"] or payload["image_hashes"] != {
            p: record["file_sha256"] for p, record in payload["geometry_receipts"].items()}:
        raise ValueError("geometry receipt/content binding mismatch")
    for relative, expected_hash in contract["image_sha256"].items():
        if digest(ROOT / relative) != expected_hash:
            raise ValueError("MRI bytes changed after lock")
    return contract, payload


def fit_controls(data, out):
    source, validation = data.select("source52"), data.select("validation13")
    latest = data.latest_decoded(source)
    truth = np.asarray([r["truth"] for r in source])
    ids = [r["patient_id"] for r in source]
    p = fit_persistence(latest, truth, ids)
    p_pred = {"mean": data.latest_decoded(validation),
              "intrinsic_scale": np.broadcast_to(p.residual_rms, (len(validation), 3)).copy()}
    p_k, p_grid = calibrate_gaussian(p_pred, validation)
    e = fit_empirical_change(latest, truth, ids, condition="gaussian_observation")
    validation_truth = np.asarray([r["truth"] for r in validation])
    grid = np.stack([e.crps(p_pred["mean"], validation_truth, k) for k in CALIBRATION_GRID])
    e_k = fit_uncertainty_scale(grid, [r["patient_id"] for r in validation])
    torch.save({"P": p, "E": e, "P_calibration": p_k, "E_calibration": e_k,
                "P_validation_grid": p_grid, "E_validation_grid": grid}, out / "controls.pt")


def fitting_identity(out, contract_digest):
    path = out / "fitting-complete.json"
    completion = json.loads(path.read_text())
    expected = {"controls.pt", "ridge/ridge.pt"} | {
        f"seed{seed}/{arm}{suffix}.pt" for seed in SEEDS for arm in ("H", "L", "G")
        for suffix in ("", "-last", "-partial")}
    if completion.get("contract_digest") != contract_digest or completion.get("seeds") != list(SEEDS):
        raise ValueError("fitting completion belongs to another contract or incomplete seeds")
    if set(completion["weight_hashes"]) != expected:
        raise ValueError("fitted artifact manifest is incomplete")
    for relative, expected_hash in completion["weight_hashes"].items():
        if digest(out / relative) != expected_hash:
            raise ValueError("fitted artifact changed after fitting")
    return digest(path), canonical_hash(completion["weight_hashes"])


def evaluate(data, out, identity):
    primary = data.select("unseen26", True)
    single = data.select("unseen26", False)
    if len({r["patient_id"] for r in primary}) < 15:
        raise ValueError("insufficient history-eligible assessment patients")
    all_bundles = {"primary": {}, "single": {}}
    controls = torch.load(out / "controls.pt", weights_only=False)
    ridge = torch.load(out / "ridge/ridge.pt", map_location="cpu", weights_only=False)
    for cohort, rows in (("primary", primary), ("single", single)):
        if not rows:
            continue
        truth = np.asarray([r["truth"] for r in rows])
        latest = data.latest_decoded(rows)
        all_bundles[cohort]["P"] = [gaussian_bundle({"mean": latest,
            "intrinsic_scale": np.broadcast_to(controls["P"].residual_rms, latest.shape).copy()}, truth, controls["P_calibration"])]
        all_bundles[cohort]["E"] = [empirical_bundle(controls["E"], latest, truth, controls["E_calibration"])]
        all_bundles[cohort]["R"] = [gaussian_bundle(predict_ridge(data, rows, ridge), truth, ridge["receipt"]["calibration"])]
        for arm in ("H", "L", "G"):
            bundles = []
            for seed in SEEDS:
                from src.model.spatial_dynamics import SpatialDynamics
                checkpoint = torch.load(out / f"seed{seed}/{arm}.pt", map_location="cpu", weights_only=False)
                model = SpatialDynamics().to(data.device).eval().requires_grad_(False)
                model.load_state_dict(checkpoint["state_dict"])
                prediction = neural_predictions(model, data, rows, arm)
                bundles.append(gaussian_bundle(prediction, truth, checkpoint["receipt"]["calibration"]))
                del model
            all_bundles[cohort][arm] = bundles
        for arm, values in all_bundles[cohort].items():
            for i, bundle in enumerate(values):
                bundle["seed"] = SEEDS[i] if arm in {"H", "L", "G"} else None
    # Canonical row order is patient then original chronological cutoff, never outcomes.
    chosen = {}
    for row in primary:
        chosen.setdefault(row["patient_id"], row)
    nuisance_rows = list(chosen.values())
    bound, _, _, _ = load_bound_inputs()
    encoder = load_encoder(bound["checkpoint"], data.device)
    overrides = {c: [data.changed_latest(r, encoder, ROOT, c) for r in nuisance_rows]
                 for c in ("identity", "intensity", "resolution")}
    for row, feature in zip(nuisance_rows, overrides["identity"]):
        if not torch.allclose(feature, data.maps[data.index[row["cutoff_visit_id"]]], atol=1e-5, rtol=1e-5):
            raise ValueError("identity nuisance re-encode does not reproduce cached map")
    nuisance_results = {arm: {} for arm in ("H", "L")}
    nt = np.asarray([r["truth"] for r in nuisance_rows])
    for arm in ("H", "L"):
        for condition in overrides:
            bundles = []
            for seed in SEEDS:
                from src.model.spatial_dynamics import SpatialDynamics
                checkpoint = torch.load(out / f"seed{seed}/{arm}.pt", map_location="cpu", weights_only=False)
                model = SpatialDynamics().to(data.device).eval().requires_grad_(False)
                model.load_state_dict(checkpoint["state_dict"])
                prediction = neural_predictions(model, data, nuisance_rows, arm, overrides=overrides[condition])
                bundles.append(gaussian_bundle(prediction, nt, checkpoint["receipt"]["calibration"]))
                del model
            for seed, bundle in zip(SEEDS, bundles):
                bundle["seed"] = seed
            nuisance_results[arm][condition] = bundles
    payload = {**identity, "cohorts": all_bundles, "primary_rows": primary, "single_rows": single,
               "nuisance": nuisance_results, "nuisance_rows": nuisance_rows}
    torch.save(payload, out / "private_predictions.pt")
    arrays = {}
    for cohort, arms in all_bundles.items():
        rows = primary if cohort == "primary" else single
        arrays[f"{cohort}_row_ids"] = np.asarray([r["row_id"] for r in rows])
        arrays[f"{cohort}_patient_ids"] = np.asarray([r["patient_id"] for r in rows])
        arrays[f"{cohort}_truth"] = np.asarray([r["truth"] for r in rows]).reshape(-1, 3)
        arrays[f"{cohort}_cutoff_measured"] = np.asarray([r["cutoff_measured"] for r in rows]).reshape(-1, 3)
        for arm, bundles in arms.items():
            arrays[f"{cohort}_{arm}_seed_ids"] = np.asarray([b["seed"] if b["seed"] is not None else -1 for b in bundles])
            for i, bundle in enumerate(bundles):
                for key, value in bundle.items():
                    if key != "seed":
                        arrays[f"{cohort}_{arm}_{i}_{key}"] = value
    arrays["nuisance_row_ids"] = np.asarray([r["row_id"] for r in nuisance_rows])
    arrays["nuisance_patient_ids"] = np.asarray([r["patient_id"] for r in nuisance_rows])
    arrays["nuisance_truth"] = nt
    arrays["nuisance_cutoff_measured"] = np.asarray([r["cutoff_measured"] for r in nuisance_rows])
    for arm, conditions in nuisance_results.items():
        for condition, bundles in conditions.items():
            arrays[f"nuisance_{arm}_{condition}_seed_ids"] = np.asarray([b["seed"] for b in bundles])
            for i, bundle in enumerate(bundles):
                for key, value in bundle.items():
                    if key != "seed":
                        arrays[f"nuisance_{arm}_{condition}_{i}_{key}"] = value
    np.savez_compressed(out / "private_predictions.npz", **arrays)
    save_json(out / "assessment-complete.json", {**identity, "status": "predictions_staged_not_graded", "primary_examples": len(primary),
        "primary_patients": len(chosen), "single_examples": len(single), "predictions_sha256": digest(out / "private_predictions.pt"),
        "predictions_npz_sha256": digest(out / "private_predictions.npz")})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["inventory", "encode", "lock", "fit", "evaluate", "report"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--contract-digest")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    configure_torch()
    gpu = args.stage in {"encode", "lock", "fit", "evaluate"}
    if gpu and (not torch.cuda.is_available() or "5090" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("operator-selected mosaic-gpu RTX 5090 is required")
    with budget_stage(args.ledger, args.stage, gpu=gpu, preflight=args.stage in {"inventory", "encode", "lock"}) as guard:
        if args.stage == "inventory":
            _, old, _, rows = load_bound_inputs()
            save_json(args.out / "inventory.json", {"counts": old["counts"], "exclusions": old["exclusion_counts"],
                "rows_sha256": canonical_hash(rows), "input_hashes": old["input_hashes"], "timing": old["timing"]})
            print(json.dumps(old["counts"], indent=2))
        elif args.stage == "encode":
            if args.review is None:
                raise ValueError("encode requires --review")
            verify_review(args.review)
            extract_cache(ROOT, args.out / "features.pt")
        elif args.stage == "lock":
            if args.review is None:
                raise ValueError("lock requires --review")
            contract = contract_for(args.out, args.review, args.ledger)
            save_json(args.out / "implementation-gate.json", json.loads(args.review.read_text()))
            save_json(args.out / "contract.json", contract)
            print(json.dumps({"contract_digest": canonical_hash(contract), "output": str(args.out)}, indent=2))
        else:
            if not args.contract_digest:
                raise ValueError("execution requires --contract-digest")
            contract, payload = verify_contract(args.out, args.contract_digest, args.ledger)
            if args.stage == "fit":
                data = SpatialData(payload)
                fit_controls(data, args.out)
                fit_ridge(data, args.out / "ridge", guard)
                for seed in SEEDS:
                    fit_neural(data, args.out / f"seed{seed}", seed, guard)
                artifacts = list(args.out.glob("seed*/*.pt")) + [args.out / "controls.pt", args.out / "ridge/ridge.pt"]
                save_json(args.out / "fitting-complete.json", {"contract_digest": args.contract_digest, "seeds": list(SEEDS),
                    "weight_hashes": {str(p.relative_to(args.out)): digest(p) for p in sorted(artifacts)}})
            elif args.stage == "evaluate":
                if not (args.out / "fitting-complete.json").is_file():
                    raise ValueError("fitting is incomplete; assessment is blocked")
                fitted_sha, manifest_sha = fitting_identity(args.out, args.contract_digest)
                identity = assessment_identity(payload, args.contract_digest, fitted_sha, manifest_sha)
                evaluate(SpatialData(payload), args.out, identity)
            else:
                assessment = json.loads((args.out / "assessment-complete.json").read_text())
                if digest(args.out / "private_predictions.pt") != assessment["predictions_sha256"] or digest(
                        args.out / "private_predictions.npz") != assessment["predictions_npz_sha256"]:
                    raise ValueError("saved assessment prediction hash mismatch")
                fitted_sha, manifest_sha = fitting_identity(args.out, args.contract_digest)
                identity = assessment_identity(payload, args.contract_digest, fitted_sha, manifest_sha)
                bundle = torch.load(args.out / "private_predictions.pt", map_location="cpu", weights_only=False)
                verify_assessment(bundle, assessment, payload, identity)
                record = analyze(bundle["cohorts"]["primary"], bundle["primary_rows"], bundle["nuisance"],
                                 bundle["nuisance_rows"])
                if bundle["single_rows"]:
                    truth = np.asarray([r["truth"] for r in bundle["single_rows"]])
                    record["single_visit_companion"] = {arm: [summarize_bundle(b, truth, bundle["single_rows"]) for b in values]
                        for arm, values in bundle["cohorts"]["single"].items()}
                record.update(**identity, reporting_approval=None,
                              classification_scope="Mechanical protocol classification pending independent result review")
                save_json(args.out / "aggregate-staged.json", record)
    if args.stage == "report":
        publish_aggregate(args.out / "aggregate.json", record, production_authority())
        print(json.dumps({k: record[k] for k in ("status", "decision", "criteria", "cohort")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
