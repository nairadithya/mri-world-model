"""Patient-cluster R36 reporting and matched source-only scalar readouts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.data.representation import COMPARTMENTS, digest, json_hash, save_json
from src.harness.analysis.representation import fit_ridge, metrics
from src.harness.analysis.delta_protocol import CONTRACT, load_contract, verify_tensor_manifest


def paired_patients(candidate: dict, baseline: dict, *, family_size: int = 3) -> dict:
    if set(candidate) != set(baseline) or not candidate:
        raise ValueError("paired bootstrap requires the same nonempty patients")
    patients = sorted(candidate)
    differences = np.asarray([candidate[p] - baseline[p] for p in patients])
    if not np.isfinite(differences).all():
        raise ValueError("nonfinite patient errors")
    rng = np.random.default_rng(2026)
    draws = differences[rng.integers(0, len(patients), (10000, len(patients)))].mean(1)
    alpha = 0.05 / family_size
    return {"difference": float(differences.mean()), "raw_95ci": np.quantile(draws, [0.025, 0.975]).tolist(),
            "family_adjusted_ci": np.quantile(draws, [alpha / 2, 1 - alpha / 2]).tolist(),
            "patients": len(patients), "draws": 10000, "seed": 2026, "family_size": family_size}


def seed_mean(values: list[dict]) -> dict:
    if len(values) != 3 or any(set(v) != set(values[0]) for v in values):
        raise ValueError("seed-mean comparison requires all three identical patient inventories")
    return {p: float(np.mean([v[p] for v in values])) for p in values[0]}


def validate_nuisance(artifact: dict, evaluation: dict, directory: Path, contract: dict,
                      arm: str, seed: int) -> None:
    kinds = (*contract["nuisance_training"], contract["held_out_nuisance"])
    provenance = artifact.get("provenance", {})
    reference = evaluation["provenance"]
    expected_visits = {}
    for row in evaluation["states"]:
        expected_visits.setdefault(row["patient_id"], row["visit_id"])
    if set(expected_visits) != set(contract["evaluation_patients"]):
        raise ValueError("nuisance reference patient coverage mismatch")
    expected = {(patient, visit, kind) for patient, visit in expected_visits.items() for kind in kinds}
    rows = artifact.get("rows", [])
    observed = [(r.get("patient_id"), r.get("visit_id"), r.get("kind")) for r in rows]
    checkpoint_hash = digest(directory / "final.pt")
    probe_hash = digest(directory / "difference/probe.pt")
    manifest = {"prepared_sha256": contract["prepared_sha256"], "tensors": reference["tensor_hashes"]}
    if (artifact.get("status") != "complete" or artifact.get("arm") != arm or artifact.get("seed") != seed
            or len(observed) != len(expected) or set(observed) != expected
            or provenance.get("contract_sha256") != json_hash(contract)
            or provenance.get("prepared_sha256") != contract["prepared_sha256"]
            or provenance.get("tensor_manifest_sha256") != json_hash(manifest)
            or provenance.get("checkpoint_sha256") != checkpoint_hash
            or reference.get("checkpoint_sha256") != checkpoint_hash
            or provenance.get("probe_sha256") != probe_hash or reference.get("probe_sha256") != probe_hash
            or reference.get("prepared_sha256") != contract["prepared_sha256"]):
        raise ValueError("nuisance artifact identity or coverage mismatch")
    fields = ("false_change", "state_drift", "excess_false_change")
    if not all(np.isfinite(r.get(field, np.nan)) for r in rows for field in fields):
        raise ValueError("nonfinite nuisance measure")
    identities = {r["patient_id"]: r["false_change"] for r in rows if r["kind"] == "identity"}
    if any(r["false_change"] < 0 or r["state_drift"] < 0 or not np.isclose(
            r["excess_false_change"], r["false_change"] - identities[r["patient_id"]], rtol=0, atol=1e-12)
           for r in rows):
        raise ValueError("inconsistent nuisance excess or negative error")
    if set(artifact.get("conditions", {})) != set(kinds):
        raise ValueError("nuisance condition summary coverage mismatch")
    for kind in kinds:
        for field in fields:
            expected_mean = np.mean([r[field] for r in rows if r["kind"] == kind])
            if not np.isclose(artifact["conditions"][kind].get(field, np.nan), expected_mean, rtol=0, atol=1e-12):
                raise ValueError("nuisance summary differs from rows")


def aggregate_matrix(root: Path, contract: dict) -> dict:
    evaluation = {}
    missing = []
    for arm in ("A", "B", "C", "D"):
        evaluation[arm] = []
        for seed in contract["seeds"]:
            path = root / f"seed{seed}" / arm / "lumiere.json"
            if not path.exists():
                missing.append(str(path))
                continue
            artifact = json.loads(path.read_text())
            if (artifact.get("status") != "complete" or artifact["arm"] != arm or artifact["seed"] != seed
                    or artifact["cohort"] != "LUMIERE" or artifact["provenance"]["contract_sha256"] != json_hash(contract)):
                raise ValueError("evaluation artifact identity mismatch")
            if ({r["row_id"] for r in artifact["rows"]} !=
                {r["row_id"] for r in contract["pair_rows"] if r["patient_id"] in contract["evaluation_patients"]}):
                raise ValueError("evaluation pair rows differ from locked contract")
            evaluation[arm].append(artifact)
    if missing:
        return {"status": "INCOMPLETE", "missing_evaluations": missing, "comparisons": {},
                "hypotheses": {"C_vs_B_frozen_accessibility": "INCONCLUSIVE", "state_retention": "INCONCLUSIVE",
                               "D_nuisance_tradeoff": "INCONCLUSIVE", "incremental_temporal_information": "INCONCLUSIVE"}}
    local = {arm: seed_mean([e["local"]["frozen_probe"]["per_patient"] for e in artifacts])
             for arm, artifacts in evaluation.items()}
    state = {arm: seed_mean([e["state"]["per_patient_mae"] for e in artifacts])
             for arm, artifacts in evaluation.items()}
    comparisons = {f"{a}-{b}": paired_patients(local[a], local[b]) for a, b in (("C", "B"), ("C", "A"), ("D", "C"))}
    state_comparison = paired_patients(state["C"], state["A"])
    gain = comparisons["C-B"]["family_adjusted_ci"][1] < 0
    retention = state_comparison["family_adjusted_ci"][1] < contract["margins"]["state_noninferiority"]
    hypotheses = {"C_vs_B_frozen_accessibility": "VERIFIED" if gain else "INCONCLUSIVE",
                  "state_retention": "VERIFIED" if retention else "INCONCLUSIVE",
                  "D_nuisance_tradeoff": "INCONCLUSIVE", "incremental_temporal_information": "INCONCLUSIVE"}
    if comparisons["C-B"]["family_adjusted_ci"][0] >= 0:
        hypotheses["C_vs_B_frozen_accessibility"] = "NOT VERIFIED"
    if state_comparison["family_adjusted_ci"][0] >= contract["margins"]["state_noninferiority"]:
        hypotheses["state_retention"] = "NOT VERIFIED"
    control_contrasts = {}
    for arm in ("C", "D"):
        control_contrasts[arm] = {}
        for control in ("target_only", "independent_state"):
            baseline = seed_mean([e["local"][control]["per_patient"] for e in evaluation[arm]])
            control_contrasts[arm][control] = paired_patients(local[arm], baseline)
    if any(all(c["family_adjusted_ci"][1] < 0 for c in controls.values()) for controls in control_contrasts.values()):
        hypotheses["incremental_temporal_information"] = "VERIFIED"
    nuisance = {}
    for arm in ("C", "D"):
        artifacts = []
        for index, seed in enumerate(contract["seeds"]):
            directory = root / f"seed{seed}" / arm
            path = directory / "nuisance.json"
            if not path.exists():
                continue
            artifact = json.loads(path.read_text())
            validate_nuisance(artifact, evaluation[arm][index], directory, contract, arm, seed)
            artifacts.append(artifact)
        if len(artifacts) == 3:
            nuisance[arm] = {kind: seed_mean([{r["patient_id"]: r["excess_false_change"] for r in e["rows"] if r["kind"] == kind}
                                              for e in artifacts])
                             for kind in ("noise", "intensity", "bias", "resolution")}
    nuisance_contrasts = {}
    if len(nuisance) == 2:
        for kind in nuisance["C"]:
            nuisance_contrasts[kind] = paired_patients(nuisance["D"][kind], nuisance["C"][kind])
        combined = {arm: {p: float(np.mean([nuisance[arm][k][p] for k in ("noise", "intensity", "bias")]))
                          for p in nuisance[arm]["noise"]} for arm in nuisance}
        trained = paired_patients(combined["D"], combined["C"])
        state_d_c = paired_patients(state["D"], state["C"])
        change_margin = contract["margins"]["change_noninferiority"]
        state_margin = contract["margins"]["state_noninferiority"]
        if (trained["family_adjusted_ci"][0] > 0
                or comparisons["D-C"]["family_adjusted_ci"][0] >= change_margin
                or state_d_c["family_adjusted_ci"][0] >= state_margin):
            hypotheses["D_nuisance_tradeoff"] = "NOT VERIFIED"
        elif (trained["family_adjusted_ci"][1] < 0
                and comparisons["D-C"]["family_adjusted_ci"][1] < change_margin
                and state_d_c["family_adjusted_ci"][1] < state_margin):
            hypotheses["D_nuisance_tradeoff"] = "VERIFIED"
        nuisance_contrasts["trained_mean"] = trained
        nuisance_contrasts["state_D-C"] = state_d_c
    return {"status": "CORE_EVALUATION_COMPLETE", "hypotheses": hypotheses, "comparisons": comparisons,
            "state_C-A": state_comparison, "control_comparisons": control_contrasts,
            "nuisance_comparisons": nuisance_contrasts,
            "arm_patient_uniform_mae": {arm: {"frozen_local": float(np.mean(list(local[arm].values()))),
                                              "state": float(np.mean(list(state[arm].values())))} for arm in local},
            "interpretation": "Three-seed mean patient errors. Scientific claims still require companion artifacts and independent review."}


def companion_tables(rows: list[dict], states: list[dict], thresholds: list) -> dict:
    def summarize(values):
        if not values:
            return {"n_rows": 0, "n_patients": 0, "patient_uniform_mae": None, "pooled_mae": None}
        patients = sorted({p for p, _ in values})
        return {"n_rows": len(values), "n_patients": len(patients),
                "patient_uniform_mae": float(np.mean([np.mean([v for q, v in values if p == q]) for p in patients])),
                "pooled_mae": float(np.mean([v for _, v in values]))}
    empty_state, pair_strata = {}, {}
    for compartment, name in enumerate(COMPARTMENTS):
        empty_state[name] = {}
        for empty in (True, False):
            selected = [(r["patient_id"], abs(r["prediction"][compartment] - r["truth"][compartment]))
                        for r in states if (r["truth"][compartment] == 0) == empty]
            empty_state[name]["empty" if empty else "nonempty"] = summarize(selected)
        pair_strata[name] = {}
        selectors = {"both_empty": lambda r: r["source_empty"][compartment] and r["target_empty"][compartment],
                     "appearance": lambda r: r["source_empty"][compartment] and not r["target_empty"][compartment],
                     "disappearance": lambda r: not r["source_empty"][compartment] and r["target_empty"][compartment],
                     "both_nonempty": lambda r: not r["source_empty"][compartment] and not r["target_empty"][compartment]}
        for quartile in range(4):
            selectors[f"change_size_q{quartile + 1}"] = lambda r, q=quartile: int(np.searchsorted(
                thresholds[compartment], r["frozen_probe"]["changed_count"][compartment], side="right")) == q
        for stratum, select in selectors.items():
            selected = [r for r in rows if select(r)]
            pair_strata[name][stratum] = {
                method: summarize([(r["patient_id"], r[method]["balanced"][compartment]) for r in selected])
                for method in ("frozen_probe", "target_only", "independent_state", "zero")}
    rng = np.random.default_rng(2026)
    permuted = []
    for patient in sorted({r["patient_id"] for r in states}):
        selected = [r for r in states if r["patient_id"] == patient]
        order = rng.permutation(len(selected))
        for i, row in enumerate(selected):
            error = np.abs(np.asarray(selected[order[i]]["prediction"]) - np.asarray(row["truth"])).mean()
            permuted.append((patient, float(error)))
    groups = np.asarray([r["patient_id"] for r in rows])
    truth = np.asarray([r["volume_truth"] for r in rows])
    scalar = {"zero": metrics(truth, np.zeros_like(truth), groups),
              "independent_state_difference": metrics(truth, np.asarray([r["volume_independent"] for r in rows]), groups)}
    if all("volume_direct" in r for r in rows):
        scalar["direct_change"] = metrics(truth, np.asarray([r["volume_direct"] for r in rows]), groups)
    return {"state_empty_nonempty": empty_state, "local_strata": pair_strata, "scalar_change": scalar,
            "posthoc_within_patient_state_permutation": {**summarize(permuted), "seed": 2026,
                "label": "visit-specificity diagnostic, not a biological-change test"},
            "thresholds": thresholds, "threshold_source": "source52 target-only audit, fixed before training"}


def scalar_readouts(training_features: str, query_features: str, prepared: str, query_prepared: str,
                    contract_path: str, out: str) -> dict:
    contract = load_contract(contract_path)
    for filename in (training_features, query_features):
        path = Path(filename)
        if digest(path) != path.with_suffix(".sha256").read_text().strip():
            raise ValueError("encoded feature artifact bytes changed after creation")
    source = torch.load(training_features, map_location="cpu", weights_only=False)
    query = torch.load(query_features, map_location="cpu", weights_only=False)
    for artifact, root in ((source, prepared), (query, query_prepared)):
        if artifact.get("contract") != CONTRACT or artifact["provenance"]["contract_sha256"] != json_hash(contract):
            raise ValueError("wrong feature contract")
        verify_tensor_manifest(root, {"prepared_sha256": artifact["provenance"]["prepared_sha256"],
                                      "tensors": artifact["provenance"]["tensor_hashes"]})
    if source["provenance"]["checkpoint_sha256"] != query["provenance"]["checkpoint_sha256"]:
        raise ValueError("readout source and query encoders differ")
    training = json.loads((Path(prepared) / "prepared.json").read_text())
    querying = json.loads((Path(query_prepared) / "prepared.json").read_text())
    if digest(Path(prepared) / "prepared.json") != contract["prepared_sha256"]:
        raise ValueError("readout source inventory differs from locked split")
    train_patients = set(contract["training_patients"])
    query_patients = (set(contract["evaluation_patients"]) if querying["cohort"] == "LUMIERE" else
                      {v["patient_id"] for v in querying["visits"]})
    if len(train_patients) != 52 or train_patients & query_patients:
        raise ValueError("source52 budget or source/query disjointness failed")
    def arrays(metadata, artifact, patients):
        visits = [v for v in metadata["visits"] if v["patient_id"] in patients]
        features = artifact["features"]
        if not all(np.asarray(features[v["visit_id"]]).shape == (128,) and
                   np.isfinite(features[v["visit_id"]]).all() for v in visits):
            raise ValueError("scalar readouts require finite pooled 128-channel features")
        x = np.stack([features[v["visit_id"]] for v in visits])
        y = np.stack([np.log1p([v["values_mm3"][c] for c in COMPARTMENTS]) for v in visits])
        lookup = {v["visit_id"]: i for i, v in enumerate(visits)}
        pairs = [p for p in metadata["pairs"] if p["patient_id"] in patients]
        a = np.asarray([lookup[p["source_visit_id"]] for p in pairs])
        b = np.asarray([lookup[p["target_visit_id"]] for p in pairs])
        return visits, pairs, x, y, x[b] - x[a], y[b] - y[a]
    train = arrays(training, source, train_patients)
    test = arrays(querying, query, query_patients)
    models, scores, predictions = {}, {}, {}
    for name, xi, yi, xq, yq, groups, qgroups in (
        ("state", train[2], train[3], test[2], test[3], [v["patient_id"] for v in train[0]], [v["patient_id"] for v in test[0]]),
        ("delta", train[4], train[5], test[4], test[5], [p["patient_id"] for p in train[1]], [p["patient_id"] for p in test[1]])):
        scaler, model, alpha = fit_ridge(xi, yi, np.asarray(groups))
        prediction = model.predict(scaler.transform(xq))
        replay = ((xq - scaler.mean_) / scaler.scale_) @ model.coef_.T + model.intercept_
        if not np.allclose(prediction, replay, atol=1e-6, rtol=1e-6):
            raise ValueError("saved scalar readout does not reproduce predictions")
        scores[name] = metrics(yq, prediction, np.asarray(qgroups))
        models[name] = {"mean": scaler.mean_, "scale": scaler.scale_, "coefficient": model.coef_,
                        "intercept": model.intercept_, "alpha": alpha}
        predictions[name + "_prediction"] = prediction
        predictions[name + "_truth"] = yq
    output = Path(out)
    provenance = {"contract_sha256": json_hash(contract), "checkpoint_sha256": source["provenance"]["checkpoint_sha256"],
                  "training_features_sha256": digest(training_features), "query_features_sha256": digest(query_features),
                  "readout_training_patients": contract["training_patients"], "query_cohort": querying["cohort"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"models": models, "provenance": provenance, "contract": CONTRACT}, output.with_suffix(".readouts.pt"))
    np.savez_compressed(output.with_suffix(".predictions.npz"), **predictions,
                        visit_ids=np.asarray([v["visit_id"] for v in test[0]]),
                        row_ids=np.asarray([p["row_id"] for p in test[1]]),
                        visit_patients=np.asarray([v["patient_id"] for v in test[0]]),
                        pair_patients=np.asarray([p["patient_id"] for p in test[1]]))
    result = {"status": "complete", "scores": scores, "provenance": provenance,
              "readout": "source52 grouped three-fold alpha selection, never query fit"}
    save_json(output, result)
    return result
