"""Fixed MRI state, observed-change, and nuisance experiments."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.data.representation import (COMPARTMENTS, CONTRACT, ObservedPairs, Visits,
                                     digest, export_inventory, prepare, save_json)
from src.model.representation import ObservedChange, SpatialState, nuisance

ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
SEEDS = (42, 43, 44)
ARMS = ("paired", "later_only", "frozen")


def source_record() -> dict:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha, dirty = None, None
    paths = ("src/data/representation.py", "src/model/representation.py",
             "src/harness/analysis/representation.py", "src/preprocessing/geometry.py")
    return {"git_sha": sha, "tracked_dirty": dirty,
            "source_hashes": {p: digest(p) for p in paths},
            "date": datetime.now(timezone.utc).isoformat(), "contract": CONTRACT}


def patient_errors(truth: np.ndarray, prediction: np.ndarray, groups: np.ndarray) -> dict:
    errors = np.abs(truth - prediction).mean(axis=1)
    return {str(p): float(errors[groups == p].mean()) for p in sorted(set(groups))}


def metrics(truth: np.ndarray, prediction: np.ndarray, groups: np.ndarray) -> dict:
    if len(truth) == 0 or truth.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError("empty, mismatched, or nonfinite predictions")
    per = patient_errors(truth, prediction, groups)
    return {"patient_uniform_mae": float(np.mean(list(per.values()))),
            "pooled_mae": float(np.abs(truth - prediction).mean()),
            "per_compartment_mae": np.abs(truth - prediction).mean(0).tolist(),
            "n_rows": len(truth), "n_patients": len(per), "per_patient_mae": per}


def paired_interval(truth: np.ndarray, candidate: np.ndarray, baseline: np.ndarray,
                    groups: np.ndarray, draws: int = 10000, family_size: int = 1) -> dict:
    a, b = patient_errors(truth, candidate, groups), patient_errors(truth, baseline, groups)
    differences = np.asarray([a[p] - b[p] for p in a])
    rng = np.random.default_rng(2026)
    means = differences[rng.integers(0, len(differences), (draws, len(differences)))].mean(1)
    adjusted = 0.05 / max(1, family_size)
    return {"difference": float(differences.mean()),
            "paired_95ci": np.quantile(means, [0.025, 0.975]).tolist(),
            "family_adjusted_ci": np.quantile(means, [adjusted / 2, 1 - adjusted / 2]).tolist(),
            "family_size": family_size, "draws": draws, "seed": 2026}


def fit_ridge(x: np.ndarray, y: np.ndarray, groups: np.ndarray):
    unique = len(set(groups))
    if unique < 3:
        raise ValueError("ridge selection requires at least three training patients")
    scores = {alpha: [] for alpha in ALPHAS}
    for train, valid in GroupKFold(min(3, unique)).split(x, y, groups):
        scaler = StandardScaler().fit(x[train])
        for alpha in ALPHAS:
            model = Ridge(alpha=alpha, solver="lsqr").fit(scaler.transform(x[train]), y[train])
            prediction = model.predict(scaler.transform(x[valid]))
            scores[alpha].append(metrics(y[valid], prediction, groups[valid])["patient_uniform_mae"])
    alpha = min(ALPHAS, key=lambda a: np.mean(scores[a]))
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=alpha, solver="lsqr").fit(scaler.transform(x), y)
    return scaler, model, alpha


def nested_predictions(x: np.ndarray, y: np.ndarray, groups: np.ndarray, protocol: dict) -> tuple:
    predictions, means = np.zeros_like(y), np.zeros_like(y)
    filled = np.zeros(len(y), dtype=bool)
    selected = []
    for fold in range(protocol["k"]):
        test = np.asarray([protocol["folds"][p] == fold for p in groups])
        if not test.any():
            continue
        train = ~test
        scaler, model, alpha = fit_ridge(x[train], y[train], groups[train])
        predictions[test] = model.predict(scaler.transform(x[test]))
        means[test] = y[train].mean(0)
        filled[test] = True
        selected.append(alpha)
    if not filled.all():
        raise ValueError("patient folds did not cover all rows")
    return predictions, means, selected


def probe_features(metadata: dict, features: dict[str, np.ndarray], out: str, *, provenance: dict,
                   pair_features: dict[str, np.ndarray] | None = None) -> dict:
    protocol = metadata["protocol"]
    unseen = set(protocol["encoder_unseen"])
    visits = [v for v in metadata["visits"] if v["patient_id"] in unseen]
    if any(v["visit_id"] not in features for v in visits):
        raise ValueError("feature inventory does not cover every unseen measured visit")
    x = np.stack([features[v["visit_id"]] for v in visits]).astype(np.float64)
    y = np.stack([np.log1p([v["values_mm3"][name] for name in COMPARTMENTS]) for v in visits])
    groups = np.asarray([v["patient_id"] for v in visits])
    state_prediction, state_mean, state_alphas = nested_predictions(x, y, groups, protocol)
    lookup = {v["visit_id"]: i for i, v in enumerate(visits)}
    pairs = [p for p in metadata["pairs"] if p["patient_id"] in unseen]
    source = np.asarray([lookup[p["source_visit_id"]] for p in pairs])
    target = np.asarray([lookup[p["target_visit_id"]] for p in pairs])
    change = y[target] - y[source]
    pair_groups = np.asarray([p["patient_id"] for p in pairs])
    predictions = {"zero": np.zeros_like(change),
                   "independent_state_difference": state_prediction[target] - state_prediction[source]}
    selected = {}
    views = {"later_only": x[target], "difference": x[target] - x[source],
             "concatenation": np.concatenate([x[source], x[target]], axis=1)}
    if pair_features is not None:
        if any(p["row_id"] not in pair_features for p in pairs):
            raise ValueError("pair features do not cover immutable evaluation rows")
        views["learned_pair"] = np.stack([pair_features[p["row_id"]] for p in pairs])
    for name, values in views.items():
        prediction, mean, alphas = nested_predictions(values, change, pair_groups, protocol)
        predictions[name] = prediction
        predictions.setdefault("train_mean_change", mean)
        selected[name] = alphas
    comparisons = {}
    for name in views:
        comparisons[name] = {baseline: paired_interval(change, predictions[name], predictions[baseline], pair_groups,
                                                       family_size=3 * len(views))
                             for baseline in ("zero", "later_only", "independent_state_difference") if name != baseline}
    output = {"provenance": provenance, "contract": CONTRACT,
              "evaluation_status": "repeatedly examined development cohort",
              "state": {"ridge": metrics(y, state_prediction, groups), "train_mean": metrics(y, state_mean, groups),
                        "comparison": paired_interval(y, state_prediction, state_mean, groups), "alphas": state_alphas},
              "observed_change": {"methods": {name: metrics(change, pred, pair_groups) for name, pred in predictions.items()},
                                  "comparisons": comparisons, "alphas": selected},
              "interpretation": "automated compartment proxies; not forecasting or confirmed biological change"}
    path = Path(out)
    save_json(path, output)
    np.savez_compressed(path.with_suffix(".predictions.npz"), state_truth=y, state_prediction=state_prediction,
                        state_mean=state_mean, state_patients=groups,
                        visit_ids=np.asarray([v["visit_id"] for v in visits]), change_truth=change,
                        pair_patients=pair_groups, row_ids=np.asarray([p["row_id"] for p in pairs]), **predictions)
    return output


def cache_features(metadata: dict, cache: str) -> tuple[dict, dict]:
    artifact = torch.load(cache, map_location="cpu", weights_only=False)
    if artifact.get("schema_version") != 2 or artifact.get("provenance", {}).get("clinical_schema") != "survival_free_v1":
        raise ValueError("controls require schema-2 survival-free features")
    features = {}
    records = {(v["patient_id"], v["visit"]): v["visit_id"] for v in metadata["visits"]}
    for patient, item in artifact["patients"].items():
        if "visits" not in item:
            raise ValueError("cache omits visit identifiers; cannot rejoin by inferred row order")
        if len(item["visits"]) != len(item["vision"]):
            raise ValueError("cached visits and image features differ")
        for index, visit in enumerate(item["visits"]):
            identifier = records.get((patient, visit))
            if identifier is not None:
                if not item["has_img"][index] or not torch.isfinite(item["vision"][index]).all():
                    raise ValueError("invalid MRI feature at a measured visit")
                features[identifier] = item["vision"][index].numpy()
    if set(features) != set(records.values()):
        raise ValueError("cache does not cover every measured visit")
    return features, {**source_record(), "cache_sha256": digest(cache), "cache_provenance": artifact["provenance"],
                      "feature_source": "cached global BRAINIAC vision; no clinical or mask inputs",
                      "checkpoint_identity": "creation records path, epoch and validation loss, not checkpoint digest"}


def cached_controls(inventory: str, cache: str, out: str) -> dict:
    metadata = json.loads(Path(inventory).read_text())
    features, provenance = cache_features(metadata, cache)
    return probe_features(metadata, features, out, provenance={**provenance, "inventory_sha256": digest(inventory)})


def cached_transfer(training_inventory: str, training_cache: str, query_inventory: str,
                    query_cache: str, out: str) -> dict:
    training_metadata = json.loads(Path(training_inventory).read_text())
    query_metadata = json.loads(Path(query_inventory).read_text())
    training_features, training_provenance = cache_features(training_metadata, training_cache)
    query_features, query_provenance = cache_features(query_metadata, query_cache)
    return transfer_features(training_metadata, training_features, query_metadata, query_features, out,
                             provenance={"training": training_provenance, "query": query_provenance})


@torch.no_grad()
def probe_nuisance(metadata: dict, features: dict, visits: Visits, encode_visit, *, encode_pair=None,
                   pair_features: dict | None = None, device: torch.device) -> dict:
    protocol = metadata["protocol"]
    unseen = set(protocol["encoder_unseen"])
    records = [v for v in metadata["visits"] if v["patient_id"] in unseen]
    groups = np.asarray([v["patient_id"] for v in records])
    truth = np.stack([np.log1p([v["values_mm3"][c] for c in COMPARTMENTS]) for v in records])
    x = np.stack([features[v["visit_id"]] for v in records])
    lookup = {v["visit_id"]: i for i, v in enumerate(records)}
    pairs = [p for p in metadata["pairs"] if p["patient_id"] in unseen]
    pair_groups = np.asarray([p["patient_id"] for p in pairs])
    source = np.asarray([lookup[p["source_visit_id"]] for p in pairs])
    target = np.asarray([lookup[p["target_visit_id"]] for p in pairs])
    delta = truth[target] - truth[source]
    pair_x = (np.stack([pair_features[p["row_id"]] for p in pairs]) if pair_features is not None
              else x[target] - x[source])
    fitted = {}
    for fold in range(protocol["k"]):
        train = np.asarray([protocol["folds"][p] != fold for p in groups])
        pair_train = np.asarray([protocol["folds"][p] != fold for p in pair_groups])
        fitted[fold] = (fit_ridge(x[train], truth[train], groups[train]),
                        fit_ridge(pair_x[pair_train], delta[pair_train], pair_groups[pair_train]))
    chosen = {}
    for record in records:
        chosen.setdefault(record["patient_id"], record)
    output = {}
    for kind in ("identity", "intensity", "noise", "resolution", "bias"):
        per_patient = {}
        for patient, record in sorted(chosen.items()):
            raw = visits.load(record["visit_id"])
            image = raw["image"][None].to(device)
            availability = raw["availability"][None].to(device)
            changed_image = nuisance(image, kind)
            original = encode_visit(image, availability).cpu().numpy()
            changed = encode_visit(changed_image, availability).cpu().numpy()
            pair = (encode_pair(image, availability, changed_image, availability).cpu().numpy()
                    if encode_pair is not None else changed - original)
            state_fit, pair_fit = fitted[protocol["folds"][patient]]
            state_scaler, state_model, _ = state_fit
            pair_scaler, pair_model, _ = pair_fit
            state_delta = state_model.predict(state_scaler.transform(changed)) - state_model.predict(state_scaler.transform(original))
            predicted_change = pair_model.predict(pair_scaler.transform(pair))
            per_patient[patient] = {"visit_id": record["visit_id"],
                                    "state_readout_drift": float(np.abs(state_delta).mean()),
                                    "false_change_mae": float(np.abs(predicted_change).mean()),
                                    "embedding_relative_drift": float(np.linalg.norm(changed - original) /
                                                                       max(np.linalg.norm(original), 1e-8))}
        output[kind] = {"n_patients": len(per_patient), "per_patient": per_patient,
                        "state_readout_drift": float(np.mean([p["state_readout_drift"] for p in per_patient.values()])),
                        "false_change_mae": float(np.mean([p["false_change_mae"] for p in per_patient.values()]))}
    output["readout_contract"] = "same nested patient-held-out ridge as real state and observed-change probes"
    return output


@torch.no_grad()
def frozen_reference(prepared: str, checkpoint_path: str, out: str) -> dict:
    from src.model.brainiac import BrainiacEncoder
    expected_hash = "7186a6267d7aa42720381e6c2780f9c86fe20e71067074f8eacdf63d48199d0a"
    if digest(checkpoint_path) != expected_hash:
        raise ValueError("wrong clean reference checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]["model"]["brainiac"]
    model = BrainiacEncoder(ckpt_path=None, lora_rank=cfg["lora_rank"],
                            lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"])
    model.load_state_dict({k.removeprefix("backbone."): v for k, v in checkpoint["model"].items()
                           if k.startswith("backbone.")}, strict=True)
    del checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    visits = Visits(prepared)
    def encode(image, availability):
        present = availability[0].bool()
        return model(image[0, present, None]).mean(0, keepdim=True)
    features = {}
    for raw in DataLoader(visits, batch_size=1, num_workers=2):
        batch = device_batch(raw, device)
        features[raw["visit_id"][0]] = encode(batch["image"], batch["availability"])[0].cpu().numpy()
    output = Path(out); output.mkdir(parents=True, exist_ok=True)
    provenance = {**source_record(), "checkpoint_sha256": expected_hash,
                  "prepared_sha256": digest(Path(prepared) / "prepared.json")}
    torch.save({"features": features, "provenance": provenance, "contract": CONTRACT}, output / "features.pt")
    if visits.metadata.get("cohort") == "SAILOR":
        return {"status": "encoded_for_developmental_transfer", "n_visits": len(features), "provenance": provenance}
    result = probe_features(visits.metadata, features, str(output / "probes.json"), provenance=provenance)
    save_json(output / "nuisance.json", probe_nuisance(visits.metadata, features, visits, encode, device=device))
    return result


def transfer_features(training_metadata: dict, training_features: dict, query_metadata: dict, query_features: dict,
                      out: str, *, training_pair_features: dict | None = None,
                      query_pair_features: dict | None = None, provenance: dict,
                      training_patients: list[str] | None = None) -> dict:
    unseen = set(training_patients or training_metadata["protocol"]["encoder_unseen"])
    train_visits = [v for v in training_metadata["visits"] if v["patient_id"] in unseen]
    query_visits = query_metadata["visits"]
    if unseen & {v["patient_id"] for v in query_visits}:
        raise ValueError("training and query patients overlap")
    def arrays(records, features):
        return (np.stack([features[v["visit_id"]] for v in records]),
                np.stack([np.log1p([v["values_mm3"][c] for c in COMPARTMENTS]) for v in records]),
                np.asarray([v["patient_id"] for v in records]))
    xa, ya, ga = arrays(train_visits, training_features)
    xb, yb, gb = arrays(query_visits, query_features)
    scaler, model, alpha = fit_ridge(xa, ya, ga)
    state_prediction = model.predict(scaler.transform(xb))
    state_mean = np.broadcast_to(ya.mean(0), yb.shape)
    def pairs(records, pairs, x, y, allowed=None):
        lookup = {v["visit_id"]: i for i, v in enumerate(records)}
        eligible = [p for p in pairs if allowed is None or p["patient_id"] in allowed]
        a = np.asarray([lookup[p["source_visit_id"]] for p in eligible])
        b = np.asarray([lookup[p["target_visit_id"]] for p in eligible])
        return eligible, a, b, y[b] - y[a], np.asarray([p["patient_id"] for p in eligible])
    pa, sa, ta, da, groups = pairs(train_visits, training_metadata["pairs"], xa, ya, unseen)
    pb, sb, tb, db, query_groups = pairs(query_visits, query_metadata["pairs"], xb, yb)
    predictions = {"zero": np.zeros_like(db), "train_mean_change": np.broadcast_to(da.mean(0), db.shape),
                   "independent_state_difference": state_prediction[tb] - state_prediction[sb]}
    views = {"later_only": (xa[ta], xb[tb]), "difference": (xa[ta] - xa[sa], xb[tb] - xb[sb]),
             "concatenation": (np.concatenate([xa[sa], xa[ta]], 1), np.concatenate([xb[sb], xb[tb]], 1))}
    if training_pair_features is not None:
        if query_pair_features is None:
            raise ValueError("query pair embeddings missing")
        views["learned_pair"] = (np.stack([training_pair_features[p["row_id"]] for p in pa]),
                                 np.stack([query_pair_features[p["row_id"]] for p in pb]))
    selected = {}
    for name, (a, b) in views.items():
        pair_scaler, pair_model, chosen = fit_ridge(a, da, groups)
        predictions[name] = pair_model.predict(pair_scaler.transform(b))
        selected[name] = chosen
    result = {"contract": CONTRACT, "provenance": provenance, "cohort_status": "patient-disjoint fixed source readouts; no query fitting",
              "state": {"ridge": metrics(yb, state_prediction, gb), "train_mean": metrics(yb, state_mean, gb),
                        "comparison": paired_interval(yb, state_prediction, state_mean, gb), "alpha": alpha},
              "observed_change": {"methods": {name: metrics(db, value, query_groups) for name, value in predictions.items()},
                                  "comparisons": {name: paired_interval(db, predictions[name], predictions["zero"],
                                                                        query_groups, family_size=len(views)) for name in views},
                                  "alphas": selected}}
    save_json(out, result)
    np.savez_compressed(Path(out).with_suffix(".predictions.npz"), state_truth=yb, state_prediction=state_prediction,
                        state_mean=state_mean, state_patients=gb, visit_ids=np.asarray([v["visit_id"] for v in query_visits]),
                        change_truth=db, pair_patients=query_groups, row_ids=np.asarray([p["row_id"] for p in pb]), **predictions)
    return result


def supervised_controls(inventory_path: str, cache: str, out: str) -> dict:
    metadata = json.loads(Path(inventory_path).read_text())
    patients = sorted(metadata["protocol"]["encoder_train"])
    random.Random(2026).shuffle(patients)
    training = sorted(patients[13:])
    features, provenance = cache_features(metadata, cache)
    query_patients = set(metadata["protocol"]["encoder_unseen"])
    query = {**metadata, "visits": [v for v in metadata["visits"] if v["patient_id"] in query_patients],
             "pairs": [p for p in metadata["pairs"] if p["patient_id"] in query_patients]}
    return transfer_features(metadata, features, query, features, out, training_patients=training,
                             provenance={**provenance, "training_patients": training,
                                         "supervision_budget": "52 patient scalar-volume labels; no dense-mask objective"})


def evaluate_transfer(prepared: str, checkpoint_path: str, training_metadata_path: str,
                      training_features_path: str, out: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("contract") != CONTRACT:
        raise ValueError("checkpoint has wrong representation task contract")
    if checkpoint["provenance"].get("dense_loss_contract") != "float32-reductions-v1":
        raise ValueError("checkpoint predates the verified mixed-precision loss contract")
    expected = checkpoint["provenance"]["source_hashes"]["src/model/representation.py"]
    if expected != digest("src/model/representation.py"):
        raise ValueError("model implementation differs from training snapshot")
    encoder = SpatialState()
    kind = checkpoint["kind"]
    model = encoder if kind == "state" else ObservedChange(encoder, kind)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    visits = Visits(prepared)
    if visits.metadata.get("cohort") != "SAILOR" or visits.metadata.get("limit") is not None:
        raise ValueError("transfer requires the full local SAILOR tensor inventory")
    features, pair_features = extract(model, visits, None if kind == "state" else ObservedPairs(visits), torch.device("cpu"))
    trained = torch.load(training_features_path, map_location="cpu", weights_only=False)
    training_metadata = json.loads(Path(training_metadata_path).read_text())
    checkpoint_hash = digest(checkpoint_path)
    checkpoint_directory = Path(checkpoint_path).resolve().parent
    probe = json.loads((checkpoint_directory / "probes.json").read_text())
    run = json.loads((checkpoint_directory.parent.parent / "run.json").read_text())
    if (Path(training_features_path).resolve() != checkpoint_directory / "features.pt"
            or probe["provenance"]["checkpoint_sha256"] != checkpoint_hash
            or trained["provenance"]["source_hashes"] != checkpoint["provenance"]["source_hashes"]
            or run["prepared_sha256"] != digest(training_metadata_path)):
        raise ValueError("training labels, features, and checkpoint provenance disagree")
    provenance = {**source_record(), "training_provenance": checkpoint["provenance"],
                  "checkpoint_sha256": digest(checkpoint_path), "training_features_sha256": digest(training_features_path)}
    return transfer_features(training_metadata, trained["features"], visits.metadata, features, out,
                             training_pair_features=trained.get("pair_features"), query_pair_features=pair_features,
                             provenance=provenance)


def state_loss(output: dict, batch: dict, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    logits, masks = output["mask_logits"].float(), batch["masks"].float()
    probability = logits.sigmoid()
    axes = (-3, -2, -1)
    dice = 1 - ((2 * (probability * masks).sum(axes) + 1) /
                (probability.sum(axes) + masks.sum(axes) + 1)).mean()
    volume = F.l1_loss(output["volume"], (batch["volume"] - mean) / scale)
    return F.binary_cross_entropy_with_logits(logits, masks) + dice + volume


def device_batch(batch: dict, device: torch.device) -> dict:
    return {key: device_batch(value, device) if isinstance(value, dict) else
            value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def train_one(model, train_data, valid_data, normalization: dict, out: Path, *, epochs: int,
              seed: int, device: torch.device, deadline: float, kind: str, artifact_context: dict) -> dict:
    model.to(device)
    mean = torch.tensor(normalization["mean"], device=device)
    scale = torch.tensor(normalization["scale"], device=device)
    change_scale = torch.tensor(normalization["change_scale"], device=device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, generator=generator,
                              num_workers=2, pin_memory=device.type == "cuda")
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=2)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, history, complete = float("inf"), [], True
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        if time.monotonic() >= deadline:
            complete = False
            break
        model.train()
        losses = []
        for raw_batch in train_loader:
            if time.monotonic() >= deadline:
                complete = False
                break
            batch = device_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                if kind == "state":
                    prediction = model(batch["image"], batch["availability"])
                    loss = state_loss(prediction, batch, mean, scale)
                else:
                    a, b = batch["source"], batch["target"]
                    prediction = model(a["image"], a["availability"], b["image"], b["availability"])
                    loss = F.l1_loss(prediction["change"], batch["change"] / change_scale)
                    loss = loss + 0.5 * state_loss(prediction, b, mean, scale)
                    weights = 1 + 4 * torch.maximum(a["masks"], b["masks"])
                    loss = loss + 0.5 * (weights * (prediction["mask_change"] - batch["mask_change"]).abs()).mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        if not complete:
            break
        model.eval()
        truths, predictions, patients = [], [], []
        with torch.no_grad():
            for raw_batch in valid_loader:
                batch = device_batch(raw_batch, device)
                if kind == "state":
                    output = model(batch["image"], batch["availability"])
                    truth, prediction = batch["volume"], output["volume"] * scale + mean
                else:
                    a, b = batch["source"], batch["target"]
                    output = model(a["image"], a["availability"], b["image"], b["availability"], dense=False)
                    truth, prediction = batch["change"], output["change"] * change_scale
                truths.append(truth.cpu().numpy()); predictions.append(prediction.cpu().numpy())
                patients += raw_batch["patient_id"]
        score = metrics(np.concatenate(truths), np.concatenate(predictions), np.asarray(patients))["patient_uniform_mae"]
        row = {"epoch": epoch + 1, "training_loss": float(np.mean(losses)), "validation_mae": score,
               "training_steps": len(losses)}
        history.append(row)
        if score < best:
            best = score
            torch.save({"state_dict": model.state_dict(), "normalization": normalization,
                        "kind": kind, "seed": seed, "epoch": epoch + 1, "validation_mae": best,
                        "contract": CONTRACT, "provenance": {**source_record(), **artifact_context}}, out / "best.pt")
        save_json(out / "training.json", {"history": history, "status": "running", "kind": kind})
        print(f"seed={seed} arm={kind} epoch={epoch + 1} train={row['training_loss']:.5f} val={score:.5f}", flush=True)
    status = "complete" if complete and len(history) == epochs else "incomplete_runtime_ceiling"
    save_json(out / "training.json", {"history": history, "status": status, "kind": kind})
    if not (out / "best.pt").exists():
        raise RuntimeError("no complete epoch before runtime ceiling")
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=False)["state_dict"])
    return {"status": status, "best_validation_mae": best, "completed_epochs": len(history)}


@torch.no_grad()
def direct_scores(model, metadata: dict, features: dict, pair_features: dict | None,
                  normalization: dict, device: torch.device, out: Path) -> dict:
    unseen = set(metadata["protocol"]["encoder_unseen"])
    visits = [v for v in metadata["visits"] if v["patient_id"] in unseen]
    groups = np.asarray([v["patient_id"] for v in visits])
    truth = np.stack([np.log1p([v["values_mm3"][c] for c in COMPARTMENTS]) for v in visits])
    encoder = model.encoder if isinstance(model, ObservedChange) else model
    embedding = torch.as_tensor(np.stack([features[v["visit_id"]] for v in visits]), device=device)
    scale = torch.tensor(normalization["scale"], device=device)
    mean = torch.tensor(normalization["mean"], device=device)
    prediction = (encoder.volume_head(embedding) * scale + mean).cpu().numpy()
    train_mean = np.broadcast_to(normalization["mean"], truth.shape)
    lookup = {v["visit_id"]: i for i, v in enumerate(visits)}
    pairs = [p for p in metadata["pairs"] if p["patient_id"] in unseen]
    source = np.asarray([lookup[p["source_visit_id"]] for p in pairs])
    target = np.asarray([lookup[p["target_visit_id"]] for p in pairs])
    delta = truth[target] - truth[source]
    pair_groups = np.asarray([p["patient_id"] for p in pairs])
    predictions = {"zero": np.zeros_like(delta), "independent_decoder_difference": prediction[target] - prediction[source]}
    if pair_features is not None:
        pair_embedding = torch.as_tensor(np.stack([pair_features[p["row_id"]] for p in pairs]), device=device)
        change_scale = torch.tensor(normalization["change_scale"], device=device)
        predictions["pair_head"] = (model.change_head(pair_embedding) * change_scale).cpu().numpy()
    result = {"readout_contract": "direct 52-patient-trained heads; no unseen readout fitting",
              "state": {"model": metrics(truth, prediction, groups), "train_mean": metrics(truth, train_mean, groups),
                        "comparison": paired_interval(truth, prediction, train_mean, groups)},
              "observed_change": {name: metrics(delta, value, pair_groups) for name, value in predictions.items()}}
    save_json(out, result)
    np.savez_compressed(out.with_suffix(".predictions.npz"), state_truth=truth, state_prediction=prediction,
                        state_mean=train_mean, state_patients=groups, visit_ids=np.asarray([v["visit_id"] for v in visits]),
                        change_truth=delta, pair_patients=pair_groups, row_ids=np.asarray([p["row_id"] for p in pairs]), **predictions)
    return result


@torch.no_grad()
def extract(model, visits: Visits, pairs: ObservedPairs | None, device: torch.device) -> tuple:
    model.eval().to(device)
    encoder = model.encoder if isinstance(model, ObservedChange) else model
    features = {}
    for raw in DataLoader(visits, batch_size=1, num_workers=2):
        batch = device_batch(raw, device)
        feature = encoder.summarize(encoder.encode(batch["image"], batch["availability"]))
        features[raw["visit_id"][0]] = feature[0].cpu().numpy()
    pair_features = None
    if pairs is not None:
        pair_features = {}
        for raw in DataLoader(pairs, batch_size=1, num_workers=2):
            batch = device_batch(raw, device)
            a, b = batch["source"], batch["target"]
            feature = model(a["image"], a["availability"], b["image"], b["availability"], dense=False)["embedding"]
            pair_features[raw["row_id"][0]] = feature[0].cpu().numpy()
    return features, pair_features


def smoke(prepared: str, out: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and "T4" not in torch.cuda.get_device_name(0):
        raise ValueError("unexpected GPU")
    visits = Visits(prepared)
    pair = next(iter(DataLoader(ObservedPairs(visits), batch_size=1)))
    batch = device_batch(pair, device)
    results = {}
    for kind in ("state", "paired"):
        torch.manual_seed(42)
        model = SpatialState() if kind == "state" else ObservedChange(SpatialState(), "paired")
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            a, b = batch["source"], batch["target"]
            output = (model(b["image"], b["availability"]) if kind == "state" else
                      model(a["image"], a["availability"], b["image"], b["availability"]))
            loss = state_loss(output, b, torch.zeros(3, device=device), torch.ones(3, device=device))
            if kind == "paired":
                loss = loss + F.l1_loss(output["change"], batch["change"])
                loss = loss + F.l1_loss(output["mask_change"], batch["mask_change"])
        if not torch.isfinite(loss):
            raise ValueError("nonfinite real MRI smoke loss")
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        scaler.step(optimizer); scaler.update()
        results[kind] = {"loss": float(loss.detach()), "gradient_norm": float(gradient)}
    result = {"device": str(device), "rows": len(pair["row_id"]), "arms": results,
              "interpretation": "real MRI optimizer check, not model quality", "provenance": source_record()}
    save_json(out, result)
    return result


def run_training(prepared: str, out: str, *, seeds: list[int], epochs: int, hours: float) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("full representation training requires GPU; use smoke for CPU checks")
    if "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("expected Tesla T4")
    device = torch.device("cuda")
    metadata = json.loads((Path(prepared) / "prepared.json").read_text())
    if metadata.get("limit") is not None:
        raise ValueError("full evaluation cannot use a truncated tensor inventory")
    patients = sorted(metadata["protocol"]["encoder_train"])
    random.Random(2026).shuffle(patients)
    validation = sorted(patients[:13]); training = sorted(patients[13:])
    train_visits, valid_visits = Visits(prepared, training), Visits(prepared, validation)
    train_pairs, valid_pairs = ObservedPairs(train_visits), ObservedPairs(valid_visits)
    all_visits = Visits(prepared)
    all_pairs = ObservedPairs(all_visits)
    unseen_visits = Visits(prepared, metadata["protocol"]["encoder_unseen"])
    values = np.stack([np.log1p([v["values_mm3"][c] for c in COMPARTMENTS]) for v in train_visits.records])
    mapping = {v["visit_id"]: np.log1p([v["values_mm3"][c] for c in COMPARTMENTS]) for v in train_visits.records}
    changes = np.stack([mapping[p["target_visit_id"]] - mapping[p["source_visit_id"]] for p in train_pairs.records])
    normalization = {"mean": values.mean(0).tolist(), "scale": values.std(0).clip(0.5).tolist(),
                     "change_scale": changes.std(0).clip(0.5).tolist()}
    result = {"provenance": source_record(), "prepared_sha256": digest(Path(prepared) / "prepared.json"),
              "training_patients": training, "validation_patients": validation,
              "evaluation_patients": metadata["protocol"]["encoder_unseen"],
              "normalization": normalization, "epochs": epochs, "seeds": seeds,
              "hours_ceiling": hours, "runs": {}}
    output = Path(out); output.mkdir(parents=True, exist_ok=True)
    save_json(output / "run.json", result)
    deadline = time.monotonic() + hours * 3600
    for seed in seeds:
        if time.monotonic() >= deadline - 900:
            result["status"] = "incomplete_runtime_ceiling"
            save_json(output / "run.json", result)
            return result
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        state = SpatialState()
        artifact_context = {"prepared_sha256": result["prepared_sha256"], "source_data_hashes": metadata["source_hashes"],
                            "training_patients": training, "validation_patients": validation,
                            "epochs": epochs, "seed": seed, "widths": [16, 32, 64, 128],
                            "dense_loss_contract": "float32-reductions-v1"}
        state_out = output / f"seed{seed}" / "state"
        status = train_one(state, train_visits, valid_visits, normalization, state_out,
                           epochs=epochs, seed=seed, device=device, deadline=deadline - 900, kind="state",
                           artifact_context={**artifact_context, "arm": "state"})
        state_weights = copy.deepcopy({k: v.cpu() for k, v in state.state_dict().items()})
        for arm in ("state", *ARMS):
            if arm == "state":
                model = state
            else:
                if time.monotonic() >= deadline - 900:
                    result["status"] = "incomplete_runtime_ceiling"
                    save_json(output / "run.json", result)
                    return result
                encoder = SpatialState(); encoder.load_state_dict(state_weights)
                torch.manual_seed(seed)
                model = ObservedChange(encoder, mode=arm)
                status = train_one(model, train_pairs, valid_pairs, normalization, output / f"seed{seed}" / arm,
                                   epochs=epochs, seed=seed, device=device, deadline=deadline - 900, kind=arm,
                                   artifact_context={**artifact_context, "arm": arm,
                                                     "initial_state_checkpoint_sha256": digest(state_out / "best.pt")})
            arm_out = output / f"seed{seed}" / arm
            features, pair_features = extract(model, all_visits, None if arm == "state" else all_pairs, device)
            checkpoint = torch.load(arm_out / "best.pt", map_location="cpu", weights_only=False)
            checkpoint_hash = digest(arm_out / "best.pt")
            torch.save({"features": features, "pair_features": pair_features, "provenance": checkpoint["provenance"],
                        "normalization": normalization, "contract": CONTRACT, "checkpoint": checkpoint_hash},
                       arm_out / "features.pt")
            del checkpoint
            direct_scores(model, metadata, features, pair_features, normalization, device, arm_out / "direct.json")
            probe = probe_features(metadata, features, str(arm_out / "probes.json"),
                                   provenance={**source_record(), "seed": seed, "arm": arm,
                                               "checkpoint_sha256": digest(arm_out / "best.pt")}, pair_features=pair_features)
            encoder = model.encoder if isinstance(model, ObservedChange) else model
            def encode_visit(image, availability):
                return encoder.summarize(encoder.encode(image, availability))
            def encode_pair(source, source_availability, target, target_availability):
                return model(source, source_availability, target, target_availability, dense=False)["embedding"]
            nuisance_result = probe_nuisance(metadata, features, unseen_visits, encode_visit,
                                            encode_pair=None if arm == "state" else encode_pair,
                                            pair_features=pair_features, device=device)
            save_json(arm_out / "nuisance.json", nuisance_result)
            result["runs"][f"{seed}/{arm}"] = {**status, "state_mae": probe["state"]["ridge"]["patient_uniform_mae"],
                                                    "probe_path": str(arm_out / "probes.json")}
            save_json(output / "run.json", result)
            print(f"evaluated seed={seed} arm={arm}", flush=True)
            if arm != "state":
                del model
                torch.cuda.empty_cache()
        del state
        torch.cuda.empty_cache()
    result["status"] = ("complete" if all(r["status"] == "complete" for r in result["runs"].values())
                        else "incomplete_runtime_ceiling")
    save_json(output / "run.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--manifest", required=True); export.add_argument("--geometry", required=True)
    export.add_argument("--protocol", default="info/eval_folds.json"); export.add_argument("--out", required=True)
    export.add_argument("--cohort", choices=("LUMIERE", "SAILOR"), default="LUMIERE")
    prep = sub.add_parser("prepare")
    prep.add_argument("--inventory", required=True); prep.add_argument("--mri-root", required=True)
    prep.add_argument("--mask-root", required=True); prep.add_argument("--out", required=True)
    prep.add_argument("--limit", type=int)
    controls = sub.add_parser("controls")
    controls.add_argument("--inventory", required=True); controls.add_argument("--cache", required=True)
    controls.add_argument("--out", required=True)
    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument("--prepared", required=True); smoke_parser.add_argument("--out", required=True)
    supervised = sub.add_parser("supervised-controls")
    supervised.add_argument("--inventory", required=True); supervised.add_argument("--cache", required=True)
    supervised.add_argument("--out", required=True)
    cached = sub.add_parser("cached-transfer")
    cached.add_argument("--training-inventory", required=True); cached.add_argument("--training-cache", required=True)
    cached.add_argument("--query-inventory", required=True); cached.add_argument("--query-cache", required=True)
    cached.add_argument("--out", required=True)
    transfer = sub.add_parser("transfer")
    transfer.add_argument("--prepared", required=True); transfer.add_argument("--checkpoint", required=True)
    transfer.add_argument("--training-metadata", required=True); transfer.add_argument("--training-features", required=True)
    transfer.add_argument("--out", required=True)
    reference = sub.add_parser("reference")
    reference.add_argument("--prepared", required=True); reference.add_argument("--checkpoint", required=True)
    reference.add_argument("--out", required=True)
    train = sub.add_parser("train")
    train.add_argument("--prepared", required=True); train.add_argument("--out", required=True)
    train.add_argument("--epochs", type=int, default=20); train.add_argument("--hours", type=float, default=8)
    train.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    if args.command == "export":
        result = export_inventory(args.manifest, args.geometry, args.protocol, args.out, args.cohort)
        print(f"exported {len(result['visits'])} visits and {len(result['pairs'])} pairs")
    elif args.command == "prepare":
        result = prepare(args.inventory, args.mri_root, args.mask_root, args.out, args.limit)
        print(f"prepared {len(result['visits'])} visits and {len(result['pairs'])} pairs")
    elif args.command == "controls":
        result = cached_controls(args.inventory, args.cache, args.out)
        print(json.dumps({"state": result["state"]["ridge"]["patient_uniform_mae"],
                          "change": {k: v["patient_uniform_mae"] for k, v in result["observed_change"]["methods"].items()}}, sort_keys=True))
    elif args.command == "smoke":
        print(json.dumps(smoke(args.prepared, args.out), sort_keys=True))
    elif args.command == "supervised-controls":
        result = supervised_controls(args.inventory, args.cache, args.out)
        print(json.dumps(result["state"]["ridge"], sort_keys=True))
    elif args.command == "cached-transfer":
        result = cached_transfer(args.training_inventory, args.training_cache, args.query_inventory, args.query_cache, args.out)
        print(json.dumps(result["state"]["ridge"], sort_keys=True))
    elif args.command == "transfer":
        result = evaluate_transfer(args.prepared, args.checkpoint, args.training_metadata, args.training_features, args.out)
        print(json.dumps(result["state"]["ridge"], sort_keys=True))
    elif args.command == "reference":
        result = frozen_reference(args.prepared, args.checkpoint, args.out)
        print(json.dumps(result.get("state", {}).get("ridge", result), sort_keys=True))
    else:
        result = run_training(args.prepared, args.out, seeds=args.seeds, epochs=args.epochs, hours=args.hours)
        print(json.dumps(result["runs"], sort_keys=True))


if __name__ == "__main__":
    main()
