"""Locked R36 objectives, source splits, and content-bound artifact checks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.data.representation import COMPARTMENTS, CONTRACT as PREPARED_CONTRACT, digest, json_hash, save_json
from src.harness.analysis.representation import state_loss

CONTRACT = "r36-signed-feature-difference-v1"
ARMS = {"A": {"change": None, "nuisance": False},
        "B": {"change": "grid", "nuisance": False},
        "C": {"change": "balanced", "nuisance": False},
        "D": {"change": "balanced", "nuisance": True}}
CHECKPOINT_HASHES = {
    42: "65a4c4db2ca1b44ebf29fa12659fbd09dea6ede606ce1d3279535144933a001a",
    43: "395e75b47b8d4d65f5993b36af41ca8ce8c4823f303baca34f9706315ba4fd61",
    44: "67747e19181fbd12758ab307ceda573143b355aa6e3420aea63f7b59ad993b6b"}
NUISANCES = ("identity", "noise", "intensity", "bias")


def partition_errors(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    error = (prediction.float() - target.float()).abs().flatten(2)
    changed = (target != 0).flatten(2)
    changed_count = changed.sum(-1)
    unchanged_count = (~changed).sum(-1)
    changed_mean = (error * changed).sum(-1) / changed_count.clamp_min(1)
    unchanged_mean = (error * ~changed).sum(-1) / unchanged_count.clamp_min(1)
    support = (changed_count > 0).float() + (unchanged_count > 0).float()
    balanced = (changed_mean + unchanged_mean) / support
    return {"balanced": balanced, "changed": changed_mean, "unchanged": unchanged_mean,
            "changed_count": changed_count, "unchanged_count": unchanged_count}


def balanced_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return partition_errors(prediction, target)["balanced"].mean()


def objective(output: dict, pair: dict, normalization: dict, arm: str) -> tuple[torch.Tensor, dict]:
    spec = ARMS[arm]
    device = output["change"].device
    mean = torch.tensor(normalization["mean"], device=device, dtype=torch.float32)
    scale = torch.tensor(normalization["scale"], device=device, dtype=torch.float32)
    change_scale = torch.tensor(normalization["change_scale"], device=device, dtype=torch.float32)
    terms = {"state": (state_loss(output["source"], pair["source"], mean, scale) +
                       state_loss(output["target"], pair["target"], mean, scale)) / 2}
    loss = terms["state"]
    if spec["change"]:
        terms["volume_delta"] = (output["change"].float() - pair["change"].float() / change_scale).abs().mean()
        terms["mask_delta"] = (balanced_loss(output["mask_change"], pair["mask_change"])
                               if spec["change"] == "balanced" else
                               (output["mask_change"].float() - pair["mask_change"].float()).abs().mean())
        loss = loss + terms["volume_delta"] + 0.5 * terms["mask_delta"]
    return loss, terms


def nuisance_loss(output: dict, normalization: dict) -> torch.Tensor:
    scale = torch.tensor(normalization["scale"], device=output["change"].device, dtype=torch.float32)
    state_drift = ((output["target"]["volume"].float() - output["source"]["volume"].float()) * scale).abs().mean()
    return (output["mask_change"].float().abs().mean() + output["change"].float().abs().mean() + state_drift) / 3


def validate_metadata(metadata: dict, run: dict) -> None:
    if (metadata.get("contract") != PREPARED_CONTRACT or metadata.get("cohort") != "LUMIERE"
            or metadata.get("limit") is not None or metadata.get("geometry_contract") != "affine-aligned-lesion-v1"):
        raise ValueError("full aligned LUMIERE inventory required")
    train, valid, query = [set(run[k]) for k in ("training_patients", "validation_patients", "evaluation_patients")]
    if (len(train), len(valid), len(query)) != (52, 13, 26) or train & valid or train & query or valid & query:
        raise ValueError("R35 fixed patient split required")
    if train | valid != set(metadata["protocol"]["encoder_train"]) or query != set(metadata["protocol"]["encoder_unseen"]):
        raise ValueError("prepared and R35 patient split disagree")
    lookup = {v["visit_id"]: v for v in metadata["visits"]}
    rows = {p["row_id"] for p in metadata["pairs"]}
    if len(lookup) != 599 or len(metadata["visits"]) != 599 or len(rows) != 496 or len(metadata["pairs"]) != 496:
        raise ValueError("full fixed visit and usable pair coverage required")
    if not {v["patient_id"] for v in lookup.values()} <= train | valid | query:
        raise ValueError("visit outside fixed patients")
    for pair in metadata["pairs"]:
        a, b = lookup[pair["source_visit_id"]], lookup[pair["target_visit_id"]]
        if (a["patient_id"] != b["patient_id"] or a["patient_id"] != pair["patient_id"]
                or not pair["usable"] or not pair["ordering_verified"] or a["pregrid"] != b["pregrid"]):
            raise ValueError("pair identity, order, or shared geometry failed")
    query_visits = [v for v in lookup.values() if v["patient_id"] in query]
    query_pairs = [p for p in metadata["pairs"] if p["patient_id"] in query]
    if len(query_visits) != 149 or len(query_pairs) != 117 or len({p["patient_id"] for p in query_pairs}) != 23:
        raise ValueError("fixed unseen assessment coverage failed")


def make_contract(metadata: dict, run: dict, *, run_hash: str, prepared_hash: str,
                  reference_metadata: dict, reference_inventory_hash: str) -> dict:
    validate_metadata(metadata, run)
    equivalent = {**metadata, "inventory_sha256": reference_metadata["inventory_sha256"]}
    if equivalent != reference_metadata or reference_inventory_hash != reference_metadata["inventory_sha256"]:
        raise ValueError("prepared metadata differs from the original R35 usable rows, targets, or visits")
    training = set(run["training_patients"])
    visits = [v for v in metadata["visits"] if v["patient_id"] in training]
    values = np.asarray([[np.log1p(v["values_mm3"][c]) for c in COMPARTMENTS] for v in visits])
    lookup = {v["visit_id"]: np.asarray([np.log1p(v["values_mm3"][c]) for c in COMPARTMENTS]) for v in visits}
    deltas = np.stack([lookup[p["target_visit_id"]] - lookup[p["source_visit_id"]]
                       for p in metadata["pairs"] if p["patient_id"] in training])
    normalization = {"mean": values.mean(0).tolist(), "scale": values.std(0).clip(0.5).tolist(),
                     "change_scale": deltas.std(0).clip(0.5).tolist()}
    for k in normalization:
        if not np.allclose(normalization[k], run["normalization"][k], atol=1e-9, rtol=1e-9):
            raise ValueError("source-only normalizers disagree with R35")
    return {"contract": CONTRACT, "schema_version": 1, "arms": ARMS, "seeds": [42, 43, 44],
            "initial_checkpoint_sha256": {str(k): v for k, v in CHECKPOINT_HASHES.items()},
            "training_patients": run["training_patients"], "validation_patients": run["validation_patients"],
            "evaluation_patients": run["evaluation_patients"], "normalization": run["normalization"],
            "r35_run_sha256": run_hash, "prepared_sha256": prepared_hash,
            "r35_prepared_sha256": run["prepared_sha256"],
            "inventory_byte_difference": "R35 compact JSON versus local indented JSON; canonical inventory bytes match R35 hash",
            "metadata_equivalence": "identical except inventory serialization SHA256; no visit, row, target, split, or geometry change",
            "visit_ids": [v["visit_id"] for v in metadata["visits"]],
            "pair_rows": [{k: p[k] for k in ("row_id", "patient_id", "source_visit_id", "target_visit_id")}
                          for p in metadata["pairs"]],
            "widths": [16, 32, 64, 128], "epochs": 20, "selection": "final_epoch_20_only",
            "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-4, "clip": 1.0, "batch_size": 1},
            "coefficients": {"state": 1.0, "volume_delta": 1.0, "mask_delta": 0.5, "nuisance": 0.25},
            "pair_order": "seed-specific randperm per epoch, shared by all arms",
            "nuisance_schedule": "alternate source and target by global pair exposure; cycle identity, noise, intensity, bias",
            "identity_training": "only D receives synthetic zero-change supervision; all arms compute matched forwards",
            "nuisance_loss": "mean of zero mask change, normalized zero volume delta, and native log-volume state consistency",
            "nuisance_training": list(NUISANCES), "held_out_nuisance": "resolution",
            "nuisance_progression_endpoint": "patient-mean excess frozen-probe false change, averaged across noise, intensity, bias",
            "nuisance_query_visits": "first inventory visit per development patient, all five conditions",
            "probe": {"epochs": 20, "seed": 2026, "lr": 1e-3, "weight_decay": 1e-4,
                      "architecture": "mean of resized voxelwise Conv3d(width,3,1), no tanh",
                      "training_patients": 52, "selection": "final_epoch_20_only",
                      "views": ["difference", "target_only"]},
            "primary_endpoint": "patient-uniform balanced signed occupancy MAE of frozen difference probe",
            "contrasts": ["C-B", "C-A", "D-C"], "primary_contrast": "C-B",
            "bootstrap": {"draws": 10000, "seed": 2026, "unit": "paired patient", "family_size": 3,
                          "seed_aggregation": "mean patient errors, not ensemble predictions"},
            "margins": {"state_noninferiority": 0.05, "change_noninferiority": 0.05},
            "ridge_alphas": [0.1, 1, 10, 100, 1000, 10000],
            "strata": "source52 quartiles of changed voxel count per compartment; empty target, both empty, appearance, disappearance",
            "qa_exclusions": [], "qa_primary": "full legacy-compatible rows",
            "max_gpu_hours": 24, "max_session_hours": 9,
            "transfer": "all 269 SAILOR visits and 242 pairs locally; no query fitting or selection",
            "information_cutoff": "observed source and target MRI, not future prediction",
            "evaluation_status": "repeatedly examined development cohorts",
            "provenance_limit": "R35 lacks creation-time MRI byte fingerprint; R36 records prepared tensor bytes before training"}


def lock_contract(path: str | Path, contract: dict) -> str:
    path = Path(path)
    expected = json_hash(contract)
    if path.exists():
        if json_hash(json.loads(path.read_text())) != expected:
            raise ValueError("immutable contract already exists with different content")
    else:
        save_json(path, contract)
        path.with_suffix(".sha256").write_text(expected + "\n")
    if path.with_suffix(".sha256").read_text().strip() != expected:
        raise ValueError("contract lock receipt mismatch")
    return expected


def load_contract(path: str | Path) -> dict:
    path = Path(path)
    contract = json.loads(path.read_text())
    if contract.get("contract") != CONTRACT or json_hash(contract) != path.with_suffix(".sha256").read_text().strip():
        raise ValueError("invalid or modified R36 contract")
    return contract


def verify_tensor_manifest(root: str | Path, manifest: dict) -> None:
    root = Path(root)
    metadata = json.loads((root / "prepared.json").read_text())
    if (digest(root / "prepared.json") != manifest["prepared_sha256"]
            or set(manifest["tensors"]) != {v["tensor_file"] for v in metadata["visits"]}):
        raise ValueError("tensor manifest metadata or coverage mismatch")
    for filename, expected in manifest["tensors"].items():
        if Path(filename).name != filename or digest(root / filename) != expected:
            raise ValueError("tensor bytes changed after creation manifest")
