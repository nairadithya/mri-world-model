"""R36 target audit, matched objectives, and source-fitted frozen probes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.data.representation import COMPARTMENTS, Visits, ObservedPairs, digest, json_hash, save_json
from src.model.representation import SpatialState, nuisance
from src.model.delta_representation import DeltaRepresentation, LocalLinearProbe
from src.harness.analysis.representation import device_batch, optimizer_step, metrics
from src.harness.analysis.delta_evaluation import aggregate_matrix, scalar_readouts, companion_tables
from src.harness.analysis.delta_targets import audit_native_targets
from src.harness.analysis.delta_protocol import (ARMS, CHECKPOINT_HASHES, CONTRACT, NUISANCES,
    balanced_loss, partition_errors, objective, nuisance_loss, make_contract, lock_contract,
    load_contract, verify_tensor_manifest)

SOURCE_FILES = ("src/model/representation.py", "src/model/delta_representation.py",
                "src/data/representation.py", "src/harness/analysis/representation.py",
                "src/harness/analysis/delta_representation.py", "src/harness/analysis/delta_protocol.py",
                "src/harness/analysis/delta_evaluation.py", "src/harness/analysis/delta_targets.py",
                "src/preprocessing/geometry.py")


def source_record() -> dict:
    return {"git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "date": datetime.now(timezone.utc).isoformat(),
            "source_hashes": {p: digest(p) for p in SOURCE_FILES}, "contract": CONTRACT}


def initialize(checkpoint: str, seed: int) -> tuple[DeltaRepresentation, dict]:
    if digest(checkpoint) != CHECKPOINT_HASHES[seed]:
        raise ValueError("R35 state checkpoint bytes differ from approved initialization")
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (artifact["kind"] != "state" or artifact["seed"] != seed
            or artifact["provenance"].get("dense_loss_contract") != "float32-reductions-v1"):
        raise ValueError("wrong R35 state graph or dense loss contract")
    if artifact["provenance"]["source_hashes"]["src/model/representation.py"] != digest("src/model/representation.py"):
        raise ValueError("R35 encoder source differs from checkpoint graph")
    torch.manual_seed(seed)
    model = DeltaRepresentation()
    model.encoder.load_state_dict(artifact["state_dict"], strict=True)
    return model, artifact["normalization"]


def bind(prepared: str, contract_path: str, audit_path: str) -> tuple[dict, dict]:
    contract = load_contract(contract_path)
    audit = json.loads(Path(audit_path).read_text())
    if (audit.get("contract_sha256") != json_hash(contract) or audit.get("status") != "targets_audited"
            or audit.get("visual_review_status") != "reviewed"):
        raise ValueError("training requires the matching completed target audit")
    verify_tensor_manifest(prepared, audit["tensor_manifest"])
    if digest(Path(prepared) / "prepared.json") != contract["prepared_sha256"]:
        raise ValueError("prepared inventory differs from contract")
    if audit.get("target_rows_sha256") != digest(Path(audit_path).parent / "target_rows.json"):
        raise ValueError("target rows or source strata changed after audit")
    for overlay in audit["overlays"].values():
        if digest(overlay["path"]) != overlay["sha256"]:
            raise ValueError("QA overlay bytes changed after review")
    return contract, audit


def audit_targets(prepared: str, r35_run: str, out: str, ratings: str | None,
                  inventory_path: str, mask_root: str) -> dict:
    visits = Visits(prepared)
    run = json.loads(Path(r35_run).read_text())
    reference_path = Path(r35_run).parents[2] / "training_prepared.json"
    if digest(reference_path) != run["prepared_sha256"]:
        raise ValueError("R35 prepared reference bytes do not match run receipt")
    reference = json.loads(reference_path.read_text())
    from hashlib import sha256
    canonical_inventory = json.dumps(json.loads(Path(inventory_path).read_text()), sort_keys=True, separators=(",", ":")).encode()
    contract = make_contract(visits.metadata, run, run_hash=digest(r35_run),
                             prepared_hash=digest(Path(prepared) / "prepared.json"), reference_metadata=reference,
                             reference_inventory_hash=sha256(canonical_inventory).hexdigest())
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    contract_hash = lock_contract(output / "r36_contract.json", contract)
    (output / "prepared.json").write_bytes((Path(prepared) / "prepared.json").read_bytes())
    tensors, occupancy, masks = {}, {}, {}
    native = {}
    for index, record in enumerate(visits.records):
        raw = visits.load(record["visit_id"])
        expected = np.log1p([record["values_mm3"][c] for c in COMPARTMENTS])
        if not np.allclose(raw["volume"].numpy(), expected, atol=1e-5, rtol=1e-6):
            raise ValueError("prepared native volumes differ from frozen manifest")
        identifier = record["visit_id"]
        tensors[record["tensor_file"]] = digest(visits.root / record["tensor_file"])
        occupancy[identifier] = raw["masks"].flatten(1).sum(1).int().tolist()
        masks[identifier] = np.packbits(raw["masks"].numpy().astype(np.uint8).reshape(3, -1), axis=1)
        native[identifier] = expected
        if (index + 1) % 50 == 0:
            print(f"audit visits {index + 1}/{len(visits)}", flush=True)
    total_voxels = 96 ** 3
    rows = []
    train_counts = []
    training = set(contract["training_patients"])
    for pair in visits.metadata["pairs"]:
        source, target = pair["source_visit_id"], pair["target_visit_id"]
        changed = np.unpackbits(masks[source] ^ masks[target], axis=1, count=total_voxels).sum(1).astype(int)
        source_count, target_count = np.asarray(occupancy[source]), np.asarray(occupancy[target])
        row = {**pair, "changed_voxels": changed.tolist(), "unchanged_voxels": (total_voxels - changed).tolist(),
               "source_occupancy": source_count.tolist(), "target_occupancy": target_count.tolist(),
               "both_empty": ((source_count == 0) & (target_count == 0)).tolist(),
               "appearance": ((source_count == 0) & (target_count > 0)).tolist(),
               "disappearance": ((source_count > 0) & (target_count == 0)).tolist(),
               "native_delta": (native[target] - native[source]).tolist(),
               "native_volume_change_without_grid_change": ((native[target] != native[source]) & (changed == 0)).tolist()}
        rows.append(row)
        if pair["patient_id"] in training:
            train_counts.append(changed)
    thresholds = np.quantile(np.stack(train_counts), [0.25, 0.5, 0.75], axis=0).T.tolist()
    save_json(output / "target_rows.json", {"rows": rows, "occupancy": occupancy, "source52_strata_thresholds": thresholds})
    tensor_manifest = {"prepared_sha256": contract["prepared_sha256"], "tensors": tensors}
    save_json(output / "tensor_manifest.json", tensor_manifest)
    native_audit = audit_native_targets(inventory_path, mask_root, visits.metadata, rows)
    save_json(output / "native_targets.json", native_audit)
    overlays = render_overlays(visits, rows, output / "overlays", ratings)
    changed = np.asarray([r["changed_voxels"] for r in rows])
    result = {"status": "targets_audited", "contract_sha256": contract_hash,
              "tensor_manifest": tensor_manifest, "tensor_manifest_sha256": digest(output / "tensor_manifest.json"),
              "target_rows_sha256": digest(output / "target_rows.json"), "source": source_record(),
              "coverage": {"visits": len(visits), "pairs": len(rows), "training_patients": 52,
                           "validation_patients": 13, "evaluation_patients": 26,
                           "evaluation_visits": 149, "evaluation_pairs": 117, "evaluation_pair_patients": 23},
              "changed_voxel_quantiles": np.quantile(changed, [0, 0.25, 0.5, 0.75, 1], axis=0).tolist(),
              "all_zero_compartment_pairs": (changed == 0).sum(0).tolist(),
              "both_empty_compartment_pairs": np.asarray([r["both_empty"] for r in rows]).sum(0).tolist(),
              "native_volume_change_without_grid_change": np.asarray([r["native_volume_change_without_grid_change"] for r in rows]).sum(0).tolist(),
              "source52_strata_thresholds": thresholds, "overlays": overlays,
              "native_spatial_changes_lost_at_96": native_audit["lost_compartment_changes"],
              "native_targets_sha256": digest(output / "native_targets.json"),
              "qa_exclusions": [], "visual_review_status": "pending_parent_pixel_review",
              "limits": ["Matching grids do not prove anatomical registration.",
                         "Native source-grid disagreement includes alignment and mask effects, not confirmed biology.",
                         "Automated target repeatability and biological attribution remain unknown."]}
    save_json(output / "audit.json", result)
    return result


def render_overlays(visits: Visits, rows: list[dict], output: Path, ratings: str | None) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import csv
    postoperative = set()
    if ratings:
        with Path(ratings).open() as stream:
            for row in csv.DictReader(stream):
                if any(value == "Post-Op" for value in row.values()):
                    postoperative.add((row["Patient"], row["Date"]))
    ordered = sorted(rows, key=lambda r: sum(r["changed_voxels"]))
    chosen = {"least_changed": ordered[0], "most_changed": ordered[-1]}
    chosen["empty_compartment"] = next(r for r in ordered if any(r["both_empty"]))
    exact = next((r for r in ordered if sum(r["changed_voxels"]) == 0), None)
    if exact:
        chosen["unchanged"] = exact
    post = next((r for r in ordered if (r["patient_id"], r["target_visit"]) in postoperative), None)
    if post:
        chosen["postoperative"] = post
    suspect = next((r for r in rows if r["patient_id"] == "Patient-068" and
                    (r["source_visit"] == "week-010" or r["target_visit"] == "week-010")), None)
    if suspect:
        chosen["known_suspect_extracranial"] = suspect
    output.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for category, row in chosen.items():
        a, b = visits.load(row["source_visit_id"]), visits.load(row["target_visit_id"])
        change = (b["masks"] - a["masks"]).numpy()
        support = np.abs(change).sum(0)
        if not support.any():
            support = (a["masks"] + b["masks"]).numpy().sum(0)
        center = tuple(int(support.sum(tuple(i for i in range(3) if i != axis)).argmax())
                       for axis in range(3)) if support.any() else (48, 48, 48)
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        for axis in range(3):
            for column, raw in enumerate((a, b)):
                image = raw["image"][int(raw["availability"].nonzero()[0])].numpy()
                plane = np.take(image, center[axis], axis=axis).T
                axes[axis, column].imshow(plane, cmap="gray", origin="lower", vmin=-1, vmax=3)
                for compartment, color in enumerate(("cyan", "orange", "lime")):
                    mask = np.take(raw["masks"][compartment].numpy(), center[axis], axis=axis).T
                    if mask.any() and not mask.all():
                        axes[axis, column].contour(mask, levels=[0.5], colors=[color], linewidths=0.7)
            signed = np.take(change, center[axis], axis=axis + 1).sum(0).T
            axes[axis, 2].imshow(signed, cmap="bwr", vmin=-3, vmax=3, origin="lower")
        for ax in axes.flat:
            ax.axis("off")
        for ax, title in zip(axes[0], ("Source MRI and masks", "Target MRI and masks", "Signed mask difference")):
            ax.set_title(title)
        fig.suptitle(f"{category}: {row['patient_id']} {row['source_visit']} to {row['target_visit']}\n"
                     f"Changed voxels {row['changed_voxels']}. Automated labels, not adjudicated change.", fontsize=11)
        fig.tight_layout()
        path = output / f"{category}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        receipts[category] = {"row_id": row["row_id"], "path": str(path), "sha256": digest(path)}
    return receipts


def smoke(prepared: str, checkpoint: str, seed: int, out: str) -> dict:
    model, normalization = initialize(checkpoint, seed)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    visits = Visits(prepared)
    pairs = ObservedPairs(visits)
    results = []
    for index, raw in enumerate(DataLoader(pairs, batch_size=1, shuffle=False)):
        if index >= 2:
            break
        a, b = raw["source"], raw["target"]
        optimizer.zero_grad(set_to_none=True)
        before = model.encoder.volume_head.weight.detach().clone()
        output = model(a["image"], a["availability"], b["image"], b["availability"])
        loss, terms = objective(output, raw, normalization, "C")
        step = optimizer_step(loss, model, optimizer, scaler)
        if not step["updated"] or torch.equal(before, model.encoder.volume_head.weight):
            raise ValueError("real smoke did not update state encoder")
        with torch.no_grad():
            identical = model.encoder.encode(a["image"], a["availability"])
            delta = [f - g for f, g in zip(identical, model.encoder.encode(a["image"], a["availability"]))]
            forward = model.decode_difference([d.detach() for d in output["differences"]])
            reverse = model.decode_difference([-d.detach() for d in output["differences"]])
        results.append({"row_id": raw["row_id"][0], "loss": float(loss.detach()),
                        "terms": {k: float(v.detach()) for k, v in terms.items()}, **step,
                        "feature_shapes": [list(d.shape) for d in output["differences"]],
                        "identity_feature_max": max(float(d.abs().max()) for d in delta),
                        "reversal_error": float((reverse["mask_change"] + forward["mask_change"]).abs().mean()),
                        "input_sha256": {v["visit_id"][0]: digest(visits.root / visits.lookup[v["visit_id"][0]]["tensor_file"])
                                         for v in (a, b)}})
        del output, loss, terms, identical, delta, forward, reverse
    result = {"status": "real_optimizer_verified", "updates": len(results), "pairs": results,
              "initial_checkpoint_sha256": digest(checkpoint), "normalization": normalization,
              "source": source_record(), "interpretation": "numerical runtime evidence, not representation quality"}
    save_json(out, result)
    return result


def train_arm(prepared: str, checkpoint: str, contract_path: str, audit_path: str,
              out: str, seed: int, arm: str, hours: float) -> dict:
    contract, audit = bind(prepared, contract_path, audit_path)
    if hours <= 0 or hours > contract["max_session_hours"]:
        raise ValueError("session ceiling must be between zero and nine hours")
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("full training requires a private Tesla T4")
    model, normalization = initialize(checkpoint, seed)
    if normalization != contract["normalization"]:
        raise ValueError("normalization differs from locked contract")
    device = torch.device("cuda")
    model.to(device)
    pairs = ObservedPairs(Visits(prepared, contract["training_patients"]))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(pairs, batch_size=1, shuffle=True, generator=generator, num_workers=2, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    output_dir = Path(out)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "training.json").exists():
        raise ValueError("training destination already contains a run; do not overwrite")
    provenance = {**source_record(), "contract_sha256": json_hash(contract),
                  "tensor_manifest_sha256": json_hash(audit["tensor_manifest"]),
                  "initial_checkpoint_sha256": digest(checkpoint), "prepared_sha256": contract["prepared_sha256"],
                  "training_patients": contract["training_patients"], "arm": arm, "seed": seed}
    history = []
    exposure, streak = 0, 0
    deadline = time.monotonic() + hours * 3600
    started = time.monotonic()
    status = "running"
    save_json(output_dir / "training.json", {"status": status, "history": history, "provenance": provenance})
    try:
        for epoch in range(20):
            model.train()
            losses, skips, kinds, order = [], 0, [], []
            epoch_start = time.monotonic()
            for raw in loader:
                if time.monotonic() >= deadline:
                    status = "incomplete_runtime_ceiling"
                    break
                batch = device_batch(raw, device)
                a, b = batch["source"], batch["target"]
                kind = NUISANCES[(exposure // 2) % len(NUISANCES)]
                visit = (a, b)[exposure % 2]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda"):
                    prediction = model(a["image"], a["availability"], b["image"], b["availability"])
                    loss, _ = objective(prediction, batch, normalization, arm)
                del prediction
                total_loss = float(loss.detach())
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite real-pair loss")
                if arm == "D":
                    scaler.scale(loss).backward()
                    del loss
                context = torch.enable_grad() if arm == "D" else torch.no_grad()
                with context, torch.autocast("cuda"):
                    synthetic = model(visit["image"], visit["availability"],
                                      nuisance(visit["image"], kind, seed=2026 + exposure), visit["availability"])
                    penalty = nuisance_loss(synthetic, normalization)
                if arm == "D":
                    loss = 0.25 * penalty
                    total_loss += float(loss.detach())
                del synthetic, penalty
                step = optimizer_step(loss, model, optimizer, scaler)
                order.append(raw["row_id"][0]); kinds.append(kind); exposure += 1
                if step["updated"]:
                    losses.append(total_loss); streak = 0
                else:
                    skips += 1; streak += 1
                    if streak >= 16:
                        raise ValueError("persistent nonfinite gradients after 16 AMP backoffs")
                del loss
            if status != "running":
                break
            if not losses:
                raise ValueError("epoch had no successful updates")
            row = {"epoch": epoch + 1, "training_loss": float(np.mean(losses)),
                   "real_pair_exposures": len(order), "nuisance_exposures": len(kinds),
                   "successful_updates": len(losses), "amp_skips": skips,
                   "pair_order_sha256": json_hash({"rows": order}), "nuisance_order_sha256": json_hash({"kinds": kinds}),
                   "seconds": time.monotonic() - epoch_start, "loss_scale": scaler.get_scale()}
            history.append(row)
            artifact = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(), "epoch": epoch + 1, "arm": arm, "seed": seed,
                        "normalization": normalization, "contract": CONTRACT, "provenance": provenance}
            torch.save(artifact, output_dir / "latest.pt")
            if epoch + 1 in (5, 10, 15, 20):
                torch.save(artifact, output_dir / f"epoch{epoch + 1}.pt")
            save_json(output_dir / "training.json", {"status": status, "history": history, "provenance": provenance})
            print(f"seed={seed} arm={arm} epoch={epoch + 1} updates={len(losses)} skips={skips} "
                  f"train={row['training_loss']:.6f} seconds={row['seconds']:.1f}", flush=True)
        if len(history) == 20:
            status = "complete"
            torch.save(artifact, output_dir / "final.pt")
            (output_dir / "final.sha256").write_text(digest(output_dir / "final.pt") + "\n")
    except Exception as error:
        status = "failed"
        save_json(output_dir / "failure.json", {"error": str(error), "provenance": provenance})
        raise
    finally:
        save_json(output_dir / "training.json", {"status": status, "history": history,
                  "provenance": provenance, "seconds": time.monotonic() - started,
                  "partial_epoch_exposures": exposure - sum(r["real_pair_exposures"] for r in history)})
    return {"status": status, "completed_epochs": len(history), "source": provenance}


def load_final(checkpoint: str, contract: dict) -> tuple[DeltaRepresentation, dict]:
    path = Path(checkpoint)
    if digest(path) != path.with_suffix(".sha256").read_text().strip():
        raise ValueError("final checkpoint bytes changed after creation")
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (artifact.get("contract") != CONTRACT or artifact.get("epoch") != 20
            or artifact["provenance"]["contract_sha256"] != json_hash(contract)
            or artifact["normalization"] != contract["normalization"]):
        raise ValueError("evaluation requires locked final epoch20 artifacts")
    for filename in ("src/model/representation.py", "src/model/delta_representation.py"):
        if artifact["provenance"]["source_hashes"][filename] != digest(filename):
            raise ValueError("model source graph differs from saved checkpoint")
    if not all(torch.isfinite(v).all() for v in artifact["state_dict"].values()):
        raise ValueError("nonfinite final weights")
    model = DeltaRepresentation()
    model.load_state_dict(artifact["state_dict"], strict=True)
    return model, artifact


def frozen_model(checkpoint: str, contract: dict, seed: int) -> tuple[DeltaRepresentation, str]:
    if digest(checkpoint) == CHECKPOINT_HASHES[seed]:
        model, normalization = initialize(checkpoint, seed)
        if normalization != contract["normalization"]:
            raise ValueError("baseline normalization mismatch")
        arm = "R35"
    else:
        model, artifact = load_final(checkpoint, contract)
        if artifact["seed"] != seed:
            raise ValueError("checkpoint seed mismatch")
        arm = artifact["arm"]
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, arm


def fit_probe(prepared: str, checkpoint: str, contract_path: str, audit_path: str,
              out: str, seed: int, view: str, hours: float) -> dict:
    contract, audit = bind(prepared, contract_path, audit_path)
    model, arm = frozen_model(checkpoint, contract, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch.manual_seed(2026)
    probe = LocalLinearProbe().to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    pairs = ObservedPairs(Visits(prepared, contract["training_patients"]))
    generator = torch.Generator().manual_seed(2026)
    loader = DataLoader(pairs, batch_size=1, shuffle=True, generator=generator, num_workers=2)
    directory = Path(out); directory.mkdir(parents=True, exist_ok=True)
    if (directory / "probe.pt").exists() or (directory / "training.json").exists():
        raise ValueError("probe destination already exists")
    history = []
    deadline = time.monotonic() + hours * 3600
    status = "running"
    provenance = {**source_record(), "contract_sha256": json_hash(contract), "checkpoint_sha256": digest(checkpoint),
                  "tensor_manifest_sha256": json_hash(audit["tensor_manifest"]), "arm": arm, "seed": seed, "view": view}
    for epoch in range(20):
        losses, order = [], []
        probe.train()
        for raw in loader:
            if time.monotonic() >= deadline:
                status = "incomplete_runtime_ceiling"
                break
            batch = device_batch(raw, device)
            with torch.no_grad():
                features = probe_input(model, batch, view)
            optimizer.zero_grad(set_to_none=True)
            prediction = probe(features)
            loss = balanced_loss(prediction, batch["mask_change"])
            optimizer_step(loss, probe, optimizer, scaler)
            losses.append(float(loss.detach())); order.append(raw["row_id"][0])
        if status != "running":
            break
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "updates": len(losses),
                        "pair_order_sha256": json_hash({"rows": order})})
        print(f"probe arm={arm} seed={seed} view={view} epoch={epoch + 1} loss={history[-1]['loss']:.6f}", flush=True)
        save_json(directory / "training.json", {"status": "running", "history": history, "provenance": provenance})
        torch.save({"state_dict": probe.state_dict(), "epoch": epoch + 1, "provenance": provenance,
                    "optimizer": optimizer.state_dict(), "contract": CONTRACT}, directory / "latest.pt")
    if len(history) == 20:
        status = "complete"
        torch.save({"state_dict": probe.state_dict(), "epoch": 20, "provenance": provenance,
                    "optimizer": optimizer.state_dict(), "contract": CONTRACT}, directory / "probe.pt")
        (directory / "probe.sha256").write_text(digest(directory / "probe.pt") + "\n")
    result = {"status": status, "history": history, "provenance": provenance}
    save_json(directory / "training.json", result)
    return result


def probe_input(model: DeltaRepresentation, pair: dict, view: str) -> list[torch.Tensor]:
    a, b = pair["source"], pair["target"]
    target = model.encoder.encode(b["image"], b["availability"])
    if view == "target_only":
        return target
    source = model.encoder.encode(a["image"], a["availability"])
    return [t - s for s, t in zip(source, target)]


def load_probe(path: str, checkpoint: str, contract: dict, view: str) -> LocalLinearProbe:
    if digest(path) != Path(path).with_suffix(".sha256").read_text().strip():
        raise ValueError("probe bytes changed after creation")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if (artifact.get("contract") != CONTRACT or artifact["epoch"] != 20
            or artifact["provenance"]["contract_sha256"] != json_hash(contract)
            or artifact["provenance"]["checkpoint_sha256"] != digest(checkpoint)
            or artifact["provenance"]["view"] != view):
        raise ValueError("probe weights do not match frozen encoder, budget, or view")
    probe = LocalLinearProbe()
    probe.load_state_dict(artifact["state_dict"], strict=True)
    probe.eval()
    return probe


def local_row(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    from sklearn.metrics import average_precision_score
    parts = partition_errors(prediction, target)
    row = {k: v[0].cpu().tolist() for k, v in parts.items()}
    row["balanced_mae"] = float(parts["balanced"].mean())
    support = (target.abs() > 0).flatten().cpu().numpy()
    scores = prediction.abs().flatten().cpu().numpy()
    row["changed_voxel_auprc"] = (float(average_precision_score(support, scores)) if support.any() else None)
    row["positive_voxels"] = int(support.sum())
    for name, selected in (("appearance", target > 0), ("disappearance", target < 0)):
        count = int(selected.sum())
        row[f"{name}_support"] = count
        row[f"{name}_mae"] = float((prediction - target).abs()[selected].mean()) if count else None
    return row


def summarize_local(rows: list[dict], key: str) -> dict:
    patients = sorted({r["patient_id"] for r in rows})
    per = {p: float(np.mean([r[key]["balanced_mae"] for r in rows if r["patient_id"] == p])) for p in patients}
    return {"patient_uniform_balanced_mae": float(np.mean(list(per.values()))), "per_patient": per,
            "n_rows": len(rows), "n_patients": len(patients),
            "pooled_pair_balanced_mae": float(np.mean([r[key]["balanced_mae"] for r in rows])),
            "changed_compartment_mae": supported_partition_mean(rows, key, "changed"),
            "unchanged_compartment_mae": supported_partition_mean(rows, key, "unchanged")}


def supported_partition_mean(rows: list[dict], key: str, partition: str) -> list:
    output = []
    for compartment in range(3):
        values = [r[key][partition][compartment] for r in rows if r[key][partition + "_count"][compartment] > 0]
        output.append({"mean": float(np.mean(values)) if values else None, "support_pairs": len(values)})
    return output


@torch.no_grad()
def encode(prepared: str, checkpoint: str, contract_path: str, out: str, seed: int) -> dict:
    contract = load_contract(contract_path)
    model, arm = frozen_model(checkpoint, contract, seed)
    visits = Visits(prepared)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if visits.metadata["cohort"] == "LUMIERE" and digest(visits.root / "prepared.json") != contract["prepared_sha256"]:
        raise ValueError("encoding source inventory differs from locked contract")
    features, predictions = {}, {}
    tensors = {v["tensor_file"]: digest(visits.root / v["tensor_file"]) for v in visits.records}
    scale = torch.tensor(contract["normalization"]["scale"], device=device)
    mean = torch.tensor(contract["normalization"]["mean"], device=device)
    for raw in DataLoader(visits, batch_size=1, num_workers=2):
        batch = device_batch(raw, device)
        maps = model.encoder.encode(batch["image"], batch["availability"])
        identifier = raw["visit_id"][0]
        embedding = model.encoder.summarize(maps)
        features[identifier] = embedding[0].cpu().numpy()
        predictions[identifier] = (model.encoder.volume_head(embedding) * scale + mean)[0].cpu().numpy()
    if any(digest(visits.root / filename) != expected for filename, expected in tensors.items()):
        raise ValueError("tensor bytes changed during feature extraction")
    provenance = {**source_record(), "contract_sha256": json_hash(contract), "checkpoint_sha256": digest(checkpoint),
                  "prepared_sha256": digest(visits.root / "prepared.json"), "tensor_hashes": tensors, "arm": arm, "seed": seed,
                  "spatial_extraction": "all four levels from encoder.encode; only pooled final embeddings persisted"}
    artifact = {"contract": CONTRACT, "features": features, "state_predictions": predictions, "provenance": provenance}
    path = Path(out); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)
    path.with_suffix(".sha256").write_text(digest(path) + "\n")
    return {"status": "encoded", "n_visits": len(features), "provenance": provenance}


@torch.no_grad()
def evaluate(prepared: str, checkpoint: str, contract_path: str, probe_path: str,
             target_probe_path: str, out: str, seed: int, cohort: str, audit_path: str) -> dict:
    contract = load_contract(contract_path)
    audit = json.loads(Path(audit_path).read_text())
    if audit["contract_sha256"] != json_hash(contract) or audit["visual_review_status"] != "reviewed":
        raise ValueError("evaluation requires reviewed source target QA")
    if digest(Path(audit_path).parent / "target_rows.json") != audit["target_rows_sha256"]:
        raise ValueError("source reporting strata changed after target audit")
    model, arm = frozen_model(checkpoint, contract, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    probe = load_probe(probe_path, checkpoint, contract, "difference").to(device)
    target_probe = load_probe(target_probe_path, checkpoint, contract, "target_only").to(device)
    visits = Visits(prepared, None if cohort == "SAILOR" else contract["evaluation_patients"])
    if visits.metadata.get("cohort") != cohort:
        raise ValueError("query cohort mismatch")
    if cohort == "LUMIERE" and digest(visits.root / "prepared.json") != contract["prepared_sha256"]:
        raise ValueError("source query inventory differs from locked rows")
    tensor_hashes = {v["tensor_file"]: digest(visits.root / v["tensor_file"]) for v in visits.metadata["visits"]}
    if cohort == "LUMIERE" and tensor_hashes != audit["tensor_manifest"]["tensors"]:
        raise ValueError("source query tensors changed after QA")
    pairs = ObservedPairs(visits)
    expected = (269, 242) if cohort == "SAILOR" else (149, 117)
    if (len(visits), len(pairs)) != expected:
        raise ValueError("query inventory has missing visits or pairs")
    rows, states = [], []
    normalization = contract["normalization"]
    scale = torch.tensor(normalization["scale"], device=device)
    mean = torch.tensor(normalization["mean"], device=device)
    change_scale = torch.tensor(normalization["change_scale"], device=device)
    for raw in DataLoader(visits, batch_size=1, num_workers=2):
        batch = device_batch(raw, device)
        output = model.encoder(batch["image"], batch["availability"])
        prediction = (output["volume"] * scale + mean)[0].cpu().numpy()
        states.append({"visit_id": raw["visit_id"][0], "patient_id": raw["patient_id"][0],
                       "truth": batch["volume"][0].cpu().tolist(), "prediction": prediction.tolist()})
    for index, raw in enumerate(DataLoader(pairs, batch_size=1, num_workers=2)):
        batch = device_batch(raw, device)
        a, b = batch["source"], batch["target"]
        output = model(a["image"], a["availability"], b["image"], b["availability"])
        differences = output["differences"]
        later = model.encoder.encode(b["image"], b["availability"])
        predictions = {"frozen_probe": probe(differences), "target_only": target_probe(later),
                       "zero": torch.zeros_like(batch["mask_change"]),
                       "independent_state": output["target"]["mask_logits"].float().sigmoid() -
                                            output["source"]["mask_logits"].float().sigmoid()}
        if arm not in ("A", "R35"):
            predictions["direct_change"] = output["mask_change"]
        row = {"row_id": raw["row_id"][0], "patient_id": raw["patient_id"][0],
               "source_visit_id": a["visit_id"][0], "target_visit_id": b["visit_id"][0],
               "volume_truth": batch["change"][0].cpu().tolist(),
               "volume_independent": ((output["target"]["volume"] - output["source"]["volume"]) * scale)[0].cpu().tolist(),
               "source_empty": (a["volume"] == 0)[0].cpu().tolist(),
               "target_empty": (b["volume"] == 0)[0].cpu().tolist(),
               "source_grid_empty": (a["masks"].flatten(2).sum(-1) == 0)[0].cpu().tolist(),
               "target_grid_empty": (b["masks"].flatten(2).sum(-1) == 0)[0].cpu().tolist()}
        row.update({key: local_row(value, batch["mask_change"]) for key, value in predictions.items()})
        if arm not in ("A", "R35"):
            row["volume_direct"] = (output["change"] * change_scale)[0].cpu().tolist()
            reverse = model.decode_difference([-d for d in differences])
            row["reversal_map_error"] = float((reverse["mask_change"] + output["mask_change"]).abs().mean())
            row["reversal_volume_error"] = float(((reverse["change"] + output["change"]) * change_scale).abs().mean())
        rows.append(row)
        if (index + 1) % 20 == 0:
            print(f"evaluate {cohort} arm={arm} pairs={index + 1}/{len(pairs)}", flush=True)
    groups = np.asarray([r["patient_id"] for r in states])
    state_metrics = metrics(np.asarray([r["truth"] for r in states]), np.asarray([r["prediction"] for r in states]), groups)
    result = {"status": "complete", "cohort": cohort, "arm": arm, "seed": seed,
              "state": state_metrics, "local": {k: summarize_local(rows, k) for k in predictions},
              "rows": rows, "states": states, "descriptive_companions": companion_tables(rows, states, audit["source52_strata_thresholds"]),
              "provenance": {**source_record(), "checkpoint_sha256": digest(checkpoint), "tensor_hashes": tensor_hashes,
                             "contract_sha256": json_hash(contract), "prepared_sha256": digest(visits.root / "prepared.json"),
                             "probe_sha256": digest(probe_path), "target_probe_sha256": digest(target_probe_path)},
              "interpretation": "automated occupancy and native log-volume proxies; development evidence, not forecasting"}
    if any(digest(visits.root / filename) != expected_hash for filename, expected_hash in tensor_hashes.items()):
        raise ValueError("query tensors changed during evaluation")
    save_json(out, result)
    return result


@torch.no_grad()
def evaluate_nuisance(prepared: str, checkpoint: str, contract_path: str, probe_path: str,
                      out: str, seed: int, audit_path: str) -> dict:
    contract, audit = bind(prepared, contract_path, audit_path)
    model, arm = frozen_model(checkpoint, contract, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    probe = load_probe(probe_path, checkpoint, contract, "difference").to(device)
    visits = Visits(prepared, contract["evaluation_patients"])
    chosen = {}
    for record in visits.records:
        chosen.setdefault(record["patient_id"], record)
    scale = torch.tensor(contract["normalization"]["scale"], device=device)
    rows = []
    for patient, record in sorted(chosen.items()):
        raw = visits.load(record["visit_id"])
        image, availability = raw["image"][None].to(device), raw["availability"][None].to(device)
        source = model.encoder.encode(image, availability)
        source_volume = model.encoder.volume_head(model.encoder.summarize(source))
        for kind in (*NUISANCES, "resolution"):
            target = model.encoder.encode(nuisance(image, kind), availability)
            difference = [b - a for a, b in zip(source, target)]
            prediction = probe(difference)
            volume = model.encoder.volume_head(model.encoder.summarize(target))
            row = {"patient_id": patient, "visit_id": record["visit_id"], "kind": kind,
                   "false_change": float(prediction.abs().mean()),
                   "state_drift": float(((volume - source_volume) * scale).abs().mean())}
            if arm not in ("A", "R35"):
                direct = model.decode_difference(difference)
                row["direct_false_change"] = float(direct["mask_change"].abs().mean())
            rows.append(row)
    identities = {r["patient_id"]: r["false_change"] for r in rows if r["kind"] == "identity"}
    for row in rows:
        row["excess_false_change"] = row["false_change"] - identities[row["patient_id"]]
    result = {"status": "complete", "arm": arm, "seed": seed, "rows": rows,
              "conditions": {kind: {k: float(np.mean([r[k] for r in rows if r["kind"] == kind]))
                                     for k in ("false_change", "state_drift", "excess_false_change")}
                             for kind in (*NUISANCES, "resolution")},
              "provenance": {**source_record(), "checkpoint_sha256": digest(checkpoint),
                             "probe_sha256": digest(probe_path), "contract_sha256": json_hash(contract),
                             "prepared_sha256": contract["prepared_sha256"],
                             "tensor_manifest_sha256": json_hash(audit["tensor_manifest"])},
              "limits": "One predetermined visit per development patient. Resolution is held out of nuisance training."}
    save_json(out, result)
    return result


def report(directory: str, contract_path: str, out: str) -> dict:
    contract = load_contract(contract_path)
    root = Path(directory)
    receipts = {}
    for seed in contract["seeds"]:
        for arm in ARMS:
            path = root / f"seed{seed}" / arm / "training.json"
            value = json.loads(path.read_text()) if path.exists() else {"status": "not_started", "history": []}
            receipts[f"{seed}/{arm}"] = {"status": value["status"], "epochs": len(value.get("history", []))}
    completed = sum(r["status"] == "complete" and r["epochs"] == 20 for r in receipts.values())
    result = {"status": "INCOMPLETE" if completed != 12 else "TRAINING_COMPLETE_EVALUATION_PENDING",
              "contract_sha256": json_hash(contract), "training": receipts,
              "hypotheses": {"C_vs_B_frozen_accessibility": "INCONCLUSIVE", "state_retention": "INCONCLUSIVE",
                             "D_nuisance_tradeoff": "INCONCLUSIVE", "incremental_temporal_information": "INCONCLUSIVE"},
              "missing_gates": ["independent design and result review", "expert repeatability and biological annotations",
                                "untouched external validation"],
              "phase2": "not_authorized; requires operator decision after Phase 1 report"}
    result.update(aggregate_matrix(root, contract))
    companions = ("lumiere.json", "sailor.json", "nuisance.json", "ridge-lumiere.json", "ridge-sailor.json",
                  "difference/probe.pt", "target_only/probe.pt", "final.pt")
    missing = [str(root / f"seed{seed}" / arm / filename) for seed in contract["seeds"] for arm in ARMS
               for filename in companions if not (root / f"seed{seed}" / arm / filename).exists()]
    baseline_missing = [str(root / f"seed{seed}" / "R35" / filename) for seed in contract["seeds"]
                        for filename in companions[:-1] if not (root / f"seed{seed}" / "R35" / filename).exists()]
    result["missing_companion_artifacts"] = missing + baseline_missing
    if completed != 12 or missing or baseline_missing:
        result["status"] = "INCOMPLETE"
    else:
        result["status"] = "RUNNABLE_ARTIFACTS_PRESENT_REVIEW_PENDING"
    save_json(out, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--prepared", required=True); audit.add_argument("--r35-run", required=True)
    audit.add_argument("--out", required=True); audit.add_argument("--ratings")
    audit.add_argument("--inventory", required=True); audit.add_argument("--mask-root", required=True)
    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument("--prepared", required=True); smoke_parser.add_argument("--checkpoint", required=True)
    smoke_parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42); smoke_parser.add_argument("--out", required=True)
    for command in ("train", "probe", "encode", "eval", "transfer", "nuisance"):
        command_parser = sub.add_parser(command)
        command_parser.add_argument("--prepared", required=True); command_parser.add_argument("--checkpoint", required=True)
        command_parser.add_argument("--contract", required=True); command_parser.add_argument("--out", required=True)
        command_parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
        if command in ("train", "probe", "eval", "transfer", "nuisance"):
            command_parser.add_argument("--audit", required=True)
        if command in ("train", "probe"):
            command_parser.add_argument("--hours", type=float, default=8)
        if command == "train":
            command_parser.add_argument("--arm", choices=tuple(ARMS), required=True)
        if command == "probe":
            command_parser.add_argument("--view", choices=("difference", "target_only"), required=True)
        if command in ("eval", "transfer", "nuisance"):
            command_parser.add_argument("--probe", required=True)
        if command in ("eval", "transfer"):
            command_parser.add_argument("--target-probe", required=True)
    ridge = sub.add_parser("ridge")
    ridge.add_argument("--training-features", required=True); ridge.add_argument("--query-features", required=True)
    ridge.add_argument("--prepared", required=True); ridge.add_argument("--query-prepared", required=True)
    ridge.add_argument("--contract", required=True); ridge.add_argument("--out", required=True)
    summary = sub.add_parser("report")
    summary.add_argument("--results", required=True); summary.add_argument("--contract", required=True); summary.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    if args.command == "audit":
        result = audit_targets(args.prepared, args.r35_run, args.out, args.ratings, args.inventory, args.mask_root)
    elif args.command == "smoke":
        result = smoke(args.prepared, args.checkpoint, args.seed, args.out)
    elif args.command == "train":
        result = train_arm(args.prepared, args.checkpoint, args.contract, args.audit, args.out, args.seed, args.arm, args.hours)
    elif args.command == "probe":
        result = fit_probe(args.prepared, args.checkpoint, args.contract, args.audit, args.out, args.seed, args.view, args.hours)
    elif args.command == "encode":
        result = encode(args.prepared, args.checkpoint, args.contract, args.out, args.seed)
    elif args.command in ("eval", "transfer"):
        result = evaluate(args.prepared, args.checkpoint, args.contract, args.probe, args.target_probe, args.out, args.seed,
                          "SAILOR" if args.command == "transfer" else "LUMIERE", args.audit)
    elif args.command == "nuisance":
        result = evaluate_nuisance(args.prepared, args.checkpoint, args.contract, args.probe, args.out, args.seed, args.audit)
    elif args.command == "ridge":
        result = scalar_readouts(args.training_features, args.query_features, args.prepared,
                                 args.query_prepared, args.contract, args.out)
    else:
        result = report(args.results, args.contract, args.out)
    print(json.dumps({k: result[k] for k in ("status", "coverage", "updates", "conditions", "local", "hypotheses") if k in result}, sort_keys=True))


if __name__ == "__main__":
    main()
