"""Matched GPU reruns of the pre-audit training rescues.

The basin experiment has four 6-epoch arms. Each objective rescue is a full
30-epoch scratch run, changing one setting from the clean P0 recipe. All arms
are evaluated on identical unfiltered next-visit rows; anatomical MAE is the
primary endpoint, with each arm's EMA persistence gate reported separately.
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.collate import make_collate
from src.data.eval_protocol import fold_patients, load_protocol
from src.harness.analysis.anatomy_baselines import (
    _bootstrap_differences, _evaluate_fold, _merge_folds, build_rows)
from src.harness.analysis.clean_rescues import pair_tensors, sha256, summary
from src.harness.data.builder import build_datasets
from src.harness.encode.encoders import features_for_batch, load_champion
from src.harness.provenance import git_sha, make_provenance
from src.harness.train.latent import load_1step_predictor

BASIN = {"fresh_accum1": (False, 1), "loaded_accum1": (True, 1),
         "fresh_accum8": (False, 8), "loaded_accum8": (True, 8)}
OBJECTIVES = ("augmentation", "surgery", "transition_weighting")


def arm_config(base, arm, output):
    cfg = yaml.safe_load(yaml.safe_dump(base))
    cfg["data"].update(include_survival=False, augment_train=arm == "augmentation")
    tr = cfg["training"]
    tr.update(batch_size=1, accumulation_steps=8, max_epochs=30, lr=1e-4,
              warmup_epochs=5, log_wandb=False, surgery_window=arm == "surgery",
              transition_weighting=arm == "transition_weighting", checkpoint_dir=str(output))
    tr.pop("transition_weights", None)
    cfg.setdefault("aux", {})["lambda"] = 0.
    if arm in BASIN:
        tr.update(accumulation_steps=BASIN[arm][1], max_epochs=6, lr=2e-5, warmup_epochs=0)
    return cfg


def encode_gpu(cfg, champion, output, config_path):
    model, checkpoint = load_champion(cfg, champion)
    model.cuda().eval()
    ds_by_split, _ = build_datasets(cfg)
    cache = {"schema_version": 2, "patients": {}, "provenance": make_provenance(
        champion, config_path, champion_sha256=sha256(champion),
        clinical_schema="survival_free_v1", target_space="ema",
        champion_epoch=checkpoint.get("epoch"), champion_val=checkpoint.get("val_loss"))}
    with torch.no_grad():
        for split, ds in ds_by_split.items():
            loader = DataLoader(ds, batch_size=1, num_workers=0,
                                collate_fn=make_collate(tuple(cfg["preprocessing"]["target_size"])))
            for batch in loader:
                pid = batch["patient_id"][0]; item = ds[ds.patients.index(pid)]
                batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
                features, meta = features_for_batch(model, batch, item, ds,
                                                    ("vision", "states", "clinical", "ema_z", "z"))
                cache["patients"][pid] = {"split": split, **{
                    k: v.cpu() if torch.is_tensor(v) else v for k, v in {**features, **meta}.items()}}
                print(f"GPU encode {len(cache['patients'])}/91", flush=True)
    torch.save(cache, output)
    del model; gc.collect(); torch.cuda.empty_cache()
    return cache


def evaluate(cache, checkpoint, manifest_path, protocol_path, output, teacher_cache):
    protocol = load_protocol(protocol_path)
    rows = build_rows(cache, json.loads(Path(manifest_path).read_text()), "LUMIERE")
    unseen = protocol["encoder_unseen"]
    methods = ("persistence", "volume_ridge", "current_image", "history_mean", "jepa_state")
    folds = []
    for i in range(protocol["k"]):
        test = fold_patients(protocol, i); train = [p for p in unseen if p not in set(test)]
        folds.append(_evaluate_fold(rows, train, test, methods))
    anatomy = _merge_folds(folds, methods); _bootstrap_differences(anatomy, 10000, 42)
    predictor = load_1step_predictor(checkpoint)
    latent = {}
    for split in ("train", "val", "test"):
        patients = sorted(p for p, r in cache["patients"].items() if r["split"] == split)
        pairs = pair_tensors(cache["patients"], patients, predictor)
        latent[split] = summary(pairs, pairs["raw"], 10000, 42)
    if teacher_cache:
        # A cross-arm feature-quality control: ridge reads into a fixed clean
        # teacher's target space, so moving each arm's EMA cannot create a win.
        import numpy as np
        from src.harness.analysis.anatomy_baselines import _fit_ridge, _flat
        teacher = teacher_cache["patients"]; fixed_rows = {}
        for pid, item in cache["patients"].items():
            reference = teacher[pid]
            if item["visits"] != reference["visits"]:
                raise ValueError(f"fixed-teacher visit mismatch: {pid}")
            for t in range(len(item["states"])):
                if not (item["has_img"][t] and item["has_img"][t + 1]): continue
                fixed_rows.setdefault(pid, []).append({"target": reference["z"][t + 1].numpy(),
                    "source": reference["z"][t].numpy(), "state": item["states"][t].numpy()})
        predictions, truth, source, groups = [], [], [], []
        for i in range(protocol["k"]):
            test = fold_patients(protocol, i); train = [p for p in unseen if p not in set(test)]
            scaler, ridge, _ = _fit_ridge(fixed_rows, train, "state")
            x, y, g, rr = _flat(fixed_rows, test, "state")
            predictions.append(ridge.predict(scaler.transform(x))); truth.append(y)
            source.append(np.stack([r["source"] for r in rr])); groups.extend(g)
        fixed = {"target": torch.from_numpy(np.concatenate(truth)),
                 "source": torch.from_numpy(np.concatenate(source)), "groups": np.array(groups)}
        fixed_teacher = summary(fixed, torch.from_numpy(np.concatenate(predictions)), 10000, 42)
    else:
        fixed_teacher = None
    payload = {"schema_version": 1, "git_sha": git_sha(), "checkpoint": str(checkpoint),
               "anatomy_row_ids": sorted(r["row_id"] for rr in rows.values() for r in rr),
               "checkpoint_sha256": sha256(checkpoint), "cache_provenance": cache["provenance"],
               "manifest_sha256": sha256(manifest_path), "anatomy": anatomy,
               "own_ema_latent_diagnostic": latent, "fixed_teacher_state_readout": fixed_teacher,
               "selection": "within-arm best validation objective; locked rows are evaluation only"}
    Path(output).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print("ANATOMY", {k: v["persistence_relative_mae"] for k, v in anatomy.items()}, flush=True)
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--champion", required=True)
    ap.add_argument("--experiment", choices=("basin", *OBJECTIVES), required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--protocol", default="info/eval_folds.json")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    torch.set_num_threads(2)
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("rescue runner requires Tesla T4")
    baseline = torch.load(args.champion, map_location="cpu", weights_only=False)
    if baseline["config"]["data"].get("include_survival", True):
        raise ValueError("baseline was trained with eventual survival")
    if not baseline.get("opt"):
        raise ValueError("basin-hold requires the optimizer-bearing clean checkpoint")
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    baseline_cfg = arm_config(cfg, "control", out / "control")
    baseline_path = out / "control.yaml"; baseline_path.write_text(yaml.safe_dump(baseline_cfg))
    teacher_cache = encode_gpu(baseline_cfg, args.champion, out / "control_cache.pt", str(baseline_path))
    control = evaluate(teacher_cache, args.champion, args.manifest, args.protocol,
                       out / "control_eval.json", teacher_cache)
    status = {"git_sha": git_sha(), "experiment": args.experiment,
              "baseline_sha256": sha256(args.champion), "arms": {"control": "COMPLETE"}}
    arms = list(BASIN) if args.experiment == "basin" else [args.experiment]
    for arm in arms:
        arm_dir = out / arm; arm_dir.mkdir(exist_ok=True)
        arm_cfg = arm_config(cfg, arm, arm_dir)
        path = arm_dir / "config.yaml"; path.write_text(yaml.safe_dump(arm_cfg))
        cmd = [sys.executable, "-u", "scripts/harness.py", "legacy", "train", "jepa",
               "--config", str(path), "--no-wandb"]
        if arm in BASIN:
            cmd += ["--resume-from", args.champion]
            if BASIN[arm][0]: cmd += ["--resume-opt"]
        status["arms"][arm] = "RUNNING"; (out / "status.json").write_text(json.dumps(status, indent=2))
        with (arm_dir / "train.log").open("w") as log:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end="", flush=True); log.write(line); log.flush()
            rc = process.wait()
        if rc:
            status["arms"][arm] = f"ERROR exit {rc}"
            (out / "status.json").write_text(json.dumps(status, indent=2))
            raise RuntimeError(f"{arm} failed; see train.log")
        cache = encode_gpu(arm_cfg, str(arm_dir / "best.pt"), arm_dir / "cache.pt", str(path))
        evaluation = evaluate(cache, str(arm_dir / "best.pt"), args.manifest, args.protocol,
                              arm_dir / "eval.json", teacher_cache)
        if evaluation["anatomy_row_ids"] != control["anatomy_row_ids"]:
            raise ValueError("matched evaluation row identities changed")
        status["arms"][arm] = "COMPLETE"; (out / "status.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__": main()
