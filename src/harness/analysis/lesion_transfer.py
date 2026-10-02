"""Evaluate a learned lesion-transition checkpoint on locked LUMIERE and SAILOR."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.collate import make_collate
from src.data.eval_protocol import load_protocol
from src.data.sailor import SAILORDataset
from src.harness.checkpoints import load_model
from src.harness.train.lesion import (build_feature_cache, datasets, load_masks,
                                      mask_indices, pair_validity, selected_gaps, visit_features)
from src.harness.data.anatomy_manifest import _stable_id
from src.harness.provenance import git_sha, make_provenance
from src.harness.analysis.clean_rescues import sha256
from src.model.jepa_model import JEPAWorldModel
from src.model.lesion_forecaster import FORECAST_INPUT_CONTRACT, LesionTransitionForecaster

SAILOR_MASKS = ("NecrosisMask-ONCO", "ContrastEnhancedMask-ONCO", "EdemaMask-ONCO")


def _find(directory: Path, stem: str):
    for suffix in (".nii.gz", ".nii"):
        path = directory / f"{stem}{suffix}"
        if path.exists(): return path
    return None


def sailor_masks(root: str, patient: str, visits: list[str]):
    all_masks, volumes, keep = [], [], []
    for i, visit in enumerate(visits):
        directory = Path(root) / patient / visit
        paths = [_find(directory, stem) for stem in SAILOR_MASKS]
        if any(path is None for path in paths): continue
        current, current_volumes = [], []
        for path in paths:
            image = nib.load(str(path)); data = np.asanyarray(image.dataobj)
            current.append(data > 0)
            current_volumes.append(np.count_nonzero(data > 0) *
                                   abs(np.linalg.det(image.affine[:3, :3])))
        all_masks.append(torch.from_numpy(np.stack(current)).float())
        volumes.append(torch.tensor(current_volumes).log1p()); keep.append(i)
    if not keep: return None, None, []
    masks = torch.nn.functional.interpolate(torch.stack(all_masks),
                                             size=(96, 96, 96), mode="nearest") > 0
    return masks, torch.stack(volumes).float(), keep


@torch.no_grad()
def sailor_cache(backbone, cfg, root, device):
    ds = SAILORDataset(root)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=make_collate(tuple(cfg["preprocessing"]["target_size"])))
    cache = {}; backbone.eval()
    for batch in loader:
        pid = batch["patient_id"][0]
        visits = ds[ds.subjects.index(pid)]["visits"]
        masks, volumes, indices = sailor_masks(root, pid, visits)
        if len(indices) < 2: continue
        values = []
        for j, t in enumerate(indices):
            present = batch["mri_mask"][0, t]
            values.append(visit_features(backbone, batch["mri"][0, t, :, 0].to(device),
                                         present.to(device), masks[j].to(device)).cpu())
        gaps = selected_gaps(batch["time_deltas"][0], indices)
        valid = pair_validity(indices, gaps)
        if not bool(valid.any()): continue
        selected = [visits[i] for i in indices]
        cache[pid] = {"features": torch.stack(values), "volumes": volumes,
                      "gaps": gaps, "visits": selected, "pair_valid": valid,
                      "row_ids": [_stable_id("SAILOR", pid, a, b)
                                  for a, b in zip(selected[:-1], selected[1:])]}
        print(f"SAILOR learned encode {len(cache)}/{len(ds)}", flush=True)
    return cache


def evaluate(cache, model, device, boot, seed):
    per, persistence = {}, {}; model.eval()
    with torch.no_grad():
        for patient, row in cache.items():
            source = row["volumes"].to(device)
            prediction = model(row["features"].to(device), row["gaps"].to(device),
                               source)["prediction"]
            target = source[1:]
            valid = row["pair_valid"].to(device)
            per[patient] = float((prediction[valid] - target[valid]).abs().mean())
            persistence[patient] = float((source[:-1][valid] - target[valid]).abs().mean())
    patients = np.array(sorted(per)); diff = np.array([per[p] - persistence[p] for p in patients])
    rng = np.random.default_rng(seed)
    draws = rng.choice(diff, (boot, len(diff)), replace=True).mean(1)
    mae, baseline = np.mean(list(per.values())), np.mean(list(persistence.values()))
    return {"patient_uniform_log_volume_mae": float(mae),
            "persistence_mae": float(baseline), "persistence_relative_mae": float(mae / baseline),
            "mae_difference": float(diff.mean()),
            "mae_difference_95ci": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
            "n_patients": len(per), "n_rows": sum(int(r["pair_valid"].sum()) for r in cache.values()),
            "per_patient_mae": per,
            "per_patient_persistence_mae": persistence}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--champion", default="checkpoints/champion_0.0081.pt")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mask-root", default="data/autoseg/extracted/Imaging")
    ap.add_argument("--sailor-root", default="data/sailor/sailor_ebrains_pseud/derivatives/mni2009c-n-s")
    ap.add_argument("--protocol", default="info/eval_folds.json")
    ap.add_argument("--out", default="outputs/p2_learned_lesion_transfer.json")
    ap.add_argument("--cache-out", default="checkpoints/learned_lesion_eval_cache.pt")
    ap.add_argument("--reuse-cache", action="store_true")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text()); device = torch.device("cpu")
    learned = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if learned.get("forecast_input_contract") != FORECAST_INPUT_CONTRACT:
        raise ValueError("legacy lesion checkpoint used target-visit inputs; retrain with history-only-lesion-v2")
    world = JEPAWorldModel(cfg); load_model(world, args.champion)
    world.backbone.load_state_dict(learned["backbone"]); world.to(device).eval()
    forecaster = LesionTransitionForecaster().to(device)
    forecaster.load_state_dict(learned["forecaster"])
    if args.reuse_cache:
        cached = torch.load(args.cache_out, map_location="cpu", weights_only=False)
        if cached.get("provenance", {}).get("champion_sha256") != sha256(args.checkpoint):
            raise ValueError("feature cache checkpoint hash mismatch; rebuild the causal cache")
        lum, sailor = cached["lumiere"], cached["sailor"]
    else:
        protocol = load_protocol(args.protocol)
        eligible = [p for p in protocol["encoder_unseen"]
                    if len(list(Path(args.mask_root).glob(
                        f"{p}/week-*/DeepBraTumIA-segmentation/atlas/segmentation/seg_mask.nii*"))) >= 2]
        lum_ds = datasets(cfg, eligible)
        lum_loader = DataLoader(lum_ds, batch_size=1, shuffle=False, num_workers=0,
                                collate_fn=make_collate(tuple(cfg["preprocessing"]["target_size"])))
        lum = build_feature_cache(world.backbone, lum_loader, args.mask_root, device)
        sailor = sailor_cache(world.backbone, cfg, args.sailor_root, device)
        Path(args.cache_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema_version": 2,
                    "provenance": make_provenance(args.checkpoint, args.config,
                        champion_sha256=sha256(args.checkpoint), forecast_input_contract=FORECAST_INPUT_CONTRACT),
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "lumiere": lum, "sailor": sailor}, args.cache_out)
    result = {"schema_version": 1, "git_sha": git_sha(),
              "checkpoint_sha256": sha256(args.checkpoint),
              "forecast_input_contract": FORECAST_INPUT_CONTRACT,
              "checkpoint": str(Path(args.checkpoint).resolve()),
              "feature_cache": str(Path(args.cache_out).resolve()),
              "lumiere_locked": evaluate(lum, forecaster, device, args.boot, args.seed),
              "sailor_transfer": evaluate(sailor, forecaster, device, args.boot, args.seed + 1)}
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for name in ("lumiere_locked", "sailor_transfer"):
        row = result[name]
        print(f"{name}: MAE={row['patient_uniform_log_volume_mae']:.4f} "
              f"rel={row['persistence_relative_mae']:.3f} CI={row['mae_difference_95ci']}")


if __name__ == "__main__": main()
