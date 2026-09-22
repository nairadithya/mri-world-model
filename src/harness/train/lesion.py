"""Two-stage P2.4 lesion encoder + elapsed-time transition GRU training."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.data.collate import make_collate
from src.data.dataset import LUMIEREDataset
from src.data.eval_protocol import load_protocol
from src.harness.checkpoints import load_model
from src.model.jepa_model import JEPAWorldModel
from src.model.lesion_forecaster import LesionTransitionForecaster
from src.model.lesion_tokens import lesion_centered_crop, pool_region_tokens, region_patch_weights

LABELS = (2, 1, 3)  # necrotic/nonenhancing, enhancing, edema


def _mask_path(root: str, patient: str, visit: str) -> Path:
    return (Path(root) / patient / visit / "DeepBraTumIA-segmentation" /
            "atlas/segmentation/seg_mask.nii.gz")


def load_masks(root: str, patient: str, visits: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    masks, volumes = [], []
    for visit in visits:
        image = nib.load(str(_mask_path(root, patient, visit)))
        data = np.asanyarray(image.dataobj)
        voxel = float(abs(np.linalg.det(image.affine[:3, :3])))
        masks.append(torch.from_numpy(np.stack([data == label for label in LABELS])))
        volumes.append(torch.tensor([np.count_nonzero(data == label) * voxel
                                     for label in LABELS]).log1p())
    tensor = torch.stack(masks).float()
    tensor = F.interpolate(tensor, size=(96, 96, 96), mode="nearest") > 0
    return tensor, torch.stack(volumes).float()


def mask_indices(root: str, patient: str, visits: list[str]) -> list[int]:
    return [i for i, visit in enumerate(visits)
            if _mask_path(root, patient, visit).exists()]


def selected_gaps(deltas: torch.Tensor, indices: list[int]) -> torch.Tensor:
    days = torch.cumsum(deltas.float(), 0)
    selected = days[indices]
    return torch.cat([selected.new_zeros(1), selected[1:] - selected[:-1]])


def configure_lora(backbone):
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    trainable = []
    for name, parameter in backbone.named_parameters():
        if "lora_" in name:
            parameter.requires_grad = True
            trainable.append(parameter)
    return trainable


def visit_features(backbone, images, present, masks, crop_size=64):
    crop_images, crop_masks = lesion_centered_crop(
        images[None], masks[None], crop_size=crop_size, output_size=96)
    tokens = images.new_zeros(4, 216, 768)
    hidden = backbone.encode_tokens(crop_images[0, present, None].float())
    hidden = hidden[:, 1:] if hidden.shape[1] == 217 else hidden
    tokens[present] = hidden
    weights = region_patch_weights(crop_masks)
    regions, contrasts = pool_region_tokens(tokens[None], weights)
    region = regions[0, present].mean(0).flatten()
    contrast = contrasts[0, present].mean(0).flatten()
    return torch.cat([region, contrast])


def datasets(cfg, patients):
    common = dict(meta_dir=cfg["data"]["meta_dir"], processed_root=cfg["data"]["root"],
                  raw_root=None, modalities=tuple(cfg["data"]["modalities"]), min_visits=2)
    return LUMIEREDataset(patients=list(patients), **common)


def batches(ds, cfg, shuffle):
    return DataLoader(ds, batch_size=1, shuffle=shuffle, num_workers=0,
                      collate_fn=make_collate(tuple(cfg["preprocessing"]["target_size"])))


@torch.no_grad()
def build_feature_cache(backbone, loader, mask_root, device):
    backbone.eval(); cache = {}
    for batch in loader:
        pid = batch["patient_id"][0]
        all_visits = loader.dataset[loader.dataset.patients.index(pid)]["visits"]
        indices = mask_indices(mask_root, pid, all_visits)
        visits = [all_visits[i] for i in indices]
        if len(visits) < 2:
            continue
        masks, volumes = load_masks(mask_root, pid, visits)
        values = []
        for j, t in enumerate(indices):
            present = batch["mri_mask"][0, t]
            values.append(visit_features(backbone, batch["mri"][0, t, :, 0].to(device),
                                         present.to(device), masks[j].to(device)).cpu())
        cache[pid] = {"features": torch.stack(values), "volumes": volumes,
                      "gaps": selected_gaps(batch["time_deltas"][0], indices),
                      "visits": visits}
    return cache


def patient_mae(cache, model, device):
    model.eval(); values, persistence = [], []
    with torch.no_grad():
        for row in cache.values():
            source = row["volumes"].to(device)
            out = model(row["features"].to(device), row["gaps"].to(device), source)
            target = source[1:]
            values.append(float((out["prediction"] - target).abs().mean()))
            persistence.append(float((source[:-1] - target).abs().mean()))
    return float(np.mean(values)), float(np.mean(persistence))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--champion", default="checkpoints/champion_0.0081.pt")
    ap.add_argument("--protocol", default="info/eval_folds.json")
    ap.add_argument("--mask-root", required=True)
    ap.add_argument("--out", default="checkpoints/lesion_learned.pt")
    ap.add_argument("--anatomy-epochs", type=int, default=8)
    ap.add_argument("--forecast-epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-patients", type=int, default=0)
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed); random.seed(args.seed)
    cfg = yaml.safe_load(Path(args.config).read_text())
    protocol = load_protocol(args.protocol)
    eligible = lambda patient: len(list(Path(args.mask_root).glob(
        f"{patient}/week-*/DeepBraTumIA-segmentation/atlas/segmentation/seg_mask.nii.gz"))) >= 2
    pool = [p for p in sorted(protocol["encoder_train"]) if eligible(p)]
    internal_dev, train_ids = pool[::5], [p for p in pool if p not in set(pool[::5])]
    unseen = [p for p in sorted(protocol["encoder_unseen"]) if eligible(p)]
    if args.max_patients:
        train_ids = train_ids[:args.max_patients]; internal_dev = internal_dev[:1]
        unseen = unseen[:1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}; representation train={len(train_ids)} internal_dev={len(internal_dev)} locked={len(unseen)}")
    world = JEPAWorldModel(cfg); _, report = load_model(world, args.champion)
    print(f"champion loaded={report.loaded} missing={len(report.missing)}")
    world.to(device)
    forecaster = LesionTransitionForecaster().to(device)
    lora = configure_lora(world.backbone)
    anatomy_params = lora + list(forecaster.observation.parameters()) + list(forecaster.anatomy.parameters())
    optimizer = torch.optim.AdamW(anatomy_params, lr=args.lr, weight_decay=0.01)
    train_loader = batches(datasets(cfg, train_ids), cfg, True)
    for epoch in range(1, args.anatomy_epochs + 1):
        world.backbone.train(); forecaster.train(); optimizer.zero_grad(); losses = []
        step = 0
        for batch in train_loader:
            pid = batch["patient_id"][0]
            all_visits = train_loader.dataset[train_loader.dataset.patients.index(pid)]["visits"]
            indices = mask_indices(args.mask_root, pid, all_visits)
            visits = [all_visits[i] for i in indices]
            if not visits:
                continue
            masks, volumes = load_masks(args.mask_root, pid, visits)
            for j, t in enumerate(indices):
                present = batch["mri_mask"][0, t]
                if not bool(present.any()): continue
                feature = visit_features(world.backbone, batch["mri"][0, t, :, 0].to(device),
                                         present.to(device), masks[j].to(device))
                prediction = forecaster.decode_anatomy(
                    forecaster.encode_observation(feature[None]))[0]
                loss = F.smooth_l1_loss(prediction, volumes[j].to(device)) / args.accum
                loss.backward(); losses.append(float(loss) * args.accum); step += 1
                if step % args.accum == 0:
                    torch.nn.utils.clip_grad_norm_(anatomy_params, 1.0)
                    optimizer.step(); optimizer.zero_grad()
        if step % args.accum: optimizer.step(); optimizer.zero_grad()
        print(f"anatomy epoch {epoch}: loss={np.mean(losses):.5f}", flush=True)
    for parameter in world.backbone.parameters(): parameter.requires_grad = False
    all_ids = sorted(set(train_ids + internal_dev + unseen))
    cache = build_feature_cache(world.backbone, batches(datasets(cfg, all_ids), cfg, False),
                                args.mask_root, device)
    train_cache = {p: cache[p] for p in train_ids if p in cache}
    dev_cache = {p: cache[p] for p in internal_dev if p in cache}
    locked_cache = {p: cache[p] for p in unseen if p in cache}
    if not train_cache or not dev_cache or not locked_cache:
        raise ValueError("mask-bearing train/dev/locked trajectories are required")
    forecast_params = list(forecaster.parameters())
    optimizer = torch.optim.AdamW(forecast_params, lr=1e-3, weight_decay=0.01)
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(1, args.forecast_epochs + 1):
        forecaster.train(); order = list(train_cache); random.shuffle(order); losses = []
        for pid in order:
            row = train_cache[pid]; source = row["volumes"].to(device)
            out = forecaster(row["features"].to(device), row["gaps"].to(device), source)
            anatomy = F.smooth_l1_loss(out["current_anatomy"], source)
            forecast = F.smooth_l1_loss(out["prediction"], source[1:])
            loss = forecast + 0.25 * anatomy
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss))
        dev, persist = patient_mae(dev_cache, forecaster, device)
        print(f"forecast epoch {epoch}: loss={np.mean(losses):.5f} dev_mae={dev:.5f} persistence={persist:.5f} relative={dev/persist:.4f}", flush=True)
        if dev < best - 1e-4:
            best, bad = dev, 0
            best_state = {k: v.detach().cpu().clone() for k, v in forecaster.state_dict().items()}
        else:
            bad += 1
            if bad >= 20: break
    forecaster.load_state_dict(best_state)
    locked, locked_persist = patient_mae(locked_cache, forecaster, device)
    result = {"internal_dev_mae": best, "locked_unseen_mae": locked,
              "locked_unseen_persistence": locked_persist,
              "locked_unseen_relative": locked / locked_persist,
              "representation_train": train_ids, "internal_dev": internal_dev,
              "locked_unseen": unseen}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"checkpoint_format_version": 2, "backbone": world.backbone.state_dict(),
                "forecaster": best_state, "result": result, "args": vars(args)}, args.out)
    Path(str(args.out) + ".json").write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result, sort_keys=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
