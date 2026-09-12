"""JEPA representation training (native harness method).

Ported verbatim from ``scripts/run_train.py`` (which is now a thin entry
shim). Stdout lines are a contract parsed by Kaggle verdict cells and
``info/`` records, so the printed strings must not change.
"""
from __future__ import annotations

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader

from ..config import find_meta
from .base import Method, register_method
from ...data.collate import make_collate
from ...data.dataset import LUMIEREDataset
from ...data.splits import patient_splits
from ...model.jepa_model import JEPAWorldModel
from ...train.trainer import evaluate, train


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--accum-steps", type=int, default=None,
                    help="gradient accumulation micros per optimizer step "
                         "(large-batch dynamics at batch-1 memory; "
                         "pair-weighted by default, see training config).")
    ap.add_argument("--no-bucket", action="store_true",
                    help="disable length-bucketed train batching (legacy "
                         "uniform shuffle; bucketing avoids OOM pairings "
                         "and padding waste).")
    ap.add_argument("--lr", type=float, default=None,
                    help="peak LR (default from config; leg 2+ typically lower, e.g. 2e-5).")
    ap.add_argument("--warmup-epochs", type=int, default=None,
                    help="linear warmup epochs (default from config).")
    ap.add_argument("--aux-lambda", type=float, default=None,
                    help="RANO aux weight (D25 joint training; 0 = JEPA only).")
    ap.add_argument("--horizon", action="store_true",
                    help="multi-horizon JEPA: state_t predicts every future "
                         "z_{t+n} via the gap-conditioned head, 1/n-weighted "
                         "loss (probe-gated; see scripts/horizon_probe.py).")
    ap.add_argument("--dynamics", action="store_true",
                    help="time-continuous dynamics: velocity field integrated "
                         "across true gaps (replaces 1-step and horizon "
                         "losses; mutually exclusive with --horizon).")
    ap.add_argument("--augment", action="store_true",
                    help="training-only augmentation; JEPA target sees the "
                         "clean view (A32).")
    ap.add_argument("--surgery-window", action="store_true",
                    help="drop JEPA pairs touching pre-/post-op or <3-months "
                         "post-surgery visits (A32/R7).")
    ap.add_argument("--transition-weighting", action="store_true",
                    help="inverse-prevalence weighting of 1-step JEPA pairs "
                         "by transition class (A32/R13).")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--random-init", action="store_true",
                    help="Skip BRAINIAC checkpoint; random init (dev/smoke-test only).")
    ap.add_argument("--patients", nargs="*", default=None,
                    help="restrict to these patient IDs (pilot runs); "
                         "splits 80/20 train/val within the list.")
    ap.add_argument("--resume-from", default=None, metavar="CKPT",
                    help="resume weights from a previous run's best.pt/last.pt "
                         "(fresh optimizer + LR schedule; for splitting long runs "
                         "across Kaggle sessions).")
    ap.add_argument("--resume-opt", action="store_true",
                    help="with --resume-from: also load the checkpoint's optimizer "
                         "state (tests whether fresh momentum ejects narrow basins, "
                         "D22/R11). Same model code required; mismatched groups "
                         "fail LOUD (no silent fresh fallback).")
    ap.add_argument("--checkpoint-dir", default=None, metavar="DIR",
                    help="override training.checkpoint_dir (multi-leg sessions "
                         "need distinct dirs per leg — a resume leg's best.pt "
                         "overwrites the staged champion, D22).")
    return ap


def main(argv=None) -> None:
    args = _parser().parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.random_init:
        cfg["model"]["brainiac"]["checkpoint"] = None
    if args.epochs is not None:
        cfg["training"]["max_epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.accum_steps is not None:
        cfg["training"]["accumulation_steps"] = args.accum_steps
    if args.no_bucket:
        cfg["training"]["bucket_batches"] = False
    if args.checkpoint_dir is not None:
        cfg["training"]["checkpoint_dir"] = args.checkpoint_dir
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.warmup_epochs is not None:
        cfg["training"]["warmup_epochs"] = args.warmup_epochs
    if args.aux_lambda is not None:
        cfg.setdefault("aux", {})["lambda"] = args.aux_lambda
    if args.horizon:
        cfg["model"].setdefault("predictor", {}).setdefault("horizon", {})["enabled"] = True
    if args.dynamics:
        if args.horizon:
            raise ValueError("--dynamics replaces the horizon loss; "
                             "pass only one of --dynamics/--horizon.")
        cfg["model"].setdefault("dynamics", {})["enabled"] = True
    if args.augment:
        cfg["data"]["augment_train"] = True
    if args.surgery_window:
        cfg["training"]["surgery_window"] = True
    if args.transition_weighting:
        cfg["training"]["transition_weighting"] = True
    if args.no_wandb:
        cfg["training"]["log_wandb"] = False

    torch.manual_seed(cfg["data"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    meta_dir = cfg["data"]["meta_dir"]
    if args.patients:
        sel = sorted(args.patients)
        n_train = max(1, int(len(sel) * 0.8))
        splits = {"train": sel[:n_train], "val": sel[n_train:] or sel[:1], "test": []}
    else:
        demo_csv = find_meta(meta_dir, "demographics")
        splits = patient_splits(
            demo_csv,
            train_frac=cfg["data"].get("train_split", 0.7),
            val_frac=cfg["data"].get("val_split", 0.15),
            seed=cfg["data"].get("seed", 42),
        )
    print({k: len(v) for k, v in splits.items()})

    size = tuple(cfg["preprocessing"].get("target_size", [96, 96, 96]))
    store_half = bool(cfg["data"].get("store_half", False))
    collate = make_collate(size, dtype=torch.float16 if store_half else torch.float32)
    train_collate = make_collate(
        size, dtype=torch.float16 if store_half else torch.float32,
        augment=bool(cfg["data"].get("augment_train", False)))
    if store_half:
        print("collate: storing fp16 inputs (backbone casts to float per chunk)")
    common = dict(
        meta_dir=meta_dir,
        processed_root=cfg["data"]["root"],
        raw_root=cfg["data"].get("raw_root"),
        modalities=tuple(cfg["data"].get("modalities", ["CT1", "T1", "T2", "FLAIR"])),
        min_visits=cfg["data"].get("min_visits", 2),
    )
    train_ds = LUMIEREDataset(patients=splits["train"], **common)
    val_ds = LUMIEREDataset(patients=splits["val"], **common)
    bs = cfg["training"].get("batch_size", 4)
    if cfg["training"].get("bucket_batches", True):
        from ...data.sampler import LengthBucketSampler
        lengths = [len(train_ds.visits[p]) for p in train_ds.patients]
        batch_sampler = LengthBucketSampler(
            lengths, bs, shuffle=True, seed=cfg["data"].get("seed", 42))
        train_loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                                  num_workers=2, collate_fn=train_collate)
        print(f"train batching: length-bucketed (bs={bs}, "
              f"maxlen={max(lengths, default=0)})")
    else:
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                  num_workers=2, collate_fn=train_collate)
        print("train batching: legacy uniform shuffle")
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=2, collate_fn=collate)
    print(f"train/val patients: {len(train_ds)}/{len(val_ds)}")
    if cfg["training"].get("transition_weighting"):
        from ...data.transitions import (CLASS_NAMES, inverse_prevalence_weights,
                                         transition_class)
        counts = [0] * 4
        for i in range(len(train_ds)):
            a = train_ds[i]["actions"].tolist()
            for x, y in zip(a[:-1], a[1:]):
                counts[transition_class(x, y)] += 1
        w = inverse_prevalence_weights(counts)
        cfg["training"]["transition_weights"] = w
        print("transition counts", dict(zip(CLASS_NAMES, counts)),
              "weights", [round(v, 2) for v in w])

    model = JEPAWorldModel(cfg)
    resume_opt_state = None
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(f"resumed weights from {args.resume_from} "
              f"(epoch {ckpt.get('epoch', '?')}, "
              f"{'loaded' if args.resume_opt else 'fresh'} optimizer/schedule)")
        if missing:
            print(f"  randomly initialized (absent in ckpt): {sorted(missing)}")
        if args.resume_opt:
            if "opt" not in ckpt or not ckpt["opt"]:
                raise ValueError(
                    f"--resume-opt given but {args.resume_from} holds no "
                    f"optimizer state (failing LOUD: a silent fresh fallback "
                    f"would corrupt the fresh-vs-loaded comparison).")
            resume_opt_state = ckpt["opt"]
            print(f"  + optimizer state WILL load "
                  f"({len(resume_opt_state.get('state', {}))} tensors)")
    elif args.resume_opt:
        raise ValueError("--resume-opt needs --resume-from.")
    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"params: {n_trainable/1e6:.1f}M trainable / {n_total/1e6:.1f}M total")

    stats = train(model, train_loader, val_loader, cfg, device,
                  resume_opt_state=resume_opt_state)
    print(f"done. best val loss: {stats['best_val_loss']:.4f}")

    # Test scorecard (PROPOSAL §6/F): reload best checkpoint, evaluate once.
    if splits["test"]:
        test_ds = LUMIEREDataset(patients=splits["test"], **common)
        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                                 num_workers=2, collate_fn=collate)
        best_path = os.path.join(cfg["training"].get("checkpoint_dir", "checkpoints/"), "best.pt")
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device)["model"])
            test_stats = evaluate(model, test_loader, device)
            print(f"test: loss={test_stats['loss']:.4f} "
                  f"std={test_stats['target_std']:.4f} rank={test_stats['target_eff_rank']:.1f}")
        else:
            print("no best.pt found; skipping test eval")
    else:
        print("no test split (pilot subset); skipping test eval")


@register_method("jepa", role="ssl",
                 produces=("vision", "fused", "states", "ema_z", "z"))
class JepaMethod(Method):
    """Registered handle for the native JEPA trainer (CLI + discovery)."""

    entry = staticmethod(main)
