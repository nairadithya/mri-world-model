"""End-to-end training entry point.

Usage:
    python scripts/run_train.py --config config/default.yaml [--epochs 10] [--batch-size 2]
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.collate import make_collate
from src.data.dataset import LUMIEREDataset
from src.data.splits import patient_splits
from src.model.jepa_model import JEPAWorldModel
from src.train.trainer import evaluate, train


def find_meta(meta_dir: str, prefix: str) -> str:
    for f in os.listdir(meta_dir):
        if f.startswith(prefix):
            return os.path.join(meta_dir, f)
    raise FileNotFoundError(f"No {prefix}* in {meta_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
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
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.random_init:
        cfg["model"]["brainiac"]["checkpoint"] = None
    if args.epochs is not None:
        cfg["training"]["max_epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.warmup_epochs is not None:
        cfg["training"]["warmup_epochs"] = args.warmup_epochs
    if args.aux_lambda is not None:
        cfg.setdefault("aux", {})["lambda"] = args.aux_lambda
    if args.horizon:
        cfg["model"].setdefault("predictor", {}).setdefault("horizon", {})["enabled"] = True
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
    collate = make_collate(size)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=2, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=2, collate_fn=collate)
    print(f"train/val patients: {len(train_ds)}/{len(val_ds)}")

    model = JEPAWorldModel(cfg)
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(f"resumed weights from {args.resume_from} "
              f"(epoch {ckpt.get('epoch', '?')}, fresh optimizer/schedule)")
        if missing:
            print(f"  randomly initialized (absent in ckpt): {sorted(missing)}")
    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"params: {n_trainable/1e6:.1f}M trainable / {n_total/1e6:.1f}M total")

    stats = train(model, train_loader, val_loader, cfg, device)
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


if __name__ == "__main__":
    main()
