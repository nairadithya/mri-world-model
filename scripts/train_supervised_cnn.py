"""Train the supervised 3D CNN comparator on LUMIERE (locked protocol).

Pair framing: (visit_t, visit_{t+1}) [4 modalities each -> 8 channels] predicts
RANO_{t+1}. Patient-level chunks, class-weighted CE, augmentation, early stop
on the 13 `dev`; optional feature encoding for the cross-site comparison.

Usage:
    python scripts/train_supervised_cnn.py --train --epochs 20
    python scripts/train_supervised_cnn.py --encode --ckpt checkpoints/cnn/best.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data.collate import make_collate
from src.data.dataset import LUMIEREDataset
from src.data.eval_protocol import load_protocol
from src.model.cnn import SupervisedCNN, pair_channels, pair_present
from src.harness.eval.metrics import macro_f1  # noqa: F401

RANO_ACTION_TO_FLAT = {3: 0, 2: 1, 5: 2, 4: 3}
CLEAN_ACTIONS = (2, 3, 4, 5)
SAILOR_ROOT = "data/sailor/sailor_ebrains_pseud/derivatives/mni2009c-n-s"
SAILOR_PROBE_MAP = {1: 0, 2: 1, 3: 2, 5: 3}


def flat_label(action: int):
    return RANO_ACTION_TO_FLAT.get(action)


def _build(cfg, patients):
    return LUMIEREDataset(patients=patients, meta_dir=cfg["data"]["meta_dir"],
                          processed_root=cfg["data"]["root"],
                          raw_root=cfg["data"].get("raw_root"),
                          modalities=tuple(cfg["data"].get("modalities",
                                                           ["CT1", "T1", "T2", "FLAIR"])),
                          min_visits=cfg["data"].get("min_visits", 2))


def _patient_pairs(batch):
    """Yield (x (1,2M,D,H,W), y) for valid labelled forecast pairs."""
    mri, mri_mask, actions = batch["mri"], batch["mri_mask"], batch["actions"]
    T = mri.shape[1]
    out = []
    for t in range(T - 1):
        if not bool(pair_present(mri_mask, t)[0]):
            continue
        lab = flat_label(int(actions[0, t + 1]))
        if lab is None:
            continue
        out.append((pair_channels(mri, t), lab))
    return out


@torch.no_grad()
def evaluate(model, pids, cfg, device, collate):
    model.eval()
    per = {}
    for pid in pids:
        ds = _build(cfg, [pid])
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=1, collate_fn=collate)
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pairs = _patient_pairs(batch)
            if not pairs:
                continue
            x = torch.cat([p[0] for p in pairs]).to(device)
            logits = model(x)
            per[pid] = (logits.argmax(1).cpu(),
                        torch.tensor([p[1] for p in pairs]))
    per = {p: v for p, v in per.items() if len(v[1])}
    if not per:
        return 0.0, per
    return macro_f1(torch.cat([v[0] for v in per.values()]),
                    torch.cat([v[1] for v in per.values()])), per


def train(args, cfg, device):
    proto = load_protocol(args.protocol)
    train_pids, dev_pids = list(proto["encoder_train"]), list(proto["dev"])
    if args.max_patients:
        train_pids, dev_pids = train_pids[:args.max_patients], dev_pids[:args.max_patients]
    size = tuple(cfg["preprocessing"].get("target_size", [96, 96, 96]))
    train_ds = _build(cfg, train_pids)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0,
                              collate_fn=make_collate(size, augment=args.augment))
    val_collate = make_collate(size, augment=False)

    counts = torch.zeros(4)
    for i in range(len(train_ds)):
        s = train_ds[i]
        for a in s["actions"][1:]:
            c = flat_label(int(a))
            if c is not None:
                counts[c] += 1
    counts = counts.clamp_min(1)
    cw = (counts.sum() / (4 * counts)).to(device)
    print(f"train class counts {counts.tolist()} weights {[round(x,2) for x in cw.tolist()]}")

    model = SupervisedCNN(n_input_channels=8, num_classes=4).to(device)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"trainable {len(train_pids)} patients")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_dev, best_ep, bad = -1.0, -1, 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pairs = _patient_pairs(batch)
            if not pairs:
                continue
            opt.zero_grad(set_to_none=True)
            nch = max(1, (len(pairs) + args.pair_batch - 1) // args.pair_batch)
            for s in range(0, len(pairs), args.pair_batch):
                chunk = pairs[s:s + args.pair_batch]
                x = torch.cat([c[0] for c in chunk]).to(device)
                y = torch.tensor([c[1] for c in chunk], device=device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = F.cross_entropy(model(x), y, weight=cw) / nch
                scaler.scale(loss).backward()
                tot += float(loss.detach()) * nch
                nb += len(chunk)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        dev_f1, _ = evaluate(model, dev_pids, cfg, device, val_collate)
        print(f"epoch {ep}: loss={tot/max(nb,1):.4f} dev_macroF1={dev_f1:.4f}", flush=True)
        if dev_f1 > best_dev:
            best_dev, best_ep, bad = dev_f1, ep, 0
            torch.save({"epoch": ep, "model": model.state_dict(),
                        "dev_macro_f1": dev_f1, "args": vars(args)},
                       os.path.join(args.checkpoint_dir, "best.pt"))
        else:
            bad += 1
            if bad >= args.patience and ep >= args.min_epochs:
                print(f"early stop at epoch {ep}")
                break

    print(f"best dev macro-F1 {best_dev:.4f} @ epoch {best_ep}")
    if args.eval_final:
        best = torch.load(os.path.join(args.checkpoint_dir, "best.pt"),
                          map_location=device, weights_only=False)
        model.load_state_dict(best["model"])
        f1, per = evaluate(model, proto["final"], cfg, device, val_collate)
        print(f"FINAL (reserved): {len(per)} patients macro-F1 {f1:.4f}")
    prov = {"config_sha1": hashlib.sha1(open(args.config, "rb").read()).hexdigest(),
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                               text=True).strip(),
            "date": time.strftime("%Y-%m-%d"), "args": vars(args),
            "best_dev_macro_f1": best_dev, "best_epoch": best_ep}
    with open(os.path.join(args.checkpoint_dir, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)


@torch.no_grad()
def encode_features(args, cfg, device):
    """Cache CNN pair features for LUMIERE (locked splits) and SAILOR."""
    model = SupervisedCNN(n_input_channels=8, num_classes=4).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    size = tuple(cfg["preprocessing"].get("target_size", [96, 96, 96]))
    collate = make_collate(size, augment=False)
    proto = load_protocol(args.protocol)
    out = {"lum": {}, "sailor": {}, "provenance": {"ckpt": os.path.abspath(args.ckpt)}}

    def _run(ds, id_key):
        loader = DataLoader(ds, batch_size=1, collate_fn=collate)
        for batch in loader:
            pid = batch["patient_id"][0]
            pairs = _patient_pairs(batch)
            if not pairs:
                continue
            x = torch.cat([p[0] for p in pairs]).to(device)
            f = model.features(x).cpu()
            out[id_key][pid] = {"features": f,
                                "labels": torch.tensor([p[1] for p in pairs])}

    if args.encode_scope in ("lum", "all"):
        for split in ("encoder_train", "dev", "final"):
            _run(_build(cfg, proto[split]), "lum")
    if args.encode_scope in ("sailor", "all"):
        if os.path.isdir(SAILOR_ROOT):
            from src.data.sailor import SAILORDataset
            _run(SAILORDataset(SAILOR_ROOT), "sailor")
        else:
            print(f"skip SAILOR encode (root not present: {SAILOR_ROOT}); "
                  f"run locally on the controlled-access data")
    torch.save(out, args.feature_cache)
    print(f"wrote {args.feature_cache}: lum {len(out['lum'])} / sailor {len(out['sailor'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--protocol", default="info/eval_folds.json")
    ap.add_argument("--checkpoint-dir", default="checkpoints/cnn")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--pair-batch", type=int, default=4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--min-epochs", type=int, default=5)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--max-patients", type=int, default=0)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval-final", action="store_true")
    ap.add_argument("--encode", action="store_true")
    ap.add_argument("--encode-scope", choices=["lum", "sailor", "all"], default="all")
    ap.add_argument("--ckpt", default="checkpoints/cnn/best.pt")
    ap.add_argument("--feature-cache", default="checkpoints/cnn_features.pt")
    args = ap.parse_args()
    random.seed(0)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if args.train:
        train(args, cfg, device)
    if args.encode:
        encode_features(args, cfg, device)


if __name__ == "__main__":
    main()
