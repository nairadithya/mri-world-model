"""2D axial-slice supervised CNN comparator (Matoso family).

Slice-level training gives ~20k samples/epoch from 65 patients (vs ~200
visit-pairs for the 3D variant), so the comparator can actually converge on
this small cohort. Pair-level predictions/features are the mean over the
pair's slices, matching the JEPA per-pair granularity for `cross_site_adapt`.

Usage:
    python scripts/train_supervised_cnn2d.py --train --eval-final --encode --encode-scope lum
    python scripts/train_supervised_cnn2d.py --encode --encode-scope sailor
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
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.collate import make_collate
from src.data.dataset import LUMIEREDataset
from src.data.eval_protocol import load_protocol
from src.model.cnn import (SupervisedCNN2D, pair_present, pair_slices,
                           roi_pair_slices)

RANO_ACTION_TO_FLAT = {3: 0, 2: 1, 5: 2, 4: 3}
SAILOR_ROOT = "data/sailor/sailor_ebrains_pseud/derivatives/mni2009c-n-s"
LUM_MASK_ROOT = "data/autoseg/extracted/Imaging"
_MASK_CACHE: dict = {}


def _load_mask96(path: str):
    import numpy as np
    import nibabel as nib

    if not os.path.exists(path):
        return None
    d = np.asanyarray(nib.load(path).dataobj).astype("float32")
    t = torch.from_numpy(d)[None, None]
    t96 = F.interpolate(t, size=(96, 96, 96), mode="nearest")[0, 0]
    return t96 > 0


def lum_mask(pid: str, visit: str):
    key = ("lum", pid, visit)
    if key not in _MASK_CACHE:
        _MASK_CACHE[key] = _load_mask96(os.path.join(
            LUM_MASK_ROOT, pid, visit,
            "DeepBraTumIA-segmentation/atlas/segmentation/seg_mask.nii.gz"))
    return _MASK_CACHE[key]


def sailor_mask(pid: str, visit: str):
    key = ("sailor", pid, visit)
    if key not in _MASK_CACHE:
        _MASK_CACHE[key] = _load_mask96(os.path.join(
            SAILOR_ROOT, pid, visit, "Segmentation-ONCO.nii.gz"))
    return _MASK_CACHE[key]


def macro_f1(pred, y, n_cls=4):
    f1s = []
    for k in range(n_cls):
        tp = int(((pred == k) & (y == k)).sum())
        fp = int(((pred == k) & (y != k)).sum())
        fn = int(((pred != k) & (y == k)).sum())
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1s.append(2 * p * r / max(1e-9, p + r))
    return sum(f1s) / n_cls


def _build(cfg, patients):
    return LUMIEREDataset(patients=patients, meta_dir=cfg["data"]["meta_dir"],
                          processed_root=cfg["data"]["root"],
                          raw_root=cfg["data"].get("raw_root"),
                          modalities=tuple(cfg["data"].get("modalities",
                                                           ["CT1", "T1", "T2", "FLAIR"])),
                          min_visits=cfg["data"].get("min_visits", 2))


def patient_examples(batch, mask_fn=None, visits=None, crop=64, min_fg=64):
    """List of ((S, 2M, H, W) slices, label) for valid labelled pairs.

    With ``mask_fn`` the pair is cropped to a tumor-centred ROI (ROI slices);
    otherwise whole-brain axial slices are returned.
    """
    mri, mri_mask, actions = batch["mri"], batch["mri_mask"], batch["actions"]
    T = mri.shape[1]
    ex = []
    for t in range(T - 1):
        if not bool(pair_present(mri_mask, t)[0]):
            continue
        lab = RANO_ACTION_TO_FLAT.get(int(actions[0, t + 1]))
        if lab is None:
            continue
        if mask_fn is not None and visits is not None:
            m0 = mask_fn(batch["patient_id"][0], visits[t])
            m1 = mask_fn(batch["patient_id"][0], visits[t + 1])
            if m0 is None and m1 is None:
                continue
            m = m0 if m1 is None else (m1 if m0 is None else (m0 | m1))
            sl = roi_pair_slices(mri[0, t, :, 0], mri[0, t + 1, :, 0], m, crop=crop)
        else:
            sl = pair_slices(mri, t, min_fg=min_fg)
        if sl.numel():
            ex.append((sl, lab))
    return ex


def _chunks(x, n):
    for i in range(0, x.shape[0], n):
        yield x[i:i + n]


@torch.no_grad()
def eval_pairs(model, pids, cfg, device, collate, mask_fn=None, crop=64,
               chunk=64):
    model.eval()
    per = {}
    for pid in pids:
        ds = _build(cfg, [pid])
        if len(ds) == 0:
            continue
        for batch in DataLoader(ds, batch_size=1, collate_fn=collate):
            ex = patient_examples(batch, mask_fn, ds.visits[pid], crop)
            if not ex:
                continue
            pl, yl = [], []
            for sl, lab in ex:
                logits = torch.cat([model(sl.to(device)[s:s + chunk]).mean(0, keepdim=True)
                                    for s in range(0, sl.shape[0], chunk)])
                pl.append(logits.mean(0).cpu())
                yl.append(lab)
            per[pid] = (torch.stack(pl).argmax(1), torch.tensor(yl))
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
    val_collate = make_collate(size, augment=False)
    train_collate = make_collate(size, augment=args.augment)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0,
                              collate_fn=train_collate)

    counts = torch.zeros(4)
    for i in range(len(train_ds)):
        s = train_ds[i]
        for a in s["actions"][1:]:
            c = RANO_ACTION_TO_FLAT.get(int(a))
            if c is not None:
                counts[c] += 1
    counts = counts.clamp_min(1)
    cw = (counts.sum() / (4 * counts)).to(device)
    print(f"train class counts {counts.tolist()} weights {[round(x,2) for x in cw.tolist()]}")

    train_mask_fn = lum_mask if args.roi else None
    model = SupervisedCNN2D(n_input_channels=8, num_classes=4).to(device)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.1f}M (2D)")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_dev, best_ep, bad = -1.0, -1, 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        buf = []          # (x, y) slices pending a step
        tot, nb = 0.0, 0

        def step():
            nonlocal buf, tot, nb
            x = torch.stack([b[0] for b in buf]).to(device)
            y = torch.tensor([b[1] for b in buf], device=device)
            if args.augment:  # random horizontal flip per slice batch
                fl = torch.rand(len(buf)) < 0.5
                if fl.any():
                    x[fl] = torch.flip(x[fl], dims=[3])
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = F.cross_entropy(model(x), y, weight=cw)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            tot += float(loss.detach())
            nb += 1
            buf = []

        for batch in train_loader:
            ex = patient_examples(batch, train_mask_fn,
                                  train_ds.visits[batch["patient_id"][0]], args.crop)
            for sl, lab in ex:
                for z in range(sl.shape[0]):
                    buf.append((sl[z], lab))
                    if len(buf) >= args.slice_batch:
                        step()
        if buf:
            step()
        sched.step()
        dev_f1, _ = eval_pairs(model, dev_pids, cfg, device, val_collate,
                                mask_fn=train_mask_fn, crop=args.crop)
        print(f"epoch {ep}: loss={tot/max(nb,1):.4f} steps={nb} dev_macroF1={dev_f1:.4f}",
              flush=True)
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
        f1, per = eval_pairs(model, proto["final"], cfg, device, val_collate,
                              mask_fn=train_mask_fn, crop=args.crop)
        print(f"FINAL (reserved): {len(per)} patients macro-F1 {f1:.4f}")
    prov = {"config_sha1": hashlib.sha1(open(args.config, "rb").read()).hexdigest(),
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                               text=True).strip(),
            "date": time.strftime("%Y-%m-%d"), "args": vars(args),
            "best_dev_macro_f1": best_dev, "best_epoch": best_ep}
    with open(os.path.join(args.checkpoint_dir, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)


@torch.no_grad()
def encode_features(args, cfg, device, chunk=64):
    model = SupervisedCNN2D(n_input_channels=8, num_classes=4).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    size = tuple(cfg["preprocessing"].get("target_size", [96, 96, 96]))
    collate = make_collate(size, augment=False)
    proto = load_protocol(args.protocol)
    out = {"lum": {}, "sailor": {}, "provenance": {"ckpt": os.path.abspath(args.ckpt)}}

    def _run(ds, key):
        mask_fn = None
        if args.roi:
            mask_fn = lum_mask if key == "lum" else sailor_mask
        for batch in DataLoader(ds, batch_size=1, collate_fn=collate):
            pid = batch["patient_id"][0]
            ex = patient_examples(batch, mask_fn, ds.visits[pid], args.crop)
            if not ex:
                continue
            feats, labs = [], []
            for sl, lab in ex:
                f = torch.cat([model.features(sl.to(device)[s:s + chunk])
                               for s in range(0, sl.shape[0], chunk)])
                feats.append(f.mean(0).cpu())
                labs.append(lab)
            out[key][pid] = {"features": torch.stack(feats),
                             "labels": torch.tensor(labs)}

    if args.encode_scope in ("lum", "all"):
        for split in ("encoder_train", "dev", "final"):
            _run(_build(cfg, proto[split]), "lum")
    if args.encode_scope in ("sailor", "all"):
        if os.path.isdir(SAILOR_ROOT):
            from src.data.sailor import SAILORDataset
            _run(SAILORDataset(SAILOR_ROOT), "sailor")
        else:
            print(f"skip SAILOR (root missing: {SAILOR_ROOT})")
    torch.save(out, args.feature_cache)
    print(f"wrote {args.feature_cache}: lum {len(out['lum'])} / sailor {len(out['sailor'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--protocol", default="info/eval_folds.json")
    ap.add_argument("--checkpoint-dir", default="checkpoints/cnn2d")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--slice-batch", type=int, default=32)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--min-epochs", type=int, default=5)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--roi", action="store_true",
                    help="tumor-centred ROI crops (mask-guided)")
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--max-patients", type=int, default=0)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval-final", action="store_true")
    ap.add_argument("--encode", action="store_true")
    ap.add_argument("--encode-scope", choices=["lum", "sailor", "all"], default="all")
    ap.add_argument("--ckpt", default="checkpoints/cnn2d/best.pt")
    ap.add_argument("--feature-cache", default="checkpoints/cnn2d_features.pt")
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
