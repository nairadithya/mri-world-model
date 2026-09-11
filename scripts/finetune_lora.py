"""Supervised LoRA finetune of the vision encoder (Step 3 / SOTA recipe).

The one untried part of the SOTA recipe: optimize the *image* representation
for the RANO label rather than only the frozen readout. Trainable = LoRA
adapters + projector + fusion + the 4-class RANO head; the temporal
transformer is frozen (gradients still flow through it to the vision tower).
Loss = class-weighted CE on the forecast framing (state_t -> RANO_{t+1}),
optionally + a JEPA regularizer that distills the online next-latent toward
the frozen champion target (lambda_jepa > 0).

Splits follow the locked protocol: train on the 65 `encoder_train`
patients, early-stop on the 13 `dev`; the 13 `final` are touched once, at
the end, and never during training. Selection metric = dev macro-F1.

Usage:
    # CPU structural smoke
    python scripts/finetune_lora.py --max-patients 2 --epochs 1 --jepa-lambda 0
    # real GPU run
    python scripts/finetune_lora.py --epochs 20 --lr 2e-5 --accum-steps 8 --augment
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
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
from src.model.heads import ACTION_TO_FLAT, CLEAN_ACTIONS
from src.model.jepa import jepa_loss
from src.model.jepa_model import JEPAWorldModel


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


def _ci(samples, lo=2.5, hi=97.5):
    s = sorted(samples)
    n = len(s)
    return s[int(lo / 100 * n)], s[min(n - 1, int(hi / 100 * n))]


def bootstrap(per, boot=10000, seed=42):
    pids = sorted(per)
    rng = random.Random(seed)
    vals = []
    for _ in range(boot):
        samp = [rng.choice(pids) for _ in pids]
        vals.append(macro_f1(torch.cat([per[p][0] for p in samp]),
                             torch.cat([per[p][1] for p in samp])))
    return _ci(vals)


def configure_trainable(model, train_fusion=True, train_head=True):
    """Freeze all, then enable LoRA adapters + projector (+fusion, +head)."""
    for p in model.parameters():
        p.requires_grad = False
    n_lora = 0
    for name, p in model.backbone.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
            n_lora += p.numel()
    for p in model.projector.parameters():
        p.requires_grad = True
    if train_fusion:
        for p in model.fusion.parameters():
            p.requires_grad = True
    if train_head:
        for p in model.rano_heads.flat.parameters():
            p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen = [p for p in model.parameters() if not p.requires_grad]
    print(f"trainable {sum(p.numel() for p in trainable)/1e6:.3f}M "
          f"(LoRA {n_lora/1e6:.3f}M) / frozen {sum(p.numel() for p in frozen)/1e6:.1f}M")
    head_ids = {id(p) for p in model.rano_heads.flat.parameters()}
    rep = [p for p in trainable if id(p) not in head_ids]
    head = [p for p in trainable if id(p) in head_ids]
    return rep, head


def forward_states(model, batch):
    mri, mri_mask = batch["mri"], batch["mri_mask"]
    visit_mask = batch["visit_mask"]
    B, T = visit_mask.shape
    v = model.encode_visits(mri, mri_mask)
    c = model.clinical(batch["clinical"])
    tokens = model.fusion(v, c.unsqueeze(1).expand(-1, T, -1))
    states, valid = model.temporal.forward_prefixes(
        tokens, batch["time_deltas"], visit_mask)
    has_img = mri_mask.any(dim=2)
    valid = valid & has_img[:, :-1] & has_img[:, 1:]
    return states, valid


def flat_targets(batch, valid):
    a_next = batch["actions"][:, 1:]
    clean = torch.zeros_like(a_next, dtype=torch.bool)
    for a in CLEAN_ACTIONS:
        clean |= a_next == a
    use = clean & valid
    flat = torch.full_like(a_next, -1)
    for a, c in ACTION_TO_FLAT.items():
        flat[a_next == a] = c
    return flat, use


def sup_loss(model, states, flat, use, cw):
    if not use.any():
        return None
    return F.cross_entropy(model.rano_heads.flat(states[use]), flat[use], weight=cw)


@torch.no_grad()
def eval_dev(model, loader, device):
    model.eval()
    per = {}
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        states, valid = forward_states(model, batch)
        flat, use = flat_targets(batch, valid)
        pred = model.rano_heads.flat(states).argmax(-1)
        pid = batch["patient_id"][0]
        per[pid] = (pred[0][use[0]].cpu(), flat[0][use[0]].cpu())
    per = {p: v for p, v in per.items() if len(v[1]) > 0}
    if not per:
        return 0.0, per
    f1 = macro_f1(torch.cat([v[0] for v in per.values()]),
                  torch.cat([v[1] for v in per.values()]))
    return f1, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--champion", default="checkpoints/champion_0.0081.pt")
    ap.add_argument("--protocol", default="info/eval_folds.json")
    ap.add_argument("--checkpoint-dir", default="checkpoints/lora")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-4,
                    help="representation LR (LoRA + projector + fusion)")
    ap.add_argument("--head-lr", type=float, default=1e-2,
                    help="RANO head LR (random init; needs the fit_linear rate)")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--accum-steps", type=int, default=8)
    ap.add_argument("--jepa-lambda", type=float, default=0.0,
                    help="weight of the JEPA distillation regularizer (0 = pure CE)")
    ap.add_argument("--augment", action="store_true",
                    help="training-only flips/intensity/noise augmentation")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min-epochs", type=int, default=3,
                    help="never early-stop before this many epochs")
    ap.add_argument("--max-patients", type=int, default=0, help="smoke subset (train+dev)")
    ap.add_argument("--random-init", action="store_true", help="skip champion (structural smoke)")
    ap.add_argument("--eval-final", action="store_true",
                    help="touch the reserved final slice ONCE at the end (default off)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.random_init:
        cfg["model"]["brainiac"]["checkpoint"] = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    proto = load_protocol(args.protocol)
    train_pids, dev_pids = list(proto["encoder_train"]), list(proto["dev"])
    if args.max_patients:
        train_pids, dev_pids = train_pids[:args.max_patients], dev_pids[:args.max_patients]
    size = tuple(cfg["preprocessing"].get("target_size", [96, 96, 96]))
    common = dict(meta_dir=cfg["data"]["meta_dir"], processed_root=cfg["data"]["root"],
                  raw_root=cfg["data"].get("raw_root"),
                  modalities=tuple(cfg["data"].get("modalities", ["CT1", "T1", "T2", "FLAIR"])),
                  min_visits=cfg["data"].get("min_visits", 2))
    train_ds = LUMIEREDataset(patients=train_pids, **common)
    val_ds = LUMIEREDataset(patients=dev_pids, **common)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0,
                              collate_fn=make_collate(size, augment=args.augment))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=make_collate(size, augment=False))
    print(f"train {len(train_ds)} / dev {len(val_ds)} patients; augment={args.augment}")

    model = JEPAWorldModel(cfg)
    if not args.random_init:
        ckpt = torch.load(args.champion, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"champion {args.champion} (epoch {ckpt.get('epoch')}, val {ckpt.get('val_loss')})")
    rep_params, head_params = configure_trainable(model)
    all_params = rep_params + head_params

    # class weights from the train label distribution (forecast framing)
    counts = torch.zeros(4)
    for i in range(len(train_ds)):
        s = train_ds[i]
        for a in s["actions"][1:]:
            c = ACTION_TO_FLAT.get(int(a))
            if c is not None:
                counts[c] += 1
    counts = counts.clamp_min(1)
    cw = (counts.sum() / (4 * counts)).to(device)
    print(f"train class counts {counts.tolist()} weights {[round(x,2) for x in cw.tolist()]}")

    model.to(device)
    opt = torch.optim.AdamW([
        {"params": rep_params, "base_lr": args.lr},
        {"params": head_params, "base_lr": args.head_lr},
    ], lr=args.lr, weight_decay=args.wd)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    steps_per_epoch = max(1, (len(train_loader) + args.accum_steps - 1) // args.accum_steps)
    total_steps = args.epochs * steps_per_epoch
    warmup = args.warmup_epochs * steps_per_epoch
    gstep = 0
    best_dev, best_ep, bad = -1.0, -1, 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        pending = 0
        run_loss, run_ce, run_j = 0.0, 0.0, 0.0
        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp):
                states, valid = forward_states(model, batch)
                flat, use = flat_targets(batch, valid)
                ce = sup_loss(model, states, flat, use, cw)
                loss = ce if ce is not None else states.sum() * 0.0
                if args.jepa_lambda > 0:
                    z_hat = model.predictor(states)
                    with torch.no_grad():
                        z_tgt = model.encode_target_visit(batch["mri"], batch["mri_mask"])[:, 1:]
                    fv = valid.reshape(-1)
                    if fv.any():
                        j = jepa_loss(z_hat.reshape(-1, z_hat.shape[-1])[fv],
                                      z_tgt.reshape(-1, z_tgt.shape[-1])[fv])
                        loss = loss + args.jepa_lambda * j
                        run_j += float(j.detach())
            scaler.scale(loss / args.accum_steps).backward()
            pending += 1
            run_loss += float(loss.detach())
            run_ce += float(ce.detach()) if ce is not None else 0.0
            if pending >= args.accum_steps:
                if gstep < warmup:
                    factor = (gstep + 1) / max(warmup, 1)
                else:
                    factor = 0.5 * (1 + math.cos(math.pi * min(
                        (gstep - warmup) / max(total_steps - warmup, 1), 1.0)))
                for pg in opt.param_groups:
                    pg["lr"] = pg["base_lr"] * factor
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                # NB: the EMA target is deliberately NOT updated — it stays the
                # frozen champion so the JEPA regularizer anchors the vision
                # tower to the self-supervised representation (not BYOL drift).
                pending = 0
                gstep += 1
        dev_f1, per = eval_dev(model, val_loader, device)
        print(f"epoch {ep}: loss={run_loss/max(len(train_loader),1):.4f} "
              f"ce={run_ce/max(len(train_loader),1):.4f} "
              f"jepa={run_j/max(len(train_loader),1):.4f} dev_macroF1={dev_f1:.4f}",
              flush=True)
        if dev_f1 > best_dev:
            best_dev, best_ep, bad = dev_f1, ep, 0
            torch.save({"epoch": ep, "model": model.state_dict(), "dev_macro_f1": dev_f1,
                        "config": cfg, "args": vars(args)},
                       os.path.join(args.checkpoint_dir, "best.pt"))
        else:
            bad += 1
            if bad >= args.patience and ep >= args.min_epochs:
                print(f"early stop at epoch {ep} (no dev gain in {args.patience})")
                break

    print(f"best dev macro-F1 {best_dev:.4f} @ epoch {best_ep}")
    prov = {
        "champion": os.path.abspath(args.champion),
        "config_sha1": hashlib.sha1(open(args.config, "rb").read()).hexdigest(),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "date": time.strftime("%Y-%m-%d"),
        "args": vars(args), "best_dev_macro_f1": best_dev, "best_epoch": best_ep,
    }
    with open(os.path.join(args.checkpoint_dir, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
    print(f"provenance -> {args.checkpoint_dir}/provenance.json")

    if args.eval_final:
        # touch the reserved final slice exactly once
        best = torch.load(os.path.join(args.checkpoint_dir, "best.pt"),
                          map_location=device, weights_only=False)
        model.load_state_dict(best["model"])
        final_ds = LUMIEREDataset(patients=proto["final"], **common)
        final_loader = DataLoader(final_ds, batch_size=1, shuffle=False, num_workers=0,
                                  collate_fn=make_collate(size, augment=False))
        f1, per = eval_dev(model, final_loader, device)
        lo, hi = bootstrap(per, seed=args.seed)
        print(f"FINAL (reserved, touched once): {len(per)} patients "
              f"macro-F1 {f1:.4f} [{lo:.4f},{hi:.4f}]")


if __name__ == "__main__":
    main()
