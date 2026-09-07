"""Treatment-conditioned velocity-field dynamics on SAILOR, frozen champion, CPU.

Option 2 (R14 core): the champion encoder never trains here. Phase 1 encodes
every SAILOR session once (target-space z + prefix states + clinical +
treatment phase + gaps). Phase 2 fits a SMALL VelocityField + PatientTempo on
cached pairs — all (t, u>t) pairs, 1/n-weighted, Euler-integrated across true
gaps (same math as JEPAWorldModel._dynamics_loss) — under k-fold
subject-wise CV. Variant A conditions on the real treatment phase
(CRT/TMZ/no/unknown); variant B (ablation) feeds a constant phase: same
capacity, same init, same protocol. The A-vs-B gap on HELD-OUT subjects is
the RQ2 test (does treatment info help?). Persistence on identical pairs is
the floor for both.

Field init: final layer scaled x0.01 so training starts AT persistence
(change must be earned, not emitted by default); velocity_norm must leave ~0
AND held-out must beat persistence, else the field just found the fixed point.

Usage:
    python scripts/train_field.py --encode            # one-time, CPU ~40 min
    python scripts/train_field.py --train             # 5-fold CV, CPU
    python scripts/train_field.py --encode --subjects sub-01 sub-02   # smoke
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data.collate import make_collate
from src.data.sailor import SAILORDataset, TREATMENT_NAMES
from src.model.dynamics_field import PatientTempo, VelocityField, integrate
from src.model.jepa_model import JEPAWorldModel

from surprise_signal import auc_mann_whitney  # noqa: E402

SAILOR_ROOT = "data/sailor/sailor_ebrains_pseud/derivatives/mni2009c-n-s"


def encode_all(cfg, champion_path, cache_path, subjects=None):
    device = torch.device("cpu")
    ds = SAILORDataset(SAILOR_ROOT, subjects=subjects)
    size = tuple(cfg["preprocessing"].get("target_size", [96, 96, 96]))
    collate = make_collate(size)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate)

    model = JEPAWorldModel(cfg)
    ckpt = torch.load(champion_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"champion: {champion_path} (epoch {ckpt.get('epoch')}, "
          f"val {ckpt.get('val_loss')})", flush=True)

    # deterministic 5-fold over subjects (repo CV convention: shuffled[i::5])
    import random
    subs = sorted(ds.subjects)
    rng = random.Random(42)
    rng.shuffle(subs)
    fold = {s: i % 5 for i, s in enumerate(subs)}

    cache = {"patients": {}}
    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            pid = batch["patient_id"][0]
            n = int(batch["n_visits"][0])
            item = ds[ds.subjects.index(pid)]
            z = model.encode_target_visit(
                batch["mri"], batch["mri_mask"])[0, :n].clone()
            v = model.encode_visits(batch["mri"], batch["mri_mask"])
            c = model.clinical(batch["clinical"])
            tok = model.fusion(v, c.unsqueeze(1).expand(-1, v.shape[1], -1))
            states, _ = model.temporal.forward_prefixes(
                tok, batch["time_deltas"], batch["visit_mask"])
            cache["patients"][pid] = {
                "fold": fold[pid],
                "z": z,
                "states": states[0, :n - 1].clone(),
                "clinical": c[0].clone(),
                "treatment": item["treatment"][:n].clone(),
                "deltas": batch["time_deltas"][0, :n].clone(),
                "has_img": batch["mri_mask"][0, :n].any(dim=-1).clone(),
                "codes": [ds.sailor_rano.get((pid, s)) for s in item["visits"][:n]],
            }
            el = time.time() - t0
            print(f"encoded {i + 1}/{len(ds)} ({el / (i + 1):.1f}s/subject)",
                  flush=True)
    torch.save(cache, cache_path)
    print(f"cache: {len(ds)} subjects -> {cache_path}")


def pairs_of(p, one_step=False):
    """All valid (t, u>t) pairs (or t+1 only): feature dict + target + meta."""
    z, st = p["z"], p["states"]
    T = len(z)
    rows = []
    for t in range(T - 1):
        if not p["has_img"][t]:
            continue
        for u in range(t + 1, T):
            if one_step and u != t + 1:
                continue
            if not p["has_img"][u]:
                continue
            gap = float(p["deltas"][t + 1:u + 1].sum())
            rows.append({"z0": z[t], "h": st[t],
                         "phase": int(p["treatment"][t]),
                         "clin": p["clinical"], "gap": gap,
                         "tgt": z[u], "n": u - t,
                         "code": p["codes"][u]})
    return rows


def fit_field(rows_tr, hidden=256, dropout=0.2, wd=0.1, lr=1e-3,
              max_epochs=400, seed=42, use_phase=True,
              steps=3):
    """Full-batch fit on train pairs for a fixed budget; returns LAST state.

    No within-fold validation split (pairs too few to split twice) and no
    best-loss selection (that would overfit-select): the fixed budget +
    strong decay + small net is the regularizer. Honest by construction;
    held-out folds judge.
    Returns (field, tempo, train_curve).
    """
    torch.manual_seed(seed)
    field = VelocityField(hidden_dim=hidden, dropout=dropout)
    tempo = PatientTempo()
    # Start at persistence: near-zero initial velocities.
    with torch.no_grad():
        field.net[-1].weight.mul_(0.01)
        field.net[-1].bias.zero_()
    opt = torch.optim.AdamW(list(field.parameters()) + list(tempo.parameters()),
                            lr=lr, weight_decay=wd)
    z0 = torch.stack([r["z0"] for r in rows_tr])
    h = torch.stack([r["h"] for r in rows_tr])
    ph = torch.tensor([r["phase"] if use_phase else 0 for r in rows_tr])
    clin = torch.stack([r["clin"] for r in rows_tr])
    gaps = torch.tensor([r["gap"] for r in rows_tr])
    tgt = torch.stack([r["tgt"] for r in rows_tr])
    nn_ = torch.tensor([r["n"] for r in rows_tr]).float()
    w = 1.0 / nn_.clamp_min(1)
    w = w / w.sum()
    curve = []
    for ep in range(max_epochs):
        field.train()
        tempo.train()
        tp = tempo(clin)
        pred = integrate(field, tp, z0, h, ph, clin, gaps, steps=steps)
        err = 1 - (F.normalize(pred, dim=-1) *
                   F.normalize(tgt, dim=-1)).sum(dim=-1)
        loss = (w * err).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        curve.append(loss.item())
    field.eval()
    tempo.eval()
    return field, tempo, curve


@torch.no_grad()
def score_pairs(field, tempo, rows, steps=3):
    out = []
    for r in rows:
        tp = tempo(r["clin"].unsqueeze(0))
        pred = integrate(field, tp, r["z0"].unsqueeze(0), r["h"].unsqueeze(0),
                         torch.tensor([r["phase"]]),
                         r["clin"].unsqueeze(0),
                         torch.tensor([r["gap"]]), steps=steps).squeeze(0)
        je = 1 - (F.normalize(pred, dim=0) *
                  F.normalize(r["tgt"], dim=0)).sum().item()
        pe = 1 - (F.normalize(r["z0"], dim=0) *
                  F.normalize(r["tgt"], dim=0)).sum().item()
        vn = (pred - r["z0"]).norm().item()
        out.append((je, pe, r["phase"], r["code"], vn))
    return out


def run_cv(cache_path, hidden=256, dropout=0.2, wd=0.1, lr=1e-3,
           max_epochs=400, steps=3, seed=42):
    cache = torch.load(cache_path, map_location="cpu",
                       weights_only=False)["patients"]
    print(f"{'fold':>4} {'variant':>8} {'held_n':>7} {'jepa':>8} "
          f"{'persist':>8} {'velnorm':>8}")
    agg: dict[str, list] = {"cond": [], "uncond": [], "persist": []}
    auc_c, auc_u, auc_p = [], [], []
    saved: dict[str, list] = {}  # test-7 within-phase slicing: fold/variant rows
    for f in range(5):
        tr_rows, te_rows = [], []
        for pid, p in cache.items():
            (te_rows if p["fold"] == f else tr_rows).extend(pairs_of(p))
        if not tr_rows or not te_rows:
            continue
        te1 = [r for _, p in cache.items() if p["fold"] == f
               for r in pairs_of(p, one_step=True)]
        for tag, use_phase in (("cond", True), ("uncond", False)):
            torch.manual_seed(seed + f)
            field, tempo, _ = fit_field(
                tr_rows, hidden=hidden, dropout=dropout, wd=wd, lr=lr,
                max_epochs=max_epochs, seed=seed + f, use_phase=use_phase,
                steps=steps)
            sc = score_pairs(field, tempo, te1, steps=steps)
            saved[f"{f}/{tag}"] = [(s[0], s[1], s[2], s[3]) for s in sc]
            je = sum(s[0] for s in sc) / len(sc)
            pe = sum(s[1] for s in sc) / len(sc)
            vn = sum(s[4] for s in sc) / len(sc)
            agg[tag].append(je)
            agg["persist"].append(pe)
            yb = torch.tensor([1 if s[3] == 1 else 0 for s in sc
                               if s[3] in (1, 2, 3, 5)])
            if yb.sum() > 0 and yb.sum() < len(yb):
                (auc_c if tag == "cond" else auc_u).append(
                    auc_mann_whitney(
                        torch.tensor([s[0] for s in sc
                                      if s[3] in (1, 2, 3, 5)]), yb))
            print(f"{f:>4} {tag:>8} {len(sc):>7} {je:>8.4f} {pe:>8.4f} "
                  f"{vn:>8.4f}", flush=True)
        ep = torch.tensor([s[1] for s in sc])
        yb = torch.tensor([1 if s[3] == 1 else 0 for s in sc
                           if s[3] in (1, 2, 3, 5)])
        if yb.sum() > 0 and yb.sum() < len(yb):
            auc_p.append(auc_mann_whitney(
                torch.tensor([s[1] for s in sc if s[3] in (1, 2, 3, 5)]), yb))
    import statistics as st
    print(f"\nCV held-out 1-step: cond {st.mean(agg['cond']):.4f}±"
          f"{st.pstdev(agg['cond']):.4f} | uncond {st.mean(agg['uncond']):.4f}±"
          f"{st.pstdev(agg['uncond']):.4f} | persist "
          f"{st.mean(agg['persist']):.4f}±{st.pstdev(agg['persist']):.4f}")
    if auc_c:
        print(f"surprise-AUC PD: cond {st.mean(auc_c):.4f} | uncond "
              f"{st.mean(auc_u):.4f} | persist {st.mean(auc_p):.4f}")
    torch.save(saved, "checkpoints/field_scores.pt")
    print("per-pair held-out scores -> checkpoints/field_scores.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--champion", default="checkpoints/champion_0.0081.pt")
    ap.add_argument("--cache", default="checkpoints/field_cache.pt")
    ap.add_argument("--encode", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-epochs", type=int, default=400)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.encode:
        encode_all(cfg, args.champion, args.cache, subjects=args.subjects)
    if args.train:
        run_cv(args.cache, hidden=args.hidden, dropout=args.dropout,
               wd=args.wd, lr=args.lr, max_epochs=args.max_epochs,
               steps=args.steps, seed=args.seed)


if __name__ == "__main__":
    main()
