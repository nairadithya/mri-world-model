"""Multi-horizon probe: predict visit t+n from state_t for all valid (t, n).

Tests whether long-range forecasting beats persistence, with the champion
encoder FROZEN throughout (only a small probe predictor trains, CPU minutes).

Phases:
  --encode : one pass over all patients, cache per-visit target-space vectors
             z (T,768), prefix states (T-1,1152), day gaps and image flags.
  --curve  : persistence error vs horizon n (1 - cos(z_t, z_{t+n})), per
             horizon and split. Zero training -- the headroom check.
  --train  : horizon-conditioned MLP probe: [state_t, log-gap] -> z_{t+n},
             trained on train-split pairs with 1/n horizon weighting (closer
             futures count more), scored per-horizon vs persistence
             on val/test. Gate: beats persistence at n>=2, matches ~1-step
             error at n=1.

Gap semantics: time_deltas[t] = days since visit t-1 (deltas[0] = 0), so the
horizon gap t -> t+n is deltas[t+1 .. t+n].sum().

Usage:
    python scripts/horizon_probe.py --encode
    python scripts/horizon_probe.py --curve
    python scripts/horizon_probe.py --train --epochs 300
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.probe_rano import build_datasets
from src.data.collate import make_collate
from src.model.jepa_model import JEPAWorldModel


def encode_all(cfg, champion_path, cache_path):
    device = torch.device("cpu")
    datasets, _ = build_datasets(cfg)
    size = tuple(cfg["preprocessing"].get("target_size", [96, 96, 96]))
    collate = make_collate(size)

    model = JEPAWorldModel(cfg)
    ckpt = torch.load(champion_path, map_location="cpu", weights_only=False)
    # Champion predates the RANO heads: missing rano_heads.* keys are expected.
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"champion: {champion_path} (epoch {ckpt.get('epoch')}, "
          f"val {ckpt.get('val_loss')})", flush=True)

    cache = {"patients": {}}
    t0 = time.time()
    done = 0
    total = sum(len(ds) for ds in datasets.values())
    with torch.no_grad():
        for split, ds in datasets.items():
            loader = DataLoader(ds, batch_size=1, shuffle=False,
                                num_workers=0, collate_fn=collate)
            for batch in loader:
                pid = batch["patient_id"][0]
                n = int(batch["n_visits"][0])
                z = model.encode_target_visit(
                    batch["mri"], batch["mri_mask"])[0, :n].clone()  # (T,768)
                v = model.encode_visits(batch["mri"], batch["mri_mask"])
                c = model.clinical(batch["clinical"])
                tok = model.fusion(v, c.unsqueeze(1).expand(-1, v.shape[1], -1))
                states, _ = model.temporal.forward_prefixes(
                    tok, batch["time_deltas"], batch["visit_mask"])
                cache["patients"][pid] = {
                    "split": split,
                    "z": z,
                    "states": states[0, :n - 1].clone(),  # state_t covers visits<=t
                    "deltas": batch["time_deltas"][0, :n].clone(),
                    "has_img": batch["mri_mask"][0, :n].any(dim=-1).clone(),
                }
                done += 1
                if done == 1 or done % 10 == 0 or done == total:
                    el = time.time() - t0
                    print(f"encoded {done}/{total} ({el / done:.1f}s/patient, "
                          f"ETA {el / done * (total - done) / 60:.0f}min)",
                          flush=True)
    torch.save(cache, cache_path)
    print(f"cache: {total} patients -> {cache_path}")


def pairs_for(patients, split_names):
    """All valid (t, t+n) pairs: (split, state_t, z_t, z_target, gap_days, n)."""
    rows = []
    for pid, p in patients.items():
        if p["split"] not in split_names:
            continue
        T = len(p["z"])
        for t in range(T - 1):
            if not p["has_img"][t]:
                continue
            gap = 0.0
            for u in range(t + 1, T):
                gap += float(p["deltas"][u])
                if not p["has_img"][u]:
                    continue
                rows.append((p["split"], p["states"][t], p["z"][t], p["z"][u],
                             gap, u - t))
    return rows


GAP_BINS = [0, 8, 15, 30, 60, 200, float("inf")]
GAP_LABELS = ["0-8d", "8-15d", "15-30d", "30-60d", "60-200d", "200d+"]


def gap_bin(g):
    for i in range(len(GAP_BINS) - 1):
        if GAP_BINS[i] <= g < GAP_BINS[i + 1]:
            return i
    return len(GAP_BINS) - 2


def cos_err(a, b):
    return (1 - (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)
                 ).sum(dim=-1)).tolist()


def persistence_curve(patients):
    rows = pairs_for(patients, ("train", "val", "test"))
    print(f"{len(rows)} valid (t, t+n) pairs")
    print(f"{'n':>4} {'split':>6} {'pairs':>7} {'persist_err':>11}")
    by = {}
    for s, _st, zt, zu, _gap, n in rows:
        by.setdefault((n, s), []).append((zt, zu))
    for (n, s) in sorted(by):
        a = torch.stack([r[0] for r in by[(n, s)]])
        b = torch.stack([r[1] for r in by[(n, s)]])
        e = torch.tensor(cos_err(a, b))
        sd = e.std().item() if len(e) > 1 else 0.0
        print(f"{n:>4} {s:>6} {len(e):>7} {e.mean():>11.4f} (+/-{sd:.4f})")


class HorizonPredictor(nn.Module):
    """[state_t (1152), standardized log-gap (1)] -> z_{t+n} (768)."""

    def __init__(self, hidden=1024, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1152 + 1, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 768),
        )

    def forward(self, s, g):
        return self.net(torch.cat([s, g.unsqueeze(-1)], dim=-1))


def train_predictor(patients, epochs=300, lr=1e-3, seed=42, weight="inv_n"):
    """weight: 'inv_n' (1/n per pair) or 'inv_gap_err' (1/mean-train-
    persistence-error of the pair's gap-days bin — day-based unit with the
    empirically correct sign: hard bins count less)."""
    g = torch.Generator().manual_seed(seed)
    tr = pairs_for(patients, ("train",))
    va = pairs_for(patients, ("val",))
    te = pairs_for(patients, ("test",))
    print(f"pairs: train {len(tr)}, val {len(va)}, test {len(te)}")

    gaps = torch.tensor([r[4] for r in tr])
    lg = torch.log1p(gaps)
    mu, sd = lg.mean(), lg.std().clamp_min(1e-6)

    def feats(rows):
        s = torch.stack([r[1] for r in rows])
        zt = torch.stack([r[2] for r in rows])
        z = torch.stack([r[3] for r in rows])
        gg = (torch.log1p(torch.tensor([r[4] for r in rows])) - mu) / sd
        nn_ = torch.tensor([r[5] for r in rows])
        return s, zt, gg, z, nn_

    Xtr, Zttr, Gtr, Ytr, Ntr = feats(tr)
    Xva, _, Gva, Yva, Nva = feats(va)
    Xte, Ztte, Gte, Yte, Nte = feats(te)

    if weight == "inv_gap_err":
        with torch.no_grad():
            perr = 1 - (F.normalize(Zttr, dim=-1) *
                        F.normalize(Ytr, dim=-1)).sum(dim=-1)
        bin_err, bin_n = {}, {}
        for gap, e in zip([r[4] for r in tr], perr.tolist()):
            b = gap_bin(gap)
            bin_err[b] = bin_err.get(b, 0.0) + e
            bin_n[b] = bin_n.get(b, 0) + 1
        bin_w = {b: 1.0 / (bin_err[b] / bin_n[b]) for b in bin_err}
        print("gap-bin mean persistence err / weight:",
              {GAP_LABELS[b]: (round(bin_err[b] / bin_n[b], 4), round(bin_w[b], 1))
               for b in sorted(bin_err)})
        Wtr = torch.tensor([bin_w[gap_bin(r[4])] for r in tr])
    else:
        Wtr = 1.0 / Ntr.float()
    print(f"weighting: {weight}")

    net = HorizonPredictor()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    B = 512
    n = len(Xtr)
    # Horizon weighting: closer futures count more, farther futures less.
    # w = 1/n per pair (renormalized per batch); eval stays per-horizon and
    # unweighted so the weighting choice itself remains auditable.
    Wtr = 1.0 / Ntr.float()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=g)
        tot, cnt = 0.0, 0
        for s in range(0, n, B):
            idx = perm[s:s + B]
            pred = net(Xtr[idx], Gtr[idx])
            err = 1 - (F.normalize(pred, dim=-1) *
                       F.normalize(Ytr[idx], dim=-1)).sum(dim=-1)
            w = Wtr[idx] / Wtr[idx].sum()
            loss = (w * err).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            cnt += len(idx)
        if (ep + 1) % 50 == 0 or ep == 0:
            net.eval()
            with torch.no_grad():
                pv = 1 - (F.normalize(net(Xva, Gva), dim=-1) *
                          F.normalize(Yva, dim=-1)).sum(dim=-1).mean().item()
            print(f"epoch {ep + 1}/{epochs}: train {tot / cnt:.4f} val {pv:.4f}",
                  flush=True)

    net.eval()
    with torch.no_grad():
        Pte = net(Xte, Gte)
    print(f"\n{'n':>4} {'pairs':>7} {'probe_err':>10} {'persist_err':>11}")
    for h in sorted(set(Nte.tolist())):
        m = Nte == h
        if m.sum() < 3:
            continue
        pe = torch.tensor(cos_err(Pte[m], Yte[m]))
        se = torch.tensor(cos_err(Ztte[m], Yte[m]))
        flag = " <-- probe wins" if pe.mean() < se.mean() else ""
        print(f"{h:>4} {m.sum():>7} {pe.mean():>10.4f} {se.mean():>11.4f}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--champion", default="checkpoints/champion_0.0081.pt")
    ap.add_argument("--cache", default="checkpoints/horizon_cache.pt")
    ap.add_argument("--encode", action="store_true")
    ap.add_argument("--curve", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight", default="inv_n", choices=["inv_n", "inv_gap_err"],
                    help="pair weighting: 1/n or 1/mean-gap-bin-persistence-error")
    args = ap.parse_args()

    if args.encode:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        encode_all(cfg, args.champion, args.cache)
        return
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    patients = cache["patients"]
    if args.curve:
        persistence_curve(patients)
    if args.train:
        train_predictor(patients, epochs=args.epochs, lr=args.lr,
                        weight=args.weight)


if __name__ == "__main__":
    main()
