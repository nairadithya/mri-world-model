"""Cross-site adaptability: LUMIERE-trained supervised CNN vs JEPA encoder.

Both encoders are trained on LUMIERE (the CNN supervised, the JEPA via SSL),
then frozen and adapted to SAILOR (27 subjects, subject-wise CV):

  zero-shot : classifier trained on LUMIERE rows -> score SAILOR rows
  K-shot    : classifier trained on K SAILOR support subjects -> held-out
              query subjects, macro-F1, for K = 3..20 (learning curves)

Features (forecast framing, pair_t -> RANO_{t+1}):
  JEPA : `sailor_cache.pt` / `interface_cache.pt` temporal states (1152-d)
  CNN  : `checkpoints/cnn_features.pt` avgpool features (512-d)

Usage:
    python scripts/cross_site_adapt.py
    python scripts/cross_site_adapt.py --cnn-cache checkpoints/cnn_features.pt
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.eval_protocol import load_protocol


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


def fit_clf(X, y, seed=0, steps=400, lr=1e-2):
    torch.manual_seed(seed)
    net = nn.Linear(X.shape[1], 4)
    counts = torch.bincount(y, minlength=4).float().clamp_min(1)
    w = counts.sum() / (4 * counts)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(net(X), y, weight=w).backward()
        opt.step()
    return net


def standardize(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def jepa_rows(cache, pids):
    out = {}
    for pid in pids:
        p = cache["patients"][pid]
        st = p["states"]
        lab = p["labels"][1:1 + st.shape[0]]
        m = lab >= 0
        if int(m.sum()):
            out[pid] = (st[m], lab[m])
    return out


def cnn_rows(cache, key, pids):
    out = {}
    for pid in pids:
        if pid in cache.get(key, {}):
            v = cache[key][pid]
            if len(v["labels"]):
                out[pid] = (v["features"], v["labels"])
    return out


def pooled(feats):
    pids = sorted(feats)
    X = torch.cat([feats[p][0] for p in pids])
    y = torch.cat([feats[p][1] for p in pids])
    return X, y


def eval_subjects(net, feats, pids, mu, sd):
    pred, y = [], []
    for p in pids:
        if p not in feats:
            continue
        with torch.no_grad():
            pred.append(net((feats[p][0] - mu) / sd).argmax(1))
        y.append(feats[p][1])
    if not pred:
        return 0.0
    return macro_f1(torch.cat(pred), torch.cat(y))


def zero_shot(lum, sa, train_pids, seed=0):
    Xtr, ytr = pooled({p: lum[p] for p in train_pids if p in lum})
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    net = fit_clf((Xtr - mu) / sd, ytr, seed=seed)
    pred, ys, per = [], [], {}
    for s in sorted(sa):
        with torch.no_grad():
            pred.append(net((sa[s][0] - mu) / sd).argmax(1))
        ys.append(sa[s][1])
        per[s] = (pred[-1], ys[-1])
    return macro_f1(torch.cat(pred), torch.cat(ys)), per


def kshot(sa, ks, repeats=20, seed=0):
    rng = random.Random(seed)
    subs = sorted(sa)
    out = {}
    for K in ks:
        scores = []
        for r in range(repeats):
            sh = subs[:]
            rng.shuffle(sh)
            sup, qry = sh[:K], sh[K:]
            if not qry:
                continue
            Xs, ytr = pooled({p: sa[p] for p in sup})
            mu, sd = Xs.mean(0), Xs.std(0).clamp_min(1e-6)
            net = fit_clf((Xs - mu) / sd, ytr, seed=seed + r)
            pred, ys = [], []
            for p in qry:
                with torch.no_grad():
                    pred.append(net((sa[p][0] - mu) / sd).argmax(1))
                ys.append(sa[p][1])
            scores.append(macro_f1(torch.cat(pred), torch.cat(ys)))
        out[K] = (statistics.mean(scores), statistics.pstdev(scores), len(scores))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="info/eval_folds.json")
    ap.add_argument("--jepa-sailor", default="checkpoints/sailor_cache.pt")
    ap.add_argument("--jepa-lum", default="checkpoints/interface_cache.pt")
    ap.add_argument("--cnn-cache", default="checkpoints/cnn_features.pt",
                    help="LUMIERE CNN features (from the Kaggle leg)")
    ap.add_argument("--cnn-sailor", default="checkpoints/cnn_sailor.pt",
                    help="SAILOR CNN features (encoded locally)")
    ap.add_argument("--ks", type=int, nargs="*", default=[3, 5, 10, 15, 20])
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    proto = load_protocol(args.protocol)
    jepa_sa = jepa_rows(torch.load(args.jepa_sailor, map_location="cpu", weights_only=False),
                        [f"sub-{i:02d}" for i in range(1, 28)])
    jepa_lum_all = jepa_rows(torch.load(args.jepa_lum, map_location="cpu", weights_only=False),
                             proto["encoder_train"] + proto["dev"] + proto["final"])
    print(f"JEPA: LUMIERE {len(jepa_lum_all)} / SAILOR {len(jepa_sa)} subjects")

    encoders = {"JEPA": (jepa_lum_all, jepa_sa)}
    cnn_lum, cnn_sa = {}, {}
    if os.path.exists(args.cnn_cache):
        cnn_lum = cnn_rows(torch.load(args.cnn_cache, map_location="cpu", weights_only=False),
                           "lum", proto["encoder_train"] + proto["dev"] + proto["final"])
    if os.path.exists(args.cnn_sailor):
        cnn_sa = cnn_rows(torch.load(args.cnn_sailor, map_location="cpu", weights_only=False),
                          "sailor", list(jepa_sa))
    elif os.path.exists(args.cnn_cache):
        cnn_sa = cnn_rows(torch.load(args.cnn_cache, map_location="cpu", weights_only=False),
                          "sailor", list(jepa_sa))
    if cnn_lum and cnn_sa:
        print(f"CNN : LUMIERE {len(cnn_lum)} / SAILOR {len(cnn_sa)} subjects")
        encoders["CNN"] = (cnn_lum, cnn_sa)
    else:
        print("(CNN features incomplete — JEPA only)")

    print(f"\n{'encoder':>7} {'zero-shot':>10} {'K=3':>7} {'K=5':>7} {'K=10':>7} {'K=15':>7} {'K=20':>7}")
    for name, (lum, sa) in encoders.items():
        z, _ = zero_shot(lum, sa, proto["encoder_train"], seed=args.seed)
        curve = kshot(sa, args.ks, repeats=args.repeats, seed=args.seed)
        cols = " ".join(f"{curve.get(k, (float('nan'),0,0))[0]:7.3f}" for k in args.ks)
        print(f"{name:>7} {z:10.3f} {cols}")
        print(f"        (K-shot std: " +
              ", ".join(f"K{k}={curve[k][1]:.3f}" for k in args.ks) + ")")


if __name__ == "__main__":
    main()
