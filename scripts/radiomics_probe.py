"""Locked-protocol in-domain eval of the radiomics/growth features.

Mirrors the A25 protocol (within-unseen CV over the 26 + transfer -> reserved
final) but with the hand-engineered volumetry+growth feature set, so it can be
read directly against the frozen JEPA numbers (0.309 CV / 0.408 transfer / 0.448
final seed-42). Features are standardized with training statistics before the
linear/MLP probe.

Usage:
    python scripts/radiomics_probe.py
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_rano import _ci, fit_linear, macro_f1  # noqa: E402
from src.data.eval_protocol import fold_patients, load_protocol  # noqa: E402


def _rows(cache, pids):
    xs, ys = [], []
    for pid in pids:
        v = cache.get(pid)
        if v is not None and len(v["labels"]):
            xs.append(v["features"])
            ys.append(v["labels"])
    if not xs:
        return None, None
    return torch.cat(xs), torch.cat(ys)


def _fit(X, y, hidden, seed=42):
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    return fit_linear((X - mu) / sd, y, hidden=hidden, seed=seed), mu, sd


def _eval(net, mu, sd, cache, pids):
    per = {}
    for pid in pids:
        v = cache.get(pid)
        if v is None or not len(v["labels"]):
            continue
        with torch.no_grad():
            pred = net((v["features"] - mu) / sd).argmax(1)
        per[pid] = (pred, v["labels"])
    if not per:
        return 0.0, per
    return macro_f1(torch.cat([p[0] for p in per.values()]),
                    torch.cat([p[1] for p in per.values()])), per


def _bootstrap(per, boot=10000, seed=42):
    pids = sorted(per)
    rng = random.Random(seed)
    vals = []
    for _ in range(boot):
        samp = [rng.choice(pids) for _ in pids]
        vals.append(macro_f1(torch.cat([per[p][0] for p in samp]),
                             torch.cat([per[p][1] for p in samp])))
    return _ci(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="checkpoints/radiomics_features.pt")
    ap.add_argument("--protocol", default="info/eval_folds.json")
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()
    proto = load_protocol(args.protocol)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)["lum"]
    unseen = sorted(proto["folds"])
    print(f"radiomics locked eval: {len(cache)} LUMIERE patients, "
          f"feat dim {next(iter(cache.values()))['features'].shape[1]}")

    # within-unseen CV (out-of-fold)
    oof = {}
    for i in range(proto["k"]):
        te = fold_patients(proto, i)
        tr = [p for p in unseen if p not in set(te)]
        X, y = _rows(cache, tr)
        net, mu, sd = _fit(X, y, args.hidden)
        _, per = _eval(net, mu, sd, cache, te)
        oof.update(per)
    f1 = macro_f1(torch.cat([p[0] for p in oof.values()]),
                  torch.cat([p[1] for p in oof.values()]))
    lo, hi = _bootstrap(oof, boot=args.boot)
    print(f"within-unseen CV ({len(oof)} patients): macro-F1 {f1:.4f} [{lo:.4f},{hi:.4f}]")

    # transfer: train on the 65, report the reserved final 13
    X, y = _rows(cache, proto["encoder_train"])
    net, mu, sd = _fit(X, y, args.hidden)
    f1, per = _eval(net, mu, sd, cache, proto["final"])
    lo, hi = _bootstrap(per, boot=args.boot)
    print(f"transfer -> final ({len(per)} patients): macro-F1 {f1:.4f} [{lo:.4f},{hi:.4f}]")


if __name__ == "__main__":
    main()
