"""Per-split JEPA-vs-persistence gate (G13/R3). No backbone forward: runs the
champion's 1-step predictor MLP over cached states (CPU seconds).

For each valid (t -> t+1) pair with pixel-backed endpoints:
  jepa  = 1 - cos(predictor(state_t), z_{t+1})
  persist = 1 - cos(z_t, z_{t+1})
Reports mean error + patient-level win counts per split. The test column is
the uncontaminated gate (A8's 0.0081/0.0218 pooled 65 train patients); the
pooled column reproduces it as a harness check.

Usage: python scripts/split_gate.py --champion checkpoints/champion_0.0081.pt
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model.jepa import Predictor


def load_predictor(champion_path):
    """The champion's own 1-step Predictor MLP."""
    ckpt = torch.load(champion_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    net = Predictor()
    net.load_state_dict({k.replace("predictor.", ""): v for k, v in sd.items()
                         if k.startswith("predictor.")})
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", default="checkpoints/champion_0.0081.pt")
    ap.add_argument("--cache", default="checkpoints/horizon_cache.pt")
    args = ap.parse_args()

    pred = load_predictor(args.champion)
    patients = torch.load(args.cache, map_location="cpu",
                          weights_only=False)["patients"]
    by: dict[str, tuple[list, list]] = {}
    wins: dict[str, list] = {}
    pmean: dict[str, tuple[list, list]] = {}
    with torch.no_grad():
        for pid, p in patients.items():
            s, z, d, him = p["split"], p["z"], p["deltas"], p["has_img"]
            T = len(z)
            je_p, pe_p = [], []
            for t in range(T - 1):
                if not him[t] or not him[t + 1]:
                    continue
                j = 1 - (F.normalize(pred(p["states"][t].unsqueeze(0)), dim=-1) *
                         F.normalize(z[t + 1].unsqueeze(0), dim=-1)).sum().item()
                e = 1 - (F.normalize(z[t], dim=0) *
                         F.normalize(z[t + 1], dim=0)).sum().item()
                je_p.append(j)
                pe_p.append(e)
                by.setdefault(s, ([], []))
                by[s][0].append(j)
                by[s][1].append(e)
            if je_p:
                wins.setdefault(s, []).append(
                    sum(a < b for a, b in zip(je_p, pe_p)) / len(je_p))
                pmean.setdefault(s, ([], []))
                pmean[s][0].append(sum(je_p) / len(je_p))
                pmean[s][1].append(sum(pe_p) / len(pe_p))
    print(f"{'split':>6} {'pairs':>7} {'jepa':>8} {'persist':>8} "
          f"{'pat-win%':>9} {'n_pat':>6} {'jepa_pu':>8} {'pers_pu':>8}")
    allj, allp = [], []
    for s in ("train", "val", "test"):
        je, pe = by.get(s, ([], []))
        allj += je
        allp += pe
        w = wins.get(s, [0])
        mj, mp = pmean.get(s, ([0], [0]))
        print(f"{s:>6} {len(je):>7} {statistics.mean(je):>8.4f} "
              f"{statistics.mean(pe):>8.4f} {statistics.mean(w):>9.3f} "
              f"{len(w):>6} {statistics.mean(mj):>8.4f} {statistics.mean(mp):>8.4f}")
    print(f"{'pool':>6} {len(allj):>7} {statistics.mean(allj):>8.4f} "
          f"{statistics.mean(allp):>8.4f}  [A8 ref: 0.0081 / 0.0218]")


if __name__ == "__main__":
    main()
