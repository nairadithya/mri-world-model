"""Apply a LUMIERE-trained gap-conditioned probe head to SAILOR, zero training.

The cross-site fix test: the champion 1-step head emits fixed-scale deltas
(flat ~0.03 error at all SAILOR gaps) with no notion of elapsed time. A head
conditioned on the day gap should scale its predictions properly. Inputs are
frozen champion states (sailor_cache) + SAILOR gap days (dataset) +
LUMIERE-fitted gap normalizer (probe weights file); targets are cached
target-space z (sailor_z_cache). Reports probe vs persistence per gap bin on
identical pairs, plus the champion 1-step reference means from the interval
eval (0.0284 / 0.0360 / 0.0283 / 0.0292).

Usage:
    python scripts/horizon_probe.py --train --epochs 300 --save checkpoints/probe_head.pt
    python scripts/sailor_gap_probe.py --head checkpoints/probe_head.pt
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
from src.data.sailor import SAILORDataset

from horizon_probe import HorizonPredictor  # noqa: E402
from sailor_interval_eval import SAILOR_ROOT, gap_bin, BIN_LABELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="checkpoints/probe_head.pt")
    ap.add_argument("--sailor-cache", default="checkpoints/sailor_cache.pt")
    ap.add_argument("--z-cache", default="checkpoints/sailor_z_cache.pt")
    args = ap.parse_args()

    saved = torch.load(args.head, map_location="cpu", weights_only=False)
    net = HorizonPredictor(hidden=saved.get("hidden", 1024),
                           layers=saved.get("layers", 2))
    net.load_state_dict(saved["net"])
    net.eval()
    mu, sd = saved["mu"], saved["sd"]

    sc = torch.load(args.sailor_cache, map_location="cpu",
                    weights_only=False)["patients"]
    zc = torch.load(args.z_cache, map_location="cpu",
                    weights_only=False)["patients"]
    ds = SAILORDataset(SAILOR_ROOT)

    by: dict[int, tuple[list, list]] = {}
    with torch.no_grad():
        for sub in sorted(sc):
            st = sc[sub]["states"]          # (T-1, 1152), visits <= t
            z = zc[sub]["z"]                # (T, 768) target space
            item = ds[ds.subjects.index(sub)]
            gaps = item["time_deltas"]      # deltas[t] = days since t-1
            T = len(z)
            assert len(st) == T - 1, sub
            for t in range(T - 1):
                if not zc[sub]["has_img"][t] or not zc[sub]["has_img"][t + 1]:
                    continue
                gap = float(gaps[t + 1])
                gg = (torch.log1p(torch.tensor(gap)) - mu) / sd
                pred = net(st[t].unsqueeze(0), gg.reshape(1)).squeeze(0)
                je = 1 - (F.normalize(pred, dim=0) *
                          F.normalize(z[t + 1], dim=0)).sum().item()
                pe = 1 - (F.normalize(z[t], dim=0) *
                          F.normalize(z[t + 1], dim=0)).sum().item()
                b = gap_bin(gap)
                by.setdefault(b, ([], []))
                by[b][0].append(je)
                by[b][1].append(pe)
    print(f"{'gap':>8} {'pairs':>7} {'gap-head':>8} {'persist':>8} "
          f"{'champ1step':>10}")
    champ = [0.0284, 0.0360, 0.0283, 0.0292]  # interval-eval reference means
    for b in sorted(by):
        je, pe = by[b]
        j, p = statistics.mean(je), statistics.mean(pe)
        print(f"{BIN_LABELS[b]:>8} {len(je):>7} {j:>8.4f} {p:>8.4f} "
              f"{champ[b]:>10.4f}{' <-- wins' if j < p else ''}")


if __name__ == "__main__":
    main()
