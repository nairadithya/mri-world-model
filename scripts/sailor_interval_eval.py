"""Interval-stratified SAILOR eval: regime vs representation decider.

The cross-site mean loss says persistence wins ~5x — but SAILOR gaps average
~14 days (near-static targets where "no change" is near-optimal). If the
failure is regime, JEPA should win on long-gap pairs where change has time
to happen; if representation, persistence wins at every gap.

Per valid (t -> t+1) pair: 1-step JEPA error (champion forward), persistence
error (1 - cos of target-space endpoints), gap days from time_deltas.
Reports means per gap bin plus gap distribution.

Usage: python scripts/sailor_interval_eval.py --champion checkpoints/champion_0.0081.pt
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.collate import make_collate
from src.data.sailor import SAILORDataset
from src.model.jepa_model import JEPAWorldModel

SAILOR_ROOT = "data/sailor/sailor_ebrains_pseud/derivatives/mni2009c-n-s"
BINS = [0, 21, 60, 180, float("inf")]
BIN_LABELS = ["0-21d", "22-60d", "61-180d", "180d+"]


def gap_bin(g):
    for i in range(len(BINS) - 1):
        if BINS[i] <= g < BINS[i + 1]:
            return i
    return len(BINS) - 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--champion", default="checkpoints/champion_0.0081.pt")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ds = SAILORDataset(SAILOR_ROOT)
    size = tuple(cfg["preprocessing"].get("target_size", [96, 96, 96]))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=make_collate(size))
    model = JEPAWorldModel(cfg)
    ckpt = torch.load(args.champion, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    by: dict[int, tuple[list, list]] = {}
    gaps_all = []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            out = model(batch)
            n = int(batch["n_visits"][0])
            z = model.encode_target_visit(
                batch["mri"], batch["mri_mask"])[0, :n]
            je = 1 - F.cosine_similarity(out["z_hat"][0], out["z_target"][0],
                                         dim=-1)
            for t in range(len(je)):
                if not bool(out["valid"][0, t]):
                    continue
                gap = float(batch["time_deltas"][0, t + 1])
                gaps_all.append(gap)
                pe = 1 - (F.normalize(z[t], dim=0) *
                          F.normalize(z[t + 1], dim=0)).sum().item()
                b = gap_bin(gap)
                by.setdefault(b, ([], []))
                by[b][0].append(float(je[t]))
                by[b][1].append(pe)
            if (bi + 1) % 5 == 0:
                print(f"{bi + 1}/{len(ds)} subjects", flush=True)
    gaps_all.sort()
    print(f"\ngap-days distribution (n={len(gaps_all)}): "
          f"median={gaps_all[len(gaps_all) // 2]:.0f} "
          f"p10={gaps_all[len(gaps_all) // 10]:.0f} "
          f"p90={gaps_all[9 * len(gaps_all) // 10]:.0f}")
    print(f"{'gap':>8} {'pairs':>7} {'jepa':>8} {'persist':>8}")
    for b in sorted(by):
        je, pe = by[b]
        j, p = statistics.mean(je), statistics.mean(pe)
        print(f"{BIN_LABELS[b]:>8} {len(je):>7} {j:>8.4f} {p:>8.4f}"
              f"{' <-- JEPA wins' if j < p else ''}")


if __name__ == "__main__":
    main()
