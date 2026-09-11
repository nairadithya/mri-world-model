"""Per-split JEPA-vs-persistence gate (G13/R3/K3-11). No backbone forward:
runs the champion's 1-step predictor MLP over cached states (CPU seconds).

For each valid (t -> t+1) pair with pixel-backed endpoints:
  jepa    = 1 - cos(predictor(state_t), z_{t+1})
  persist = 1 - cos(z_t, z_{t+1})
where z is the EMA-target latent — SAME space on both sides (D28/A8
addendum). Reports per split:
  - pooled means over all pairs;
  - patient-uniform means (each patient weighted equally);
  - patient win counts (patients whose mean JEPA < their mean persistence);
and 95% CIs from a patient-level cluster bootstrap (K3-11). The test column
is the uncontaminated gate; the pooled train row reproduces A8's harness.

Usage:
    python scripts/split_gate.py --champion checkpoints/champion_0.0081.pt
"""
from __future__ import annotations

import argparse
import os
import random
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


def _ci(samples, lo=2.5, hi=97.5):
    s = sorted(samples)
    n = len(s)
    return s[int(lo / 100 * n)], s[min(n - 1, int(hi / 100 * n))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", default="checkpoints/champion_0.0081.pt")
    ap.add_argument("--cache", default="checkpoints/horizon_cache.pt")
    ap.add_argument("--boot", type=int, default=10000,
                    help="patient-level bootstrap resamples (0 = skip)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pred = load_predictor(args.champion)
    patients = torch.load(args.cache, map_location="cpu",
                          weights_only=False)["patients"]

    # split -> pid -> (jepa errors, persistence errors)
    data: dict[str, dict[str, tuple[list, list]]] = {}
    with torch.no_grad():
        for pid, p in patients.items():
            s, z, him = p["split"], p["z"], p["has_img"]
            T = len(z)
            jl, pl = [], []
            for t in range(T - 1):
                if not him[t] or not him[t + 1]:
                    continue
                j = 1 - (F.normalize(pred(p["states"][t].unsqueeze(0)), dim=-1) *
                         F.normalize(z[t + 1].unsqueeze(0), dim=-1)).sum().item()
                e = 1 - (F.normalize(z[t], dim=0) *
                         F.normalize(z[t + 1], dim=0)).sum().item()
                jl.append(j)
                pl.append(e)
            if jl:
                data.setdefault(s, {})[pid] = (jl, pl)

    rng = random.Random(args.seed)

    def report(label, pid_map):
        pids = list(pid_map)
        pj = [statistics.mean(pid_map[pid][0]) for pid in pids]
        pp = [statistics.mean(pid_map[pid][1]) for pid in pids]
        allj = [x for pid in pids for x in pid_map[pid][0]]
        allp = [x for pid in pids for x in pid_map[pid][1]]
        pool = (statistics.mean(allj), statistics.mean(allp))
        pu = (statistics.mean(pj), statistics.mean(pp))
        wins = sum(a < b for a, b in zip(pj, pp))
        print(f"{label} ({len(pids)} patients, {len(allj)} pairs)")
        if args.boot:
            bp_j, bp_p, bu_j, bu_p, bw = [], [], [], [], []
            for _ in range(args.boot):
                samp = [rng.choice(pids) for _ in pids]
                sj = [statistics.mean(pid_map[pid][0]) for pid in samp]
                sp = [statistics.mean(pid_map[pid][1]) for pid in samp]
                bp_j.append(statistics.mean(x for pid in samp
                                            for x in pid_map[pid][0]))
                bp_p.append(statistics.mean(x for pid in samp
                                            for x in pid_map[pid][1]))
                bu_j.append(statistics.mean(sj))
                bu_p.append(statistics.mean(sp))
                bw.append(sum(a < b for a, b in zip(sj, sp)) / len(samp))
            # paired (same patient-resample) difference; negative = JEPA better
            bd_pool = [j - p for j, p in zip(bp_j, bp_p)]
            bd_pu = [j - p for j, p in zip(bu_j, bu_p)]
            dlo, dhi = _ci(bd_pool)
            dul, duh = _ci(bd_pu)
            sig_pool = "SIG" if dhi < 0 or dlo > 0 else "n.s."
            sig_pu = "SIG" if duh < 0 or dul > 0 else "n.s."
            print(f"  pooled      JEPA {pool[0]:.4f} [{_ci(bp_j)[0]:.4f},{_ci(bp_j)[1]:.4f}]"
                  f"  persist {pool[1]:.4f} [{_ci(bp_p)[0]:.4f},{_ci(bp_p)[1]:.4f}]")
            print(f"  patient-u   JEPA {pu[0]:.4f} [{_ci(bu_j)[0]:.4f},{_ci(bu_j)[1]:.4f}]"
                  f"  persist {pu[1]:.4f} [{_ci(bu_p)[0]:.4f},{_ci(bu_p)[1]:.4f}]")
            print(f"  paired diff {pool[0] - pool[1]:+.4f} [{dlo:+.4f},{dhi:+.4f}] {sig_pool}"
                  f"   pat-u {pu[0] - pu[1]:+.4f} [{dul:+.4f},{duh:+.4f}] {sig_pu}")
            lo, hi = _ci(bw)
            print(f"  wins        {wins}/{len(pids)} ({100 * wins / len(pids):.0f}%)"
                  f"  [CI {100 * lo:.0f},{100 * hi:.0f}%]")
        else:
            print(f"  pooled      JEPA {pool[0]:.4f}  persist {pool[1]:.4f}")
            print(f"  patient-u   JEPA {pu[0]:.4f}  persist {pu[1]:.4f}")
            print(f"  wins        {wins}/{len(pids)}")

    for s in ("train", "val", "test"):
        report(s, data.get(s, {}))
    overall = {pid: v for s in ("train", "val", "test")
               for pid, v in data.get(s, {}).items()}
    report("overall", overall)
    print("[A8 addendum reference: train pooled 0.0082/0.0065, test "
          "0.0070/0.0088; same-space overall win 41/91]")


if __name__ == "__main__":
    main()
