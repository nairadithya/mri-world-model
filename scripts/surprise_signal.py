"""Surprise-as-signal (D24.c): does JEPA next-visit prediction error anticipate PD?

For each valid (t -> t+1) pair: err = 1 - cos(predictor(state_t), target_{t+1})
from the frozen champion, label = clean RANO of visit t+1. Reports mean error
by class and ROC-AUC of error-as-PD-score (Mann-Whitney, no sklearn needed).

Usage: python scripts/surprise_signal.py --champion checkpoints/champion_0.0081.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/probe_rano
from src.data.collate import make_collate
from src.data.dataset import LUMIEREDataset
from src.model.jepa_model import JEPAWorldModel

from probe_rano import RANO_PROBE_MAP  # noqa: E402  (scripts/ on path via cwd)


def auc_mann_whitney(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """P(score_pos > score_neg) via rank sum. labels binary {0,1}."""
    pos = scores[labels == 1].sort().values
    neg = scores[labels == 0].sort().values
    # rank all, sum ranks of positives (average ties)
    order = torch.argsort(torch.cat([neg, pos]), stable=True).float() + 1
    n0, n1 = len(neg), len(pos)
    r1 = order[n0:].sum().item()
    return (r1 - n1 * (n1 + 1) / 2) / (n0 * n1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--champion", default="checkpoints/champion_0.0081.pt")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ds = LUMIEREDataset(
        meta_dir=cfg["data"]["meta_dir"], processed_root=cfg["data"]["root"],
        raw_root=cfg["data"].get("raw_root"),
        modalities=tuple(cfg["data"].get("modalities", ["CT1", "T1", "T2", "FLAIR"])),
        min_visits=cfg["data"].get("min_visits", 2))
    size = tuple(cfg["preprocessing"].get("target_size", [96, 96, 96]))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=make_collate(size))

    model = JEPAWorldModel(cfg)
    model.load_state_dict(torch.load(args.champion, map_location="cpu")["model"])
    model.eval()

    errs, y_bin, y_4 = [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            e = 1 - F.cosine_similarity(out["z_hat"][0], out["z_target"][0], dim=-1)
            pid = batch["patient_id"][0]
            item = ds[ds.patients.index(pid)]
            for t in range(len(e)):
                if not bool(out["valid"][0, t]):
                    continue
                rating = ds.rano.get((pid, item["visits"][t + 1]), "")
                if rating not in RANO_PROBE_MAP:
                    continue
                errs.append(float(e[t]))
                y_4.append(RANO_PROBE_MAP[rating])
                y_bin.append(1 if rating == "PD" else 0)
    errs, y_bin, y_4 = map(torch.tensor, (errs, y_bin, y_4))
    print(f"pairs: {len(errs)} PD-rate={y_bin.float().mean():.3f}")
    names = ["PD", "SD", "PR", "CR"]
    for k, nm in enumerate(names):
        m = y_4 == k
        print(f"mean err {nm}: {errs[m].mean():.4f} (n={int(m.sum())})")
    print(f"AUC(err -> next-visit PD): {auc_mann_whitney(errs, y_bin):.4f}")


if __name__ == "__main__":
    main()
