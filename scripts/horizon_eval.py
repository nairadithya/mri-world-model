"""Per-horizon JEPA vs persistence eval for a horizon-trained checkpoint (CPU).

Loads best.pt with horizon enabled, runs forward per patient, groups the
model's own (t, t+n) pair errors by n against persistence (z_t as prediction)
on the identical pair set. Split-aware via the standard patient splits.

Usage: .venv/bin/python scripts/horizon_eval.py --ckpt outputs/horizon-leg/checkpoints/best.pt
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
from scripts.probe_rano import build_datasets
from src.data.collate import make_collate
from src.model.jepa_model import JEPAWorldModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg["model"]["predictor"].setdefault("horizon", {})["enabled"] = True
    size = tuple(cfg["preprocessing"].get("target_size", [96, 96, 96]))
    collate = make_collate(size)
    datasets, _ = build_datasets(cfg)

    model = JEPAWorldModel(cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    print(f"ckpt epoch {ckpt.get('epoch')}, val {ckpt.get('val_loss')}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    by = {}  # (n, split) -> [jepa_err], [persist_err]
    done = total = sum(len(ds) for ds in datasets.values())
    with torch.no_grad():
        for split, ds in datasets.items():
            loader = DataLoader(ds, batch_size=1, shuffle=False,
                                num_workers=0, collate_fn=collate)
            for batch in loader:
                out = model(batch)
                hz = out["horizon"]
                done_cur = True
                if hz is not None:
                    pe = 1 - (F.normalize(hz["zt"], dim=-1) *
                              F.normalize(hz["zu"], dim=-1)).sum(dim=-1)
                    for n_, je, pe_ in zip(hz["n"].tolist(), hz["err"].tolist(),
                                           pe.tolist()):
                        by.setdefault((int(n_), split), [[], []])
                        by[(int(n_), split)][0].append(je)
                        by[(int(n_), split)][1].append(pe_)
                total -= 1
                if total % 10 == 0:
                    print(f"{done - total}/{done} patients", flush=True)
    print(f"{'n':>4} {'split':>6} {'pairs':>7} {'jepa':>8} {'persist':>8}")
    for (n_, s) in sorted(by):
        je, pe = by[(n_, s)]
        j = sum(je) / len(je)
        p = sum(pe) / len(pe)
        print(f"{n_:>4} {s:>6} {len(je):>7} {j:>8.4f} {p:>8.4f}"
              f"{' <-- wins' if j < p else ''}")


if __name__ == "__main__":
    main()
