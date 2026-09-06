"""Volume probes (D24.b): tumour volumetry from frozen latents.

Labels: DeepBraTumIA `measured_volumes_in_mm3.json` (auto-masks, 599 studies).
Tasks (all log-mm3, closed-form least squares, hero splits):
  readout:   log(vol_t) from fused_t            (does the latent hold size?)
  forecast:  log(vol_{t+1}) from state_t        (dynamics → future burden)
Baselines: mean (readout), persistence vol_t (forecast).

Usage: python scripts/volume_probe.py --cache checkpoints/probe_cache.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import LUMIEREDataset


def load_volumes(root="data/autoseg/vols/Imaging"):
    vols = {}
    for p in os.listdir(root):
        pdir = os.path.join(root, p)
        if not os.path.isdir(pdir):
            continue
        for v in os.listdir(pdir):
            jf = os.path.join(pdir, v, "DeepBraTumIA-segmentation/atlas/"
                              "segmentation/measured_volumes_in_mm3.json")
            if os.path.exists(jf):
                d = json.load(open(jf))
                tot = sum(float(x) for x in d.values())
                vols[(p, v)] = {"total": tot, "enhancing": float(d.get("Enhancing_Core", 0.0))}
    return vols


def fit_ridge_cv(x_tr, y_tr, lams=(0.1, 1.0, 10.0, 100.0, 1000.0), seed=42):
    """Ridge with standardized features; λ picked by 5-fold CV on train rows."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(x_tr), generator=g)
    folds = [idx[i::5] for i in range(5)]
    mu, sd = x_tr.mean(0), x_tr.std(0).clamp_min(1e-6)
    z = (x_tr - mu) / sd
    d = z.shape[1]
    I = torch.eye(d)
    best, best_mse = lams[0], float("inf")
    for lam in lams:
        mses = []
        for i in range(5):
            te, tr = folds[i], torch.cat([folds[j] for j in range(5) if j != i])
            w = torch.linalg.solve(z[tr].T @ z[tr] + lam * I, z[tr].T @ y_tr[tr].unsqueeze(1))
            pred = (z[te] @ w).squeeze(1)
            mses.append(((y_tr[te] - pred) ** 2).mean().item())
        m = sum(mses) / 5
        if m < best_mse:
            best, best_mse = lam, m
    w = torch.linalg.solve(z.T @ z + best * I, z.T @ y_tr.unsqueeze(1)).squeeze(1)
    b = y_tr.mean() - (mu / sd * w).sum()
    return w / sd, b, best


def report(name, y_te, pred, base):
    mae = (y_te - pred).abs().mean().item()
    mae_b = (y_te - base).abs().mean().item()
    ss_res = ((y_te - pred) ** 2).sum().item()
    ss_tot = ((y_te - y_te.mean()) ** 2).sum().item()
    print(f"{name}: n={len(y_te)} MAE={mae:.4f} (base {mae_b:.4f}) "
          f"R2={1 - ss_res / max(1e-9, ss_tot):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="checkpoints/probe_cache.pt")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--vols", default="data/autoseg/vols/Imaging")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    vols = load_volumes(args.vols)
    print(f"volume studies: {len(vols)}")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)["patients"]
    ds = LUMIEREDataset(meta_dir=cfg["data"]["meta_dir"],
                        processed_root=cfg["data"]["root"],
                        raw_root=cfg["data"].get("raw_root"))

    rows = []  # (split, fused_t, state_t_or_None, logvol_t, logvol_next_or_None)
    for pid, p in cache.items():
        item = ds[ds.patients.index(pid)]
        names = item["visits"]
        assert len(names) == len(p["fused"]), (pid, len(names), len(p["fused"]))
        vv = [vols.get((pid, v)) for v in names]
        st = p.get("states")
        for t in range(len(names)):
            if vv[t] is None:
                continue
            lv = torch.log(torch.tensor(vv[t]["total"]) + 1.0)
            nxt = None
            if t + 1 < len(names) and vv[t + 1] is not None:
                nxt = torch.log(torch.tensor(vv[t + 1]["total"]) + 1.0)
            rows.append((p["split"], p["fused"][t], st[t] if st is not None and t < len(st) else None,
                         lv, nxt))
    print(f"rows with volumes: {len(rows)}")

    for key in ["total"]:
        # readout: fused_t -> logvol_t
        tr = [(f, lv) for s, f, _, lv, _ in rows if s == "train"]
        te = [(f, lv) for s, f, _, lv, _ in rows if s == "test"]
        x_tr, y_tr = torch.stack([r[0] for r in tr]), torch.tensor([r[1] for r in tr])
        x_te, y_te = torch.stack([r[0] for r in te]), torch.tensor([r[1] for r in te])
        w, b, lam = fit_ridge_cv(x_tr, y_tr)
        report(f"readout log-{key} from fused [lam={lam}]", y_te, x_te @ w + b,
               y_tr.mean().expand_as(y_te))
        # forecast: state_t -> logvol_{t+1}, baseline persistence
        tr = [(st, lv, nxt) for s, _, st, lv, nxt in rows
              if s == "train" and st is not None and nxt is not None]
        te = [(st, lv, nxt) for s, _, st, lv, nxt in rows
              if s == "test" and st is not None and nxt is not None]
        x_tr = torch.stack([r[0] for r in tr])
        y_tr, c_tr = torch.tensor([r[2] for r in tr]), torch.tensor([r[1] for r in tr])
        x_te = torch.stack([r[0] for r in te])
        y_te, c_te = torch.tensor([r[2] for r in te]), torch.tensor([r[1] for r in te])
        w, b, lam = fit_ridge_cv(x_tr, y_tr)
        report(f"forecast log-{key} from state [lam={lam}]", y_te, x_te @ w + b, c_te)


if __name__ == "__main__":
    main()
