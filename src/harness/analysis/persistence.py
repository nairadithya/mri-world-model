"""Persistence-error baselines for the surprise-as-signal AUCs.

The open confound: does JEPA prediction error anticipate progression, or does
*any* change measure (e.g. trivial persistence error = distance from last
visit) do the same? This scores 1 - cos(z_t, z_{t+1}) as the PD signal on the
identical pair sets and compares against the reported JEPA-error AUCs
(LUMIERE 0.77, SAILOR 0.87).

  --lumiere       : from horizon_cache (target-space z) + probe_cache (labels)
  --sailor-encode : cache SAILOR target-space z (CPU, ~30 min, one-time)
  --sailor        : persistence AUC from the z cache (PD = raw code 1)

Usage:
    python scripts/harness.py run persistence --lumiere
    python scripts/harness.py run persistence --sailor-encode
    python scripts/harness.py run persistence --sailor
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.data.collate import make_collate
from src.data.sailor import SAILORDataset
from src.model.jepa_model import JEPAWorldModel

from .surprise import auc_mann_whitney  # noqa: E402

SAILOR_ROOT = "data/sailor/sailor_ebrains_pseud/derivatives/mni2009c-n-s"


def lumiere(horizon_path, probe_path):
    hz = torch.load(horizon_path, map_location="cpu",
                    weights_only=False)["patients"]
    pr = torch.load(probe_path, map_location="cpu",
                    weights_only=False)["patients"]
    errs, yb, yc = [], [], []
    for pid, p in hz.items():
        lab = pr[pid]["labels"]
        assert len(lab) == len(p["z"]), pid
        T = len(p["z"])
        for t in range(T - 1):
            if not p["has_img"][t] or not p["has_img"][t + 1]:
                continue
            if int(lab[t + 1]) < 0:
                continue
            e = 1 - (F.normalize(p["z"][t], dim=0) *
                     F.normalize(p["z"][t + 1], dim=0)).sum().item()
            errs.append(e)
            yc.append(int(lab[t + 1]))
            yb.append(1 if int(lab[t + 1]) == 0 else 0)  # probe label 0 = PD
    errs, yb, yc = map(torch.tensor, (errs, yb, yc))
    print(f"LUMIERE pairs: {len(errs)} PD-rate={yb.float().mean():.3f}")
    for k, nm in enumerate(["PD", "SD", "PR", "CR"]):
        m = yc == k
        print(f"mean persistence err {nm}: {errs[m].mean():.4f} (n={int(m.sum())})")
    print(f"AUC(persistence-err -> next-visit PD): "
          f"{auc_mann_whitney(errs, yb):.4f}  [JEPA-err reference: 0.7677]")


def lumiere_probe(probe_path):
    """Run the persistence control from the canonical current probe cache."""
    patients = torch.load(probe_path, map_location="cpu",
                          weights_only=False)["patients"]
    errs, yb, yc = [], [], []
    for p in patients.values():
        z, lab = p["z"], p["labels"]
        T = min(len(z), len(lab), len(p["has_img"]))
        for t in range(T - 1):
            if not p["has_img"][t] or not p["has_img"][t + 1]:
                continue
            if int(lab[t + 1]) < 0:
                continue
            e = 1 - (F.normalize(z[t], dim=0) *
                     F.normalize(z[t + 1], dim=0)).sum().item()
            errs.append(e)
            yc.append(int(lab[t + 1]))
            yb.append(1 if int(lab[t + 1]) == 0 else 0)
    errs, yb, yc = map(torch.tensor, (errs, yb, yc))
    print(f"LUMIERE pairs: {len(errs)} PD-rate={yb.float().mean():.3f}")
    for k, nm in enumerate(["PD", "SD", "PR", "CR"]):
        m = yc == k
        print(f"mean persistence err {nm}: {errs[m].mean():.4f} (n={int(m.sum())})")
    print(f"AUC(persistence-err -> next-visit PD): "
          f"{auc_mann_whitney(errs, yb):.4f}  [canonical probe cache]")


def sailor_encode(cfg, champion_path, cache_path):
    ds = SAILORDataset(SAILOR_ROOT)
    size = tuple(cfg["preprocessing"].get("target_size", [96, 96, 96]))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=make_collate(size))
    model = JEPAWorldModel(cfg)
    ckpt = torch.load(champion_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    cache = {"patients": {}}
    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            pid = batch["patient_id"][0]
            n = int(batch["n_visits"][0])
            z = model.encode_target_visit(
                batch["mri"], batch["mri_mask"])[0, :n].clone()
            item = ds[ds.subjects.index(pid)]
            codes = [ds.sailor_rano.get((pid, s)) for s in item["visits"]]
            cache["patients"][pid] = {
                "z": z,
                "visits": item["visits"][:n],
                "codes": codes[:n],
                "has_img": batch["mri_mask"][0, :n].any(dim=-1).clone(),
            }
            el = time.time() - t0
            print(f"encoded {i + 1}/{len(ds)} ({el / (i + 1):.1f}s/subject)",
                  flush=True)
    torch.save(cache, cache_path)
    print(f"cache -> {cache_path}")


def sailor(cache_path):
    cache = torch.load(cache_path, map_location="cpu",
                       weights_only=False)["patients"]
    errs, yb = [], []
    per_class: dict[int, list] = {}
    for pid, p in cache.items():
        T = len(p["z"])
        for t in range(T - 1):
            if not p["has_img"][t] or not p["has_img"][t + 1]:
                continue
            code = p["codes"][t + 1]
            if code not in (1, 2, 3, 5):
                continue
            e = 1 - (F.normalize(p["z"][t], dim=0) *
                     F.normalize(p["z"][t + 1], dim=0)).sum().item()
            errs.append(e)
            yb.append(1 if code == 1 else 0)
            per_class.setdefault(code, []).append(e)
    errs, yb = map(torch.tensor, (errs, yb))
    print(f"SAILOR pairs: {len(errs)} PD-rate={yb.float().mean():.3f}")
    for code, nm in [(1, "PD"), (2, "SD"), (3, "PR"), (5, "CR")]:
        ee = per_class.get(code, [])
        if ee:
            print(f"mean persistence err {nm}: "
                  f"{torch.tensor(ee).mean():.4f} (n={len(ee)})")
    print(f"AUC(persistence-err -> next-visit PD): "
          f"{auc_mann_whitney(errs, yb):.4f}  [JEPA-err reference: 0.87]")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--champion", default="checkpoints/champion_0.0081.pt")
    ap.add_argument("--horizon-cache", default="checkpoints/horizon_cache.pt")
    ap.add_argument("--probe-cache", default="checkpoints/probe_cache.pt")
    ap.add_argument("--z-cache", default="checkpoints/sailor_z_cache.pt")
    ap.add_argument("--lumiere", action="store_true")
    ap.add_argument("--sailor-encode", action="store_true")
    ap.add_argument("--sailor", action="store_true")
    args = ap.parse_args(argv)
    if args.lumiere:
        probe = torch.load(args.probe_cache, map_location="cpu",
                           weights_only=False)
        first = next(iter(probe.get("patients", {}).values()), {})
        if probe.get("schema_version") == 2 and "z" in first:
            lumiere_probe(args.probe_cache)
        else:
            lumiere(args.horizon_cache, args.probe_cache)
        return
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.sailor_encode:
        sailor_encode(cfg, args.champion, args.z_cache)
    if args.sailor:
        sailor(args.z_cache)


if __name__ == "__main__":
    main()
