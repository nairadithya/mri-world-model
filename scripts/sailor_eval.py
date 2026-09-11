"""SAILOR cross-site eval (D26): frozen champion, zero training.

Phase --encode: SAILOR visits -> per-patient cache (vision/fused/states +
raw RANO codes mapped to probe labels {1:0,2:1,3:2,5:3}).
Phase --eval:
  (a) JEPA-vs-persistence over all SAILOR pairs, SAME EMA-target space both
      sides (K3-2/D28; the old row used the online projector over
      encode_visits and was mixed-space);
  (b) LUMIERE-trained forecast probes applied to SAILOR rows (transfer)
      + SAILOR-fit probe (ceiling);
  (d) surprise-AUC with SAILOR RANO.
(c) volume probes: follow-up (ONCO masks).
--pairs: only the (a) row (no transfer/surprise passes).

Usage:
    python scripts/sailor_eval.py --champion checkpoints/champion_0.0081.pt --encode
    python scripts/sailor_eval.py --pairs
    python scripts/sailor_eval.py --eval --lum-cache checkpoints/probe_cache.pt
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data.collate import make_collate
from src.data.sailor import SAILORDataset
from src.model.jepa_model import JEPAWorldModel

from probe_rano import RANO_PROBE_NAMES, fit_linear, rows_for, scores  # noqa: E402

SAILOR_ROOT = "data/sailor/sailor_ebrains_pseud/derivatives/mni2009c-n-s"
# SAILOR numeric code -> probe label {PD:0, SD:1, PR:2, CR:3} (D26 codebook).
SAILOR_PROBE_MAP = {1: 0, 2: 1, 3: 2, 5: 3}


def encode_sailor(cfg, champion_path, cache_path, root=SAILOR_ROOT):
    ds = SAILORDataset(root)
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
            v = model.encode_visits(batch["mri"], batch["mri_mask"])
            c = model.clinical(batch["clinical"])
            tok = model.fusion(v, c.unsqueeze(1).expand(-1, v.shape[1], -1))
            states, _ = model.temporal.forward_prefixes(
                tok, batch["time_deltas"], batch["visit_mask"])
            n = int(batch["n_visits"][0])
            item = ds[ds.subjects.index(pid)]
            labels = [SAILOR_PROBE_MAP.get(ds.sailor_rano.get((pid, s)), -1)
                      for s in item["visits"]]
            cache["patients"][pid] = {
                "split": "sailor", "vision": v[0, :n].clone(),
                "fused": tok[0, :n].clone(), "clinical": c[0].clone(),
                "labels": torch.tensor(labels, dtype=torch.long),
                "states": states[0, :n - 1].clone(),
            }
            el = time.time() - t0
            print(f"encoded {i + 1}/{len(ds)} ({el / (i + 1):.1f}s/subject)",
                  flush=True)
    print(f"cache -> {cache_path}")
    torch.save(cache, cache_path)


def _pairs_same_space(model, loader):
    """(a) JEPA vs persistence, SAME EMA-target space on both sides (K3-2).

    JEPA = 1 - cos(predictor(state_t), EMA_target_{t+1}); persistence =
    1 - cos(EMA_target_t, EMA_target_{t+1}). The old row compared the EMA
    JEPA error against `PersistenceBaseline(online projector)` over
    `encode_visits` (online backbone) — mixed space, the original-A8 error.
    """
    je, pe, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            valid = out["valid"][0]
            if not bool(valid.any()):
                continue
            z = model.encode_target_visit(batch["mri"], batch["mri_mask"])[0]
            je_t = 1 - F.cosine_similarity(out["z_hat"][0], out["z_target"][0],
                                           dim=-1)
            pe_t = 1 - F.cosine_similarity(z[:-1], z[1:], dim=-1)
            nv = int(valid.sum())
            je += float(je_t[valid].sum())
            pe += float(pe_t[valid].sum())
            n += nv
    return je / n, pe / n, n


def eval_pairs(cfg, champion_path, root=SAILOR_ROOT):
    """Quick mode: only the same-space (a) row (no transfer/surprise passes)."""
    ds = SAILORDataset(root)
    size = tuple(cfg["preprocessing"].get("target_size", [96, 96, 96]))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=make_collate(size))
    model = JEPAWorldModel(cfg)
    model.load_state_dict(torch.load(champion_path, map_location="cpu")["model"],
                          strict=False)
    model.eval()
    je, pe, n = _pairs_same_space(model, loader)
    print(f"(a) SAILOR pairs (same-space): n={n} "
          f"JEPA={je:.4f} persist={pe:.4f}")
    return je, pe, n


def eval_all(cfg, champion_path, cache_path, lum_cache_path, root=SAILOR_ROOT):
    device = torch.device("cpu")
    ds = SAILORDataset(root)
    size = tuple(cfg["preprocessing"].get("target_size", [96, 96, 96]))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=make_collate(size))
    model = JEPAWorldModel(cfg)
    model.load_state_dict(torch.load(champion_path, map_location="cpu")["model"],
                          strict=False)
    model.eval()

    je, pe, n = _pairs_same_space(model, loader)
    print(f"(a) SAILOR pairs (same-space): n={n} "
          f"JEPA={je:.4f} persist={pe:.4f}")

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)["patients"]
    lum = torch.load(lum_cache_path, map_location="cpu", weights_only=False)["patients"]
    lsplit = {}
    for pid, p in lum.items():
        lsplit.setdefault(p["split"], []).append(pid)
    sp = {"sailor": sorted(cache)}
    # (b) transfer: LUMIERE-train forecast-mlp -> SAILOR rows
    x_tr, y_tr = rows_for(lum, lsplit, ["train"], "states_forecast")
    x_sa, y_sa = rows_for(cache, sp, ["sailor"], "states_forecast")
    net = fit_linear(x_tr, y_tr, hidden=256)
    acc, f1, rec, cm = scores(net, x_sa, y_sa)
    maj = torch.bincount(y_sa, minlength=4).argmax().item()
    print(f"(b) transfer LUM->SAILOR forecast-mlp: n={len(y_sa)} acc={acc:.4f} "
          f"(maj {RANO_PROBE_NAMES[maj]} {(y_sa == maj).float().mean():.4f}) macro-F1={f1:.4f}")
    print(f"    recall={ {k: round(v, 3) for k, v in rec.items()} }")
    # (b2) SAILOR-fit ceiling: train on SAILOR rows themselves (optimistic)
    net2 = fit_linear(x_sa, y_sa, hidden=256)
    acc2, f12, _, _ = scores(net2, x_sa, y_sa)
    print(f"(b2) SAILOR-fit (train=test, ceiling): acc={acc2:.4f} macro-F1={f12:.4f}")
    # (d) surprise-AUC with SAILOR RANO
    from surprise_signal import auc_mann_whitney  # noqa: E402
    errs, yb = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            e = 1 - F.cosine_similarity(out["z_hat"][0], out["z_target"][0], dim=-1)
            pid = batch["patient_id"][0]
            item = ds[ds.subjects.index(pid)]
            for t in range(len(e)):
                if not bool(out["valid"][0, t]):
                    continue
                code = ds.sailor_rano.get((pid, item["visits"][t + 1]))
                if code not in SAILOR_PROBE_MAP:
                    continue
                errs.append(float(e[t]))
                yb.append(1 if code == 1 else 0)
    errs, yb = torch.tensor(errs), torch.tensor(yb)
    print(f"(d) SAILOR surprise-AUC(err->PD): {auc_mann_whitney(errs, yb):.4f} "
          f"(n={len(errs)} PD-rate={yb.float().mean():.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--champion", default="checkpoints/champion_0.0081.pt")
    ap.add_argument("--cache", default="checkpoints/sailor_cache.pt")
    ap.add_argument("--lum-cache", default="checkpoints/probe_cache.pt")
    ap.add_argument("--encode", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--pairs", action="store_true",
                    help="only the same-space (a) JEPA/persist row")
    ap.add_argument("--root", default=SAILOR_ROOT)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.encode:
        encode_sailor(cfg, args.champion, args.cache, args.root)
    if args.pairs:
        eval_pairs(cfg, args.champion, args.root)
    if args.eval:
        eval_all(cfg, args.champion, args.cache, args.lum_cache, args.root)


if __name__ == "__main__":
    main()
