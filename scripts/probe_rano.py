"""Frozen RANO probe (D23): does the champion representation encode progression?

Phase 1 --encode: freeze champion, encode every visit (vision 768-d +
fused 1152-d latents), cache per-patient tensors + clean RANO labels.
Phase 2 --probe: train linear/MLP probes on train-split patients, score
macro-F1/accuracy/confusion on test-split patients, with majority and
clinical-only baselines.

Usage:
    python scripts/probe_rano.py --champion <best.pt> --cache probe_cache.pt --encode
    python scripts/probe_rano.py --cache probe_cache.pt --probe
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.collate import make_collate
from src.data.dataset import LUMIEREDataset
from src.data.splits import patient_splits
from src.model.jepa_model import JEPAWorldModel

# Clean RANO response classes only (operative states + missing excluded).
RANO_PROBE_MAP = {"PD": 0, "SD": 1, "PR": 2, "CR": 3}
RANO_PROBE_NAMES = ["PD", "SD", "PR", "CR"]


def build_datasets(cfg):
    meta_dir = cfg["data"]["meta_dir"]
    demo_csv = next(os.path.join(meta_dir, f) for f in os.listdir(meta_dir)
                    if f.startswith("demographics"))
    splits = patient_splits(demo_csv,
                            train_frac=cfg["data"].get("train_split", 0.7),
                            val_frac=cfg["data"].get("val_split", 0.15),
                            seed=cfg["data"].get("seed", 42))
    common = dict(
        meta_dir=meta_dir,
        processed_root=cfg["data"]["root"],
        raw_root=cfg["data"].get("raw_root"),
        modalities=tuple(cfg["data"].get("modalities", ["CT1", "T1", "T2", "FLAIR"])),
        min_visits=cfg["data"].get("min_visits", 2),
    )
    datasets = {k: LUMIEREDataset(patients=v, **common) for k, v in splits.items()}
    return datasets, splits


def encode_all(cfg, champion_path, cache_path):
    device = torch.device("cpu")
    datasets, splits = build_datasets(cfg)
    size = tuple(cfg["preprocessing"].get("target_size", [96, 96, 96]))
    collate = make_collate(size)

    model = JEPAWorldModel(cfg)
    ckpt = torch.load(champion_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"champion: {champion_path} (epoch {ckpt.get('epoch')}, "
          f"val {ckpt.get('val_loss')})")

    cache = {"patients": {}}
    t0 = time.time()
    done = 0
    total = sum(len(ds) for ds in datasets.values())
    with torch.no_grad():
        for split, ds in datasets.items():
            loader = DataLoader(ds, batch_size=1, shuffle=False,
                                num_workers=0, collate_fn=collate)
            for batch in loader:
                pid = batch["patient_id"][0]
                item = ds[ds.patients.index(pid)]
                v = model.encode_visits(batch["mri"], batch["mri_mask"])  # (1,T,768)
                c = model.clinical(batch["clinical"])                     # (1,384)
                tok = model.fusion(v, c.unsqueeze(1).expand(-1, v.shape[1], -1))
                n = int(batch["n_visits"][0])
                labels = []
                for t in range(n):
                    rating = ds.rano.get((pid, item["visits"][t]), "")
                    labels.append(RANO_PROBE_MAP.get(rating, -1))
                cache["patients"][pid] = {
                    "split": split,
                    "vision": v[0, :n].clone(),
                    "fused": tok[0, :n].clone(),
                    "clinical": c[0].clone(),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "n_labeled": sum(l >= 0 for l in labels),
                }
                # Temporal states: state s_t summarizes visits 0..t.
                # Cached alongside so probes test the history summary
                # (thesis: RANO/volume read out from temporal latent).
                states, _ = model.temporal.forward_prefixes(
                    tok, batch["time_deltas"], batch["visit_mask"])
                cache["patients"][pid]["states"] = states[0, :n - 1].clone()
                done += 1
                if done == 1 or done % 10 == 0 or done == total:
                    el = time.time() - t0
                    print(f"encoded {done}/{total} ({el / done:.1f}s/patient, "
                          f"ETA {el / done * (total - done) / 60:.0f}min)",
                          flush=True)
    n_lab = sum(int((p["labels"] >= 0).sum()) for p in cache["patients"].values())
    print(f"cache: {total} patients, {n_lab} RANO-labelled visits -> {cache_path}")
    torch.save(cache, cache_path)


def rows_for(patients, splits, split_names, feat):
    """Feature rows + clean RANO labels for a probe config.

    Snapshot feats (fused/vision/clinical): one row per labelled visit.
    states_current: row s_t (history<=t) with label RANO_t.
    states_forecast: row s_t with label RANO_{t+1} (future status).
    """
    xs, ys = [], []
    wanted = [split_names] if isinstance(split_names, str) else split_names
    for pid in [p for s in wanted for p in splits[s]]:
        p = patients[pid]
        if feat in ("states_current", "states_forecast"):
            off = 0 if feat == "states_current" else 1
            idx = [t for t in range(len(p["states"])) if p["labels"][t + off] >= 0]
            if not idx:
                continue
            xs.append(p["states"][idx])
            ys.append(p["labels"][[t + off for t in idx]])
        else:
            keep = p["labels"] >= 0
            if feat == "fused":
                xs.append(p["fused"][keep])
            elif feat == "vision":
                xs.append(p["vision"][keep])
            else:  # clinical: broadcast static vector per labelled visit
                xs.append(p["clinical"].unsqueeze(0).expand(int(keep.sum()), -1))
            ys.append(p["labels"][keep])
    return torch.cat(xs), torch.cat(ys)


def fit_linear(x_tr, y_tr, n_cls=4, hidden=0, steps=500, lr=1e-2, seed=42):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    if hidden:
        net = nn.Sequential(nn.Linear(x_tr.shape[1], hidden), nn.GELU(),
                            nn.Linear(hidden, n_cls))
    else:
        net = nn.Linear(x_tr.shape[1], n_cls)
    counts = torch.bincount(y_tr, minlength=n_cls).float().clamp_min(1)
    weight = counts.sum() / (n_cls * counts)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        idx = torch.randperm(len(x_tr), generator=g)
        opt.zero_grad()
        loss = nn.functional.cross_entropy(net(x_tr[idx]), y_tr[idx], weight=weight)
        loss.backward()
        opt.step()
    return net


def scores(net, x, y):
    with torch.no_grad():
        pred = net(x).argmax(1)
    acc = (pred == y).float().mean().item()
    f1s, recs, cm = [], [], torch.zeros(4, 4, dtype=torch.long)
    for k in range(4):
        tp = int(((pred == k) & (y == k)).sum())
        fp = int(((pred == k) & (y != k)).sum())
        fn = int(((pred != k) & (y == k)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1s.append(2 * prec * rec / max(1e-9, prec + rec))
        recs.append(rec)
        for j in range(4):
            cm[k, j] = int(((y == k) & (pred == j)).sum())
    return acc, sum(f1s) / 4, dict(zip(RANO_PROBE_NAMES, recs)), cm


def run_probe(cache_path):
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    patients = cache["patients"]
    splits = {}
    for pid, p in patients.items():
        splits.setdefault(p["split"], []).append(pid)
    n_lab = lambda s: sum(int((patients[p]["labels"] >= 0).sum()) for p in splits[s])
    print(f"labelled visits: train={n_lab('train')} val={n_lab('val')} test={n_lab('test')}")

    maj = torch.bincount(rows_for(cache["patients"], splits, ["test"], "fused")[1],
                         minlength=4).argmax().item()
    _, y_te = rows_for(cache["patients"], splits, ["test"], "fused")
    print(f"majority baseline (predict {RANO_PROBE_NAMES[maj]}): "
          f"acc={(y_te == maj).float().mean():.4f} (macro-F1 ~0 by construction)")

    for feat in ["fused", "vision", "clinical", "states_current", "states_forecast"]:
        x_tr, y_tr = rows_for(cache["patients"], splits, ["train"], feat)
        x_te, y_te = rows_for(cache["patients"], splits, ["test"], feat)
        for hidden in [0, 256]:
            tag = f"{feat}-{'mlp' if hidden else 'linear'}"
            net = fit_linear(x_tr, y_tr, hidden=hidden)
            acc, f1, rec, cm = scores(net, x_te, y_te)
            print(f"{tag}: acc={acc:.4f} macro-F1={f1:.4f} "
                  f"recall={ {k: round(v, 3) for k, v in rec.items()} }")
            print(f"  confusion (true x pred):\n{cm}")


def run_cv(cache_path, k=5, feat="states_forecast", hidden=256, seed=42):
    """k-fold patient-wise CV of one probe config (field protocol)."""
    import random
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    patients = cache["patients"]
    pids = sorted(patients)
    rng = random.Random(seed)
    rng.shuffle(pids)
    folds = [pids[i::k] for i in range(k)]
    accs, f1s = [], []
    for i in range(k):
        te, tr = folds[i], [p for j in range(k) if j != i for p in folds[j]]
        sp = {"tr": tr, "te": te}
        x_tr, y_tr = rows_for(patients, sp, "tr", feat)
        x_te, y_te = rows_for(patients, sp, "te", feat)
        net = fit_linear(x_tr, y_tr, hidden=hidden)
        acc, f1, _, _ = scores(net, x_te, y_te)
        accs.append(acc)
        f1s.append(f1)
        print(f"fold {i}: n_te={len(y_te)} acc={acc:.4f} macro-F1={f1:.4f}", flush=True)
    import statistics
    print(f"CV {feat}-{'mlp' if hidden else 'linear'}: "
          f"acc={statistics.mean(accs):.4f}±{statistics.pstdev(accs):.4f} "
          f"macro-F1={statistics.mean(f1s):.4f}±{statistics.pstdev(f1s):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--champion", default="checkpoints/champion_0.0081.pt")
    ap.add_argument("--cache", default="checkpoints/probe_cache.pt")
    ap.add_argument("--encode", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--cv", action="store_true",
                    help="5-fold patient-wise CV of states_forecast-mlp")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.encode:
        encode_all(cfg, args.champion, args.cache)
    if args.probe or (not args.encode and not args.cv):
        run_probe(args.cache)
    if args.cv:
        run_cv(args.cache)


if __name__ == "__main__":
    main()
