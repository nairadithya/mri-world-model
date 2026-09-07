"""Lead-time surprise (R1) + transition error atlas (R2). Frozen champion, CPU.

R1: does JEPA prediction error anticipate *future* PD? For each valid
(t -> t+1) pair with error e_t, the k-step label is incident PD newly
appearing at visit t+k: labels[t+1 .. t+k] all clean, label[t+k] == PD, no
PD in between. k=1 reproduces the contemporaneous A10 AUC (0.7677) as a
harness check. Persistence error on the identical pairs is the control:
JEPA must beat it at k>=1 to claim anticipation beyond change-detection.

R2: per-pair errors by (RANO_t -> RANO_{t+1}) transition class (16 clean
cells + dirty/operative-involved rows): means, counts, JEPA vs persistence.
Designs R13's transition weights empirically; quantifies the G15
surgery-transition anomaly (massive change at ~1-day model gaps).

Usage: python scripts/leadtime.py --champion checkpoints/champion_0.0081.pt
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data.collate import make_collate
from src.data.dataset import LUMIEREDataset
from src.model.jepa_model import JEPAWorldModel

from probe_rano import RANO_PROBE_MAP  # noqa: E402
from surprise_signal import auc_mann_whitney  # noqa: E402

NAMES = ["PD", "SD", "PR", "CR"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--champion", default="checkpoints/champion_0.0081.pt")
    ap.add_argument("--max-k", type=int, default=3)
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
    ckpt = torch.load(args.champion, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # per-patient: full visit label array + valid-pair rows (with pair index
    # t; rows need NOT be consecutive — invalid pairs are skipped, so future
    # labels are read from the visit array, never chained across rows).
    plab = []
    pats = []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            out = model(batch)
            n = int(batch["n_visits"][0])
            # Target endpoints without a second full encode: the forward
            # already computed visits 1..n-1 (out["z_target"]); only visit 0
            # needs encoding (~1 visit instead of ~n).
            z0 = model.encode_target_visit(
                batch["mri"][:, :1], batch["mri_mask"][:, :1])[0, :1]
            z_full = torch.cat([z0, out["z_target"][0][:n - 1]], dim=0)
            je = 1 - F.cosine_similarity(out["z_hat"][0], out["z_target"][0], dim=-1)
            pid = batch["patient_id"][0]
            item = ds[ds.patients.index(pid)]
            labs = []
            for t in range(n):
                labs.append(RANO_PROBE_MAP.get(
                    ds.rano.get((pid, item["visits"][t]), ""), -1))
            rows = []
            for t in range(len(je)):
                if not bool(out["valid"][0, t]):
                    continue
                pe = 1 - (F.normalize(z_full[t], dim=0) *
                          F.normalize(z_full[t + 1], dim=0)).sum().item()
                rows.append((t, float(je[t]), pe,
                             float(batch["time_deltas"][0, t + 1]),
                             labs[t], labs[t + 1]))
            plab.append(labs)
            pats.append(rows)
            if (bi + 1) % 20 == 0:
                print(f"{bi + 1}/{len(ds)} patients", flush=True)

    # ---- R1: lead-time AUCs ----
    print("\n== R1: incident-PD AUC by lead k (JEPA err vs persistence err) ==")
    print(f"{'k':>3} {'pairs':>7} {'PD-rate':>8} {'JEPA-AUC':>9} {'pers-AUC':>9}")
    for k in range(1, args.max_k + 1):
        ej, ep, yb = [], [], []
        for rows, seq in zip(pats, plab):
            for (t, je_, pe_, _gap, _a, _b) in rows:
                fut = seq[t + 1:t + 1 + k]
                if len(fut) < k or any(v < 0 for v in fut):
                    continue
                if any(v == 0 for v in fut[:-1]):
                    continue  # PD already present -> not incident
                ej.append(je_)
                ep.append(pe_)
                yb.append(1 if fut[-1] == 0 else 0)
        ejt, ept, ybt = map(torch.tensor, (ej, ep, yb))
        if len(ybt) and ybt.sum() > 0 and ybt.sum() < len(ybt):
            aj = auc_mann_whitney(ejt, ybt)
            ap_ = auc_mann_whitney(ept, ybt)
            print(f"{k:>3} {len(ybt):>7} {ybt.float().mean():>8.3f} "
                  f"{aj:>9.4f} {ap_:>9.4f}")
        else:
            print(f"{k:>3} {len(ybt):>7} {'n/a':>8} {'n/a':>9} {'n/a':>9}")

    # ---- R2: transition atlas ----
    print("\n== R2: error by (RANO_t -> RANO_{t+1}) transition ==")
    print(f"{'trans':>9} {'n':>6} {'jepa':>8} {'persist':>8} {'gap_med':>8}")
    cells: dict[tuple, tuple[list, list, list]] = {}
    for rows in pats:
        for (_t, je_, pe_, gap_, a, b) in rows:
            key = (NAMES[a] if a >= 0 else "X", NAMES[b] if b >= 0 else "X")
            cells.setdefault(key, ([], [], []))
            cells[key][0].append(je_)
            cells[key][1].append(pe_)
            cells[key][2].append(gap_)
    for key in sorted(cells, key=lambda k: -len(cells[k][0])):
        je_, pe_, gap_ = cells[key]
        print(f"{key[0]+'>'+key[1]:>9} {len(je_):>6} "
              f"{statistics.mean(je_):>8.4f} {statistics.mean(pe_):>8.4f} "
              f"{statistics.median(gap_):>8.0f}")


if __name__ == "__main__":
    main()
