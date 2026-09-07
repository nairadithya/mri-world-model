"""Freezing battery (tests 1, 2, 5, 8). All CPU, all on existing caches.

1. --separability : logistic site probe (LUMIERE vs SAILOR) on frozen z and
   states. Near-perfect separation quantifies the "different neighborhood";
   per-subject mean logits show whether it is uniform or driven by outliers.
2. --coral : CORAL-align SAILOR latents to LUMIERE covariance (fold-wise:
   fit on train folds, apply to held-out, both z and states spaces), then
   score the FROZEN LUMIERE 1-step head on aligned latents. If aligned head
   error drops to persistence, second-order statistics explain the transfer
   failure and no unfreezing is ever needed. Transductive (fit all SAILOR)
   line included as a labeled upper bound.
5. --plhm : image-space consecutive-visit |Δ| (resized 96³, brain mask) for
   SAILOR base-T1c vs LUMIERE CT1. Damped in image space too -> pipeline
   (PLHM suspect stands); similar images but different latents -> encoder
   shift (freezing suspect). Plus a look for non-PLHM SAILOR sources.
8. --interval-match : LUMIERE vs SAILOR pairs binned by matched gap-days:
   latent drift + frozen-head error per bin per cohort. Same gaps, different
   behavior? (follows the A14 gap reframe).

Usage: python scripts/freeze_battery.py --separability --coral --plhm --interval-match
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

from surprise_signal import auc_mann_whitney  # noqa: E402

HZ = "checkpoints/horizon_cache.pt"
FLD = "checkpoints/field_cache.pt"


def load():
    hz = torch.load(HZ, map_location="cpu", weights_only=False)["patients"]
    fl = torch.load(FLD, map_location="cpu", weights_only=False)["patients"]
    return hz, fl


def visit_rows(cache, key="z"):
    """(X, cohort) with one row per imaged visit; SAILOR rows tagged by subject."""
    xs, yh, who = [], [], []
    for pid, p in cache.items():
        for t in range(len(p[key])):
            if key == "z" and not p["has_img"][t]:
                continue
            if key == "states" and (t >= len(p["states"]) or not p["has_img"][t]):
                continue
            xs.append(p[key][t])
            who.append(pid)
    return torch.stack(xs), who


def separability():
    hz, fl = load()
    for key in ("z", "states"):
        xl, _ = visit_rows(hz, key)
        xs, _ = visit_rows(fl, key)
        X = torch.cat([xl, xs])
        y = torch.cat([torch.zeros(len(xl)), torch.ones(len(xs))])
        mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
        Xs = (X - mu) / sd
        torch.manual_seed(0)
        lin = torch.nn.Linear(Xs.shape[1], 1)
        opt = torch.optim.Adam(lin.parameters(), lr=0.1, weight_decay=1.0)
        for _ in range(300):
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                lin(Xs).squeeze(1), y)
            loss.backward()
            opt.step()
        with torch.no_grad():
            s = lin(Xs).squeeze(1)
        acc = ((s > 0) == (y > 0)).float().mean().item()
        print(f"site probe on {key}: acc={acc:.4f} "
              f"AUC={auc_mann_whitney(s, y.long()):.4f} "
              f"(n_lum={len(xl)}, n_sai={len(xs)})")
    # per-subject margin spread on z
    xl, _ = visit_rows(hz, "z")
    xs, ws = visit_rows(fl, "z")
    X = torch.cat([xl, xs])
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    Xs = (X - mu) / sd
    torch.manual_seed(0)
    lin = torch.nn.Linear(Xs.shape[1], 1)
    opt = torch.optim.Adam(lin.parameters(), lr=0.1, weight_decay=1.0)
    y = torch.cat([torch.zeros(len(xl)), torch.ones(len(xs))])
    for _ in range(300):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(
            lin(Xs).squeeze(1), y).backward()
        opt.step()
    with torch.no_grad():
        s = lin(Xs).squeeze(1)[len(xl):]
    per_sub: dict[str, list] = {}
    for w, v in zip(ws, s.tolist()):
        per_sub.setdefault(w, []).append(v)
    m = [sum(v) / len(v) for v in per_sub.values()]
    print(f"SAILOR per-subject mean logit: min={min(m):.2f} "
          f"max={max(m):.2f} (uniform shift if tight, outliers if spread)")


def coral_map(Xs, Xt, eps=1e-2):
    """CORAL: return (A, mu_s, mu_t) mapping source rows onto target geometry.

    Standardize-agnostic closed form on covariances + eps*I shrinkage
    (rank-deficient at 768/1152-d with ~200 rows — shrinkage is load-bearing).
    """
    mu_s, mu_t = Xs.mean(0), Xt.mean(0)
    Cs = torch.cov(Xs.T) + eps * torch.eye(Xs.shape[1])
    Ct = torch.cov(Xt.T) + eps * torch.eye(Xt.shape[1])
    Ds, Vs = torch.linalg.eigh(Cs)
    Dt, Vt = torch.linalg.eigh(Ct)
    Cs_inv_sqrt = (Vs * Ds.clamp_min(1e-12).rsqrt()).mm(Vs.T)
    Ct_sqrt = (Vt * Dt.clamp_min(1e-12).sqrt()).mm(Vt.T)
    return Cs_inv_sqrt.mm(Ct_sqrt), mu_s, mu_t


def coral():
    from split_gate import load_predictor
    hz, fl = load()
    pred = load_predictor("checkpoints/champion_0.0081.pt")
    # reference pools: all LUMIERE imaged visits
    zl, _ = visit_rows(hz, "z")
    sl, _ = visit_rows(hz, "states")
    print(f"{'fold':>4} {'jepa_un':>8} {'pers_un':>8} "
          f"{'jepa_al':>8} {'pers_al':>8}")
    agg = {"ju": [], "pu": [], "ja": [], "pa": [], "jt": []}
    subs = sorted(fl)
    for f in range(5):
        tr = [fl[s] for s in subs if fl[s].get("fold", 0) != f]
        te = [fl[s] for s in subs if fl[s].get("fold", 0) == f]
        zs = torch.cat([p["z"][p["has_img"]] for p in tr])
        ss = torch.cat([p["states"][[t for t in range(len(p["states"]))
                                     if p["has_img"][t]]] for p in tr])
        Az, mz_s, mz_t = coral_map(zs, zl)
        As, ms_s, ms_t = coral_map(ss, sl)
        ju, pu, ja, pa = [], [], [], []
        with torch.no_grad():
            for p in te:
                T = len(p["z"])
                for t in range(T - 1):
                    if not p["has_img"][t] or not p["has_img"][t + 1]:
                        continue
                    zt, zu, st = p["z"][t], p["z"][t + 1], p["states"][t]
                    ju.append(1 - (F.normalize(pred(st.unsqueeze(0)), dim=-1) *
                                   F.normalize(zu.unsqueeze(0), dim=-1)
                                   ).sum().item())
                    pu.append(1 - (F.normalize(zt, dim=0) *
                                   F.normalize(zu, dim=0)).sum().item())
                    zta = (zt - mz_s) @ Az + mz_t
                    zua = (zu - mz_s) @ Az + mz_t
                    sta = (st - ms_s) @ As + ms_t
                    ja.append(1 - (F.normalize(pred(sta.unsqueeze(0)), dim=-1) *
                                   F.normalize(zua.unsqueeze(0), dim=-1)
                                   ).sum().item())
                    pa.append(1 - (F.normalize(zta, dim=0) *
                                   F.normalize(zua, dim=0)).sum().item())
        print(f"{f:>4} {sum(ju)/len(ju):>8.4f} {sum(pu)/len(pu):>8.4f} "
              f"{sum(ja)/len(ja):>8.4f} {sum(pa)/len(pa):>8.4f}")
        agg["ju"] += ju
        agg["pu"] += pu
        agg["ja"] += ja
        agg["pa"] += pa
    # transductive upper bound (fit on ALL sailor, same scoring)
    zs = torch.cat([p["z"][p["has_img"]] for p in fl.values()])
    ss = torch.cat([p["states"][[t for t in range(len(p["states"]))
                                 if p["has_img"][t]]] for p in fl.values()])
    Az, mz_s, mz_t = coral_map(zs, zl)
    As, ms_s, ms_t = coral_map(ss, sl)
    jt = []
    with torch.no_grad():
        for p in fl.values():
            T = len(p["z"])
            for t in range(T - 1):
                if not p["has_img"][t] or not p["has_img"][t + 1]:
                    continue
                sta = (p["states"][t] - ms_s) @ As + ms_t
                zua = (p["z"][t + 1] - mz_s) @ Az + mz_t
                jt.append(1 - (F.normalize(pred(sta.unsqueeze(0)), dim=-1) *
                               F.normalize(zua.unsqueeze(0), dim=-1)
                               ).sum().item())
    agg["jt"] = jt
    m = statistics.mean
    print(f"CV: head-unaligned {m(agg['ju']):.4f} vs persist-unaligned "
          f"{m(agg['pu']):.4f} | head-aligned {m(agg['ja']):.4f} vs "
          f"persist-aligned {m(agg['pa']):.4f} | transductive-head "
          f"{m(agg['jt']):.4f}")


def plhm():
    from src.preprocessing.transforms import runtime_transform
    from src.data.sailor import SAILORDataset
    ds = SAILORDataset(SAILOR_ROOT_PLHM)
    pairs_s, pairs_l = [], []
    for sub in sorted(ds.subjects)[:10]:
        item = ds[ds.subjects.index(sub)]
        v = item["visits"]
        if len(v) >= 3:
            for t in (0, 1):
                a = ds._image_path(sub, v[t], "CT1")
                b = ds._image_path(sub, v[t + 1], "CT1")
                if a and b:
                    pairs_s.append((a, b))
    import glob as _glob
    proot = "data/lumiere_preprocessed"
    pairs_l = []
    for p in sorted(_glob.glob(proot + "/Patient-*")):
        vs = sorted(v for v in _glob.glob(p + "/week-*"
                                          ) if os.path.isdir(v))
        # skip the two earliest visits (peri-operative inflation); take the
        # first later consecutive pair with both CT1 files present
        for a, b in zip(vs[2:-1], vs[3:]):
            fa, fb = os.path.join(a, "CT1.nii.gz"), os.path.join(b, "CT1.nii.gz")
            if os.path.exists(fa) and os.path.exists(fb):
                pairs_l.append((fa, fb))
                break  # one pair per patient
        if len(pairs_l) >= 10:
            break
    print(f"image pairs: SAILOR={len(pairs_s)} LUMIERE={len(pairs_l)} "
          f"(LUMIERE skips 2 earliest visits vs peri-op inflation)")
    for tag, pairs in (("SAILOR", pairs_s), ("LUMIERE", pairs_l)):
        ds_ = []
        for a, b in pairs:
            ta = runtime_transform(a, (96, 96, 96))
            tb = runtime_transform(b, (96, 96, 96))
            mask = (ta != 0) | (tb != 0)
            ds_.append(float((ta - tb).abs()[mask].mean()))
        ds_.sort()
        print(f"{tag}: mean|d| median={ds_[len(ds_)//2]:.4f} "
              f"p10={ds_[len(ds_)//10]:.4f} p90={ds_[9*len(ds_)//10]:.4f} "
              f"(n={len(ds_)})")


SAILOR_ROOT_PLHM = "data/sailor/sailor_ebrains_pseud/derivatives/mni2009c-n-s"


def interval_match():
    from split_gate import load_predictor
    hz, fl = load()
    pred = load_predictor("checkpoints/champion_0.0081.pt")
    BINS = [0, 30, 90, 180, float("inf")]
    LAB = ["0-30d", "30-90d", "90-180d", "180d+"]

    def grab(cache):
        rows = []  # (gap, drift, headerr, cohort)
        with torch.no_grad():
            for pid, p in cache.items():
                T = len(p["z"])
                for t in range(T - 1):
                    if not p["has_img"][t] or not p["has_img"][t + 1]:
                        continue
                    gap = float(p["deltas"][t + 1])
                    dr = 1 - (F.normalize(p["z"][t], dim=0) *
                              F.normalize(p["z"][t + 1], dim=0)).sum().item()
                    # frozen LUMIERE 1-step head on these states
                    he = 1 - (F.normalize(pred(p["states"][t].unsqueeze(0)), dim=-1) *
                              F.normalize(p["z"][t + 1].unsqueeze(0), dim=-1)
                              ).sum().item() if t < len(p["states"]) else None
                    rows.append((gap, dr, he))
        return rows

    print(f"{'gap':>8} {'cohort':>8} {'n':>6} {'drift':>8} {'headerr':>8}")
    for rows, tag in ((grab(hz), "LUMIERE"), (grab(fl), "SAILOR")):
        by: dict[int, list] = {}
        for gap, dr, he in rows:
            b = next(i for i in range(len(BINS) - 1)
                     if BINS[i] <= gap < BINS[i + 1])
            by.setdefault(b, []).append((dr, he))
        for b in sorted(by):
            dd = [r[0] for r in by[b] if r[1] is not None]
            hh = [r[1] for r in by[b] if r[1] is not None]
            print(f"{LAB[b]:>8} {tag:>8} {len(dd):>6} "
                  f"{statistics.mean(dd):>8.4f} {statistics.mean(hh):>8.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--separability", action="store_true")
    ap.add_argument("--coral", action="store_true")
    ap.add_argument("--plhm", action="store_true")
    ap.add_argument("--interval-match", action="store_true")
    args = ap.parse_args()
    if args.separability:
        print("== 1. site separability ==")
        separability()
    if args.coral:
        print("== 2. CORAL alignment ==")
        coral()
    if args.plhm:
        print("== 5. image-space damping (PLHM check) ==")
        plhm()
    if args.interval_match:
        print("== 8. interval-matched comparison ==")
        interval_match()


if __name__ == "__main__":
    main()
