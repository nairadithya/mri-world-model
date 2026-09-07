"""Raw-vs-MNI visit-to-visit deltas on identical SAILOR session pairs.

Question: how much of SAILOR's damped visit-to-visit change (MNI median
~0.24 vs LUMIERE ~0.76) comes from the MNI pipeline (denoise + intra-patient
registration + PLHM + affine + uint8) rather than the disease?

Method: for every consecutive MNI pair (ses-k -> ses-k+1) with both sessions
linked in raw-mni-link.tsv, compute mean|delta| on 96^3 volumes for:
  (a) the two RAW sessions (t1wc/t1w/t2w/t2wflair, canonical reorientation,
      crude foreground mask = nonzero & >10th pct of nonzero, per-pair mask
      intersection, z-score within mask),
  (b) the two MNI sessions (base variants T1c/T1/T2/Flair, BrainExtractionMask,
      z-score within mask).
Same pairs, same metric shape — only the pipeline differs. Caveats: raw slabs
are anisotropic and unregistered, so raw deltas include position/skull signal
registration legitimately removes; this measures TOTAL pipeline effect, not
PLHM alone (isolating PLHM needs intermediates we don't have).

Usage: python scripts/raw_mni_deltas.py [--subjects sub-01 ...]
Writes JSON summary to stdout; --save writes checkpoints/raw_mni_deltas.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MNI_ROOT = "data/sailor/sailor_ebrains_pseud/derivatives/mni2009c-n-s"
RAW_ROOT = "data/sailor/sailor_ebrains_pseud/rawdata"
LINK = os.path.join(MNI_ROOT, "raw-mni-link.tsv")
# raw-name -> MNI base file
SLOTS = [("t1wc", "T1c"), ("t1w", "T1"), ("t2w", "T2"), ("t2wflair", "Flair")]


def load_link():
    link = {}
    with open(LINK) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["mni session"] != "no":
                link[(r["subject"], r["mni session"])] = r["raw session"]
    return link


def to96(x: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(x).float()[None, None]
    z = [96 / s for s in x.shape]
    y = F.interpolate(t, scale_factor=z, mode="trilinear", align_corners=False)
    return y[0, 0].numpy()


def prep(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = to96(vol)
    m = to96(mask) > 0.5
    m &= v != 0
    vals = v[m]
    if vals.size < 100 or vals.std() < 1e-9:
        return None, None
    return (v - vals.mean()) / vals.std(), m


def crude_mask(vol: np.ndarray) -> np.ndarray:
    nz = vol[vol != 0]
    if nz.size == 0:
        return vol != 0
    return (vol != 0) & (vol > np.percentile(nz, 10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--save", default="checkpoints/raw_mni_deltas.pt")
    args = ap.parse_args()
    link = load_link()
    subjects = sorted({s for s, _ in link}) if not args.subjects else args.subjects

    import torch as _t

    raw_vals, mni_vals, dropped = [], [], 0
    per_slot = {m: ([], []) for _, m in SLOTS}
    for sub in subjects:
        mni_ses = sorted({m for s, m in link if s == sub})
        for a, b in zip(mni_ses, mni_ses[1:]):
            ra, rb = link[(sub, a)], link[(sub, b)]
            for raw_name, mni_name in SLOTS:
                rp_a = os.path.join(RAW_ROOT, sub, ra, raw_name + ".nii.gz")
                rp_b = os.path.join(RAW_ROOT, sub, rb, raw_name + ".nii.gz")
                mp_a = os.path.join(MNI_ROOT, sub, a, mni_name + ".nii.gz")
                mp_b = os.path.join(MNI_ROOT, sub, b, mni_name + ".nii.gz")
                mask_p = os.path.join(MNI_ROOT, sub, a, "BrainExtractionMask.nii.gz")
                if not all(os.path.exists(p) for p in (rp_a, rp_b, mp_a, mp_b, mask_p)):
                    dropped += 1
                    continue
                try:
                    # ascontiguousarray: canonical reorientation can return
                    # negative-stride views (axis flips), which torch rejects.
                    r_a = np.ascontiguousarray(np.asanyarray(
                        nib.as_closest_canonical(nib.load(rp_a)).dataobj,
                        dtype=np.float64))
                    r_b = np.ascontiguousarray(np.asanyarray(
                        nib.as_closest_canonical(nib.load(rp_b)).dataobj,
                        dtype=np.float64))
                    m_a = np.asanyarray(nib.load(mp_a).dataobj, dtype=np.float64)
                    m_b = np.asanyarray(nib.load(mp_b).dataobj, dtype=np.float64)
                    mk = np.asanyarray(nib.load(mask_p).dataobj) > 0
                    zr_a, mr_a = prep(r_a, crude_mask(r_a))
                    zr_b, mr_b = prep(r_b, crude_mask(r_b))
                    zm_a, mm_a = prep(m_a, mk)
                    zm_b, mm_b = prep(m_b, mk)
                    if any(v is None for v in (zr_a, zr_b, zm_a, zm_b)):
                        dropped += 1
                        continue
                    mr = mr_a & mr_b
                    mm = mm_a & mm_b
                    if mr.sum() < 100 or mm.sum() < 100:
                        dropped += 1
                        continue
                    rd = float(np.abs(zr_a[mr] - zr_b[mr]).mean())
                    md = float(np.abs(zm_a[mm] - zm_b[mm]).mean())
                    if not (np.isfinite(rd) and np.isfinite(md)):
                        dropped += 1
                        continue
                except Exception:
                    dropped += 1
                    continue
                raw_vals.append(rd)
                mni_vals.append(md)
                per_slot[mni_name][0].append(rd)
                per_slot[mni_name][1].append(md)
    def q(v):
        v = sorted(v)
        if not v:
            return {"n": 0, "median": None, "p10": None, "p90": None, "mean": None}
        return {"n": len(v), "median": round(v[len(v) // 2], 4),
                "p10": round(v[len(v) // 10], 4), "p90": round(v[9 * len(v) // 10], 4),
                "mean": round(statistics.mean(v), 4)}
    out = {"raw": q(raw_vals), "mni": q(mni_vals), "dropped": dropped,
           "per_slot": {k: {"raw": q(a), "mni": q(b)} for k, (a, b) in per_slot.items()}}
    print(json.dumps(out, indent=1))
    if args.save:
        _t.save(out, args.save)
        print(f"saved {args.save}", file=sys.stderr)


if __name__ == "__main__":
    main()
