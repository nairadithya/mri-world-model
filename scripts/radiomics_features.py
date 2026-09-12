"""Hand-engineered volumetry + growth features (the SOTA hybrid's edge).

Tikhonov 2025's 0.50 macro-F1 is carried by longitudinal volumetry/radiomics,
not the CNN (their ablation: volumes-only F1 0.30, +growth/shrinkage F1 0.45,
ResNet-alone AUC 0.74). This builds the closest in-repo version: per visit-pair
region volumes, log-growth, ratios, and nadir-relative features, for LUMIERE
(DeepBraTumIA measured volumes) and SAILOR (ONCO masks), in the cache format
`cross_site_adapt.py` consumes.

Features per pair t -> t+1 (label = RANO_{t+1}), regions {necrotic, enhancing,
edema} + total:
  log1p(v_t), log1p(v_{t+1}), dlog, ratio, v_t/nadir_t, v_{t+1}/nadir_{t+1},
  plus total-volume versions of the first four.

Usage:
    python scripts/radiomics_features.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import LUMIEREDataset
from src.data.eval_protocol import load_protocol
from src.model.heads import ACTION_TO_FLAT

VOL_ROOT = "data/autoseg/vols/Imaging"
SAILOR_ROOT = "data/sailor/sailor_ebrains_pseud/derivatives/mni2009c-n-s"
SAILOR_MASKS = ("NecrosisMask-ONCO", "ContrastEnhancedMask-ONCO", "EdemaMask-ONCO")


def lum_volumes(root=VOL_ROOT):
    """(patient, week) -> (necrotic, enhancing, edema) mm^3."""
    vols = {}
    for pid in os.listdir(root):
        pdir = os.path.join(root, pid)
        if not os.path.isdir(pdir):
            continue
        for week in os.listdir(pdir):
            jf = os.path.join(pdir, week, "DeepBraTumIA-segmentation/atlas/"
                              "segmentation/measured_volumes_in_mm3.json")
            if not os.path.exists(jf):
                continue
            d = json.load(open(jf))
            vols[(pid, week)] = (
                float(d.get("Necrotic_NonEnhancing", 0.0)),
                float(d.get("Enhancing_Core", 0.0)),
                float(d.get("Edema_Compartment", 0.0)),
            )
    return vols


def _mask_volume_mm3(path: str) -> float | None:
    if not os.path.exists(path):
        return None
    import nibabel as nib
    import numpy as np

    img = nib.load(path)
    d = np.asanyarray(img.dataobj)
    vox = float(abs(np.linalg.det(img.affine[:3, :3])))
    return float((d > 0).sum()) * vox


def sailor_volumes(root=SAILOR_ROOT):
    """(subject, session) -> (necrotic, edema, enhancing) mm^3."""
    vols = {}
    for sub in os.listdir(root):
        sdir = os.path.join(root, sub)
        if not os.path.isdir(sdir):
            continue
        for ses in os.listdir(sdir):
            d = os.path.join(sdir, ses)
            if not os.path.isdir(d):
                continue
            got = []
            ok = True
            for name in SAILOR_MASKS:
                v = None
                for ext in (".nii.gz", ".nii"):
                    v = _mask_volume_mm3(os.path.join(d, name + ext))
                    if v is not None:
                        break
                if v is None:
                    ok = False
                    break
                got.append(v)
            if ok:
                vols[(sub, ses)] = tuple(got)  # necrotic, enhancing, edema
    return vols


def _pair_feature(a, b, nadir):
    f = []
    for r in range(3):
        va, vb, nd = a[r], b[r], nadir[r]
        f += [math.log1p(va), math.log1p(vb),
              math.log1p(vb) - math.log1p(va),
              (vb + 1.0) / (va + 1.0),
              (va + 1.0) / (nd + 1.0),
              (vb + 1.0) / (nd + 1.0)]
    ta, tb = sum(a), sum(b)
    f += [math.log1p(ta), math.log1p(tb),
          math.log1p(tb) - math.log1p(ta), (tb + 1.0) / (ta + 1.0)]
    return f


def build_series(vols, patient, visits, labels_next):
    """Rows of (features, label) for consecutive pairs with volumes on both."""
    series = [vols.get((patient, v)) for v in visits]
    nadir = [float("inf")] * 3
    nadir_at = []
    for s in series:
        if s is not None:
            nadir = [min(nadir[r], s[r]) for r in range(3)]
        nadir_at.append(list(nadir))
    feats, labs = [], []
    for t in range(len(visits) - 1):
        a, b = series[t], series[t + 1]
        lab = labels_next[t]
        if a is None or b is None or lab is None:
            continue
        feats.append(_pair_feature(a, b, nadir_at[t + 1]))
        labs.append(lab)
    return feats, labs


def _label(action_id: int):
    return ACTION_TO_FLAT.get(int(action_id))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--protocol", default="info/eval_folds.json")
    ap.add_argument("--out", default="checkpoints/radiomics_features.pt")
    args = ap.parse_args()
    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    proto = load_protocol(args.protocol)
    out = {"lum": {}, "sailor": {}, "feature_dim": 22}

    common = dict(meta_dir=cfg["data"]["meta_dir"], processed_root=cfg["data"]["root"],
                  raw_root=cfg["data"].get("raw_root"),
                  modalities=tuple(cfg["data"].get("modalities", ["CT1", "T1", "T2", "FLAIR"])),
                  min_visits=cfg["data"].get("min_visits", 2))
    lv = lum_volumes()
    print(f"LUMIERE volume studies: {len(lv)}")
    for split, key in (("encoder_train", "train"), ("dev", "dev"), ("final", "test")):
        ds = LUMIEREDataset(patients=proto[split], **common)
        for i in range(len(ds)):
            s = ds[i]
            pid = s["patient_id"]
            labs = [_label(a) for a in s["actions"].tolist()[1:]] + [None]
            feats, labs = build_series(lv, pid, s["visits"], labs)
            if feats:
                out["lum"][pid] = {"features": torch.tensor(feats, dtype=torch.float32),
                                   "labels": torch.tensor(labs, dtype=torch.long),
                                   "split": key}

    sv = sailor_volumes()
    print(f"SAILOR volume studies: {len(sv)}")
    from src.data.sailor import SAILORDataset
    sds = SAILORDataset(SAILOR_ROOT)
    for i in range(len(sds)):
        s = sds[i]
        sub = s["patient_id"]
        labs = [_label(a) for a in s["actions"].tolist()[1:]] + [None]
        feats, labs = build_series(sv, sub, s["visits"], labs)
        if feats:
            out["sailor"][sub] = {"features": torch.tensor(feats, dtype=torch.float32),
                                  "labels": torch.tensor(labs, dtype=torch.long)}

    torch.save(out, args.out)
    print(f"wrote {args.out}: lum {len(out['lum'])} / sailor {len(out['sailor'])} "
          f"patients, dim {out['feature_dim']}")


if __name__ == "__main__":
    main()
