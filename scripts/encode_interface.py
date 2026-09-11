"""Encode the representation-interface feature cache (Step 2).

Runs the frozen champion once per volume, keeping representations the
existing `probe_cache` throws away, so interface variants can be scored on
the locked protocol (info/eval_protocol.md):

  vision_mod      (T, M, 768)  first-token latent, per modality (R9)
  vision_roi_mod  (T, M, 768)  DeepBraTumIA-atlas ROI-pooled latent, per modality (R5)
  vision_roi      (T, 768)     mean over modalities of vision_roi_mod
  vision          (T, 768)     mean first-token (existing baseline)
  clinical        (384,)
  fused           (T, 1152)    fusion(vision_mean, clinical) — existing baseline
  states          (T-1, 1152)  frozen temporal over fused (existing baseline)
  states_roi      (T-1, 1152)  frozen temporal over fusion(vision_roi, clinical)
  volumes         (T, 3)       log1p DeepBraTumIA region volumes (present flag)
  labels          (T,)         clean RANO probe labels

ROI alignment is APPROXIMATE: the atlas mask is MNI152 1mm (182x218x182),
resized nearest to 96^3 the same way the registered image is (the registration
transform is not saved, K3-5). Treat vision_roi as a hypothesis probe, not a
precise segmentation read; validate before citing.

Usage:
    python scripts/encode_interface.py --patients Patient-014 Patient-030  # smoke
    python scripts/encode_interface.py                                     # full
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.collate import make_collate
from src.data.dataset import LUMIEREDataset
from src.data.splits import patient_splits
from src.model.jepa_model import JEPAWorldModel

RANO_PROBE_MAP = {"PD": 0, "SD": 1, "PR": 2, "CR": 3}
MASK_ROOT = "data/autoseg/extracted/Imaging"
VOL_ROOT = "data/autoseg/vols/Imaging"
PATCH = 16
ENCODE_CHUNK = 8


def roi_patch_occupancy(patient, visit, size=96):
    """(216,) patch occupancy + (96^3) resized mask, or (None, None)."""
    p = os.path.join(MASK_ROOT, patient, visit,
                     "DeepBraTumIA-segmentation/atlas/segmentation/seg_mask.nii.gz")
    if not os.path.exists(p):
        return None, None
    import nibabel as nib

    d = np.asanyarray(nib.load(p).dataobj).astype(np.float32)
    t = torch.from_numpy(d)[None, None]
    t96 = F.interpolate(t, size=(size, size, size), mode="nearest")[0, 0]
    occ = F.avg_pool3d((t96 > 0).float()[None, None], kernel_size=PATCH)[0, 0]
    return occ.flatten(), t96


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def load_volumes(root=VOL_ROOT):
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
                vols[(p, v)] = (float(d.get("Necrotic_NonEnhancing", 0.0)),
                                float(d.get("Enhancing_Core", 0.0)),
                                float(d.get("Edema_Compartment", 0.0)))
    return vols


def build_datasets(cfg):
    meta_dir = cfg["data"]["meta_dir"]
    demo = next(os.path.join(meta_dir, f) for f in os.listdir(meta_dir)
                if f.startswith("demographics"))
    splits = patient_splits(demo, train_frac=cfg["data"].get("train_split", 0.7),
                            val_frac=cfg["data"].get("val_split", 0.15),
                            seed=cfg["data"].get("seed", 42))
    common = dict(meta_dir=meta_dir, processed_root=cfg["data"]["root"],
                  raw_root=cfg["data"].get("raw_root"),
                  modalities=tuple(cfg["data"].get("modalities", ["CT1", "T1", "T2", "FLAIR"])),
                  min_visits=cfg["data"].get("min_visits", 2))
    return {k: LUMIEREDataset(patients=v, **common) for k, v in splits.items()}


@torch.no_grad()
def encode_tokens(model, flat):
    """(N,1,96,96,96) -> (N, tokens, 768) in chunks."""
    outs = []
    for s in range(0, flat.shape[0], ENCODE_CHUNK):
        inp = flat[s:s + ENCODE_CHUNK].float()
        h = model.backbone.encode_tokens(inp)
        outs.append(h)
    return torch.cat(outs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--champion", default="checkpoints/champion_0.0081.pt")
    ap.add_argument("--cache", default="checkpoints/interface_cache.pt")
    ap.add_argument("--patients", nargs="*", default=None)
    ap.add_argument("--no-roi", action="store_true")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    size = tuple(cfg["preprocessing"].get("target_size", [96, 96, 96]))
    collate = make_collate(size)
    datasets = build_datasets(cfg)
    if args.patients:
        datasets = {k: LUMIEREDataset(patients=args.patients,
                                      meta_dir=cfg["data"]["meta_dir"],
                                      processed_root=cfg["data"]["root"],
                                      raw_root=cfg["data"].get("raw_root"),
                                      modalities=tuple(cfg["data"].get("modalities")),
                                      min_visits=cfg["data"].get("min_visits", 2))
                    for k in datasets}

    model = JEPAWorldModel(cfg)
    ckpt = torch.load(args.champion, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"champion {args.champion} (epoch {ckpt.get('epoch')}, val {ckpt.get('val_loss')})")
    vols = load_volumes()

    cache = {"patients": {}}
    t0 = time.time()
    total = sum(len(ds) for ds in datasets.values())
    done = 0
    for split, ds in datasets.items():
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=collate)
        for batch in loader:
            pid = batch["patient_id"][0]
            item = ds[ds.patients.index(pid)]
            visits = item["visits"]
            n = int(batch["n_visits"][0])
            mri, mri_mask = batch["mri"], batch["mri_mask"]  # (1,T,M,1,96^3),(1,T,M)
            B, T, M = mri.shape[:3]
            flat = mri.reshape(B * T * M, *mri.shape[3:])
            toks = encode_tokens(model, flat)          # (T*M, 217, 768)
            if toks.dim() == 2:
                toks = toks.unsqueeze(1)
            # MONAI ViT (classification off) emits patch tokens only; if a
            # cls token is ever prepended, drop it (216 patches + 1 cls).
            patches = toks[:, 1:] if toks.shape[1] == 217 else toks
            v_first = toks[:, 0].view(T, M, -1)

            occ, roi_ok = None, None
            if not args.no_roi:
                occs, ok = [], []
                for v in visits:
                    o, _ = roi_patch_occupancy(pid, v, size=size[0])
                    ok.append(o is not None)
                    occs.append(o if o is not None else torch.zeros(patches.shape[1]))
                occ = torch.stack(occs).view(T, 1, -1)          # (T,1,216)
                roi_ok = torch.tensor(ok)
                w = occ.unsqueeze(-1)                           # (T,1,216,1)
                pr = patches.view(T, M, patches.shape[1], 768)
                v_roi_mod = (pr * w).sum(2) / w.sum(2).clamp_min(1e-9)
                v_roi = ((v_roi_mod * mri_mask[0].unsqueeze(-1)).sum(1)
                         / mri_mask[0].unsqueeze(-1).sum(1).clamp_min(1e-9))
            else:
                v_roi_mod = torch.zeros(T, M, 768)

            wm = mri_mask[0].float().unsqueeze(-1)              # (T,M,1)
            v_mean = (v_first * wm).sum(1) / wm.sum(1).clamp_min(1e-9)
            if args.no_roi:
                v_roi = v_mean

            c = model.clinical(batch["clinical"])               # (1,384)            ce = c.unsqueeze(1).expand(-1, T, -1)
            fused = model.fusion(v_mean.unsqueeze(0), ce)[0]     # (T,1152)
            states, _ = model.temporal.forward_prefixes(
                fused.unsqueeze(0), batch["time_deltas"], batch["visit_mask"])
            fused_roi = model.fusion(v_roi.unsqueeze(0), ce)[0]
            states_roi, _ = model.temporal.forward_prefixes(
                fused_roi.unsqueeze(0), batch["time_deltas"], batch["visit_mask"])

            labels = [RANO_PROBE_MAP.get(ds.rano.get((pid, v), ""), -1) for v in visits]
            vv = []
            for v in visits:
                d = vols.get((pid, v))
                vv.append([float(np.log1p(x)) for x in d] if d else [0.0, 0.0, 0.0])
            cache["patients"][pid] = {
                "split": split,
                "visits": list(visits),
                "vision": v_mean[:n].clone(),
                "vision_mod": v_first[:n].clone(),
                "vision_roi": v_roi[:n].clone(),
                "vision_roi_mod": v_roi_mod[:n].clone(),
                "roi_present": (roi_ok[:n].clone() if roi_ok is not None
                                else torch.zeros(n, dtype=torch.bool)),
                "clinical": c[0].clone(),
                "fused": fused[:n].clone(),
                "states": states[0, :n - 1].clone(),
                "states_roi": states_roi[0, :n - 1].clone(),
                "volumes": torch.tensor(vv, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
            done += 1
            el = time.time() - t0
            print(f"encoded {done}/{total} ({el / done:.1f}s/patient, "
                  f"ETA {el / done * (total - done) / 60:.0f}min)", flush=True)
    prov = {
        "champion": os.path.abspath(args.champion),
        "champion_epoch": ckpt.get("epoch"),
        "champion_val": ckpt.get("val_loss"),
        "config": os.path.abspath(args.config),
        "config_sha1": hashlib.sha1(open(args.config, "rb").read()).hexdigest(),
        "git_sha": _git_sha(),
        "date": time.strftime("%Y-%m-%d"),
        "mask_root": MASK_ROOT,
        "roi_alignment": "approximate (MNI152 atlas resized to 96^3)",
    }
    cache["provenance"] = prov
    torch.save(cache, args.cache)
    print(f"provenance: {prov}")
    nroi = sum(int(p["roi_present"].sum()) for p in cache["patients"].values())
    print(f"wrote {args.cache}: {done} patients, {nroi} visits with ROI mask")


if __name__ == "__main__":
    main()
