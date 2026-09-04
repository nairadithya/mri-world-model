"""Offline preprocessing: raw LUMIERE NIfTIs -> BRAINIAC-contract volumes.

Usage:
    python scripts/preprocess.py --config config/default.yaml --template data/templates/MNI152_T1_1mm.nii.gz
    python scripts/preprocess.py --patients Patient-067 Patient-031 --workers 4 --template ...

Walks <raw_root>/<Patient>/<visit>/*.nii.gz for the 4 modalities and writes
mirrored outputs under <root>/<Patient>/<visit>/*.nii.gz. Skips files that
already exist (resume-safe). Multiprocessing over volumes.
"""
from __future__ import annotations

import argparse
import os
import sys
from functools import partial
from multiprocessing import Pool

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing.brainiac_contract import preprocess_sequence


MODALITIES = ("CT1", "T1", "T2", "FLAIR")


def _run_one(job: tuple[str, str, str, tuple[int, int, int]]) -> str | None:
    src, dst, template, target_size = job
    try:
        import torch

        torch.set_num_threads(2)  # avoid oversubscription across pool workers
        preprocess_sequence(src, dst, template_path=template,
                            device="cpu", target_size=target_size)
        return f"{src} -> ok"
    except Exception as e:
        return f"{src} -> FAILED: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--template", default=None)
    ap.add_argument("--patients", nargs="*", default=None,
                    help="restrict to these patient IDs (default: all)")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    raw_root = cfg["data"]["raw_root"]
    out_root = cfg["data"]["root"]
    target_size = tuple(cfg["preprocessing"]["target_size"])
    if args.template is None:
        print("WARNING: no --template given; rigid MNI registration (step 3 of the "
              "BRAINIAC contract) is SKIPPED. Outputs will be N4+iso+skull-stripped "
              "but not in MNI space.")

    jobs = []
    for patient in sorted(os.listdir(raw_root)):
        if args.patients and patient not in args.patients:
            continue
        pdir = os.path.join(raw_root, patient)
        if not os.path.isdir(pdir):
            continue
        for visit in sorted(os.listdir(pdir)):
            vdir = os.path.join(pdir, visit)
            if not os.path.isdir(vdir):
                continue
            for mod in MODALITIES:
                src = os.path.join(vdir, f"{mod}.nii.gz")
                dst = os.path.join(out_root, patient, visit, f"{mod}.nii.gz")
                if os.path.exists(src) and not os.path.exists(dst):
                    jobs.append((src, dst, args.template, target_size))
    print(f"{len(jobs)} volumes to process ({args.workers} workers)")

    if args.workers > 1:
        with Pool(args.workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_run_one, jobs), 1):
                print(f"[{i}/{len(jobs)}] {res}")
    else:
        for i, job in enumerate(jobs, 1):
            print(f"[{i}/{len(jobs)}] {_run_one(job)}")
    print("Done.")


if __name__ == "__main__":
    main()
