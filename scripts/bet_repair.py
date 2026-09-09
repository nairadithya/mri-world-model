"""HD-BET repair for the SAILOR reprocess: real skull-strip, isolated env.

Background: the executed LUMIERE contract on SAILOR raws silently used the
percentile fallback (hd-bet 2.x unrunnable in the repo venv — torchvision
ban; wrapper flags exit 2), retaining skull (finals nzfrac med 0.49 vs
LUMIERE 0.17). This script redoes ONLY the BET stage on the existing
`_reg.nii.gz` intermediates with hd-bet 2.x from an isolated venv (torchvision
allowed there — no peft import), then finalizes (96^3 + nonzero z-score, the
exact contract step 5) into a NEW root. Nothing under
`data/sailor_reprocessed/` finals is modified (only `_bet2.nii.gz`
intermediates are added beside them).

Efficiency: hd-bet reloads its model per invocation, so sessions are
processed one-invocation-per-session (symlinked `_reg` files into a temp
dir, folder-mode call) instead of one-invocation-per-volume.

Usage:
  scripts/bet_repair.py --hdbet-bin ~/.venvs/hdbet/bin/hd-bet \
    --subjects sub-01 sub-02 --workers 2 --threads 2
  (omit --subjects for all; finals -> data/sailor_reprocessed_bet/)

Resume-safe: existing finals/_bet2 outputs are skipped. Sidecar .txt files
(intervals / RANO / treatment) are copied from the reprocessed session dirs
so eval scripts find metadata. T1c/Flair alias links are created for the
adapter.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

REPROC = "data/sailor_reprocessed"
NEWROOT = "data/sailor_reprocessed_bet"
MODS = ("CT1", "T1", "T2", "FLAIR")
ALIASES = {"T1c": "CT1", "Flair": "FLAIR"}


def bet_session(args) -> str:
    tmpd, ndir, sub, ses, hdbet, threads = args
    regs = {}
    for mod in MODS:
        p = os.path.join(tmpd, f"{mod}_reg.nii.gz")
        if os.path.exists(p):
            regs[mod] = p
    if not regs:
        return f"empty {sub}/{ses}"
    outs = {m: os.path.join(tmpd, f"{m}_bet2.nii.gz") for m in regs}
    if all(os.path.exists(o) for o in outs.values()):
        return f"skip {sub}/{ses}"
    env = dict(os.environ, OMP_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads))
    work = tempfile.mkdtemp(prefix="betses_")
    try:
        # abspath: relative targets would dangle (resolved from the WORK
        # dir, not cwd) -> empty input scan -> nnU-Net IndexError. This
        # exact bug failed all 268 sessions of the first full run.
        for m, p in regs.items():
            os.symlink(os.path.abspath(p),
                       os.path.join(work, f"{m}_reg.nii.gz"))
        outd = os.path.join(work, "out")
        subprocess.run(
            [hdbet, "-i", work, "-o", outd, "-device", "cpu",
             "--disable_tta"],
            check=True, capture_output=True, env=env,
            timeout=2400,
        )
        got = {}
        for f in glob.glob(os.path.join(outd, "*.nii.gz")):
            base = os.path.basename(f)
            for m in regs:
                if base.startswith(m + "_reg"):
                    got[m] = f
        missing = [m for m in regs if m not in got]
        if missing:
            return f"FAIL {sub}/{ses}: no output for {missing}"
        for m, f in got.items():
            # shutil.move: /tmp and data/ are different filesystems
            # (os.replace raises EXDEV).
            if os.path.exists(outs[m]):
                os.remove(outs[m])
            shutil.move(f, outs[m])
        return f"ok {sub}/{ses} ({len(regs)} mods)"
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode(errors="replace")[-300:]
        return f"FAIL {sub}/{ses}: rc={e.returncode} {err}"
    except Exception as e:
        return f"FAIL {sub}/{ses}: {type(e).__name__} {str(e)[:160]}"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def finalize(bet_path: str, final_path: str) -> str:
    img = nib.load(bet_path)
    v = np.asanyarray(img.dataobj, dtype=np.float64)
    t = torch.from_numpy(np.ascontiguousarray(v)).float()[None, None]
    z = [96 / s for s in v.shape]
    y = F.interpolate(t, scale_factor=z, mode="trilinear",
                      align_corners=False)[0, 0].numpy()
    m = y != 0
    vals = y[m]
    if vals.size < 100 or float(vals.std()) < 1e-9:
        return f"EMPTY {final_path}"
    y[m] = (vals - vals.mean()) / max(float(vals.std()), 1e-6)
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    nib.save(nib.Nifti1Image(y.astype(np.float32), np.eye(4)), final_path)
    return f"final {final_path}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdbet-bin", required=True)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--bet-only", action="store_true",
                    help="run BET stage only (skip finalize)")
    args = ap.parse_args()

    jobs, finals = [], []
    subs = sorted(args.subjects) if args.subjects else sorted(
        d for d in os.listdir(REPROC)
        if os.path.isdir(os.path.join(REPROC, d)) and d.startswith("sub-"))
    for sub in subs:
        for ses in sorted(os.listdir(os.path.join(REPROC, sub))):
            sdir = os.path.join(REPROC, sub, ses)
            tmpd = os.path.join(sdir, "_tmp")
            if not os.path.isdir(tmpd):
                continue
            ndir = os.path.join(NEWROOT, sub, ses)
            jobs.append((tmpd, ndir, sub, ses, args.hdbet_bin,
                         args.threads))
            for mod in MODS:
                if os.path.exists(os.path.join(tmpd, f"{mod}_reg.nii.gz")):
                    finals.append((os.path.join(tmpd, f"{mod}_bet2.nii.gz"),
                                   os.path.join(ndir, f"{mod}.nii.gz")))
    print(f"{len(jobs)} BET sessions, {len(finals)} finals", flush=True)
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for i, res in enumerate(pool.imap_unordered(bet_session, jobs), 1):
                if i % 10 == 0 or res.startswith("FAIL"):
                    print(f"[{i}/{len(jobs)}] {res}", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            print(f"[{i}/{len(jobs)}] {bet_session(j)}", flush=True)
    print("BET stage done", flush=True)
    if args.bet_only:
        return
    for bout, fout in finals:
        if not os.path.exists(bout):
            print(f"MISSING bet output for {fout}", flush=True)
            continue
        if os.path.exists(fout):
            continue
        print(finalize(bout, fout), flush=True)
    for sub in subs:  # sidecars + adapter aliases
        # subject-level sidecars (intervals/age/survival — gaps read 0
        # without intervals-days.txt; lesson learned 2026-09-09)
        for f in glob.glob(os.path.join(REPROC, sub, "*.txt")):
            dst = os.path.join(NEWROOT, sub, os.path.basename(f))
            if not os.path.exists(dst):
                shutil.copy(f, dst)
        for ses in sorted(os.listdir(os.path.join(REPROC, sub))):
            sdir = os.path.join(REPROC, sub, ses)
            ndir = os.path.join(NEWROOT, sub, ses)
            if not os.path.isdir(os.path.join(sdir, "_tmp")):
                continue
            os.makedirs(ndir, exist_ok=True)
            for f in glob.glob(os.path.join(sdir, "*.txt")):
                dst = os.path.join(ndir, os.path.basename(f))
                if not os.path.exists(dst):
                    shutil.copy(f, dst)
            for alias, mod in ALIASES.items():
                src, dst = os.path.join(ndir, f"{mod}.nii.gz"), \
                    os.path.join(ndir, f"{alias}.nii.gz")
                if os.path.exists(src) and not os.path.lexists(dst):
                    os.symlink(f"{mod}.nii.gz", dst)
    print("finalize done", flush=True)


if __name__ == "__main__":
    main()
