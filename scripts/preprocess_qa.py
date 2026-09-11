"""Post-preprocess QA gate: integrity per volume + drift vs a reference cohort.

Integrity (FAIL, exit 1): wrong shape, non-finite voxels, empty volume,
near-empty (nzfrac < 0.05), or foreground bounding box < 5% of the cube.
NCC-to-template is WARNING-only: it fires on healthy T2s (contrast-driven)
so it can never gate, but total collapse reads ≈0 and is worth listing.

Drift (report only): nzfrac distribution vs --ref root. A median shift
flags a different effective contract (e.g. skull retained: SAILOR
reprocess med ~0.49 vs LUMIERE ~0.17) even when every volume passes
integrity. Sessions with <4 modalities are listed (expected: raw-missing
+ registration casualties), not failed.

SAILOR content gate (--sailor-root): the derivatives tree holds
present-but-empty modality files (existence ≠ presence, K3-1) that the
96³-named walk above cannot see because those files are native-resolution
and use T1c/Flair aliases. This mode delegates to the adapter's own
`scan_empty_modalities` so the QA gate and the loader agree (exit 1 if any).

Usage:
    python scripts/preprocess_qa.py --root data/sailor_reprocessed \\
        --template data/templates/MNI152_T1_1mm.nii.gz --ref data/lumiere_preprocessed
    python scripts/preprocess_qa.py --sailor-root \\
        data/sailor/sailor_ebrains_pseud/derivatives/mni2009c-n-s \\
        data/sailor_reprocessed data/sailor_reprocessed_bet
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODALITIES = ("CT1", "T1", "T2", "FLAIR")
NCC_FAIL = 0.01
NZ_FAIL = 0.05
BBOX_FAIL = 0.05


def ncc_union(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    m = (a != 0) | (b != 0)
    a, b = a[m], b[m]
    if a.size == 0:
        return 0.0
    denom = a.std() * b.std()
    if denom == 0:
        return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean() / denom)


def bbox_frac(d: np.ndarray) -> float:
    idx = np.argwhere(d != 0)
    if len(idx) == 0:
        return 0.0
    ext = (idx.max(0) - idx.min(0) + 1)
    return float(np.prod(ext) / d.size)


def scan_root(root: str, size: tuple[int, int, int]) -> dict:
    vols: list[dict] = []
    partial: list[str] = []
    for pat in sorted(os.listdir(root)):
        pdir = os.path.join(root, pat)
        if not os.path.isdir(pdir):
            continue
        for vis in sorted(os.listdir(pdir)):
            vdir = os.path.join(pdir, vis)
            if not os.path.isdir(vdir):
                continue
            have = [m for m in MODALITIES
                    if os.path.exists(os.path.join(vdir, f"{m}.nii.gz"))]
            if len(have) < len(MODALITIES):
                partial.append(f"{pat}/{vis}: {have}")
            for m in have:
                vols.append({"id": f"{pat}/{vis}/{m}",
                             "path": os.path.join(vdir, f"{m}.nii.gz")})
    return {"vols": vols, "partial": partial}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--template", default=None)
    ap.add_argument("--ref", default=None,
                    help="reference preprocessed root for drift report")
    ap.add_argument("--size", type=int, nargs=3, default=[96, 96, 96])
    ap.add_argument("--sailor-root", nargs="*", default=None,
                    help="SAILOR tree(s): flag present-but-empty modality "
                         "files via the adapter's content metric")
    args = ap.parse_args()
    size = tuple(args.size)

    if args.sailor_root is not None:
        if not args.sailor_root:
            ap.error("--sailor-root given with no roots")
        from src.data.sailor import scan_empty_modalities

        rc = 0
        for sr in args.sailor_root:
            rows = scan_empty_modalities(sr)
            print(f"{sr}: {len(rows)} present-but-empty modality files")
            for r in rows[:60]:
                print(f"  {r['subject']}/{r['session']}/{r['slot']}: "
                      f"nz={r['nz']:.4f} ({r['path']})")
            if rows:
                rc = 1
        return rc
    if not args.root:
        ap.error("--root is required (or use --sailor-root)")

    tpl = None
    if args.template:
        import torch
        import torch.nn.functional as F

        d = nib.load(args.template).get_fdata(dtype=np.float32)
        t = torch.from_numpy(d).unsqueeze(0).unsqueeze(0)
        tpl = F.interpolate(t, size=size, mode="trilinear",
                            align_corners=False).squeeze().numpy()

    info = scan_root(args.root, size)
    vols = info["vols"]
    print(f"{args.root}: {len(vols)} volumes, "
          f"{len(info['partial'])} partial sessions")
    for p in info["partial"][:20]:
        print(f"  partial: {p}")

    fails: list[str] = []
    warns: list[str] = []
    nz_all: list[float] = []
    for i, v in enumerate(vols, 1):
        try:
            d = nib.load(v["path"]).get_fdata(dtype=np.float32)
        except Exception as e:
            fails.append(f"{v['id']}: unreadable ({e})")
            continue
        nz = float((d != 0).mean())
        nz_all.append(nz)
        problems = []
        if tuple(d.shape) != size:
            problems.append(f"shape {d.shape}")
        if not np.isfinite(d).all():
            problems.append("non-finite")
        if nz == 0:
            problems.append("empty")
        elif nz < NZ_FAIL:
            problems.append(f"near-empty nz={nz:.4f}")
        if nz > 0 and bbox_frac(d) < BBOX_FAIL:
            problems.append(f"bbox<5% nz={nz:.4f}")
        if tpl is not None and not problems:
            # Informational only: fires on healthy T2s (contrast-driven),
            # so it must never fail the run. Kept because total collapse
            # (nz~0 volumes excluded above) reads ≈0 here.
            try:
                import torch
                import torch.nn.functional as F

                t = torch.from_numpy(d).unsqueeze(0).unsqueeze(0)
                g = F.interpolate(t, size=size, mode="trilinear",
                                  align_corners=False).squeeze().numpy()
                if ncc_union(g, tpl) < NCC_FAIL:
                    warns.append(f"{v['id']}: ncc-to-template<0.01")
            except Exception:
                pass
        if problems:
            fails.append(f"{v['id']}: {'; '.join(problems)}")
        if i % 250 == 0:
            print(f"  ...{i}/{len(vols)}", flush=True)

    nz_all.sort()
    n = len(nz_all)
    med = nz_all[n // 2] if n else 0.0
    print(f"nzfrac: n={n} min={nz_all[0]:.4f} p5={nz_all[n // 20]:.4f} "
          f"med={med:.4f} max={nz_all[-1]:.4f}" if n else "no volumes")
    if args.ref:
        ref = scan_root(args.ref, size)["vols"]
        rnz = []
        for v in ref:
            try:
                d = nib.load(v["path"]).get_fdata(dtype=np.float32)
                rnz.append(float((d != 0).mean()))
            except Exception:
                pass
        rnz.sort()
        rmed = rnz[len(rnz) // 2]
        print(f"ref {args.ref}: n={len(rnz)} med={rmed:.4f} "
              f"p5={rnz[len(rnz) // 20]:.4f} max={rnz[-1]:.4f}")
        print(f"drift: target/ref median = {med / (rmed + 1e-12):.2f}x "
              f"({'SKULL-SUSPECT' if med > 2 * rmed else 'ok'})")

    print(f"FAILURES: {len(fails)}")
    for f in fails[:30]:
        print(f"  {f}")
    print(f"WARNINGS (info only): {len(warns)}")
    for w in warns[:15]:
        print(f"  {w}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
