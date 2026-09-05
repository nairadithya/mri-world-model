# Preprocessing: the BRAINIAC contract

All training inputs pass through `src/preprocessing/brainiac_contract.py`
(`preprocess_sequence`), driven offline by `scripts/preprocess.py`. Training
never preprocesses on the fly (D3). Purpose: the BRAINIAC ViT-B backbone was
pretrained on exactly one input distribution — 96³ voxels @ 1 mm isotropic,
N4-corrected, rigidly MNI-registered, skull-stripped — so feeding it anything
else is out-of-distribution by construction. The pipeline reproduces the
official repo's transforms line-for-line (verified against their
`dataset.py`: trilinear resize + `NormalizeIntensityd(nonzero=True)`
identical); A4 confirmed it preserves longitudinal signal (raw vs
preprocessed drift indistinguishable, ~2–3e-5 per sequence).

## The 5 stages (per sequence: CT1, T1, T2, FLAIR independently)

**1. N4 bias-field correction** (`n4_bias.py`). MRI coil sensitivity leaves a
smooth low-frequency intensity gradient across the scan — two identical
tissues can read different values at opposite ends of the volume. N4 estimates
that field and divides it out. Implementation: cast to Float32, Otsu-threshold
a foreground mask (200 bins), shrink 2× for speed, fit N4 (4 levels × 50
iterations), upsample the log bias field to full resolution, divide.
Intermediate: `<MOD>_n4.nii.gz`.

**2. Resample to 1 mm isotropic** (`registration.py::resample_to_iso`).
Clinical scans arrive with anisotropic voxels (e.g. 1×1×5 mm slabs). The
backbone's 16³ patches assume cubic 1 mm voxels, so the volume is resampled
with linear interpolation, preserving origin/direction, zero background.
Intermediate: `<MOD>_iso.nii.gz`.

**3. Rigid registration to MNI152** (`registration.py`,
template `data/templates/MNI152_T1_1mm.nii.gz`, D4). Aligns each scan to a
common anatomical frame so a voxel index means the same brain location across
visits and patients — without this, "change" is dominated by head position.
Rigid-only (Euler3D: rotation + translation, no scaling/shear): anatomy is
never warped, so tumor size/shape — the actual signal — is preserved.
Optimizer: Mattes mutual information (50 histogram bins, 5% random sampling),
gradient descent, 200 iterations over a 3-level pyramid (shrink 4/2/1,
smoothing 2/1/0), geometry-based centered initialization. Skipped with an
explicit warning if `--template` is omitted — never silently. Intermediate:
`<MOD>_reg.nii.gz`.

**4. HD-BET skull-stripping** (`skull_strip.py`). Removes skull, scalp, eyes —
tissues the backbone never saw in pretraining and that would dominate
intensity statistics. Runs as the `hd-bet` subprocess (`-device cpu -mode
fast -tta 0`), keeping the heavy dependency out of the import path. If
HD-BET is missing/broken, falls back to a crude 10th-percentile intensity
mask (better than nothing, worse than BET — check logs if you suspect it).
This is the slowest stage on CPU (neural-net inference per volume) and
dominates wall-clock. Intermediate: `<MOD>_bet.nii.gz`.

**5. Resize to 96³ + nonzero z-score** (in `preprocess_sequence`). Trilinear
resize to the backbone's exact input grid, then per-volume normalization over
nonzero voxels only (`mean/std` of brain, background excluded): without the
nonzero guard, the vast zero background would drag the mean and crush tissue
contrast. Output affine is reset to identity (downstream treats volumes as
pre-aligned arrays, not world-coordinate images). Final:
`data/lumiere_preprocessed/<Patient>/<visit>/<MOD>.nii.gz`.

## Orchestration (`scripts/preprocess.py`)

- Mirrors `<raw_root>/<Patient>/<visit>/` into `<root>/` (roots from
  `--config`, default `config/default.yaml`); `--patients` restricts the set.
- Resume-safe: files with existing outputs are skipped at scan time, so
  crashed runs lose no completed work (proven repeatedly — files accumulate
  across restarts).
- Multiprocessing over volumes (`--workers`, 2 torch threads per worker, D5);
  per-stage intermediates land in `<visit>/_tmp/` (kept, ~4× file count —
  only final `<MOD>.nii.gz` matter downstream).
- Measured ~30 s/volume wall uncontended; the 2455-volume full run is an
  overnight job on 2 workers.

## Ops lessons (earned 2026-09-04/05)

- The pool uses forkserver workers (Python 3.14). After any killed run,
  orphaned workers keep spinning — hunt `multiprocessing.forkserver` with
  the script's `main_path` and kill them, or they hold load ~40 idle.
- Never run two preprocess invocations concurrently: both queue the same
  unfinished volumes and race on output paths (caught on Patient-025;
  outputs verified clean after).
- A "workers 4+ BrokenPipe crash" was reported and then exonerated (D18
  correction): 4 workers ran 12+ min clean; the crashes were collateral of
  a process-group kill and a load-38 machine, not a worker-count bug.
