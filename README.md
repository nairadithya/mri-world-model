# world-model

JEPA world model over longitudinal glioma MRI: predict the *latent* of the next
visit from prior scans + clinical context, never pixels. See `PROPOSAL.md` for
the science and `info/` for the decision log (D-IDs), ablations (A-/I-IDs),
and pilot notes.

- **Training corpus:** LUMIERE — 91 glioblastoma patients, weekly MRI during
  chemo-RT (modalities `CT1, T1, T2, FLAIR` + RANO ratings, demographics,
  acquisition params).
- **Held-out eval:** SAILOR v1 — 27 high-grade-glioma patients, 3–19 timepoints
  each (EBRAINS, controlled access).
- **Backbone:** BRAINIAC 3D ViT-B + LoRA, per-sequence latents fused with
  clinical MLPs, temporal transformer with time-delta encodings, EMA target
  encoder, cosine JEPA loss.

## Layout

- `scripts/` — runnable entry points: `lumiere_fetch.py`, `sailor_fetch.py`,
  `ebrains_auth.py`, `sailor_request_access.py`, `access_probe.py`,
  `preprocess.py`, `run_train.py`.
- `src/` — library code: `data/` (dataset/collate/splits), `model/`
  (BRAINIAC+LoRA, clinical encoders, fusion, temporal transformer, predictor,
  EMA target, JEPA loss), `train/` (trainer, baselines), `preprocessing/`
  (BRAINIAC-contract pipeline).
- `config/` — `default.yaml` (full run), `pilot.yaml` (5-patient CPU pilot).
- `info/` — append-only docs with IDs (`decisions.md`, `ablations.md`,
  `pilot.md`).
- `data/`, `checkpoints/`, `.venv/`, `.env` — local only, NEVER committed
  (gitignored).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Notes: requires `transformers<5`; never install `torchvision` (a skewed build
breaks the `peft` import with a misleading error). You additionally need the
artifacts below — all gitignored, documented not committed.

## Downloading data (`scripts/`)

### LUMIERE imaging — `lumiere_fetch.py` (no auth, CC0)

The full cohort is one 32.6 GB zip on Figshare. The script reads the zip's
central directory with tiny ranged fetches, then pulls only the entries you
ask for — 5 patients cost ~1–2 GB, not 32 GB.

```bash
python scripts/lumiere_fetch.py --list                                   # archive contents
python scripts/lumiere_fetch.py --n 5 --out data/lumiere_sample/Imaging  # first 5 patients
python scripts/lumiere_fetch.py --patients Patient-067 Patient-031 --out data/lumiere_sample/Imaging
python scripts/lumiere_fetch.py --full --out data/lumiere_sample/Imaging # entire 32.6 GB (resumable)
```

Layout on disk: `data/lumiere_sample/Imaging/Patient-XXX/<visit>/{CT1,T1,T2,FLAIR}.nii.gz`
plus `DeepBraTumIA-` / `HD-GLIO-AUTO-segmentation/` masks.

### LUMIERE metadata — already fetched pattern

`config/*.yaml` expect CSVs under `data/lumiere_meta/`:

| file | contents |
|---|---|
| `rano_rating__*.csv` | 616 expert RANO ratings (PD/SD/PR/CR per patient+week) |
| `demographics__*.csv` | survival (weeks), sex, age, IDH, MGMT |
| `mri_params__*.csv` | per-image acquisition parameters (vendor, TR/TE, flip…) |
| `completeness__*.csv` | which sequences/segmentations each visit has |

These ship inside the same Figshare archive / XNAT export; place them under
`data/lumiere_meta/` (the name after `__` must start with `demographics`,
`rano_rating`, `mri_params`, `completeness` — see `run_train.py::find_meta`).

### SAILOR (held-out eval) — `sailor_fetch.py` (EBRAINS auth, controlled access)

SAILOR lives on EBRAINS behind auth (`cae85bcb-8526-442d-b0d8-a866425efff8`).
Auth is shared via `ebrains_auth.py`: `KG_TOKEN` > `KG_REFRESH_TOKEN` >
`KG_CLIENT_ID`/`KG_CLIENT_SECRET` > interactive device flow, cached in `.env`
(0600). Device codes expire in ~5 min; refresh tokens rotate — exactly one
consumer at a time or the chain invalidates (HTTP 400).

```bash
python scripts/sailor_fetch.py --list                  # file tree + sizes (authenticates first)
python scripts/sailor_fetch.py --all --out data/sailor # whole dataset (resumable, token refreshes mid-run)
python scripts/sailor_fetch.py --match 'rawdata_BIDS*' --out data/sailor
```

No access yet? File a request and follow whatever ToS URL it prints:

```bash
python scripts/sailor_request_access.py
```

Data-proxy API used: `GET /v1/datasets/{id}` (list), `GET /v1/datasets/{id}/{object}`
(bytes). There is no `/files` endpoint.

### Other prerequisites

- **BRAINIAC weights** — official `BrainIAC.ckpt` ONLY (community ports have
  proven silently corrupt: matching keys, degenerate outputs). Download from
  the authors' link (see `src/model/brainiac.py::load_simclr_weights` /
  https://github.com/AIM-KannLab/BrainIAC) and place at
  `checkpoints/BrainIAC.ckpt` (path configured in `config/*.yaml`). Verify
  behaviorally: real-vs-noise drift must be ≫ 0.
- **MNI template** — `data/templates/MNI152_T1_1mm.nii.gz` (MNI152NLin2009cAsym
  1 mm via TemplateFlow) for the rigid-registration step of preprocessing.
- **Network probe** — `python scripts/access_probe.py` prints which dataset
  hosts are reachable from your network.

## Preprocess

Converts raw LUMIERE NIfTIs to the BRAINIAC input contract (N4 → 1 mm iso →
rigid MNI → HD-BET skull-strip → 96³ + nonzero z-score) under
`data/lumiere_preprocessed/<Patient>/<visit>/*.nii.gz`. Resume-safe (skips
existing outputs).

```bash
# Pilot subset (shown); drop --patients for the full 91-patient / ~2455-volume cohort
python scripts/preprocess.py --patients Patient-067 Patient-031 \
  --workers 4 --template data/templates/MNI152_T1_1mm.nii.gz
# Full run
python scripts/preprocess.py --workers 8 --template data/templates/MNI152_T1_1mm.nii.gz
# --config defaults to config/default.yaml (sets raw_root, out root, target_size)
```

Omitting `--template` warns and skips MNI registration — never silently.
~30 s/volume; pilot subset (352 vols) vs full cohort (~2455 vols).

## Training run

`scripts/run_train.py` builds patient-level train/val/test splits, loads
preprocessed volumes + clinical CSVs, trains the JEPA model with collapse
monitors (target std / effective rank), checkpoints `best.pt`, and evaluates
the test scorecard when a test split exists.

```bash
# Pilot: 5 richest patients, CPU-friendly (config/pilot.yaml: batch 1, 5 epochs, AMP off)
python scripts/run_train.py --config config/pilot.yaml \
  --patients Patient-067 Patient-031 Patient-073 Patient-078 Patient-029 --no-wandb

# Full run (patient splits from demographics CSV, checkpoint_dir: checkpoints/)
python scripts/run_train.py --config config/default.yaml --no-wandb

# Smoke test, no weights / no data needed (random init, 1 epoch)
python scripts/run_train.py --epochs 1 --batch-size 1 --no-wandb --random-init
```

Useful overrides: `--epochs N`, `--batch-size N`, `--no-wandb`,
`--random-init` (skip checkpoint), `--patients ...` (pilot subset, 80/20
train/val split within the list). Batch 4 fits 24 GB VRAM (chunked encoding);
there is no GPU on the dev host — pilot on CPU (~8 min/epoch), hero run needs
one (~6 h all-in, see `info/pilot.md` scale-up checklist).

**Success criteria** (a falling loss alone proves nothing): loss drops AND
monitors healthy (target std ≫ 0, rank > 1) AND JEPA beats the persistence
baseline (last visit's latent as prediction; pilot reference 0.0043 —
5 epochs on 4 patients are not expected to beat near-static targets).

## Research practice conventions

1. **Never commit data or model weights.** Keep them in `data/` /
   `checkpoints/`, which are gitignored. Document how to obtain them instead.
2. **Small, atomic commits.** One logical change per commit; messages in the
   imperative ("add lumiere fetch script").
3. **Code over notes.** Prefer executable scripts and self-documenting code to
   free-form markdown. Docs that must exist live in `info/` with D/A/I IDs.
4. **Reproducibility.** Pin dependencies, record seeds/configs, and make every
   result traceable to a script + commit hash.
5. **Branch per experiment.** Use feature branches (`exp/...`) and open PRs
   rather than committing straight to `main`.
