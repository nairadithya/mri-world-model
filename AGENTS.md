# AGENTS.md — working guide for this repo

Research repo: JEPA world models over longitudinal glioma MRI (LUMIERE
training corpus, SAILOR held-out eval). Read `PROPOSAL.md` for the science
and `info/` for why things are the way they are.

## Layout

- `src/` — library code. `data/` (LUMIERE dataset/collate/splits),
  `model/` (BRAINIAC+LoRA, clinical encoders, fusion, temporal transformer,
  predictor, EMA target, JEPA loss), `train/` (trainer, baselines),
  `preprocessing/` (BRAINIAC contract pipeline).
- `scripts/` — runnable entry points. `run_train.py`, `preprocess.py`,
  `lumiere_fetch.py`, `sailor_fetch.py`, `ebrains_auth.py`,
  `sailor_request_access.py`, `access_probe.py`.
- `config/` — `default.yaml` (full run), `pilot.yaml` (5-patient CPU pilot).
- `kaggle/` — hero-run notebook. `hero_run.py` is the source of truth;
  never edit the `.ipynb` directly (JSON churn breaks diffs). Regenerate
  with `jupytext --to ipynb kaggle/hero_run.py` after editing.
- `info/` — decision log (`decisions.md`, IDs D0–), ablations (`ablations.md`,
  IDs A–/I–), pilot notes (`pilot.md`). Append-only; reference IDs.
- `data/`, `checkpoints/`, `.venv/`, `.env` — NEVER commit (gitignored).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

You additionally need (all gitignored, documented not committed):

- `checkpoints/BrainIAC.ckpt` — official weights ONLY (see gotchas).
- `data/lumiere_sample/Imaging/` — LUMIERE NIfTIs (`lumiere_fetch.py`).
- `data/lumiere_preprocessed/` — built by `preprocess.py`.
- `.env` — EBRAINS tokens for SAILOR (auto-managed by `ebrains_auth.py`).
- `data/templates/MNI152_T1_1mm.nii.gz` — TemplateFlow (see `info/`).

## Common commands

```bash
# Preprocess (pilot subset shown; drop --patients for full cohort)
python scripts/preprocess.py --patients Patient-067 Patient-031 \
  --workers 4 --template data/templates/MNI152_T1_1mm.nii.gz
# Train
python scripts/run_train.py --config config/pilot.yaml \
  --patients Patient-067 Patient-031 Patient-073 Patient-078 Patient-029 --no-wandb
# Smoke test (no weights needed)
python scripts/run_train.py --epochs 1 --batch-size 1 --no-wandb --random-init
```

## Conventions (from README, enforced)

1. Never commit data, weights, checkpoints, or secrets.
2. Small atomic commits, imperative messages ("add X").
3. Code over notes; docs live in `info/` with D/A/I IDs.
4. Reproducibility: pin deps, record seeds/configs, tie results to commits.
5. Branch per experiment (`exp/...`), PRs not direct-to-main.

## Gotchas (earned the hard way — see info/ablations.md)

- **Weights**: use official `BrainIAC.ckpt` only. Community ports can be
  silently corrupt (matching key names, degenerate outputs). Verify any
  checkpoint behaviorally: real-vs-noise drift must be ≫ 0.
- **Deps**: pin `transformers<5`; never install `torchvision` (a skewed
  build breaks peft import with a misleading error).
- **Data**: pandas NaN is truthy — coerce clinical numerics via explicit
  `isna` guards. Drop imageless visits; require pixels on both sides of
  every loss pair.
- **Gitignore**: anchor data-dir rules (`/data/`, not `data/`) — an
  unanchored pattern silently unmatched `src/data/` and the whole data
  layer went uncommitted until the first fresh clone (Kaggle) failed.
  Audit with `git ls-files` after ignore changes.
- **Kaggle**: T4 only (P100/sm_60 has no torch kernels in the image);
  batch 1 on 16 GB; ingestion gunzips `.nii.gz`→`.nii` in place (dataset
  accepts both); uninstall torchao (0.10 breaks fresh peft imports).
- **Auth**: EBRAINS device codes expire in 5 min. Refresh tokens rotate —
  exactly one consumer at a time or the chain invalidates (400).
  SAILOR is controlled-access; the data-proxy v1 object API is
  `GET /v1/datasets/{id}` (list) and `GET /v1/datasets/{id}/{object}`
  (bytes). There is no `/files` endpoint.
- **Scale**: checkpoints are ~1.5GB each; batch 4 fits 24GB VRAM
  (chunked encoding). No GPU on the dev host — pilot on CPU, hero run
  needs one (≈6 h all-in, see `info/pilot.md` scale-up checklist).

## Verification bar

- `python -m py_compile` on touched files.
- Smoke test above must pass; forward + backward + EMA + baselines finite.
- Clinical vectors swept NaN-free over all 91 patients after data changes.
- Success criteria for training changes: loss drops, monitors healthy
  (target std ≫ 0, rank > 1), JEPA beats persistence (currently 0.0043
  pilot) — a falling loss alone proves nothing.
