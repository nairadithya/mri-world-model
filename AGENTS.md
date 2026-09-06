# AGENTS.md — working guide for this repo

Research repo: JEPA world models over longitudinal glioma MRI (LUMIERE
training corpus, SAILOR held-out eval). Read `PROPOSAL.md` for the science
and `info/` for why things are the way they are.

## Layout

- `src/` — library code. `data/` (LUMIERE dataset/collate/splits,
  SAILOR adapter `sailor.py`), `model/` (BRAINIAC+LoRA, clinical encoders,
  fusion, temporal transformer, predictor, gap-conditioned horizon head,
  velocity-field dynamics `dynamics_field.py`, EMA target, JEPA loss, RANO aux
  `heads.py`), `train/` (trainer, baselines), `preprocessing/` (BRAINIAC
  contract pipeline).
- `scripts/` — runnable entry points. `run_train.py` (+`--aux-lambda`,
  `--resume-from`, `--horizon`, `--dynamics`), `preprocess.py`, `probe_rano.py` (frozen
  RANO probes + `--cv`), `surprise_signal.py` (error→PD AUC), `volume_probe.py`
  (auto-mask volumetry), `sailor_eval.py` (cross-site eval),
  `horizon_probe.py` (`--encode`/`--curve`/`--train`: multi-horizon gate),
  `horizon_eval.py` (per-horizon JEPA-vs-persistence for a trained leg),
  fetch/auth scripts.
- `config/` — `default.yaml` (full run; `aux:` section, lambda 0 = JEPA
  only), `pilot.yaml` (5-patient CPU pilot).
- `kaggle/` — hero-run notebook. `hero_run.py` is the source of truth;
  never edit the `.ipynb` directly (JSON churn breaks diffs). Regenerate
  with `jupytext --to ipynb kaggle/hero_run.py` after editing.
  `kernel-horizon/` is the pushable-run variant (own `.py` source +
  `kernel-metadata.json`; shell commands LIVE — push executes the notebook
  as-is, so never `py_compile` it, only `jupytext --to ipynb`).
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
- **Probes**: hero-split probe numbers are noise (0.45→0.51 across encoders
  while CV sits at 0.33) — only 5-fold CV counts. Ridge + CV-λ mandatory
  above ~100-d features (plain LSQ on 1152-d/450-row gives R²≈−100).
- **SAILOR**: `-icor` files carry background NaNs (~200 sessions, up to 79%
  of voxels) — prefer base variants; finite-check any new site before first
  encode. RANO codes {1:PD, 2:SD, 3:PR, 5:CR} are empirical (volume deltas);
  3-vs-5 tentative. Intervals ~14 days → persistence regime (A12).
- **Gitignore**: anchor data-dir rules (`/data/`, not `data/`) — an
  unanchored pattern silently unmatched `src/data/` and the whole data
  layer went uncommitted until the first fresh clone (Kaggle) failed.
  Audit with `git ls-files` after ignore changes.
- **Kaggle**: T4 only (P100/sm_60 has no torch kernels in the image);
  batch 1 on 16 GB; ingestion gunzips `.nii.gz`→`.nii` in place (dataset
  accepts both); uninstall torchao (0.10 breaks fresh peft imports).
- **Kaggle notebook**: `%env` swallows trailing `#` comments into the value
  (keep comments on separate lines); datasets may mount at
  `/kaggle/input/datasets/<user>/<slug>/` — glob recursively.
  A resume leg's `best.pt` overwrites the staged champion: label downloads
  with (leg, epoch, val).
- **Auth**: EBRAINS device codes expire in 5 min. Refresh tokens rotate —
  exactly one consumer at a time or the chain invalidates (400).
  SAILOR is controlled-access; the data-proxy v1 object API is
  `GET /v1/datasets/{id}` (list) and `GET /v1/datasets/{id}/{object}`
  (bytes). There is no `/files` endpoint. Read 401s literally: missing
  scopes (token needs `roles`, `email`, `team`, `profile`) means re-auth;
  "access has expired, please request access again" means file a request
  (`sailor_request_access.py`), not a token bug.
- **Scale**: checkpoints are ~1.5GB each; batch 4 fits 24GB VRAM
  (chunked encoding). No GPU on the dev host — pilot on CPU, hero run
  needs one (≈6 h all-in, see `info/pilot.md` scale-up checklist).

## Programmatic Kaggle runs (verified 2026-09-06, horizon leg)

The CLI (`kaggle`, creds in `~/.kaggle/`) can push, monitor, and fetch a GPU
run end-to-end — no browser needed except for live logs (the API exposes
status only; the kernel page streams the live log).

```bash
kaggle quota                                            # GPU hours remaining
kaggle kernels push -p kaggle/kernel-horizon \
  -t 32400 --accelerator NvidiaTeslaT4                  # push + start (v1, v2, …)
kaggle kernels status -k nairadithya/horizon-leg        # RUNNING / COMPLETE / ERROR
kaggle kernels logs -k nairadithya/horizon-leg > logs_run.json   # JSON array AFTER finish
kaggle kernels output -k nairadithya/horizon-leg -p outputs/horizon-leg/  # best.pt + logs (~1.5GB)
```

Rules, all earned:

- **T4 selection needs `machine_shape`.** `enable_gpu: true` alone lands on
  P100, which this image's torch build cannot use (no sm_60 kernels — CUDA
  available but every op fails). Set `"machine_shape": "NvidiaTeslaT4"` in
  `kernel-metadata.json` AND pass `--accelerator NvidiaTeslaT4` (belt and
  suspenders; the flag alone has mixed reports).
- **Push executes the notebook as-is.** Training shell commands must be LIVE
  in the pushed `.ipynb` (unlike `hero_run.py`, where they stay commented).
  Keep the `.py` jupytext source as truth; regenerate after editing.
- **Fail fast on the wrong GPU.** First code cell asserts `torch.cuda` and
  `'T4' in device name` — a P100 session aborts in seconds instead of burning
  quota. Verify from the log (`device: Tesla T4`), never assume.
- **Persist CLI-set flags to `kaggle.yaml`.** `run_train.py` flags mutate the
  in-memory config only; later eval cells re-read the file. The horizon leg
  lost its whole eval table to this (training completed, assert tripped on a
  stale file). Write every flag the run depends on into `kaggle.yaml` at wire-up.
- **Datasets by slug + recursive glob.** Attach inputs by dataset name; mounts
  land at varying depths (`/kaggle/input/<slug>/` vs
  `/kaggle/input/datasets/<user>/<slug>/`). Never hardcode one depth.
- **Background launches need `setsid`.** A tool-timeout process-group kill
  takes plain `nohup … &` children with it (killed the first local eval).
  Use `setsid nohup … & disown` for anything outliving the call.
- **Outputs are gitignored, always.** `outputs/` holds ~1.5GB `best.pt` files;
  judge the gate locally (`scripts/horizon_eval.py` runs CPU-only), commit
  only code + notes.

## Verification bar

- `python -m py_compile` on touched files.
- Smoke test above must pass; forward + backward + EMA + baselines finite.
- Clinical vectors swept NaN-free over all 91 patients after data changes.
- Success criteria for training changes: loss drops, monitors healthy
  (target std ≫ 0, rank > 1), JEPA beats persistence (A8: 0.0081 vs 0.0218
  in-domain; A12: SAILOR short-interval regime favors persistence) —
  a falling loss alone proves nothing.
