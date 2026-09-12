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
  `--resume-from`, `--horizon`, `--dynamics`), `preprocess.py`, `probe_rano.py`
  (frozen RANO probes; `--cv-unseen` locked protocol + `--cohort`/`--readout-seed`;
  legacy `--cv` is leaky, K3-16), `lock_eval.py` (materialize/verify the locked
  folds), `surprise_signal.py` (error→PD AUC), `volume_probe.py` (auto-mask
  volumetry), `sailor_eval.py` (cross-site eval),
  `horizon_probe.py` (`--encode`/`--curve`/`--train`: multi-horizon gate),
  `horizon_eval.py` (per-horizon JEPA-vs-persistence for a trained leg),
  `pred_latent_probe.py` (predicted-vs-EMA latent probes; `--refit` field
  models + held-out preds, `--probe` classifier tables),
  `encode_interface.py` (per-modality/ROI/volumetry feature cache),
  `task_train.py` (frozen-backbone temporal+head task training),
  `finetune_lora.py` (supervised LoRA vision finetune),
  `train_supervised_cnn.py` (MONAI 3D ResNet-18 comparator),
  `cross_site_adapt.py` (zero-shot + K-shot CNN-vs-JEPA on SAILOR),
  `view_scans.py` (local browser NIfTI explorer),
  `shot_viewer.py` (headless screenshot validator for it; dev-only),
  fetch/auth scripts.
- `config/` — `default.yaml` (full run; `aux:` section, lambda 0 = JEPA
  only), `pilot.yaml` (5-patient CPU pilot).
- `kaggle/` — hero-run notebook. `hero_run.py` is the source of truth;
  never edit the `.ipynb` directly (JSON churn breaks diffs). Regenerate
  with `jupytext --to ipynb kaggle/hero_run.py` after editing.
  `kaggle/kernel-*/` are pushable-run variants (own `.py` source +
  `kernel-metadata.json`; shell commands LIVE — push executes the notebook
  as-is, so never `py_compile` them, only `jupytext --to ipynb`).
  `kernel-lora/` (supervised vision finetune, `finetune_lora.py`) and
  `kernel-cnn/` (supervised ResNet-18 comparator, `train_supervised_cnn.py`)
  are the current pushable legs.
- `info/` — decision log (`decisions.md`, IDs D0–), ablations (`ablations.md`,
  IDs A–/I–), pilot notes (`pilot.md`). Append-only; reference IDs.
  Plot numbers live in `info/plots/metrics.json` (single source of truth);
  `info/plots/make_plots.py` regenerates every plot from it — never
  hardcode numbers in a plot script.
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
# Local scan explorer (client-side NiiVue; binds 127.0.0.1 only)
python scripts/view_scans.py                                   # SAILOR derivatives
python scripts/view_scans.py --root data/lumiere_preprocessed  # LUMIERE 96³
#   deep-link: /?p=<rel>&ov=<rel>&ov=<rel>  (repeat ov for overlaid masks)
# Headless render check (viewer running with --no-browser):
python scripts/shot_viewer.py \
  "http://127.0.0.1:8765/?p=sub-01/ses-05/T1c.nii.gz&ov=sub-01/ses-05/EdemaMask-ONCO.nii.gz" \
  /tmp/opencode/shot.png
```

## Local scan explorer (`view_scans.py`)

Browser UI over a scan tree. NiiVue renders NIfTI entirely client-side, so the
bytes only travel disk → localhost → your own browser; nothing is uploaded.
Auto-detects the SAILOR derivatives layout (`sub-XX/ses-YY/*.nii.gz`) and the
LUMIERE preprocessed layout (`Patient-XXX/week-*/*.nii.gz`). Base modalities
(T1c/T1/T2/Flair/CT1/FLAIR) load as the background; masks, segmentations and
`-icor`/`-zscore` variants load as overlays. Deep-link for scripting/screenshots:
`/?p=<rel>&ov=<rel>&ov=<rel>` (repeat `ov`, comma-separate also works).

Per-patient/session **metadata** is read from the tree sidecars and shown in a
bottom panel (and age/OS inline in the patient list):
- SAILOR: subject `age-years.txt` / `overall-survival-months.txt` /
  `intervals-days.txt`; session `treatment.txt` / `RANO.txt` (codes decoded
  PD/SD/PR/CR).
- LUMIERE: `--meta-dir` (default `data/lumiere_meta`) joins demographics,
  RANO rating/rationale, and per-timepoint modality completeness onto the
  `Patient-XXX/week-*` tree.

- SAILOR is controlled-access: keep it on `127.0.0.1`. Do not port-forward
  beyond an SSH tunnel, and never upload the files to a cloud viewer.
- The NiiVue bundle is fetched once from jsDelivr into
  `~/.cache/world-model-viewer/` (pinned `0.69.0`); after that it runs offline.
- Validate viewer changes with `scripts/shot_viewer.py` (headless
  Chromium/WebGL via SwiftShader). It is a dev-only tool: `pip install
  playwright && playwright install chromium` (NOT in `requirements.txt`). Start
  the viewer with `--no-browser`, point the harness at a deep link; it polls
  until volumes load (software-WebGL shader compile is slow) and exits nonzero
  if nothing rendered — screenshot the result before trusting a change.

NiiVue 0.69 API gotchas (each cost a blank canvas, found via `shot_viewer.py`):

- `attachTo(id)` wants the id of a `<canvas>`, not a `<div>`.
- `attachTo` is async and resets volumes when it resolves — load volumes only
  after awaiting it, or they get wiped (looks like "0 volumes").
- `loadVolumes` APPENDS and 0.69 has no `removeAllVolumes`; clear via
  `removeVolume` first. The explorer serializes refreshes through a promise
  queue so concurrent deep-link loads cannot stack.
- Passing option objects with a `name` breaks `getFileExt(name||url)`; load by
  `{url}` only, then set `name`/`colormap`/`opacity` on `nv.volumes[i]`.

## Conventions (from README, enforced)

1. Never commit data, weights, checkpoints, or secrets.
2. Small atomic commits, imperative messages ("add X").
3. Code over notes; docs live in `info/` with D/A/I IDs.
4. Reproducibility: pin deps, record seeds/configs, tie results to commits.
5. Branch per experiment (`exp/...`), PRs not direct-to-main.
6. Artifacts carry provenance: every encoded cache records {champion path,
   config hash, git sha, date} at creation; loaders assert it. Pre-2026-09-10
   caches are filename-only (same bug class as the D22 champion overwrite).
7. Persist trained weights, not just scores, whenever a follow-up might need
   predictions (`train_field.py` saved scores only; recovering field
   predictions cost a full refit — now `checkpoints/field_models.pt`).
8. Update the record with the result: every experiment lands in `info/` (new
   D/A/R ID, append-only) and, when plots or conclusions change,
   `info/explainer.html` too. An unlogged result might as well not have happened.

## Gotchas (earned the hard way — see info/ablations.md)

- **Weights**: use official `BrainIAC.ckpt` only. Community ports can be
  silently corrupt (matching key names, degenerate outputs). Verify any
  checkpoint behaviorally: real-vs-noise drift must be ≫ 0.
- **Deps**: pin `transformers<5`; never install `torchvision` (a skewed
  build breaks peft import with a misleading error).
- **Data**: pandas NaN is truthy — coerce clinical numerics via explicit
  `isna` guards. Drop imageless visits; require pixels on both sides of
  every loss pair.
- **Probes**: single-split probe numbers are noise, and the legacy all-91
  `--cv` leaks encoder-train patients into probe folds (K3-16) — use the locked
  protocol (`info/eval_protocol.md`, `--cv-unseen`); even the readout seed
  alone spans 0.34–0.45 (A27). Ridge + CV-λ mandatory above ~100-d features
  (plain LSQ on 1152-d/450-row gives R²≈−100).
- **SAILOR**: `-icor` files carry background NaNs (~200 sessions, up to 79%
  of voxels) — prefer base variants; finite-check any new site before first
  encode. RANO codes {1:PD, 2:SD, 3:PR, 5:CR} are empirical (volume deltas);
  3-vs-5 tentative. Median inter-visit gap ~76 days (A14; earlier "~14 days"
  was wrong) → persistence regime (A12).
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
  with (leg, epoch, val). Select resume checkpoints BY DEFINITION (lowest
  val_loss), never by filename. Ferried checkpoints are opt-stripped —
  resume-opt designs need an optimizer-bearing file or must drop those legs.
- **Kaggle notebook parsing**: test every log-regex against a REAL log line
  before push (an untested `" epoch"`-vs-`"(epoch"` mismatch voided the R11
  verdict cell on complete logs). Verdict/eval cells record UNKNOWN, never
  assert-fail the session. Run train commands with `python -u`.
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
  (target std ≫ 0, rank > 1), and the same-space persistence gate is
  re-passed (A24: in-domain test pooled is a tie, train pooled loses;
  cross-site loses 7–10.5×) — a falling loss alone proves nothing.
