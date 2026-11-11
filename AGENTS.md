# AGENTS.md — working guide for this repo

**Study closed November 11, 2026.** Preserve the code and historical record; no new experiment, training, or reserved-cohort inspection is queued. Reopening requires a new explicit research decision. Historical host paths and execution instructions below describe prior work, not a claim that those resources remain available.

Research repo: longitudinal glioma MRI using LUMIERE for development and SAILOR for developmental transfer. The research direction is **world models for MRI**: maintaining a patient-state representation from available observations and predicting aspects of future observations (with uncertainty) while separating acquisition nuisance.

Read `info/README.md` for the documentation map and `info/RESEARCH_SUMMARY.md` for the dated synthesis through R62. Use `info/TODO.md` for deferred dependencies and the work log, `info/EXPERIMENTS.md` for completed outcomes, `info/METHODS.md` for protocols, and `info/NOTES_AND_INSIGHTS.md` for reasoning. Before implementing a new experiment, follow [`info/research_guardrails.md`](info/research_guardrails.md).

R62 added MU-Glioma-Post for development. Its forty hash-reserved patients remain unopened; do not load their images or segmentation measurements as part of routine development or verification. The fixed membership and cohort roles are documented in `info/METHODS.md`.

## Research direction and workflow rules

1. **Hypothesis-driven world-model research.** Start a new experiment with one capability under test, a falsifiable hypothesis, and matched baselines (including latest-scan-only and persistence controls when testing longitudinal benefit). Respect the prediction-time information cutoff and evaluate anatomically meaningful outputs, not only latent error.
2. **Bound data and target checks to the experiment.** Never bypass known leakage, invalid targets, privacy terms, or broken geometry, but do not let data auditing or infrastructure perfection become an open-ended prerequisite program. Automated masks are development proxies, not expert-verified biological truth.
3. **Keep execution lean — no compliance theater:**
   - **Direct implementation on one branch.** Do not split one experiment across multiple git worktrees, spawn competing design subagents for straightforward code, or run multi-round review gates before testing if the code runs.
   - **Never build custom process/budget-killing supervisors.** Compute budgets are scoping guidelines, not software lockouts. Do not wrap training runs in custom process-tree monitors or hash-locked budget ledgers that abort long GPU runs on transient errors.
   - **Run a quick real-data smoke test first.** Verify 1–2 real batches end-to-end (load → train step → save/reload → score) on `mosaic-gpu` before launching a multi-seed training matrix.

## Compute & execution (`mosaic-gpu` first)

- **Primary GPU host (`mosaic-gpu` over SSH):** Do not run heavy tests, feature extraction, or training on this laptop (CPU-only). Run GPU workloads, tests, and experiments over SSH on `mosaic-gpu` (including workloads that would previously have gone to Kaggle).
- **Local laptop:** Use only for lightweight editing, git operations, doc checks, and SSH orchestration.
- **Privacy:** SAILOR is controlled-access. Do not upload SAILOR data to cloud viewers or public hosts. Never commit `data/`, `checkpoints/`, `outputs/`, `.venv/`, or `.env`.

## Layout

- `src/` — library code:
  - `data/` — LUMIERE dataset/collate/splits, SAILOR adapter (`sailor.py`).
  - `model/` — spatial state & dynamics (`spatial_dynamics.py`), BRAINIAC+LoRA, clinical encoders, fusion, temporal transformer, predictor, gap-conditioned horizon head, velocity-field dynamics (`dynamics_field.py`), EMA target, JEPA loss, RANO aux (`heads.py`).
  - `train/` — trainer, baselines.
  - `preprocessing/` — BRAINIAC contract pipeline.
  - `harness/` — modular train/eval harness (`cli`, registries, `ReadoutEvaluator`, `analysis/` for R35–R37 spatial state, delta representation, and spatial dynamics runners).
- `scripts/` — runnable entry points:
  - `run_spatial_dynamics.py` & `run_spatial_noise_ablation.py` — R37 frozen-state spatial dynamics pilot and empirical-change ablation.
  - `harness.py` — unified CLI (`encode`, `eval`, `train <jepa|field|head|lora|cnn3d|cnn2d|readout>`, `run <name>`, `list`).
  - `refresh_research_run.py` — regenerates `info/research_run.html` from aggregate JSON metrics.
  - `build_technical_handbook.py` — builds and checks `info/technical_handbook.html` from `info/handbook/`.
  - `documentation_sources.py` & `render_documentation_coverage.py` — verifies archive hashes, local links, and historical ID coverage.
  - Data & viewer scripts: `preprocess.py`, `preprocess_qa.py`, `bet_repair.py`, `raw_mni_deltas.py`, `view_scans.py`, `shot_viewer.py`.
- `info/` — canonical research record:
  - `TODO.md` — closure status, deferred dependencies, and dated work log.
  - `EXPERIMENTS.md` — measurements, controls, uncertainty, corrections, and withdrawals (preserving historical D, A, I, R, G, V, E, and K3 IDs).
  - `METHODS.md` — definitions, fixed protocols, evaluation rules, and reproduction commands.
  - `NOTES_AND_INSIGHTS.md` — reasoning, interpretation, and open questions.
  - `technical_handbook.html` — generated prerequisite-first educational guide (edit `info/handbook/` and run `python3 -B scripts/build_technical_handbook.py --check`).
  - `archive/` & `documentation_sources/` — immutable historical records and snapshots. Never edit archived files in place.
- `kaggle/` — legacy/pushable Kaggle notebooks (`.py` is source of truth; keep `.ipynb` synchronized via `uv run --extra dev jupytext --to ipynb <file>.py`).

## Maintain the research documentation

When recording new experimental results or updating docs:

1. **Put each fact in its owning page:** actions/priorities in `info/TODO.md`, measurements/outcomes in `info/EXPERIMENTS.md`, protocols in `info/METHODS.md`, and interpretation in `info/NOTES_AND_INSIGHTS.md`.
2. **Preserve historical integrity:** Keep existing D, A, I, R, G, V, E, and K3 IDs. Do not erase earlier negative or withdrawn results; supersede old statuses with dated entries. Never modify files inside `info/archive/` or `info/documentation_sources/versions/`.
3. **Write plain, direct English:** Define shorthand and keep tables readable. Keep private participant inventories, raw images, predictions, and weights out of git.
4. **Run the fast verification suite before committing doc updates:**
   ```bash
   python3 -B -m unittest discover -s tests -p 'test_documentation*.py' -v
   python3 -B scripts/documentation_sources.py verify
   python3 -B scripts/build_technical_handbook.py --check
   ```

## Setup & common commands

```bash
uv sync                  # create .venv from uv.lock (uv >= 0.9.8)
uv sync --extra dev      # + jupytext (for notebook pair checks)
uv sync --extra viewer   # + playwright for shot_viewer.py
```

```bash
# Run spatial dynamics unit tests
.venv/bin/python -B -m unittest discover -s tests -p 'test_spatial*.py' -v

# Check notebook pairs (.py <-> .ipynb)
uv run --extra dev python -B scripts/check_notebook_pairs.py

# Local scan explorer (client-side NiiVue; binds 127.0.0.1 only)
.venv/bin/python scripts/view_scans.py                                   # SAILOR derivatives
.venv/bin/python scripts/view_scans.py --root data/lumiere_preprocessed  # LUMIERE 96³
```

## Key codebase gotchas

- **Deps**: pin `transformers<5`; never install `torchvision` in the local CPU venv (a skewed build breaks `peft` imports; see D16/D37).
- **Data**: pandas `NaN` is truthy — coerce clinical numerics via explicit `isna` guards. Drop imageless visits; require pixels on both sides of every loss pair.
- **LUMIERE timing**: LUMIERE visit labels use week bins (`week-XXX`), not exact scan-day intervals; suffixes (`-1`, `-2`) indicate visit order within a week, not 1-day gaps.
- **SAILOR**: `-icor` files carry background NaNs (~200 sessions) — prefer base variants; finite-check any new site before encoding. Median inter-visit gap is ~76 days (A14).
- **Gitignore**: anchor data-dir rules (`/data/`, not `data/`) so `src/data/` is never ignored.
- **Figures & HTML**: generate plots from JSON metrics (`info/plots/make_plots.py`), never hardcoded numbers. Visually inspect rendered plots or HTML pages with the image-aware read tool before committing layout changes.

## Enforced agent rules

| Rule | Enforcement |
|---|---|
| Kaggle `.py` notebook sources and executable `.ipynb` pairs must stay synchronized; `.py` is the source of truth. | `uv run --extra dev python scripts/check_notebook_pairs.py` (also run in `.github/workflows/notebook-pairs.yml`); regression fixture: `tests/test_notebook_pair_check.py`. |
| Consolidated documentation links, archive hashes, and baseline IDs must remain valid. | `python3 -B scripts/documentation_sources.py verify` and `python3 -B -m unittest discover -s tests -p 'test_documentation*.py' -v`. |
