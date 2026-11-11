# world-model

**Study closed November 11, 2026.** This repository preserves research code and
records for longitudinal glioma MRI; no further experiment or validation is
queued. Start with the
[detailed research summary through R62](info/RESEARCH_SUMMARY.md), dated November 1,
2026. The work supports limited anatomical learning but has not established a
useful, transferable longitudinal world model.

Later development improved current MU anatomy with cavity supervision, but native
LUMIERE anatomy and missing-FLAIR lesion detail showed tradeoffs on reused
selection data ([observation comparison](info/EXPERIMENTS.md#cavity-supervision-completed-mu-anatomy-improves-joint-retention-unresolved-november-6-2026)).
The subsequent [quantity comparison](info/EXPERIMENTS.md#regularized-scalar-versus-pca-mri-quantity-study-failed-both-gates-november-8-2026)
failed its forecast gates. In the [acquisition study](info/EXPERIMENTS.md#acquisition-jepa-completed-no-adaptive-acquisition-gain-november-9-2026),
all policies chose the same sequence path; the acquisition claim failed and
anatomical retention was not established, while missing-FLAIR completion
retention passed. These are developmental results, not independent confirmation.

The [current conclusion through R71](info/NOTES_AND_INSIGHTS.md#current-conclusion-through-r71-november-4-2026)
is that the models learn useful but task-dependent representations. R71 demonstrated
learned transfer but failed the required retention against direct-image adaptation,
despite a 334.75 MiB-per-visit state. The representation-study sequence is closed.
The subsequent [R67 comparison](info/EXPERIMENTS.md#r67-failed-the-joint-patient-history-benefit-claim)
did not establish useful added history beyond current panel C. Its joint claim
failed, with lesion improvement unresolved and gradient error worse than both
trained controls. This is not proof that history contains no additional information.

The authoritative report is the private local [LaTeX source](docs/report/report.tex)
and its [PDF](docs/report/report.pdf). The report package includes controlled
figures and is not a public distribution.

The [research index](info/README.md) links results, protocols, interpretations,
and operating guides. [Research TODO](info/TODO.md#current-status-and-open-decisions)
records deferred dependencies and completed work. The
[HTML technical handbook](info/technical_handbook.html) teaches prerequisites and
the earlier program through R36. The [archive](info/archive/2026-10-12/README.md)
preserves the original documentation.

LUMIERE is the main development cohort. SAILOR is controlled-access developmental
transfer data, not an untouched final test set. R62 added MU-Glioma-Post for a
reviewed-mask development pilot and reserved forty patients whose images remain
unopened. The research objective and experiment guardrails are in
[`info/research_guardrails.md`](info/research_guardrails.md).

## Setup

The last integrated build was the [integrated patient-state system](info/METHODS.md#integrated-patient-state-system-november-5-2026): one saved spatial state, current anatomy, sequence completion, and a patient timeline in the existing viewer. The initial demonstration used R69/R71 checkpoints and explicitly labeled persistence forecasts; later work trained a shared observation model, added cavity supervision, and evaluated learned forecasts without establishing the predictive-state milestone. Its milestones and outcomes are recorded in [TODO](info/TODO.md#integrated-patient-state-development-november-5-2026); further development is deferred.

These historical entry points require separately restored inputs, weights, and an appropriate compute environment:

```bash
python scripts/patient_state.py --help
python -m src.patient_state.training --help
python scripts/evaluate_patient_state.py --help
python scripts/view_scans.py --root <timeline-output> --port 8775 --no-browser
```

The GPU environment used for this build added [requirements-patient-state.txt](requirements-patient-state.txt) to its existing CUDA/PyTorch environment. See the system protocol for data preparation, checkpoint initialization, training, evaluation and SSH viewer access.

```bash
uv sync
uv sync --extra dev       # notebook tools
uv sync --extra viewer    # scan viewer screenshot checks
```

The repository needs local data, official BRAINIAC weights, and an MNI152
template for full preprocessing and training. These files are not stored in
Git. See [`scripts/README.md`](scripts/README.md) for data acquisition and
[`info/preprocessing.md`](info/preprocessing.md) for the preprocessing
contract. Never commit data, checkpoints, or secrets.

## Common commands

```bash
# List registered harness commands.
uv run python scripts/harness.py list

# Run a one-epoch legacy JEPA smoke test with prepared LUMIERE data.
uv run python scripts/harness.py legacy train jepa --epochs 1 --batch-size 1 \
  --no-wandb --random-init

# Reproduce the RANO-free baseline evaluation.
uv run python scripts/harness.py anatomy baseline --boot 10000
```

Use `uv run python scripts/harness.py --help` for current CLI options. The
local scan viewer is `uv run python scripts/view_scans.py`; keep SAILOR access
on `127.0.0.1`.

## Repository map

- `src/` contains data, model, preprocessing, and harness code.
- `scripts/` contains runnable entry points.
- `config/` contains training configurations.
- `info/` contains current research contracts, results, and historical records.
- `kaggle/` contains notebook sources. Edit the `.py` source and regenerate
  notebooks with `jupytext`.

See [`info/README.md`](info/README.md) for research documents and
[`AGENTS.md`](AGENTS.md) for repository-specific development and verification
rules.
