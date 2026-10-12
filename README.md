# world-model

Research code for longitudinal glioma MRI. Start with
[Notes and insights](info/NOTES_AND_INSIGHTS.md) for the research questions and
findings. [Experiments](info/EXPERIMENTS.md) records results and corrections,
[Methods](info/METHODS.md) defines protocols, and
[Research TODO](info/TODO.md) lists work left at the archived snapshot.
The [research index](info/README.md) links the source history and operating guides.
The [archive](info/archive/2026-10-12/README.md) preserves the original documentation.

LUMIERE is the development cohort. SAILOR is controlled-access data used for
developmental transfer, not an untouched final test set. The research objective
and binding experiment guardrails are in
[`info/research_guardrails.md`](info/research_guardrails.md).

## Setup

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
