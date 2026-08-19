# world-model

Research repo for building / studying world models. Starting fresh with a clean, reproducible setup.

## Layout

- `data/` — datasets and raw inputs. **Not tracked by git** (see `.gitignore`); fetch or symlink locally.
- `scripts/` — runnable code (data acquisition, probes, utilities).
- `src/` — library code (to be added).
- `experiments/` — per-experiment configs, logs, and results (to be added).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r scripts/requirements.txt   # once added
```

## Research practice conventions

1. **Never commit data or model weights.** Keep them in `data/` / `checkpoints/`, which are gitignored. Document how to obtain them instead.
2. **Small, atomic commits.** One logical change per commit; write messages in the imperative ("add lumiere fetch script").
3. **Code over notes.** Prefer executable scripts and self-documenting code to free-form markdown notes. If a doc is needed, keep it close to the code it describes.
4. **Reproducibility.** Pin dependencies, record seeds/configs, and make every result traceable to a script + commit hash.
5. **Branch per experiment.** Use feature branches (`exp/...`) and open PRs rather than committing straight to `main`.
