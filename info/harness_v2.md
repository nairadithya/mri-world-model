# Anatomy-first harness v2 migration

Status: first compatibility boundary implemented (2026-09-22).

The default CLI now presents the active anatomy workflow:

```bash
python scripts/harness.py list
python scripts/harness.py anatomy manifest
python scripts/harness.py anatomy baseline --boot 10000
python scripts/harness.py checkpoint inspect checkpoints/best.pt
```

Historical RANO workflows remain reproducible under an explicit namespace:

```bash
python scripts/harness.py legacy list
python scripts/harness.py legacy eval --task rano4_forecast ...
python scripts/harness.py legacy train jepa ...
python scripts/harness.py legacy run forecast-baselines ...
```

The old top-level `encode`, `eval`, `train`, and `run` forms remain temporary
aliases and print a deprecation notice. No historical implementation has been
deleted.

## Checkpoint compatibility contract

`src/harness/checkpoints.py` accepts:

- historical `{model: state_dict, opt: ..., epoch: ..., val_loss: ...}` files;
- `{state_dict: ...}` and `{weights: ...}` containers;
- raw state dictionaries;
- `module.`-prefixed data-parallel tensors.

It loads only name-and-shape-compatible tensors and returns a structured audit
containing loaded, missing, unexpected, ignored-legacy, and shape-mismatched
keys. Shape conflicts fail loudly. A historical `rano_heads.*` branch may be
ignored when loading into a future anatomy model that does not define it; all
other unexpected tensors are reported. New lesion heads can be explicitly
declared optional when an old encoder initializes them from scratch.

The compatibility loader is now used by the canonical frozen encoder,
interface cache builder, JEPA resume path, and best-checkpoint reload. Further
legacy analysis modules may migrate to it as they are touched; their explicit
legacy namespace already prevents them from being mistaken for P2 commands.

## Anatomy data and residual slice

The second slice is implemented:

```bash
python scripts/harness.py anatomy features
python scripts/harness.py anatomy residual
```

The first command builds the history-only `lesion-physical-v1` feature cache
from native masks/affines. The second exposes continuous anatomy rows, runs the
locked residual gate, and persists the final model plus feature/scaler
provenance. This remains a deterministic structured forecaster, not yet a
lesion-aware encoder trainer. The RANO heads stay in the historical model
temporarily for binary checkpoint compatibility, with zero auxiliary weight in
active configurations.
