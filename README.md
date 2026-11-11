# world-model

**Study closed November 11, 2026.** This repository preserves research code and evidence for longitudinal glioma MRI. No further experiment, validation, or reserved-patient inspection is queued.

The [final study synthesis](info/STUDY_SUMMARY.md) summarizes results through November 9: the study found useful current-anatomy and observed-change signals, but did not establish a useful, transferable longitudinal world model. Its tested history, quantity, clinical-objective, and adaptive-acquisition claims have task-specific limits; they do not establish an MRI information limit.

The authoritative private report is the local [LaTeX source](docs/report/report.tex) and [PDF](docs/report/report.pdf). The package contains controlled figures and is not a public distribution.

- [Research index](info/README.md): evidence, protocols, interpretation, and historical records.
- [Command guide](scripts/README.md): research entry points and data access.
- [Artifact catalog](info/ARTIFACTS.md): preserved materials and cleanup status.
- [Study closure and deferred dependencies](info/TODO.md).
- [Figures](info/plots/README.md): figure sources and interpretation.

## Setup

The source requires an appropriate environment and, for full reproduction, authorized datasets, external weights, and retained artifacts. Availability is not guaranteed by the source code. See the [command guide](scripts/README.md), [preprocessing contract](info/preprocessing.md), and [repository guide](AGENTS.md). Never commit data, checkpoints, or secrets.

```bash
uv sync
uv sync --extra dev       # notebook tools
uv sync --extra viewer    # scan viewer screenshot checks
```

The historical [through-R62 synthesis](info/RESEARCH_SUMMARY.md) and [earlier workshop manuscript](info/workshop_manuscript.typ) remain available as dated records.
