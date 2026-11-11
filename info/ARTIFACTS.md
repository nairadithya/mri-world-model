# Artifact catalog

The study closed November 11, 2026. This catalog maps the retained evidence and private preservation set; it does not authorize new training or opening the forty reserved MU patients. Private archives, models, individual predictions, and controlled figures stay outside Git.

## Authoritative report

| Material | Location |
|---|---|
| Current LaTeX source and build instructions | [Report source](../docs/report/report.tex), [build instructions](../docs/report/README.txt) |
| Current compiled report | [Report PDF](../docs/report/report.pdf) |
| Verified backup of the complete current package | `outputs/cleanup-20261111/final-report-source.tar.gz` and [verification receipt](../outputs/cleanup-20261111/final-report-source.receipt.json) |
| Superseded reports and build intermediates | `outputs/cleanup-20261111/report-archive/`; [original paths and hashes](../outputs/cleanup-20261111/report-archive/manifest.json) |

The report package contains controlled research figures. Its earlier versions and the separate historical manuscript are not the authoritative report.

## Research preservation

Preservation uses the verified November 6 archives plus the November 11 compact archive. Retain both layers and their receipts together.

| Layer | Contents and verification |
|---|---|
| `outputs/mosaic-archive-20261106/` | Original inputs, selected observation models, earlier study code, measurements, and provenance. [Archive receipts](../outputs/mosaic-archive-20261106/archives-complete.json), [recorded local digest verification](../outputs/mosaic-archive-20261106/laptop-verified.json), and [reconstruction index](../outputs/mosaic-archive-20261106/reconstruction-index.jsonl). |
| `outputs/cleanup-20261111/study-closeout-compact-20261111.tar.zst` | Subsequent checkpoints, evaluations, code, configuration, and provenance. Local transfer and final GPU retirement are still in progress. |
| [Per-file retention plan](../outputs/cleanup-20261111/remote-compact-preservation-plan.json) | Distinguishes new retained material, previously archived originals, and regenerable input/state/context/timeline caches. |
| [Historical archive inventory](../outputs/cleanup-20261111/retained-historical-archive-index.json) | All fourteen earlier archives remain present with their recorded sizes; digest verification is retained from the earlier receipt. |

The compact archive was compared against its GPU source files before transfer. Its local checksum must pass before remote removal. The temporary full snapshot and transfer chunks are working files, not additional preservation layers.

### Earlier archive map

All files below are under `outputs/mosaic-archive-20261106/` and have the suffix `.tar.zst`.

| Archive stem | Study family |
|---|---|
| `world-model-observability-20261103` | Observability program and its source records |
| `world-model-r65-20261103` | R65 fitting and preparation |
| `world-model-r65-lumiere-20261103` | LUMIERE source and preparation |
| `world-model-r66-20261103` | R66 shared-state study |
| `world-model-r66-ce-diagnostic-20261104` | R66 weighting diagnostic |
| `world-model-r66-flair-assessment-20261104` | R66 FLAIR assessment |
| `world-model-r67-preparation-20261104` | R67 preparation |
| `world-model-r67-caliper8-20261104` | R67 history comparison |
| `world-model-state-fitting-20261104` | Privileged state-fitting diagnostic |
| `world-model-spatial-access-20261104` | R69 spatial access |
| `world-model-frozen-readout-20261104` | R70 compact readout |
| `world-model-complete-state-footprint-20261104` | Complete-state footprint |
| `world-model-complete-reuse-20261104` | R71 complete-state reuse |
| `world-model-patient-state-20261105` | Integrated observation system, source inputs, and observation models |

### Later study map

The compact archive retains the original experiment-root names. Existing local result copies provide shorter entry points:

| Study family | Local result records | Public evidence |
|---|---|---|
| Current-only cavity forecasting | `outputs/mu-cavity-forecast-20261107/` | [Results](patient_cavity_forecasting_metrics.json), [closure](patient_cavity_forecast_closure.json) |
| Amount-first and Gaussian quantity | `outputs/mu-amount-first-20261108/`, `outputs/mu-gaussian-count-20261108/` | [Amount-first](patient_amount_first_metrics.json), [Gaussian](patient_gaussian_count_metrics.json) |
| Frozen-state information program | `outputs/mu-information-program-20261108/` | [Completed findings](EXPERIMENTS.md#frozen-state-information-program-no-eligible-predictive-configuration-november-8-2026); [recorded protocol](METHODS.md#frozen-state-information-program-november-8-2026) |
| Scalar versus PCA-MRI quantity | `outputs/mu-linear-quantity-20261108/` | [Results](patient_linear_quantity_metrics.json) |
| Acquisition JEPA | `outputs/acq-jepa-v1/` | [Results](patient_acquisition_jepa_metrics.json) |
| R68 clinical-context preparation | `outputs/r68-preparation-20261109/` | [Results](r68_preparation_metrics.json) |

The [experiment ledger](EXPERIMENTS.md) owns outcomes and the [protocol ledger](METHODS.md) owns reproduction details. Other private historical fits and source snapshots remain under `outputs/`, `checkpoints/`, and `scratch/`; their paths in the research record are historical. This catalog does not imply that every historical weight or array is available.

## Restoring a study

1. Start with its recorded protocol and aggregate result. Identify the selected checkpoint and information cutoff before restoring files.
2. Consult both archive layers and the reconstruction index. Restore into a new authorized compute environment; earlier absolute host paths are historical.
3. Recreate omitted caches from the archived inputs and code. Restore symlinks using the [recorded link map](../outputs/cleanup-20261111/remote-symlinks.txt) where needed.
4. Check identities and saved-artifact replay before numerical work. Do not use reserved patients for restoration or verification.

Use the [command index](../scripts/README.md) to find tools. Source code alone does not supply data access, external weights, an approved environment, or authorization to reopen the study.

## Historical documentation and figures

The [dated work log](archive/2026-11-11/WORK_LOG.md), [source coverage](SOURCE_COVERAGE.md), and immutable earlier archives preserve decisions and corrections. The [figure guide](plots/README.md) separates the curated gallery, report figures, and historical exports. The [November 1 synthesis](RESEARCH_SUMMARY.md), [workshop draft](workshop_manuscript.typ), and [R36/R37 page](research_run.html) retain their original scope.
