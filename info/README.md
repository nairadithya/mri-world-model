# Research documentation

**Study closed November 11, 2026.** No experiment, training, independent assessment, or reserved-cohort inspection is queued. The [integrated patient-state system](METHODS.md#integrated-patient-state-system-november-5-2026) was the last development direction; its milestones and outcomes are retained in the [dated work log](TODO.md#integrated-patient-state-development-november-5-2026). Open dependencies are deferred, not commitments to continue. Historical host paths and artifact availability describe their time of writing.

Start with the [detailed research summary through R62](RESEARCH_SUMMARY.md), dated November 1, 2026, and use the records below for subsequent work. The [November 3 observability program](TODO.md#observability-program-november-3-2026) and its [protocols](METHODS.md#six-question-observability-program-november-3-2026) are historical. Forty MU-Glioma-Post patients remain unopened.

The [conclusion through R71](NOTES_AND_INSIGHTS.md#current-conclusion-through-r71-november-4-2026) is useful but task-dependent representation learning, not an established MRI observability limit. The through-R62 summary remains a closed, dated synthesis. [R71](EXPERIMENTS.md#r71-retained-learned-segmentation-signal-but-failed-the-direct-image-retention-margin) is closed as learned transfer demonstrated, with the required direct-image retention failed. The representation-study sequence is closed. The subsequent [R67 history comparison](EXPERIMENTS.md#r67-failed-the-joint-patient-history-benefit-claim) completed with its joint claim failed. Genuine history did not establish lesion improvement beyond current panel C and had worse gradient error than both trained controls. This does not prove absent historical information.

## Canonical records

| Document | Owns |
|---|---|
| [Experiments](EXPERIMENTS.md) | Measurements, outcomes, controls, uncertainty, and withdrawals |
| [Methods](METHODS.md) | Definitions, fixed protocols, information cutoffs, and reproduction commands |
| [Notes and insights](NOTES_AND_INSIGHTS.md) | Reasoning, interpretation, and open scientific questions |
| [Research TODO](TODO.md#current-status-and-open-decisions) | Closure status, deferred dependencies, and the dated work log |

The summary is a synthesis of these records, not a separate result ledger. Update a fact in its owning page first. Historical experiment IDs and dated corrections remain intact.

## Find the evidence by question

| Question | Main records |
|---|---|
| Does the model recover current anatomy? | [R35 spatial state](EXPERIMENTS.md#r35), [R49 matched pretraining](EXPERIMENTS.md#matched-pretraining-result), [R52 token interfaces](EXPERIMENTS.md#r52-completed-without-an-established-token-replacement) |
| Does paired modeling add useful observed change? | [R35 paired controls](EXPERIMENTS.md#r35), [R36 frozen-readout comparison](EXPERIMENTS.md#r36-completed) |
| Does history help predict a future observation? | [R37 frozen spatial dynamics](EXPERIMENTS.md#r37-full-pilot), [R38 joint training](EXPERIMENTS.md#r38-completed), [R40 supplied horizon](EXPERIMENTS.md#r40-completed-trainable-spatial-horizon-pilot), [R47 measured-state controls](EXPERIMENTS.md#r47-anatomical-signal-robustness-and-locked-transfer-october-31-2026) |
| Does JEPA add value beyond direct supervision? | [R50/R51 clinical comparisons](EXPERIMENTS.md#r50-v3-and-r51-v3-limited-adaptation-clinical-comparison), [R58 anatomical forecasting](EXPERIMENTS.md#r58-did-not-establish-a-future-feature-or-history-forecasting-gain) |
| Can the spatial interface recover images? | [R53 completion and forecasting](EXPERIMENTS.md#r53-assessment-found-no-established-forecasting-benefit), [R54 fitting diagnosis](EXPERIMENTS.md#r54-learned-corrections-but-failed-substantial-individual-fitting), [R55 encoder bypass](EXPERIMENTS.md#r55-code-bypass-did-not-resolve-image-fidelity) |
| Do regional assimilation and context help? | [R56 assimilation](EXPERIMENTS.md#r56-did-not-establish-a-useful-learned-assimilation-gain), [R57 graph context](EXPERIMENTS.md#r57-regional-context-helped-without-established-adjacency-or-history-benefit) |
| Are the regional measurements valid? | [R59 diagnosis](EXPERIMENTS.md#r59-source-mask-inspection-confirmed-coarse-target-information-loss), [R60 source-volume correction](EXPERIMENTS.md#r60-quantified-and-corrected-coarse-regional-target-discrepancies), [target contract](METHODS.md#r60-source-volume-regional-targets) |
| Does corrected measured history help? | [R61 LUMIERE comparison](EXPERIMENTS.md#r61-corrected-targets-did-not-establish-useful-history-forecasting), [R62 MU comparison](EXPERIMENTS.md#r62-did-not-establish-useful-history-forecasting-on-mu) |
| Which early claims were corrected or withdrawn? | [Withdrawals](EXPERIMENTS.md#explicit-withdrawals-and-limits), [historical experiment index](EXPERIMENTS.md#historical-experiment-index), [direction index](METHODS.md#r-directions) |

Each experiment section links its aggregate and protocol. LUMIERE and SAILOR results are developmental evidence. MU's pilot-scoring patients are now development evidence too; its reserved patients remain unopened.

## Evidence and presentation

- The authoritative study report is the private local [LaTeX source](../docs/report/report.tex) and [PDF](../docs/report/report.pdf). The report package includes controlled figures; treat it as private.
- The final LaTeX package is also preserved in `outputs/cleanup-20261111/final-report-source.tar.gz`; superseded report versions are in `outputs/cleanup-20261111/report-archive`. Preservation is layered: verified November 6 archives retain original materials including selected observation models; the new compact snapshot, `outputs/cleanup-20261111/study-closeout-compact-20261111.tar.zst`, is planned to retain subsequent checkpoints, evidence, source, and configurations. It excludes already archived originals and regenerable timelines, state, context, and prepared caches. The per-file plan is `outputs/cleanup-20261111/remote-compact-preservation-plan.json`. Preservation and remote cleanup remain in progress.

- The [figure gallery through R62](plots/current/README.md) presents aggregate-based comparisons with captions and print-ready exports. The [figure review](plots/REVIEW.md) records coverage, corrected presentations, and historical figures that are withdrawn or superseded.
- The [HTML technical handbook](technical_handbook.html) teaches prerequisites and the earlier research through R36. Its [maintenance guide](handbook/README.md) describes the build. Use the R62 summary for the later work.
- The [generated R36/R37 research page](research_run.html) presents those studies. `scripts/refresh_research_run.py` generates its current-result sections. It is not a full-program dashboard.
- The [workshop manuscript](workshop_manuscript.typ) and [compiled PDF](workshop_manuscript.pdf) are a draft centered on earlier studies. The [publication assessment](NOTES_AND_INSIGHTS.md#workshop-manuscript-feasibility) retains the limits on novelty and acceptance.
- R35 has a [result review](reviews/r35_result_review.md), [artifact audit](reviews/r35_artifact_audit.md), and [closure receipt](reviews/r35_closure_receipt.json).
- R36 Phase 1 has a [result review](reviews/r36_phase1_result_review.md), [artifact audit](reviews/r36_phase1_artifact_audit.md), and [closure receipt](reviews/r36_closure_receipt.json).
- R37 has a [result review](documentation_sources/updates/2026-10-26/reviews/result-v1.md), [executed checker](documentation_sources/updates/2026-10-26/reviews/result-replay-v1.py), [replay receipt](documentation_sources/updates/2026-10-26/reviews/result-replay-v1.json), and [closure receipt](documentation_sources/updates/2026-10-26/closure.json).

These reviews permit scoped reporting. Later numerical replay receipts do not imply the same independent scientific review. Raw generator-time status fields remain historical where a separate closure receipt supersedes them. Individual inventories, predictions, images, and weights remain private.

## Source history

[Source coverage](SOURCE_COVERAGE.md) maps the original 72 captured Markdown versions across 48 paths and ten frozen branch refs. The [catalog](documentation_sources/catalog.json) records their origins. These are dated capture counts, not totals for all later research.

The [October 13 coverage update](documentation_sources/updates/2026-10-13/coverage.json), [October 13 archive](archive/2026-10-13/README.md), and [October 12 archive](archive/2026-10-12/README.md) preserve the earlier records. The [preparation record](documentation_sources/checkpoint.md) and [consolidation plan](documentation_sources/updates/2026-10-13/plan.md) explain the original documentation work. Never edit archived files or versioned source snapshots in place.

## Code and reproduction index

- [Patient-state CLI](../scripts/patient_state.py) and [training module](../src/patient_state/training.py): integrated observation and timeline work; see the [system protocol](METHODS.md#integrated-patient-state-system-november-5-2026).
- [Harness CLI](../scripts/harness.py): registered earlier training, evaluation, and analysis commands; see [Methods](METHODS.md) for study-specific settings.
- [Scripts guide](../scripts/README.md) and [preprocessing contract](preprocessing.md): data preparation and access.

Reproduction depends on authorized data, external weights, retained artifacts, and a suitable environment; their availability is not guaranteed by the presence of source code. Controlled-access data, individual predictions, and weights are not part of the public research record.

## Operating guides and maintenance

- [Project README](../README.md): setup and common commands.
- [Repository guide](../AGENTS.md#maintain-the-research-documentation): documentation ownership and verification.
- [Research guardrails](research_guardrails.md): experiment scope and the decision brief.
- [Preprocessing](preprocessing.md): BRAINIAC input contract.
- [SAILOR reprocessing](../src/preprocessing/reprocess_sailor.md): that procedure and its limits.
- [Acquisition scripts](../scripts/README.md): data access tools.
- [CLI verification](../.agents/skills/verify-world-model-cli/SKILL.md): real command-line checks.

Run the documentation checks from the repository root:

```bash
python3 -B -m unittest discover -s tests -p 'test_documentation*.py' -v
python3 -B scripts/documentation_sources.py verify
python3 -B scripts/build_technical_handbook.py --check
```

These checks verify documentation structure and preserved evidence bytes. They do not establish scientific acceptance.
