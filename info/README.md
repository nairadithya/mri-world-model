# Research documentation

Start with the [HTML technical handbook](technical_handbook.html) for a prerequisite-first primer, a detailed experiment account, and a candid publication assessment. It is an educational guide, not a fifth result ledger. Its [maintenance guide](handbook/README.md) defines the deterministic build and review procedure.

The four canonical documents own the research record:

- [Notes and insights](NOTES_AND_INSIGHTS.md) explains the questions, findings, and limits.
- [Experiments](EXPERIMENTS.md) records measurements, controls, corrections, and withdrawals.
- [Methods](METHODS.md) defines terms, protocols, reproduction commands, and historical IDs.
- [Research TODO](TODO.md) lists current remaining work and its authorization.

The latest R36 closure is October 13, 2026. Phase 1 computation and scoped reporting review are complete. The primary encoder gain remains inconclusive, C passes the specified direct-volume retention margin, and Phase 2 remains unauthorized. [Completed findings](EXPERIMENTS.md#r36-completed) and [D58 interpretation](NOTES_AND_INSIGHTS.md#d58) explain those distinctions. Earlier October 11 runtime snapshots remain historical, not live job status.

The [R37 frozen-state dynamics pilot](METHODS.md#r37) completed on October 26, 2026, with independent approval for scoped reporting of **INCONCLUSIVE_FOR_FULL_CAPABILITY**. All fixed arms and three neural seeds completed, but added-history, earlier-spatial-detail and secondary latest-state benefits were not established. [Completed findings](EXPERIMENTS.md#r37-full-pilot), [interpretation](NOTES_AND_INSIGHTS.md#r37-completed-without-an-established-history-benefit), [result review](documentation_sources/updates/2026-10-26/reviews/result-v1.md) and [closure receipt](documentation_sources/updates/2026-10-26/closure.json) give the controls and limits. The pilot is closed; no rescue training or historical R36 Phase 2 follows.

The October 24, 2026, [workshop manuscript assessment](NOTES_AND_INSIGHTS.md#workshop-manuscript-feasibility) narrows the earlier unverified publication framing. Primary literature already covers paired-versus-state comparisons and frozen change probing. A matched R36 objective study remains a conditional short-paper candidate, not established novelty or a demonstrated world model. The [bounded source check](documentation_sources/updates/2026-10-24/literature-check.md) and [unrendered figure specification](documentation_sources/updates/2026-10-24/figure-spec.json) preserve its evidence and limits. No training or submission was authorized by that check.

The operator-authorized [R37 continuation](documentation_sources/updates/2026-10-26/r37-continuation.md) ran on SSH `mosaic-gpu`, not the laptop or a hosted Kaggle GPU. Earlier expenditure remains separate, including unknown original v1 cost. Archival acceptance is preferred; non-archival acceptance is a fallback requiring agreement. Computational completion does not establish novelty, a confirmed archival route or acceptance.

## Evidence and presentation

The [current research page](research_run.html) presents the aggregate findings. Its generator is `scripts/refresh_research_run.py`. [Aggregate R36 measurements](r36_phase1_metrics.json) retain all arms, companions, and input hashes. The exact [R37 raw aggregate](r37_full_metrics.json) retains seed, calibration, uncertainty, nuisance, single-visit and empty-compartment companions. Its generator-time pending-review marker is preserved; the separate bound [R37 result review](documentation_sources/updates/2026-10-26/reviews/result-v1.md) supplies scoped approval. The [independent checker](documentation_sources/updates/2026-10-26/reviews/result-replay-v1.py), [replay receipt](documentation_sources/updates/2026-10-26/reviews/result-replay-v1.json) and [execution guide](documentation_sources/updates/2026-10-26/r37-execution.md) preserve the checked scope. Individual inventories, predictions and weights remain private.

The [independent result review](reviews/r36_phase1_result_review.md), [artifact audit](reviews/r36_phase1_artifact_audit.md), and [closure receipt](reviews/r36_closure_receipt.json) retain their exact bytes and checked scope. They permit bounded reporting, not global representation-quality, biological-change, or forecasting claims.

R35 computation and scoped result review also closed on October 13, 2026. Its [result review](reviews/r35_result_review.md), [corrected audit](reviews/r35_artifact_audit.md), [closure receipt](reviews/r35_closure_receipt.json), and [evidence guide](reviews/evidence/r35/README.md) preserve the checked scope and prior audit corrections. [Canonical findings](EXPERIMENTS.md#r35) retain unresolved paired benefit, failed or unresolved developmental transfer, and target/provenance limits. Earlier unavailable-review statements remain historical.

## Source history

[Source coverage](SOURCE_COVERAGE.md) maps the original 72 captured Markdown versions across 48 paths and ten frozen branch refs. The [catalog](documentation_sources/catalog.json) records their origins. That is a dated capture, not a count of every later document.

The [October 13 update](documentation_sources/updates/2026-10-13/coverage.json) maps the newly incorporated records and restored duplicates. The [October 13 archive](archive/2026-10-13/README.md) preserves ten relocated files. The [October 12 archive](archive/2026-10-12/README.md) preserves 32 earlier research records and five presentation assets. Both retain original bytes.

Old plans, runtime claims, and embedded links are historical. Later corrections govern interpretation. The [original preparation record](documentation_sources/checkpoint.md) and [new update plan](documentation_sources/updates/2026-10-13/plan.md) disclose review and verification limits.

## Operating guides and maintenance

Keep operating documents separate:

- [Project README](../README.md) covers setup and common commands.
- [Agent documentation procedure](../AGENTS.md#maintain-the-research-documentation) defines ownership, archival, and update checks.
- [Research guardrails](research_guardrails.md) define the objective and required experiment decision brief.
- [Preprocessing](preprocessing.md) defines BRAINIAC inputs.
- [SAILOR reprocessing](../src/preprocessing/reprocess_sailor.md) preserves that procedure and its limits.
- [Acquisition scripts](../scripts/README.md) describe data access tools.
- [CLI verification skill](../.agents/skills/verify-world-model-cli/SKILL.md) defines real command-line checks.

Add measurements to Experiments, reasoning to Notes and insights, protocols to Methods, and actions to TODO. Keep historical IDs, corrections, archive bytes, and review scope intact. Do not create a second active research narrative.

Regenerate original coverage with `python3 -B scripts/render_documentation_coverage.py`. Verify the canonical package and all archives with `python3 -B scripts/documentation_sources.py verify`. The later update's reproducible checks are documented in its plan. Passing these checks is not scientific acceptance.
