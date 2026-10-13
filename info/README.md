# Research documentation

The four canonical documents own the research record:

- [Notes and insights](NOTES_AND_INSIGHTS.md) explains the questions, findings, and limits.
- [Experiments](EXPERIMENTS.md) records measurements, controls, corrections, and withdrawals.
- [Methods](METHODS.md) defines terms, protocols, reproduction commands, and historical IDs.
- [Research TODO](TODO.md) lists current remaining work and its authorization.

The latest R36 closure is October 13, 2026. Phase 1 computation and scoped reporting review are complete. The primary encoder gain remains inconclusive, C passes the specified direct-volume retention margin, and Phase 2 remains unauthorized. [Completed findings](EXPERIMENTS.md#r36-completed) and [D58 interpretation](NOTES_AND_INSIGHTS.md#d58) explain those distinctions. Earlier October 11 runtime snapshots remain historical, not live job status.

## Evidence and presentation

The [current research page](research_run.html) presents the aggregate findings. Its generator is `scripts/refresh_research_run.py`. [Aggregate R36 measurements](r36_phase1_metrics.json) retain all arms, companions, and input hashes.

The [independent result review](reviews/r36_phase1_result_review.md), [artifact audit](reviews/r36_phase1_artifact_audit.md), and [closure receipt](reviews/r36_closure_receipt.json) retain their exact bytes and checked scope. They permit bounded reporting, not global representation-quality, biological-change, or forecasting claims.

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
