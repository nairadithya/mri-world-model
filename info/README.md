# Research documentation

The four consolidated documents are the starting point for the research record:

- [Notes and insights](NOTES_AND_INSIGHTS.md) explains the research questions, findings, and limits.
- [Experiments](EXPERIMENTS.md) records measurements, corrections, withdrawals, and incomplete results.
- [Methods](METHODS.md) defines terms, protocols, and historical decision and audit IDs.
- [Research TODO](TODO.md) lists the remaining work at the archived research snapshot.

R36 runtime statements in the consolidated documents describe October 11, 2026, at 18:30:51 UTC unless a later addendum says otherwise. They are not live job status. The [R36 execution record](r36_delta_representation_results.md) preserves later progress. Document approval does not accept scientific results or authorize Phase 2.

## Source history

[Source coverage](SOURCE_COVERAGE.md) maps all 72 captured Markdown versions across 48 paths and ten frozen branch refs. The [source catalog](documentation_sources/catalog.json) records their origins. Research citations use immutable source copies so conflicting branch versions and dirty or untracked records remain traceable.

The [October 12 archive](archive/2026-10-12/README.md) contains 32 original research records and five supporting presentation assets moved without changing their bytes. Their old plans and conclusions are historical. Later corrections and withdrawals take priority when interpreting them.

The [preparation record](documentation_sources/checkpoint.md) records comparisons, reviews, writing changes, and verification limits. A documentation check is not scientific acceptance.

## Operating guides and binding rules

Operating documents remain separate:

- [Project README](../README.md) covers setup and common commands.
- [Agent instructions](../AGENTS.md) define repository development and verification rules.
- [Research guardrails](research_guardrails.md) define the objective and required experiment decision brief.
- [Preprocessing](preprocessing.md) defines the BRAINIAC input pipeline.
- [SAILOR reprocessing](../src/preprocessing/reprocess_sailor.md) preserves that procedure and its limits.
- [Acquisition scripts](../scripts/README.md) document data access tools.
- [CLI verification skill](../.agents/skills/verify-world-model-cli/SKILL.md) defines command discovery and bounded execution checks.

Add new measurements to Experiments, their reasoning to Notes and insights, and protocol changes to Methods. Preserve historical IDs and corrections. Keep the archive and frozen source bytes unchanged.

Regenerate the coverage page with `python3 scripts/render_documentation_coverage.py`. Verify the current four-document package with `python3 scripts/documentation_sources.py verify`.
