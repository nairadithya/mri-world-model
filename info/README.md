# Research documentation

Start with the project [README](../README.md) for setup and common commands. The [research guardrails](research_guardrails.md) state the objective and the decision brief required before substantial research implementation. The documents below have different jobs. Use the active contract and result records for current claims. Treat dated experiment logs as evidence, not as current guidance.

## Current research

- [`p2_execution.md`](p2_execution.md) defines the active RANO-free next-observed-visit anatomy task, its gates, and the latest results recorded on `main`.
- [`anatomy_harmonization.md`](anatomy_harmonization.md) defines the `lesion-state-v1` measurements, eligible rows, and timing limits.
- [`p1_anatomy_baselines.md`](p1_anatomy_baselines.md) records the closed prospective baseline comparison.
- [`external_supervision_audit.md`](external_supervision_audit.md) records cohort access and data-use constraints for future work.

A later cutoff audit and clean reruns are recorded on the local `exp/clean-data-rescues` branch in `info/clean_rescues.md` at commit `067ed6c`. That branch withdraws the learned lesion forecast claims reported in [`p2_execution.md`](p2_execution.md) and [`p2_seed_sensitivity.md`](p2_seed_sensitivity.md). It is not part of `main`.

## Research record

- [`decisions.md`](decisions.md) is the append-only decision history. Cite its D-IDs.
- [`ablations.md`](ablations.md) is the append-only experiment and diagnostic ledger. Cite its A-IDs.
- [`pilot.md`](pilot.md) records the initial training runs.
- [`results.md`](results.md) is a historical narrative. Its older conclusions are superseded by the dated P0, P1, and P2 records above.

## Historical protocols and audits

These documents preserve work on the retired RANO-classification program. They are not the active anatomy task contract.

- [`eval_protocol.md`](eval_protocol.md) and [`p0_audit.md`](p0_audit.md) record the former RANO evaluation protocol and validity audit.
- [`p0_refresh.md`](p0_refresh.md), [`p1_baselines.md`](p1_baselines.md), and [`p1_sota_audit.md`](p1_sota_audit.md) record the former RANO-era refresh and comparisons.
- [`directions.md`](directions.md), [`frontier_lumiere.md`](frontier_lumiere.md), [`ASTRA_PROBE.md`](../ASTRA_PROBE.md), and [`K3_PROBE.md`](../K3_PROBE.md) preserve literature reviews and adversarial research audits.
- [`stale_code_audit.md`](stale_code_audit.md) records which code paths belong to the retired task.
- [`harness_v2.md`](harness_v2.md) records the historical CLI migration. Use `uv run python scripts/harness.py --help` for current commands.

## Operational guides

- [`preprocessing.md`](preprocessing.md) documents the BRAINIAC preprocessing contract.
- [`../src/preprocessing/reprocess_sailor.md`](../src/preprocessing/reprocess_sailor.md) documents the SAILOR reprocessing procedure.
- [`../scripts/README.md`](../scripts/README.md) documents data acquisition scripts.

The shortened map does not replace experiment records. Decisions and ablations remain append-only so older claims and their corrections stay traceable.
