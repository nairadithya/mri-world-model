# info/

Decision log, ablation records, and pilot notes for the longitudinal-MRI
world-model program. Code is the primary artifact; these docs exist to record
*why* things are the way they are, with the evidence behind each call.

- [`decisions.md`](decisions.md) — every significant decision, its rationale,
  alternatives considered, and status (decided / provisional).
- [`ablations.md`](ablations.md) — ablation/diagnostic experiments: setup,
  numbers, and the inference drawn. Append-only; never rewrite old entries.
- [`pilot.md`](pilot.md) — the CPU pilot protocol, live results, and the
  scale-up verdict.
- [`anatomy_harmonization.md`](anatomy_harmonization.md) — the active P0
  `lesion-state-v1` target, timing, and pair contract.
- [`p1_anatomy_baselines.md`](p1_anatomy_baselines.md) — the closed P1
  prospective floor and negative JEPA result.
- [`p2_execution.md`](p2_execution.md) — the active lesion-aware forecasting
  program, staged gates, and evaluation contract.
- [`stale_code_audit.md`](stale_code_audit.md) — code paths superseded by the
  RANO-free endpoint and what remains reusable.
- [`harness_v2.md`](harness_v2.md) — anatomy-first CLI and historical
  checkpoint compatibility contract.
- [`p0_audit.md`](p0_audit.md) and [`p1_baselines.md`](p1_baselines.md) —
  superseded RANO-era validity/baseline records retained for provenance.
- [`eval_protocol.md`](eval_protocol.md) — the historical locked RANO readout
  protocol; patient assignments remain reusable, but its task and metrics are
  superseded for P2.
- [`directions.md`](directions.md) — research directions and the current
  ASTRA/K3 execution triage.
- [`frontier_lumiere.md`](frontier_lumiere.md) — literature/benchmark frontier
  and assessment-versus-forecasting caveats.

Conventions:

1. Each decision gets an ID (`D1`, `D2`, …), each ablation an ID (`A1`, …),
   each inference an ID (`I1`, …). Reference IDs, not prose, when linking.
2. Numbers beat adjectives. Record the measurement, the command or script
   that produced it, and the date.
3. A superseded decision is marked as such, never deleted.
