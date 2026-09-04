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

Conventions:

1. Each decision gets an ID (`D1`, `D2`, …), each ablation an ID (`A1`, …),
   each inference an ID (`I1`, …). Reference IDs, not prose, when linking.
2. Numbers beat adjectives. Record the measurement, the command or script
   that produced it, and the date.
3. A superseded decision is marked as such, never deleted.
