## Review

**Scope:** Read-only, post-implementation protocol/recovery review. No jobs, edits, commits, or test commands were run. This is not retroactive pre-training approval or independent result review.

### Correct
- **Patient-clean fitting:** Normalizers use source52 only; representation training and both probe views use the locked training patients. Ridge scaling and alpha selection remain grouped and source-only (`delta_protocol.py:100–117`; `delta_representation.py:260,401`; `delta_evaluation.py:193–214`; `representation.py:71–86`).
- **Matched primary contrast:** B and C differ only in grid versus balanced occupancy loss. Seed-matched initialization, ordered exposures, coefficients, and final20 grading are preserved (`delta_protocol.py:41–57`; `delta_representation.py:39–51,280–344`). Matrix receipts document matched schedules and retain AMP skips.
- **Controls and inference separation:** Change decoding receives signed feature differences; target-only probes receive target features. Independent-state and zero controls are evaluated separately, and A’s untrained change head is excluded (`src/model/delta_representation.py:18–32`; `delta_representation.py:445–452,578–597`). Masks do not enter model inference.
- **Patient bootstrap:** Errors are averaged across three seeds before paired-patient resampling; raw and family-three intervals are reported (`delta_evaluation.py:15–34,58–65`).
- **Recovery:** Completed seeds require bound, finite, final20 probes with matched schedules. Exclusive write-ahead launch intent survives ambiguous push failures and prevents duplicate pushes; quota and sequential-launch guards remain (`scripts/continue_delta_representation_run.py:70–124,149–172`). Relevant behavioral tests cover these paths (`tests/test_delta_continuation.py:43–118`).

### Findings
1. **P1 — Report aggregation does not adequately verify nuisance artifact identity.**  
   `src/harness/analysis/delta_evaluation.py:88–107` checks only the nuisance report’s contract hash before using it for the D-versus-C verdict. It does not check complete status, arm, seed, checkpoint/probe hashes, or expected patient-condition coverage. A report from another arm under the same contract passes this guard. This contradicts the artifact-verification requirement in handoff **Unit 4**.  
   **Smallest correction:** Validate these fields against the expected arm/seed, local bound artifacts, and locked nuisance coverage; reject duplicates, omissions, and mismatches. Add aggregation rejection fixtures. No existing corrupted nuisance result was observed—query results do not yet exist.

2. **P2 — Decisively failed D tradeoffs remain “INCONCLUSIVE.”**  
   `delta_evaluation.py:66–68,103–107` initializes D’s verdict as INCONCLUSIVE and only implements the successful branch. Even an interval wholly exceeding the state/change noninferiority margin cannot produce NOT VERIFIED. The handoff requires rejecting demonstrated adverse tradeoffs (`info/r36_delta_representation_handoff.md:189,197`).  
   **Smallest correction:** Add explicit negative branches for demonstrated nuisance worsening or margin violations; retain INCONCLUSIVE for overlapping intervals or missing evidence. Test those classifications.

### Decision-trail audit
`.audit/r36-delta-objective.tsv:4–13` distinguishes target/numerical verification, computational completion, and missing scientific acceptance. Latest entries explicitly identify this as post-implementation review, with no retroactive approval or Phase2 authorization. **No unsupported scientific acceptance claim found.** Historical test-pass statements were not independently rerun.

### Limitations and remaining gates
- Seeds43/44 readouts and all query-result review remain outstanding.
- Automated targets, anatomical correspondence, repeatability, and biological attribution remain unresolved. SAILOR remains developmental and local-only.
- Affine intensity perturbation is largely removed by the retained per-sequence standardization (`src/model/representation.py:121–132`); it is not a strong independent intensity stress test.
- The initial bounded diff showed unrelated documentation changes only. A later check reported changed HEAD; no subsequent committed delta was reviewed.

**Merge verdict: OK with notes for recovery/query generation.** No evidenced blocker requiring a GPU stop was found. Scientific acceptance remains pending the reporting corrections, completed companion evaluations, and independent result review.