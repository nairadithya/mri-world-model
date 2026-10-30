## Review

**Scope:** Read-only review of current v5 runner, focused tests, analysis helpers, relevant artifact producers, and amended protocol. No commands, private-data access, network, edits, or assessments performed.

Parent reports focused tests passing and real preflight PASS with digest `36db2dd08b54668eb6990a2267142da11c550d2d93e7989674d41ed34ebf1271`: 599 prepared visits, 638 original visits, 423 eligible rows, split into 262 source / 60 validation / 101 unseen rows, with 18 history-eligible assessment patients. These execution receipts were not independently reproduced.

### Correct

- **Compartment order fixed:** targets now use authoritative `COMPARTMENTS`, matching the frozen decoder (`scripts/run_spatial_noise_ablation.py:13,218`; `src/data/representation.py:17`). The mocked-run fixture checks asymmetric target coordinates and source changes (`tests/test_spatial_noise_ablation.py:100–136`).
- **Relational provenance fixed:** prepared/inventory binding, source-data hashes, checkpoint/cache agreement, normalization, and original source/validation membership are checked (`scripts/run_spatial_noise_ablation.py:109–133`). The provenance-mismatch fixture exercises rejection.
- **Helper and row binding fixed:** the contract binds runner, imported helpers, representation definitions, input hashes, effective configuration, compartment order, and ordered eligible-row identities/history membership (`scripts/run_spatial_noise_ablation.py:153–165`).
- **No future-target-gap eligibility:** actual row construction derives historical gaps solely from week components; suffixes only order visits. Same-week targets remain eligible, while a retained zero historical gap excludes the subsequent row (`scripts/run_spatial_noise_ablation.py:24–87`). The four-visit regression covers that distinction (`tests/test_spatial_noise_ablation.py:32–45`). The contract explicitly discloses approximate week-bin timing rather than exact elapsed days.
- **Replay preservation fixed:** private outputs retain identities, decoded cutoffs, truths, source changes and patient weights, validation-grid scores, assessment-grid scores, and both bootstrap distributions. Single-visit identities, inputs, truths, and common-setting scores are retained, including safe empty arrays (`scripts/run_spatial_noise_ablation.py:238–250`). Together these arrays reconstruct calibrated predictions and scores without another encoder pass.
- **Both paired comparisons fixed:** common `k=1` and independently validation-selected settings receive patient-paired estimates and 95% bootstrap intervals (`scripts/run_spatial_noise_ablation.py:227–232,251`). Public aggregate contrasts exclude participant-keyed `patient_values`.
- **Scientific scope remains bounded:** identical empirical changes and weights, unchanged source mean during calibration, validation-only multiplier selection, and no assessment winner selection. This is descriptive E-only evidence—not H/L training or a world-model capability result.
- **Budget accounting substantially improved:** v5 imports the v4 ledger’s cumulative CPU totals and original wall start, subtracts recorded expenditure, enforces wall limits, and records ordinary exceptions (`scripts/run_spatial_noise_ablation.py:169–209,253–257`). Preservation of actual v2–v4 expenditure is parent-reported. Unknown v1 preflight expenditure is disclosed honestly; that historical uncertainty alone is not a blocker.

### Finding: P1 — CPU exhaustion bypasses failure accounting and leaves execution restartable

**Location:** `scripts/run_spatial_noise_ablation.py:176–189,207–211,236,253–257`

`_with_limits()` installs a handler only for `SIGALRM`, then sets `RLIMIT_CPU`. Under normal Linux execution, reaching the CPU soft limit delivers **SIGXCPU**, whose default action terminates the process. It does not raise an exception that reaches the failed-ledger handler.

The reachable consequence is:

1. Run startup changes `mode` but leaves ledger status `preflight_complete`.
2. Scoring exhausts the CPU allowance before `target.mkdir()`.
3. SIGXCPU terminates execution without charging that run’s CPU or marking failure.
4. A subsequent launch accepts the unchanged ledger and receives the previously recorded allowance again.

Thus cumulative accounting and failed-run preservation remain incomplete specifically at the CPU ceiling—the boundary they must enforce.

**Smallest fix:** Install a SIGXCPU handler that raises a caught timeout and permits failure-ledger persistence. Persist a non-restartable running state before work, with conservative accounting for interrupted executions. Add focused coverage proving CPU-limit failure records expenditure and cannot silently restart. No private-data assessment is necessary for that test.

### Merge verdict: **BLOCK — scoped E-only execution**

The scientific comparison, provenance binding, row construction, and replay corrections are satisfactory for this scope. Fix the CPU-exhaustion ledger path before passing the digest for execution.

This is not scientific-result approval, approval of changed hashes, or authorization for the broader R37 pilot.