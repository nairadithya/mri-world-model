## Review

**Scope:** Read-only inspection of the runner, its tests, both used analysis helpers, the R37 amendment, and relevant source producers. No commands, model runs, private-data inspection, network access, or edits. Parent-reported artifact verification and preflight PASS are context, not independently reproduced execution evidence.

### Correct

- The empirical controls implement the amended law: identical joint source changes and patient weights, calibration around the unchanged source mean, and condition-specific observation noise (`src/harness/analysis/spatial_dynamics_baselines.py:103–148`). Nondegenerate synthetic reference tests cover all calibration values (`tests/test_spatial_dynamics_baselines.py:100–135`).
- History construction prevents bridging unusable visits, resets the first retained gap, and checks retained historical gaps (`src/harness/analysis/spatial_dynamics_protocol.py:87–117`).
- Calibration uses validation only; assessment does not select a condition. The common-`k=1` comparison uses paired, patient-level bootstrap resampling (`scripts/run_spatial_noise_ablation.py:228–243`).
- No encoder forward pass or neural fitting occurs. Selecting the eventual E condition for the full R37 capability decision is **not** required for this descriptive E-only ablation.
- The all-true availability placeholders (`scripts/run_spatial_noise_ablation.py:89–90`) do not affect current E scores or eligibility: the history builder uses usability and historical gaps, and E consumes decoded cutoff volumes. They are not valid covariates for a future H/L/R/G runner, but are not an E-only blocker.

### Findings

#### P1 — Target coordinates disagree with the frozen decoder

**Location:** `scripts/run_spatial_noise_ablation.py:225`

Targets are assembled as:

`(edema_flair, enhancing, necrotic_non_enhancing)`

The saved decoder and normalization use:

`(necrotic_non_enhancing, enhancing, edema_flair)`

This order is defined in `src/data/representation.py:17` and used when producing training targets, normalization, and direct predictions (`src/data/representation.py:145–158`; `src/harness/analysis/representation.py:548–553,657–660`).

Consequently, fitted changes subtract decoded necrotic volume from measured edema volume, and vice versa. This changes the predictive law, not merely output labels.

**Smallest fix:** Use one authoritative compartment order for target assembly and explicitly record it in the contract and saved arrays. Add an asymmetric end-to-end fixture that would fail under this reversal.

#### P1 — Input hashes do not establish the required provenance relationships

**Location:** `scripts/run_spatial_noise_ablation.py:142–148,167–173,179–185`

The loader verifies the checkpoint hash, run-to-prepared hash, normalization equality, cache visit IDs, and inventory-to-anatomy hash. It does **not** check:

- the feature artifact’s recorded checkpoint binding;
- cache/checkpoint provenance agreement;
- checkpoint provenance against run patient membership and prepared/source metadata;
- the prepared inventory binding and source-hash relationships.

These fields exist in the producers: feature artifacts record `checkpoint` and checkpoint provenance (`src/harness/analysis/representation.py:677–680,704–708`); prepared metadata records `inventory_sha256` and `source_hashes` (`src/data/representation.py:173–179`).

The direct-prediction check does not close this gap: its producer saves only unseen-patient predictions (`src/harness/analysis/representation.py:547–550`). Thus it cannot authenticate source/validation embeddings. Likewise, checking only split sizes and disjointness does not prove the run uses the checkpoint’s original membership.

**Smallest fix:** Validate those recorded relationships, including original split membership. Where inventory serialization was intentionally relocated or reformatted, use the established original binding plus canonical-equivalence check—not an unnecessary new raw-data audit. Do not require current encoder source equality merely to decode an already-frozen cache.

#### P1 — The approved digest does not bind executable helper behavior or exact rows

**Location:** `scripts/run_spatial_noise_ablation.py:190–205,214–216`

The contract hashes the runner and input files, but not either imported analysis helper. It also records counts rather than an ordered eligible-row manifest/hash.

A concrete bypass is changing `OBSERVATION_SD` in `spatial_dynamics_baselines.py:12`: execution changes, while `make_contract()` still reports literal `observation_sd: 0.05` and produces the same digest. Changes to history construction can likewise alter row membership without necessarily changing recorded counts.

This contradicts the code/config/row binding required by `info/METHODS.md:418`.

**Smallest fix:** Bind both helper files, the ordered row identities/history membership, compartment order, and actual effective configuration. Add negative tests proving helper/config/row changes invalidate approval.

#### P1 — The runner adds a target-gap eligibility restriction absent from the locked protocol

**Location:** `scripts/run_spatial_noise_ablation.py:97–98`

The shared builder requires positive **historical** gaps, not a positive cutoff-to-target week difference (`src/harness/analysis/spatial_dynamics_protocol.py:59–61,101–107`; `info/METHODS.md:386`). The runner additionally drops same-week targets.

This is reachable even with a prepared pair marked usable: the original pair producer computes gaps from `day`, and the historical parser adds suffixes to that value (`src/harness/data/anatomy_manifest.py:318–337`; `src/data/dataset.py:41–48`).

For example, usable visits `week-000 → week-001 → week-001-1` give the middle cutoff a valid positive historical gap. The shared builder permits its adjacent target, whereas the runner drops it solely because both endpoints occupy the same week bin.

**Smallest fix:** Remove this additional target-gap filter while retaining prepared-pair usability and valid historical gaps. Alternatively, explicitly amend and independently review the eligibility rule before freezing rows. I did not inspect private data, so I make no claim about the number of affected actual rows.

#### P1 — Resource ceilings are declarations, not enforced execution limits

**Location:** `scripts/run_spatial_noise_ablation.py:200,208–216,259`

The contract declares four preflight CPU-hours, sixteen run CPU-hours, and two elapsed days. Neither preflight nor execution checks a deadline or CPU limit. The sole process-time measurement is reported after scoring finishes.

Moreover, METHODS specifies sixteen CPU-hours **including** preliminary checks (`info/METHODS.md:410`), whereas the contract presents a separate sixteen-hour run allowance without accounting for preflight expenditure.

**Smallest fix:** Persist cumulative budget accounting from execution start, subtract preliminary expenditure, and enforce CPU and elapsed-time ceilings through monitored execution or an equivalent enforced launch limit. Preserve failure/timeout expenditure rather than resetting the allowance.

#### P1 — Saved artifacts cannot independently replay the fitted comparison

**Location:** `scripts/run_spatial_noise_ablation.py:246–261`

`private_predictions_scores.npz` contains assessment score matrices, common-setting deltas, and bootstrap draws—not predictions or their identities. It omits:

- ordered row/patient/cutoff/target/history IDs;
- truth and decoded cutoff arrays;
- fitted source changes and their patient weights;
- validation per-grid scores supporting calibration selection.

The aggregate saves selected multipliers, but not their validation evidence. Single-visit results survive only as aggregate scores. Neither contract contains the missing row manifest.

This falls short of the private counts/exclusions and prediction/selection preservation requirements (`info/METHODS.md:408,418`).

**Smallest fix:** Save a private identity manifest and the compact arrays needed to reconstruct both mixture laws, validation selections, and patient-level assessment results. Preserve exclusion reasons. No image copying or further model execution is necessary.

#### P2 — The calibrated comparison lacks an explicit paired estimate and interval

**Location:** `scripts/run_spatial_noise_ablation.py:239–258`

The report preserves all-grid assessment means and each condition’s validation-selected multiplier, so selected means are recoverable. However, its only paired delta and bootstrap interval compare `k=1` against `k=1`; there is no corresponding paired result using each condition’s independently selected multiplier.

**Smallest fix:** Also calculate and label the secondary validation-selected paired contrast and patient-bootstrap interval, without selecting a condition from assessment outcomes. Preserve the common-`k=1` comparison as the noise-isolating primary descriptive comparison.

### Test coverage and review limits

The seven runner tests (`tests/test_spatial_noise_ablation.py:16–67`) exercise helpers and generic hashing, but never invoke `build_rows`, `load_inputs`, `make_contract`, or `run`. They therefore do not establish runner acceptance or detect the blockers above.

After corrections, the supervisor should run the focused synthetic unittest suites for the runner, protocol, and baselines, including new mocked full-run/provenance/budget/replay fixtures. No real-data assessment is needed to test these corrections.

### Merge verdict: **BLOCK — scoped E-only execution**

The amended baseline mathematics is suitable for the bounded comparison, but this runner is not yet acceptable for assessment. Preserve the existing v1 preflight/review receipt unchanged; mint and independently review a corrected **v2 contract** before execution. This review grants no scientific-result approval.