# R35 independent scientific result review

**Verdict: BLOCKED — independent artifact verification is incomplete.**

The saved results support reporting a **descriptive, in-domain automated-volume proxy improvement from the tested dense-supervised spatial-state pipeline**. They do not establish architecture-only superiority, general encoder quality, incremental benefit from a dedicated paired network, successful transfer, nuisance separation, forecasting, or biological validity.

**The R35 RESULT REVIEW thread cannot close as independently accepted on the present verification receipt.** This is an evidence-completion block, not a conclusion that the results are incorrect.

## Independently reread scope

Review target: completed **R35**, not R36. Repository identity supplied by the handoff: `integration/r36-report-closure`, HEAD `b6b20c8d8254788a8e6888bd795eda68ccb61867`.

I independently read:

- `/tmp/r35-closure-review/handoff.json`.
- Full `AGENTS.md` and `info/research_guardrails.md`.
- All 173 lines of `info/archive/2026-10-13/info/r35_representation_comparison.md`.
- Full archived `r35_representation_results.html`, as HTML source—not rendered pixels.
- Canonical R35 sections: `info/EXPERIMENTS.md:327–379` and `info/METHODS.md:177–191`.
- **Full actual** `outputs/representation-comparison/r35-final/summary.json`, paginated continuously through its end, including every direct arm, both source-readout budgets, every nuisance condition, and all state audits. Private participant-level sections were inspected but are not reproduced here.
- Actual `r35-final/closure.json`, `retry-complete/summary.json`, and `retry-complete/verification.json`.
- Saved LUMIERE nested-probe metrics for all twelve corrected arms, plus frozen-reference metrics and relevant comparison intervals.
- Full `scripts/report_representation_run.py`, `scripts/audit_representation_predictions.py`, `src/harness/analysis/representation.py`, `src/model/representation.py`, and `src/data/representation.py`; relevant normalization source, renderer, and report tests.
- The completed independent verifier artifact at the supplied output reference, from run `01a106ca-2f1c-4165-8866-e489053e6531`.

I performed **no commands, executable hashing, jobs, fitting, image/model inference, network access, edits, or delegation**. Numerical values below were read from saved artifacts; they were not independently recomputed by me.

## Scientific assessment

### 1. The in-domain state-proxy finding is justified descriptively

On 149 visits from 26 encoder-unseen development patients:

- Spatial-state direct MAE: **1.114490**.
- Fresh frozen scalar-source52 readout: **1.499949**.
- Difference: **−0.385459**, descriptive patient 95% interval **[−0.518427, −0.248109]**.
- Source-trained mean: **1.705059**; spatial minus mean **−0.590569 [−0.785399, −0.398129]**.

The older comparison’s rounded **1.5000** and **−0.3855** agree substantively with the fresh final summary. The pooled spatial-state MAE is **1.318016**, not 1.114490; these aggregation choices must remain distinct.

Evidence: final summary `LUMIERE_direct`; `info/EXPERIMENTS.md:341–347`; report aggregation implementation, `scripts/report_representation_run.py:38–72`.

The defensible claim concerns the **whole tested supervised pipeline and its automated compartment targets**. The spatial candidate receives dense masks during training, while the frozen scalar comparator does not. Pretraining histories also differ (`info/METHODS.md:185–191`). Consequently:

- The contrast does **not** identify architecture as the causal source of improvement.
- Direct-decoder accuracy does **not** establish encoder-wide or multi-capability superiority.
- Dense training diagnostics are not independent biological validation.
- The seed-average intervals are descriptive, not retrospective satisfaction of a newly prespecified gate.

### 2. Dedicated paired modeling has not established incremental value

All values below are means of patient-level errors across seeds, not ensemble errors.

| Cohort / arm | State | Independent decoded-state difference | Pair head |
|---|---:|---:|---:|
| LUMIERE state | 1.1145 | 1.2589 | — |
| LUMIERE frozen spatial encoder | 1.1145 | 1.2589 | 1.2654 |
| LUMIERE later-only | 1.0130 | 1.2001 | 1.4381 |
| LUMIERE paired | 0.9922 | 1.2348 | 1.2696 |
| SAILOR state | 2.2401 | 0.9665 | — |
| SAILOR frozen spatial encoder | 2.2401 | 0.9665 | 1.0159 |
| SAILOR later-only | 2.0255 | 0.9629 | 1.1954 |
| SAILOR paired | 2.3156 | 0.9965 | 1.0508 |

The “frozen” candidate is the frozen **spatial** encoder, not the separate BRAINIAC reference.

**LUMIERE:** 117 observed pairs from 23 patients; zero-change MAE **1.544612**.

- Paired versus zero: **−0.274972 [−0.495299, −0.095657]**.
- Paired versus later-only: **−0.168471 [−0.343809, −0.015474]**; family-12 interval **[−0.425773, +0.049483]**.
- Paired versus independent state: **+0.010751 [−0.079122, +0.106620]**.
- Paired versus its own encoder’s decoded-state difference: **+0.034818 [−0.015639, +0.083483]**.
- Paired versus frozen-spatial pair head: **+0.004260 [−0.079815, +0.088112]**.

**SAILOR:** 242 observed pairs from 27 development subjects; zero-change MAE **0.943004**.

- Independent state differences: **0.966484**; versus zero **+0.023480 [−0.036820, +0.105219]**.
- Paired versus zero: **+0.107766 [ +0.023841, +0.215180]**; family-12 interval crosses zero.
- Paired versus independent state: **+0.084286**, family-12 interval **[+0.029480, +0.144016]**.
- Paired versus its own decoded-state difference: **+0.054226**, family-12 interval **[+0.037620, +0.072704]**.
- Paired versus later-only: **−0.144612**, ordinary 95% interval **[−0.254617, −0.033050]**, but family-12 interval **[−0.300108, +0.022352]**.

Thus, the dedicated paired head’s advantage over a weak later-only control cannot override its lack of advantage over independent state differences—and its transfer disadvantage against that control.

### 3. Nested probes do not rescue a paired-network claim

I inspected every candidate arm’s saved nested-probe metrics. Across seeds:

| LUMIERE arm | State ridge range | Feature-difference range | Learned-pair range | Independent state-readout difference range |
|---|---:|---:|---:|---:|
| State | 1.1785–1.2075 | 1.1679–1.2942 | — | 1.2780–1.4288 |
| Frozen spatial | 1.1785–1.2075 | 1.1679–1.2942 | 1.2001–1.3950 | 1.2780–1.4288 |
| Later-only | 1.0865–1.1623 | 1.1538–1.1808 | 1.4519–1.5265 | 1.3046–1.4018 |
| Paired | 1.0138–1.1535 | 1.1554–1.2273 | 1.2151–1.3006 | 1.2561–1.4553 |

Concatenation, later-only, train-mean-change, and zero controls were also inspected. In the paired arm, learned-pair error exceeds simple feature-difference error in **every seed**.

The frozen BRAINIAC reference’s nested state score is **1.600834** versus fold-trained mean **1.835922**. Its feature-difference score is **1.527149** versus zero **1.544612**, with an interval crossing zero.

These nested readouts fit on folds of the 26 encoder-unseen patients. They are **not** the source52 direct-head protocol and must not be substituted selectively.

### 4. Both fixed-source transfer budgets remain negative or unresolved

All feature views and controls in both budgets were inspected. Relevant SAILOR patient-uniform MAEs:

| Budget / arm | State | Difference | Concatenation | Later-only | Learned pair | Independent state-readout difference |
|---|---:|---:|---:|---:|---:|---:|
| 26 / state | 1.9715 | 0.9536 | 0.9967 | 1.2599 | — | 0.9262 |
| 26 / frozen spatial | 1.9715 | 0.9536 | 0.9967 | 1.2599 | 0.9933 | 0.9262 |
| 26 / later-only | 1.9777 | 0.9465 | 1.0314 | 1.3542 | 1.4166 | 0.9289 |
| 26 / paired | 1.9845 | 0.9422 | 1.0018 | 1.2327 | 1.0192 | 0.9446 |
| 26 / BRAINIAC reference | 4.7075 | 0.9791 | 1.0731 | 1.2817 | — | 1.1403 |
| 52 / state | 1.9525 | 0.9804 | 1.0448 | 1.4483 | — | 0.9490 |
| 52 / frozen spatial | 1.9525 | 0.9804 | 1.0448 | 1.4483 | 1.0143 | 0.9490 |
| 52 / later-only | 1.9971 | 0.9663 | 1.0401 | 1.6163 | 1.5100 | 0.9713 |
| 52 / paired | 2.1164 | 0.9911 | 0.9996 | 1.4550 | 1.0544 | 0.9730 |
| 52 / BRAINIAC reference | 4.6876 | 0.9801 | 1.0275 | 1.4142 | — | 1.1951 |

Matched state means are **2.239059** for source26 and **2.110533** for source52. State-arm contrasts against these means are respectively:

- **−0.267598 [−0.618959, +0.076044]**.
- **−0.158051 [−0.496610, +0.175920]**.

Neither resolves superiority. Direct state also fails to establish superiority over the source mean: **2.240098 versus 2.110533**.

Zero change is **0.943004** in both budgets. Train-mean change is **0.947107** and **0.949948**, respectively. The paired source26 difference’s tiny point advantage over zero is **−0.000848 [−0.044173, +0.052217]**, not a transfer success.

Acquisition differences and automated-label-source differences remain confounded. These results cannot isolate the mechanism of transfer failure.

### 5. Nuisance and state-specificity limitations remain material

All five conditions were read for all four candidate arms and BRAINIAC reference:

| Arm | Identity | Intensity | Noise | Bias | Resolution |
|---|---:|---:|---:|---:|---:|
| State | 0.0445 | 0.0445 | 0.0433 | 0.0845 | 0.3275 |
| Frozen spatial | 0.1828 | 0.1828 | 0.1830 | 0.2013 | 0.4082 |
| Later-only | 0.6282 | 0.6282 | 0.6277 | 0.6331 | 0.6898 |
| Paired | 0.1504 | 0.1504 | 0.1495 | 0.1711 | 0.4192 |
| BRAINIAC reference | 0.0589 | 0.0589 | 0.0670 | 0.0901 | 0.3570 |

These are LUMIERE held-out-ridge false-change scores, not direct-head transfer scores.

For the state arm, resolution excess is **+0.283079 [0.198591, 0.374659]**, with state-readout drift **0.622798**. Resolution excess is positive for every candidate arm. Separation from acquisition perturbations is not established; blur may remove anatomical information as well.

Affine intensity is substantially canceled by explicit foreground standardization (`src/model/representation.py:107–135`), so its stability is not learned robustness.

The 100-trial within-patient shuffle is **post hoc**. All seed-specific original-minus-shuffle intervals are below zero for every arm in both cohorts. Spatial-state mean error rises **1.1145→1.5162** in LUMIERE and **2.2401→2.3361** in SAILOR. This demonstrates visit-specific information, not biological attribution.

Empty-target failures remain large. LUMIERE spatial-state seed-average MAEs are **7.0523** for necrotic/non-enhancing, **5.7072** for enhancing, and **9.5619** for edema. SAILOR state-arm seed-specific empty-target errors span approximately **9.79–10.77**, **8.24–9.81**, and **11.56–12.45**, respectively. These failures are not removed by the favorable aggregate state result.

## Protocol and verification assessment

The source implements appropriate separation of contracts:

- MAE averages compartments within rows, rows within patients, then patients; bootstrap units are patients (`src/harness/analysis/representation.py:43–68`).
- Ridge scaling and regularizer selection use training-only patient-grouped folds (`:71–105`).
- State normalization uses source52; internal source13 validation selects minimum-MAE checkpoints (`:459–540`, `:651–710`). This is **validation-selected best**, not R36 final20.
- Pair arms receive another 20 epochs after state initialization. Fitting patients, validation-selection patients, readout budgets, and pretraining exposure are different quantities.
- Both scans are observed inputs. Later-only explicitly ignores the source image in learned inference (`src/model/representation.py:89–103`).
- Masks are supervision, not learned MRI inference inputs.
- Float32 dense reductions and AMP backoff handling are present; transfer rejects pre-repair dense-loss checkpoints (`src/harness/analysis/representation.py:365–366`, `:427–456`, `:680–681`).

Historical invalid AMP runs are explicitly excluded from model claims in the archived chronology and canonical record (`info/EXPERIMENTS.md:379`). Their exclusion is scientifically necessary. I did **not** independently verify all twelve artifact bindings to that exclusion.

**Executable verification belongs to the verifier, not me.** Its completed report records:

- 245 input hashes checked before and after, zero mismatches.
- Correct branch/HEAD and clean worktree.
- 16 representation tests and four report tests passed.
- One state checkpoint metadata inspection passed.

It explicitly did **not** complete all-checkpoint selection/finiteness checks, prediction identity and truth audits, 24 ridge-bundle/two-reference replay, metric/bootstrap recomputation, or nuisance/audit coverage verification. Historical `verification.json` and `closure.json` assertions do not substitute for this missing fresh independent audit.

## Review

- **Correct:** The central bounded interpretation is supported by the saved aggregate evidence: descriptive in-domain state-proxy improvement; no established dedicated-pair increment; unsuccessful/unresolved developmental transfer; incomplete nuisance separation.
- **Fixed:** None. Review-only.

### Finding P1 — required independent artifact acceptance remains incomplete

**Location:** supplied `independent-artifact-verification.md`, “Evidence read and scope not verified”; historical assertions in `info/EXPERIMENTS.md:379` and `r35-final/closure.json`.

**Evidence:** The verifier returned **BLOCKED**, explicitly withholding the broad saved-artifact replay and recomputation needed for scoped acceptance.

**Smallest correction:** Retain the computational-completion receipt as historical and keep current independent result acceptance open. Obtain the outstanding bounded saved-artifact verification through an authorized path before claiming accepted closure. No retraining, new experiment, or MRI inference is implied or authorized by this review.

### Finding P2 — canonical interval wording mixes ordinary and adjusted summaries

**Location:** `info/EXPERIMENTS.md:341–367`, especially `:363`.

**Evidence:** The canonical section omits the source’s explicit descriptive-interval qualification. It gives the ordinary LUMIERE later-only interval, then says the SAILOR interval crosses zero without naming the switch to family-12. SAILOR’s ordinary interval **[−0.2546, −0.0331]** excludes zero; its family-12 interval **[−0.3001, +0.0224]** crosses zero. LUMIERE’s family-12 interval also crosses zero.

**Smallest correction:** Add a sentence identifying seed-average intervals as descriptive, not new prespecified gates; name both interval types consistently for the later-only contrast. Preserve the unchanged “no incremental advantage over independent state differences” conclusion. Do not rewrite immutable archived receipts.

### Finding P2 — presentation’s “every feature view” statement omits shortcut controls

**Location:** `scripts/render_representation_report.py:40–49`; archived HTML’s fixed-source-readout tables.

**Evidence:** Those tables omit saved `independent_state_difference` and `train_mean_change` controls despite claiming every feature view is reported. These controls matter: source26 state independent differences score **0.9262**, whereas its learned frozen-spatial pair scores **0.9933**.

**Smallest correction:** Include these controls in any current successor presentation, or qualify its completeness statement and provide discoverable aggregate controls. Preserve the immutable historical HTML.

- **Merge verdict: BLOCK.**
- **R35 result-review closure: not accepted yet.** Scientific interpretation has been independently reviewed here; artifact-level acceptance remains incomplete, with the reporting qualifications above outstanding.

No retrospective pretraining-design approval is granted. Historical statements that independent review was unavailable remain valid dated history. No experiment family is selected, no future-forecasting evidence is claimed, and Phase 2 is not authorized.