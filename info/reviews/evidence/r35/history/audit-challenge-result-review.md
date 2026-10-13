# R35 final scientific RESULT review

**Verdict: NEEDS_FIXES.**

The scientific interpretation remains supportable for scoped reporting, and both original P2 reporting defects are corrected. However, inspection of the resumed verifier’s actual audit source reveals gaps behind several claimed checks. Its **PASS_FOR_SCOPED_REPORT cannot yet support accepted closure without a narrow supplemental audit and corrected receipt**.

**The R35 RESULT REVIEW thread remains unaccepted.** No incorrect scientific result or corrupt artifact has been demonstrated.

## Scope and verification attribution

This continues the prior independent R35 review, whose initial report is preserved at `/tmp/r35-closure-review/received/initial-result-review.md`. It is not an R36 review.

The retained prior scope includes full `AGENTS.md`, research guardrails, the 173-line archived comparison, archived HTML, canonical R35 methods/results, the complete paginated actual `r35-final/summary.json`, all arms and both readout budgets, nuisance/state audits, and relevant data/model/analysis/reporting source.

In this continuation I independently reread:

- `final-review-handoff.json`.
- Full resumed verifier reports at both supplied locations.
- Full `audit_r35.py`, `audit-output.txt`, `metrics_r35.py`, `metrics-output.txt`, `contrast_extra.py`, and `correct_independent.py`.
- The supplied three-file patch and actual bounded working-tree diffs.
- Updated canonical R35 results, full renderer and report test.
- Full generated `/tmp/r35-closure-review/render/r35.html`.
- Parent-generated `controls.png`, using the image-aware read tool. Both source-budget tables and added controls are legible in that image.

The handoff declares unchanged HEAD `b6b20c8d8254788a8e6888bd795eda68ccb61867` and three changed input hashes. I cannot execute hashing and **do not independently attest the expected handoff hash**.

I performed no commands, executable verification, fitting, image/model inference, edits, network access, or delegation. Executed checks belong to the verifier or parent. No participant identities or inventories are reproduced here.

## Scientific result retained

### State improvement: justified, narrowly

For LUMIERE’s 149 visits from 26 encoder-unseen development patients:

- Direct spatial-state MAE: **1.114490**.
- Fresh frozen scalar-source52 control: **1.499949**.
- Difference: **−0.385459**, descriptive patient 95% interval **[−0.518427, −0.248109]**.
- Source-trained mean: **1.705059**.

The verifier’s independent NumPy calculation reproduces the matched frozen-control contrast in `metrics-output.txt`. It does not substitute the easier training-mean comparator.

This supports **lower in-domain automated-volume proxy error for the tested dense-supervised spatial-state pipeline**. It does not isolate architecture, because supervision and pretraining differ. It does not establish encoder-wide quality or biological validity.

### Dedicated paired modeling: incremental benefit not established

| Direct observed-change control | LUMIERE | SAILOR |
|---|---:|---:|
| Zero change | 1.5446 | 0.9430 |
| Independent state-arm differences | 1.2589 | 0.9665 |
| Frozen-spatial pair head | 1.2654 | 1.0159 |
| Later-only pair head | 1.4381 | 1.1954 |
| Paired head | 1.2696 | 1.0508 |
| Paired encoder’s independent decoded differences | 1.2348 | 0.9965 |

LUMIERE paired-minus-zero is **−0.2750 [−0.4953, −0.0957]**, but paired-minus-independent-state is **+0.0108 [−0.0791, +0.1066]**.

SAILOR paired-minus-independent-state is **+0.0843**, family-12 interval **[+0.0295, +0.1440]**. Dedicated pairing transfers worse than that simple control.

The corrected canonical account now names both ordinary and family-12 later-only intervals:

- LUMIERE: ordinary **[−0.3438, −0.0155]**; family-12 **[−0.4258, +0.0495]**.
- SAILOR: ordinary **[−0.2546, −0.0331]**; family-12 **[−0.3001, +0.0224]**.

Both adjusted intervals cross zero. Neither contrast overrides failure against independent state differences.

The retained all-arm nested-probe inspection also remains unfavorable to a dedicated comparison network: paired learned features score **1.2151–1.3006**, whereas simple differences from that encoder score **1.1554–1.2273**, with lower difference-probe error in every seed. These nested unseen-cohort readouts are not interchangeable with source52 direct heads.

### Both developmental transfer budgets remain unresolved or unfavorable

| SAILOR state-arm ridge budget | State MAE | Matched mean | State-minus-mean 95% interval |
|---|---:|---:|---|
| Source26 | 1.9715 | 2.2391 | [−0.6190, +0.0760] |
| Source52 | 1.9525 | 2.1105 | [−0.4966, +0.1759] |

Direct spatial state also fails to establish superiority over the mean: **2.2401 versus 2.1105**.

For paired transfer readouts:

- Source26 learned-pair **1.0192** versus independent-state **0.9446**; difference **+0.07455**, family-12 interval **[+0.04818, +0.10441]**.
- Source52 learned-pair **1.0544** versus independent-state **0.9730**; difference **+0.08143**, family-12 interval **[+0.04012, +0.13796]**.

Zero change is **0.9430** in both budgets; training-mean change is **0.9471/0.9499**. The tiny source26 paired feature-difference advantage over zero remains unresolved. No favorable view or budget is promoted after inspection.

### Nuisance and state audits remain cautionary

All arms and five declared perturbations were considered in the retained scientific review. Spatial-state false-change scores are:

- Identity **0.0445**.
- Intensity **0.0445**.
- Noise **0.0433**.
- Bias **0.0845**.
- Resolution **0.3275**.

Resolution excess **+0.283079 [0.198591, 0.374659]** is independently reproduced in the verifier’s saved metric output. State drift is **0.6228**. Resolution degradation may discard useful anatomy; it is not proof of nuisance-only removal. Affine-intensity stability largely follows normalization.

Empty-compartment failures remain substantial: source spatial-state errors **7.0523**, **5.7072**, and **9.5619** for the three reported compartments. Transfer empty-target errors are also large.

Shuffle results remain **post-hoc saved diagnostics**, not independently replayed evidence: state error increases **1.1145→1.5162** on LUMIERE and **2.2401→2.3361** on SAILOR. They suggest visit-specific information, not biological attribution or a prespecified success gate.

## Review and finding dispositions

### Correct

The revised reporting preserves the necessary scientific boundaries:

- Both scans are observed inputs—not future forecasting.
- Source52 fitting and source13 checkpoint selection are distinct budgets.
- R35 uses validation-selected best checkpoints, not R36 final20.
- Invalid historical AMP runs are excluded from graded model claims.
- Dense supervision and pretraining prevent architecture-only attribution.
- SAILOR remains developmental, with acquisition/measurement-source confounding.

### Fixed — original P2 interval wording

**Resolved.** `info/EXPERIMENTS.md:339` labels seed-average intervals descriptive, with patients as bootstrap units. Line **365** supplies ordinary and family-12 contrasts for both cohorts without changing the independent-state conclusion.

### Fixed — original P2 omitted presentation controls

**Resolved.** `scripts/render_representation_report.py:38–53` includes independent-state differences and training-mean change for every arm and both budgets, alongside zero controls.

`tests/test_representation_report.py:45–73` checks headers and distinctive values. The actual successor HTML and supplied rendered image show the new columns, including source26 frozen-spatial **0.9262/0.9471** and source52 **0.9490/0.9499**. Five passing tests are **parent-reported**, not executed by me.

Historical archived HTML and old unavailable-review fields remain historical; they should not be rewritten.

### Finding P1 — nuisance “exact patient coverage” is not established by the supplied audit

**Location:** `/tmp/r35-closure-review/audit_r35.py:105–118`; resumed verifier’s “Executable audit” section.

**Evidence:** Candidate and reference checks require `n_patients == 26` and dictionary length 26. They do not compare keys with the expected encoder-unseen patient set. Subsequent candidate arithmetic aligns dictionaries across seeds, but does not establish membership in the expected cohort. Cardinality is not exact identity.

**Smallest correction:** Check exact expected patient membership for all **65** condition records, and preserve an executed aggregate receipt. Correct the verifier’s claim to the actual checked scope until then.

### Finding P1 — frozen-reference provenance and budget checks are overstated

**Location:** `/tmp/r35-closure-review/audit_r35.py:89–101`, especially **94**; resumed verifier’s “Limits retained” section.

**Evidence:** The reference prepared-provenance assertion explicitly reduces to `True` through `if False else True`. Reference replay checks predictions but does not enforce the source26/source52 training identities, query disjointness, or readout/feature checkpoint and prepared-metadata provenance as the candidate loop does.

Replay agreement alone cannot establish that the reference fit used the declared source budget.

**Smallest correction:** Add and execute the narrow reference identity/provenance checks for both budgets against the recorded contracts. Distinguish legitimate metadata binding from the unavailable creation-time MRI-byte fingerprint; do not manufacture the latter.

### Finding P1 — LUMIERE summary prediction binding is silently skipped

**Location:** `/tmp/r35-closure-review/audit_r35.py:59–61`; retained summary `artifact_hashes`.

**Evidence:** The loop checks an optional `predictions` key. The actual summary key is **`source_predictions`**. Therefore the reported source prediction-to-summary binding is not enforced by this loop. Audit-file digest checks elsewhere do not replace this particular summary binding.

**Smallest correction:** Require `source_predictions` and compare it with each actual LUMIERE direct prediction archive digest for all twelve arms. Avoid silently skipping a required binding when its key is absent.

The supervisor confirmed that no supplemental executed receipt proving these missing checks had yet been supplied. These are **audit/acceptance defects**, not demonstrated numerical-result defects.

## What the resumed verifier does establish

Its source and actual `audit-output.txt` substantiate substantial progress:

- Twelve finite checkpoint/feature checkpoint bindings and repaired dense-loss contracts.
- Complete ordered validation histories with best epoch matching the minimum.
- **240 epochs, 74,592 updates, 48 backoffs**.
- **48 candidate prediction archives** checked against metadata identities and truths.
- Candidate source-budget identities and query disjointness.
- **122 saved readout-view replays**, including reference prediction replay.
- **24 state-audit input digest matches**, with empty-target prediction errors recomputed.
- `FAILURES 0` for the assertions actually implemented.

Independent metric source/output supports the principal ordinary/bootstrap calculations. `correct_independent.py` correctly distinguishes the separately trained state arm from the paired encoder’s own state differences. Its separate executed output was not included among the supplied text outputs; corresponding contrasts remain verifier-attested rather than output-reproduced by this reviewer.

The verifier reports **228 private artifact hashes unchanged**. That executable hashing is attributed to it, not me.

## Closure and remaining limits

- **Merge verdict: BLOCK pending the narrow audit corrections.**
- **R35 RESULT REVIEW: cannot close as accepted yet.**
- Both original P2 report fixes are accepted; the scientific scoped interpretation remains unchanged.
- No broad scientific rerun is warranted by these findings.

Even after audit repair, approval would remain limited by absent creation-time MRI-byte fingerprints, no new dense behavioral replay, unreplayed shuffle values, automated-target reliability, missing expert biological annotations/repeatability, and absence of untouched validation.

This review grants no retrospective pretraining-design approval, selects no new experiment family, and authorizes no Phase 2.