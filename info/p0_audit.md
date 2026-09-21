# P0 evaluation audit

Status: in progress. This records the P0 validity work without changing the
historical result tables.

## Label audit

Run against the local LUMIERE tree after the shared label contract was added.

- 91 patients, 638 visits after 11 imageless visits were dropped.
- Raw LUMIERE ratings: PD 251, SD 95, PR 20, CR 27, Pre-Op 91,
  Post-Op 116, Post-Op/PD 1, blank 36, NaN 1.
- 255 visits are not clean response labels and must not silently become SD.
- The legacy action field now uses -1 for missing/unknown/operative-ambiguous
  ratings; clean evaluation uses `response_labels` and `response_valid`.
  Existing checkpoints remain numerically loadable, but must not be treated as
  trained under this corrected label contract.
- SAILOR: 27 subjects / 270 sessions; codes 1/2/3/5 occur 75/115/27/23
  times, with 30 sessions lacking a RANO code. Treatment phases are
  CRT/TMZ/no/unknown = 103/109/17/41.

The shared contract is implemented in `src/data/labels.py`, with separate
response, treatment, operative-event, and validity fields in both dataset
adapters and collate output.

## Task manifests

`src/harness/data/manifest.py` defines immutable row IDs and explicit input
cutoffs for assessment, forecast, anatomy, and surprise tasks. A provisional
LUMIERE manifest was generated at `info/lumiere_task_manifest.json` from the
existing interface/horizon caches. It is labeled `lumiere_legacy_cache` and
must be regenerated after current-schema caches are encoded.

## Metric repair

`auc_mann_whitney` now uses average ranks and the exact Mann–Whitney definition,
including ties. Tests cover separated, reversed, interleaved, and tied scores.
The old implementation returned 0.75 for scores `[.2,.4,.1,.3]` with labels
`[0,0,1,1]`; the corrected value is 0.25. It returned 1.0 for all ties; the
correct value is 0.5.

The evaluator now preserves score logits and supports AUROC, AUPRC, Brier,
log-loss, and ECE with metric-specific bootstrap CIs. The legacy cached LUMIERE recheck gave persistence-error AUC **0.4574** and
JEPA-error AUC **0.4973**. The survival-free current-schema refresh gives:

- persistence-error AUC for next-visit PD: **0.4611** (393 pairs)
- champion JEPA-error AUC: **0.4971** (393 pairs)
- lead-time JEPA/persistence AUCs: k=1 **0.4971/0.4611**, k=2
  **0.4289/0.5932**, k=3 **0.4907/0.5752**
- SAILOR persistence-error AUC: **0.6600** (240 pairs)
- corrected legacy site-probe AUCs: z **0.9996**, states **0.8518**

These replace the old AUC values for current development purposes. Full
provenance and the refreshed SAILOR transfer table are in `info/p0_refresh.md`.

## Blocked items

- **Independent SAILOR codebook verification:** blocked by lack of an
  independent expert label/reference from the data owner. The current mapping
  remains empirical and codebook sensitivity is not a validated clinical
  analysis.
- **RANO 2.0 timing reconciliation:** blocked by missing/uncertain
  radiotherapy-completion and confirmation metadata aligned to every LUMIERE
  label. Do not silently treat the surgery-window mask as RANO 2.0.
- **Current-schema cache rebuild:** blocked by the expensive full-cohort
  re-encoding/retraining step; existing legacy caches remain explicitly
  labeled and are not suitable for the final prospective claim.
- **New external/hidden holdout:** blocked by access/challenge rules and the
  absence of a locked new cohort assignment. Historical `final` is now
  development-only.
