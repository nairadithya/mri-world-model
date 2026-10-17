# R35 closure evidence

The [final result review](../../r35_result_review.md) and [corrected artifact audit](../../r35_artifact_audit.md) support the dated closure in [Experiments](../../../EXPERIMENTS.md#r35). The [closure receipt](../../r35_closure_receipt.json) binds their exact bytes, run IDs, reviewed inputs, and scope.

## Evidence ownership

- `audit_r35.py` and `audit-output.txt` preserve the base executable audit. Its disabled reference check and skipped source-prediction key are historical defects, not passing checks.
- `supplemental_r35.py` and `supplemental_r35.output.txt` supply the actual replacement assertions for exact nuisance membership, declared reference provenance and budgets, and required summary digests.
- `metrics_r35.py`, `contrast_extra.py`, and `correct_independent.py` retain the numerical sources. Each has its own receipt, stdout, stderr, and exit file. The separate-state-arm contrast is in `correct_independent.py.stdout.txt`; same-encoder controls are in `contrast_extra.py.stdout.txt`.
- `history/` preserves the initial blocked review, the audit challenge, and the superseded verifier report unchanged.
- The two review handoffs preserve the released input hashes. The private original handoff is not committed because it contains an artifact inventory.
- `reviewed-report.html` and `controls.png` preserve the aggregate presentation reviewed before approval. They are evidence snapshots, not a second active research page. Their old unavailable-review fields and relative links retain their original context.
- `parent-checks.json` records final repository checks. It is parent evidence, not another independent review.

All copied review and executable evidence retains its original bytes. The scripts preserve the original repository and scratch paths. Reproduction requires the private artifacts and original private handoff, plus restoring the sources in the scratch directory recorded in the execution receipts. They do not train or refit models. Do not substitute these historical scripts for a portable public-data pipeline.

Reference checkpoint equality is declared metadata equality, not a fresh checkpoint-byte hash. The reference feature cache has a current digest but no original review-manifest hash. Earlier MRI-byte provenance, dense behavioral replay, shuffled-array replay, biological target reliability, and untouched validation remain limits. Neither approval nor this evidence directory authorizes Phase 2 or another experiment.
