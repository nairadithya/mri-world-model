# R37 E noise-ablation execution receipt

October 19, 2026. Source is the operator's current-session instruction, "Go ahead and run." It follows the authorized with-noise versus without-noise amendment. This receipt records execution and provenance, not a second result narrative. [Experiments](../../../EXPERIMENTS.md#r37-noise-ablation) owns measurements and [Methods](../../../METHODS.md#r37) owns the comparison.

## Scope and execution

The E-only comparison ran once against content-bound contract digest `1d0fc791c9340487670f951a88b5362095c9feb32e559a2e0347244667e0ba86`. The private bundle is `outputs/r37-noise-ablation/v6/`. No encoder forward pass or neural training ran. Per-example arrays and participant inventories remain private and ignored.

```bash
OMP_NUM_THREADS=1 .venv/bin/python -B scripts/run_spatial_noise_ablation.py \
  --run --contract-digest 1d0fc791c9340487670f951a88b5362095c9feb32e559a2e0347244667e0ba86
```

The command exited zero and the ledger records `completed`. The [public aggregate copy](../../../r37_noise_ablation_metrics.json) contains no participant identifiers. The runner verifies source/cache/checkpoint relationships, exact split membership, the original 638-row curated chronology, positive week-bin historical gaps, target adjacency, and pregrid equality. It excludes invalid historical windows without bridging unusable visits. Same-week targets are not excluded merely for having a zero binned target gap.

## Corrections and review boundaries

The first artifact search incorrectly reported missing directories and failed to find the saved original checkpoint and predictions. Later exact inspection corrected that finding. The pooled feature cache covers all 599 prepared visits and binds to the original seed-42 checkpoint. Earlier direct-prediction files cover only development patients; source and validation decoding was separately replayed from cached embeddings and the frozen affine head.

The initial runner was an unconditional blocked placeholder. Subsequent pre-execution review found reversed compartment coordinates, incomplete relational provenance, missing helper and row bindings, an unsupported target-gap exclusion, unenforced resource limits, incomplete private replay arrays, and a missing calibrated paired interval. All were corrected before assessment. A later CPU-limit review found that SIGXCPU could bypass failure accounting. The final runner handles that signal and persists a non-restartable running state before work. No flawed assessment was run to discover or fix these defects.

Preflight contracts v1 through v5 are superseded and preserved privately. The executed v6 contract is not a retry of an assessment. Earlier preflight revisions are not current approvals. The initial v1 receipt lacks resource accounting; its expenditure is unavailable. Later ledger totals carry forward measured preflight expenditure and elapsed time. Remaining R37 work must not reset these resource ceilings.

## Independent reference checks

The independent reviewer authored two read-only replay scripts. The reviewer lacked an execution tool, so the parent executed those scripts unchanged and returned their receipts. This is independently authored computation executed on the parent host, not a claim that the reviewer personally executed commands.

- [Score, eligibility, provenance, and bootstrap reference](reviews/noise-result-replay.py).
- [Score-reference receipt](reviews/noise-result-replay.json).
- [Frozen affine decoder reference](reviews/noise-decoder-replay.py).
- [Decoder-reference receipt](reviews/noise-decoder-replay.json).

Both commands exited zero. The score reference imports no project helpers. It reconstructs row eligibility and independently recomputes patient weights, source changes, all validation and assessment grid scores, both selected multipliers, single-visit scores, and both paired bootstrap intervals. The decoder reference recomputes every saved source, validation, primary, and single-visit cutoff prediction using the frozen head. Neither replay changes scientific inputs or retrains a model.

The [final independent result review](reviews/noise-result-final.md) returned `APPROVED_FOR_SCOPED_ABLATION_REPORTING`, with merge verdict `OK`, limited to the completed E ablation. No issues were found. It explicitly distinguishes reviewer-authored references from parent-host execution and records the reviewed input hashes. The approval covers descriptive reporting of these two conditions, not full R37 completion, capability success, selection of a future primary E control, or newly edited documentation bytes. Review of execution code alone was not scientific-result approval. Automated masks, reused development patients, approximate timing, limited marginal scoring, and unknown v1 expenditure remain reporting limits. No full R37 history benefit or general world-model result is established.
