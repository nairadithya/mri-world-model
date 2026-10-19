# R37 protocol-first update, October 19, 2026

## Source and disposition

The source is the operator's current-session request for one bounded spatial-state dynamics pilot. It selects the capability of next-observed-visit prediction using cutoff-available MRI history. It authorizes frozen encoders and small predictors, not a backbone sweep or the historical R36 Phase 2 encoder-objective recipe. The immediate deliverable is the protocol before fitting.

The [one-page decision brief and locked specification](../../../METHODS.md#r37) own R37's hypothesis, comparisons, input cutoff, patient split, probabilistic endpoint, fitting budget, margins, and stopping rule. [TODO](../../../TODO.md#r37-protocol-first-dynamics-pilot-october-19-2026) owns authorization and remaining gates. [Notes and insights](../../../NOTES_AND_INSIGHTS.md#r37-asks-whether-spatial-history-predicts-an-unseen-observation) owns the distinction between future prediction, current-state prediction, and R36's observed-change assessment. The [research index](../../../README.md) links the new pilot.

No measurement is added to Experiments because no model fitting, data check, or predictive assessment ran in this protocol task. Existing R35/R36 findings, corrections, reviews, archives, and source snapshots remain unchanged. No standalone result narrative is created, no duplicate is relocated, and the original 72-version catalog is not extended or overwritten.

## Grounding and review boundary

The parent compared the conditional proposal and completed R35/R36 protocol sections in Methods, the saved R36 handoff's opening brief, the current TODO, and the R35/R36 closure references. Direct code inspection checked `SpatialState.encode`, `volume_head`, and the seed-42 checkpoint hash in `delta_protocol.py`. It did not inspect checkpoint bytes or participant inventories, rerun historical experiments, or establish availability of compute or private training inputs.

The [initial independent design review](reviews/initial-design.md) returned `CHANGES_REQUIRED` and merge verdict `BLOCK`. Both P1 findings were accepted. The [dispositions](reviews/dispositions.md) preserve the corrections. L now matches past observation covariates while withholding earlier MRI maps. G now retains the current spatial pathway while pooling only earlier maps. Neither correction adds an arm or expands the budget.

The [fresh bounded re-review](reviews/final-design.md) returns `APPROVED_FOR_PILOT_DESIGN` and merge verdict `OK — protocol design only.` It checked the two corrections and their implications, not a new full review of every settled choice. Together, the initial review and bounded re-review cover the protocol. [The review receipt](review-receipt.json) binds exact report bytes and reviewed hashes. Hash computation and documentation checks were parent-executed, not independently executed by the reviewer. The subsequent TODO checkbox closure is parent-authored administration, not independently reviewed text.

Design approval does not approve implementation, certify data or geometry, establish computational feasibility, or approve results. No R37 training, data QA, or assessment occurred. Implementation review, cutoff rejection fixtures, a content-bound run contract, bounded preflight checks, and aggregate-result review remain open.

[Final checks](final-checks.json) record the documentation verification commands and their outcomes. These establish documentation integrity only. No Python, notebook, plot, active HTML, archive, or existing review bytes changed, so compilation, notebook synchronization, and visual rendering are outside this update's scope.
