# R37 observation-noise amendment receipt

October 19, 2026. Source is the operator's current-session instruction, "I say we go ahead and ablate this with and without."

The immediately preceding discussion concerned E's added Gaussian observation noise and the lack of any measured with-noise versus without-noise comparison. The instruction authorizes that bounded comparison, not a new neural sweep or a larger resource budget.

`info/METHODS.md` owns the exact amended comparison. `info/TODO.md` owns remaining implementation and launch gates. This receipt is not a second results narrative. No participant data or experimental results are recorded here.

## Disposition

The new code requires explicit `discrete` or `gaussian_observation` selection. Both retain joint three-compartment empirical change rows, patient-uniform weights, and the same source-fitted mean. The Gaussian condition adds independent observation noise with standard deviation 0.05 per log-volume coordinate. Compare common multiplier 1 to isolate noise, and preserve validation-selected and all-grid scores separately.

The amendment has not selected a condition for the original five-contrast capability decision. That choice and multiplicity treatment remain pre-fitting review gates. No assessment-based winner selection is allowed.

## Review limits

The immutable design review at `reviews/final-design.md` remains approval of its original protocol inputs only. The earlier independent partial implementation reviews did not approve a full runnable pilot. The P/E review blocked adopting an unspecified E law. The operator's new instruction authorizes comparing both laws, but it does not establish feasibility, source chronology, geometry, or result quality.

The [independent amendment review](reviews/noise-amendment.md) returned `OK with notes — bounded noise-ablation design and partial baseline code only.` It did not compute input hashes or independently execute tests. It resolved the unspecified-law blocker for comparing both conditions, not for selecting the eventual primary control. Its two P2 notes identified missing calibrated nondegenerate score tests and a missing source-ledger disposition. Subsequent edits added both. The review does not automatically approve those later edits.

Thirty-one focused synthetic tests passed in the parent session after those corrections. Compilation and the documentation tests, source verification, and coverage regeneration also passed. This is numerical and boundary test evidence, not an executed data ablation. No fitting, feature extraction, or assessment ran. Full implementation and launch review remained pending at that implementation checkpoint.

## Later execution status, October 19, 2026

The operator subsequently instructed, "Go ahead and run." The E-only comparison completed once after further runner and timing reviews. The [execution receipt](r37-noise-execution.md) supersedes the earlier unexecuted status for this ablation only. The full pilot remains incomplete. The earlier review verdicts retain their original input scope.
