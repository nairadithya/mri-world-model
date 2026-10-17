# Local integration handoff, October 13, 2026

The operator authorized committing and locally integrating the approved documentation package. This handoff records that administrative closure. It is not another scientific assessment or an independently reviewed addition.

## Local repository state

- Documentation commit: `a1a9d1d679729c54a10f5e87bb0d9d2fb4815a31`, on `docs/r36-record-update`.
- Local merge: `4f24f9f73a8de8b05a3dbe226b0e48af548c1379`, on `integration/r36-report-closure`.
- Before this handoff was added, the merge tree exactly matched the documentation commit.
- Local `main` was not moved. Nothing was pushed, published, or submitted to a forge.
- The earlier documentation, R35, and R36 sibling checkouts retained their exact recorded HEADs and Git statuses. Their dirty files were not cleaned up.

The four canonical pages now contain the completed findings rather than an active October 11 launch checklist. Start with [TODO](../../../TODO.md), [Experiments](../../../EXPERIMENTS.md), [Methods](../../../METHODS.md), and [Notes and insights](../../../NOTES_AND_INSIGHTS.md). Follow [AGENTS.md](../../../../AGENTS.md#maintain-the-research-documentation) for later maintenance.

## Closed computation and reporting

**R35 computation is closed.** All runnable training, transfer, nuisance, and control experiments are complete. Better in-domain state estimation does not establish forecasting, biological change, or a paired-model advantage over independent state differences. The [canonical R35 results](../../../EXPERIMENTS.md#r35) and [archived original](../../../archive/2026-10-13/info/r35_representation_comparison.md) retain those limits. This documentation closure supplies no new separate R35 scientific result review.

**R36 Phase 1 is closed for computation and scoped reporting.** Twelve training arms, thirty source-only probes, all required assessments, both reporting corrections, independent result review, and the executable artifact audit are complete. The [closure receipt](../../../reviews/r36_closure_receipt.json) and [canonical findings](../../../EXPERIMENTS.md#r36-completed) bind the accepted reporting scope.

The primary C-versus-B encoder gain, D's intended nuisance tradeoff, and superiority over independent state differences remain inconclusive. C passes only the specified direct-volume retention margin. Descriptive nonlinear-decoder gains are not a frozen-encoder win. Do not repeat completed jobs because an archived checkbox or the raw aggregation CLI still says review-pending.

## Verification for this integration

Before committing, the parent reran the update checker, pure documentation verifier, 22 documentation tests, three delta-summary tests, all 17 notebook-pair checks, and compilation of the five touched ordinary Python files. All passed. No training smoke test, inference, artifact-scoring rerun, data acquisition, or research-process control was performed.

After the merge, the parent verified all 44 approved input hashes, three preserved review-artifact hashes, and 116 immutable evidence hashes. The canonical verifier passed for 72 frozen source versions, 392 links, and 47 archive records. The independent approval and its limits remain in [completion.json](completion.json), [independent-review.md](independent-review.md), and [independent-verification.json](independent-verification.json).

Byte-exact review evidence and captured `.diff` files were excluded from the staged whitespace check. Their Markdown line breaks and literal patch context spaces were preserved, not normalized. All other staged paths passed. No private dataset, weights, tensor, prediction, secret, or bytecode path was staged.

The original update checker compares protected checkout HEADs and statuses with its pre-integration baseline. The authorized commit and merge deliberately change those identities. Do not rewrite the historical baseline to conceal that change or treat its later HEAD-mismatch report as a scientific regression. Post-merge checks verified immutable bytes and unchanged sibling checkouts separately.

## Still open, without execution authorization

Expert target reliability, segmentation repeatability, reliable biological attribution, and untouched independent validation remain unresolved. The next scientific question and method family are undecided. R36 Phase 2 and other proposed experiments remain unauthorized.

No experiment was reopened, and no public push was performed as part of this closure.
