# R36 Phase 1 report

Status, October 13, 2026. Phase 1 computation and scoped reporting review are complete. The final reporting corrections passed independent review. The executable artifact audit passed with zero failed assertions. The [closure receipt](reviews/r36_closure_receipt.json) binds both completed native runs and their evidence. This approves reporting the bounded findings, not a representation-quality win or progression to Phase 2.

## Question and fixed comparison

Does balancing changed and unchanged occupancy regions improve local access to signed observed change in a frozen MRI encoder? Both scans are observed inputs. This is not a future-prediction experiment.

The four arms share the R35 spatial backbone and seed-matched initialization. A continues state-only training. B adds whole-grid signed-change supervision. C balances changed and unchanged regions. D adds nuisance constraints. All twelve arms completed the fixed 20-epoch recipe. Thirty final-epoch difference and target-only probes were fitted on the same 52 source patients. The original R35 encoder is a reference, not a newly selected competitor.

The primary endpoint is balanced signed-occupancy MAE, averaged over pairs within patients and then over patients. The report averages errors across seeds 42, 43, and 44, not ensemble predictions. Paired patients are the bootstrap units, with 10,000 draws, seed 2026, and the locked family-three adjustment.

LUMIERE assessment includes 149 visits from 26 encoder-unseen development patients and 117 pairs from 23 patients. SAILOR includes 269 visits and 242 pairs from 27 development subjects. No SAILOR fitting or checkpoint selection occurred. Neither cohort is untouched independent validation.

## Fixed hypothesis classifications

Independent result review and executable raw-row recomputation support these protocol classifications.

| Hypothesis | Difference and family-adjusted interval | Classification |
|---|---|---|
| C versus B frozen local accessibility | -0.003067, [-0.007995, +0.002644] | INCONCLUSIVE |
| C versus A direct-state retention | -0.029774, [-0.073244, +0.017074], upper allowance +0.05 | VERIFIED for this noninferiority endpoint |
| D nuisance tradeoff | Trained-perturbation excess improves, but direct-state noninferiority remains unresolved | INCONCLUSIVE |
| Incremental observed temporal information | C and D beat target-only probes, but neither establishes superiority over independent decoded-state differences | INCONCLUSIVE |

C versus B's raw 95% interval also crosses zero. Failure to establish a gain is not proof of equivalence or universal representation failure. C's state-retention classification is not a verified state improvement or retention of every anatomical capability.

## All arms and required companions

The [research page](research_run.html) presents all five encoders on both cohorts. The [aggregate-only metrics](r36_phase1_metrics.json) preserve state, frozen local and nonlinear head errors, target-only and independent-state controls, zero change, changed/unchanged partitions, appearance/disappearance support, AUPRC, empty/nonempty state, both-empty pairs, source-fixed change-size strata, native scalar change, source52 ridge readouts, reversal, nuisance responses, and input artifact hashes. No participant inventories or raw images are included. Companion tables are descriptive, not additional selected hypothesis tests.

The local LUMIERE frozen-probe means for A, B, C, D, and original R35 are 0.376955, 0.381971, 0.378904, 0.381402, and 0.373396. The balanced objectives improve the descriptive nonlinear change-head means relative to B, from 0.483508 to 0.270405 for C and 0.274940 for D. That is a decoder result, not the primary frozen-encoder gain.

On SAILOR, every frozen difference probe has a higher descriptive error than its independent-state control. Every source52 ridge-delta mean exceeds zero observed native-volume change. These negative transfer controls remain part of the report.

### D's unresolved tradeoff includes measured harm

D minus C mean excess false change over identity, averaged across noise, intensity, and bias, is -0.000220, with adjusted interval [-0.000416, -0.000083]. Bias excess falls, but noise excess rises by +0.0000301, with interval [+0.0000182, +0.0000436]. Held-out resolution excess has an interval crossing zero.

D minus C direct-state error rises by +0.047505, with interval [+0.004175, +0.098686]. The interval establishes worsening, but overlaps the locked +0.05 allowance. Thus the intended state-preserving nuisance tradeoff is unresolved, not VERIFIED and not a demonstrated violation of the specific margin.

The affine intensity transform is largely canceled by per-sequence foreground standardization. Near-zero intensity drift is not evidence of learned intensity invariance. Preserve the locked endpoint and name this limitation rather than replacing the stress after examining results.

### Target and empty-compartment limits

The target audit retained all 599 visits and 496 usable pairs without model-selected exclusions. Sparse changed occupancy motivated balancing but did not establish background dilution as the cause of prior results. One native enhancing disagreement disappeared completely after resizing; partial boundary-detail loss remains unquantified.

Empty compartments still have large state errors. Correct affines and native volumes do not establish accurate segmentation or anatomical correspondence. The inspected suspect extracranial contour remains in the primary analysis. Automated differences are not expert-verified biological changes. No repeatability estimate or noise ceiling is available.

## Reporting corrections and reproduction

The nuisance validator binds complete arm/seed identity, actual final checkpoint and difference-probe bytes, corresponding evaluation provenance, prepared and canonical tensor-manifest hashes, exact first-visit patient/condition tuples, finite errors, identity subtraction, and condition-summary consistency. Duplicate, missing, substituted, and mismatched evidence is rejected. The D classifier now returns NOT VERIFIED for demonstrated aggregate nuisance worsening or a demonstrated state/change margin violation, while retaining INCONCLUSIVE for overlapping intervals or missing evidence.

`tests/test_delta_reporting.py` covers success, negative state/change/nuisance tradeoffs, overlapping intervals, missing artifacts, content replacement, and 18 identity/coverage/metric tamper cases. `tests/test_delta_summary.py` checks privacy filtering, aggregate shapes, and the generated page's scientific scope.

```bash
.venv/bin/python -B -m unittest discover -s tests -p 'test_delta*.py' -v
.venv/bin/python scripts/harness.py delta-representation report \
  --results outputs/r36-delta-objective/matched-final \
  --contract outputs/r36-delta-objective/qa/r36_contract.json \
  --out /tmp/r36-report-reproduction.json
.venv/bin/python scripts/summarize_delta_representation.py \
  --results outputs/r36-delta-objective/matched-final \
  --report /tmp/r36-report-reproduction.json \
  --out /tmp/r36-public-metrics.json
```

The saved metrics bind their inputs and generator by SHA256. Original private weights, predictions, tensors, and recovered assessments remain under ignored outputs. A fresh report is a cheap aggregation of existing artifacts, not another training or inference run.

## Review and remaining gates

The interrupted review draft was not accepted as a completed native run. Its two reporting notes prompted stronger manifest/visit binding and regression tests on the integration branch. The resumed native reviewer and verifier retained their original model/tool contracts and completed successfully.

The [result reviewer](reviews/r36_phase1_result_review.md) returned APPROVED_FOR_SCOPED_REPORT and reporting-patch OK. The [artifact audit](reviews/r36_phase1_artifact_audit.md) verified 12 checkpoints, 30 probes, all 15 assessment sets, 1,950 nuisance rows, and 30 saved ridge bundles. Saved-coefficient replay and independent raw-row seed-mean bootstraps reproduced the report. Its [scratch source](reviews/evidence/r36_artifact_audit.py) and [aggregate results](reviews/evidence/r36_artifact_audit_results.json) are preserved as evidence, not a substitute for reading the audit's checked scope. Raw aggregation differs by about 1e-9 through floating-point reduction, without a conclusion change.

The artifact audit binds declared tensor provenance; it does not newly replay every dense occupancy prediction, PR/direction measure, or original MRI tensor. The audit's managed output contained only a completion summary because its full report was written to a misrouted artifact directory. The parent recovered the exact final report from the child-recorded path, without changing its contents, and bound it by hash in the closure receipt.

The raw CLI continues to label its aggregation review-pending because it does not consume review artifacts. The separate closure receipt and this report record completed review. Historical runtime receipts were not rewritten as if their old unavailable-review status were current.

Independent review cannot provide retrospective pre-training approval. Expert target adjudication, segmentation repeatability, reliable biological attribution, and untouched external validation remain unavailable. No forecast result follows from this two-observation assessment. No follow-up family is selected, and Phase 2 remains unauthorized.
