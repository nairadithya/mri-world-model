# Documentation review disposition

The earlier content-consolidation revision received APPROVED_FOR_OPERATOR_REVIEW. Its exact draft bytes are preserved in [the writing baseline](writing_revision/baseline.json).

The later [plain-English rewrite](writing_revision/README.md) changes the draft wording and organization. The earlier independent verdict does not cover the revised hashes. The revised wording has parent editorial review only. [Writing checks](writing_revision/checks.json) record its integrity checks.

Those reviews preceded documentation promotion. The operator subsequently approved the wording and requested a commit and archival of the originals. The promoted documents are now under `info/`, with the original pages in the dated archive. [The checkpoint](checkpoint.md) records that later action. Neither review nor documentation promotion establishes scientific acceptance, a complete independent reread of all sources, or permission for research execution or R36 Phase 2.

## Independent evidence

The [post-integration reviewer](received_reports/post-integration.md) resumed after explicit barrier release. It read all four drafts and all 72 disposition records, and performed a targeted reread of high-risk evidence. The report gives exact independently read source ranges and states the sources and reports not reread in full. Its verdict is APPROVED_FOR_OPERATOR_REVIEW.

The fresh [read-only verifier](received_reports/read-only-verification.md) independently matched all 109 handoff input hashes, reran all 19 regression tests, and called the pure package verifier. Its result is PASS+NOTES. It inspected protection evidence without accessing protected checkouts or outputs.

The [received-report manifest](received_reports/manifest.json) binds both reports to their managed workflow and child IDs and byte hashes. The original BLOCKED review and incomplete comparison reports remain preserved. Workflow completion alone is not approval.

## Finding dispositions

1. Stale contracts-receipt status: fixed. The manifest now states the completed parent disposition and points to the integrated contracts report and decision row. `disposition_at_receipt` retains the original pending wording. The received contracts report itself is unchanged.
2. Frozen versus live branch count: clarified here without changing the reviewed verifier. `verification.json.branch_refs` is the count of frozen catalog refs, not a fresh recount. `protection_check.json.branch_refs_match_capture` records the separate read-only comparison. Capture scope and the date of protection evidence must accompany these counts.
3. Link-parser limitations: retained, not expanded into an unrelated parser rewrite. Current drafts have no fenced link examples or non-Markdown fragment targets that expose those specific blind spots. Regex parsing is not a complete Markdown implementation; external URLs were not fetched. Archived sources retain their original embedded links.
4. Parent assertions versus independent semantic evidence: disclosed. All 72 source dispositions are parent-checked assertions backed by recorded reads or complete variant diffs. The mechanical verifier checks their structure and frozen bytes, not their semantic truth. Independent semantic review was targeted, and the reviewer explicitly states its actual read scope.

The verifier additionally notes that supplementary hashes prove integrity of listed artifacts, not completeness or current runtime status. Identifier checking proves baseline identifier presence, not correct interpretation of every identifier across every source. These are retained limits, not scientific-result gates.

## Stability and protection

At closure of the earlier consolidation review, only audit metadata changed. Scientific prose, source snapshots, coverage decisions, supplementary bytes, verification code, and tests were unchanged at that point. The later writing revision changes prose and moves reference indexes. Its baseline and checks are recorded separately. The original [review handoff](review_handoff.json) remains immutable evidence of the reviewed inputs. [Final checks](final_checks.json) record the final verification and identify any post-review administrative input changes.

The protected-checkout comparison is limited to observed HEADs, Git statuses, baseline tracked bytes, and captured Markdown. It does not certify shared outputs or process state, and differences cannot be attributed to this task solely from that comparison. Historical R36 snapshots must not be quoted as current job status.

No canonical promotion, source deletion, commit, push, merge, or research job occurred during the preparation and writing stages described above. The later operator request separately authorizes the documentation commit and archive.
