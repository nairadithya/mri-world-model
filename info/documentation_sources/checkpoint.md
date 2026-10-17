# Documentation preparation checkpoint

The operator approved the wording and requested a commit and archival of the old documentation. The consolidated documents are now at `info/TODO.md`, `info/EXPERIMENTS.md`, `info/METHODS.md`, and `info/NOTES_AND_INSIGHTS.md`.

The [archive](../archive/2026-10-12/README.md) preserves 32 original records and five supporting presentation assets byte-for-byte. Its manifest records the old and new paths, hashes, and tracked status. Operating guides and research guardrails remain separate. Only the documentation checkout is changed. No protected research checkout or run output is moved.

## Evidence and history

[Source coverage](../SOURCE_COVERAGE.md) and the [catalog](catalog.json) retain all 72 captured Markdown versions across 48 paths and ten frozen branch refs. The initial incomplete child reports and BLOCKED review remain in [received reports](received_reports/manifest.json).

The later reviewer approved the content-consolidation revision after a targeted semantic reread and review of all disposition records. A fresh verifier matched 109 input hashes and passed 19 tests. Those reviews did not independently reread every source. Their exact scope remains in [the review disposition](review_summary.md).

The operator then requested a plain-English rewrite. [The writing baseline](writing_revision/baseline.json) preserves the reviewed pre-rewrite bytes. [Writing checks](writing_revision/checks.json) record the rewritten draft hashes and set-presence checks. Historical decision and audit indexes moved into Methods. The rewrite received parent editorial review and operator approval, not a new independent scientific or semantic review.

## Promotion and verification

The commit and archival request authorizes documentation promotion. It does not accept scientific results, authorize Phase 2, or refresh the October 11 R36 runtime snapshot.

[Promotion checks](promotion/checks.json) record current package verification, archive integrity, local links, Python compilation, notebook synchronization, protection evidence, and staged-file inspection. Earlier test and review artifacts retain their original paths and hashes as historical evidence. They are not silently relabeled as checks of the promoted files.

The verifier checks frozen source bytes, listed supplementary bytes, archive bytes, coverage-record structure, baseline identifier presence, and local link targets. It does not prove semantic completeness, remote URL validity, current job status, or biological validity.

The protection comparison covers HEADs, Git statuses, baseline tracked-file bytes, and captured Markdown in the three protected checkouts. It does not certify shared outputs or process state. No experiment-output traversal or process control is part of this migration.

## Applied principles

The following principle leaf skills were read during preparation:

- Separate Before Serializing Shared State: the parent alone writes the isolated documentation checkout. Research checkouts and outputs are protected.
- Foundational Thinking: documentation promotion remains separate from research execution and scientific acceptance.
- Model the Domain: source versions, historical IDs, withdrawals, operational exceptions, and archive mappings remain explicit.
- Build the Lever: the reusable verifier now checks canonical document paths and archive hashes.
- Prove It Works: current checks accompany the migration, and earlier review scope is preserved rather than extended to new hashes.

No research job, push, or merge is authorized by this commit request. The commit belongs on `docs/portable-research-records`, not directly on main.
