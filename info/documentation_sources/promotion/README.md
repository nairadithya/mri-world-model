# Documentation promotion and archive

The operator approved the wording and requested a commit and archival of the old documentation. The four research documents and source coverage now live directly under `info/`.

The [archive manifest](../../archive/2026-10-12/manifest.json) binds 32 original research records and five supporting presentation assets. All 37 files retain their original bytes. Frozen cross-branch snapshots and the saved pre-rewrite drafts also remain unchanged.

Operating guides, CLI verification skills, and research guardrails remain discoverable. Links in active documentation and code-facing comments now point to the consolidated pages or archive. Notebook comment edits were regenerated through Jupytext with `--update`, preserving cell IDs and synchronized source pairs. No training or evaluation logic changed.

The earlier independent verdict applies to its recorded draft hashes. The later rewrite and this migration have operator approval and parent checks, not a new independent scientific review. The R36 runtime snapshot is not refreshed. Phase 2 remains unauthorized.

[Promotion checks](checks.json) record tests, compilation, package verification, local-link checks, notebook synchronization, archive hashes, protected-checkout comparison, and staged-file inspection.

Three unchanged evidence files contain whitespace that the standard Git check flags: Markdown hard breaks in two received reports and a final blank line in one frozen source. Their exact paths and hashes are recorded in the checks. The check for changed authored files excludes only those three evidence files. They are not normalized because their bytes are part of the provenance.

The commit is made on `docs/portable-research-records`. No push or merge is part of the request. No research job, data acquisition, dataset move, checkpoint change, or process control occurred.
