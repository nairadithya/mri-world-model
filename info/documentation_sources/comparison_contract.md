# Cross-branch documentation comparison contract

The operator wants every Markdown source checked across all existing branches and included in the next four-document research draft. This is an additive preparation for later operator review. Canonical promotion, source deletion, research execution, commits, and publishing are outside this run.

The only writable checkout is `/home/adithya/projects/world-model-docs`, branch `docs/portable-research-records`, starting at main `389b8ab`. Only the parent writes it. You are read-only. Return your comparison through the managed output artifact. Do not edit project files, use write through bash, stage, commit, switch branches, change configs, touch processes, launch jobs, or delegate.

The CPU worktree `/home/adithya/projects/world-model-r36` and shared artifact trees `/home/adithya/projects/world-model/outputs/` are protected. Do not traverse them. All branch and live Markdown evidence needed for your task is frozen in `info/documentation_sources/versions/`. No datasets, private patient inventories, checkpoint bytes, or experiment outputs are needed. Consult the frozen catalog rather than live sources.

Read `catalog.json`, `slices.json`, and the four actual drafts in `info/drafts/`. Your assignment is one named slice in slices.json. Every source ID in that slice needs a separate comparison result. Identical bytes with different branch origins are already deduplicated. Alternative bytes of the same path remain separate IDs. Read complete primary documents with pagination; inspect every alternative version or its complete diff against an already-read version. Do not claim a version was checked merely because a filename or identifier appeared.

Compare each section with the four drafts. Preserve valid conclusions, withdrawals, invalid or incomplete outcomes, method limitations, D/A/I/R/G/K3/V/E identifiers, task changes, and reproducibility details. Historical plans must stay historical. Newer source status does not automatically establish a quality verdict. A40 withdraws the learned forecasting headlines. SAILOR remains developmental transfer. Scientific result acceptance is separate from documentation review.

Return one JSON object with these fields, without fences:

- `slice`: your assigned slice.
- `coverage`: an array containing one object per assigned ID with `id`, `disposition`, `destinations`, `evidence`, and `read_mode`. Disposition must explain inclusion, supersession, or a retained operational role. Destinations are draft filenames. Evidence is an array of draft section names and frozen source line references. Read mode states full read or complete diff against an identified source.
- `missing_material`: an array of concrete omissions or contradictions, each with source IDs and lines, draft filename and section, priority, proposed insertion or exact old/new replacement, and why it matters. Do not generate stylistic rewrites or fill absent scientific results with guesses.
- `branch_differences`: an array of materially different source versions, their origins, and how those differences should be represented.
- `unread`: any unread IDs, with reasons. Completeness requires this to be empty.
- `verdict`: COMPLETE, NEEDS_INTEGRATION, or BLOCKED.

The parent coverage ledger adds `checked_by` and `comparison_status` to each per-source result. `comparison_status` is `checked`, `unread`, or `missing_material`. The verifier rejects the latter two states and requires `read_mode`. Every destination set must match the catalog or carry a `destination_override` rationale. Identifier preservation covers every frozen version of all four draft baselines across the combined draft package. An identifier can move between drafts, but cannot disappear from the package. These mechanical checks do not prove that a comparison or disposition is correct.

All citations use actual catalog IDs and snapshot paths. The parent will replace draft links deterministically with frozen source paths after comparisons. Suggested prose must use plain sentences, no long-dash characters, and no unsupported numerical claims. Contact the supervisor only if a real decision or source-access failure prevents completion.
