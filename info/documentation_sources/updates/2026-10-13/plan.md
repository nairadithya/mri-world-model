# Fold the completed R36 record into the documentation

## Session pickup

1. Locate the prior trail from the user's session ID, Pi resume history, `PI_SESSION_FILE`, a `pi-subagents` run or mission ID, or a pushed branch. Read the session metadata and last messages first, then scan the decision points. Do not search unrelated projects' session files. For a long transcript, delegate a bounded summary only when authorized.
2. Reconstruct operational state. The branch and worktree, what already landed (`git log`, `git diff` against the base), the open todos, the decisions made. The prior trail is authoritative input. Resist the bias to re-derive it.
3. Diff done vs pending. Compare what shipped against what was planned, name the resume point, do not re-run the prior repro or redo completed work. A "let me verify from scratch" pass means you're treating the trail as untrustworthy when it's authoritative.
4. Route remaining work to the matching playbook and decide whether to continue, deliver a finished recommendation, ratify or override a conclusion, or postmortem a failed run. The pickup playbook ends here. The routed playbook owns the rest.
5. Verify the inherited claims against the original goal on the real artifact (the **principle-prove-it-works** skill). A passing prior self-report is not the proof.

## Bespoke consolidation workflow

- [x] Read the Principles section of poteto-mode.
- [x] Phase A: Frame. The base is `36a9e28`. The prior session completed reporting review, not a representation-quality success. The original 72-version source capture is already complete and will not be repeated.
- [x] Phase B: Design the workflow. The parent alone writes `world-model-docs-update` on `docs/r36-record-update`. A bounded scout reads the previous session. A fresh reviewer waits for a hash-bound stable-input handoff. No research checkout, output, dataset, process, or existing archive is changed.
- [ ] Phase C: Run the loop.
- [ ] Capture hashes and destinations for the new narratives, restored drafts, and presentations. Compare existing source bytes before duplicating material.
- [ ] Integrate completed outcomes into Experiments, definitions and reporting rules into Methods, implications into Notes, and actionable remaining work into TODO.
- [ ] Archive standalone historical narratives without changing their bytes. Keep review evidence and aggregate machine-readable measurements separate and discoverable.
- [ ] Update active navigation and presentation generators where relocation changes links. Do not change measured values or research logic.
- [ ] Add the documentation-maintenance procedure to AGENTS.md.
- [ ] Run the existing documentation tests and verifier, archive hashes, active-link checks, notebook-pair check, and rendered presentation checks if active HTML changes.
- [ ] Release stable inputs to the independent reviewer, preserve its report, and resolve concrete findings.
- [ ] Phase D: Keep the audit trail. Record decisions in decisions.tsv as work proceeds.
- [ ] Phase E: Verify and hand back. Inspect the final diff and report checks and remaining limits. No push, merge, experiment, or Phase 2 run is part of this task.

## Definition of done

The four canonical documents contain the completed R36 classifications, material controls, protocol and audit limits, and current remaining work. Every relocated file retains its original hash. The 72 frozen source versions and October 12 archive remain unchanged. Active links resolve. Review evidence retains its exact bytes and scope. The original dated runtime snapshots remain historical. AGENTS.md specifies where each kind of documentation belongs and how to verify an update. A fresh reviewer returns an explicit documentation verdict against stable inputs.

## Throughput checkpoint

The uncertain work is semantic comparison, not moving files. The initial inventory contains five standalone research narratives, four restored drafts, two presentations, and two independent review reports. Existing verification scripts and source mappings cover the earlier consolidation. Reuse those tools. Keep one parent writer because the four documents share result and protocol references. Separate read-only session reconstruction and final review from editing. No writer fanout or design competition is needed for this existing document structure.
