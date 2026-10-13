# Review retry diagnosis

## First failure: workflow serialization

Workflow `1409a27f-8479-4c11-a6cc-328f88c4ac6b` completed the context scout, then failed before launching the reviewer:

`emit.outputPathMapping must be a JSON value; received undefined.`

The script emitted an optional child-result field without normalizing absent data. The authorized retry used `?? null`. The completed scout was reused. This was an orchestration error, not a source-comparison verdict.

## Second failure: reviewer launched before its inputs were ready

Workflow `71534ef6-618c-49d1-8715-14d552d5e18b` and reviewer `cd1c2c5c-6349-4aa4-874c-2580486b6ef0` are FAILED. The reviewer started at 2026-10-13 04:04:06.162 UTC. Its run-level deadline was 04:34:05.525 UTC, with timeout 1,800,000 ms.

The event log records four blocking `contact_supervisor` calls:

| UTC start | UTC end | Activity |
|---|---|---|
| 04:04:11 | 04:14:11 | Wait for stable-input release; request timed out |
| 04:14:16 | 04:15:00 | Wait; parent permits immutable-source preparation only |
| 04:15:11 | 04:25:11 | Request final release; request timed out |
| 04:25:16 | 04:34:05 | Wait; run-level deadline cancels request |

Between the second and third calls, the reviewer read the full archived Phase 1 report, scientific result review, artifact audit, and closure receipt. It did not inspect mutable target documents and did not issue a verdict. Almost the entire deadline was spent waiting for the parent, which was still editing and verifying the documents.

The supervisor tool is exempt from the per-tool timeout, but not the child's run-level deadline. Replies permitting preparation did not release the final-input barrier or reset that deadline. The parent should have completed and frozen the inputs before launching the final reviewer. The timeout is a parent scheduling error, not a reviewer reasoning failure or rejection of the science.

The recovery listing of changed tracked files describes the shared checkout delta. Those are parent-owned edits, not evidence that the read-only reviewer modified them.

## Capability mismatch

The reviewer has read, search, diff, and supervisor tools, but no shell or SHA256 execution tool. The original task required hash verification without assigning executable verification to a capable lane. The reviewer correctly disclosed that limitation. Parent evidence alone must not be presented as independently recomputed hashes.

## Recovery

The user authorized diagnosis and another review attempt. The parent inspected exact-run status and the event/transcript evidence. Status identifies the retained reviewer as resumable. Partial state was captured at `/tmp/world-model-docs-update-failure-cd1c2c5c/`. The current protected-checkout comparison passes for all four recorded HEADs/statuses, and all 116 immutable evidence hashes match.

The next workflow will first execute independent hash and focused-check verification using the read-only verifier. Only after that succeeds will it resume the known reviewer under its stored role, model, and tools. The resume task will explicitly release the stable-input barrier, bind the handoff hash, and provide the verifier's actual output reference. It will not ask the reviewer to wait while editing continues.

No external CLI, foreground fallback, new scientific execution, commit, push, or merge is part of this recovery. HEAD/status checks do not certify ignored research outputs or process state. A completed workflow is not itself a documentation verdict.
