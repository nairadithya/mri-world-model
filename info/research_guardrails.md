# Research guardrails

## Objective

Build useful longitudinal MRI representations that preserve anatomical and pathological state and capture meaningful longitudinal dynamics while separating acquisition nuisance. Learn per-scan state and model temporal dynamics as distinct questions, and evaluate both. Lesion awareness, JEPA, forecast heads, and whole-brain methods are hypotheses, not the objective.

## Scientific standards

1. **Patient-clean splits and information cutoff.** Never leak future scans, masks, volumes, embeddings, inter-visit gaps ending at future visits, or future treatment into prediction-time inputs. Teacher observations are training targets only.
2. **Meaningful baselines.** When claiming a longitudinal benefit, compare against matched latest-scan-only and persistence controls on the same cohort and horizon. When claiming a representation improvement, compare against a simpler direct method.
3. **Evaluate anatomical or clinical outputs, not just latent loss.** A falling latent or training loss does not establish representation quality or learned dynamics.
4. **Check acquisition nuisance.** Test sensitivity to resolution or intensity perturbations so scanner differences are not mistaken for disease progression.
5. **Keep evidence labels honest.** Automated segmentation masks are proxy targets, not expert-adjudicated biological truth. Reused LUMIERE splits and SAILOR transfer cohorts are development evidence, not untouched final validation. Do not reopen completed negative or inconclusive sweeps to manufacture a positive result.

## Execution discipline (keep it lean)

- **One branch, direct execution.** Implement an experiment directly on one branch in the main checkout. Do not fan out across multiple worktrees, run competing architecture bakeoffs for routine plumbing, or block on multi-round subagent review ceremonies.
- **Run a fast real-data smoke test early.** Before launching a full multi-seed GPU run on `mosaic-gpu`, run 1–2 real batches end-to-end (data load → forward/backward → checkpoint save/load → evaluation scorer) to catch shape, normalization, or path mismatches immediately.
- **No custom budget-policing wrappers.** Compute and time budgets are planning bounds for scoping experiments. Never build custom process-tree supervisors, wall-clock kill-switches, or hash-bound ledger authorities that can crash a GPU training run mid-flight.
- **Dialogue stays dialogue.** Discussing ideas, literature, or next steps does not require writing a formal decision brief or modifying canonical docs until an experiment is selected.

## Experiment decision brief

Before starting a new training experiment, agree on this short brief with the operator:

```text
Question & capability tested:
Hypothesis & method:
Baselines (latest-scan-only, persistence, simpler comparator):
Information cutoff & split:
Primary metric & nuisance check:
Compute target & approximate runtime (e.g. mosaic-gpu):
Stop rule / what falsifies the hypothesis:
```
