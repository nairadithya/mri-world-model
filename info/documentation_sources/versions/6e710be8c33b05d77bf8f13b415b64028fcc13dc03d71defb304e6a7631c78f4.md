# Research guardrails

## Objective

Build useful longitudinal MRI representations that preserve anatomical and pathological state and meaningful change over time while separating acquisition nuisance. Learn per-scan representations and model temporal change as distinct questions. Evaluate both. Lesion awareness, JEPA, forecast heads, and whole-brain alternatives are hypotheses, not the objective.

The current P2 lesion forecast contract remains valid for that scoped experiment. It does not define representation quality or serve as a universal gate. A failed lesion forecast or latent-error score does not show that all representations fail. Do not reopen unchanged failed sweeps. Do not turn one lesion endpoint or score into a general claim.

Keep patient-clean splits and respect each task's information cutoff. Compare with persistence when it fits the outcome and horizon. State timing limits. SAILOR remains developmental transfer, not final independent validation. Multi-capability evaluation is a planning requirement, not a claim that such an evaluation already exists.

## Before substantial research implementation

Write a brief with the current candidate, a materially different approach, and a simple baseline. Name the capability under test and explain the rationale and assumptions. Specify matched comparisons, costs, independent checks, and falsifiers or stop rules. The operator decides changes to the scientific target, method-family priority, and whether a proxy counts as success. Agents may implement or analyze an approved hypothesis without asking about routine reversible steps. A general request to improve the project does not authorize silently choosing a method family or changing the target.

Do not treat training loss as evaluation. When claiming longitudinal benefit, test whether history adds value beyond the current scan. Check nuisance and shortcut signals. Keep the per-scan representation question separate from temporal modeling and evaluation. Reports distinguish measured evidence, interpretation, and unknowns.

One failure mode illustrates the distinction. A temporal model may reduce lesion forecast error by exploiting current lesion burden, without encoding useful change from history. That result supports the scoped forecast claim only if its comparisons and cutoff are valid; it does not establish longitudinal representation benefit.

Tiny QA and mechanical edits do not need a research portfolio. These guardrails guide research judgment. They do not ban lesion methods or use a linter to infer methodological bias.

## Experiment decision brief

Copy and fill this brief before substantial research implementation.

```text
Question and capability tested:
Current candidate:
Materially different approach:
Simple baseline:
Rationale and assumptions:
Matched comparisons:
Information cutoff and patient-clean split:
Evaluation measures and persistence comparison, if appropriate:
Costs:
Independent checks, including history-vs-current-scan and nuisance checks when claiming longitudinal benefit:
Timing limits and cohort status:
Falsifiers and stop rules:
Expected unknowns:
```

Report what was measured, how it was interpreted, and what remains unknown.