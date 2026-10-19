# R37 initial design-review dispositions

The [initial review](initial-design.md) returned `CHANGES_REQUIRED` and merge verdict `BLOCK`. Both P1 findings are accepted. That report remains unchanged. It reviewed the initial hashes, not this revision.

1. H versus L changed both earlier MRI information and past observation timing. L now receives the same completed gaps, visit masks, and historical sequence-availability flags as H. Only earlier standardized MRI maps become constant zeros. The current map is unchanged. No artificial repeated current scan is introduced. The revised interpretation is MRI-history information conditional on these observation covariates.
2. H versus G changed current-scan processing as well as earlier spatial detail. G now uses H's exact architecture and current-scan spatial pathway. Only earlier maps become spatially pooled and broadcast. The claim is restricted to the value of earlier spatial detail. Anatomical correspondence and physical mechanisms are not identified.

The review also noted that standardization largely cancels the affine-intensity perturbation. The protocol now explicitly forbids treating that condition as learned intensity invariance. The resolution adds no arm, data check, resource allowance, or scientific result.

A fresh review of the revised inputs is required before marking the design-review task complete. Passing documentation checks does not resolve this review gate.
