# P2.3 lesion-local architecture gate

The frozen representation experiments did not test a compartment-specific
observation encoder. They pooled the whole tumor on BRAINIAC's whole-volume
6×6×6 patch grid, and the matched forecasting ablation PCA-compressed global
features. A null result there cannot exclude local enhancing, necrotic, edema,
or peritumoral appearance signal.

The executable audit `python scripts/harness.py anatomy coverage` measured all
868 mask-bearing visits. Enhancing disease has fewer than two effective tokens
in 40.4% of LUMIERE and 42.4% of SAILOR visits. Necrotic/nonenhancing disease
has fewer than two in 31.7% and 64.7%, respectively. Median effective support
is only 2.53/2.32 tokens for enhancing and 2.62/1.65 for necrotic disease
(LUMIERE/SAILOR). Edema is better resolved (median 6.44/6.19), and the
one-token adjacent ring has median 78/84 patches.

The audit is descriptive; its threshold was not preregistered before looking at
these support statistics. We now freeze the operational rule for subsequent
comparisons: more than 25% of visits below two effective tokens for either core
compartment in either cohort forces a crop. It is exceeded in both cohorts, so
whole-volume patch pooling is rejected as the primary lesion representation.
P2.3 moves to a 64-voxel lesion-centred crop resized to the 96-voxel BRAINIAC
input. The model will retain separate compartment, adjacent-ring, and
compartment-minus-ring tokens by modality. Whole-volume/global features remain
controls. This decision changes spatial sampling, not the frozen endpoint,
patient folds, or persistence-centred evaluation.
