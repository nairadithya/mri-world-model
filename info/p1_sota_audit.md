# P1 observed-scan SOTA audit

Status: protocol audit completed at headline level; full reproduction remains
blocked where primary methods/code or challenge rules are unavailable. This
track is **assessment** (the observed follow-up scan is allowed), not the
past-only forecast track.

## References and reproducible status

- **Tikhonov et al.** — [arXiv:2509.06511](https://arxiv.org/abs/2509.06511).
  The reported LUMIERE result is approximately macro-F1 0.50, ROC-AUC 0.81,
and accuracy 0.72 from patient-wise validation using a pretrained image arm,
engineered growth/shrinkage features, radiomics, and CatBoost. The local A35
comparator is only a 22-dimensional volumetry/growth reduction; it is not a
faithful 4,800-feature reproduction. Full pair inclusion, feature filtering,
and tuning details still need primary-method verification.

- **TRACE** — [arXiv:2606.30313](https://arxiv.org/abs/2606.30313). The
reported four-class macro-F1 is approximately 0.477 and PD-vs-rest macro-F1
approximately 0.709. The method uses paired longitudinal inputs, a pretrained
MedicalNet-style encoder, explicit RANO concepts/measurements, metadata, and a
task head. A local concept concatenation (A36) is not a TRACE reproduction:
it lacks the supplied/predicted concept distinction and the paired-image
training protocol.

- **Amato et al.** — [DARE report, p.92](https://online.flippingbook.com/view/701322381/205/)
reports the TRACE comparison value 0.616 +/- 0.048. This is recorded as a
higher reported reference, but the primary method, folds, tuning budget, and
metric averaging have not been independently verified. Do not use it as an
apples-to-apples target yet.

- **Maurya et al.** — [arXiv:2504.18268](https://arxiv.org/abs/2504.18268).
  The reported balanced accuracy is 0.6415 in cross-validation and 0.5118 on
a hidden multicentric test. These are not macro-F1 values. The challenge code,
exact patient aggregation, and hidden-test access are not available locally.

- **BraTS progression evaluation** — challenge rules/code/results and a new
hidden cohort assignment are not available in this workspace. This remains
blocked; historical `final` patients cannot substitute for a new test.

## Local implementation status

Already recorded from A25--A36:

- locked patient-wise evaluation protocol and minority-class metrics;
- observed paired-image CNN experiments, including ROI-cropped from-scratch
  controls;
- reduced volumetry/growth radiomics comparator;
- concept and late-interface probes;
- explicit warning that these are not faithful Tikhonov, TRACE, or Maurya
  reproductions.

Still actionable after the refreshed forecast controls:

1. build a scalar volumetry + nadir + interval tabular baseline;
2. add spatial growth/shrinkage-region radiomics rather than only scalar
   volume features;
3. obtain a behaviorally verified pretrained paired-image ROI encoder;
4. perform out-of-fold late probability fusion on identical assessment rows;
5. report minority recall and train/validation curves under matched budgets.
