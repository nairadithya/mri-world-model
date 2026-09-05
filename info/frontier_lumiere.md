# LUMIERE research frontier — survey (2026-09-05)

Notation: **ON LUMIERE** = trained/evaluated on LUMIERE; **NOT LUMIERE** =
related glioma work on other data. (Survey via research subagent; verify
primary sources before citing in a paper.)

## 0. The dataset

Suter et al., "The LUMIERE dataset: Longitudinal Glioblastoma MRI with expert
RANO evaluation," *Scientific Data* 9:768 (2022). DOI
**10.1038/s41597-022-01881-7** — https://www.nature.com/articles/s41597-022-01881-7
91 GBM patients (Bern, resected 2008–2013, Stupp chemoradiation), 638 study
dates, 2487 images (T1 pre/post-contrast, T2, FLAIR). Expert per-visit RANO
labels + rationale + bi-dimensional (Macdonald) measurements; age/sex/OS;
MGMT and IDH1 subsets. Data: https://doi.org/10.6084/m9.figshare.c.5904905.v1.
Usable RANO counts (after dropping pre/post-op, <3-month): **253 PD / 97 SD /
20 PR / 27 CR** (PD-dominated). Code: https://github.com/ysuter/gbm-data-longitudinal

## 1. Closest reference: TaDiff-Net

Liu Q. et al., "Treatment-Aware Diffusion Probabilistic Model for Longitudinal
MRI Generation and Diffuse Glioma Growth Prediction," *IEEE TMI* 44(6):2449–2462
(2025). DOI **10.1109/TMI.2025.3533038**; preprint arXiv:2309.05406
(https://arxiv.org/abs/2309.05406); code https://github.com/samleoqh/TaDiff-Net.
- Method: conditional DDPM + segmentation head; conditions on prior MRIs +
  treatment (CRT vs TMZ) + treatment-day; generates future MRI at arbitrary
  future timepoint AND tumour masks, with ensemble uncertainty.
- LUMIERE is its **external test set only** (37 HGG pts / 132 exams). Training
  is a private Oslo 27-pt cohort (= SAILOR, §5). States LUMIERE has no expert
  masks → on LUMIERE reports **only image-generation metrics**.
- Numbers (local Oslo test): SSIM **0.919±0.03**, PSNR **27.9±1.2**; future-tumour
  Dice **0.719** w/ treatment vs **0.556** w/o (+16.3pp). External (LUMIERE):
  −7.1% SSIM, −4.6% PSNR, +6.8% MSE. Worst window SSIM 0.877 (days 221–365).
- Does **not** predict RANO; no representation-quality eval.

## 2. RANO/progression classification ON LUMIERE (visit-pair → {CR,PR,SD,PD})

- **Matoso et al., arXiv:2504.18268 (2025)** — https://arxiv.org/abs/2504.18268 ;
  code https://github.com/anamatoso/RANO-classification ; ISMRM 2025 abstract.
  Self-described first DL 4-class RANO pipeline. 5-fold CV stratified 80/20.
  Best: **DenseNet264 on T1w+T2w+FLAIR pairs (no pretraining), median balanced
  accuracy 51%** (>55% in 2 folds). Pretraining + clinical data *hurt*.
- **Tikhonov et al. (MBZUAI), arXiv:2509.06511 (2025), BraTS-Lighthouse entry** —
  https://arxiv.org/abs/2509.06511. LUMIERE as BraTS Task-11 training data,
  patient-wise stratified 5-fold CV. Fine-tuned ResNet-18 (2D ROI, 4 mods) +
  >4,800 radiomic/engineered features (growth/shrinkage-mask radiomics,
  nadir-relative volumetry, centroid shift) → CatBoost. **Macro F1 0.50±0.08,
  mean ROC-AUC 0.81±0.08, accuracy 0.72±0.05** — best reported 4-class numbers.
  Ablation: volumes-only F1 0.30/AUC 0.59; +growth/shrinkage F1 0.45; ResNet-18
  alone AUC 0.74.
- **TRACE (Basha et al.), arXiv:2606.30313 (2026)** — https://arxiv.org/abs/2606.30313.
  RANO-2.0 concept-bottleneck (3D encoder → measurement concepts → deterministic
  rules). 5-fold patient-wise CV: **4-class macro F1 0.4769; binary
  progression-vs-rest 0.7085.**
- **Amato et al., IEEE BIBM 2025** — https://ieeexplore.ieee.org/document/11356532
  (paywalled; record https://iris.unipa.it/handle/10447/698665). DL + radiomics →
  RANO; no public numbers verifiable — claim-only.
- **RECAP-Net (Kakkar et al.), BraTS-Lighthouse/MICCAI 2025 proceedings** —
  https://link.springer.com/chapter/10.1007/978-3-032-16370-7_23. Spectral pipeline
  for BraTSPRO (public train = LUMIERE, hidden multi-centric test). Numbers not
  publicly extractable.
- Context: **BraTS 2025 formalized LUMIERE as the progression-task training set**
  (Task 11/BraTSPRO; train = LUMIERE + auto-segmentations, hidden test) —
  https://zenodo.org/records/10991975. Expect a wave of LUMIERE RANO papers.

## 3. Best RANO numbers ON LUMIERE (scoreboard)

| Framing | Best | Source |
|---|---|---|
| 4-class, paired visits, patient-wise 5-fold CV, ~91 pts | **macro F1 0.50, AUC 0.81, acc 0.72** | Tikhonov 2025 |
| Same | macro F1 **0.477** | TRACE 2026 |
| Same | median **balanced acc 0.51** | Matoso 2025 |
| Binary progression vs rest | macro F1 **0.71** | TRACE 2026 |
| Surrogate: volume-trend vs expert trend | **81.1%** (HD-GLIO) | Suter 2023 |

Uniform framing: consecutive-visit-pair → 4 classes, patient-wise splits
(BraTS Task 11 labels 0=CR,1=PR,2=SD,3=PD). Nobody reports per-class AUCs or a
held-out multi-centre test yet (BraTS hidden test pending).

## 4. Volumetry / masks / growth forecasting

- **Masks in LUMIERE: automated only, zero manual voxel labels.** Shipped:
  **DeepBraTumIA** (necrosis / enhancement / edema) and **HD-GLIO-AUTO**
  (enhancing + T2/FLAIR abnormality) for 599/638 studies with all 4 sequences,
  native space, + PyRadiomics + CoLIAGe features. Sources: dataset paper +
  https://springernature.figshare.com/articles/dataset/LUMIERE_dataset_-_MRI_data_and_automated_segmentations/21249516
  + https://github.com/ysuter/gbm-data-longitudinal. TaDiff-Net confirms.
- **Growth forecasting ON LUMIERE:** only TaDiff-Net, qualitative/external only.
  No published volumetric-forecasting benchmark (next-visit volume / TTP from
  latents) exists.
- Manual longitudinal GBM masks live elsewhere (Meier 2016 14-pt set;
  BraTS-Reg/progression) — not LUMIERE.

## 5. SAILOR

- "Brain tumour MRI: Serial assessments in longitudinal oncological research
  (SAILOR) (v1)," Hovden et al., Oct 2023, EBRAINS KG —
  https://search.kg.ebrains.eu/instances/cae85bcb-8526-442d-b0d8-a866425efff8.
  **27 HGG patients, Oslo, Stupp** (pre/post-op MRI + 3–19 follow-ups; 257
  timepoints in downstream use). Age 32–68 (median 56), F/M 8/19, median OS
  19 months. Controlled-access (EBRAINS account + data-proxy request). No RANO
  labels advertised (treatment-course + RT dose maps feature downstream).
- Papers: (a) **TaDiff-Net's training cohort is this Oslo 27-pt cohort**
  (same n/protocol, overlapping authors — SAILOR published months after the
  TaDiff preprint); (b) **Huisman et al., arXiv:2603.08385 (2026)** —
  https://arxiv.org/html/2603.08385 — trains on public SAILOR (25/27 after
  dose-map exclusions; 21/2/2 split), conditions on RT dose maps from pre-RT
  MRI only; positions as fixing TaDiff's limits (global treatment vector;
  needs multiple priors). No RANO-probe or SAILOR↔LUMIERE joint study exists.

## 6. NOT LUMIERE (do not confuse)

Nalepa 2023 auto-RANO (closed+open cohorts, ICC 0.7); Chang 2019 AutoRANO;
Gomaa/Moassefi/Li/Jang/Bacchi pseudo-progression classifiers (AUC 0.75–0.95,
closed DTI/DWI cohorts); Kickingereder 2019 *Lancet Oncol.* (Heidelberg,
HD-GLIO lineage). Matoso Table 1 fences these from true RANO-4-class work.

## 7. Gap verdict (see D24)

Unreported: (i) frozen JEPA-latent RANO probe as representation benchmark;
(ii) next-visit latent prediction error as progression signal; (iii) any
LUMIERE→SAILOR / hidden-test generalization of a RANO readout; (iv) volumetric
forecasting from latents vs shipped auto-masks. A JEPA world model + RANO probe
with (i)–(iii) is the first entry in the empty intersection of
treatment-aware generation × RANO classification.
