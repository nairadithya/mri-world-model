# Proposal: World Models for Longitudinal MRI with Clinical Annotations

**Status:** Draft v0.2 — proposal stage; architecture direction decided (JEPA,
see §4)
**Goal:** Define a research program that learns *world models* over longitudinal
medical imaging, conditioned on and evaluated against attached clinical
annotations.

---

## 1. Motivation

Longitudinal MRI is the backbone of monitoring for chronic and progressive
disease (e.g. glioma), yet current practice largely reduces rich 3D+t image
series to a single scalar read (progression / stable / response). This throws
away the temporal structure that clinicians implicitly reason over: *given the
current scan and the patient's history and treatment, what should the next scan
look like, and is what we observe surprising?*

A **world model** — a learned system that predicts future observations and
rewards from past observations and actions — is a natural fit:

- Imaging is sequential and partially observed.
- Treatments (surgery, chemo, radiation) are *actions* with delayed,
  heterogeneous effects.
- Clinical annotations (RANO ratings, demographics, pathology, labs) are
  *structured context and reward signals* that can ground and supervise the model.

Attaching clinical annotations turns an unsupervised imaging-forecasting problem
into a grounded one: predictions can be checked against expert ratings, and the
model's internal state can be probed for clinically meaningful latent factors.

## 2. Objective

Learn a model `p(o_{t+1} | o_{≤t}, a_{≤t}, c)` where:

- `o_t` = MRI volume (and derived modalities) at visit `t`
- `a_t` = clinical actions / interventions between `t` and `t+1`
- `c` = static + time-varying clinical context (demographics, pathology,
  RANO/expert ratings, labs, treatment regimen)

Such that the model supports:

1. **Forecasting** future scans and trajectories under observed or hypothetical
   treatment.
2. **Surprise / anomaly detection** — flag observations that diverge from the
   patient-specific predicted trajectory (early progression warning).
3. **Latent probing** — recover clinically meaningful factors (tumour burden,
   oedema, treatment effect) in a disentangled, interpretable state.

## 3. Data

Two complementary longitudinal glioma cohorts:

### 3.1 LUMIERE — breadth

Multi-site longitudinal glioma MRI cohort:

- `completeness__*.csv` — per-visit data availability.
- `demographics__*.csv` — demographics + pathology.
- `mri_params__*.csv` — acquisition parameters (critical for the physical-
  acquisition confound path).
- `rano_rating__*.csv` — expert RANO-style progression ratings (reward /
  supervision signal).
- Imaging (`./data/lumiere_sample/Imaging`, raw volumes to be fetched).

**Role:** main training corpus for the JEPA dynamics (patient scale,
acquisition variation for axis B).

### 3.2 SAILOR v1 — depth

27 high-grade-glioma patients, 3–19 timepoints each through the full Stupp
protocol (pre-surgery → post-surgery → CRT → TMZ cycles), median OS 19 months
(Hovden et al., EBRAINS; descriptor: `data-descriptor_a866425efff8.pdf`):

- **Per-timepoint treatment status (CRT / TMZ / none / unknown)** — the `a_t`
  actions of §2, natively categorical.
- **RANO classes per timepoint** — clinical grounding / supervision.
- **Expert + ONCOHabitats tissue masks per timepoint** (enhancing tumour,
  necrosis, oedema, NAWM, brain) — ground truth for latent probing (RQ3) and
  tumour-burden correlation.
- Structural sequences (t1w, t1wc, t2w, t2wflair) + functional /
  physiological (dti, adc, trace, dce, dsc with derived cbv/cbf, t1wll).
- Radiation dose distribution maps from CRT; overall survival; inter-visit
  day counts.
- Versions: DICOM `sourcedata`, NIfTI `rawdata`, BIDS 1.8, and an MNI152
  (ICBM 2009c) `derivatives` version — skull-stripped, 1 mm iso — whose
  pipeline already performs **rigid intra-patient registration** (visits
  aligned to a patient reference before MNI) and **longitudinal intensity
  normalisation (PLHM)**, i.e. the visit-consistency handling stage A needs.
- `missing.tsv` — documented per-session sequence availability; the
  per-sequence encoding + missing-token design absorbs this naturally.

**Role:** rich-annotation training signal and primary held-out evaluation set
(actions, masks, RANO, dose maps). N = 27 forbids training-scale use.

### 3.3 Cross-dataset cautions

- SAILOR MNI-version inter-visit intervals may be inaccurate — derive
  intervals from source/raw exam dates via `raw-mni-link.tsv`.
- SAILOR MNI intensities are scaled to uint8 0–255 — re-normalise to the
  BRAINIAC input contract before the backbone.
- Verify template compatibility (SAILOR: ICBM MNI152 2009c nonlinear
  symmetric; confirm BRAINIAC's registration target).
- Functional sequences (dce/dsc/dti/t1wll/adc/trace) lie outside BRAINIAC's
  structural pretraining — see open decisions (§9). Note Larsson et al. (2020)
  found perfusion changes predictive of progression, so cbv/cbf may warrant a
  separate encoding path.

**Data governance:** raw data stays out of git (`.gitignore`). Provenance,
cohort definitions, and a fetch/derivation script are the committed artifacts.

## 4. Approach

### 4.1 Architecture: multimodal JEPA

The dynamics model is a **JEPA** (Joint-Embedding Predictive Architecture): we
predict the *latent* of the next scan, never its pixels. Per timepoint, a
vision backbone and clinical encoders are fused; a temporal transformer rolls
the sequence forward; a predictor is trained to match an EMA target encoder's
representation of the next scan.

```
CONTEXT (online, gradient-updated)               TARGET (EMA, stop-grad)
──────────────────────────────────               ────────────────────────
V_i = BRAINIAC(MRI_i)          i ≤ n            Z_{n+1} = proj_EMA(backbone_EMA(MRI_{n+1}))
C_i = clinMLP(clinical_i)                          ↑ image only — no clinical input
token_i = concat(LN(V_i), LN(C_i))
h_n = TemporalTransformer(token_1..n)            (time-delta encodings for
Ẑ_{n+1} = predictor(h_n)      ──── loss ────►    Z_{n+1}                  irregular visits)
```

Components:

- **Vision encoder (BRAINIAC, LoRA-finetuned).** 3D ViT-B pretrained with
  SimCLR contrastive SSL on 32k brain MRIs (Tak et al., *Nat. Neurosci.* 2026;
  code/weights: `github.com/AIM-KannLab/BrainIAC`). Key facts that constrain
  our design:
  - **Single-sequence model**: one forward pass per sequence; multi-channel
    stacking is explicitly future work in the paper. → per-sequence latents +
    fusion is the only supported option, and pairs cleanly with LoRA (stem
    untouched).
  - **Input contract**: 96×96×96 voxels @ 1 mm iso; N4 bias correction,
    1 mm resample, rigid MNI registration, HD-BET skull-strip. Our
    preprocessing (stage A) must reproduce this exactly.
  - **Pretraining coverage**: T1W 24.5k, FLAIR 15.4k, T2W 5.4k, T1CE 3.3k —
    T1CE latents are expected to be the weakest, exactly where glioma RANO
    assessment leans (T1CE + FLAIR). A per-sequence latent-quality check is a
    stage-A deliverable.
  - **No temporal modeling** in BRAINIAC (longitudinal scans were treated as
    independent images during pretraining) — the temporal JEPA here is the
    novelty, not a re-implementation.
  - Check the repo's code license before building on it (paper states only its
    own CC BY-NC-ND).
- **Clinical encoders (MLPs).** One per annotation family (demographics,
  pathology, labs, treatment), mapped to a shared clinical latent `C_i`.
  Clinical data is *conditioning only* — it never enters the target branch.
- **Fusion.** LayerNorm each branch, concatenate. (Learnable fusion /
  cross-attention is an upgrade path if the concat bottleneck shows.)
- **Temporal transformer.** Consumes fused tokens with **time-delta encodings**
  (visits are irregularly spaced). Outputs the longitudinal state `h_n`.
- **Target encoder.** EMA twin of (backbone → image projector). Updated by
  EMA of online weights, stop-gradient; **takes the n+1-th MRI only** — no
  clinical input (rationale in §4.2).
- **Predictor.** MLP/small transformer mapping `h_n` → `Ẑ_{n+1}` in the target
  projection space (output dim = `dim(proj)`, not the fused dim).
- **Optional auxiliary head.** `ĉ_{n+1} = aux(h_n)` predicting the next
  *clinical* embedding under a separate, separately-weighted loss — keeps
  clinical forecasting as a signal without polluting the JEPA target.
- **Training objective.** Cosine / L2-on-normalised-embeddings between
  `Ẑ_{n+1}` and `Z_{n+1}` (unnormalised L2 invites scale shrinkage). Predict
  **every timepoint from its prefix**, not only the last — multiplies targets
  ~4–6× per patient on short (2–8 visit) sequences.

### 4.2 Failure modes and design mitigations

| Failure mode | Mechanism | Mitigation |
| --- | --- | --- |
| Representational collapse | Predictor + target conspire to constants; loss → 0 silently | EMA target + stop-grad; monitor per-dim std / effective rank of target embeddings each epoch |
| Clinical shortcut | Autocorrelated annotations extrapolated trivially; model ignores imaging | Clinical data excluded from target; image-only `Z_{n+1}`; optional separate clinical aux head |
| Regression to the mean | Small cohort, hard task → predictor emits population-average latent | Report persistence baseline (last visit's latent as prediction); check per-patient variance of predictions |
| Acquisition confound | Latent deltas reflect scanner/protocol drift, not biology | Condition context on `mri_params`; harmonisation study (axis B) |
| Temporal leakage | n+1 clinical annotations encode treatments applied before n+1 | Only t ≤ n data enters context; actions encoded as occurring in `(t, t+1]` |
| Scale imbalance at fusion | Vision and clinical latents have different norms; one branch dominates | LayerNorm per branch pre-concat; clinical-only baseline to verify `C_i` isn't near-constant |
| Transformer overfits short sequences | 2–8 visits per patient | Prefix-wise targets; GRU / last-visit-MLP baseline must be beaten to justify the transformer |

### 4.3 Program stages

- **A. Anatomy / representation.** Reproduce BRAINIAC's preprocessing contract
  (N4 → 1 mm iso resample → rigid MNI registration → HD-BET) with
  visit-consistency checks. Reuse SAILOR's MNI derivatives (already
  intra-patient registered, PLHM-normalised, skull-stripped, 1 mm iso) as the
  visit-consistent reference pipeline; re-normalise intensities to the
  BRAINIAC contract. Compare against independent per-visit MNI registration
  to quantify visit-to-visit preprocessing jitter. Deliverable:
  per-sequence latent-quality report (T1CE is the at-risk sequence).
- **B. Physical acquisition.** Model / correct scanner- and protocol-level
  confounds in `mri_params` so predictions reflect biology, not acquisition
  drift.
- **C. Dynamics (the JEPA world model).** As specified in §4.1.
- **D. Safety / calibration.** Uncertainty, out-of-distribution behaviour, and
  failure modes before any clinical-facing use.
- **E. Agent (optional).** Treat treatment selection as action; explore
  counterfactual trajectories through the learned dynamics.
- **F. Benchmark.** Fixed evaluation against RANO ratings and a held-out
  forecasting task; a reproducible scorecard.

## 5. Research questions

1. Can a world model forecast next-visit imaging well enough that deviations
   from prediction precede expert RANO progression calls?
2. Does conditioning on clinical actions (`a`) and context (`c`) materially
   improve forecast accuracy vs. an unconditional prior?
3. Are the learned latent dimensions interpretable / aligned with clinical
   factors (tumour burden, oedema, treatment effect)?
4. How badly do acquisition-parameter shifts (B) corrupt forecasts if
   uncorrected, and does explicit correction close the gap?

## 6. Evaluation

- **Forecast metrics:** latent-space error vs. ground-truth next visit
  (cosine / normalised L2), plus decoder-based voxel/region error and tumour-
  region overlap where a decoder is available.
- **Baselines (mandatory):** persistence (last visit's latent as the
  prediction), GRU / last-visit-MLP dynamics, clinical-only forecaster. The
  JEPA model must beat these to justify its complexity.
- **Clinical grounding:** AUROC / lead-time of "surprise" (target-vs-predicted
  latent distance) vs. subsequent RANO progression; correlation of latent
  tumour-burden factor with RANO.
- **Ablations:** unconditional vs. action-conditioned; with/without acquisition
  correction; with/without clinical context; frozen vs. finetuned backbone.
- **Training health:** per-dim std / effective rank of target embeddings
  (collapse monitor); prediction variance across patients (regression-to-mean
  monitor).
- **Splits:** patient-level train/val/test — no patient spans splits.
- **Reproducibility:** fixed seeds, pinned deps, every scorecard tied to a
  commit hash.

## 7. Milestones (proposal → execution)

1. Data fetch + cohort/derivation script; canonical preprocessing.
2. Baseline unconditional next-visit forecaster (establish floor).
3. Action/context-conditioned dynamics model.
4. Acquisition-correction module + ablation.
5. Latent-probing + clinical grounding analysis.
6. Benchmark scorecard + safety review.

## 8. Risks & ethics

- **Confounding by acquisition** (B) can masquerade as progression.
- **Small, imbalanced cohorts** → careful splits, uncertainty reporting.
- **No clinical deployment** without prospective validation; this is a modeling
  research program, not a diagnostic device.
- **Privacy / governance:** data access, pseudonymisation, audit trail.

## 9. Open decisions

- ~~Transition-model family~~ — **decided: JEPA with temporal transformer (§4.1).**
- ~~BRAINIAC backbone: frozen vs. LoRA / partial finetune~~ — **decided: LoRA**
  (adapters on attention/MLP blocks; input stem untouched).
- ~~Multi-sequence visits: per-sequence latents + fusion vs. channel stacking~~
  — **decided: per-sequence latents + fusion.** BRAINIAC is single-sequence
  (one forward pass per sequence); multi-channel input is explicitly future
  work in the paper, and channel stacking would require a new stem that LoRA
  does not cover.
- How to encode "actions": **resolved for SAILOR** (categorical per-timepoint
  treatment status: CRT / TMZ / none / unknown). LUMIERE treatment encoding,
  if used, remains open (free-text regimens).
- SAILOR functional sequences (dce/dsc/dti/t1wll/adc/trace; outside BRAINIAC's
  structural pretraining): separate small encoders vs. auxiliary prediction
  targets vs. defer to a follow-up. Include cbv/cbf if pursued (perfusion
  changes are progression-predictive per Larsson et al. 2020).
- Train/val/test split ratio and cohort-inclusion criteria (patient-level
  independence is fixed; the exact split protocol is not).
