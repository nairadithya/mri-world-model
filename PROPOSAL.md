# Proposal: World Models for Longitudinal MRI with Clinical Annotations

**Status:** Draft v0.1 — proposal stage
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

Primary: **LUMIERE** longitudinal glioma MRI cohort, which ships the annotations
this proposal is built around:

- `completeness__*.csv` — per-visit data availability.
- `demographics__*.csv` — demographics + pathology.
- `mri_params__*.csv` — acquisition parameters (critical for the physical-
  acquisition confound path).
- `rano_rating__*.csv` — expert RANO-style progression ratings (reward /
  supervision signal).
- Imaging (`./data/lumiere_sample/Imaging`, raw volumes to be fetched).

**Data governance:** raw data stays out of git (`.gitignore`). Provenance,
cohort definitions, and a fetch/derivation script are the committed artifacts.

## 4. Approach

A staged program (mirrors the earlier investigation axes):

- **A. Anatomy / representation.** Build a robust longitudinal-aware encoder
  that aligns scans across visits (registration, resampling, modality
  normalisation) and produces a compact latent per visit.
- **B. Physical acquisition.** Model / correct for scanner- and protocol-level
  confounds in `mri_params` so predictions reflect biology, not acquisition
  drift.
- **C. Dynamics (the world model).** Recurrent / latent-ODE / diffusion
  transition model over the visit latents, conditioned on actions `a` and
  context `c`.
- **D. Safety / calibration.** Uncertainty, out-of-distribution behaviour, and
  failure modes before any clinical-facing use.
- **E. Agent (optional).** Treat treatment selection as action; explore
  counterfactual trajectories.
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

- **Forecast metrics:** voxel/region error, structural similarity, tumour-
  region overlap vs. ground-truth next visit.
- **Clinical grounding:** AUROC / lead-time of "surprise" vs. subsequent RANO
  progression; correlation of latent tumour-burden factor with RANO.
- **Ablations:** unconditional vs. action-conditioned; with/without acquisition
  correction; with/without clinical context.
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

- Transition-model family (latent ODE vs. diffusion vs. autoregressive).
- How to encode "actions" from free-text treatment regimens.
- Train/val/test split strategy that respects patient-level independence.
