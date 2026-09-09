# Research directions + codebase audit

**Status (2026-09-07):** DONE since writing — R1 (A13: no JEPA lead),
R2 (A14: atlas + R13 design + gap reframe), R3 (split_gate + A8 same-space
correction), R14-core (A15: treatment tie, site-refit flips gate), batch
fixes (D27: accumulation/bucketing/fp16/G2/G3), freezing battery (A16:
statistics + damping + within-phase tie), R11 partial (A17: accum-8 holds,
batch-1 ejects, momentum half dead by opt-strip). NEXT: R11-repair
(C-extended durability + accum-4 dose), R5 mask pooling, R13
transition-weighted retrain (gated on R11-repair).

The full record is `info/results.md` — this file does not repeat it. It contains (1) a code-and-data audit against the actual source (every claim has a `file:line`), and (2) research directions ranked by cost, each with hypothesis → protocol → success bar.

ID convention: **G** = gotcha/audit finding, **R** = research direction. Append-only; reference IDs like the D/A/I logs.

---

## Part 1 — Codebase audit (verified today)

### G1. Overall survival is in the clinical vector — future leakage (medium)
`src/data/dataset.py:149` (`surv = Survival time (weeks)/200`) and `src/data/sailor.py:121` (OS months → weeks) feed **total** survival into every visit token. OS is known only at end of follow-up; no prospective deployment would have it. It does not leak into targets (image-only branch), and the clinical-only probe is weak (~0.17), so numeric damage is small — but a reviewer will spot it in one glance at the code. Fix: drop the column (or ablate with/without once and cite the delta). Cost: one CPU re-encode.

### G2. The EMA target runs in *train* mode during training (low)
`src/model/jepa.py:114` sets `self.eval()` at init, but `model.train()` in the loop recurses into all children and flips the target back — nothing re-evals it. MONAI ViT dropout defaults to 0.0 so the only live effect is LoRA dropout (0.05) noising the targets. Benign today, hazardous the moment anyone adds backbone dropout. Fix: override `train()` in `JEPAWorldModel` to pin `self.target.eval()`. Two lines.

### G3. `'Post-Op '` (trailing space) mislabels 3 visits (trivial)
Verified in the CSV just now: two distinct strings, `'Post-Op'` (121) and `'Post-Op '` (3). `RANO_ACTION_MAP.get(rating, 2)` (`dataset.py:172`) sends the 3 spaced ones to action 2 (SD) instead of 1. Fix: `.strip()` ratings on load (dataset + probe map). Negligible numerically; signals care.

### G4. SAILOR gap/session misalignment after drops (low–medium, verify)
`sailor.py:114-115` builds `deltas = [0] + gaps[:n_kept-1]` — the *first* gaps, not the gaps *at kept positions*. LUMIERE recomputes deltas from kept week numbers (`dataset.py:166-167`, correct); SAILOR does not. If dropped sessions sit mid-sequence, every later gap is wrong. Base-variant coverage is 99.6% so drops are few — count middle-drops once, then index gaps by kept session order. Real bug, probably small effect.

### G5. SAILOR sex hardcoded female (low)
`sailor.py:122`: clinical vector `[0, age, ...]` — sex bit 0 = female for all 27 subjects, 19 of whom are male. The embedding only admits {0,1} (`clinical.py:49` clamps), so there is no UNK slot; use 0.5-style soft input or drop the bit cross-site. Small noise source on transfer.

### G6. `max_visits` is dead config (hygiene)
Present in both yamls, never read anywhere (`grep` confirms). Longest patient has 21 visits and trains whole — fine, but delete or implement.

### G7. Patient-uniform, not pair-uniform, training weight (medium)
`evaluate()` (`trainer.py:19-35`) averages per-batch means; with hero-run batch-size 1, each patient = one gradient step of its own mean-pair loss. A 2-visit static patient steers as hard as a 21-visit dynamic one — yet A8 shows the dynamic patients are where all learning lives. Consider pair-weighted sampling or loss. Same note covers the horizon/dynamics per-batch `w/w.sum()` renormalization (`jepa_model.py:193,238`).

### G8. Resume ignores `ckpt["opt"]` — the D22 suspect is directly testable (medium)
Checkpoints *save* optimizer state (`trainer.py:115-119`); resume loads only `ckpt["model"]` (`run_train.py:124-130`) by documented choice. D22's leading theory (fresh momentum ejects the narrow basin) has therefore never actually been tested against its alternative (load opt state, continue don't restart schedule). One flag, one short leg, kills or confirms the theory.

### G9. No gradient accumulation at batch-size 1 (medium)
All hero legs ran batch 1 (16 GB T4). Accumulating 4–8 steps costs zero memory and tests whether "batch-1 noise" (D22) is the basin-ejector. Pair with G8: {fresh opt, loaded opt} × {batch 1, accum 8} is a 4-cell answer.

### G10. Ridge λ-selection folds are row-random, not patient-wise (low)
`volume_probe.py:41-63`: visits from one patient land on both sides of λ-CV folds. Final eval is patient-clean, so only λ choice is mildly optimistic. Fix when touching the file.

### G11. Probe CV mixes encoder-train patients into probe-test folds (low–medium)
`probe_rano.py:209-232` reshuffles all 91 patients; the encoder saw 65 of them. Hero-split probe test (13 clean patients) avoids this; CV (0.33) does not. True-unseen CV should fold over the 26 non-train patients, or go nested. The 0.33 headline is likely *slightly* optimistic as a representation claim.

### G12. The hero test set is a validation set now (process)
It has scored ~10 configs (fused/vision/delta/states/MLP/aux-re-gate/ horizon-re-gate/head sweeps). The repo already learned "only CV counts" — extend the rule: lock the test set, or freeze a second held-out slice for the next claim that matters.

### G13. Missing cell: test-only JEPA-vs-persistence (cheap)
A8's 2.7× pools 91 patients (65 train). Test JEPA (0.0074) is quoted; test persistence is not. Fill it before any writeup cites the ratio.

### G14. GRU / last-visit-MLP baselines were never trained (mandate gap)
Proposal §6 requires beating them; A6 logged *untrained* floors (0.9993 / 1.0027, i.e. chance). A trained GRU + a no-history last-visit-MLP isolate the value of history for *forecasting* (A9 did it for RANO, not for JEPA error). CPU-cheap. No paper without these.

### G15. Pre/post-op pairs poison the time model (medium, verified pattern)
47 patient-weeks carry ≥2 distinct dates (checked today), concentrated at week-000: pre-op → post-op resection at model-gap ≈ 1 day
(`parse_week_to_days`: suffix = +1 day). The 0–8d bin is the hardest (persistence err 0.0108, "mostly surgery transitions" — results.md). The model is being taught that 1-day gaps contain resections. Treat surgery as an intervention (reset/flag), or at minimum ablate those pairs out and re-score. Directly relevant to the horizon/dynamics programs.

### G16–G19. Untapped columns/files on disk (all verified today)
- **G16. Rationale column** (RANO CSV): CRET 81 / PRET 39 / T2-Progr. 17+ / new-lesion codes / *"Progression probably within the irradiated area"* (6) — fine-grained, free auxiliary targets. Never parsed.
- **G17. `mri_params` (2487 rows, all images): 3 field strengths, 4 manufacturers, 21 models** — and the same visit can mix scanners across
  sequences (CT1 3T Siemens + FLAIR 1.5T Philips observed). Axis B's raw material, completely unused.
- **G18. SAILOR `treatment.txt` per session: CRT 103 / TMZ 109 / no 17 / unknown 41** — the proposal's "natively categorical `a_t`" — and `sailor.py` never reads it. RQ2 (do actions help?) has never been tested with real treatment labels.
- **G19. Perfusion + dose on SAILOR disk, unused**: `rCBF/rCBV.nii.gz`, `DoseMap.nii.gz`, fastsurfer + ONCO/CL masks per session. (Larsson 2020: perfusion predicts progression; Huisman 2026: dose-map conditioning.)

### G20. Stale "Open work" section in results.md
It lists interval-stratified eval + persistence baselines as open; both are since done and reported two sections up. Hygiene when next editing.

---

## Part 2 — Data problems (what the numbers sit on)

1. **PD 64% imbalance shapes the dynamics, not just the probes.** The JEPA loss sees mostly progression transitions; stable physiology is under-represented — consistent with the diagnosed transfer mechanism (champion over-predicts change on stable SAILOR, flat ~0.03 error at all gaps). Transition-type balancing (R13) is the direct fix, never tried.
2. **n=13 test / n=49 probe visits.** CV spread ±0.06 on F1; every single-split headline so far was the lucky tail. Already internalized —
   keep it internalized (G12).
3. **Volume labels are unvalidated auto-masks** (DeepBraTumIA, 599 studies; enhancing-core untested; growth-rate framing open). R² 0.15 ceiling may be label noise, not model failure — SAILOR's expert/CL masks are the cross-check.
4. **SAILOR codebook 3-vs-5 is tentative** (D26) and transfer F1 0.37 rests on it. One swap-and-rescore sensitivity run (R-bundle in Tier 0).
5. **PLHM dampening is the prime unverified suspect** for "SAILOR targets stay near-static even 76 days apart." Testable: compare PLHM'd vs raw intensities' visit-to-visit deltas on a few subjects; or train a tiny site-discriminator on latents (if site is linearly separable in target space, dynamics never had a chance — and you know exactly what to remove).
6. **Registration jitter is an unmeasured noise floor.** Per-visit independent rigid MNI registration turns head-position noise into "change"; the persistence floor (0.004–0.02) *includes* it. Intra-patient registration (visits → baseline → MNI, cf. SAILOR's own pipeline) was
   proposal stage A and was never tried.
7. **Cohort definition drift**: LUMIERE GBM 2008–2013 predates the 2016 IDH-based reclassification; some "GBM" may be IDH-mutant (now astrocytoma). IDH is in the vector — stratify once, check nothing funny.
8. **SAILOR uint8 quantization** (0–255 PLHM) vs LUMIERE float32 — minor, and z-score absorbs scale but not quantization texture.

---

## Part 3 — Reviewer-view methodology gaps (before any paper)

- G14 (trained GRU/MLP baselines), G13 (test-only persistence), G1 (survival leakage — ablate or drop), G11 (CV contamination note), G12 (test reuse statement), G4/G5 (SAILOR adapter fixes), G3 (strip fix).
- Report bootstrap CIs on every headline; pre-register the next eval protocol (horizon bins, probe config, codebook) *before* running it — the repo's split-luck history (0.45→0.33, 0.509→0.328) demands it.
- Contemporaneous surprise AUC ≠ early warning. RQ1 says *precede*; R1 below measures exactly that.

---

## Part 4 — Research directions

### Tier 0 — zero GPU, CPU days (do these first; several are writeup-grade)

- **R1. Lead-time: does surprise *precede* RANO?** The stated RQ1 was never measured — current AUCs are contemporaneous (err t→t+1 vs RANO_{t+1}). Protocol: err at pair (t−k → t−k+1) vs PD newly appearing at t, k=1..3; persistence-error same framing as control. Success: JEPA lead-AUC > persistence lead-AUC at k≥1 (separates "anticipates" from "change-detection"). Cost: hours.
- **R2. Transition-type error atlas.** Score JEPA error by transition class (PD→PD, SD→PD, SD→SD, post-op→*, …) in-domain. Quantifies G15/G7-balance claims with zero training and designs R13's weights empirically.
- **R3. Honest-numbers pass.** G13 + bootstrap CIs on all six scorecard rows + pre-registered protocol doc. Boring, load-bearing.
- **R4. Robustness bundle.** Codebook 3/5 swap (G-data-4), rating strip (G3), SAILOR gap realignment (G4), sex fix (G5), survival-drop ablation (G1): re-run frozen evals, report deltas. Kills five reviewer objections in one pass.
- **R5. Mask-guided spatial pooling.** The global mean dilutes focal change
  (a 2 mL enhancing focus in a whole-brain latent) — plausibly *the* reason
  volume reads at R² 0.15. Protocol: pool BRAINIAC patch tokens inside the
  DeepBraTumIA enhancing mask (599 studies on disk) vs whole-brain; repeat
  readout/forecast probes. Success: readout R² clearly above 0.15.
  Opens the spatial program without any training.
- **R6. Rationale codes as targets.** CRET/PRET (surgical signal),
  T2-Progr. (non-enhancing progression), irradiated-area (pseudo-progression
  flag, n=6 — tiny but precious): probe states → rationale classes.
  Free supervision, zero GPU.
- **R7. Surgery-reset ablation.** Drop (or flag-and-reset) pre/post-op pairs
  (G15), re-score 1-step + horizon tables. If near-horizon error drops and
  the 0–8d anomaly shrinks, surgery-as-intervention is proven necessary —
  and every dynamics run must condition on it.
- **R8. Scanner-stratified errors from mri_params.** Error by field
  strength / manufacturer / model (G17). If 1.5T-vs-3T gaps dominate,
  axis B is quantified before a line of modeling — and it doubles as the
  harmonization benchmark (R17's target metric).
- **R9. Per-sequence (un-averaged) probes.** Which modality carries progression/volume signal? If CT1+FLAIR dominate and T1/T2 add noise, the mean-fusion (D9) is provably lossy → R18 is earned, not speculative.
- **R10. Survival-from-trajectory.** Cox/MLP head: states → overall survival (the legitimate use of OS — as *target*, fixing G1 by construction). Clinically legible downstream; SAILOR OS validates it cross-site.

### Tier 1 — one Kaggle session each (sequenced)

- **R11. Basin-hold test (G8×G9).** {fresh opt, loaded opt} × {batch 1, accum 8} from the champion, 10 epochs. Decides whether LUMIERE training is truly exhausted (D22) or just badly resumed. Do before any other training.
  - **R11-recreate proposal (2026-09-08, unrun).** The loaded-opt half is
    untestable from any existing file (all ferried champions opt-stripped;
    leg-1 Adam moments died with that session). Recreate instead of reusing:
    re-run Run-4-like settings (batch 1, LR 1e-4, 5-ep warmup, `--no-bucket`
    for legacy ordering, pinned commit) ~7–8 epochs, stop at the val
    minimum, keep the checkpoint WITH optimizer state (~2 GB ferry, not
    stripped). From that recreated point run 5-epoch cells of
    {loaded-opt, fresh-opt} × {batch 1, accum 8}; compare drift slopes, not
    endpoints. Replication, not reproduction (bucketing/G2/GPU drift mean
    near-0.008, not 0.0081 — the comparison is within-run, so this is
    fine). If loaded holds where fresh ejects, D22's momentum suspect is
    confirmed and all future resumes must carry opt state. Value-gated:
    the operational question (can training continue?) is already answered
    YES via accum-8; this buys the mechanism only — run it iff further legs
    depend on the answer.
- **R12. Mandated baselines, trained.** GRU + last-visit-MLP to convergence (G14). The only result that can retire the transformer's justification question either way.
- **R13. Transition-balanced JEPA.** Reweight/resample pairs by transition class (weights from R2) — the direct attack on the diagnosed transfer mechanism (over-predicts change). Re-gate: in-domain persistence margin + SAILOR gap table. If SAILOR error un-flattens, the mechanism is confirmed.
- **R14. Real action conditioning (RQ2, finally).** Wire `treatment.txt` → phase embedding (G18); SAILOR-fit or LUMIERE→SAILOR transfer with treatment-phase as input. Control for time-confounding (treatment correlates with visit index — shuffle-phase control). The proposal's central conditioning question, untested for the whole program.
- **R15. Run the --dynamics leg.** Implemented, never executed (latest commit). Watch `velocity_norm` (v≈0 collapse = learned persistence) and the SAILOR flat-error signature. Euler-3-step memory on T4 is the ops risk — chunk pairs if needed.
- **R16. Site-adapter rescue.** Freeze champion, fit only LayerNorm/FiLM or LoRA-scale site parameters on SAILOR (N=27 forbids more), re-score the gap table. Localizes transfer failure above/below the head with finality: if adapters close it, the failure was normalization-scale all along.
- **R17. Acquisition conditioning/harmonization.** FiLM on the mri_params fingerprint (G17) or latent ComBat; benchmark = R8's stratified gaps close *and* persistence margin holds. Axis B, the proposal's untouched stage.
- **R18. Stop averaging modalities.** Concat (4×768) or cross-attention fusion; gate on R9's evidence + volume readout. D19's additive path and this are the two fusion candidates — test both, keep one.
- **R19. Sharpen the predictor.** Cosine-mean invites regression-to-mean (the horizon docstring says so). Candidates: InfoNCE over in-batch futures (needs batch > 1 — pairs with R11-accum), VICReg variance term on *predictions*, or a probabilistic head (NLL over a predicted distribution). Add prediction-side monitors (pred std/rank) regardless — currently only targets are watched.

### Tier 2 — program bets (paper-grade, multi-session)

- **R20. BraTS Lighthouse Task 11 submission.** The field just converged on LUMIERE-as-benchmark with a hidden multi-centric test (frontier §2–3). `states_forecast`-MLP is a submission-shaped object; hidden-test numbers end every split-luck debate permanently. Highest external-validation value per unit effort in this file.
- **R21. Visit-consistent registration.** Intra-patient rigid alignment (visits → baseline → MNI) + jitter quantification; re-preprocess, re-train or re-probe. Attacks the noise floor itself (G-data-6). Heavy compute, one-time payoff across all future runs.
- **R22. Continuous-time program.** Neural CDE/ODE over visits with true gaps — the principled parent of --dynamics (R15) and the horizon head. Justified only if R15/R7 show time-modeling is the binding constraint.
- **R23. Prospective simulation.** Walk-forward eval with strictly past data (no OS, no future-derived features): the honest clinical framing of RQ1. Mostly protocol + R1 extended; writeup-grade.
- **R24. Pseudo-progression disambiguation.** MGMT-conditioned dynamics + irradiated-area rationale labels (G16) + post-CRT window focus. The clinically hardest error mode, currently invisible to the model. Needs the R6 labels + R14 machinery.
- **R25. Perfusion second tower.** Small encoder over SAILOR rCBF/rCBV (+ dose map) fused late — structural BRAINIAC stays frozen. Direct lineage to Larsson 2020 / Huisman 2026, and LUMIERE can't do it (no perfusion) so it's SAILOR-native work.
- **R26. Uncertainty + OOD (stage D).** Ensemble/dropout surprise intervals; new-site detection via target-space typicality (pairs with the R-data-5 site-discriminator). Required before any clinical-facing claim.
- **R27. Treatment counterfactuals (stage E).** Reachable only after R14: roll the action-conditioned dynamics under CRT-vs-TMZ-swapped phases. Simulation, not prescription — but the program's namesake payoff.

### Suggested order (next two weeks)

R3 + R4 (honest baseline, days) → R1 + R2 (lead-time + atlas; designs R13)
→ R5 (mask pooling; biggest cheap shot at the volume half) → R8 + R9
(quantify acquisition + modality evidence for R17/R18) → R11 (basin-hold;
decides all further training) → R14 (real actions; the program's open
thesis question) → R20 in parallel (external validation while GPUs run).

UPDATE (2026-09-07): R1/R2/R3/R14-core/A16-battery DONE (see Status).
R11-repair OUTCOME (A17/D32): noise confirmed by dose-response; hold
transient (ejects ep6+); EXPLOIT PROTOCOL = accum-8 short legs (≤6 ep),
best-tracking, stop — found 0.0077 < champion. R13 UNBLOCKED under that
protocol only; batch-1/long legs stay banned. NEXT: R13 short legs, R5
mask pooling, legC2-best re-gate probes.

---

## Fixes applied (2026-09-06, batch-size/OOM thread)

- **Pair-weighted gradient accumulation** (`src/train/trainer.py`,
  `training.accumulation_steps` / `accumulate_pair_weighted` /
  `accumulate_pair_norm`, CLI `--accum-steps`): micro *i* contributes
  `loss_i·(P_i/P_NORM)`, grads divided by `(P_total/P_NORM)` at step time —
  exactly the pooled-batch gradient. EMA, LR schedule, clip and logging
  moved to per-optimizer-step (schedule auto-stretches via optimizer-step
  counting). `accumulation_steps: 1` keeps the legacy code path untouched.
  Validated: accumulated grads match the joint pooled gradient to 2.6e-06
  max rel diff on a P=[2,1,2] synthetic case, while classic 1/K differs at
  0.66 (test sensitive, fix exact). Enables the full R11 matrix at batch-1
  peak memory.
- **Length-bucketed train batching** (`src/data/sampler.py`,
  `training.bucket_batches`, CLI `--no-bucket`): sorts by visit count,
  chunks, shuffles batch order; deterministic per (seed, epoch). Cuts the
  dense-input worst case (1.1 GB at B=4/T=21) by avoiding long+short
  pairings. Validated: exact coverage, `len()` match, tight buckets
  (T_max [4,8,21] on a mixed list), determinism + cross-epoch reshuffle.
  Val/test loaders unchanged (sequential).
- **Opt-in fp16 collate storage** (`data.store_half`, default false):
  halves input footprint; backbone casts per chunk. Validated finite with
  max diff 9.7e-04 vs fp32. Off by default to preserve eval comparability.
- **G2**: `JEPAWorldModel.train()` pins the EMA target to eval mode (LoRA
  dropout no longer noises targets). Validated: `target.training is False`
  under `model.train()`, target outputs bit-deterministic.
- **G3**: RANO ratings stripped on load — the 3 `'Post-Op '` visits now map
  to action 1 (verified end-to-end on the real dataset object).
- Smokes (CPU, random-init, 3 lean patients): legacy path and `--accum-steps
  2` both run end-to-end, finite losses, `best.pt`/`last.pt` written.
- Queued, not done: G1 (dims/checkpoint compat — needs a decision), G4/G5
  (would shift published SAILOR numbers — fold into the R4 bundle).
