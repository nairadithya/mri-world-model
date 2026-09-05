# Decision log

Date convention: 2026-09-03/04 (build session). `PROPOSAL.md` decisions are
referenced, not repeated — only session decisions are recorded here in full.

## Architecture (from PROPOSAL.md, prior art)

- **D0 — JEPA over generative forecasting.** Predict next-visit *latent*,
  never pixels. Rationale: pixel forecasting on 3D MRI wastes capacity on
  irrelevant detail; latent prediction targets semantic change (§4.1).
  Status: decided, unchanged.

## Training framework

- **D1 — Pure PyTorch, no Lightning.** BRAINIAC uses Lightning, but our loop
  needs custom EMA updates, per-prefix losses, and collapse monitors that
  Lightning callbacks would obscure. Pure torch keeps every mechanism
  visible in `src/train/trainer.py` (~120 lines). Revisit if multi-node
  training demands Lightning/Accelerate scaffolding.

## Clinical conditioning

- **D2 — RANO ratings as categorical actions for LUMIERE.** Mapping
  (`src/data/dataset.py::RANO_ACTION_MAP`): Pre-Op 0, Post-Op 1, SD 2,
  PD 3, CR 4, PR 5; `Post-Op/PD` → 3, missing → 2 (stable). Rationale: the
  proposal resolves actions only for SAILOR (CRT/TMZ/none); LUMIERE has no
  treatment column, and RANO is the closest per-visit intervention/response
  proxy. Provisional — revisit once SAILOR adapter lands (categorical
  treatment status is cleaner). Actions are currently logged, not yet
  conditioned on (stage C+ hook in `JEPAWorldModel.forward`).

## Preprocessing

- **D3 — Full BRAINIAC contract now, not deferred.** N4 → 1mm iso →
  rigid MNI → HD-BET → 96³ + nonzero z-score (`src/preprocessing/`).
  Offline script writes `data/lumiere_preprocessed/`; training never
  preprocesses on the fly. Rationale: the backbone is a fixed contract —
  feeding it anything else is out-of-distribution by construction.
  Verified faithful against the official repo's `dataset.py`
  (Resize-trilinear + `NormalizeIntensityd(nonzero=True)` identical).
- **D4 — Template: MNI152NLin2009cAsym 1mm** (TemplateFlow). Same template
  family as SAILOR's MNI derivatives (ICBM 2009c), removing one
  cross-dataset confound (§3.3 caution). `scripts/preprocess.py` warns and
  skips registration if `--template` is omitted — never silently.
- **D5 — Multiprocessing with capped threads** (`--workers`, 2 torch
  threads/worker). Measured single-volume time ~30 s wall; 352 pilot
  volumes completed with 0 failures. Required for the 2455-volume full run.

## Vision backbone

- **D6 — LoRA (r=8, α=16) on frozen ViT-B, peft.** Targets qkv/proj/
  linear1/linear2 (verified 96 adapters, ~1.18M params). Rationale: full
  finetune of 86M params on ≤91 patients invites overfitting; frozen +
  adapters preserves the 32k-scan prior while allowing change-sensitivity
  to be learned (see I3).
- **D7 — First-patch-token pooling.** Matches the official `model.py`
  (`features[0][:, 0]`). Ablation A3 confirmed mean-pooling carries no
  more longitudinal signal, so the official convention stands.
- **D8 — Official `BrainIAC.ckpt`, never community ports.** The HF
  `backbone.safetensors` port proved corrupt (A2): identical outputs for
  MRI/noise/zeros with 150× activation blowup. Official weights load
  strict-clean (minus `norm_cross_attn`, a MONAI-1.6 addition absent from
  the 1.3.2-era checkpoint, and the SimCLR `projection_head`, correctly
  ignored). Lesson recorded as I4.

## Temporal model

- **D9 — Per-sequence latents, mean over available modalities.** BRAINIAC
  is single-sequence; channel stacking would need a new stem LoRA doesn't
  cover (proposal §9, decided). Missing modalities zero-fill + boolean
  mask; visits average only over present sequences.
- **D10 — Time-delta encodings: sinusoidal + learned residual.**
  Irregular visits (days to months apart) can't use positional indices.
  Sinusoid carries scale-free relative time; the 0.1-weighted residual MLP
  learns dataset-specific tempo.
- **D11 — Prefix-wise targets.** Every prefix predicts its next visit
  (~4–6× targets on 2–8 visit sequences). `forward_prefixes` re-encodes
  prefixes (O(T) passes, T ≤ 21 — negligible vs ViT cost).
- **D12 — Cosine loss on normalized embeddings.** Unnormalized L2 invites
  scale shrinkage (proposal §4.1). Degenerate batches (no valid pairs)
  contribute `0 × grad` loss — backward-safe, no NaN.

## Data handling

- **D13 — Imageless visits dropped eagerly at dataset init.**
  Patient-025's 8 upstream-missing weeks (proven absent from the Figshare
  archive, not a download bug) plus 3 more cohort-wide would otherwise
  train the predictor toward the projector-bias constant. Counts are exact
  under multiprocess loading; loss masks additionally require pixels on
  both sides of every pair (covers load-time failures too).
- **D14 — NaN-safe clinical parsing.** pandas NaN is truthy, so `float(x)
  or 0` leaks NaN (31/91 patients affected via survival/MGMT-quantitative).
  Explicit `isna` coercion; swept clean over the full cohort.
- **D15 — Chunked backbone encoding (8 vols/chunk).** A full batch can
  hold 200+ volumes in one ViT forward — OOM on ≤24GB GPUs. Chunked
  index-assign preserves gradients.

## Dependencies

- **D16 — `transformers<5`, no torchvision.** HD-BET's install pulled
  transformers 5 + a skewed torchvision 0.29, which broke peft's import
  (`torchvision::nms does not exist` surfacing as a fake Bloom error —
  the lazy loader masked the root cause). torchvision serves nothing in
  this stack; it is banned in `requirements.txt` with the reason attached.

## Pilot scoping

- **D17 — CPU pilot: 5 richest patients, 5 epochs, batch 1.**
  Patients 067/031/073/078/029 (top imaged-volume counts). Same code path
  as full-scale; only config + patient list change. Purpose: prove the
  loop learns (loss drops, beats persistence) before GPU spend.

## Scale-up execution

- **D18 — Hero run on free Kaggle GPU, not a paid 4090.** A7 showed epochs
  are spent (plateau ~13) and patients are the lever — the paid 4090 buys
  speed, not capability. Kaggle free tier (30 h GPU/week, ~9 h sessions,
  16 GB VRAM) fits: batch 2 (chunked encoding, D15; drop to 1 on OOM),
  one private input dataset (`lumiere_preprocessed/` + `lumiere_meta/` +
  official `BrainIAC.ckpt`), `kaggle/hero_run.ipynb` wires paths into a
  generated `kaggle.yaml`. Consequences: (a) new `--resume-from` flag
  (weights only, fresh optimizer/schedule) to split across sessions —
  trainer already checkpoints every epoch; (b) no hd-bet in the Kaggle
  env (preprocessing stays local; hd-bet would reintroduce the D16
  transformers/torchvision conflict); (c) torchvision left at the image's
  matched version unless peft's import errors; (d)   full local preprocess
  (~2051 remaining volumes, `--workers 2` — 4+ workers crash the pool
  with BrokenPipe at startup, cause still open) must finish before the
  dataset upload.

  - **D18 correction (same night) — no worker-count bug.** A 4-worker
    diagnostic on Patient-025 ran 12+ min with zero errors; the earlier
    BrokenPipes were collateral, not a code bug: (1) the 8-worker pool
    was murdered mid-flight by a tool-timeout process-group kill, and
    its orphaned workers spun at 80% for 20+ min; (2) the 4-worker
    restart launched straight into that load-38 wreckage. The 2-worker
    run continues by choice (stability + laptop headroom), not
    necessity. Collateral lessons: kill orphaned forkserver workers
    after any killed pool run (`ps` for `multiprocessing.forkserver`
    with the script's `main_path`); never run two preprocess
    invocations concurrently (both raced Patient-025 — duplicate killed,
    finals verified 16/16 valid 96³ finite via nibabel). Note the
  pipeline keeps `_tmp/` intermediates (~4× file count) next to final
  `{CT1,T1,T2,FLAIR}.nii.gz` — only finals matter downstream.

- **D20 — Leg 2 resumes from best.pt at 5× lower LR, never from last.pt.**
  Run 4 (leg 1, 30 epochs, LR 1e-4) found its optimum at epoch ~7 then
  overfit the tail, so `last.pt` holds overfit weights while `best.pt`
  (val 0.0081) is the lowest-val checkpoint. Leg 2 (`--resume-from
  best.pt`, `--lr 2e-5`, fresh optimizer/schedule per the documented
  `--resume-from` limitation) exploits with small steps. `--epochs` per
  leg means ADDITIONAL epochs (counter restarts); best.pt tracking makes
  over-long legs safe. Previous-session outputs ferry via a
  `prev-checkpoints` input dataset (opt-stripped best.pt, ~1 GB).

- **D21 — Gradient-checkpoint temporal prefixes; T4 peak is one long
  patient, not fragmentation.** Leg 2 OOMed twice at batch 12/65 on a
  14.56 GiB T4, 20 MB short both times — first in the ViT backbone,
  then (with `PYTORCH_ALLOC_CONF=expandable_segments:True`) in the
  temporal encoder. Root cause: `forward_prefixes` ran the full 6-layer
  encoder once per prefix (T−1 encodes, all graphs held for backward),
  so a long-history patient spiked GBs. Fix (`b8af83b`): each prefix
  encode is `torch.utils.checkpoint`ed (`use_reentrant=False`) —
  activations freed after forward, recomputed at backward, peak drops
  from T−1 live encodes to ~1 with identical math (RNG preserved, so
  dropout matches) at ~30% slower temporal steps. Same commit sets
  `enable_nested_tensor=False` (the pre-norm encoder was already
  falling back; silences the UserWarning, no behavior change). Leg 1
  survived the same patient on allocator luck; batch 12 is the memory
  worst-case, so passing it clears the leg. Collateral Kaggle lessons
  from the same night: datasets may mount at
  `/kaggle/input/datasets/<user>/<slug>/` instead of
  `/kaggle/input/<slug>/` (resume glob is now recursive), and `%env`
  swallows trailing `#` comments into the value (keep them on separate
  lines) — both bit leg-2 startup.

  - **D21 correction (same night) — the prefixes were not the hog.** The
    checkpointed run OOMed again at the same batch inside a SINGLE
    prefix, 20 MB short: ViT activations for every volume of the
    long-history patient, held for backward, dominate the peak — not
    the T−1 prefix graphs. Follow-up (`80f7b10`, verified bit-identical
    forward + finite backward): `encode_chunked` now checkpoints the
    backbone call on the grad path only (target no-grad path keeps the
    plain call). Peak is ~1 chunk transient + ~1 prefix instead of all
    volumes + all prefixes, at ~30–40% slower steps — still inside the
    session budget.

- **D22 — Leg-2 verdict: the optimum is not holdable; stop training
  LUMIERE, probe once, then pivot.** Run 5 (resume 0.0081 champion, 30
  epochs flat 2e-5) ascended monotonically 0.0089 → 0.031 plateau;
  Run 4's tail did the same at decaying LR (0.0081 → 0.0704). Two
  schedules, same direction: continued gradient steps cannot hold or
  improve the epoch-6/7 basin. Leading suspect is the documented
  `--resume-from` limitation — fresh optimizer discards the momentum
  that held the narrow basin (drift starts in ep1 even at warmup-tiny
  LR, so it is direction, not size; monitors stay healthy, so it is
  not collapse). Consequences: (a) no more 30-epoch legs; (b) one
  final 10-epoch probe at 5e-6 from the 0.0081 champion (~25 min) —
  holds means an exploit path exists, drifts means the champion
  (val 0.0081 / test 0.0074, merit ~2.6× banked) is the final model;
  (c) all remaining effort pivots to SAILOR held-out eval + writeup.
  Housekeeping rule from the same night: a resume leg's `best.pt`
  overwrites the staged champion with that leg's best — always label
  downloads with (leg, epoch, val) and ferry the true champion file.

- **D24 — Literature survey pivots the program to representation
  quality (2026-09-05).** Frontier on LUMIERE: TaDiff-Net (TMI 2025,
  10.1109/TMI.2025.3533038) generates future MRI+masks but never
  touches RANO and reports no representation metric; RANO-4-class
  SOTA is Tikhonov 2025 hybrid (ResNet-18 + radiomics/volumetry →
  CatBoost, macro F1 0.50 / AUC 0.81 / acc 0.72, patient-wise CV),
  then TRACE 2026 (F1 0.477) and Matoso 2025 (balanced acc 0.51).
  Our frozen probe (A9, F1 0.31) trails — but all SOTA trains
  end-to-end on visit-pairs with volumetry, ours reads frozen
  single-visit snapshots. Consequences: (a) adopt the field's
  pair-framing (consecutive-visit-pair → 4-class, patient-wise CV) so
  numbers read against 0.50; (b) volume unblocked — LUMIERE ships
  automated segmentations (DeepBraTumIA + HD-GLIO-AUTO, ~599 studies,
  zero expert masks), enabling volume-from-latent and
  next-visit-volume probes nobody has published; (c) cheapest new
  experiment is surprise-as-signal (per-patient JEPA errors from A8 ×
  RANO labels: does prediction error anticipate PD? zero GPU);
  (d) SAILOR access becomes a priority track — it is TaDiff's Oslo
  training cohort, making LUMIERE→SAILOR the cross-site test the
  field lacks; (e) D19 retrain gated on frozen probes approaching
  0.50 on pair-framing. Unchanged: 5e-6 probe, no more 30-epoch legs.
  Glossary: PD = Progressive Disease (RANO; tumour grew/new lesions);
  SD/PR/CR = stable/partial/complete. PD is our majority class
  (253/397, ~64%), hence macro-F1 over accuracy.

- **D23 — Frozen RANO probe: first downstream test, runs local on CPU.**
  JEPA loss (~0.008, cosine ≈ 0.992) is saturated as an objective;
  tumour-size/RANO performance is the metric now. Labels exist:
  `rano_rating__LUMIERE-ExpertRating-v202211.csv`, 616 rows / 91
  patients — PD 253, SD 97, CR 27, PR 20 (+ operative states Pre-Op 92,
  Post-Op ~124, Post-Op/PD 1). Task: 4-class response classification
  {PD, SD, PR, CR} (n=397 visits; operative states excluded — surgical
  status, not response). No tumour-size labels exist anywhere (no
  masks/volumes/diameters); RANO-as-proxy, since response categories
  ARE size-change categories. Design: freeze 0.0081 champion, per-visit
  features = fused tokens (vision+clinical, 1152-d), ablate vision-only
  (768-d) vs clinical-only; linear probe (class-weighted CE) + optional
  1-hidden MLP. Splits: patient-level, reuse the 65/13/13 hero split
  (probe trains on train patients, scores test — no leakage).
  Baselines: majority-PD (64% acc floor), clinical-only probe
  (separates "representation knows progression" from "demographics
  knows it"). Metrics: macro-F1 primary (PD-heavy imbalance), accuracy,
  confusion matrix. Local runbook: (0) `kaggle datasets download
  nairadithya/prev-checkpoints` — the 0.0081 champion is NOT on local
  disk (only leg-2 files are); (1) encode-and-cache all visit latents
  overnight on CPU (one-time, hours; chunk 1–2 for laptop RAM, save
  `.pt` cache, subset to RANO-labelled visits optional); (2) probe
  training = minutes on CPU. Success bar: fused probe clearly beats
  clinical-only + majority on macro-F1 → representation earns its keep
  and decides D19/multi-task RANO loss (awin) vs straight to SAILOR.
  Failure mode it rules out first: representation holds no progression
  signal, in which case no JEPA-loss squeezing would ever have fixed
  it.

## Future work (after hero leg 2)

- **D19 — Additive clinical conditioning (fusion upgrade).** Today fusion
  is concat of LayerNormed vision + clinical branches (proposal §4.1;
  the LayerNorm-per-branch is the standing mitigation for scale
  imbalance). The upgrade: an MLP mapping the clinical vector to the
  vision latent dim, ADDED to the ViT latent (residual-style offset),
  with the joint representation learned around that sum. Rationale:
  addition keeps the backbone dim (no concat bottleneck), lets clinical
  context steer the representation without overwhelming it, and is a
  smaller step than cross-attention (still reserved as the further
  upgrade). Evaluation stays the same gate: beat persistence, ablate
  with/without the additive path, confirm `C_i` isn't near-constant
  (clinical-only baseline must stay weak). Status: parked until leg 2
  lands — no architecture churn mid-hero-run.
