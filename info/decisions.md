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
  matched version unless peft's import errors; (d) full local preprocess
  (~2051 remaining volumes, `--workers 2` — 4+ workers crash the pool
  with BrokenPipe at startup, cause still open) must finish before the
  dataset upload.
