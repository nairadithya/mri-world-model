# CPU pilot

## Protocol

- Patients: 067, 031, 073, 078, 029 (top-5 imaged-volume counts;
  352 volumes). Split 4 train / 1 val (sorted order). No test split.
- Config: `config/pilot.yaml` — batch 1, lr 1e-4, warmup 1, 5 epochs,
  AMP off (CPU), `checkpoints/pilot/`, real `BrainIAC.ckpt` weights.
- Command: `run_train.py --config config/pilot.yaml --patients Patient-067 Patient-031 Patient-073 Patient-078 Patient-029 --no-wandb`
  (Runs 1–2 used `max_epochs: 5` from `pilot.yaml`; Run 3 overrides `--epochs 25`.)
- Success criteria: (a) train/val loss drops substantially from ~1.0;
  (b) collapse monitors stay healthy (target std ≫ 0, rank > 1);
  (c) trained JEPA beats persistence (A6).

## Run 1 (2026-09-03/04) — INVALID (corrupt weights)

- Weights: HF community port (later proven corrupt, A2).
- Result: val 0.37 → 0.03, but target std 0.0024 / rank 1.1 throughout —
  constant-target collapse in the heads. Checkpoints deleted.
- Lesson: a falling loss without healthy monitors proves nothing (I2).

## Run 2 (2026-09-04) — complete, official weights

- Weights: official `BrainIAC.ckpt` (362MB). Pre-train drift audit:
  longitudinal mean 0.0046 (max 0.028) — small but real signal for LoRA
  to amplify.
- Epoch 1: val loss 0.4256, std 0.0791, rank 2.2.
- Epoch 2: val loss 0.1868, std 0.0791, rank 2.2.
- Epoch 3: val loss 0.0889, std 0.0791, rank 2.2.
- Epoch 4: val loss 0.0614, std 0.0791, rank 2.2.
- Epoch 5: val loss 0.0564, std 0.0791, rank 2.2. Done; best 0.0564.

## Run 2 verdict

- Success (a) PASS: 0.43 → 0.056 from chance (~1.0).
- Success (b) PASS: monitors healthy throughout (std 0.079, rank 2.2).
- Success (c) FAIL (expected): JEPA 0.0576 vs persistence 0.0043 (A6).
  Five epochs on 4 patients cannot beat near-static targets — this is the
  scale gate for the hero run, not a redesign trigger.
- Pilot objective met: the loop learns real dynamics on real
  preprocessed data with official weights. Cleared to scale.

## Scale-up checklist (after pilot passes)

1. GPU host (no GPU on this machine — pilot epochs take ~8 min on CPU).
2. Full preprocess: 2455 volumes, `--workers 8`, same script.
   → DONE 2026-09-05: 2051 volumes, 0 failures (`logs_preprocess_full.log`);
   91/91 patients, 2487 finals (CT1 632 / T1 617 / T2 626 / FLAIR 612),
   60-file nibabel spot-check clean. Ran `--workers 2` (see D18).
3. SAILOR adapter dataset (`sub-XX/ses-YY` + `RANO.txt`) for held-out eval.
4. `config/default.yaml` unchanged code path; raise batch size to GPU fit.

## Run 3 (2026-09-04/05) — extended 25-epoch pilot, official weights — COMPLETE

- Purpose: de-risk the hero run on free CPU before spending (own money).
  Gate: does the JEPA-vs-persistence gap (Run 2: 0.0576 vs 0.0043, A6)
  narrow with more epochs, or plateau far above persistence?
- Code state: commit `8193bde`, `config/pilot.yaml` unchanged (batch 1,
  lr 1e-4, warmup 1, AMP off, seed 42); epochs via CLI override.
- Pre-step (trainer starts fresh, overwrites `best.pt`/`last.pt` — no
  resume): `cp checkpoints/pilot/best.pt checkpoints/pilot/best_run2_ep5.pt`
- Command (background, log ignored by `*.log` gitignore rule):

  ```bash
  nohup .venv/bin/python scripts/run_train.py --config config/pilot.yaml \
    --patients Patient-067 Patient-031 Patient-073 Patient-078 Patient-029 \
    --no-wandb --epochs 25 > logs_pilot_extended.log 2>&1 &
  ```

- ETA: ~8 min/epoch, ~3.5 h total. Per-epoch lines (`val epoch N:
  loss=... std=... rank=...`) stream to `logs_pilot_extended.log`.
- Results: best val 0.0093 (plateau from epoch ~13; monitors healthy:
  std 0.0791, rank 2.2). Per-patient JEPA vs persistence — 067:
  0.0100/0.0047; 031: 0.0051/0.0043; 073: 0.0042/0.0045 (first JEPA
  win); 078: 0.0093/0.0041; 029 (held-out val): 0.0262/0.0040. Means:
  JEPA 0.0110 vs persistence 0.0043. Full numbers + inference: A7.
  Gate verdict: gap narrowed 13× → 2.5× but persistence unbeaten on
  average and the held-out patient is worst — more epochs are spent
  (plateaued); what remains is more patients, i.e. the hero run.

## Run 4 (2026-09-05) — hero leg 1, full cohort on Kaggle T4 — COMPLETE

- Setup: 65/13/13 patient split, batch 1, 30 epochs, AMP on, official
  weights. Code ≥ `bbc2a36` (pulled for the `.nii` fallback — Kaggle
  gunzips ingestion in place). Notebook path, `kaggle.yaml` from
  `config/default.yaml`. First run at scale: 65 train patients (vs 4).
- Command:

  ```bash
  python scripts/run_train.py --config kaggle.yaml --epochs 30 --batch-size 1 --no-wandb
  ```

- Val trajectory: ep1 0.0830 → ep5 0.0089 → ep7 0.0081 → ep10 0.0129 →
  ep15 0.0316 → ep20 0.0510 → ep25 0.0667 → ep30 0.0704.
  Best val 0.0081. Monitors healthy throughout (std 0.085→0.29,
  rank 1.7→2.5 — no collapse). Train loss stays ~0.01–0.04.
- Test scorecard (13 held-out patients, `best.pt` reloaded): loss 0.0074,
  std 0.0700, rank 1.6.
- Inference: U-shaped val = best model at epoch ~7–8, then steady
  overfit climb while train holds — more epochs at this LR are spent;
  capacity (65M trainable) still dwarfs 65 patients. Test 0.0074 is the
  first real held-out scorecard (A7's n=1 val is retired as the
  reference). Eval verdict (A8): mean JEPA 0.0081 vs persistence 0.0218
  over all 91 (~2.7×, 82/91 wins, biggest on the most dynamic
  patients) — the mandatory baseline is beaten at scale.
- Next: leg 2 resumes from `best.pt` (epoch ~8 weights), NOT `last.pt`
  (epoch-30 overfit weights).

## Run 5 (2026-09-05) — hero leg 2, resume best.pt at LR 2e-5 — COMPLETE, exploit FAILED

- Setup: resume `/kaggle/working/checkpoints/best.pt` (Run 4 champion,
  val 0.0081) via `prev-checkpoints` dataset, `--epochs 30 --batch-size
  1 --lr 0.00002`, fresh optimizer/schedule, code `80f7b10` (D21
  checkpointing — batch-12 long-history patient passes, 1.4 s/it).
  `PYTORCH_ALLOC_CONF=expandable_segments:True` kept from the OOM
  fight (harmless, possibly helpful).
- Val trajectory: ep1 0.0089 → ep5 0.0167 → ep10 0.0210 → ep15 0.0267 →
  ep20 0.0291 → ep25 0.0309 → ep30 0.0308. Monotonic rise then plateau;
  this leg's best is ep1 (0.0089), champion 0.0081 never touched.
  Monitors healthy throughout (std 0.10→0.19, rank 1.7→2.2 — drift,
  not collapse). Train loss bouncy 0.007–0.042, no trend.
- Isolated `std=0.0000 rank=1.0` train rows (ep 4/24/27/29) are
  degenerate batches with no valid pairs (monitor prints 0 per
  `collapse_metrics` empty-input guard) — logging artifact, val
  unaffected those epochs.
- Eval (leg-2 `best.pt` = ep1 weights ≈ champion + 65 warmup-tiny
  steps): MEAN JEPA 0.0085 vs persistence 0.0221 over 91 (~2.6×).
  Merit intact (cf. A8 0.0081/0.0218).
- Verdict (D22): the epoch-6/7 optimum is NOT holdable by continued
  training. Leg-1 tail ascended at decaying LR (→ 0.0704), leg 2
  ascends at flat 2e-5 (→ 0.031 plateau) — same direction, LR only sets
  drift speed. Suspect: fresh optimizer discards leg-1 momentum that
  held the narrow basin (batch-1 noise ejects from ep1, even during
  warmup). No more 30-epoch legs. Remaining LUMIERE question is one
  10-epoch 5e-6 probe from the 0.0081 champion (~25 min, confirmatory):
  holds → exploit path exists; drifts → champion is final, effort
  pivots to SAILOR eval + writeup.
- Housekeeping: leg-2 `best.pt` on disk is ep1 weights (val 0.0089),
  NOT the 0.0081 champion — it overwrote the staged copy. Champion
  survives in `prev-checkpoints` + leg-1 outputs. Label all future
  downloads with (leg, epoch, val).

## Run 6 (2026-09-06) — aux fine-tune, champion + RANO heads, 10 epochs — COMPLETE, verdict PENDING re-gate

- Setup (D25, path B): resume 0.0081 champion, joint JEPA + λ=1.0
  (flat/prog/resp heads, fresh random), `--lr 5e-6 --warmup-epochs 1`,
  batch 1, T4. Code `f6a4ffe`. Resume line correctly listed the six
  `rano_heads.*` keys as randomly initialized.
- Train: total loss aux-dominated (1.43 → 7.01 spike ep2 as random heads
  engage → 0.66 ep10); aux_flat 3.08 → 0.37, aux_prog 0.53 → 0.23,
  aux_resp 3.37 → 0.04. Val total 1.89 → 1.69 (best ep6, 1.6850).
  Monitors healthy throughout (std 0.09→0.15, rank 1.7→2.0 — no
  collapse). Isolated std=0 blank train rows (ep 4/7/9) = degenerate
  batches again, val unaffected.
- Eval (aux-tuned best.pt, all 91): MEAN JEPA 0.0191 vs persistence
  0.0220 — still beats persistence on mean, but the margin collapsed
  (was 0.0081/0.0218): JEPA error more than doubled while persistence
  stood still. Patient-level losses multiplied (004/008/010/011/018/
  019/029/032/034/037/043/045/046/054/061/063/064/066/068/072/075/077/
  080/082/083/086/087/091); mean survives on remaining big wins
  (076/084/002/074/...). Predicted tension confirmed: encoder tilted
  toward the classifier, dynamics paid.
- Verdict PENDING: worth it iff the frozen RANO re-probe with the NEW
  encoder jumps (target 0.33 → 0.40+). Needs downloaded best.pt +
  train_aux.log → local re-gate (frozen-probe F1, JEPA val decomposed,
  persistence from this table). If probe flat: aux run was pure damage,
  champion unchanged, path A (joint-from-scratch) likely not worth its
  session either.

### Run 6 verdict — RE-GATE FAILED, champion stands (2026-09-06, local CPU)

- Re-gate (`checkpoints/aux10/`, re-encoded cache, frozen probes):
  hero-split states_forecast-mlp F1 0.509 (up from 0.448) — but CV
  (5-fold, same protocol) 0.328±0.038 vs pre-aux 0.334±0.060:
  UNCHANGED. The hero-split gain is split luck, same trap as A9.
  JEPA-only val 0.0205 (was 0.0081), test 0.0174 (was 0.0074):
  dynamics damaged ~2.5× while the honest task number didn't move.
- Inference: path B failed its gate. Aux pressure tilted the encoder
  (hero-split numbers dance) without improving generalizable RANO
  signal, and charged 2.5× JEPA error. Champion (0.0081/0.0074) stands
  untouched. Path A (joint-from-scratch, full session) is NOT earned —
  same pressure, more instability, no evidence of headroom. D19 fusion
  retrain stays gated. Representation program continues via SAILOR
  generalization + volume framings, not more LUMIERE gradient steps.
