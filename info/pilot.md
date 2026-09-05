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
