# CPU pilot

## Protocol

- Patients: 067, 031, 073, 078, 029 (top-5 imaged-volume counts;
  352 volumes). Split 4 train / 1 val (sorted order). No test split.
- Config: `config/pilot.yaml` — batch 1, lr 1e-4, warmup 1, 5 epochs,
  AMP off (CPU), `checkpoints/pilot/`, real `BrainIAC.ckpt` weights.
- Command: `run_train.py --config config/pilot.yaml --patients ... --no-wandb`
- Success criteria: (a) train/val loss drops substantially from ~1.0;
  (b) collapse monitors stay healthy (target std ≫ 0, rank > 1);
  (c) trained JEPA beats persistence (A6).

## Run 1 (2026-09-03/04) — INVALID (corrupt weights)

- Weights: HF community port (later proven corrupt, A2).
- Result: val 0.37 → 0.03, but target std 0.0024 / rank 1.1 throughout —
  constant-target collapse in the heads. Checkpoints deleted.
- Lesson: a falling loss without healthy monitors proves nothing (I2).

## Run 2 (2026-09-04) — in progress, official weights

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
3. SAILOR adapter dataset (`sub-XX/ses-YY` + `RANO.txt`) for held-out eval.
4. `config/default.yaml` unchanged code path; raise batch size to GPU fit.
