# Ablations and diagnostics

Append-only. Each entry: setup → numbers → inference. IDs referenced from
`decisions.md` and `pilot.md`.

## A1 — Random-init sanity (2026-09-03)

- Setup: full JEPA model, `BrainiacEncoder(checkpoint=None)`, Patient-001,
  real NIfTIs, CPU eval.
- Numbers: JEPA loss 0.978 (chance ≈ 1.0 for cosine), z shapes (1,3,768)
  = 3 prefix targets from 4 visits, train step + EMA update OK.
- Inference: plumbing correct; loss scale calibrated. Baselines run:
  persistence 0.636 < random JEPA — the gap the model must close (I1: the
  evaluation harness can resolve a learning signal when one exists).

## A2 — HF-port vs official weights (2026-09-04)

- Setup: `BrainiacEncoder` with `eugenehp/brainiac:backbone.safetensors`
  (137/137 keys matched) vs official `BrainIAC.ckpt` from the authors'
  Dropbox; probe = real CT1 vs pure noise vs all-zeros, cosine drift.
- Numbers:

  | weights | real-vs-noise | CT1-vs-FLAIR | output norm |
  |---|---|---|---|
  | HF port | 0.0002 | 0.0000 | 30.02 (identical for all inputs) |
  | official | 0.0709 | 0.0858 | 27.75 |
  | random-init | 0.8707 | — | — |

  HF-port block activations explode 150× (std 2.6 → 385); tensors
  finite, shapes matching — silent corruption, invisible to key-count
  checks (I4).
- Inference: key-name matching does not validate a weight port. Real
  behavioral probes (noise/contrast discrimination) are mandatory for any
  third-party checkpoint. → D8.

## A3 — First-token vs mean pooling (2026-09-04)

- Setup: official weights, all 5 pilot patients, per-sequence
  consecutive-visit drift under both poolings.
- Numbers: CT1 0.000022/0.000019, T1 0.000021/0.000019,
  T2 0.000031/0.000028, FLAIR 0.000024/0.000020 (first/mean).
- Inference: pooling carries no additional signal → keep the official
  first-token convention (D7).

## A4 — Raw+minimal vs full preprocessing (2026-09-04)

- Setup: same drift metric on raw NIfTIs (resize + z-score only) vs
  full-contract preprocessed volumes, Patient-067, all sequences.
- Numbers: raw drift 0.000024–0.000036 vs preprocessed 0.000021–0.000031
  per sequence — indistinguishable.
- Inference: our preprocessing preserves whatever signal the backbone
  can see; it neither destroys nor creates longitudinal contrast. The
  pipeline is faithful (matches official transforms line-for-line).

## A5 — Backbone drift: fresh vs trained (2026-09-04, broken-weights era)

- Setup: raw backbone features (no projector), fresh pretrained net vs
  online/EMA backbones after the 5-epoch pilot on corrupt weights.
- Numbers: all three give drift 0.000036, feat-std 1.0832 — identical to
  6 decimals.
- Inference: training did not touch the backbone (frozen + negligible
  LoRA movement). The first pilot's collapse signature (target std
  0.0024, rank 1.1, val loss → 0.03) was constant-target collapse in the
  *heads*, downstream of constant features from corrupt weights — not a
  flaw in the training loop. The monitors (std/rank) detected it
  correctly; the misreading was human (I2).

## A6 — Persistence baseline, official weights (2026-09-04)

- Setup: 5-epoch pilot model (`checkpoints/pilot/best.pt`), all 5 pilot
  patients, per-patient loss averaged. GRU/clinical baselines untrained
  (reference floors only).
- Numbers: JEPA 0.0576, persistence 0.0043, GRU 0.9993, clinical-only
  1.0027. (Persistence 0.0043 ≈ pre-train drift mean 0.0046 — consistent.)
- Inference: **learning signal proven** (JEPA far below chance and below
  untrained dynamics), but **persistence not beaten** — expected after 5
  epochs on 4 patients against near-static targets. Beating persistence
  is the explicit gate for the hero run (proposal §6); it needs scale
  (more patients, more epochs, LoRA fully warmed), not a redesign.

## A7 — Extended 25-epoch pilot (2026-09-04/05)

- Setup: same 5 patients + `config/pilot.yaml` as Run 2, `--epochs 25`,
  official weights, CPU. Best val 0.0093 (plateau ~0.0093–0.0100 from
  epoch 13 on; monitors healthy throughout: std 0.0791, rank 2.2).
  Per-patient JEPA vs persistence on `checkpoints/pilot/best.pt`:
- Numbers:

  | patient | JEPA | persistence | split |
  |---|---|---|---|
  | Patient-067 | 0.0100 | 0.0047 | train |
  | Patient-031 | 0.0051 | 0.0043 | train |
  | Patient-073 | 0.0042 | 0.0045 | train, JEPA wins |
  | Patient-078 | 0.0093 | 0.0041 | train |
  | Patient-029 | 0.0262 | 0.0040 | val (held-out) |
  | mean | 0.0110 | 0.0043 | — |

- Inference: gap narrowed 13× → ~2.5×, with the first patient-level win
  (073) and a near-tie (031). But the only held-out patient (029) is the
  worst by far — the model fit train dynamics, not general ones. Mean
  persistence (0.0043, identical to A6) still unbeaten. Scale (more
  patients, not more epochs — epochs stopped helping at ~13) is the
  remaining lever; the train/val generalization gap is what the hero run
  must close.
