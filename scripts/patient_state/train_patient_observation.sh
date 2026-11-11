#!/usr/bin/env bash
# A normal sequential job, without a process supervisor or budget kill switch.
set -euo pipefail
RUN_ROOT=${1:?Usage: train_patient_observation.sh RUN_ROOT PYTHON}
PYTHON_BIN=${2:?Supply the GPU environment Python}
OBSERVATION_RUN=${3:-observation-v2}
DEMO_PATIENT=${4:?Supply demonstration patient identifier}
DEMO_VISIT=${5:?Supply demonstration visit identifier}
cd "$RUN_ROOT/code"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
"$PYTHON_BIN" -B -m src.patient_state.training \
  --catalogue "$RUN_ROOT/mu-prepared/catalogue.json" \
  --pretrained "$RUN_ROOT/pretrained/openmind-resenc-l-mae.pth" \
  --out "$RUN_ROOT/$OBSERVATION_RUN" --steps 37500 --select-every 1250 \
  --patch-size 128 --accumulate 2 --seed 17 --lr 0.0002
"$PYTHON_BIN" -B scripts/patient_state/evaluate_patient_state.py \
  --catalogue "$RUN_ROOT/mu-prepared/catalogue.json" \
  --checkpoint "$RUN_ROOT/$OBSERVATION_RUN/best.pt" \
  --r69 "$(dirname "$RUN_ROOT")/world-model-spatial-access-20261104" \
  --r71 "$(dirname "$RUN_ROOT")/world-model-complete-reuse-20261104" \
  --group selection --out "$RUN_ROOT/observation-selection-full"
# The fixed demonstration uses fitting visits and its own output tree.
"$PYTHON_BIN" -B scripts/patient_state.py timeline \
  --catalogue "$RUN_ROOT/mu-prepared/catalogue.json" \
  --checkpoint "$RUN_ROOT/$OBSERVATION_RUN/best.pt" \
  --patient "$DEMO_PATIENT" --through "$DEMO_VISIT" --out "$RUN_ROOT/timeline-joint"
"$PYTHON_BIN" -B scripts/patient_state.py timeline \
  --catalogue "$RUN_ROOT/mu-prepared/catalogue.json" \
  --checkpoint "$RUN_ROOT/$OBSERVATION_RUN/best.pt" \
  --patient "$DEMO_PATIENT" --through "$DEMO_VISIT" --hide FLAIR --out "$RUN_ROOT/timeline-joint-missing-flair"
