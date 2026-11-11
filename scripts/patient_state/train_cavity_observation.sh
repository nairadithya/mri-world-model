#!/usr/bin/env bash
# One ordinary sequential development job. No architecture matrix or reserved data.
set -euo pipefail
RUN_ROOT=${1:?Supply run root}
PYTHON_BIN=${2:?Supply GPU environment Python}
LUMIERE_PATIENT=${3:?Supply LUMIERE patient identifier}
LUMIERE_VISIT=${4:?Supply LUMIERE visit identifier}
MU_PATIENT=${5:?Supply MU patient identifier}
MU_VISIT=${6:?Supply MU visit identifier}
cd "$RUN_ROOT/code"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
"$PYTHON_BIN" -B -m src.patient_state.training \
  --catalogue "$RUN_ROOT/cavity-targets-v1/catalogue.json" \
  --initialize-observation "$RUN_ROOT/observation-preserved-v1/best.pt" \
  --out "$RUN_ROOT/observation-cavity-v1" --steps 12500 --select-every 1250 \
  --patch-size 128 --accumulate 2 --seed 17 --lr 0.0001 --full-volume-selection
"$PYTHON_BIN" -B scripts/patient_state/evaluate_cavity_observation.py \
  --catalogue "$RUN_ROOT/cavity-targets-v1/catalogue.json" \
  --checkpoint "$RUN_ROOT/observation-preserved-v1/best.pt" --out "$RUN_ROOT/cavity-reference-selection-v1"
"$PYTHON_BIN" -B scripts/patient_state/evaluate_cavity_observation.py \
  --catalogue "$RUN_ROOT/cavity-targets-v1/catalogue.json" \
  --checkpoint "$RUN_ROOT/observation-cavity-v1/best.pt" --out "$RUN_ROOT/cavity-selection-v1"
"$PYTHON_BIN" -B scripts/patient_state.py timeline \
  --catalogue "$RUN_ROOT/cavity-targets-v1/catalogue.json" \
  --checkpoint "$RUN_ROOT/observation-cavity-v1/best.pt" \
  --patient "$LUMIERE_PATIENT" --through "$LUMIERE_VISIT" --out "$RUN_ROOT/timeline-cavity-v1"
"$PYTHON_BIN" -B scripts/patient_state.py timeline \
  --catalogue "$RUN_ROOT/cavity-targets-v1/catalogue.json" \
  --checkpoint "$RUN_ROOT/observation-cavity-v1/best.pt" \
  --patient "$LUMIERE_PATIENT" --through "$LUMIERE_VISIT" --hide FLAIR --out "$RUN_ROOT/timeline-cavity-missing-flair-v1"
"$PYTHON_BIN" -B scripts/patient_state.py timeline \
  --catalogue "$RUN_ROOT/cavity-targets-v1/catalogue.json" \
  --checkpoint "$RUN_ROOT/observation-cavity-v1/best.pt" \
  --patient "$MU_PATIENT" --through "$MU_VISIT" --out "$RUN_ROOT/timeline-cavity-v1"
