#!/usr/bin/env bash
# Ordinary sequential training/evaluation. No process supervisor or runtime kill switch.
set -euo pipefail
run_root=$1
python_path=$2
cd "$run_root/code"
"$python_path" -B -m src.patient_state.forecast_training \
  --catalogue "$run_root/mu-prepared/catalogue.json" \
  --observation "$run_root/observation-preserved-v1/best.pt" \
  --arm current --steps 8000 --select-every 500 --out "$run_root/temporal-current-v1"
"$python_path" -B -m src.patient_state.forecast_training \
  --catalogue "$run_root/mu-prepared/catalogue.json" \
  --observation "$run_root/observation-preserved-v1/best.pt" \
  --arm history --steps 8000 --select-every 500 --out "$run_root/temporal-history-v1"
"$python_path" -B scripts/patient_state/evaluate_patient_forecasts.py \
  --catalogue "$run_root/mu-prepared/catalogue.json" \
  --observation "$run_root/observation-preserved-v1/best.pt" \
  --current "$run_root/temporal-current-v1/best.pt" \
  --history "$run_root/temporal-history-v1/best.pt" \
  --samples 16 --out "$run_root/temporal-selection-v1"
