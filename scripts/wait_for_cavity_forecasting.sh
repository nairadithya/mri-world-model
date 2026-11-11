#!/usr/bin/env bash
# One-time capacity wait before starting; never monitors or kills running work.
set -euo pipefail
cd /home/aditya/experiments/world-model-mu-cavity-forecast-20261107/code
printf 'Waiting for at least 12500 MiB free GPU memory before saved-state replay and observation preparation.\n'
while (( $(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits) < 12500 )); do
    sleep 30
done
printf 'GPU capacity available; starting the fixed adaptation sequence.\n'
exec bash scripts/train_cavity_forecasting.sh --continue-after-smoke-timeline
