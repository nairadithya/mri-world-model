#!/usr/bin/env bash
# One sequential adaptation, using immutable old inputs and an isolated new root.
set -euo pipefail
run_root=/home/aditya/experiments/world-model-mu-cavity-forecast-20261107
python_bin=/home/aditya/experiments/world-model-observability-20261103/.venv/bin/python
cd "$run_root/code"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ "${1:-}" != "--continue-after-smoke-timeline" && "${1:-}" != "--continue-after-verified-smoke" ]]; then
# Engineering checks precede cohort fitting. No process/budget supervisor.
"$python_bin" -B -m unittest discover -s tests -p 'test_anatomy_readouts.py' -v
"$python_bin" -B -m unittest discover -s tests -p 'test_structured_forecasting.py' -v
"$python_bin" -B -m unittest discover -s tests -p 'test_cavity_supervision.py' -v
"$python_bin" -u scripts/prepare_structured_forecasting.py \
  --catalogue ../baseline/cavity-targets-v1/catalogue.json \
  --observation ../baseline/observation-cavity-v1/best.pt \
  --fixed-mu-pairs ../cleanup-records/next-mu-pairs.json --out ../contexts-v2 --smoke-pair
for kind in volume anatomy; do
  "$python_bin" -u -m src.patient_state.structured_training \
    --contexts ../contexts-v2 --out "../smoke-v2/$kind-current" --kind "$kind" \
    --arm current --compartments 4 --smoke
done
"$python_bin" -u scripts/evaluate_cavity_forecasts.py \
  --contexts ../contexts-v2 --runs ../smoke-v2 --preserved ../baseline/structured-selection-v1 \
  --out ../smoke-evaluation-v2 --smoke
"$python_bin" -u scripts/build_cavity_forecast_timeline.py \
  --contexts ../contexts-v2 --checkpoint ../baseline/observation-cavity-v1/best.pt \
  --transition ../smoke-v2/anatomy-current/last.pt --evaluation ../smoke-evaluation-v2 \
  --out ../timeline-smoke-v2 --smoke --catalogue ../baseline/cavity-targets-v1/catalogue.json
fi
if [[ "${1:-}" != "--continue-after-verified-smoke" ]]; then
"$python_bin" -u scripts/verify_cavity_forecast_timeline.py \
  --root ../timeline-smoke-v2 --checkpoint ../baseline/observation-cavity-v1/best.pt
fi
"$python_bin" -u scripts/prepare_structured_forecasting.py \
  --catalogue ../baseline/cavity-targets-v1/catalogue.json \
  --observation ../baseline/observation-cavity-v1/best.pt \
  --fixed-mu-pairs ../cleanup-records/next-mu-pairs.json --out ../contexts-v2
for kind in volume anatomy; do
  "$python_bin" -u -m src.patient_state.structured_training \
    --contexts ../contexts-v2 --out "../runs/$kind-current" --kind "$kind" \
    --arm current --compartments 4 --steps 6000 --select-every 500 --seed 17
done
"$python_bin" -u scripts/evaluate_cavity_forecasts.py \
  --contexts ../contexts-v2 --runs ../runs --preserved ../baseline/structured-selection-v1 --out ../evaluation
"$python_bin" -u scripts/build_cavity_forecast_timeline.py \
  --contexts ../contexts-v2 --checkpoint ../baseline/observation-cavity-v1/best.pt \
  --transition ../runs/anatomy-current/best.pt --evaluation ../evaluation \
  --out ../timeline --catalogue ../baseline/cavity-targets-v1/catalogue.json
"$python_bin" -u scripts/verify_cavity_forecast_timeline.py \
  --root ../timeline --checkpoint ../baseline/observation-cavity-v1/best.pt
"$python_bin" - <<'PY'
import json,time
from pathlib import Path
Path('../pipeline-complete.json').write_text(json.dumps({'status':'completed','completed_unix_time':time.time(),'scope':'development selection, no predictive success assumed','reserved_images_opened':0},indent=2)+'\n')
PY
