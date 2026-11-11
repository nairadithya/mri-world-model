#!/usr/bin/env bash
# Ordinary sequential job: preparation -> real MU smoke -> matched fits -> evaluation -> timeline.
# No process killing, retry supervisor, or budget lockout.
set -euo pipefail
run_root=/home/aditya/experiments/world-model-patient-state-20261105
python_bin=/home/aditya/experiments/world-model-observability-20261103/.venv/bin/python
DEMO_PATIENT=${DEMO_PATIENT:?Set demonstration patient identifier}
DEMO_VISIT=${DEMO_VISIT:?Set demonstration visit identifier}
cd "$run_root/code"
# An already-running preparation may be allowed to finish before this job starts.
if [[ $# -gt 0 ]]; then
    tail --pid="$1" -f /dev/null
fi
"$python_bin" -u scripts/patient_state/prepare_structured_forecasting.py \
    --catalogue "$run_root/mu-timed-v1/catalogue.json" \
    --observation "$run_root/observation-preserved-v1/best.pt" \
    --out "$run_root/structured-context-v1"
"$python_bin" -u -m src.patient_state.structured_training \
    --contexts "$run_root/structured-context-v1" --out "$run_root/structured-smoke-mu-v1" \
    --kind anatomy --arm history --smoke --smoke-dataset MU
for kind in volume anatomy; do
    for arm in current history; do
        "$python_bin" -u -m src.patient_state.structured_training \
            --contexts "$run_root/structured-context-v1" \
            --out "$run_root/structured-v1/$kind-$arm" \
            --kind "$kind" --arm "$arm" --steps 6000 --select-every 500 --seed 17
    done
done
"$python_bin" -u scripts/patient_state/evaluate_structured_forecasts.py \
    --contexts "$run_root/structured-context-v1" --runs "$run_root/structured-v1" \
    --out "$run_root/structured-selection-v1"
"$python_bin" -u scripts/patient_state.py timeline \
    --catalogue "$run_root/mu-timed-v1/catalogue.json" \
    --checkpoint "$run_root/observation-preserved-v1/best.pt" \
    --structured-current "$run_root/structured-v1/anatomy-current/best.pt" \
    --structured-history "$run_root/structured-v1/anatomy-history/best.pt" \
    --patient "$DEMO_PATIENT" --through "$DEMO_VISIT" --out "$run_root/structured-timeline-v1"
