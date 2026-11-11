# Script index

Use the study-specific protocols in [Methods](../info/METHODS.md) and the [closure record](../info/TODO.md) before any separately authorized reproduction. Numerical workloads require an authorized external compute host with the environment and necessary data restored; mosaic-gpu has been retired. Protect controlled-access data and the unopened reserved cohort.

- **Preparation:** [preprocess.py](preprocess.py), [prepare_patient_state_mu.py](patient_state/prepare_patient_state_mu.py), [prepare_cavity_supervision.py](patient_state/prepare_cavity_supervision.py), [prepare_structured_forecasting.py](patient_state/prepare_structured_forecasting.py), and [prepare_r68_episodes.py](patient_state/prepare_r68_episodes.py).
- **Observation:** [patient_state.py](patient_state.py), [train_patient_observation.sh](patient_state/train_patient_observation.sh), and [train_cavity_observation.sh](patient_state/train_cavity_observation.sh).
- **Forecasting:** [train_patient_forecasting.sh](patient_state/train_patient_forecasting.sh), [train_structured_forecasting.sh](patient_state/train_structured_forecasting.sh), and [train_cavity_forecasting.sh](patient_state/train_cavity_forecasting.sh).
- **Quantity:** [fit_linear_quantity.py](experiments/quantity/fit_linear_quantity.py), [predict_linear_quantity.py](experiments/quantity/predict_linear_quantity.py), and [train_amount_first.py](experiments/quantity/train_amount_first.py).
- **Acquisition:** [prepare_acquisition_study.py](experiments/acquisition/prepare_acquisition_study.py), [train_acquisition_jepa.py](experiments/acquisition/train_acquisition_jepa.py), and [evaluate_acquisition_jepa.py](experiments/acquisition/evaluate_acquisition_jepa.py). The [historical data access guide](DATA_ACCESS.md) records earlier download instructions.
- **Evaluation:** [evaluate_patient_state.py](patient_state/evaluate_patient_state.py), [evaluate_patient_forecasts.py](patient_state/evaluate_patient_forecasts.py), [evaluate_structured_forecasts.py](patient_state/evaluate_structured_forecasts.py), and [evaluate_cavity_forecasts.py](patient_state/evaluate_cavity_forecasts.py).
- **Viewers:** [view_scans.py](view_scans.py) and [shot_viewer.py](viewer/shot_viewer.py); keep controlled data local.
- **Documentation and figures:** [build_figures.py](maintenance/build_figures.py), [documentation_sources.py](maintenance/documentation_sources.py), [build_technical_handbook.py](maintenance/build_technical_handbook.py), and [refresh_research_run.py](maintenance/refresh_research_run.py). See the [figure index](../info/plots/README.md).
- **Historical harness:** [harness.py](harness.py), [run_spatial_dynamics.py](experiments/spatial/run_spatial_dynamics.py), and [run_treatment_forecast.py](experiments/spatial/run_treatment_forecast.py). Follow the relevant historical protocol in Methods.

Individual tools provide their own options where applicable. This is an index, not a replacement for the recorded protocols.

## Directory layout

Only the four main entry points remain at the top level: `harness.py`, `patient_state.py`, `preprocess.py`, and `view_scans.py`.

| Folder | Contents |
|---|---|
| `data/` | Acquisition, preprocessing, restoration and geometry tools |
| `patient_state/` | Observation, forecast preparation, training, evaluation and timelines |
| `experiments/quantity/` | Closed quantity and information comparisons |
| `experiments/acquisition/` | Closed acquisition-conditioned JEPA study |
| `experiments/representation/` | Earlier representation studies and reporting |
| `experiments/spatial/` | Spatial dynamics, treatment and legacy anatomical runs |
| `maintenance/` | Repository checks, figures and generated documentation |
| `viewer/` | Viewer screenshot helper |

[Old-to-new path map](PATHS.json) translates historical command locations. Exact archived replays may require the original source layout and hashes; see the [artifact catalog](../info/ARTIFACTS.md#restoring-a-study).
