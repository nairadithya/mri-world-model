# Script index

Use the study-specific protocols in [Methods](../info/METHODS.md) and the [closure record](../info/TODO.md) before any separately authorized reproduction. Numerical workloads require an authorized external compute host with the environment and necessary data restored; mosaic-gpu is retiring. Protect controlled-access data and the unopened reserved cohort.

- **Preparation:** [preprocess.py](preprocess.py), [prepare_patient_state_mu.py](prepare_patient_state_mu.py), [prepare_cavity_supervision.py](prepare_cavity_supervision.py), [prepare_structured_forecasting.py](prepare_structured_forecasting.py), and [prepare_r68_episodes.py](prepare_r68_episodes.py).
- **Observation:** [patient_state.py](patient_state.py), [train_patient_observation.sh](train_patient_observation.sh), and [train_cavity_observation.sh](train_cavity_observation.sh).
- **Forecasting:** [train_patient_forecasting.sh](train_patient_forecasting.sh), [train_structured_forecasting.sh](train_structured_forecasting.sh), and [train_cavity_forecasting.sh](train_cavity_forecasting.sh).
- **Quantity:** [fit_linear_quantity.py](fit_linear_quantity.py), [predict_linear_quantity.py](predict_linear_quantity.py), and [train_amount_first.py](train_amount_first.py).
- **Acquisition:** [prepare_acquisition_study.py](prepare_acquisition_study.py), [train_acquisition_jepa.py](train_acquisition_jepa.py), and [evaluate_acquisition_jepa.py](evaluate_acquisition_jepa.py). The [historical data access guide](DATA_ACCESS.md) records earlier download instructions.
- **Evaluation:** [evaluate_patient_state.py](evaluate_patient_state.py), [evaluate_patient_forecasts.py](evaluate_patient_forecasts.py), [evaluate_structured_forecasts.py](evaluate_structured_forecasts.py), and [evaluate_cavity_forecasts.py](evaluate_cavity_forecasts.py).
- **Viewers:** [view_scans.py](view_scans.py) and [shot_viewer.py](shot_viewer.py); keep controlled data local.
- **Documentation and figures:** [build_figures.py](build_figures.py), [documentation_sources.py](documentation_sources.py), [build_technical_handbook.py](build_technical_handbook.py), and [refresh_research_run.py](refresh_research_run.py). See the [figure index](../info/plots/README.md).
- **Historical harness:** [harness.py](harness.py), [run_spatial_dynamics.py](run_spatial_dynamics.py), and [run_treatment_forecast.py](run_treatment_forecast.py). Follow the relevant historical protocol in Methods.

Individual tools provide their own options where applicable. This is an index, not a replacement for the recorded protocols.
