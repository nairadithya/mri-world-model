# P0 current-schema refresh

The survival-free retrain and cache refresh completed on 2026-09-21.

- Champion: `checkpoints/p0_clean_jepa/best.pt`, epoch 19, validation loss
  `0.00915965`.
- LUMIERE probe cache: schema 2, 91 patients, 393 RANO-labelled visits,
  `clinical_schema=survival_free_v1`.
- LUMIERE interface cache: schema 2, 91 patients,
  `clinical_schema=survival_free_v1`.
- Radiomics features: schema 2, 79 usable patients.
- SAILOR cache: schema 2, 27 subjects, 270 sessions,
  `clinical_schema=survival_free_v1`.

The refreshed P0 controls are complete:

- LUMIERE surprise-error AUC for next-visit PD: **0.4971** (393 pairs).
- LUMIERE persistence-error AUC: **0.4611** (393 pairs; PD prevalence 0.639).
- Incident-PD lead-time AUCs (JEPA / persistence): k=1 **0.4971 / 0.4611**
  (393 pairs), k=2 **0.4289 / 0.5932** (124), k=3 **0.4907 / 0.5752** (60).
- SAILOR persistence-error AUC: **0.6600** (240 pairs; PD prevalence 0.312).
- SAILOR zero-shot/K-shot JEPA macro-F1: **0.123 / 0.252 / 0.241 / 0.253 /
  0.264 / 0.263** for zero-shot/K=3/5/10/15/20. CNN control:
  **0.261 / 0.228 / 0.225 / 0.261 / 0.264 / 0.227**.

The full execution log is `logs/p0_refreshed_analyses.log`. The LUMIERE
forecast-baseline run is recorded in `info/p1_baselines.md` and its full
bootstrap output is in the ignored file
`outputs/p1_forecast_baselines_clean.json`.

## P0 scorecard closure (2026-09-21)

`scripts/p0_scorecard.py` now regenerates the complete local scorecard from
the current-schema probe cache. It reports pooled and patient-uniform metrics,
10,000-resample patient-cluster CIs, prespecified paired comparisons, and a
3-seed × 3-width readout sensitivity grid in `outputs/p0_scorecard.json`.

For the primary locked unseen forecast comparison, `states_forecast` versus
`fused` gives macro-F1 **0.2718 vs 0.2095 pooled** and **0.2247 vs 0.1428
patient-uniform**, with paired difference **+0.0623 [0.0003, 0.1273]**.
Development comparisons are now frozen to the MLP readout with hidden width
256, seed 42, and no feature standardization. The remaining P0 blockers are
independent SAILOR label confirmation, aligned RANO/treatment timing metadata,
and a genuinely new external or hidden holdout.
