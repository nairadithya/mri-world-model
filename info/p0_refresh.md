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

The current-cache persistence control uses the canonical LUMIERE target
embeddings and clean next-visit labels:

- 393 pairs; PD prevalence 0.639.
- Mean persistence error: PD 0.0054, SD 0.0076, PR 0.0058, CR 0.0032.
- Persistence-error AUC for next-visit PD: **0.4611**.

The LUMIERE forecast-baseline run is recorded in `info/p1_baselines.md` and
its full bootstrap output is in the ignored file
`outputs/p1_forecast_baselines_clean.json`. Surprise, lead-time, SAILOR
persistence, and cross-site analyses are being rerun against the refreshed
champion/caches before the final P0 scorecard is closed.
