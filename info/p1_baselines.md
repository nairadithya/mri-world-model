# P1 prospective baseline suite

Status: implementation complete; authoritative run pending the survival-free
current-schema cache. The suite is `src/harness/analysis/forecast_baselines.py`
and is registered as:

```bash
python scripts/harness.py run forecast-baselines \
  --cache checkpoints/p0_interface_cache.pt \
  --out results/p1_forecast_baselines.json
```

Every row is a clean-labelled `t -> t+1` forecast row. Features use only the
source visit: last observed RANO, smoothed RANO transition counts, current
image features, current volumes, or past volume trend. The clinical baseline
is refused unless cache provenance says `clinical_schema=survival_free_v1`.

## Current survival-free LUMIERE run

Run on `checkpoints/p0_interface_cache.pt` (schema 2, provenance
`clinical_schema=survival_free_v1`; 79 patients contribute forecast rows):

| method | within-unseen CV | transfer -> historical final |
|---|---:|---:|
| majority | 0.1952 | 0.1975 |
| last RANO | 0.2566 | 0.3868 |
| transition | 0.1952 | 0.3212 |
| clinical | 0.1625 | 0.1726 |
| current image | 0.2388 | 0.2619 |
| current volume | 0.2180 | 0.2509 |
| past volume trend | 0.1597 | 0.2725 |

Bootstrap intervals and prevalence are preserved in
`outputs/p1_forecast_baselines_clean.json` (gitignored). These are still
assessment/development diagnostics: the historical-final slice is not an
untouched test, and the JEPA forecast readout must be evaluated on the same
current cache before claiming an advantage.

The temporal baseline suite is implemented as
`src/harness/analysis/temporal_baselines.py` and registered as
`run temporal-baselines`. It evaluates mean-history pooling, a last-visit MLP,
a GRU, order-shuffled GRU, and truncated-history GRU on the same 393 forecast
rows. A 3-step smoke run passed; the full prespecified run is queued behind the
current P0 refresh analysis.

## Exploratory legacy-cache run

Run on `checkpoints/interface_cache.pt` (schema 1, 79 patients; not citable as
the final result because the cache predates the survival-free contract):

| method | within-unseen CV | transfer -> historical final |
|---|---:|---:|
| majority | 0.1952 | 0.1975 |
| last RANO | 0.2566 | 0.3868 |
| transition | 0.1952 | 0.3212 |
| current image | 0.2439 | 0.2953 |
| current volume | 0.2180 | 0.2509 |
| past volume trend | 0.1597 | 0.2725 |
| clinical | **blocked** | **blocked** |

These legacy values are development diagnostics only. The within-unseen value
is the mean of the five locked fold-pooled scores; the transfer value is
evaluated on the historical final slice, which is now development-only.
