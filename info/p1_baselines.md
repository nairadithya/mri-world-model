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

These values are development diagnostics only. The within-unseen value is the
mean of the five locked fold-pooled scores; the transfer value is evaluated on
the historical final slice, which is now development-only. Re-run after the
Kaggle checkpoint and current-schema cache rebuild before interpreting any
JEPA comparison.
