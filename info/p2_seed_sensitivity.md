# P2.4 training-seed sensitivity

Question: does the held-out LUMIERE gain from the lesion-aware observation and
transition model persist under training-seed changes?

## Fixed protocol

The original seed-42 run and repeats use trainer commit `18d211d`, the same
65-patient encoder-training pool, the same 13-patient internal development
set, the same 23 mask-eligible locked LUMIERE patients, and the same training
settings (8 anatomy epochs; up to 200 forecast epochs; patience 20). Internal
development selects the checkpoint. The locked LUMIERE scores do not select
the seed or checkpoint. These are seed-sensitivity repeats on an already
examined evaluation cohort, not independent validation.

| Seed | Anatomy | Forecast | LUMIERE MAE (rel.) | LUMIERE Δ 95% CI | SAILOR MAE (rel.) | SAILOR Δ 95% CI | Exec. time |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 8 | 48 | 1.1833 (0.759) | [−0.638, −0.154] | 0.9872 (1.047) | [−0.082, +0.166] | 25m09s |
| 43 | 8 | 66 | 1.2125 (0.778) | [−0.608, −0.130] | 1.1132 (1.181) | [+0.047, +0.288] | 41m18s |
| 44 | 8 | 49 | 1.2375 (0.794) | [−0.552, −0.120] | 0.9997 (1.060) | [−0.058, +0.165] | 25m49s |

MAEs are patient-uniform log-volume MAE. Relative MAE is versus persistence;
Δ is learned minus persistence, with 10,000 patient bootstrap draws and fixed
seed 42. Cohort sizes are 23 locked LUMIERE patients and 27 SAILOR patients.
All three LUMIERE CIs exclude zero in favor of the learned model. SAILOR seed
43 is worse than persistence by paired bootstrap; seed 42 and 44 CIs include
zero. The transfer result is therefore seed-sensitive and does not meet the
external-site gate consistently.

## Seed 43 runtime status

Seeds 43 and 44 were submitted in parallel to private Tesla T4 kernels on
2026-09-28. Seed 44 completed in 25m49s. Seed 43's first attempt remained
`RUNNING` as of 2026-09-28 10:14 UTC, after 2h07m from its Kaggle
`lastRunTime` of 08:07:29 UTC. This was substantially longer than the
completed runs. The training code has a finite bound of 8 anatomy epochs plus
at most 200 forecast epochs, with early stopping after 20 non-improving
development epochs. The Kaggle push set a 32,400-second (9-hour) run limit.
Kaggle did not expose partial output, and its live log stream returned an HTTP
500 when queried, so epoch progress was unknown. The runtime discrepancy is
unexplained; do not attribute it to ordinary seed variation without
observing progress.

At 2026-09-28 10:20:55 UTC, version 2 of the same seed-43 kernel was pushed to
restart the run. Later the kernel disappeared from the owner's Kaggle listing;
the status endpoint returned a permission error and file lookup returned 403.
The local source remained intact. At 10:25:16 UTC, it was pushed again and
Kaggle created version 1, confirming the old kernel had been deleted. Kaggle
now reports the recreated kernel as `RUNNING`; this run has a 9-hour limit
(outer deadline 19:25:16 UTC). Check logs and outputs before interpreting
completion.

At 2026-09-28 11:03 UTC (37m56s after the recreated run started), Kaggle still
reported `RUNNING`. `kaggle kernels logs` returned no log content, so progress
is unknown. This has already exceeded the 25m09s and 25m49s runtimes of seeds
42 and 44; it remains inside the configured 9-hour limit.

At 2026-09-28 11:24 UTC (59m13s elapsed), it still reported `RUNNING`; the
kernel listing retained the same start time, and `kaggle kernels logs` again
returned no content.

The run completed normally. Its final log timestamp was 2,478.1 seconds
(41m18s of notebook execution), and its run note was emitted at 11:42 UTC,
about 1h17m after Kaggle's 10:25:16 UTC `lastRunTime` (submission-to-finish
includes startup delay). It trained 8 anatomy epochs and 66 forecast epochs,
then early-stopped. Locked LUMIERE MAE was 1.2125 versus persistence 1.5590
(relative MAE 0.7778; paired 95% CI on the difference [−0.608, −0.130]). The
run artifact records commit `18d211d`, seed 43, and the expected settings. Its
trained checkpoint and result JSON were downloaded under
`outputs/lesion-transition-seed43/`.

The unchanged SAILOR transfer evaluation used the same local evaluator and
27-patient cohort for each trained checkpoint. Seed 43 scored MAE 1.1132 vs
0.9430 persistence (relative 1.181; paired 95% CI on the difference [+0.047,
+0.288]). Seed 44 scored 0.9997 vs 0.9430 (relative 1.060; CI [−0.058,
+0.165]). Per-seed result JSON and separate encoder feature caches are saved
under `outputs/` and `checkpoints/`; SAILOR data remained local.

## Interpretation

All three seed runs beat persistence on locked LUMIERE, with paired CIs below
zero. This supports stability across these training seeds on this already
examined cohort; it is not independent validation. SAILOR transfer is not
stable across seeds: seed 43 is significantly worse than persistence, while
seed 42 and 44 are statistically indistinguishable from it under this
patient-level bootstrap. Do not claim external-site success. Report every
seed without selecting the best one.
