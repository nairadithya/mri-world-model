# SAILOR reprocess: SAILOR raw through the LUMIERE (BRAINIAC-contract) pipeline

Goal: build a SAILOR variant in LUMIERE's preprocessing contract to decide
whether the cross-site dynamics failure is pipeline mismatch or true site
difference. Everything else held fixed: same sessions, same pairs, same
frozen champion, same evals. Hypothesis: reprocessed SAILOR restores
LUMIERE-scale change and the transfer gate flips back. Success bar:
champion-vs-persistence on reprocessed SAILOR at/below the LUMIERE test
margin, with discriminative readouts holding (F1 ≈ 0.33–0.37, AUC ≈ 0.87).
If dynamics still lose while readouts hold, the residual is site
physiology/scanner, not processing — also a closing answer.

Deliberate non-goal: do NOT "improve" the pipeline for this run (no
intra-patient registration, no PLHM, no denoise). Harmonization means the
same artifacts and the same noise floor the model was calibrated on —
including per-visit registration jitter.

## 0. Inputs (all on disk except where noted)

- Raw sessions: `data/sailor/sailor_ebrains_pseud/rawdata/sub-XX/ses-YY/`
  (`t1w/t1wc/t2w/t2wflair.nii.gz`, selectively extracted from
  `rawdata.tar.bz2`; 1311 files, 7.5 GB).
- Session map: `derivatives/mni2009c-n-s/raw-mni-link.tsv` (raw ↔ MNI ids;
  243 consecutive linked MNI pairs = the eval set).
- Template: `data/templates/MNI152_T1_1mm.nii.gz` (same file as the
  LUMIERE run — non-negotiable, never omit `--template`).
- HD-BET available as `hd-bet` subprocess (as in the LUMIERE run);
  without it the pipeline falls back to a crude percentile mask — check
  logs if skull-strip stage looks instant.
- Reference numbers: raw-vs-MNI total pipeline damping ~3.9×
  (`info/ablations.md` A18; raw 0.673 ≈ LUMIERE 0.76).

## 1. Stage linked raws into LUMIERE layout (symlinks, free)

`scripts/preprocess.py` walks `<raw_root>/<Patient>/<visit>/{CT1,T1,T2,FLAIR}.nii.gz`
(`MODALITIES` constant), so translate names and session ids up front. Stage
ONLY linked sessions under their MNI ids — the derivatives eval sees MNI
sessions only, and unlinked pre-surgery sessions would change history
lengths (use `raw-mni-link.tsv`, skip `mni session == no`):

```
t1wc → CT1,  t1w → T1,  t2w → T2 (absent in a few sessions: skip),
t2wflair → FLAIR
staging/sub-XX/ses-YY/{CT1,T1,T2,FLAIR}.nii.gz  (symlink → rawdata/…)
```

Symlinks, not copies (rawdata stays the archive). Missing `t2w` sessions
stage with 3 links — same drop-and-mask handling as always; record how
many pairs this costs vs the derivatives eval (expect ≈243 minus
T2-missing; derivatives had base-T2 ≈99.6%).

## 2. Config (new file, do not edit `default.yaml` in place)

Copy `config/default.yaml` → `config/sailor_reprocess.yaml` with:

```yaml
data:
  root: data/sailor_reprocessed        # outputs (gitignored)
  raw_root: data/sailor_staging        # step-1 symlinks
```

Everything else (96³ target, seed, splits) stays identical.

## 3. Run (background, resume-safe)

```bash
setsid nohup .venv/bin/python scripts/preprocess.py \
  --config config/sailor_reprocess.yaml \
  --template data/templates/MNI152_T1_1mm.nii.gz \
  --workers 2 > logs_reprocess_sailor.log 2>&1 < /dev/null & disown
```

- `--workers 2` is the stability choice (same as the LUMIERE full run);
  ~1000 volumes × ~30 s ÷ 2 ≈ 4–5 h. Resume-safe: existing outputs are
  skipped, so restarts lose nothing; never run two invocations
  concurrently (they race on output paths).
- Ops hygiene from the LUMIERE run: after any killed run, hunt orphaned
  `multiprocessing.forkserver` workers before restarting; disk budget
  ≈ outputs (~4 GB) + `_tmp/` intermediates (~4×) — check `df` first.
- Spot-check like before: ~60-file nibabel pass (all finals 96³, finite;
  `16/16`-style per-patient verification on at least one full subject).

## 4. Adapt names for the SAILOR eval scripts

`src/data/sailor.py` expects MNI-derivative names (`T1c/T1/T2/Flair`),
preprocess writes LUMIERE names (`CT1/T1/T2/FLAIR`). After step 3,
symlink per session dir: `T1c.nii.gz → CT1.nii.gz`,
`Flair.nii.gz → FLAIR.nii.gz` (`T1`/`T2` already match).

## 5. Eval (needs a one-line root override)

`scripts/sailor_interval_eval.py` and `scripts/sailor_eval.py` hardcode
`SAILOR_ROOT`. Add a `--root` CLI flag to both (5 lines each, default =
current constant) rather than temp-editing constants, then:

```bash
.venv/bin/python scripts/sailor_interval_eval.py \
  --champion checkpoints/champion_0.0081.pt --root data/sailor_reprocessed
.venv/bin/python scripts/sailor_eval.py --encode --eval   # same --root
```

Report, same-space both sides, per gap bin + pooled: JEPA vs persistence
(the gate), transferred-probe F1, surprise AUC. Reuse
`scripts/split_gate.py`-style accounting; record pair counts next to the
derivatives-eval counts (any T2-missing delta must be visible, not silent).

## 6. Read the outcome

| Observation | Reading |
|---|---|
| Dynamics gate flips (beats persistence), readouts hold | Pipeline mismatch convicted; transfer story closed as processing + scale |
| Dynamics still lose, readouts hold | Residual is site physiology/scanner (denoise/PLHM excluded by construction — reprocessed never saw them); mismatch demoted to contributor |
| Readouts drop too | Something else broke (staging/mapping bug or genuine content shift) — stop, audit staging before any claim |
| T2-heavy recovery only | Modality-specific damping (consistent with A18's T2 tail) — route to per-sequence fusion work, not site work |

## 7. Aftermath (same conventions as every run)

Numbers → `info/plots/metrics.json` + `make_plots.py` (next plot #),
writeup → `info/ablations.md` (next A-ID, append-only), downloaded or
staged files labeled with (config, date, val). `data/sailor_reprocessed/`
and `data/sailor_staging/` stay gitignored like all data dirs.
