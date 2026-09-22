# Code staleness audit after the RANO-free P0/P1 reset

Date: 2026-09-22. “Stale” here means **not valid for the active P2 primary
claim without redesign**. It does not mean the file should be deleted: much of
it is retained to reproduce historical RANO experiments.

## Active and reusable

- `src/harness/data/anatomy_manifest.py`: source of truth for
  `lesion-state-v1` visits, pairs, provenance, geometry QA, and timing scope.
- `src/harness/analysis/anatomy_baselines.py`: frozen P1 floor and P2
  regression test.
- `src/harness/encode/interface.py`: current schema-2 representation cache;
  its image/state tensors remain usable, although its RANO label fields are
  no longer primary.
- `src/data/eval_protocol.py` and `info/eval_folds.json`: patient identities
  and folds remain reusable. The old RANO metrics attached to that protocol do
  not.
- BRAINIAC encoding, fusion, temporal transformer, EMA target, JEPA loss, and
  general cache/provenance infrastructure remain reusable components.
- Native preprocessing and recovered mask artifacts remain reusable, provided
  physical feature extraction reads the native affine rather than a resized
  identity-affine tensor.

## Stale primary-task/evaluation layer

These modules implement the superseded RANO classification endpoint and must
not appear in a P2 primary scorecard:

- `src/harness/data/tasks.py` (`rano4_*`, progression/response tasks).
- `src/harness/data/manifest.py`: even its `anatomy` rows require a valid RANO
  label and use `rano-clean-v1` in row IDs; use `anatomy_manifest.py` instead.
- `src/harness/analysis/forecast_baselines.py` and
  `temporal_baselines.py`: prospective RANO classifiers.
- `src/harness/analysis/tabular_baseline.py`: observed-follow-up RANO
  assessment, not forecasting.
- `src/harness/analysis/sailor.py` and `cross_site.py`: map the unverified
  SAILOR numeric RANO codebook and report cross-site macro-F1.
- `src/harness/analysis/surprise.py`, `leadtime.py`, `pred_latent.py`, and the
  RANO-readout portion of `concept.py`.
- `scripts/p0_scorecard.py` and the CLI default `rano4_forecast` evaluation.

The generic registry/evaluator framework can be extended with continuous
anatomy tasks; its existing task registrations and classification reporting
are historical.

## Stale supervised training paths

- `src/model/heads.py` and the `rano_heads` branch in
  `src/model/jepa_model.py` train PD/SD/PR/CR auxiliaries.
- `src/harness/train/head.py`, `lora.py`, `cnn2d.py`, and `cnn3d.py` optimize
  RANO classification.
- `config/default.yaml::aux` and the corresponding CLI `--aux-lambda` control
  the obsolete RANO auxiliary loss.

These should be disabled for P2, then replaced—not silently repurposed—with a
versioned continuous lesion-state/residual head and explicit loss weights.
Checkpoint loading must remain backward compatible with historical heads.

## Stale or unsafe data semantics

- `src/data/sailor.py::SAILOR_RANO_TO_ACTION` encodes the tentative numeric
  mapping. Keep the raw code for provenance/viewing, but do not transform it
  into a primary label.
- `src/data/dataset.py::RANO_ACTION_MAP` is still used as an `actions` channel.
  A response assessment is not a treatment action and must not condition a P2
  causal or treatment-aware model.
- `src/model/jepa_model.py::_dynamics_loss` falls back from missing treatment
  to RANO `actions`. This is unsafe for P2. Missing treatment should be an
  explicit unknown category; response must not substitute for intervention.
- `src/data/collate.py` always materializes `actions` and comments still frame
  treatment as a preferred alternative to RANO. P2 needs explicit, separately
  named response and treatment channels.
- The exact-gap horizon and velocity-field paths consume `time_deltas` as
  days. They remain useful experimental machinery, but current LUMIERE week
  bins and partly estimated SAILOR intervals do not justify primary fixed-time
  or rate claims.

## Feature code that is useful but must be rewritten

`src/harness/encode/radiomics.py` computes valuable volume/nadir/change
features, but it is an observed-pair RANO feature builder: it includes target
visit measurements in `X`, labels rows with future RANO, and contains separate
cohort loaders rather than consuming the audited P0 manifest. Its physical
mask-volume helper is reusable; its row contract is not. P2 needs a new
history-only lesion feature builder keyed by immutable anatomy row IDs.

Similarly, `src/harness/encode/interface.py` correctly caches representations
and log-volumes, but reports RANO-labelled visit counts and carries labels as
first-class fields. P2 should add an anatomy-oriented cache/view rather than
mutating old cache semantics in place.

## Documentation and interface drift

- `PROPOSAL.md` still describes RANO as an action/reward and SAILOR as the
  primary held-out RANO evaluation set.
- `info/eval_protocol.md`, `p0_audit.md`, `p0_refresh.md`, `p1_baselines.md`,
  `p1_sota_audit.md`, and much of `explainer.html` describe the historical
  classification program. They are retained evidence, not the current plan.
- `scripts/harness.py` examples and `src/harness/cli.py` usage/defaults still
  foreground `rano4_forecast`.

The active documentation chain is now `anatomy_harmonization.md` →
`p1_anatomy_baselines.md` → `p2_execution.md`.

## Recommended cleanup order

1. **In progress:** anatomy-first CLI and audited checkpoint compatibility are
   implemented (`info/harness_v2.md`); old commands live under `legacy`.
2. Add the P2 lesion feature cache and continuous task/evaluator without
   deleting historical RANO paths.
3. Separate `response` from `treatment` throughout dataset/collate/model APIs;
   remove the response-as-action fallback from new training.
4. Add a versioned residual lesion head and lesion-aware objective.
5. After reproducing archived results, remove the temporary deprecated
   top-level aliases; keep RANO-only commands under the legacy namespace.
