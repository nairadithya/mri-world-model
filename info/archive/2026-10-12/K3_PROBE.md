# K3_PROBE — adversarial review of every conclusion in `info/`

Date: 2026-09-09. Method: read all of `info/` (`decisions.md`, `ablations.md`,
`pilot.md`, `results.md`, `preprocessing.md`, `directions.md`,
`frontier_lumiere.md`), then verified each load-bearing claim against the
actual code and data on disk, and re-ran the key numbers from the stored
caches. No subagents. New measurements are marked **[NEW]**; everything else
is a code/data check of an existing claim.

Reading guide: findings are ordered by severity within each part. Each item
gives the claim, what was checked (`file:line` or command), the verdict, and
the cheapest falsifying/confirming follow-up. Part 6 records what survived
intact — several conclusions got *stronger* under probe. Part 7 ranks next
steps.

**One-line summary:** the headline conclusions survive in *direction* but are
thinner and less well-identified than stated. Two genuinely new problems were
found: (1) the SAILOR eval inputs contain 19 empty-T2 + 2 empty-T1 sessions
that the adapter counts as present (10.3% of eval pairs touch one), and
(2) the cross-site dynamics failure decomposes, with zero training, into a
~5× scalar gain overshoot plus an uninformative predicted direction — a
sharper localization than anything in `info/`. The in-domain gate, re-scored
same-space, passes on means but loses the patient majority in every split
(41/91 wins, including 26/65 on its own training patients).

---

## Part 1 — Preprocessing alignment (the main suspicion)

### K3-1 [NEW, significant]. Phantom modalities in the SAILOR eval inputs.
`src/data/sailor.py::_has_any_image` (line 121–122) and `_image_path`
(111–119) check file **existence** only. `src/data/collate.py` (44–45) marks
any loadable file `mri_mask=True`. Nothing anywhere checks file **content**.

Scanning all 270 SAILOR sessions' base variants (central-slab nonzero
fraction): **19 T2 and 2 T1 volumes are effectively empty** (nzfrac < 0.05,
most exactly 0), concentrated in sub-23 (10 empty T2 sessions — its entire
T2 series), sub-01, sub-02, sub-24, sub-20, sub-27. The `-icor` variants of
the same sessions are NaN-filled or also empty, so this is genuinely missing
data at source — but the adapter feeds the all-zero base file to the encoder
with mask True. An all-zero input produces a constant "empty" embedding that
enters the per-visit modality mean on both the online and EMA-target sides
(`jepa_model.py::encode_visits`, `encode_target_visit` — no content guard
either side).

Impact, measured: 25/243 consecutive eval pairs (10.3%, across 6 subjects)
touch a phantom-modality visit. sub-23's cached vision norms sit at 24.8 vs
the 26.1 cohort mean — the phantom drags its latents systematically. Because
the constant pulls both endpoints of a pair toward the same point, affected
pairs' persistence errors are biased *downward*, i.e. this artifact
strengthens the "persistence wins on SAILOR" headline it contaminates.
LUMIERE is clean (A20 QA gate exits 0), so the corruption is SAILOR-only and
asymmetric. The SAILOR derivatives tree was never QA'd — `preprocess_qa.py`
only scans 96³ preprocessed roots.

Fix (hours): treat all-zero/near-zero volumes as absent in `_image_path` or
collate (one-line content check), rebuild SAILOR caches, re-run the three
arm tables. Expect small pooled shifts but large per-subject corrections
(sub-23). Until then, every SAILOR number in `info/` carries ~10% corrupted
pairs.

### K3-2 [significant]. The arm headline tables violate the D28 same-space rule.
D28/A8-addendum mandates same-space both sides and asserts transfer results
were "already same-space". Code check:

- `scripts/sailor_eval.py::eval_all` (a), lines 90/99–101: JEPA from
  `model(batch)` (EMA-target space) vs persistence from
  `PersistenceBaseline(model.projector)` over `model.encode_visits` (**online**
  projector over the **online** backbone) — **mixed space**, exactly the
  original-A8 error. The repo-root logs prove the published arm numbers came
  from this path (`logs_sailor_reprocessed_eval.log`:
  `(a) SAILOR pairs: n=243 JEPA=0.0245 persist=0.0037`).
- `scripts/sailor_interval_eval.py`, lines 69–79: persistence from
  `model.encode_target_visit` — EMA space, same as JEPA. **Same space.**

So A12's (0.0290/0.0056), A19's (0.0245/0.0037) and A21's (0.0339/0.0059)
headline rows are mixed-space; only the per-bin rows are same-space. On
SAILOR-derivatives data the online/EMA persistence ratio is ~1.4× (0.0056 vs
~0.0039 EMA, cf. A16's table), so the "~5×" headline is overstated by roughly
that factor — direction unchanged, verdict unchanged, but the rule the repo
imposes on itself is broken in the very tables used to compare preprocessing
arms, and A16's persistence row (0.0039, EMA) is silently a different space
from A12's (0.0056, online). Fix: route the (a) comparison through cached
EMA endpoints (as `split_gate.py` does) and re-issue the three arm rows.

### K3-3 [NEW, major]. The transfer failure = scalar gain overshoot ×
uninformative direction. Measured, zero training.
Using the champion predictor over cached EMA states/targets
(`horizon_cache.pt`, `field_cache.pt` — harness check: reproduces A12/A16
numbers exactly):

| cohort | pred ‖Δ‖ | true ‖Δ‖ | ratio | dir-cos(pred Δ, true Δ) |
|---|---|---|---|---|
| LUMIERE (n=547) | 6.80 | 2.85 | **2.38×** | 0.19 |
| SAILOR (n=243) | 9.31 | 1.89 | **4.92×** | 0.105 |

The champion does not merely "over-predict change" (A12/A14 phrasing) — it
predicts large, nearly *orthogonal* displacements (dir-cos ~0.1), which is
why the error sits flat ~0.03 at all gaps. Counterfactual that isolates the
two faults (keep predicted direction, rescale to the true magnitude —
oracle gain, diagnostic only):

| cohort | orig err | persist | magnitude-fixed |
|---|---|---|---|
| LUMIERE | 0.0080 | 0.0069 | **0.0047** |
| SAILOR | 0.0290 | 0.0039 | **0.0041** |

On SAILOR, fixing *only* the scalar gain recovers 86% of the gap
(0.0290 → 0.0041 ≈ persistence). The residual ≈ persistence means the
predicted *direction* is worth ~nothing cross-site. In-domain, direction is
genuinely informative (gain-fixed 0.0047 < persistence 0.0069) but magnitude
still overshoots 2.4×. So: the predictor learned an in-domain-useful
direction plus a LUMIERE-scale magnitude prior (regression-to-the-mean under
cosine loss); cross-site the direction dies and the magnitude prior fires
anyway. This reframes A15 ("dynamics scale", correct but coarse — "scale"
turns out to be *literally a scalar gain*, and A15's field refit beating
persistence 0.0036-vs-0.0040 must come from its directional half, since no
gain alone can beat persistence), and it reframes the damping debate: damped
inputs (smaller true ‖Δ‖) and a fixed-magnitude predictor are the *same*
failure viewed from image vs latent space (see K3-8). A per-site/per-gap
gain calibration on the frozen head — one scalar, no training of directions
— is the obvious experiment the program skipped on its way to heavier
machinery (CORAL, fields, reprocessing).

### K3-4 [moderate]. CORAL "covariance shift" over-attributes — the mean was never isolated.
A16(2)'s map (`freeze_battery.py::coral_map`, 110–123) aligns the mean
*and* the covariance (`(x − μ_s)@A + μ_t`), but the inference credits
"second-order statistics" for the full 0.0290 → 0.0119 recovery. A
mean-only translation ablation was never run — yet A16(1)'s own
separability result (all 27 SAILOR per-subject logits positive, 3.7–15.7)
argues the shift is predominantly a *uniform translation*, which an MLP head
trained in LUMIERE's neighborhood would be maximally sensitive to. The
"2/3 is covariance shift" number (arithmetically correct: 0.0171/0.0251)
cannot distinguish "covariance fixed it" from "translation fixed it".
Cheap decider: score the frozen head after mean-translation only. (Also
noted in passing: the reported separability acc/AUC 0.956/0.9998 are
*in-sample* — logistic trained and scored on all rows; the claim survives
CV — my ridge row-CV gives acc 0.969/AUC 1.0 — but the quoted numbers are
the overfit-flavored ones.)

### K3-5 [moderate]. Same code ≠ same behavior: three unquantified residuals inside the "harmonized" reprocess.
(a) **N4 Otsu asymmetry.** `n4_bias.py` fits the Otsu mask on full-head
SAILOR raws (skull included) vs source-stripped LUMIERE raws (verified:
LUMIERE raw CT1 nzfrac 0.098, min exactly 0.0 — A19's "source-stripped"
claim checks out). Different foreground ⇒ different bias fields ⇒ different
intensity correction, from identical code. Never compared (e.g. bias-field
magnitude distributions per cohort).
(b) **Full-head → brain-template registration.** The reprocess registered
full-head SAILOR raws against the brain-only MNI template with Mattes MI;
skull edges drive part of the alignment (failure mode observed:
sub-07/ses-03 hard-fail + 2 near-empty regs, correctly flagged as
negligible pooled). No registration-quality check on the successes.
(c) **Transforms are discarded** (`brainiac_contract.py:61` keeps only the
resampled image) and **MI sampling is unseeded**
(`registration.py:47–49`, 5% RANDOM, no `SetMetricSamplingSeed`), so
per-visit registration jitter — the mechanism behind G-data-6 and part of
the K3-3 magnitude prior — is unmeasured *by construction* and
non-reproducible run-to-run. Recording per-visit transform magnitudes is a
one-line change that would turn the jitter theory from narrative into a
number.

### K3-6 [low–moderate]. D4's template claim overstates; affine-vs-rigid never harmonized.
D4 says the TemplateFlow template removes "one cross-dataset confound".
Checked: our template is 193×229×193 (MNI152NLin2009c**Asym** geometry);
SAILOR's is ICBM 2009c nonlinear **symmetric** (descriptor). Same family,
*different templates* (asymmetric vs symmetrized anatomy). Further, SAILOR
brains are affine-scaled to template size while LUMIERE brains keep native
scale (rigid-only) — apparent brain/tumor size differs systematically
cross-site, on top of which the encoder was trained. Neither difference was
quantified (e.g. brain-volume distributions in the 96³ frame per cohort).
Unlikely to explain a 5× error ratio alone; belongs in the honest-residual
list, not out of it.

### K3-7 [low]. A19's staging audit rests on a thin correlation margin.
"Mapping vindicated" cites T2 matched-session corr 0.58 vs crossed 0.46
(Δ0.12) with no null calibration. A same-style null I computed on LUMIERE
T2 center-box correlations shows this metric is noisy at small samples
(cross-patient ≈ −0.10, same-patient different-visit ≈ −0.01 on two pairs
— i.e. uncalibrated). The audit's exact pair/gap/label accounting
(med-76 gaps, PD-rate 0.312 reproduced) is the strong half of the
vindication; the correlation half should have carried a null. Verdict on
A19's conclusion unchanged, confidence interval wider than presented.

### K3-8 [measured — result below]. The missing cell: drift magnitude per preprocessing arm.
A19/A21 report *errors* per arm but never the *true change magnitudes*
(latent ‖Δ‖ or image drift) for derivatives vs skull-in vs skull-out trees.
That number decides whether the reprocess restored LUMIERE-scale change
(damping = pipeline) or not (damping = acquisition) — and, via K3-3,
whether the residual gain mismatch is miscalibration or input damping.
Measured 2026-09-09 (matched consecutive pairs, image-space mean|Δ| on 96³,
z-scored, same metric shape as A18; 60 SAILOR pairs × 4 slots = 240
pair-slots per arm + 60 LUMIERE CT1 pairs):

| arm | n | median | p10/p90 |
|---|---|---|---|
| SAILOR derivatives | 240 | **0.19** | 0.11/0.48 |
| SAILOR skull-out reprocess (LUMIERE contract) | 240 | **0.75** | 0.48/1.02 |
| LUMIERE reference | 60 | **0.72** | 0.45/0.87 |

The reprocess **fully restores LUMIERE-scale image change** (0.75 ≈ 0.72;
derivatives reproduce A18's ~0.17–0.24). Pipeline-side damping confirmed at
image level. But pair it with the *latent* drift per arm (same-space
persistence, 61–180d bin): derivatives 0.0035 → skull-out 0.0039 →
LUMIERE ~0.005–0.006 (A16(8): 0.0058 at 90–180d). **Image drift went 4×;
latent drift barely moved.** The restored component is essentially
encoder-invisible — most plausibly registration jitter (the reprocess doc
deliberately restored "the same noise floor the model was calibrated on",
and a ViT substantially absorbs 1-voxel rigid shifts at patch level),
not biological change. Two consequences: (i) it explains *why* A21's gate
didn't budge despite LUMIERE-scale images — the model's magnitude prior
needs biological-scale latent change, and jitter is not that; (ii) it
strengthens A21's closure in one direction (no pipeline-texture fix was
ever going to work) while localizing the true residual to
*biological trajectory quietness (phase physiology) × encoder
under-response to SAILOR texture* — i.e. a site×encoder interaction, which
is exactly what K3-3's gain + direction decomposition measures from the
other end. The "damping" and "scale" stories were the same failure viewed
from image vs latent space all along.

### K3-9 [checked, no objection]. uint8 exoneration is correctly scoped
("ruled out as necessary, not proven zero-effect", A21 addendum). Agreed.

### K3-10 [low]. LUMIERE raw's registration state was never established.
G-data-6's "per-visit independent registration turns head-position noise
into change" assumes LUMIERE raws arrive unaligned. If the Figshare raws are
already longitudinally co-registered at source (unverified — nothing in the
repo checks), the in-domain jitter floor is smaller than assumed and
SAILOR's intra-patient registration is *not* a differentiator. One cheap
check: translation/rotation magnitudes between consecutive LUMIERE raws
(pre-registration) vs post-registration residuals.

---

## Part 2 — The in-domain gate is thinner than stated

### K3-11 [major]. Same-space, the model loses the patient majority in every split — including train.
Recomputed from `horizon_cache.pt` with the champion predictor (matches the
A8 addendum to 4 decimals):

| split | pooled JEPA/persist | patient-uniform | patient wins |
|---|---|---|---|
| train (65) | 0.0082 / 0.0065 loses | 0.0083 / 0.0071 loses | **26/65 (40%)** |
| val (13) | 0.0078 / 0.0075 loses | 0.0081 / 0.0086 wins | 7/13 |
| test (13) | 0.0070 / 0.0088 wins | 0.0074 / 0.0160 wins | 8/13 |

Three consequences. (i) Pooled-train loses 26%: on its own training data,
average pair, the model is worse than copying — the regression-to-the-mean
tax on static pairs (cf. G7). (ii) The published "82/91 wins" (A8) was a
mixed-space artifact; same-space overall wins are **41/91 (45%)**, never
stated anywhere. (iii) The surviving gate is: test means (both
aggregations) + overall patient-uniform by Δ0.0005 (0.0081 vs 0.0086) — a
margin with no reported uncertainty, on the exact aggregation the val
selection metric optimizes (G7). The test patient-uniform persist (0.0160)
is visibly inflated by few-pair dynamic patients. The honest headline is
"beats persistence on held-out means, loses the patient majority
everywhere, never beats it pooled-on-train" — a mean-effect claim, not the
"exactly the right pattern" majority claim A8 makes. Required: bootstrap
CIs on all six cells + the per-patient table before any writeup cites a
ratio.

### K3-12 [low]. The rank monitor is vacuous at batch 1; std does all the work.
`collapse_metrics` (`jepa.py:141–149`) is computed per batch; at hero batch
size 1 that is one patient (≤20 pairs) in 768-d, where rank ~1–3 is
automatic. "Rank 1.7–2.5 healthy" cannot distinguish health from
small-batch rank deficiency — only total collapse (rank 1.0). Run 1's
collapse was caught by std (0.0024), correctly. The (b)-criterion of the
triple gate therefore rests on std alone in every hero leg. Practice is
fine (std is the informative one); the criterion as written over-credits
rank.

### K3-13 [moderate]. LoRA's contribution was never isolated.
A5 proves the *backbone* didn't move (fresh-vs-trained drift identical to 6
decimals) — so all learned change-sensitivity lives in the projector +
temporal stack (a learned metric over frozen features), and D6's "LoRA lets
change-sensitivity be learned" is unproven: LoRA movement was "negligible"
by the same measurement. The open question in the A8 addendum ("how much of
the online-vs-EMA spread is LoRA change-amplification") has a cheap answer
that was never run: frozen-backbone + trained projector/temporal vs LoRA,
same protocol. If the gap is ~zero, 1.18M LoRA params (and the D6
narrative) are load-bearing in name only.

### K3-14 [moderate]. Best-tracking on n=69 val pairs invites winner's curse — twice.
The champion (min over 30 epochs, val 0.0081) and the legC2/legE "0.0077 <
champion" exploit models (min over 20/10 epochs, Δ0.0004 ≈ 5%) are selected
on 69 val pairs / 13 patients with no uncertainty on the selection metric
itself. I verified the R11 trajectories from the raw logs (dose-response
accum-8 < accum-4 < accum-1 is real — mechanism evidence stands), but the
*models* banked by the exploit protocol may be val-noise selections: C2's
ep3/ep5 0.0077 vs ep1 0.0078 differs by 0.0001. "Re-gate probes pending" is
correctly flagged; add: bootstrap the val metric before crowning any
best.pt, and report test (not val) for the 0.0077 legs.

### K3-15 [low, structural]. G7 cuts both ways.
Patient-uniform training (each patient = one gradient step of its own mean
loss at batch 1) is also the selection metric — consistent — but it means
the model is *optimized* for the one aggregation it passes overall, while
pooled-all (the natural deployment average) loses 0.0080 vs 0.0069. If
deployment cares about pairs, the gate fails; if it cares about patients,
it thinly passes. The docs should state which claim is being made instead
of letting "beats persistence" float between aggregations.

---

## Part 3 — Probes, surprise, and cross-site readouts

### K3-16 [moderate]. CV 0.33 mixes encoder-train patients into probe-test folds.
Confirmed in code (`probe_rano.py::run_cv`, 209–232: reshuffles all 91
patients; encoder saw 65). G11 acknowledges it; the scorecard's "honest
headline" doesn't carry the caveat, and the transfer comparison ("0.37 ≈
0.33") leans on it. True-unseen CV over the 26 non-train patients was never
reported. Direction of bias: mildly optimistic for 0.33 (encoder trained
next-latent prediction on 65/91 probe Dixon... on probe patients — indirect
leakage, small but nonzero). Run it; it costs CPU minutes.

### K3-17 [moderate]. The SAILOR codebook is load-bearing and untested.
{1:PD, 2:SD, 3:PR, 5:CR} is empirical (volume deltas, n small) with 3-vs-5
"tentative" — yet every SAILOR RANO number (transfer F1 0.37, surprise-AUC
0.87/0.89, A15 AUCs, within-phase ties) conditions on it. The R4
swap-and-rescore sensitivity was never run. Worse, the transfer F1 is
unstable across preprocessing arms: 0.37 → 0.25 → 0.32 (A12/A19/A21) — a
±0.06 swing from input processing alone, on a number quoted to two
decimals against CV 0.33±0.060. "0.37 ≈ 0.33, representation transfers" is
two noisy numbers touching, not an equality. Report the transfer probe with
a CI (bootstrap over the 240 rows) and the codebook swap before citing.

### K3-18 [moderate]. A21's ceiling anomaly undermines its own transfer number.
SAILOR-fit (train=test) F1 0.30 sits *below* transfer 0.32 on the skull-out
cache — memorization failing where transfer succeeds smells like the fit
path, as the doc honestly flags ("needs a look before anyone cites it").
But the doc then cites the transfer half anyway ("readouts recovered 0.25
→ 0.32", "transfer story is closed"). Both numbers come from the same
`fit_linear` path on the same cache; if the path is suspect, neither is
citable. Resolve the anomaly first (prime suspect: degenerate/constant
feature columns or label misalignment in `sailor_betfix_cache.pt`, whose
provenance is filename-only per K3-31).

### K3-19 [low, agreed-with-caveat]. Surprise ≈ change-detection, fully conceded — keep it conceded.
JEPA-AUC vs persistence-error AUC: 0.7677/0.7521 in-domain, 0.87/0.86
derivatives cross-site, 0.89 skull-out (persistence control unreported on
that arm — fill it before citing 0.89 anywhere). The docs already conclude "change-detection, not a
JEPA-specific signal". The sole JEPA-specific residue (PR/CR surprise most,
n=20–27) is too small to carry any claim — ensure the writeup never lets
"surprise-AUC 0.87" appear without its persistence control. (Technical nit:
`auc_mann_whitney` claims tie-averaging but `argsort` assigns distinct
ranks; immaterial for continuous errors.)

### K3-20 [moderate]. A15's exact tie smells like a dead input channel.
cond ≡ uncond to ~4 decimals on held-out error in all 5 folds *and* within
every phase (A16.7). Two explanations fit: (i) phase truly redundant given
the 1152-d history state (the doc's reading); (ii) the phase embedding never
receives/uses gradient under the fixed 400-epoch, no-val-split protocol
(weight decay 0.1, same budget both arms), in which case *any* channel —
informative or not — would tie. The disambiguators were never run: phase
embedding gradient/weight norms, a shuffle-phase control (dismissed as
"moot", but it separates (i) — shuffle should hurt — from (ii) — shuffle
ties), or a phase-from-state probe (direct test of the redundancy story).
Hedge in A15 ("no operational gain at N=27") is correctly narrow; A16(7)'s
"treatment is redundant given history, homogeneously" is broader than the
evidence. (Correction, 2026-09-09: an earlier draft of this note wrongly
claimed the folds were unshuffled — `train_field.py:66–70` shuffles with
seed 42 before `i % 5`. The fold assignment is honest; withdrawn.)

### K3-21 [checked, CLEARED]. Volume readout is vision-driven, not OS-driven.
My one new ablation (ridge λ-CV, hero splits, n=73, reproduces A11 exactly):
fused R² 0.154, **vision-only 0.108**, clinical-only 0.002. The suspicion
that OS in the clinical branch inflates the size signal is dead — the
clinical branch carries no volume information and the vision latent holds
the (weak, diffuse, λ=1000-shrunk) signal alone. A11's "latent holds weak
size signal" is *strengthened* by this probe; add vision-only to the table.
(Remaining open as stated: enhancing-core and growth-rate framings;
`volume_probe.py` loads `Enhancing_Core` but the loop only runs "total".)

### K3-22 [low]. Lead-time k=2 is suggestive, correctly labeled — one extra caveat.
JEPA 0.699 vs persist 0.810 at k=2 (n=124, SE~0.06): ~1.8 SE, suggestive,
not conclusive — fair as written. Add: the incident-PD definition requires
clean intermediate labels, which *selects* stable trajectories; the k=2
pool is therefore enriched for cases where raw volatility (persistence)
has signal by construction. RQ1-as-stated answering NO stands; no further
early-warning work justified (agreed).

### K3-23 [low]. A15's fixed budget + no within-fold val split (disclosed).
Both variants share the protocol so the tie comparison is fair to first
order; the hidden-128 robustness rerun is the right control and was done.
No objection beyond K3-20.

---

## Part 4 — Time, gaps, labels, and internal contradictions

### K3-24 [moderate]. MNI gap values "may be inaccurate" — never verified.
The descriptor states MNI-version intervals were manually extracted, edited,
and "may be inaccurate", while source/raw intervals from exam dates are
accurate. Every gap-stratified conclusion (regime exoneration, gap bins,
gap head, horizon conditioning cross-site, K3-3's gap-independence of the
flat error) rests on `intervals-days.txt`. No cross-check against DICOM
exam dates (sourcedata, unextracted) was ever run. Concrete protocol: read
AcquisitionDate/SeriesDate from sourcedata DICOMs for linked sessions,
recompute gaps, report agreement (median abs deviation) — half a day, and
it either retires or detonates the quantitative gap claims.

### K3-25 [low–moderate]. LUMIERE time deltas are nominal weeks, not dates.
`parse_week_to_days` (`dataset.py:39–46`): weeks×7, same-week duplicates
ordered by a +1-day suffix hack. G15 proves weeks carry multiple distinct
real dates (47 patient-weeks, concentrated week-000). So in-domain time
conditioning is quantized to 7 days with known violations at exactly the
surgical transitions where timing matters most (the 0–8d bin). If real
dates are recoverable from the RANO CSV/meta, re-derive deltas from dates;
otherwise bound the error (what fraction of pairs sit in multi-date
weeks?).

### K3-26 [moderate, internal contradiction]. "Weekly MRI" vs "~90d medians".
`results.md` describes LUMIERE as "weekly MRI during chemo-radiotherapy"
and its scorecard mechanism as "LUMIERE-calibrated dynamics expecting
weekly on-treatment volatility, applied to mostly-stable disease". But A14
established cell medians of 84–98d ("LUMIERE pairs are NOT predominantly
weekly... SAILOR's 76d median is a similar regime"). Both wordings cannot
stand: either the weekly framing goes (and the mechanism sentence with it —
phase composition PD-64% vs SD-48% survives alone), or A14's medians need
reconciliation with the protocol (weekly during 6-week CRT + sparse
follow-up would explain both — in which case the *training distribution is
a mixture of two regimes* and the predictor's magnitude prior (K3-3) is a
mixture average, a sharper and more testable mechanism than either
phrasing).

### K3-27 [low]. Surgery pairs remain in all training and eval (R7 open).
G15's resection-at-1-day-gap pairs still train every dynamics run and score
every gate (the 0–8d hardest bin both cohorts). The atlas (A14) quantifies
them; the intervention/reset treatment (R7) was never applied. Any future
dynamics leg should condition or exclude first — currently the headline
numbers all include a known-unlearnable discontinuity class.

### K3-28 [low]. G1 survival leakage: verified present, damage argued small.
`dataset.py:151` (surv/200) and `sailor.py:155` (months→weeks/200) still
feed total OS into every visit token; the R4 survival-drop ablation was
never run. Mitigating facts I verified: the target branch is image-only
(`jepa_model.py::encode_target_visit` — clinical never enters targets, so
OS cannot leak into the JEPA objective's ground truth, only into the
predictor's inputs); clinical-only RANO probe ≈ 0.17; my K3-21 ablation
(clinical volume R² 0.002). A reviewer will still flag it; dropping the
column (or the one-run ablation) remains the cheapest close-out.

### K3-29 [checked, minor]. G3/G4/G5/D2 status verified in code.
G3 strip fix present (`dataset.py:79`); G4 index-aligned gaps present and
correct (`sailor.py:127–142`, sums spanned gaps — genuinely fixed);
G5 sex=0-for-all still present (`sailor.py:156`, embedding admits no UNK —
fix as documented when touching); D2 actions logged-not-conditioned
(`jepa_model.py:8/303` — hook only); SAILOR ses-01→action-2 default is
inert (actions unused; probes skip unlabeled). No action needed beyond the
R4 bundle already planned.

### K3-30 [trivial]. Bookkeeping nits.
A9/D23 n=397 response visits vs A10/surprise n=393 pairs (4-visit gap from
filtering, unexplained in one line — likely G3/min-visits interplay);
pilot.md "2051 volumes processed" vs "2487 finals" vs "2455-volume full
run" (2051 must be one invocation's job count under resume-safe restarts,
but no log line says so); D26's "sessions with any missing quad modality
dropped" contradicts the code (partial sessions kept, mask False —
`sailor.py:72`). None load-bearing; each deserves one clarifying line.

---

## Part 5 — Process and infrastructure

### K3-31 [moderate]. Caches carry zero provenance.
`probe_cache.pt`, `horizon_cache.pt`, `sailor*.pt`, `field_cache.pt` store
no source-checkpoint path, config hash, commit, or timestamp. Every
cache-dependent conclusion (gates, probes, CORAL, fields) rests on filename
convention — the same failure class as D22's champion overwrite, which cost
a leg. `split_gate.py`'s default cache (`horizon_cache.pt`) is consumed
without any check that its states/z came from the `--champion` predictor
being scored. Fix: write `{champion_path, config_hash, git_sha, date}` into
every cache at encode time and assert on load (minutes of work).

### K3-32 [low, acknowledged]. The hero test set is now a validation set (G12).
~10+ configs scored on the same 13 patients/49 probe visits; the repo
learned "only CV counts" but keeps spending the test set (horizon re-gate,
aux re-gate, head sweeps). Freeze a second holdout slice before the next
claim that matters.

### K3-33 [low, mandate gap]. Trained GRU/MLP baselines still missing (G14/R12).
Proposal §6 requires beating them; only untrained floors (A6: 0.9993/1.0027)
exist. The transformer justification rests on persistence alone. CPU-cheap;
no paper without it.

### K3-34 [low]. `collate_fn` swallows every load exception silently.
`collate.py:46–47` (`except Exception: continue`, mask False). Load-time
failures are invisible — modality coverage can decay without a trace (and
did, in effect, via K3-1's empties, which load *successfully* and are
therefore doubly invisible). Count skips per modality per run.

### K3-35 [verified, no action]. Weight loading, A2 lesson, stuck.
137/161 keys + 12 unused + 24 random `norm_cross_attn` (MONAI-version
delta), strict-clean otherwise — matches D8; behavioral-probe rule intact.

---

## Part 6 — What survived the probe (credit)

- **Same-space correction (D28/A8-addendum).** `split_gate.py` is the right
  instrument; I reproduced its table exactly (K3-11). The 2.7× retraction
  was honest and the "narrow win / pooled loss" revision is accurate as far
  as it goes — my complaint is only that it stopped short of the patient
  win counts the same script prints.
- **CV discipline.** The 0.45→0.33 correction and "only CV counts" rule are
  exemplary; the aux re-gate applied the same standard to kill its own
  headline (0.509→0.328). This is what makes the remaining G11 caveat (K3-16)
  worth fixing rather than damning.
- **Lead-time design (A13).** Incident-PD framing + persistence control is
  exactly right; the negative RQ1 answer was accepted without spin.
- **Pre-registered decider (A19).** The §6 outcome table + audit-before-claim
  is the reason the skull confound got caught instead of published.
- **Scoping discipline.** "Ruled out as necessary, not proven zero-effect"
  (uint8); "upper bound, NOT PLHM-alone" (A18); "no operational gain at
  N=27, NOT proof useless in principle" (A15-narrow). The hedges are in the
  right places — except where noted (K3-4, K3-20).
- **R11 mechanism evidence.** Dose-response monotonic in accumulation,
  verified from raw kernel logs; exploit protocol honestly labeled
  transient.
- **Fixes verified in code:** G2 target-eval pin (`jepa_model.py:128–136`),
  G3 strip, G4 gap alignment, pair-weighted accumulation math, bucketing
  determinism. The engineering is careful.

---

## Part 7 — Ranked next steps

1. **Content-check + rebuild SAILOR inputs (K3-1).** Empty→mask-False in
   adapter/collate; extend QA to the derivatives tree; rebuild caches;
   re-issue all SAILOR tables. Hours. Nothing cross-site is citable until
   this is done.
2. **Same-space arm rows (K3-2) + win counts and bootstrap CIs (K3-11).**
   Recompute (a)-rows through EMA endpoints; publish per-patient wins and
   CIs for every gate cell; state the claim as mean-effect or
   majority-effect explicitly. Hours.
3. **Gain-calibration experiment (K3-3).** Fit a scalar gain per site (then
   per gap-bin, then state+gap-predicted) on the frozen head; compare
   against A15's field. Predicts: site-gain alone reaches ~persistence;
   directional refit adds the 0.0040→0.0036 edge. CPU day, potentially the
   cleanest result in the transfer story.
4. **CORAL mean-only ablation (K3-4).** Hours; decides whether "covariance"
   survives as a claim.
5. **Phase-channel aliveness (K3-20).** Embedding norms, shuffle control,
   phase-from-state probe. CPU hours; decides whether A16(7)'s redundancy
  reading stands.
6. **True-unseen probe CV + codebook swap (K3-16/17).** CPU hours; both are
   pre-registered-style robustness the writeup needs.
7. **Interval verification vs DICOM dates (K3-24).** Half day; retires or
   reframes every gap claim.
8. **LoRA ablation + cache provenance + GRU baselines (K3-13/31/33).**
   Small each; all mandate- or hygiene-grade.
9. **Reconcile "weekly" vs "~90d" (K3-26) and run R7 surgery conditioning.**
   Writing + one eval pass; the mixture-regime reading is directly testable
   (per-regime magnitude priors).
10. **Drift-by-arm measurement (K3-8, DONE — filled in).** Image change
    restored 0.19 → 0.75 by reprocessing; latent drift stayed damped
    (0.0035 → 0.0039 vs LUMIERE ~0.006). The restored component is
    encoder-invisible jitter, not biology — use this to close the
    damping-vs-acquisition loop as written in K3-8.

---

*Appendix — new numbers and their provenance (all CPU, champion
`checkpoints/champion_0.0081.pt`, config `config/default.yaml`):*
- *Phantom-modality scan: central-slab [80:110, 100:130, 80:110] nzfrac on
  all 1080 SAILOR base files; 19 T2 + 2 T1 with nzfrac < 0.05; 25/243
  consecutive pairs touch one (§K3-1). sub-23 vision-norm 24.81 vs 26.11
  cohort from `sailor_cache.pt`.*
- *Gain/direction decomposition: champion `Predictor` over cached states vs
  cached EMA targets (`horizon_cache.pt` LUMIERE, `field_cache.pt` SAILOR);
  oracle-gain counterfactual rescales predicted Δ to true ‖Δ‖ (§K3-3).*
- *Same-space win counts: recomputed per-patient means from
  `horizon_cache.pt`, matches A8-addendum means to 4 decimals (§K3-11).*
- *Volume ablation: `fit_ridge_cv` on `probe_cache.pt` rows, hero splits,
  n=73; fused R² 0.154 reproduces A11 exactly (§K3-21).*
- *Site separability: ridge λ=100 row-CV acc 0.969±0.017, AUC 1.0 on 638+270
  vision rows (§K3-4).*
- *Drift-by-arm: matched consecutive pairs, image-space mean|Δ| on
  trilinear-96³ with per-volume z-score (A18 metric shape); 60 SAILOR pairs
  × {T1c,T1,T2,Flair} = 240 slots for derivatives and skull-out trees,
  60 LUMIERE CT1 pairs post-peri-op (§K3-8).*
