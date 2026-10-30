# Timing protocol decision

**PERMITTED_WITH_DISCLOSURE** — approximate, positive **week-bin past gaps** are compatible with R37. Exact calendar intervals and `timing_verified=True` are not requirements stated by the governing protocol. **Raw manifest `day` differences require a suffix correction before they qualify.**

## Review

- **Correct — governing protocol permits this interpretation.** `info/METHODS.md:374,377,387` defines next-observed-visit prediction, completed historical gaps encoded as `log1p(days)`, and exclusion of examples with missing or nonpositive historical gaps. It explicitly excludes future gaps from predictor inputs. It does not require calendar-date verification. The existing timing contract also distinguishes approximate timing from exact horizons (`info/METHODS.md:109–111`).

- **Correct — uncertainty must remain explicit.** The source identifies its timing as curated week bins, sets `timing_verified=False`, verifies ordering, and states that same-week suffixes provide order only (`src/harness/data/anatomy_manifest.py:213–217`). Using that documented resolution does not change the next-observation target or method family. Guardrails require timing limits to be reported and permit engineering of approved hypotheses (`info/research_guardrails.md:9,13`). Bounded checks—not indefinite biological validation—are required (`AGENTS.md:43–48`).

- **Finding: P1 — raw `day` differences can contain artificial suffix durations.** `anatomy_manifest.py:213` calls `parse_week_to_days`; that function returns `weeks * 7 + suffix` (`src/data/dataset.py:41–48`). Thus, as a code-level example, `week-010` to `week-010-1` produces a positive difference of 1 despite the manifest declaring the suffix order-only. Cross-bin differences can also be distorted by suffixes. Merely excluding nonpositive **raw** differences is insufficient.  
  **Smallest fix:** retain original visit identities and ordering, but derive approximate elapsed coordinates from the **week-bin component only**, multiplied by seven. Do not treat suffix increments or legacy substituted gaps as elapsed time. Reject affected examples when a required retained historical gap is missing, nonfinite, zero, or negative; do not delete intervening visits to manufacture valid histories.

- **Correct — worker wording is unnecessarily strict.** “Measured preceding-visit gap evidence” and “Actual elapsed days” appear in `src/harness/analysis/spatial_dynamics_protocol.py:3–5,19–20`; TODO repeats the measured-gap concern (`info/TODO.md:13`). METHODS owns the protocol (`info/TODO.md:9`). These implementation/status descriptions should be clarified, not treated as an additional operator-imposed exact-date requirement. The builder’s positive-gap rejection remains appropriate (`spatial_dynamics_protocol.py:101–108`).

## Minimal clarification

No operator-approved scientific amendment is needed for this timing interpretation. Before binding the implementation, clarify its timing contract:

> Historical gaps may use approximate curated week-bin offsets, computed from the week component × 7. Same-week suffixes establish ordering only and never supply elapsed days. Preserve the full original visit chronology, reject examples with invalid required historical gaps, and never bridge unusable visits. Report timing as approximate and unverified; infer no target gap or exact-day horizon.

The full-chronology requirement remains: the source generator enumerates visits from processed and volume directories (`anatomy_manifest.py:149–156`); that enumeration alone does not prove all original visits are represented.

## Execution boundary

For the authorized descriptive E noise/no-noise ablation, apply the same frozen eligibility rows to both conditions even though E does not consume historical gap values. Preserve identical source rows and patient weights (`info/METHODS.md:404–408`).

**Stop before execution** if suffix-only/artificial durations remain, required chronology cannot be established from existing evidence, or eligibility requires bridging omissions. Do not waive those defects because E ignores gap values.

**Merge verdict: OK with notes — timing interpretation only.** This is not runner approval, provenance verification, scientific approval, or evidence that an ablation launched. The supplied HEAD/cache/checkpoint assertions were not independently verified in this read-only review.