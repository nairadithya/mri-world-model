# Notes → formalized directions

Formalized 2026-09-13 into `directions.md` (R29–R32). Originals kept in
parentheses so nothing is lost; the R-ID is now the reference.

- **R29** — Train/val generalization-gap monitor. Persist per-epoch train loss
  next to val, plot both, split "overfit" vs "underfit" with evidence.
  *(orig: plot training and validation loss together to find overfitting or not)*
- **R30** — Imbalance handling beyond class-weighted CE: weighted *sampler*
  (distinct from A34's loss reweighting), focal/logit-adjusted readout, and the
  binary progression-vs-not target.
  *(orig: data imbalance techniques, class-specific / class-weighted losses)*
- **R31** — Feature-alignment *loss* for cross-domain dynamics (CORAL/MMD/
  adversarial on the projector); preprocessing side is closed by A18/A19/A21.
  *(orig: cross-domain adaptation — loss, preprocessing)*
- **R32** — Pretrained comparator (executes R28) plus an alternative-encoder
  arm testing whether the edge is the objective or BRAINIAC.
  *(orig: using pre-trained models for comparison with our results)*

Hygiene: R13 is answered negative in its loss-side form (A34); R7's
`--surgery-window` is implemented but only tested bundled — isolate before
citing. See `directions.md` for hypothesis → protocol → success bar → cost.
