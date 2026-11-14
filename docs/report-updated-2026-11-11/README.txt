WORLD MODELS FOR LONGITUDINAL GLIOMA MRI

Main source: report.tex
Compiled document: report.pdf
This is the updated report package. It retains the institutional template and
author details. The original docs/report package remains unchanged.

READING THE REPORT
The abstract asks whether a useful state of current anatomy also benefits
from patient history when predicting future anatomy. Chapters 6-8 follow
that question through verification, results and interpretation. Their prose
is organized by mechanisms and evidence rather than execution dates.

PACKAGE CONTENTS
  report.tex, frontmatter.tex, references.tex and chapters/: LaTeX sources
  amrita_logo.png: institutional logo
  figures/: all embedded figures and retained illustration assets
  metrics/: 30 canonical aggregate snapshots and their SHA256 provenance
  make_figures.py: four aggregate plots, saved as PDF and PNG
  make_forecast_attribution.py: the MU 24-week Dice attribution plot
  EVIDENCE_MAP.json: paths and hashes for the canonical research records
  SHA256SUMS: integrity checks for every packaged file except this manifest
  EDITORIAL_REVIEW.txt: editorial decisions and verification notes

From this directory, verify the packaged files with:
  sha256sum -c SHA256SUMS

The package contains all images and data needed to compile the report and
regenerate its five aggregate plots. Other result comparisons are supported
by the bundled aggregate snapshots and the tracked canonical records listed
in EVIDENCE_MAP.json. Source paths there are relative to the repository root.
The package does not require a training run or access to patient datasets.

BUILD
From this directory, run XeLaTeX three times:
  xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error report.tex
A working TeX Live installation needs XeLaTeX and the packages listed in
report.tex. The supplied PDF uses Liberation Serif and Liberation Sans;
installed fonts can change pagination. Build intermediates are ignored.

On this laptop, the prepared format and font tree are cached in
~/.cache/world-model-report-tex. Run from this directory:
  (
    export TEXMFHOME="$HOME/.cache/world-model-report-tex/texmf-home"
    export TEXMFVAR="$HOME/.cache/world-model-report-tex/texmf-var"
    report_build=$(mktemp -d /tmp/world-model-report-build-XXXXXX)
    for pass in 1 2 3; do
      xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory="$report_build" report.tex || exit 1
    done
    cp "$report_build/report.pdf" report.pdf
  )

REGENERATING AGGREGATE FIGURES
From the repository root, with Matplotlib and NumPy installed:
  .venv/bin/python docs/report-updated-2026-11-11/make_figures.py
  .venv/bin/python docs/report-updated-2026-11-11/make_forecast_attribution.py
The scripts verify aggregate hashes and write vector PDFs and 300-dpi PNGs.
They use installed fonts, so font versions can change text rasterization.
The supplied illustrations remain in figures/ and need no external files.
Rebuilding or regenerating outputs changes their hashes; SHA256SUMS records
the exact delivered package, rather than an arbitrary later rendering.

Aggregate plot inputs:
  updated_completion: r65_lumiere_metrics.json, r69_spatial_access_metrics.json
  updated_cavity: patient_cavity_observation_metrics.json
  updated_forecast: patient_structured_forecasting_metrics.json
  updated_acquisition: patient_acquisition_jepa_metrics.json
  updated_forecast_attribution: patient_forecast_interpretation_metrics.json

Other bundled results include current anatomy, observed change, matched
pretraining, history forecasts, regional-volume corrections, fitting
diagnostics, clinical objectives, the shape-and-activity simulator, frozen
readouts and quantity distributions. Each added source has a report-use
note in metrics/provenance.json. The canonical records remain authoritative
for protocols, qualifications, corrections and historical interpretation.

STORAGE
This package retains controlled-access figure assets from the original
report and belongs in authorized local/private storage. Its supporting data
are existing aggregate records; raw MRI, participant inventories, model
weights and individual prediction archives are not part of the package.
