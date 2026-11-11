# Research figures

- [Curated gallery through R62](current/README.md): figures and captions derived from recorded aggregate results.
- [Report aggregate figures and captions](../../report/README.md): architecture, timeline, and aggregate comparisons. The final LaTeX report and its embedded case figures are private and must remain in authorized storage.
- [Historical figure review](REVIEW.md): older figures, coverage, and limitations; these are historical rather than the current gallery.

From the repository root, run `.venv/bin/python -B scripts/build_figures.py` to rebuild the curated gallery and report aggregate figures. Add `--historical` to also rebuild the older plot sets. Use `--list` to see the commands without rendering. Existing individual figure scripts remain available. The figure builders use aggregate records; the private report case figures are outside this command.
