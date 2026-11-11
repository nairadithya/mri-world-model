#!/usr/bin/env python3
"""Build the curated research figures, optionally including historical plots."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CURATED = ("info/plots/make_current_plots.py", "report/make_figures.py")
HISTORICAL = ("info/plots/make_plots.py", "info/plots/make_plots_review.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", action="store_true", help="also build historical plot sets")
    parser.add_argument("--list", action="store_true", help="list selected commands without rendering")
    args = parser.parse_args()
    # The original historical builder also emits the curated gallery.
    selected = HISTORICAL + ("report/make_figures.py",) if args.historical else CURATED
    for script in selected:
        command = [sys.executable, "-B", str(ROOT / script)]
        if args.list:
            print(f"{sys.executable} -B {script}")
        else:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
