"""Prepare private R39 cutoff rows or run the fixed CPU-only comparison."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.harness.analysis.treatment_forecast import prepare, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("prepare")
    build.add_argument("--inventory", type=Path, required=True)
    build.add_argument("--predictions", type=Path, required=True)
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    fit = sub.add_parser("run")
    fit.add_argument("--rows", type=Path, required=True)
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--canary", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.inventory, args.predictions, args.root, args.out)
    else:
        result = run(args.rows, args.out, args.canary)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
