"""Materialize / verify the locked eval protocol (Step 1).

Writes ``info/eval_folds.json`` (frozen patient-wise folds over the 26
encoder-unseen patients) and asserts encoder-train/probe disjointness.

Usage:
    python scripts/lock_eval.py --write
    python scripts/lock_eval.py            # verify an existing file
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.eval_protocol import (DEFAULT_PROTOCOL, assert_disjoint,
                                    build_protocol, load_protocol,
                                    write_protocol)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--fold-seed", type=int, default=2026)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    if args.write:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        proto = build_protocol(cfg, k=args.k, fold_seed=args.fold_seed)
        path = write_protocol(proto, args.protocol)
        print(f"wrote {path}")
    else:
        proto = load_protocol(args.protocol)

    assert_disjoint(proto)
    print(f"encoder_train  : {len(proto['encoder_train'])}")
    print(f"encoder_unseen : {len(proto['encoder_unseen'])} "
          f"(dev {len(proto['dev'])}, final {len(proto['final'])})")
    print(f"folds          : {proto['k']} "
          f"({[len([p for p, f in proto['folds'].items() if f == i]) for i in range(proto['k'])]})")
    print(f"primary metric : {proto['primary_metric']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
