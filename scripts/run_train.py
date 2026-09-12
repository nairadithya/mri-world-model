#!/usr/bin/env python
"""End-to-end JEPA training entry point (legacy shim).

The implementation now lives in :mod:`src.harness.train.jepa`; this file keeps
the historical ``scripts/run_train.py`` path working for Kaggle/hero_run.

Usage:
    python scripts/run_train.py --config config/default.yaml [--epochs 10] [--batch-size 2]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.harness.train.jepa import main  # noqa: E402

if __name__ == "__main__":
    main()
