#!/usr/bin/env python
"""Unified train/eval harness entry point.

Usage:
    python scripts/harness.py list
    python scripts/harness.py encode --cache checkpoints/interface_cache.pt
    python scripts/harness.py eval --task rano4_forecast --view states_forecast
    python scripts/harness.py train jepa --config kaggle.yaml --epochs 30 ...
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.harness.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
