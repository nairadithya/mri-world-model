#!/usr/bin/env python
"""Anatomy-first harness entry point with an explicit legacy namespace.

Usage:
    python scripts/harness.py anatomy manifest
    python scripts/harness.py anatomy baseline
    python scripts/harness.py checkpoint inspect checkpoints/best.pt
    python scripts/harness.py legacy eval --task rano4_forecast ...
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.harness.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
