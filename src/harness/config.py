"""Config loading + the CLI-override resolution shared by every entry point."""
from __future__ import annotations

import os

import yaml

from . import paths


def find_meta(meta_dir: str, prefix: str) -> str:
    """First file in ``meta_dir`` starting with ``prefix`` (legacy helper)."""
    for f in os.listdir(meta_dir):
        if f.startswith(prefix):
            return os.path.join(meta_dir, f)
    raise FileNotFoundError(f"No {prefix}* in {meta_dir}")


def load_config(path: str = paths.DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
