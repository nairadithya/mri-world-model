from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml

from src.harness.analysis import horizon_eval
from src.model.temporal import TemporalTransformer


class TinyWorld(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        basis = cfg["model"]["temporal"].get("time_basis", "previous_gap")
        self.temporal = TemporalTransformer(8, 2, 1, time_basis=basis)


def _run_horizon_eval(tmp_path, checkpoint_basis):
    cfg = {"model": {"temporal": {}, "predictor": {}},
           "preprocessing": {"target_size": [8, 8, 8]}}
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "checkpoint.pt"
    config.write_text(yaml.safe_dump(cfg))
    saved_cfg = {"model": {"temporal": {"time_basis": checkpoint_basis}}}
    torch.save({"model": TinyWorld(saved_cfg).state_dict(), "config": saved_cfg}, checkpoint)
    with patch.object(horizon_eval, "JEPAWorldModel", TinyWorld), \
            patch.object(horizon_eval, "build_datasets", return_value=({}, {})):
        horizon_eval.main(["--config", str(config), "--ckpt", str(checkpoint)])


def test_horizon_eval_rejects_elapsed_checkpoint_under_default_config(tmp_path):
    with pytest.raises(ValueError, match="time_basis='elapsed'.*'previous_gap'"):
        _run_horizon_eval(tmp_path, "elapsed")


def test_horizon_eval_accepts_matching_legacy_checkpoint(tmp_path):
    _run_horizon_eval(tmp_path, "previous_gap")


def test_active_jepa_raw_checkpoint_consumers_validate_semantics():
    root = Path(__file__).parents[1] / "src/harness"
    files = [
        "analysis/horizon_eval.py", "analysis/horizon_probe.py", "analysis/leadtime.py",
        "analysis/sailor.py", "analysis/persistence.py", "analysis/surprise.py",
        "analysis/sailor_interval.py", "train/head.py", "train/field.py",
    ]
    for relative in files:
        assert "validate_temporal_semantics" in (root / relative).read_text(), relative
