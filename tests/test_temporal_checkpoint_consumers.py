"""Run with .venv/bin/python tests/test_temporal_checkpoint_consumers.py."""
import ast
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from src.harness.analysis import horizon_eval
from src.model.temporal import TemporalTransformer


class TinyWorld(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        basis = cfg["model"]["temporal"].get("time_basis", "previous_gap")
        self.temporal = TemporalTransformer(8, 2, 1, time_basis=basis)


def run_horizon_eval(directory, checkpoint_basis):
    cfg = {"model": {"temporal": {}, "predictor": {}},
           "preprocessing": {"target_size": [8, 8, 8]}}
    config = directory / "config.yaml"
    checkpoint = directory / "checkpoint.pt"
    config.write_text(yaml.safe_dump(cfg))
    saved_cfg = {"model": {"temporal": {"time_basis": checkpoint_basis}}}
    torch.save({"model": TinyWorld(saved_cfg).state_dict(), "config": saved_cfg}, checkpoint)
    with patch.object(horizon_eval, "JEPAWorldModel", TinyWorld), \
            patch.object(horizon_eval, "build_datasets", return_value=({}, {})):
        horizon_eval.main(["--config", str(config), "--ckpt", str(checkpoint)])


class CheckpointConsumerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_horizon_eval_rejects_elapsed_checkpoint_under_default_config(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "time_basis='elapsed'.*'previous_gap'"):
                run_horizon_eval(Path(directory), "elapsed")

    def test_horizon_eval_accepts_matching_legacy_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            run_horizon_eval(Path(directory), "previous_gap")

    def test_all_inventoried_raw_world_loads_have_a_prior_guard(self):
        root = Path(__file__).resolve().parents[1] / "src/harness"
        counts = {"analysis/horizon_eval.py": 1, "analysis/horizon_probe.py": 1,
                  "analysis/leadtime.py": 1, "analysis/sailor.py": 3,
                  "analysis/persistence.py": 1, "analysis/surprise.py": 1,
                  "analysis/sailor_interval.py": 1, "train/head.py": 1, "train/field.py": 1}
        for relative, expected in counts.items():
            tree = ast.parse((root / relative).read_text())
            count = 0
            for function in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
                calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)]
                loads = [c for c in calls if isinstance(c.func, ast.Attribute)
                         and c.func.attr == "load_state_dict" and isinstance(c.func.value, ast.Name)
                         and c.func.value.id == "model"]
                guards = [c for c in calls if isinstance(c.func, ast.Name)
                          and c.func.id == "validate_temporal_semantics"]
                for load in loads:
                    count += 1
                    self.assertTrue(any(g.lineno < load.lineno for g in guards), relative)
            self.assertEqual(count, expected, relative)


if __name__ == "__main__":
    unittest.main()
