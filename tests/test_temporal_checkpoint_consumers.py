"""Run with .venv/bin/python tests/test_temporal_checkpoint_consumers.py."""
import ast
import json
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

    def run_lora(self, configured, saved):
        from src.harness.train import lora
        cfg = {"model": {"temporal": {"time_basis": configured}},
               "preprocessing": {"target_size": [8, 8, 8]},
               "data": {"meta_dir": "unused", "root": "unused"}}
        saved_cfg = {"model": {"temporal": {"time_basis": saved}}}
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            checkpoint = Path(directory) / "checkpoint.pt"
            config.write_text(yaml.safe_dump(cfg))
            torch.save({"model": TinyWorld(saved_cfg).state_dict(), "config": saved_cfg}, checkpoint)
            with patch.object(lora, "load_protocol", return_value={"encoder_train": ["p1"], "dev": ["p2"]}), \
                    patch.object(lora, "LUMIEREDataset", return_value=[]), \
                    patch.object(lora, "DataLoader", return_value=[]), \
                    patch.object(lora, "JEPAWorldModel", TinyWorld), \
                    patch.object(lora, "configure_trainable", side_effect=RuntimeError("training sentinel")):
                lora.main(["--config", str(config), "--champion", str(checkpoint), "--protocol", "unused"])

    def test_lora_rejects_mismatched_basis_before_configuring_training(self):
        for configured, saved in (("elapsed", "previous_gap"), ("previous_gap", "elapsed")):
            with self.assertRaisesRegex(ValueError, "temporal time_basis"):
                self.run_lora(configured, saved)

    def test_lora_accepts_matching_basis_before_configuring_training(self):
        for basis in ("previous_gap", "elapsed"):
            with self.assertRaisesRegex(RuntimeError, "training sentinel"):
                self.run_lora(basis, basis)

    def test_notebook_world_loads_have_prior_semantic_guards(self):
        root = Path(__file__).resolve().parents[1]
        for name, expected in (("kaggle/hero_run.ipynb", 2),
                               ("kaggle/kernel-horizon/horizon_leg.ipynb", 1)):
            notebook = json.loads((root / name).read_text())
            count = 0
            for cell in notebook["cells"]:
                source = "".join(cell["source"])
                if cell["cell_type"] != "code" or "model.load_state_dict(" not in source:
                    continue
                calls = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)]
                guards = [c for c in calls if isinstance(c.func, ast.Name)
                          and c.func.id == "validate_temporal_semantics"]
                for load in calls:
                    if not (isinstance(load.func, ast.Attribute) and load.func.attr == "load_state_dict"
                            and isinstance(load.func.value, ast.Name) and load.func.value.id == "model"):
                        continue
                    count += 1
                    self.assertTrue(any(g.lineno < load.lineno for g in guards), name)
            self.assertEqual(count, expected, name)

    def test_all_inventoried_raw_world_loads_have_a_prior_guard(self):
        root = Path(__file__).resolve().parents[1] / "src/harness"
        expected = {"analysis/horizon_eval.py": 1, "analysis/horizon_probe.py": 1,
                    "analysis/leadtime.py": 1, "analysis/sailor.py": 3,
                    "analysis/persistence.py": 1, "analysis/surprise.py": 1,
                    "analysis/sailor_interval.py": 1, "train/head.py": 1,
                    "train/field.py": 1, "train/lora.py": 2}
        observed = {}
        for path in root.rglob("*.py"):
            relative = str(path.relative_to(root))
            tree = ast.parse(path.read_text())
            for function in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
                worlds = {target.id for node in ast.walk(function) if isinstance(node, ast.Assign)
                          and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                          and node.value.func.id == "JEPAWorldModel" for target in node.targets
                          if isinstance(target, ast.Name)}
                calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)]
                guards = [c for c in calls if isinstance(c.func, ast.Name)
                          and c.func.id == "validate_temporal_semantics"]
                for load in calls:
                    if not (isinstance(load.func, ast.Attribute) and load.func.attr == "load_state_dict"
                            and isinstance(load.func.value, ast.Name) and load.func.value.id in worlds):
                        continue
                    observed[relative] = observed.get(relative, 0) + 1
                    payload = load.args[0].value.id
                    self.assertTrue(any(g.lineno < load.lineno and g.args[0].id == load.func.value.id
                                        and g.args[1].id == payload for g in guards), relative)
        self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
