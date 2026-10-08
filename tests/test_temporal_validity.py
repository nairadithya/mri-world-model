"""Structural checks for causal temporal coordinates and checkpoint compatibility."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.model.temporal import TemporalTransformer


class TemporalValidityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_elapsed_coordinates_are_cumulative_and_swap_changes_state(self):
        model = TemporalTransformer(d_model=8, nhead=2, num_layers=1, dropout=0,
                                    time_basis="elapsed").eval()
        gaps = torch.tensor([[0., 10., 20.]])
        self.assertTrue(torch.equal(model._time_coordinates(gaps),
                                    torch.tensor([[0., 10., 30.]])))
        tokens = torch.zeros(1, 3, 8)
        mask = torch.ones(1, 3, dtype=torch.bool)
        swapped = torch.tensor([[0., 20., 10.]])
        self.assertFalse(torch.allclose(model(tokens, gaps, mask),
                                        model(tokens, swapped, mask)))
        history = torch.randn(1, 4, 8)
        equal_gaps = torch.tensor([[0., 30., 30., 30.]])
        valid = torch.ones(1, 4, dtype=torch.bool)
        self.assertFalse(torch.allclose(model(history, equal_gaps, valid),
                                        model(history[:, [0, 2, 1, 3]], equal_gaps, valid)))

    def test_legacy_equal_gap_swaps_remain_equal(self):
        model = TemporalTransformer(d_model=8, nhead=2, num_layers=1, dropout=0,
                                    time_basis="previous_gap").eval()
        tokens = torch.tensor([[[1., 0, 0, 0, 0, 0, 0, 0],
                                [0., 1, 0, 0, 0, 0, 0, 0],
                                [0., 0, 1, 0, 0, 0, 0, 0]]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        gaps = torch.tensor([[10., 10., 10.]])
        swapped_tokens = tokens[:, [1, 0, 2]]
        self.assertTrue(torch.allclose(model(tokens, gaps, mask),
                                       model(swapped_tokens, gaps, mask), atol=1e-6))

    def test_prefixes_are_causal_in_both_modes(self):
        for basis in ("previous_gap", "elapsed"):
            model = TemporalTransformer(d_model=8, nhead=2, num_layers=1, dropout=0,
                                        time_basis=basis).eval()
            tokens = torch.randn(1, 4, 8)
            gaps = torch.tensor([[0., 5., 10., 15.]])
            mask = torch.ones(1, 4, dtype=torch.bool)
            states, _ = model.forward_prefixes(tokens, gaps, mask)
            changed = tokens.clone()
            changed[:, 3] += 100
            changed_gaps = gaps.clone()
            changed_gaps[:, 3] += 100
            states_changed, _ = model.forward_prefixes(changed, changed_gaps, mask)
            self.assertTrue(torch.allclose(states[:, :2], states_changed[:, :2]))

    def test_state_dict_keys_and_values_are_legacy_compatible(self):
        torch.manual_seed(4)
        legacy = TemporalTransformer(d_model=8, nhead=2, num_layers=1, dropout=0)
        state = legacy.state_dict()
        elapsed = TemporalTransformer(d_model=8, nhead=2, num_layers=1,
                                      dropout=0, time_basis="elapsed")
        self.assertEqual(set(state), set(elapsed.state_dict()))
        elapsed.load_state_dict(state, strict=True)
        self.assertTrue(all(torch.equal(v, elapsed.state_dict()[k])
                            for k, v in state.items()))
        self.assertTrue(all(torch.equal(v, legacy.state_dict()[k])
                            for k, v in state.items()))

    def test_default_output_is_exactly_explicit_legacy(self):
        default = TemporalTransformer(d_model=8, nhead=2, num_layers=1, dropout=0).eval()
        explicit = TemporalTransformer(d_model=8, nhead=2, num_layers=1, dropout=0,
                                       time_basis="previous_gap").eval()
        explicit.load_state_dict(default.state_dict(), strict=True)
        tokens = torch.randn(2, 4, 8)
        gaps = torch.tensor([[0., 10., 20., 30.], [0., 7., 0., 0.]])
        mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
        self.assertTrue(torch.equal(default(tokens, gaps, mask), explicit(tokens, gaps, mask)))
        for a, b in zip(default.forward_prefixes(tokens, gaps, mask),
                        explicit.forward_prefixes(tokens, gaps, mask)):
            self.assertTrue(torch.equal(a, b))

    def test_checkpoint_loader_rejects_same_shape_semantic_mismatch(self):
        from src.harness.checkpoints import load_model
        legacy = TemporalTransformer(d_model=8, nhead=2, num_layers=1)
        elapsed = TemporalTransformer(d_model=8, nhead=2, num_layers=1, time_basis="elapsed")
        payload = {"model": legacy.state_dict()}
        load_model(legacy, payload)
        with self.assertRaisesRegex(ValueError, "temporal time_basis"):
            load_model(elapsed, payload)
        payload["config"] = {"model": {"temporal": {"time_basis": "elapsed"}}}
        load_model(elapsed, payload)
        with self.assertRaisesRegex(ValueError, "temporal time_basis"):
            load_model(legacy, payload)

    def test_invalid_basis_fails_and_jepa_config_is_wired(self):
        with self.assertRaises(ValueError):
            TemporalTransformer(d_model=8, nhead=2, num_layers=1, time_basis="bad")
        from src.model import jepa_model
        cfg = {
            "model": {
                "brainiac": {}, "clinical": {"hidden_dim": 4, "dropout": 0},
                "fusion": {"vision_dim": 8},
                "temporal": {"d_model": 20, "nhead": 2, "num_layers": 1,
                             "dim_feedforward": 24, "dropout": 0, "time_basis": "elapsed"},
                "predictor": {"hidden_dim": 20, "output_dim": 8, "dropout": 0},
                "target": {"projection_dim": 8},
            }, "actions": {"classes": []}
        }
        with patch.object(jepa_model, "BrainiacEncoder", return_value=torch.nn.Identity()):
            model = jepa_model.JEPAWorldModel(cfg)
        self.assertEqual(model.temporal.time_basis, "elapsed")


if __name__ == "__main__":
    unittest.main()
