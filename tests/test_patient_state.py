"""Run numerical tests on mosaic-gpu; no patient images needed for these contracts."""
import copy
from pathlib import Path
import tempfile
import unittest
import random

import numpy as np
import torch

from src.patient_state.data import AFFINE, horizon_inventory
from src.patient_state.pipeline import volumes
from src.patient_state.state import PatientState, check_update, persistence
from src.patient_state.training import hide
from src.patient_state.labels import harmonize_labels
from src.patient_state.temporal import ensemble_crps, forecast_pairs


def state():
    return PatientState({"feature0": torch.ones(1, 2, 8, 8, 8), "feature1": torch.ones(1, 4, 4, 4, 4)},
        {"patient": "example", "visit": "week-000", "nominal_week": 0, "visit_order": [0, 0],
         "availability": [True]*4, "affine": AFFINE, "shape_xyz": [8]*3, "model_id": "fixed",
         "backend": "test", "time_unit": "nominal_week"})


class PatientStateTests(unittest.TestCase):
    def test_complete_serialization(self):
        original = state()
        original.memory["previous"] = torch.tensor([1.])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"state.pt"
            original.save(path)
            restored = PatientState.load(path)
            self.assertEqual(restored.metadata, original.metadata)
            self.assertEqual(restored.payload_bytes, original.payload_bytes)
            self.assertEqual(set(restored.features), set(original.features))
            self.assertTrue(torch.equal(restored.memory["previous"], original.memory["previous"]))
            with self.assertRaises(FileExistsError):
                original.save(path)

    def test_update_cutoff_and_geometry(self):
        previous = state()
        current = copy.deepcopy(previous)
        current.metadata.update(visit="week-012", nominal_week=12, visit_order=[12, 0])
        check_update(previous, current)
        for key, bad in (("patient", "another"), ("visit_order", [0, 0]), ("model_id", "new-model"), ("shape_xyz", [7]*3)):
            changed = copy.deepcopy(current)
            changed.metadata[key] = bad
            with self.assertRaises(ValueError):
                check_update(previous, changed)

    def test_same_week_suffix_is_order_not_elapsed_time(self):
        a, b = state(), state()
        b.metadata.update(visit="week-000-1", visit_order=[0, 1])
        check_update(a, b)
        self.assertEqual(b.metadata["nominal_week"]-a.metadata["nominal_week"], 0)

    def test_persistence_has_no_invented_uncertainty(self):
        original = state()
        predicted = persistence(original, 24)
        self.assertEqual(predicted.metadata["forecast_week"], 24)
        self.assertEqual(predicted.metadata["forecast_method"], "persistence")
        self.assertIn("unavailable", predicted.metadata["uncertainty_status"])
        self.assertNotIn("forecast_week", original.metadata)
        self.assertTrue(torch.equal(predicted.features["feature0"], original.features["feature0"]))

    def test_nonfinite_and_singular_geometry_rejected(self):
        broken = state()
        broken.features["feature1"][0, 0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            broken.validate()
        broken = state()
        broken.metadata["affine"] = [[0.]*4]*4
        with self.assertRaises(ValueError):
            broken.validate()

    def test_volume_uses_affine_voxel_size(self):
        labels = np.array([[[1, 2], [1, 3]]], dtype=np.uint8)
        affine = np.diag([2., 3., 4., 1.])
        result = volumes(labels, affine)
        self.assertAlmostEqual(result["enhancing"], .048)

    def test_hidden_sequence_values_cannot_enter_input(self):
        original = torch.randn(1, 4, 8, 8, 8)
        altered = original.clone()
        altered[:, 3] = 1000
        a, availability, mask = hide(original, 3)
        b, _, _ = hide(altered, 3)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(bool(availability[0, 3]))
        self.assertTrue(bool(mask[:, 3].all()))
        self.assertTrue(torch.equal(a[:, :3], original[:, :3]))

    def test_hidden_region_always_intersects_brain(self):
        x = torch.ones(1, 4, 32, 32, 32)
        support = torch.zeros(1, 1, 32, 32, 32, dtype=torch.bool)
        support[:, :, 0, 0, 0] = True
        for seed in range(12):
            random.seed(seed)
            corrupted, availability, hidden = hide(x, support=support)
            self.assertTrue(bool((hidden & support).any()))
            self.assertTrue(bool(availability.any()))

    def test_horizon_inventory_uses_only_metadata(self):
        rows = [{"patient": "p", "visit_order": [w, 0]} for w in (0, 12, 24)]
        result = horizon_inventory({"groups": {"fitting": rows}})
        self.assertEqual(result["fitting"]["12"]["cutoffs"], 2)
        self.assertEqual(result["fitting"]["24"]["cutoffs"], 1)
        self.assertEqual(result["fitting"]["12"]["with_history"], 1)

    def test_mu_cavity_is_not_background(self):
        mapped = harmonize_labels(np.array([0, 1, 2, 3, 4]), "MU")
        np.testing.assert_array_equal(mapped, [0, 2, 3, 1, -1])
        with self.assertRaises(ValueError):
            harmonize_labels(np.array([5]), "MU")

    def test_forecast_pairs_keep_future_out_of_history(self):
        rows = [{"patient": "p", "visit_order": [w, 0], "visit": str(w)} for w in (0, 4, 17, 28)]
        pairs = forecast_pairs(rows)
        self.assertTrue(pairs)
        for pair in pairs:
            cutoff = pair["history"][-1]["visit_order"][0]
            self.assertGreater(pair["target"]["visit_order"][0], cutoff)
            self.assertNotIn(pair["target"], pair["history"])
            self.assertIn(pair["horizon"], (12, 24))
            self.assertLessEqual(abs(pair["target_offset"]), 4)

    def test_crps_point_forecast_and_spread(self):
        self.assertEqual(float(ensemble_crps(torch.tensor([[2.], [2.]]), torch.tensor([5.]))), 3.)
        self.assertEqual(float(ensemble_crps(torch.tensor([[0.], [2.]]), torch.tensor([1.]))), .5)

class ForecastContractTests(unittest.TestCase):
    def test_current_only_memory_is_independent_of_history_and_gap(self):
        from src.patient_state.temporal import SpatialPredictor
        from src.patient_state.observation import FEATURES
        torch.manual_seed(17)
        net=SpatialPredictor('current')
        fs=[torch.randn(1,c,4,4,4) for c in FEATURES]
        with torch.no_grad():
            a=net.update(fs,torch.randn(1,32,4,4,4),12.)
            b=net.update(fs,None,0.)
        self.assertTrue(torch.equal(a,b))
        with self.assertRaises(ValueError):net.update(fs,None,-1)

    def test_availability_is_an_explicit_temporal_input(self):
        from src.patient_state.temporal import SpatialPredictor
        from src.patient_state.observation import FEATURES
        net=SpatialPredictor('history')
        fs=[torch.zeros(1,c,4,4,4) for c in FEATURES]
        with torch.no_grad():
            a=net.update(fs,availability=torch.ones(1,4))
            b=net.update(fs,availability=torch.tensor([[0,1,1,1]]))
        self.assertFalse(torch.equal(a,b))

    def test_volume_intervals_report_width_and_misses(self):
        from scripts.evaluate_patient_forecasts import distribution_scores
        result=distribution_scores([[2,3,4],[2,3,4]],[2,5,4])
        self.assertEqual(result['coverage90'],[1.,0.,1.])
        self.assertEqual(result['width90_ml'],[0.,0.,0.])
        self.assertEqual(result['crps_ml'],[0.,2.,0.])
        self.assertEqual(result['interval_score90_ml'],[0.,40.,0.])

    def test_same_week_order_never_becomes_future_elapsed_time(self):
        rows=[{'patient':'p','visit_order':[w,s],'visit':f'{w}-{s}'} for w,s in ((0,0),(0,1),(12,0),(24,0))]
        pair=forecast_pairs(rows)[0]
        self.assertEqual(pair['history'][-1]['visit_order'],[0,1])
        self.assertEqual(pair['target_offset'],0)

    def test_forecast_noise_is_consistent_in_overlapping_patches(self):
        from src.patient_state.forecast_inference import sample_state
        from src.patient_state.observation import FEATURES
        original=state()
        original.metadata.update(padded_shape=[160,128,128],shape_xyz=[160,128,128],
            patches=[{'prefix':'a_','origin':[0,0,0],'size':128},{'prefix':'b_','origin':[32,0,0],'size':128}])
        original.features={prefix+f'feature{i}':torch.zeros(1,c,4,4,4)
                           for prefix in ('a_','b_') for i,c in enumerate(FEATURES)}
        lv={k:torch.zeros_like(v) for k,v in original.features.items()}
        original.features['brain_support']=torch.ones(1,1,1,1,1,dtype=torch.bool)
        drawn=sample_state(original,lv,17)
        for i in range(6):
            self.assertTrue(torch.equal(drawn.features[f'a_feature{i}'][:,:,1:],drawn.features[f'b_feature{i}'][:,:,:3]))

if __name__ == '__main__':
    unittest.main()
