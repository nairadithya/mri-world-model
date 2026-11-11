"""Numerical readout contracts; run on mosaic-gpu with the patient-state suite."""
import unittest

import numpy as np
import torch

from src.patient_state.anatomy_readouts import anatomy_readouts, readout_contract
from src.patient_state.pipeline import volumes


class AnatomyReadoutTests(unittest.TestCase):
    def test_cavity_exclusion_is_conditional_not_relabeling(self):
        logits = torch.tensor([0., 1., 3., 2., 4.]).reshape(1, 5, 1, 1, 1)
        maps = anatomy_readouts(logits)
        self.assertEqual(maps['native'].item(), 4)
        self.assertEqual(maps['conditional_legacy'].item(), 2)
        labels = readout_contract(5, 'LUMIERE')['native']['labels']
        affine = np.diag([2., 3., 4., 1.])
        measured = volumes(maps['native'].numpy(), affine, labels)
        self.assertAlmostEqual(measured['resection_cavity'], .024)
        self.assertEqual(measured['necrotic_non_enhancing'], 0)
        self.assertIn('UNVALIDATED', readout_contract(5, 'LUMIERE')['native']['cavity_status'])

    def test_legacy_masking_is_unchanged_and_five_way_is_unmasked(self):
        logits = torch.tensor([0., 1., 3., 2., 4.]).reshape(1, 5, 1, 1, 1)
        support = torch.zeros((1, 1, 1, 1), dtype=torch.bool)
        for value in anatomy_readouts(logits[:, :4], support).values():
            self.assertEqual(value.item(), 0)
        self.assertEqual(anatomy_readouts(logits, support)['native'].item(), 4)


if __name__ == '__main__':
    unittest.main()
