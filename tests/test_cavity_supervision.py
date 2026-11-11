"""Numerical contracts for partial annotation and released-label transitions."""
import unittest
import numpy as np
import torch
from src.patient_state.losses import joint_loss
from scripts.patient_state.inspect_cavity_transitions import transition_record


class CavitySupervisionTests(unittest.TestCase):
    def test_unannotated_cavity_has_zero_loss_gradient(self):
        logits = torch.randn(1, 5, 4, 4, 4, requires_grad=True)
        image = torch.randn(1, 4, 4, 4, 4)
        labels = torch.randint(0, 4, (1, 4, 4, 4))
        support = torch.ones(1, 1, 4, 4, 4, dtype=torch.bool)
        loss, _ = joint_loss(logits, image, image, labels, support, support.expand_as(image), 4)
        loss.backward()
        self.assertEqual(float(logits.grad[:, 4].abs().max()), 0.)
        changed = logits.detach().clone(); changed[:, 4] += 100
        second, _ = joint_loss(changed, image, image, labels, support, support.expand_as(image), 4)
        self.assertEqual(float(loss.detach()), float(second.detach()))

    def test_cavity_is_supervised_when_released(self):
        logits = torch.zeros(1, 5, 4, 4, 4, requires_grad=True)
        image = torch.zeros(1, 4, 4, 4, 4)
        labels = torch.full((1, 4, 4, 4), 4)
        support = torch.ones(1, 1, 4, 4, 4, dtype=torch.bool)
        loss, _ = joint_loss(logits, image, image, labels, support, support.expand_as(image), 5)
        loss.backward()
        self.assertLess(float(logits.grad[:, 4].mean()), 0.)
        with self.assertRaises(ValueError):
            joint_loss(logits, image, image, labels, support, support.expand_as(image), 4)

    def test_cavity_outside_image_support_is_not_discarded(self):
        logits = torch.zeros(1, 5, 4, 4, 4, requires_grad=True)
        image = torch.zeros(1, 4, 4, 4, 4)
        labels = torch.zeros(1, 4, 4, 4, dtype=torch.long)
        labels[0, 0, 0, 0] = 4
        support = torch.ones(1, 1, 4, 4, 4, dtype=torch.bool)
        support[0, 0, 0, 0, 0] = False
        loss, _ = joint_loss(logits, image, image, labels, support, support.expand_as(image), 5)
        loss.backward()
        self.assertLess(float(logits.grad[0, 4, 0, 0, 0]), 0.)

    def test_exchange_not_silently_called_resolution(self):
        current = np.full((20, 20, 20), 2, dtype=np.int8)
        future = np.full_like(current, 4)
        result = transition_record(current, future, np.ones_like(current, dtype=bool))
        self.assertEqual(result['core_change_ml'], -8.)
        self.assertEqual(result['cavity_change_ml'], 8.)
        self.assertEqual(result['lost_core_to_cavity_fraction'], 1.)
        self.assertIn('core_cavity_exchange_review', result['flags'])
        self.assertIn('unreviewed', result['review_status'])


if __name__ == '__main__':
    unittest.main()
