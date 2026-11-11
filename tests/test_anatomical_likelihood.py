"""Regression checks for confidently wrong anatomical mixture predictions."""
import unittest

import torch
from torch.nn import functional as F

from src.patient_state.structured import anatomical_mixture_cross_entropy


class AnatomicalLikelihoodTests(unittest.TestCase):
    def test_absent_current_core_receives_corrective_gradient(self):
        logits = torch.tensor([0., -32., -32., -32., -32.]).reshape(1, 5, 1, 1, 1).requires_grad_()
        truth = torch.tensor([0., 0., 1., 0., 0.]).reshape(5, 1, 1, 1)
        valid = torch.ones(1, 1, 1, 1)
        weights = torch.ones(1)
        old = -(truth * logits.softmax(1)[0].clamp_min(1e-6).log()).sum()
        old_gradient = torch.autograd.grad(old, logits, retain_graph=True)[0]
        loss = anatomical_mixture_cross_entropy(logits, weights, truth, valid, valid)
        gradient = torch.autograd.grad(loss, logits)[0]
        self.assertEqual(float(old_gradient[0, 2]), 0.)
        self.assertLess(float(gradient[0, 2]), -.99)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_matches_probability_likelihood_without_flooring(self):
        torch.manual_seed(17)
        logits = torch.randn(3, 5, 2, 2, 2, dtype=torch.float64, requires_grad=True)
        raw_weights = torch.tensor([-.5, 0., .5], dtype=torch.float64, requires_grad=True)
        weights = raw_weights.softmax(0)
        truth = F.one_hot(torch.arange(8).reshape(2, 2, 2) % 5, 5).movedim(-1, 0).double()
        valid = torch.ones(1, 2, 2, 2, dtype=torch.float64)
        old = -(truth * (weights[:, None, None, None, None] * logits.softmax(1)).sum(0).log()).sum() / 8
        new = anatomical_mixture_cross_entropy(logits, weights, truth, valid, valid, raw_weights.log_softmax(0))
        torch.testing.assert_close(new, old, rtol=1e-12, atol=1e-12)
        old_gradients = torch.autograd.grad(old, (logits, raw_weights), retain_graph=True)
        new_gradients = torch.autograd.grad(new, (logits, raw_weights))
        for a, b in zip(old_gradients, new_gradients):
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)

    def test_fixed_outside_domain_remains_finite_and_has_no_gradient(self):
        logits = torch.zeros(2, 5, 1, 1, 2, requires_grad=True)
        truth = torch.zeros(5, 1, 1, 2)
        truth[0, 0, 0, 0] = 1
        truth[2, 0, 0, 1] = 1
        valid = torch.ones(1, 1, 1, 2)
        support = torch.zeros_like(valid)
        loss = anatomical_mixture_cross_entropy(logits, torch.tensor([.5, .5]), truth, valid, support)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(logits.grad.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
