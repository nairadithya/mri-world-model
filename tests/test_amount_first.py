"""Numerical and conservation contracts for the amount-first forecasting path."""
import unittest
import torch
from src.patient_state.amount_first import (AmountFirst, ORDER, count_log_mass,
    sample_count, rank_domain, allocate_counts, fixed_uniforms)
from src.patient_state.structured import VolumeChangePredictor


class AmountFirstContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.control=VolumeChangePredictor('current',compartments=4).requires_grad_(False)
        # A fitted scalar has nonzero final weights; the architecture's blank initializer does not.
        with torch.no_grad():self.control.network[-1].weight.normal_(0,.01)
        self.model=AmountFirst(self.control)
        self.x=torch.randn(1,17);self.vol=torch.tensor([[1.,2.,3.,4.]])

    def test_backbone_is_adapted_not_frozen_with_control(self):
        self.assertTrue(all(p.requires_grad for p in self.model.backbone.parameters()))
        self.assertFalse(any(p.requires_grad for p in self.control.parameters()))
        loss,_=self.model.nll(self.x,self.vol,torch.tensor([[1,2,3,4]]),torch.tensor([100]))
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.model.backbone.parameters()),0)

    def test_discrete_likelihood_normalizes_at_capacity_and_extreme_tails(self):
        n=torch.arange(1,20)
        for mu in (-100.,0.,100.):
            probability=count_log_mass(n,torch.full_like(n,19),torch.full(n.shape,mu),torch.full(n.shape,.2)).exp()
            self.assertAlmostEqual(float(probability.sum()),1.,places=10)
            samples=sample_count(torch.ones(64),torch.full((64,),mu),torch.full((64,),.2),fixed_uniforms('test','cpu')[:,0])
            self.assertTrue(bool((samples==1).all()))

    def test_feasible_samples_and_absence(self):
        # All presence patterns and counts must remain feasible even for a four-voxel domain.
        u=fixed_uniforms('small-capacity','cpu')
        n=self.model.sample(self.x,self.vol,4,u)
        self.assertTrue(bool((n>=0).all()))
        self.assertTrue(bool((n.sum(1)<=4).all()))
        self.assertTrue(torch.equal(n,self.model.sample(self.x,self.vol,4,u)))
        for head in self.model.presence:
            with torch.no_grad():head.bias.fill_(-100)
        self.assertTrue(bool((self.model.sample(self.x,self.vol,4,u)==0).all()))

    def test_exact_exclusive_allocator_ties_and_infeasibility(self):
        domain=torch.ones(2,2,2,dtype=torch.bool);fields=torch.zeros(4,2,2,2)
        ranks=rank_domain(fields,domain);counts=torch.tensor([2,1,2,1])
        decoded=allocate_counts(ranks,counts,domain.shape,8)
        self.assertEqual(decoded.flatten().tolist(),[2,1,1,4,3,3,0,0])
        with self.assertRaises(ValueError):allocate_counts(ranks,torch.tensor([3,3,3,3]),domain.shape,8)
        with self.assertRaises(ValueError):
            self.model.nll(self.x,self.vol,torch.tensor([[3,3,3,3]]),torch.tensor([4]))

    def test_exact_joint_presence_marginals(self):
        for i,head in enumerate(self.model.presence):
            with torch.no_grad():head.weight.zero_();head.bias.fill_(i-2)
        expected=torch.sigmoid(torch.tensor([-2.,-1.,0.,1.]))[list(ORDER)]
        self.assertTrue(torch.allclose(self.model.marginal_presence(self.x,self.vol),expected,atol=1e-7))


if __name__=='__main__':unittest.main()
