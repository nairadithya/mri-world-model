"""Family replacement, tail gradients and deterministic CDF reference contracts."""
import math
import unittest
import torch
from src.patient_state.amount_first import AmountFirst,count_log_mass,sample_count,fixed_uniforms
from src.patient_state.core_cdf import core_parameters,core_cdf_scores
from src.patient_state.structured import VolumeChangePredictor


class GaussianCountContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(17)
        self.control=VolumeChangePredictor('current',compartments=4)
        with torch.no_grad():self.control.network[-1].weight.normal_(0,.01)
        self.x=torch.randn(1,17);self.v=torch.tensor([[1.,2.,3.,4.]])

    def test_identical_initial_weights_and_matched_log_variance(self):
        torch.manual_seed(17);a=AmountFirst(self.control,family='logistic')
        torch.manual_seed(17);b=AmountFirst(self.control,family='gaussian')
        for k in a.state_dict():self.assertTrue(torch.equal(a.state_dict()[k],b.state_dict()[k]),k)
        z=torch.tensor([[1.,0.,1.,1.]])
        for i in range(4):
            h,mu0,s0=a.embedding(self.x,self.v);ma,sa=a.count_parameters(h,z,torch.ones(1,i),mu0,s0,i)
            h,mu0,s0=b.embedding(self.x,self.v);mb,sb=b.count_parameters(h,z,torch.ones(1,i),mu0,s0,i)
            self.assertTrue(torch.equal(ma,mb));self.assertTrue(torch.allclose(sa.square()*math.pi**2/3,sb.square(),rtol=3e-7))
        self.assertTrue(torch.equal(a.marginal_presence(self.x,self.v),b.marginal_presence(self.x,self.v)))

    def test_gaussian_normalization_tail_sampling_and_nonzero_gradients(self):
        n=torch.arange(1,102)
        for loc in (-100.,0.,100.):
            mu=torch.tensor(loc,dtype=torch.double,requires_grad=True);scale=torch.tensor(.3,dtype=torch.double,requires_grad=True)
            logs=count_log_mass(n,torch.full_like(n,101),mu.expand_as(n),scale.expand_as(n),'gaussian')
            self.assertAlmostEqual(float(logs.exp().sum().detach()),1.,places=8)
            (-logs[16]).backward()
            self.assertTrue(torch.isfinite(mu.grad));self.assertTrue(torch.isfinite(scale.grad))
            self.assertGreater(abs(float(mu.grad)),0)
            u=fixed_uniforms('gaussian-tails','cpu')[:,0]
            for cap in (1,101,1500000):
                sampled=sample_count(torch.full((64,),cap),mu.expand(64),scale.expand(64),u,'gaussian')
                self.assertTrue(bool(((sampled>=1)&(sampled<=cap)).all()))

    def test_inverse_cdf_matches_discrete_probabilities(self):
        # Deterministic interior quantiles, no statistical tolerance or new evaluation stream.
        u=(torch.arange(10000,dtype=torch.double)+.5)/10000
        maximum=torch.full((10000,),19);mu=torch.full((10000,),2.);sigma=torch.full((10000,),1.)
        samples=sample_count(maximum,mu,sigma,u,'gaussian')
        n=torch.arange(1,20)
        pmf=count_log_mass(n,torch.full_like(n,19),torch.full((19,),2.),torch.ones(19),'gaussian').exp()
        observed=torch.bincount(samples,minlength=20)[1:].double()/10000
        self.assertLess(float((observed-pmf).abs().max()),.0002)

    def test_deterministic_core_scores_match_exhaustive_pmf(self):
        for family in ('logistic','gaussian'):
            model=AmountFirst(self.control,family=family)
            params=core_parameters(model,self.x,self.v,19)
            pmf=torch.zeros(20,dtype=torch.double);pmf[0]=params['absence_probability']
            for i in range(8):
                maximum=int(params['maximum'][i]);n=torch.arange(1,maximum+1)
                pmf[1:maximum+1]+=params['log_weights'][i].exp()*count_log_mass(n,torch.full_like(n,maximum),params['mu'][i].expand_as(n),params['scale'][i].expand_as(n),family).exp()
            self.assertAlmostEqual(float(pmf.sum()),1.,places=10)
            values=torch.arange(20,dtype=torch.double)
            for truth in (0,7,19):
                direct=((values-truth).abs()*pmf).sum()-.5*((values[:,None]-values[None,:]).abs()*pmf[:,None]*pmf[None,:]).sum()
                result=core_cdf_scores(model,self.x,self.v,19,truth,chunk=7)
                self.assertAlmostEqual(result['mean_ml'],float((values*pmf).sum())/1000,places=11)
                self.assertAlmostEqual(result['crps_ml'],float(direct)/1000,places=11)


if __name__=='__main__':unittest.main()
