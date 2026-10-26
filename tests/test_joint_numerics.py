"""Portable checks for R38 numerical contracts; not run in this lane."""
import unittest
import numpy as np
from src.harness.analysis.joint_dynamics.assessment import gaussian_crps, mixture_crps
from src.harness.analysis.joint_dynamics.baselines import fit_persistence

class JointNumericsTests(unittest.TestCase):
    def test_gaussian_zero_residual(self):
        self.assertAlmostEqual(float(gaussian_crps(np.zeros((1,3)),np.ones((1,3)),np.zeros((1,3)))[0,0]),(np.sqrt(2)-1)/np.sqrt(np.pi),places=12)
    def test_single_component_mixture(self):
        got=mixture_crps(np.zeros((1,1,3)),np.ones(1),np.zeros((1,3)),1.0)
        self.assertAlmostEqual(float(got[0,0]),float(gaussian_crps(np.zeros((1,3)),np.ones((1,3)),np.zeros((1,3)))[0,0]),places=12)
    def test_patient_weighted_persistence(self):
        e=np.zeros((3,3)); y=np.array([[1]*3,[1]*3,[3]*3],float)
        np.testing.assert_allclose(fit_persistence(e,y,['a','a','b']),np.sqrt(np.array([5,5,5])))

if __name__=='__main__': unittest.main()
