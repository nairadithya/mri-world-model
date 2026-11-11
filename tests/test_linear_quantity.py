import unittest
import numpy as np
import torch
from scipy.integrate import quad
from src.patient_state.linear_quantity import gaussian_log_interval,count_log_mass,fit_transform,transform


class LinearQuantityTests(unittest.TestCase):
    def test_narrow_interval_and_gradient(self):
        for mean in (-30.,-1.,0.,1.,30.):
            for width in (1e-7,1e-4,.02):
                m=torch.tensor([mean],dtype=torch.double,requires_grad=True);w=torch.tensor([width],dtype=torch.double)
                value=gaussian_log_interval(m,w)
                # Integrate relative to midpoint density, avoiding underflow.
                integral=quad(lambda t:np.exp(-mean*t-.5*t*t),-width/2,width/2,epsabs=1e-20)[0]
                expected=np.log(integral)-.5*mean**2-.5*np.log(2*np.pi)
                self.assertAlmostEqual(float(value.detach()),expected,places=9)
                self.assertTrue(torch.autograd.gradcheck(lambda z:gaussian_log_interval(z,w),(m,),eps=1e-5,atol=1e-7))

    def test_native_count_precision_and_normalization(self):
        for size in (1,7,57):
            n=torch.arange(1,size+1,dtype=torch.double)
            for mu in (-4.,1.,10.):
                p=count_log_mass(n,torch.full_like(n,size),torch.full_like(n,mu),torch.ones_like(n))
                self.assertAlmostEqual(float(p.exp().sum()),1.,places=11)
        n=torch.tensor([900000.],dtype=torch.double);maximum=torch.tensor([1000000.],dtype=torch.double)
        mu=torch.tensor([12.],dtype=torch.double,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda z:count_log_mass(n,maximum,z,torch.ones_like(z)),(mu,),eps=1e-5,atol=1e-7))

    def test_fold_transform_and_common_rms(self):
        rng=np.random.default_rng(17);rows=[]
        for i in range(24):
            raw=rng.normal(size=(6,257));raw[:,256]=0
            if i%3==0:raw[0,256]=1;raw[0,:256]=12345
            rows.append(dict(patient=str(i//2),stem=str(i),horizon=12+12*(i%2),current_counts=np.arange(1,5)*(i+1),descriptor=raw.ravel()))
        recipe=fit_transform(rows,True);s=fit_transform(rows,False)
        x=transform(rows,recipe).numpy();xs=transform(rows,s).numpy()
        np.testing.assert_allclose(x[:,:11],xs,atol=0,rtol=0)
        self.assertAlmostEqual(float(np.mean(x[:,11:]**2)),1.,places=12)
        altered=[dict(r,descriptor=r['descriptor'].copy()) for r in rows]
        for r in altered:
            raw=r['descriptor'].reshape(6,257)
            if raw[0,256]:raw[0,:256]=-999999
        np.testing.assert_allclose(transform(altered,recipe),x,atol=0,rtol=0)
        self.assertFalse(recipe['whiten'])


if __name__=='__main__':unittest.main()
