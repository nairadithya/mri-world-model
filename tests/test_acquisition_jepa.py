import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from src.patient_state.acquisition_data import observed_images,t1_support
from src.patient_state.acquisition import AcquisitionObservation
from src.patient_state.observation import FEATURES
from src.patient_state.losses import joint_loss


class AcquisitionBoundaryTests(unittest.TestCase):
    def test_unopened_arrays_do_not_enter_loader_or_support(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'images.npy';x=np.ones((4,4,4,4),np.float16);x[1,0]=0;np.save(path,x)
            row=dict(images=str(path),shape_xyz=[4,4,4],support='/must/not/be/read')
            before=observed_images(row,(1,));support=t1_support(row)
            x[[0,2,3]]=np.nan;np.save(path,x)
            self.assertTrue(torch.equal(before,observed_images(row,(1,))))
            self.assertTrue(torch.equal(support,t1_support(row)))

    def test_actual_set_fusion_order(self):
        model=AcquisitionObservation().eval()
        with torch.no_grad():
            evidence={i:[torch.randn(1,c,2,2,2) for c in FEATURES] for i in (1,0,2)}
            a=model.fuse(evidence);b=model.fuse({i:evidence[i] for i in (2,1,0)})
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,b)))
            self.assertTrue(all(not p.requires_grad for p in model.target_encoder.parameters()))

    def test_unknown_cavity_and_all_observed_loss(self):
        x=torch.randn(1,4,4,4,4);y=torch.zeros(1,4,4,4,dtype=torch.long);b=torch.ones(1,1,4,4,4,dtype=torch.bool)
        logits=torch.randn(1,5,4,4,4,requires_grad=True);prediction=torch.randn_like(x,requires_grad=True)
        loss,parts=joint_loss(logits,prediction,x,y,b,torch.zeros_like(x,dtype=torch.bool),4,allow_all_observed=True)
        loss.backward();self.assertEqual(parts['hidden_mae'],0.);self.assertTrue(torch.isfinite(loss));self.assertEqual(float(logits.grad[:,4].abs().sum()),0.)
        logits=torch.randn(1,5,4,4,4,requires_grad=True);b[:,:,:,2:]=False;y[:,:,3,3]=4
        loss,_=joint_loss(logits,prediction,x,y,b,torch.ones_like(x,dtype=torch.bool),5)
        loss.backward();self.assertGreater(float(logits.grad[:,4,:,3,3].abs().sum()),0.)


if __name__=='__main__':unittest.main()
