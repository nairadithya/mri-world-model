import unittest
import numpy as np
import torch
from src.patient_state.structured import weighted_quantile,VolumeChangePredictor
from src.patient_state.information_quantity import InformationQuantity
from src.patient_state.state_descriptors import DIMENSION,regional_descriptor,patch_geometry
from scripts.experiments.quantity.train_amount_first import draw,balanced_groups


class InformationContracts(unittest.TestCase):
    def test_equal_weight_boundary(self):
        x=torch.arange(40,dtype=torch.float32)[:,None];w=torch.ones(40)/40
        self.assertEqual(float(weighted_quantile(x,w,.95)),37.)
        self.assertEqual(float(weighted_quantile(x,w,.05)),1.)

    def test_padding_occupancy(self):
        seg=np.zeros((7,9,5),np.uint8);seg[:3,:4,:2]=2
        feature=torch.ones((128,8,8,8))
        vector,counts=regional_descriptor(feature,seg,np.ones(seg.shape,bool))
        self.assertEqual(counts[1],24);self.assertEqual(counts[-1],315)
        x=vector.reshape(6,257)
        self.assertTrue(np.array_equal(x[1,:128],np.ones(128)))
        self.assertTrue(np.array_equal(x[1,128:256],np.zeros(128)))
        self.assertEqual(x[0,-1],1)
        padded,origins=patch_geometry((181,217,181))
        self.assertEqual(padded,(192,224,192))
        self.assertTrue(all(all(o%4==0 for o in origin) for origin in origins))

    def test_seed_draws(self):
        data=[dict(pair=dict(horizon=h,patient=str(p)),key=(h,p,v)) for h in (12,24) for p in range(5) for v in range(3)]
        groups=balanced_groups(data)
        a=[draw(groups,i,17)['key'] for i in range(30)]
        self.assertEqual(a,[draw(groups,i,17)['key'] for i in range(30)])
        self.assertNotEqual(a,[draw(groups,i,23)['key'] for i in range(30)])

    def test_information_gates(self):
        torch.manual_seed(17);control=VolumeChangePredictor('current',compartments=4)
        model=InformationQuantity(control,current_mri=False,memory=0)
        x=torch.randn(1,17+2*DIMENSION+10);v=torch.ones(1,4)
        altered=x.clone();altered[:,17:]=torch.randn_like(altered[:,17:])*10
        a=model.embedding(x,v);b=model.embedding(altered,v)
        for u,w in zip(a,b):self.assertTrue(torch.equal(u,w))
        model.current_mri=True
        self.assertFalse(torch.equal(model.embedding(x,v)[0],model.embedding(altered,v)[0]))
        # The warm-start amount offsets remain scalar-only in every condition.
        for u,w in zip(model.embedding(x,v)[1:],a[1:]):self.assertTrue(torch.equal(u,w))


if __name__=='__main__':unittest.main()
