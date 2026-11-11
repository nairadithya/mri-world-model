"""Run on mosaic-gpu alongside the real full-volume save/reload smoke."""
import copy
import unittest
import torch
from src.patient_state.structured import (AnatomicalTransition, context, weighted_crps,
                                         weighted_quantile, anatomical_loss, VolumeChangePredictor)
from src.patient_state.timing import timed_pairs


def record():
    sdf=torch.full((3,16,16,16),-4.)
    sdf[0,4:8,4:8,4:8]=1
    return {'sdf':sdf,'images':torch.randn(4,16,16,16),'support':torch.ones(1,16,16,16),
            'availability':torch.ones(4),'predicted_volumes':torch.tensor([.5,0,0]),
            'row':{'dataset':'LUMIERE'},'voxel_ml':.008}


class StructuredTests(unittest.TestCase):
    def test_cavity_distribution_is_coherent_and_target_blind(self):
        now=record()
        now['sdf']=torch.cat((now['sdf'],torch.full_like(now['sdf'][:1],-4.)))
        now['sdf'][3,9:12,9:12,9:12]=1
        now['predicted_volumes']=torch.tensor([.5,0.,0.,.216])
        x,s=context(now,None,0,24,'current')
        self.assertEqual(tuple(s.shape),(1,17))
        self.assertEqual(x.shape[1],14)
        now['target_volumes']=torch.full((4,),999.)
        now['target_occupancy']=torch.randn(5,16,16,16)
        x2,s2=context(now,None,0,24,'current')
        self.assertTrue(torch.equal(x,x2));self.assertTrue(torch.equal(s,s2))
        net=AnatomicalTransition('current',compartments=4)
        params=net.parameters_for(x,s)
        _,maps,images=net.render(params,now['sdf'],now['images'],now['support'])
        self.assertEqual(tuple(maps.shape),(8,5,16,16,16))
        self.assertTrue(torch.equal(maps.sum(1),torch.ones_like(maps[:,0])))
        volumes,w=VolumeChangePredictor('current',compartments=4)(s,now['predicted_volumes'])
        self.assertEqual(tuple(volumes.shape),(8,4));self.assertAlmostEqual(float(w.sum().detach()),1.)
        params['delta']=torch.zeros_like(params['delta'])
        _,maps,images=net.render(params,now['sdf'],now['images'],now['support'])
        seg=torch.cat((torch.zeros_like(now['sdf'][:1]),now['sdf'])).argmax(0)
        self.assertTrue(torch.equal(maps.argmax(1),seg[None].expand(8,-1,-1,-1)))
        self.assertTrue(torch.equal(images,now['images'][None].expand_as(images)))

    def test_mu_exact_days_not_directory_spacing(self):
        rows=[{'patient':'x','dataset':'MU','visit':f'Timepoint_{i}', 'visit_order':[i,0],
               'time_unit':'diagnosis_relative_days','diagnosis_relative_day':d} for i,d in ((1,0),(2,50),(3,134),(4,218))]
        pairs=timed_pairs(rows)
        self.assertEqual([p['target_offset'] for p in pairs],[0,0,0])
        for p in pairs:self.assertNotIn(p['target'],p['history'])
        rows[1]['time_unit']='visit_order_only_not_elapsed_time'
        self.assertEqual(timed_pairs(rows),[])

    def test_current_only_excludes_history_timing_and_labels(self):
        now,previous=record(),record()
        a=context(now,previous,12,24,'current')
        previous['sdf']*=100
        now['target_volumes']=torch.ones(3)*999
        now['target_occupancy']=torch.rand(4,16,16,16)
        b=context(now,previous,50,24,'current')
        for x,y in zip(a,b):self.assertTrue(torch.equal(x,y))

    def test_zero_boundary_change_reproduces_anatomy_and_image(self):
        now=record();sdf=now['sdf'];seg=torch.cat((torch.zeros_like(sdf[:1]),sdf),0).argmax(0)
        params={'delta':torch.zeros(2,3,2,2,2),'appearance':torch.randn(4,3),'weights':torch.tensor([.5,.5])}
        _,maps,images=AnatomicalTransition.render(params,sdf,now['images'],now['support'],seg)
        self.assertTrue(torch.equal(maps.argmax(1),seg[None].expand(2,-1,-1,-1)))
        self.assertTrue(torch.equal(images,now['images'][None].expand(2,-1,-1,-1,-1)))
        self.assertTrue(torch.equal(maps.sum(1),torch.ones_like(maps[:,0])))

    def test_volume_crps_is_learned_and_intervals_have_width(self):
        samples=torch.tensor([[0.,2.],[2.,4.]],requires_grad=True)
        weights=torch.tensor([.5,.5])
        loss=weighted_crps(samples,weights,torch.tensor([1.,3.])).mean()
        self.assertAlmostEqual(float(loss.detach()),.5)
        loss.backward();self.assertGreater(float(samples.grad.abs().sum()),0)
        self.assertTrue(torch.equal(weighted_quantile(samples,weights,.95)-weighted_quantile(samples,weights,.05),torch.tensor([2.,2.])))

    def test_full_brain_loss_reaches_spatial_and_distribution_parameters(self):
        now=record();target=copy.deepcopy(now)
        target['target_occupancy']=torch.zeros(4,16,16,16);target['target_occupancy'][0]=1
        target['target_occupancy'][0,3:10,3:10,3:10]=0;target['target_occupancy'][1,3:10,3:10,3:10]=1
        target['valid']=torch.ones(1,16,16,16);target['target_volumes']=torch.tensor([2.744,0,0])
        model=AnatomicalTransition()
        x,s=context(now,None,0,12,'history');params=model.parameters_for(x,s)
        loss,_=anatomical_loss(params,now,target);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for module in (model.spatial,model.global_head):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters()),0)

    def test_spatial_energy_does_not_turn_ignored_cavity_into_background(self):
        now=record();target=copy.deepcopy(now)
        target['valid']=torch.ones(1,16,16,16);target['valid'][:,:8,:8,:8]=0
        target['target_occupancy']=torch.zeros(4,16,16,16);target['target_occupancy'][0]=target['valid'][0]
        target['target_volumes']=torch.zeros(3)
        parameters={'delta':torch.zeros(2,3,16,16,16),'appearance':torch.zeros(4,3),'weights':torch.tensor([.5,.5])}
        _,a=anatomical_loss(parameters,now,target)
        parameters['delta'][:,1,:8,:8,:8]=20
        _,b=anatomical_loss(parameters,now,target)
        self.assertEqual(a['spatial_energy'],b['spatial_energy'])


if __name__=='__main__':unittest.main()
