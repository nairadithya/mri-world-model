"""Optional second stage: quantity gate must pass before this module fits anything."""
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from src.patient_state.amount_first import LocationReadout, native_priors, rank_domain, allocate_counts, location_loss
from src.patient_state.structured import context
from src.patient_state.structured_training import load_context, save
from src.patient_state.state import write_json, digest
from train_amount_first import native_domain, reference, average, paired, balanced_groups, draw, rate


@torch.no_grad()
def evaluate_location(location, contexts, data, quantity_records, save_first=None):
    location.eval() if location is not None else None
    records=[]
    for item,q in zip(data,quantity_records):
        if item['key']!=q['key']:raise ValueError('Quantity/location case mismatch')
        current=load_context(contexts,item['pair']['history'][-1]);domain=torch.from_numpy(native_domain(current)).cuda()
        fields=torch.from_numpy(native_priors(current['predicted_segmentation'].numpy())).cuda()
        if location is not None:
            spatial,_=context(current,None,0,item['pair']['horizon'],'current')
            fields+=F.interpolate(location(spatial.cuda()),size=domain.shape,mode='trilinear',align_corners=False)[0]
        rankings=rank_domain(fields,domain)
        # Future mask is consumed only after fixed forecast rankings exist.
        y=torch.from_numpy(reference(item['pair']['target'])).cuda();truth=q['truth_ml']
        occupancies=torch.zeros((5,*domain.shape),device='cuda',dtype=torch.float32)
        dices=[]
        for index,counts in enumerate(q['sample_counts']):
            counts=torch.tensor(counts,device='cuda')
            pred=allocate_counts(rankings,counts,domain.shape,int(item['capacity']))
            values=[]
            for c in range(5):occupancies[c]+=(pred==c)/64
            for c in range(1,5):
                denom=int(counts[c-1])+int((y==c).sum())
                values.append(2*int(((pred==c)&(y==c)).sum())/denom if denom else 1.)
            dices.append(values)
            if index==0 and save_first is not None:
                from src.patient_state.pipeline import save_nifti
                save_nifti(save_first/(item['pair']['history'][-1]['stem']+'-h'+str(item['pair']['horizon'])+'.nii.gz'),pred.cpu().numpy(),current['row']['affine'])
        brier=sum((occupancies[c]-(y==c)).square().sum() for c in range(5))/y.numel()
        current_vol=q['cutoff_volumes_ml'];change=np.abs(np.array(truth)-current_vol)
        records.append({'key':q['key'],'patient':q['patient'],'horizon':q['horizon'],'truth_ml':truth,
            'expected_dice':np.mean(dices,axis=0).tolist(),'spatial_brier':float(brier),
            'persistence_dice':q['persistence_dice'],'absolute_change_from_current_estimate_ml':change.tolist(),
            'native_maps_verified':64,'quantities_unchanged':True})
    return records


def summarize_location(records,prior,quantities):
    summary={};all_pass=True
    for h in (12,24):
        rows=[r for r in records if r['horizon']==h];core=[r for r in rows if r['truth_ml'][1]>0]
        dice=average(core,lambda r:r['expected_dice'][1]);persist=average(core,lambda r:r['persistence_dice'][1])
        passed=dice>=persist and quantities[str(h)]['quantity_gate']['passed'];all_pass &= passed
        control={r['key']:r for r in prior}
        compartments={}
        for c in range(4):
            compartments[str(c+1)]={}
            for name,rs in [('all',rows),('nonempty',[r for r in rows if r['truth_ml'][c]>0]),('empty',[r for r in rows if r['truth_ml'][c]==0])]:
                compartments[str(c+1)][name]={'expected_dice':average(rs,lambda r:r['expected_dice'][c]),
                    'persistence_dice':average(rs,lambda r:r['persistence_dice'][c]),
                    'minus_persistence':paired(rs,lambda r:r['expected_dice'][c],lambda r:r['persistence_dice'][c]),
                    'minus_current_anatomy_allocation':paired(rs,lambda r:r['expected_dice'][c],lambda r:control[r['key']]['expected_dice'][c])}
        summary[str(h)]={'core_expected_dice':dice,'core_persistence_dice':persist,'eligible':passed,
            'compartments':compartments,'spatial_brier':average(rows,lambda r:r['spatial_brier']),
            'balanced_expected_dice':average(rows,lambda r:float(np.mean(r['expected_dice'])))}
    return summary,all_pass


def fit_location(args,quantity,control,data,quantity_records,result):
    if not result['quantity_gate_passed']:raise ValueError('Quantity gate required before spatial fitting')
    quantity.requires_grad_(False);quantity.zero_grad(set_to_none=True);quantity.eval()
    result['location_training_started']=True
    write_json(args.out/'results.json',result)
    prior=evaluate_location(None,args.contexts,data['selection'],quantity_records)
    write_json(args.out/'prior-allocation-cases.json',prior)
    torch.manual_seed(17);model=LocationReadout().cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    groups=balanced_groups(data['fitting']);history=[];best=None;started=time.monotonic()
    for step in range(1,6001):
        item=draw(groups,step);current=load_context(args.contexts,item['pair']['history'][-1],'cuda')
        target=load_context(args.contexts,item['pair']['target'],'cuda')
        spatial,_=context(current,None,0,item['pair']['horizon'],'current')
        rng=np.random.default_rng(17+step);axes=[i for i in range(3) if rng.random()<.5]
        if axes:
            spatial=spatial.flip([i+2 for i in axes]);current['sdf']=current['sdf'].flip([i+1 for i in axes])
            for k in ('target_occupancy','valid'):target[k]=target[k].flip([i+1 for i in axes])
        model.train();optimizer.zero_grad(set_to_none=True)
        for g in optimizer.param_groups:g['lr']=rate(step)
        loss=location_loss(model(spatial),current,target);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),12,error_if_nonfinite=True);optimizer.step()
        if any(p.grad is not None for p in quantity.parameters()):raise AssertionError('Location gradient reached quantity model')
        if step%100==0:print(json.dumps({'location_step':step,'loss':float(loss.detach()),'seconds':time.monotonic()-started}),flush=True)
        del current,target,spatial,loss
        if step%1000==0:
            records=evaluate_location(model,args.contexts,data['selection'],quantity_records)
            scores,eligible=summarize_location(records,prior,result['horizons'])
            ranking=float(np.mean([s['balanced_expected_dice'] for s in scores.values()]))
            record={'step':step,'eligible':eligible,'balanced_expected_dice':ranking,'horizons':scores}
            history.append(record);write_json(args.out/'location-selection.json',history)
            payload={'identity':result['identity'],'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'step':step,'quantity_sha256':result['quantity_checkpoint_sha256']}
            save(args.out/'location-last.pt',payload)
            write_json(args.out/('location-cases-'+str(step)+'.json'),records)
            # Brier and volume CRPS are identical at every spatial checkpoint because quantities are frozen;
            # an exact Dice tie therefore retains the earlier candidate.
            if eligible and (best is None or ranking>best):
                best=ranking;save(args.out/'location-selected.pt',payload)
                result['selected_checkpoint']={'step':step,'sha256':digest(args.out/'location-selected.pt')}
            result['location_selection']=history;result['status']='eligible_development_candidate' if best is not None else 'no_eligible_checkpoint'
            write_json(args.out/'results.json',result)
            print('LOCATION_SELECTION '+json.dumps(record),flush=True)
    result['location_training_seconds']=time.monotonic()-started
    write_json(args.out/'results.json',result)
