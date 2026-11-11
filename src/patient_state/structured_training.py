"""Full-brain anatomical-distribution training and bounded 1mm selection."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from .data import load_visit
from .state import digest, write_json
from .timing import elapsed_weeks
from .structured import (AnatomicalTransition, VolumeChangePredictor, context,
                         anatomical_loss, weighted_crps, weighted_quantile)


def load_context(root, row, device='cpu'):
    value = torch.load(root/(row['stem']+'.pt'),map_location='cpu',weights_only=True)
    if value['row'] != row:
        raise ValueError('Context and population disagree')
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in value.items()}


def inputs(root, pair, arm, device='cuda'):
    current = load_context(root,pair['history'][-1],device)
    earlier = [r for r in pair['history'][:-1] if elapsed_weeks(r)<elapsed_weeks(pair['history'][-1])]
    previous = load_context(root,earlier[-1],device) if earlier and arm=='history' else None
    gap = elapsed_weeks(current['row'])-elapsed_weeks(previous['row']) if previous is not None else 0
    spatial, scalar = context(current,previous,gap,pair['horizon'],arm)
    return current, spatial, scalar


def bounded_selection(pairs, patients_per_cohort=6):
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[pair['dataset'],pair['horizon'],pair['patient']].append(pair)
    selected = []
    for dataset in ('LUMIERE','MU'):
        for h in (12,24):
            keys = [k for k in grouped if k[:2]==(dataset,h)]
            keys.sort(key=lambda k: hashlib.sha256(('structured-v1:'+k[2]).encode()).hexdigest())
            selected.extend(grouped[k][len(grouped[k])//2] for k in keys[:patients_per_cohort])
    return selected


def distribution_metrics(volumes, weights, truth):
    lo, hi = weighted_quantile(volumes,weights,.05), weighted_quantile(volumes,weights,.95)
    point = (weights[:,None]*volumes).sum(0)
    crps=weighted_crps(volumes,weights,truth)
    coverage=((truth>=lo)&(truth<=hi)).float()
    interval=hi-lo+20*(lo-truth).clamp_min(0)+20*(truth-hi).clamp_min(0)
    return {'volume_crps_ml':float(crps.mean()),
            'volume_mae_ml':float((point-truth).abs().mean()),
            'coverage90':float(coverage.mean()),'interval_score90_ml':float(interval.mean()),
            'width90_ml':float((hi-lo).mean()),'mean_volumes_ml':point.tolist(),
            'lower90_ml':lo.tolist(),'upper90_ml':hi.tolist(),
            'scenario_volumes_ml':volumes.tolist(),'weights':weights.tolist(),
            'compartment_crps_ml':crps.tolist(),'compartment_coverage90':coverage.tolist(),
            'compartment_width90_ml':(hi-lo).tolist(),'compartment_volume_error_ml':(point-truth).tolist()}


@torch.no_grad()
def evaluate_case(model, root, pair, full_resolution=True, nuisance=None):
    current, spatial, scalar = inputs(root,pair,model.arm)
    if nuisance == 'intensity_1.05':
        spatial=spatial.clone();spatial[:,:4]*=1.05
    elif nuisance == 'resolution_4mm':
        spatial=spatial.clone();spatial[:,:4]=F.interpolate(F.avg_pool3d(spatial[:,:4],2,ceil_mode=True),size=spatial.shape[2:],mode='trilinear',align_corners=False)
    elif nuisance is not None:
        raise ValueError('Unknown cutoff-context nuisance')
    target_context = load_context(root,pair['target'])
    truth = target_context['target_volumes'].cuda()
    if isinstance(model,VolumeChangePredictor):
        samples, weights = model(scalar,current['predicted_volumes'])
        return distribution_metrics(samples,weights,truth)
    parameters = model.parameters_for(spatial,scalar)
    if not full_resolution:
        loss, parts = anatomical_loss(parameters,current,{k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in target_context.items()})
        return {'loss':float(loss),**parts}
    # Selection always uses actual 1mm maps and exact derived compartment volumes.
    from scipy.ndimage import distance_transform_edt
    seg = current['predicted_segmentation'].cpu().numpy()
    fields = []
    for c in range(1,model.compartments+1):
        a = seg==c
        fields.append(np.clip(distance_transform_edt(a)-distance_transform_edt(~a),-32,32) if a.any() else np.full(seg.shape,-32.))
    sdf = torch.from_numpy(np.stack(fields).astype(np.float32)).cuda()
    now = load_visit(pair['history'][-1])
    future = load_visit(pair['target'],labels=True)
    images, support = now['images'][0].cuda(), now['support'][0].cuda()
    if model.compartments==4:support=support | (current['predicted_segmentation']>0)[None]
    y=future['labels'][0].cuda()
    valid=(y>=0) if model.compartments==4 else future['support'][0,0].cuda() & (y>=0)
    mean_probability = images.new_zeros((model.compartments+1,*images.shape[1:]))
    mean_image = torch.zeros_like(images)
    volumes = []
    weights = parameters['weights']
    for i in range(model.scenarios):
        single = {**parameters,'delta':parameters['delta'][i:i+1]}
        probability, maps, sample_image = model.render(single,sdf,images,support,current['predicted_segmentation'])
        volumes.append(maps[0,1:].sum((1,2,3))*.001)
        mean_probability += weights[i]*maps[0]
        mean_image += weights[i]*sample_image[0]
    values = torch.stack(volumes)
    result = distribution_metrics(values,weights,truth)
    predicted = mean_probability.argmax(0)
    dices, compartments = [], {}
    for c in range(1,model.compartments+1):
        p,t=(predicted==c)&valid,(y==c)&valid
        denom=int(p.sum()+t.sum())
        d=2*int((p&t).sum())/denom if denom else 1.
        dices.append(d)
        compartments[str(c)]={'dice':d,'point_volume_ml':int((predicted==c).sum())*.001,
                             'target_volume_ml':float(truth[c-1]),'absent_target_fp_ml':int(p.sum())*.001 if not t.any() else None}
    error=(mean_image-future['images'][0].cuda()).abs()
    brain=future['support'][0].cuda().expand_as(error)
    lesion=((y>0)&(y<4))[None].expand_as(error)&brain
    residual=mean_image-future['images'][0].cuda()
    grads=[]
    for axis in (1,2,3):
        aa,bb=[slice(None)]*4,[slice(None)]*4
        aa[axis],bb[axis]=slice(1,None),slice(None,-1)
        a,b=tuple(aa),tuple(bb);m=brain[a]&brain[b]
        grads.append(float((residual[a]-residual[b]).abs()[m].mean()))
    result.update(dice=float(np.mean(dices)),compartments=compartments,
                  brain_mae=float(error[brain].mean()),lesion_mae=float(error[lesion].mean()) if lesion.any() else None,
                  gradient_mae=float(np.mean(grads)),
                  anatomy_brier=float(sum(((mean_probability[c]-(y==c).float()).square()*valid).sum() for c in range(model.compartments+1))/valid.sum().clamp_min(1)))
    if model.compartments==4:
        result.update(anatomy_detail(predicted,y,valid,future['support'][0,0].cuda()))
        cavity=(y==4)[None].expand_as(error)&brain
        result['cavity_image_mae']=float(error[cavity].mean()) if cavity.any() else None
    return result


def anatomy_detail(predicted,y,valid,legacy_support=None):
    out={}
    for name,mask in (('native',valid),('legacy_ignored_cavity',valid & (y!=4) & (legacy_support if legacy_support is not None else valid))):
        scores={}
        for c in (1,2,3,4) if name=='native' else (1,2,3):
            p,t=(predicted==c)&mask,(y==c)&mask
            denom=int(p.sum()+t.sum())
            scores[str(c)]={'dice':2*int((p&t).sum())/denom if denom else 1.,
                           'target_nonempty':bool(t.any()),
                           'absent_target_fp_ml':int(p.sum())*.001 if not t.any() else None}
        out[name+'_compartments']=scores
        out[name+'_original_three_dice']=float(np.mean([scores[str(c)]['dice'] for c in (1,2,3)]))
    return out


@torch.no_grad()
def select(model, root, pairs):
    model.eval();records=[]
    for pair in pairs:
        metrics=evaluate_case(model,root,pair)
        records.append({'patient':pair['patient'],'dataset':pair['dataset'],'horizon':pair['horizon'],**metrics})
    # Equal cohort/horizon, then equal patients (one cutoff per patient/horizon).
    groups=defaultdict(list)
    for r in records:groups[r['dataset'],r['horizon']].append(r)
    summaries={f'{d}:{h}':{k:float(np.mean([r[k] for r in v if r.get(k) is not None])) for k in
                 ('volume_crps_ml','volume_mae_ml','coverage90','width90_ml','dice','brain_mae','lesion_mae','gradient_mae') if any(r.get(k) is not None for r in v)}
               for (d,h),v in groups.items()}
    score=np.mean([.02*s['volume_crps_ml']+(1-s['dice'] if 'dice' in s else 0) for s in summaries.values()])
    return {'score':float(score),'groups':summaries,'records':records}


def save(path, value):
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contexts',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--kind',choices=('anatomy','volume'),required=True)
    p.add_argument('--arm',choices=('current','history'),required=True)
    p.add_argument('--steps',type=int,default=6000)
    p.add_argument('--select-every',type=int,default=500)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--smoke-dataset',choices=('MU','LUMIERE'))
    p.add_argument('--resume',action='store_true')
    p.add_argument('--compartments',type=int,choices=(3,4),default=3)
    args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    population=json.loads((args.contexts/'population.json').read_text())
    fitting=population['fitting'];selection=bounded_selection(population['selection'])
    if args.smoke:
        fitting=[p for p in fitting if all((args.contexts/(r['stem']+'.pt')).exists() for r in ((p['history'] if args.arm=='history' else [p['history'][-1]])+[p['target']]))]
        if args.smoke_dataset:fitting=[p for p in fitting if p['dataset']==args.smoke_dataset]
        if not fitting:raise ValueError('No complete real fitting pair prepared yet')
    elif args.smoke_dataset:
        raise ValueError('Dataset restriction is only for engineering smoke tests')
    elif not all((args.contexts/(r['stem']+'.pt')).exists() for p in fitting+selection for r in ((p['history'] if args.arm=='history' else [p['history'][-1]])+[p['target']])):
        raise ValueError('Complete context preparation before cohort training')
    args.out.mkdir(parents=True,exist_ok=args.resume)
    if (args.out/'completed.json').exists():raise ValueError('Completed run is preserved')
    identity=json.loads((args.contexts/'identity.json').read_text())
    config={**vars(args),'contexts':str(args.contexts),'out':str(args.out),'identity':identity,
            'formulation':'eight learned anatomical scenarios, three boundary offsets and two learned regional bases per compartment',
            'compartments':args.compartments, 'training_grid_mm':2,'selection_grid_mm':1,'selection_cases':len(selection),
            'augmentation':'shared random spatial flips of observed context and supervised target, anatomy arm only',
            'source_sha256':{'training':digest(__file__),'model':digest(Path(__file__).with_name('structured.py'))}}
    config.pop('resume')
    if identity.get('compartments',3)!=args.compartments:raise ValueError('Context anatomy definition differs')
    model=(AnatomicalTransition(args.arm,compartments=args.compartments) if args.kind=='anatomy' else VolumeChangePredictor(args.arm,compartments=args.compartments)).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    steps=2 if args.smoke else args.steps
    offset,curve,best=0,[],float('inf')
    if args.resume:
        ck=torch.load(args.out/'last.pt',weights_only=True,map_location='cpu')
        if ck['config']!=config:raise ValueError('Resume configuration changed')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);offset=ck['step']
        torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state(ck['cuda_rng'])
        if (args.out/'curve.json').exists():
            curve=json.loads((args.out/'curve.json').read_text());best=min(r['score'] for r in curve)
    write_json(args.out/'config.json',config);write_json(args.out/'selection-population.json',selection)
    grouped=defaultdict(list)
    for pair in fitting:grouped[pair['dataset'],pair['horizon'],pair['patient']].append(pair)
    strata=defaultdict(list)
    for (d,h,pat),v in grouped.items():strata[d,h].append(v)
    keys=sorted(strata);times=[];started=time.monotonic()
    for step in range(offset+1,steps+1):
        tick=time.monotonic();model.train()
        rng=np.random.default_rng(args.seed+step)
        groups=strata[keys[int(rng.integers(len(keys)))]];group=groups[int(rng.integers(len(groups)))];pair=group[int(rng.integers(len(group)))]
        current,spatial,scalar=inputs(args.contexts,pair,args.arm)
        target=load_context(args.contexts,pair['target'],'cuda')
        if args.kind=='anatomy':
            axes=[axis for axis in range(3) if rng.random()<.5]
            if axes:
                spatial=spatial.flip([axis+2 for axis in axes])
                for key in ('images','sdf','support'):current[key]=current[key].flip([axis+1 for axis in axes])
                for key in tuple(k for k in ('images','support','image_support','valid','target_occupancy') if k in target):target[key]=target[key].flip([axis+1 for axis in axes])
        optimizer.zero_grad(set_to_none=True)
        lr=5e-4*min(1,step/100)*(.01+.99*.5*(1+math.cos(math.pi*max(0,step-100)/max(1,steps-100))))
        for group in optimizer.param_groups:group['lr']=lr
        if args.kind=='anatomy':
            parameters=model.parameters_for(spatial,scalar)
            loss,parts=anatomical_loss(parameters,current,target)
        else:
            volumes,weights=model(scalar,current['predicted_volumes'])
            loss=weighted_crps(volumes,weights,target['target_volumes']).mean()/50
            parts={'volume_crps_ml':float(loss.detach())*50}
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),12,error_if_nonfinite=True);optimizer.step()
        times.append(time.monotonic()-tick)
        if step%25==0 or args.smoke:
            print(json.dumps({'step':step,'loss':float(loss.detach()),**parts,'lr':lr,'mean_seconds':float(np.mean(times[-100:]))}),flush=True)
        payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':config,'step':step,
                 'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}
        if step%250==0 or step%args.select_every==0 or step==steps:save(args.out/'last.pt',payload)
        if not args.smoke and (step%args.select_every==0 or step==steps):
            summary=select(model,args.contexts,selection);summary['step']=step;curve.append(summary)
            write_json(args.out/'curve.json',curve)
            print('SELECTION '+json.dumps({k:v for k,v in summary.items() if k!='records'}),flush=True)
            if summary['score']<best:best=summary['score'];save(args.out/'best.pt',payload)
        if args.smoke and step==steps:
            restored=(AnatomicalTransition(args.arm,compartments=args.compartments) if args.kind=='anatomy' else VolumeChangePredictor(args.arm,compartments=args.compartments)).cuda()
            restored.load_state_dict(torch.load(args.out/'last.pt',weights_only=True)['model'])
            a=evaluate_case(model,args.contexts,pair);b=evaluate_case(restored,args.contexts,pair)
            if a!=b:raise AssertionError('Checkpoint full-volume replay changed')
            write_json(args.out/'smoke.json',{'passed':True,'updates':2,'full_volume_replay_exact':True,
                        'metrics':a,'mean_update_seconds':float(np.mean(times)),
                        'peak_cuda_bytes':torch.cuda.max_memory_allocated()})
        del current,spatial,scalar,target,loss
    write_json(args.out/'completed.json',{'steps':steps,'elapsed_seconds':time.monotonic()-started,
               'mean_update_seconds':float(np.mean(times)),'best_score':best if math.isfinite(best) else None,
               'forecast_usefulness_established':False,'boundary_review_required':True})


if __name__=='__main__':main()
