#!/usr/bin/env python3
"""Fixed 24-fit information program; terminal quantity candidates only."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
from src.patient_state.information_quantity import InformationQuantity
from src.patient_state.state_descriptors import DIMENSION
from src.patient_state.structured import VolumeChangePredictor
from src.patient_state.amount_first import fixed_uniforms
from src.patient_state.core_cdf import core_cdf_scores
from src.patient_state.state import write_json,digest
from src.patient_state.structured_training import save
from scripts.experiments.quantity.train_amount_first import balanced_groups,draw,prevalence,rate,case_key,metrics,presence_scores

ARMS={
 'old_scalar':dict(pool='old',current_mri=False,memory=0,reference=False),
 'old_mri':dict(pool='old',current_mri=True,memory=0,reference=False),
 'expanded_scalar':dict(pool='expanded',current_mri=False,memory=0,reference=False),
 'expanded_mri':dict(pool='expanded',current_mri=True,memory=0,reference=False),
 'reference_scalar':dict(pool='expanded',current_mri=False,memory=0,reference=True),
 'memory_metadata':dict(pool='expanded',current_mri=True,memory=1,reference=False),
 'memory_volumes':dict(pool='expanded',current_mri=True,memory=2,reference=False),
 'memory_mri':dict(pool='expanded',current_mri=True,memory=3,reference=False)}


def normalization(pairs,cache):
    stems=sorted({p['history'][-1]['stem'] for p in pairs})
    rows=np.stack([cache[s]['descriptor'] for s in stems]).reshape(-1,6,257)
    mean=np.zeros((6,257),np.float32);scale=np.ones_like(mean)
    for i in range(6):
        present=rows[:,i,256]==0
        if present.any():
            mean[i,:256]=rows[present,i,:256].mean(0,dtype=np.float64)
            std=rows[present,i,:256].std(0,dtype=np.float64)
            scale[i,:256]=np.where(std>1e-6,std,1.)
    return dict(mean=mean.ravel().tolist(),scale=scale.ravel().tolist(),fitting_stems=stems)


def standardized(raw,norm):
    x=(raw-np.array(norm['mean'],np.float32))/np.array(norm['scale'],np.float32)
    x=x.reshape(6,257);x[raw.reshape(6,257)[:,256]>0,:256]=0
    return x.ravel()


def prepare(data,descriptors,out,original_stage=False):
    population_path=data/('original-population.json' if original_stage else 'population.json')
    population=json.loads(population_path.read_text())
    original=json.loads((data/'original-population.json').read_text())
    assert population['fitting'][:87]==original['fitting'] and population['selection']==original['selection']
    write_json(out/('protocol-original-stage.json' if original_stage else 'protocol.json'),dict(identity='mu-frozen-state-information-v1',arms=ARMS,seeds=[17,23,31],updates=6000,
        original_stage_only=original_stage,population_sha256=digest(population_path),original_population_sha256=digest(data/'original-population.json'),
        observation_sha256='d006424493442bd1bcd8fd5b15888b197aa3a95ef3f87aeed5df2f52e86ccacb',
        family='gaussian',draw_seed='training run_seed + update; shared original evaluation streams keyed cutoff',
        gate='Core deterministic CDF mean MAE <= estimated persistence and CRPS <= full frozen scalar, both horizons, seed17 and seed average',
        eligible_ranking='Lowest equal-horizon patient-balanced joint NLL; reference-input diagnostic excluded',
        source_sha256={str(p.relative_to(Path(__file__).resolve().parents[3])):digest(p) for p in
            [Path(__file__),Path(__file__).resolve().parents[3]/'src/patient_state/information_quantity.py',Path(__file__).resolve().parents[3]/'src/patient_state/state_descriptors.py']}))
    rows={r['stem']:r for pairs in population.values() for p in pairs for r in p['history'][-2:]}
    cache={s:dict(np.load(descriptors/(s+'.npz'))) for s in rows if (descriptors/(s+'.npz')).exists()}
    norms={pool:normalization(pairs,cache) for pool,pairs in [('old',original['fitting']),('expanded',population['fitting'])]}
    normpath=out/('normalization-original-stage.json' if original_stage else 'normalization.json')
    if not original_stage and (out/'normalization-original-stage.json').exists():
        assert norms['old']==json.loads((out/'normalization-original-stage.json').read_text())['old']
    write_json(normpath,norms)
    previous_cases=json.loads(Path('/home/aditya/experiments/world-model-mu-gaussian-count-20261108/build/quantity-cases.json').read_text())
    fixed={r['key']:r for r in previous_cases}
    targets={};result={}
    for split,pairs in population.items():
        result[split]=[]
        for pair in pairs:
            now=pair['history'][-1];a=cache[now['stem']];target=pair['target']
            if target['stem'] not in targets:
                y=np.load(target['mask'],mmap_mode='r');targets[target['stem']]=np.array([(y==c).sum() for c in range(1,5)],np.int64)
            counts=targets[target['stem']]
            if counts.sum()>a['capacity']:raise ValueError('Infeasible target quantities')
            if target['affine']!=now['affine'] or target['shape_xyz']!=now['shape_xyz']:raise ValueError('Pair geometry changed')
            previous=pair['history'][-2] if len(pair['history'])>1 else None
            available=previous is not None and previous['stem'] in cache
            gap=(now['diagnosis_relative_day']-previous['diagnosis_relative_day'])/7 if available else 0
            if available and gap<=0:raise ValueError('Invalid previous visit timing')
            key=case_key(pair)
            item=dict(pair=pair,key=key,random_key='|'.join(map(str,(pair['patient'],now['visit'],pair['horizon']))),
                truth=torch.tensor(counts,device='cuda')[None],capacity=torch.tensor([int(a['capacity'])],device='cuda'),
                estimated=torch.tensor(a['estimated_counts']/1000,dtype=torch.float32,device='cuda')[None],
                reference=torch.tensor(a['reference_counts']/1000,dtype=torch.float32,device='cuda')[None],
                history_available=available,history_gap_weeks=gap,inputs={})
            for pool,norm in norms.items():
                for reference in (False,True):
                    # Keep original CPU log1p arithmetic for scalar/control equivalence.
                    v=torch.tensor(a['reference_counts' if reference else 'estimated_counts']/1000,dtype=torch.float32)
                    scalar=torch.cat((torch.log1p(v)/5,torch.zeros(4),torch.tensor([pair['horizon']/24,0]),torch.ones(4),torch.tensor([1.,1.,0.])))
                    past=np.zeros(8,np.float32);past_mri=np.zeros(DIMENSION,np.float32)
                    if available:
                        prev=cache[previous['stem']];pv=torch.tensor(prev['estimated_counts']/1000,dtype=torch.float32)
                        ev=torch.tensor(a['estimated_counts']/1000,dtype=torch.float32)
                        past=torch.cat((torch.log1p(pv)/5,(torch.log1p(ev)-torch.log1p(pv))/max(gap,1.))).numpy()
                        past_mri=standardized(prev['descriptor'],norm)
                    packed=np.concatenate((scalar.numpy(),standardized(a['descriptor'],norm),
                        np.array([float(available),np.log1p(gap)/5],np.float32),past,past_mri))
                    item['inputs'][(pool,reference)]=torch.tensor(packed,dtype=torch.float32,device='cuda')[None]
            if split=='selection':
                old=fixed[key]
                if not np.allclose(a['estimated_counts']/1000,old['cutoff_volumes_ml'],atol=1e-5):raise AssertionError('Current quantities changed')
                if not np.allclose(counts/1000,old['truth_ml'],atol=1e-5):raise AssertionError('Selection truth changed')
                item['old']=old
            result[split].append(item)
    save(out/('prepared-original.pt' if original_stage else 'prepared.pt'),result)
    return result


def arm_data(data,config):
    selected={}
    for split,items in data.items():
        selected[split]=[]
        for item in (items[:87] if split=='fitting' and config['pool']=='old' else items):
            x=dict(item)
            x['scalar']=item['inputs'][(config['pool'],config['reference'])]
            x['current']=item['reference' if config['reference'] else 'estimated']
            selected[split].append(x)
    return selected


def fit(model,data,out,seed,steps=6000):
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-4,weight_decay=1e-4)
    groups=balanced_groups(data);start=time.monotonic();curve=[];draws=[]
    for step in range(1,steps+1):
        item=draw(groups,step,seed);draws.append(item['key']);optimizer.zero_grad(set_to_none=True)
        for g in optimizer.param_groups:g['lr']=rate(step,6000)
        loss,parts=model.nll(item['scalar'],item['current'],item['truth'],item['capacity'])
        if not torch.isfinite(loss):raise ArithmeticError('Nonfinite likelihood')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),12,error_if_nonfinite=True);optimizer.step()
        if step%1000==0 or steps<=2:
            curve.append(dict(step=step,loss=float(loss.detach()),seconds=time.monotonic()-start,**parts));print(json.dumps(curve[-1]),flush=True)
    save(out/'quantity-terminal.pt',dict(model=model.state_dict(),step=steps,seed=seed,family='gaussian',identity=model.identity))
    write_json(out/'training.json',dict(curve=curve,seconds=time.monotonic()-start,updates=steps,draws=draws))


@torch.no_grad()
def evaluate(model,control,data,prior,out):
    records=[];start=time.monotonic()
    for item in data:
        old=item['old'];pair=item['pair'];truth=item['truth'][0].float()/1000
        samples=model.sample(item['scalar'],item['current'],int(item['capacity']),fixed_uniforms(item['random_key']))
        p=model.marginal_presence(item['scalar'],item['current']);loss,parts=model.nll(item['scalar'],item['current'],item['truth'],item['capacity'])
        deterministic=core_cdf_scores(model,item['scalar'],item['current'],int(item['capacity']),int(item['truth'][0,1]))
        record={k:old[k] for k in ('key','patient','horizon','cutoff','target','outside_domain_counts','persistence_dice')}
        record.update(truth_ml=truth.tolist(),cutoff_volumes_ml=item['current'][0].tolist(),
            estimated_current_ml=item['estimated'][0].tolist(),reference_current_ml=item['reference'][0].tolist(),
            capacity=int(item['capacity']),history_available=item['history_available'],history_gap_weeks=item['history_gap_weeks'],
            sample_counts=samples.tolist(),presence_probability=p.tolist(),joint_nll=float(loss),**parts,core_cdf=deterministic,
            methods=dict(amount_first=metrics(samples.float()/1000,torch.full((64,),1/64,device='cuda'),truth),
                scalar=old['methods']['scalar'],persistence=old['methods']['persistence'],
                reference_persistence=metrics(item['reference'],torch.ones(1,device='cuda'),truth)),presence={})
        for name,prob in [('amount_first',p.cpu().numpy()),('fitting_prevalence',prior[pair['horizon']]),('current_persistence',(item['estimated'][0]>0).cpu().numpy())]:
            record['presence'][name]=presence_scores(prob,truth.cpu().numpy()>0)
        records.append(record)
    write_json(out/'quantity-cases.json',records)
    write_json(out/'evaluation.json',dict(seconds=time.monotonic()-start,targets=len(records),quantity_checkpoint_sha256=digest(out/'quantity-terminal.pt')))
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--prepare',action='store_true');p.add_argument('--original-stage',action='store_true')
    args=p.parse_args();root=args.root;torch.set_num_threads(4);root.mkdir(parents=True,exist_ok=True)
    if args.prepare:prepare(root/'data',root/'descriptors',root,args.original_stage);return
    data=torch.load(root/('prepared-original.pt' if args.original_stage else 'prepared.pt'),weights_only=False)
    control=VolumeChangePredictor('current',compartments=4).cuda().eval()
    scalar=Path('/home/aditya/experiments/world-model-mu-cavity-forecast-20261107/runs/volume-current/best.pt')
    if digest(scalar)!='cf064a1356fc2f5fd0284ac324a837eef4660575d32e43923ebcdb9f13c885ae':raise ValueError('Warm start changed')
    control.load_state_dict(torch.load(scalar,map_location='cuda',weights_only=True)['model'])
    control.requires_grad_(False)
    for arm,config in ARMS.items():
        if args.smoke and arm!='memory_mri':continue
        if args.original_stage and not args.smoke and config['pool']!='old':continue
        config=dict(config)
        if args.original_stage and args.smoke:config['pool']='old'
        selected=arm_data(data,config);prior=prevalence(selected['fitting']);initial=np.mean(list(prior.values()),axis=0)
        for seed in ((17,) if args.smoke else (17,23,31)):
            out=root/('smoke' if args.smoke else 'runs')/f'{arm}-s{seed}';out.mkdir(parents=True,exist_ok=True)
            if (out/'evaluation.json').exists():continue
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=InformationQuantity(control,initial,current_mri=config['current_mri'],memory=config['memory']).cuda()
            write_json(out/'configuration.json',dict(arm=arm,seed=seed,**config,scalar_sha256=digest(scalar)))
            if not (out/'quantity-terminal.pt').exists():
                save(out/'quantity-initial.pt',dict(model=model.state_dict(),seed=seed,step=0))
                print('FIT '+arm+' '+str(seed),flush=True);fit(model,selected['fitting'],out,seed,2 if args.smoke else 6000)
            else:model.load_state_dict(torch.load(out/'quantity-terminal.pt',weights_only=True)['model'])
            model.eval()
            if args.smoke:
                clone=InformationQuantity(control,current_mri=True,memory=3).cuda();clone.load_state_dict(torch.load(out/'quantity-terminal.pt',weights_only=True)['model'])
                item=selected['selection'][0];u=fixed_uniforms(item['random_key'])
                assert torch.equal(model.sample(item['scalar'],item['current'],int(item['capacity']),u),clone.sample(item['scalar'],item['current'],int(item['capacity']),u))
                # Named inputs have no target field: changing supervision cannot change samples.
                changed=dict(item,truth=torch.zeros_like(item['truth']))
                assert torch.equal(model.sample(item['scalar'],item['current'],int(item['capacity']),u),model.sample(changed['scalar'],changed['current'],int(changed['capacity']),u))
                evaluate(model,control,selected['selection'][:1],prior,out)
                write_json(out/'smoke.json',dict(train_steps=2,save_reload_exact=True,cutoff_only_samples=True,real_data=True))
            else:evaluate(model,control,selected['selection'],prior,out)
            print('DONE '+arm+' '+str(seed),flush=True)


if __name__=='__main__':main()
