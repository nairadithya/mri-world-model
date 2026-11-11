#!/usr/bin/env python3
"""One bounded MU amount-first build; quantity failure prevents location fitting."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.patient_state.amount_first import (AmountFirst, LocationReadout, fixed_uniforms,
    native_priors, rank_domain, allocate_counts, count_log_mass, sample_count, location_loss)
from src.patient_state.structured import VolumeChangePredictor, context
from src.patient_state.structured_training import load_context, distribution_metrics, save
from src.patient_state.state import digest, write_json

IDENTITY='mu-amount-first-v1'
NAMES=('enhancing','nonenhancing_core','flair_abnormality','cavity')


def case_key(pair):
    return '|'.join(map(str,(pair['patient'],pair['history'][-1]['visit'],pair['target']['visit'],pair['horizon'])))


def native_domain(current):
    row=current['row']
    support=np.asarray(np.load(row['support'],mmap_mode='r').transpose(1,2,0),dtype=bool)
    seg=current['predicted_segmentation'].cpu().numpy()
    if support.shape!=seg.shape or abs(abs(np.linalg.det(np.array(row['affine'])[:3,:3]))-1)>1e-6:
        raise ValueError('Expected shared native 1mm geometry')
    return support | (seg>0)


def reference(row):
    y=np.array(np.load(row['mask'],mmap_mode='r').transpose(1,2,0),dtype=np.int16)
    if row['supervised_classes']!=5 or not np.isin(y,(0,1,2,3,4)).all():
        raise ValueError('MU quantity likelihood requires known exclusive five-class annotation')
    return y


def prepare(contexts):
    population=json.loads((contexts/'population.json').read_text())
    if len(population['fitting'])!=87 or len(population['selection'])!=58:raise ValueError('Fixed pair population changed')
    membership=Path('/home/aditya/experiments/world-model-patient-state-20261105/mu/membership.json')
    if digest(membership)!='4df898761ecdb299bc87f99749fbebc8faab294043d454c4055850fcc676e2d6':raise ValueError('Membership changed')
    members=json.loads(membership.read_text()); result={}
    for group in ('fitting','selection'):
        result[group]=[]
        for pair in population[group]:
            if pair['patient'] not in members[group] or pair['patient'] in members['reserved']:raise ValueError('Unauthorized patient')
            current=load_context(contexts,pair['history'][-1]); domain=native_domain(current)
            _, scalar=context(current,None,0,pair['horizon'],'current')
            # Labels are read only on this supervision/evaluation side.
            y=reference(pair['target']);counts=np.bincount(y.ravel(),minlength=5)[1:]
            if y.shape!=domain.shape or pair['target']['affine']!=current['row']['affine']:raise ValueError('Pair geometry mismatch')
            if counts.sum()>domain.sum():raise ValueError('Infeasible reference quantity; cannot train by clipping')
            seg=current['predicted_segmentation'].numpy(); now=np.bincount(seg.ravel(),minlength=5)[1:]
            if not np.allclose(now/1000,current['predicted_volumes'],atol=1e-4):raise ValueError('Saved map/volume mismatch')
            pd=[]
            for c in range(1,5):
                denom=int(now[c-1]+counts[c-1]);pd.append(2*int(((seg==c)&(y==c)).sum())/denom if denom else 1.)
            result[group].append({'pair':pair,'key':case_key(pair),'random_key':'|'.join(map(str,(pair['patient'],pair['history'][-1]['visit'],pair['horizon']))),'scalar':scalar.cuda(),
                'current':torch.tensor(now/1000,dtype=torch.float32,device='cuda')[None],
                'truth':torch.tensor(counts,dtype=torch.long,device='cuda')[None],
                'capacity':torch.tensor([int(domain.sum())],device='cuda'),
                'outside_domain_counts':[int(((y==c)&~domain).sum()) for c in range(1,5)],
                'persistence_dice':pd})
        print('PREPARED '+group+' '+str(len(result[group])),flush=True)
    return result


def balanced_groups(data):
    groups=defaultdict(lambda:defaultdict(list))
    for item in data:groups[item['pair']['horizon']][item['pair']['patient']].append(item)
    return {h:list(g.values()) for h,g in groups.items()}


def draw(groups,step,run_seed=17):
    rng=np.random.default_rng(run_seed+step);keys=sorted(groups)
    patients=groups[keys[int(rng.integers(len(keys)))]]
    visits=patients[int(rng.integers(len(patients)))];return visits[int(rng.integers(len(visits)))]


def prevalence(data):
    result={}
    for horizon,patients in balanced_groups(data).items():
        result[horizon]=np.mean([np.mean([(x['truth'][0]>0).cpu().numpy() for x in visits],axis=0) for visits in patients],axis=0)
    return result


def rate(step,total=6000):
    return 5e-4*min(1,step/100)*(.01+.99*.5*(1+math.cos(math.pi*max(0,step-100)/max(1,total-100))))


def fit_quantity(model,data,out,steps=6000):
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-4,weight_decay=1e-4)
    groups=balanced_groups(data);start=time.monotonic();curve=[]
    for step in range(1,steps+1):
        item=draw(groups,step);optimizer.zero_grad(set_to_none=True)
        for g in optimizer.param_groups:g['lr']=rate(step,steps)
        loss,parts=model.nll(item['scalar'],item['current'],item['truth'],item['capacity'])
        if not torch.isfinite(loss):raise ArithmeticError('Nonfinite joint likelihood')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),12,error_if_nonfinite=True);optimizer.step()
        if step%100==0 or steps<=2:
            entry={'step':step,'loss':float(loss.detach()),**parts,'elapsed_seconds':time.monotonic()-start}
            curve.append(entry);print(json.dumps(entry),flush=True)
        if step%1000==0 or step==steps:
            save(out/'quantity-terminal.pt',{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,
                'identity':model.identity,'family':model.family,'terminal_candidate':step==6000,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()})
    write_json(out/'quantity-curve.json',curve)
    return {'updates':steps,'seconds':time.monotonic()-start}


def presence_scores(probability,truth):
    p=np.array(probability,dtype=float);z=np.array(truth,dtype=bool)
    correct=np.where(z,p,1-p)
    return {'brier':((p-z)**2).tolist(),
            'log_score':[float(-math.log(x)) if x>0 else None for x in correct],
            'infinite_log_score':(correct==0).tolist(),
            'false_absence':((p<.5)&z).astype(float).tolist(),
            'false_presence':((p>=.5)&~z).astype(float).tolist(),'probability':p.tolist()}


def metrics(samples,weights,truth):
    out=distribution_metrics(samples,weights,truth)
    out['mean_volume_mae_ml']=np.abs(out['compartment_volume_error_ml']).tolist()
    out['conditional_positive']={}
    for c in range(4):
        if truth[c]<=0:continue
        positive=samples[:,c]>0
        if not positive.any():out['conditional_positive'][str(c+1)]={'defined':False};continue
        w=weights[positive];w=w/w.sum()
        m=distribution_metrics(samples[positive,c:c+1],w,truth[c:c+1])
        out['conditional_positive'][str(c+1)]={'defined':True,'crps_ml':m['volume_crps_ml'],
            'mae_ml':m['volume_mae_ml'],'coverage90':m['coverage90'],'width90_ml':m['width90_ml']}
    # Proper multivariate energy score on the same joint quantities.
    first=(weights*torch.linalg.vector_norm(samples-truth,dim=1)).sum()
    second=(weights[:,None]*weights[None,:]*torch.cdist(samples,samples)).sum()
    out['joint_energy_ml']=float(first-.5*second)
    return out


@torch.no_grad()
def evaluate_quantity(model,control,data,prior,out):
    records=[]
    for item in data:
        pair=item['pair'];truth=item['truth'][0].float()/1000
        samples=model.sample(item['scalar'],item['current'],int(item['capacity']),fixed_uniforms(item['random_key']))
        p=model.marginal_presence(item['scalar'],item['current'])
        old,ow=control(item['scalar'],item['current'][0])
        nll,parts=model.nll(item['scalar'],item['current'],item['truth'],item['capacity'])
        record={'key':item['key'],'patient':pair['patient'],'horizon':pair['horizon'],
            'cutoff':pair['history'][-1]['visit'],'target':pair['target']['visit'],
            'truth_ml':truth.tolist(),'cutoff_volumes_ml':item['current'][0].tolist(),
            'capacity':int(item['capacity']),'outside_domain_counts':item['outside_domain_counts'],
            'sample_counts':samples.tolist(),'presence_probability':p.tolist(),'joint_nll':float(nll),**parts,
            'methods':{'amount_first':metrics(samples.float()/1000,torch.full((64,),1/64,device='cuda'),truth),
                       'scalar':metrics(old,ow,truth),'persistence':metrics(item['current'],torch.ones(1,device='cuda'),truth)},
            'presence':{},'persistence_dice':item['persistence_dice']}
        for name,prob in [('amount_first',p.cpu().numpy()),('fitting_prevalence',prior[pair['horizon']]),
                          ('current_persistence',(item['current'][0]>0).cpu().numpy())]:
            record['presence'][name]=presence_scores(prob,truth.cpu().numpy()>0)
        records.append(record)
    write_json(out/'quantity-cases.json',records)
    return records


def patient_values(records,fn):
    grouped=defaultdict(list)
    for r in records:
        value=fn(r)
        if value is not None and math.isfinite(value):grouped[r['patient']].append(value)
    return {p:float(np.mean(v)) for p,v in grouped.items()}


def average(records,fn):
    v=patient_values(records,fn)
    return float(np.mean(list(v.values()))) if v else None


def paired(records,a,b):
    patients=patient_values(records,lambda r:a(r)-b(r));values=np.array(list(patients.values()))
    if not len(values):return {'patients':0,'difference':None,'ci95':None}
    rng=np.random.default_rng(17);means=values[rng.integers(len(values),size=(4000,len(values)))].mean(1)
    return {'patients':len(values),'difference':float(values.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),
            'patients_worse':int((values>0).sum()),'patients_better':int((values<0).sum())}


def summarize(records):
    result={}
    for h in (12,24):
        rows=[r for r in records if r['horizon']==h];nonempty=[r for r in rows if r['truth_ml'][1]>0]
        methods={}
        for method in ('amount_first','scalar','persistence'):
            compartments={}
            for c,name in enumerate(NAMES):
                entry={}
                for subset,rs in [('all',rows),('nonempty',[r for r in rows if r['truth_ml'][c]>0]),('empty',[r for r in rows if r['truth_ml'][c]==0])]:
                    entry[subset]={'targets':len(rs),'patients':len({r['patient'] for r in rs})}
                    for metric in ('mean_volume_mae_ml','compartment_crps_ml','compartment_coverage90','compartment_width90_ml'):
                        entry[subset][metric]=average(rs,lambda r:r['methods'][method][metric][c])
                    if subset=='empty':entry[subset]['expected_fp_ml']=average(rs,lambda r:r['methods'][method]['mean_volumes_ml'][c])
                    if subset=='nonempty':
                        for key in ('crps_ml','mae_ml','coverage90','width90_ml'):
                            entry[subset]['conditional_positive_'+key]=average(rs,lambda r:r['methods'][method]['conditional_positive'].get(str(c+1),{}).get(key))
                        entry[subset]['conditional_undefined_targets']=sum(not r['methods'][method]['conditional_positive'].get(str(c+1),{}).get('defined',False) for r in rs)
                compartments[name]=entry
            methods[method]=compartments
        presence={}
        for method in ('amount_first','fitting_prevalence','current_persistence'):
            presence[method]={}
            for c,name in enumerate(NAMES):
                inf=sum(r['presence'][method]['infinite_log_score'][c] for r in rows)
                presence[method][name]={k:average(rows,lambda r:r['presence'][method][k][c]) for k in ('brier','false_absence','false_presence')}
                presence[method][name].update(log_score=None if inf else average(rows,lambda r:r['presence'][method]['log_score'][c]),
                    infinite_log_score_targets=inf,log_score_status='positive_infinity' if inf else 'finite')
                bins=[]
                for low in (0,.2,.4,.6,.8):
                    rs=[r for r in rows if low<=r['presence'][method]['probability'][c] and (r['presence'][method]['probability'][c]<low+.2 or low==.8)]
                    bins.append({'lower':low,'upper':low+.2,'targets':len(rs),'patients':len({r['patient'] for r in rs}),
                        'mean_probability':average(rs,lambda r:r['presence'][method]['probability'][c]),
                        'observed_frequency':average(rs,lambda r:float(r['truth_ml'][c]>0))})
                presence[method][name]['reliability']=bins
        q=methods['amount_first']['nonenhancing_core']['nonempty'];p=methods['persistence']['nonenhancing_core']['nonempty'];s=methods['scalar']['nonenhancing_core']['nonempty']
        gate={'targets':len(nonempty),'patients':len({r['patient'] for r in nonempty}),
            'mean_mae_ml':q['mean_volume_mae_ml'],'persistence_mae_ml':p['mean_volume_mae_ml'],
            'crps_ml':q['compartment_crps_ml'],'scalar_crps_ml':s['compartment_crps_ml']}
        gate['mae_pass']=gate['mean_mae_ml']<=gate['persistence_mae_ml']
        gate['crps_pass']=gate['crps_ml']<=gate['scalar_crps_ml'];gate['passed']=gate['mae_pass'] and gate['crps_pass']
        contrasts={}
        for c,name in enumerate(NAMES):
            contrasts[name]={}
            for subset,rs in [('all',rows),('nonempty',[r for r in rows if r['truth_ml'][c]>0])]:
                contrasts[name][subset]={}
                for comparator in ('scalar','persistence'):
                    for metric in ('mean_volume_mae_ml','compartment_crps_ml'):
                        contrasts[name][subset][metric+'_minus_'+comparator]=paired(rs,
                            lambda r:r['methods']['amount_first'][metric][c],lambda r:r['methods'][comparator][metric][c])
                for comparator in ('fitting_prevalence','current_persistence'):
                    contrasts[name][subset]['presence_brier_minus_'+comparator]=paired(rs,
                        lambda r:r['presence']['amount_first']['brier'][c],lambda r:r['presence'][comparator]['brier'][c])
        result[str(h)]={'targets':len(rows),'patients':len({r['patient'] for r in rows}),'quantity_gate':gate,
            'compartments':methods,'presence':presence,'paired_patient_intervals':contrasts,
            'outside_domain_target_voxels':np.sum([r['outside_domain_counts'] for r in rows],axis=0).tolist(),
            'joint_energy_ml':{m:average(rows,lambda r:r['methods'][m]['joint_energy_ml']) for m in methods}}
    return result


def smoke(model,control,item,contexts,out):
    started=time.monotonic();model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4)
    for _ in range(2):
        optimizer.zero_grad();loss,_=model.nll(item['scalar'],item['current'],item['truth'],item['capacity']);loss.backward();optimizer.step()
    save(out/'smoke-quantity.pt',{'model':model.state_dict()})
    restored=AmountFirst(control,family=model.family).cuda();restored.load_state_dict(torch.load(out/'smoke-quantity.pt',weights_only=True)['model'])
    u=fixed_uniforms(item['random_key']);a=model.sample(item['scalar'],item['current'],int(item['capacity']),u)
    b=restored.sample(item['scalar'],item['current'],int(item['capacity']),u)
    if not torch.equal(a,b):raise AssertionError('Save/reload altered samples')
    # Analytic discrete normalization, one-voxel support and both far-tail cases.
    for mu in (-100.,0.,5.,100.):
        n=torch.arange(1,102,device='cuda');m=torch.full_like(n,101)
        mass=count_log_mass(n,m,n.new_full(n.shape,mu,dtype=torch.double),n.new_full(n.shape,.3,dtype=torch.double),model.family).exp().sum()
        if not torch.allclose(mass,torch.ones_like(mass),atol=1e-9):raise AssertionError('Discrete probability did not normalize')
        draw=sample_count(torch.ones(64,device='cuda'),torch.full((64,),mu,device='cuda'),torch.full((64,),.3,device='cuda'),u[:,0],model.family)
        if not bool((draw==1).all()):raise AssertionError('One-voxel support failed')
    current=load_context(contexts,item['pair']['history'][-1]);domain=torch.from_numpy(native_domain(current)).cuda()
    fields=torch.from_numpy(native_priors(current['predicted_segmentation'].numpy())).cuda();rankings=rank_domain(fields,domain)
    for counts in a:
        decoded=allocate_counts(rankings,counts,domain.shape,int(domain.sum()))
        if bool((decoded[~domain]!=0).any()):raise AssertionError('Map outside cutoff domain')
    from src.patient_state.pipeline import save_nifti
    import nibabel as nib
    path=out/'smoke-native-sample0.nii.gz'
    first=allocate_counts(rankings,a[0],domain.shape,int(domain.sum())).cpu().numpy()
    save_nifti(path,first,current['row']['affine'])
    loaded=np.asanyarray(nib.load(path).dataobj);counts=np.bincount(loaded.astype('int64').ravel(),minlength=5)[1:]
    if not np.array_equal(counts,a[0].cpu().numpy()):raise AssertionError('Native NIfTI roundtrip changed counts')
    # Spatial train/save/reload smoke is engineering only; discard these weights.
    location=LocationReadout().cuda();spatial,_=context(current,None,0,item['pair']['horizon'],'current')
    cur={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in current.items()};target=load_context(contexts,item['pair']['target'],'cuda')
    opt=torch.optim.AdamW(location.parameters(),lr=5e-4);opt.zero_grad()
    sloss=location_loss(location(spatial.cuda()),cur,target);sloss.backward();opt.step()
    save(out/'smoke-location.pt',{'model':location.state_dict()})
    loc2=LocationReadout().cuda();loc2.load_state_dict(torch.load(out/'smoke-location.pt',weights_only=True)['model'])
    with torch.no_grad():
        if not torch.equal(location(spatial.cuda()),loc2(spatial.cuda())):raise AssertionError('Location replay changed')
    receipt={'passed':True,'quantity_updates':2,'location_smoke_updates':1,'native_maps_verified':64,
             'fixed_samples_replay_exact':True,'native_nifti_count_replay_exact':True,'discrete_normalization_and_tail_checks':True,
             'seconds':time.monotonic()-started,'quantity_smoke_weights_discarded_before_fit':True,'spatial_smoke_weights_discarded':True}
    write_json(out/'smoke.json',receipt);return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contexts',type=Path,required=True);p.add_argument('--scalar',type=Path,required=True)
    p.add_argument('--family',choices=('logistic','gaussian'),default='logistic')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke-only',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(17);np.random.seed(17)
    if digest(a.scalar)!='cf064a1356fc2f5fd0284ac324a837eef4660575d32e43923ebcdb9f13c885ae':raise ValueError('Frozen scalar identity changed')
    identity=json.loads((a.contexts/'identity.json').read_text())
    if identity['observation_sha256']!='d006424493442bd1bcd8fd5b15888b197aa3a95ef3f87aeed5df2f52e86ccacb':raise ValueError('Observation identity changed')
    run_identity='mu-amount-first-gaussian-v1' if a.family=='gaussian' else IDENTITY
    protocol={'identity':run_identity,'family':a.family,'variance_matching':'Gaussian sigma=logistic scale*pi/sqrt(3); identical raw scale parameterization',
        'core_numerical_reference':'deterministic full discrete CDF sum over feasible counts and eight core-present patterns; does not change 64-sample gates',
        'seed':17,'quantity_updates':6000,'location_updates_if_quantity_passes':6000,
        'samples':64,'random_stream':'SHA256 amount-first-v1:17:patient|cutoff|horizon; 52-bit midpoint uniforms',
        'representative_sample':0,'order':['core','enhancing','cavity','flair_abnormality'],'optimizer':'AdamW, lr5e-4, warmup100, cosine to1%, wd1e-4, clip12',
        'sampling':'uniform horizon, then patient, then target within patient/horizon','selection':'fixed terminal quantity; eligible spatial candidates every1000',
        'initialization':'copy scalar 17-32-32 backbone and frozen eight-scenario projection; count log median starts at log(max(1,1000*scalar mean)), logistic scale from scalar log variance floor0.1 cap2; trainable AR residual heads',
        'training_losses':'joint Bernoulli and conditional native-count NLL; no location gradient into quantity',
        'presence_initialization':'horizon-balanced fitting patient prevalence; no selection targets',
        'domain':'native cutoff image support union current predicted anatomy; no future expansion',
        'observation_context':identity,'scalar_sha256':digest(a.scalar),'population_sha256':digest(a.contexts/'population.json'),
        'source_sha256':{x.name:digest(x) for x in [Path(__file__),Path(__file__).parents[1]/'src/patient_state/amount_first.py',Path(__file__).with_name('amount_first_location.py'),Path(__file__).parents[1]/'src/patient_state/gaussian_count.py']},
        'reserved_patients_opened':0,'forecast_image_synthesis_claim':False}
    write_json(a.out/'protocol.json',protocol)
    control=VolumeChangePredictor('current',compartments=4).cuda()
    control.load_state_dict(torch.load(a.scalar,map_location='cuda',weights_only=True)['model']);control.eval();control.requires_grad_(False)
    data=prepare(a.contexts);prior=prevalence(data['fitting']);initial=np.mean(list(prior.values()),axis=0)
    receipt=smoke(AmountFirst(control,initial,family=a.family).cuda(),control,data['fitting'][0],a.contexts,a.out)
    print('SMOKE '+json.dumps(receipt),flush=True)
    if a.smoke_only:return
    torch.manual_seed(17)
    model=AmountFirst(control,initial,family=a.family).cuda()
    if a.family=='gaussian':
        with torch.random.fork_rng():
            torch.manual_seed(17)
            original=AmountFirst(control,initial,family='logistic').cuda()
        if any(not torch.equal(v,original.state_dict()[k]) for k,v in model.state_dict().items()):
            raise AssertionError('Gaussian did not start with original raw initialization')
        item=data['fitting'][0];z=item['scalar'].new_ones(1,4)
        hidden,mu0,s0=model.embedding(item['scalar'],item['current'])
        maximum_variance_error=0.
        for i in range(4):
            prefix=hidden.new_zeros(1,i)
            mg,sg=model.count_parameters(hidden,z,prefix,mu0,s0,i)
            ml,sl=original.count_parameters(hidden,z,prefix,mu0,s0,i)
            if not torch.equal(mg,ml) or not torch.allclose(sg.square(),sl.square()*math.pi**2/3,rtol=3e-7):
                raise AssertionError('Initial log-space variance mismatch')
            maximum_variance_error=max(maximum_variance_error,float(((sg.square()/(sl.square()*math.pi**2/3))-1).abs().max().detach()))
        write_json(a.out/'initialization-verification.json',{'identical_original_raw_parameters':True,'original_scalar_sha256':digest(a.scalar),
            'log_variance_matched':True,'maximum_relative_variance_error':maximum_variance_error,'warm_started_from_fitted_logistic':False})
        del original
    save(a.out/'quantity-initial.pt',{'model':model.state_dict(),'family':model.family,'identity':model.identity,'step':0})
    timing=fit_quantity(model,data['fitting'],a.out)
    records=evaluate_quantity(model,control,data['selection'],prior,a.out)
    summary=summarize(records);passed=all(v['quantity_gate']['passed'] for v in summary.values())
    result={'identity':run_identity,'status':'quantity_gate_passed' if passed else 'no_eligible_checkpoint',
        'quantity_gate_passed':passed,'location_training_started':False,'selected_checkpoint':None,
        'protocol':protocol,'smoke':receipt,'quantity_training':timing,'quantity_checkpoint_sha256':digest(a.out/'quantity-terminal.pt'),
        'horizons':summary,'forecast_usefulness_established':False,'infeasible_samples':[],
        'retained_image_completion_tradeoff':'Frozen observation model: previously measured approximately 2.9% higher LUMIERE missing-FLAIR lesion MAE; no retraining or substituted checkpoint'}
    write_json(a.out/'results.json',result)
    print('QUANTITY_GATE '+json.dumps({h:v['quantity_gate'] for h,v in summary.items()}),flush=True)
    if not passed:
        print('EARLY EXIT: quantity gate failed; no spatial fitting.',flush=True)
        return
    # Defined here only after the early gate; no quantity gradients in this phase.
    from amount_first_location import fit_location
    fit_location(a,model,control,data,records,result)


if __name__=='__main__':main()
