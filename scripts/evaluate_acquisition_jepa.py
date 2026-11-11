#!/usr/bin/env python3
"""Frozen subset enumeration, P-only selectors, V locking, then A assessment."""
import argparse,json,time
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.patient_state.acquisition import AcquisitionObservation,SUBSETS,PANELS
from src.patient_state.acquisition_data import observed_images,labels,state_metadata
from src.patient_state.acquisition_policy import (ORDERS,key,subset_state,descriptors,fit_preprocessing,fit_ridge,trajectory)
from src.patient_state.observation import JointObservation
from src.patient_state.state import digest,write_json
from train_acquisition_jepa import score

REFERENCE=Path('/home/aditya/experiments/world-model-patient-state-20261105/observation-cavity-v1/best.pt')


def jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,dict):return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [jsonable(v) for v in value]
    return value


@torch.inference_mode()
def enumerate_role(root,role,reference=REFERENCE):
    checkpoint=root/'representation/best.pt';model_hash=digest(checkpoint)
    if role=='A':
        lock=json.loads((root/'policy-lock.json').read_text())
        if lock['representation_sha256']!=model_hash:raise ValueError('Representation changed after lock')
        if digest(root/'selectors.json')!=lock['selectors_sha256']:raise ValueError('Selectors changed after lock')
    rows=json.loads((root/'data'/f'{role}.json').read_text());out=root/'policy'/role;out.mkdir(parents=True,exist_ok=True)
    model=AcquisitionObservation.from_checkpoint(checkpoint);baseline=JointObservation.from_checkpoint(reference) if role=='A' else None
    for number,row in enumerate(rows):
        path=out/(row['stem']+'.json')
        if path.exists():
            old=json.loads(path.read_text())
            if old['representation_sha256']!=model_hash:raise ValueError('Stale subset predictions')
            continue
        started=time.monotonic();x=observed_images(row,(0,1,2,3));y=labels(row)
        torch.cuda.synchronize();tick=time.monotonic();full=model.encode(x,state_metadata(row,(0,1,2,3)));torch.cuda.synchronize();encoding=time.monotonic()-tick
        record=dict(patient=row['patient'],visit=row['visit'],stem=row['stem'],role=role,representation_sha256=model_hash,subsets={},reference={},timing=dict(four_sequence_encoding_seconds=encoding),full_evidence_state_bytes=full.payload_bytes)
        for subset in SUBSETS:
            state=full if len(subset)==4 else subset_state(model,full,subset)
            tick=time.monotonic();output=model.read(state);torch.cuda.synchronize();read_seconds=time.monotonic()-tick
            scored=score(output['logits'],output['images'],x,y,state.memory['starting_t1_support'],5)
            tick=time.monotonic();descriptor=descriptors(model,state,output);torch.cuda.synchronize();query_seconds=time.monotonic()-tick
            record['subsets'][key(subset)]=dict(availability=state.metadata['availability'],score=scored,descriptor=jsonable(descriptor),state_bytes=state.payload_bytes,readout_seconds=read_seconds,descriptor_query_seconds=query_seconds)
            del output,state
        del full
        if baseline is not None:
            for panel in (*PANELS,(0,1,2,3)):
                current=observed_images(row,panel)
                state=baseline.encode(current.to(next(baseline.parameters()).device),state_metadata(row,panel));output=baseline.read(state)
                record['reference'][key(panel)]=score(output['logits'],output['images'],x,y,x[:,1:2]!=0,5)
                del state,output,current
        record['timing']['visit_total_seconds']=time.monotonic()-started;record['reference_sha256']=digest(reference) if baseline is not None else None
        write_json(path,record);print(json.dumps(dict(role=role,completed=number+1,total=len(rows),seconds=record['timing']['visit_total_seconds'])),flush=True)
    write_json(out/'complete.json',dict(role=role,visits=len(rows),patients=len({r['patient'] for r in rows}),representation_sha256=model_hash))


def load_role(root,role):
    rows=json.loads((root/'data'/f'{role}.json').read_text());result=[]
    for row in rows:
        r=json.loads((root/'policy'/role/(row['stem']+'.json')).read_text())
        if r['patient']!=row['patient'] or set(r['subsets'])!={key(s) for s in SUBSETS}:raise ValueError('Missing patient/action outcome')
        result.append(r)
    return result


def patient_means(rows,fn):
    values=defaultdict(list)
    for r in rows:
        v=fn(r)
        if v is not None:values[r['patient']].append(v)
    return {p:float(np.mean(v)) for p,v in sorted(values.items())}


def mean(rows,fn):
    values=patient_means(rows,fn)
    return float(np.mean(list(values.values()))) if values else None


def paired(rows,fn):
    values=patient_means(rows,fn);x=np.array(list(values.values()))
    if not len(x):return dict(patients=0,difference=None,ci95=None)
    rng=np.random.default_rng(17);means=x[rng.integers(len(x),size=(10000,len(x)))].mean(1)
    return dict(patients=len(x),difference=float(x.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),patient_differences=values)


def fit_selectors(root):
    p,v=load_role(root,'P'),load_role(root,'V');recipe=fit_preprocessing(p);models={};receipts={}
    for name,jepa in [('direct',False),('jepa',True)]:
        candidates=[]
        for penalty in (.1,1.,10.,100.):
            model=fit_ridge(p,recipe,penalty,jepa);utility=mean(v,lambda r:trajectory(r,recipe,model)['primary'])
            candidates.append(dict(model=model,V_primary=utility))
        best=max(candidates,key=lambda c:(c['V_primary'],c['model']['penalty']));models[name]=best['model'];receipts[name]=candidates
    fixed=[dict(order=list(order),V_primary=mean(v,lambda r:trajectory(r,order=order)['primary'])) for order in ORDERS]
    chosen=max(fixed,key=lambda c:c['V_primary'])
    write_json(root/'selectors.json',dict(recipe=recipe,models=models,fixed_order=chosen['order'],selection=receipts,fixed_order_selection=fixed,
        objective='patient mean of D at two and three observations on V',feature_scope='current predicted ROIs and T1 support only; hypothetical query features never actual evidence',fit_role='P'))
    write_json(root/'policy-lock.json',dict(representation_sha256=digest(root/'representation/best.pt'),selectors_sha256=digest(root/'selectors.json'),
        fixed_order=chosen['order'],assessment_opened=False,assessment_role_sha256=digest(root/'manifests/roles.json')))
    print(json.dumps(dict(locked=True,fixed_order=chosen['order'],penalties={n:m['penalty'] for n,m in models.items()})),flush=True)


def assess(root):
    lock=json.loads((root/'policy-lock.json').read_text());selectors=json.loads((root/'selectors.json').read_text())
    if digest(root/'selectors.json')!=lock['selectors_sha256']:raise ValueError('Locked policy changed')
    rows=load_role(root,'A');recipe=selectors['recipe'];methods=('fixed','direct','jepa','oracle')
    for r in rows:
        r['trajectories']={n:trajectory(r,recipe,selectors['models'][n]) for n in ('direct','jepa')}
        r['trajectories']['fixed']=trajectory(r,order=selectors['fixed_order'])
        r['trajectories']['oracle']=max([trajectory(r,order=order) for order in ORDERS],key=lambda t:t['primary'])
    aggregate={n:dict(primary=mean(rows,lambda r:r['trajectories'][n]['primary']),budgets=[mean(rows,lambda r:r['trajectories'][n]['budgets'][i]) for i in range(2)],
        acquisition_frequency={str(b+2):{str(q):mean(rows,lambda r:float(r['trajectories'][n]['actions'][b]==q)) for q in (0,3,2)} for b in range(2)}) for n in methods}
    contrasts={name:paired(rows,lambda r:r['trajectories'][left]['primary']-r['trajectories'][right]['primary']) for name,left,right in
        [('jepa_minus_fixed','jepa','fixed'),('jepa_minus_direct','jepa','direct'),('oracle_minus_fixed','oracle','fixed')]}
    for name,left,right in [('jepa_minus_fixed','jepa','fixed'),('jepa_minus_direct','jepa','direct')]:
        contrasts[name]['budgets']=[paired(rows,lambda r:r['trajectories'][left]['budgets'][i]-r['trajectories'][right]['budgets'][i]) for i in range(2)]
    tradeoffs={};harm=[]
    def at(r,method,budget):return r['subsets'][key([1]+r['trajectories'][method]['actions'][:budget-1])]['score']
    for budget in (2,3):
        tradeoffs[str(budget)]={}
        for c in range(1,5):
            k=str(c);present=[r for r in rows if at(r,'fixed',budget)['anatomy'][k]['nonempty_dice'] is not None];empty=[r for r in rows if r not in present]
            dice=paired(present,lambda r:at(r,'jepa',budget)['anatomy'][k]['nonempty_dice']-at(r,'fixed',budget)['anatomy'][k]['nonempty_dice'])
            fp=paired(empty,lambda r:at(r,'jepa',budget)['anatomy'][k]['empty_reference_fp_ml']-at(r,'fixed',budget)['anatomy'][k]['empty_reference_fp_ml'])
            if dice['ci95'] and dice['ci95'][1]<0:harm.append(dict(budget=budget,compartment=c,kind='lower_nonempty_dice'))
            if fp['ci95'] and fp['ci95'][0]>0:harm.append(dict(budget=budget,compartment=c,kind='higher_empty_false_positive_volume'))
            tradeoffs[str(budget)][k]=dict(nonempty_targets=len(present),empty_targets=len(empty),nonempty_dice={n:mean(present,lambda r:at(r,n,budget)['anatomy'][k]['nonempty_dice']) for n in methods},
                empty_fp_ml={n:mean(empty,lambda r:at(r,n,budget)['anatomy'][k]['empty_reference_fp_ml']) for n in methods},empty_fp_rate={n:mean(empty,lambda r:float(at(r,n,budget)['anatomy'][k]['empty_reference_fp_ml']>0)) for n in methods},jepa_minus_fixed_dice=dice,jepa_minus_fixed_fp_ml=fp)
    retention={};image_pass=True
    for panel in PANELS:
        k=key(panel);rs=[r for r in rows if r['reference'][k]['lesion_flair_mae'] is not None]
        deficit=paired(rs,lambda r:r['subsets'][k]['score']['lesion_flair_mae']-r['reference'][k]['lesion_flair_mae'])
        passed=bool(deficit['ci95'] and deficit['ci95'][1]<=.05);image_pass &= passed
        retention[k]=dict(lesion_deficit=deficit,passed=passed,
            new={metric:mean(rows,lambda r:r['subsets'][k]['score'][metric]) for metric in ('D','lesion_flair_mae','brain_flair_mae','gradient_flair_mae')},
            reference={metric:mean(rows,lambda r:r['reference'][k][metric]) for metric in ('D','lesion_flair_mae','brain_flair_mae','gradient_flair_mae')})
    native_nonempty=[r for r in rows if not r['reference']['0123']['foreground_empty']]
    anatomy=paired(native_nonempty,lambda r:r['subsets']['0123']['score']['D']-r['reference']['0123']['D']);anatomy_pass=bool(anatomy['ci95'] and anatomy['ci95'][0]>=-.05)
    jfixed=contrasts['jepa_minus_fixed'];jdirect=contrasts['jepa_minus_direct'];claim=bool(jfixed['difference']>=.02 and jfixed['ci95'][0]>0 and jdirect['ci95'][0]>0)
    result=dict(identity='acq-jepa-v1',status='completed',populations=json.loads((root/'manifests/roles.json').read_text())['counts'],assessment_visits=len(rows),assessment_patients=len({r['patient'] for r in rows}),
        acquisition=aggregate,primary_contrasts=contrasts,acquisition_joint_claim=claim,retained_image_completion=image_pass,retained_all_four_anatomy=anatomy_pass,
        practical_improvement_supported=claim and image_pass and anatomy_pass and not harm,compartment_tradeoffs=tradeoffs,exploratory_harm_flags=harm,fixed_panel_retention=retention,all_four_anatomy_difference=anatomy,
        entirely_empty_visits=[dict(patient=r['patient'],visit=r['visit'],trajectories=r['trajectories']) for r in rows if r['subsets']['1']['score']['foreground_empty']],
        full_state_bytes=patient_means(rows,lambda r:r['full_evidence_state_bytes']),mean_four_sequence_encoding_seconds=mean(rows,lambda r:r['timing']['four_sequence_encoding_seconds']),
        representation_runtime=json.loads((root/'representation/completed.json').read_text()),policy_selection=selectors['selection'],fixed_order_selection=selectors['fixed_order_selection'],lock=lock,
        limitations=['Offline sequence reveal on released upstream geometry','Unit cost is one reveal, not scan time or a contrast-injection schedule','One development seed; conditional bootstrap intervals','No calibrated uncertainty or biological forecasting claim','JEPA causal training effect unisolated; reference observer saw broader fitting population'],reserved_patients_opened=0,unopened_MU_scoring_patients_opened=0)
    write_json(root/'assessment.json',result)
    write_json(root/'assessment-patients.json',[dict(patient=r['patient'],visit=r['visit'],trajectories=r['trajectories']) for r in rows])
    print(json.dumps(dict(claim=claim,retained_image=image_pass,retained_anatomy=anatomy_pass,contrasts={k:{f:v[f] for f in ('difference','ci95','patients')} for k,v in contrasts.items()})),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('enumerate','fit-policy','assess'));p.add_argument('--root',type=Path,required=True);p.add_argument('--role',choices=('P','V','A'));a=p.parse_args();torch.set_num_threads(4)
    if a.stage=='enumerate':enumerate_role(a.root,a.role)
    elif a.stage=='fit-policy':fit_selectors(a.root)
    else:assess(a.root)


if __name__=='__main__':main()
