#!/usr/bin/env python3
"""Matched MU adaptation assessment, preserving both annotation conventions."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.patient_state.state import digest,write_json
from src.patient_state.structured import AnatomicalTransition,VolumeChangePredictor
from src.patient_state.structured_training import evaluate_case,load_context,bounded_selection,distribution_metrics


class NativePersistence:
    arm='current'
    scenarios=1
    compartments=4
    render=staticmethod(AnatomicalTransition.render)
    def parameters_for(self,x,s):
        return {'delta':x.new_zeros(1,4,1,1,1),'appearance':x.new_zeros(4,4),'weights':x.new_ones(1)}


def key(r):
    return r['patient'],r['cutoff'],r['target'],r['horizon']


def summarize_value(records,fn):
    grouped=defaultdict(list)
    for r in records:
        v=fn(r)
        if v is not None:grouped[r['patient']].append(float(v))
    v=np.array([np.mean(v) for _,v in sorted(grouped.items())])
    if not len(v):return {'mean':None,'patients':0,'patient_bootstrap95':None}
    rng=np.random.default_rng(17)
    samples=v[rng.integers(len(v),size=(4000,len(v)))].mean(1)
    return {'mean':float(v.mean()),'patients':len(v),'patient_bootstrap95':np.quantile(samples,[.025,.975]).tolist()}


def enrich(value,current_estimate,current_truth,target_truth):
    n=len(value['mean_volumes_ml'])
    error=(np.array(value['mean_volumes_ml'])-np.array(current_estimate)[:n])-(np.array(target_truth)[:n]-np.array(current_truth)[:n])
    value['compartment_change_error_ml']=error.tolist()
    value['volume_change_mae_ml']=float(np.abs(error).mean())
    value['original_three_volume_change_mae_ml']=float(np.abs(error[:3]).mean())
    for field,source in (('original_three_volume_crps_ml','compartment_crps_ml'),('original_three_volume_mae_ml','compartment_volume_error_ml')):
        a=np.array(value[source])[:3]
        value[field]=float(np.abs(a).mean() if 'mae' in field else a.mean())
    if 'legacy_ignored_cavity_original_three_dice' in value:
        value['matched_legacy_dice']=value['legacy_ignored_cavity_original_three_dice']
    elif value.get('dice') is not None:
        value['matched_legacy_dice']=value['dice']
    return value


METRICS=('matched_legacy_volume_crps_ml','matched_legacy_volume_mae_ml','matched_legacy_volume_change_mae_ml','native_original_three_dice','matched_legacy_dice','dice','original_three_volume_crps_ml',
         'original_three_volume_mae_ml','original_three_volume_change_mae_ml','volume_crps_ml',
         'volume_mae_ml','coverage90','width90_ml','interval_score90_ml','lesion_mae','brain_mae','gradient_mae','cavity_image_mae')


def summarize(records):
    horizons={}
    for h in (12,24):
        rows=[r for r in records if r['horizon']==h]
        if not rows:continue
        methods={}
        for name in rows[0]['methods']:
            metrics={m:summarize_value(rows,lambda r:r['methods'][name].get(m)) for m in METRICS}
            compartments={}
            n=4 if name in ('current','persistence','volume_current') else 3
            for c in range(1,n+1):
                def score(r,k):
                    m=r['methods'][name]
                    if k=='dice':return m.get('compartments',{}).get(str(c),{}).get('dice')
                    if k=='absent_target_fp_ml':
                        v=m.get('compartments',{}).get(str(c),{})
                        return v.get(k,v.get('empty_reference_fp_ml'))
                    a=m.get('compartment_'+k)
                    if a is None:return None
                    return abs(a[c-1]) if k in ('volume_error_ml','change_error_ml') else a[c-1]
                compartments[str(c)]={k:summarize_value(rows,lambda r,k=k:score(r,k)) for k in
                    ('dice','crps_ml','volume_error_ml','change_error_ml','coverage90','width90_ml','absent_target_fp_ml')}
                # Presence is based on reference annotation, never model prediction.
                for g in ('empty','nonempty'):
                    part=[r for r in rows if (r['target_volumes_ml'][c-1]>0)==(g=='nonempty')]
                    compartments[str(c)][g]={'targets':len(part),**{k:summarize_value(part,lambda r,k=k:score(r,k)) for k in ('dice','volume_error_ml','crps_ml','coverage90','width90_ml')}}
            methods[name]={'metrics':metrics,'compartments':compartments,
                'by_observed_change':{g:{'targets':len(part:=[r for r in rows if r['change_group']==g]),
                    'metrics':{m:summarize_value(part,lambda r,m=m:r['methods'][name].get(m)) for m in METRICS}} for g in ('stable','changed')}}
        contrasts={}
        for control in rows[0]['methods']:
            if control=='current':continue
            def delta(r,m):
                a=r['methods']['current'].get(m);b=r['methods'][control].get(m)
                return a-b if a is not None and b is not None else None
            comparable=METRICS if not control.startswith('preserved_') else ('matched_legacy_dice','matched_legacy_volume_crps_ml','matched_legacy_volume_mae_ml','matched_legacy_volume_change_mae_ml','lesion_mae','brain_mae','gradient_mae')
            contrasts[control]={m:summarize_value(rows,lambda r,m=m:delta(r,m)) for m in comparable}
        horizons[str(h)]={'targets':len(rows),'patients':len({r['patient'] for r in rows}),
                         'methods':methods,'current_minus_control':contrasts}
    return horizons


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('contexts','runs','preserved','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();torch.set_num_threads(4)
    identity=json.loads((a.contexts/'identity.json').read_text())
    if identity.get('compartments')!=4:raise ValueError('Expected explicit cavity state')
    models={'persistence':NativePersistence()};hashes={}
    for kind in ('anatomy','volume'):
        path=a.runs/(kind+'-current')/('last.pt' if a.smoke else 'best.pt')
        ck=torch.load(path,map_location='cpu',weights_only=True)
        if ck['config']['identity']!=identity or ck['config']['arm']!='current' or ck['config'].get('compartments')!=4:raise ValueError('Forecast identity mismatch')
        if ck['config']['smoke']!=a.smoke:raise ValueError('Smoke/selected checkpoint mismatch')
        net=(AnatomicalTransition if kind=='anatomy' else VolumeChangePredictor)('current',compartments=4).cuda().eval()
        net.load_state_dict(ck['model'])
        name='current' if kind=='anatomy' else 'volume_current';models[name]=net;hashes[name]=digest(path)
    old={}
    for path in a.preserved.glob('case-*.json'):
        r=json.loads(path.read_text())
        if r['dataset']=='MU':
            if key(r) in old:raise ValueError('Duplicate preserved target')
            old[key(r)]=(r,path)
    population=json.loads((a.contexts/'population.json').read_text())
    pairs=population['fitting'][:1] if a.smoke else population['selection']
    if not a.smoke and len(pairs)!=58:raise ValueError('Original selection targets changed')
    a.out.mkdir(parents=True,exist_ok=a.resume)
    receipt={'observation_context':identity,'models':hashes,'smoke':a.smoke,'population_sha256':digest(a.contexts/'population.json'),
             'definitions':'Native five-category MU, plus separate legacy comparison ignoring reference cavity voxels. Conditional legacy observation readout is a different view and is not substituted here.'}
    if (a.out/'identity.json').exists() and json.loads((a.out/'identity.json').read_text())!=receipt:raise ValueError('Evaluation changed')
    write_json(a.out/'identity.json',receipt);records=[];start=time.monotonic()
    with torch.inference_mode():
        for i,pair in enumerate(pairs):
            path=a.out/f'case-{i:04d}.json'
            if path.exists():records.append(json.loads(path.read_text()));continue
            cur=load_context(a.contexts,pair['history'][-1]);target=load_context(a.contexts,pair['target'])
            now=cur['predicted_volumes'].tolist();truth=target['target_volumes'].tolist();past=cur['target_volumes'].tolist()
            methods={name:enrich(evaluate_case(net,a.contexts,pair),now,past,truth) for name,net in models.items()}
            legacy_truth=target['legacy_target_volumes'];legacy_past=cur['legacy_target_volumes']
            for value in methods.values():
                scores=distribution_metrics(torch.tensor(value['scenario_volumes_ml'])[:,:3],torch.tensor(value['weights']),legacy_truth)
                for metric in ('volume_crps_ml','volume_mae_ml'):value['matched_legacy_'+metric]=scores[metric]
                change=(np.array(value['mean_volumes_ml'][:3])-now[:3])-(legacy_truth.numpy()-legacy_past.numpy())
                value['matched_legacy_volume_change_mae_ml']=float(np.abs(change).mean())
            if not np.allclose(methods['persistence']['mean_volumes_ml'],now,atol=1e-4,rtol=0):raise AssertionError('Persistence altered native current anatomy')
            # A point mass is a CRPS control, not trained uncertainty.
            methods['persistence'].update(coverage90=None,width90_ml=None,interval_score90_ml=None,
                compartment_coverage90=None,compartment_width90_ml=None,uncertainty_status='point persistence; no fitted interval')
            r={'patient':pair['patient'],'dataset':'MU','cutoff':pair['history'][-1]['visit'],'target':pair['target']['visit'],
               'horizon':pair['horizon'],'target_offset':pair['target_offset'],'target_volumes_ml':truth,
               'current_reference_volumes_ml':past,'current_estimated_volumes_ml':now,
               'change_group':'changed' if (np.abs(np.array(truth[:3])-past[:3])>np.maximum(2,.2*np.array(past[:3]))).any() else 'stable',
               'cavity_change_group':'changed' if abs(truth[3]-past[3])>max(2,.2*past[3]) else 'stable','methods':methods}
            if not a.smoke:
                preserved,source=old[key(r)]
                if preserved['target_offset']!=r['target_offset']:raise ValueError('Target matching changed')
                if not np.allclose(preserved['methods']['persistence']['target_volumes_ml'],legacy_truth.numpy(),atol=1e-4,rtol=0):raise ValueError('Reference volumes changed')
                for name in ('persistence','current','history','volume_current'):
                    value=json.loads(json.dumps(preserved['methods'][name]))
                    # Historical change metrics already use that model's observed baseline.
                    value['original_three_volume_change_mae_ml']=value.get('volume_change_mae_ml')
                    value['original_three_volume_crps_ml']=value['volume_crps_ml']
                    value['original_three_volume_mae_ml']=value['volume_mae_ml']
                    value['matched_legacy_dice']=value.get('dice')
                    for metric in ('volume_crps_ml','volume_mae_ml','volume_change_mae_ml'):value['matched_legacy_'+metric]=value.get(metric)
                    methods['preserved_'+name]=value
                r['preserved_record']={'name':source.name,'sha256':digest(source)}
            write_json(path,r);records.append(r)
            print(json.dumps({'complete':i+1,'total':len(pairs),'seconds':time.monotonic()-start}),flush=True)
        sensitivity=[]
        if not a.smoke:
            for pair in bounded_selection(pairs,patients_per_cohort=1):
                for perturbation in ('intensity_1.05','resolution_4mm'):
                    sensitivity.append({'patient':pair['patient'],'horizon':pair['horizon'],'perturbation':perturbation,
                        'metrics':evaluate_case(models['current'],a.contexts,pair,nuisance=perturbation)})
        write_json(a.out/'context-sensitivity.json',sensitivity)
    write_json(a.out/'aggregate.json',{'status':'smoke_completed' if a.smoke else 'completed','horizons':summarize(records),
        'scope':'reused development selection; no independent confirmation or history-benefit claim',
        'intervals':'90% finite-mixture intervals; coverage and width measured separately',
        'observation_tradeoff':'LUMIERE missing-FLAIR lesion MAE +2.9%; retained observation measurement, not a forecast-image synthesis claim',
        'nuisance_scope':'downstream observed-context perturbation, not full observation re-encoding',
        'reserved_images_opened':0,'seconds':time.monotonic()-start})


if __name__=='__main__':main()
