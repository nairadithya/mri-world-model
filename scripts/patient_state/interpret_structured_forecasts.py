#!/usr/bin/env python3
"""Bounded, no-fit interpretation of the existing MU forecast population."""
import argparse
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from src.patient_state.data import load_visit,catalogue
from src.patient_state.state import digest,write_json
from src.patient_state.structured_training import inputs,load_context
from src.patient_state.structured_inference import load_transition,signed_distances

DOMAINS=('unchanged_reference','changed_reference','current_cavity','other_current_unscored')


def dice_counts(counts):
    tp,fp,fn=counts
    return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.


def counts(pred,truth,mask):
    return np.array([np.count_nonzero(pred&truth&mask),np.count_nonzero(pred&~truth&mask),
                     np.count_nonzero(~pred&truth&mask)],dtype=np.int64)


def attribution(before,after):
    """Exact additive Dice attribution over disjoint reference-defined domains.

    Shapley averaging handles Dice's nonadditive denominator and empty/empty=1.
    These are retrospective reference partitions, never model inputs or causes.
    """
    n=len(before);scores={}
    for bits in itertools.product((0,1),repeat=n):
        scores[bits]=dice_counts(sum((after[i] if b else before[i]) for i,b in enumerate(bits)))
    result=[]
    for i in range(n):
        value=0.
        for bits in scores:
            if bits[i]:continue
            k=sum(bits);changed=list(bits);changed[i]=1
            weight=math.factorial(k)*math.factorial(n-k-1)/math.factorial(n)
            value+=weight*(scores[tuple(changed)]-scores[bits])
        result.append(value)
    if not np.isclose(sum(result),scores[(1,)*n]-scores[(0,)*n],atol=1e-12):
        raise AssertionError('Attribution does not conserve Dice gain')
    return dict(zip(DOMAINS,result))


def patient_mean(rows,key):
    groups=defaultdict(list)
    for r in rows:
        if r.get(key) is not None:groups[r['patient']].append(r[key])
    return float(np.mean([np.mean(v) for v in groups.values()])) if groups else None


def interval(rows,key):
    groups=defaultdict(list)
    for r in rows:
        if r.get(key) is not None:groups[r['patient']].append(r[key])
    if not groups:return {'mean':None,'patient_bootstrap95':None,'patients':0,'targets':len(rows)}
    values=np.array([np.mean(v) for _,v in sorted(groups.items())]);rng=np.random.default_rng(17)
    if len(values)==1:
        return {'mean':float(values[0]),'patient_bootstrap95':None,'patients':1,'targets':len(rows),
                'warning':'One patient cannot support a population uncertainty interval'}
    draws=values[rng.integers(len(values),size=(4000,len(values)))].mean(1)
    return {'mean':float(values.mean()),'patient_bootstrap95':np.quantile(draws,[.025,.975]).tolist(),
            'patients':len(values),'targets':len(rows)}


def analyze_compartment(p,f,y0,y1,s0,s1,c,record,method):
    valid0=s0&(y0>=0);valid1=s1&(y1>=0);common=valid0&valid1
    b=p==c;a=f==c;t0=y0==c;t1=y1==c
    domains=[common&(t0==t1),common&(t0!=t1),valid1&s0&(y0==-1),valid1&~valid0&~(s0&(y0==-1))]
    before=[counts(b,t1,m) for m in domains];after=[counts(a,t1,m) for m in domains]
    d0=dice_counts(sum(before));d1=dice_counts(sum(after))
    v0=np.count_nonzero(t0&valid0)/1000;v1=np.count_nonzero(t1&valid1)/1000
    group=('persistent_nonempty' if v0 and v1 else 'appeared' if v1 else 'resolved' if v0 else 'both_empty')
    a0=dice_counts(counts(a,t0,common));b0=dice_counts(counts(b,t0,common))
    af=dice_counts(counts(a,t1,common));bf=dice_counts(counts(b,t1,common))
    oracle=dice_counts(counts(t0,t1,common))
    ss=record['methods'][method];lo=ss['lower90_ml'][c-1];hi=ss['upper90_ml'][c-1]
    distmean=ss['mean_volumes_ml'][c-1]
    result={'compartment':c,'presence_group':group,'current_reference_ml':v0,'future_reference_ml':v1,
        'future_empty':v1==0,'future_at_most_0_1ml':v1<=.1,'meaningful_volume_change':abs(v1-v0)>max(2,.2*v0),
        'persistence_dice':d0,'forecast_dice':d1,'gain':d1-d0,
        'current_dice_persistence':b0,'current_dice_forecast':a0,'current_anatomy_gain':a0-b0,
        'common_future_gain':af-bf,'future_specific_gain':(af-bf)-(a0-b0),
        'reference_persistence_dice':oracle,'forecast_common_dice':af,'gain_over_reference_persistence':af-oracle,
        'current_cavity_prediction_ml':np.count_nonzero(b&s0&(y0==-1))/1000,
        'forecast_in_current_cavity_ml':np.count_nonzero(a&s0&(y0==-1))/1000,
        'current_cavity_future_scored_ml':np.count_nonzero(domains[2])/1000,
        'current_cavity_reference_both_ignored_ml':np.count_nonzero(s0&s1&(y0==-1)&(y1==-1))/1000,
        'coverage90':float(lo-1e-6<=v1<=hi+1e-6),
        'below_interval':float(v1<lo-1e-6),'above_interval':float(v1>hi+1e-6),
        'width90_ml':hi-lo,'lower90_ml':lo,'upper90_ml':hi,
        'volume_crps_ml':ss['compartment_crps_ml'][c-1],
        'current_volume_abs_error_ml':abs(np.count_nonzero(b)/1000-v0),
        'forecast_volume_abs_error_to_current_ml':abs(distmean-v0),
        'forecast_volume_abs_error_to_future_ml':abs(distmean-v1),
        'reference_absolute_change_ml':abs(v1-v0)}
    for name,value in attribution(before,after).items():result['attribution_'+name]=value
    for name,m in zip(DOMAINS,domains):
        result['error_reduction_'+name+'_ml']=(np.count_nonzero((b!=t1)&m)-np.count_nonzero((a!=t1)&m))/1000
    changed=domains[1];correct_change=(a!=b)&changed&(a==t1);wrong_change=(a!=b)&changed&(b==t1)
    result.update(correct_change_edits_ml=np.count_nonzero(correct_change)/1000,wrong_change_edits_ml=np.count_nonzero(wrong_change)/1000,
                  reference_changed_ml=np.count_nonzero(changed)/1000)
    # These replay matches are required before interpreting any decomposition.
    for name,value in (('persistence',d0),(method,d1)):
        if abs(value-record['methods'][name]['compartments'][str(c)]['dice'])>1e-7:
            raise AssertionError('Replayed compartment Dice differs from saved evaluation')
    return {k:v.item() if isinstance(v,np.generic) else v for k,v in result.items()}


@torch.inference_mode()
def replay(model,root,pair,now,seg,sdf,record):
    _,spatial,scalar=inputs(root,pair,model.arm)
    parameters=model.parameters_for(spatial,scalar);marginal=torch.zeros_like(now['images'][0],device='cuda')
    volumes=[]
    for i in range(model.scenarios):
        _,maps,_=model.render({**parameters,'delta':parameters['delta'][i:i+1]},sdf,
                            now['images'][0].cuda(),now['support'][0].cuda(),seg.cuda())
        marginal+=parameters['weights'][i]*maps[0]
        volumes.append((maps[0,1:].sum((1,2,3))*.001).cpu().numpy())
    if not np.allclose(volumes,record['methods'][model.arm]['scenario_volumes_ml'],atol=1e-4,rtol=0):
        raise AssertionError('Saved scenario volumes do not replay')
    return marginal.argmax(0).byte().cpu().numpy()


def summarize(rows):
    out={}
    for h in (12,24):
        for method in ('current','history'):
            subset=[r for r in rows if r['horizon']==h and r['method']==method]
            bycase=defaultdict(list)
            for r in subset:bycase[r['patient']].append(r)
            # Fixed original denominator: contribution strata sum to original mean gain.
            weights={p:1/(len(bycase)*len(v)) for p,v in bycase.items()}
            contribution=lambda ss,key:float(sum(weights[r['patient']]*r[key] for r in ss))
            compartments={}
            for c in (1,2,3):
                rs=[r for r in subset if r['compartment']==c]
                strata={'all':rs,'future_empty':[r for r in rs if r['future_empty']],
                        'future_nonempty':[r for r in rs if not r['future_empty']],
                        'future_over_0_1ml':[r for r in rs if not r['future_at_most_0_1ml']],
                        'meaningful_change':[r for r in rs if r['meaningful_volume_change']],
                        'persistent_nonempty_meaningful_change':[r for r in rs if r['meaningful_volume_change'] and r['presence_group']=='persistent_nonempty']}
                strata.update({g:[r for r in rs if r['presence_group']==g] for g in ('both_empty','appeared','resolved','persistent_nonempty')})
                compartments[str(c)]={'mean_gain':interval(rs,'gain'),
                    'contribution_to_overall_gain':contribution(rs,'gain'),
                    'strata':{g:{'gain':interval(v,'gain'),'gain_over_reference_persistence':interval(v,'gain_over_reference_persistence'),
                                'future_specific_gain':interval(v,'future_specific_gain'),
                                'contribution_to_overall_gain':contribution(v,'gain'),
                                **{k:patient_mean(v,k) for k in ('persistence_dice','forecast_dice','current_anatomy_gain','common_future_gain',
                                   'coverage90','below_interval','above_interval','width90_ml','lower90_ml','upper90_ml',
                                   'current_cavity_prediction_ml','forecast_in_current_cavity_ml','reference_absolute_change_ml',
                                   'current_volume_abs_error_ml','forecast_volume_abs_error_to_current_ml','forecast_volume_abs_error_to_future_ml')}} for g,v in strata.items()},
                    'spatial_attribution':{d:interval(rs,'attribution_'+d) for d in DOMAINS},
                    'changed_region_edits':{k:patient_mean(rs,k) for k in ('correct_change_edits_ml','wrong_change_edits_ml','reference_changed_ml')}}
            out[f'MU:{h}:{method}']={'targets':len(subset)//3,'patients':len(bycase),'overall_gain':contribution(subset,'gain'),
                'gain_contribution_by_future_presence':{g:contribution([r for r in subset if r['presence_group']==g],'gain') for g in ('both_empty','appeared','resolved','persistent_nonempty')},
                'gain_contribution_by_domain':{d:contribution(subset,'attribution_'+d) for d in DOMAINS},'compartments':compartments}
    return out


def empty_details(rows,evaluation):
    result={}
    for h in (12,24):
        for method in ('current','history'):
            subset=[r for r in rows if r['horizon']==h and r['method']==method and r['compartment']==2 and r['future_empty']]
            measurements=[]
            for r in subset:
                old=json.loads((evaluation/f"case-{r['evaluation_index']:04d}.json").read_text())
                measurements.append({**r,'persistence_scored_fp_ml':old['methods']['persistence']['compartments']['2']['empty_reference_fp_ml'],
                                     'forecast_scored_fp_ml':old['methods'][method]['compartments']['2']['absent_target_fp_ml']})
            result[f'MU:{h}:{method}']={'targets':len(subset),'patients':len({r['patient'] for r in subset}),
                'correct_empty_persistence_targets':sum(r['persistence_dice']==1 for r in subset),
                'correct_empty_forecast_targets':sum(r['forecast_dice']==1 for r in subset),
                'scored_false_positive_ml':{k:patient_mean(measurements,k) for k in ('persistence_scored_fp_ml','forecast_scored_fp_ml')},
                'successful_resolution_cutoff_reference_ml':sorted(r['current_reference_ml'] for r in subset if r['presence_group']=='resolved' and r['gain']>0),
                'meaningful_resolution_targets':sum(r['presence_group']=='resolved' and r['meaningful_volume_change'] for r in subset),
                'successful_meaningful_resolution_targets':sum(r['presence_group']=='resolved' and r['meaningful_volume_change'] and r['forecast_dice']==1 for r in subset)}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--limit',type=int);p.add_argument('--resume',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);root=args.run_root;contexts=root/'structured-context-v1';evaluation=root/'structured-selection-v1'
    cat=catalogue(root/'mu-timed-v1/catalogue.json');allowed={r['patient'] for r in cat['groups']['selection'] if r.get('dataset')=='MU'}
    pairs=json.loads((evaluation/'population.json').read_text());saved_identity=json.loads((evaluation/'identity.json').read_text())
    models={arm:load_transition(root/f'structured-v1/anatomy-{arm}/best.pt',saved_identity['contexts']['observation_sha256']) for arm in ('current','history')}
    if any(m.model_id!=saved_identity['checkpoints'][arm] for arm,m in models.items()):raise ValueError('Selected model changed')
    identity={'source_evaluation_sha256':digest(evaluation/'aggregate.json'),'source_identity':saved_identity,
              'checkpoint_sha256':{arm:m.model_id for arm,m in models.items()},'limit':args.limit,'scope':'Existing MU selection targets only; no fitting or reserved patients'}
    args.out.mkdir(parents=True,exist_ok=args.resume)
    if (args.out/'identity.json').exists() and json.loads((args.out/'identity.json').read_text())!=identity:raise ValueError('Interpretation identity changed')
    write_json(args.out/'identity.json',identity)
    work=[(i,pair) for i,pair in enumerate(pairs) if pair['dataset']=='MU']
    if args.limit:work=work[:args.limit]
    rows=[];started=time.monotonic()
    for n,(i,pair) in enumerate(work,1):
        if pair['patient'] not in allowed:raise ValueError('Outside authorized selection population')
        path=args.out/f'case-{i:04d}.json'
        if path.exists():rows.extend(json.loads(path.read_text()));continue
        now=load_visit(pair['history'][-1],labels=True);future=load_visit(pair['target'],labels=True)
        cached=load_context(contexts,pair['history'][-1]);seg=cached['predicted_segmentation'];sdf=signed_distances(seg.numpy()).cuda()
        reference0=now['labels'][0].numpy();reference1=future['labels'][0].numpy();support0=now['support'][0,0].numpy();support1=future['support'][0,0].numpy()
        original=json.loads((evaluation/f'case-{i:04d}.json').read_text());entries=[];maps={}
        for method,model in models.items():
            forecast=replay(model,contexts,pair,now,seg,sdf,original);maps[method]=forecast
            for c in (1,2,3):
                value=analyze_compartment(seg.numpy(),forecast,reference0,reference1,support0,support1,c,original,method)
                entries.append({'patient':pair['patient'],'horizon':pair['horizon'],'cutoff':pair['history'][-1]['visit'],'target':pair['target']['visit'],
                                'evaluation_index':i,'method':method,**value})
        np.savez_compressed(args.out/f'maps-{i:04d}.npz',**maps)
        write_json(path,entries);rows.extend(entries)
        print(json.dumps({'completed':n,'targets':len(work),'elapsed_seconds':time.monotonic()-started}),flush=True)
    summary={'status':'limited_check' if args.limit else 'complete','targets':len(work),'scope':identity['scope'],
             'replay':'All compartment Dice and eight scenario volumes matched the preserved evaluation',
             'attribution':'Exact retrospective Dice decomposition, not causal biological attribution; cavity treated separately',
             'reference_persistence':'Privileged current-reference map, evaluation only; never a prediction input',
             'reserved_images_opened':0,'new_training_updates':0,'results':summarize(rows),
             'nonenhancing_empty_targets':empty_details(rows,evaluation)}
    write_json(args.out/'aggregate.json',summary)


if __name__=='__main__':main()
