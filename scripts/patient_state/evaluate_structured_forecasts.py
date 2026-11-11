#!/usr/bin/env python3
"""Matched full-volume anatomy, uncertainty and scalar controls after selection."""
import argparse,json,sys,time
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from src.patient_state.structured import AnatomicalTransition,VolumeChangePredictor
from src.patient_state.structured_training import load_context,evaluate_case,distribution_metrics,bounded_selection
from src.patient_state.data import load_visit
from src.patient_state.state import digest,write_json
from src.patient_state.timing import elapsed_weeks
from scripts.patient_state.evaluate_patient_forecasts import metrics as image_metrics


def aggregate(records):
    result={}
    keys=('dice','volume_mae_ml','volume_crps_ml','volume_change_mae_ml','coverage90','width90_ml','interval_score90_ml','brain_mae','lesion_mae','gradient_mae','anatomy_brier')
    for d in ('LUMIERE','MU'):
        for h in (12,24):
            subset=[r for r in records if r['dataset']==d and r['horizon']==h]
            if not subset:continue
            methods={}
            for method in subset[0]['methods']:
                def average(k,rs):
                    bypatient=defaultdict(list)
                    for r in rs:
                        if r['methods'][method].get(k) is not None:bypatient[r['patient']].append(r['methods'][method][k])
                    return float(np.mean([np.mean(v) for v in bypatient.values()])) if bypatient else None
                methods[method]={k:average(k,subset) for k in keys}
                compartments={}
                for c in ('1','2','3'):
                    values={}
                    for key in ('dice','absent_target_fp_ml','crps_ml','coverage90','width90_ml','volume_error_ml','change_error_ml'):
                        grouped=defaultdict(list)
                        for r in subset:
                            m=r['methods'][method]
                            if key in ('dice','absent_target_fp_ml'):
                                value=m.get('compartments',{}).get(c,{}).get(key)
                                if key=='absent_target_fp_ml' and value is None:
                                    value=m.get('compartments',{}).get(c,{}).get('empty_reference_fp_ml')
                            else:
                                array=m.get('compartment_'+key);value=array[int(c)-1] if array is not None else None
                            if value is not None:grouped[r['patient']].append(abs(value) if key in ('volume_error_ml','change_error_ml') else value)
                        values[key]=float(np.mean([np.mean(v) for v in grouped.values()])) if grouped else None
                    compartments[c]=values
                methods[method]['compartments']=compartments
                methods[method]['by_observed_change']={g:{'targets':len(s:=[r for r in subset if r['change_group']==g]),**{k:average(k,s) for k in keys}} for g in ('stable','changed')}
            contrasts={}
            for control in ('current','persistence','volume_history'):
                contrasts[control]={}
                for k in ('dice','volume_mae_ml','volume_crps_ml','volume_change_mae_ml','lesion_mae','gradient_mae'):
                    grouped=defaultdict(list)
                    for r in subset:
                        a,b=r['methods']['history'].get(k),r['methods'][control].get(k)
                        if a is not None and b is not None:grouped[r['patient']].append(a-b)
                    values=np.array([np.mean(v) for _,v in sorted(grouped.items())])
                    if len(values):
                        rng=np.random.default_rng(17);sample=values[rng.integers(len(values),size=(2000,len(values)))].mean(1)
                        contrasts[control][k]={'difference':float(values.mean()),'patient_bootstrap95':np.quantile(sample,[.025,.975]).tolist(),'patients':len(values)}
            result[f'{d}:{h}']={'targets':len(subset),'patients':len({r['patient'] for r in subset}),
                                'targets_with_positive_past_interval':sum(r['history_with_positive_elapsed_time'] for r in subset),
                                'methods':methods,'history_minus_control':contrasts}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contexts',type=Path,required=True);p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--limit',type=int);p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true',help='Explicit instrument check using last.pt, restricted to at most two targets')
    args=p.parse_args();torch.set_num_threads(4)
    if args.smoke and (args.limit is None or not 1<=args.limit<=2):raise ValueError('Smoke must explicitly limit evaluation to one or two targets')
    identity=json.loads((args.contexts/'identity.json').read_text());models={};hashes={}
    for kind in ('anatomy','volume'):
        for arm in ('current','history'):
            path=args.runs/f'{kind}-{arm}'/('last.pt' if args.smoke else 'best.pt');ck=torch.load(path,map_location='cpu',weights_only=True)
            if ck['config']['identity']!=identity or (ck['config']['smoke'] and not args.smoke):raise ValueError('Mismatched or smoke checkpoint')
            net=(AnatomicalTransition(arm) if kind=='anatomy' else VolumeChangePredictor(arm)).cuda().eval()
            net.load_state_dict(ck['model']);name=arm if kind=='anatomy' else f'volume_{arm}';models[name]=net;hashes[name]=digest(path)
    args.out.mkdir(parents=True,exist_ok=args.resume)
    receipt={'contexts':identity,'checkpoints':hashes,'limit':args.limit,'smoke':args.smoke}
    if (args.out/'identity.json').exists() and json.loads((args.out/'identity.json').read_text())!=receipt:raise ValueError('Evaluation identity changed')
    write_json(args.out/'identity.json',receipt)
    pairs=json.loads((args.contexts/'population.json').read_text())['selection'];pairs=pairs[:args.limit] if args.limit else pairs
    write_json(args.out/'population.json',pairs)
    records=[];started=time.monotonic()
    with torch.inference_mode():
        for i,pair in enumerate(pairs):
            path=args.out/f'case-{i:04d}.json'
            if path.exists():records.append(json.loads(path.read_text()));continue
            current=load_context(args.contexts,pair['history'][-1]);target=load_context(args.contexts,pair['target'])
            now=load_visit(pair['history'][-1]);future=load_visit(pair['target'],labels=True)
            vv=current['predicted_volumes'].numpy();yy=target['target_volumes'];previous_truth=current['target_volumes'].numpy()
            persistence=image_metrics(current['predicted_segmentation'].numpy(),now['images'],future,vv,previous_truth)
            persistence.update(distribution_metrics(current['predicted_volumes'][None],torch.ones(1),yy))
            persistence.update(coverage90=None,width90_ml=None,interval_score90_ml=None,
                               compartment_coverage90=None,compartment_width90_ml=None,
                               uncertainty_status='point-mass CRPS comparator; no fitted uncertainty interval')
            methods={'persistence':persistence}
            for name,net in models.items():
                value=evaluate_case(net,args.contexts,pair)
                pred=np.array(value['mean_volumes_ml']);truth=yy.numpy()
                value['volume_change_mae_ml']=float(np.abs((pred-vv)-(truth-previous_truth)).mean())
                value['compartment_change_error_ml']=((pred-vv)-(truth-previous_truth)).tolist()
                methods[name]=value
            change=np.abs(yy.numpy()-previous_truth)>np.maximum(2,.2*previous_truth)
            record={'patient':pair['patient'],'dataset':pair['dataset'],'horizon':pair['horizon'],'cutoff':pair['history'][-1]['visit'],
                    'history_with_positive_elapsed_time':any(elapsed_weeks(r)<elapsed_weeks(pair['history'][-1]) for r in pair['history'][:-1]),
                    'target':pair['target']['visit'],'target_offset':pair['target_offset'],'change_group':'changed' if change.any() else 'stable','methods':methods}
            write_json(path,record);records.append(record)
            print(json.dumps({'complete':i+1,'total':len(pairs),'elapsed_seconds':time.monotonic()-started}),flush=True)
    # These perturbations test the downstream temporal context, not a complete
    # re-encoding of the frozen observation model or clinical acquisition robustness.
    sensitivity=[]
    for pair in bounded_selection(pairs,patients_per_cohort=1):
        for nuisance in ('intensity_1.05','resolution_4mm'):
            sensitivity.append({'dataset':pair['dataset'],'patient':pair['patient'],'horizon':pair['horizon'],'perturbation':nuisance,
                                'metrics':evaluate_case(models['history'],args.contexts,pair,nuisance=nuisance)})
    write_json(args.out/'private-context-sensitivity.json',sensitivity)
    write_json(args.out/'aggregate.json',{'status':'smoke_completed' if args.smoke else 'completed','scope':'development selection; finite learned mixture, no independent generalization claim',
               'timing':'LUMIERE nominal weeks; MU verified diagnosis-relative day differences',
               'horizons':aggregate(records),'elapsed_seconds':time.monotonic()-started,
               'nuisance_scope':'downstream temporal-context perturbation only','reserved_images_opened':0})


if __name__=='__main__':main()
