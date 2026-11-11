#!/usr/bin/env python3
"""Matched official draws and deterministic fitted core CDFs; no fitting."""
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
from src.patient_state.amount_first import AmountFirst,fixed_uniforms
from src.patient_state.structured import VolumeChangePredictor
from src.patient_state.core_cdf import core_cdf_scores
from src.patient_state.state import digest,write_json
from train_amount_first import average,paired


def scalar_input(record):
    # Exact current-only MU context contract, reconstructed from saved volumes.
    # Its adequacy is verified by exact replay of every official sample below.
    # Original preparation computes log1p on CPU before copying the scalar to CUDA.
    v=torch.tensor(record['cutoff_volumes_ml'],device='cpu',dtype=torch.float32)
    x=torch.cat((torch.log1p(v)/5,torch.zeros_like(v),v.new_tensor([record['horizon']/24,0]),v.new_ones(4),v.new_tensor([1,1,0])))[None]
    return x.cuda(),v[None].cuda()


def weighted_auc(rows,family):
    groups=defaultdict(list)
    for r in rows:groups[r['patient']].append(r)
    present=[];absent=[]
    for visits in groups.values():
        w=1/(len(groups)*len(visits))
        for r in visits:
            (present if r['truth_ml']>0 else absent).append((r[family]['presence_probability'],w))
    denominator=sum(w for _,w in present)*sum(w for _,w in absent)
    if not denominator:return None
    return sum(w*v*(float(p>q)+.5*float(p==q)) for p,w in present for q,v in absent)/denominator


def subset_summary(rows,family):
    return {'targets':len(rows),'patients':len({r['patient'] for r in rows}),
        'official_mean_mae_ml':average(rows,lambda r:r[family]['official_mean_mae_ml']),
        'official_crps_ml':average(rows,lambda r:r[family]['official_crps_ml']),
        'cdf_mean_mae_ml':average(rows,lambda r:r[family]['cdf']['mean_volume_mae_ml']),
        'cdf_crps_ml':average(rows,lambda r:r[family]['cdf']['crps_ml']),
        'cdf_crps_underprediction_ml':average(rows,lambda r:r[family]['cdf']['crps_underprediction_ml']),
        'cdf_crps_overprediction_ml':average(rows,lambda r:r[family]['cdf']['crps_overprediction_ml']),
        'absolute_64_mean_minus_cdf_mean_ml':average(rows,lambda r:abs(r[family]['official_mean_ml']-r[family]['cdf']['mean_ml'])),
        'signed_64_mean_minus_cdf_mean_ml':average(rows,lambda r:r[family]['official_mean_ml']-r[family]['cdf']['mean_ml']),
        'signed_64_crps_minus_cdf_crps_ml':average(rows,lambda r:r[family]['official_crps_ml']-r[family]['cdf']['crps_ml']),
        'presence_probability':average(rows,lambda r:r[family]['presence_probability']),
        'presence_brier':average(rows,lambda r:r[family]['presence_brier']),
        'presence_log_score':average(rows,lambda r:r[family]['presence_log_score']),
        'top1_sample_volume_share':average(rows,lambda r:r[family]['top1_share']),
        'top4_sample_volume_share':average(rows,lambda r:r[family]['top4_share']),
        'sample_median_ml':average(rows,lambda r:r[family]['sample_median_ml']),
        'sample_mean_ml':average(rows,lambda r:r[family]['official_mean_ml']),
        'positive_sample_median_ml':average(rows,lambda r:r[family]['positive_sample_median_ml']),
        'positive_sample_mean_ml':average(rows,lambda r:r[family]['positive_sample_mean_ml']),
        'positive_sample_coverage90':average(rows,lambda r:r[family]['positive_sample_coverage90']) if rows and all(r['truth_ml']>0 for r in rows) else None,
        'positive_sample_width90_ml':average(rows,lambda r:r[family]['positive_sample_width90_ml']),
        'maximum_sample_ml':max((r[family]['maximum_sample_ml'] for r in rows),default=None),
        'persistence_mae_ml':average(rows,lambda r:r['persistence_mae_ml']),
        'scalar_crps_ml':average(rows,lambda r:r['scalar_crps_ml'])}


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--logistic',type=Path,required=True);p.add_argument('--gaussian',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);start=time.monotonic();sources={'logistic':args.logistic,'gaussian':args.gaussian}
    cases={f:json.loads((root/'quantity-cases.json').read_text()) for f,root in sources.items()}
    assert len(cases['logistic'])==len(cases['gaussian'])==58
    models={}
    for family,root in sources.items():
        checkpoint=torch.load(root/'quantity-terminal.pt',map_location='cpu',weights_only=True)
        if checkpoint['step']!=6000 or checkpoint.get('family','logistic')!=family:raise ValueError('Expected preserved terminal family candidate')
        model=AmountFirst(VolumeChangePredictor('current',compartments=4),family=family).cuda().eval()
        model.load_state_dict(checkpoint['model']);models[family]=model
    records=[]
    for i,(old,new) in enumerate(zip(cases['logistic'],cases['gaussian'])):
        if old['key']!=new['key'] or old['truth_ml']!=new['truth_ml'] or old['capacity']!=new['capacity']:raise ValueError('Unmatched targets')
        record={k:old[k] for k in ('patient','horizon','key','cutoff','target')}
        record.update(truth_ml=old['truth_ml'][1],persistence_mae_ml=old['methods']['persistence']['mean_volume_mae_ml'][1],scalar_crps_ml=old['methods']['scalar']['compartment_crps_ml'][1])
        for family,case in [('logistic',old),('gaussian',new)]:
            scalar,volumes=scalar_input(case)
            key='|'.join(map(str,(case['patient'],case['cutoff'],case['horizon'])))
            replay=models[family].sample(scalar,volumes,case['capacity'],fixed_uniforms(key)).tolist()
            if replay!=case['sample_counts']:raise AssertionError('Reconstructed scalar or saved sampler differs: '+family+' '+case['key'])
            reference=core_cdf_scores(models[family],scalar,volumes,case['capacity'],round(case['truth_ml'][1]*1000))
            values=sorted(n[1]/1000 for n in replay);positive=[v for v in values if v>0]
            quant=lambda p:positive[max(0,math.ceil(p*len(positive))-1)] if positive else None
            total=sum(values);m=case['methods']['amount_first'];ps=case['presence']['amount_first']
            record[family]={'official_mean_ml':m['mean_volumes_ml'][1],'official_mean_mae_ml':m['mean_volume_mae_ml'][1],
                'official_crps_ml':m['compartment_crps_ml'][1],'cdf':reference,
                'presence_probability':case['presence_probability'][1],'presence_brier':ps['brier'][1],'presence_log_score':ps['log_score'][1],
                'top1_share':values[-1]/total if total else None,'top4_share':sum(values[-4:])/total if total else None,
                'sample_median_ml':values[31],'maximum_sample_ml':values[-1],
                'positive_sample_median_ml':quant(.5),'positive_sample_mean_ml':float(np.mean(positive)) if positive else None,
                'positive_sample_coverage90':float(quant(.05)<=record['truth_ml']<=quant(.95)) if positive and record['truth_ml']>0 else None,
                'positive_sample_width90_ml':quant(.95)-quant(.05) if positive else None}
        records.append(record)
        write_json(args.out/'core-cdf-cases.json',records)
        print(json.dumps({'cdf_targets_complete':i+1,'seconds':time.monotonic()-start}),flush=True)
    summary={};patient_results=[]
    for h in (12,24):
        rows=[r for r in records if r['horizon']==h]
        groups=defaultdict(list)
        for r in rows:groups[r['patient']].append(r)
        horizon={}
        for family in sources:
            horizon[family]={s:subset_summary(rs,family) for s,rs in [('all',rows),('future_present',[r for r in rows if r['truth_ml']>0]),('future_empty',[r for r in rows if r['truth_ml']==0])]}
            horizon[family]['presence_discrimination']={'patient_balanced_auc':weighted_auc(rows,family),
                'probability_range':[min(r[family]['presence_probability'] for r in rows),max(r[family]['presence_probability'] for r in rows)],
                'false_absence_targets_at_half':sum(r['truth_ml']>0 and r[family]['presence_probability']<.5 for r in rows),
                'false_presence_targets_at_half':sum(r['truth_ml']==0 and r[family]['presence_probability']>=.5 for r in rows)}
            positive=horizon[family]['future_present']
            horizon[family]['numerical_reference_gate_not_used_for_selection']={
                'mae_pass':positive['cdf_mean_mae_ml']<=positive['persistence_mae_ml'],
                'crps_pass':positive['cdf_crps_ml']<=positive['scalar_crps_ml']}
        for patient,visits in groups.items():
            item={'patient':patient,'horizon':h,'families':{}}
            for family in sources:
                item['families'][family]={s:subset_summary(rs,family) for s,rs in [('all',visits),('future_present',[r for r in visits if r['truth_ml']>0]),('future_empty',[r for r in visits if r['truth_ml']==0])]}
                pos=item['families'][family]['future_present']
                if pos['targets']:
                    item['families'][family]['nonempty_failures']={mode:{'mae_worse_than_persistence':pos[mode+'_mean_mae_ml']>pos['persistence_mae_ml'],
                        'crps_worse_than_scalar':pos[mode+'_crps_ml']>pos['scalar_crps_ml']} for mode in ('official','cdf')}
            patient_results.append(item)
        horizon['nonempty_patient_failures']={family:{mode:{metric:sum(p['families'][family].get('nonempty_failures',{}).get(mode,{}).get(metric,False) for p in patient_results if p['horizon']==h)
            for metric in ('mae_worse_than_persistence','crps_worse_than_scalar')} for mode in ('official','cdf')} for family in sources}
        horizon['gaussian_minus_logistic_paired_intervals']={}
        for subset,rs in [('all',rows),('future_present',[r for r in rows if r['truth_ml']>0]),('future_empty',[r for r in rows if r['truth_ml']==0])]:
            horizon['gaussian_minus_logistic_paired_intervals'][subset]={
                'official_mae_ml':paired(rs,lambda r:r['gaussian']['official_mean_mae_ml'],lambda r:r['logistic']['official_mean_mae_ml']),
                'official_crps_ml':paired(rs,lambda r:r['gaussian']['official_crps_ml'],lambda r:r['logistic']['official_crps_ml']),
                'cdf_mae_ml':paired(rs,lambda r:r['gaussian']['cdf']['mean_volume_mae_ml'],lambda r:r['logistic']['cdf']['mean_volume_mae_ml']),
                'cdf_crps_ml':paired(rs,lambda r:r['gaussian']['cdf']['crps_ml'],lambda r:r['logistic']['cdf']['crps_ml']),
                'presence_brier':paired(rs,lambda r:r['gaussian']['presence_brier'],lambda r:r['logistic']['presence_brier'])}
        summary[str(h)]=horizon
    write_json(args.out/'core-cdf-patients.json',patient_results)
    write_json(args.out/'core-cdf-comparison.json',{'horizons':summary,'targets':58,'patient_horizon_rows':len(patient_results),
        'exact_sample_replays':116,'cdf_method':'float64 sums over every feasible integer count and all16 presence patterns; no quadrature or Monte Carlo',
        'official_gates_unchanged':True,'cdf_reference_does_not_select_checkpoints':True,
        'conditional_intervals':'exact equal-weight ranks within saved positive draws; descriptive, not a change to official scores',
        'source_hashes':{family:{'checkpoint':digest(root/'quantity-terminal.pt'),'cases':digest(root/'quantity-cases.json')} for family,root in sources.items()},
        'seconds':time.monotonic()-start})


if __name__=='__main__':main()
