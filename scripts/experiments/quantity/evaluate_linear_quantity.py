#!/usr/bin/env python3
"""One assessment of both locked regularized quantity forecasters."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
from scripts.experiments.quantity.fit_linear_quantity import records,SOURCE,score_rows,summarize_scores
from src.patient_state.linear_quantity import load_bundle,transform,patient_horizon_weights,count_log_mass
from src.patient_state.amount_first import fixed_uniforms,ORDER
from src.patient_state.core_cdf import core_cdf_scores
from src.patient_state.state import write_json,digest
from scripts.experiments.quantity.train_amount_first import metrics,presence_scores,average,patient_values,summarize,NAMES


def interval(values):
    x=np.array([values[p] for p in sorted(values)])
    if not len(x):return None
    rng=np.random.default_rng(17);boot=x[rng.integers(len(x),size=(4000,len(x)))].mean(1)
    loo=(x.sum()-x)/(len(x)-1) if len(x)>1 else x
    return dict(patients=len(x),difference=float(x.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
        patients_better=int((x<0).sum()),patients_worse=int((x>0).sum()),
        leave_one_patient_out_range=[float(loo.min()),float(loo.max())],patient_differences=values)


def auc(rows,c,method='amount_first'):
    weights=patient_horizon_weights(rows);positive=[];negative=[]
    for r,w in zip(rows,weights):
        p=r['presence'][method]['probability'][c]
        (positive if r['truth_ml'][c]>0 else negative).append((p,w))
    denominator=sum(w for p,w in positive)*sum(w for p,w in negative)
    if not denominator:return None
    return sum(w*v*(float(p>q)+.5*float(p==q)) for p,w in positive for q,v in negative)/denominator


@torch.no_grad()
def evaluate(model,recipe,rows,prior):
    result=[]
    for item in rows:
        old=item['old'];x=transform([item],recipe,'cuda')
        current=torch.tensor(item['current_counts'],dtype=torch.double,device='cuda')[None]
        truth=torch.tensor(item['truth'],device='cuda')[None];capacity=torch.tensor([item['capacity']],device='cuda')
        samples=model.sample(x,current,item['capacity'],fixed_uniforms(item['random_key']))
        if (samples<0).any() or (samples.sum(1)>item['capacity']).any():raise AssertionError('Infeasible sampled counts')
        logs=model.presence_log_marginals(x,current);prob=logs[:,1].exp();z=truth[0]>0
        terms=model.terms(x,current,truth,capacity)
        p=float(terms['presence'].sum());q=float(terms['amount'].sum())
        target=truth[0].double()/1000
        r={k:old[k] for k in ('key','patient','horizon','cutoff','target','outside_domain_counts','persistence_dice')}
        r.update(truth_ml=target.tolist(),current_counts=item['current_counts'].tolist(),capacity=item['capacity'],
            sample_counts=samples.tolist(),presence_probability=prob.tolist(),joint_nll=p+q,presence_nll=p,amount_nll=q,
            core_cdf=core_cdf_scores(model,x,current,item['capacity'],int(item['truth'][1])),
            methods=dict(amount_first=metrics(samples.double()/1000,torch.full((64,),1/64,dtype=torch.double,device='cuda'),target),
                scalar=old['methods']['scalar'],persistence=old['methods']['persistence']),presence={})
        for name,value in [('amount_first',prob.cpu().numpy()),('fitting_prevalence',prior[item['horizon']]),
                           ('current_persistence',item['current_counts']>0)]:
            r['presence'][name]=presence_scores(value,z.cpu().numpy())
        # Evaluate the learned marginal log score without rounding probabilities to one.
        r['presence']['amount_first']['log_score']=(-logs[torch.arange(4,device='cuda'),z.long()]).tolist()
        r['presence']['amount_first']['infinite_log_score']=[False]*4
        result.append(r)
    return result


def aggregate(rows):
    result=summarize(rows)
    for h in (12,24):
        rs=[r for r in rows if r['horizon']==h];entry=result[str(h)]
        entry['likelihood']={k:average(rs,lambda r:r[k]) for k in ('joint_nll','presence_nll','amount_nll')}
        entry['presence_auc']={method:{name:auc(rs,c,method) for c,name in enumerate(NAMES)}
            for method in ('amount_first','fitting_prevalence','current_persistence')}
        entry['core_cdf']={}
        for label,subset in [('all',rs),('nonempty',[r for r in rs if r['truth_ml'][1]>0]),('empty',[r for r in rs if r['truth_ml'][1]==0])]:
            s={k:average(subset,lambda r:r['core_cdf'][k]) for k in ('mean_ml','mean_volume_mae_ml','crps_ml','crps_underprediction_ml','crps_overprediction_ml')}
            s.update(targets=len(subset),patients=len({r['patient'] for r in subset}))
            s['absolute_sample_mean_error_ml']=average(subset,lambda r:abs(r['core_cdf']['mean_ml']-r['methods']['amount_first']['mean_volumes_ml'][1]))
            for metric,control,key in [('mean_volume_mae_ml','persistence','mean_volume_mae_ml'),('crps_ml','scalar','compartment_crps_ml')]:
                s[metric+'_minus_'+control]=interval(patient_values(subset,lambda r:r['core_cdf'][metric]-r['methods'][control][key][1]))
            entry['core_cdf'][label]=s
        nonempty=[r for r in rs if r['truth_ml'][1]>0];core=entry['core_cdf']['nonempty']
        gate=dict(targets=len(nonempty),patients=len({r['patient'] for r in nonempty}),mean_mae_ml=core['mean_volume_mae_ml'],
            persistence_mae_ml=average(nonempty,lambda r:r['methods']['persistence']['mean_volume_mae_ml'][1]),crps_ml=core['crps_ml'],
            scalar_crps_ml=average(nonempty,lambda r:r['methods']['scalar']['compartment_crps_ml'][1]))
        gate.update(mae_pass=gate['mean_mae_ml']<=gate['persistence_mae_ml'],crps_pass=gate['crps_ml']<=gate['scalar_crps_ml'])
        gate['passed']=gate['mae_pass'] and gate['crps_pass'];entry['quantity_gate']=gate
        entry['numerical_scope']='Core gate and core_cdf use the fitted discrete CDF. Other amount_first compartment statistics/paired intervals use the same 64 joint samples; controls retain original full distributions.'
        entry['targets_with_outside_domain_tissue']=sum(any(r['outside_domain_counts']) for r in rs)
    return result


def paired_arms(cases):
    result={};patient_rows=[]
    for h in (12,24):
        result[str(h)]={}
        for subset in ('all','nonempty_core','empty_core'):
            select=lambda r:r['horizon']==h and (subset=='all' or (r['truth_ml'][1]>0)==(subset=='nonempty_core'))
            functions={k:(lambda r,key=k:r[key]) for k in ('joint_nll','presence_nll','amount_nll')}
            functions.update(core_cdf_mae=lambda r:r['core_cdf']['mean_volume_mae_ml'],core_cdf_crps=lambda r:r['core_cdf']['crps_ml'])
            for c,name in enumerate(NAMES):
                functions[name+'_brier']=lambda r,c=c:r['presence']['amount_first']['brier'][c]
                for metric in ('mean_volume_mae_ml','compartment_crps_ml','compartment_coverage90','compartment_width90_ml'):
                    functions[name+'_'+metric]=lambda r,c=c,key=metric:r['methods']['amount_first'][key][c]
            result[str(h)][subset]={}
            for name,fn in functions.items():
                s=patient_values([r for r in cases['S'] if select(r)],fn);m=patient_values([r for r in cases['M'] if select(r)],fn)
                assert s.keys()==m.keys()
                result[str(h)][subset][name]=interval({p:m[p]-s[p] for p in s})
                for p in s:patient_rows.append(dict(patient=p,horizon=h,subset=subset,metric=name,S=s[p],M=m[p],M_minus_S=m[p]-s[p]))
    return result,patient_rows


def oof_calibration_summary(directory,metadata):
    rows=json.loads((directory/'outer-predictions.json').read_text());out=[]
    sigma=torch.tensor(metadata['sigmas_autoregressive_order'],dtype=torch.double)
    for r in rows:
        n=torch.tensor(r['counts']);mu=torch.tensor(r['mu'],dtype=torch.double);maximum=torch.tensor(r['maximum']);positive=n>0
        q=float(-count_log_mass(n[positive],maximum[positive],mu[positive],sigma[positive],'gaussian').sum())
        out.append(dict(r,amount_nll=q,joint_nll=r['presence_nll']+q))
    return dict(score=summarize_scores(out),scope='In-sample scale calibration on outer-excluded locations; not independent validation of fitted sigmas or coverage')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,required=True);p.add_argument('--source',type=Path,default=SOURCE);a=p.parse_args()
    started=time.monotonic();torch.set_num_threads(1)
    lock=json.loads((a.build/'locked-final-models.json').read_text())
    for arm in ('S','M'):
        if digest(a.build/lock['models'][arm]['checkpoint'])!=lock['models'][arm]['sha256']:raise ValueError('Locked model changed')
    # Both files have been locked and checked before selection records are accessed.
    rows=records(a.source,'selection');fitting=records(a.source,'fitting')
    if len(rows)!=58 or {r['patient'] for r in rows}&{r['patient'] for r in fitting}:raise AssertionError('Population or split changed')
    prior={h:patient_horizon_weights([r for r in fitting if r['horizon']==h])@np.stack([r['truth']>0 for r in fitting if r['horizon']==h]) for h in (12,24)}
    cases={};results={};fit={}
    for arm in ('S','M'):
        model,recipe,metadata=load_bundle(a.build/lock['models'][arm]['checkpoint'],'cuda')
        assert set(recipe['fitting_patients'])=={r['patient'] for r in fitting}
        cases[arm]=evaluate(model,recipe,rows,prior)
        write_json(a.build/arm/'selection-cases.json',cases[arm]);results[arm]=aggregate(cases[arm])
        fit[arm]=dict(metadata,outer_attached_scale_calibration=oof_calibration_summary(a.build/arm,metadata))
    contrasts,patients=paired_arms(cases);write_json(a.build/'patient-results.json',patients)
    eligible=[arm for arm in ('S','M') if all(results[arm][str(h)]['quantity_gate']['passed'] for h in (12,24))]
    comparison=dict(identity='mu-linear-count-v1',status='quantity_eligible' if eligible else 'no_eligible_quantity_model',eligible_arms=eligible,
        models=lock['models'],results=results,paired_M_minus_S=contrasts,fitting=fit,
        targets=58,fitting_pairs=156,fitting_patients=56,location_training_started=False,reserved_patients_opened=0,
        selection_scope='Repeatedly used developmental selection; head-level fitting CV conditional on a fitting-supervised frozen observer',
        aggregation='Targets within patients, patients equally, horizons separately; 4000 paired patient-bootstrap draws seed17 and leave-one-patient-out ranges',
        gate='Each deterministic final arm: fitted-CDF nonempty-core MAE <= persistence and CRPS <= full frozen scalar at both horizons. No seed-average rule.',
        training_seconds=lock['training_seconds'],evaluation_seconds=time.monotonic()-started,
        retained_completion_tradeoff='Frozen observation model retains previously measured approximately 2.9% higher missing-FLAIR lesion MAE on LUMIERE; no retraining or substitution',
        protocol=json.loads((a.build/'protocol.json').read_text()),smoke=json.loads((a.build/'smoke.json').read_text()))
    write_json(a.build/'comparison.json',comparison)
    print(json.dumps(dict(status=comparison['status'],eligible_arms=eligible,gates={arm:{h:results[arm][h]['quantity_gate'] for h in ('12','24')} for arm in ('S','M')},seconds=comparison['evaluation_seconds'])),flush=True)


if __name__=='__main__':main()
