#!/usr/bin/env python3
"""Patient-balanced matched information contrasts and prospective CDF gates."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
from src.patient_state.state import write_json
from scripts.experiments.quantity.train_amount_first import average,patient_values,summarize,NAMES
from scripts.experiments.quantity.train_information_program import ARMS


def interval(patient_differences):
    values=np.array([patient_differences[p] for p in sorted(patient_differences)])
    if not len(values):return None
    rng=np.random.default_rng(17);boot=values[rng.integers(len(values),size=(4000,len(values)))].mean(1)
    loo=(values.sum()-values)/(len(values)-1) if len(values)>1 else values
    return dict(patients=len(values),difference=float(values.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
        patients_better=int((values<0).sum()),patients_worse=int((values>0).sum()),
        leave_one_patient_out_range=[float(loo.min()),float(loo.max())],patient_differences=patient_differences)


def gate(rows):
    selected=[r for r in rows if r['truth_ml'][1]>0]
    a=average(selected,lambda r:r['core_cdf']['mean_volume_mae_ml']);b=average(selected,lambda r:r['methods']['persistence']['mean_volume_mae_ml'][1])
    c=average(selected,lambda r:r['core_cdf']['crps_ml']);d=average(selected,lambda r:r['methods']['scalar']['compartment_crps_ml'][1])
    return dict(mean_mae_ml=a,persistence_mae_ml=b,crps_ml=c,scalar_crps_ml=d,mae_pass=a<=b,crps_pass=c<=d,passed=a<=b and c<=d,
        targets=len({r['key'] for r in selected}),patients=len({r['patient'] for r in selected}))


def presence_auc(rows):
    # Averaged seed probabilities for each target; equal patient mass.
    targets=defaultdict(list)
    for r in rows:targets[r['key']].append(r)
    bypatient=defaultdict(list)
    for rs in targets.values():bypatient[rs[0]['patient']].append((np.mean([r['presence_probability'][1] for r in rs]),rs[0]['truth_ml'][1]>0))
    yes=[];no=[]
    for values in bypatient.values():
        for p,z in values:(yes if z else no).append((p,1/len(values)))
    denom=sum(w for _,w in yes)*sum(w for _,w in no)
    return sum(w*v*(float(p>q)+.5*float(p==q)) for p,w in yes for q,v in no)/denom if denom else None


def aggregate(rows):
    summary=summarize(rows)
    for h in (12,24):
        rs=[r for r in rows if r['horizon']==h];s=summary[str(h)]
        s['targets']=len({r['key'] for r in rs});s['seed_target_records']=len(rs)
        s['quantity_gate']=gate(rs);s['joint_nll']=average(rs,lambda r:r['joint_nll'])
        s['presence_nll']=average(rs,lambda r:r['presence_nll']);s['conditional_count_nll']=average(rs,lambda r:r['conditional_count_nll'])
        s['core_presence_auc_seed_mean_probability']=presence_auc(rs)
        s['core_cdf']={}
        for label,subset in [('all',rs),('nonempty',[r for r in rs if r['truth_ml'][1]>0]),('empty',[r for r in rs if r['truth_ml'][1]==0])]:
            s['core_cdf'][label]={k:average(subset,lambda r:r['core_cdf'][k]) for k in ('mean_ml','mean_volume_mae_ml','crps_ml','crps_underprediction_ml','crps_overprediction_ml')}
            s['core_cdf'][label]['absolute_64_sample_mean_error_ml']=average(subset,
                lambda r:abs(r['methods']['amount_first']['mean_volumes_ml'][1]-r['core_cdf']['mean_ml']))
            s['core_cdf'][label]['top4_sample_volume_share']=average(subset,
                lambda r:sum(sorted(n[1] for n in r['sample_counts'])[-4:])/sum(n[1] for n in r['sample_counts'])
                if sum(n[1] for n in r['sample_counts']) else None)
            for metric,control in [('mean_volume_mae_ml','persistence'),('crps_ml','scalar')]:
                key='mean_volume_mae_ml' if metric=='mean_volume_mae_ml' else 'compartment_crps_ml'
                s['core_cdf'][label][metric+'_minus_'+control]=interval(patient_values(subset,lambda r:r['core_cdf'][metric]-r['methods'][control][key][1]))
        # Repeated seeds are not additional targets or outside-domain tissue.
        for method in s['compartments'].values():
            for c,name in enumerate(NAMES):
                for subset in ('all','nonempty','empty'):
                    method[name][subset]['targets']=len({r['key'] for r in rs if subset=='all' or (r['truth_ml'][c]>0)==(subset=='nonempty')})
                    entry=method[name][subset]
                    if 'conditional_undefined_targets' in entry:
                        entry['conditional_undefined_seed_target_records']=entry.pop('conditional_undefined_targets')
        for method,compartments in s['presence'].items():
            for c,name in enumerate(NAMES):
                entry=compartments[name]
                entry['infinite_log_score_targets']=len({r['key'] for r in rs if r['presence'][method]['infinite_log_score'][c]})
                for bin in entry['reliability']:
                    bin['seed_target_records']=bin.pop('targets')
        unique={r['key']:r for r in rs}
        s['outside_domain_target_voxels']=np.sum([r['outside_domain_counts'] for r in unique.values()],axis=0).tolist()
        s['targets_with_outside_domain_tissue']=sum(any(r['outside_domain_counts']) for r in unique.values())
        s['reference_persistence']={}
        for c,name in enumerate(NAMES):
            s['reference_persistence'][name]={}
            for label,subset in [('all',rs),('nonempty',[r for r in rs if r['truth_ml'][c]>0]),('empty',[r for r in rs if r['truth_ml'][c]==0])]:
                s['reference_persistence'][name][label]={key:average(subset,lambda r:r['methods']['reference_persistence'][key][c]) for key in
                    ('mean_volume_mae_ml','compartment_crps_ml','compartment_coverage90','compartment_width90_ml')}
                if c==1:
                    for metric,key in [('mean_volume_mae_ml','mean_volume_mae_ml'),('crps_ml','compartment_crps_ml')]:
                        s['reference_persistence'][name][label]['learned_cdf_'+metric+'_minus_reference_persistence']=interval(patient_values(subset,
                            lambda r:r['core_cdf'][metric]-r['methods']['reference_persistence'][key][c]))
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    cases={};runs={};patients=[]
    for arm in ARMS:
        cases[arm]=[];runs[arm]={}
        for seed in (17,23,31):
            directory=root/'runs'/f'{arm}-s{seed}'
            rows=json.loads((directory/'quantity-cases.json').read_text())
            assert len(rows)==58
            for r in rows:r['seed']=seed;r['arm']=arm
            cases[arm]+=rows;runs[arm][str(seed)]=aggregate(rows)
        runs[arm]['seed_mean']=aggregate(cases[arm])
        for h in (12,24):
            rs=[r for r in cases[arm] if r['horizon']==h]
            for patient in sorted({r['patient'] for r in rs}):
                subset=[r for r in rs if r['patient']==patient];nonempty=[r for r in subset if r['truth_ml'][1]>0]
                patients.append(dict(arm=arm,horizon=h,patient=patient,joint_nll=average(subset,lambda r:r['joint_nll']),
                    nonempty_core=gate(nonempty) if nonempty else None,
                    presence_brier=average(subset,lambda r:r['presence']['amount_first']['brier'][1]),
                    reference_present_targets=len({r['key'] for r in nonempty})))
    metric_functions={'joint_nll':lambda r:r['joint_nll'],'core_cdf_mae':lambda r:r['core_cdf']['mean_volume_mae_ml'],
        'core_cdf_crps':lambda r:r['core_cdf']['crps_ml'],'core_presence_brier':lambda r:r['presence']['amount_first']['brier'][1]}
    pairs={'MRI_old':('old_mri','old_scalar'),'MRI_expanded':('expanded_mri','expanded_scalar'),
        'data_scalar':('expanded_scalar','old_scalar'),'data_MRI':('expanded_mri','old_mri'),
        'reference_quantities':('reference_scalar','expanded_scalar'),
        'past_volumes':('memory_volumes','memory_metadata'),'past_MRI':('memory_mri','memory_volumes'),
        'all_history':('memory_mri','memory_metadata')}
    contrasts={}
    for label,(left,right) in pairs.items():
        contrasts[label]={}
        for h in (12,24):
            contrasts[label][str(h)]={}
            for subset in ('all','nonempty_core'):
                result={}
                for metric,fn in metric_functions.items():
                    values=[]
                    for arm in (left,right):values.append(patient_values([r for r in cases[arm] if r['horizon']==h and (subset=='all' or r['truth_ml'][1]>0)],fn))
                    assert values[0].keys()==values[1].keys()
                    result[metric]=interval({p:values[0][p]-values[1][p] for p in values[0]})
                contrasts[label][str(h)][subset]=result
    interaction={}
    for h in ('12','24'):
        interaction[h]={}
        for metric in metric_functions:
            expanded=contrasts['MRI_expanded'][h]['all'][metric]['patient_differences'];old=contrasts['MRI_old'][h]['all'][metric]['patient_differences']
            interaction[h][metric]=interval({p:expanded[p]-old[p] for p in old})
    decomposition=[]
    for r in cases['expanded_scalar']:
        if r['seed']!=17:continue
        estimated=np.array(r['estimated_current_ml']);reference=np.array(r['reference_current_ml']);future=np.array(r['truth_ml'])
        measurement=estimated-reference;change=future-reference;persistence=estimated-future
        assert np.allclose(persistence,measurement-change)
        decomposition.append(dict(key=r['key'],patient=r['patient'],horizon=r['horizon'],
            estimation_error_ml=measurement.tolist(),reference_change_ml=change.tolist(),estimated_persistence_error_ml=persistence.tolist(),
            reference_persistence_absolute_error_ml=np.abs(change).tolist(),
            cancellation_ml=(np.abs(measurement)+np.abs(change)-np.abs(persistence)).tolist()))
    decomposition_summary={}
    for h in (12,24):
        rows=[r for r in decomposition if r['horizon']==h]
        decomposition_summary[str(h)]={name:{metric:average(rows,lambda r:r[metric][c]) for metric in
            ('estimation_error_ml','reference_change_ml','estimated_persistence_error_ml','reference_persistence_absolute_error_ml','cancellation_ml')}
            for c,name in enumerate(NAMES)}
    eligible=[]
    for arm in ARMS:
        if ARMS[arm]['reference']:continue
        if all(runs[arm][seed][str(h)]['quantity_gate']['passed'] for seed in ('17','seed_mean') for h in (12,24)):
            score=np.mean([runs[arm]['seed_mean'][str(h)]['joint_nll'] for h in (12,24)]);eligible.append((float(score),arm))
    selected=min(eligible)[1] if eligible else None
    result=dict(identity='mu-frozen-state-information-v1',status='quantity_eligible' if selected else 'no_eligible_quantity_configuration',
        selected_arm=selected,eligible_arms=eligible,location_training_started=False,configuration_results=runs,contrasts=contrasts,
        MRI_by_data_interaction=interaction,patient_results=patients,measurement_change_decomposition=decomposition,
        measurement_change_summary=decomposition_summary,
        aggregation='Average seeds and targets within each patient/horizon, then patients equally. Shared 4000 patient-bootstrap streams seed17. Reused developmental selection.',
        gate='Deterministic core CDF MAE <= estimated persistence and CRPS <= preserved full scalar; each horizon, seed17 and mean of three seeds.',
        targets=58,seeds=[17,23,31],fits=24,reserved_patients_opened=0,joint_samples_per_target=64,sample_streams_shared=True,
        protocol=json.loads((root/'protocol.json').read_text()),
        run_artifacts={arm:{str(seed):dict(checkpoint=str(Path('runs')/f'{arm}-s{seed}'/'quantity-terminal.pt'),
            evaluation=json.loads((root/'runs'/f'{arm}-s{seed}'/'evaluation.json').read_text()),
            configuration=json.loads((root/'runs'/f'{arm}-s{seed}'/'configuration.json').read_text())) for seed in (17,23,31)} for arm in ARMS},
        retained_completion_tradeoff='Frozen observation: approximately 2.9% higher missing-FLAIR lesion MAE on LUMIERE than original comparator; no retraining.',
        restoration=json.loads((root/'data/restoration.json').read_text()),
        feature_extraction={p.stem:json.loads(p.read_text()) for p in (root/'descriptors').glob('*extraction.json')},
        timing={arm:{str(seed):dict(training_seconds=json.loads((root/'runs'/f'{arm}-s{seed}'/'training.json').read_text())['seconds'],evaluation_seconds=json.loads((root/'runs'/f'{arm}-s{seed}'/'evaluation.json').read_text())['seconds']) for seed in (17,23,31)} for arm in ARMS})
    write_json(root/'comparison.json',result)
    print(json.dumps(dict(status=result['status'],selected_arm=selected,eligible_arms=eligible,
        gates={arm:{h:runs[arm]['seed_mean'][h]['quantity_gate'] for h in ('12','24')} for arm in ARMS})),flush=True)


if __name__=='__main__':main()
