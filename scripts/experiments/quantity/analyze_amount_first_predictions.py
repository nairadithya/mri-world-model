#!/usr/bin/env python3
"""Read-only, small-record core interpretation; no model loading or fitting."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics as st


def mean(values):
    values=[v for v in values if v is not None]
    return st.mean(values) if values else None


def quantile(values,p):
    values=sorted(values)
    return values[max(0,math.ceil(p*len(values))-1)] if values else None


def patient_average(rows,key):
    groups=defaultdict(list)
    for row in rows:groups[row['patient']].append(row['metrics'].get(key))
    return mean([mean(values) for values in groups.values()])


def summarize(rows):
    keys=sorted({k for row in rows for k in row['metrics']})
    averaged={key:patient_average(rows,key) for key in keys}
    for method in ('amount_first','fitting_prevalence','current_persistence'):
        if averaged.get('presence_infinite_log_score_'+method,0):
            averaged['presence_log_score_'+method]=None
    return {'targets':len(rows),'patients':len({r['patient'] for r in rows}),
            'patient_averaged':averaged,
            'predicted_presence_probability_range':[min(r['metrics']['presence_probability'] for r in rows),max(r['metrics']['presence_probability'] for r in rows)] if rows else None,
            'positive_sample_count_range':[min(r['metrics']['positive_samples'] for r in rows),max(r['metrics']['positive_samples'] for r in rows)] if rows else None,
            'targets_top4_contribute_over_half':sum(r['metrics']['top4_volume_share']>.5 for r in rows),
            'targets_conditional_median_below_truth':sum(r['metrics']['positive_median_ml']<r['metrics']['truth_ml'] for r in rows),
            'largest_core_sample_ml':max((r['metrics']['maximum_sample_ml'] for r in rows),default=None)}


def analyze(source,private,aggregate):
    raw=json.loads(source.read_text())
    assert len(raw)==58 and len({r['key'] for r in raw})==58
    records=[]
    for row in raw:
        values=sorted(n[1]/1000 for n in row['sample_counts'])
        assert len(values)==64 and values[0]>=0
        positive=[x for x in values if x>0]
        assert positive, 'Retain and report undefined conditional samples instead of dropping a case'
        y=row['truth_ml'][1];total=sum(values);mu=mean(values);pmedian=quantile(positive,.5);pmean=mean(positive)
        lo,hi=quantile(positive,.05),quantile(positive,.95)
        m={'truth_ml':y,'current_estimate_ml':row['cutoff_volumes_ml'][1],
           'presence_probability':row['presence_probability'][1],'observed_presence':float(y>0),
           'positive_samples':len(positive),'sampled_presence_frequency':len(positive)/64,
           'positive_median_ml':pmedian,'positive_mean_ml':pmean,
           'positive_lower90_ml':lo,'positive_upper90_ml':hi,'positive_width90_ml':hi-lo,
           'positive_coverage90':float(lo<=y<=hi) if y>0 else None,
           'positive_median_mae_ml':abs(pmedian-y) if y>0 else None,
           'positive_mean_mae_ml':abs(pmean-y) if y>0 else None,
           'positive_mean_bias_ml':pmean-y if y>0 else None,
           'positive_median_bias_ml':pmedian-y if y>0 else None,
           'positive_interval_miss_below':float(y<lo) if y>0 else None,
           'positive_interval_miss_above':float(y>hi) if y>0 else None,
           'unconditional_mean_ml':mu,'unconditional_median_ml':quantile(values,.5),
           'unconditional_mean_bias_ml':mu-y,'unconditional_median_mae_ml':abs(quantile(values,.5)-y),
           'top1_volume_share':values[-1]/total,'top4_volume_share':sum(values[-4:])/total,
           'top7_volume_share':sum(values[-7:])/total,'maximum_sample_ml':values[-1],
           'fraction_samples_below_mean':mean([float(x<mu) for x in values]),
           'mean_without_largest_sample_ml':mean(values[:-1]),
           'mae_without_largest_sample_ml':abs(mean(values[:-1])-y),
           'mean_without_top4_samples_ml':mean(values[:-4]),
           'mae_without_top4_samples_ml':abs(mean(values[:-4])-y)}
        for method,score in row['presence'].items():
            m['presence_brier_'+method]=score['brier'][1]
            m['presence_log_score_'+method]=score['log_score'][1]
            m['presence_infinite_log_score_'+method]=float(score['infinite_log_score'][1])
            m['presence_probability_'+method]=score['probability'][1]
        for method,score in row['methods'].items():
            for metric in ('mean_volume_mae_ml','compartment_crps_ml','compartment_coverage90','compartment_width90_ml'):
                m[method+'_'+metric]=score[metric][1]
            m[method+'_all_four_compartment_mae_ml']=score['volume_mae_ml']
            m[method+'_all_four_compartment_crps_ml']=score['volume_crps_ml']
            m[method+'_conditional_positive_crps_ml']=score['conditional_positive'].get('2',{}).get('crps_ml')
        assert abs(mu-row['methods']['amount_first']['mean_volumes_ml'][1])<1e-4
        records.append({k:row[k] for k in ('key','patient','horizon','cutoff','target')}|{'metrics':m})
    horizons={};patients=[]
    for h in (12,24):
        rows=[r for r in records if r['horizon']==h]
        horizons[str(h)]={name:summarize(rs) for name,rs in (
            ('all',rows),('future_present',[r for r in rows if r['metrics']['truth_ml']>0]),
            ('future_empty',[r for r in rows if r['metrics']['truth_ml']==0]))}
        for pat in sorted({r['patient'] for r in rows}):
            pr=[r for r in rows if r['patient']==pat]
            patients.append({'patient':pat,'horizon':h,'all':summarize(pr),
                'future_present':summarize([r for r in pr if r['metrics']['truth_ml']>0]),
                'future_empty':summarize([r for r in pr if r['metrics']['truth_ml']==0])})
        present=[p for p in patients if p['horizon']==h and p['future_present']['targets']]
        differences=[p['future_present']['patient_averaged']['amount_first_mean_volume_mae_ml']-
                     p['future_present']['patient_averaged']['persistence_mean_volume_mae_ml'] for p in present]
        horizons[str(h)]['nonempty_core_mae_vs_persistence_patients']={'better':sum(x<0 for x in differences),'worse':sum(x>0 for x in differences),'tied':sum(x==0 for x in differences),
            'difference_ml':mean(differences),'difference_without_most_improved_patient_ml':mean(sorted(differences)[1:])}
        # Deterministic zero/one scores can be infinite. Never present their finite-only average as full log loss.
        for subset in ('all','future_present','future_empty'):
            stats=horizons[str(h)][subset]['patient_averaged']
            for method in raw[0]['presence']:
                if stats.get('presence_infinite_log_score_'+method,0):stats['presence_log_score_'+method]=None
    private.mkdir(parents=True,exist_ok=True)
    for name,data in [('core-case-analysis.json',records),('core-patient-analysis.json',patients)]:
        (private/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    result={'analysis':'amount-first saved core predictions; one bounded interpretation',
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'targets':58,'horizons':horizons,
        'definitions':{'aggregation':'targets averaged within each patient and horizon/subset, then equal patients; patient sets overlap between empty/nonempty subsets',
            'positive_amount':'empirical distribution of saved positive core samples; no conditioning on future presence at inference; future-present references used only for evaluation',
            'median_and_interval':'inverse empirical CDF; 50th, 5th and 95th percentiles; each target summarized before patient averaging',
            'tail_share':'sum of largest 1,4,7 core samples divided by all 64 core samples; zeros included',
            'mean_deletion':'descriptive sensitivity only; no sample removal in official predictions, scores or gate',
            'probabilities':'exact saved autoregressive marginal probability, distinct from 64-draw presence frequency',
            'conditional_monte_carlo':'only the positive subset of the original 64 draws; no additional samples or quantile precision claim',
            'overall':'all selection targets including empty core; four-compartment mean metrics also retained',
            'infinite_presence_log_score':'null with positive infinite-score frequency means infinite loss, not an undefined event omitted from the score'},
        'private_patient_results':str(private/'core-patient-analysis.json'),
        'private_case_results':str(private/'core-case-analysis.json'),
        'conditional_interval_correction':{'scope':'one 24-week future-present target; saved predictions unchanged',
            'reason':'GPU float32 cumulative weight at rank38 of40 was0.949999928 < float32(0.95), selecting rank39 instead of38',
            'old_case_upper95_ml':24.276,'exact_equal_weight_case_upper95_ml':8.071,
            'old_patient_averaged_positive_width24_ml':34.263231217861176,
            'corrected_patient_averaged_positive_width24_ml':horizons['24']['future_present']['patient_averaged']['positive_width90_ml'],
            'coverage_unchanged':True,'unconditional_gate_unchanged':True},
        'new_fit_launched':False,'new_predictions_generated':False,'reserved_data_accessed':False,
        'recommended_single_correction':'Replace only the positive-count log-logistic family with a discretized log-normal truncated to the same feasible native-count capacity; retain explicit presence, autoregression, zero atoms, exact quantity allocation and fixed gates. This is a motivated hypothesis, not an established fix or authorization to fit.'}
    aggregate.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({h:{s:v[s] for s in ['all','future_present','future_empty']} for h,v in horizons.items()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('outputs/mu-amount-first-20261108/build/quantity-cases.json'))
    parser.add_argument('--private-out',type=Path,default=Path('outputs/mu-amount-first-20261108/analysis'))
    parser.add_argument('--out',type=Path,default=Path('info/patient_amount_first_interpretation.json'))
    args=parser.parse_args();analyze(args.source,args.private_out,args.out)
