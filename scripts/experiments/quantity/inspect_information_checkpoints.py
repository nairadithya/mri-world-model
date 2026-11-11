#!/usr/bin/env python3
"""Seed-17 saved-input diagnostic only: no fitting, sampling, or MRI loading."""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
from torch.nn import functional as F
from src.patient_state.amount_first import ORDER,count_log_mass
from src.patient_state.information_quantity import InformationQuantity
from src.patient_state.structured import VolumeChangePredictor
from src.patient_state.state_descriptors import DIMENSION,REGIONS
from src.patient_state.state import digest,write_json
from scripts.experiments.quantity.train_information_program import ARMS,arm_data


def weights(rows):
    counts=Counter((r['horizon'],r['patient']) for r in rows)
    patients=defaultdict(set)
    for h,p in counts:patients[h].add(p)
    return np.array([1/(len(patients)*len(patients[r['horizon']])*counts[r['horizon'],r['patient']]) for r in rows])


def distribution(values,w):
    v=np.asarray(values,dtype=float).reshape(-1);w=np.asarray(w,dtype=float).reshape(-1)
    if not len(v):return None
    if not np.isfinite(v).all():raise ValueError('Nonfinite diagnostic value')
    w=w/w.sum();order=np.argsort(v);cumulative=np.cumsum(w[order])
    q=[float(v[order[min(np.searchsorted(cumulative,p),len(v)-1)]]) for p in (.05,.5,.95,.99)]
    return dict(mean=float(v@w),rms=float(np.sqrt((v*v)@w)),
                p05=q[0],median=q[1],p95=q[2],p99=q[3],min=float(v.min()),max=float(v.max()))


def summarize_scalar(rows,key):
    return distribution([r[key] for r in rows],weights(rows)) if rows else None


def scopes(rows):
    return {'12':[r for r in rows if r['horizon']==12],'24':[r for r in rows if r['horizon']==24],'balanced_horizons':rows}


@torch.no_grad()
def model_diagnostics(model,data,saved):
    rows=[];replay_difference=0.
    for item in data:
        x=item['scalar'];v=item['current'];n=item['truth'][:,ORDER];z=(n>0).float()
        hidden,mu0,scale0=model.embedding(x,v)
        presence=[];amount=[];conditional=[];remaining=item['capacity'].clone()
        for i in range(4):
            logit=model.presence[i](torch.cat((hidden,z[:,:i]),1))[:,0]
            presence.append(float(F.binary_cross_entropy_with_logits(logit,z[:,i])))
            mu,sigma=model.count_parameters(hidden,z,n[:,:i],mu0,scale0,i)
            maximum=remaining-z[:,i+1:].sum(1).long()
            if n[0,i]>0:
                loss=-count_log_mass(n[:,i],maximum,mu,sigma,'gaussian')
                residual=float(n[0,i].double().log()-mu[0].double());s=float(sigma[0])
                conditional.append(dict(residual_log_count=residual,absolute_residual_log_count=abs(residual),
                    sigma_log_count=s,log_sigma=math.log(s),absolute_standardized_residual=abs(residual)/s,
                    near_scale_floor=float(s<=1.05*.01*math.pi/math.sqrt(3)),sigma_below_point05=float(s<.05),
                    sigma_below_point1=float(s<.1)))
                amount.append(float(loss))
            else:conditional.append(None);amount.append(0.)
            remaining-=n[:,i]
        joint=sum(presence)+sum(amount)
        if item['key'] in saved:
            old=saved[item['key']]
            replay_difference=max(replay_difference,abs(joint-old['joint_nll']),abs(sum(amount)-old['conditional_count_nll']))
        rows.append(dict(patient=item['pair']['patient'],horizon=item['pair']['horizon'],joint_nll=joint,
            presence_nll=sum(presence),conditional_amount_nll=sum(amount),presence_terms=presence,amount_terms=amount,
            conditional=conditional,fused_hidden_l2=float(hidden.norm())))
    result={}
    names=('core','enhancing','cavity','flair_abnormality')
    for scope,items in scopes(rows).items():
        w=weights(items)
        s={'pairs':len(items),'patients':len({r['patient'] for r in items}),
            **{k:float(np.array([r[k] for r in items])@w) for k in ('joint_nll','presence_nll','conditional_amount_nll')},
            'likelihood_distribution':{k:summarize_scalar(items,k) for k in ('joint_nll','presence_nll','conditional_amount_nll')},
            'fused_hidden_l2':summarize_scalar(items,'fused_hidden_l2'),'compartments':{}}
        for i,name in enumerate(names):
            positive=[dict(r,**r['conditional'][i]) for r in items if r['conditional'][i] is not None]
            s['compartments'][name]={'present_pairs':len(positive),'present_patients':len({r['patient'] for r in positive}),
                'presence_nll_all':float(np.array([r['presence_terms'][i] for r in items])@w),
                'conditional_amount_nll_all':float(np.array([r['amount_terms'][i] for r in items])@w),
                'teacher_forced_present':{key:summarize_scalar(positive,key) for key in
                    ('residual_log_count','absolute_residual_log_count','sigma_log_count','log_sigma','absolute_standardized_residual')},
                'teacher_forced_present_fractions':{key:float(np.array([r[key] for r in positive])@weights(positive)) if positive else None
                    for key in ('near_scale_floor','sigma_below_point05','sigma_below_point1')}}
        result[scope]=s
    return result,replay_difference


def input_diagnostics(data,root):
    cache={}
    def cached(stem):
        if stem not in cache:
            with np.load(root/'descriptors'/(stem+'.npz')) as f:
                # Read only existing small descriptors/counts, never cached maps.
                cache[stem]=(f['descriptor'],f['region_counts'])
        return cache[stem]
    norm=json.loads((root/'normalization.json').read_text())['expanded']
    fitting=[cached(s) for s in norm['fitting_stems']]
    scale=np.array(norm['scale']).reshape(6,257)
    normalization_support={}
    for i,name in enumerate(REGIONS):
        raw=np.array([d[0].reshape(6,257)[i,:256] for d in fitting if d[1][i]>0])
        sd=raw.std(0,dtype=np.float64) if len(raw) else np.zeros(256)
        normalization_support[name]=dict(fitting_unique_current_visits_nonempty=len(raw),
            fitting_unique_current_visits_empty=len(fitting)-len(raw),
            features_using_unit_scale_fallback=int((sd<=1e-6).sum()),
            scale_min=float(scale[i,:256].min()),scale_median=float(np.median(scale[i,:256])),scale_max=float(scale[i,:256].max()))
    result={}
    for split,items in data.items():
        rows=[]
        for item in items:
            x=item['inputs']['expanded',False][0].cpu().numpy()
            hist=item['pair']['history'];now=hist[-1];past=hist[-2] if item['history_available'] else None
            parts={'base_scalar':x[:17],'current_descriptor':x[17:17+DIMENSION],
                   'history_metadata':x[17+DIMENSION:19+DIMENSION],'previous_volumes_rates':x[19+DIMENSION:27+DIMENSION],
                   'previous_descriptor':x[27+DIMENSION:]}
            row=dict(patient=item['pair']['patient'],horizon=item['pair']['horizon'],history_available=item['history_available'],
                norms={name:float(np.linalg.norm(v)) for name,v in parts.items()},
                current=(now['stem'],parts['current_descriptor'].reshape(6,257),cached(now['stem'])[1]),
                previous=(past['stem'],parts['previous_descriptor'].reshape(6,257),cached(past['stem'])[1]) if past else None)
            rows.append(row)
        result[split]={}
        for scope,subset in scopes(rows).items():
            w=weights(subset);entry={'pairs':len(subset),'patients':len({r['patient'] for r in subset}),
                'history_available_fraction':float(np.array([r['history_available'] for r in subset])@w),
                'input_l2_norms':{key:distribution([r['norms'][key] for r in subset],w) for key in subset[0]['norms']},'descriptors':{}}
            available=[r for r in subset if r['history_available']]
            entry['input_l2_norms_given_history_available']={key:distribution([r['norms'][key] for r in available],weights(available))
                for key in ('previous_volumes_rates','previous_descriptor')} if available else {}
            for which in ('current','previous'):
                observed=[r for r in subset if r[which] is not None];entry['descriptors'][which]={}
                for i,name in enumerate(REGIONS):
                    present=[r for r in observed if r[which][2][i]>0]
                    unique={r[which][0]:r for r in observed}
                    vals=np.stack([r[which][1][i,:256] for r in present]) if present else np.empty((0,256))
                    absvals=np.abs(vals);pw=weights(present) if present else np.array([])
                    cw=np.repeat(pw/256,256)
                    entry['descriptors'][which][name]=dict(observed_pairs=len(observed),unique_visits=len(unique),
                        nonempty_pairs=len(present),nonempty_patients=len({r['patient'] for r in present}),
                        empty_unique_visits=sum(r[which][2][i]==0 for r in unique.values()),
                        nonempty_unique_visits=sum(r[which][2][i]>0 for r in unique.values()),
                        rare_nonempty_unique_visits_le64voxels=sum(0<r[which][2][i]<=64 for r in unique.values()),
                        rare_nonempty_unique_visits_le512voxels=sum(0<r[which][2][i]<=512 for r in unique.values()),
                        empty_pair_fraction=float(np.array([r[which][2][i]==0 for r in observed])@weights(observed)) if observed else None,
                        nonempty_native_voxels=distribution([r[which][2][i] for r in present],pw) if present else None,
                        nonempty_abs_z=distribution(absvals.ravel(),cw) if present else None,
                        nonempty_fraction_abs_z_gt5=float((absvals.ravel()>5)@cw) if present else None,
                        nonempty_fraction_abs_z_gt10=float((absvals.ravel()>10)@cw) if present else None,
                        empty_moment_max_abs=max((float(np.abs(r[which][1][i,:256]).max()) for r in observed if r[which][2][i]==0),default=0.))
            result[split][scope]=entry
    return dict(normalizer_fitting_support=normalization_support,by_split=result)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);data=torch.load(a.root/'prepared.pt',weights_only=False)
    result=dict(scope='Saved seed17 checkpoints and prepared fitting/selection inputs only; no training, sampling, image loading or gate changes.',
        aggregation='Within each horizon: mean pairs within patient, then equal patients. Balanced_horizons averages the two horizon distributions equally. Positive-only summaries reweight eligible patients within each horizon. Quantiles use that weighted empirical distribution.',
        teacher_forcing='NLL and residual/scale diagnostics condition on true future presence patterns and preceding true future counts. Diagnostic supervision contexts only, not deployed predictions or forecasts.',
        residual_definition='ln(reference native count) minus untruncated Gaussian log-count location mu; sigma_log_count is Gaussian standard deviation, log_sigma is its natural logarithm. Residual/sigma is diagnostic, not truncated-distribution coverage.',
        input_definition='Prepared expanded-pool standardized inputs. Z summaries exclude empty-region moments and the binary indicator. Each of the 256 moment channels has equal weight within a pair. Norms include indicators; previous input norms include missing-history zeros. Rare thresholds are one/8 native 4mm cells (64/512 one-mm voxels), descriptive only.',
        historical_comparison_sha256=digest(a.root/'comparison.json'),prepared_inputs_sha256=digest(a.root/'prepared.pt'),models={},inputs=input_diagnostics(data,a.root))
    for arm in ('expanded_scalar','expanded_mri','memory_mri'):
        config=ARMS[arm];model=InformationQuantity(VolumeChangePredictor('current',compartments=4),current_mri=config['current_mri'],memory=config['memory']).cuda().eval()
        checkpoint=a.root/'runs'/f'{arm}-s17'/'quantity-terminal.pt'
        model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cuda')['model'])
        saved={r['key']:r for r in json.loads((checkpoint.parent/'quantity-cases.json').read_text())}
        selected=arm_data(data,config);result['models'][arm]={'checkpoint_sha256':digest(checkpoint)}
        for split,items in selected.items():
            summary,error=model_diagnostics(model,items,saved if split=='selection' else {})
            result['models'][arm][split]=summary
            if split=='selection':
                result['models'][arm]['max_saved_selection_nll_difference']=error
                if error>1e-4:raise AssertionError('Diagnostic likelihood differs from saved assessment')
    # JSON supports no numpy scalars; summaries contain no participant identifiers.
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False,default=lambda x:x.item())+'\n')
    print(json.dumps({arm:{split:{h:{k:v[k] for k in ('joint_nll','presence_nll','conditional_amount_nll')} for h,v in m[split].items()} for split in ('fitting','selection')} for arm,m in result['models'].items()}),flush=True)


if __name__=='__main__':main()
