"""Validation and staged scoring of R38 saved prediction arrays."""
from collections.abc import Mapping
import numpy as np

ARMS=('OH','OL','JH','JL'); SEEDS=(42,43,44); CONDITIONS=('identity','intensity','resolution')

def gaussian_crps(mean, sd, truth):
    from scipy.special import ndtr
    z=(truth-mean)/sd
    return sd*(z*(2*ndtr(z)-1)+2*np.exp(-z*z/2)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))

def mixture_crps(centers, weights, truth, observation_sd=.05):
    c=np.asarray(centers,float); w=np.asarray(weights,float); y=np.asarray(truth,float)
    if c.ndim!=3 or c.shape[0]!=len(y) or c.shape[2]!=3 or w.shape!=(c.shape[1],) or (w<0).any() or not np.isclose(w.sum(),1) or observation_sd<=0: raise ValueError('invalid mixture')
    # E|X-y| - 1/2 E|X-X'|, Gaussian components.
    def absnorm(delta, sd):
        from scipy.special import ndtr
        z=delta/sd
        return sd*np.sqrt(2/np.pi)*np.exp(-z*z/2)+delta*(2*ndtr(z)-1)
    first=np.sum(w[None,:,None]*absnorm(c-y[:,None,:],observation_sd),axis=1)
    delta=c[:,:,None,:]-c[:,None,:,:]; pair_sd=np.sqrt(2)*observation_sd
    second=np.sum(w[None,:,None,None]*w[None,None,:,None]*absnorm(delta,pair_sd),axis=(1,2))
    return first-.5*second

def _need(a,k,shape):
    if k not in a: raise ValueError(f'missing array {k}')
    x=np.asarray(a[k])
    if x.shape!=shape: raise ValueError(f'{k} shape {x.shape}, expected {shape}')
    if x.dtype.kind in 'fc' and not np.isfinite(x).all(): raise ValueError(f'nonfinite {k}')
    return x

def _weighted(values, patients):
    p=np.asarray(patients).astype(str); v=np.asarray(values,float)
    u,c=np.unique(p,return_counts=True); counts=dict(zip(u,c))
    return float(np.mean([np.mean(v[p==x]) for x in u]))

def _crps_rows(a,prefix,truth):
    m=np.asarray(a[prefix+'_mean'],float); s=np.asarray(a[prefix+'_intrinsic_scale'],float); k=float(np.asarray(a[prefix+'_calibration']))
    if m.shape!=truth.shape or s.shape!=truth.shape or (s<=0).any() or k not in (.5,1.,2.): raise ValueError(f'invalid Gaussian {prefix}')
    return gaussian_crps(m,np.sqrt((s*k)**2+.05**2),truth)

def analyze(arrays: Mapping[str,np.ndarray], identity: dict) -> dict:
    if not isinstance(arrays,Mapping): raise ValueError('arrays must be mapping')
    if not all(identity.get(k) for k in ('contract_digest','source_sha256','fitted_manifest_sha256')): raise ValueError('incomplete identity')
    if not np.array_equal(_need(arrays,'seeds',(3,)),SEEDS): raise ValueError('seed mismatch')
    rows=identity.get('cohort_rows')
    if not isinstance(rows,dict): raise ValueError('missing cohort identities')
    result={'status':'staged_only','identity':{k:identity[k] for k in ('contract_digest','source_sha256','fitted_manifest_sha256')},'protocol':{'arms':list(ARMS),'seeds':list(SEEDS),'bootstrap_draws':10000,'bootstrap_seed':2026},'cohorts':{}}
    for cohort in ('primary','single'):
        records=rows.get(cohort)
        if not records: raise ValueError(f'missing {cohort} identity')
        n=len(records); ids=np.array([r['row_id'] for r in records]); patients=np.array([r['patient_id'] for r in records]); truth=_need(arrays,cohort+'_truth',(n,3)).astype(float)
        if not np.array_equal(_need(arrays,cohort+'_row_ids',(n,)),ids) or not np.array_equal(_need(arrays,cohort+'_patient_ids',(n,)),patients): raise ValueError(f'{cohort} identity mismatch')
        scores={}; seed_scores={}
        for arm in ARMS:
            for seed in SEEDS:
                p=f'{cohort}_{arm}_{seed}'
                cr=_crps_rows(arrays,p,truth); seed_scores[f'{arm}_{seed}']=np.mean(cr,axis=1)
                for suffix in ('current_mean','current_dice','future_dice'):
                    x=_need(arrays,p+'_'+suffix,(n,3)).astype(float)
                    if suffix.endswith('dice') and ((x<0).any() or (x>1).any()): raise ValueError('dice outside [0,1]')
            scores[arm]=np.mean([_weighted(seed_scores[f'{arm}_{s}'],patients) for s in SEEDS])
        result['cohorts'][cohort]={'rows':n,'patients':len(set(patients)),'seed_scores':{k:float(np.mean(v)) for k,v in seed_scores.items()},'patient_uniform_crps':scores}
    if not rows.get('nuisance'): raise ValueError('missing nuisance identity')
    # Nuisance all fixed conditions/arms/seeds; no recalibration is permitted here.
    records=rows['nuisance']; n=len(records); patients=np.array([r['patient_id'] for r in records]); truth=_need(arrays,'nuisance_truth',(n,3)).astype(float); nuisance={}
    for arm in ARMS:
        for condition in CONDITIONS:
            for seed in SEEDS:
                p=f'nuisance_{arm}_{condition}_{seed}'; mean=_need(arrays,p+'_mean',(n,3)).astype(float); scale=_need(arrays,p+'_intrinsic_scale',(n,3)).astype(float)
                if (scale<=0).any(): raise ValueError('nonpositive nuisance scale')
                cal=float(np.asarray(arrays[p+'_calibration']))
                nuisance[f'{arm}_{condition}_{seed}']={'mae':_weighted(np.abs(mean-truth).mean(axis=1),patients),'drift':_weighted(np.abs(mean-np.asarray(arrays['nuisance_cutoff_measured'])).mean(axis=1),patients),'calibration':cal}
    result['nuisance']=nuisance
    return result
