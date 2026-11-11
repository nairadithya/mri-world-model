#!/usr/bin/env python3
"""Fixed nested patient CV, excluded-patient scale fitting, then locked S/M models."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time
import os
from contextlib import nullcontext
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import minimize
from src.patient_state.linear_quantity import (LinearQuantity,patient_horizon_weights,fit_transform,transform,save_bundle,load_bundle,predict,count_log_mass)
from src.patient_state.amount_first import ORDER
from src.patient_state.core_cdf import core_cdf_scores
from src.patient_state.state import digest,write_json

SOURCE=Path('/home/aditya/experiments/world-model-information-program-20261108')
PENALTIES=(.01,.1,1.,10.,100.)
SOLVER=dict(method='L-BFGS-B',options=dict(gtol=1e-9,ftol=1e-14,maxiter=2000,maxls=50,maxcor=20))
REFINEMENT=dict(method='trust-exact',options=dict(gtol=1e-6,maxiter=100))


def records(source,split):
    prepared=torch.load(source/'prepared.pt',map_location='cpu',weights_only=False)[split]
    cache={};rows=[]
    for item in prepared:
        pair=item['pair'];stem=pair['history'][-1]['stem']
        if stem not in cache:
            with np.load(source/'descriptors'/(stem+'.npz')) as f:
                cache[stem]=(f['descriptor'],f['estimated_counts'])
        descriptor,counts=cache[stem]
        rows.append(dict(patient=pair['patient'],horizon=pair['horizon'],stem=stem,key=item['key'],random_key=item['random_key'],
            current_counts=counts,descriptor=descriptor,truth=item['truth'][0].numpy(),capacity=int(item['capacity']),
            old=item.get('old')))
    return rows


def folds(rows,k,seed=17):
    patients=sorted({r['patient'] for r in rows});rng=np.random.default_rng(seed)
    shuffled=np.array(patients)[rng.permutation(len(patients))]
    return {str(p):i%k for i,p in enumerate(shuffled)}


def tensors(rows,recipe):
    return (transform(rows,recipe),torch.tensor(np.stack([r['current_counts'] for r in rows]),dtype=torch.double),
        torch.tensor(np.stack([r['truth'] for r in rows]),dtype=torch.long),torch.tensor([r['capacity'] for r in rows]),
        torch.tensor(patient_horizon_weights(rows),dtype=torch.double))


def optimize(fun,start):
    def objective(value):
        theta=torch.tensor(value,dtype=torch.double,requires_grad=True)
        loss=fun(theta)
        gradient,=torch.autograd.grad(loss,theta)
        if not torch.isfinite(loss) or not torch.isfinite(gradient).all():raise ArithmeticError('Nonfinite exact likelihood objective/gradient')
        return float(loss.detach()),gradient.detach().numpy()
    initial=minimize(objective,np.asarray(start,dtype=float),jac=True,**SOLVER)
    def hessian(value):
        theta=torch.tensor(value,dtype=torch.double,requires_grad=True)
        return torch.autograd.functional.hessian(fun,theta).detach().numpy()
    # Apply the same standard Newton trust-region refinement to every fit.
    # This addresses relative-function stopping before the gradient criterion;
    # it does not relax the criterion or change the fitted objective.
    result=minimize(objective,initial.x,jac=True,hess=hessian,**REFINEMENT)
    loss,grad=objective(result.x);norm=float(np.max(np.abs(grad)))
    receipt=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit),evaluations=int(result.nfev),
        objective=loss,gradient_infinity_norm=norm,accepted=bool(result.success and norm<=1e-6),
        initial_lbfgs=dict(success=bool(initial.success),message=str(initial.message),iterations=int(initial.nit),evaluations=int(initial.nfev)))
    return result.x,receipt


def fit(rows,recipe,penalty):
    model=LinearQuantity(recipe['mri']);x,current,truth,capacity,w=tensors(rows,recipe)
    n=truth[:,ORDER];z=(n>0).double();offset=current.clamp_min(1).log()[:,ORDER];remaining=capacity.clone();receipts=[]
    for i in range(4):
        design=torch.cat((x,z[:,:i],torch.ones(len(x),1)),1)
        objective=lambda theta:(w*F.binary_cross_entropy_with_logits(design@theta,z[:,i],reduction='none')).sum()+penalty*.5*theta.square().sum()
        coef,receipt=optimize(objective,np.zeros(design.shape[1]));receipts.append(dict(kind='presence',compartment=i,**receipt))
        if not receipt['accepted']:return None,receipts
        with torch.no_grad():model.presence[i].weight.copy_(torch.tensor(coef[:-1])[None]);model.presence[i].bias.copy_(torch.tensor(coef[-1:]))
        design=torch.cat((x,z,torch.log1p(n[:,:i].double())/16,torch.ones(len(x),1)),1)
        maximum=remaining-z[:,i+1:].sum(1).long();positive=n[:,i]>0
        def objective(theta):
            mu=offset[:,i]+design@theta
            likelihood=-count_log_mass(n[positive,i],maximum[positive],mu[positive],torch.ones(int(positive.sum()),dtype=torch.double),'gaussian')
            return (w[positive]*likelihood).sum()+penalty*.5*theta.square().sum()
        coef,receipt=optimize(objective,np.zeros(design.shape[1]));receipts.append(dict(kind='amount',compartment=i,positive_examples=int(positive.sum()),**receipt))
        if not receipt['accepted']:return None,receipts
        with torch.no_grad():model.amount[i].weight.copy_(torch.tensor(coef[:-1])[None]);model.amount[i].bias.copy_(torch.tensor(coef[-1:]))
        remaining-=n[:,i]
    return model.eval(),receipts


@torch.no_grad()
def score_rows(model,recipe,rows):
    x,current,truth,capacity,w=tensors(rows,recipe);t=model.terms(x,current,truth,capacity)
    p=t['presence'].sum(1);q=t['amount'].sum(1)
    return [dict(key=r['key'],patient=r['patient'],horizon=r['horizon'],presence_nll=float(p[i]),amount_nll=float(q[i]),joint_nll=float(p[i]+q[i]),
        mu=t['mu'][i].tolist(),maximum=t['maximum'][i].tolist(),counts=t['counts'][i].tolist(),
        presence_probability=model.presence_log_marginals(x[i:i+1],current[i:i+1])[:,1].exp().tolist()) for i,r in enumerate(rows)]


def summarize_scores(rows):
    result={}
    for label,rs in [('12',[r for r in rows if r['horizon']==12]),('24',[r for r in rows if r['horizon']==24]),('balanced',rows)]:
        w=patient_horizon_weights(rs)
        result[label]=dict(pairs=len(rs),patients=len({r['patient'] for r in rs}),**{k:float(w@np.array([r[k] for r in rs])) for k in ('joint_nll','presence_nll','amount_nll')})
    return result


def choose_penalty(rows,assignment,mri,out):
    candidates={str(p):[] for p in PENALTIES};receipts=[];failures=[]
    for fold in sorted(set(assignment.values())):
        training=[r for r in rows if assignment[r['patient']]!=fold];validation=[r for r in rows if assignment[r['patient']]==fold]
        recipe=fit_transform(training,mri)
        assert set(recipe['fitting_patients']).isdisjoint(r['patient'] for r in validation)
        for penalty in PENALTIES:
            model,convergence=fit(training,recipe,penalty)
            receipts.append(dict(fold=fold,penalty=penalty,heads=convergence))
            if model is None:failures.append(dict(fold=fold,penalty=penalty));continue
            candidates[str(penalty)].extend(score_rows(model,recipe,validation))
    summaries={p:summarize_scores(v) for p,v in candidates.items() if len(v)==len(rows)}
    write_json(out,dict(assignment=assignment,penalties=summaries,convergence=receipts,failures=failures))
    if failures:raise RuntimeError('Convergence failure in fixed penalty comparison; see '+str(out))
    best=min(v['balanced']['joint_nll'] for v in summaries.values())
    # A numerical equality tolerance only; no performance-margin search.
    chosen=max(float(p) for p,v in summaries.items() if abs(v['balanced']['joint_nll']-best)<=1e-10)
    return chosen,summaries[str(chosen)],candidates[str(chosen)]


def fit_scales(oof):
    w=torch.tensor(patient_horizon_weights(oof),dtype=torch.double)
    n=torch.tensor([r['counts'] for r in oof],dtype=torch.long);maximum=torch.tensor([r['maximum'] for r in oof],dtype=torch.long)
    mu=torch.tensor([r['mu'] for r in oof],dtype=torch.double);sigmas=[];receipts=[]
    for i in range(4):
        valid=n[:,i]>0
        if not valid.any():raise ValueError('No excluded-patient positive outcomes for scale fitting')
        pw=w[valid]/w[valid].sum()
        def objective(log_sigma):
            sigma=log_sigma[0].exp().expand(int(valid.sum()))
            return -(pw*count_log_mass(n[valid,i],maximum[valid,i],mu[valid,i],sigma,'gaussian')).sum()
        value,receipt=optimize(objective,[0.])
        sigma=float(np.exp(value[0]))
        profile={str(delta):float(objective(torch.tensor(value+delta)).detach()) for delta in (-2.,-.5,.5,2.)}
        supported=receipt['accepted'] and np.isfinite(sigma) and sigma>0 and all(v>receipt['objective']+1e-8 for v in profile.values())
        receipts.append(dict(compartment=i,sigma=sigma,log_sigma=float(value[0]),positive_pairs=int(valid.sum()),
            local_profile=profile,supported_finite_optimum=bool(supported),**receipt))
        if not supported:raise RuntimeError('Unsupported finite scale optimum: '+json.dumps(receipts[-1]))
        sigmas.append(sigma)
    return sigmas,receipts


def smoke(rows,out):
    outer=folds(rows,5);outer_training=[r for r in rows if outer[r['patient']]!=0]
    inner=folds(outer_training,3);training=[r for r in outer_training if inner[r['patient']]!=0]
    excluded=[r for r in rows if outer[r['patient']]==0]
    recipe=fit_transform(training,True)
    assert set(recipe['fitting_patients']).isdisjoint(r['patient'] for r in excluded)
    assert all(inner[r['patient']]!=0 for r in training)
    frozen=json.dumps(recipe,sort_keys=True)
    transform([dict(r,descriptor=r['descriptor']*123) for r in excluded],recipe)
    assert json.dumps(recipe,sort_keys=True)==frozen
    model,receipts=fit(training,recipe,.1)
    if model is None:raise RuntimeError('Smoke optimizer failed: '+json.dumps(receipts))
    save_bundle(out/'smoke.pt',model,recipe,dict(smoke=True))
    restored,restored_recipe,_=load_bundle(out/'smoke.pt','cuda');model=model.cuda();item=excluded[0]
    args=[item['current_counts'],item['descriptor'],item['capacity'],item['horizon'],item['random_key']]
    a=predict(model,recipe,*args);b=predict(restored,restored_recipe,*args)
    assert a==b
    counts=np.array(a['sample_counts']);assert (counts>=0).all() and (counts.sum(1)<=item['capacity']).all()
    x=transform([item],recipe,'cuda');current=torch.tensor(item['current_counts'],dtype=torch.double,device='cuda')[None]
    score=core_cdf_scores(model,x,current,item['capacity'],int(item['truth'][1]))
    write_json(out/'smoke.json',dict(training_pairs=len(training),excluded_fitting_pairs=len(excluded),nested_patient_separation=True,
        transform_fit_patients=recipe['fitting_patients'],fit_complete=True,convergence=receipts,reload_exact=True,
        feasible_joint_samples=64,core_cdf=score,selection_used=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    started=time.monotonic()
    with nullcontext():
        rows=records(a.source,'fitting')
        if len(rows)!=156 or len({r['patient'] for r in rows})!=56:raise ValueError('Fitting population changed')
        if a.smoke:smoke(rows,a.out);return
        if not (a.out/'smoke.json').exists():raise ValueError('Real-data smoke required')
        outer=folds(rows,5);inner={str(f):folds([r for r in rows if outer[r['patient']]!=f],3) for f in range(5)}
        write_json(a.out/'folds.json',dict(outer=outer,inner=inner,final_cv=outer,seed=17))
        write_json(a.out/'protocol.json',dict(identity=LinearQuantity.identity,arms=['S','M'],penalties=PENALTIES,solver=SOLVER,refinement=REFINEMENT,
            convergence='scipy success and gradient infinity norm <=1e-6; no global convexity claim',training_sigma=1.,pca_components=8,
            scalar_features='patient/unique-visit standardized log1p native counts, 24w indicator, six ROI-empty flags',
            MRI='region centering from nonempty training visits, mean imputation; patient-balanced deterministic SVD, no whitening, one common score RMS',
            calibration='four global sigmas from outer-excluded teacher-forced location predictions; no location refit after attachment',
            source_population_sha256=digest(a.source/'data/population.json'),observation_sha256='d006424493442bd1bcd8fd5b15888b197aa3a95ef3f87aeed5df2f52e86ccacb',
            validation_scope='head-level cross-fitting conditional on an observer supervised on fitting observations; selection repeatedly developmental',
            selection_evaluated=False))
        locked={}
        for arm in ('S','M'):
            directory=a.out/arm;directory.mkdir(exist_ok=True);oof=[];outer_results=[]
            for fold in range(5):
                training=[r for r in rows if outer[r['patient']]!=fold];held=[r for r in rows if outer[r['patient']]==fold]
                penalty,cv,_=choose_penalty(training,inner[str(fold)],arm=='M',directory/f'outer-{fold}-inner-cv.json')
                recipe=fit_transform(training,arm=='M');model,convergence=fit(training,recipe,penalty)
                if model is None:raise RuntimeError('Outer refit did not converge')
                prediction=score_rows(model,recipe,held);oof.extend(prediction)
                save_bundle(directory/f'outer-{fold}.pt',model,recipe,dict(penalty=penalty,fold=fold,convergence=convergence))
                outer_results.append(dict(fold=fold,penalty=penalty,inner_score=cv,convergence=convergence,
                    training_patients=recipe['fitting_patients'],excluded_patients=sorted({r['patient'] for r in held})))
                print(json.dumps(dict(arm=arm,outer_complete=fold+1,seconds=time.monotonic()-started)),flush=True)
            if len({r['key'] for r in oof})!=156:raise AssertionError('Outer predictions missing or duplicated')
            write_json(directory/'outer-predictions.json',oof)
            try:
                sigmas,calibration=fit_scales(oof)
            except Exception as error:
                write_json(directory/'scale-failure.json',dict(error=str(error),locations_fixed=True))
                raise
            penalty,cv,cv_records=choose_penalty(rows,outer,arm=='M',directory/'final-penalty-cv.json')
            recipe=fit_transform(rows,arm=='M');model,convergence=fit(rows,recipe,penalty)
            if model is None:raise RuntimeError('Final refit did not converge')
            before=score_rows(model,recipe,rows)
            with torch.no_grad():model.sigmas.copy_(torch.tensor(sigmas,dtype=torch.double))
            after=score_rows(model,recipe,rows)
            metadata=dict(arm=arm,penalty=penalty,sigmas_autoregressive_order=sigmas,calibration=calibration,
                outer_folds=outer_results,outer_fixed_sigma_score=summarize_scores(oof),final_cv_fixed_sigma_score=cv,
                final_fit_fixed_sigma_score=summarize_scores(before),final_fit_attached_sigma_score=summarize_scores(after),
                convergence=convergence,observer_sha256=json.loads((a.out/'protocol.json').read_text())['observation_sha256'])
            save_bundle(a.out/f'final-{arm}.pt',model,recipe,metadata)
            write_json(directory/'fit-summary.json',metadata)
            locked[arm]=dict(checkpoint=f'final-{arm}.pt',sha256=digest(a.out/f'final-{arm}.pt'))
        write_json(a.out/'locked-final-models.json',dict(models=locked,selection_evaluated=False,training_seconds=time.monotonic()-started,
            condition='Both final models fixed before selection evaluation; no location refit after attaching excluded-patient scales'))
        print('LOCKED '+json.dumps(locked),flush=True)


if __name__=='__main__':main()
