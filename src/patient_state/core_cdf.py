"""Deterministic discrete core moments/CRPS, integrating all presence patterns."""
import itertools
import math
import torch
from torch.nn import functional as F
from .amount_first import log_cdf_difference
from .gaussian_count import gaussian_log_cdf_difference


@torch.no_grad()
def core_parameters(model,scalar,volumes,capacity):
    hidden,mu0,scale0=model.embedding(scalar,volumes)
    if len(hidden)!=1 or capacity<4:raise ValueError('One valid cutoff required')
    z=hidden.new_tensor(list(itertools.product((0.,1.),repeat=4)))
    expanded=hidden.expand(16,-1);logp=torch.zeros(16,device=hidden.device,dtype=torch.double)
    for i in range(4):
        logit=model.presence[i](torch.cat((expanded,z[:,:i]),1))[:,0].double()
        logp+=torch.where(z[:,i].bool(),F.logsigmoid(logit),F.logsigmoid(-logit))
    positive=z[:,0].bool();present=z[positive]
    mu,scale=model.count_parameters(hidden.expand(8,-1),present,hidden.new_zeros(8,0),mu0.expand(8,-1),scale0.expand(8,-1),0)
    maximum=capacity-present[:,1:].sum(1).long()
    return {'log_weights':logp[positive],'absence_probability':float(logp[~positive].exp().sum()),
            'mu':mu.double(),'scale':scale.double(),'maximum':maximum,'capacity':capacity,'family':model.family}


@torch.no_grad()
def core_survival(parameters,counts):
    """P(core count > n) at integer n, including absence and all later presences."""
    n=counts.double();mu=parameters['mu'];scale=parameters['scale'];maximum=parameters['maximum']
    difference=gaussian_log_cdf_difference if parameters['family']=='gaussian' else log_cdf_difference
    log_survival=torch.full((len(n),8),-torch.inf,device=n.device,dtype=torch.double)
    for i in range(8):
        supported=n<maximum[i]
        v=n[supported];lower=(math.log(.5)-mu[i])/scale[i];upper=(torch.log(maximum[i].double()+.5)-mu[i])/scale[i]
        cut=(torch.log(v+.5)-mu[i])/scale[i]
        log_survival[supported,i]=parameters['log_weights'][i]+difference(cut,upper)-difference(lower,upper)
    log_s=torch.logsumexp(log_survival,dim=1)
    return log_s.exp(),-torch.expm1(log_s)


@torch.no_grad()
def core_cdf_scores(model,scalar,volumes,capacity,truth_count,chunk=65536):
    if not 0<=truth_count<=capacity:raise ValueError('Core reference outside feasible count domain')
    parameters=core_parameters(model,scalar,volumes,capacity)
    mean=scalar.new_zeros((),dtype=torch.double);crps=mean.clone();under=mean.clone();over=mean.clone()
    previous=1.;max_violation=0.
    for start in range(0,capacity,chunk):
        n=torch.arange(start,min(start+chunk,capacity),device=scalar.device)
        survival,cdf=core_survival(parameters,n)
        if not torch.isfinite(survival).all() or not torch.isfinite(cdf).all():raise ArithmeticError('Nonfinite core CDF')
        violation=max(float(survival[0])-previous,float((survival[1:]-survival[:-1]).max()) if len(n)>1 else 0.)
        max_violation=max(max_violation,violation);previous=float(survival[-1])
        if float(cdf.min()) < -1e-12 or float(cdf.max()) > 1+1e-12 or violation>1e-12:
            raise ArithmeticError('Invalid core CDF; no numerical clipping')
        mean+=survival.sum()
        below=(n<truth_count);under+=cdf[below].square().sum();over+=survival[~below].square().sum()
    crps=under+over
    return {'mean_ml':float(mean)/1000,'mean_volume_mae_ml':abs(float(mean)-truth_count)/1000,
        'crps_ml':float(crps)/1000,'crps_underprediction_ml':float(under)/1000,'crps_overprediction_ml':float(over)/1000,
        'presence_probability':1-parameters['absence_probability'],'feasible_core_counts_evaluated':capacity,
        'presence_patterns_integrated':16,'positive_patterns_integrated':8,'max_cdf_monotonicity_violation':max_violation}
