"""Stable Gaussian log-count likelihood and inverse-CDF sampling; no floors."""
import numpy as np
import torch


def gaussian_log_cdf_difference(lo, hi):
    # Reflect positive-tail intervals before evaluating either log CDF. Selecting
    # transformed inputs avoids NaN gradients from an unused saturated branch.
    reflect=lo>0
    upper=torch.where(reflect,-lo,hi)
    lower=torch.where(reflect,-hi,lo)
    log_upper=torch.special.log_ndtr(upper)
    log_lower=torch.special.log_ndtr(lower)
    return log_upper+torch.log(-torch.expm1(log_lower-log_upper))


def gaussian_sample_count(maximum,mu,sigma,uniform):
    from scipy.special import log_ndtr, ndtri_exp
    original_device=mu.device
    maximum,mu,sigma,uniform=[x.detach().double().cpu().numpy() for x in (maximum,mu,sigma,uniform)]
    if np.any(maximum<1) or np.any((uniform<=0)|(uniform>=1)) or np.any(sigma<=0):
        raise ValueError('Invalid truncated Gaussian count support')
    low=(np.log(.5)-mu)/sigma;high=(np.log(maximum+.5)-mu)/sigma
    logp=np.logaddexp(log_ndtr(low)+np.log1p(-uniform),log_ndtr(high)+np.log(uniform))
    logq=np.logaddexp(log_ndtr(-low)+np.log1p(-uniform),log_ndtr(-high)+np.log(uniform))
    # Invert the smaller tail directly in log probability; neither exp(logp)
    # underflow nor subtraction from one can destroy an extreme probability.
    z=np.empty_like(logp);left=logp<=logq
    z[left]=ndtri_exp(logp[left]);z[~left]=-ndtri_exp(logq[~left])
    result=np.floor(np.exp(mu+sigma*z)+.5).astype(np.int64)
    if np.any(result<1) or np.any(result>maximum):
        raise ArithmeticError('Gaussian inverse CDF produced infeasible counts; no clipping or resampling')
    return torch.from_numpy(result).to(original_device)
