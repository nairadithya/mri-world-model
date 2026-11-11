"""Explicit presence, feasible native counts, and quantity-preserving location.

Tensor/report order is legacy label order (1,2,3,4). Autoregression and spatial
allocation use core, enhancing, cavity, FLAIR (2,1,4,3). No target enters sample().
"""
import copy
import hashlib
import itertools
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ORDER = (1, 0, 3, 2)  # zero-based legacy channels
LABEL_ORDER = (2, 1, 4, 3)


def log_cdf_difference(lo, hi):
    """log(sigmoid(hi)-sigmoid(lo)), stable also in either tail."""
    return F.logsigmoid(hi) + F.logsigmoid(-lo) + torch.log(-torch.expm1(lo-hi))


def count_log_mass(n, maximum, mu, scale, family="logistic"):
    # Double precision protects the very narrow bins at large native counts.
    n, maximum, mu, scale = [x.double() for x in (n, maximum, mu, scale)]
    if bool(((n < 1) | (n > maximum)).any()):
        raise ValueError('Reference count outside conditional feasible support')
    difference=log_cdf_difference
    if family=='gaussian':
        from .gaussian_count import gaussian_log_cdf_difference
        difference=gaussian_log_cdf_difference
    elif family!='logistic':raise ValueError('Unknown count family')
    low, high = torch.log(n-.5), torch.log(n+.5)
    return (difference((low-mu)/scale, (high-mu)/scale)
            - difference((math.log(.5)-mu)/scale, (torch.log(maximum+.5)-mu)/scale))


def sample_count(maximum, mu, scale, uniform, family="logistic"):
    if family=='gaussian':
        from .gaussian_count import gaussian_sample_count
        return gaussian_sample_count(maximum,mu,scale,uniform)
    if family!='logistic':raise ValueError('Unknown count family')
    maximum, mu, scale, uniform = [x.double() for x in (maximum, mu, scale, uniform)]
    if bool(((maximum < 1) | (uniform <= 0) | (uniform >= 1)).any()):
        raise ValueError('Invalid truncated-count support or random variate')
    low, high = (math.log(.5)-mu)/scale, (torch.log(maximum+.5)-mu)/scale
    log_p = torch.logaddexp(F.logsigmoid(low)+torch.log1p(-uniform), F.logsigmoid(high)+uniform.log())
    log_q = torch.logaddexp(F.logsigmoid(-low)+torch.log1p(-uniform), F.logsigmoid(-high)+uniform.log())
    result = torch.floor(torch.exp(mu+scale*(log_p-log_q))+.5).long()
    # Never repair a draw by clipping or rejection resampling.
    if bool(((result < 1) | (result > maximum)).any()):
        raise ArithmeticError('Inverse CDF produced an infeasible native count')
    return result


class AmountFirst(nn.Module):
    """Two-layer 32-unit scalar backbone with two small autoregressive heads."""
    def __init__(self, scalar_control, prevalence=None, family="logistic"):
        super().__init__()
        if family not in ('logistic','gaussian'):raise ValueError('Unknown count family')
        self.family=family
        self.identity='mu-amount-first-gaussian-v1' if family=='gaussian' else 'mu-amount-first-v1'
        self.backbone = copy.deepcopy(scalar_control.network[:4]).requires_grad_(True)
        self.legacy_projection = copy.deepcopy(scalar_control.network[4])
        self.legacy_projection.requires_grad_(False)
        self.presence = nn.ModuleList([nn.Linear(32+i, 1) for i in range(4)])
        self.amount = nn.ModuleList([nn.Linear(32+4+i, 2) for i in range(4)])
        for i, (ph, ah) in enumerate(zip(self.presence, self.amount)):
            nn.init.zeros_(ph.weight); nn.init.zeros_(ah.weight); nn.init.zeros_(ah.bias)
            p = .5 if prevalence is None else float(prevalence[ORDER[i]])
            nn.init.constant_(ph.bias, math.log(max(.01,min(.99,p))/(1-max(.01,min(.99,p)))))

    def embedding(self, scalar, volumes):
        hidden = self.backbone(scalar)
        raw = self.legacy_projection(hidden).reshape(-1,8,5)
        old = torch.expm1((torch.log1p(volumes)[:,None]+raw[:,:,:4]).clamp(0,8))
        weight = raw[:,:,4].softmax(1)
        # Initial positive log-count median matches scalar predictive mean (>=1 voxel).
        mean = (old*weight[:,:,None]).sum(1)
        var = ((old-mean[:,None]).square()*weight[:,:,None]).sum(1)
        base_mu = (mean*1000).clamp_min(1).log()
        base_scale = (torch.log1p(var/(mean.square()+1e-6)).sqrt()*math.sqrt(3)/math.pi).clamp(.1,2.)
        return hidden, base_mu[:,ORDER], base_scale[:,ORDER]

    def count_parameters(self, hidden, z, preceding, mu0, scale0, index):
        raw = self.amount[index](torch.cat((hidden,z,torch.log1p(preceding)/16),1))
        scale=.01+F.softplus(torch.log(torch.expm1(scale0[:,index]-.01))+raw[:,1])
        # Identical raw scale parameterization: match untruncated log-space
        # variance, including initialization, without a second narrowing change.
        if self.family=='gaussian':scale=scale*(math.pi/math.sqrt(3))
        return mu0[:,index]+raw[:,0], scale

    def nll(self, scalar, current_volumes, counts, capacity):
        n = counts[:,ORDER].long(); z = (n>0).float()
        if bool(((n.sum(1)>capacity) | (capacity<4)).any()):
            raise ValueError('Infeasible supervised count vector; do not clip reference')
        hidden, mu0, scale0 = self.embedding(scalar,current_volumes)
        presence_loss = hidden.new_zeros(()); amount_loss = hidden.double().new_zeros(())
        remaining = capacity.clone()
        for i in range(4):
            logit = self.presence[i](torch.cat((hidden,z[:,:i]),1))[:,0]
            presence_loss = presence_loss+F.binary_cross_entropy_with_logits(logit,z[:,i],reduction='mean')
            mu, scale = self.count_parameters(hidden,z,n[:,:i],mu0,scale0,i)
            present = z[:,i].bool()
            maximum = remaining-z[:,i+1:].sum(1).long()
            if present.any():
                # Divide by whole batch: joint count likelihood, not an oracle positive-only objective.
                amount_loss = amount_loss-count_log_mass(n[present,i],maximum[present],mu[present],scale[present],self.family).sum()/len(n)
            remaining = remaining-n[:,i]
        return presence_loss+amount_loss, {'presence_nll':float(presence_loss.detach()),'conditional_count_nll':float(amount_loss.detach())}

    @torch.no_grad()
    def marginal_presence(self, scalar, volumes):
        hidden, _, _ = self.embedding(scalar,volumes)
        if len(hidden)!=1: raise ValueError('Exact marginal enumeration expects one cutoff')
        patterns = torch.tensor(list(itertools.product((0.,1.),repeat=4)),device=hidden.device)
        h = hidden.expand(16,-1); logp = h.new_zeros(16)
        for i in range(4):
            logits = self.presence[i](torch.cat((h,patterns[:,:i]),1))[:,0]
            logp -= F.binary_cross_entropy_with_logits(logits,patterns[:,i],reduction='none')
        marginal = (logp.exp()[:,None]*patterns).sum(0)
        return marginal[list(ORDER)]  # permutation is self-inverse

    @torch.no_grad()
    def sample(self, scalar, volumes, capacity, uniforms):
        if int(capacity)<4: raise ValueError('Cutoff domain must have capacity for all four compartments')
        k=len(uniforms); hidden, mu0, scale0=self.embedding(scalar,volumes)
        hidden=hidden.expand(k,-1); mu0=mu0.expand(k,-1); scale0=scale0.expand(k,-1)
        z=hidden.new_zeros(k,4); n=torch.zeros(k,4,device=hidden.device,dtype=torch.long)
        for i in range(4):
            p=self.presence[i](torch.cat((hidden,z[:,:i]),1))[:,0].sigmoid()
            z[:,i]=(uniforms[:,i]<p).float()
        remaining=torch.full((k,),int(capacity),device=hidden.device,dtype=torch.long)
        for i in range(4):
            present=z[:,i].bool(); maximum=remaining-z[:,i+1:].sum(1).long()
            mu,scale=self.count_parameters(hidden,z,n[:,:i],mu0,scale0,i)
            if present.any():n[present,i]=sample_count(maximum[present],mu[present],scale[present],uniforms[present,4+i],self.family)
            remaining-=n[:,i]
        if bool(((n>0)!=z.bool()).any()) or bool((remaining<0).any()):raise AssertionError('Joint feasibility failed')
        return n[:,list(ORDER)]


def fixed_uniforms(key, device='cuda', count=64):
    seed=int.from_bytes(hashlib.sha256(('amount-first-v1:17:'+key).encode()).digest()[:8],'little')%(2**63-1)
    generator=torch.Generator(device='cpu').manual_seed(seed)
    # Strict interior midpoint uniforms; stable independently of global training RNG.
    u=(torch.randint(0,2**52,(count,8),generator=generator,dtype=torch.int64).double()+.5)/2**52
    return u.to(device)


def native_priors(segmentation):
    from scipy.ndimage import distance_transform_edt
    fields=[]
    for c in range(1,5):
        mask=segmentation==c
        fields.append(np.clip(distance_transform_edt(mask)-distance_transform_edt(~mask),-32,32).astype('float32')
                      if mask.any() else np.full(segmentation.shape,-32.,dtype='float32'))
    return np.stack(fields)


class LocationReadout(nn.Module):
    def __init__(self):
        super().__init__()
        # Same small spatial architecture, no additional 4x pooling of 2mm input.
        self.spatial=nn.Sequential(nn.Conv3d(14,16,3,padding=1),nn.SiLU(),nn.Conv3d(16,16,3,padding=1),nn.SiLU(),nn.Conv3d(16,4,1))
        nn.init.zeros_(self.spatial[-1].weight);nn.init.zeros_(self.spatial[-1].bias)

    def forward(self, spatial):
        return self.spatial(spatial)


def rank_domain(fields, domain):
    indices=torch.nonzero(domain.flatten(),as_tuple=False)[:,0]
    # Ascending flat index is the explicit deterministic secondary key.
    return [indices[torch.argsort(field.flatten()[indices],descending=True,stable=True)] for field in fields]


def allocate_counts(rankings, counts, shape, capacity):
    counts=counts.long()
    if bool((counts<0).any()) or int(counts.sum())>capacity:
        raise ValueError(f'Infeasible allocation: requested={counts.tolist()}, capacity={capacity}')
    result=torch.zeros(math.prod(shape),device=rankings[0].device,dtype=torch.uint8)
    for channel in ORDER:
        requested=int(counts[channel]);order=rankings[channel]
        selected=order[result[order]==0][:requested]
        if len(selected)!=requested:raise ValueError('Rankings cannot realize requested quantities')
        result[selected]=channel+1
    measured=torch.bincount(result.long(),minlength=5)[1:]
    if not torch.equal(measured,counts):raise AssertionError('Native allocation did not preserve quantities')
    return result.reshape(shape)


def location_loss(residual, current, target):
    logits=torch.cat((torch.zeros_like(residual[:,:1]),current['sdf'].float()[None]+residual),1)
    truth=target['target_occupancy'].float()[None]; valid=target['valid'].float()[None]
    logp=F.log_softmax(logits.float(),1);prob=logp.exp()
    ce=[];dice=[]
    for c in range(5):
        occupancy=truth[:,c];total=occupancy.sum()
        if total>0:ce.append(-(occupancy*logp[:,c]).sum()/total)
        if c>0 and total>0:dice.append(1-(2*(prob[:,c]*occupancy).sum()+1e-5)/((prob[:,c]*valid[:,0]).sum()+total+1e-5))
    return .5*(torch.stack(ce).mean()+torch.stack(dice).mean())
