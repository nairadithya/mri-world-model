"""A small distribution over anatomical boundary changes, not feature rewrites.

Each of eight learned scenarios is one map, its derived volumes, and an image
conditioned on that map. Scenario weights and spatial change bases are learned.
The finite mixture is a development approximation, not calibrated uncertainty.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def weighted_crps(samples, weights, target):
    first = (weights[:,None]*(samples-target).abs()).sum(0)
    second = (weights[:,None,None]*weights[None,:,None]*(samples[:,None]-samples[None,:]).abs()).sum((0,1))
    return first-.5*second


def weighted_quantile(samples, weights, q):
    """Inverse empirical CDF; integer ranks for equal-weight samples.

    Float32 prefix sums can move exact equal-weight boundaries by a rank.
    Unequal scenario weights are normalized and accumulated in float64.
    """
    if not 0 <= q <= 1 or len(weights)==0 or len(samples)!=len(weights):
        raise ValueError('Invalid quantile input')
    if not torch.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0:
        raise ValueError('Invalid distribution weights')
    values, order = samples.sort(0)
    if bool((weights==weights[0]).all()):
        rank=max(0,math.ceil(q*len(weights))-1)
        return values[rank]
    normalized=weights.double()/weights.double().sum()
    cumulative=normalized[order].cumsum(0)
    index=(cumulative<q).sum(0).clamp_max(len(weights)-1)
    return values.gather(0,index[None])[0]


def anatomical_mixture_cross_entropy(logits, weights, truth, valid, support, log_weights=None):
    """Mixture likelihood without clipping gradients for unlikely true tissue.

    The renderer forces background outside the cutoff-observed domain. Keep
    that fixed-domain behavior (including its finite impossible-target cost)
    while evaluating learnable probabilities inside the domain in log space.
    """
    if log_weights is None:
        log_weights = weights.log()
    log_probability = torch.logsumexp(
        F.log_softmax(logits, dim=1) + log_weights[:, None, None, None, None], dim=0)
    outside = torch.full_like(log_probability, math.log(1e-6))
    outside[0] = 0.
    log_probability = torch.where(support > .5, log_probability, outside)
    return -(truth * log_probability).sum() / valid.sum().clamp_min(1)


def context(current, previous, gap, horizon, arm):
    if horizon not in (12,24) or gap < 0:
        raise ValueError('Invalid forecasting interval')
    history = arm == 'history' and previous is not None and gap > 0
    images, sdf = current['images'].float(), current['sdf'].float()
    delta = (sdf-previous['sdf'].float())/max(gap,1.) if history else torch.zeros_like(sdf)
    v = current['predicted_volumes'].float()
    rate = (torch.log1p(v)-torch.log1p(previous['predicted_volumes'].float()))/max(gap,1.) if history else torch.zeros_like(v)
    spatial = torch.cat((images,sdf/32,delta.clamp(-4,4)/4,current['support'].float(),
                         torch.full_like(sdf[:1],math.log1p(gap)/5 if history else 0)),0)[None]
    scalar = torch.cat((torch.log1p(v)/5,rate,horizon*v.new_ones(1)/24,
                        v.new_tensor([math.log1p(gap)/5 if history else 0]),current['availability'].float(),
                        v.new_tensor([current['row'].get('dataset')=='MU',current['row'].get('time_unit')=='diagnosis_relative_days',history])))[None]
    if scalar.shape[1] != 2 * len(v) + 9:
        raise AssertionError('Context contract changed')
    return spatial, scalar


class VolumeChangePredictor(nn.Module):
    """Small scalar reference: correlated log-volume change distribution."""
    def __init__(self, arm='history', scenarios=8, compartments=3):
        super().__init__()
        self.arm, self.scenarios, self.compartments = arm, scenarios, compartments
        if compartments not in (3,4):
            raise ValueError('Expected original compartments with optional explicit cavity')
        self.network = nn.Sequential(nn.Linear(2*compartments+9,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU(),nn.Linear(32,scenarios*(compartments+1)))
        nn.init.zeros_(self.network[-1].weight)
        with torch.no_grad():
            bias = self.network[-1].bias.reshape(scenarios,compartments+1)
            bias.zero_()
            for c in range(compartments):
                bias[:,c] = .15*torch.sin(torch.arange(scenarios)*2.4+c)

    def forward(self, scalar, current_volumes):
        raw = self.network(scalar).reshape(self.scenarios,self.compartments+1)
        volumes = torch.expm1((torch.log1p(current_volumes)[None]+raw[:,:self.compartments]).clamp(0,8))
        return volumes, raw[:,self.compartments].softmax(0)


class AnatomicalTransition(nn.Module):
    def __init__(self, arm='history', scenarios=8, compartments=3):
        super().__init__()
        if arm not in ('history','current'):
            raise ValueError('Unknown arm')
        self.arm, self.scenarios, self.compartments = arm, scenarios, compartments
        if compartments not in (3,4):
            raise ValueError('Expected original compartments with optional explicit cavity')
        self.spatial = nn.Sequential(nn.Conv3d(6+2*compartments,16,3,padding=1),nn.SiLU(),
                                     nn.Conv3d(16,16,3,padding=1),nn.SiLU(),nn.Conv3d(16,2*compartments,1),nn.Tanh())
        self.global_head = nn.Sequential(nn.Linear(2*compartments+9,48),nn.SiLU(),nn.Linear(48,scenarios*(3*compartments+1)+4*compartments))
        nn.init.zeros_(self.global_head[-1].weight)
        nn.init.zeros_(self.global_head[-1].bias)
        with torch.no_grad():
            bias = self.global_head[-1].bias[:scenarios*(3*compartments+1)].reshape(scenarios,3*compartments+1)
            for c in range(compartments):
                bias[:,c*3] = 2*torch.sin(torch.arange(scenarios)*2.4+c)
                bias[:,c*3+1:c*3+3] = .1*torch.randn(scenarios,2)

    def parameters_for(self, spatial, scalar):
        coarse = F.avg_pool3d(spatial,4,ceil_mode=True)
        bases = self.spatial(coarse).reshape(1,self.compartments,2,*coarse.shape[2:])
        raw = self.global_head(scalar)[0]
        modes = raw[:self.scenarios*(3*self.compartments+1)].reshape(self.scenarios,3*self.compartments+1)
        coefficients = modes[:,:3*self.compartments].reshape(self.scenarios,self.compartments,3)
        delta = coefficients[:,:,0,None,None,None]+torch.einsum('kcr,crxyz->kcxyz',coefficients[:,:,1:],bases[0])
        return {'delta':delta.clamp(-48,48),'weights':modes[:,-1].softmax(0),
                'log_weights':F.log_softmax(modes[:,-1],dim=0),
                'appearance':raw[self.scenarios*(3*self.compartments+1):].reshape(4,self.compartments)}

    @staticmethod
    def render(parameters, sdf, images, support, current_seg=None, hard_gradient=False, return_logits=False):
        """The same map determines volume and image; no image-only future branch."""
        delta = F.interpolate(parameters['delta'],size=sdf.shape[1:],mode='trilinear',align_corners=False)
        logits = torch.cat((torch.zeros_like(delta[:,:1]),(sdf[None]+delta)),1)
        probabilities = logits.softmax(1)
        # Support is cutoff-observed, never taken from the future scan.
        inside = support > .5
        probabilities = probabilities*inside[None]
        probabilities[:,0] += (~inside[0]).to(probabilities)
        hard = F.one_hot(probabilities.argmax(1),sdf.shape[0]+1).movedim(-1,1).float()
        maps = hard+(probabilities-probabilities.detach()) if hard_gradient else hard
        if current_seg is None:
            current_seg = torch.cat((torch.zeros_like(sdf[:1]),sdf),0).argmax(0)
        now = F.one_hot(current_seg.long(),sdf.shape[0]+1).movedim(-1,0).float()
        # Class contrasts are derived only from current images; learned residuals
        # adapt their appearance when sampled compartments grow or shrink.
        prototypes = []
        brain = inside[0]
        background = brain & (current_seg==0)
        base = images[:,background].mean(1) if background.any() else images.new_zeros(4)
        for c in range(1,sdf.shape[0]+1):
            mask = brain & (current_seg==c)
            prototypes.append(images[:,mask].mean(1)-base if mask.any() else images.new_zeros(4))
        contrast = torch.stack(prototypes,1)+parameters['appearance']
        future_images = images[None]+torch.einsum('kcxyz,ic->kixyz',maps[:,1:]-now[None,1:],contrast)
        if return_logits:
            return probabilities, maps, future_images, logits
        return probabilities, maps, future_images


def anatomical_loss(parameters, current, target):
    probabilities, maps, images, logits = AnatomicalTransition.render(parameters,current['sdf'].float(),current['images'].float(),current['support'],hard_gradient=True,return_logits=True)
    w = parameters['weights']
    marginal = (w[:,None,None,None,None]*probabilities).sum(0)
    truth, valid = target['target_occupancy'].float(), target['valid'].float()
    ce = anatomical_mixture_cross_entropy(logits,w,truth,valid,current['support'],parameters.get('log_weights'))
    dice = (2*(marginal[1:]*truth[1:]).sum((1,2,3))+1)/( (marginal[1:]*valid).sum((1,2,3))+truth[1:].sum((1,2,3))+1)
    volumes = maps[:,1:].sum((2,3,4))*current['voxel_ml']
    crps = weighted_crps(volumes,w,target['target_volumes']).mean()
    # Proper energy score on a small full-brain occupancy grid trains joint
    # spatial scenarios in addition to marginal anatomical probabilities.
    pooled = F.avg_pool3d(maps[:,1:]*valid[None],4,ceil_mode=True).flatten(1)
    reference = F.avg_pool3d(truth[None,1:],4,ceil_mode=True).flatten(1)
    distance = torch.linalg.vector_norm(pooled-reference,dim=1)/math.sqrt(pooled.shape[1])
    pair_distance = torch.cdist(pooled,pooled)/math.sqrt(pooled.shape[1])
    energy = (w*distance).sum()-.5*(w[:,None]*w[None,:]*pair_distance).sum()
    # Image accuracy is a secondary objective on the same scenario distribution.
    mean_image = (w[:,None,None,None,None]*images).sum(0)
    error = (mean_image-target['images']).abs()
    brain = target.get('image_support',target['support']).expand_as(error)
    lesion = truth[1:].sum(0,keepdim=True).expand_as(error)
    image_loss = (error*brain).sum()/brain.sum().clamp_min(1)+ (error*lesion).sum()/lesion.sum().clamp_min(1)
    loss = ce+.5*(1-dice.mean())+.02*crps+energy+.05*image_loss
    return loss, {'ce':float(ce.detach()),'soft_dice':float(dice.mean().detach()),'volume_crps_ml':float(crps.detach()),
                  'spatial_energy':float(energy.detach()),'image_loss':float(image_loss.detach())}
