"""Capacity-controlled native-count forecasts with fold-local transforms."""
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .amount_first import AmountFirst,ORDER,fixed_uniforms
from .gaussian_count import gaussian_log_cdf_difference
from .core_cdf import core_cdf_scores
from .structured import weighted_quantile


def gaussian_log_interval(midpoint,width):
    """Normal interval mass, including narrow bins to float64 precision.

    The symmetric integral series through h^6 has error below roundoff when
    width*(1+abs(midpoint)) < .01. It avoids subtracting nearby log-CDF values
    and avoids cancellation in width when a location parameter changes.
    """
    small=width*(1+midpoint.abs())<.01
    result=torch.empty_like(midpoint)
    m=midpoint[small];h=width[small]/2;m2=m.square();h2=h.square()
    correction=(m2-1)*h2/6+(m2.square()-6*m2+3)*h2.square()/120+(m2.pow(3)-15*m2.square()+45*m2-15)*h2.pow(3)/5040
    result[small]=width[small].log()-.5*m2-.5*np.log(2*np.pi)+torch.log1p(correction)
    m=midpoint[~small];h=width[~small]/2
    result[~small]=gaussian_log_cdf_difference(m-h,m+h)
    return result


def count_log_mass(n,maximum,mu,scale,family='gaussian'):
    """Same exact discrete/truncated law, with stable narrow-bin arithmetic."""
    if family!='gaussian':raise ValueError('Linear study supports Gaussian counts only')
    n,maximum,mu,scale=[v.double() for v in (n,maximum,mu,scale)]
    if ((n<1)|(n>maximum)|(scale<=0)).any():raise ValueError('Invalid count support/scale')
    width=torch.log1p(1/(n-.5));midpoint=(torch.log(n-.5)+width/2-mu)/scale
    denominator_width=torch.log1p(2*maximum)
    denominator_midpoint=(np.log(.5)+denominator_width/2-mu)/scale
    return gaussian_log_interval(midpoint,width/scale)-gaussian_log_interval(denominator_midpoint,denominator_width/scale)


def patient_horizon_weights(rows):
    counts=Counter((r['horizon'],r['patient']) for r in rows)
    horizons=sorted({h for h,p in counts})
    patients={h:len({p for hh,p in counts if hh==h}) for h in horizons}
    return np.array([1/(len(horizons)*patients[r['horizon']]*counts[r['horizon'],r['patient']]) for r in rows])


def fit_transform(rows,mri):
    unique={r['stem']:r for r in rows};items=[unique[s] for s in sorted(unique)]
    counts=Counter(r['patient'] for r in items)
    w=np.array([1/(len(counts)*counts[r['patient']]) for r in items])
    v=np.log1p(np.stack([r['current_counts'] for r in items]).astype(float))
    mean=w@v;sd=np.sqrt(w@((v-mean)**2));sd=np.where(sd>1e-12,sd,1.)
    result=dict(mri=bool(mri),scalar_mean=mean.tolist(),scalar_sd=sd.tolist(),
        fitting_patients=sorted(counts),fitting_stems=sorted(unique),pca_dimension=8 if mri else 0)
    if mri:
        raw=np.stack([r['descriptor'] for r in items]).reshape(-1,6,257).astype(float)
        centered=np.zeros((len(items),6,256));region_mean=np.zeros((6,256));support=[]
        for i in range(6):
            valid=raw[:,i,256]==0;support.append(int(valid.sum()))
            if valid.any():region_mean[i]=(w[valid,None]*raw[valid,i,:256]).sum(0)/w[valid].sum()
            centered[valid,i]=raw[valid,i,:256]-region_mean[i]
        x=centered.reshape(len(items),1536)
        # Conditional region centering plus mean imputation makes this weighted
        # overall center zero up to roundoff. Preserve the actual center too.
        center=w@x;x=x-center
        from scipy.linalg import svd
        _,singular,vt=svd(np.sqrt(w[:,None])*x,full_matrices=False,lapack_driver='gesvd')
        if len(singular)<8 or singular[7]<=1e-12:raise ValueError('Eight PCA components unsupported')
        basis=vt[:8].copy()
        for row in basis:
            if row[np.argmax(np.abs(row))]<0:row*=-1
        scores=x@basis.T;scale=float(np.sqrt(np.sum(w[:,None]*scores**2)/8))
        if not np.isfinite(scale) or scale<=0:raise ValueError('Degenerate common PCA RMS')
        result.update(region_mean=region_mean.tolist(),region_nonempty_fitting_visits=support,
            pca_center=center.tolist(),pca_components=basis.tolist(),pca_common_rms=scale,
            singular_values=singular[:8].tolist(),whiten=False)
    return result


def transform(rows,recipe,device='cpu'):
    counts=np.stack([r['current_counts'] for r in rows]).astype(float)
    raw=np.stack([r['descriptor'] for r in rows]).reshape(-1,6,257).astype(float)
    flags=raw[:,:,256]
    x=[(np.log1p(counts)-recipe['scalar_mean'])/recipe['scalar_sd'],
       np.array([r['horizon']==24 for r in rows],dtype=float)[:,None],flags]
    if recipe['mri']:
        centered=raw[:,:,:256]-np.array(recipe['region_mean'])[None]
        centered[flags>0]=0
        scores=(centered.reshape(-1,1536)-recipe['pca_center'])@np.array(recipe['pca_components']).T
        x.append(scores/recipe['pca_common_rms'])
    return torch.tensor(np.concatenate(x,axis=1),dtype=torch.double,device=device)


class LinearQuantity(nn.Module):
    family='gaussian'
    identity='mu-linear-count-v1'
    def __init__(self,mri=False):
        super().__init__();self.mri=bool(mri);d=19 if mri else 11
        self.presence=nn.ModuleList([nn.Linear(d+i,1,dtype=torch.double) for i in range(4)])
        self.amount=nn.ModuleList([nn.Linear(d+4+i,1,dtype=torch.double) for i in range(4)])
        self.register_buffer('sigmas',torch.ones(4,dtype=torch.double)) # autoregressive order
        for p in self.parameters():nn.init.zeros_(p)

    def embedding(self,x,current_counts):
        x=x.double();mu=current_counts.double().clamp_min(1).log()[:,ORDER]
        return x,mu,self.sigmas[None].expand(len(x),-1)

    def count_parameters(self,hidden,z,preceding,mu0,scale0,index):
        correction=self.amount[index](torch.cat((hidden,z.double(),torch.log1p(preceding.double())/16),1))[:,0]
        return mu0[:,index]+correction,scale0[:,index]

    def terms(self,x,current_counts,truth,capacity):
        hidden,mu0,sigma0=self.embedding(x,current_counts);n=truth[:,ORDER].long();z=(n>0).double()
        remaining=capacity.clone();ps=[];qs=[];locations=[];maxima=[]
        for i in range(4):
            logit=self.presence[i](torch.cat((hidden,z[:,:i]),1))[:,0]
            ps.append(F.binary_cross_entropy_with_logits(logit,z[:,i],reduction='none'))
            mu,sigma=self.count_parameters(hidden,z,n[:,:i],mu0,sigma0,i)
            maximum=remaining-z[:,i+1:].sum(1).long();positive=n[:,i]>0
            loss=torch.zeros(len(x),dtype=torch.double,device=x.device)
            loss[positive]=-count_log_mass(n[positive,i],maximum[positive],mu[positive],sigma[positive],'gaussian')
            qs.append(loss);locations.append(mu);maxima.append(maximum);remaining-=n[:,i]
        return dict(presence=torch.stack(ps,1),amount=torch.stack(qs,1),mu=torch.stack(locations,1),maximum=torch.stack(maxima,1),counts=n)

    sample=AmountFirst.sample
    marginal_presence=AmountFirst.marginal_presence

    @torch.no_grad()
    def presence_log_marginals(self,x,current_counts):
        import itertools
        h,_,_=self.embedding(x,current_counts)
        z=h.new_tensor(list(itertools.product((0.,1.),repeat=4)))
        lp=h.new_zeros(16)
        for i in range(4):
            logits=self.presence[i](torch.cat((h.expand(16,-1),z[:,:i]),1))[:,0]
            lp+=torch.where(z[:,i].bool(),F.logsigmoid(logits),F.logsigmoid(-logits))
        return torch.stack([torch.stack((torch.logsumexp(lp[z[:,i]==0],0),torch.logsumexp(lp[z[:,i]==1],0))) for i in ORDER])


def save_bundle(path,model,recipe,metadata):
    torch.save(dict(identity=model.identity,model=model.state_dict(),transform=recipe,metadata=metadata),path)


def load_bundle(path,device='cpu'):
    value=torch.load(path,map_location=device,weights_only=True)
    if value['identity']!=LinearQuantity.identity:raise ValueError('Wrong forecast identity')
    model=LinearQuantity(value['transform']['mri']).to(device).eval();model.load_state_dict(value['model'])
    return model,value['transform'],value['metadata']


@torch.no_grad()
def predict(model,recipe,current_counts,descriptor,capacity,horizon,key):
    """Cutoff-only quantity inference; coherent samples can feed existing allocation."""
    if horizon not in (12,24):raise ValueError('Unsupported horizon')
    device=next(model.parameters()).device
    x=transform([dict(current_counts=current_counts,descriptor=descriptor,horizon=horizon)],recipe,device)
    current=torch.as_tensor(current_counts,dtype=torch.double,device=device)[None]
    samples=model.sample(x,current,int(capacity),fixed_uniforms(key,device=str(device)))
    values=samples.double()/1000;w=torch.full((64,),1/64,dtype=torch.double,device=device)
    # The scoring helper's truth argument is zero only to obtain its distribution
    # mean; no score against that artificial reference is returned to inference.
    mean=core_cdf_scores(model,x,current,int(capacity),0)['mean_ml']
    return dict(sample_counts=samples.cpu().tolist(),presence_probability=model.presence_log_marginals(x,current)[:,1].exp().cpu().tolist(),
        representative_sample=0,representative_volumes_ml=values[0].cpu().tolist(),exact_core_mean_ml=mean,
        sample_mean_volumes_ml=values.mean(0).cpu().tolist(),lower90_ml=weighted_quantile(values,w,.05).cpu().tolist(),
        upper90_ml=weighted_quantile(values,w,.95).cpu().tolist(),
        numerical_status='64 fixed coherent samples for non-core quantities/intervals; core mean exact discrete CDF; no calibration guarantee')
