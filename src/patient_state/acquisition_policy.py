"""Small policy descriptors and matched ridge selectors; no outcomes at inference."""
import itertools
import numpy as np
import torch
from torch.nn import functional as F
from .state import PatientState
from .acquisition import STAGES
from .state_descriptors import windows

ORDERS=tuple(itertools.permutations((0,3,2))) # lexical sequence-name tie order: CT1, FLAIR, T2


def key(subset):return ''.join(map(str,sorted(subset)))


@torch.inference_mode()
def subset_state(model,full,subset):
    device=next(model.parameters()).device;stored={}
    for patch in full.metadata['patches']:
        prefix=patch['prefix'];evidence={}
        for i in sorted(subset):
            evidence[i]=[]
            for j in range(6):
                name=f'{prefix}observed{i}_feature{j}';stored[name]=full.features[name];evidence[i].append(stored[name].to(device))
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):fused=model.fuse(evidence)
        for j,f in enumerate(fused):stored[prefix+f'feature{j}']=f.cpu()
    return PatientState(stored,dict(full.metadata,availability=[i in subset for i in range(4)]),dict(full.memory))


@torch.inference_mode()
def merged_features(model,state,query=None):
    stages=(2,) if query is None else STAGES;shape=state.metadata['padded_shape'];native,_=windows()
    sums={i:torch.zeros((1,(128,256,320)[STAGES.index(i)],*(n//(2**i) for n in shape))) for i in stages}
    weights={i:torch.zeros((1,1,*(n//(2**i) for n in shape))) for i in stages}
    for patch in state.metadata['patches']:
        predicted=model.hypothetical_patch(state,patch,query)['features'] if query is not None else None
        for i in stages:
            f=(predicted[STAGES.index(i)] if predicted is not None else state.features[patch['prefix']+f'feature{i}']).float().cpu()
            scale=2**i;window=F.avg_pool3d(native,scale);sl=(slice(None),slice(None),*(slice(o//scale,o//scale+128//scale) for o in patch['origin']))
            sums[i][sl]+=f*window;weights[i][sl]+=window
    return {i:(sums[i]/weights[i])[0] for i in stages}


def moments(features,masks):
    values=[]
    for stage,f in sorted(features.items()):
        scale=2**stage;shape=masks.shape[1:];pad=tuple(v for s,p in zip(reversed(shape),reversed(f.shape[1:])) for v in (0,p*scale-s))
        weights=F.avg_pool3d(F.pad(masks[None].float(),pad),scale)[0]
        for w in weights:
            if not w.any():values.extend([0.]*(2*len(f)));continue
            mean=(f*w).sum((1,2,3))/w.sum();var=((f-mean[:,None,None,None])**2*w).sum((1,2,3))/w.sum()
            values.extend(mean.tolist()+var.sqrt().tolist())
    return np.array(values,dtype=np.float32)


@torch.inference_mode()
def descriptors(model,state,readout=None):
    output=model.read(state) if readout is None else readout
    probabilities=output['logits'].float().softmax(1)[0];segmentation=probabilities.argmax(0)
    masks=torch.stack([segmentation==c for c in range(1,5)]+[state.memory['starting_t1_support'][0,0].bool()])
    counts=masks.flatten(1).sum(1);entropy=-(probabilities*probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()).sum(0)
    # Entropy is a descriptor, not a calibrated uncertainty claim.
    regional_entropy=[float(entropy[m].mean()) if m.any() else 0. for m in masks]
    explicit=np.concatenate((np.log1p(counts[:4].numpy()/1000),regional_entropy,(counts==0).numpy().astype(float)))
    current=moments(merged_features(model,state),masks)
    query={str(i):moments(merged_features(model,state,i),masks) for i in (0,3,2) if not state.metadata['availability'][i]}
    return dict(current=current,explicit=explicit.astype(np.float32),queries=query)


def patient_weights(rows):
    from collections import Counter
    counts=Counter(r['patient'] for r in rows)
    return np.array([1/(len(counts)*counts[r['patient']]) for r in rows])


def pca_fit(x,w,dimension):
    # Center with patient-uniform weight; one common RMS, no channel whitening.
    mean=w@x;centered=x-mean
    _,s,vt=np.linalg.svd(np.sqrt(w[:,None])*centered,full_matrices=False)
    dimension=min(dimension,len(s));basis=vt[:dimension].copy()
    for row in basis:
        if row[np.argmax(abs(row))]<0:row*=-1
    scores=centered@basis.T;scale=float(np.sqrt(np.sum(w[:,None]*scores**2)/dimension))
    if not np.isfinite(scale) or scale<=0:raise ValueError('Degenerate policy PCA')
    return dict(mean=mean.tolist(),basis=basis.tolist(),scale=scale,whiten=False)


def pca_apply(x,recipe):return (np.asarray(x)-recipe['mean'])@np.array(recipe['basis']).T/recipe['scale']


def fit_preprocessing(visits):
    states=[dict(patient=v['patient'],state=s) for v in visits for s in v['subsets'].values()]
    queries=[dict(patient=v['patient'],value=q) for v in visits for s in v['subsets'].values() for q in s['descriptor']['queries'].values()]
    w=patient_weights(states);explicit=np.array([r['state']['descriptor']['explicit'] for r in states]);mean=w@explicit;sd=np.sqrt(w@((explicit-mean)**2));sd=np.where(sd>1e-6,sd,1.)
    return dict(current=pca_fit(np.array([r['state']['descriptor']['current'] for r in states]),w,16),
        query=pca_fit(np.array([r['value'] for r in queries]),patient_weights(queries),8),explicit_mean=mean.tolist(),explicit_sd=sd.tolist(),
        fitting_patients=sorted({v['patient'] for v in visits}),scope='P only, patient-uniform subsets/actions; no PCA whitening')


def policy_input(state,query,recipe,jepa):
    d=state['descriptor'];a=np.zeros(4);a[query]=1
    q=pca_apply(d['queries'][str(query)],recipe['query']) if jepa else np.zeros(len(recipe['query']['basis']))
    return np.concatenate((pca_apply(d['current'],recipe['current']),q,(np.array(d['explicit'])-recipe['explicit_mean'])/recipe['explicit_sd'],state['availability'],a,[1.]))


def fit_ridge(visits,recipe,penalty,jepa):
    rows=[];design=[];target=[]
    for v in visits:
        for sk,s in v['subsets'].items():
            for query in (0,3,2):
                if s['availability'][query]:continue
                next_key=key([int(c) for c in sk]+[query])
                design.append(policy_input(s,query,recipe,jepa));target.append(v['subsets'][next_key]['score']['D']-s['score']['D']);rows.append(v)
    x=np.array(design);y=np.array(target);w=patient_weights(rows)
    penalty_matrix=np.eye(x.shape[1])*penalty;penalty_matrix[-1,-1]=0
    beta=np.linalg.solve(x.T@(w[:,None]*x)+penalty_matrix,x.T@(w*y))
    return dict(coefficients=beta.tolist(),jepa=jepa,penalty=penalty)


def action_scores(state,recipe,selector):
    # State carries only the current descriptor/flags; no utility or next state.
    return {str(q):float(policy_input(state,q,recipe,selector['jepa'])@np.array(selector['coefficients'])) for q in (0,3,2) if not state['availability'][q]}


def choose(state,recipe,selector):
    scores=action_scores(state,recipe,selector)
    return max((q for q in (0,3,2) if str(q) in scores),key=lambda q:scores[str(q)])


def trajectory(visit,recipe=None,selector=None,order=None):
    subset=[1];result=[];actions=[]
    for step in range(2):
        state=visit['subsets'][key(subset)]
        q=order[step] if order is not None else choose(state,recipe,selector)
        subset.append(q);actions.append(q);result.append(visit['subsets'][key(subset)]['score']['D'])
    return dict(primary=float(np.mean(result)),budgets=result,actions=actions)
