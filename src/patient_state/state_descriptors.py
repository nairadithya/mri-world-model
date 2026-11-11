"""Frozen feature2 regional moments with normalized overlap and native ROIs."""
from itertools import product
import numpy as np
import torch
from torch.nn import functional as F

REGIONS=('enhancing','core','flair_abnormality','cavity','outer_shell_8mm','brain')
DIMENSION=6*(128*2+1)


def patch_geometry(shape):
    padded=tuple((s+31)//32*32 for s in shape)
    axes=[list(range(0,s-128+1,96)) if s>128 else [0] for s in padded]
    for a,s in zip(axes,padded):
        if s>128 and a[-1]!=s-128:a.append(s-128)
    return padded,list(product(*axes))


def windows():
    axis=(1-torch.linspace(-1,1,128).abs()).clamp_min(.05)
    native=(axis[:,None,None]*axis[None,:,None]*axis[None,None,:])[None,None]
    return native,F.avg_pool3d(native,4)


@torch.inference_mode()
def merge_saved_feature2(state):
    shape=state.metadata.get('padded_shape',[((s+31)//32)*32 for s in state.metadata['shape_xyz']])
    if 'patches' not in state.metadata:return state.features['feature2'][0].float().cpu()
    total=torch.zeros((1,128,*(s//4 for s in shape)),dtype=torch.float32)
    weight=torch.zeros((1,1,*(s//4 for s in shape)),dtype=torch.float32);_,window=windows()
    for patch in state.metadata['patches']:
        origin=patch['origin'];f=state.features[patch['prefix']+'feature2'].float().cpu()
        if any(o%4 for o in origin):raise ValueError('Feature2 origin not aligned')
        sl=(slice(None),slice(None),*(slice(o//4,o//4+f.shape[2+i]) for i,o in enumerate(origin)))
        total[sl]+=f*window;weight[sl]+=window
    if (weight<=0).any():raise ValueError('Uncovered feature2 cells')
    return (total/weight)[0]


@torch.inference_mode()
def encode_feature2(model,images,known_segmentation=None):
    """Stream patches; no feature pyramid export. Decode anatomy only if needed."""
    shape=tuple(images.shape[2:]);padded,origins=patch_geometry(shape)
    if min(padded)<128:raise ValueError('This bounded MU extractor expects >=128 padded dimensions')
    padding=tuple(v for n in reversed(shape) for v in (0,(-n)%32))
    x=F.pad(images,padding);availability=torch.ones((1,4),device=x.device)
    feature_sum=torch.zeros((1,128,*(n//4 for n in padded)),dtype=torch.float32)
    feature_weight=torch.zeros((1,1,*(n//4 for n in padded)),dtype=torch.float32)
    native_window,window=windows()
    logit_sum=torch.zeros((1,5,*padded),dtype=torch.float32) if known_segmentation is None else None
    native_weight=torch.zeros((1,1,*padded),dtype=torch.float32) if known_segmentation is None else None
    for origin in origins:
        sl=(slice(None),slice(None),*(slice(o,o+128) for o in origin))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            features=model.features(x[sl],availability)
            logits=model.segmentation_decoder(features) if known_segmentation is None else None
        f=features[2].float().cpu()
        fs=(slice(None),slice(None),*(slice(o//4,o//4+32) for o in origin))
        feature_sum[fs]+=f*window;feature_weight[fs]+=window
        if logits is not None:
            logit_sum[sl]+=logits.float().cpu()*native_window;native_weight[sl]+=native_window
        del features,logits
    if (feature_weight<=0).any():raise ValueError('Uncovered feature2 cells')
    merged=(feature_sum/feature_weight)[0]
    if known_segmentation is None:
        crop=(slice(None),slice(None),*(slice(0,n) for n in shape))
        seg=(logit_sum/native_weight)[crop].argmax(1)[0].numpy().astype(np.uint8)
    else:seg=np.asarray(known_segmentation,dtype=np.uint8)
    return merged,seg


@torch.inference_mode()
def regional_descriptor(feature2,segmentation,brain_support):
    from scipy.ndimage import distance_transform_edt
    shape=segmentation.shape;union=segmentation>0
    shell=(distance_transform_edt(~union)<=8)&~union&brain_support if union.any() else np.zeros(shape,dtype=bool)
    masks=[segmentation==c for c in range(1,5)]+[shell,brain_support]
    pad=tuple(v for s in reversed(shape) for v in (0,(-s)%4))
    # Native occupancy fractions count each actual voxel once, including edge
    # cells truncated by padding. Feature overlap was normalized before moments.
    weights=F.avg_pool3d(F.pad(torch.from_numpy(np.stack(masks)).float()[None],pad),4)[0]
    if tuple(weights.shape[1:])!=tuple(feature2.shape[1:]):
        weights=F.pad(weights,tuple(v for a,b in zip(reversed(weights.shape[1:]),reversed(feature2.shape[1:])) for v in (0,b-a)))
    feature2=feature2.float();descriptor=[];counts=[]
    for mask,w in zip(masks,weights):
        n=int(mask.sum());counts.append(n)
        if not n:descriptor.append(torch.cat((torch.zeros(256),torch.ones(1))));continue
        total=w.sum()
        if abs(float(total)*64-n)>1e-3:raise AssertionError('Padded ROI counts changed')
        mean=(feature2*w).sum((1,2,3))/total
        variance=((feature2-mean[:,None,None,None]).square()*w).sum((1,2,3))/total
        descriptor.append(torch.cat((mean,variance.sqrt(),torch.zeros(1))))
    vector=torch.cat(descriptor).numpy()
    if vector.shape!=(DIMENSION,) or not np.isfinite(vector).all():raise ValueError('Invalid descriptor')
    return vector,np.array(counts,dtype=np.int64)
