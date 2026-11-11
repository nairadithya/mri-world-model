"""Structured prediction from declared, serializable observation context."""
import numpy as np
import torch
from torch.nn import functional as F
from .state import PatientState, digest
from .structured import AnatomicalTransition, context, weighted_quantile
from .timing import elapsed_weeks


def signed_distances(segmentation, compartments=3):
    from scipy.ndimage import distance_transform_edt
    seg=np.asarray(segmentation)
    result=[]
    for c in range(1,compartments+1):
        inside=seg==c
        result.append(np.clip(distance_transform_edt(inside)-distance_transform_edt(~inside),-32,32) if inside.any() else np.full(seg.shape,-32.))
    return torch.from_numpy(np.stack(result).astype(np.float32))


def load_transition(path, observation_id):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck['config']['kind']!='anatomy' or ck['config']['identity']['observation_sha256']!=observation_id:
        raise ValueError('Structured transition belongs to another observation model')
    net=AnatomicalTransition(ck['config']['arm'],compartments=ck['config'].get('compartments',3)).cuda().eval()
    net.load_state_dict(ck['model'],strict=True)
    net.model_id=digest(path);net.is_smoke=ck['config']['smoke']
    return net


def make_context_state(observation, row, segmentation, images, support):
    compartments=observation.metadata.get('anatomical_classes',4)-1
    sdf=signed_distances(segmentation,compartments)
    if compartments==4:support=support.cpu() | torch.from_numpy(segmentation>0)[None,None]
    pad=tuple(v for size in reversed(segmentation.shape) for v in (0,size%2))
    def reduce(x):return F.avg_pool3d(F.pad(x.float(),pad),2)
    features={'images_2mm':reduce(images.cpu()).half(),'sdf_2mm':reduce(sdf[None]).half(),
              'support_2mm':reduce(support.cpu()).half(),'images':images.cpu().float(),
              'segmentation':torch.from_numpy(segmentation)[None,None], 'support':support.cpu(), 'sdf':sdf[None]}
    row={k:row[k] for k in ('patient','visit','visit_order','dataset','time_unit','diagnosis_relative_day') if k in row}
    meta=dict(observation.metadata,backend='structured-observation-context-v1',source_row=row,
              context_format='structured-context-v1',observed_sequences_only=True)
    memory={'predicted_volumes':torch.tensor([(segmentation==c).sum()/1000 for c in range(1,compartments+1)]).float(),
            'availability':torch.tensor(observation.metadata['availability']).float()}
    return PatientState(features,meta,memory).validate()


def context_record(state,device='cuda'):
    f=state.features
    return {'images':f['images_2mm'][0].to(device),'sdf':f['sdf_2mm'][0].to(device),'support':f['support_2mm'][0].to(device),
            'predicted_volumes':state.memory['predicted_volumes'].to(device),'availability':state.memory['availability'].to(device),
            'row':state.metadata['source_row'],'voxel_ml':.008}


@torch.inference_mode()
def predict_context(model,state,previous,horizon):
    current=context_record(state)
    old=context_record(previous) if previous is not None else None
    if previous is not None:
        if previous.metadata['patient']!=state.metadata['patient'] or previous.metadata['model_id']!=state.metadata['model_id']:
            raise ValueError('Incompatible patient or observation identity')
        if previous.metadata['shape_xyz']!=state.metadata['shape_xyz'] or previous.metadata['affine']!=state.metadata['affine']:
            raise ValueError('Register observed states to common geometry')
    gap=elapsed_weeks(current['row'])-elapsed_weeks(old['row']) if old is not None else 0
    x,s=context(current,old,gap,horizon,model.arm);parameters=model.parameters_for(x,s)
    weights=parameters['weights'];representative=int(weights.argmax())
    volumes=[];chosen_seg=chosen_image=None
    f=state.features
    for i in range(model.scenarios):
        _,maps,images=model.render({**parameters,'delta':parameters['delta'][i:i+1]},
                   f['sdf'][0].float().cuda(),f['images'][0].float().cuda(),f['support'][0].cuda(),f['segmentation'][0,0].cuda())
        volumes.append(maps[0,1:].sum((1,2,3))*.001)
        if i==representative:
            chosen_seg=maps.argmax(1)[0].byte().cpu().numpy();chosen_image=images[0,3].cpu().numpy()
    values=torch.stack(volumes)
    lo,hi=weighted_quantile(values,weights,.05),weighted_quantile(values,weights,.95)
    return {'segmentation':chosen_seg,'FLAIR':chosen_image,
            'distribution':{'samples_ml':values.tolist(),'weights':weights.tolist(),
                 'mean_ml':(values*weights[:,None]).sum(0).tolist(),'lower90_ml':lo.tolist(),'upper90_ml':hi.tolist(),
                 'width90_ml':(hi-lo).tolist(),'representative_scenario':representative},
            'parameters':{k:v.detach().cpu() for k,v in parameters.items()}}
