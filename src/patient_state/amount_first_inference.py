"""Amount-first forecasts from the existing serialized observation context."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from .amount_first import AmountFirst, LocationReadout, fixed_uniforms, rank_domain, allocate_counts
from .structured import VolumeChangePredictor, context, weighted_quantile
from .structured_inference import context_record
from .state import digest


def load_quantity(path, observation_id):
    path=Path(path);protocol=json.loads((path.parent/'protocol.json').read_text())
    if protocol['observation_context']['observation_sha256']!=observation_id:raise ValueError('Wrong observation model')
    checkpoint=torch.load(path,weights_only=True,map_location='cpu')
    if checkpoint['identity'] not in ('mu-amount-first-v1','mu-amount-first-gaussian-v1') or checkpoint['step']!=6000 or not checkpoint['terminal_candidate']:
        raise ValueError('Expected the fixed terminal quantity candidate')
    family=checkpoint.get('family','logistic')
    if family!=protocol.get('family','logistic'):raise ValueError('Checkpoint family and protocol disagree')
    model=AmountFirst(VolumeChangePredictor('current',compartments=4),family=family).cuda().eval()
    model.load_state_dict(checkpoint['model']);model.model_id=digest(path);model.observation_id=observation_id
    model.status=json.loads((path.parent/'results.json').read_text())['status']
    return model


@torch.inference_mode()
def predict_amount_state(quantity,state,horizon,location=None):
    if state.metadata['model_id']!=quantity.observation_id:raise ValueError('Observation state mismatch')
    if state.metadata.get('context_format')!='structured-context-v1':raise ValueError('Expected declared observation context')
    current=context_record(state);spatial,scalar=context(current,None,0,horizon,'current')
    domain=state.features['support'][0,0].bool().cuda();capacity=int(domain.sum())
    key='|'.join(map(str,(state.metadata['patient'],state.metadata['visit'],horizon)))
    counts=quantity.sample(scalar,current['predicted_volumes'][None],capacity,fixed_uniforms(key))
    presence=quantity.marginal_presence(scalar,current['predicted_volumes'][None])
    fields=state.features['sdf'][0].float().cuda()
    if location is not None:fields=fields+F.interpolate(location(spatial),size=domain.shape,mode='trilinear',align_corners=False)[0]
    rankings=rank_domain(fields,domain);representative=None
    for i,count in enumerate(counts):
        prediction=allocate_counts(rankings,count,domain.shape,capacity)
        if i==0:representative=prediction.cpu().numpy()
    values=counts.float()/1000;weights=torch.full((64,),1/64,device='cuda')
    low,high=weighted_quantile(values,weights,.05),weighted_quantile(values,weights,.95)
    conditional=[]
    for c in range(4):
        positive=values[:,c]>0
        if not positive.any():conditional.append({'defined':False,'positive_samples':0});continue
        v=values[positive,c:c+1];w=torch.ones(len(v),device='cuda')/len(v)
        conditional.append({'defined':True,'positive_samples':len(v),'mean_ml':float(v.mean()),
            'lower90_ml':float(weighted_quantile(v,w,.05)[0]),'upper90_ml':float(weighted_quantile(v,w,.95)[0])})
    return {'segmentation':representative,'sample_counts':counts.cpu(),
        'presence_probability':presence.tolist(),'native_maps_verified':64,'capacity_voxels':capacity,
        'distribution':{'samples_ml':values.tolist(),'weights':weights.tolist(),'mean_ml':values.mean(0).tolist(),
            'lower90_ml':low.tolist(),'upper90_ml':high.tolist(),'width90_ml':(high-low).tolist(),
            'representative_scenario':0,'conditional_positive':conditional}}
