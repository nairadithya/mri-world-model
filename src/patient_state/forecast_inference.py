"""Streaming complete-state forecasts; no cohort feature caches or future inputs."""
import torch
from torch.nn import functional as F
from .data import load_visit, metadata
from .state import PatientState, check_update, digest
from .observation import FEATURES
from .temporal import SpatialPredictor


def load_predictor(path, observation_id, device='cuda'):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck['observation_sha256']!=observation_id:
        raise ValueError('Forecast checkpoint belongs to another observation model')
    net=SpatialPredictor(ck['arm']).to(device).eval()
    net.load_state_dict(ck['model'],strict=True)
    net.model_id=digest(path)
    net.is_smoke=bool(ck.get('config',{}).get('smoke',False))
    return net


@torch.inference_mode()
def observed_history(observation, rows, history_model, cutoff_gain=1.):
    previous=None
    for index,row in enumerate(rows[-4:]):
        case=load_visit(row)
        gain=cutoff_gain if index==len(rows[-4:])-1 else 1.
        state=observation.encode(case['images'].cuda()*gain,metadata(row))
        state.features['brain_support']=case['support']
        if previous is not None:check_update(previous,state)
        state=history_model.assimilate_state(state,previous)
        state.metadata['temporal_model_id']=history_model.model_id
        state.metadata['history_limit']=4
        state.memory={k:v.cpu() for k,v in state.memory.items()}
        previous=state
    return state


@torch.inference_mode()
def forecast_distribution(model,state,horizon):
    if model.arm=='history' and state.metadata.get('temporal_model_id')!=model.model_id:
        raise ValueError('Recurrent state belongs to another temporal model')
    means,logvars={},{}
    for patch in state.metadata['patches']:
        prefix=patch['prefix']
        fs=[state.features[prefix+f'feature{i}'].cuda() for i in range(6)]
        if model.arm=='current':
            memory=model.update(fs,availability=torch.tensor([state.metadata['availability']],device='cuda'))
        else:memory=state.memory[prefix+'recurrent'].cuda()
        mm,ll=model.predict(fs,memory,horizon)
        for i,(m,l) in enumerate(zip(mm,ll)):
            key=prefix+f'feature{i}'
            means[key]=m.to(dtype=torch.bfloat16,device='cpu')
            logvars[key]=l.to(dtype=torch.bfloat16,device='cpu')
    means['brain_support']=state.features['brain_support']
    meta=dict(state.metadata,forecast_method=model.arm,horizon_nominal_weeks=horizon,
              forecast_week=state.metadata['nominal_week']+horizon,
              uncertainty_status='learned Gaussian marginals, 32mm block-correlated samples; development calibration measured separately')
    # Marginal uncertainty is part of the declared predicted state, not a hidden
    # decoder side channel. The second return is a view for scoring convenience.
    return PatientState(means,meta,{'logvar:'+k:v for k,v in logvars.items()}),logvars


@torch.inference_mode()
def sample_state(mean_state,logvars,seed):
    generator=torch.Generator().manual_seed(seed)
    shape=mean_state.metadata['padded_shape']
    fields=[torch.randn((1,c,*(s//32 for s in shape)),generator=generator) for c in FEATURES]
    features={}
    for patch in mean_state.metadata['patches']:
        origin=patch['origin'];prefix=patch['prefix']
        for i in range(6):
            key=prefix+f'feature{i}';mean=mean_state.features[key]
            slices=(slice(None),slice(None),*(slice(s//32,s//32+4) for s in origin))
            noise=F.interpolate(fields[i][slices],size=mean.shape[2:],mode='nearest')
            features[key]=(mean.float()+noise*(.5*logvars[key].float()).exp()).to(torch.bfloat16)
    features['brain_support']=mean_state.features['brain_support']
    return PatientState(features,dict(mean_state.metadata))
