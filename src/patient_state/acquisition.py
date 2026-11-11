"""Actual sequence evidence and separately named hypothetical JEPA queries."""
import copy
from itertools import product
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .observation import JointObservation,FEATURES
from .state import PatientState,SEQUENCES,digest

STAGES=(2,3,4) # zero-based hierarchy indices: 4, 8 and 16 mm
PANELS=((1,),(0,1),(1,2),(0,1,2))
SUBSETS=tuple(tuple(sorted((1,)+s)) for s in ((),(0,),(2,),(3,),(0,2),(0,3),(2,3),(0,2,3)))


class AcquisitionObservation(JointObservation):
    backend='resenc-l-acquisition-jepa-v1'

    def __init__(self):
        super().__init__(cavity=True);self.backend=type(self).backend
        self.fusion=nn.ModuleList([nn.Conv3d(4*c+4,c,1) for c in FEATURES])
        for layer in self.fusion:nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        self.predictors=nn.ModuleList([nn.Sequential(nn.Conv3d(FEATURES[i]+8,64,1),nn.GroupNorm(8,64),nn.SiLU(),nn.Conv3d(64,64,3,padding=1),nn.SiLU(),nn.Conv3d(64,FEATURES[i],1)) for i in STAGES])
        self.target_encoder=copy.deepcopy(self.encoder).eval().requires_grad_(False)
        self.use_checkpointing=True

    def initialize_pretrained(self,path):
        info=super().initialize_pretrained(path)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        info.update(observation_warm_start=False,single_sequence='named image slot, one availability flag; shared encoder')
        return info

    def train(self,mode=True):
        super().train(mode);self.target_encoder.eval();return self

    def encode_sequence(self,image,index,target=False):
        if index not in range(4) or image.ndim!=5 or image.shape[1]!=1:raise ValueError('One named actual sequence required')
        slots=[torch.zeros_like(image) for _ in range(8)];slots[index]=image;slots[index+4]=torch.ones_like(image)
        packet=torch.cat(slots,1);encoder=self.target_encoder if target else self.encoder
        if self.training and not target and self.use_checkpointing:return checkpoint(encoder,packet,use_reentrant=False)
        return encoder(packet)

    def fuse(self,evidence):
        if 1 not in evidence:raise ValueError('T1 must be observed')
        flags=next(iter(evidence.values()))[0].new_tensor([i in evidence for i in range(4)])[None]
        fused=[]
        for stage,c in enumerate(FEATURES):
            observed=[evidence[i][stage] for i in sorted(evidence)];zero=torch.zeros_like(observed[0])
            a=flags[...,None,None,None].expand(len(zero),4,*zero.shape[2:])
            packet=torch.cat([evidence[i][stage] if i in evidence else zero for i in range(4)]+[a],1)
            residual=checkpoint(self.fusion[stage],packet,use_reentrant=False) if self.training and self.use_checkpointing else self.fusion[stage](packet)
            fused.append(torch.stack(observed).mean(0)+residual)
        return fused

    def features(self,images,availability):
        if len(images)!=1 or availability.shape!=(1,4):raise ValueError('One visit per microbatch')
        evidence={i:self.encode_sequence(images[:,i:i+1],i) for i in range(4) if bool(availability[0,i])}
        return self.fuse(evidence)

    def predict_query(self,fused,availability,query):
        condition=fused[0].new_zeros(len(fused[0]),8);condition[:,:4]=availability;condition[:,4+query]=1
        result=[]
        for layer,stage in zip(self.predictors,STAGES):
            x=fused[stage];c=condition[...,None,None,None].expand(len(x),8,*x.shape[2:]);result.append(layer(torch.cat((x,c),1)))
        return result

    def decode_features(self,features):
        if self.training and self.use_checkpointing:
            return checkpoint(self.segmentation_decoder,features,use_reentrant=False),checkpoint(self.image_decoder,features,use_reentrant=False)
        return super().decode_features(features)

    @torch.no_grad()
    def update_target(self):
        for target,source in zip(self.target_encoder.parameters(),self.encoder.parameters()):target.mul_(.996).add_(source,alpha=.004)
        for target,source in zip(self.target_encoder.buffers(),self.encoder.buffers()):target.copy_(source)

    @torch.inference_mode()
    def encode(self,images,metadata):
        """Only flagged tensor slots are read; unopened slots can even contain NaNs."""
        if not metadata['availability'][1]:raise ValueError('Starting T1 required')
        shape=tuple(images.shape[2:]);padded=tuple(max(128,((s+31)//32)*32) for s in shape)
        axes=[list(range(0,s-128+1,96)) for s in padded]
        for axis,s in zip(axes,padded):
            if axis[-1]!=s-128:axis.append(s-128)
        patches=[dict(prefix=f'patch{n:03d}_',origin=list(origin),size=128) for n,origin in enumerate(product(*axes))]
        meta=dict(metadata,model_id=self.model_id,backend=self.backend,shape_xyz=list(shape),padded_shape=list(padded),patches=patches,
            anatomical_classes=5,evidence='actual per-sequence feature hierarchies plus their fused hierarchy; no raw-image readout path',
            hypothetical_status='separate query output; never observed evidence',uncertainty_status='not calibrated',geometry_condition='released upstream geometry; offline sequence reveal')
        state=PatientState({},meta)
        # Reuse exactly the actual-update path, beginning with T1.
        state.metadata['availability']=[False]*4
        for i in (1,0,2,3):
            if metadata['availability'][i]:state=self.reveal(state,i,images[:,i:i+1])
        return state

    @torch.inference_mode()
    def reveal(self,state,index,image):
        if state.metadata['model_id']!=self.model_id or state.metadata['availability'][index]:raise ValueError('Wrong model or sequence already observed')
        if index!=1 and not state.metadata['availability'][1]:raise ValueError('T1 must arrive first')
        shape=state.metadata['shape_xyz']
        if list(image.shape[2:])!=shape or image.shape[:2]!=(1,1):raise ValueError('Reveal geometry mismatch')
        memory=dict(state.memory)
        if index==1:memory['starting_t1_support']=(image!=0).cpu()
        pad=tuple(v for s,p in reversed(list(zip(shape,state.metadata['padded_shape']))) for v in (0,p-s))
        image=F.pad(image,pad);device=next(self.parameters()).device;stored=dict(state.features)
        flags=list(state.metadata['availability']);flags[index]=True
        for patch in state.metadata['patches']:
            prefix=patch['prefix'];sl=(slice(None),slice(None),*(slice(s,s+128) for s in patch['origin']))
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                actual=self.encode_sequence(image[sl].to(device),index)
                for j,f in enumerate(actual):stored[f'{prefix}observed{index}_feature{j}']=f.cpu()
                evidence={i:[stored[f'{prefix}observed{i}_feature{j}'].to(device) for j in range(6)] for i in range(4) if flags[i]}
                fused=self.fuse(evidence)
                for j,f in enumerate(fused):stored[f'{prefix}feature{j}']=f.cpu()
        return PatientState(stored,dict(state.metadata,availability=flags),memory)

    @torch.inference_mode()
    def hypothetical_patch(self,state,patch,query):
        if state.metadata['availability'][query]:raise ValueError('Acquisition query must be missing')
        device=next(self.parameters()).device
        fused=[state.features[patch['prefix']+f'feature{i}'].to(device) for i in range(6)]
        flags=torch.tensor([state.metadata['availability']],device=device)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):features=self.predict_query(fused,flags,query)
        return dict(kind='hypothetical_only',query=SEQUENCES[query],features=features)

    @classmethod
    def from_checkpoint(cls,path,device='cuda'):
        value=torch.load(path,map_location='cpu',weights_only=False)
        if value['backend']!=cls.backend:raise ValueError('Wrong acquisition model')
        model=cls();model.load_state_dict(value['model']);model.model_id=digest(path)
        return model.to(device).eval().requires_grad_(False)


def normalized_feature_loss(predicted,actual):
    terms=[]
    for p,t in zip(predicted,actual):
        p=F.layer_norm(p.float().movedim(1,-1),(p.shape[1],));t=F.layer_norm(t.detach().float().movedim(1,-1),(t.shape[1],))
        terms.append((p-t).abs().mean())
    return torch.stack(terms).mean()
