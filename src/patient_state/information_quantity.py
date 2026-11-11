"""Matched scalar/MRI/history information gates on the frozen-state quantity head.

Packed cutoff input: original 17 scalars, current descriptor, history availability
and elapsed gap, previous volumes/rates, previous descriptor. No future fields.
"""
import torch
from torch import nn
from .amount_first import AmountFirst
from .state_descriptors import DIMENSION


class InformationQuantity(AmountFirst):
    def __init__(self, control, prevalence=None, *, current_mri=False, memory=0):
        super().__init__(control, prevalence, family='gaussian')
        if memory not in (0,1,2,3):raise ValueError('Unknown memory condition')
        self.identity='mu-information-quantity-v1'
        self.current_mri=bool(current_mri)
        # 0=no memory experiment; 1=metadata; 2=metadata+volumes; 3=+MRI.
        self.memory=memory
        self.mri_projection=nn.Sequential(nn.Linear(DIMENSION,32),nn.SiLU())
        self.history_metadata=nn.Sequential(nn.Linear(2,32),nn.SiLU())
        self.history_volumes=nn.Sequential(nn.Linear(8,32),nn.SiLU())
        self.previous_mri_projection=nn.Sequential(nn.Linear(DIMENSION,32),nn.SiLU())

    def embedding(self, cutoff, volumes):
        if cutoff.shape[1]!=17+2*DIMENSION+10:raise ValueError('Cutoff input contract changed')
        hidden,mu,scale=super().embedding(cutoff[:,:17],volumes)
        current=cutoff[:,17:17+DIMENSION]
        metadata=cutoff[:,17+DIMENSION:19+DIMENSION]
        previous=cutoff[:,19+DIMENSION:27+DIMENSION]
        previous_mri=cutoff[:,27+DIMENSION:]
        # Gate each information source immediately before fusion. All arms have
        # identical parameter construction and scalar warm start for each seed.
        hidden=hidden+float(self.current_mri)*self.mri_projection(current)
        hidden=hidden+float(self.memory>0)*self.history_metadata(metadata)
        available=metadata[:,:1]
        hidden=hidden+float(self.memory>=2)*available*self.history_volumes(previous)
        hidden=hidden+float(self.memory>=3)*available*self.previous_mri_projection(previous_mri)
        return hidden,mu,scale
