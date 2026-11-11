"""Cutoff-only image access and T1-only support for sequence-reveal episodes."""
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
from .data import metadata


def observed_images(row,sequences):
    raw=np.load(row['images'],mmap_mode='r',allow_pickle=False)
    expected=row.get('shape_xyz',[182,218,182])
    if raw.shape!=(4,expected[2],expected[0],expected[1]):raise ValueError('Invalid released geometry')
    images=torch.zeros((1,4,*expected),dtype=torch.float32)
    for i in sequences:
        x=np.array(raw[i].transpose(1,2,0),dtype=np.float32,order='C')
        if not np.isfinite(x).all():raise ValueError('Nonfinite observed sequence')
        images[0,i]=torch.from_numpy(x)
    return images


def t1_support(row):
    return observed_images(row,(1,))[:,1:2]!=0


def labels(row):
    raw=np.load(row['mask'],mmap_mode='r')
    y=np.array(raw.transpose(1,2,0),dtype=np.int64,order='C')
    classes=row.get('supervised_classes',4)
    if not np.isin(y,range(classes)).all():raise ValueError('Unknown native label')
    return torch.from_numpy(y)[None]


def state_metadata(row,sequences):
    m=metadata(row);m.update(dataset=row.get('dataset','LUMIERE'),availability=[i in sequences for i in range(4)],
        normalization='existing independent per-sequence nonzero z-score',local_support='starting T1 nonzero only; old CT1 support never read')
    return m


class AcquisitionPatches(Dataset):
    def __init__(self,rows,length,offset=0,selection=False):
        self.rows=rows;self.length=length;self.offset=offset;self.selection=selection
        groups=defaultdict(list)
        for row in rows:groups[row['patient']].append(row)
        self.groups=[sorted(groups[p],key=lambda r:r['visit_order']) for p in sorted(groups)]

    def __len__(self):return self.length

    def __getitem__(self,index):
        index+=self.offset;rng=np.random.default_rng((7100 if self.selection else 17)+index)
        group=self.groups[index//2%len(self.groups)] if self.selection else self.groups[int(rng.integers(len(self.groups)))]
        row=group[0] if self.selection else group[int(rng.integers(len(group)))]
        raw=np.load(row['images'],mmap_mode='r');y=np.load(row['mask'],mmap_mode='r')
        # Validation locations use only starting T1. Training may use labels.
        support=raw[1]!=0
        lesion=not self.selection and rng.random()<.5
        candidate=y[::4,::4,::4]>0 if lesion else support[::4,::4,::4]
        coords=np.argwhere(candidate)
        if not len(coords):coords=np.argwhere(support[::4,::4,::4])
        if not len(coords):raise ValueError('Empty starting T1 support')
        center=coords[int(rng.integers(len(coords)))]*4
        start=[int(np.clip(c-64,0,s-128)) for c,s in zip(center,y.shape)]
        sl=tuple(slice(s,s+128) for s in start)
        x=np.array(raw[(slice(None),*sl)].transpose(0,2,3,1),dtype=np.float32,order='C')
        target=np.array(y[sl].transpose(1,2,0),dtype=np.int64,order='C');b=np.array(support[sl].transpose(1,2,0),dtype=bool,order='C')
        if not np.isfinite(x).all() or not np.isin(target,range(row.get('supervised_classes',4))).all():raise ValueError('Invalid supervised patch')
        return torch.from_numpy(x),torch.from_numpy(target),torch.from_numpy(b)[None],row['patient'],row.get('supervised_classes',4)
