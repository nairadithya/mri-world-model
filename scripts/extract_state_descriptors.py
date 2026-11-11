#!/usr/bin/env python3
"""Extract only bounded-program cutoff and previous-visit descriptors."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.patient_state.state_descriptors import encode_feature2,regional_descriptor,DIMENSION,REGIONS
from src.patient_state.observation import JointObservation
from src.patient_state.data import load_visit
from src.patient_state.state import digest,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--observation',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--available-only',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    population=json.loads((args.data/('candidate-population.json' if (args.smoke or args.available_only) else 'population.json')).read_text())
    needed={r['stem']:r for pairs in population.values() for pair in pairs for r in pair['history'][-2:]}
    old=Path('/home/aditya/experiments/world-model-mu-cavity-forecast-20261107/contexts-v2')
    if digest(args.observation)!='d006424493442bd1bcd8fd5b15888b197aa3a95ef3f87aeed5df2f52e86ccacb':raise ValueError('Observation changed')
    receipt_path=args.out/('smoke.json' if args.smoke else ('available-extraction.json' if args.available_only else 'extraction.json'))
    previous=json.loads(receipt_path.read_text()) if receipt_path.exists() else {'records':[],'seconds':0.}
    model=JointObservation.from_checkpoint(args.observation);records=[];started=time.monotonic()
    items=sorted(needed.items())
    if args.smoke:
        row=population['fitting'][0]['history'][-1];items=[(row['stem'],row)]
    for i,(stem,row) in enumerate(items):
        output=args.out/(stem+'.npz')
        if output.exists():continue
        affine=np.asarray(row['affine'])[:3,:3]
        if not np.allclose(affine.T@affine,np.eye(3),atol=1e-6,rtol=0):
            raise ValueError('Descriptor shell and count units require released 1mm geometry')
        if not all(Path(row[k]).exists() for k in ('images','support','mask')):
            if args.available_only:continue
            if any(p['history'][-1]['stem']==stem for pairs in population.values() for p in pairs):raise ValueError('Missing current cutoff')
            continue
        tick=time.monotonic();case=load_visit(row)
        known=None
        if (old/(stem+'.pt')).exists():known=torch.load(old/(stem+'.pt'),map_location='cpu',weights_only=True)['predicted_segmentation'].numpy()
        feature,seg=encode_feature2(model,case['images'].cuda(),None if args.smoke else known)
        if args.smoke and known is not None and not np.array_equal(seg,known):raise AssertionError('Streaming observation anatomy changed')
        vector,counts=regional_descriptor(feature,seg,case['support'][0,0].numpy())
        domain=case['support'][0,0].numpy()|(seg>0)
        predicted=np.array([(seg==c).sum() for c in range(1,5)],dtype=np.int64)
        # Current reference counts are a separate diagnostic field, never an ROI.
        reference=np.load(row['mask'],mmap_mode='r')
        if not np.isin(reference,(0,1,2,3,4)).all():raise ValueError('Unknown current reference')
        reference_counts=np.array([(reference==c).sum() for c in range(1,5)],dtype=np.int64)
        kwargs={'descriptor':vector,'region_counts':counts,'estimated_counts':predicted,'capacity':np.array(int(domain.sum())),
            'reference_counts':reference_counts,'observation_sha256':np.array(model.model_id),'stem':np.array(stem)}
        # Compact masks are current outputs needed only if a deployable arm earns spatial fitting.
        if known is None:kwargs['segmentation']=seg
        temporary=output.with_suffix('.partial.npz')
        np.savez_compressed(temporary,**kwargs);temporary.replace(output)
        records.append({'stem':stem,'seconds':time.monotonic()-tick,'bytes':output.stat().st_size,'current_mask_reused':known is not None})
        print(json.dumps({'completed':i+1,'total':len(items),**records[-1]}),flush=True)
        del feature,seg,case,domain
    receipt={'observation_sha256':model.model_id,'dimension':DIMENSION,'regions':REGIONS,
        'feature':'feature2,128 channels,stride4','overlap':'native triangular windows averaged to4mm; normalized global blend before regional moments',
        'ROI':'four current predicted compartments,8mm outer shell within current brain support,brain support; native occupancy weights; six explicit empty indicators',
        'records':previous['records']+records,'seconds':previous['seconds']+time.monotonic()-started,
        'required_unique_visits':len(needed),'cached_unique_visits':sum((args.out/(s+'.npz')).exists() for s in needed),
        'feature_pyramids_exported':False,'smoke':args.smoke,'reserved_images_opened':0}
    write_json(receipt_path,receipt)


if __name__=='__main__':main()
