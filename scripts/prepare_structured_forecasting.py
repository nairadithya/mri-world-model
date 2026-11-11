#!/usr/bin/env python3
"""Compact decoded context for structured forecasting; no feature-pyramid export."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
from torch.nn import functional as F
from src.patient_state.data import catalogue, load_visit, metadata
from src.patient_state.observation import JointObservation
from src.patient_state.state import digest, write_json
from src.patient_state.timing import timed_pairs


def signed_distances(seg, spacing=1., compartments=3):
    fields = []
    for c in range(1,compartments+1):
        inside = seg == c
        if not inside.any():
            fields.append(np.full(seg.shape, -32., dtype=np.float32))
        else:
            fields.append(np.clip(distance_transform_edt(inside, sampling=spacing)-distance_transform_edt(~inside, sampling=spacing), -32, 32).astype(np.float32))
    return np.stack(fields)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalogue', type=Path, required=True)
    p.add_argument('--observation', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--smoke-pair',action='store_true')
    p.add_argument('--fixed-mu-pairs',type=Path,help='Preserved original pairs; no eligibility recomputation')
    args = p.parse_args()
    torch.set_num_threads(4)
    cat = catalogue(args.catalogue)
    groups = {g: timed_pairs(cat['groups'][g]) for g in ('fitting','selection')}
    if args.fixed_mu_pairs:
        original=json.loads(args.fixed_mu_pairs.read_text())
        lookup={r['stem']:r for rows in cat['groups'].values() for r in rows}
        groups={}
        for group,pairs in original.items():
            if group not in ('fitting','selection'):raise ValueError('Development groups only')
            allowed={r['stem'] for r in cat['groups'][group]}
            groups[group]=[]
            for pair in pairs:
                if pair['dataset']!='MU':raise ValueError('MU adaptation only')
                for row in pair['history']+[pair['target']]:
                    if row['stem'] not in allowed:raise ValueError('Allocation changed')
                    for key in ('patient','visit','visit_order','diagnosis_relative_day','time_unit','images','support'):
                        if lookup[row['stem']].get(key)!=row.get(key):raise ValueError('Timing/input changed: '+key)
                groups[group].append(dict(pair,history=[lookup[r['stem']] for r in pair['history']],target=lookup[pair['target']['stem']]))
        if (len(groups['fitting']),len(groups['selection']))!=(87,58):raise ValueError('Pair population changed')
    rows = {r['stem']: r for pairs in groups.values() for p in pairs
            for r in ([p['history'][-1],p['target']] if args.fixed_mu_pairs else p['history']+[p['target']])}
    obs = JointObservation.from_checkpoint(args.observation)
    args.out.mkdir(parents=True, exist_ok=True)
    compartments=obs.num_classes-1
    if compartments==4 and not args.fixed_mu_pairs:raise ValueError('Five-way forecasting requires fixed MU adaptation population')
    identity = {'catalogue_sha256': digest(args.catalogue), 'observation_sha256': obs.model_id,
                'format': 'structured-context-v1', 'spacing_mm': 2, 'scope': 'fitting and selection only'}
    if args.fixed_mu_pairs:identity.update(compartments=compartments,fixed_pairs_sha256=digest(args.fixed_mu_pairs),anatomy_readout='native_five_category',model_identity='mu-cavity-current-structured-v1',support_contract='observed brain union native predicted anatomy; zero change preserves native map',target_contract='all native five-category reference voxels, including cavity outside image support')
    if (args.out/'identity.json').exists() and json.loads((args.out/'identity.json').read_text()) != identity:
        raise ValueError('Preparation identity changed')
    write_json(args.out/'identity.json', identity)
    write_json(args.out/'population.json', groups)
    started = time.monotonic()
    ordered=sorted(rows.items())
    if args.smoke_pair:
        first=groups['fitting'][0]
        ordered=[(r['stem'],r) for r in (first['history'][-1],first['target'])]
    for n, (key, row) in enumerate(ordered):
        if args.limit is not None and n >= args.limit:
            break
        path = args.out/(key+'.pt')
        if path.exists():
            continue
        tick = time.monotonic()
        case = load_visit(row, labels=True)
        state = obs.encode(case['images'].cuda(), metadata(row))
        read = obs.read(state)
        seg = read['logits'].argmax(1)[0].cpu().numpy().astype(np.uint8)
        if compartments==3:seg[~case['support'][0,0].numpy()] = 0
        # Full-resolution predicted maps remain available for exact 1mm selection.
        sdf = signed_distances(seg,compartments=compartments)
        shape = list(seg.shape)
        pad = tuple(v for size in reversed(shape) for v in (0, size % 2))
        def reduce(x):
            return F.avg_pool3d(F.pad(x.float(), pad), 2)
        observed_domain=case['support'] | torch.from_numpy(seg>0)[None,None] if compartments==4 else case['support']
        support = reduce(observed_domain)
        labels = case['labels']
        target = torch.stack([(labels == c) for c in range(compartments+1)], 1)
        valid = (labels[:,None] >= 0) if compartments==4 else case['support'] & (labels[:,None] >= 0)
        record = {'identity': identity, 'row': row, 'shape': shape,
                  'images': reduce(case['images'])[0].half(),
                  'sdf': reduce(torch.from_numpy(sdf)[None])[0].half(),
                  'predicted_segmentation': torch.from_numpy(seg),
                  'predicted_volumes': torch.tensor([(seg==c).sum()/1000 for c in range(1,compartments+1)]).float(),
                  'support': support[0].half(), 'image_support':reduce(case['support'])[0].half(), 'valid': reduce(valid)[0].half(),
                  'target_occupancy': reduce(target & valid)[0].half(),
                  'target_volumes': torch.tensor([int(((labels==c)&valid[:,0]).sum())/1000 for c in range(1,compartments+1)]),
                  'availability': torch.ones(4), 'voxel_ml': .008}
        if compartments==4:
            record['legacy_target_volumes']=torch.tensor([int(((labels==c)&case['support'][:,0]).sum())/1000 for c in (1,2,3)])
        tmp = path.with_suffix('.tmp');torch.save(record,tmp);tmp.replace(path)
        del state, read, case, record, sdf
        print(json.dumps({'prepared':n+1,'total':len(rows),'visit':key,'seconds':time.monotonic()-tick}),flush=True)
    write_json(args.out/'preparation.json', {'eligible_visits':len(rows), 'available_contexts':sum((args.out/(k+'.pt')).exists() for k in rows),
               'elapsed_seconds':time.monotonic()-started,'limited':args.limit is not None or args.smoke_pair,
               'feature_pyramids_exported':False,'reserved_images_opened':0})


if __name__ == '__main__': main()
