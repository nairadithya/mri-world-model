#!/usr/bin/env python3
"""Bounded MU fitting recovery, preserving original pair and selection files."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.patient_state.state import digest,write_json
from src.patient_state.data import catalogue

BASE=Path('/home/aditya/experiments')
OBS=BASE/'world-model-patient-state-20261105'
OLD=BASE/'world-model-mu-cavity-forecast-20261107/contexts-v2'


def pair_key(pair):return (pair['patient'],pair['history'][-1]['visit'],pair['horizon'])


def expanded_pairs(rows,old):
    grouped=defaultdict(list)
    for row in rows:
        if row.get('dataset')=='MU':grouped[row['patient']].append(row)
    result=list(old);seen={pair_key(p) for p in old}
    for patient,visits in sorted(grouped.items()):
        visits.sort(key=lambda r:r['visit_order']);history=[]
        for i,row in enumerate(visits):
            if row.get('time_unit')!='diagnosis_relative_days':history=[];continue
            day=row['diagnosis_relative_day']
            if int(day)!=day:raise ValueError('MU matching requires integer days')
            if history and day<=history[-1]['diagnosis_relative_day']:raise ValueError('Invalid chronology')
            history.append(row);later=[]
            for target in visits[i+1:]:
                if target.get('time_unit')!='diagnosis_relative_days':break
                gap=target['diagnosis_relative_day']-day
                if int(gap)!=gap:raise ValueError('MU matching requires integer days')
                if gap>0:later.append((int(gap),target))
            for horizon in (12,24):
                if (patient,row['visit'],horizon) in seen or not later:continue
                gap,target=min(later,key=lambda p:(abs(p[0]-7*horizon),p[1]['visit_order'],p[1]['stem']))
                if abs(gap-7*horizon)<=28:
                    pair={'patient':patient,'dataset':'MU','history':list(history),'target':target,'horizon':horizon,'target_offset':gap/7-horizon}
                    result.append(pair);seen.add(pair_key(pair))
    return result


def inventory(pairs):
    return {'pairs':len(pairs),'patients':len({p['patient'] for p in pairs}),
        'horizons':{str(h):{'pairs':sum(p['horizon']==h for p in pairs),'patients':len({p['patient'] for p in pairs if p['horizon']==h})} for h in (12,24)}}


def manifest(out):
    out.mkdir(parents=True,exist_ok=True)
    cat=catalogue(OBS/'cavity-targets-v1/catalogue.json');old=json.loads((OLD/'population.json').read_text())
    assert len(old['fitting'])==87 and len(old['selection'])==58
    shutil.copyfile(OLD/'population.json',out/'original-population.json')
    pairs=expanded_pairs(cat['groups']['fitting'],old['fitting'])
    assert pairs[:87]==old['fitting']
    members=json.loads((OBS/'mu/membership.json').read_text());rows={}
    for split,items in [('fitting',pairs),('selection',old['selection'])]:
        for pair in items:
            assert pair['patient'] in members[split] and pair['patient'] not in members['reserved']
            # Recover one immediately preceding valid visit for the fixed memory comparison.
            needed=pair['history'][-2:]+[pair['target']]
            for row in needed:
                rows[row['stem']]={'row':row,'split':split}
    paths=[];missing=[]
    for stem,entry in sorted(rows.items()):
        row=entry['row'];absent=[k for k in ('images','support','mask') if not Path(row[k]).exists()]
        if absent:
            missing.append({'stem':stem,'missing':absent})
            for suffix in ('brain_t1c','brain_t1n','brain_t2w','brain_t2f','tumorMask'):
                paths.append(f'world-model-patient-state-20261105/mu/images/{row["patient"]}/{row["visit"]}/{stem}_{suffix}.nii.gz')
    write_json(out/'candidate-population.json',{'fitting':pairs,'selection':old['selection']})
    write_json(out/'required-visits.json',rows)
    (out/'restore-files0').write_bytes(b'\0'.join(p.encode() for p in paths)+b'\0')
    write_json(out/'manifest-inventory.json',{'original':inventory(old['fitting']),'candidate':inventory(pairs),
        'selection':inventory(old['selection']),'required_visits':len(rows),'visits_requiring_restoration':len(missing),'restore_source_files':len(paths),
        'missing':missing,'original_population_sha256':digest(OLD/'population.json'),'catalogue_sha256':digest(OBS/'cavity-targets-v1/catalogue.json'),
        'original_old_pool_file_preserved_byte_for_byte':digest(out/'original-population.json')==digest(OLD/'population.json'),
        'reserved_paths_requested':0,'candidate_not_frozen_until_actual_availability_checked':True})
    print(json.dumps({k:v for k,v in json.loads((out/'manifest-inventory.json').read_text()).items() if k!='missing'}),flush=True)


def restore_arrays(out,partial=False):
    import nibabel as nib
    import numpy as np
    started=time.monotonic();rows=json.loads((out/'required-visits.json').read_text());records=[]
    prior=json.loads((out/'restoration-partial.json').read_text()) if (out/'restoration-partial.json').exists() else {'records':[],'seconds':0.}
    for stem,entry in sorted(rows.items()):
        row=entry['row'];missing=[k for k in ('images','support','mask') if not Path(row[k]).exists()]
        if not missing:continue
        paths=[OBS/'mu/images'/row['patient']/row['visit']/(stem+'_'+s+'.nii.gz') for s in ('brain_t1c','brain_t1n','brain_t2w','brain_t2f','tumorMask')]
        if not all(p.exists() for p in paths):
            records.append({'stem':stem,'status':'missing archived source'});continue
        if digest(paths[-1])!=row['source_mask_sha256']:
            if partial:continue  # The last archive member can still be transferring.
            raise ValueError('Released source mask hash changed')
        images=[nib.load(p) for p in paths];first=images[0]
        if any(im.shape!=first.shape or not np.allclose(im.affine,first.affine,atol=1e-5,rtol=0) for im in images):raise ValueError('Geometry mismatch')
        transform=nib.orientations.ornt_transform(nib.orientations.io_orientation(first.affine),nib.orientations.axcodes2ornt(('L','A','S')))
        affine=first.affine@nib.orientations.inv_ornt_aff(transform,first.shape)
        if not np.allclose(affine,row['affine'],atol=1e-5,rtol=0):raise ValueError('Catalogue geometry mismatch')
        arrays=[];support=None
        for i,im in enumerate(images[:4]):
            x=nib.orientations.apply_orientation(np.asarray(im.dataobj,dtype=np.float32),transform)
            if list(x.shape)!=row['shape_xyz'] or not np.isfinite(x).all():raise ValueError('Invalid restored image')
            foreground=x!=0
            if i==0:support=foreground
            sd=float(x[foreground].std(dtype=np.float64))
            if sd<=0:raise ValueError('Degenerate normalization')
            z=np.zeros_like(x);z[foreground]=(x[foreground]-x[foreground].mean(dtype=np.float64))/sd
            arrays.append(z.transpose(2,0,1).astype(np.float16))
        raw=nib.orientations.apply_orientation(np.asanyarray(images[-1].dataobj),transform)
        if not np.isin(raw,(0,1,2,3,4)).all():raise ValueError('Unknown MU labels')
        labels=np.asarray([0,2,3,1,4],dtype=np.int8)[raw.astype(np.int64)].transpose(2,0,1)
        for key,value in [('images',np.stack(arrays)),('support',support.transpose(2,0,1).astype(np.uint8)),('mask',labels)]:
            path=Path(row[key]);path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():np.save(path,value)
        records.append({'stem':stem,'status':'restored','source_mask_verified':True})
        print(json.dumps({'restored':len(records),'stem':stem,'seconds':time.monotonic()-started}),flush=True)
    merged={r['stem']:r for r in prior['records'] if r['status']=='restored'}
    merged.update({r['stem']:r for r in records if r['status']=='restored'})
    if partial:
        write_json(out/'restoration-partial.json',{'records':list(merged.values()),'seconds':prior['seconds']+time.monotonic()-started,'manifest_frozen':False})
        return
    records=list(merged.values())+[r for r in records if r['status']!='restored']
    population=json.loads((out/'candidate-population.json').read_text());old=json.loads((out/'original-population.json').read_text())
    available=lambda row:all(Path(row[k]).exists() for k in ('images','support','mask'))
    fitting=[p for p in population['fitting'] if available(p['history'][-1]) and available(p['target'])]
    if fitting[:87]!=old['fitting'] or population['selection']!=old['selection']:raise ValueError('Original targets were changed')
    # Any missing historical source is explicitly unavailable, rather than removing its cutoff pair.
    write_json(out/'population.json',{'fitting':fitting,'selection':old['selection']})
    write_json(out/'restoration.json',{'records':records,'seconds':prior['seconds']+time.monotonic()-started,'frozen_fitting':inventory(fitting),
        'selection_unchanged':True,'pairs_removed_for_unavailable_images':len(population['fitting'])-len(fitting),
        'previous_visits_unavailable':sorted({p['history'][-2]['stem'] for p in fitting+old['selection'] if len(p['history'])>1 and not available(p['history'][-2])}),
        'frozen_manifest_sha256':digest(out/'population.json'),'reserved_images_opened':0})
    print('FROZEN '+json.dumps(inventory(fitting)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('manifest','restore'));parser.add_argument('--out',type=Path,required=True);parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    if args.stage=='manifest':manifest(args.out)
    else:restore_arrays(args.out,args.partial)
