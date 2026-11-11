#!/usr/bin/env python3
"""Freeze acquisition roles from prior complete-visit metadata; restore named sources."""
import argparse
from collections import defaultdict,Counter
import hashlib,json,math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def manifest(catalogue,archive_plan,out):
    value=json.loads(catalogue.read_text());entries={r['path']:r for r in json.loads(archive_plan.read_text())['entries']}
    mu={g:sorted({r['patient'] for r in value['groups'][g] if r.get('dataset')=='MU'},key=lambda p:hashlib.sha256(('acq-jepa-v1|'+p).encode()).hexdigest()) for g in ('fitting','selection')}
    policy=set(mu['fitting'][:math.ceil(.2*len(mu['fitting']))]);selection=set(mu['selection'][:len(mu['selection'])//2])
    roles={r:[] for r in ('R','P','V','A')};source_root='world-model-patient-state-20261105';lumiere_root='world-model-r65-lumiere-20261103'
    for group in ('fitting','selection'):
        for original in value['groups'][group]:
            row=dict(original);is_mu=row.get('dataset')=='MU'
            if not is_mu and group!='fitting':continue
            role=('P' if row['patient'] in policy else 'R') if group=='fitting' else ('V' if row['patient'] in selection else 'A')
            row['role']=role
            if is_mu:
                prefix=f"{source_root}/mu/images/{row['patient']}/{row['visit']}/{row['stem']}_"
                paths=[prefix+s+'.nii.gz' for s in ('brain_t1c','brain_t1n','brain_t2w','brain_t2f','tumorMask')]
            else:
                prefix=f"{lumiere_root}/atlas-all/Imaging/{row['patient']}/{row['visit']}/DeepBraTumIA-segmentation/atlas/"
                paths=[prefix+'skull_strip/'+s+'_skull_strip.nii.gz' for s in ('ct1','t1','t2','flair')]+[prefix+'segmentation/seg_mask.nii.gz']
            if any(p not in entries or not entries[p]['archive'] for p in paths):raise ValueError('Previously eligible source missing from archive manifest: '+row['stem'])
            row['released_sources']=paths;roles[role].append(row)
    patients={role:{r['patient'] for r in rows} for role,rows in roles.items()}
    for a in patients:
        for b in patients:
            if a!=b and patients[a]&patients[b]:raise ValueError('Patient overlap')
    counts={role:dict(patients=len(patients[role]),visits=len(rows),datasets=dict(Counter(r.get('dataset','LUMIERE') for r in rows))) for role,rows in roles.items()}
    write(out/'roles.json',dict(identity='acq-jepa-v1',roles=roles,counts=counts,assignment='SHA256(acq-jepa-v1|patient); ceil20% fitting P; floor half selection V',eligibility='All complete visits in prior cavity catalogue; archived files restored, not filtered by longitudinal eligibility or current cache existence',catalogue_sha256=hashlib.sha256(catalogue.read_bytes()).hexdigest()))
    for phase,rs in [('development',sum([roles[k] for k in ('R','P','V')],[])),('assessment',roles['A'])]:
        paths={p for row in rs for p in row['released_sources']}
        if phase=='development':paths.add(source_root+'/pretrained/openmind-resenc-l-mae.pth')
        for root in (source_root,lumiere_root):
            selected=sorted(p for p in paths if p.startswith(root+'/'))
            (out/f'{phase}-{root}.files0').write_bytes(b'\0'.join(p.encode() for p in selected)+b'\0' if selected else b'')
    print(json.dumps(counts),flush=True)


def restore(roles_path,base,out,assessment=False):
    import nibabel as nib
    import numpy as np
    data=json.loads(roles_path.read_text());roles=('A',) if assessment else ('R','P','V')
    if assessment and not (out.parent/'policy-lock.json').exists():raise ValueError('Assessment remains closed until policy lock')
    for role in roles:
        for row in data['roles'][role]:
            # Existing prepared images are per-sequence independently normalized.
            # Their CT1-derived support file is never read by this study.
            if Path(row['images']).is_file() and Path(row['mask']).is_file():continue
            paths=[base/p for p in row['released_sources']]
            if not all(p.is_file() for p in paths):raise FileNotFoundError(row['stem'])
            ims=[nib.load(p) for p in paths];first=ims[0]
            if any(im.shape!=first.shape or not np.allclose(im.affine,first.affine,atol=1e-5) for im in ims[1:]):
                raise ValueError('Released image/annotation geometry mismatch')
            orientation=nib.orientations.ornt_transform(nib.orientations.io_orientation(first.affine),nib.orientations.axcodes2ornt(('L','A','S')))
            arrays=[]
            for im in ims[:4]:
                if im.shape!=first.shape or not np.allclose(im.affine,first.affine,atol=1e-5):raise ValueError('Released sequence geometry changed')
                x=nib.orientations.apply_orientation(np.asarray(im.dataobj,dtype=np.float32),orientation)
                fg=x!=0
                if not np.isfinite(x).all() or not fg.any():raise ValueError('Nonfinite or empty image')
                sd=x[fg].std(dtype=np.float64)
                if sd<=0:raise ValueError('Degenerate sequence')
                z=np.zeros_like(x);z[fg]=(x[fg]-x[fg].mean(dtype=np.float64))/sd
                arrays.append(z.transpose(2,0,1).astype(np.float16))
            y=nib.orientations.apply_orientation(np.asarray(ims[-1].dataobj),orientation)
            if row.get('dataset')=='MU':
                if not np.isin(y,(0,1,2,3,4)).all():raise ValueError('Invalid MU label')
                y=np.asarray([0,2,3,1,4],np.int8)[y.astype(int)]
            else:
                # Released DeepBraTumIA: 1 enhancing, 2 nonenhancing, 3 FLAIR.
                if not np.isin(y,(0,1,2,3)).all():raise ValueError('Invalid LUMIERE label')
                y=y.astype(np.int8)
            directory=out/role;directory.mkdir(parents=True,exist_ok=True)
            row['images']=str(directory/(row['stem']+'_images.npy'));row['mask']=str(directory/(row['stem']+'_mask.npy'))
            np.save(row['images'],np.stack(arrays));np.save(row['mask'],y.transpose(2,0,1))
            print(json.dumps(dict(role=role,restored=row['stem'])),flush=True)
        write(out/f'{role}.json',data['roles'][role])
    write(out/'restoration.json',dict(roles=list(roles),CT1_support_loaded=False,assessment_opened=assessment))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('manifest','restore'));p.add_argument('--catalogue',type=Path);p.add_argument('--archive-plan',type=Path);p.add_argument('--roles',type=Path);p.add_argument('--base',type=Path,default=Path('/home/aditya/experiments'));p.add_argument('--out',type=Path,required=True);p.add_argument('--assessment',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.stage=='manifest':manifest(a.catalogue,a.archive_plan,a.out)
    else:restore(a.roles,a.base,a.out,a.assessment)


if __name__=='__main__':main()
