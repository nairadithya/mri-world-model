#!/usr/bin/env python3
"""Bounded metadata-chosen MU correspondence inspection before spatial fitting."""
import argparse,json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.signal import fftconvolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.patient_state.data import catalogue,load_visit
from src.patient_state.timing import timed_pairs
from src.patient_state.state import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalogue',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();cat=catalogue(args.catalogue);chosen=[]
    for group in ('fitting','selection'):
        pairs=[p for p in timed_pairs(cat['groups'][group]) if p['dataset']=='MU']
        patients=sorted({p['patient'] for p in pairs},key=lambda s:hashlib.sha256(('geometry-v1:'+s).encode()).hexdigest())[:3]
        chosen.extend((group,next(p for p in pairs if p['patient']==patient)) for patient in patients)
    fig,axes=plt.subplots(len(chosen),3,figsize=(10,3*len(chosen)))
    records=[]
    for n,(group,pair) in enumerate(chosen):
        a,b=pair['history'][-1],pair['target']
        if a['shape_xyz']!=b['shape_xyz'] or not np.allclose(a['affine'],b['affine'],atol=1e-5):
            raise ValueError('MU longitudinal geometry mismatch')
        ca,cb=load_visit(a),load_visit(b)
        x,y=ca['images'][0,1].numpy(),cb['images'][0,1].numpy()
        sa,sb=ca['support'][0,0].numpy(),cb['support'][0,0].numpy()
        mask=sa&sb
        correlation=float(np.corrcoef(x[mask],y[mask])[0,1])
        support_dice=2*int((sa&sb).sum())/int(sa.sum()+sb.sum())
        # A coarse translation diagnostic is not a fitted registration transform.
        xx,yy=x[::4,::4,::4]*sa[::4,::4,::4],y[::4,::4,::4]*sb[::4,::4,::4]
        corr=fftconvolve(xx,yy[::-1,::-1,::-1],mode='full')
        peak=(np.array(np.unravel_index(np.argmax(corr),corr.shape))-(np.array(yy.shape)-1))*4
        records.append({'group':group,'patient':pair['patient'],'cutoff':a['visit'],'target':b['visit'],
                        'horizon':pair['horizon'],'support_dice':support_dice,'t1_correlation':correlation,
                        'coarse_translation_peak_mm':peak.tolist()})
        z=int(np.median(np.argwhere(mask)[:,2]));ii,jj=np.indices(x.shape[:2]);checker=(ii//16+jj//16)%2==0
        panels=[x[:,:,z],y[:,:,z],np.where(checker,x[:,:,z],y[:,:,z])]
        for ax,array,title in zip(axes[n],panels,('Cutoff T1','Later T1','Checkerboard')):
            ax.imshow(array.T,origin='lower',cmap='gray',vmin=-1.5,vmax=2.5);ax.axis('off');ax.set_title(f'{group} case {n+1}: {title}')
    args.out.mkdir(parents=True,exist_ok=False)
    fig.tight_layout();fig.savefig(args.out/'correspondence.png',dpi=140);plt.close(fig)
    write_json(args.out/'private-cases.json',records)
    summary={'cases':len(records),'support_dice_range':[min(r['support_dice'] for r in records),max(r['support_dice'] for r in records)],
             't1_correlation_range':[min(r['t1_correlation'] for r in records),max(r['t1_correlation'] for r in records)],
             'maximum_coarse_translation_mm':max(max(abs(x) for x in r['coarse_translation_peak_mm']) for r in records),
             'geometry_headers_match':True,'scope':'Six metadata-selected development cases; no cohort-wide registration guarantee',
             'source_registration':'SRI24 rigid 1mm, https://doi.org/10.1038/s41597-025-06011-7',
             'reserved_images_opened':0}
    write_json(args.out/'summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
