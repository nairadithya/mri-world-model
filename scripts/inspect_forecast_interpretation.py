#!/usr/bin/env python3
"""Three explicitly selected diagnostic slices; not a representative outcome panel."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.patient_state.data import load_visit
from src.patient_state.structured_training import load_context
from src.patient_state.state import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True)
    args=p.parse_args();root=args.run_root
    rows=[r for path in sorted(args.analysis.glob('case-*.json')) for r in json.loads(path.read_text()) if r['method']=='current' and r['horizon']==24]
    criteria=[('Both-empty core: largest gain, then current volume error',[r for r in rows if r['compartment']==2 and r['presence_group']=='both_empty'],lambda r:(r['gain'],r['current_volume_abs_error_ml'])),
              ('Nonempty core with substantial change: largest gain',[r for r in rows if r['compartment']==2 and r['presence_group']=='persistent_nonempty' and r['meaningful_volume_change']],lambda r:r['gain']),
              ('FLAIR: widest interval',[r for r in rows if r['compartment']==3],lambda r:r['width90_ml'])]
    chosen=[(label,max(candidates,key=key)) for label,candidates,key in criteria if candidates]
    pairs=json.loads((root/'structured-selection-v1/population.json').read_text());fig,axes=plt.subplots(len(chosen),5,figsize=(15,3.7*len(chosen)),squeeze=False)
    receipt=[]
    for n,(criterion,row) in enumerate(chosen):
        i=row['evaluation_index'];pair=pairs[i];c=row['compartment']
        now=load_visit(pair['history'][-1],labels=True);future=load_visit(pair['target'],labels=True)
        pred=load_context(root/'structured-context-v1',pair['history'][-1])['predicted_segmentation'].numpy()
        forecast=np.load(args.analysis/f'maps-{i:04d}.npz')['current']
        a,b=now['labels'][0].numpy(),future['labels'][0].numpy();s0=now['support'][0,0].numpy();s1=future['support'][0,0].numpy()
        x,y=now['images'][0,1].numpy(),future['images'][0,1].numpy()
        relevant=(a==c)|(b==c)|(pred==c)|(forecast==c)
        z=int(np.argmax(relevant.sum((0,1))))
        coords=np.argwhere(relevant[:,:,z]);low=np.maximum(coords.min(0)-20,0);high=np.minimum(coords.max(0)+21,np.array(a.shape[:2]))
        sl=(slice(low[0],high[0]),slice(low[1],high[1]),z)
        def overlay(ax,im,mask,color,alpha=.55):
            m=mask[sl].T;rgba=np.zeros((*m.shape,4));rgba[m,:3]=color;rgba[m,3]=alpha;ax.imshow(rgba,origin='lower')
        for j,ax in enumerate(axes[n]):
            ax.imshow((x if j<2 else y)[sl].T,cmap='gray',origin='lower',vmin=-1.5,vmax=2.5);ax.axis('off')
        for j,label in enumerate(('Current reference','Current model','Current-only forecast','Later reference','Edits vs persistence')):
            axes[n,j].set_title(label,fontsize=10)
        for j,mask in ((0,a==c),(1,pred==c),(2,forecast==c),(3,b==c)):
            overlay(axes[n,j],x,mask,(1,.6,0) if j in (0,3) else (0,.8,1))
        overlay(axes[n,0],x,a==-1,(.7,.3,.9),.4);overlay(axes[n,3],y,b==-1,(.7,.3,.9),.4)
        valid=s1&(b>=0);common=valid&s0&(a>=0);edit=(pred==c)!=(forecast==c)
        corrected=edit&((forecast==c)==(b==c));stable=common&((a==c)==(b==c));change=common&~stable
        overlay(axes[n,4],y,corrected&stable,(.1,.9,.2));overlay(axes[n,4],y,corrected&change,(.1,.6,1))
        overlay(axes[n,4],y,edit&~corrected&common,(1,.15,.15));overlay(axes[n,4],y,edit&valid&~common,(.7,.3,.9))
        axes[n,0].text(0,-.08,f'Case {n+1}, 24 weeks\n{criterion}',transform=axes[n,0].transAxes,fontsize=9,va='top')
        receipt.append({'criterion':criterion,'patient':row['patient'],'evaluation_index':i,'compartment':c,'slice':z,'dice_gain':row['gain']})
    fig.suptitle('Bounded MU interpretation: selected diagnostic slices, current-only model',fontsize=14)
    fig.text(.5,.015,'Reference tissue: orange; cavity: purple. Edits: green = correct unchanged anatomy; blue = correct reference change; red = harmful; purple = current reference unscored.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.96));fig.savefig(args.analysis/'diagnostic-slices.png',dpi=140);plt.close(fig)
    write_json(args.analysis/'private-visual-cases.json',receipt)


if __name__=='__main__':main()
