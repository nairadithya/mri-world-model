#!/usr/bin/env python3
"""Preserve the working baseline and inspect six specific MU compartment failures."""
import argparse,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import nibabel as nib
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from src.patient_state.data import catalogue,load_visit,metadata
from src.patient_state.labels import harmonize_labels
from src.patient_state.observation import JointObservation
from src.patient_state.state import digest,write_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
a=p.parse_args();r=a.root
out=r/'observation-preserved-v1';out.mkdir(exist_ok=False)
for name in ('best.pt','config.json','initialization.json','curve.json','completed.json'):
 shutil.copy2(r/'observation-v2'/name,out/name)
cat=catalogue(r/'mu-prepared/catalogue.json')
rows=cat['groups']['selection']
common=[{'patient':x['patient'],'visit':x['visit'],'stem':x['stem']} for x in rows if x.get('dataset','LUMIERE')=='LUMIERE']
write_json(out/'comparison-population.json',common)
files=[out/'best.pt',out/'config.json',r/'observation-selection-full/aggregate.json']
for name in ('timeline-joint','timeline-joint-missing-flair'):
 files.extend(sorted((r/name).rglob('timeline.json')))
 files.extend(sorted((r/name).rglob('state.pt')))
write_json(out/'manifest.json',{'files':{str(x):digest(x) for x in files},
 'common_population':{'patients':len({x['patient'] for x in common}),'visits':len(common)},
 'comparison':'same 178 LUMIERE visits, same support/labels, patient-uniform aggregation; CT1/T1/T2 for all matched arms',
 'inference':'128-cube stride-aligned overlap; positive triangular blending; all pyramid features saved',
 'scope':'working development baseline; jointly trained capability, not unseen-task transfer'})
records=[]
for row in rows:
 if row.get('dataset')=='MU':
  score=json.loads((r/'observation-selection-full'/(row['stem']+'.json')).read_text())
  records.append((row,score))
nonempty=[x for x in records if x[1]['segmentation']['2']['nonempty_dice'] is not None]
absent=[x for x in records if x[1]['segmentation']['2']['empty_reference_fp_ml'] is not None]
selected=[]
for category, candidates in [('nonenhancing_miss',sorted(nonempty,key=lambda x:x[1]['segmentation']['2']['nonempty_dice'])),
 ('absent_compartment_false_positive',sorted(absent,key=lambda x:-x[1]['segmentation']['2']['empty_reference_fp_ml'])),
 ('cavity_overlap',sorted(records,key=lambda x:-int((np.load(x[0]['mask'],mmap_mode='r')==-1).sum())))]:
 count=0
 for row,score in candidates:
  if row['stem'] in {x[1]['stem'] for x in selected}:continue
  selected.append((category,row,score));count+=1
  if count==2:break
obs=JointObservation.from_checkpoint(out/'best.pt');torch.set_num_threads(4)
fig,axes=plt.subplots(len(selected),4,figsize=(14,4*len(selected)),squeeze=False)
cm=ListedColormap([(0,0,0,0),(.95,.15,.1,.8),(1,.65,.05,.8),(.15,.5,1,.65),(.9,.1,.9,.8)])
reports=[]
mu=Path(cat['mu_membership_path']).parent
for i,(category,row,old) in enumerate(selected):
 case=load_visit(row,labels=True);y=case['labels'][0].numpy();b=case['support'][0,0].numpy()
 raw=nib.load(mu/'images'/row['patient']/row['visit']/(row['stem']+'_tumorMask.nii.gz'))
 tr=nib.orientations.ornt_transform(nib.orientations.io_orientation(raw.affine),nib.orientations.axcodes2ornt(('L','A','S')))
 source=nib.orientations.apply_orientation(np.asarray(raw.dataobj),tr)
 mapped=harmonize_labels(source,'MU')
 if not np.array_equal(mapped,y):raise AssertionError('Source mapping differs from prepared target')
 state=obs.encode(case['images'].cuda(),metadata(row));outputs=obs.read(state)
 pred=outputs['logits'].argmax(1)[0].cpu().numpy();pred[~b]=0
 valid=b&(y>=0);cavity=y==-1
 confusion=[[int(((y==a)&(pred==c)&valid).sum()) for c in range(4)] for a in range(4)]
 report={'category':category,'patient':row['patient'],'visit':row['visit'],'mapping_exact':True,
  'nonenhancing_score':old['segmentation']['2'],'confusion_voxels_reference_rows':confusion,
  'cavity_ml':float(cavity.sum()/1000),'cavity_predicted_as_ml':{str(c):float(((pred==c)&cavity).sum()/1000) for c in range(4)},
  'nonenhancing_missed_ml':float(((y==2)&(pred!=2)&b).sum()/1000),
  'nonenhancing_false_positive_outside_cavity_ml':float(((y!=2)&(pred==2)&valid).sum()/1000)}
 reports.append(report)
 attention=((y==2)|(pred==2)|cavity).sum((0,1));z=int(attention.argmax())
 flair=case['images'][0,3,:,:,z].numpy().T;ct=case['images'][0,0,:,:,z].numpy().T
 label=y[:,:,z].copy();label[label<0]=4
 for j in range(4):
  ax=axes[i,j];ax.imshow(ct if j==1 else flair,cmap='gray',origin='lower',vmin=-1,vmax=2.5)
  if j==2:ax.imshow(label.T,cmap=cm,origin='lower',vmin=0,vmax=4,interpolation='nearest')
  if j==3:ax.imshow(pred[:,:,z].T,cmap=cm,origin='lower',vmin=0,vmax=4,interpolation='nearest')
  ax.set_title([category+' · FLAIR','Contrast T1','Reference (cavity magenta)','Prediction'][j],fontsize=9);ax.axis('off')
 del state,outputs,case
fig.suptitle('Bounded MU review: enhancing red, nonenhancing orange, FLAIR abnormality blue; cavity ignored in shared score',fontsize=11)
fig.tight_layout();fig.savefig(out/'mu-failure-inspection.png',dpi=120);plt.close(fig)
write_json(out/'mu-failure-inspection.json',{'cases':reports,'reserved_opened':0,
 'cavity_policy':'Ignored in shared segmentation loss/metrics; cavity overlap separately quantified, not counted as background truth',
 'annotation_limit':'MU nonenhancing core and separate cavity; LUMIERE combined necrotic/nonenhancing proxy and no separate cavity',
 'mapping_defects_found':False,'selection_rule':'two worst nonempty necrotic, two worst absent necrotic FP, two largest remaining cavities'})
print(json.dumps({'preserved':str(out),'cases':len(reports),'mapping_defects_found':False}),flush=True)
