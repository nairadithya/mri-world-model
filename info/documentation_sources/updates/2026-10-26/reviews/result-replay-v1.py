#!/usr/bin/env python3
"""Independent saved-artifact R37 audit. No fitting/encoder extraction/assessment rerun.
Only project API called: budget_stage, solely for authorized resource metering.
Numeric tolerances fixed before execution: scores 1e-8, fixed-weight forward 1e-4.
CPU-only; first canonical primary and first canonical single row per neural arm/seed.
"""
import os
os.environ['OMP_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys, json, hashlib, math, re, time, signal, resource, traceback
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.special import ndtr
from statistics import NormalDist
import torch
import torch.nn.functional as F
BASE=Path('/home/aditya/experiments/world-model-r37-20261026')
ROOT=BASE/'repo'; OUT=ROOT/'outputs/r37-full/continuation-20261026'; HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.harness.analysis.spatial_dynamics_runtime import budget_stage
STAGE='independent-result-review-v1'; LEDGER=BASE/'receipts/continuation_budget.json'
CONTRACT='6f73bb278bfbb4afd5778e24181f78485d80aa2cb792a82732900bfbc4dc9f1c'
AGG='bf36341fbe581329a9bd2cc11a49deef8a552a455dcab0fb66b4b13074b12b9f'
REVIEW='72a85f07f597fff07bb55ed2b41ce362b6102265d48ea5066d8ba1ae3928a5bf'
LEGACY='cb717e6205a4af7e912e18854f19892c0a3f9169206cb48a22163aed3386c695'
ARMS=('H','L','P','E','R','G'); NEURAL=('H','L','G'); SEEDS=(42,43,44); GRID=(.5,1.,2.)
LEVELS=(.5,.8,.95); COMPS=('necrotic_non_enhancing','enhancing','edema_flair')
checks=0; discrepancies=[]; max_errors={}; artifacts={}; result={}
torch.set_num_threads(1); torch.set_grad_enabled(False)

def J(p): return json.loads(Path(p).read_text())
def canonical(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def require(ok,label):
 global checks
 checks+=1
 if not bool(ok): raise AssertionError(label)
def close(a,b,label,tol=1e-8):
 global checks
 checks+=1
 a,b=np.asarray(a),np.asarray(b)
 if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
  discrepancies.append(label+':shape/nonfinite'); return
 e=float(np.max(np.abs(a-b),initial=0)); max_errors[label.split('/')[0]]=max(max_errors.get(label.split('/')[0],0),e)
 if not np.allclose(a,b,atol=tol,rtol=tol): discrepancies.append(label)
def tree(a,b,label):
 if isinstance(a,dict):
  for k,v in a.items():
   require(k in b,label+':missing-key'); tree(v,b[k],label+'/'+k)
 elif isinstance(a,(list,tuple)):
  require(len(a)==len(b),label+':length')
  for i,(x,y) in enumerate(zip(a,b)): tree(x,y,label+'/'+str(i))
 elif a is None or isinstance(a,(str,bool)):
  require(a==b,label+':value')
 else: close(a,b,label)
def load(p): return torch.load(p,map_location='cpu',weights_only=False)
def weights(ids):
 ids=np.asarray(ids); unique,counts=np.unique(ids,return_counts=True); lookup=dict(zip(unique,counts))
 return np.array([1/(len(unique)*lookup[p]) for p in ids])
def pmean(x,ids):
 x=np.asarray(x,dtype=float); return np.stack([x[np.asarray(ids)==p].mean(0) for p in sorted(set(ids))]).mean(0)
def crps(m,s,y):
 z=(np.asarray(y)-m)/s
 return s*(z*(2*ndtr(z)-1)+2*np.exp(-z*z/2)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))
def boot(a,b,ids,conf=.95):
 d=np.asarray(a)-np.asarray(b)
 if d.ndim>1: d=d.mean(axis=tuple(range(1,d.ndim)))
 d=np.array([d[np.asarray(ids)==p].mean() for p in sorted(set(ids))])
 ix=np.random.default_rng(2026).integers(0,len(d),size=(10000,len(d)))
 samples=d[ix].mean(1); q=(1-conf)/2
 return dict(difference=float(d.mean()),ci=np.quantile(samples,[q,1-q]).tolist(),confidence=conf,patients=len(d),examples=len(ids),draws=10000,seed=2026)
def moment_abs(d,s):
 z=d/s; return s*np.sqrt(2/np.pi)*np.exp(-z*z/2)+d*(2*ndtr(z)-1)
def mixture(latest,truth,changes,ids,k):
 w=weights(ids); avg=w@changes; shift=avg+k*(changes-avg)
 centers=latest[:,None,:]+shift[None,:,:]
 pair=moment_abs(shift[:,None,:]-shift[None,:,:],np.sqrt(2)*.05)
 second=np.einsum('i,j,ijc->c',w,w,pair)
 scores=(moment_abs(truth[:,None,:]-centers,.05)*w[None,:,None]).sum(1)-second[None,:]/2
 return scores,centers,w

def summary(b,y,cut,ids):
 c=b['crps']; error=np.abs(b['mean']-y)
 v={'patients':len(set(ids)),'examples':len(ids),'crps':float(pmean(c,ids).mean()),'compartment_crps':pmean(c,ids).tolist(),
    'uncalibrated_crps':float(pmean(b['uncalibrated_crps'],ids).mean()),'mae':float(pmean(error,ids).mean()),
    'compartment_mae':pmean(error,ids).tolist(),'coverage':{},'width':{},'subsets':{}}
 for level in LEVELS:
  lo,hi=b[f'lower_{level}'],b[f'upper_{level}']; v['coverage'][str(level)]=pmean(((y>=lo)&(y<=hi)).astype(float),ids).tolist(); v['width'][str(level)]=pmean(hi-lo,ids).tolist()
 if 'latent_mse' in b: v['latent_mse_diagnostic']=float(pmean(b['latent_mse'],ids))
 for name,mask in {'empty_target':y==0,'appearance':(cut==0)&(y>0),'disappearance':(cut>0)&(y==0),'both_empty':(cut==0)&(y==0)}.items():
  values=[]
  for j in range(3):
   ix=np.flatnonzero(mask[:,j]); chosen=np.asarray(ids)[ix]
   values.append({'patients':len(set(chosen)),'examples':len(ix),'crps':float(pmean(c[ix,j],chosen)) if len(ix) else None,'mae':float(pmean(error[ix,j],chosen)) if len(ix) else None})
  v['subsets'][name]=values
 return v

def order(v):
 m=re.fullmatch(r'week-(\d+)(?:-(\d+))?',v); require(m is not None,'chronology/label'); return int(m[1]),int(m[2] or 0)
def conv(x,state,name,pad=0): return F.conv3d(x,state[name+'.weight'],state[name+'.bias'],padding=pad)
def forward(maps,avail,gaps,mask,state,arm):
 current=maps[:,-1]; maps=maps.clone()
 if arm=='L': maps[:,:-1]=0
 if arm=='G': maps[:,:-1]=maps[:,:-1].mean((-3,-2,-1),keepdim=True)
 hidden=torch.zeros((1,16,*maps.shape[-3:]))
 for t in range(4):
  reduced=conv(maps[:,t],state,'reduce'); cov=torch.cat((avail[:,t,:,None,None,None].expand(-1,-1,*maps.shape[-3:]),gaps[:,t,None,None,None,None].expand(-1,1,*maps.shape[-3:])),1)
  reset,update=torch.sigmoid(conv(torch.cat((reduced,hidden,cov),1),state,'gates',1)).chunk(2,1)
  proposal=torch.tanh(conv(torch.cat((reduced,reset*hidden,cov),1),state,'candidate',1))
  hidden=torch.where(mask[:,t,None,None,None,None].bool(),(1-update)*hidden+update*proposal,torch.zeros_like(hidden))
 mu=current+conv(hidden,state,'residual'); sigma=F.softplus(F.linear(hidden.mean((-3,-2,-1)),state['scale.weight'],state['scale.bias']))+1e-6
 return mu,sigma

def audit(guard):
 started=time.monotonic(); sigtime=signal.getitimer(signal.ITIMER_REAL)[0]; signal.setitimer(signal.ITIMER_REAL,min(sigtime,300))
 soft,hard=resource.getrlimit(resource.RLIMIT_CPU); resource.setrlimit(resource.RLIMIT_CPU,(min(soft,math.ceil(time.process_time()+600)),hard))
 require(sha(OUT/'aggregate.json')==AGG,'aggregate/hash')
 agg=J(OUT/'aggregate.json'); contract=J(OUT/'contract.json'); require(canonical(contract)==CONTRACT,'contract/digest')
 reviewpath=ROOT/contract['review']['review_path']; require(sha(reviewpath)==REVIEW,'implementation/approved-report')
 text=reviewpath.read_text(); approved=json.loads(re.findall(r'```json\s*(.*?)\s*```',text,re.S)[-1]); require(approved['verdict']=='APPROVED_FOR_R37_IMPLEMENTATION','implementation/verdict')
 require(approved['reviewed_code']==contract['code_sha256'],'implementation/code-binding'); require(len(contract['code_sha256'])==13,'implementation/scope')
 snap=OUT/'locked-inputs-snapshot'
 for p,h in contract['code_sha256'].items(): require(sha(ROOT/p)==h and sha(snap/p)==h,'implementation/source-snapshot')
 require(sha(ROOT/'info/METHODS.md')==contract['protocol_sha256']==sha(snap/'info/METHODS.md'),'protocol/locked')
 cont='info/documentation_sources/updates/2026-10-26/r37-continuation.md'; require(sha(ROOT/cont)==sha(snap/cont),'continuation/snapshot')
 require(sha(snap/contract['review']['review_path'])==REVIEW,'implementation/review-snapshot')
 require(sha(ROOT/'info/documentation_sources/updates/2026-10-26/reviews/implementation-v1.md')=='bdb9a17a5ada777ee1481440fff655e156c62d34410f8030136571a1eba533de','implementation/v1-preserved')
 gate=J(OUT/'implementation-gate.json'); require(sha(OUT/'implementation-approval.json')==contract['review_gate_sha256'] and J(OUT/'implementation-approval.json')==gate==contract['review'],'implementation/gate')
 completion=J(OUT/'fitting-complete.json'); require(completion['contract_digest']==CONTRACT and completion['seeds']==list(SEEDS),'fitting/contract-seeds')
 expected={'controls.pt','ridge/ridge.pt'}|{f'seed{s}/{a}{t}.pt' for s in SEEDS for a in NEURAL for t in ('','-last','-partial')}
 require(set(completion['weight_hashes'])==expected,'fitting/29-artifacts')
 for p,h in completion['weight_hashes'].items(): require(sha(OUT/p)==h,'fitting/weight-hash'); artifacts[p]=h
 files=['implementation-approval.json','aggregate.json','aggregate-staged.json','assessment-complete.json','contract.json','fitting-complete.json','implementation-gate.json','features.pt','features.json','private_predictions.pt','private_predictions.npz','ridge/source_oof.npz','ridge/complete.json']
 for p in files: artifacts[p]=sha(OUT/p)
 require(artifacts['features.pt']==contract['cache_sha256'],'cache/hash')
 cache=load(OUT/'features.pt'); bundle=load(OUT/'private_predictions.pt'); marker=J(OUT/'assessment-complete.json'); arrays=np.load(OUT/'private_predictions.npz',allow_pickle=False)
 for key,wanted in {'contract_digest':CONTRACT,'fitting_completion_sha256':artifacts['fitting-complete.json'],'fitted_artifact_manifest_sha256':canonical(completion['weight_hashes'])}.items():
  require(bundle[key]==marker[key]==agg[key]==wanted,'binding/'+key)
 require(marker['predictions_sha256']==artifacts['private_predictions.pt'] and marker['predictions_npz_sha256']==artifacts['private_predictions.npz'],'binding/predictions')
 require(canonical(cache['rows'])==contract['row_manifest_sha256'] and cache['splits']==contract['splits'],'binding/cache-rows-splits')
 require(canonical(cache['geometry_receipts'])==contract['geometry_receipts_sha256']==cache['geometry_receipts_sha256'],'geometry/receipt-digest')
 require(contract['image_sha256']==cache['image_hashes']=={p:v['file_sha256'] for p,v in cache['geometry_receipts'].items()},'geometry/file-binding')
 require(len(cache['geometry_receipts'])==2396 and len(cache['visit_ids'])==599,'geometry/counts')
 require(cache['reconstruction']['atol']==cache['reconstruction']['rtol']==1e-4,'reconstruction/fixed-tolerances')
 old=ROOT/'outputs/representation-comparison/retry-complete/unpacked'
 inputpaths={'prepared':old/'training_prepared.json','checkpoint':old/'results/comparison/seed42/state/best.pt','features':old/'results/comparison/seed42/state/features.pt','direct_predictions':old/'results/comparison/seed42/state/direct.predictions.npz','run':old/'results/comparison/run.json','inventory':ROOT/'outputs/representation-comparison/inventory.json','anatomy_manifest':ROOT/'outputs/anatomy_harmonization.json','chronology_manifest':ROOT/'info/lumiere_task_manifest.json'}
 require(contract['original_input_sha256']==cache['input_hashes']=={k:sha(p) for k,p in inputpaths.items()},'binding/original-inputs')
 prep=J(inputpaths['prepared']); original=J(inputpaths['anatomy_manifest']); run=J(inputpaths['run']); cp=load(inputpaths['checkpoint']); feat=load(inputpaths['features'])
 splits={k:set(v) for k,v in cache['splits'].items()}; require(tuple(map(len,splits.values()))==(52,26,13) or sorted(map(len,splits.values()))==[13,26,52],'splits/sizes')
 require(all(not splits[a]&splits[b] for a in splits for b in splits if a<b),'splits/disjoint')
 require(splits['source52']==set(run['training_patients']) and splits['validation13']==set(run['validation_patients']) and splits['unseen26']==set(run['evaluation_patients']),'splits/original-membership')
 pv={v['visit_id']:v for v in prep['visits']}; orig={};
 for v in original['visits']:
  if v['cohort']=='LUMIERE': orig.setdefault(v['patient_id'],[]).append(v)
 require(sum(map(len,orig.values()))==638,'chronology/full-count')
 pairs={(p['patient_id'],p['source_visit'],p['target_visit']):p for p in prep['pairs']}; rows=[]
 for patient,vis in sorted(orig.items()):
  vis=sorted(vis,key=lambda v:order(v['visit']))
  for i in range(len(vis)-1):
   a,b=vis[i],vis[i+1]
   if a['visit_id'] not in pv or b['visit_id'] not in pv: continue
   start=i
   while start>0 and vis[start-1]['visit_id'] in pv: start-=1
   hist=vis[max(start,i-3):i+1]; gaps=[7*(order(y['visit'])[0]-order(x['visit'])[0]) for x,y in zip(hist[:-1],hist[1:])]
   if any(g<=0 for g in gaps): continue
   pair=pairs.get((patient,a['visit'],b['visit']))
   if pair is None: continue
   require(pair['usable'] and not pair.get('exclusions'),'chronology/pair-usable'); require(pv[a['visit_id']]['pregrid']==pv[b['visit_id']]['pregrid'],'chronology/pair-grid')
   rows.append({'row_id':a['visit_id']+'->'+b['visit_id'],'patient_id':patient,'cutoff_visit_id':a['visit_id'],'target_visit_id':b['visit_id'],'primary_history':len(hist)>=2,'history_visit_ids':[v['visit_id'] for v in hist],'gap_log1p':[0.]*(4-len(hist))+[0.]+[math.log1p(g) for g in gaps],'history_mask':[False]*(4-len(hist))+[True]*len(hist),'truth':[math.log1p(pv[b['visit_id']]['values_mm3'][c]) for c in COMPS],'cutoff_measured':[math.log1p(pv[a['visit_id']]['values_mm3'][c]) for c in COMPS]})
 require(canonical(rows)==canonical(cache['rows']) and len(rows)==423,'chronology/independent-rows')
 subsets={k:[r for r in rows if r['patient_id'] in v] for k,v in splits.items()}; src=subsets['source52']; val=subsets['validation13']; assess=subsets['unseen26']
 cohorts={'primary':[r for r in assess if r['primary_history']],'single':[r for r in assess if not r['primary_history']]}
 earliest={}
 for r in cohorts['primary']: earliest.setdefault(r['patient_id'],r)
 cohorts['nuisance']=list(earliest.values()); require(len(earliest)>=15,'cohorts/minimum')
 for name,rr in cohorts.items():
  require(canonical(rr)==canonical(bundle[name+'_rows'])==bundle['cohort_row_sha256'][name]==marker['cohort_row_sha256'][name]==agg['cohort_row_sha256'][name],'cohorts/exact-binding')
  require(np.array_equal(arrays[name+'_row_ids'],[r['row_id'] for r in rr]) and np.array_equal(arrays[name+'_patient_ids'],[r['patient_id'] for r in rr]),'cohorts/npz-identities')
  close(arrays[name+'_truth'],[r['truth'] for r in rr],'targets/'+name); close(arrays[name+'_cutoff_measured'],[r['cutoff_measured'] for r in rr],'cutoff/'+name)
 require(not {r['row_id'] for r in cohorts['primary']}&{r['row_id'] for r in cohorts['single']},'cohorts/separate')
 raw=cache['maps']; ix={v:i for i,v in enumerate(cache['visit_ids'])}; sourceix=[ix[v] for v in cache['visit_ids'] if pv[v]['patient_id'] in splits['source52']]
 require(len(sourceix)==353 and set(cache['source_normalizer_visit_ids'])=={cache['visit_ids'][i] for i in sourceix},'normalization/source-visits')
 zz=raw[sourceix].double(); close(zz.mean((0,2,3,4)).float(),cache['feature_mean'],'normalization/mean',1e-6); close(zz.std((0,2,3,4),correction=0).float().clamp_min(1e-6),cache['feature_std'],'normalization/std',1e-6); del zz
 require(tuple(cache['compartment_order'])==COMPS and cache['normalization']==run['normalization']==cp['normalization']==feat['normalization'],'decoder/order-normalizer')
 require(torch.equal(cache['decoder_weight'],cp['state_dict']['volume_head.weight']) and torch.equal(cache['decoder_bias'],cp['state_dict']['volume_head.bias']),'decoder/frozen-weights')
 pooled=raw.mean((-3,-2,-1)); norm=cache['normalization']; decoded=F.linear(pooled,cache['decoder_weight'],cache['decoder_bias']).numpy()*np.asarray(norm['scale'])+np.asarray(norm['mean'])
 close(pooled.numpy(),np.stack([feat['features'][v] for v in cache['visit_ids']]),'reconstruction/saved-pooled',1e-4)
 latest=lambda rr: decoded[[ix[r['cutoff_visit_id']] for r in rr]]
 truth=lambda rr: np.array([r['truth'] for r in rr]); ids=lambda rr: np.array([r['patient_id'] for r in rr]); controls=load(OUT/'controls.pt'); empirical=controls['E']; source_ids=ids(src); changes=truth(src)-latest(src)
 close(empirical.changes,changes,'E/source-changes',1e-6); require(tuple(empirical.patient_ids)==tuple(source_ids) and empirical.condition=='gaussian_observation','E/source-law')
 rms=np.sqrt(pmean((truth(src)-latest(src))**2,source_ids)); close(rms,controls['P'].residual_rms,'P/source-rms',1e-6)
 pgrid=[float(pmean(crps(latest(val),np.sqrt((k*rms)**2+.0025),truth(val)),ids(val)).mean()) for k in GRID]
 close(pgrid,controls['P_validation_grid'],'P/validation-grid',1e-6); require(controls['P_calibration']==GRID[int(np.argmin(pgrid))],'P/validation-choice')
 egrid=np.stack([mixture(latest(val),truth(val),changes,source_ids,k)[0] for k in GRID]); close(egrid,controls['E_validation_grid'],'E/validation-grid',1e-6)
 require(controls['E_calibration']==GRID[int(np.argmin([pmean(s,ids(val)).mean() for s in egrid]))],'E/validation-choice')
 stdmaps=(raw-cache['feature_mean'][None,:,None,None,None])/cache['feature_std'][None,:,None,None,None]
 ridge=load(OUT/'ridge/ridge.pt'); receipt=ridge['receipt']; oof=np.load(OUT/'ridge/source_oof.npz',allow_pickle=False)
 require(receipt==J(OUT/'ridge/complete.json'),'ridge/receipt'); require(np.array_equal(oof['row_ids'],[r['row_id'] for r in src]) and np.array_equal(oof['patient_ids'],source_ids),'ridge/oof-identities'); close(oof['truth'],truth(src),'ridge/oof-truth')
 chosen=int(np.argmin(receipt['source_cv_crps'])); require(receipt['alpha_grid']==[.1,1.,10.,100.,1000.,10000.] and receipt['alpha']==receipt['alpha_grid'][chosen],'ridge/alpha-selection')
 require(oof['means'].shape==(6,len(src),3) and np.isfinite(oof['means']).all(),'ridge/oof-complete')
 r_rms=np.sqrt(pmean((truth(src)-oof['means'][chosen])**2,source_ids)); close(r_rms,ridge['residual_rms'],'ridge/oof-rms'); close(r_rms,receipt['residual_rms'],'ridge/receipt-rms')
 cv=np.zeros((len(src),3)); heldall=[]
 for fold in receipt['folds']:
  trainset,held=set(fold['fitting_patients']),set(fold['held_patients']); require(not trainset&held and trainset|held==set(source_ids),'ridge/grouped-fold'); heldall+=list(held)
  select=np.isin(source_ids,list(held)); noise=np.array(fold['training_residual_rms']); require(np.isfinite(noise).all() and (noise>=0).all(),'ridge/fold-noise'); cv[select]=crps(oof['means'][chosen,select],np.sqrt(noise**2+.0025),truth(src)[select])
 require(len(heldall)==len(set(heldall))==len(set(source_ids)) and len(receipt['folds'])==3,'ridge/fold-partition'); close(pmean(cv,source_ids).mean(),receipt['source_cv_crps'][chosen],'ridge/selected-cv-score')
 def rmean(rr):
  means=[]
  for r in rr:
   hist=[ix[v] for v in r['history_visit_ids']]; maps=torch.zeros((4,128,*raw.shape[-3:])); maps[-len(hist):]=stdmaps[hist]
   av=torch.zeros(4,4); av[-len(hist):]=cache['availability'][hist]
   cov=torch.cat((av.reshape(-1),torch.tensor(r['gap_log1p']),torch.tensor(r['history_mask'],dtype=torch.float32))).double()
   design=torch.cat((maps.double().mean((-3,-2,-1)).reshape(-1),cov,torch.ones(1,dtype=torch.float64)))
   current=stdmaps[ix[r['cutoff_visit_id']]].mean((-3,-2,-1)).double(); native=(current+design@ridge['beta']).float()*cache['feature_std']+cache['feature_mean']
   m=F.linear(native,cache['decoder_weight'],cache['decoder_bias']).numpy()*np.array(norm['scale'])+np.array(norm['mean']); means.append(m)
  return np.stack(means)
 r_val=rmean(val); rgrid=[float(pmean(crps(r_val,np.sqrt((k*r_rms)**2+.0025),truth(val)),ids(val)).mean()) for k in GRID]
 close(rgrid,receipt['validation_grid_crps'],'ridge/validation-grid',1e-4); require(receipt['calibration']==GRID[int(np.argmin(rgrid))],'ridge/validation-choice')
 neural={}; selections=[]; replay_count=0
 for seed in SEEDS:
  orderfile=OUT/f'seed{seed}/sample_order.npy'; orders=np.load(orderfile,allow_pickle=False)
  rng=np.random.default_rng(seed); sampled=np.stack([rng.choice(len(src),len(src),replace=True,p=weights(source_ids)) for _ in range(20)])
  require(np.array_equal(orders,sampled) and orders.shape==(20,262),'neural/patient-uniform-orders'); artifacts[f'seed{seed}/sample_order.npy']=sha(orderfile)
  torch.manual_seed(seed); init={}
  for name,module in [('reduce',torch.nn.Conv3d(128,16,1)),('gates',torch.nn.Conv3d(37,32,3,padding=1)),('candidate',torch.nn.Conv3d(37,16,3,padding=1)),('residual',torch.nn.Conv3d(16,128,1)),('scale',torch.nn.Linear(16,128))]:
   init.update({name+'.'+k:v.tolist() for k,v in module.state_dict().items()})
  ini=canonical(init); seedcomplete=J(OUT/f'seed{seed}/complete.json')
  for arm in NEURAL:
   checkpoint=load(OUT/f'seed{seed}/{arm}.pt'); rec=checkpoint['receipt']; neural[(arm,seed)]=checkpoint
   require(rec==J(OUT/f'seed{seed}/{arm}.json')==seedcomplete[arm],'neural/receipt-consistency')
   require(rec['seed']==seed and rec['arm']==arm and rec['initialization_sha256']==ini and rec['sample_order_sha256']==canonical(orders.tolist()),'neural/matched-initialization-orders')
   history=rec['history']; require(len(history)==20 and [h['epoch'] for h in history]==list(range(1,21)),'neural/20-epochs')
   best=int(np.argmin([h['validation_crps'] for h in history])); require(rec['selected_epoch']==best+1 and rec['epochs']==20 and rec['updates']==5240,'neural/earliest-minimum-updates')
   close(rec['selection_validation_crps'],history[best]['validation_crps'],'neural/selected-score'); close(rec['validation_grid_crps'][1],rec['selection_validation_crps'],'neural/uncalibrated-grid')
   require(rec['calibration']==GRID[int(np.argmin(rec['validation_grid_crps']))],'neural/validation-calibration')
   last=load(OUT/f'seed{seed}/{arm}-last.pt'); partial=load(OUT/f'seed{seed}/{arm}-partial.pt')
   require(last['epoch']==20 and 'optimizer_state_dict' in last,'neural/optimizer-final')
   opt=last['optimizer_state_dict']; require(all(float(s['step'])==5240 for s in opt['state'].values()),'neural/optimizer-update-steps'); require(all(g['lr']==3e-4 and g['weight_decay']==1e-4 for g in opt['param_groups']),'neural/optimizer-settings')
   require(all(torch.equal(v,partial['state_dict'][k]) for k,v in checkpoint['state_dict'].items()),'neural/partial-selected-weights')
   require(J(OUT/f'seed{seed}/{arm}-progress.json')['history']==history,'neural/progress-history')
   selections.append({'arm':arm,'seed':seed,'selected_epoch':rec['selected_epoch'],'calibration':rec['calibration']})
   for name in ('primary','single'):
    if not cohorts[name]: continue
    r=cohorts[name][0]; hist=[ix[v] for v in r['history_visit_ids']]; maps=torch.zeros(1,4,128,*raw.shape[-3:]); maps[0,-len(hist):]=stdmaps[hist]
    av=torch.zeros(1,4,4); av[0,-len(hist):]=cache['availability'][hist]
    mu,sigma=forward(maps,av,torch.tensor([r['gap_log1p']]),torch.tensor([r['history_mask']]),checkpoint['state_dict'],arm)
    native=mu.mean((-3,-2,-1))*cache['feature_std']+cache['feature_mean']; m=F.linear(native,cache['decoder_weight'],cache['decoder_bias']).numpy()*np.array(norm['scale'])+np.array(norm['mean'])
    effective=cache['decoder_weight']*torch.tensor(norm['scale'])[:,None]; var=(effective[None]**2*(sigma*cache['feature_std'])[:,None,:]**2).sum(2).numpy()
    s=np.sqrt(var*rec['calibration']**2+.0025); saved=bundle['cohorts'][name][arm][SEEDS.index(seed)]
    close(m[0],saved['mean'][0],'forward/mean',1e-4); close(s[0],saved['scale'][0],'forward/scale',1e-4)
    target=stdmaps[ix[r['target_visit_id']]][None]; close(float((mu-target).square().mean()),saved['latent_mse'][0],'forward/latent-mse',1e-4); replay_count+=1
  guard()
 # All raw NPZ/PT bundles and independent Gaussian/empirical marginal laws.
 for name,rr in cohorts.items():
  groups=bundle['cohorts'][name] if name!='nuisance' else {a+'_'+c:v for a,conditions in bundle['nuisance'].items() for c,v in conditions.items()}
  yy=truth(rr)
  for label,bb in groups.items():
   arm=label.split('_')[0]; expectedseeds=list(SEEDS) if arm in NEURAL else [None]
   require([b['seed'] for b in bb]==expectedseeds,'arrays/seed-labels'); require(np.array_equal(arrays[name+'_'+label+'_seed_ids'],[s if s is not None else -1 for s in expectedseeds]),'arrays/npz-seed-labels')
   for j,b in enumerate(bb):
    for k,v in b.items():
     if k=='seed': continue
     close(v,arrays[f'{name}_{label}_{j}_{k}'],'arrays/pt-npz/'+k,0.)
    if arm!='E':
     k=neural[(arm,SEEDS[j])]['receipt']['calibration'] if arm in NEURAL else controls['P_calibration'] if arm=='P' else receipt['calibration']
     close(crps(b['mean'],b['scale'],yy),b['crps'],'gaussian/calibrated')
     unc=np.sqrt(np.maximum(0,np.asarray(b['scale'])**2-.0025)/k**2+.0025); close(crps(b['mean'],unc,yy),b['uncalibrated_crps'],'gaussian/uncalibrated',1e-6)
     for lev in LEVELS:
      z=NormalDist().inv_cdf((1+lev)/2); close(b['mean']-z*b['scale'],b[f'lower_{lev}'],'gaussian/lower'); close(b['mean']+z*b['scale'],b[f'upper_{lev}'],'gaussian/upper')
    else:
     score,centers,w=mixture(latest(rr),yy,changes,source_ids,controls['E_calibration']); close(score,b['crps'],'E/assessment-crps',1e-6); close(mixture(latest(rr),yy,changes,source_ids,1.)[0],b['uncalibrated_crps'],'E/uncalibrated',1e-6); close(np.einsum('nsc,s->nc',centers,w),b['mean'],'E/mean',1e-6)
     for lev in LEVELS:
      for side,p in [('lower',(1-lev)/2),('upper',(1+lev)/2)]:
       low=centers.min(1)-.6; high=centers.max(1)+.6
       for _ in range(60):
        mid=(low+high)/2; cdf=(ndtr((mid[:,None,:]-centers)/.05)*w[None,:,None]).sum(1); low=np.where(cdf<p,mid,low); high=np.where(cdf>=p,mid,high)
       close((low+high)/2,b[f'{side}_{lev}'],'E/interval',1e-6)
    # Replace copied CRPS with independently recomputed scores for every subsequent summary.
    b['crps']=crps(b['mean'],b['scale'],yy) if arm!='E' else score
   if name!='nuisance':
    if arm=='P': close(bb[0]['mean'],latest(rr),'P/persistence-mean',1e-6); close(bb[0]['scale'],np.broadcast_to(np.sqrt((controls['P_calibration']*rms)**2+.0025),yy.shape),'P/scale',1e-6)
    if arm=='R': close(bb[0]['mean'],rmean(rr),'ridge/assessment-mean',1e-4); close(bb[0]['scale'],np.broadcast_to(np.sqrt((receipt['calibration']*r_rms)**2+.0025),yy.shape),'ridge/scale')
   got=[summary(b,yy,np.array([r['cutoff_measured'] for r in rr]),ids(rr)) for b in bb]
   expected_summary=agg['arm_seed_summaries'][arm] if name=='primary' else agg['single_visit_companion'][arm] if name=='single' else agg['nuisance'][arm][label.split('_',1)[1]]['seed_summaries']
   tree(got,expected_summary,'summaries/'+name+'/'+label)
  guard()
 pp=cohorts['primary']; pi=ids(pp); yy=truth(pp); aa=bundle['cohorts']['primary']
 scores={a:np.mean([b['crps'] for b in aa[a]],0) for a in ARMS}; errors={a:np.mean([abs(b['mean']-yy) for b in aa[a]],0) for a in ARMS}
 arm_scores={a:{'crps':float(pmean(scores[a],pi).mean()),'mae':float(pmean(errors[a],pi).mean()),'compartment_crps':pmean(scores[a],pi).tolist(),'compartment_mae':pmean(errors[a],pi).tolist()} for a in ARMS}; tree(arm_scores,agg['arm_scores'],'aggregate/arms')
 contrasts={}
 for a in ('L','P','E','R','G'):
  v=boot(scores['H'],scores[a],pi,.99); v['unadjusted_ci']=boot(scores['H'],scores[a],pi)['ci']; v['relative_crps_reduction']=1-arm_scores['H']['crps']/arm_scores[a]['crps']; contrasts['H-'+a]=v
 tree(contrasts,agg['primary_comparisons'],'bootstrap/primary')
 margins={'H-'+a:boot(errors['H'],errors[a],pi) for a in ('L','P')}; tree(margins,agg['mae_margins'],'bootstrap/mae')
 secondary={}
 for a in ('P','E'):
  v=boot(scores['L'],scores[a],pi); v['relative_crps_reduction']=1-arm_scores['L']['crps']/arm_scores[a]['crps']; secondary['L-'+a]=v
 tree(secondary,agg['secondary_latest_comparisons'],'bootstrap/secondary')
 coverage=np.mean([pmean(((yy>=b['lower_0.8'])&(yy<=b['upper_0.8'])).astype(float),pi) for b in aa['H']],0); close(coverage,agg['H_80_coverage_by_compartment'],'coverage/H80')
 nr=cohorts['nuisance']; ny=truth(nr); ni=ids(nr); nuisance={}
 for arm in ('H','L'):
  nuisance[arm]={}; base=bundle['nuisance'][arm]['identity']; bc=np.mean([b['crps'] for b in base],0); be=np.mean([abs(b['mean']-ny) for b in base],0)
  for condition,bb in bundle['nuisance'][arm].items():
   cc=np.mean([b['crps'] for b in bb],0); ee=np.mean([abs(b['mean']-ny) for b in bb],0); drift=np.mean([abs(b['mean']-z['mean']) for b,z in zip(bb,base)],0)
   v={'crps_difference':boot(cc,bc,ni),'mae_difference':boot(ee,be,ni),'relative_crps_worsening':float(pmean(cc,ni).mean()/pmean(bc,ni).mean()-1),'mean_absolute_volume_drift':pmean(drift,ni).tolist(),'mean_absolute_volume_drift_ci':boot(drift,np.zeros_like(drift),ni),'coverage_difference_ci':{}}
   for lev in LEVELS:
    cov=np.mean([((ny>=b[f'lower_{lev}'])&(ny<=b[f'upper_{lev}'])).astype(float) for b in bb],0); bcov=np.mean([((ny>=b[f'lower_{lev}'])&(ny<=b[f'upper_{lev}'])).astype(float) for b in base],0)
    v['coverage_difference_ci'][str(lev)]=[boot(cov[:,c],bcov[:,c],ni) for c in range(3)]
   tree(v,agg['nuisance'][arm][condition],'nuisance/'+arm+'/'+condition); nuisance[arm][condition]=v
 criteria={'primary_contrasts':contrasts['H-L']['relative_crps_reduction']>=.05 and all(contrasts['H-'+a]['ci'][1]<0 for a in ('L','P','E','R')),'mae_noninferiority':all(v['ci'][1]<.05 for v in margins.values()),'coverage':bool(((coverage>=.7)&(coverage<=.9)).all()),'nuisance':all(nuisance['H'][c]['relative_crps_worsening']<=.1 and nuisance['H'][c]['mae_difference']['difference']<=.05 for c in ('intensity','resolution'))}
 tree(criteria,agg['criteria'],'decision/gates'); history=all(criteria.values()); spatial=history and contrasts['H-G']['ci'][1]<0; current=all(v['ci'][1]<0 and v['relative_crps_reduction']>=.05 for v in secondary.values())
 require(agg['history_benefit']==history and agg['earlier_spatial_detail_benefit']==spatial and agg['secondary_current_state_prediction_benefit']==current,'decision/benefits')
 require(agg['decision']==('ADDED_HISTORY_BENEFIT' if history else 'INCONCLUSIVE_FOR_FULL_CAPABILITY'),'decision/classification'); harms=[k for k,v in contrasts.items() if v['ci'][0]>0]; require(agg['demonstrated_harms']==harms,'decision/harms'); require(agg['automatic_followup'] is False and agg['reporting_approval'] is None and agg['stopping_decision']=='STOP_AFTER_ONE_ANALYSIS_REGARDLESS_OF_OUTCOME','decision/stopping-review-status')
 # Independent authority/journal and execution receipt inspection, without authority helper APIs.
 anchor=J(BASE/'receipts/continuation_authority.json'); ledger=J(LEDGER); events=[json.loads(line) for line in (BASE/'receipts/continuation_journal.jsonl').read_text().splitlines()]; prev='0'*64
 for i,e in enumerate(events):
  unsigned={k:v for k,v in e.items() if k!='event_sha256'}; require(e['sequence']==i and e['previous']==prev and canonical(unsigned)==e['event_sha256'] and canonical(e['snapshot'])==e['ledger_sha256'],'budget/journal-chain'); prev=e['event_sha256']
 require(events[-1]['ledger_sha256']==canonical(ledger) and ledger['authority_sha256']==canonical(anchor),'budget/current-binding')
 require(anchor['initial_ledger_sha256']==LEGACY and anchor['initial_snapshot']['setup_cpu_charge_seconds']==900 and anchor['initial_snapshot']['stages'][0]['cpu_seconds']==.844038823,'budget/legacy-retained')
 previous=ledger['stages'][:-1]; require([s['stage'] for s in previous]==['inventory','encode','lock','fit','evaluate','report'] and all(s['status']=='completed' for s in previous),'budget/six-complete-no-retry')
 require(ledger['stages'][-1]['stage']==STAGE and ledger['stages'][-1]['status']=='running' and not ledger['stages'][-1]['gpu'],'budget/review-reservation')
 require(ledger['started_at']=='2026-10-26T10:00:04+05:30' and ledger['deadline_unix']==datetime.fromisoformat('2026-10-28T10:00:04+05:30').timestamp() and ledger['gpu_seconds_limit']==28800 and ledger['cpu_seconds_limit']==57600,'budget/authorization')
 require(contract['budget_identity']['locked_journal_tip'] in {e['event_sha256'] for e in events} and agg['completed_budget_journal_tip'] in {e['event_sha256'] for e in events},'budget/locked-history')
 cpusec=sum(s['cpu_seconds'] for s in previous); gpusec=sum(s['gpu_seconds'] for s in previous); execution=J(BASE/'receipts/execution-complete-v1.json')
 close(cpusec,execution['resources']['scientific_cpu_seconds'],'budget/scientific-cpu'); close(gpusec,execution['resources']['gpu_seconds'],'budget/gpu'); close(cpusec+900,execution['resources']['total_charged_cpu_seconds'],'budget/total-cpu')
 require(all(s['started_wall_unix']+s['wall_seconds']<ledger['deadline_unix'] for s in previous) and cpusec+900<57600 and gpusec<28800,'budget/in-window')
 for p,h in execution['artifact_sha256'].items(): require(sha(OUT/p)==h,'execution/artifact-receipt')
 for p in ['authority-initialization.json','encoding-gate.json','fitting-gate.json','execution-complete-v1.json','execution-summary-v1.json']:
  artifacts['receipts/'+p]=sha(BASE/'receipts'/p)
 enc=J(BASE/'receipts/encoding-gate.json'); require(enc['cache_sha256']==artifacts['features.pt'] and enc['geometry_receipts_sha256']==contract['geometry_receipts_sha256'],'encoding/gate-binding'); tree(cache['reconstruction'],enc['reconstruction'],'encoding/reconstruction-receipt')
 guard(); require(time.monotonic()-started<300,'verification/wall-target')
 for p,h in artifacts.items(): require(sha((BASE/p) if p.startswith('receipts/') else OUT/p)==h,'integrity/after-review')
 result.update(status='PASS' if not discrepancies else 'CHANGES_REQUIRED',checks=checks,discrepancies=discrepancies,max_absolute_errors=max_errors,
  contract_digest=CONTRACT,aggregate_sha256=AGG,source_sha256=contract['code_sha256'],artifact_sha256=artifacts,
  units='log1p(mm3)',cohorts={k:{'patients':len(set(ids(v))),'examples':len(v)} for k,v in cohorts.items()},arm_scores=arm_scores,primary_comparisons=contrasts,mae_margins=margins,secondary_latest_comparisons=secondary,H_80_coverage_by_compartment=coverage.tolist(),nuisance=nuisance,criteria=criteria,decision=agg['decision'],demonstrated_harms=harms,selections=selections,
  fixed_weight_forward_replays=replay_count,forward_tolerance={'atol':1e-4,'rtol':1e-4},score_tolerance={'atol':1e-8,'rtol':1e-8},
  pre_review_resources={'scientific_cpu_seconds':cpusec,'setup_cpu_seconds':900.,'gpu_seconds':gpusec},
  executed_scope=['independent raw-array Gaussian and E-mixture CRPS/intervals','all arm/seed patient aggregation and bootstrap','all saved MAE/coverage/width/subset/nuisance/single summaries','eighteen first-row CPU fixed-weight neural forward replays','source normalization and original chronological rows','P/E validation choices and ridge validation/selected-CV/OOF-noise arithmetic','neural initialization/order regeneration and saved epoch/optimizer/selection receipts','contract/artifact/history bindings'],
  limits=['No real-image geometry or encoder extraction replay','No refit or full assessment rerun','Neural validation epoch/calibration choice verified from complete saved histories/grid receipts, not a new validation forward sweep','All-alpha ridge selection checked against recorded CV grid; selected-alpha CV recomputed from saved OOF means and recorded fold-training noise; no fold refit'])
 (HERE/'verification-staged.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
 if (HERE/'verification-receipt.json').exists(): raise FileExistsError('verification output already exists; no retry')
 try:
  with budget_stage(LEDGER,STAGE,gpu=False) as guard: audit(guard)
  final=J(LEDGER); entry=final['stages'][-1]
  require(entry['stage']==STAGE and entry['status']=='completed','verification/authoritative-completion')
  result.update(checker_sha256=sha(__file__),interpreter=sys.executable,torch_version=torch.__version__,numpy_version=np.__version__,stage=entry,
    final_ledger_sha256=sha(LEDGER),final_journal_sha256=sha(BASE/'receipts/continuation_journal.jsonl'),authority_anchor_sha256=sha(BASE/'receipts/continuation_authority.json'),staged_receipt_sha256=sha(HERE/'verification-staged.json'))
  (HERE/'verification-receipt.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
  print(json.dumps({'status':result['status'],'checks':result['checks'],'discrepancies':result['discrepancies'],'decision':result['decision'],'checker_sha256':result['checker_sha256'],'receipt_sha256':sha(HERE/'verification-receipt.json'),'stage_cpu_seconds':entry['cpu_seconds'],'stage_wall_seconds':entry['wall_seconds'],'artifactPaths':[str(Path(__file__).resolve()),str(HERE/'verification-receipt.json'),str(HERE/'verification-staged.json')]},sort_keys=True,indent=2))
 except BaseException as exc:
  (HERE/'failure.json').write_text(json.dumps({'status':'FAILED_STOP_NO_RETRY','checker_sha256':sha(__file__),'error_type':type(exc).__name__,'error':str(exc)},indent=2)+'\n')
  print('FAILED_STOP_NO_RETRY: '+type(exc).__name__+': '+str(exc),file=sys.stderr)
  raise
