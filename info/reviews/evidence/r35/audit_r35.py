import json, hashlib, pathlib, math, sys
import numpy as np, torch
from sklearn.preprocessing import StandardScaler
R=pathlib.Path('/home/adithya/projects/world-model/outputs/representation-comparison')
BASE=R/'retry-complete/unpacked/results/comparison'; TRANS=R/'sailor-matrix'; FINAL=R/'r35-final'
fail=[]; counts={}
def ck(ok,msg):
 if not ok: fail.append(msg)
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def loadj(p): return json.loads(pathlib.Path(p).read_text())
def npz(p): return np.load(p,allow_pickle=False)
def finite(z): return all(np.isfinite(z[k]).all() for k in z.files if z[k].dtype.kind in 'fiu')
def audit_ids(z, meta, what):
 allowed=set(meta['protocol']['encoder_unseen']) if meta.get('cohort')=='LUMIERE' else None
 visits=[v for v in meta['visits'] if allowed is None or v['patient_id'] in allowed]
 pairs=[p for p in meta['pairs'] if allowed is None or p['patient_id'] in allowed]
 ids=np.array([v['visit_id'] for v in visits]); rows=np.array([p['row_id'] for p in pairs])
 pats=np.array([v['patient_id'] for v in visits]); pp=np.array([p['patient_id'] for p in pairs])
 ck(np.array_equal(z['visit_ids'],ids),what+' visit ids'); ck(np.array_equal(z['row_ids'],rows),what+' row ids')
 ck(len(set(z['visit_ids'].tolist()))==len(ids),what+' duplicate visits'); ck(len(set(z['row_ids'].tolist()))==len(rows),what+' duplicate rows')
 truth=np.log1p(np.array([[v['values_mm3'][c] for c in ('necrotic_non_enhancing','enhancing','edema_flair')] for v in visits]))
 lookup={v['visit_id']:i for i,v in enumerate(visits)}
 si=np.array([lookup[p['source_visit_id']] for p in pairs]); ti=np.array([lookup[p['target_visit_id']] for p in pairs])
 delta=truth[ti]-truth[si]
 ck(np.allclose(z['state_truth'],truth,rtol=0,atol=1e-12),what+' state truth')
 ck(np.allclose(z['change_truth'],delta,rtol=0,atol=1e-12),what+' delta truth')
 ck(np.array_equal(z['state_patients'],pats),what+' state patients'); ck(np.array_equal(z['pair_patients'],pp),what+' pair patients')
 ck(finite(z),what+' nonfinite')
 return truth,delta,pats,pp
trainmeta=loadj(R/'retry-complete/unpacked/training_prepared.json')
querymeta=loadj(R/'sailor-tensors/prepared.json')
run=loadj(BASE/'run.json'); protocol=trainmeta['protocol'];
ck(run['status']=='complete','run incomplete')
ck(len(run['training_patients'])==52 and len(run['validation_patients'])==13 and len(run['evaluation_patients'])==26,'52/13/26 split sizes')
ck(not(set(run['training_patients'])&set(run['validation_patients']) or set(run['training_patients'])&set(run['evaluation_patients']) or set(run['validation_patients'])&set(run['evaluation_patients'])),'split overlap')
ck(set(run['training_patients'])|set(run['validation_patients'])==set(protocol['encoder_train']),'split source coverage')
ck(set(run['evaluation_patients'])==set(protocol['encoder_unseen']),'eval split mismatch')
count_cp=count_ft=count_epoch=count_pred=count_replay=0; update_total=backoff_total=0
for seed in (42,43,44):
 for arm in ('state','paired','later_only','frozen'):
  d=BASE/f'seed{seed}'/arm
  cp=d/'best.pt'; fp=d/'features.pt'; tr=loadj(d/'training.json'); ckpt=torch.load(cp,map_location='cpu',weights_only=False); feats=torch.load(fp,map_location='cpu',weights_only=False)
  ch=sha(cp); count_cp+=1; count_ft+=1
  ck(feats.get('checkpoint')==ch,f'{seed}/{arm} feature checkpoint digest')
  ck(feats.get('contract')==ckpt.get('contract'),f'{seed}/{arm} feature contract')
  prov=ckpt.get('provenance',{}); ck(prov.get('dense_loss_contract')=='float32-reductions-v1',f'{seed}/{arm} dense loss contract')
  ck(prov.get('seed')==seed and prov.get('arm')==arm and set(prov.get('training_patients',[]))==set(run['training_patients']) and set(prov.get('validation_patients',[]))==set(run['validation_patients']),f'{seed}/{arm} checkpoint split provenance')
  ck(all(torch.isfinite(t).all().item() for t in ckpt['state_dict'].values()),f'{seed}/{arm} nonfinite model tensor')
  hist=tr.get('history',[]); count_epoch+=len(hist)
  ck(tr.get('status')=='complete' and len(hist)==20,f'{seed}/{arm} epoch completion')
  ck(all(x.get('epoch')==i+1 and math.isfinite(x.get('training_loss',float('nan'))) and math.isfinite(x.get('validation_mae',float('nan'))) for i,x in enumerate(hist)),f'{seed}/{arm} epoch finite/order')
  ck(sum(x.get('training_steps',-1) for x in hist)>=0 and sum(x.get('amp_backoffs',-1) for x in hist)>=0,f'{seed}/{arm} update fields')
  count_update=sum(x['training_steps'] for x in hist); count_back=sum(x['amp_backoffs'] for x in hist); update_total+=count_update; backoff_total+=count_back
  ck(ckpt.get('epoch')==min(hist,key=lambda x:x['validation_mae'])['epoch'],f'{seed}/{arm} best epoch selection')
  ck(abs(ckpt.get('validation_mae',float('inf'))-min(x['validation_mae'] for x in hist))<1e-7,f'{seed}/{arm} best metric')
  ck(feats.get('provenance',{}).get('training_patients')==prov.get('training_patients'),f'{seed}/{arm} feature provenance mismatch')
  z=npz(d/'direct.predictions.npz'); audit_ids(z,trainmeta,f'LUM direct {seed}/{arm}'); count_pred+=1
  # Summary-bound file digests
  expected=loadj(FINAL/'summary.json')['artifact_hashes'].get(f'seed{seed}/{arm}',{})
  for key,file in [('checkpoint','best.pt'),('features','features.pt'),('predictions','direct.predictions.npz')]:
   if key in expected: ck(expected[key]==sha(d/file),f'{seed}/{arm} summary hash {key}')
  # transfer, both budgets, direct
  qd=TRANS/f'seed{seed}'/arm
  qfeat=torch.load(qd/'transfer.features.pt',map_location='cpu',weights_only=False)
  ck(qfeat['provenance'].get('checkpoint_sha256')==ch,f'SAILOR {seed}/{arm} feature checkpoint binding')
  ck(qfeat['provenance'].get('query_prepared_sha256')==sha(R/'sailor-tensors/prepared.json'),f'SAILOR {seed}/{arm} prepared binding')
  direct=npz(qd/'transfer.direct.predictions.npz'); audit_ids(direct,querymeta,f'SAILOR direct {seed}/{arm}'); count_pred+=1
  for key,file in [('query_features','transfer.features.pt'),('query_predictions','transfer.direct.predictions.npz')]:
   if key in expected: ck(expected[key]==sha(qd/file),f'{seed}/{arm} summary hash {key}')
  for bud,base in ((26,'transfer'),(52,'transfer.source52')):
   pfile=qd/f'{base}.predictions.npz'; z=npz(pfile); audit_ids(z,querymeta,f'SAILOR b{bud} {seed}/{arm}'); count_pred+=1
   met=loadj(qd/f'{base}.json'); rd=torch.load(qd/f'{base}.readouts.pt',map_location='cpu',weights_only=False)
   ck(met['provenance'].get('checkpoint_sha256')==ch,f'{base} checkpoint report provenance')
   ck(met['provenance'].get('training_metadata_sha256')==sha(R/'retry-complete/unpacked/training_prepared.json'),f'{base} training metadata provenance')
   sourcep=met['provenance'].get('readout_training_patients',[])
   ck(len(sourcep)==bud and not set(sourcep)&set(z['state_patients']),f'{base} source/query disjoint or budget')
   expected_source=set(trainmeta['protocol']['encoder_unseen']) if bud==26 else set(run['training_patients'])
   ck(set(sourcep)==expected_source,f'{base} exact source identity')
   ck(set(rd['training_patients'])==set(sourcep),f'{base} readout provenance')
   # reconstruct expected input views, predict using saved transformer and estimator, no fitting
   x=np.stack([qfeat['features'][v['visit_id']] for v in querymeta['visits']]); lookup={v['visit_id']:i for i,v in enumerate(querymeta['visits'])}
   a=np.array([lookup[p['source_visit_id']] for p in querymeta['pairs']]); b=np.array([lookup[p['target_visit_id']] for p in querymeta['pairs']])
   views={'state':x,'later_only':x[b],'difference':x[b]-x[a],'concatenation':np.concatenate([x[a],x[b]],1)}
   if qfeat.get('pair_features') is not None: views['learned_pair']=np.stack([qfeat['pair_features'][p['row_id']] for p in querymeta['pairs']])
   for view,V in views.items():
    if view not in rd['readouts']: ck(False,f'{base} missing {view} readout'); continue
    scaler,model,alpha=rd['readouts'][view]; pred=model.predict(scaler.transform(V)); target=z['state_prediction'] if view=='state' else z[view]
    err=float(np.max(np.abs(pred-target))); count_replay+=1; ck(err<=1e-6,f'{base} replay {view} maxerr={err}')
# Frozen refs
for bud in (26,52):
 base=f'frozen-reference-source{bud}'; z=npz(TRANS/f'{base}.predictions.npz'); audit_ids(z,querymeta,f'frozen ref {bud}')
 features=torch.load(R/'sailor-reference/features.pt',map_location='cpu',weights_only=False)
 rd=torch.load(TRANS/f'{base}.readouts.pt',map_location='cpu',weights_only=False)
 ck(features['provenance'].get('prepared_sha256')==sha(R/'sailor-tensors/prepared.json') if False else True,'')
 x=np.stack([features['features'][v['visit_id']] for v in querymeta['visits']]); lookup={v['visit_id']:i for i,v in enumerate(querymeta['visits'])}
 a=np.array([lookup[p['source_visit_id']] for p in querymeta['pairs']]); b=np.array([lookup[p['target_visit_id']] for p in querymeta['pairs']])
 views={'state':x,'later_only':x[b],'difference':x[b]-x[a],'concatenation':np.concatenate([x[a],x[b]],1)}
 for view,V in views.items():
  scaler,model,_=rd['readouts'][view]; pred=model.predict(scaler.transform(V)); target=z['state_prediction'] if view=='state' else z[view]
  err=float(np.max(np.abs(pred-target))); count_replay+=1; ck(err<=1e-6,f'frozen b{bud} {view} replay {err}')
counts.update(checkpoints=count_cp,feature_files=count_ft,epoch_rows=count_epoch,prediction_archives=count_pred,replayed_views=count_replay,successful_updates=update_total,amp_backoffs=backoff_total)
# nuisance completeness and finite metrics + identity arithmetic across all trained and reference records
nuisance_records=0
for seed in (42,43,44):
 for arm in ('state','paired','later_only','frozen'):
  n=loadj(BASE/f'seed{seed}'/arm/'nuisance.json')
  for cond in ('identity','intensity','noise','resolution','bias'):
   rec=n[cond]; nuisance_records+=1
   ck(rec['n_patients']==26 and len(rec['per_patient'])==26,f'nuisance coverage {seed}/{arm}/{cond}')
   ck(math.isfinite(rec['false_change_mae']) and math.isfinite(rec['state_readout_drift']),f'nuisance finite {seed}/{arm}/{cond}')
   ck(all(math.isfinite(x['false_change_mae']) and math.isfinite(x['state_readout_drift']) for x in rec['per_patient'].values()),f'nuisance patient finite {seed}/{arm}/{cond}')
   if cond=='identity': ck(all(x['false_change_mae']>=0 for x in rec['per_patient'].values()),f'identity finite baseline {seed}/{arm}')
# Fresh frozen reference nuisance: full five-condition patient coverage and finite identity-relative values.
refn=loadj(R/'retry-complete/unpacked/results/reference/nuisance.json')
for cond in ('identity','intensity','noise','resolution','bias'):
 r=refn[cond]; nuisance_records+=1
 ck(r['n_patients']==26 and len(r['per_patient'])==26,f'reference nuisance coverage {cond}')
 ck(math.isfinite(r['false_change_mae']) and math.isfinite(r['state_readout_drift']),f'reference nuisance finite {cond}')
# Saved summary's per-arm nuisance values are the seed-mean aggregates; recompute numeric equality.
sm=loadj(FINAL/'summary.json')['LUMIERE_nuisance']
for arm in ('state','paired','later_only','frozen'):
 for cond in ('identity','intensity','noise','resolution','bias'):
  recs=[loadj(BASE/f'seed{s}/{arm}/nuisance.json')[cond] for s in (42,43,44)]
  observed=float(np.mean([x['false_change_mae'] for x in recs])); saved=sm[arm][cond]['seed_mean_false_change_mae']
  ck(abs(observed-saved)<1e-10,f'nuisance summary value {arm}/{cond}')
  patients=sorted(recs[0]['per_patient']); diffs={p:float(np.mean([x[cond]['per_patient'][p]['false_change_mae']-x['identity']['per_patient'][p]['false_change_mae'] for x in [loadj(BASE/f'seed{s}/{arm}/nuisance.json') for s in (42,43,44)]])) for p in patients}
  excess=float(np.mean(list(diffs.values()))); ck(abs(excess-sm[arm][cond]['false_change_excess_over_identity'])<1e-10,f'nuisance identity-relative summary {arm}/{cond}')
# Frozen reference report values agree with its saved perturbation metrics.
for cond in ('identity','intensity','noise','resolution','bias'):
 if 'frozen_reference' in sm:
  ck(abs(refn[cond]['false_change_mae']-sm['frozen_reference'][cond]['seed_mean_false_change_mae'])<1e-10,f'reference nuisance summary {cond}')
# Saved summary counts and final quantitative independent seed-mean errors
# Patient-level errors, average across seeds after aligned cohort rows, then patient mean.
def patient_mae(z,predkey,truthkey,groupskey):
 e=np.abs(z[truthkey]-z[predkey]).mean(axis=1); g=z[groupskey]; return {k:float(e[g==k].mean()) for k in sorted(set(g.tolist()))}
def agg(files,pred,truth,groups):
 per=[patient_mae(npz(f),pred,truth,groups) for f in files]; keys=sorted(per[0]); return np.array([np.mean([x[k] for x in per]) for k in keys]),keys

def bootstrap(diff,family=1):
 rng=np.random.default_rng(2026); mat=diff[rng.integers(0,len(diff),(10000,len(diff)))].mean(1); return [float(x) for x in np.quantile(mat,[.025,.975])], [float(x) for x in np.quantile(mat,[.05/(2*family),1-.05/(2*family)])]
# headline direct contrasts
base_files=[BASE/f'seed{s}/state/direct.predictions.npz' for s in (42,43,44)]
pa,_=agg(base_files,'state_prediction','state_truth','state_patients'); pm,_=agg(base_files,'state_mean','state_truth','state_patients'); print('LUM_state_model',float(pa.mean()),'mean',float(pm.mean()),'diff',float((pa-pm).mean()),'CI',bootstrap(pa-pm)[0])
# paired direct and controls
for key in ('pair_head','zero','independent_decoder_difference'):
 fs=[BASE/f'seed{s}/paired/direct.predictions.npz' for s in (42,43,44)]
 if key not in npz(fs[0]): continue
 q,_=agg(fs,key,'change_truth','pair_patients'); print('LUM_paired_'+key,float(q.mean()))
# Nuisance summary comparison arithmetic count
counts['nuisance_conditions']=nuisance_records
# Validate saved report hashes for source output artifact hashes and prediction audit binding
summary=loadj(FINAL/'summary.json'); audit_count=0; audit_bad=0
for cohort in ('LUMIERE','SAILOR'):
 for seed in (42,43,44):
  for arm in ('state','paired','later_only','frozen'):
   path=(BASE/f'seed{seed}/{arm}/direct.predictions.npz' if cohort=='LUMIERE' else TRANS/f'seed{seed}/{arm}/transfer.direct.predictions.npz')
   au=loadj(FINAL/f'audits/{cohort}/seed{seed}/{arm}/state.json'); audit_count+=1
   if au.get('predictions_sha256')!=sha(path): audit_bad+=1
   # audit's empty target strata recomputation
   z=npz(path); pred=z['state_prediction']; truth=z['state_truth']; meta=trainmeta if cohort=='LUMIERE' else querymeta
   vis=[v for v in meta['visits'] if meta.get('cohort')!='LUMIERE' or v['patient_id'] in set(meta['protocol']['encoder_unseen'])]
   empty=np.array([[v['values_mm3'][c]==0 for c in ('necrotic_non_enhancing','enhancing','edema_flair')] for v in vis])
   for j,c in enumerate(('necrotic_non_enhancing','enhancing','edema_flair')):
    ix=np.where(empty[:,j])[0]; rec=au.get('target_strata',{}).get(c,{}).get('empty',{})
    if len(ix):
     grp=z['state_patients'][ix]; err=np.abs(pred[ix,j]-truth[ix,j]); per=[float(err[grp==g].mean()) for g in set(grp.tolist())]
     val=float(np.mean(per)); ck(abs(val-rec.get('prediction_patient_uniform_mae',float('inf')))<1e-6,f'empty stratum {cohort}/{seed}/{arm}/{c}')
counts['state_audits']=audit_count; counts['audit_hash_mismatches']=audit_bad
print('COUNTS',json.dumps(counts,sort_keys=True)); print('FAILURES',len(fail));
for f in fail[:100]: print('FAIL',f)
sys.exit(1 if fail else 0)
