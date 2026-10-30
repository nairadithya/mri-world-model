#!/usr/bin/env python3
"""Run the bounded R37 E discrete-versus-Gaussian observation-noise ablation."""
from __future__ import annotations
import argparse, hashlib, json, math, os, re, signal, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
BASE=ROOT/'outputs/representation-comparison/retry-complete/unpacked'
OUT=ROOT/'outputs/r37-noise-ablation/v6'
EXPECTED_CHECKPOINT='65a4c4db2ca1b44ebf29fa12659fbd09dea6ede606ce1d3279535144933a001a'
from src.data.representation import COMPARTMENTS
from src.harness.analysis.spatial_dynamics_baselines import CALIBRATION_GRID, OBSERVATION_SD, fit_empirical_condition_receipts, patient_uniform_crps, patient_weights
from src.harness.analysis.spatial_dynamics_protocol import ChronologyVisit, FullChronology, build_history_rows


def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
 return h.hexdigest()
def json_hash(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def week_order(visit):
 m=re.fullmatch(r'week-(\d+)(?:-(\d+))?',visit)
 if not m: raise ValueError(f'unrecognized visit ordering label: {visit!r}')
 return int(m.group(1)),int(m.group(2) or 0)
def gap_days(previous,current):
 weeks=week_order(current)[0]-week_order(previous)[0]
 return float(weeks*7) if weeks>0 else 0.
def rows_by_identity(rows):
 result={}
 for row in rows:
  key=(row['patient_id'],row['visit'])
  if key in result: raise ValueError(f'duplicate patient/visit identity: {key}')
  result[key]=row
 return result

def build_rows(manifest_visits,inventory_visits,prepared_visits,prepared_pairs):
 inventory=rows_by_identity(inventory_visits); prepared=rows_by_identity(prepared_visits)
 ids=[(v['patient_id'],v['visit']) for v in manifest_visits]
 if len(ids)!=len(set(ids)): raise ValueError('duplicate original chronology patient/visit identity')
 chronology_keys=set(ids)
 if not set(inventory)<=chronology_keys or not set(prepared)<=chronology_keys: raise ValueError('inventory/prepared visit outside original chronology')
 pairmap={}
 for p in prepared_pairs:
  key=(p['patient_id'],p['source_visit'],p['target_visit'])
  if key in pairmap: raise ValueError(f'duplicate prepared pair: {key}')
  pairmap[key]=p
 by_patient={}
 for v in manifest_visits: by_patient.setdefault(v['patient_id'],[]).append(v)
 chronology=[]; visit_by_id={v['visit_id']:v for v in manifest_visits}
 for patient in sorted(by_patient):
  visits=sorted(by_patient[patient],key=lambda v:week_order(v['visit']))
  for i,v in enumerate(visits):
   ident=(patient,v['visit']); inv=inventory.get(ident); use=prepared.get(ident)
   if use and (inv is None or use['visit_id']!=inv['visit_id'] or use['values_mm3']!=inv['values_mm3'] or use['mask_relative']!=inv['mask_relative']): raise ValueError(f'prepared/inventory measurement identity mismatch: {ident}')
   if i==0: gap=None
   else:
    gap=gap_days(visits[i-1]['visit'],v['visit'])
   chronology.append(ChronologyVisit(v['visit_id'],patient,use is not None,gap,(True,)*4 if use else (False,)*4,use))
 built=build_history_rows(FullChronology(tuple(chronology),True),max_visits=4); result=[]
 reasons={}
 def exclude(reason): reasons[reason]=reasons.get(reason,0)+1
 built_keys={(r.cutoff_visit_id,r.target_visit_id) for r in built}
 for patient, visits in by_patient.items():
  ordered=sorted(visits,key=lambda v:week_order(v['visit']))
  for i in range(len(ordered)-1):
   cutoff,target=ordered[i],ordered[i+1]; key=(cutoff['visit_id'],target['visit_id'])
   if key in built_keys: continue
   if cutoff['patient_id']!=target['patient_id'] or (patient,cutoff['visit']) not in prepared or (patient,target['visit']) not in prepared:
    exclude('unusable_or_missing_adjacent_endpoint'); continue
   start=i
   while start>0 and (patient,ordered[start-1]['visit']) in prepared: start-=1
   hist=ordered[max(start,i-3):i+1]
   gaps=[gap_days(hist[j-1]['visit'],hist[j]['visit']) for j in range(1,len(hist))]
   if any(g is None or not math.isfinite(g) or g<=0 for g in gaps): exclude('invalid_historical_gap'); continue
   exclude('history_protocol_exclusion')
 for row in built:
  cutoff=visit_by_id[row.cutoff_visit_id]['visit']; target_visit=visit_by_id[row.target_visit_id]['visit']
  pair=pairmap.get((row.patient_id,cutoff,target_visit))
  if pair is None: exclude('prepared_pair_missing'); continue
  if not pair.get('usable') or pair.get('exclusions'): raise ValueError(f'chronology-adjacent prepared pair is explicitly unusable: {row.row_id}')
  source=prepared.get((row.patient_id,cutoff)); target=prepared.get((row.patient_id,target_visit))
  if source is None or target is None: exclude('prepared_endpoint_missing'); continue
  if source['pregrid']!=target['pregrid']: raise ValueError(f'known prepared-pair geometry mismatch: {row.row_id}')
  result.append({'row_id':row.row_id,'patient_id':row.patient_id,'cutoff_visit_id':row.cutoff_visit_id,'target_visit_id':row.target_visit_id,'history_visit_ids':row.history_visit_ids,'primary_history':row.primary_history,'cutoff':source,'target':target})
 return result,reasons

def decode_features(features,checkpoint,normalization):
 if not isinstance(features,dict) or not features: raise ValueError('feature cache must map visit IDs to embeddings')
 state=checkpoint.get('state_dict',{}); w=state.get('volume_head.weight'); b=state.get('volume_head.bias')
 if w is None or b is None: raise ValueError('frozen checkpoint lacks volume head')
 w=w.detach().cpu().numpy(); b=b.detach().cpu().numpy(); mean=np.asarray(normalization['mean'],float); scale=np.asarray(normalization['scale'],float)
 if w.shape!=(3,128) or b.shape!=(3,) or mean.shape!=(3,) or scale.shape!=(3,) or not all(np.isfinite(x).all() for x in (w,b,mean,scale)) or (scale<=0).any(): raise ValueError('frozen decoder or normalization has invalid shape/values')
 return {k:(w@np.asarray(x,float)+b)*scale+mean for k,x in features.items() if np.asarray(x).shape==(128,) and np.isfinite(x).all()}

def _inventory_digest_matches(prepared_hash,inventory,raw_inventory):
 # R35 stores SHA256 of compact canonical inventory JSON; tolerate formatting relocation only.
 raw=hashlib.sha256(raw_inventory).hexdigest(); canonical=json_hash(inventory)
 return prepared_hash in {raw,canonical}

def load_inputs():
 files={'prepared':BASE/'training_prepared.json','checkpoint':BASE/'results/comparison/seed42/state/best.pt','features':BASE/'results/comparison/seed42/state/features.pt','direct_predictions':BASE/'results/comparison/seed42/state/direct.predictions.npz','run':BASE/'results/comparison/run.json','inventory':ROOT/'outputs/representation-comparison/inventory.json','anatomy_manifest':ROOT/'outputs/anatomy_harmonization.json','chronology_manifest':ROOT/'info/lumiere_task_manifest.json'}
 missing=[str(p) for p in files.values() if not p.is_file()]
 if missing: raise FileNotFoundError('missing required input(s): '+', '.join(missing))
 raw={k:p.read_bytes() for k,p in files.items()}; obj={k:json.loads(raw[k]) for k in ('prepared','run','inventory','anatomy_manifest','chronology_manifest')}
 if sha256(files['checkpoint'])!=EXPECTED_CHECKPOINT: raise ValueError('checkpoint SHA256 differs from frozen seed-42 checkpoint')
 run=obj['run']; prep=obj['prepared']; inventory=obj['inventory']
 if run.get('prepared_sha256')!=sha256(files['prepared']): raise ValueError('prepared metadata hash differs from frozen run receipt')
 if not _inventory_digest_matches(prep.get('inventory_sha256'),inventory,raw['inventory']): raise ValueError('prepared inventory_sha256 matches neither raw nor canonical inventory bytes')
 source_hashes=inventory.get('source_hashes',{})
 if prep.get('source_hashes')!=source_hashes: raise ValueError('prepared source_hashes differ from inventory')
 for key,path in [('manifest',files['anatomy_manifest'])]:
  if source_hashes.get(key)!=sha256(path): raise ValueError(f'inventory does not bind original {key} source')
 feat=torch.load(files['features'],map_location='cpu',weights_only=False); checkpoint=torch.load(files['checkpoint'],map_location='cpu',weights_only=False)
 if feat.get('checkpoint')!=EXPECTED_CHECKPOINT: raise ValueError('feature artifact does not bind frozen checkpoint')
 if feat.get('provenance',{}).get('prepared_sha256')!=run['prepared_sha256'] or checkpoint.get('provenance',{}).get('prepared_sha256')!=run['prepared_sha256']: raise ValueError('cache/checkpoint prepared provenance disagrees with run receipt')
 if feat.get('provenance',{}).get('source_data_hashes')!=source_hashes or checkpoint.get('provenance',{}).get('source_data_hashes')!=source_hashes: raise ValueError('cache/checkpoint source-data provenance disagrees with prepared inventory')
 if feat.get('provenance',{}).get('source_hashes')!=checkpoint.get('provenance',{}).get('source_hashes'): raise ValueError('feature cache/checkpoint code provenance differs')
 if feat.get('contract')!='mri-only-state-observed-change-v1' or checkpoint.get('contract')!='mri-only-state-observed-change-v1': raise ValueError('cache/checkpoint contract mismatch')
 if feat.get('normalization')!=run.get('normalization') or checkpoint.get('normalization')!=run.get('normalization'): raise ValueError('saved normalization differs across run/cache/checkpoint')
 prov=feat['provenance']; cp=checkpoint['provenance']
 if prov.get('training_patients')!=run['training_patients'] or prov.get('validation_patients')!=run['validation_patients'] or cp.get('training_patients')!=run['training_patients'] or cp.get('validation_patients')!=run['validation_patients']: raise ValueError('checkpoint/cache original source or validation membership differs from run')
 if cp.get('seed')!=42 or cp.get('arm')!='state' or checkpoint.get('seed')!=42: raise ValueError('wrong original R35 checkpoint membership/arm')
 prepared_ids=[v['visit_id'] for v in prep['visits']]
 if len(prepared_ids)!=len(set(prepared_ids)): raise ValueError('duplicate prepared visit IDs')
 inventory_ids=[v['visit_id'] for v in inventory['visits']]
 if len(inventory_ids)!=len(set(inventory_ids)) or set(inventory_ids)!=set(prepared_ids): raise ValueError('prepared visits do not exactly cover unique inventory visits')
 if set(feat['features'])!=set(prepared_ids): raise ValueError('feature cache visit IDs do not exactly match prepared visits')
 if set(prep.get('source_hashes',{}))!={'manifest','geometry','protocol'}: raise ValueError('prepared source-hash inventory malformed')
 original=[v for v in obj['anatomy_manifest']['visits'] if v.get('cohort')=='LUMIERE']
 if len(original)!=638: raise ValueError(f'expected 638 original LUMIERE chronology visits, found {len(original)}')
 disk=set(); roots=[ROOT/'data/lumiere_preprocessed',ROOT/'data/autoseg/vols/Imaging']
 existing=0
 for root in roots:
  if root.exists():
   existing+=1; disk.update((p.parent.name,p.name) for p in root.glob('Patient-*/week-*') if p.is_dir())
 manifest_keys={(v['patient_id'],v['visit']) for v in original}
 if not existing or not disk: raise ValueError('required prepared/source directory identity evidence is empty')
 if disk!=manifest_keys: raise ValueError(f'directory visit identities differ from original chronology: disk={len(disk)}, manifest={len(manifest_keys)}, missing={len(manifest_keys-disk)}, extra={len(disk-manifest_keys)}')
 if len({v['visit_id'] for v in original})!=len(original): raise ValueError('duplicate original chronology visit IDs')
 rows,exclusions=build_rows(original,inventory['visits'],prep['visits'],prep['pairs'])
 if not rows: raise ValueError('no chronology-adjacent rows have usable prepared pairs')
 predicted=decode_features(feat['features'],checkpoint,run['normalization'])
 if len(predicted)!=len(prepared_ids): raise ValueError('invalid embedding omitted during frozen decode')
 direct=np.load(files['direct_predictions'],allow_pickle=False); direct_ids=direct['visit_ids'].astype(str); direct_values=direct['state_prediction']
 if len(set(direct_ids))!=len(direct_ids) or direct_values.shape!=(len(direct_ids),3): raise ValueError('direct prediction identity table malformed')
 for vid,val in zip(direct_ids,direct_values):
  if vid not in predicted or not np.allclose(predicted[vid],val,atol=1e-5,rtol=1e-5): raise ValueError(f'cached frozen decoder disagrees with direct prediction for {vid}')
 return {'files':files,'objects':obj,'feature_obj':feat,'checkpoint':checkpoint,'predicted':predicted,'rows':rows,'row_exclusions':exclusions,'disk_identity_count':len(disk),'manifest_forecast_count':len(obj['chronology_manifest']['tasks']['forecast'])}

def row_manifest(rows,splits):
 return [{'row_id':r['row_id'],'patient_id':r['patient_id'],'cutoff_visit_id':r['cutoff_visit_id'],'target_visit_id':r['target_visit_id'],'history_visit_ids':list(r['history_visit_ids']),'primary_history':bool(r['primary_history']),'split':next(k for k,v in splits.items() if r['patient_id'] in v)} for r in rows if any(r['patient_id'] in v for v in splits.values())]

def make_contract(data):
 run=data['objects']['run']; splits={'source52':sorted(run['training_patients']),'validation13':sorted(run['validation_patients']),'unseen26':sorted(run['evaluation_patients'])}
 if tuple(map(len,splits.values()))!=(52,13,26): raise ValueError('frozen source/validation/unseen split sizes are not 52/13/26')
 ids={k:set(v) for k,v in splits.items()}
 if any(ids[a]&ids[b] for a in ids for b in ids if a<b): raise ValueError('patient splits overlap')
 selected=[r for r in data['rows'] if r['patient_id'] in set.union(*ids.values())]; hist=[r for r in selected if r['primary_history']]
 qp=sorted({r['patient_id'] for r in hist if r['patient_id'] in ids['unseen26']})
 if len(qp)<15: raise ValueError(f'only {len(qp)} unseen patients have eligible history; minimum is 15')
 manifest=row_manifest(selected,ids)
 return {'schema':'r37-e-noise-ablation-v6','runner_sha256':sha256(Path(__file__).resolve()),'input_hashes':{k:sha256(v) for k,v in data['files'].items()},'helper_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'src/harness/analysis/spatial_dynamics_baselines.py',ROOT/'src/harness/analysis/spatial_dynamics_protocol.py',ROOT/'src/data/representation.py']},'compartment_order':list(COMPARTMENTS),'source_provenance':data['feature_obj']['provenance']['source_hashes'],'source_paths':sorted(data['feature_obj']['provenance']['source_hashes']),'checkpoint_sha256':EXPECTED_CHECKPOINT,'splits':splits,'eligible_row_manifest':manifest,'eligible_row_manifest_sha256':json_hash(manifest),'counts':{'prepared_visits':len(data['objects']['prepared']['visits']),'original_visits':638,'eligible_rows':len(selected),'source_rows':sum(r['patient_id'] in ids['source52'] for r in selected),'validation_rows':sum(r['patient_id'] in ids['validation13'] for r in selected),'assessment_rows':sum(r['patient_id'] in ids['unseen26'] for r in selected),'assessment_history_patients':len(qp)},'assessment_history_patients':qp,'exclusion_counts':data['row_exclusions'],'normalization':data['objects']['run']['normalization'],'timing':'historical gaps must be positive; target gap is not an eligibility criterion; week-bin difference * 7 days; suffix only orders visits','geometry':'prepared/inventory identity and adjacent usable pair with equal verified pregrid required','decoder':'frozen cached embedding through checkpoint volume_head, then saved scale and mean; no encoder forward','budget':{'total_cpu_hours':16,'preflight_cpu_hours_max':4,'wall_hours':48,'automatic_retry':False,'v1_expenditure':'unavailable; v1 receipt has no resource ledger','superseded_contracts':['outputs/r37-noise-ablation/v2/run_contract.json','outputs/r37-noise-ablation/v3/run_contract.json'],'carryover_note':'measured v2/v3/v4/v5 CPU and wall expenditure is carried into the v6 ledger'},'protocol':{'conditions':['discrete','gaussian_observation'],'common_calibration':1.0,'validation_calibration_grid':list(CALIBRATION_GRID),'observation_sd':OBSERVATION_SD,'primary':'history rows (>=2 visits), patient-uniform mean CRPS in log1p(mm3)','bootstrap':{'replicates':10000,'seed':2026,'confidence':0.95}}}

def _write_json(path,value):
 path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
def _ledger_start(mode):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/'budget_ledger.json'
 if p.exists(): ledger=json.loads(p.read_text())
 else:
  prior_path=ROOT/'outputs/r37-noise-ablation/v5/budget_ledger.json'
  prior=json.loads(prior_path.read_text()) if prior_path.is_file() else None
  ledger={'version':3,'status':'started','started_wall_unix':prior.get('started_wall_unix',time.time()) if prior else time.time(),'started_cpu_seconds':time.process_time(),'preflight_cpu_seconds':prior.get('preflight_cpu_seconds',0.) if prior else 0.,'run_cpu_seconds':prior.get('run_cpu_seconds',0.) if prior else 0.,'mode':mode,'v1_expenditure':'unavailable; v1 receipt contains no resource ledger','v2_carryover_cpu_seconds':prior.get('preflight_cpu_seconds',0.)+prior.get('run_cpu_seconds',0.) if prior else 0.,'v2_carryover_wall_seconds':time.time()-prior.get('started_wall_unix',time.time()) if prior else 0.}; _write_json(p,ledger)
 if ledger.get('status')=='completed' or (mode=='preflight' and ledger.get('status')!='started') or (mode=='run' and ledger.get('status') not in ('preflight_complete','started')): raise FileExistsError('v2 budget ledger already exists for another/finished operation')
 if mode=='run': ledger.update(mode='run',status='running',run_started_wall_unix=time.time(),run_started_process_cpu=time.process_time(),reserved_cpu_seconds=16*3600-ledger.get('preflight_cpu_seconds',0)-ledger.get('run_cpu_seconds',0)); _write_json(p,ledger)
 return p,ledger
def _budget_guard(deadline,cpu_deadline):
 if time.monotonic()>deadline or time.process_time()>cpu_deadline: raise TimeoutError('R37 cumulative CPU or wall budget exhausted')
def _with_limits(cpu_seconds,wall_seconds,fn):
 old=signal.getsignal(signal.SIGALRM); old_cpu=signal.getsignal(signal.SIGXCPU)
 def alarm(_s,_f): raise TimeoutError('R37 wall-clock budget exhausted')
 def cpu_alarm(_s,_f): raise TimeoutError('R37 CPU budget exhausted')
 signal.signal(signal.SIGALRM,alarm); signal.signal(signal.SIGXCPU,cpu_alarm); signal.alarm(max(1,int(wall_seconds)))
 import resource
 old_limits=resource.getrlimit(resource.RLIMIT_CPU)
 try:
  soft=max(1,math.ceil(time.process_time()+cpu_seconds))
  if old_limits[1]!=resource.RLIM_INFINITY: soft=min(soft,old_limits[1])
  resource.setrlimit(resource.RLIMIT_CPU,(soft,old_limits[1]))
  return fn()
 finally:
  resource.setrlimit(resource.RLIMIT_CPU,old_limits)
  signal.alarm(0); signal.signal(signal.SIGALRM,old); signal.signal(signal.SIGXCPU,old_cpu)
def preflight():
 ledger_path,ledger=_ledger_start('preflight'); cpu0=time.process_time(); wall0=time.time(); deadline=time.monotonic()+min(4*3600,16*3600-ledger.get('preflight_cpu_seconds',0)-ledger.get('run_cpu_seconds',0))
 try:
  def work():
   if time.monotonic()>deadline: raise TimeoutError('preflight wall budget exhausted')
   data=load_inputs(); return make_contract(data)
  contract=_with_limits(max(1,min(4*3600,16*3600-ledger.get('preflight_cpu_seconds',0)-ledger.get('run_cpu_seconds',0))),max(1,48*3600-(time.time()-ledger['started_wall_unix'])),work)
  elapsed=time.process_time()-cpu0
  if elapsed>4*3600: raise TimeoutError('preflight exceeded four CPU-hours')
  ledger.update(status='preflight_complete',preflight_cpu_seconds=ledger['preflight_cpu_seconds']+elapsed,preflight_wall_seconds=time.time()-wall0); _write_json(ledger_path,ledger)
  return contract
 except BaseException as e:
  ledger.update(status='preflight_failed',preflight_cpu_seconds=ledger['preflight_cpu_seconds']+time.process_time()-cpu0,last_error=f'{type(e).__name__}: {e}'); _write_json(ledger_path,ledger); raise

def run(contract_digest):
 target=OUT/'run'
 if target.exists(): raise FileExistsError(f'refusing to overwrite existing run receipt: {target}')
 ledger_path,ledger=_ledger_start('run'); cpu0=time.process_time(); start=time.monotonic(); remaining=16*3600-ledger.get('preflight_cpu_seconds',0)-ledger.get('run_cpu_seconds',0)
 if remaining<=0: raise TimeoutError('cumulative 16 CPU-hour budget exhausted')
 deadline=start+max(0,48*3600-(time.time()-ledger['started_wall_unix']))
 def work():
  _budget_guard(deadline,cpu0+remaining)
  data=load_inputs(); contract=make_contract(data); digest=json_hash(contract)
  if digest!=contract_digest: raise ValueError(f'contract digest mismatch: supplied {contract_digest}, current {digest}')
  splits={k:set(v) for k,v in contract['splits'].items()}; rows=data['rows']
  def arrays(group,history_only=False):
   rr=[r for r in rows if r['patient_id'] in splits[group] and (not history_only or r['primary_history'])]
   latest=np.stack([data['predicted'][r['cutoff']['visit_id']] for r in rr])
   truth=np.stack([[math.log1p(r['target']['values_mm3'][k]) for k in COMPARTMENTS] for r in rr])
   return rr,latest,truth,np.array([r['patient_id'] for r in rr],object)
  sr,sd,sy,sp=arrays('source52'); vr,vd,vy,vp=arrays('validation13'); qr,qd,qy,qp=arrays('unseen26',True)
  if not len(sr) or not len(vr) or not len(qr): raise ValueError('empty source, validation, or assessment rows')
  receipts=fit_empirical_condition_receipts(sd,sy,sp,vd,vy,vp); scores={}; val_scores={}; models={}
  for condition,receipt in receipts.items():
   c=receipt['control']; models[condition]=c; val_scores[condition]=receipt['scores_by_scale']
   scores[condition]={}
   for scale in sorted(set(CALIBRATION_GRID)|{1.,receipt['calibration']}): scores[condition][str(scale)]=c.crps(qd,qy,scale)
  patients=sorted(set(qp)); rng=np.random.default_rng(2026); paired={}
  for label,ga,ds in [('common_k1',1.,1.),('validation_selected',receipts['gaussian_observation']['calibration'],receipts['discrete']['calibration'])]:
   delta=scores['gaussian_observation'][str(ga)].mean(axis=1)-scores['discrete'][str(ds)].mean(axis=1)
   pv={p:float(delta[qp==p].mean()) for p in patients}; boots=np.array([np.mean([pv[p] for p in rng.choice(patients,len(patients),replace=True)]) for _ in range(10000)])
   paired[label]={'delta':delta,'patient_values':pv,'bootstrap':boots,'mean':float(np.mean(list(pv.values()))),'ci':[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))]}
  summary={c:{s:patient_uniform_crps(v,qp) for s,v in vals.items()} for c,vals in scores.items()}
  sr0,sl,st,si=arrays('unseen26',False); single=[i for i,r in enumerate(sr0) if not r['primary_history']]; single_scores={}
  if single:
   for c,receipt in receipts.items(): single_scores[c]=patient_uniform_crps(receipt['control'].crps(sl[single],st[single],1.),si[single])
  target.mkdir(parents=True)
  manifest=contract['eligible_row_manifest']; lookup={r['row_id']:r for r in rows}
  arrays_save={'assessment_ids':np.asarray([r['row_id'] for r in qr],dtype='U'),'assessment_patient_ids':np.asarray(qp,dtype='U'),'assessment_latest_decoded':qd,'assessment_truth_log1p':qy,'validation_ids':np.asarray([r['row_id'] for r in vr],dtype='U'),'validation_patient_ids':np.asarray(vp,dtype='U'),'validation_latest_decoded':vd,'validation_truth_log1p':vy,'source_ids':np.asarray([r['row_id'] for r in sr],dtype='U'),'source_patient_ids':np.asarray(sp,dtype='U'),'source_patient_weights':patient_weights(sp),'source_latest_decoded':sd,'source_truth_log1p':sy,'source_changes':sy-sd}
  single_idx=[i for i,r in enumerate(sr0) if not r['primary_history']]
  arrays_save['single_visit_assessment_ids']=np.asarray([sr0[i]['row_id'] for i in single_idx],dtype='U')
  arrays_save['single_visit_assessment_patient_ids']=np.asarray(si[single_idx],dtype='U')
  arrays_save['single_visit_assessment_latest_decoded']=sl[single_idx]
  arrays_save['single_visit_assessment_truth_log1p']=st[single_idx]
  for c,receipt in receipts.items(): arrays_save[f'single_visit_assessment_crps_{c}']=receipt['control'].crps(sl[single_idx],st[single_idx],1.) if single_idx else np.empty((0,3))
  for c,v in val_scores.items(): arrays_save[f'validation_grid_{c}']=v
  for c,scales in scores.items():
   for s,a in scales.items(): arrays_save[f'assessment_crps_{c}_{s}']=a
  for label,d in paired.items(): arrays_save[f'paired_{label}_delta']=d['delta']; arrays_save[f'paired_{label}_bootstrap']=d['bootstrap']
  np.savez_compressed(target/'private_predictions_scores.npz',**arrays_save)
  _write_json(target/'private_row_manifest.json',{'rows':manifest,'excluded_row_counts':contract['exclusion_counts'],'source_ids':[r['row_id'] for r in sr],'validation_ids':[r['row_id'] for r in vr],'assessment_ids':[r['row_id'] for r in qr]})
  report={'contract_digest':digest,'status':'completed','primary_metric':'patient-uniform CRPS in log1p(mm3)','query_rows':len(qr),'query_patients':len(patients),'validation_selected_calibration':{c:r['calibration'] for c,r in receipts.items()},'validation_grid_scores':{c:[patient_uniform_crps(row,vp) for row in val_scores[c]] for c in val_scores},'patient_uniform_scores':summary,'single_visit_common_k1_scores':single_scores,'paired_common_k1':{k:v for k,v in paired['common_k1'].items() if k not in ('delta','bootstrap','patient_values')},'paired_validation_selected':{k:v for k,v in paired['validation_selected'].items() if k not in ('delta','bootstrap','patient_values')},'assessment_winner_selection':False,'input_hashes':contract['input_hashes'],'row_manifest_sha256':contract['eligible_row_manifest_sha256']}
  _write_json(target/'aggregate.json',report); _write_json(target/'contract.json',contract); return report
 try:
  limit=min(remaining,48*3600); result=_with_limits(limit,deadline-start,work); elapsed=time.process_time()-cpu0
  ledger.update(status='completed',run_cpu_seconds=ledger.get('run_cpu_seconds',0)+elapsed,run_wall_seconds=time.monotonic()-start); _write_json(ledger_path,ledger); return result
 except BaseException as e:
  ledger.update(status='run_failed',run_cpu_seconds=ledger.get('run_cpu_seconds',0)+time.process_time()-cpu0,last_error=f'{type(e).__name__}: {e}'); _write_json(ledger_path,ledger); raise

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__); g=p.add_mutually_exclusive_group(required=True); g.add_argument('--preflight',action='store_true'); g.add_argument('--run',action='store_true'); p.add_argument('--contract-digest'); a=p.parse_args(argv)
 try:
  if a.preflight:
   if OUT.exists(): raise FileExistsError(f'refusing to overwrite existing v2 receipt directory: {OUT}')
   contract=preflight(); digest=json_hash(contract); _write_json(OUT/'run_contract.json',contract); _write_json(OUT/'preflight_receipt.json',{'status':'PASS','contract_digest':digest,'contract_sha256':sha256(OUT/'run_contract.json')}); print(json.dumps({'status':'PASS','contract_digest':digest,'counts':contract['counts'],'output':str(OUT.relative_to(ROOT))},indent=2)); return 0
  if not a.contract_digest: p.error('--run requires --contract-digest')
  print(json.dumps(run(a.contract_digest),indent=2)); return 0
 except Exception as e: print(f'FAILED: {type(e).__name__}: {e}',file=sys.stderr); return 1
if __name__=='__main__': raise SystemExit(main())
