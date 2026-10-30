import os,time,signal,resource,json,hashlib,math,re
from pathlib import Path
import numpy as np
start=time.process_time(); wall=time.monotonic()
resource.setrlimit(resource.RLIMIT_CPU,(240,240)); signal.alarm(270)
R=Path('/home/adithya/projects/world-model'); V=R/'outputs/r37-noise-ablation/v6'; B=R/'outputs/representation-comparison/retry-complete/unpacked'
def load(p): return json.loads(p.read_text())
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def canon(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def close(a,b,tol=2e-12):
 a=np.asarray(a); b=np.asarray(b); assert a.shape==b.shape; assert np.isfinite(a).all() and np.isfinite(b).all(); assert np.max(np.abs(a-b),initial=0)<tol
A=load(V/'run/aggregate.json'); C=load(V/'run/contract.json'); P=load(V/'run_contract.json'); F=load(V/'preflight_receipt.json'); L=load(V/'budget_ledger.json'); M=load(V/'run/private_row_manifest.json')
assert C==P and canon(C)==A['contract_digest']==F['contract_digest']=='1d0fc791c9340487670f951a88b5362095c9feb32e559a2e0347244667e0ba86'
assert digest(V/'run_contract.json')==F['contract_sha256']
assert C['eligible_row_manifest']==M['rows'] and canon(M['rows'])==C['eligible_row_manifest_sha256']==A['row_manifest_sha256']
assert digest(R/'scripts/run_spatial_noise_ablation.py')==C['runner_sha256']
for p,h in C['helper_hashes'].items(): assert digest(R/p)==h
paths={'prepared':B/'training_prepared.json','checkpoint':B/'results/comparison/seed42/state/best.pt','features':B/'results/comparison/seed42/state/features.pt','direct_predictions':B/'results/comparison/seed42/state/direct.predictions.npz','run':B/'results/comparison/run.json','inventory':R/'outputs/representation-comparison/inventory.json','anatomy_manifest':R/'outputs/anatomy_harmonization.json','chronology_manifest':R/'info/lumiere_task_manifest.json'}
assert C['input_hashes']==A['input_hashes']
for key,p in paths.items(): assert digest(p)==C['input_hashes'][key]
assert A['status']==L['status']=='completed' and F['status']=='PASS' and A['assessment_winner_selection'] is False
assert L['preflight_cpu_seconds']<=14400 and L['preflight_cpu_seconds']+L['run_cpu_seconds']<=57600
assert L['run_started_wall_unix']+L['run_wall_seconds']-L['started_wall_unix']<172800
splits={k:set(v) for k,v in C['splits'].items()}; keys=list(splits)
assert [len(splits[k]) for k in ('source52','validation13','unseen26')]==[52,13,26]
assert all(not splits[a]&splits[b] for i,a in enumerate(keys) for b in keys[i+1:])
orig=load(paths['run']); assert splits['source52']==set(orig['training_patients']) and splits['validation13']==set(orig['validation_patients']) and splits['unseen26']==set(orig['evaluation_patients'])
prep=load(paths['prepared']); pv={v['visit_id']:v for v in prep['visits']}; anat=load(paths['anatomy_manifest']); chronology=[v for v in anat['visits'] if v.get('cohort')=='LUMIERE']; assert len(chronology)==638
by={}
for v in chronology: by.setdefault(v['patient_id'],[]).append(v)
def order(v):
 m=re.fullmatch(r'week-(\d+)(?:-(\d+))?',v['visit']); assert m; return int(m[1]),int(m[2] or 0)
expected=[]; eligible_pairs={(p['patient_id'],p['source_visit'],p['target_visit']):p for p in prep['pairs']}
for patient in sorted(by):
 vv=sorted(by[patient],key=order)
 for i in range(len(vv)-1):
  cut,tar=vv[i:i+2]
  if cut['visit_id'] not in pv or tar['visit_id'] not in pv: continue
  s=i
  while s>0 and vv[s-1]['visit_id'] in pv: s-=1
  hh=vv[max(s,i-3):i+1]
  if any(order(hh[j])[0]<=order(hh[j-1])[0] for j in range(1,len(hh))): continue
  pair=eligible_pairs.get((patient,cut['visit'],tar['visit']))
  if pair is None: continue
  assert pair['usable'] and not pair['exclusions'] and pv[cut['visit_id']]['pregrid']==pv[tar['visit_id']]['pregrid']
  group=[k for k in keys if patient in splits[k]]
  if not group: continue
  expected.append({'row_id':cut['visit_id']+'->'+tar['visit_id'],'patient_id':patient,'cutoff_visit_id':cut['visit_id'],'target_visit_id':tar['visit_id'],'history_visit_ids':[v['visit_id'] for v in hh],'primary_history':len(hh)>=2,'split':group[0]})
assert expected==M['rows']
Z=np.load(V/'run/private_predictions_scores.npz',allow_pickle=False); comps=['necrotic_non_enhancing','enhancing','edema_flair']; assert comps==C['compartment_order']
for group,prefix,hist in [('source52','source',False),('validation13','validation',False),('unseen26','assessment',True),('unseen26','single_visit_assessment',False)]:
 rows=[r for r in expected if r['split']==group and (r['primary_history'] if hist else (not r['primary_history'] if prefix=='single_visit_assessment' else True))]
 assert [r['row_id'] for r in rows]==Z[prefix+'_ids'].tolist()
 assert [r['patient_id'] for r in rows]==Z[prefix+'_patient_ids'].tolist()
 if prefix!='single_visit_assessment': assert M[prefix+'_ids']==Z[prefix+'_ids'].tolist()
 truth=np.asarray([[math.log1p(pv[r['target_visit_id']]['values_mm3'][c]) for c in comps] for r in rows]); close(truth,Z[prefix+'_truth_log1p'])
assert len(Z['assessment_ids'])==A['query_rows']==75 and len(set(Z['assessment_patient_ids']))==A['query_patients']==18 and A['query_patients']>=15
sid=Z['source_patient_ids']; w=np.asarray([1/(len(set(sid))*np.count_nonzero(sid==p)) for p in sid]); close(w,Z['source_patient_weights']); assert abs(w.sum()-1)<1e-14
changes=Z['source_truth_log1p']-Z['source_latest_decoded']; close(changes,Z['source_changes']); w=w/w.sum(); m=w@changes
# Independent analytic E|Normal(a,s)| using erf, not project CDF or CRPS helper.
erf=np.vectorize(math.erf,otypes=[float])
def absolute_normal(a,s):
 return np.abs(a) if s==0 else s*math.sqrt(2/math.pi)*np.exp(-a*a/(2*s*s))+a*erf(a/(s*math.sqrt(2)))
def ref_scores(prefix,k,sd):
 shift=m+k*(changes-m)
 # Translation cancels in the pair term: compute once, not per query.
 pair=absolute_normal(shift[:,None,:]-shift[None,:,:],math.sqrt(2)*sd)
 correction=.5*np.einsum('i,j,ijc->c',w,w,pair)
 distance=Z[prefix+'_truth_log1p'][:,None,:]-Z[prefix+'_latest_decoded'][:,None,:]-shift[None,:,:]
 first=np.einsum('nsc,s->nc',absolute_normal(distance,sd),w)
 return first-correction

def patient_average(scores,ids): return float(np.mean([scores[ids==p].mean() for p in sorted(set(ids))]))
grid=[.5,1.,2.]; val={}; scores={}; maxerr=0.
for condition,sd in [('discrete',0.),('gaussian_observation',.05)]:
 val[condition]=[]; scores[condition]={}
 for j,k in enumerate(grid):
  vs=ref_scores('validation',k,sd); qs=ref_scores('assessment',k,sd)
  close(vs,Z['validation_grid_'+condition][j]); close(qs,Z['assessment_crps_'+condition+'_'+str(k)])
  maxerr=max(maxerr,float(np.max(np.abs(qs-Z['assessment_crps_'+condition+'_'+str(k)]))),float(np.max(np.abs(vs-Z['validation_grid_'+condition][j]))))
  val[condition].append(patient_average(vs,Z['validation_patient_ids'])); scores[condition][str(k)]=patient_average(qs,Z['assessment_patient_ids'])
  close(scores[condition][str(k)],A['patient_uniform_scores'][condition][str(k)])
 close(val[condition],A['validation_grid_scores'][condition]); selected=grid[int(np.argmin(val[condition]))]; assert selected==A['validation_selected_calibration'][condition]==1.
 single=ref_scores('single_visit_assessment',1.,sd); close(single,Z['single_visit_assessment_crps_'+condition]); close(patient_average(single,Z['single_visit_assessment_patient_ids']),A['single_visit_common_k1_scores'][condition])
# Paired unit is the patient after averaging examples and compartments; RNG stream continues for second contrast.
ids=Z['assessment_patient_ids']; patients=sorted(set(ids)); rng=np.random.default_rng(2026); paired={}; booterr=0.
delta=(ref_scores('assessment',1.,.05)-ref_scores('assessment',1.,0.)).mean(axis=1)
for label in ['common_k1','validation_selected']:
 close(delta,Z['paired_'+label+'_delta']); per={p:float(delta[ids==p].mean()) for p in patients}
 boots=np.asarray([np.mean([per[p] for p in rng.choice(patients,len(patients),replace=True)]) for _ in range(10000)])
 close(boots,Z['paired_'+label+'_bootstrap']); booterr=max(booterr,float(np.max(np.abs(boots-Z['paired_'+label+'_bootstrap']))))
 receipt={'mean':float(np.mean(list(per.values()))),'ci':np.quantile(boots,[.025,.975]).tolist()}; close(receipt['mean'],A['paired_'+label]['mean']); close(receipt['ci'],A['paired_'+label]['ci']); paired[label]=receipt
# Check saved direct frozen-decoder outputs wherever coverage permits; do not load encoder or MRI.
D=np.load(paths['direct_predictions'],allow_pickle=False); dd=dict(zip(D['visit_ids'].astype(str),D['state_prediction'])); direct_coverage={}
for prefix in ['source','validation','assessment','single_visit_assessment']:
 rows={r['row_id']:r for r in expected}; n=0
 for i,rid in enumerate(Z[prefix+'_ids']):
  vid=rows[str(rid)]['cutoff_visit_id']
  if vid in dd: close(Z[prefix+'_latest_decoded'][i],dd[vid],tol=1e-4); n+=1
 direct_coverage[prefix]=n
relative=100*(scores['discrete']['1.0']-scores['gaussian_observation']['1.0'])/scores['discrete']['1.0']
files=['scripts/run_spatial_noise_ablation.py','src/harness/analysis/spatial_dynamics_baselines.py','src/harness/analysis/spatial_dynamics_protocol.py','src/data/representation.py','info/METHODS.md','outputs/r37-noise-ablation/v6/run_contract.json','outputs/r37-noise-ablation/v6/preflight_receipt.json','outputs/r37-noise-ablation/v6/budget_ledger.json','outputs/r37-noise-ablation/v6/execution.log','outputs/r37-noise-ablation/v6/run/aggregate.json','outputs/r37-noise-ablation/v6/run/contract.json','outputs/r37-noise-ablation/v6/run/private_predictions_scores.npz','outputs/r37-noise-ablation/v6/run/private_row_manifest.json']
print(json.dumps({'status':'PASS','method':'reviewer-authored independent reference; no project helper imports','contract_digest':canon(C),'counts':C['counts'],'scored_rows':{'source':len(sid),'validation':len(Z['validation_ids']),'query':len(ids),'query_patients':len(patients),'single_visit':len(Z['single_visit_assessment_ids'])},'validation_scores':val,'assessment_scores':scores,'paired':paired,'relative_reduction_percent':relative,'max_crps_array_abs_error':maxerr,'max_bootstrap_abs_error':booterr,'direct_decoder_covered_rows':direct_coverage,'recorded_input_hashes_verified':C['input_hashes'],'sha256':{p:digest(R/p) for p in files},'ledger':{k:L[k] for k in ['status','preflight_cpu_seconds','run_cpu_seconds','run_wall_seconds','v1_expenditure']},'review_cpu_seconds':time.process_time()-start,'review_wall_seconds':time.monotonic()-wall},indent=2))
