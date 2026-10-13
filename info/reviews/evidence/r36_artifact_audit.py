import json, hashlib, sys, os, numpy as np, torch
from pathlib import Path
sys.path.insert(0,'/home/adithya/projects/world-model')
osroot=Path('/home/adithya/projects/world-model')
os.chdir(osroot)
from src.data.representation import digest,json_hash
from src.harness.analysis.delta_protocol import load_contract,CHECKPOINT_HASHES
from src.harness.analysis.delta_representation import load_final,load_probe
from src.harness.analysis.delta_evaluation import paired_patients,seed_mean,validate_nuisance
from src.harness.analysis.delta_representation import initialize
root=Path('outputs/r36-delta-objective/matched-final'); c=load_contract('outputs/r36-delta-objective/qa/r36_contract.json')
seeds=c['seeds']; arms=['A','B','C','D']; refs=['R35']; errs=[]; stats={}
def check(ok,msg):
 if not ok: errs.append(msg)
def loadj(p):return json.loads(p.read_text())
# base hash and contract inventories
check(len(c['visit_ids'])==599 and len(set(c['visit_ids']))==599,'contract visit inventory')
check(len(c['pair_rows'])==496 and len({r['row_id'] for r in c['pair_rows']})==496,'contract pair inventory')
query=[r for r in c['pair_rows'] if r['patient_id'] in c['evaluation_patients']]
query_ids={x['row_id'] for x in query}; query_patients={x['patient_id'] for x in query}
lumeta=json.load(open('outputs/r36-delta-objective/lumiere-tensors/prepared.json')); sailmeta=json.load(open('outputs/representation-comparison/sailor-tensors/prepared.json'))
lum_vis={v['visit_id'] for v in lumeta['visits'] if v['patient_id'] in c['evaluation_patients']}; lum_pairs={p['row_id'] for p in lumeta['pairs'] if p['patient_id'] in c['evaluation_patients']}
sail_vis={v['visit_id'] for v in sailmeta['visits']}; sail_pairs={p['row_id'] for p in sailmeta['pairs']}
check(len(query)==117 and len(query_patients)==23,'locked query pairs/patients')
# checkpoint/probe/histories
for seed in seeds:
 histories={}
 for arm in arms:
  d=root/f'seed{seed}'/arm; ck=d/'final.pt'; rec=loadj(d/'training.json'); art=torch.load(ck,map_location='cpu',weights_only=False)
  check(digest(ck)==(d/'final.sha256').read_text().strip(),f'final sha {seed}-{arm}')
  try: model,loaded=load_final(str(ck),c)
  except Exception as e: errs.append(f'load_final {seed}-{arm}: {type(e).__name__}'); loaded=art
  check(rec.get('status')=='complete' and len(rec.get('history',[]))==20 and rec.get('history',[])[-1].get('epoch')==20,f'final20 receipt {seed}-{arm}')
  check(rec.get('provenance',{}).get('arm')==arm and rec.get('provenance',{}).get('seed')==seed and rec.get('provenance',{}).get('contract_sha256')==json_hash(c) and rec.get('provenance',{}).get('tensor_manifest_sha256')==json_hash({'prepared_sha256':c['prepared_sha256'],'tensors':loadj(Path('outputs/r36-delta-objective/qa/audit.json'))['tensor_manifest']['tensors']}),f'training provenance {seed}-{arm}')
  check(art.get('seed')==seed and art.get('arm')==arm and art.get('epoch')==20,f'checkpoint identity {seed}-{arm}')
  histories[arm]=rec['history']
  for view in ('difference','target_only'):
   pd=d/view; pr=loadj(pd/'training.json'); pfile=pd/'probe.pt'; pa=torch.load(pfile,map_location='cpu',weights_only=False)
   check(digest(pfile)==(pd/'probe.sha256').read_text().strip(),f'probe sha {seed}-{arm}-{view}')
   try: load_probe(str(pfile),str(ck),c,view)
   except Exception as e: errs.append(f'load_probe {seed}-{arm}-{view}: {type(e).__name__}')
   check(pr.get('status')=='complete' and len(pr.get('history',[]))==20 and pr['history'][-1].get('epoch')==20,f'probe final20 {seed}-{arm}-{view}')
   prov=pr['provenance']; check(prov.get('arm')==arm and prov.get('seed')==seed and prov.get('view')==view and prov.get('checkpoint_sha256')==digest(ck) and prov.get('contract_sha256')==json_hash(c),f'probe provenance {seed}-{arm}-{view}')
   check(all(h['updates']==len(c['pair_rows'])-sum(1 for r in c['pair_rows'] if r['patient_id'] not in c['training_patients']) for h in pr['history']),f'probe source52 rows {seed}-{arm}-{view}')
  # nuisance hash, rows and provenance validated against evaluation at matched row IDs
  n=loadj(d/'nuisance.json'); ev=loadj(d/'lumiere.json'); expvis={}
  for s in ev['states']: expvis.setdefault(s['patient_id'],s['visit_id'])
  expected={(p,v,k) for p,v in expvis.items() for k in (*c['nuisance_training'],c['held_out_nuisance'])}
  observed=[(r['patient_id'],r['visit_id'],r['kind']) for r in n['rows']]
  check(len(observed)==len(set(observed)) and set(observed)==expected and len(n['rows'])==130,f'nuisance coverage {seed}-{arm}')
  manifest=json_hash({'prepared_sha256':c['prepared_sha256'],'tensors':ev['provenance']['tensor_hashes']})
  try: validate_nuisance(n,ev,d,c,arm,seed)
  except Exception as e: errs.append(f'final nuisance validator {seed}-{arm}: {type(e).__name__}')
  check(n['provenance']['tensor_manifest_sha256']==manifest,f'nuisance tensor manifest {seed}-{arm}')
  check(n['provenance']['checkpoint_sha256']==digest(ck) and n['provenance']['probe_sha256']==digest(d/'difference/probe.pt') and n['provenance']['contract_sha256']==json_hash(c),f'nuisance checkpoint/probe binding {seed}-{arm}')
 # matched history hash/order across arms, epoch-by-epoch
 for a in arms[1:]:
  check([h['pair_order_sha256'] for h in histories[a]]==[h['pair_order_sha256'] for h in histories['A']],f'pair schedule {seed}-{a}')
  check([h['nuisance_order_sha256'] for h in histories[a]]==[h['nuisance_order_sha256'] for h in histories['A']],f'nuisance schedule {seed}-{a}')
# assessment matrices & independent measures
local={a:[] for a in arms}; state={a:[] for a in arms}; controls={a:{x:[] for x in ('target_only','independent_state')} for a in ('C','D')}; nmetric={a:{k:[] for k in ('noise','intensity','bias')} for a in ('C','D')}
assessment_count=0; ridge_count=0; rows_count=0; state_count=0; probe_receipts=0; probe_histories={}
for seed in seeds:
 for arm in arms+refs:
  d=root/f'seed{seed}'/arm
  if arm=='R35':
   ck=Path(f'outputs/representation-comparison/retry-complete/unpacked/results/comparison/seed{seed}/state/best.pt')
  else: ck=d/'final.pt'
  if arm=='R35':
   check(ck.exists(),f'R35 checkpoint exists {seed}')
   check(digest(ck)==CHECKPOINT_HASHES[seed],f'R35 initial checkpoint hash {seed}')
   try: initialize(str(ck),seed)
   except Exception as e: errs.append(f'R35 checkpoint strict graph load {seed}: {type(e).__name__}')
  ckh=digest(ck) if ck.exists() else ''
  for cohort in ('lumiere','sailor'):
   p=d/(cohort+'.json'); x=loadj(p); assessment_count+=1
   check(x['status']=='complete' and x['arm']==arm and x['seed']==seed and x['cohort'].upper()==cohort.upper(),f'assessment identity {seed}-{arm}-{cohort}')
   pr=x['provenance']; check(pr['contract_sha256']==json_hash(c) and pr['checkpoint_sha256']==ckh,f'assess binding {seed}-{arm}-{cohort}')
   feat=torch.load(d/f'{cohort}-features.pt',map_location='cpu',weights_only=False); fp=feat['provenance']
   check(pr['prepared_sha256']==fp['prepared_sha256'] and pr['tensor_hashes']==fp['tensor_hashes'] and pr['source_hashes']==fp['source_hashes'],f'assess feature/tensor/source provenance {seed}-{arm}-{cohort}')
   if arm!='R35': check(pr['source_hashes']==torch.load(d/'final.pt',map_location='cpu',weights_only=False)['provenance']['source_hashes'],f'assessment frozen graph hashes {seed}-{arm}-{cohort}')
   if cohort=='lumiere': check(pr['prepared_sha256']==c['prepared_sha256'],f'LUMIERE prepared contract {seed}-{arm}')
   else: check(pr['prepared_sha256']==digest('outputs/representation-comparison/sailor-tensors/prepared.json'),f'SAILOR prepared digest {seed}-{arm}')
   # view probe content binding
   view='difference' if arm!='R35' else 'difference'
   probe=d/'difference/probe.pt'; check(pr['probe_sha256']==digest(probe),f'assessment diff probe binding {seed}-{arm}-{cohort}')
   check(pr['target_probe_sha256']==digest(d/'target_only/probe.pt'),f'assessment target probe binding {seed}-{arm}-{cohort}')
   rows=x['rows']; states=x['states']; rows_count+=len(rows);state_count+=len(states)
   rid=[r['row_id'] for r in rows]; vids=[s['visit_id'] for s in states]
   check(len(rid)==len(set(rid)) and len(vids)==len(set(vids)),f'assess duplicate ids {seed}-{arm}-{cohort}')
   if cohort=='lumiere':
    check(set(rid)==lum_pairs==query_ids and len(rows)==117 and set(vids)==lum_vis and len(states)==149,f'LUMIERE pair/visit inventory {seed}-{arm}')
    lmrows={z['row_id']:z for z in lumeta['pairs'] if z['patient_id'] in c['evaluation_patients']}
    check(all(r['patient_id']==lmrows[r['row_id']]['patient_id'] and r['source_visit_id']==lmrows[r['row_id']]['source_visit_id'] and r['target_visit_id']==lmrows[r['row_id']]['target_visit_id'] for r in rows),f'LUMIERE row identity mapping {seed}-{arm}')
    lmvis={v['visit_id']:v['patient_id'] for v in lumeta['visits'] if v['patient_id'] in c['evaluation_patients']}
    check(all(st['patient_id']==lmvis[st['visit_id']] for st in states),f'LUMIERE visit identity mapping {seed}-{arm}')
    if arm in arms:
     def groupmean(vals):
      g={}
      for patient,value in vals:g.setdefault(patient,[]).append(float(value))
      return {p:float(np.mean(v)) for p,v in g.items()}
     local_raw=groupmean([(r['patient_id'],np.mean(r['frozen_probe']['balanced'])) for r in rows])
     state_raw=groupmean([(r['patient_id'],np.abs(np.asarray(r['prediction'])-np.asarray(r['truth'])).mean()) for r in states])
     check(set(local_raw)==set(x['local']['frozen_probe']['per_patient']) and all(np.isclose(local_raw[p],x['local']['frozen_probe']['per_patient'][p],atol=1e-7) for p in local_raw),f'raw local vs cache {seed}-{arm}')
     check(set(state_raw)==set(x['state']['per_patient_mae']) and all(np.isclose(state_raw[p],x['state']['per_patient_mae'][p],atol=1e-7) for p in state_raw),f'raw state vs cache {seed}-{arm}')
     local[arm].append(local_raw);state[arm].append(state_raw)
     if arm in ('C','D'):
      for co in controls[arm]:
       local_control=groupmean([(r['patient_id'],np.mean(r[co]['balanced'])) for r in rows])
       check(all(np.isclose(local_control[p],x['local'][co]['per_patient'][p],atol=1e-7) for p in local_control),f'raw control vs cache {seed}-{arm}-{co}')
       controls[arm][co].append(local_control)
   else:
    check(set(rid)==sail_pairs and set(vids)==sail_vis and len(rows)==242 and len(states)==269,f'SAILOR pair/visit inventory {seed}-{arm}')
    smap={z['row_id']:z for z in sailmeta['pairs']}
    check(all(r['patient_id']==smap[r['row_id']]['patient_id'] and r['source_visit_id']==smap[r['row_id']]['source_visit_id'] and r['target_visit_id']==smap[r['row_id']]['target_visit_id'] for r in rows),f'SAILOR row identity mapping {seed}-{arm}')
    svmap={v['visit_id']:v['patient_id'] for v in sailmeta['visits']}
    check(all(st['patient_id']==svmap[st['visit_id']] for st in states),f'SAILOR visit identity mapping {seed}-{arm}')
  # All 30 assessment probes, including frozen R35, with content locks and source52 histories.
  for view in ('difference','target_only'):
   pd=d/view; pfile=pd/'probe.pt'; trainj=loadj(pd/'training.json'); pa=torch.load(pfile,map_location='cpu',weights_only=False); probe_receipts+=1
   try: load_probe(str(pfile),str(ck),c,view)
   except Exception as e: errs.append(f'assessment load_probe {seed}-{arm}-{view}: {type(e).__name__}')
   check(digest(pfile)==(pd/'probe.sha256').read_text().strip(),f'assessment probe sha {seed}-{arm}-{view}')
   hp=trainj.get('history',[]); pp=trainj.get('provenance',{})
   check(trainj.get('status')=='complete' and len(hp)==20 and hp[-1].get('epoch')==20,f'assessment probe final20 {seed}-{arm}-{view}')
   check(pp.get('arm')==arm and pp.get('seed')==seed and pp.get('view')==view and pp.get('checkpoint_sha256')==ckh and pp.get('contract_sha256')==json_hash(c),f'assessment probe binding {seed}-{arm}-{view}')
   if arm!='R35': check(pp.get('source_hashes')==art['provenance'].get('source_hashes'),f'probe frozen graph hashes {seed}-{arm}-{view}')
   check(all(h.get('updates')==len([p for p in lumeta['pairs'] if p['patient_id'] in c['training_patients']]) for h in hp),f'assessment probe source52 updates {seed}-{arm}-{view}')
   check(pa.get('epoch')==20 and pa.get('provenance')==pp,f'assessment probe checkpoint receipt {seed}-{arm}-{view}')
   probe_histories[(seed,arm,view)]=[h['pair_order_sha256'] for h in hp]
  # All 15 nuisance companions, including R35 with its distinct frozen checkpoint path.
  nj=loadj(d/'nuisance.json'); evj=loadj(d/'lumiere.json'); nr=nj['rows']; evis={}
  for sr in evj['states']:evis.setdefault(sr['patient_id'],sr['visit_id'])
  nkinds=(*c['nuisance_training'],c['held_out_nuisance']); nexpect={(p,v,k) for p,v in evis.items() for k in nkinds}
  nobs=[(z['patient_id'],z['visit_id'],z['kind']) for z in nr]
  nmanifest=json_hash({'prepared_sha256':c['prepared_sha256'],'tensors':evj['provenance']['tensor_hashes']})
  check(nj.get('status')=='complete' and nj.get('arm')==arm and nj.get('seed')==seed and len(nobs)==len(set(nobs)) and set(nobs)==nexpect,f'all nuisance coverage {seed}-{arm}')
  check(nj['provenance'].get('contract_sha256')==json_hash(c) and nj['provenance'].get('prepared_sha256')==c['prepared_sha256'] and nj['provenance'].get('tensor_manifest_sha256')==nmanifest and nj['provenance'].get('checkpoint_sha256')==ckh and nj['provenance'].get('probe_sha256')==digest(d/'difference/probe.pt'),f'all nuisance provenance {seed}-{arm}')
  idbase={z['patient_id']:z['false_change'] for z in nr if z['kind']=='identity'}
  check(all(np.isfinite(z.get(k,np.nan)) for z in nr for k in ('false_change','state_drift','excess_false_change')) and all(z['false_change']>=0 and z['state_drift']>=0 and np.isclose(z['excess_false_change'],z['false_change']-idbase[z['patient_id']],atol=1e-12,rtol=0) for z in nr),f'all nuisance metric identities {seed}-{arm}')
  check(set(nj.get('conditions',{}))==set(nkinds) and all(np.isclose(nj['conditions'][k][m],np.mean([z[m] for z in nr if z['kind']==k]),atol=1e-12,rtol=0) for k in nkinds for m in ('false_change','state_drift','excess_false_change')),f'all nuisance summaries {seed}-{arm}')
  if arm in ('C','D'):
   n=loadj(d/'nuisance.json')
   for kind in nmetric[arm]:
    nmetric[arm][kind].append({r['patient_id']:r['excess_false_change'] for r in n['rows'] if r['kind']==kind})
  # ridge receipts and saved readout model replay (two cohorts)
  for cohort in ('lumiere','sailor'):
   rj=loadj(d/f'ridge-{cohort}.json'); modelpack=torch.load(d/f'ridge-{cohort}.readouts.pt',map_location='cpu',weights_only=False); arr=np.load(d/f'ridge-{cohort}.predictions.npz',allow_pickle=False); features=torch.load(d/f'{cohort}-features.pt',map_location='cpu',weights_only=False); ridge_count+=1
   rp=rj['provenance']; mp=modelpack['provenance']; check(modelpack['contract']=='r36-signed-feature-difference-v1' and rp==mp,f'ridge receipt equality {seed}-{arm}-{cohort}')
   check(rp['contract_sha256']==json_hash(c) and rp['checkpoint_sha256']==ckh and rp['query_cohort']==cohort.upper(),f'ridge provenance {seed}-{arm}-{cohort}')
   check(rp['training_features_sha256']==digest(d/'lumiere-features.pt') and rp['query_features_sha256']==digest(d/f'{cohort}-features.pt'),f'ridge features bytes {seed}-{arm}-{cohort}')
   check(set(rp['readout_training_patients'])==set(c['training_patients']) and len(rp['readout_training_patients'])==52,f'ridge source52 patients {seed}-{arm}-{cohort}')
   check(all(np.isfinite(np.asarray(modelpack['models'][k][field])).all() for k in ('state','delta') for field in ('mean','scale','coefficient','intercept')),f'ridge model finite {seed}-{arm}-{cohort}')
   models=modelpack['models']
   for kind,featname,truthname,groupname in [('state','state_prediction','state_truth','visit_ids'),('delta','delta_prediction','delta_truth','row_ids')]:
    ids=arr[groupname].astype(str).tolist(); vals=[]
    for ident in ids:
     if kind=='state': vals.append(features['features'][ident])
     else:
      # obtain paired feature deltas from assessment row source/target IDs
      rr=loadj(d/f'{cohort}.json')['rows']; rowmap={r['row_id']:r for r in rr}; row=rowmap[ident]
      vals.append(np.asarray(features['features'][row['target_visit_id']])-np.asarray(features['features'][row['source_visit_id']]))
    X=np.asarray(vals); m=models[kind]; pred=((X-np.asarray(m['mean']))/np.asarray(m['scale']))@np.asarray(m['coefficient']).T+np.asarray(m['intercept'])
    check(pred.shape==arr[featname].shape and np.allclose(pred,arr[featname],rtol=2e-6,atol=2e-6),f'ridge prediction replay {seed}-{arm}-{cohort}-{kind}')
    check(np.array_equal(arr[truthname],arr['state_truth'] if kind=='state' else arr['delta_truth']),f'ridge truth shape')
# Source-only probe epoch orders must be matched across every arm/reference for a seed/view.
for seed in seeds:
 for view in ('difference','target_only'):
  base=probe_histories[(seed,'A',view)]
  for arm in arms[1:]+refs:
   check(probe_histories[(seed,arm,view)]==base,f'matched probe epoch row hashes {seed}-{arm}-{view}')
check(probe_receipts==30,'total 30 probe artifacts')
# Verify all recorded graph hashes against the frozen R36 snapshot (evaluation implementation intentionally patched afterward).
for seed in seeds:
 for arm in arms:
  pr=loadj(root/f'seed{seed}'/arm/'training.json')['provenance']
  for name,h in pr['source_hashes'].items():
   if name=='src/harness/analysis/delta_evaluation.py':
    expected=hashlib.sha256(__import__('subprocess').check_output(['git','-C','/home/adithya/projects/world-model-r36','show','HEAD:'+name])).hexdigest()
   else: expected=digest(name)
   check(h==expected,f'frozen source hash {name}')
# independently derive named contrasts
lm={a:seed_mean(local[a]) for a in arms}; sm={a:seed_mean(state[a]) for a in arms}
comp={k:paired_patients(lm[a],lm[b]) for k,a,b in [('C-B','C','B'),('C-A','C','A'),('D-C','D','C')]}
sc=paired_patients(sm['C'],sm['A'])
cc={a:{co:paired_patients(lm[a],seed_mean(controls[a][co])) for co in controls[a]} for a in ('C','D')}
nm={a:{k:seed_mean(nmetric[a][k]) for k in nmetric[a]} for a in ('C','D')}
train={p:float(np.mean([np.mean([nm['D'][k][p]-nm['C'][k][p] for k in ('noise','intensity','bias')])])) for p in nm['C']['noise']}
dcstate=paired_patients(sm['D'],sm['C'])
report={'checks_failed':errs,'counts':{'checkpoints':12,'probes':probe_receipts,'probe trainings':probe_receipts,'arm assessments':15,'assessment artifacts':30,'lumiere_rows':117*15,'lumiere_states':149*15,'sailor_rows':242*15,'sailor_states':269*15,'nuisance_rows':130*15,'ridge_receipts':ridge_count},'contrasts':comp,'state_C-A':sc,'controls':cc,'nuisance_trained_mean':paired_patients({p:float(np.mean([nm['D'][k][p] for k in ('noise','intensity','bias')])) for p in nm['D']['noise']},{p:float(np.mean([nm['C'][k][p] for k in ('noise','intensity','bias')])) for p in nm['C']['noise']}),'state_D-C':dcstate}
Path('/tmp/r36-audit-results.json').write_text(json.dumps(report,indent=2,sort_keys=True,default=float))
print(json.dumps({'fail_count':len(errs),'failures':errs[:30],'counts':report['counts'],'contrasts':comp,'state_C-A':sc,'nuisance_trained_mean':report['nuisance_trained_mean'],'state_D-C':dcstate,'controls':cc},indent=2,default=float))
