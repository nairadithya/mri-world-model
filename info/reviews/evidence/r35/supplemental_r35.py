"""Read-only supplemental assertions addressing reviewer audit gaps; aggregate-only output."""
import json, hashlib, pathlib, math, sys
import numpy as np, torch
ROOT=pathlib.Path('/home/adithya/projects/world-model')
R=ROOT/'outputs/representation-comparison'; BASE=R/'retry-complete/unpacked/results/comparison'; TRANS=R/'sailor-matrix'; FINAL=R/'r35-final'
errors=[]; counts={'nuisance_maps':0,'reference_budgets':0,'source_summary_digests':0,'query_summary_digests':0,'private_hashes':0}
def check(ok,msg):
 if not ok: errors.append(msg)
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def j(p): return json.loads(pathlib.Path(p).read_text())
# Exact participant membership is compared in-memory; never print identifiers.
run=j(BASE/'run.json'); evalset=set(run['evaluation_patients']); expected_conditions={'identity','intensity','noise','resolution','bias','readout_contract'}
summary=j(FINAL/'summary.json'); nuis_summary=summary['LUMIERE_nuisance']
for seed in (42,43,44):
 for arm in ('state','paired','later_only','frozen'):
  n=j(BASE/f'seed{seed}'/arm/'nuisance.json')
  check(set(n)==expected_conditions,f'{seed}/{arm}: unexpected/missing nuisance conditions')
  identity=n['identity']; ids=set(identity['per_patient'])
  check(ids==evalset and identity['n_patients']==len(evalset),f'{seed}/{arm}/identity exact coverage')
  for condition in ('identity','intensity','noise','resolution','bias'):
   counts['nuisance_maps']+=1; rec=n[condition]; rids=set(rec.get('per_patient',{}))
   check(rec.get('n_patients')==len(evalset) and rids==evalset,f'{seed}/{arm}/{condition}: exact evaluation patient membership')
   check(rids==ids,f'{seed}/{arm}/{condition}: condition patient set differs from identity')
   check(math.isfinite(rec.get('false_change_mae',float('nan'))) and math.isfinite(rec.get('state_readout_drift',float('nan'))),f'{seed}/{arm}/{condition}: nonfinite aggregate')
   values=list(rec['per_patient'].values())
   check(all(math.isfinite(x.get('false_change_mae',float('nan'))) and math.isfinite(x.get('state_readout_drift',float('nan'))) for x in values),f'{seed}/{arm}/{condition}: nonfinite per-patient metric')
   # Summary's seed-mean metric and patient-paired identity-relative excess are independently reconstructed.
   srec=nuis_summary[arm][condition]
   check(abs(np.mean([j(BASE/f'seed{s}/{arm}/nuisance.json')[condition]['false_change_mae'] for s in (42,43,44)])-srec['seed_mean_false_change_mae'])<1e-10,f'{arm}/{condition}: saved seed-mean false-change summary')
   paired_delta=[]
   for patient in sorted(evalset):
    means=[]
    for s in (42,43,44):
     sn=j(BASE/f'seed{s}/{arm}/nuisance.json')
     means.append(sn[condition]['per_patient'][patient]['false_change_mae']-sn['identity']['per_patient'][patient]['false_change_mae'])
    paired_delta.append(float(np.mean(means)))
   excess=float(np.mean(paired_delta))
   check(abs(excess-srec['false_change_excess_over_identity'])<1e-10,f'{arm}/{condition}: identity-relative summary arithmetic')
   counts['nuisance_maps']+=0
# Frozen reference is one model, five nuisance conditions; exact cohort membership and same map across conditions.
refn=j(R/'retry-complete/unpacked/results/reference/nuisance.json')
check(set(refn)==expected_conditions,'reference: unexpected/missing nuisance conditions')
refids=set(refn['identity']['per_patient']); check(refids==evalset and refn['identity']['n_patients']==len(evalset),'reference identity exact evaluation coverage')
for c in ('identity','intensity','noise','resolution','bias'):
 rec=refn[c]; check(set(rec['per_patient'])==evalset and rec['n_patients']==len(evalset),f'reference/{c}: exact patient coverage')
 check(set(rec['per_patient'])==refids,f'reference/{c}: condition differs from identity map')
 check(math.isfinite(rec['false_change_mae']) and math.isfinite(rec['state_readout_drift']),f'reference/{c}: nonfinite metrics')
 if 'frozen_reference' in nuis_summary:
  check(abs(rec['false_change_mae']-nuis_summary['frozen_reference'][c]['seed_mean_false_change_mae'])<1e-10,f'reference/{c}: saved summary metric')
  if 'frozen_reference' in nuis_summary:
   ex=float(np.mean([rec['per_patient'][pid]['false_change_mae']-refn['identity']['per_patient'][pid]['false_change_mae'] for pid in sorted(evalset)]))
   check(abs(ex-nuis_summary['frozen_reference'][c]['false_change_excess_over_identity'])<1e-10,f'reference/{c}: identity-relative arithmetic')
 counts['nuisance_maps']+=1
# Direct source prediction hashes and exact query artifact keys.
for seed in (42,43,44):
 for arm in ('state','paired','later_only','frozen'):
  key=f'seed{seed}/{arm}'; entry=summary['artifact_hashes'][key]
  source=BASE/f'seed{seed}'/arm/'direct.predictions.npz'
  query=TRANS/f'seed{seed}'/arm
  qfeat=query/'transfer.features.pt'; qpred=query/'transfer.direct.predictions.npz'
  check(entry.get('source_predictions')==sha(source),f'{key}: source_predictions digest')
  check(entry.get('query_features')==sha(qfeat),f'{key}: query_features digest')
  check(entry.get('query_predictions')==sha(qpred),f'{key}: query_predictions digest')
  counts['source_summary_digests']+=1; counts['query_summary_digests']+=2
# Frozen-reference provenance semantics: source readout fits on LUMIERE inventories, query embeddings on SAILOR prepared inventory.
trainmeta=j(R/'retry-complete/unpacked/training_prepared.json'); sailorprep=R/'sailor-tensors/prepared.json'
source26=set(trainmeta['protocol']['encoder_unseen']); source52=set(run['training_patients'])
querymeta=j(sailorprep); querypatients={v['patient_id'] for v in querymeta['visits']}
ref_features=torch.load(R/'sailor-reference/features.pt',map_location='cpu',weights_only=False)
qsha=sha(sailorprep); expected_cp='7186a6267d7aa42720381e6c2780f9c86fe20e71067074f8eacdf63d48199d0a'
check(ref_features['provenance'].get('prepared_sha256')==qsha,'sailor-reference/features prepared_sha256 binding')
check(ref_features['provenance'].get('checkpoint_sha256')==expected_cp,'sailor-reference/features checkpoint binding (recorded clean-reference digest)')
check(set(ref_features['features'])=={v['visit_id'] for v in querymeta['visits']},'sailor-reference/features visit-id coverage')
train_sha=sha(R/'retry-complete/unpacked/training_prepared.json')
for budget,expected_src in ((26,source26),(52,source52)):
 counts['reference_budgets']+=1
 stem=f'frozen-reference-source{budget}'
 readout=torch.load(TRANS/f'{stem}.readouts.pt',map_location='cpu',weights_only=False)
 report=j(TRANS/f'{stem}.json'); rp=report['provenance']; pp=readout['provenance']
 check(set(readout['training_patients'])==expected_src,f'{stem}: exact saved readout training membership')
 check(set(pp['readout_training_patients'])==expected_src and set(rp['readout_training_patients'])==expected_src,f'{stem}: provenance source membership')
 check(not expected_src & querypatients,f'{stem}: source/query patient disjointness')
 check(pp['readout_training_patients']==rp['readout_training_patients'],f'{stem}: report/readout source identities')
 check(pp['training']==rp['training'] and pp['query']==rp['query'],f'{stem}: JSON and PT provenance equality')
 # The source's training prepared digest must match the actual training-prepared metadata; query digest matches actual SAILOR prepared metadata.
 trainprov=pp['training']; queryprov=pp['query']
 check(trainprov.get('prepared_sha256')==train_sha,f'{stem}: training prepared metadata digest')
 check(queryprov.get('prepared_sha256')==qsha,f'{stem}: query prepared metadata digest')
 check(trainprov.get('checkpoint_sha256')==expected_cp and queryprov.get('checkpoint_sha256')==expected_cp,f'{stem}: reference checkpoint digests')
 check(queryprov.get('source_hashes')==ref_features['provenance'].get('source_hashes'),f'{stem}: query/reference feature source hashes')
 check(readout.get('contract')==ref_features.get('contract')==report.get('contract'),f'{stem}: representation contract')
 z=np.load(TRANS/f'{stem}.predictions.npz',allow_pickle=False)
 expected_vis=[v['visit_id'] for v in querymeta['visits']]; expected_rows=[p['row_id'] for p in querymeta['pairs']]
 check(np.array_equal(z['visit_ids'],expected_vis) and np.array_equal(z['row_ids'],expected_rows),f'{stem}: query prediction identities')
# Private frozen inputs recheck against original handoff, excluding public files that parent was authorized to edit.
handoff=j(pathlib.Path('/tmp/r35-closure-review/handoff.json'))
for p,h in handoff['input_sha256'].items():
 if p.startswith('outputs/representation-comparison/'):
  check(sha(ROOT/p)==h,f'private hash changed: {p}'); counts['private_hashes']+=1
print('REFERENCE_FEATURES_SHA256',sha(R/'sailor-reference/features.pt'))
print('COUNTS',json.dumps(counts,sort_keys=True))
print('ASSERTION_FAILURES',len(errors))
for e in errors: print('FAIL',e)
sys.exit(bool(errors))
