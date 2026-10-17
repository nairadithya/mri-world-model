exec(open('/tmp/r35-closure-review/audit_r35.py').read().split('# Validate saved report hashes')[0].split('# nuisance completeness')[0])
# Aggregate per-patient absolute errors by seed, then average seed errors patient-wise.
def pererr(z,pred,truth,groups):
 e=np.abs(z[truth]-z[pred]).mean(1); g=z[groups]; return {x:float(e[g==x].mean()) for x in sorted(set(g.tolist()))}
def meanerr(paths,pred,truth,groups):
 vals=[pererr(npz(p),pred,truth,groups) for p in paths]; keys=sorted(vals[0]); return {k:float(np.mean([v[k] for v in vals])) for k in keys}
def m(x): return float(np.mean(list(x.values())))
def paired(a,b):
 d=np.array([a[k]-b[k] for k in a]); rng=np.random.default_rng(2026); q=d[rng.integers(0,len(d),(10000,len(d)))].mean(1); return float(d.mean()),np.quantile(q,[.025,.975]).tolist(),np.quantile(q,[.05/24,.95/1 if False else 1-.05/24]).tolist()
def paths(cohort,arm,kind='direct'):
 if cohort=='LUM': return [BASE/f'seed{s}/{arm}/direct.predictions.npz' for s in (42,43,44)]
 return [TRANS/f'seed{s}/{arm}/transfer.direct.predictions.npz' for s in (42,43,44)]
# Main in-domain direct and baselines
p=paths('LUM','state'); state=meanerr(p,'state_prediction','state_truth','state_patients'); zero=meanerr(p,'state_mean','state_truth','state_patients')
print('LUM state direct vs run-train-mean',m(state),m(zero),paired(state,zero))
ref=R/'retry-complete/fresh-frozen-source52.predictions.npz'; rz=npz(ref); frozen=pererr(rz,'state_prediction','state_truth','state_patients');print('LUM state vs frozen source52',m(state),m(frozen),paired(state,frozen))
# Paired arm scores and baselines
pp=paths('LUM','paired');
for key in ('pair_head','zero','independent_decoder_difference'):
 e=meanerr(pp,key,'change_truth','pair_patients');print('LUM paired',key,m(e))
head=meanerr(pp,'pair_head','change_truth','pair_patients');
for arm,key in [('paired','zero'),('later_only','pair_head'),('paired','independent_decoder_difference')]:
 if arm=='paired': base=meanerr(pp,key,'change_truth','pair_patients')
 else: base=meanerr(paths('LUM','later_only'),'pair_head','change_truth','pair_patients')
 print('LUM contrast',arm,key,paired(head,base))
# SAILOR direct state and observed change
for arm in ('state','paired','later_only'):
 f=paths('SAI',arm); key='state_prediction' if arm=='state' else 'independent_state_difference'
 if arm=='state':
  a=meanerr(f,key,'state_truth','state_patients'); b=meanerr(f,'state_mean','state_truth','state_patients'); print('SAI state',m(a),'trainmean',m(b),paired(a,b))
 else:
  c='pair_head' if arm=='paired' else 'independent_decoder_difference'; x=meanerr(f,c,'change_truth','pair_patients'); print('SAI change',arm,m(x))
# Source budgets: state, paired, independent for all arms; paired vs zero and independent on paired
for budget,prefix in ((26,'transfer'),(52,'transfer.source52')):
 for arm in ('state','paired','later_only','frozen'):
  files=[TRANS/f'seed{s}/{arm}/{prefix}.predictions.npz' for s in (42,43,44)]
  z=meanerr(files,'state_prediction','state_truth','state_patients'); print('SAI ridge state',budget,arm,m(z))
  if arm=='paired':
   for key in ('learned_pair','independent_state_difference','zero'):
    x=meanerr(files,key,'change_truth','pair_patients');print('SAI ridge change',budget,key,m(x))
# nuisance resolution excess vs identity (patient/seed averaging before bootstrap)
for arm in ('state','paired','later_only','frozen'):
 diffs=[]
 for seed in (42,43,44):
  n=loadj(BASE/f'seed{seed}/{arm}/nuisance.json'); a=n['resolution']['per_patient']; b=n['identity']['per_patient']; diffs.append({k:a[k]['false_change_mae']-b[k]['false_change_mae'] for k in sorted(a)})
 patients=sorted(diffs[0]); d={k:float(np.mean([x[k] for x in diffs])) for k in patients}; print('nuisance excess',arm,m(d),paired(d,{k:0. for k in d}))
