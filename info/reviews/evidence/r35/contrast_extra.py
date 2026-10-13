exec(open('/tmp/r35-closure-review/metrics_r35.py').read().split('# Main in-domain')[0])
def bootdiff(a,b,fam=1):
 d=np.array([a[k]-b[k] for k in a]); x=np.random.default_rng(2026).integers(0,len(d),(10000,len(d))); v=d[x].mean(1)
 return float(d.mean()),np.quantile(v,[.025,.975]).tolist(),np.quantile(v,[.05/(2*fam),1-.05/(2*fam)]).tolist()
# SAIL direct paired - independent state decoder; zero comparator
fs=[TRANS/f'seed{s}/paired/transfer.direct.predictions.npz' for s in (42,43,44)]
a=meanerr(fs,'pair_head','change_truth','pair_patients');b=meanerr(fs,'independent_decoder_difference','change_truth','pair_patients');z=meanerr(fs,'zero','change_truth','pair_patients')
print('SAIL direct pair-independent',bootdiff(a,b,12)); print('SAIL direct pair-zero',bootdiff(a,z,12))
# SAIL readout budgets: paired feature vs independent state difference, and state vs mean controls
for budget,prefix in ((26,'transfer'),(52,'transfer.source52')):
 fs=[TRANS/f'seed{s}/paired/{prefix}.predictions.npz' for s in (42,43,44)]
 a=meanerr(fs,'learned_pair','change_truth','pair_patients');b=meanerr(fs,'independent_state_difference','change_truth','pair_patients');z=meanerr(fs,'zero','change_truth','pair_patients')
 print('SAIL budget',budget,'pair-independent',bootdiff(a,b,12),'pair-zero',bootdiff(a,z,12))
 # state arm summary compared to saved train mean (both directly in transfer prediction)
 ss=[TRANS/f'seed{s}/state/{prefix}.predictions.npz' for s in (42,43,44)]
 a=meanerr(ss,'state_prediction','state_truth','state_patients');b=meanerr(ss,'state_mean','state_truth','state_patients')
 print('SAIL budget state-mean',budget, m(a),m(b),bootdiff(a,b,1))
