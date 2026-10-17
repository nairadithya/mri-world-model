exec(open('/tmp/r35-closure-review/contrast_extra.py').read().split('# SAIL direct')[0])
for cohort,root in [('LUM',BASE),('SAI',TRANS)]:
 d=[]
 for s in (42,43,44):
  if cohort=='LUM':
   a=npz(root/f'seed{s}/paired/direct.predictions.npz'); b=npz(root/f'seed{s}/state/direct.predictions.npz')
  else:
   a=npz(root/f'seed{s}/paired/transfer.direct.predictions.npz'); b=npz(root/f'seed{s}/state/transfer.direct.predictions.npz')
  # paired model head vs separately trained state model differenced output, both same patient rows
  d.append((a,b))
 ap=[]; bp=[]
 for a,b in d:
  ap.append({k:float(np.abs(a['change_truth'][a['pair_patients']==k]-a['pair_head'][a['pair_patients']==k]).mean()) for k in sorted(set(a['pair_patients'].tolist()))})
  bp.append({k:float(np.abs(b['change_truth'][b['pair_patients']==k]-b['independent_decoder_difference'][b['pair_patients']==k]).mean()) for k in sorted(set(b['pair_patients'].tolist()))})
 keys=sorted(ap[0]); A={k:np.mean([x[k] for x in ap]) for k in keys};B={k:np.mean([x[k] for x in bp]) for k in keys}
 print(cohort,'paired',np.mean(list(A.values())),'independent_state_arm',np.mean(list(B.values())),'diff/ordinary/family12',bootdiff(A,B,12))
