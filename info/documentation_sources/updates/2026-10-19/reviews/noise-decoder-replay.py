import json,hashlib,time,resource,signal
from pathlib import Path
import numpy as np
import torch
resource.setrlimit(resource.RLIMIT_CPU,(230,230)); signal.alarm(250); t=time.process_time(); wall=time.monotonic()
R=Path('/home/adithya/projects/world-model'); B=R/'outputs/representation-comparison/retry-complete/unpacked'; V=R/'outputs/r37-noise-ablation/v6'; state=B/'results/comparison/seed42/state'
f=torch.load(state/'features.pt',map_location='cpu',weights_only=False); ck=torch.load(state/'best.pt',map_location='cpu',weights_only=False); original=json.loads((B/'results/comparison/run.json').read_text()); prep=json.loads((B/'training_prepared.json').read_text()); contract=json.loads((V/'run/contract.json').read_text()); man=json.loads((V/'run/private_row_manifest.json').read_text()); z=np.load(V/'run/private_predictions_scores.npz',allow_pickle=False)
assert f['checkpoint']==contract['checkpoint_sha256']; assert f['normalization']==ck['normalization']==original['normalization']==contract['normalization']; assert f['provenance']['source_hashes']==ck['provenance']['source_hashes']==contract['source_provenance']
for obj in [f,ck]:
 p=obj['provenance']; assert p['prepared_sha256']==original['prepared_sha256']==contract['input_hashes']['prepared']; assert p['source_data_hashes']==prep['source_hashes']; assert p['training_patients']==original['training_patients']; assert p['validation_patients']==original['validation_patients']
assert ck['seed']==ck['provenance']['seed']==42 and ck['provenance']['arm']=='state'; assert f['contract']==ck['contract']=='mri-only-state-observed-change-v1'
w=ck['state_dict']['volume_head.weight'].detach().cpu().numpy(); bias=ck['state_dict']['volume_head.bias'].detach().cpu().numpy(); mean=np.asarray(original['normalization']['mean'],float); scale=np.asarray(original['normalization']['scale'],float); rows={r['row_id']:r for r in man['rows']}; errors={}
assert w.shape==(3,128) and bias.shape==(3,) and np.all(scale>0)
for prefix in ['source','validation','assessment','single_visit_assessment']:
 ids=[rows[str(r)]['cutoff_visit_id'] for r in z[prefix+'_ids']]; x=np.asarray([np.asarray(f['features'][i],dtype=float) for i in ids]); decoded=(x@w.T+bias)*scale+mean
 error=float(np.max(np.abs(decoded-z[prefix+'_latest_decoded']),initial=0)); assert error<1e-11; errors[prefix]={'rows':len(ids),'max_absolute_error':error}
assert np.isfinite(w).all() and np.isfinite(bias).all()
print(json.dumps({'status':'PASS','method':'independent batched affine frozen-cache decode; no encoder forward','decoded_rows':errors,'cache_checkpoint_provenance_and_normalization':'PASS','review_cpu_seconds':time.process_time()-t,'review_wall_seconds':time.monotonic()-wall},indent=2))
