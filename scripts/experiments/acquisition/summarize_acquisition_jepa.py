#!/usr/bin/env python3
"""Finish the fixed acquisition report from saved predictions, without inference."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import argparse
import json
from collections import Counter
from pathlib import Path
from scripts.experiments.acquisition.evaluate_acquisition_jepa import load_role, mean, paired
from src.patient_state.state import write_json


def scrub(value):
    if isinstance(value,dict):
        return {k:scrub(v) for k,v in value.items() if k!='patient_differences'}
    if isinstance(value,list):return [scrub(v) for v in value]
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    result=json.loads((root/'assessment.json').read_text());rows=load_role(root,'A')
    trajectories=json.loads((root/'assessment-patients.json').read_text())
    result['chosen_path_verification']={
        'targets':len(trajectories),
        'identical_fixed_direct_jepa_paths':sum(r['trajectories']['fixed']['actions']==r['trajectories']['direct']['actions']==r['trajectories']['jepa']['actions'] for r in trajectories),
        'paths':{m:dict(Counter(','.join(map(str,r['trajectories'][m]['actions'])) for r in trajectories)) for m in ('fixed','direct','jepa')},
        'sequence_index':{'0':'CT1','1':'T1','2':'T2','3':'FLAIR'},
    }
    def score(r,method):return r['subsets']['0123']['score'] if method=='new' else r['reference']['0123']
    native={'nonempty_dice':{m:mean(rows,lambda r:score(r,m)['nonempty_dice']) for m in ('new','reference')},'compartments':{}}
    for c in ('1','2','3','4'):
        present=[r for r in rows if score(r,'new')['anatomy'][c]['nonempty_dice'] is not None]
        empty=[r for r in rows if score(r,'new')['anatomy'][c]['nonempty_dice'] is None]
        native['compartments'][c]={
            'nonempty_targets':len(present),'nonempty_patients':len({r['patient'] for r in present}),
            'empty_targets':len(empty),'empty_patients':len({r['patient'] for r in empty}),
            'nonempty_dice':{m:mean(present,lambda r:score(r,m)['anatomy'][c]['nonempty_dice']) for m in ('new','reference')},
            'empty_fp_ml':{m:mean(empty,lambda r:score(r,m)['anatomy'][c]['empty_reference_fp_ml']) for m in ('new','reference')},
            'empty_fp_rate':{m:mean(empty,lambda r:float(score(r,m)['anatomy'][c]['empty_reference_fp_ml']>0)) for m in ('new','reference')},
            'all_target_volume_mae_ml':{m:mean(rows,lambda r:score(r,m)['anatomy'][c]['volume_error_ml']) for m in ('new','reference')},
            'dice_difference':paired(present,lambda r:score(r,'new')['anatomy'][c]['nonempty_dice']-score(r,'reference')['anatomy'][c]['nonempty_dice']),
            'empty_fp_difference':paired(empty,lambda r:score(r,'new')['anatomy'][c]['empty_reference_fp_ml']-score(r,'reference')['anatomy'][c]['empty_reference_fp_ml']),
        }
    result['all_four_absolute_anatomy']=native
    for panel,entry in result['fixed_panel_retention'].items():
        entry['additional_paired_differences']={k:paired(rows,lambda r:r['subsets'][panel]['score'][k]-r['reference'][panel][k]) for k in ('D','brain_flair_mae','gradient_flair_mae')}
    result['execution_measurements']={
        'role_total_seconds':{role:sum(r['timing']['visit_total_seconds'] for r in load_role(root,role)) for role in ('P','V','A')},
        'patient_mean_four_sequence_encoding_seconds':mean(rows,lambda r:r['timing']['four_sequence_encoding_seconds']),
        'patient_mean_all_four_readout_seconds':mean(rows,lambda r:r['subsets']['0123']['readout_seconds']),
        'patient_mean_T1_query_descriptor_seconds':mean(rows,lambda r:r['subsets']['1']['descriptor_query_seconds']),
        'patient_mean_state_bytes':{k:mean(rows,lambda r:r['subsets'][k]['state_bytes']) for k in ('1','01','013','0123')},
        'example_files_bytes':sum(p.stat().st_size for p in (root/'examples').rglob('*') if p.is_file()),
    }
    examples=json.loads((root/'examples/examples.json').read_text());verification=json.loads((root/'examples/verification.json').read_text())
    viewer=json.loads((root/'examples/viewer-verification.json').read_text());assert viewer['passed']
    result['examples']={'count':len(examples['examples']),'rule':examples['rule'],'different_case_exists':examples['different_case_exists'],
        'verification':verification,'viewer_verified':viewer['passed'],'viewer_checks':viewer['checks'],
        'recovered_from_host_oom':True,'timing_limitation':'Initial encoding and pre-interruption update timings were not persisted; one resumed actual update took '+str(examples['examples'][1]['actual_update_seconds'][0])+' seconds.'}
    result['execution_corrections']=['Corrected raw-versus-window-blended smoke assertion; serialized readouts remain exact',
        'Transferred fixed reference observer inputs to CUDA; no models, scores or policies changed',
        'Example batch exhausted host memory after one full example and two steps of the second; resumed only the missing final step with mapped saved-state storage']
    result['status']='completed_no_acquisition_benefit_anatomy_retention_not_established'
    result['interpretation']='No incremental adaptive sequence-choice benefit under this trained observer. Low oracle headroom is conditional on this observer and released offline reveal task, not a finding that MRI acquisition has no value. Image completion improved against the preserved reference; native all-four anatomy did not pass retention acceptance. Anatomy was statistically worse than reference, but the interval straddles the -0.05 margin: retention is not established, and inferiority beyond that margin is not established either. One seed, terminal selected checkpoint, reused development population; no causal JEPA training-effect or independent generalization claim.'
    write_json(root/'final-results-private.json',result)
    public=scrub(result)
    public.pop('full_state_bytes');public['entirely_empty_visit_count']=len(public.pop('entirely_empty_visits'))
    public['policy_selection']={m:[{'penalty':r['model']['penalty'],'V_primary':r['V_primary']} for r in candidates] for m,candidates in public['policy_selection'].items()}
    write_json(root/'final-results.json',public)
    print(json.dumps({'status':public['status'],'identical_paths':public['chosen_path_verification']['identical_fixed_direct_jepa_paths'],'native_dice':native['nonempty_dice'],'examples_verified':verification['passed']}))


if __name__=='__main__':main()
