#!/usr/bin/env python3
"""Attach amount-first samples to preserved observation/completion timelines."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import nibabel as nib
import torch
from src.patient_state.amount_first import LocationReadout
from src.patient_state.amount_first_inference import load_quantity,predict_amount_state
from src.patient_state.pipeline import save_nifti,volumes
from src.patient_state.state import PatientState,write_json,digest,LABELS


def copy_artifact(src,dst):
    # Mutable JSON receives a new inode; large immutable observations can share disk blocks.
    if str(src).endswith('.json'):return shutil.copy2(src,dst)
    os.link(src,dst);return dst


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4)
    result=json.loads((args.build/'results.json').read_text());cases=json.loads((args.build/'quantity-cases.json').read_text())
    observation=result['protocol']['observation_context']['observation_sha256']
    quantity=load_quantity(args.build/'quantity-terminal.pt',observation);location=None
    if result['selected_checkpoint']:
        location=LocationReadout().cuda().eval();location.load_state_dict(torch.load(args.build/'location-selected.pt',weights_only=True)['model'])
    shutil.copytree(args.source,args.out,copy_function=copy_artifact)
    labels={**LABELS,4:'resection_cavity'};verified=0;matched=0
    for path in (args.out/'predictive').glob('*/timeline.json'):
        timeline=json.loads(path.read_text());base=path.parent
        entries={v['visit']:v for v in timeline['visits']}
        # Preserve old forecast results on disk but remove their issued-forecast UI entries from this new demonstration.
        for entry in timeline['visits']:
            entry['forecasts']=[f for f in entry['forecasts'] if f['method']=='persistence']
            entry['previous_forecast_comparisons']=[c for c in entry['previous_forecast_comparisons'] if c['method'].startswith('persistence')]
        candidates=[r for r in cases if r['patient']==timeline['patient'] and r['cutoff']==timeline['visits'][0]['visit'] and r['target'] in entries]
        for record in candidates:
            entry=entries[record['cutoff']];target=entries[record['target']];h=record['horizon']
            statepath=base/entry['visit']/'forecast_context.pt'
            if not statepath.exists():raise ValueError('Preserved serialized cutoff context missing')
            state=PatientState.load(statepath,'cpu')
            pred=predict_amount_state(quantity,state,h,location)
            if pred['sample_counts'].tolist()!=record['sample_counts']:raise AssertionError('Timeline/evaluation quantity samples differ')
            relative=f"{entry['visit']}/amount-first-h{h}-sample0.nii.gz"
            save_nifti(base/relative,pred['segmentation'],state.metadata['affine'])
            measured=volumes(np.asanyarray(nib.load(base/relative).dataobj),state.metadata['affine'],labels)
            if not np.allclose(list(measured.values()),np.array(record['sample_counts'][0])/1000,rtol=0,atol=1e-10):raise AssertionError('Displayed map volumes differ')
            method=result['identity']+(' learned location' if location is not None else ' prior allocation (quantity gate failed)')
            f={'method':method,'horizon_nominal_weeks':h,'forecast_week':entry['nominal_week']+h,
                'segmentation':relative,'volumes_ml':measured,'distribution':pred['distribution'],
                'presence_probability':pred['presence_probability'],'representative_sample':0,
                'uncertainty_status':'64 joint samples; empirical 90% intervals; calibration not established. '+result['status'],
                'history_used':[entry['visit']],'quantity_model_id':quantity.model_id,
                'quantity_gate_passed':result['quantity_gate_passed'],'forecast_image_status':'Anatomy overlay on observed cutoff FLAIR; no future-image synthesis',
                'source_state':str(statepath.relative_to(base))}
            entry['forecasts'].append(f)
            ref=target['reference_anatomy'];truth=np.array(list(ref['volumes_ml'].values()))
            metric=record['methods']['amount_first']
            target['previous_forecast_comparisons'].append({'issued_at':entry['visit'],'method':method+' vs reference labels',
                'horizon_nominal_weeks':h,'target_offset_nominal_weeks':target['nominal_week']-entry['nominal_week']-h,
                'comparison_kind':'reference_labels','reference_segmentation':ref['segmentation'],'forecast_segmentation':relative,
                'volume_error_ml':{name:measured[name]-ref['volumes_ml'][name] for name in labels.values()},
                'predictive_mean_volume_error_ml':metric['compartment_volume_error_ml'],
                'coverage90':metric['compartment_coverage90'],'width90_ml':metric['compartment_width90_ml'],
                'change_group':'measured change from cutoff estimate','cavity_change_group':'reported separately',
                'observed_change_ml':(truth-np.array(list(entry['current_volumes_ml'].values()))).tolist(),
                'predicted_change_ml':(np.array(pred['distribution']['mean_ml'])-np.array(list(entry['current_volumes_ml'].values()))).tolist(),
                'evaluation_metrics':metric})
            torch.save({'identity':result['identity'],'quantity_sha256':quantity.model_id,'horizon':h,
                'source_context_sha256':digest(statepath),'sample_counts':pred['sample_counts'],'representative_sample':0},
                base/entry['visit']/f'amount-first-h{h}-samples.pt')
            verified+=64;matched+=1
        timeline['forecast_uncertainty']='64 joint presence/count samples; empirical 90% volume intervals; calibration not established. Persistence has no fitted uncertainty.'
        timeline['scope']='MU development '+result['identity']+' anatomy. '+result['status']+'. Sample0 map volumes are distinct from distribution means. Observation and missing-sequence outputs preserved. No future-image claim.'
        for entry in timeline['visits']:write_json(base/entry['visit']/'outputs.json',entry)
        write_json(path,timeline)
    if matched!=2:raise ValueError('Expected two predetermined-horizon demonstration targets')
    receipt={'matched_horizons':matched,'native_maps_verified':verified,'same_quantity_samples_as_evaluation':True,
        'representative_sample':0,'nifti_volume_roundtrip_exact':True,'observation_and_completion_files_preserved':True,
        'future_image_synthesis':False,'status':result['status']}
    write_json(args.out/'amount-first-demonstration.json',receipt)
    result['timeline']=receipt;write_json(args.build/'results.json',result)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
