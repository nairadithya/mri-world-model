#!/usr/bin/env python3
"""Replay issued anatomy scenarios from saved context and compare viewer outputs."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import nibabel as nib
import numpy as np
import torch
from src.patient_state.state import PatientState,write_json
from src.patient_state.structured import AnatomicalTransition


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--patient-root',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(4)
    timeline=json.loads((args.patient_root/'timeline.json').read_text())
    checked=0;maximum=0.;replayed=0
    with torch.inference_mode():
        for visit in timeline['visits']:
            for f in visit['forecasts']:
                if f['method']=='persistence':continue
                d=f['distribution'];w=np.array(d['weights']);v=np.array(d['samples_ml'])
                assert np.isclose(w.sum(),1,atol=1e-6) and (w>=0).all()
                assert np.allclose((v*w[:,None]).sum(0),d['mean_ml'],atol=1e-4)
                seg=np.asarray(nib.load(args.patient_root/f['segmentation']).dataobj)
                vv=np.array([(seg==c).sum()/1000 for c in (1,2,3)])
                error=float(np.abs(vv-v[d['representative_scenario']]).max());maximum=max(maximum,error)
                assert error<1e-4
                checked+=1
                # Replay the last cutoff's four issued distributions only.
                if visit is not timeline['visits'][-1]:continue
                saved=torch.load(args.patient_root/f['state'],weights_only=True,map_location='cpu')
                state=PatientState.load(args.patient_root/saved['context_state'])
                params={k:v.cuda() for k,v in saved['parameters'].items()}
                i=int(params['weights'].argmax());params['delta']=params['delta'][i:i+1]
                fs=state.features
                _,maps,images=AnatomicalTransition.render(params,fs['sdf'][0].float().cuda(),fs['images'][0].float().cuda(),
                                      fs['support'][0].cuda(),fs['segmentation'][0,0].cuda())
                assert np.array_equal(maps.argmax(1)[0].byte().cpu().numpy(),seg)
                expected=np.asarray(nib.load(args.patient_root/f['image']).dataobj)
                assert np.array_equal(images[0,3].cpu().numpy(),expected)
                replayed+=1
    result={'passed':True,'visits':len(timeline['visits']),'distributions_checked':checked,
            'exact_map_and_image_replays':replayed,'maximum_map_distribution_volume_difference_ml':maximum,
            'comparisons_after_latest_observation':len(timeline['visits'][-1]['previous_forecast_comparisons']),
            'scope':'Engineering consistency; not predictive capability'}
    write_json(args.patient_root/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
