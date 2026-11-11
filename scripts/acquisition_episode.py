#!/usr/bin/env python3
"""Actual reveal -> saved evidence state -> readouts and hypothetical decisions."""
import argparse,hashlib,json,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import nibabel as nib
import numpy as np
import torch
from src.patient_state.acquisition import AcquisitionObservation
from src.patient_state.acquisition_data import observed_images,state_metadata
from src.patient_state.acquisition_policy import descriptors,action_scores,choose,trajectory
from src.patient_state.state import PatientState,SEQUENCES,write_json,digest
from evaluate_acquisition_jepa import load_role

NAMES=('enhancing','necrotic_non_enhancing','edema_flair','resection_cavity')


def recommend(model,state,selectors):
    desc=descriptors(model,state)
    current=dict(descriptor=desc,availability=state.metadata['availability'])
    return choose(current,selectors['recipe'],selectors['models']['jepa']),action_scores(current,selectors['recipe'],selectors['models']['jepa'])


def episode(model,row,selectors,out,check_boundary=False,resume=False):
    patient=f"{row['patient']}_{row['visit']}";directory=out/patient;directory.mkdir(parents=True,exist_ok=resume)
    steps=[];updates=[];initial=None;recovered=False
    if resume:
        for budget in (1,2,3):
            path=directory/f'acq-{budget:02d}'/'outputs.json'
            if not path.exists():break
            entry=json.loads(path.read_text())
            if entry['model_id']!=model.model_id:raise ValueError('Example checkpoint changed')
            steps.append(entry)
        recovered=bool(steps)
    if len(steps)<3:
        if steps:
            # Read-only mapped storage lets the OS reclaim completed evidence
            # pages during this one unfinished update after host-memory OOM.
            state=PatientState(**torch.load(directory/steps[-1]['state'],map_location='cpu',weights_only=True,mmap=True)).validate()
            observed_indices=[i for i,v in enumerate(state.metadata['availability']) if v]
            x=observed_images(row,observed_indices);observed={i:x[:,i:i+1] for i in observed_indices}
            candidate=SEQUENCES.index(steps[-1]['acquisition']['chosen_next'])
            new=observed_images(row,(candidate,))[:,candidate:candidate+1]
            tick=time.monotonic();state=model.reveal(state,candidate,new);torch.cuda.synchronize();updates.append(time.monotonic()-tick);observed[candidate]=new
        else:
            x=observed_images(row,(1,));torch.cuda.synchronize();tick=time.monotonic();state=model.encode(x,state_metadata(row,(1,)));torch.cuda.synchronize();initial=time.monotonic()-tick
            observed={1:x[:,1:2]}
        affine=np.array(state.metadata['affine'])
    for budget in range(len(steps)+1,4):
        step=f'acq-{budget:02d}';target=directory/step;target.mkdir();state.save(target/'state.pt')
        tick=time.monotonic();output=model.read(state);torch.cuda.synchronize();read_seconds=time.monotonic()-tick
        segmentation=output['logits'].argmax(1)[0].numpy().astype(np.uint8);volumes={name:float((segmentation==i+1).sum())/1000 for i,name in enumerate(NAMES)}
        nib.save(nib.Nifti1Image(segmentation,affine),target/'segmentation.nii.gz')
        for i,image in observed.items():nib.save(nib.Nifti1Image(image[0,0].numpy(),affine),target/(SEQUENCES[i]+'.nii.gz'))
        for i,name in enumerate(SEQUENCES):nib.save(nib.Nifti1Image(output['images'][0,i].float().numpy(),affine),target/('predicted_'+name+'.nii.gz'))
        candidate=None;scores={}
        if budget<3:candidate,scores=recommend(model,state,selectors)
        entry=dict(visit=step,nominal_week=None,time_unit='sequence_reveal_step_not_elapsed_time',state=f'{step}/state.pt',model_id=model.model_id,backend=model.backend,
            input_sequences=[SEQUENCES[i] for i in sorted(observed)],current_volumes_ml=volumes,displayed_anatomy='native',
            anatomy_readouts={'native':dict(label='Native 5-category anatomy',labels={str(i+1):n for i,n in enumerate(NAMES)},segmentation=f'{step}/segmentation.nii.gz',volumes_ml=volumes,state=f'{step}/state.pt',model_id=model.model_id,cavity_status='MU released cavity supervision')},
            image_readout_state=f'{step}/state.pt',forecasts=[],previous_forecast_comparisons=[],state_payload_bytes=state.payload_bytes,state_file_bytes=(target/'state.pt').stat().st_size,
            acquisition=dict(original_visit=row['visit'],budget=budget,actually_observed=[SEQUENCES[i] for i in sorted(observed)],
                hypothetical_candidates={SEQUENCES[int(k)]:v for k,v in scores.items()},chosen_next=None if candidate is None else SEQUENCES[candidate],
                predicted_images='Image-completion outputs, not acquired measurements',uncertainty='No calibrated uncertainty',cost='one revealed sequence, not scan time',readout_seconds=read_seconds))
        write_json(target/'outputs.json',entry);steps.append(entry)
        if check_boundary and budget==1:
            altered=x.clone();altered[:,[0,2,3]]=float('nan');perturbed=model.encode(altered,state_metadata(row,(1,)))
            assert all(torch.equal(state.features[k],perturbed.features[k]) for k in state.features)
            other,_=recommend(model,perturbed,selectors);assert candidate==other;del perturbed,altered
            replay=PatientState.load(target/'state.pt');decoded=model.read(replay)
            assert torch.equal(decoded['logits'],output['logits']);del decoded,replay
        if candidate is not None:
            new=observed_images(row,(candidate,))[:,candidate:candidate+1]
            tick=time.monotonic();state=model.reveal(state,candidate,new);torch.cuda.synchronize();updates.append(time.monotonic()-tick);observed[candidate]=new
    write_json(directory/'timeline.json',dict(version=2,patient=patient,visits=steps,scope='Offline acquisition episode within one visit; sequence steps are not longitudinal time',anatomy_note='Saved observed evidence supports anatomy and completion; hypothetical features only inform choice'))
    return dict(patient=row['patient'],visit=row['visit'],example=patient,initial_encoding_seconds=initial,actual_update_seconds=updates,state_bytes=[s['state_payload_bytes'] for s in steps],
        actions=[s['acquisition']['chosen_next'] for s in steps[:2]],unavailable_perturbation_checked=check_boundary,reload_checked=check_boundary,
        resumed_from_completed_artifacts=recovered,timing_note='Pre-interruption encoding/update times were not persisted; null initial time and listed update times cover only resumed work' if recovered else None)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();torch.set_num_threads(4)
    lock=json.loads((a.root/'policy-lock.json').read_text());selectors=json.loads((a.root/'selectors.json').read_text())
    if digest(a.root/'selectors.json')!=lock['selectors_sha256']:raise ValueError('Policy changed')
    visits=load_role(a.root,'V');ordered=sorted(visits,key=lambda r:hashlib.sha256(('acq-example-v1|'+r['stem']).encode()).hexdigest())
    first=ordered[0];different=[r for r in ordered[1:] if trajectory(r,selectors['recipe'],selectors['models']['jepa'])['actions']!=selectors['fixed_order'][:2]]
    second=different[0] if different else ordered[1]
    rows={r['stem']:r for r in json.loads((a.root/'data/V.json').read_text())}
    model=AcquisitionObservation.from_checkpoint(a.root/'representation/best.pt');a.out.mkdir(parents=True,exist_ok=True)
    results=[episode(model,rows[r['stem']],selectors,a.out,check_boundary=i==0,resume=a.resume) for i,r in enumerate((first,second))]
    write_json(a.out/'examples.json',dict(rule='First V visit by SHA256(acq-example-v1|stem), then first remaining differing learned order if present, otherwise next hash; no accuracy selection',examples=results,fixed_order=selectors['fixed_order'],different_case_exists=bool(different)))


if __name__=='__main__':main()
