#!/usr/bin/env python3
"""Identical-target full-volume forecasting evaluation and predictive viewer exports."""
import argparse,json,sys,time
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.patient_state.data import catalogue,load_visit,metadata
from src.patient_state.observation import JointObservation
from src.patient_state.temporal import forecast_pairs,ensemble_crps
from src.patient_state.forecast_inference import load_predictor,observed_history,forecast_distribution,sample_state
from src.patient_state.losses import segmentation_metrics
from src.patient_state.pipeline import save_nifti,volumes
from src.patient_state.state import write_json,digest,LABELS,SEQUENCES,PatientState


def distribution_scores(samples,truth):
    a=np.asarray(samples,dtype=float);y=np.asarray(truth,dtype=float)
    lo,hi=np.quantile(a,[.05,.95],axis=0)
    crps=np.abs(a-y).mean(0)-.5*np.abs(a[:,None]-a[None,:]).mean((0,1))
    width=hi-lo
    interval=width+20*np.maximum(lo-y,0)+20*np.maximum(y-hi,0)
    return {'samples_ml':a.tolist(),'mean_ml':a.mean(0).tolist(),'lower90_ml':lo.tolist(),'upper90_ml':hi.tolist(),
            'coverage90':((y>=lo)&(y<=hi)).astype(float).tolist(),'width90_ml':width.tolist(),
            'crps_ml':crps.tolist(),'interval_score90_ml':interval.tolist()}


def metrics(seg,images,case,current_pred,current_truth):
    y=case['labels'];support=case['support'];valid=support[:,0]&(y>=0)
    pred=torch.from_numpy(seg)[None]
    anatomy=segmentation_metrics(pred,y,support[:,0])
    vv=np.array([(seg==c).sum()/1000 for c in (1,2,3)])
    yy=np.array([int(((y==c)&support[:,0]).sum())/1000 for c in (1,2,3)])
    residual=images.float().cpu()-case['images'];error=residual.abs()
    lesion=(y>0)[:,None].expand_as(error)&support.expand_as(error)
    grads=[]
    for axis in (2,3,4):
        aa,bb=[slice(None)]*5,[slice(None)]*5
        aa[axis],bb[axis]=slice(1,None),slice(None,-1);aa,bb=tuple(aa),tuple(bb)
        mask=support.expand_as(error)[aa]&support.expand_as(error)[bb]
        grads.append(float((residual[aa]-residual[bb]).abs()[mask].mean()))
    return {'dice':float(np.mean([v['dice'] for v in anatomy.values()])), 'compartments':anatomy,
            'volumes_ml':vv.tolist(),'target_volumes_ml':yy.tolist(),'volume_mae_ml':float(np.abs(vv-yy).mean()),
            'predicted_change_ml':(vv-current_pred).tolist(),'observed_change_ml':(yy-current_truth).tolist(),
            'volume_change_mae_ml':float(np.abs((vv-current_pred)-(yy-current_truth)).mean()),
            'brain_mae':float(error[support.expand_as(error)].mean()),
            'lesion_mae':float(error[lesion].mean()) if lesion.any() else None,
            'gradient_mae':float(np.mean(grads))}


def bootstrap(records,horizon,metric,comparator):
    grouped=defaultdict(list)
    for r in records:
        if r['horizon']!=horizon:continue
        a=r['methods']['history'].get(metric);b=r['methods'][comparator].get(metric)
        if a is not None and b is not None:grouped[r['patient']].append(a-b)
    v=np.array([np.mean(x) for _,x in sorted(grouped.items())]);rng=np.random.default_rng(17)
    if not len(v):
        return {'difference':None,'patient_bootstrap95':None,'patients':0}
    samples=v[rng.integers(len(v),size=(2000,len(v)))].mean(1)
    return {'difference':float(v.mean()),'patient_bootstrap95':np.quantile(samples,[.025,.975]).tolist(),'patients':len(v)}


def aggregate(records):
    summary={}
    keys=('dice','volume_mae_ml','volume_change_mae_ml','brain_mae','lesion_mae','gradient_mae',
          'volume_crps_ml','coverage90','width90_ml','interval_score90_ml','anatomy_brier','feature_mse')
    for h in (12,24):
        subset=[r for r in records if r['horizon']==h]
        if not subset:continue
        result={'targets':len(subset),'patients':len({r['patient'] for r in subset}),
                'offsets':{str(d):sum(r['target_offset']==d for r in subset) for d in range(-4,5)},'methods':{}}
        for method in ('persistence','current','history'):
            def average(k,selected):
                grouped=defaultdict(list)
                for r in selected:
                    value=r['methods'][method].get(k)
                    if value is not None:grouped[r['patient']].append(value)
                return float(np.mean([np.mean(v) for v in grouped.values()])) if grouped else None
            result['methods'][method]={k:average(k,subset) for k in keys}
            result['methods'][method]['by_observed_change']={name:{'targets':sum(r['change_group']==name for r in subset),
                 **{k:average(k,[r for r in subset if r['change_group']==name]) for k in keys}}
                 for name in ('stable','changed')}
            result['methods'][method]['compartments']={str(c):{
                k:float(np.mean([np.mean([r['methods'][method]['distribution'][k][c-1] for r in subset if r['patient']==patient])
                  for patient in sorted({r['patient'] for r in subset})]))
                for k in ('coverage90','width90_ml','crps_ml','interval_score90_ml')} for c in (1,2,3)}
            for c in ('1','2','3'):
                for metric in ('dice','nonempty_dice','volume_error_ml','empty_reference_fp_ml'):
                    values=defaultdict(list)
                    for r in subset:
                        value=r['methods'][method]['compartments'][c][metric]
                        if value is not None:
                            values[r['patient']].append(abs(value) if metric=='volume_error_ml' else value)
                    result['methods'][method]['compartments'][c][metric]=float(np.mean([np.mean(v) for v in values.values()])) if values else None
        result['history_minus_control']={control:{k:bootstrap(records,h,k,control) for k in
           ('dice','volume_mae_ml','volume_change_mae_ml','lesion_mae','gradient_mae','volume_crps_ml')}
           for control in ('current','persistence')}
        summary[str(h)]=result
    return summary


def export_observation(obs,row,patient_root,entry_map):
    if row['visit'] in entry_map:return
    case=load_visit(row,labels=True);state=obs.encode(case['images'].cuda(),metadata(row))
    state.features['brain_support']=case['support'];read=obs.read(state)
    seg=read['logits'].argmax(1)[0].cpu().numpy().astype(np.uint8);seg[~case['support'][0,0].numpy()]=0
    destination=patient_root/row['visit'];destination.mkdir(parents=True,exist_ok=True)
    affine=state.metadata['affine']
    for i,name in enumerate(SEQUENCES):save_nifti(destination/(name+'.nii.gz'),case['images'][0,i].numpy(),affine)
    save_nifti(destination/'predicted_FLAIR.nii.gz',read['images'][0,3].float().cpu().numpy(),affine)
    save_nifti(destination/'segmentation.nii.gz',seg,affine)
    save_nifti(destination/'reference.nii.gz',case['labels'][0].numpy().astype(np.int16),affine)
    entry_map[row['visit']]={'visit':row['visit'],'nominal_week':row['visit_order'][0],
        'input_sequences':list(SEQUENCES),'current_volumes_ml':volumes(seg,affine),
        'forecasts':[],'previous_forecast_comparisons':[],'model_id':obs.model_id}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalogue',type=Path,required=True);p.add_argument('--observation',type=Path,required=True)
    p.add_argument('--current',type=Path,required=True);p.add_argument('--history',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--samples',type=int,default=16)
    p.add_argument('--limit',type=int);p.add_argument('--resume',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=args.resume)
    torch.set_num_threads(4);obs=JointObservation.from_checkpoint(args.observation)
    models={arm:load_predictor(getattr(args,arm),obs.model_id) for arm in ('current','history')}
    if any(net.arm!=arm for arm,net in models.items()):
        raise ValueError('Current-only and history checkpoint arguments are reversed')
    selection_rows=catalogue(args.catalogue)['groups']['selection']
    pairs=forecast_pairs(selection_rows)
    # Demo selected by timing coverage alone, before forecast outcomes are known.
    groups=defaultdict(list)
    for pair in pairs:groups[pair['patient'],pair['history'][-1]['visit']].append(pair)
    demo_key=next((key for key,pp in sorted(groups.items()) if {q['horizon'] for q in pp}=={12,24}),None)
    if args.limit:
        pairs=sorted(pairs,key=lambda x:0 if (x['patient'],x['history'][-1]['visit'])==demo_key else 1)[:args.limit]
    identity={'observation':obs.model_id,'current':digest(args.current),'history':digest(args.history),
              'catalogue':digest(args.catalogue),'samples':args.samples,'limit':args.limit}
    if (args.out/'identity.json').exists() and json.loads((args.out/'identity.json').read_text())!=identity:
        raise ValueError('Resume identity changed')
    write_json(args.out/'identity.json',identity);write_json(args.out/'population.json',pairs)
    records=[];entries={};demo_root=args.out/'timeline'/demo_key[0] if demo_key else None
    timeline_path=demo_root/'timeline.json' if demo_root else None
    if timeline_path and timeline_path.exists():
        entries={v['visit']:v for v in json.loads(timeline_path.read_text())['visits']}
    for index,pair in enumerate(pairs):
        key=f"{pair['patient']}_{pair['history'][-1]['visit']}_h{pair['horizon']}"
        record_path=args.out/(key+'.json')
        if args.resume and record_path.exists():records.append(json.loads(record_path.read_text()));continue
        start=time.monotonic();state=observed_history(obs,pair['history'],models['history'])
        current=obs.read(state);support=state.features['brain_support'][0,0].numpy()
        curseg=current['logits'].argmax(1)[0].cpu().numpy().astype(np.uint8);curseg[~support]=0
        current_volume=np.array([(curseg==c).sum()/1000 for c in (1,2,3)])
        current_case=load_visit(pair['history'][-1],labels=True)
        current_truth=np.array([int(((current_case['labels']==c)&current_case['support'][:,0]).sum())/1000 for c in (1,2,3)])
        # Future data are loaded only after the cutoff state is finalized.
        case=load_visit(pair['target'],labels=True)
        target_state=obs.encode(case['images'].cuda(),metadata(pair['target']))
        truth=np.array([int(((case['labels']==c)&case['support'][:,0]).sum())/1000 for c in (1,2,3)])
        changed=bool((np.abs(truth-current_truth)>np.maximum(2.,.2*current_truth)).any())
        record={'patient':pair['patient'],'cutoff':pair['history'][-1]['visit'],'target':pair['target']['visit'],
                'horizon':pair['horizon'],'target_offset':pair['target_offset'],
                'change_group':'changed' if changed else 'stable','methods':{}}
        demo=(pair['patient'],pair['history'][-1]['visit'])==demo_key
        if demo:
            for row in pair['history'][-4:]+[pair['target']]:export_observation(obs,row,demo_root,entries)
            # One saved cutoff state includes recurrent memory; both heads reload it.
            state_file=demo_root/pair['history'][-1]['visit']/'state.pt'
            if not state_file.exists():state.save(state_file)
            state=PatientState.load(state_file)
            nuisance_state=observed_history(obs,pair['history'],models['history'],cutoff_gain=1.05)
        for arm in ('persistence','current','history'):
            if arm=='persistence':
                mean_state=state;lv=None;out=current;seg=curseg
                # Persistence carries the observed image, not a reconstructed image.
                image_mean=current_case['images']
            else:
                mean_state,lv=forecast_distribution(models[arm],state,pair['horizon'])
                out=obs.read(mean_state);seg=out['logits'].argmax(1)[0].cpu().numpy().astype(np.uint8);seg[~support]=0
                image_mean=out['images']
            result=metrics(seg,image_mean,case,current_volume,current_truth)
            result['feature_mse']=float(np.mean([float((mean_state.features[k].float()-target_state.features[k].float()).square().mean())
                  for k in target_state.features if 'feature' in k]))
            probability=torch.zeros_like(out['logits'],dtype=torch.float32,device='cpu')
            samples=[]
            for s in range(1 if arm=='persistence' else args.samples):
                if arm=='persistence':sp=seg
                else:
                    sampled_state=sample_state(mean_state,lv,17000+s)
                    sampled=obs.read(sampled_state)
                    sp=sampled['logits'].argmax(1)[0].cpu().numpy().astype(np.uint8);sp[~support]=0
                    del sampled,sampled_state
                samples.append([(sp==c).sum()/1000 for c in (1,2,3)])
                for c in range(4):probability[0,c]+=torch.from_numpy(sp==c)
            probability/=len(samples)
            valid=case['support'][:,0]&(case['labels']>=0)
            brier=sum((probability[:,c]-(case['labels']==c).float()).square() for c in range(4))
            result['anatomy_brier']=float(brier[valid].mean())
            result['distribution']=distribution_scores(samples,truth)
            for key2,source in [('volume_crps_ml','crps_ml'),('coverage90','coverage90'),('width90_ml','width90_ml'),('interval_score90_ml','interval_score90_ml')]:
                result[key2]=float(np.mean(result['distribution'][source]))
            record['methods'][arm]=result
            if demo and arm!='persistence':
                nuisance_mean,nuisance_lv=forecast_distribution(models[arm],nuisance_state,pair['horizon'])
                nuisance_out=obs.read(nuisance_mean)
                nuisance_seg=nuisance_out['logits'].argmax(1)[0].cpu().numpy().astype(np.uint8)
                nuisance_seg[~support]=0
                agreement=segmentation_metrics(torch.from_numpy(nuisance_seg)[None],torch.from_numpy(seg)[None],torch.from_numpy(support)[None])
                result['cutoff_intensity_times_1_05']={'scope':'metadata-selected demonstration cutoff only',
                    'mean_compartment_agreement_dice':float(np.mean([v['dice'] for v in agreement.values()])),
                    'volume_shift_ml':[(int((nuisance_seg==c).sum())-int((seg==c).sum()))/1000 for c in (1,2,3)],
                    'image_shift_mae':float((nuisance_out['images'].float()-image_mean.float()).abs()[state.features['brain_support'].expand_as(image_mean)].mean())}
                del nuisance_mean,nuisance_lv,nuisance_out
            if demo:
                issued_distribution={k:result['distribution'][k] for k in
                    ('samples_ml','mean_ml','lower90_ml','upper90_ml','width90_ml')}
                cutoff=pair['history'][-1]['visit'];target=pair['target']['visit'];name=f'{arm}_h{pair["horizon"]}'
                destination=demo_root/cutoff;affine=state.metadata['affine']
                save_nifti(destination/(name+'_segmentation.nii.gz'),seg,affine)
                save_nifti(destination/(name+'_FLAIR.nii.gz'),image_mean[0,3].float().cpu().numpy(),affine)
                entropy=-(probability*probability.clamp_min(1e-7).log()).sum(1)[0].numpy()
                save_nifti(destination/(name+'_entropy.nii.gz'),entropy,affine)
                write_json(destination/(name+'_forecast.json'),{'cutoff':cutoff,'horizon':pair['horizon'],
                    'observation_sha256':obs.model_id,'predictor_sha256':identity.get(arm),
                    'history_used':[r['visit'] for r in pair['history'][-4:]] if arm=='history' else [cutoff],
                    'distribution':issued_distribution,'scope':'retrospective development forecast; no future inputs'})
                entry=entries[cutoff]
                entry['forecasts'].append({'method':arm,'horizon_nominal_weeks':pair['horizon'],
                    'forecast_week':pair['history'][-1]['visit_order'][0]+pair['horizon'],
                    'segmentation':cutoff+'/'+name+'_segmentation.nii.gz','image':cutoff+'/'+name+'_FLAIR.nii.gz',
                    'history_used':[r['visit'] for r in pair['history'][-4:]] if arm=='history' else [cutoff],
                    'entropy':cutoff+'/'+name+'_entropy.nii.gz','volumes_ml':dict(zip(LABELS.values(),result['volumes_ml'])),
                    'distribution':issued_distribution,'uncertainty_status':'point mass' if arm=='persistence' else f'uncalibrated learned distribution; {args.samples} draws'})
                entries[target]['previous_forecast_comparisons'].append({'issued_at':cutoff,'method':arm,
                    'horizon_nominal_weeks':pair['horizon'],'target_offset_nominal_weeks':pair['target_offset'],
                    'forecast_segmentation':cutoff+'/'+name+'_segmentation.nii.gz','reference_segmentation':target+'/reference.nii.gz',
                    'reference':'subsequent automated development labels','dice':result['dice'],
                    'change_group':record['change_group'],'observed_change_ml':result['observed_change_ml'],
                    'predicted_change_ml':result['predicted_change_ml'],'coverage90':result['distribution']['coverage90'],
                    'volume_error_ml':dict(zip(LABELS.values(),np.array(result['volumes_ml'])-truth))})
            if arm!='persistence':del mean_state,lv,out
        if demo:
            write_json(timeline_path,{'version':2,'patient':pair['patient'],'visits':sorted(entries.values(),key=lambda v:(v['nominal_week'],v['visit'])),
                'scope':('INSTRUMENT CHECK: smoke checkpoints, not trained forecasts. ' if args.limit else '')+'Retrospective development forecasting. Nominal weeks, ±4 week matching. Automated reference labels. Stability: every compartment changes ≤max(2 mL,20%); descriptive, not clinical.',
                'forecast_uncertainty':f'90% empirical volume intervals from {args.samples} draws; persistence has zero-width intervals; calibration is evaluated, not assumed'})
            del nuisance_state
        record['seconds']=time.monotonic()-start;write_json(record_path,record);records.append(record)
        print(json.dumps({'evaluated':index+1,'total':len(pairs),'seconds':record['seconds']}),flush=True)
        del state,target_state,current,case,current_case
        write_json(args.out/'progress.json',{'completed':len(records),'total':len(pairs)})
    if timeline_path and timeline_path.exists():
        timeline=json.loads(timeline_path.read_text())
        patient_rows=sorted([r for r in selection_rows if r['patient']==demo_key[0]],key=lambda r:r['visit_order'])
        by_name={r['visit']:i for i,r in enumerate(patient_rows)}
        for entry in timeline['visits']:
            upto=patient_rows[:by_name[entry['visit']]+1]
            state_file=demo_root/entry['visit']/'state.pt'
            if not state_file.exists():
                updated=observed_history(obs,upto,models['history']);updated.save(state_file);del updated
            entry.update(state=entry['visit']+'/state.pt', assimilated_observations=[r['visit'] for r in upto[-4:]])
        write_json(timeline_path,timeline)
    write_json(args.out/'aggregate.json',{'scope':'instrument check' if args.limit else 'development-selection, frozen observation',
        'horizons':aggregate(records),'samples':args.samples,'population_identical_across_methods':True,
        'persistence_uncertainty':'degenerate point distribution; zero width, no fitted interval',
        'intervals':'empirical 5th/95th percentiles of decoded samples; no selection-target recalibration',
        'useful_forecasting_established':False,'next':'Inspect matched controls, change/stability and calibration; boundary trend; no independent-generalization claim'})

if __name__=='__main__':main()
