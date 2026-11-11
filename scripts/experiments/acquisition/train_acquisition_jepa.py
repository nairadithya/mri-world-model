#!/usr/bin/env python3
"""One seed-17 acquisition-conditioned representation fit, selected on V only."""
import argparse,json,math,random,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.patient_state.acquisition import AcquisitionObservation,PANELS,STAGES,normalized_feature_loss
from src.patient_state.acquisition_data import AcquisitionPatches
from src.patient_state.losses import joint_loss,segmentation_metrics
from src.patient_state.training import augment,atomic_checkpoint
from src.patient_state.state import PatientState,write_json,digest


def score(logits,reconstruction,x,y,brain,classes=5):
    pred=logits[:,:classes].float().argmax(1)
    anatomy=segmentation_metrics(pred,y,y>=0,range(1,classes))
    nonempty=[v['nonempty_dice'] for v in anatomy.values() if v['nonempty_dice'] is not None]
    false_positive=float((pred>0).sum())/1000
    d=float(np.mean(nonempty)) if nonempty else -min(false_positive,1.)
    b=brain[:,0].bool();lesion=(y>0)&(y<4)&b;residual=reconstruction[:,3].float()-x[:,3]
    gradients=[]
    for axis in (1,2,3):
        a,z=[slice(None)]*4,[slice(None)]*4;a[axis],z[axis]=slice(1,None),slice(None,-1);a,z=tuple(a),tuple(z)
        valid=b[a]&b[z]
        if valid.any():gradients.append(float((residual[a]-residual[z]).abs()[valid].mean()))
    return dict(D=d,nonempty_dice=float(np.mean(nonempty)) if nonempty else None,foreground_empty=not nonempty,anatomy=anatomy,brain_flair_mae=float(residual[b].abs().mean()),
        lesion_flair_mae=float(residual[lesion].abs().mean()) if lesion.any() else None,gradient_flair_mae=float(np.mean(gradients)),predicted_foreground_ml=false_positive)


@torch.inference_mode()
def select(model,rows):
    patients=sorted({r['patient'] for r in rows});dataset=AcquisitionPatches(rows,2*len(patients),selection=True);result=[];model.eval()
    for x,y,b,p,classes in DataLoader(dataset,batch_size=1,num_workers=0):
        x,y,b=x.cuda(),y.cuda(),b.cuda()
        for panel in PANELS:
            flags=torch.tensor([[i in panel for i in range(4)]],device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):logits,reconstruction=model(x,flags)
            result.append(dict(patient=p[0],panel=list(panel),**score(logits,reconstruction,x,y,b,int(classes[0]))))
    keys=('nonempty_dice','lesion_flair_mae','brain_flair_mae','gradient_flair_mae')
    perpatient=[]
    for patient in patients:
        rs=[r for r in result if r['patient']==patient]
        perpatient.append({k:float(np.mean([r[k] for r in rs if r[k] is not None])) if any(r[k] is not None for r in rs) else None for k in keys})
    average={k:float(np.mean([r[k] for r in perpatient if r[k] is not None])) for k in keys}
    average['selection_score']=1-average['nonempty_dice']+.25*average['lesion_flair_mae']+.1*average['brain_flair_mae']+.1*average['gradient_flair_mae']
    average['patients_with_reference_nonempty_selection_patch']=sum(r['nonempty_dice'] is not None for r in perpatient)
    model.train();return dict(**average,records=result,scope='fixed first V visit; two fixed T1-support patches and four FLAIR-withheld panels')


def batch_loss(model,x,y,b,classes,force_subset=None):
    subset=force_subset
    if subset is None:
        size=4 if random.random()<.25 else random.choice((1,2,3))
        subset=tuple(sorted([1]+random.sample([0,2,3],size-1)))
    flags=torch.tensor([[i in subset for i in range(4)]],device=x.device)
    fused=model.features(x,flags);logits,reconstruction=model.decode_features(fused)
    hidden=(~flags)[...,None,None,None].expand_as(x)
    loss,logs=joint_loss(logits,reconstruction,x,y,b,hidden,classes,allow_all_observed=True)
    observed=random.choice(list(subset));missing=[i for i in range(4) if i not in subset]
    terms=[]
    for query,weight,kind in [(observed,.10,'context')]+([(random.choice(missing),.25,'missing')] if missing else []):
        with torch.no_grad():target=model.encode_sequence(x[:,query:query+1],query,target=True)
        prediction=model.predict_query(fused,flags,query)
        term=normalized_feature_loss(prediction,[target[i] for i in STAGES]);loss=loss+weight*term;logs[kind+'_jepa']=float(term.detach());terms.append(query)
    logs.update(subset=list(subset),queries=terms,total=float(loss.detach()))
    return loss,logs


def smoke_checks(model,x,y,b,classes,out):
    model.eval();device=x.device;flags=torch.tensor([[False,True,False,False]],device=device)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        fused=model.features(x,flags);changed=x.clone();changed[:,[0,2,3]]=float('nan');other=model.features(changed,flags)
        assert all(torch.equal(a,c) for a,c in zip(fused,other))
        expected=model.decode_features(fused)
    model.model_id='smoke-acq';meta=dict(patient='smoke-development',visit='one',nominal_week=0,visit_order=[0,0],time_unit='nominal_week',availability=flags[0].tolist(),affine=np.eye(4).tolist())
    state=model.encode(changed,meta);state.save(out/'smoke-state.pt');loaded=PatientState.load(out/'smoke-state.pt')
    actual=model.read(loaded);before_save_readout=model.read(state)
    assert all(torch.equal(actual[k],before_save_readout[k]) for k in ('logits','images'))
    # Patch-window multiplication/division introduces float32 rounding even
    # for one patch. Serialization itself must remain bit-for-bit exact.
    for raw,key in zip(expected,('logits','images')):
        torch.testing.assert_close(raw.float().cpu(),actual[key].float(),rtol=1e-6,atol=1e-6)
    blending_max_abs_error=max(float((raw.float().cpu()-actual[key].float()).abs().max()) for raw,key in zip(expected,('logits','images')))
    a=model.reveal(model.reveal(state,0,x[:,0:1]),2,x[:,2:3]);c=model.reveal(model.reveal(state,2,x[:,2:3]),0,x[:,0:1])
    assert a.metadata['availability']==c.metadata['availability']
    assert all(torch.equal(a.features[k],c.features[k]) for k in a.features)
    before=state.metadata['availability'].copy();keys=set(state.features)
    hypothetical=model.hypothetical_patch(state,state.metadata['patches'][0],3)
    assert hypothetical['kind']=='hypothetical_only' and state.metadata['availability']==before and set(state.features)==keys
    native=score(actual['logits'],actual['images'],x.cpu(),y.cpu(),b.cpu(),classes)
    return dict(unavailable_nan_invariance=True,reveal_order_exact=True,hypothetical_separation=True,saved_state_readout_exact=True,
        blending_max_abs_error=blending_max_abs_error,retained_all_native_target_voxels=int((y>=0).sum()),cavity_voxels=int((y==4).sum()),cavity_outside_t1_support=int(((y==4)&~b[:,0]).sum()),score=native,state_payload_bytes=state.payload_bytes)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--pretrained',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);random.seed(17);np.random.seed(17);torch.manual_seed(17);torch.backends.cudnn.benchmark=True
    a.out.mkdir(parents=True,exist_ok=True);rows=json.loads((a.data/'R.json').read_text());validation=json.loads((a.data/'V.json').read_text())
    if {r['patient'] for r in rows}&{r['patient'] for r in validation}:raise ValueError('Role overlap')
    model=AcquisitionObservation().cuda();initialization=model.initialize_pretrained(a.pretrained);opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    steps=2 if a.smoke else 12500;start=0;best=float('inf');best_step=0;curve=[]
    config=dict(seed=17,steps=steps,patch_size=128,effective_batch=2,lr=1e-4,warmup=1000,weight_decay=1e-4,ema=.996,select_every=1250,
        roles_sha256=digest(a.data.parent/'manifests/roles.json'),checkpointing=True,loss='0.5(CE+Dice)+0.5 hidden+0.1 visible+0.1 gradient+0.25 missing JEPA+0.10 context JEPA',source_sha256=digest(Path(__file__)))
    if a.resume:
        value=torch.load(a.out/'last.pt',map_location='cpu',weights_only=False)
        if value['config']!=config:raise ValueError('Resume configuration changed')
        model.load_state_dict(value['model']);opt.load_state_dict(value['optimizer']);start=value['step'];best=value['best'];best_step=value['best_step'];curve=value['curve']
        random.setstate(value['random']);torch.set_rng_state(value['torch']);torch.cuda.set_rng_state_all(value['cuda'])
    elif (a.out/'last.pt').exists():raise FileExistsError('Use resume or fresh run')
    write_json(a.out/'config.json',dict(config,initialization=initialization))
    if a.smoke:
        rows=[next(r for r in rows if r.get('dataset','LUMIERE')==d) for d in ('LUMIERE','MU')]
    loader=iter(DataLoader(AcquisitionPatches(rows,(steps-start)*2,offset=start*2),batch_size=1,num_workers=2,pin_memory=True))
    started=time.monotonic();times=[];model.train()
    def save(path,step):atomic_checkpoint(path,dict(backend=model.backend,model=model.state_dict(),optimizer=opt.state_dict(),step=step,best=best,best_step=best_step,curve=curve,config=config,initialization=initialization,random=random.getstate(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
    for step in range(start+1,steps+1):
        tick=time.monotonic();fraction=step/1000 if step<=1000 else .5*(1+math.cos(math.pi*(step-1000)/11500))
        for g in opt.param_groups:g['lr']=1e-4*max(.01,fraction)
        opt.zero_grad(set_to_none=True)
        for micro in range(2):
            x,y,b,patient,classes=next(loader);x,y,b=x.cuda(),y.cuda(),b.cuda();x,y,b=augment(x,y,b)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,logs=batch_loss(model,x,y,b,int(classes[0]),force_subset=(0,1,2,3) if a.smoke and step==1 else None)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            (loss/2).backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],12,error_if_nonfinite=True);opt.step();model.update_target();torch.cuda.synchronize();times.append(time.monotonic()-tick)
        if step%25==0 or a.smoke:print(json.dumps(dict(step=step,seconds_per_update=float(np.mean(times[-25:])),peak_bytes=torch.cuda.max_memory_allocated(),**logs)),flush=True)
        if not a.smoke and step%1250==0:
            result=select(model,validation);improved=result['selection_score']<best-1e-10;best=min(best,result['selection_score']);curve.append(dict(step=step,elapsed_seconds=time.monotonic()-started,**result))
            if improved:best_step=step
            write_json(a.out/'curve.json',curve);save(a.out/'last.pt',step)
            if improved:save(a.out/'best.pt',step)
            print(json.dumps(dict(step=step,selection_score=result['selection_score'],best=best)),flush=True)
        elif step%250==0 or a.smoke:save(a.out/'last.pt',step)
    if a.smoke:
        model.eval()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):expected=model(x,torch.ones((1,4),device='cuda',dtype=torch.bool))[0].cpu()
        # Reload into the same allocation; avoid a second GPU-sized model.
        value=torch.load(a.out/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(value['model']);del value
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):actual=model(x,torch.ones((1,4),device='cuda',dtype=torch.bool))[0].cpu()
        assert torch.equal(expected,actual)
        checks=smoke_checks(model,x,y,b,int(classes[0]),a.out)
        write_json(a.out/'smoke.json',dict(passed=True,checkpoint_reload_exact=True,updates=steps,mean_seconds_per_update=float(np.mean(times)),peak_bytes=torch.cuda.max_memory_allocated(),estimated_training_hours=float(np.mean(times))*12500/3600,**checks))
    write_json(a.out/'completed.json',dict(smoke=a.smoke,updates=steps,elapsed_seconds=time.monotonic()-started,mean_update_seconds=float(np.mean(times)),peak_bytes=torch.cuda.max_memory_allocated(),best_selection_score=None if a.smoke else best,best_step=best_step,best_terminal=bool(best_step==steps)))


if __name__=='__main__':main()
