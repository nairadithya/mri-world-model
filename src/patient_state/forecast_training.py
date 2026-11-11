"""Frozen observation forecasting: matched inputs, targets, and development selection."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .data import catalogue
from .observation import JointObservation
from .temporal import SpatialPredictor, forecast_pairs, feature_nll, ensemble_crps
from .losses import joint_loss, segmentation_metrics
from .state import digest, write_json


class TemporalPatches(Dataset):
    """All sampling coordinates and augmentations are fixed before reading the target."""
    def __init__(self, pairs, length, seed=17, fixed=False, offset=0):
        self.pairs, self.length, self.seed, self.fixed, self.offset = pairs, length, seed, fixed, offset
        groups = defaultdict(list)
        for p in pairs:
            groups[p['horizon'], p['patient']].append(p)
        self.groups = {h: [v for (hh, _), v in groups.items() if hh == h] for h in (12, 24)}

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index >= self.length:
            raise IndexError(index)
        rng = np.random.default_rng(self.seed + self.offset + index)
        if self.fixed:
            pair = self.pairs[index // 2 % len(self.pairs)]
        else:
            groups = self.groups[(12, 24)[int(rng.integers(2))]]
            group = groups[int(rng.integers(len(groups)))]
            pair = group[int(rng.integers(len(group)))]
        history = pair['history'][-4:]
        current = history[-1]
        support = np.load(current['support'], mmap_mode='r')
        # Current annotations guide training crops only; validation uses brain support.
        mask = np.load(current['mask'], mmap_mode='r') if not self.fixed else None
        coarse = (mask[::4, ::4, ::4] > 0) if mask is not None and rng.random() < .5 else support[::4, ::4, ::4] > 0
        coords = np.argwhere(coarse & (support[::4, ::4, ::4] > 0))
        if not len(coords):
            coords = np.argwhere(support[::4, ::4, ::4] > 0)
        center = coords[int(rng.integers(len(coords)))]*4
        start = [int(np.clip(c-64, 0, s-128)) for c, s in zip(center, support.shape)]
        sl = tuple(slice(s, s+128) for s in start)
        flips = [] if self.fixed else [d for d in range(3) if rng.random() < .5]
        def image(row):
            a = np.array(np.load(row['images'], mmap_mode='r')[(slice(None), *sl)].transpose(0, 2, 3, 1), dtype=np.float32)
            for axis in flips:
                a = np.flip(a, axis+1)
            if not np.isfinite(a).all():
                raise ValueError('Nonfinite temporal image')
            return torch.from_numpy(a.copy())[None]
        xs = [image(r) for r in history]
        availability = []
        for _ in history:
            a = torch.ones((1, 4), dtype=torch.bool)
            if not self.fixed and rng.random() < .25:
                a[0, int(rng.integers(3))] = False  # current FLAIR always available
            availability.append(a)
        target = pair['target']
        tx = image(target)
        y = np.array(np.load(target['mask'], mmap_mode='r')[sl].transpose(1, 2, 0), dtype=np.int64)
        b = np.array(np.load(target['support'], mmap_mode='r')[sl].transpose(1, 2, 0), dtype=bool)
        for axis in flips:
            y, b = np.flip(y, axis), np.flip(b, axis)
        if not b.any():
            raise ValueError('No target brain in cutoff-selected patch')
        gaps = [0]+[b['visit_order'][0]-a['visit_order'][0] for a,b in zip(history, history[1:])]
        return {'images': xs, 'availability': availability, 'gaps': gaps, 'target': tx,
                'labels': torch.from_numpy(y.copy())[None], 'support': torch.from_numpy(b.copy())[None,None],
                'horizon': pair['horizon'], 'patient': pair['patient']}


def encode_batch(observation, batch, arm):
    indices = list(range(len(batch['images']))) if arm == 'history' else [len(batch['images'])-1]
    features = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in indices:
            features.append(observation.features(batch['images'][i].cuda(), batch['availability'][i].cuda()))
        target = observation.features(batch['target'].cuda(), torch.ones((1,4), dtype=torch.bool, device='cuda'))
    return features, target, indices


def predict_batch(model, features, indices, batch):
    memory = None
    for fs, i in zip(features, indices):
        memory = model.update(fs, memory, batch['gaps'][i], batch['availability'][i].cuda())
    return model.predict(features[-1], memory, batch['horizon'])


def draw_features(means, logvars, generator=None):
    # A Gaussian marginal field correlated in 32mm blocks, rather than independent
    # voxel noise whose errors disappear when aggregated into compartment volumes.
    result = []
    for mean, lv in zip(means, logvars):
        noise = torch.randn((*mean.shape[:2], 4,4,4), device=mean.device, generator=generator)
        noise = torch.nn.functional.interpolate(noise, size=mean.shape[2:], mode='nearest')
        result.append(mean+noise*(.5*lv).exp())
    return result


def decoded_loss(observation, means, batch):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits, images = observation.decode_features(means)
        loss, parts = joint_loss(logits, images, batch['target'].cuda(), batch['labels'].cuda(),
                                 batch['support'].cuda(), torch.ones_like(batch['target'],dtype=torch.bool,device='cuda'))
    return loss, parts, logits, images


@torch.no_grad()
def select(observation, model, pairs):
    # One fixed cutoff per patient/horizon, two brain patches; the full common
    # population is evaluated after selection, with no feature export cache.
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[pair['horizon'],pair['patient']].append(pair)
    fixed = [v[len(v)//2] for _,v in sorted(grouped.items())]
    data = TemporalPatches(fixed, 2*len(fixed), seed=203, fixed=True)
    records = []
    for batch in data:
        features, targets, indices = encode_batch(observation, batch, model.arm)
        means, lv = predict_batch(model, features, indices, batch)
        loss, parts, logits, images = decoded_loss(observation, means, batch)
        seg = segmentation_metrics(logits.argmax(1).cpu(), batch['labels'], batch['support'][:,0])
        error = (images.float().cpu()-batch['target']).abs()
        lesion = (batch['labels']>0)[:,None].expand_as(error) & batch['support'].expand_as(error)
        lesion_mae = float(error[lesion].mean()) if lesion.any() else parts['hidden_mae']
        samples = []
        generator = torch.Generator(device='cuda').manual_seed(731)
        for _ in range(4):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                slogits, _ = observation.decode_features(draw_features(means, lv, generator))
            sp = slogits.argmax(1).cpu()
            samples.append(torch.tensor([int(((sp==c)&batch['support'][:,0]).sum())/1000 for c in (1,2,3)]))
        truth = torch.tensor([int(((batch['labels']==c)&batch['support'][:,0]).sum())/1000 for c in (1,2,3)])
        crps = float(ensemble_crps(torch.stack(samples), truth).mean())
        dice = np.mean([v['dice'] for v in seg.values()])
        records.append({'patient':batch['patient'],'horizon':batch['horizon'],'dice':float(dice),
                        'lesion_mae':lesion_mae,'brain_mae':parts['hidden_mae'],'gradient_mae':parts['gradient_mae'],
                        'volume_crps_ml':crps,'feature_nll':float(feature_nll(means,lv,targets)),
                        'score':float(1-dice+.25*lesion_mae+.1*parts['gradient_mae']+.005*crps)})
    summaries = {}
    for h in (12,24):
        summaries[str(h)] = {k: float(np.mean([r[k] for r in records if r['horizon']==h]))
                            for k in ('score','dice','lesion_mae','brain_mae','gradient_mae','volume_crps_ml','feature_nll')}
    return {'score':np.mean([v['score'] for v in summaries.values()]),'horizons':summaries,'patches':len(records)}


def save_checkpoint(path, value):
    tmp = path.with_suffix('.tmp')
    torch.save(value,tmp)
    tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalogue',type=Path,required=True)
    p.add_argument('--observation',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--arm',choices=('current','history'),required=True)
    p.add_argument('--steps',type=int,default=8000)
    p.add_argument('--select-every',type=int,default=500)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=args.resume)
    if (args.out/'completed.json').exists():
        raise ValueError('Completed run is preserved; choose a new run or schedule')
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed)
    obs=JointObservation.from_checkpoint(args.observation)
    obs_hash=obs.model_id
    model=SpatialPredictor(args.arm).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    cat=catalogue(args.catalogue)
    fitting=forecast_pairs(cat['groups']['fitting']); selection=forecast_pairs(cat['groups']['selection'])
    config={**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            'observation_sha256':obs_hash,'catalogue_sha256':digest(args.catalogue),'horizons':[12,24],
            'max_history':4,'tolerance':4,'correlation':'independent channels, 32mm Gaussian blocks',
            'selection':'equal horizon and patient, fixed cutoff two brain patches',
            'loss':'0.1 feature marginal Gaussian NLL + mean decoded joint loss + 0.1 sample decoded joint loss',
            'uncertainty':'uncalibrated learned Gaussian marginals, empirical decoded volume distribution',
            'fitting_pairs':len(fitting),'selection_pairs':len(selection)}
    config.pop('resume')
    offset=0;curve=[];best=float('inf')
    if args.resume:
        saved=torch.load(args.out/'last.pt',map_location='cpu',weights_only=False)
        if saved['config']!=config:
            raise ValueError('Resume configuration differs')
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        offset=saved['step'];torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
        if (args.out/'curve.json').exists():
            curve=json.loads((args.out/'curve.json').read_text())
            best=min(v['score'] for v in curve)
    write_json(args.out/'config.json',config)
    write_json(args.out/'population.json',{'fitting':fitting,'selection':selection})
    steps=2 if args.smoke else args.steps
    loader=DataLoader(TemporalPatches(fitting,steps-offset,args.seed,offset=offset),batch_size=None,num_workers=2,
                      prefetch_factor=2,generator=torch.Generator().manual_seed(args.seed))
    started=time.monotonic();update_times=[]
    for step,batch in enumerate(loader,offset+1):
        tick=time.monotonic();model.train()
        warmup=min(1.,step/250)
        lr=2e-4*warmup*(.01+.99*.5*(1+math.cos(math.pi*max(0,step-250)/max(1,steps-250))))
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True)
        features,targets,indices=encode_batch(obs,batch,args.arm)
        means,lv=predict_batch(model,features,indices,batch)
        nll=feature_nll(means,lv,targets)
        decoded,parts,logits,images=decoded_loss(obs,means,batch)
        sampled,_,_,_=decoded_loss(obs,draw_features(means,lv),batch)
        loss=.1*nll+decoded+.1*sampled
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),12.,error_if_nonfinite=True);opt.step()
        update_times.append(time.monotonic()-tick)
        if step%25==0 or args.smoke:
            print(json.dumps({'step':step,'loss':float(loss.detach()),'nll':float(nll.detach()),'decoded':float(decoded.detach()),
                              'lr':lr,'mean_seconds':float(np.mean(update_times[-100:]))}),flush=True)
        if step%250==0 or step%args.select_every==0 or step==steps:
            payload={'model':model.state_dict(),'optimizer':opt.state_dict(),'arm':args.arm,'step':step,
                     'config':config,'observation_sha256':obs_hash,'torch_rng':torch.get_rng_state(),
                     'cuda_rng':torch.cuda.get_rng_state(),'scope':'development forecasting'}
            save_checkpoint(args.out/'last.pt',payload)
        if not args.smoke and (step%args.select_every==0 or step==steps):
            model.eval();summary=select(obs,model,selection);summary['step']=step;curve.append(summary)
            write_json(args.out/'curve.json',curve);print('SELECTION '+json.dumps(summary),flush=True)
            if summary['score']<best:
                best=summary['score'];save_checkpoint(args.out/'best.pt',payload)
        if args.smoke and step==steps:
            restored=SpatialPredictor(args.arm).cuda()
            restored.load_state_dict(torch.load(args.out/'last.pt',weights_only=False)['model'])
            with torch.no_grad():
                a=predict_batch(model,features,indices,batch);b=predict_batch(restored,features,indices,batch)
                difference=max(float((u-v).abs().max()) for u,v in zip(a[0]+a[1],b[0]+b[1]))
                if difference!=0:raise AssertionError('Checkpoint replay differs')
                _,_,pred,_=decoded_loss(obs,a[0],batch)
                scored=segmentation_metrics(pred.argmax(1).cpu(),batch['labels'],batch['support'][:,0])
            write_json(args.out/'smoke.json',{'passed':True,'updates':steps,'reload_difference':difference,
                       'decoded_segmentation':scored,'mean_update_seconds':float(np.mean(update_times)),
                       'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'observation_unchanged':digest(args.observation)==obs_hash})
        del features,targets,means,lv,loss,decoded,sampled,logits,images,nll
    if digest(args.observation)!=obs_hash or any(v.requires_grad or v.grad is not None for v in obs.parameters()):
        raise AssertionError('Frozen observation changed')
    write_json(args.out/'completed.json',{'steps':steps,'best_selection_score':best if not args.smoke else None,
               'elapsed_seconds':time.monotonic()-started,'mean_update_seconds':float(np.mean(update_times)),
               'boundary_review_required':True,'forecast_usefulness_established':False})

if __name__=='__main__':main()
