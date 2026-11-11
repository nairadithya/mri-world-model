#!/usr/bin/env python3
"""Check whether empty ensemble-vote maps also have empty sampled scenarios."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
from src.patient_state.data import load_visit
from src.patient_state.structured_training import inputs,load_context
from src.patient_state.structured_inference import load_transition,signed_distances
from src.patient_state.state import write_json
from scripts.patient_state.interpret_structured_forecasts import patient_mean


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True)
    args=p.parse_args();root=args.run_root;torch.set_num_threads(4)
    pairs=json.loads((root/'structured-selection-v1/population.json').read_text())
    identity=json.loads((root/'structured-selection-v1/identity.json').read_text())
    models={m:load_transition(root/f'structured-v1/anatomy-{m}/best.pt',identity['contexts']['observation_sha256']) for m in ('current','history')}
    for m,net in models.items():
        if net.model_id!=identity['checkpoints'][m]:raise ValueError('Preserved checkpoint changed')
    selected=[r for f in sorted(args.analysis.glob('case-*.json')) for r in json.loads(f.read_text()) if r['compartment']==2 and r['future_empty']]
    records=[]
    for index in sorted({r['evaluation_index'] for r in selected}):
        pair=pairs[index];now=load_visit(pair['history'][-1]);future=load_visit(pair['target'],labels=True)
        valid=(future['support'][0,0]&(future['labels'][0]>=0)).cuda()
        seg=load_context(root/'structured-context-v1',pair['history'][-1])['predicted_segmentation'].cuda();sdf=signed_distances(seg.cpu().numpy()).cuda()
        for m,net in models.items():
            _,x,s=inputs(root/'structured-context-v1',pair,m);parameters=net.parameters_for(x,s)
            scored=[];whole=[]
            for i in range(net.scenarios):
                _,maps,_=net.render({**parameters,'delta':parameters['delta'][i:i+1]},sdf,now['images'][0].cuda(),now['support'][0].cuda(),seg)
                scored.append(float((maps[0,2]*valid).sum())*.001);whole.append(float(maps[0,2].sum())*.001)
            row=next(r for r in selected if r['evaluation_index']==index and r['method']==m)
            weights=parameters['weights'].cpu().numpy();empty=np.array(scored)==0
            records.append({'patient':pair['patient'],'horizon':pair['horizon'],'method':m,'evaluation_index':index,
                'vote_correct_empty':row['forecast_dice']==1,'empty_scored_probability':float(weights[empty].sum()),
                'highest_weight_scenario_empty':bool(empty[int(weights.argmax())]),
                'scored_scenario_volumes_ml':scored,'whole_scenario_volumes_ml':whole})
        print(json.dumps({'checked_targets':len(records)//2,'total_targets':len(selected)//2}),flush=True)
    summary={}
    for h in (12,24):
        for m in models:
            rs=[r for r in records if r['horizon']==h and r['method']==m];good=[r for r in rs if r['vote_correct_empty']]
            summary[f'MU:{h}:{m}']={'empty_targets':len(rs),'correct_empty_votes':len(good),
                'correct_empty_votes_with_no_empty_scenario':sum(r['empty_scored_probability']==0 for r in good),
                'highest_weight_scenario_correct_empty_targets':sum(r['highest_weight_scenario_empty'] for r in rs),
                'mean_probability_empty_given_correct_vote':patient_mean(good,'empty_scored_probability')}
    write_json(args.analysis/'private-empty-scenarios.json',records)
    write_json(args.analysis/'empty-scenario-summary.json',summary)


if __name__=='__main__':main()
