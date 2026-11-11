#!/usr/bin/env python3
"""Copy saved count reports with exact equal-weight interval ranks; no replay."""
import argparse
import copy
import json
import math
from pathlib import Path
import hashlib


def bounds(values):
    values=sorted(values)
    return [values[max(0,math.ceil(p*len(values))-1)] for p in (.05,.95)]


def repair(source,out):
    out.mkdir(parents=True,exist_ok=True)
    path=source/'build/quantity-cases.json';rows=json.loads(path.read_text());changes=[]
    for r in rows:
        for c,old in r['methods']['amount_first']['conditional_positive'].items():
            j=int(c)-1;v=[n[j]/1000 for n in r['sample_counts'] if n[j]>0]
            if not v:continue
            lo,hi=bounds(v);width=hi-lo;coverage=float(lo<=r['truth_ml'][j]<=hi)
            if round(width*1000)!=round(old['width90_ml']*1000):
                changes.append(dict(key=r['key'],compartment=c,old_width_ml=old['width90_ml'],width_ml=width,coverage_changed=coverage!=old['coverage90']))
            old.update(width90_ml=width,coverage90=coverage,lower90_ml=lo,upper90_ml=hi)
    (out/'quantity-cases.json').write_text(json.dumps(rows,indent=2)+'\n')
    timeline_changes=[]
    for path in (source/'timeline').rglob('*.json'):
        value=json.loads(path.read_text());changed=[0]
        def visit(obj):
            if isinstance(obj,list):
                for v in obj:visit(v)
            elif isinstance(obj,dict):
                if 'samples_ml' in obj and 'conditional_positive' in obj:
                    for j,cond in enumerate(obj['conditional_positive']):
                        v=[row[j] for row in obj['samples_ml'] if row[j]>0]
                        if v:
                            lo,hi=bounds(v)
                            cond.update(lower90_ml=lo,upper90_ml=hi)
                            changed[0]+=1
                if 'scenario_volumes_ml' in obj and isinstance(obj.get('conditional_positive'),dict):
                    for label,cond in obj['conditional_positive'].items():
                        j=int(label)-1
                        v=[round(row[j]*1000)/1000 for row in obj['scenario_volumes_ml'] if row[j]>0]
                        if v:
                            lo,hi=bounds(v)
                            cond.update(lower90_ml=lo,upper90_ml=hi,width90_ml=hi-lo)
                            changed[0]+=1
                for v in obj.values():visit(v)
        visit(value)
        if changed[0]:
            dest=out/'timeline'/path.relative_to(source/'timeline');dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_text(json.dumps(value,indent=2)+'\n');timeline_changes.append(str(dest))
    return dict(source=str(source),source_cases_sha256=hashlib.sha256((source/'build/quantity-cases.json').read_bytes()).hexdigest(),
        changed_intervals=changes,timeline_json_copies=timeline_changes,originals_unchanged=True,no_model_replay=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result={family:repair(Path('outputs')/folder,a.out/family) for family,folder in [('logistic','mu-amount-first-20261108'),('gaussian','mu-gaussian-count-20261108')]}
    (a.out/'repair.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:dict(changed=len(v['changed_intervals']),coverage_changes=sum(r['coverage_changed'] for r in v['changed_intervals'])) for k,v in result.items()}))


if __name__=='__main__':main()
