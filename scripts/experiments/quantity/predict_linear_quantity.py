#!/usr/bin/env python3
"""Cutoff-only inference from a saved linear quantity model and descriptor NPZ."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
from src.patient_state.linear_quantity import load_bundle,predict
from src.patient_state.state import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--descriptor',type=Path,required=True)
    p.add_argument('--horizon',type=int,choices=(12,24),required=True);p.add_argument('--key',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    model,recipe,metadata=load_bundle(a.checkpoint,a.device)
    with np.load(a.descriptor) as data:
        result=predict(model,recipe,data['estimated_counts'],data['descriptor'],int(data['capacity']),a.horizon,a.key)
    result.update(model_identity=model.identity,arm=metadata['arm'],horizon=a.horizon)
    write_json(a.out,result)


if __name__=='__main__':main()
