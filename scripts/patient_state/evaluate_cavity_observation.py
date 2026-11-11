#!/usr/bin/env python3
"""Compare full-volume anatomy/completion with explicit cavity annotation availability."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from src.patient_state.data import catalogue
from src.patient_state.observation import JointObservation
from src.patient_state.observation_selection import bounded_rows, evaluate_visit, summarize
from src.patient_state.state import digest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalogue', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--bounded', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(4)
    rows = catalogue(a.catalogue)['groups']['selection']
    if a.bounded:
        rows = bounded_rows(rows)
    a.out.mkdir(parents=True, exist_ok=False)
    model = JointObservation.from_checkpoint(a.checkpoint)
    write_json(a.out/'population.json', rows)
    records = []
    for row in rows:
        record = evaluate_visit(model, row, nuisance=True)
        records.append(record)
        write_json(a.out/(row['stem']+'.json'), record)
        print(f'Evaluated {len(records)}/{len(rows)} visits', flush=True)
    write_json(a.out/'aggregate.json', dict(summarize(records), checkpoint_sha256=digest(a.checkpoint),
        catalogue_sha256=digest(a.catalogue), scope='bounded selection check' if a.bounded else 'full development selection',
        cavity_unknown_on_LUMIERE=True, reserved_images_opened=0))


if __name__ == '__main__':
    main()
