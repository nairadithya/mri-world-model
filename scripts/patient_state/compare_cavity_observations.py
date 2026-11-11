#!/usr/bin/env python3
"""Paired patient-level interpretation of the completed cavity observation build."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import nibabel as nib
from src.patient_state.state import write_json


def paired(pairs, getter):
    grouped = defaultdict(list)
    for before, after in pairs:
        x, y = getter(before), getter(after)
        if (x is None) != (y is None):
            raise ValueError('Different eligible reference populations')
        if x is not None:
            grouped[before['patient']].append((float(x), float(y)))
    if not grouped:
        return None
    values = np.array([np.mean(grouped[k], axis=0) for k in sorted(grouped)])
    delta = values[:, 1]-values[:, 0]
    rng = np.random.default_rng(17)
    draws = delta[rng.integers(len(delta), size=(4000, len(delta)))].mean(1)
    return {'baseline': float(values[:, 0].mean()), 'new': float(values[:, 1].mean()),
            'new_minus_baseline': float(delta.mean()), 'paired_patient_ci95': np.quantile(draws, [.025, .975]).tolist(),
            'patients': len(grouped), 'visits': sum(map(len, grouped.values()))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); root = a.run_root
    dirs = [root/'cavity-reference-selection-v1', root/'cavity-selection-v1']
    aggregates = [json.loads((d/'aggregate.json').read_text()) for d in dirs]
    populations = [json.loads((d/'population.json').read_text()) for d in dirs]
    if populations[0] != populations[1] or aggregates[0]['catalogue_sha256'] != aggregates[1]['catalogue_sha256']:
        raise ValueError('Evaluation targets do not match')
    pairs = []
    for row in populations[0]:
        values = [json.loads((d/(row['stem']+'.json')).read_text()) for d in dirs]
        if any(v['patient'] != row['patient'] or v['visit'] != row['visit'] for v in values):
            raise ValueError('Visit identity mismatch')
        pairs.append(values)
    def mean_dice(record, key):
        return float(np.mean([v['dice'] for v in record[key].values()]))
    metrics = {'scored_compartment_dice': lambda r: mean_dice(r, 'segmentation'),
               'native_available_label_dice': lambda r: mean_dice(r, 'native_available_label_scores'),
               'common_foreground_three_compartment_dice': lambda r: mean_dice(r, 'common_foreground_three_compartments'),
               'core_in_reference_cavity_ml': lambda r: r['core_in_reference_cavity_ml'],
               'cavity_in_reference_core_ml': lambda r: r['cavity_in_reference_core_ml'],
               'intensity_times_1_05_dice': lambda r: mean_dice(r, 'intensity_times_1_05')}
    output = {}
    for dataset in ('LUMIERE', 'MU'):
        selected = [pair for pair in pairs if pair[0]['dataset'] == dataset]
        result = {name: paired(selected, fn) for name, fn in metrics.items()}
        for c in selected[0][0]['segmentation']:
            for key in ('dice', 'nonempty_dice', 'volume_error_ml', 'empty_reference_fp_ml'):
                result[f'class_{c}_{key}'] = paired(selected, lambda r, c=c, key=key: abs(r['segmentation'][c][key]) if r['segmentation'][c][key] is not None else None)
        for sequence in ('CT1', 'T1', 'T2', 'FLAIR'):
            for key in ('brain_mae', 'lesion_mae', 'cavity_mae', 'gradient_mae'):
                result[f'missing_{sequence}_{key}'] = paired(selected, lambda r, sequence=sequence, key=key: r['missing'][sequence][key])
        for i, name in enumerate(('baseline', 'new')):
            for key, aggregate_key in (('scored_compartment_dice', 'dice'), ('native_available_label_dice', 'native_available_label_dice')):
                if not np.isclose(result[key][name], aggregates[i]['by_dataset'][dataset][aggregate_key], atol=1e-10):
                    raise ValueError('Patient means do not reproduce existing aggregate')
        output[dataset] = result
    timeline_checks = []
    for tree in ('timeline-cavity-v1', 'timeline-cavity-missing-flair-v1'):
        for path in sorted((root/tree).glob('*/timeline.json')):
            timeline = json.loads(path.read_text()); count = 0
            for visit in timeline['visits']:
                im = nib.load(path.parent/visit['visit']/'segmentation.nii.gz')
                labels = np.asanyarray(im.dataobj)
                voxel_ml = abs(np.linalg.det(im.affine[:3, :3]))/1000.
                for c, name in enumerate(('enhancing', 'necrotic_non_enhancing', 'edema_flair', 'resection_cavity'), 1):
                    if not np.isclose((labels == c).sum()*voxel_ml, visit['current_volumes_ml'][name], atol=1e-5):
                        raise ValueError('Timeline map and volume disagree')
                if any(f['method'] != 'persistence' for f in visit['forecasts']):
                    raise ValueError('Unexpected learned forecast attached to changed observation model')
                count += 1
            timeline_checks.append({'tree': tree, 'visits': count, 'map_volume_agreement': True, 'forecast_method': 'persistence'})
    curve = json.loads((root/'observation-cavity-v1/curve.json').read_text())
    write_json(a.out, {'date': '2026-11-06', 'status': 'training, matched full selection evaluation and timeline exports complete',
        'scope': 'Development selection; single seed; conditional LUMIERE scores do not validate native cavity output',
        'checkpoints': {name: v['checkpoint_sha256'] for name, v in zip(('baseline', 'new'), aggregates)},
        'catalogue_sha256': aggregates[0]['catalogue_sha256'], 'datasets': output,
        'intervals': '4000 paired patient bootstrap draws, seed 17; patient means across eligible visits; exploratory unadjusted intervals',
        'selected_update': min(curve, key=lambda x:x['selection_score'])['step'],
        'boundary': [{'step': c['step'], 'selection_score': c['selection_score']} for c in curve[-3:]],
        'timelines': timeline_checks, 'reserved_images_opened': 0, 'new_learned_forecast_assessment': False})


if __name__ == '__main__':
    main()
