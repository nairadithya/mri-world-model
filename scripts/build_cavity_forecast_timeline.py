#!/usr/bin/env python3
"""Build a bounded current-only cavity timeline from fixed development pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.patient_state.data import catalogue
from src.patient_state.observation import JointObservation
from src.patient_state.pipeline import append_visit, save_nifti, volumes
from src.patient_state.state import LABELS, digest, write_json
from src.patient_state.structured_inference import load_transition

LABELS5 = {**LABELS, 4: 'resection_cavity'}


def key(record):
    return (record['patient'], record['cutoff'], record['target'], record['horizon'])


def choose_pairs(population, smoke=False):
    pairs = population['fitting' if smoke else 'selection']
    if smoke:
        return pairs[:1]
    if not pairs:
        raise ValueError('No fixed selection pairs')
    groups = {}
    for pair in pairs:
        groups.setdefault((pair['patient'], pair['history'][-1]['visit']), []).append(pair)
    # Prefer a cutoff with both predeclared horizons; tie-break without outcomes.
    ordered = sorted(groups, key=lambda k: (-len({p['horizon'] for p in groups[k]}),
                     hashlib.sha256(k[0].encode()).hexdigest(), k[1]))
    chosen = groups[ordered[0]]
    return [sorted([p for p in chosen if p['horizon'] == h], key=lambda p: p['target']['visit_order'])[0]
            for h in sorted({p['horizon'] for p in chosen})]


def attach_references(root, pairs, evaluations):
    patient = pairs[0]['patient']
    base = root/patient
    path = base/'timeline.json'
    timeline = json.loads(path.read_text())
    entries = {entry['visit']: entry for entry in timeline['visits']}
    for pair in pairs:
        current, target = pair['history'][-1], pair['target']
        record = evaluations[(patient, current['visit'], target['visit'], pair['horizon'])]
        entry = entries[target['visit']]
        reference = np.asarray(np.load(target['mask'], mmap_mode='r').transpose(1, 2, 0), dtype=np.int16)
        if not np.isin(reference, (-1, 0, 1, 2, 3, 4)).all():
            raise ValueError('Unexpected native reference categories')
        reference_path = f"{target['visit']}/reference_native_segmentation.nii.gz"
        save_nifti(base/reference_path, reference, target['affine'])
        reference_volumes = volumes(reference, target['affine'], LABELS5)
        entry['reference_anatomy'] = {'segmentation': reference_path, 'volumes_ml': reference_volumes,
            'source': target['label_source'], 'status': 'Released five-category reference; not adjudicated biological truth'}
        forecasts = entries[current['visit']]['forecasts']
        for method in ('persistence', 'current'):
            forecast = next(f for f in forecasts if f['method'] == method and f['horizon_nominal_weeks'] == pair['horizon'])
            metric = record['methods'][method]
            comparison = {'issued_at': current['visit'], 'horizon_nominal_weeks': pair['horizon'],
                'target_offset_nominal_weeks': pair['target_offset'],
                'method': method+' vs reference labels', 'comparison_kind': 'reference_labels',
                'reference': 'Released native five-category reference labels',
                'reference_segmentation': reference_path, 'forecast_segmentation': forecast['segmentation'],
                'volume_error_ml': {k: forecast['volumes_ml'][k]-v for k, v in reference_volumes.items()},
                'volume_error_definition': 'Displayed forecast map minus displayed reference map; evaluation-domain metrics retained separately',
                'change_group': record['change_group'], 'cavity_change_group': record.get('cavity_change_group'),
                'evaluation_metrics': metric,
                'evaluation_scope': 'Frozen matched evaluation; reference-defined stable/changed groups, distinct from model assimilation',
                'coverage90': metric.get('compartment_coverage90'),
                'width90_ml': metric.get('compartment_width90_ml'),
                'evaluation_source': record['_source']}
            if forecast.get('distribution'):
                d = forecast['distribution']
                truth = np.array(list(reference_volumes.values()))
                comparison['displayed_reference_coverage90'] = ((truth >= d['lower90_ml']) & (truth <= d['upper90_ml'])).tolist()
                comparison['displayed_reference_width90_ml'] = d['width90_ml']
            entry['previous_forecast_comparisons'].append(comparison)
        entry.setdefault('reference_evaluations', []).append(record)
    for entry in timeline['visits']:
        for comparison in entry['previous_forecast_comparisons']:
            comparison.setdefault('comparison_kind', 'model_assimilation')
        write_json(base/entry['visit']/'outputs.json', entry)
    timeline['scope'] += ' Reference-label comparisons are listed separately from model-assimilation comparisons. Stable/changed labels come from the matched evaluation.'
    write_json(path, timeline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('contexts', 'checkpoint', 'transition', 'evaluation', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--catalogue', type=Path, help='Optional retained LUMIERE observation catalogue')
    parser.add_argument('--lumiere-patient', help='Patient identifier for the optional LUMIERE observation')
    parser.add_argument('--lumiere-visit', help='Visit identifier for the optional LUMIERE observation')
    parser.add_argument('--smoke', action='store_true', help='First fixed fitting pair; explicitly marked instrument check')
    args = parser.parse_args()
    if bool(args.catalogue) != bool(args.lumiere_patient and args.lumiere_visit) or (not args.catalogue and (args.lumiere_patient or args.lumiere_visit)):
        parser.error('--catalogue, --lumiere-patient and --lumiere-visit must be supplied together')
    torch.set_num_threads(4)
    population = json.loads((args.contexts/'population.json').read_text())
    pairs = choose_pairs(population, args.smoke)
    if not pairs or any(p['dataset'] != 'MU' for p in pairs):
        raise ValueError('Fixed MU development pairs required')
    identity = json.loads((args.contexts/'identity.json').read_text())
    if identity['observation_sha256'] != digest(args.checkpoint) or identity.get('compartments') != 4:
        raise ValueError('Contexts must use this five-category observation checkpoint')
    receipt = json.loads((args.evaluation/'identity.json').read_text())
    if receipt['observation_context'] != identity or receipt['models']['current'] != digest(args.transition) or receipt['smoke'] != args.smoke:
        raise ValueError('Matched evaluation must use these exact contexts, transition and smoke status')
    if receipt['population_sha256'] != digest(args.contexts/'population.json'):
        raise ValueError('Evaluation population changed')
    evaluations = {}
    for path in sorted(args.evaluation.glob('case-*.json')):
        record = json.loads(path.read_text())
        if key(record) in evaluations:
            raise ValueError('Duplicate evaluation target')
        evaluations[key(record)] = dict(record, _source=str(path.resolve()))
    for pair in pairs:
        if (pair['patient'], pair['history'][-1]['visit'], pair['target']['visit'], pair['horizon']) not in evaluations:
            raise ValueError('Matching frozen evaluation is required for each demo pair')
    observation = JointObservation.from_checkpoint(args.checkpoint)
    transition = load_transition(args.transition, observation.model_id)
    if observation.num_classes != 5 or transition.arm != 'current' or transition.compartments != 4:
        raise ValueError('Native five-category current-only models required')
    if (observation.is_smoke or transition.is_smoke) and not args.smoke:
        raise ValueError('Smoke checkpoints require --smoke')
    args.out.mkdir(parents=True, exist_ok=False)
    rows = {r['stem']: r for pair in pairs for r in (pair['history'][-1], pair['target'])}
    ordered = sorted(rows.values(), key=lambda row: row['visit_order'])
    horizons = tuple(sorted({pair['horizon'] for pair in pairs}))
    for row in ordered:
        append_visit(observation, row, args.out/'predictive', horizons=horizons,
                     structured_predictors={'current': transition})
    attach_references(args.out/'predictive', pairs, evaluations)
    append_visit(observation, ordered[0], args.out/'missing-flair', horizons=(), hidden_sequences=('FLAIR',))
    lumiere_status = 'Not requested'
    if args.catalogue:
        cat = catalogue(args.catalogue)
        candidates = [row for rows in cat['groups'].values() for row in rows
                      if row['patient'] == args.lumiere_patient and row['visit'] == args.lumiere_visit and row.get('dataset', 'LUMIERE') == 'LUMIERE']
        if len(candidates) != 1:
            raise ValueError('Exactly one matching retained LUMIERE row required')
        append_visit(observation, candidates[0], args.out/'lumiere', horizons=())
        lumiere_status = 'One retained visit; native cavity predictions UNVALIDATED'
    write_json(args.out/'demonstration.json', {'scope': 'Instrument check' if args.smoke else 'Bounded development demonstration; not an independent capability test',
        'observation_sha256': observation.model_id, 'transition_sha256': transition.model_id,
        'pair_selection': 'First fitting pair' if args.smoke else 'Prefer both horizons, then patient identifier hash and cutoff; no outcomes used',
        'pairs': [{k: p[k] for k in ('patient', 'horizon', 'target_offset')} | {'cutoff': p['history'][-1]['visit'], 'target': p['target']['visit']} for p in pairs],
        'lumiere': lumiere_status, 'reserved_images_opened': 0})
    print(str(args.out.resolve()), flush=True)


if __name__ == '__main__':
    main()
