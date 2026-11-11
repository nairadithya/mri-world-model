#!/usr/bin/env python3
"""Verify native/conditional maps, point volumes and saved-state provenance."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nibabel as nib
import numpy as np
import torch
from src.patient_state.anatomy_readouts import anatomy_readouts
from src.patient_state.pipeline import volumes
from src.patient_state.state import LABELS, PatientState, SEQUENCES, write_json


LABELS5 = {**LABELS, 4: 'resection_cavity'}


def checked_map(root, path, expected):
    image = nib.load(root/path)
    segmentation = np.asarray(image.dataobj)
    labels = LABELS5 if 'resection_cavity' in expected else LABELS
    measured = volumes(segmentation, image.affine, labels)
    if set(measured) != set(expected) or not all(np.isclose(measured[k], expected[k], atol=1e-4, rtol=0) for k in measured):
        raise AssertionError('Displayed map and point volumes differ: '+str(path))
    return segmentation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, help='Also replay all saved observation anatomy/image readouts on CUDA')
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = None
    if args.checkpoint:
        from src.patient_state.observation import JointObservation
        model = JointObservation.from_checkpoint(args.checkpoint)
    paths = sorted(args.root.rglob('timeline.json'))
    if not paths:
        raise ValueError('No demonstration timelines found')
    visits = maps = distributions = replays = reference_comparisons = 0
    for path in paths:
        base = path.parent
        timeline = json.loads(path.read_text())
        for entry in timeline['visits']:
            visits += 1
            state = PatientState.load(base/entry['state'])
            assert entry['model_id'] == state.metadata['model_id']
            assert entry['image_readout_state'] == entry['state']
            assert entry['displayed_anatomy'] == 'native'
            assert entry['current_volumes_ml'] == entry['anatomy_readouts']['native']['volumes_ml']
            displayed = {}
            for name, readout in entry['anatomy_readouts'].items():
                assert readout['model_id'] == entry['model_id'] and readout['state'] == entry['state']
                displayed[name] = checked_map(base, readout['segmentation'], readout['volumes_ml'])
                maps += 1
            if state.metadata.get('dataset') == 'LUMIERE':
                assert 'UNVALIDATED' in entry['anatomy_readouts']['native']['cavity_status']
            for sequence in SEQUENCES:
                assert (base/entry['visit']/f'predicted_{sequence}.nii.gz').is_file()
            if model is not None:
                assert model.model_id == entry['model_id']
                output = model.read(state)
                decoded = anatomy_readouts(output['logits'], state.features['brain_support'][:, 0].to(output['logits'].device))
                for name, prediction in decoded.items():
                    assert np.array_equal(prediction[0].cpu().numpy(), displayed[name])
                for i, sequence in enumerate(SEQUENCES):
                    saved = np.asarray(nib.load(base/entry['visit']/f'predicted_{sequence}.nii.gz').dataobj)
                    assert np.array_equal(output['images'][0, i].float().cpu().numpy(), saved)
                replays += 1
            for forecast in entry['forecasts']:
                checked_map(base, forecast['segmentation'], forecast['volumes_ml'])
                maps += 1
                if forecast['method'] == 'persistence':
                    assert forecast['state'] == entry['state']
                    assert forecast['volumes_ml'] == entry['current_volumes_ml']
                    continue
                saved = torch.load(base/forecast['state'], map_location='cpu', weights_only=True)
                context = PatientState.load(base/saved['context_state'])
                assert context.metadata['model_id'] == entry['model_id']
                assert context.metadata['visit'] == entry['visit']
                assert np.array_equal(context.features['segmentation'][0, 0].numpy(), displayed['native'])
                assert saved['predictor_id'] == forecast['predictor_id']
                assert forecast['history_used'] == [entry['visit']]
                distribution = forecast['distribution']
                samples = np.asarray(distribution['samples_ml'])
                weights = np.asarray(distribution['weights'])
                assert samples.shape[1] == 4 and np.isclose(weights.sum(), 1) and (weights >= 0).all()
                assert np.allclose((samples*weights[:, None]).sum(0), distribution['mean_ml'], atol=1e-4)
                assert np.allclose(samples[distribution['representative_scenario']], list(forecast['volumes_ml'].values()), atol=1e-4)
                assert np.allclose(np.asarray(distribution['upper90_ml'])-distribution['lower90_ml'], distribution['width90_ml'], atol=1e-4)
                distributions += 1
                del context
            if 'reference_anatomy' in entry:
                reference = entry['reference_anatomy']
                checked_map(base, reference['segmentation'], reference['volumes_ml'])
                maps += 1
            for comparison in entry['previous_forecast_comparisons']:
                assert comparison['comparison_kind'] in ('reference_labels', 'model_assimilation')
                if comparison['comparison_kind'] == 'reference_labels':
                    assert comparison['reference_segmentation'] == entry['reference_anatomy']['segmentation']
                    image = nib.load(base/comparison['forecast_segmentation'])
                    predicted = volumes(np.asarray(image.dataobj), image.affine, LABELS5)
                    assert all(np.isclose(comparison['volume_error_ml'][k], predicted[k]-v, atol=1e-4)
                               for k, v in entry['reference_anatomy']['volumes_ml'].items())
                    assert 'evaluation_metrics' in comparison
                    reference_comparisons += 1
            del state
    result = {'passed': True, 'timelines': len(paths), 'visits': visits, 'maps_checked': maps,
              'distributions_checked': distributions, 'reference_comparisons': reference_comparisons,
              'exact_saved_state_observation_replays': replays,
              'scope': 'Engineering consistency; no predictive capability or calibration claim'}
    write_json(args.root/'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
