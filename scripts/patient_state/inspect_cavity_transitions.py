#!/usr/bin/env python3
"""Reference-only development review queue; never feeds future labels into inputs."""
import argparse
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from src.patient_state.data import catalogue
from src.patient_state.state import digest, write_json
from src.patient_state.timing import timed_pairs


def transition_record(current, future, support):
    valid = support & (current >= 0) & (future >= 0)
    matrix = np.bincount((5*current[valid]+future[valid]).astype(np.int64), minlength=25).reshape(5, 5)/1000.
    before, after = matrix.sum(1), matrix.sum(0)
    core_change = after[2]-before[2]
    cavity_change = after[4]-before[4]
    flags = []
    if before[2] > 0 and after[2] > 0:
        flags.append('core_persistent_nonempty')
    elif before[2] > 0:
        flags.append('core_resolved_in_annotation')
    elif after[2] > 0:
        flags.append('core_appeared_in_annotation')
    else:
        flags.append('core_both_empty')
    if core_change < -max(2., .2*before[2]):
        flags.append('substantial_core_reduction')
    if abs(cavity_change) > max(2., .2*before[4]):
        flags.append('substantial_cavity_volume_change')
    if matrix[2, 4]+matrix[4, 2] > 2.:
        flags.append('core_cavity_exchange_review')
    # Both directions can reflect annotation inconsistency or residual alignment,
    # but neither can be diagnosed automatically from masks alone.
    if min(matrix[2, 4], matrix[4, 2]) > 1.:
        flags.append('bidirectional_exchange_review')
    lost = matrix[2].sum()-matrix[2, 2]
    return {'common_support_transition_ml': matrix.tolist(), 'current_common_volume_ml': before[1:].tolist(),
            'future_common_volume_ml': after[1:].tolist(), 'core_change_ml': float(core_change),
            'cavity_change_ml': float(cavity_change), 'lost_core_to_cavity_fraction': float(matrix[2, 4]/lost) if lost else None,
            'flags': flags, 'review_status': 'unreviewed; reference-defined changes, not adjudicated biological transitions'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalogue', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    cat = catalogue(a.catalogue)
    a.out.mkdir(parents=True, exist_ok=False)
    records, counts = [], {}
    for group in ('fitting', 'selection'):
        pairs = [p for p in timed_pairs(cat['groups'][group]) if p['dataset'] == 'MU']
        for pair in pairs:
            current, future = pair['history'][-1], pair['target']
            if any(r.get('supervised_classes') != 5 for r in (current, future)):
                raise ValueError('Explicit cavity targets required')
            y0, y1 = [np.load(r['mask'], mmap_mode='r') for r in (current, future)]
            if y0.shape != y1.shape or not np.allclose(current['affine'], future['affine'], atol=1e-5, rtol=0):
                raise ValueError('Cannot compare masks on different grids')
            support = np.load(current['support'], mmap_mode='r').astype(bool) & np.load(future['support'], mmap_mode='r').astype(bool)
            record = transition_record(y0, y1, support)
            record.update(group=group, patient=pair['patient'], cutoff=current['visit'], target=future['visit'],
                          horizon=pair['horizon'], target_offset=pair['target_offset'],
                          current_mask=current['mask'], future_mask=future['mask'],
                          current_images=current['images'], future_images=future['images'])
            records.append(record)
        for h in (12, 24):
            rs = [r for r in records if r['group'] == group and r['horizon'] == h]
            counts[group+'_'+str(h)] = {'targets': len(rs), 'patients': len({r['patient'] for r in rs}),
                'flags': dict(Counter(f for r in rs for f in r['flags']))}
    # A bounded queue: one case per patient, prioritized by ambiguity, not outcomes of a new model.
    priority = sorted(records, key=lambda r: (-int('bidirectional_exchange_review' in r['flags']),
        -int('core_cavity_exchange_review' in r['flags']), -abs(r['core_change_ml']), r['patient'], r['cutoff']))
    queue, seen = [], set()
    for record in priority:
        if record['patient'] not in seen:
            queue.append(record); seen.add(record['patient'])
        if len(queue) == 6:
            break
    write_json(a.out/'private-transitions.json', records)
    write_json(a.out/'private-review-queue.json', queue)
    write_json(a.out/'aggregate.json', {'groups': counts, 'review_queue_cases': len(queue),
        'catalogue_sha256': digest(a.catalogue), 'reserved_images_opened': 0,
        'scope': 'Reference-only retrospective supervision inspection; no new model fitted, labels not changed',
        'limitations': 'Common-support voxel transitions need image review; registration and annotation inconsistency remain possible. LUMIERE cavity is unknown. No expert adjudication completed.'})
    print(counts, flush=True)


if __name__ == '__main__':
    main()
