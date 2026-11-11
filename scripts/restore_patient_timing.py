#!/usr/bin/env python3
"""Join official MU MRI timing without opening any images or segmentation values."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.patient_state.data import catalogue
from src.patient_state.state import digest, write_json
from src.patient_state.timing import mu_clinical_days, timed_pairs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalogue', type=Path, required=True)
    p.add_argument('--clinical', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    cat = catalogue(args.catalogue)
    allowed = {r['patient'] for g in ('fitting', 'selection') for r in cat['groups'][g] if r.get('dataset') == 'MU'}
    days, issues, headers = mu_clinical_days(args.clinical, allowed)
    source = {'url': 'https://www.cancerimagingarchive.net/wp-content/uploads/MU-Glioma-Post_ClinicalData-July2025.xlsx',
              'sha256': digest(args.clinical), 'headers': headers,
              'join': 'PatientID and exact Timepoint_N; no ordinal-to-day interpolation'}
    cat['mu_timing_source'] = source
    counts = {}
    for group in ('fitting', 'selection'):
        for row in cat['groups'][group]:
            if row.get('dataset') != 'MU':
                continue
            day = days.get(row['patient'], {}).get(row['visit'])
            if day is None:
                issues.append({'patient': row['patient'], 'visit': row['visit'], 'reason': 'no usable exact timing join'})
                continue
            row.update(diagnosis_relative_day=day, time_unit='diagnosis_relative_days')
        pairs = timed_pairs(cat['groups'][group])
        counts[group] = {dataset: {str(h): {'targets': len(q := [p for p in pairs if p['dataset'] == dataset and p['horizon'] == h]),
                       'patients': len({p['patient'] for p in q})} for h in (12,24)} for dataset in ('LUMIERE', 'MU')}
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out/'catalogue.json', cat)
    write_json(args.out/'private-join-issues.json', issues)
    summary = {'source': source, 'counts': counts, 'issue_counts': dict(Counter(x['reason'] for x in issues)),
               'reserved_images_opened': 0, 'segmentation_measurements_read': 0,
               'scope': 'metadata eligibility only; spatial correspondence still requires verification'}
    write_json(args.out/'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
