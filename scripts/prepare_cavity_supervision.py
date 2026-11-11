#!/usr/bin/env python3
"""Version cavity targets without overwriting existing masks, images, or timing."""
import argparse
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nibabel as nib
import numpy as np
from src.patient_state.data import catalogue
from src.patient_state.state import digest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalogue', type=Path, required=True)
    p.add_argument('--mu', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    value = catalogue(a.catalogue)  # Fixed fitting/selection membership checked before IO.
    a.out.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    for group in ('fitting', 'selection'):
        for row in value['groups'][group]:
            if row.get('dataset', 'LUMIERE') != 'MU':
                row.update(supervised_classes=4, cavity_annotation='unknown', label_schema='conditional-four-class-v2')
                continue
            source = a.mu/'images'/row['patient']/row['visit']/(row['stem']+'_tumorMask.nii.gz')
            im = nib.load(source)
            transform = nib.orientations.ornt_transform(nib.orientations.io_orientation(im.affine), nib.orientations.axcodes2ornt(('L', 'A', 'S')))
            affine = im.affine @ nib.orientations.inv_ornt_aff(transform, im.shape)
            raw = nib.orientations.apply_orientation(np.asanyarray(im.dataobj), transform)
            if list(raw.shape) != row['shape_xyz'] or not np.allclose(affine, row['affine'], atol=1e-5, rtol=0):
                raise ValueError('Released-mask and prepared geometry differ')
            if not np.isin(raw, (0, 1, 2, 3, 4)).all():
                raise ValueError('Unexpected released MU label')
            labels = np.asarray([0, 2, 3, 1, 4], dtype=np.int8)[raw.astype(np.int64)].transpose(2, 0, 1)
            old = np.load(row['mask'], mmap_mode='r')
            if not np.array_equal(np.where(labels == 4, -1, labels), old):
                raise ValueError('New target changes more than the previously ignored cavity')
            path = a.out/(row['stem']+'_mask.npy')
            np.save(path, labels)
            row.update(mask=str(path), supervised_classes=5, cavity_annotation='released explicit label 4',
                       label_schema='exclusive-five-class-v2', label_source='released MU labels including cavity; no new adjudication',
                       source_mask_sha256=digest(source))
            counts[group+'_visits'] += 1
            counts[group+'_cavity_nonempty_visits'] += int((labels == 4).any())
    value.update(label_protocol='MU exclusive five-way; LUMIERE conditional four-way with cavity unknown',
                 parent_catalogue_sha256=digest(a.catalogue))
    write_json(a.out/'catalogue.json', value)
    write_json(a.out/'preparation.json', dict(counts, reserved_images_opened=0,
        source_masks_verified='all prepared MU fitting/selection labels exactly agree outside formerly ignored cavity',
        images_and_timing='unchanged; referenced without copying', parent_catalogue_sha256=digest(a.catalogue)))
    print(dict(counts), flush=True)


if __name__ == '__main__':
    main()
