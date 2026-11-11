#!/usr/bin/env python3
"""Bounded image review material: estimation error beside reference-defined change."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.patient_state.state import write_json

BASELINE = '87e2349c54fafd133a08246fd0d920956687faf640c8a080edc2f2cd1dc3ae59'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--contexts', type=Path, required=True)
    a = p.parse_args()
    queue = json.loads((a.review/'private-review-queue.json').read_text())
    results = []
    fig, axes = plt.subplots(len(queue), 4, figsize=(12, 3*len(queue)), squeeze=False)
    for i, row in enumerate(queue):
        y0, y1 = [np.load(row[k], mmap_mode='r').transpose(1, 2, 0) for k in ('current_mask', 'future_mask')]
        x0, x1 = [np.load(row[k], mmap_mode='r')[1].transpose(1, 2, 0) for k in ('current_images', 'future_images')]
        path = a.contexts/(row['patient']+'_'+row['cutoff']+'.pt')
        context = torch.load(path, map_location='cpu', weights_only=False)
        if context['identity']['observation_sha256'] != BASELINE:
            raise ValueError('Observation reference changed')
        pred = context['predicted_segmentation'].numpy()
        exchange = ((y0 == 2) & (y1 == 4)) | ((y0 == 4) & (y1 == 2))
        relevant = exchange if exchange.any() else (y0 == 2) | (y1 == 2) | (y0 == 4) | (y1 == 4)
        z = int(relevant.sum((0, 1)).argmax())
        coords = np.argwhere(relevant[:, :, z])
        if not len(coords):
            coords = np.argwhere((x0 != 0)[:, :, z])
        low = np.maximum(coords.min(0)-15, 0); high = np.minimum(coords.max(0)+16, y0.shape[:2])
        sl = (slice(low[0], high[0]), slice(low[1], high[1]), z)
        for j, (image, labels) in enumerate(((x0, y0), (x0, pred), (x1, y1), (x1, None))):
            ax = axes[i, j]
            ax.imshow(image[sl].T, cmap='gray', origin='lower', vmin=-1.5, vmax=2.5)
            ax.axis('off')
            if labels is not None:
                for c, color in ((2, (1., .55, 0)), (4, (.65, .35, 1.))):
                    mask = (labels[sl] == c).T
                    rgba = np.zeros((*mask.shape, 4)); rgba[mask, :3] = color; rgba[mask, 3] = .55
                    ax.imshow(rgba, origin='lower')
            else:
                for mask, color in ((((y0 == 2) & (y1 == 4)), (.1, .9, 1.)), (((y0 == 4) & (y1 == 2)), (1., .2, .2))):
                    mask = mask[sl].T
                    rgba = np.zeros((*mask.shape, 4)); rgba[mask, :3] = color; rgba[mask, 3] = .65
                    ax.imshow(rgba, origin='lower')
            if i == 0:
                ax.set_title(('Current reference', 'Preserved current model', 'Later reference', 'Reference exchanges')[j], fontsize=11)
        axes[i, 0].text(0, -.08, f'Case {i+1}: {row["group"]}, {row["horizon"]} weeks\nCore Δ {row["core_change_ml"]:+.1f} mL; cavity Δ {row["cavity_change_ml"]:+.1f} mL', transform=axes[i, 0].transAxes, fontsize=9, va='top')
        results.append({'patient': row['patient'], 'cutoff': row['cutoff'], 'target': row['target'], 'slice': z,
                        'baseline_core_in_current_cavity_ml': float(((pred == 2) & (y0 == 4)).sum())/1000.,
                        'baseline_core_false_positive_outside_cavity_ml': float(((pred == 2) & (y0 != 2) & (y0 != 4)).sum())/1000.,
                        'reference_core_change_ml': row['core_change_ml'], 'reference_cavity_change_ml': row['cavity_change_ml'],
                        'flags': row['flags'], 'review_status': 'Prepared for bounded visual inspection; no expert adjudication or target relabeling'})
    fig.suptitle('Core/cavity supervision: six ambiguity-prioritized development cases', fontsize=14)
    fig.text(.5, .01, 'Orange: core; purple: cavity. Exchanges: cyan core → cavity; red cavity → core.\nReleased labels, residual alignment and annotation uncertainty; these are not verified biological transitions.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .97))
    fig.savefig(a.review/'diagnostic-slices.png', dpi=130)
    plt.close(fig)
    write_json(a.review/'private-current-estimation-review.json', results)


if __name__ == '__main__':
    main()
