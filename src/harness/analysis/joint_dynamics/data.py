"""Rebuild, bind and consume the existing R38 images and automated targets."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable

import numpy as np
import torch

from src.data.representation import SEQUENCES, find_image
from src.model.representation import SpatialState, nuisance
from src.preprocessing.geometry import GEOMETRY_CONTRACT, Grid, align_masks
from src.preprocessing.transforms import runtime_transform
from .protocol import (CHECKPOINT_SHA256, COMPARTMENTS, PLAN, ROW_SHA256, ForecastRow,
                       Normalizer, canonical_hash, digest, gap_weeks, read_json,
                       save_json, week_order)

PREPARATION_SOURCES = ('src/harness/analysis/joint_dynamics/protocol.py',
                       'src/harness/analysis/joint_dynamics/data.py',
                       'src/harness/analysis/joint_dynamics/model.py',
                       'src/model/representation.py', 'src/data/representation.py',
                       'src/preprocessing/geometry.py', 'src/preprocessing/transforms.py')


def unique(records: list[dict], key: str) -> dict[str, dict]:
    result = {record[key]: record for record in records}
    if len(result) != len(records):
        raise ValueError(f'duplicate {key}')
    return result


def measured(visit: dict) -> list[float]:
    values = [float(visit['values_mm3'][c]) for c in COMPARTMENTS]
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('invalid measured native target')
    return [math.log1p(x) for x in values]


def derive_rows(anatomy: dict, inventory: dict, prepared: dict, splits: dict) -> list[dict]:
    """Re-derive original adjacency and history, including unusable boundaries."""
    original = [v for v in anatomy['visits'] if v['cohort'] == 'LUMIERE']
    unique(original, 'visit_id')
    inv = unique(inventory['visits'], 'visit_id')
    usable = unique(prepared['visits'], 'visit_id')
    if set(inv) != set(usable) or not set(usable) <= {v['visit_id'] for v in original}:
        raise ValueError('prepared/inventory/full chronology identity mismatch')
    identities = [(v['patient_id'], v['visit']) for v in original]
    if len(set(identities)) != len(identities):
        raise ValueError('duplicate chronology patient/visit')
    for vid, visit in usable.items():
        for key in ('patient_id', 'visit', 'values_mm3', 'mask_relative', 'labels'):
            if visit[key] != inv[vid][key]:
                raise ValueError(f'prepared and inventory disagree: {vid}/{key}')
    pairs = {}
    for pair in prepared['pairs']:
        key = (pair['source_visit_id'], pair['target_visit_id'])
        if key in pairs:
            raise ValueError('duplicate prepared pair')
        pairs[key] = pair
    result = []
    allowed = set().union(*(set(x) for x in splits.values()))
    for patient in sorted({v['patient_id'] for v in original}):
        ordered = sorted([v for v in original if v['patient_id'] == patient],
                         key=lambda v: week_order(v['visit']))
        for i, cutoff in enumerate(ordered[:-1]):
            target = ordered[i + 1]
            if patient not in allowed or any(v['visit_id'] not in usable for v in (cutoff, target)):
                continue
            start = i
            while start > 0 and ordered[start - 1]['visit_id'] in usable:
                start -= 1
            history = ordered[max(start, i - 3):i + 1]
            intervals = [gap_weeks(a['visit'], b['visit']) * 7
                         for a, b in zip(history[:-1], history[1:])]
            # Only historical gaps affect inherited eligibility, never the outgoing gap.
            if any(g <= 0 for g in intervals):
                continue
            pair = pairs.get((cutoff['visit_id'], target['visit_id']))
            if pair is None:
                continue
            if not pair['usable'] or pair.get('exclusions') or pair['patient_id'] != patient:
                raise ValueError('adjacent pair explicitly unusable or cross-patient')
            a, b = usable[cutoff['visit_id']], usable[target['visit_id']]
            if a['pregrid'] != b['pregrid']:
                raise ValueError('known broken pair geometry')
            pad = 4 - len(history)
            result.append({'row_id': f"{a['visit_id']}->{b['visit_id']}", 'patient_id': patient,
                'cutoff_visit_id': a['visit_id'], 'target_visit_id': b['visit_id'],
                'history_visit_ids': [v['visit_id'] for v in history],
                'gap_log1p': [0.] * (pad + 1) + [math.log1p(g) for g in intervals],
                'history_mask': [False] * pad + [True] * len(history),
                'truth': measured(b), 'cutoff_measured': measured(a),
                'primary_history': len(history) >= 2})
    return result


def load_metadata(bindings: dict) -> dict:
    paths = {k: Path(v) for k, v in bindings['inputs'].items()}
    hashes = {k: digest(p) for k, p in paths.items() if p.is_file()}
    for key, expected in bindings['known_sha256'].items():
        if key in paths and hashes.get(key) != expected:
            raise ValueError(f'changed frozen input: {key}')
    names = ('inventory', 'prepared_metadata', 'original_run', 'geometry', 'task_manifest',
             'r37_contract', 'r37_feature_metadata', 'mask_transfer')
    objects = {k: read_json(paths[k]) for k in names}
    inv, prep, run = (objects[k] for k in ('inventory', 'prepared_metadata', 'original_run'))
    old, meta = objects['r37_contract'], objects['r37_feature_metadata']
    if hashes['checkpoint'] != CHECKPOINT_SHA256 or hashes['r37_feature_cache'] != old['cache_sha256']:
        raise ValueError('frozen checkpoint or R37 cache hash mismatch')
    if run['prepared_sha256'] != hashes['prepared_metadata']:
        raise ValueError('run does not bind prepared metadata')
    if prep['inventory_sha256'] not in (hashes['inventory'], canonical_hash(inv)):
        raise ValueError('prepared inventory identity mismatch')
    if prep['source_hashes'] != inv['source_hashes'] or inv['source_hashes']['manifest'] != hashes['geometry']:
        raise ValueError('source target inventory mismatch')
    alias = {'prepared': 'prepared_metadata', 'run': 'original_run', 'features': 'original_features',
             'checkpoint': 'checkpoint', 'inventory': 'inventory', 'direct_predictions': 'original_predictions',
             'anatomy_manifest': 'geometry', 'chronology_manifest': 'task_manifest'}
    for old_key, new_key in alias.items():
        if meta['input_hashes'][old_key] != hashes[new_key]:
            raise ValueError(f'original feature input mismatch: {old_key}')
    splits = {'source52': sorted(run['training_patients']),
              'validation13': sorted(run['validation_patients']),
              'unseen26': sorted(run['evaluation_patients'])}
    if tuple(map(len, splits.values())) != (52, 13, 26):
        raise ValueError('wrong split sizes')
    if any(set(splits[a]) & set(splits[b]) for a in splits for b in splits if a < b):
        raise ValueError('overlapping splits')
    if splits != old['splits'] or splits != meta['splits']:
        raise ValueError('original split binding mismatch')
    if (set(splits['source52']) | set(splits['validation13']) != set(inv['protocol']['encoder_train'])
            or set(splits['unseen26']) != set(inv['protocol']['encoder_unseen'])):
        raise ValueError('split differs from original encoder partition')
    rows = derive_rows(objects['geometry'], inv, prep, splits)
    if (len(rows) != 423 or rows != meta['rows'] or canonical_hash(rows) != ROW_SHA256
            or old['row_manifest_sha256'] != ROW_SHA256):
        raise ValueError('independently reconstructed rows differ from immutable row object')
    if len(prep['visits']) != 599 or len([v for v in objects['geometry']['visits']
                                        if v['cohort'] == 'LUMIERE']) != 638:
        raise ValueError('incomplete prepared/full chronology')
    transfer = objects['mask_transfer']
    if transfer['prepared_sha256'] != hashes['prepared_metadata'] or transfer['inventory_sha256'] != hashes['inventory']:
        raise ValueError('mask transfer metadata binding mismatch')
    mask_records = unique(transfer['records'], 'visit_id')
    if set(mask_records) != {v['visit_id'] for v in prep['visits']}:
        raise ValueError('incomplete transferred masks')
    for visit in prep['visits']:
        record = mask_records[visit['visit_id']]
        if record['relative'] != visit['mask_relative'] or record['labels'] != visit['labels']:
            raise ValueError('transfer changed mask source or labels')
        if Path(record['relative']).is_absolute() or '..' in Path(record['relative']).parts:
            raise ValueError('unsafe mask path')
    return {'paths': paths, 'input_sha256': hashes, 'objects': objects, 'rows': rows,
            'splits': splits, 'mask_records': mask_records}


def source_normalizer(visits: list[dict], source_patients: list[str], saved: dict) -> Normalizer:
    unique(visits, 'visit_id')
    if len(source_patients) != 52 or len(set(source_patients)) != 52:
        raise ValueError('normalizer requires the exact source52 patient set')
    selected = sorted([v for v in visits if v['patient_id'] in source_patients], key=lambda v: v['visit_id'])
    if len(selected) != 353 or {v['patient_id'] for v in selected} != set(source_patients):
        raise ValueError('normalizer requires every one of the 353 source52 prepared visits')
    values = np.array([measured(v) for v in selected], dtype=np.float64)
    mean, scale = values.mean(0), values.std(0, ddof=0).clip(0.5)
    if not all(np.allclose(x, saved[k], atol=1e-6, rtol=1e-6)
               for k, x in (('mean', mean), ('scale', scale))):
        raise ValueError('independent source-only normalizer does not reproduce R35')
    return Normalizer(tuple(v['visit_id'] for v in selected), tuple(sorted(source_patients)),
                      tuple(mean.tolist()), tuple(scale.tolist()))


def verify_raw_masks(meta: dict) -> dict[str, str]:
    result = {}
    for vid, record in meta['mask_records'].items():
        path = meta['paths']['mask_root'] / record['relative']
        actual = digest(path)
        if actual != record['sha256'] or path.stat().st_size != record['bytes']:
            raise ValueError(f'raw mask changed: {vid}')
        result[record['relative']] = actual
    return result


def verify_geometry(path: Path, geometry: dict, pregrid: dict) -> dict:
    import nibabel as nib
    if geometry.get('geometry_contract') != GEOMETRY_CONTRACT or geometry.get('schema_version') != 1:
        raise ValueError('wrong geometry contract')
    key = '/'.join((path.parent.parent.name, path.parent.name,
                    path.name.removesuffix('.gz').removesuffix('.nii')))
    record = geometry['images'][key]
    image = nib.load(str(path))
    grid = Grid(tuple(image.shape), image.affine)
    values = image.get_fdata(dtype=np.float32)
    content = hashlib.sha256(json.dumps(grid.record(), sort_keys=True).encode())
    content.update(np.ascontiguousarray(values, dtype='<f4').tobytes())
    if (not np.isfinite(values).all() or record['tensor_grid'] != grid.record()
            or record['content_sha256'] != content.hexdigest()):
        raise ValueError('MRI content/grid differs from frozen geometry')
    physical = Grid(tuple(record['pregrid']['shape']), record['pregrid']['affine'])
    if not physical.same_as(Grid(tuple(pregrid['shape']), pregrid['affine'])):
        raise ValueError('prepared pregrid differs from frozen physical image grid')
    if np.allclose(grid.affine, np.eye(4), atol=1e-6, rtol=0):
        if (record.get('reconstruction_verified') is not True
                or record.get('resize') != 'trilinear-align_corners-false-then-nonzero-zscore'):
            raise ValueError('unverified identity-affine image')
    elif not physical.same_as(grid):
        raise ValueError('native affine differs from pregrid')
    return {'geometry_record_sha256': canonical_hash(record), 'pregrid': physical.record()}


def _load_frozen(meta: dict) -> tuple[dict, dict, dict]:
    paths = meta['paths']
    checkpoint = torch.load(paths['checkpoint'], map_location='cpu', weights_only=False)
    original = torch.load(paths['original_features'], map_location='cpu', weights_only=False)
    cache = torch.load(paths['r37_feature_cache'], map_location='cpu', weights_only=False)
    prep, run = (meta['objects'][k] for k in ('prepared_metadata', 'original_run'))
    for payload in (checkpoint, original):
        provenance = payload['provenance']
        if (payload['contract'] != 'mri-only-state-observed-change-v1'
                or payload['normalization'] != run['normalization']
                or provenance['prepared_sha256'] != meta['input_sha256']['prepared_metadata']
                or provenance['source_data_hashes'] != prep['source_hashes']
                or provenance['training_patients'] != run['training_patients']
                or provenance['validation_patients'] != run['validation_patients']):
            raise ValueError('original checkpoint/features provenance mismatch')
    if (checkpoint.get('seed') != 42 or checkpoint['provenance']['arm'] != 'state'
            or checkpoint['provenance']['seed'] != 42
            or original['checkpoint'] != CHECKPOINT_SHA256
            or original['provenance']['source_hashes'] != checkpoint['provenance']['source_hashes']):
        raise ValueError('wrong R35 control identity')
    metadata = meta['objects']['r37_feature_metadata']
    if any(cache[k] != metadata[k] for k in ('rows', 'visit_ids', 'splits', 'checkpoint_sha256', 'normalization')):
        raise ValueError('cached R37 private metadata mismatch')
    if set(original['features']) != {v['visit_id'] for v in prep['visits']}:
        raise ValueError('incomplete original pooled features')
    return checkpoint, original, cache


def prepare_inputs(meta: dict, out: Path, guard: object) -> dict:
    """Called only inside the runner's charged GPU stage; no historical writes."""
    import nibabel as nib
    out.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.monotonic(), time.process_time()
    verify_raw_masks(meta)
    checkpoint, original, cache = _load_frozen(meta)
    encoder = SpatialState().cuda().float().eval().requires_grad_(False)
    encoder.load_state_dict(checkpoint['state_dict'], strict=True)
    norm = meta['objects']['original_run']['normalization']
    canonical = {}
    ids = cache['visit_ids']
    with torch.inference_mode():
        decoded = torch.nn.functional.linear(cache['maps'].float().mean((-3, -2, -1)),
                   cache['decoder_weight'].float(), cache['decoder_bias'].float()).numpy()
    decoded = decoded * np.asarray(norm['scale']) + np.asarray(norm['mean'])
    if decoded.shape != (599, 3) or not np.isfinite(decoded).all():
        raise ValueError('invalid canonical cached native decode')
    canonical.update(zip(ids, decoded.tolist()))
    with np.load(meta['paths']['original_predictions'], allow_pickle=False) as direct:
        if set(direct['visit_ids'].astype(str)) != set(ids):
            raise ValueError('incomplete direct frozen predictions')
        for vid, val in zip(direct['visit_ids'].astype(str), direct['state_prediction']):
            if not np.allclose(canonical[vid], val, atol=1e-4, rtol=1e-4):
                raise ValueError('canonical cached decode differs from immutable direct values')
    receipts, pooled_errors, native_errors = {}, [], []
    anatomy = {v['visit_id']: v for v in meta['objects']['geometry']['visits'] if v['cohort'] == 'LUMIERE'}
    visits = meta['objects']['prepared_metadata']['visits']
    for visit in sorted(visits, key=lambda v: v['visit_id']):
        guard.check()
        vid = visit['visit_id']
        directory = meta['paths']['mri_root'] / visit['patient_id'] / visit['visit']
        paths = [find_image(directory, name) for name in SEQUENCES]
        if not any(paths):
            raise ValueError('usable visit has no MRI')
        raw_before = {str(p): digest(p) for p in paths if p is not None}
        geometry = {str(p): verify_geometry(p, meta['objects']['inventory']['geometry'], visit['pregrid'])
                    for p in paths if p is not None}
        if any(digest(p) != sha for p, sha in raw_before.items()):
            raise ValueError('MRI changed during geometry reading')
        image = np.zeros((4, 96, 96, 96), dtype=np.float16)
        for slot, path in enumerate(paths):
            if path is not None:
                image[slot] = runtime_transform(str(path))[0].numpy().astype(np.float16)
                if not np.isfinite(image[slot]).all() or not (image[slot] != 0).any():
                    raise ValueError('nonfinite/empty float16 prepared MRI')
        mask_record = meta['mask_records'][vid]
        mask_path = meta['paths']['mask_root'] / mask_record['relative']
        if digest(mask_path) != mask_record['sha256']:
            raise ValueError('mask changed before alignment')
        mask_image = nib.load(str(mask_path))
        labels = np.asanyarray(mask_image.dataobj)
        if (not np.isfinite(labels).all() or not (labels == np.floor(labels)).all()
                or not set(np.unique(labels)) <= {0, *visit['labels'].values()}):
            raise ValueError('mask is not a finite declared integer label array')
        declared = anatomy[vid]['measurement']['mask']
        if (declared['labels'] != visit['labels'] or not declared['finite']
                or not declared['geometry_valid']
                or not declared['path'].endswith(visit['mask_relative'])):
            raise ValueError('mask differs from frozen source declaration')
        aligned = align_masks([(mask_path, visit['labels'][c]) for c in COMPARTMENTS],
                              [p for p in paths if p is not None],
                              manifest=meta['objects']['inventory']['geometry'])
        if aligned.provenance['image_pregrid'] != visit['pregrid']:
            raise ValueError('new mask alignment changed image pregrid')
        native = aligned.native_volumes_mm3.double().numpy()
        if not np.allclose(native, [visit['values_mm3'][c] for c in COMPARTMENTS], atol=1, rtol=1e-6):
            raise ValueError('native mask volumes differ from frozen measurements')
        available = np.array([p is not None for p in paths], dtype=np.float32)
        with torch.inference_mode():
            features = encoder.encode(torch.from_numpy(image.astype(np.float32))[None].cuda(),
                                      torch.from_numpy(available)[None].cuda())
            pooled = features[-1].mean((-3, -2, -1))
            predicted = encoder.volume_head(pooled).float().cpu().numpy()[0]
            pooled = pooled.float().cpu().numpy()[0]
        predicted = predicted * np.asarray(norm['scale']) + np.asarray(norm['mean'])
        reference = np.asarray(original['features'][vid])
        if (not np.allclose(pooled, reference, atol=1e-4, rtol=1e-4)
                or not np.allclose(predicted, canonical[vid], atol=1e-4, rtol=1e-4)):
            raise ValueError('newly prepared bytes fail all-visit frozen reconstruction')
        pooled_errors.append(float(np.max(np.abs(pooled - reference))))
        native_errors.append(float(np.max(np.abs(predicted - canonical[vid]))))
        if any(digest(p) != sha for p, sha in raw_before.items()) or digest(mask_path) != mask_record['sha256']:
            raise ValueError('MRI/mask changed across transformation and encoding')
        image_file, mask_file = f'{vid}.image.npy', f'{vid}.mask.npy'
        np.save(out / image_file, image, allow_pickle=False)
        np.save(out / mask_file, aligned.masks.numpy().astype(np.uint8), allow_pickle=False)
        receipts[vid] = {'visit': visit, 'availability': available.tolist(), 'raw_mri': raw_before,
            'geometry': geometry, 'raw_mask_sha256': mask_record['sha256'],
            'native_mm3': native.tolist(), 'image_file': image_file, 'mask_file': mask_file,
            'image_sha256': digest(out / image_file), 'mask_sha256': digest(out / mask_file),
            'alignment': aligned.provenance}
    verify_raw_masks(meta)
    normalizer = source_normalizer(visits, meta['splits']['source52'], norm)
    # Recompute from the newly read masks, not merely from inherited measurements.
    native_visits = [{**v, 'values_mm3': dict(zip(COMPARTMENTS, receipts[v['visit_id']]['native_mm3']))}
                     for v in visits]
    rebuilt = source_normalizer(native_visits, meta['splits']['source52'], norm)
    if not all(np.allclose(getattr(rebuilt, k), getattr(normalizer, k), atol=1e-6, rtol=1e-6)
               for k in ('mean', 'scale')):
        raise ValueError('new mask normalizer disagrees with original source measurements')
    torch.cuda.synchronize()
    result = {'schema': 'r38-prepared-v3', 'rows': meta['rows'], 'splits': meta['splits'],
        'visits': receipts, 'input_sha256': meta['input_sha256'], 'normalizer': asdict(rebuilt),
        'frozen_normalization': norm, 'canonical_control_estimates': canonical,
        'roundtrip': 'runtime-transform/float16-storage/float32-load',
        'raw_source_normalizer_visit_ids': list(normalizer.visit_ids),
        'prepared_source_normalizer_visit_ids': list(rebuilt.visit_ids),
        'historical_dense_tensor_hash': None,
        'historical_dense_identity_limit': 'No historical dense-mask tensor hash exists; native volumes do not prove byte equivalence.',
        'reconstruction': {'count': 599, 'atol': 1e-4, 'rtol': 1e-4,
            'max_pooled_absolute_error': max(pooled_errors), 'max_native_absolute_error': max(native_errors)},
        'timing': {'wall_seconds': time.monotonic() - started,
                   'process_cpu_seconds': time.process_time() - cpu_started}}
    save_json(out / 'prepared.json', result)
    return result


class PreparedData:
    def __init__(self, root: Path, device: str = 'cuda', loader: Callable = np.load):
        self.root, self.device, self.loader = root, device, loader
        self.payload = read_json(root / 'prepared.json')
        if self.payload.get('schema') != 'r38-prepared-v3':
            raise ValueError('wrong prepared input schema')
        self.rows = tuple(ForecastRow.from_dict(r) for r in self.payload['rows'])
        self.by_row = {r.row_id: r for r in self.rows}
        self.visits = self.payload['visits']
        self.normalizer = self.payload['normalizer']
        self.canonical = self.payload['canonical_control_estimates']
        self.splits = self.payload['splits']

    def select(self, name: str) -> tuple[ForecastRow, ...]:
        if name in ('primary', 'single'):
            return tuple(r for r in self.rows if r.patient_id in self.splits['unseen26']
                         and r.primary_history == (name == 'primary'))
        if name == 'nuisance':
            # Bound row order is the inherited order, already chronological per patient.
            selected = {}
            for row in self.select('primary'):
                selected.setdefault(row.patient_id, row)
            return tuple(selected.values())
        return tuple(r for r in self.rows if r.patient_id in self.splits[name])

    def check_row(self, row: ForecastRow) -> None:
        if self.by_row.get(row.row_id) != row:
            raise ValueError('row history, target or metadata differs from bound cohort')

    def _array(self, vid: str, kind: str) -> torch.Tensor:
        record = self.visits[vid]
        path = self.root / record[f'{kind}_file']
        if digest(path) != record[f'{kind}_sha256']:
            raise ValueError('prepared bytes changed before consumption')
        values = self.loader(path, allow_pickle=False)
        result = torch.as_tensor(np.array(values, copy=True), device=self.device).float()
        if digest(path) != record[f'{kind}_sha256'] or not torch.isfinite(result).all():
            raise ValueError('prepared bytes changed/nonfinite during consumption')
        return result

    def inputs(self, row: ForecastRow, arm: object, condition: str = 'identity'):
        from .model import ForecastInputs
        self.check_row(row)
        if condition not in ('identity', 'intensity', 'resolution'):
            raise ValueError('undeclared perturbation')
        images = torch.zeros((1, 4, 4, 96, 96, 96), device=self.device)
        availability = torch.zeros((1, 4, 4), device=self.device)
        present = torch.zeros((1, 4), device=self.device, dtype=torch.bool)
        pad = 4 - len(row.history_visit_ids)
        for j, vid in enumerate(row.history_visit_ids, pad):
            availability[0, j] = torch.tensor(self.visits[vid]['availability'], device=self.device)
            if arm.history or vid == row.cutoff_visit_id:
                images[0, j] = self._array(vid, 'image')
                present[0, j] = True
        if condition != 'identity':
            images[:, -1] = nuisance(images[:, -1], condition)
        return ForecastInputs(images, availability,
            torch.tensor([row.gap_log1p], device=self.device, dtype=torch.float32),
            torch.tensor([row.history_mask], device=self.device, dtype=torch.bool), present)

    def target(self, row: ForecastRow, *, current: bool = False):
        from .model import AnatomyTargets
        self.check_row(row)
        vid = row.cutoff_visit_id if current else row.target_visit_id
        volume = row.cutoff_measured if current else row.truth
        return AnatomyTargets(torch.tensor([volume], device=self.device, dtype=torch.float32),
                              self._array(vid, 'mask')[None])

    def controls(self, rows: tuple[ForecastRow, ...]):
        from .baselines import ControlInputs, FilterHistory
        return ControlInputs(np.array([self.canonical[r.cutoff_visit_id] for r in rows], dtype=np.float64),
            tuple(FilterHistory(np.array([self.canonical[v] for v in r.history_visit_ids], dtype=np.float64),
              np.array([0.] + [math.expm1(x) / 7 for x in r.gap_log1p[-len(r.history_visit_ids) + 1:]],
                       dtype=np.float64) if len(r.history_visit_ids) > 1 else np.zeros(1, dtype=np.float64))
              for r in rows))

    def source_chains(self, anatomy: dict) -> tuple[tuple, dict]:
        from .baselines import SourceChain
        chains, singletons, breaks = [], [], []
        for patient in self.splits['source52']:
            ordered = sorted([v for v in anatomy['visits'] if v['cohort'] == 'LUMIERE'
                              and v['patient_id'] == patient], key=lambda v: week_order(v['visit']))
            chain = []
            def finish() -> None:
                if len(chain) == 1:
                    singletons.append(chain[0]['visit_id'])
                elif chain:
                    chains.append(SourceChain(patient, tuple(v['visit_id'] for v in chain),
                        np.array([self.canonical[v['visit_id']] for v in chain], dtype=np.float64),
                        np.array([0.] + [gap_weeks(a['visit'], b['visit'])
                          for a, b in zip(chain[:-1], chain[1:])], dtype=np.float64)))
            for visit in ordered:
                if visit['visit_id'] not in self.visits:
                    finish(); chain = []
                    breaks.append({'visit_id': visit['visit_id'], 'reason': 'unusable'})
                    continue
                if chain and gap_weeks(chain[-1]['visit'], visit['visit']) <= 0:
                    finish(); chain = []
                    breaks.append({'visit_id': visit['visit_id'], 'reason': 'nonpositive_interval'})
                chain.append(visit)
            finish()
        return tuple(chains), {'singleton_visit_ids': singletons, 'breaks': breaks,
                              'source_patient_ids': self.splits['source52']}

    def control_fit_data(self, rows: tuple[ForecastRow, ...], anatomy: dict | None = None):
        from .baselines import ControlFitData
        chains = self.source_chains(anatomy)[0] if anatomy is not None else ()
        outgoing = np.array([gap_weeks(self.visits[r.cutoff_visit_id]['visit']['visit'],
                                      self.visits[r.target_visit_id]['visit']['visit']) for r in rows],
                            dtype=np.float64) if anatomy is not None else np.empty(0, dtype=np.float64)
        return ControlFitData(self.controls(rows), np.array([r.truth for r in rows], dtype=np.float64),
            np.array([r.patient_id for r in rows], dtype='U'),
            np.array([r.row_id for r in rows], dtype='U'), chains, outgoing)

    def verify_all_bytes(self, meta: dict) -> None:
        if meta['input_sha256'] != self.payload['input_sha256']:
            raise ValueError('source inputs changed since preparation')
        verify_raw_masks(meta)
        for record in self.visits.values():
            for kind in ('image', 'mask'):
                if digest(self.root / record[f'{kind}_file']) != record[f'{kind}_sha256']:
                    raise ValueError('prepared artifact changed')
            if any(digest(path) != sha for path, sha in record['raw_mri'].items()):
                raise ValueError('raw MRI changed since preparation')
