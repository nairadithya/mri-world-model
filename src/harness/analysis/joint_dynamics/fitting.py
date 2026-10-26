from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import math
import os
from pathlib import Path
import random
import time
from typing import Iterator
import warnings

import numpy as np
from scipy.special import ndtr
import torch

from .data import PreparedData
from .model import JointDynamics, binary_dice, configure_torch, finite_step, losses
from .protocol import (ARMS, CALIBRATION_GRID, CONDITIONS, PLAN, Arm, ForecastRow,
                       SourceIdentity, canonical_hash, digest, read_json, save_json)

RESERVES = {'numerical_review': (1800., 600.), 'documentation': (900., 0.),
            'contingency': (900., 300.)}
MEASURED_PHASES = ('calibration_controls', 'source_cohort_mask_inference',
                   'nuisance_inference', 'serialization', 'assessment')


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_hash(state: dict[str, torch.Tensor]) -> str:
    content = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        content.update(name.encode())
        content.update(str(value.dtype).encode())
        content.update(str(tuple(value.shape)).encode())
        content.update(value.view(torch.uint8).numpy().tobytes())
    return content.hexdigest()


def replacement_order(rows: tuple[ForecastRow, ...], seed: int) -> tuple[tuple[str, ...], ...]:
    by_patient = {p: tuple(r.row_id for r in rows if r.patient_id == p)
                  for p in sorted({r.patient_id for r in rows})}
    if not by_patient or len({r.row_id for r in rows}) != len(rows):
        raise ValueError('empty/duplicate training rows')
    rng = np.random.default_rng(seed)
    patients = tuple(by_patient)
    return tuple(tuple(by_patient[p][int(rng.integers(len(by_patient[p])))]
                 for p in (patients[int(rng.integers(len(patients)))]
                           for _ in range(PLAN.updates_per_epoch))) for _ in range(PLAN.epochs))


def make_model(data: PreparedData) -> JointDynamics:
    return JointDynamics(data.normalizer['mean'], data.normalizer['scale']).cuda()


def make_optimizer(model: JointDynamics) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)


@contextmanager
def warning_receipt(path: Path) -> Iterator[None]:
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('once')
        try:
            yield
        finally:
            unique = {(w.category.__name__, str(w.message), w.filename, w.lineno) for w in recorded}
            save_json(path, [{'category': k, 'message': msg, 'filename': file, 'line': line}
                             for k, msg, file, line in sorted(unique)])


def train_step(data: PreparedData, row: ForecastRow, arm: Arm, model: JointDynamics,
               optimizer: torch.optim.Optimizer) -> dict:
    optimizer.zero_grad(set_to_none=True)
    inputs, current, future = data.inputs(row, arm), data.target(row, current=True), data.target(row)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        prediction = model(inputs, arm)
        parts = losses(prediction, current, future, model.volume_mean, model.volume_scale)
    if not all(torch.isfinite(t).all() for t in (prediction.mean, prediction.intrinsic_scale,
                   prediction.current_mean, prediction.current_logits, prediction.future_logits)):
        raise ValueError('nonfinite training prediction')
    norms = finite_step(model, optimizer, parts['total'])
    return {'loss': {k: float(v.detach()) for k, v in parts.items()}, 'gradient_norm_before_clip': norms,
            'current_supervision_visit_id': row.cutoff_visit_id, 'earlier_state_supervision': 0,
            'future_image_forwards': 0}


def _prediction_values(prediction) -> dict[str, np.ndarray]:
    return {key: getattr(prediction, key).detach().float().cpu().numpy()
            for key in ('mean', 'intrinsic_scale', 'current_mean', 'current_standardized',
                        'current_logits', 'future_logits')}


def prediction_roundtrip(model: JointDynamics, data: PreparedData, row: ForecastRow,
                         arm: Arm, out: Path, optimizer: torch.optim.Optimizer) -> dict:
    model.eval()
    inputs = data.inputs(row, arm)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        before = _prediction_values(model(inputs, arm))
    payload = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
               'arm': arm.name, 'normalizer': data.normalizer}
    torch.save(payload, out)
    saved = torch.load(out, map_location='cpu', weights_only=False)
    restored = make_model(data)
    restored.load_state_dict(saved['state_dict'], strict=True)
    restored_optimizer = make_optimizer(restored)
    restored_optimizer.load_state_dict(saved['optimizer'])
    restored.eval()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        after = _prediction_values(restored(inputs, arm))
    if any(not np.isfinite(v).all() or not np.array_equal(v, after[k]) for k, v in before.items()):
        raise ValueError('real-data exact prediction save/load failed')
    result = {'exact': True, 'fields': list(before), 'checkpoint_sha256': digest(out),
              'tensor_sha256': tensor_hash(saved['state_dict']), 'optimizer_roundtrip': True}
    del restored, restored_optimizer
    model.train()
    return result


def preflight(data: PreparedData, out: Path, guard: object) -> dict:
    configure_torch()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('fixed BF16 CUDA policy is unavailable; no precision fallback')
    out.mkdir(parents=True, exist_ok=False)
    rows = data.select('source52')
    max_history = max(len(r.history_visit_ids) for r in rows)
    if max_history != 4:
        raise ValueError('source-only maximum-history fixture is not four visits')
    row = sorted((r for r in rows if len(r.history_visit_ids) == 4), key=lambda r: r.row_id)[0]
    cells, roundtrips = {}, {}
    warmup, samples = 2, 5
    with warning_receipt(out / 'warnings.json'):
        for arm in ARMS:
            seed_all(42)
            model = make_model(data)
            optimizer = make_optimizer(model)
            initial = tensor_hash(model.state_dict())
            torch.cuda.reset_peak_memory_stats()
            times, cpu_times = [], []
            for index in range(warmup + samples):
                guard.check()
                torch.cuda.synchronize()
                start, cpu = time.monotonic(), time.process_time()
                step = train_step(data, row, arm, model, optimizer)
                torch.cuda.synchronize()
                elapsed, cpu_elapsed = time.monotonic() - start, time.process_time() - cpu
                if index >= warmup:
                    times.append(elapsed); cpu_times.append(cpu_elapsed)
                if index == 0:
                    if initial == tensor_hash(model.state_dict()):
                        raise ValueError('finite real backward did not change parameters')
                    roundtrips[arm.name] = prediction_roundtrip(model, data, row, arm,
                                                       out / f'{arm.name}.roundtrip.pt', optimizer)
            cells[arm.name] = {'step_wall_seconds': times, 'step_cpu_seconds': cpu_times,
                'warmup_steps': warmup, 'samples': samples, 'failure_count': 0,
                'peak_device_memory_bytes': torch.cuda.max_memory_allocated(), 'last_step': step,
                'initial_tensor_sha256': initial}
            if arm.name == 'OH':
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                       torch.profiler.ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
                    train_step(data, row, arm, model, optimizer)
                    torch.cuda.synchronize()
                prof.export_chrome_trace(str(out / 'profiler.json'))
                (out / 'profiler.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=40))
            del model, optimizer
            torch.cuda.empty_cache()
    if len({c['initial_tensor_sha256'] for c in cells.values()}) != 1:
        raise ValueError('preflight initializations differ between arms')
    result = {'status': 'engineering_pass_not_fit_approval', 'source_fixture_row_id': row.row_id,
        'history_visit_ids': list(row.history_visit_ids), 'cells': cells, 'roundtrip': roundtrips,
        'warmup': warmup, 'timing_samples': samples, 'profiler_samples': 1,
        'loadavg': list(os.getloadavg()), 'torch_threads': torch.get_num_threads(),
        'torch_version': torch.__version__, 'device': torch.cuda.get_device_name(),
        'precision': PLAN.precision, 'deterministic_policy': PLAN.deterministic,
        'preflight_states_discarded': True}
    save_json(out / 'preflight.json', result)
    return result


def verify_resource_plan(plan: dict, preflight_record: dict, authority: object,
                         source_identity: dict) -> dict:
    if (plan.get('source_sha256') != source_identity['source_sha256']
            or plan.get('packet_sha256') != source_identity['packet_sha256']
            or plan.get('total_updates') != 62880 or plan.get('batch_size') != 1
            or plan.get('preflight_sha256') != canonical_hash(preflight_record)
            or not plan.get('supervisor_decision_sha256')):
        raise ValueError('resource plan is not bound to this full fixed workload')
    snapshot = plan['authority']
    phases = plan['phases']
    if set(phases) != {'updates', *MEASURED_PHASES, *RESERVES}:
        raise ValueError('whole remaining-workload projection has missing or extra phases')
    totals = {'cpu_seconds': 0., 'gpu_seconds': 0.}
    for name, phase in phases.items():
        if name in RESERVES:
            cpu, gpu = RESERVES[name]
            if (phase.get('kind') != 'hard_reserve' or phase.get('cpu_seconds') != cpu
                    or phase.get('gpu_seconds') != gpu):
                raise ValueError('fixed hard reserves cannot be reduced or double counted')
        else:
            factor = phase.get('uncertainty_factor', 0)
            if (phase.get('kind') != 'measured_extrapolation'
                    or not isinstance(factor, (int, float)) or not math.isfinite(factor) or factor < 1
                    or not phase.get('work_counts') or not phase.get('evidence')):
                raise ValueError(f'unsupported measured phase estimate: {name}')
            for evidence in phase['evidence']:
                stage = [s for s in snapshot['stages'] if s['name'] == evidence['stage_name']]
                if (len(stage) != 1 or stage[0]['status'] != 'succeeded'
                        or stage[0]['scientific'] or not all(isinstance(stage[0][k], (int, float))
                        and math.isfinite(stage[0][k]) and stage[0][k] >= 0
                        for k in ('cpu_seconds', 'gpu_seconds'))
                        or digest(evidence['path']) != evidence['sha256']):
                    raise ValueError('resource evidence is not an intact successful charged engineering stage')
            if name == 'updates':
                factor = float(phase['uncertainty_factor'])
                cells = preflight_record['cells']
                if set(cells) != {a.name for a in ARMS}:
                    raise ValueError('complete-step timing evidence must include all four arms')
                for cell in cells.values():
                    for key in ('step_wall_seconds', 'step_cpu_seconds'):
                        values = cell[key]
                        if (len(values) < 5 or not all(isinstance(v, (int, float))
                                and math.isfinite(v) and v >= 0 for v in values)):
                            raise ValueError('missing finite synchronized complete-step samples')
                expected_gpu = sum(max(cells[a.name]['step_wall_seconds']) * 15720 for a in ARMS) * factor
                expected_cpu = sum(max(cells[a.name]['step_cpu_seconds']) * 15720 for a in ARMS) * factor
                if (phase['gpu_seconds'] < expected_gpu or phase['cpu_seconds'] < expected_cpu
                        or phase['work_counts'] != {'updates_per_cell': 5240, 'cells': 12}):
                    raise ValueError('update projection undercounts complete measured steps')
            elif phase.get('information_scope') not in ('source_only', 'production_shaped_synthetic'):
                raise ValueError('preflight timing evidence must not use assessment outcomes')
            if name == 'assessment':
                wall = phase.get('wall_seconds')
                if not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall <= 0:
                    raise ValueError('assessment requires an explicit supported wall-time cap')
        for key in totals:
            value = phase[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('invalid resource upper bound')
            totals[key] += value
    return {'cpu_seconds': totals['cpu_seconds'], 'gpu_seconds': totals['gpu_seconds'],
            'resource_plan_sha256': canonical_hash(plan), 'already_charged_preparation': True,
            'hard_reserve_cpu_seconds': 3600., 'hard_reserve_gpu_seconds': 900.}


class WorkBudget:
    """Enforce phase caps as well as global protection of all not-yet-run work."""
    def __init__(self, guard: object, resource_plan: dict):
        self.guard, self.plan = guard, resource_plan
        self.pending = set(resource_plan['phases'])
        self.active = None

    def check(self, remaining_updates: int | None = None) -> None:
        cpu, gpu = 0., 0.
        for name in self.pending:
            phase = self.plan['phases'][name]
            factor = remaining_updates / 62880 if name == 'updates' and remaining_updates is not None else 1.
            cpu += phase['cpu_seconds'] * factor
            gpu += phase['gpu_seconds'] * factor
        self.guard.check(projected_cpu_seconds=cpu, projected_gpu_seconds=gpu)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if name not in self.pending or self.active is not None:
            raise ValueError('duplicate or nested workload phase')
        self.check()
        self.active = name
        torch.cuda.synchronize()
        start, cpu0 = time.monotonic(), time.process_time()
        try:
            yield
            torch.cuda.synchronize()
            cap = self.plan['phases'][name]
            if time.process_time() - cpu0 > cap['cpu_seconds'] or time.monotonic() - start > cap['gpu_seconds']:
                raise TimeoutError(f'INCOMPLETE_NOT_GRADED: phase cap exceeded: {name}')
            self.pending.remove(name)
        finally:
            self.active = None


def patient_weights(patients: np.ndarray) -> np.ndarray:
    names, counts = np.unique(patients.astype(str), return_counts=True)
    table = dict(zip(names, counts))
    return np.array([1 / (len(names) * table[str(p)]) for p in patients], dtype=np.float64)


def calibration_scores(mean: np.ndarray, intrinsic: np.ndarray, truth: np.ndarray,
                       patients: np.ndarray) -> list[float]:
    weights = patient_weights(patients)
    scores = []
    for multiplier in CALIBRATION_GRID:
        sigma = np.sqrt((multiplier * intrinsic) ** 2 + 0.05 ** 2)
        z = (truth - mean) / sigma
        crps = sigma * (z * (2 * ndtr(z) - 1) + 2 * np.exp(-0.5 * z ** 2) /
                         math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi))
        scores.append(float(np.sum(weights * crps.mean(1))))
    return scores


def evaluate(model: JointDynamics, data: PreparedData, rows: tuple[ForecastRow, ...], arm: Arm,
             guard: object, condition: str = 'identity', *, masks: bool = True) -> dict[str, np.ndarray]:
    result = {k: [] for k in ('mean', 'intrinsic_scale', 'current_mean')}
    if masks:
        result.update(current_dice=[], future_dice=[])
    model.eval()
    with torch.inference_mode():
        for row in rows:
            guard.check()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(data.inputs(row, arm, condition), arm)
            for key in ('mean', 'intrinsic_scale', 'current_mean'):
                result[key].append(getattr(prediction, key)[0].float().cpu().numpy())
            if masks:
                result['current_dice'].append(binary_dice(prediction.current_logits,
                                       data.target(row, current=True).masks)[0].cpu().numpy())
                result['future_dice'].append(binary_dice(prediction.future_logits,
                                       data.target(row).masks)[0].cpu().numpy())
    arrays = {k: np.asarray(v, dtype=np.float64) for k, v in result.items()}
    if any(v.shape != (len(rows), 3) or not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('incomplete/nonfinite native predictions or mask companions')
    return arrays


def _record(path: Path, root: Path) -> dict:
    return {'path': str(path.relative_to(root)), 'sha256': digest(path)}


def _shared_rows(arrays: dict, name: str, rows: tuple[ForecastRow, ...]) -> list[dict]:
    arrays[f'{name}_row_ids'] = np.array([r.row_id for r in rows], dtype='U')
    arrays[f'{name}_patient_ids'] = np.array([r.patient_id for r in rows], dtype='U')
    arrays[f'{name}_truth'] = np.array([r.truth for r in rows], dtype=np.float64)
    arrays[f'{name}_cutoff_measured'] = np.array([r.cutoff_measured for r in rows], dtype=np.float64)
    return [{'row_id': r.row_id, 'patient_id': r.patient_id, 'cutoff_visit_id': r.cutoff_visit_id,
             'target_visit_id': r.target_visit_id, 'truth': list(r.truth),
             'cutoff_measured': list(r.cutoff_measured)} for r in rows]


def fit_matrix(data: PreparedData, out: Path, contract: dict, guard: object,
               anatomy: dict, *, canary: bool = False) -> dict:
    from .baselines import (ControlFits, calibrate_gaussian, fit_controls, fit_persistence,
                            gaussian_calibration_receipt, predict_controls)
    configure_torch()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('fixed CUDA BF16 unavailable')
    out.mkdir(parents=True, exist_ok=False)
    contract_digest = canonical_hash(contract)
    budget = guard
    source = SourceIdentity.from_dict(contract['implementation_source'])
    scope = 'engineering_canary' if canary else 'fixed_r38_study'
    if contract['execution_scope'] != scope:
        raise ValueError('training scope differs from the bound execution contract')
    epochs, per_epoch = (1, 7) if canary else (PLAN.epochs, PLAN.updates_per_epoch)
    per_cell = epochs * per_epoch
    expected_updates = len(ARMS) * len(PLAN.seeds) * per_cell
    training, validation = data.select('source52'), data.select('validation13')
    fixture = min((r for r in training if len(r.history_visit_ids) == 4), key=lambda r: r.row_id)
    measurements, roundtrips = {}, {}
    cohorts = {k: data.select(k) for k in ('primary', 'single', 'nuisance')}
    arrays = {'seeds': np.array(PLAN.seeds, dtype=np.int64)}
    cohort_rows = {name: _shared_rows(arrays, name, rows) for name, rows in cohorts.items()}
    source_identity = {'contract_digest': contract_digest, 'implementation_source': source.to_dict(),
                       'execution_scope': scope,
                       'row_manifest_sha256': canonical_hash(data.payload['rows'])}
    cells, calibration, orders, initializations = {}, {}, {}, {}
    for seed in PLAN.seeds:
        seed_all(seed)
        model = make_model(data)
        initial = out / f'initialization_{seed}.pt'
        torch.save({'state_dict': model.state_dict(), **source_identity, 'seed': seed}, initial)
        initializations[str(seed)] = {**_record(initial, out), 'tensor_sha256': tensor_hash(model.state_dict())}
        order = ((fixture.row_id,) * per_epoch,) if canary else replacement_order(training, seed)
        order_file = out / f'order_{seed}.json'
        save_json(order_file, {'seed': seed, 'orders': order, **source_identity})
        orders[seed] = order
        del model
    with warning_receipt(out / 'warnings.json'):
        with budget.phase('updates'):
            successful = 0
            for seed in PLAN.seeds:
                saved = torch.load(out / f'initialization_{seed}.pt', map_location='cpu', weights_only=False)
                for arm in ARMS:
                    label = f'{arm.name}_{seed}'
                    directory = out / label
                    directory.mkdir()
                    seed_all(seed)
                    model = make_model(data)
                    model.load_state_dict(saved['state_dict'], strict=True)
                    if tensor_hash(model.state_dict()) != initializations[str(seed)]['tensor_sha256']:
                        raise ValueError('cell initialization differs from matched seed initialization')
                    optimizer = make_optimizer(model)
                    torch.cuda.reset_peak_memory_stats()
                    update_file = directory / 'updates.jsonl'
                    measurements[label] = []
                    with update_file.open('w') as stream:
                        for epoch, epoch_order in enumerate(orders[seed], 1):
                            for update, row_id in enumerate(epoch_order, 1):
                                budget.check(remaining_updates=expected_updates - successful)
                                row = data.by_row[row_id]
                                if canary and update > 2:
                                    with budget.sample(f'{label}/update', synchronize=torch.cuda.synchronize):
                                        t0, cpu0 = time.monotonic(), time.process_time()
                                        step = train_step(data, row, arm, model, optimizer)
                                        budget.completed(f'{label}/update')
                                        torch.cuda.synchronize()
                                        measurements[label].append({'wall_seconds': time.monotonic() - t0,
                                            'cpu_seconds': time.process_time() - cpu0})
                                else:
                                    step = train_step(data, row, arm, model, optimizer)
                                    budget.completed(f'{label}/update')
                                successful += 1
                                record = {'epoch': epoch, 'update': update, 'row_id': row_id,
                                    'cutoff_mask_sha256': data.visits[row.cutoff_visit_id]['mask_sha256'],
                                    **step}
                                import json
                                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
                            stream.flush(); os.fsync(stream.fileno())
                    checkpoint = directory / f'epoch{epochs}.pt'
                    torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'epoch': epochs, 'updates': per_cell, 'arm': arm.name, 'seed': seed,
                        'initialization': initializations[str(seed)],
                        'order_sha256': digest(out / f'order_{seed}.json'), 'normalizer': data.normalizer,
                        **source_identity}, checkpoint)
                    if canary:
                        if tensor_hash(model.state_dict()) == initializations[str(seed)]['tensor_sha256']:
                            raise ValueError('actual-data updates did not change parameters')
                        roundtrips[label] = prediction_roundtrip(model, data, fixture, arm,
                            directory / 'roundtrip.pt', optimizer)
                        roundtrips[label]['peak_device_memory_bytes'] = torch.cuda.max_memory_allocated()
                    cells[label] = {'epoch': epochs, 'updates': per_cell, **source_identity,
                        'initialization': initializations[str(seed)],
                        'weights': _record(checkpoint, out), 'updates_file': _record(update_file, out)}
                    del model, optimizer
                    torch.cuda.empty_cache()
            if successful != expected_updates:
                raise ValueError('incomplete successful-update matrix')
        with budget.phase('source_cohort_mask_inference'):
            for seed in PLAN.seeds:
                for arm in ARMS:
                    label = f'{arm.name}_{seed}'
                    model = _restore_cell(data, out, cells[label], contract_digest)
                    source = evaluate(model, data, training, arm, guard, masks=False)
                    valid = evaluate(model, data, validation, arm, guard, masks=False)
                    raw = {}
                    for name, rows, prediction in (('source', training, source), ('validation', validation, valid)):
                        _shared_rows(raw, name, rows)
                        raw.update({f'{name}_{k}': v for k, v in prediction.items()})
                    np.savez_compressed(out / label / 'calibration_inputs.npz', **raw)
                    for name in ('primary', 'single'):
                        prediction = evaluate(model, data, cohorts[name], arm, guard)
                        arrays.update({f'{name}_{label}_{k}': v for k, v in prediction.items()})
                    del model
                    torch.cuda.empty_cache()
        with budget.phase('calibration_controls'):
            source_data = data.control_fit_data(training, anatomy)
            validation_data = data.control_fit_data(validation)
            controls = fit_controls(source_data, validation_data)
            control_record = controls.to_dict()
            controls = ControlFits.from_dict(control_record)
            if controls.to_dict() != control_record:
                raise ValueError('control envelope does not roundtrip exactly')
            save_json(out / 'control_fits.json', {'fits': control_record, **source_identity,
                      'source_row_ids': [r.row_id for r in training],
                      'validation_row_ids': [r.row_id for r in validation],
                      'chains': data.source_chains(anatomy)[1]})
            for name in ('primary', 'single'):
                predictions = predict_controls(controls, data.controls(cohorts[name]))
                if set(predictions) != {'P', 'E', 'FP', 'SSM'}:
                    raise ValueError('missing common controls')
                for control, values in predictions.items():
                    if control == 'E':
                        for key in ('centers', 'weights', 'observation_sd'):
                            arrays[f'{name}_E_{key}'] = np.asarray(values[key])
                    else:
                        for key in ('mean', 'intrinsic_scale', 'calibration'):
                            arrays[f'{name}_{control}_{key}'] = np.asarray(values[key])
                    if name == 'primary':
                        calibration[control] = float(control_record['state']['calibration'][control])
            for seed in PLAN.seeds:
                for arm in ARMS:
                    label = f'{arm.name}_{seed}'
                    with np.load(out / label / 'calibration_inputs.npz', allow_pickle=False) as raw:
                        forecast_scores = calibration_scores(raw['validation_mean'], raw['validation_intrinsic_scale'],
                                                             raw['validation_truth'], raw['validation_patient_ids'])
                        forecast = calibrate_gaussian(raw['validation_mean'], raw['validation_intrinsic_scale'],
                                                      raw['validation_truth'], raw['validation_patient_ids'])
                        rms = fit_persistence(raw['source_current_mean'], raw['source_truth'], raw['source_patient_ids'])
                        repeated = np.broadcast_to(rms, raw['validation_current_mean'].shape)
                        persistence_scores = calibration_scores(raw['validation_current_mean'], repeated,
                                                               raw['validation_truth'], raw['validation_patient_ids'])
                        persistence = calibrate_gaussian(raw['validation_current_mean'], repeated,
                                                        raw['validation_truth'], raw['validation_patient_ids'])
                        forecast_receipt = gaussian_calibration_receipt(raw['validation_mean'],
                            raw['validation_intrinsic_scale'], raw['validation_truth'], raw['validation_patient_ids'])
                        persistence_receipt = gaussian_calibration_receipt(raw['validation_current_mean'],
                            repeated, raw['validation_truth'], raw['validation_patient_ids'])
                        if (forecast != CALIBRATION_GRID[int(np.argmin(forecast_scores))]
                                or persistence != CALIBRATION_GRID[int(np.argmin(persistence_scores))]):
                            raise ValueError('calibration selection differs from bound full-grid scores')
                    receipt = {**source_identity, 'arm': arm.name, 'seed': seed, 'epoch': epochs,
                        'inputs': _record(out / label / 'calibration_inputs.npz', out),
                        'source_row_ids': [r.row_id for r in training], 'source_patient_ids': [r.patient_id for r in training],
                        'validation_row_ids': [r.row_id for r in validation],
                        'validation_patient_ids': [r.patient_id for r in validation],
                        'forecast_receipt': forecast_receipt, 'persistence_receipt': persistence_receipt,
                        'grid': list(CALIBRATION_GRID), 'forecast': float(forecast),
                        'forecast_grid_scores': forecast_scores, 'persistence': float(persistence),
                        'persistence_grid_scores': persistence_scores, 'persistence_rms': np.asarray(rms).tolist()}
                    save_json(out / label / 'calibration.json', receipt)
                    cells[label]['calibration_file'] = _record(out / label / 'calibration.json', out)
                    calibration[label] = {k: receipt[k] for k in ('forecast', 'persistence', 'persistence_rms')}
                    for name in ('primary', 'single'):
                        arrays[f'{name}_{label}_calibration'] = np.asarray(forecast)
                        arrays[f'{name}_{label}_persistence_rms'] = np.asarray(rms)
                        arrays[f'{name}_{label}_persistence_calibration'] = np.asarray(persistence)
        with budget.phase('nuisance_inference'):
            for seed in PLAN.seeds:
                for arm in ARMS:
                    label = f'{arm.name}_{seed}'
                    model = _restore_cell(data, out, cells[label], contract_digest)
                    for condition in CONDITIONS:
                        prediction = evaluate(model, data, cohorts['nuisance'], arm, guard, condition, masks=False)
                        prefix = f'nuisance_{arm.name}_{condition}_{seed}'
                        arrays.update({f'{prefix}_{k}': v for k, v in prediction.items()})
                        arrays[f'{prefix}_calibration'] = np.asarray(calibration[label]['forecast'])
                    del model
                    torch.cuda.empty_cache()
        with budget.phase('serialization'):
            for key, values in arrays.items():
                if values.dtype.hasobject or (values.dtype.kind in 'fc' and not np.isfinite(values).all()):
                    raise ValueError(f'object/nonfinite prediction array: {key}')
            np.savez_compressed(out / 'private_predictions.npz', **arrays)
            with np.load(out / 'private_predictions.npz', allow_pickle=False) as saved:
                if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
                    raise ValueError('prediction serialization roundtrip mismatch')
            fitted = {**source_identity, 'epochs': epochs, 'updates_per_epoch': per_epoch,
                'seeds': list(PLAN.seeds), 'total_updates': successful, 'cells': cells,
                'timing_samples': measurements, 'checkpoint_roundtrips': roundtrips,
                'runtime': {'loadavg': list(os.getloadavg()), 'torch_threads': torch.get_num_threads(),
                    'torch_version': torch.__version__, 'device': torch.cuda.get_device_name(),
                    'precision': PLAN.precision, 'deterministic_policy': PLAN.deterministic},
                'canary_fixture_row_id': fixture.row_id if canary else None,
                'initializations': initializations, 'calibration': calibration, 'cohort_rows': cohort_rows,
                'control_fits': control_record, 'control_fits_sha256': digest(out / 'control_fits.json'),
                'prediction_sha256': digest(out / 'private_predictions.npz'),
                'orders': {str(seed): _record(out / f'order_{seed}.json', out) for seed in PLAN.seeds}}
            save_json(out / 'fitted.json', fitted)
    budget.check()
    torch.cuda.synchronize()
    return fitted


def _restore_cell(data: PreparedData, out: Path, cell: dict, contract_digest: str) -> JointDynamics:
    record = cell['weights']
    if digest(out / record['path']) != record['sha256']:
        raise ValueError('changed checkpoint before prediction')
    payload = torch.load(out / record['path'], map_location='cpu', weights_only=False)
    schedule = {'engineering_canary': (1, 7), 'fixed_r38_study': (20, 5240)}
    expected = schedule.get(cell.get('execution_scope'))
    if (expected is None or (cell['epoch'], cell['updates']) != expected
            or payload.get('contract_digest') != contract_digest
            or (payload.get('epoch'), payload.get('updates')) != expected
            or payload.get('execution_scope') != cell['execution_scope']
            or payload.get('normalizer') != data.normalizer):
        raise ValueError('mismatched checkpoint identity, selection or normalization')
    model = make_model(data)
    model.load_state_dict(payload['state_dict'], strict=True)
    return model
