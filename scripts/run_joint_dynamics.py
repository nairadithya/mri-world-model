#!/usr/bin/env python3
"""Run the fixed R38 study or its explicitly engineering-only actual-data canary."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.harness.analysis.joint_dynamics.protocol import (
    PLAN, PLANNING, REPO, WORKSPACE, ArtifactRef, Cost, SourceIdentity, StageLimits,
    StageRequest, canonical_hash, digest, read_json, save_json)

STUDY = WORKSPACE / 'study'
STAGES = STUDY / 'stages'
FIT_PHASES = ('updates', 'source_cohort_mask_inference', 'calibration_controls',
              'nuisance_inference', 'serialization')


def require_remote() -> None:
    if ROOT != REPO or Path(sys.executable).absolute() != WORKSPACE / '.venv/bin/python':
        raise RuntimeError('use only the fixed remote R38 repo and Python; no local computation')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'


@contextmanager
def runner_lock():
    STUDY.mkdir(parents=True, exist_ok=True)
    with (STUDY / 'runner.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('another R38 runner owns stage transitions') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def stage_path(name: str) -> Path:
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in name):
        raise ValueError('stage name must be a single safe identity')
    return STAGES / name


def completion(name: str) -> dict:
    path = stage_path(name)
    value = read_json(path.with_name(path.name + '.control') / 'complete.json')
    if value['schema'] != 'r38-finalized-stage-v4' or value['stage_name'] != name:
        raise ValueError('stage completion identity mismatch')
    return value


def _caps(cpu: float, wall: float, gpu: bool) -> Cost:
    return Cost(cpu, wall, wall if gpu else 0.)


def _cleanup(gpu: bool) -> Cost:
    return _caps(max(300., 3. * (os.cpu_count() or 1)), 8., gpu)


def run_stage(authority, name: str, operation: str, payload: dict, phases: dict,
              *, gpu: bool, scientific: bool, future: Cost | None = None, ancestors=()):
    from src.harness.analysis.joint_dynamics.execution import execute_stage
    for ancestor in ancestors:
        authority.verify_extension(ancestor['authority'])
    request_path = STUDY / 'requests' / name / 'payload.json'
    request_path.parent.mkdir(parents=True, exist_ok=False)
    save_json(request_path, payload)
    refs = [ArtifactRef(request_path, digest(request_path)),
            ArtifactRef(Path(payload['bindings']), digest(payload['bindings']))]
    for ancestor in ancestors:
        path = stage_path(ancestor['stage_name'])
        marker = path.with_name(path.name + '.control') / 'complete.json'
        refs.append(ArtifactRef(marker, digest(marker)))
    snapshot = authority.snapshot()
    limits = StageLimits(gpu, scientific, phases,
        future if future is not None else Cost(3600., 3600., 900.), _cleanup(gpu),
        datetime.fromisoformat(snapshot['identity']['deadline']))
    request = StageRequest(name, operation, SourceIdentity.capture(), tuple(refs), limits, stage_path(name))
    finalized = execute_stage(authority, request)
    return dict(finalized.operation_result)


def execution_bindings() -> Path:
    path = STUDY / 'execution-input-bindings.json'
    original = read_json(PLANNING / 'input-bindings.json')
    original['known_sha256']['mask_transfer'] = 'd42259bbb47f6a9926892d1203dce3300cdee098fee8bbf1109da70482274666'
    original['execution_note'] = 'Add the diagnosed original manifest pin; the frozen v3 input packet is preserved unchanged.'
    if path.exists():
        if read_json(path) != original:
            raise ValueError('execution bindings changed; no replacement')
    else:
        save_json(path, original)
    return path


def prepare(authority, name: str) -> dict:
    if any(s['scientific'] for s in authority.snapshot()['stages']):
        raise RuntimeError('scientific work already started; preparation cannot restart')
    return run_stage(authority, name, 'prepare',
        {'purpose': 'prepare', 'bindings': str(execution_bindings())},
        {'startup': _caps(300., 60., True), 'preparation': _caps(3600., 1200., True),
         'teardown': _caps(300., 90., True)}, gpu=True, scientific=False,
        future=Cost(10000., 5500., 2400.))


def make_contract(prepared: str, *, canary: bool) -> dict:
    return {'schema': 'r38-concrete-execution-v4',
        'execution_scope': 'engineering_canary' if canary else 'fixed_r38_study',
        'implementation_source': SourceIdentity.capture().to_dict(),
        'scientific_plan': json.loads(json.dumps(asdict(PLAN))),
        'protocol_v3_sha256': digest(PLANNING / 'protocol-v3.md'),
        'preparation': prepared, 'preparation_completion_sha256': canonical_hash(completion(prepared)),
        'bindings_sha256': digest(execution_bindings()),
        'selection': 'explicit seven-update source-only max-history engineering fixture' if canary
                     else 'fixed epoch20; all12 cells x5240 updates; no selection or rescue'}


def score(authority, fit_name: str, name: str, *, canary: bool, cpu: float, wall: float) -> dict:
    fitted_stage = completion(fit_name)
    fitted_payload = read_json(STUDY / 'requests' / fit_name / 'payload.json')
    if fitted_stage['operation_result']['execution_scope'] != fitted_payload['contract']['execution_scope']:
        raise ValueError('fit completion scope differs')
    return run_stage(authority, name, 'verify' if canary else 'assess',
        {**fitted_payload, 'purpose': 'canary_score' if canary else 'study_score',
         'fit': str(stage_path(fit_name))},
        {'startup': _caps(300., 60., False), 'assessment': _caps(cpu, wall, False),
         'teardown': _caps(300., 60., False)}, gpu=False, scientific=not canary,
        ancestors=(completion(fitted_payload['prepared_stage']), fitted_stage))


def canary(authority, name: str, prepared: str) -> dict:
    if any(s['scientific'] for s in authority.snapshot()['stages']):
        raise RuntimeError('engineering canary cannot follow scientific work')
    phases = {'startup': _caps(300., 60., True), 'essential_checks': _caps(600., 180., True),
        'updates': _caps(600., 180., True),
        'source_cohort_mask_inference': _caps(900., 480., True),
        'calibration_controls': _caps(300., 90., True),
        'nuisance_inference': _caps(600., 180., True),
        'serialization': _caps(300., 60., True), 'teardown': _caps(600., 180., True)}
    contract = make_contract(prepared, canary=True)
    result = run_stage(authority, name, 'preflight', {'purpose': 'canary_fit',
        'bindings': str(execution_bindings()), 'prepared_stage': prepared,
        'preparation': str(stage_path(prepared)), 'contract': contract}, phases,
        gpu=True, scientific=False, future=Cost(5400., 4500., 900.), ancestors=(completion(prepared),))
    scored = score(authority, name, name + '-score', canary=True, cpu=900., wall=300.)
    return {'status': 'actual_data_canary_completed_not_scientific_evidence',
            'fit': result, 'score': scored,
            'remaining_authority': {k: authority.snapshot()[k] for k in
                ('charged_cpu_seconds', 'charged_gpu_seconds', 'remaining_cpu_seconds', 'remaining_gpu_seconds')}}


def projection(authority, name: str) -> dict:
    fit_stage, scored = completion(name), completion(name + '-score')
    if (fit_stage['operation_result']['actual_updates'] != 84
            or scored['operation_result']['status'] != 'engineering_canary_scored_not_scientific_evidence'):
        raise ValueError('a complete actual-data canary and real scoring are required')
    source = SourceIdentity.capture().to_dict()
    if fit_stage['request']['implementation_source'] != source or scored['request']['implementation_source'] != source:
        raise ValueError('source changed since the scored canary')
    authority.verify_extension(fit_stage['authority'])
    authority.verify_extension(scored['authority'])
    observations = read_json(fit_stage['observation'])
    score_observations = read_json(scored['observation'])
    samples = observations['samples']
    update_cpu, update_wall = 0., 0.
    for arm in PLAN.arms:
        values = [s['observed'] for s in samples if s['unit'].startswith(arm.name + '_')]
        if len(values) != 15:
            raise ValueError('five synchronized max-history samples per seed and arm are required')
        update_cpu += 15720 * max(v['cpu_seconds'] for v in values)
        update_wall += 15720 * max(v['wall_seconds'] for v in values)
    # Bounds, not promises: 1.5x the slowest observed per-arm step plus measured
    # initialization overhead; all other full-shaped phases get 2x plus headroom.
    phases = {'startup': _caps(600., 120., True)}
    observed = observations['phase_costs']
    for phase in FIT_PHASES:
        if phase == 'updates':
            cpu = 1.5 * update_cpu + 2 * observed[phase]['cpu_seconds'] + 150.
            wall = 1.5 * update_wall + 2 * observed[phase]['wall_seconds'] + 30.
        else:
            cpu = max(150., 2 * observed[phase]['cpu_seconds'] + 150.)
            wall = max(30., 2 * observed[phase]['wall_seconds'] + 15.)
        phases[phase] = _caps(cpu, wall, True)
    phases['teardown'] = _caps(600., 180., True)
    score_cost = score_observations['phase_costs']['assessment']
    assessment = _caps(max(300., 2 * score_cost['cpu_seconds'] + 150.),
                       max(90., 2 * score_cost['wall_seconds'] + 30.), False)
    future = Cost(4200. + assessment.cpu_seconds, 3720. + assessment.wall_seconds, 900.)
    cleanup = _cleanup(True)
    total_cpu = sum(c.cpu_seconds for c in phases.values()) + future.cpu_seconds + cleanup.cpu_seconds
    total_gpu = sum(c.gpu_seconds for c in phases.values()) + future.gpu_seconds + cleanup.gpu_seconds
    total_wall = sum(c.wall_seconds for c in phases.values()) + future.wall_seconds + cleanup.wall_seconds
    current = authority.snapshot()
    fits = (total_cpu < current['remaining_cpu_seconds'] and total_gpu < current['remaining_gpu_seconds']
        and datetime.now().timestamp() + total_wall < datetime.fromisoformat(current['identity']['deadline']).timestamp())
    return {'status': 'fits_fixed_workload' if fits else 'BLOCKED_FIXED_WORKLOAD_EXCEEDS_REMAINING_ALLOWANCE',
        'total_updates': PLAN.total_updates, 'phases': {k: v.to_dict() for k, v in phases.items()},
        'assessment': assessment.to_dict(), 'future': future.to_dict(),
        'required_cpu_with_reserves_cleanup': total_cpu, 'required_gpu_with_reserves_cleanup': total_gpu,
        'required_wall_with_reserves_cleanup': total_wall, 'authority': current,
        'evidence': {'fit_completion': canonical_hash(fit_stage), 'score_completion': canonical_hash(scored),
                     'observations_sha256': digest(fit_stage['observation']), 'canary_updates': 84,
                     'unit_samples': len(samples), 'inflation_update': 1.5, 'inflation_other': 2.}}


def fit(authority, canary_name: str, name: str) -> dict:
    if any(s['scientific'] for s in authority.snapshot()['stages']):
        raise RuntimeError('scientific work already started; no retry or rescue')
    bound = projection(authority, canary_name)
    save_json(STUDY / ('projection-' + name + '.json'), bound)
    if bound['status'] != 'fits_fixed_workload':
        return bound
    previous = read_json(STUDY / 'requests' / canary_name / 'payload.json')
    prepared = previous['prepared_stage']
    contract = make_contract(prepared, canary=False)
    result = run_stage(authority, name, 'fit', {**previous, 'purpose': 'study_fit', 'contract': contract},
        {k: Cost.from_dict(v) for k, v in bound['phases'].items()}, gpu=True, scientific=True,
        future=Cost.from_dict(bound['future']), ancestors=(completion(prepared), completion(canary_name),
                                                        completion(canary_name + '-score')))
    # The fixed study includes assessment; its CPU-only stage never changes GPU tagging.
    assessment = bound['assessment']
    report = score(authority, name, name + '-score', canary=False,
                   cpu=assessment['cpu_seconds'], wall=assessment['wall_seconds'])
    return {'status': 'fixed_study_executed_pending_scientific_interpretation', 'fit': result, 'score': report}


def perform_operation(request, work):
    from src.harness.analysis.joint_dynamics.operations import perform_operation as execute
    return execute(request, work)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    command = sub.add_parser('prepare')
    command.add_argument('--stage-name', required=True)
    command = sub.add_parser('canary')
    command.add_argument('--stage-name', required=True)
    command.add_argument('--prepared-stage', required=True)
    command = sub.add_parser('project')
    command.add_argument('--canary-stage', required=True)
    command = sub.add_parser('fit')
    command.add_argument('--stage-name', required=True)
    command.add_argument('--canary-stage', required=True)
    args = parser.parse_args(argv)
    try:
        require_remote()
        from src.harness.analysis.joint_dynamics.authority import production_authority
        authority = production_authority()
        authority.snapshot()
        with runner_lock():
            if args.mode == 'prepare':
                result = prepare(authority, args.stage_name)
            elif args.mode == 'canary':
                result = canary(authority, args.stage_name, args.prepared_stage)
            elif args.mode == 'project':
                result = projection(authority, args.canary_stage)
            else:
                result = fit(authority, args.canary_stage, args.stage_name)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0 if not result['status'].startswith('BLOCKED') else 2
    except Exception as error:
        print(f'INCOMPLETE_NOT_GRADED: {type(error).__name__}: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
