#!/usr/bin/env python3
"""One fixed-root R38 stage runner; computation is remote and authority-charged."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.harness.analysis.joint_dynamics.protocol import (
    PLAN, PLANNING, REPO, WORKSPACE, canonical_hash, digest, implementation_identity,
    read_json, save_json, seal_stage, source_hashes, validate_completed_fit,
    verify_approval, verify_stage)

STUDY = WORKSPACE / 'study'
STAGES = STUDY / 'stages'


def require_remote() -> None:
    if ROOT != REPO or Path(sys.executable).absolute() != WORKSPACE / '.venv/bin/python':
        raise RuntimeError('use only the fixed remote R38 repo and Python; no local computation')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


@contextmanager
def runner_lock() -> Iterator[None]:
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
        raise ValueError('stage name must be a single safe identity, not an output directory')
    return STAGES / name


def reserve_directory(name: str) -> Path:
    path = stage_path(name)
    path.mkdir(parents=True, exist_ok=False)
    save_json(path / 'started.json', {'stage_name': name, 'status': 'started_not_complete'})
    return path


@contextmanager
def charged_stage(authority, name: str, *, gpu: bool, scientific: bool,
                  prior_paths: tuple[Path, ...] = (), prior_snapshots: tuple[dict, ...] = ()):
    for path in prior_paths:
        authority.verify_extension(read_json(path / 'complete.json')['authority'])
    for snapshot in prior_snapshots:
        authority.verify_extension(snapshot)
    entry = authority.snapshot()
    with authority.stage(name, gpu=gpu, scientific=scientific) as guard:
        current = guard.snapshot()
        stages = current['stages']
        if (any(current[k] != entry[k] for k in
                ('authority_id', 'authorization_sha256', 'identity'))
                or len(stages) != len(entry['stages']) + 1
                or stages[:-1] != entry['stages']
                or stages[-1]['name'] != name or stages[-1]['status'] != 'running'
                or stages[-1]['scientific'] != scientific):
            raise RuntimeError('authority identity or exact stage prefix changed at entry')
        yield guard


def locked_paths() -> tuple[Path, Path]:
    path = stage_path(read_json(STUDY / 'locked.json')['stage'])
    contract = read_json(path / 'contract.json')
    return path, stage_path(contract['prepared_stage'])


def verify_counts(data) -> None:
    for name, rows, patients in PLAN.cohort_counts:
        selected = data.select(name)
        if len(selected) != rows or len({r.patient_id for r in selected}) != patients:
            raise ValueError(f'wrong ordered cohort: {name}')


def prepare(authority, name: str) -> dict:
    if (STUDY / 'locked.json').exists():
        raise RuntimeError('preparation cannot restart after scientific lock')
    path = reserve_directory(name)
    with charged_stage(authority, name, gpu=True, scientific=False) as guard:
        from src.harness.analysis.joint_dynamics.data import (
            PREPARATION_SOURCES, PreparedData, load_metadata, prepare_inputs)
        from src.harness.analysis.joint_dynamics.model import configure_torch
        configure_torch()
        bindings = read_json(PLANNING / 'input-bindings.json')
        meta = load_metadata(bindings)
        dependencies = {p: digest(REPO / p) for p in PREPARATION_SOURCES}
        result = prepare_inputs(meta, path / 'prepared', guard)
        data = PreparedData(path / 'prepared')
        verify_counts(data)
        data.verify_all_bytes(load_metadata(bindings))
        preparation_contract = {'preparation_source_sha256': dependencies,
            'bindings_sha256': digest(PLANNING / 'input-bindings.json'),
            'input_sha256': meta['input_sha256'], 'prepared_sha256': digest(path / 'prepared/prepared.json')}
        save_json(path / 'preparation_contract.json', preparation_contract)
        guard.check()
    # Complete markers are never emitted inside a reservation that can still fail.
    snapshot = authority.snapshot()
    seal_stage(path, 'prepare', canonical_hash(preparation_contract), snapshot)
    return {'status': 'prepared_not_fit_approved', 'stage': name,
            'contract_digest': canonical_hash(preparation_contract), 'reconstruction': result['reconstruction']}


def load_prepared(authority, name: str):
    from src.harness.analysis.joint_dynamics.data import PREPARATION_SOURCES, PreparedData, load_metadata
    path = stage_path(name)
    verify_stage(path, 'prepare')
    receipt = read_json(path / 'preparation_contract.json')
    if receipt['preparation_source_sha256'] != {p: digest(REPO / p) for p in PREPARATION_SOURCES}:
        raise ValueError('preparation dependency bytes changed')
    meta = load_metadata(read_json(PLANNING / 'input-bindings.json'))
    data = PreparedData(path / 'prepared')
    data.verify_all_bytes(meta)
    verify_counts(data)
    return data, meta


def preflight(authority, name: str, prepared_stage: str) -> dict:
    if (STUDY / 'locked.json').exists():
        raise RuntimeError('preflight cannot restart after scientific lock')
    path = reserve_directory(name)
    with charged_stage(authority, name, gpu=True, scientific=False,
                       prior_paths=(stage_path(prepared_stage),)) as guard:
        from src.harness.analysis.joint_dynamics.fitting import preflight as run_preflight
        identity = implementation_identity()
        data, meta = load_prepared(authority, prepared_stage)
        # These tests run remotely only, and all child CPU and stage GPU wall are charged.
        command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests',
                   '-p', 'test_joint_dynamics.py', '-v']
        completed = subprocess.run(command, cwd=REPO, text=True, capture_output=True)
        (path / 'tests.stdout').write_text(completed.stdout)
        (path / 'tests.stderr').write_text(completed.stderr)
        save_json(path / 'tests.json', {'command': command, 'returncode': completed.returncode,
                                      'identity': identity})
        if completed.returncode:
            raise RuntimeError('gradient/input/artifact verification tests failed')
        result = run_preflight(data, path / 'probe', guard)
        data.verify_all_bytes(meta)
        save_json(path / 'identity.json', identity)
        save_json(path / 'inputs.json', {'prepared_stage': prepared_stage,
                  'preparation_marker_sha256': digest(stage_path(prepared_stage) / 'complete.json')})
        guard.check()
    seal_stage(path, 'preflight', canonical_hash(identity), authority.snapshot())
    return {'status': 'preflight_measured_pending_whole_budget_gate', 'stage': name,
            'preflight_sha256': canonical_hash(result)}


def lock(authority, name: str, prepared_stage: str, preflight_stage: str,
         approval_path: Path, resource_path: Path) -> dict:
    if (STUDY / 'locked.json').exists():
        raise FileExistsError('scientific contract already locked')
    path = reserve_directory(name)
    before = authority.snapshot()
    if any(s['scientific'] for s in before['stages']):
        raise RuntimeError('started scientific work prevents a new lock')
    resource_snapshot = read_json(resource_path)['authority']
    with charged_stage(authority, name, gpu=False, scientific=False,
                       prior_paths=(stage_path(prepared_stage), stage_path(preflight_stage)),
                       prior_snapshots=(resource_snapshot,)) as guard:
        from src.harness.analysis.joint_dynamics.fitting import verify_resource_plan
        identity = implementation_identity()
        approval, resource = read_json(approval_path), read_json(resource_path)
        verify_approval(approval, identity)
        data, meta = load_prepared(authority, prepared_stage)
        preflight_path = stage_path(preflight_stage)
        verify_stage(preflight_path, 'preflight', canonical_hash(identity))
        if resource['authority'] != resource_snapshot:
            raise ValueError('resource ancestry changed after entry verification')
        if (read_json(preflight_path / 'identity.json') != identity
                or read_json(preflight_path / 'inputs.json') != {
                    'prepared_stage': prepared_stage,
                    'preparation_marker_sha256': digest(stage_path(prepared_stage) / 'complete.json')}):
            raise ValueError('preflight source/input identities differ from lock')
        projection = verify_resource_plan(resource, read_json(preflight_path / 'probe/preflight.json'),
                                          authority, identity)
        guard.check(projected_cpu_seconds=projection['cpu_seconds'],
                    projected_gpu_seconds=projection['gpu_seconds'])
        contract = {**identity, 'schema': 'r38-joint-dynamics-contract-v3',
            'prepared_stage': prepared_stage, 'preflight_stage': preflight_stage,
            'prepared_marker_sha256': digest(stage_path(prepared_stage) / 'complete.json'),
            'preflight_marker_sha256': digest(preflight_path / 'complete.json'),
            'input_sha256': meta['input_sha256'], 'normalizer': data.normalizer,
            'rows': data.payload['rows'], 'splits': data.splits,
            'approval_sha256': digest(approval_path), 'resource_plan_sha256': digest(resource_path),
            'resource_projection': projection, 'authority': before}
        save_json(path / 'contract.json', contract)
        shutil.copyfile(approval_path, path / 'approval.json')
        shutil.copyfile(resource_path, path / 'resource_plan.json')
    seal_stage(path, 'lock', canonical_hash(contract), authority.snapshot())
    save_json(STUDY / 'locked.json', {'stage': name, 'contract_digest': canonical_hash(contract),
              'complete_sha256': digest(path / 'complete.json')})
    return {'status': 'locked_not_started', 'contract_digest': canonical_hash(contract), 'projection': projection}


def load_lock(authority):
    locked = read_json(STUDY / 'locked.json')
    path = stage_path(locked['stage'])
    if digest(path / 'complete.json') != locked['complete_sha256']:
        raise ValueError('lock marker changed')
    verify_stage(path, 'lock', locked['contract_digest'])
    contract = read_json(path / 'contract.json')
    if canonical_hash(contract) != locked['contract_digest']:
        raise ValueError('lock contract changed')
    identity = implementation_identity()
    if any(contract[k] != identity[k] for k in ('source_sha256', 'packet_sha256', 'plan')):
        raise ValueError('actual implementation/protocol differs from stable approval and locked bytes')
    verify_approval(read_json(path / 'approval.json'), identity)
    return contract, read_json(path / 'resource_plan.json')


def scientific_history(authority, *, assessing: bool) -> None:
    scientific = [s for s in authority.snapshot()['stages'] if s['scientific']]
    expected = ['r38-fit'] if assessing else []
    if [s['name'] for s in scientific] != expected or any(s['status'] != 'succeeded' for s in scientific):
        raise RuntimeError('started, failed, partial or repeated scientific stage; no retry')


def fit(authority) -> dict:
    scientific_history(authority, assessing=False)
    path = reserve_directory('r38-fit')
    with charged_stage(authority, 'r38-fit', gpu=True, scientific=True,
                       prior_paths=locked_paths()) as guard:
        from src.harness.analysis.joint_dynamics.fitting import fit_matrix
        contract, resource = load_lock(authority)
        data, meta = load_prepared(authority, contract['prepared_stage'])
        if meta['input_sha256'] != contract['input_sha256'] or data.normalizer != contract['normalizer']:
            raise ValueError('locked source or normalization changed')
        guard.check(projected_cpu_seconds=contract['resource_projection']['cpu_seconds'],
                    projected_gpu_seconds=contract['resource_projection']['gpu_seconds'])
        result = fit_matrix(data, path / 'fit', contract, resource, guard, meta['objects']['geometry'])
        data.verify_all_bytes(meta)
        guard.check(projected_cpu_seconds=3600. + resource['phases']['assessment']['cpu_seconds'],
                    projected_gpu_seconds=900. + resource['phases']['assessment']['gpu_seconds'])
        import torch
        torch.cuda.synchronize()
    # fit payload has its own exact manifest; stage envelope binds the same contract.
    snapshot = authority.snapshot()
    seal_stage(path / 'fit', 'fit', canonical_hash(contract), snapshot)
    validate_completed_fit(path / 'fit', canonical_hash(contract))
    seal_stage(path, 'fit-stage', canonical_hash(contract), snapshot)
    return {'status': 'fitted_not_assessed', 'updates': result['total_updates'],
            'contract_digest': canonical_hash(contract)}


def assess(authority) -> dict:
    scientific_history(authority, assessing=True)
    path = reserve_directory('r38-assess')
    with charged_stage(authority, 'r38-assess', gpu=False, scientific=True,
                       prior_paths=(*locked_paths(), stage_path('r38-fit') / 'fit')) as guard:
        import time
        import numpy as np
        from src.harness.analysis.joint_dynamics.assessment import analyze
        from src.harness.analysis.joint_dynamics.baselines import ControlFits
        contract, resource = load_lock(authority)
        data, meta = load_prepared(authority, contract['prepared_stage'])
        fitted_root = stage_path('r38-fit') / 'fit'
        marker = validate_completed_fit(fitted_root, canonical_hash(contract))
        if marker['contract_digest'] != canonical_hash(contract):
            raise ValueError('fitted contract differs from locked study')
        fitted = read_json(fitted_root / 'fitted.json')
        portable = ControlFits.from_dict(fitted['control_fits']).to_dict()
        if portable != fitted['control_fits']:
            raise ValueError('portable controls do not roundtrip exactly')
        control_receipt = read_json(fitted_root / 'control_fits.json')
        if (control_receipt['fits'] != portable
                or control_receipt['contract_digest'] != canonical_hash(contract)
                or fitted['source_sha256'] != contract['source_sha256']
                or fitted['control_fits_sha256'] != digest(fitted_root / 'control_fits.json')
                or control_receipt['source_row_ids'] != [r.row_id for r in data.select('source52')]
                or control_receipt['validation_row_ids'] != [r.row_id for r in data.select('validation13')]):
            raise ValueError('control fitting identity mismatch')
        identity = {'contract_digest': canonical_hash(contract), 'source_sha256': source_hashes(),
            'fitted_manifest_sha256': digest(fitted_root / 'fitted.json'), 'control_fits': portable,
            'cohort_rows': fitted['cohort_rows'], 'calibration': fitted['calibration']}
        _verify_fitted_identity(data, fitted_root, fitted, identity)
        start, cpu = time.monotonic(), time.process_time()
        with np.load(fitted_root / 'private_predictions.npz', allow_pickle=False) as bundle:
            arrays = {key: bundle[key] for key in bundle.files}
        report = analyze(arrays, identity)
        cap = resource['phases']['assessment']
        if time.process_time() - cpu > cap['cpu_seconds'] or time.monotonic() - start > cap['wall_seconds']:
            raise TimeoutError('INCOMPLETE_NOT_GRADED: assessment phase cap exceeded')
        data.verify_all_bytes(meta)
        if source_hashes() != contract['source_sha256']:
            raise ValueError('source changed during assessment')
        if report.get('status') != 'staged_only':
            raise ValueError('assessment must return a staged-only aggregate')
        report['identity'] = identity
        save_json(path / 'staged_aggregate.json', report)
        guard.check(projected_cpu_seconds=3600., projected_gpu_seconds=900.)
    report['status'] = 'completed_not_independently_reviewed'
    save_json(path / 'aggregate.json', report)
    (path / 'staged_aggregate.json').unlink()
    seal_stage(path, 'assessment', canonical_hash(contract), authority.snapshot())
    return {'status': 'completed_not_independently_reviewed', 'aggregate_sha256': digest(path / 'aggregate.json')}


def _verify_fitted_identity(data, root: Path, fitted: dict, identity: dict) -> None:
    from src.harness.analysis.joint_dynamics.fitting import _shared_rows
    for name in ('primary', 'single', 'nuisance'):
        expected = _shared_rows({}, name, data.select(name))
        if expected != identity['cohort_rows'][name]:
            raise ValueError('fitted measured targets/ordered cohorts differ from prepared source')
    import numpy as np
    from src.harness.analysis.joint_dynamics.baselines import gaussian_calibration_receipt
    for control in ('P', 'E', 'FP', 'SSM'):
        if identity['calibration'][control] != fitted['control_fits']['state']['calibration'][control]:
            raise ValueError('common control calibration differs from validated fit envelope')
    for label, cell in fitted['cells'].items():
        calibration = read_json(root / cell['calibration_file']['path'])
        raw = calibration['inputs']
        if digest(root / raw['path']) != raw['sha256']:
            raise ValueError('changed source/validation calibration arrays')
        with np.load(root / raw['path'], allow_pickle=False) as values:
            for name, cohort in (('source', 'source52'), ('validation', 'validation13')):
                expected = {}
                _shared_rows(expected, name, data.select(cohort))
                if any(not np.array_equal(values[k], v) for k, v in expected.items()):
                    raise ValueError('own-persistence source/validation binding mismatch')
                if (calibration[f'{name}_row_ids'] != expected[f'{name}_row_ids'].tolist()
                        or calibration[f'{name}_patient_ids'] != expected[f'{name}_patient_ids'].tolist()):
                    raise ValueError('own-persistence receipt membership mismatch')
            forecast_receipt = gaussian_calibration_receipt(values['validation_mean'],
                values['validation_intrinsic_scale'], values['validation_truth'], values['validation_patient_ids'])
            persistence_receipt = gaussian_calibration_receipt(values['validation_current_mean'],
                np.broadcast_to(calibration['persistence_rms'], values['validation_current_mean'].shape),
                values['validation_truth'], values['validation_patient_ids'])
            if (forecast_receipt != calibration['forecast_receipt']
                    or persistence_receipt != calibration['persistence_receipt']):
                raise ValueError('own-cell calibration raw-grid receipt mismatch')
        if (calibration['contract_digest'] != identity['contract_digest']
                or {k: calibration[k] for k in ('forecast', 'persistence', 'persistence_rms')}
                    != identity['calibration'][label]):
            raise ValueError('fitted neural calibration mismatch')
    if (identity['calibration'] | {}) != fitted['calibration']:
        raise ValueError('calibration identity changed')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    for mode in ('prepare', 'preflight', 'lock'):
        command = sub.add_parser(mode)
        command.add_argument('--stage-name', required=True)
        if mode != 'prepare':
            command.add_argument('--prepared-stage', required=True)
        if mode == 'lock':
            command.add_argument('--preflight-stage', required=True)
            command.add_argument('--approval', required=True, type=Path)
            command.add_argument('--resource-plan', required=True, type=Path)
    sub.add_parser('fit')
    sub.add_parser('assess')
    args = parser.parse_args(argv)
    try:
        require_remote()
        from src.harness.analysis.joint_dynamics.authority import production_authority
        authority = production_authority()  # Never initialize or adopt the existing real ledger here.
        authority.snapshot()
        with runner_lock():
            if args.mode == 'prepare':
                result = prepare(authority, args.stage_name)
            elif args.mode == 'preflight':
                result = preflight(authority, args.stage_name, args.prepared_stage)
            elif args.mode == 'lock':
                result = lock(authority, args.stage_name, args.prepared_stage, args.preflight_stage,
                              args.approval, args.resource_plan)
            elif args.mode == 'fit':
                result = fit(authority)
            else:
                result = assess(authority)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except Exception as error:
        print(f'INCOMPLETE_NOT_GRADED: {type(error).__name__}: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
