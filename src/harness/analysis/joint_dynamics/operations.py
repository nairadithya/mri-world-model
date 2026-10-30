from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import os
import subprocess
import sys

from .protocol import (PLAN, REPO, ArtifactRef, SourceIdentity, canonical_hash,
                       digest, read_json, save_json)


def verify_counts(data) -> None:
    for name, rows, patients in PLAN.cohort_counts:
        selected = data.select(name)
        row_patients = {r.patient_id for r in selected}
        if name in ('source52', 'validation13'):
            nominal = set(data.splits[name])
            membership_valid = len(nominal) == patients and row_patients <= nominal
        else:
            membership_valid = len(row_patients) == patients
        if len(selected) != rows or not membership_valid:
            raise ValueError(f'wrong canonical cohort: {name}')


def load_prepared(payload: dict):
    from .data import PREPARATION_SOURCES, PreparedData, load_metadata
    preparation = Path(payload['preparation'])
    completion = read_json(preparation.with_name(preparation.name + '.control') / 'complete.json')
    if completion['schema'] != 'r38-finalized-stage-v4' or completion['request']['operation'] != 'prepare':
        raise ValueError('preparation was not successfully finalized')
    for name in ('preparation_contract.json', 'prepared/prepared.json'):
        if digest(preparation / name) != completion['manifest'][name]:
            raise ValueError('preparation metadata changed')
    receipt = read_json(preparation / 'preparation_contract.json')
    if receipt['preparation_source_sha256'] != {p: digest(REPO / p) for p in PREPARATION_SOURCES}:
        raise ValueError('preparation primitives changed')
    if digest(payload['bindings']) != receipt['bindings_sha256']:
        raise ValueError('preparation input binding changed')
    meta = load_metadata(read_json(payload['bindings']))
    data = PreparedData(preparation / 'prepared')
    data.verify_all_bytes(meta)
    verify_counts(data)
    return data, meta


def essential_checks(out: Path, *, preparation: bool = False) -> None:
    command = [sys.executable, '-B', '-m', 'unittest', '-v',
        'tests.test_joint_reference',
        'tests.test_joint_numerics',
        'tests.test_joint_execution',
        'tests.test_joint_authority.AuthorityTests.test_cost_observes_live_workers_reaped_grandchild_and_final_single_charge',
        'tests.test_joint_dynamics.JointDynamicsModelTests.test_forecast_detaches_every_observation_path_including_native_anchor',
        'tests.test_joint_dynamics.JointDynamicsModelTests.test_current_supervision_equal_for_oh_ol_after_identical_updates',
        'tests.test_joint_dynamics.JointDynamicsModelTests.test_named_boundary_rejects_targets_future_covariates_wrong_shapes_and_dtypes',
        'tests.test_joint_dynamics.JointDynamicsModelTests.test_fixed_cuda_bf16_complete_step_cutoff_only_and_successful_updates',
        'tests.test_joint_dynamics.JointDynamicsDataTests.test_loader_spy_latest_never_reads_earlier_or_future_mri',
        'tests.test_joint_dynamics.JointDynamicsDataTests.test_source_only_353_normalizer_includes_noncutoff_visits',
        'tests.test_joint_dynamics.JointDynamicsDataTests.test_outgoing_zero_gap_is_not_row_eligibility_filter',
        'tests.test_joint_dynamics.JointDynamicsDataTests.test_fixed_schedule_patient_uniform_replacement_and_matched_orders']
    # The dedicated subprocess lets the lifecycle test own only its own children.
    selected = [part.removeprefix('tests.') for part in command[5:]]
    preparation_tests = ('test_joint_reference', 'test_joint_execution', 'test_joint_authority')
    selected = [part for part in selected if part.startswith(preparation_tests) == preparation]
    command = command[:5] + selected
    result = subprocess.run(command, cwd=REPO, text=True, capture_output=True,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                                     PYTHONPATH=str(REPO / 'tests') + os.pathsep + str(REPO)))
    (out / 'checks.stdout').write_text(result.stdout)
    (out / 'checks.stderr').write_text(result.stderr)
    save_json(out / 'checks.json', {'command': command, 'returncode': result.returncode,
        'scope': 'repaired reference/scorer contracts, gradients, cutoff, loading and owned limits; not scientific evidence',
        'fixture_optimizer_updates': {'CPU_synthetic': 0 if preparation else 4,
                                      'CUDA_synthetic': 0 if preparation else 4},
        'actual_data_matrix_updates': 0})
    if result.returncode:
        raise RuntimeError('relevant essential execution checks failed; see checks.stderr')
    import py_compile
    for path in sorted((REPO / 'src/harness/analysis/joint_dynamics').glob('*.py')):
        py_compile.compile(str(path), cfile=str(out / (path.name + 'c')), doraise=True)
    py_compile.compile(str(REPO / 'scripts/run_joint_dynamics.py'),
                       cfile=str(out / 'run_joint_dynamics.pyc'), doraise=True)


def validate_fitted(data, root: Path, contract: dict, *, canary: bool) -> dict:
    import json
    import torch
    from .baselines import ControlFits, gaussian_calibration_receipt
    from .fitting import _shared_rows
    import numpy as np
    fitted = read_json(root / 'fitted.json')
    source = SourceIdentity.from_dict(contract['implementation_source'])
    expected = (1, 7) if canary else (20, 5240)
    labels = {f'{a.name}_{s}' for a in PLAN.arms for s in PLAN.seeds}
    if (fitted['implementation_source'] != source.to_dict()
            or fitted['contract_digest'] != canonical_hash(contract)
            or fitted['execution_scope'] != contract['execution_scope']
            or set(fitted['cells']) != labels or fitted['seeds'] != list(PLAN.seeds)
            or fitted['total_updates'] != len(labels) * expected[1]
            or (fitted['epochs'], fitted['updates_per_epoch']) != (expected[0], 7 if canary else 262)):
        raise ValueError('actual fit scope, source, schedule or completed cells mismatch')
    for label, cell in fitted['cells'].items():
        if (cell['epoch'], cell['updates']) != expected:
            raise ValueError('incorrect actual per-cell updates')
        for key in ('weights', 'updates_file', 'calibration_file'):
            record = cell[key]
            if digest(root / record['path']) != record['sha256']:
                raise ValueError('changed fit artifact')
        saved = torch.load(root / cell['weights']['path'], map_location='cpu', weights_only=False)
        if ((saved['epoch'], saved['updates']) != expected
                or saved['execution_scope'] != contract['execution_scope']
                or saved['implementation_source'] != source.to_dict()
                or saved['contract_digest'] != canonical_hash(contract)
                or saved['normalizer'] != data.normalizer
                or saved['arm'] != label.split('_')[0] or saved['seed'] != int(label.split('_')[1])
                or not saved['optimizer']['state']
                or not all(torch.isfinite(t).all() for t in saved['state_dict'].values())):
            raise ValueError('checkpoint does not contain the intended trained state')
        if any(int(state['step']) != expected[1] for state in saved['optimizer']['state'].values()):
            raise ValueError('optimizer step count disagrees with successful training updates')
        with (root / cell['updates_file']['path']).open() as stream:
            updates = [json.loads(line) for line in stream]
        if (len(updates) != expected[1]
                or any((r['epoch'], r['update']) != (i // (7 if canary else 262) + 1,
                        i % (7 if canary else 262) + 1) for i, r in enumerate(updates))
                or any(r['future_image_forwards'] != 0 or r['earlier_state_supervision'] != 0
                       for r in updates)):
            raise ValueError('successful update log or supervision scope differs')
        calibration = read_json(root / cell['calibration_file']['path'])
        raw = calibration['inputs']
        if digest(root / raw['path']) != raw['sha256']:
            raise ValueError('calibration arrays changed')
        with np.load(root / raw['path'], allow_pickle=False) as values:
            for name, cohort in (('source', 'source52'), ('validation', 'validation13')):
                membership = {}
                _shared_rows(membership, name, data.select(cohort))
                if any(not np.array_equal(values[k], v) for k, v in membership.items()):
                    raise ValueError('calibration membership or truths changed')
            forecast = gaussian_calibration_receipt(values['validation_mean'],
                values['validation_intrinsic_scale'], values['validation_truth'], values['validation_patient_ids'])
            persistence = gaussian_calibration_receipt(values['validation_current_mean'],
                np.broadcast_to(calibration['persistence_rms'], values['validation_current_mean'].shape),
                values['validation_truth'], values['validation_patient_ids'])
        if (forecast != calibration['forecast_receipt'] or persistence != calibration['persistence_receipt']
                or calibration['contract_digest'] != canonical_hash(contract)
                or {k: calibration[k] for k in ('forecast', 'persistence', 'persistence_rms')}
                    != fitted['calibration'][label]):
            raise ValueError('calibration receipt differs from actual inputs')
    for name in ('primary', 'single', 'nuisance'):
        if fitted['cohort_rows'][name] != _shared_rows({}, name, data.select(name)):
            raise ValueError('assessment rows differ from complete canonical cohorts')
    controls = ControlFits.from_dict(fitted['control_fits'])
    receipt = read_json(root / 'control_fits.json')
    if (controls.to_dict() != fitted['control_fits'] or receipt['fits'] != controls.to_dict()
            or receipt['contract_digest'] != canonical_hash(contract)
            or receipt['source_row_ids'] != [r.row_id for r in data.select('source52')]
            or receipt['validation_row_ids'] != [r.row_id for r in data.select('validation13')]
            or digest(root / 'control_fits.json') != fitted['control_fits_sha256']
            or digest(root / 'private_predictions.npz') != fitted['prediction_sha256']):
        raise ValueError('control or prediction artifact differs from fitted identity')
    return fitted


def perform_operation(request, work):
    payload = read_json(request.inputs[0].path)
    if payload['purpose'] == 'prepare':
        from .data import PREPARATION_SOURCES, PreparedData, load_metadata, prepare_inputs
        from .model import configure_torch
        configure_torch()
        with work.phase('preparation'):
            essential_checks(request.output, preparation=True)
            meta = load_metadata(read_json(payload['bindings']))
            prepared = prepare_inputs(meta, request.output / 'prepared', work)
            data = PreparedData(request.output / 'prepared')
            verify_counts(data)
            data.verify_all_bytes(meta)
            save_json(request.output / 'preparation_contract.json', {
                'preparation_source_sha256': {p: digest(REPO / p) for p in PREPARATION_SOURCES},
                'bindings_sha256': digest(payload['bindings']),
                'input_sha256': meta['input_sha256'],
                'prepared_sha256': digest(request.output / 'prepared/prepared.json')})
            result = {'status': 'prepared_not_trained', 'reconstruction': prepared['reconstruction'],
                      'direct_reference_count': prepared['direct_reference_coverage']['supplied_count'],
                      'normalizer_visits': len(prepared['normalizer']['visit_ids'])}
    elif payload['purpose'] == 'prepare_recovery':
        import shutil
        from .data import PREPARATION_SOURCES, PreparedData, load_metadata
        with work.phase('preparation'):
            source_root = Path(payload['recovery_prepared'])
            if digest(source_root / 'prepared.json') != payload['recovery_metadata_sha256']:
                raise ValueError('diagnosed partial preparation metadata changed')
            meta = load_metadata(read_json(payload['bindings']))
            original = PreparedData(source_root, device='cpu')
            if (original.payload['reconstruction']['count'] != 599
                    or len(original.normalizer['visit_ids']) != 353):
                raise ValueError('recovery is missing completed numerical preparation')
            verify_counts(original)
            original.verify_all_bytes(meta)
            command = [sys.executable, '-B', '-m', 'unittest', '-v',
                       'test_joint_execution.NominalSplitCountTests']
            checked = subprocess.run(command, cwd=REPO, text=True, capture_output=True,
                env=dict(os.environ, PYTHONPATH=str(REPO / 'tests') + os.pathsep + str(REPO)))
            (request.output / 'count-check.stdout').write_text(checked.stdout)
            (request.output / 'count-check.stderr').write_text(checked.stderr)
            if checked.returncode:
                raise RuntimeError('nominal-versus-active source membership regression failed')
            shutil.copytree(source_root, request.output / 'prepared', copy_function=shutil.copyfile)
            data = PreparedData(request.output / 'prepared', device='cpu')
            verify_counts(data)
            data.verify_all_bytes(meta)
            if digest(source_root / 'prepared.json') != payload['recovery_metadata_sha256']:
                raise ValueError('recovery metadata changed during finalization')
            save_json(request.output / 'preparation_contract.json', {
                'preparation_source_sha256': {p: digest(REPO / p) for p in PREPARATION_SOURCES},
                'bindings_sha256': digest(payload['bindings']), 'input_sha256': meta['input_sha256'],
                'prepared_sha256': digest(request.output / 'prepared/prepared.json'),
                'recovery': {'original_stage': payload['recovery_stage'],
                    'original_status': 'failed_after_all599_numeric_reconstruction_before_final_count_check',
                    'metadata_sha256': payload['recovery_metadata_sha256'],
                    'action': 'preserve failed output; copy completed preparation bytes; recheck counts and every raw/prepared hash',
                    'reencoded': False}})
            result = {'status': 'prepared_not_trained', 'reconstruction': data.payload['reconstruction'],
                'normalizer_visits': len(data.normalizer['visit_ids']), 'source_nominal_patients': 52,
                'source_patients_with_rows': len({r.patient_id for r in data.select('source52')}),
                'direct_reference_count': data.payload['direct_reference_coverage']['supplied_count'],
                'recovered_from_failed_stage': payload['recovery_stage']}
    elif payload['purpose'] in ('canary_fit', 'study_fit'):
        import torch
        from .fitting import fit_matrix
        from .model import configure_torch
        configure_torch()
        canary = payload['purpose'] == 'canary_fit'
        if canary:
            with work.phase('essential_checks'):
                essential_checks(request.output)
        data, meta = load_prepared(payload)
        contract = payload['contract']
        if contract['implementation_source'] != request.source.to_dict():
            raise ValueError('fitting contract source differs from supervised request')
        fitted = fit_matrix(data, request.output / 'fit', contract, work,
                            meta['objects']['geometry'], canary=canary)
        validate_fitted(data, request.output / 'fit', contract, canary=canary)
        data.verify_all_bytes(meta)
        torch.cuda.synchronize()
        result = {'status': 'engineering_fit_not_scientific_result' if canary else 'fitted_not_assessed',
                  'actual_updates': fitted['total_updates'], 'cells': len(fitted['cells']),
                  'execution_scope': contract['execution_scope'], 'contract_digest': canonical_hash(contract)}
    elif payload['purpose'] in ('canary_score', 'study_score'):
        import numpy as np
        from .assessment import AssessmentIdentity, analyze
        from .model import configure_torch
        configure_torch()
        with work.phase('assessment'):
            data, meta = load_prepared(payload)
            root, contract = Path(payload['fit']) / 'fit', payload['contract']
            fitted = validate_fitted(data, root, contract, canary=payload['purpose'] == 'canary_score')
            identity = AssessmentIdentity(request.source, canonical_hash(contract), digest(root / 'fitted.json'),
                fitted['control_fits'], fitted['cohort_rows'], fitted['calibration'])
            identity = AssessmentIdentity.from_dict(identity.to_dict())
            with np.load(root / 'private_predictions.npz', allow_pickle=False) as bundle:
                arrays = {key: bundle[key] for key in bundle.files}
            report = analyze(arrays, identity)
            if report['status'] != 'staged_only':
                raise ValueError('scorer must not promote a result before authority finalization')
            save_json(request.output / 'staged_aggregate.json', report)
            data.verify_all_bytes(meta)
            result = {'status': 'engineering_canary_scored_not_scientific_evidence'
                        if payload['purpose'] == 'canary_score' else 'scored_pending_result_interpretation',
                      'actual_updates': fitted['total_updates'],
                      'scorer_status': report['status'], 'contract_digest': canonical_hash(contract),
                      'aggregate_sha256': digest(request.output / 'staged_aggregate.json')}
    elif payload['purpose'] == 'diagnose_counts':
        from .data import PreparedData
        with work.phase('diagnosis'):
            path = Path(payload['partial_prepared'])
            before = digest(path / 'prepared.json')
            data = PreparedData(path, device='cpu')
            result = {'status': 'metadata_diagnosis_only_not_input_approval',
                'prepared_metadata_sha256': before, 'rows': len(data.rows), 'visits': len(data.visits),
                'normalizer_visits': len(data.normalizer['visit_ids']),
                'reconstruction': data.payload['reconstruction'],
                'direct_reference_count': data.payload['direct_reference_coverage']['supplied_count'],
                'cohorts': {name: {'expected_rows': rows, 'expected_nominal_patients': patients,
                    'actual_rows': len(data.select(name)),
                    'actual_row_patients': len({r.patient_id for r in data.select(name)}),
                    'nominal_split_patients': len(data.splits[name]) if name in data.splits else None}
                    for name, rows, patients in PLAN.cohort_counts}}
            if digest(path / 'prepared.json') != before:
                raise ValueError('partial metadata changed during diagnosis')
    else:
        raise ValueError('unsupported concrete R38 operation')
    save_json(request.output / 'operation_result.json', result)
    return {str(p.relative_to(request.output)): digest(p)
            for p in sorted(request.output.rglob('*')) if p.is_file()}
