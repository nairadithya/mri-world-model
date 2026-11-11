from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, fields, replace
import copy
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from src.harness.analysis.joint_dynamics import data, fitting, model, protocol


def inputs(history: bool = True) -> model.ForecastInputs:
    image = torch.zeros(1, 4, 4, 96, 96, 96)
    image[:, -1] = 0.3
    if history:
        image[:, -2] = 0.7
    return model.ForecastInputs(image, torch.tensor([[[0.] * 4, [0.] * 4, [1.] * 4, [1.] * 4]]),
        torch.tensor([[0., 0., 0., math.log1p(7.)]]),
        torch.tensor([[False, False, True, True]]),
        torch.tensor([[False, False, history, True]]))


def targets() -> model.AnatomyTargets:
    masks = torch.zeros(1, 3, 96, 96, 96)
    masks[:, :, 30:50, 32:54, 35:55] = 1
    return model.AnatomyTargets(torch.tensor([[1., 2., 3.]]), masks)


def row(patient: str = 'source', suffix: str = '') -> protocol.ForecastRow:
    return protocol.ForecastRow(f'r{suffix}', patient, 'cutoff', 'future', ('earlier', 'cutoff'),
        (0., 0., 0., math.log1p(7.)), (False, False, True, True), (2., 3., 4.), (1., 2., 3.), True)


def runner_module():
    path = Path(__file__).resolve().parents[1] / 'scripts/experiments/spatial/run_joint_dynamics.py'
    spec = importlib.util.spec_from_file_location('r38_runner_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class JointDynamicsModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model.configure_torch()

    def make_model(self):
        torch.manual_seed(42)
        return model.JointDynamics((1., 2., 3.), (0.5, 1., 2.))

    def test_exact_architecture_shapes_fp32_losses_and_current_gradients(self):
        network = self.make_model()
        self.assertEqual(network.state.widths, (16, 32, 64, 128))
        self.assertEqual(network.head.recurrent.input_size, 247)
        self.assertEqual(network.head.recurrent.hidden_size, 64)
        self.assertEqual(tuple(network.head.dense[0].weight.shape), (32, 128, 3, 3, 3))
        seen = []
        hook = network.head.register_forward_pre_hook(lambda _, args: seen.append(
            ([tuple(p.shape) for p in args[0]], tuple(args[1].shape), tuple(args[2].shape))))
        prediction = network(inputs(), protocol.ARMS[0])
        hook.remove()
        self.assertEqual(seen, [([(1, 240)] * 4, (1, 64, 24, 24, 24), (1, 3))])
        for name in ('mean', 'intrinsic_scale', 'current_mean', 'current_standardized'):
            self.assertEqual(tuple(getattr(prediction, name).shape), (1, 3))
            self.assertEqual(getattr(prediction, name).dtype, torch.float32)
        for name in ('current_logits', 'future_logits'):
            self.assertEqual(tuple(getattr(prediction, name).shape), (1, 3, 96, 96, 96))
        target = targets()
        parts = model.losses(prediction, target, target, network.volume_mean, network.volume_scale)
        expected_current = F.binary_cross_entropy_with_logits(prediction.current_logits, target.masks)
        expected_current += model.soft_dice_loss(prediction.current_logits, target.masks)
        expected_current += F.l1_loss(prediction.current_standardized,
                                     (target.native_volume - network.volume_mean) / network.volume_scale)
        expected_forecast = model.gaussian_crps(prediction.mean, prediction.total_scale, target.native_volume).mean()
        expected_forecast += 0.1 * model.soft_dice_loss(prediction.future_logits, target.masks)
        self.assertTrue(torch.equal(parts['current'], expected_current))
        self.assertTrue(torch.equal(parts['forecast'], expected_forecast))
        self.assertTrue(torch.equal(parts['total'], expected_current + expected_forecast))
        self.assertTrue(all(v.dtype == torch.float32 and torch.isfinite(v) for v in parts.values()))
        parts['current'].backward()
        for component in (network.state.blocks, network.state.decoder, network.state.volume_head):
            gradients = [p.grad for p in component.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)
        self.assertTrue(all(p.grad is None for p in network.head.parameters()))

    def test_forecast_detaches_every_observation_path_including_native_anchor(self):
        for arm in protocol.ARMS:
            with self.subTest(arm=arm.name):
                network = self.make_model()
                seen = []
                handle = network.head.register_forward_pre_hook(lambda _, args: seen.append(
                    ([p.requires_grad for p in args[0]], args[1].requires_grad, args[2].requires_grad)))
                prediction = network(inputs(arm.history), arm)
                handle.remove()
                self.assertEqual(seen[0][1:], (arm.joint, arm.joint))
                self.assertEqual(seen[0][0][-1], arm.joint)
                objective = model.gaussian_crps(prediction.mean, prediction.total_scale, targets().native_volume).mean()
                objective += 0.1 * model.soft_dice_loss(prediction.future_logits, targets().masks)
                objective.backward()
                self.assertTrue(all(p.grad is None for p in network.state.decoder.parameters()))
                state_gradients = [p.grad for p in network.state.parameters() if p.grad is not None]
                if arm.joint:
                    self.assertTrue(state_gradients)
                    self.assertTrue(all(torch.isfinite(g).all() for g in state_gradients))
                    self.assertGreater(sum(float(g.abs().sum()) for g in state_gradients), 0.)
                    self.assertGreater(float(network.state.volume_head.weight.grad.abs().sum()), 0.)
                else:
                    self.assertEqual(state_gradients, [])
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in network.head.parameters()
                                       if p.grad is not None), 0.)

    def test_current_supervision_equal_for_oh_ol_after_identical_updates(self):
        initial = self.make_model().state_dict()
        final = []
        for arm in protocol.ARMS[:2]:
            network = self.make_model()
            network.load_state_dict(initial)
            optimizer = fitting.make_optimizer(network)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                prediction = network(inputs(arm.history), arm)
                parts = model.losses(prediction, targets(), targets(), network.volume_mean, network.volume_scale)
                model.finite_step(network, optimizer, parts['total'])
            final.append({k: v.clone() for k, v in network.state.state_dict().items()})
        self.assertEqual(final[0].keys(), final[1].keys())
        self.assertTrue(all(torch.equal(final[0][k], final[1][k]) for k in final[0]))

    def test_latest_substitution_zero_and_cross_patient_exchange_are_bit_identical(self):
        network = self.make_model().eval()
        for arm in (protocol.ARMS[1], protocol.ARMS[3]):
            original = inputs(False)
            with torch.inference_mode():
                reference = network(original, arm)
                other_patient = inputs().images.clone()
                other_patient[:, :-1] = -3.
                variants = (torch.zeros_like(original.images[:, :-1]),
                            torch.full_like(original.images[:, :-1], 17.), other_patient[:, :-1])
                for replacement in variants:
                    changed = original.images.clone()
                    changed[:, :-1] = replacement
                    prediction = network(replace(original, images=changed), arm)
                    for field in ('mean', 'intrinsic_scale', 'current_mean', 'current_logits', 'future_logits'):
                        self.assertTrue(torch.equal(getattr(reference, field), getattr(prediction, field)))

    def test_controlled_history_path_is_usable_not_a_trained_success_requirement(self):
        head = model.ForecastHead()
        with torch.no_grad():
            for parameter in head.parameters():
                parameter.zero_()
            head.recurrent.weight_ih[128:192, 0] = 1.
            head.change.weight[:, 0] = 1.
        pooled = [torch.zeros(1, 240) for _ in range(4)]
        spatial, anchor, covariates = torch.zeros(1, 64, 24, 24, 24), torch.zeros(1, 3), inputs()
        with torch.inference_mode():
            before = head(pooled, spatial, anchor, covariates)[0]
            pooled[2][:, 0] = 2.
            after = head(pooled, spatial, anchor, covariates)[0]
        self.assertFalse(torch.equal(before, after))

    def test_named_boundary_rejects_targets_future_covariates_wrong_shapes_and_dtypes(self):
        network = self.make_model()
        with self.assertRaises(TypeError):
            network(targets(), protocol.ARMS[0])
        with self.assertRaises(TypeError):
            model.ForecastInputs(**asdict(inputs()), future_gap=torch.ones(1))
        for changed in (replace(inputs(), images=torch.zeros(1, 5, 4, 96, 96, 96)),
                        replace(inputs(), images=inputs().images.half()),
                        replace(inputs(), gap_log1p_days=torch.tensor([[0., 0., 0., -1.]])),
                        replace(inputs(), image_present=torch.ones(1, 4, dtype=torch.bool))):
            with self.assertRaises(ValueError):
                changed.validate()

    def test_fixed_cuda_bf16_complete_step_cutoff_only_and_successful_updates(self):
        self.assertTrue(torch.cuda.is_available(), 'remote fixed CUDA policy required')
        self.assertTrue(torch.cuda.is_bf16_supported(), 'no precision fallback')
        class Fixture:
            def __init__(self):
                self.calls = []
            def inputs(self, selected, arm):
                self.calls.append(('images', selected.cutoff_visit_id))
                value = inputs(arm.history)
                return model.ForecastInputs(**{f.name: getattr(value, f.name).cuda() for f in fields(value)})
            def target(self, selected, *, current=False):
                self.calls.append(('target', selected.cutoff_visit_id if current else selected.target_visit_id))
                value = targets()
                return model.AnatomyTargets(value.native_volume.cuda(), value.masks.cuda())
        for arm in protocol.ARMS:
            fixture = Fixture()
            network = self.make_model().cuda()
            optimizer = fitting.make_optimizer(network)
            initial = fitting.tensor_hash(network.state_dict())
            receipt = fitting.train_step(fixture, row(), arm, network, optimizer)
            self.assertNotEqual(initial, fitting.tensor_hash(network.state_dict()))
            self.assertEqual(fixture.calls, [('images', 'cutoff'), ('target', 'cutoff'), ('target', 'future')])
            self.assertEqual(receipt['current_supervision_visit_id'], 'cutoff')
            self.assertEqual(receipt['earlier_state_supervision'], 0)
            self.assertEqual(receipt['future_image_forwards'], 0)
            self.assertTrue(all(math.isfinite(v) for v in receipt['loss'].values()))
            self.assertEqual({int(v['step']) for v in optimizer.state.values()}, {1})
            del network, optimizer
            torch.cuda.empty_cache()

    def test_literal_losses_and_empty_dice(self):
        value = model.gaussian_crps(torch.zeros(1), torch.ones(1), torch.zeros(1))
        self.assertAlmostEqual(float(value), (math.sqrt(2.) - 1.) / math.sqrt(math.pi), places=6)
        self.assertTrue(torch.equal(model.binary_dice(torch.full((1, 3, 2, 2, 2), -1.),
                                                     torch.zeros(1, 3, 2, 2, 2)), torch.ones(1, 3)))

    def test_separate_norm_one_clipping(self):
        network = self.make_model()
        optimizer = fitting.make_optimizer(network)
        objective = sum(p.sum() for p in network.parameters())
        with patch.object(torch.nn.utils, 'clip_grad_norm_', wraps=torch.nn.utils.clip_grad_norm_) as clip:
            norms = model.finite_step(network, optimizer, objective)
        self.assertEqual(clip.call_count, 2)
        self.assertEqual(set(norms), {'state', 'head'})
        state_ids = {id(p) for p in network.state.parameters()}
        head_ids = {id(p) for p in network.head.parameters()}
        self.assertEqual({id(p) for p in clip.call_args_list[0].args[0]}, state_ids)
        self.assertEqual({id(p) for p in clip.call_args_list[1].args[0]}, head_ids)
        self.assertTrue(state_ids.isdisjoint(head_ids))

    def test_prototype_optimizer_and_prediction_save_load(self):
        network = self.make_model()
        optimizer = fitting.make_optimizer(network)
        original = inputs(False)
        prediction = network(original, protocol.ARMS[1])
        parts = model.losses(prediction, targets(), targets(), network.volume_mean, network.volume_scale)
        model.finite_step(network, optimizer, parts['total'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'prototype.pt'
            torch.save({'state_dict': network.state_dict(), 'optimizer': optimizer.state_dict()}, path)
            payload = torch.load(path, weights_only=False, map_location='cpu')
            restored = self.make_model()
            restored.load_state_dict(payload['state_dict'], strict=True)
            restored_optimizer = fitting.make_optimizer(restored)
            restored_optimizer.load_state_dict(payload['optimizer'])
            self.assertEqual(fitting.tensor_hash(network.state_dict()), fitting.tensor_hash(restored.state_dict()))
            self.assertEqual(len(optimizer.state), len(restored_optimizer.state))
            with torch.inference_mode():
                before, after = network(original, protocol.ARMS[1]), restored(original, protocol.ARMS[1])
            for field in fields(before):
                self.assertTrue(torch.equal(getattr(before, field.name), getattr(after, field.name)))


class JointDynamicsDataTests(unittest.TestCase):
    def test_loader_spy_latest_never_reads_earlier_or_future_mri(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            visits = {}
            for vid in ('earlier', 'cutoff', 'future'):
                path = root / f'{vid}.npy'
                np.save(path, np.full((4, 96, 96, 96), 0.3, dtype=np.float16), allow_pickle=False)
                visits[vid] = {'availability': [1.] * 4, 'image_file': path.name,
                               'image_sha256': protocol.digest(path)}
            protocol.save_json(root / 'prepared.json', {'schema': 'r38-prepared-v3', 'rows': [asdict(row())],
                'visits': visits, 'normalizer': {}, 'canonical_control_estimates': {}, 'splits': {}})
            calls = []
            def loader(path, **kwargs):
                calls.append(path.name)
                return np.load(path, **kwargs)
            prepared = data.PreparedData(root, device='cpu', loader=loader)
            for arm in protocol.ARMS:
                calls.clear()
                values = prepared.inputs(row(), arm)
                values.validate()
                self.assertEqual(calls, ['earlier.npy', 'cutoff.npy'] if arm.history else ['cutoff.npy'])
                self.assertEqual(values.images.dtype, torch.float32)
            for changed in (replace(row(), history_visit_ids=('earlier', 'future')),
                            replace(row(), gap_log1p=(0., 0., 0., 99.)),
                            replace(row(), target_visit_id='earlier')):
                with self.assertRaises(ValueError):
                    prepared.inputs(changed, protocol.ARMS[0])

    def test_source_only_353_normalizer_includes_noncutoff_visits(self):
        patients = [f'p{i:02}' for i in range(52)]
        visits = [{'visit_id': f'v{i:03}', 'patient_id': patients[i % 52],
                   'values_mm3': dict.fromkeys(protocol.COMPARTMENTS, float(i + 1))} for i in range(353)]
        values = np.array([data.measured(v) for v in visits])
        saved = {'mean': values.mean(0), 'scale': values.std(0, ddof=0).clip(0.5)}
        outsider = {'visit_id': 'validation-only', 'patient_id': 'validation',
                    'values_mm3': dict.fromkeys(protocol.COMPARTMENTS, 1e30)}
        result = data.source_normalizer(visits + [outsider], patients, saved)
        self.assertEqual(result.visit_ids, tuple(v['visit_id'] for v in visits))
        self.assertNotIn('validation-only', result.visit_ids)
        for invalid in (visits[:-1], visits[:-1] + [visits[0]], visits + [dict(outsider, patient_id=patients[0])]):
            with self.assertRaises(ValueError):
                data.source_normalizer(invalid, patients, saved)
        with self.assertRaises(ValueError):
            data.source_normalizer(visits, patients, {'mean': [0.] * 3, 'scale': [0.5] * 3})

    def test_outgoing_zero_gap_is_not_row_eligibility_filter(self):
        visits = []
        for index, visit in enumerate(('week-0', 'week-1', 'week-1-1')):
            visits.append({'cohort': 'LUMIERE', 'visit_id': str(index), 'patient_id': 'p', 'visit': visit,
                'values_mm3': dict.fromkeys(protocol.COMPARTMENTS, 1.), 'mask_relative': f'{index}.nii.gz',
                'labels': dict(zip(protocol.COMPARTMENTS, (1, 2, 3))), 'pregrid': {'same': True}})
        prepared = {'visits': visits, 'pairs': [
            {'source_visit_id': str(i), 'target_visit_id': str(i + 1), 'usable': True,
             'exclusions': [], 'patient_id': 'p'} for i in range(2)]}
        rows = data.derive_rows({'visits': visits}, {'visits': visits}, prepared, {'source52': ['p']})
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]['target_visit_id'], '2')
        broken = copy.deepcopy(visits)
        broken.insert(1, dict(visits[0], visit_id='unusable', visit='week-0-1'))
        rows = data.derive_rows({'visits': broken}, {'visits': visits}, prepared, {'source52': ['p']})
        self.assertEqual([r['row_id'] for r in rows], ['1->2'])
        self.assertEqual(rows[0]['history_visit_ids'], ['1'])

    def test_fixed_schedule_patient_uniform_replacement_and_matched_orders(self):
        rows = tuple(row('a', str(i)) for i in range(100)) + (row('b', 'b'),)
        order = fitting.replacement_order(rows, 42)
        self.assertEqual(order, fitting.replacement_order(rows, 42))
        self.assertNotEqual(order, fitting.replacement_order(rows, 43))
        self.assertEqual(len(order), 20)
        self.assertTrue(all(len(epoch) == 262 for epoch in order))
        fraction = sum(r == 'rb' for epoch in order for r in epoch) / 5240
        self.assertGreater(fraction, 0.45)
        self.assertLess(fraction, 0.55)
        self.assertEqual(protocol.PLAN.total_updates, 62880)
        self.assertEqual(protocol.PLAN.seeds, (42, 43, 44))


class JointDynamicsArtifactTests(unittest.TestCase):
    def test_incomplete_stale_extra_and_mismatched_marker_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                protocol.verify_stage(root, 'fit', 'contract')
            protocol.save_json(root / 'payload.json', {'value': 1})
            protocol.seal_stage(root, 'fit', 'contract', {'stages': [{'status': 'succeeded'}]})
            protocol.verify_stage(root, 'fit', 'contract')
            with self.assertRaises(ValueError):
                protocol.verify_stage(root, 'fit', 'other-contract')
            with self.assertRaises(ValueError):
                protocol.verify_stage(root, 'prepare', 'contract')
            protocol.save_json(root / 'payload.json', {'value': 2})
            with self.assertRaises(ValueError):
                protocol.verify_stage(root, 'fit', 'contract')
            protocol.save_json(root / 'payload.json', {'value': 1})
            (root / 'extra').write_text('unbound')
            with self.assertRaises(ValueError):
                protocol.verify_stage(root, 'fit', 'contract')
            with self.assertRaises(FileExistsError):
                protocol.seal_stage(root, 'fit', 'contract', {})

    def test_sealed_partial_matrix_is_not_a_completed_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol.save_json(root / 'fitted.json', {'cells': {}, 'total_updates': 62880,
                                'seeds': [42, 43, 44], 'contract_digest': 'contract'})
            protocol.seal_stage(root, 'fit', 'contract', {})
            with self.assertRaises(ValueError):
                protocol.validate_completed_fit(root, 'contract')

    def test_implementation_plan_identity_survives_json_lock_roundtrip(self):
        with patch.object(protocol, 'source_hashes', return_value={}), \
                patch.object(protocol, 'packet_identity', return_value={}):
            identity = protocol.implementation_identity()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'identity.json'
            protocol.save_json(path, identity)
            self.assertEqual(protocol.read_json(path), identity)

    def test_resource_phase_set_requires_assessment_and_protected_policy_caps(self):
        self.assertIn('assessment', fitting.MEASURED_PHASES)
        self.assertEqual(sum(v[0] for v in fitting.RESERVES.values()), 3600.)
        self.assertEqual(sum(v[1] for v in fitting.RESERVES.values()), 900.)


class JointDynamicsRunnerTests(unittest.TestCase):
    def test_actual_cli_dispatch_uses_existing_factory_never_initializes(self):
        runner = runner_module()
        from src.harness.analysis.joint_dynamics import authority
        commands = [(['prepare', '--stage-name', 'p'], 'prepare'),
                    (['preflight', '--stage-name', 'p', '--prepared-stage', 's'], 'preflight'),
                    (['lock', '--stage-name', 'l', '--prepared-stage', 'p', '--preflight-stage', 'f',
                      '--approval', 'approval.json', '--resource-plan', 'plan.json'], 'lock'),
                    (['fit'], 'fit'), (['assess'], 'assess')]
        from unittest.mock import Mock
        existing = Mock()
        for argv, target in commands:
            with patch.object(runner, 'require_remote'), patch.object(runner, 'runner_lock') as lock, \
                    patch.object(authority, 'production_authority', return_value=existing), \
                    patch.object(runner, target, return_value={'status': 'dispatch-only'}) as dispatch:
                lock.return_value.__enter__.return_value = None
                self.assertEqual(runner.main(argv), 0)
                self.assertIs(dispatch.call_args.args[0], existing)
        existing.initialize.assert_not_called()
        existing.adopt_initialized_authority.assert_not_called()

    def test_stage_entry_no_reentrant_verification_and_exact_prefix(self):
        runner = runner_module()
        class Authority:
            held = False
            corrupt = False
            def verify_extension(self, snapshot):
                if self.held:
                    raise AssertionError('non-reentrant authority lock')
            def snapshot(self):
                return {'authority_id': 'id', 'authorization_sha256': 'sha', 'identity': {}, 'stages': []}
            @contextmanager
            def stage(self, name, **kwargs):
                self.held = True
                try:
                    yield self
                finally:
                    self.held = False
        authority = Authority()
        entry = authority.snapshot()
        def current():
            return {**entry, 'authority_id': 'wrong' if authority.corrupt else 'id',
                    'stages': [{'name': 'test', 'status': 'running', 'scientific': False}]}
        with patch.object(authority, 'snapshot', side_effect=lambda: current() if authority.held else entry):
            with runner.charged_stage(authority, 'test', gpu=False, scientific=False, prior_snapshots=(entry,)):
                self.assertTrue(authority.held)
            authority.corrupt = True
            with self.assertRaises(RuntimeError):
                with runner.charged_stage(authority, 'test', gpu=False, scientific=False):
                    self.fail('identity mismatch must stop before substantive work')

    def test_assessment_promotes_only_after_successful_budget_exit(self):
        runner = runner_module()
        from src.harness.analysis.joint_dynamics import assessment, baselines
        from unittest.mock import Mock
        for fail_exit in (False, True):
            with self.subTest(fail_exit=fail_exit), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fit_root = root / 'r38-fit' / 'fit'
                fit_root.mkdir(parents=True)
                contract = {'prepared_stage': 'prepared', 'source_sha256': {}}
                protocol.save_json(fit_root / 'control_fits.json', {'fits': {},
                    'contract_digest': protocol.canonical_hash(contract),
                    'source_row_ids': [], 'validation_row_ids': []})
                protocol.save_json(fit_root / 'fitted.json', {'control_fits': {}, 'cohort_rows': {},
                    'calibration': {}, 'source_sha256': {},
                    'control_fits_sha256': protocol.digest(fit_root / 'control_fits.json')})
                prepared = Mock()
                prepared.select.return_value = ()
                np.savez(fit_root / 'private_predictions.npz', seeds=np.array([42, 43, 44]))
                report = {'status': 'staged_only'}
                guard = Mock()
                @contextmanager
                def charged(*args, **kwargs):
                    yield guard
                    self.assertFalse((root / 'r38-assess' / 'aggregate.json').exists())
                    self.assertEqual(protocol.read_json(root / 'r38-assess' / 'staged_aggregate.json')['status'], 'staged_only')
                    if fail_exit:
                        raise TimeoutError('charged final exit rejected')
                portable = Mock()
                portable.to_dict.return_value = {}
                resource = {'phases': {'assessment': {'cpu_seconds': 1e6, 'wall_seconds': 1e6}}}
                with patch.object(runner, 'STAGES', root), patch.object(runner, 'scientific_history'), \
                        patch.object(runner, 'locked_paths', return_value=()), \
                        patch.object(runner, 'charged_stage', side_effect=charged), \
                        patch.object(runner, 'load_lock', return_value=(contract, resource)), \
                        patch.object(runner, 'load_prepared', return_value=(prepared, {})), \
                        patch.object(runner, 'validate_completed_fit', return_value={'contract_digest': protocol.canonical_hash(contract)}), \
                        patch.object(runner, '_verify_fitted_identity'), \
                        patch.object(runner, 'source_hashes', return_value={}), \
                        patch.object(baselines.ControlFits, 'from_dict', return_value=portable), \
                        patch.object(assessment, 'analyze', return_value=report):
                    with self.subTest(bound_contract=True):
                        if fail_exit:
                            with self.assertRaises(TimeoutError):
                                runner.assess(Mock())
                            self.assertFalse((root / 'r38-assess' / 'aggregate.json').exists())
                            self.assertFalse((root / 'r38-assess' / 'complete.json').exists())
                        else:
                            authority = Mock()
                            authority.snapshot.return_value = {}
                            runner.assess(authority)
                            self.assertEqual(protocol.read_json(root / 'r38-assess' / 'aggregate.json')['status'],
                                             'completed_not_independently_reviewed')
                            self.assertTrue((root / 'r38-assess' / 'complete.json').exists())


if __name__ == '__main__':
    unittest.main()
