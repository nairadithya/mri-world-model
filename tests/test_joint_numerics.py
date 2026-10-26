from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import ndtri

from src.harness.analysis.joint_dynamics import assessment as score
from src.harness.analysis.joint_dynamics import baselines as controls
from src.harness.analysis.joint_dynamics.baselines import (
    ControlFitData, ControlFits, ControlInputs, FilterHistory, SourceChain,
    calibrate_gaussian, fit_controls, fit_persistence, gaussian_calibration_receipt,
    predict_controls,
)


def _inputs(x: np.ndarray) -> ControlInputs:
    return ControlInputs(x, tuple(FilterHistory(row[None, :].copy(), np.zeros(1)) for row in x))


def _chain(patient: str, names: tuple[str, ...], values: list[float], gaps: list[float]) -> SourceChain:
    return SourceChain(patient, names, np.repeat(np.asarray(values, float)[:, None], 3, axis=1),
                       np.asarray(gaps, float))


def _fit_data() -> tuple[ControlFitData, ControlFitData]:
    x = np.zeros((3, 3))
    source = ControlFitData(_inputs(x), np.asarray([[1, 1, 1], [1, 1, 1], [3, 3, 3]], float),
                            np.asarray(["a", "a", "b"]), np.asarray(["s0", "s1", "s2"]),
                            (_chain("a", ("a0", "a1", "a2"), [0, 1, 4], [0, 1, 2]),
                             _chain("a", ("a3", "a4"), [0, 2], [0, 3]),
                             _chain("b", ("b0", "b1"), [0, 3], [0, 2]),
                             _chain("b", ("isolated",), [7], [0])), np.asarray([0, 2, 4], float))
    validation = ControlFitData(_inputs(np.zeros((2, 3))), np.asarray([[0.2]*3, [3.0]*3]),
                                np.asarray(["v0", "v1"]), np.asarray(["vrow0", "vrow1"]),
                                (), np.zeros(0))
    return source, validation


def _production_bundle(fits: ControlFits) -> tuple[dict[str, np.ndarray], dict]:
    arrays = {"seeds": np.asarray(score.SEEDS, dtype=np.int64)}
    identity = {"contract_digest": "a"*64, "source_sha256": "b"*64,
                "fitted_manifest_sha256": "c"*64, "control_fits": fits.to_dict(),
                "cohort_rows": {}, "calibration": dict(fits.state["calibration"])}
    for arm_index, arm in enumerate(score.ARMS):
        for seed_index, seed in enumerate(score.SEEDS):
            identity["calibration"][f"{arm}_{seed}"] = {
                "forecast": controls.GRID[(arm_index+seed_index) % 3],
                "persistence": controls.GRID[seed_index], "persistence_rms": [1.0, 1.5, 2.0]}
    for cohort, (n, patients) in (("primary", (75, 18)), ("single", (26, 23))):
        rows = []
        for i in range(n):
            p = f"p{i % patients:02d}"
            cutoff = [0.0 if (i+c) % 3 == 0 else 1.0 for c in range(3)]
            truth = [0.0 if (i+c) % 4 == 0 else 2.0 for c in range(3)]
            rows.append({"row_id": f"{cohort}{i}", "patient_id": p,
                         "cutoff_visit_id": f"{p}/{cohort}/cutoff{i}",
                         "target_visit_id": f"{p}/{cohort}/target{i}",
                         "truth": truth, "cutoff_measured": cutoff})
        identity["cohort_rows"][cohort] = rows
        truth, cutoff = np.asarray([r["truth"] for r in rows]), np.asarray([r["cutoff_measured"] for r in rows])
        arrays[cohort+"_truth"] = truth
        arrays[cohort+"_cutoff_measured"] = cutoff
        arrays[cohort+"_row_ids"] = np.asarray([r["row_id"] for r in rows])
        arrays[cohort+"_patient_ids"] = np.asarray([r["patient_id"] for r in rows])
        q = cutoff+0.25
        common = predict_controls(fits, _inputs(q))
        for name, prediction in common.items():
            for field in (("centers", "weights", "observation_sd") if name == "E"
                          else ("mean", "intrinsic_scale", "calibration")):
                arrays[f"{cohort}_{name}_{field}"] = np.asarray(prediction[field], dtype=float)
        for arm in score.ARMS:
            for seed_index, seed in enumerate(score.SEEDS):
                prefix = f"{cohort}_{arm}_{seed}"
                receipt = identity["calibration"][f"{arm}_{seed}"]
                error = {"OH": 0.3, "OL": 0.5, "JH": 0.1, "JL": 0.2}[arm]+seed_index*0.01
                values = {"mean": truth+error, "intrinsic_scale": np.full((n, 3), 0.5),
                          "calibration": receipt["forecast"], "current_mean": cutoff+0.1,
                          "current_dice": np.where(cutoff == 0, 1.0, 0.8),
                          "future_dice": np.where(truth == 0, 1.0, 0.7),
                          "persistence_rms": receipt["persistence_rms"],
                          "persistence_calibration": receipt["persistence"]}
                for field, value in values.items():
                    arrays[prefix+"_"+field] = np.asarray(value, dtype=float)
    nuisance_rows = deepcopy(identity["cohort_rows"]["primary"][:18])
    identity["cohort_rows"]["nuisance"] = nuisance_rows
    for field in ("truth", "cutoff_measured", "row_ids", "patient_ids"):
        arrays["nuisance_"+field] = arrays["primary_"+field][:18].copy()
    for arm in score.ARMS:
        for condition in score.CONDITIONS:
            for seed in score.SEEDS:
                prefix = f"nuisance_{arm}_{condition}_{seed}"
                for field in ("mean", "intrinsic_scale", "calibration", "current_mean"):
                    value = arrays[f"primary_{arm}_{seed}_{field}"]
                    arrays[prefix+"_"+field] = value.copy() if value.ndim == 0 else value[:18].copy()
                if condition != "identity":
                    arrays[prefix+"_mean"] += 0.01 if condition == "intensity" else 0.02
    return arrays, identity


class LiteralScoreTests(unittest.TestCase):
    def test_literal_gaussian_values_and_invalid_scales(self):
        got = score.gaussian_crps([0.0, 0.0], [1.0, 1.0], [0.0, 1.0])
        np.testing.assert_allclose(got, [0.23369497725510913, 0.6024413576276163], atol=1e-14)
        for scale in (0.0, -1.0, np.nan):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                score.gaussian_crps(0.0, scale, 0.0)

    def test_literal_weighted_gaussian_mixture(self):
        centers = np.asarray([[[-1.0]*3, [1.0]*3]])
        got = score.mixture_crps(centers, np.asarray([0.25, 0.75]), np.zeros((1, 3)), 1.0)
        np.testing.assert_allclose(got, np.full((1, 3), 0.4201669983355203), atol=1e-14)
        single = score.mixture_crps(np.zeros((1, 1, 3)), np.ones(1), np.zeros((1, 3)), 1.0)
        np.testing.assert_allclose(single, np.full((1, 3), 0.23369497725510913), atol=1e-14)

    def test_mixture_rejects_bad_weights_noise_and_nonfinite_values(self):
        for weights, noise in (([0.2, 0.2], 0.05), ([-0.1, 1.1], 0.05), ([0.5, 0.5], 0.0)):
            with self.subTest(weights=weights, noise=noise), self.assertRaises(ValueError):
                score.mixture_crps(np.zeros((1, 2, 3)), weights, np.zeros((1, 3)), noise)
        with self.assertRaises(ValueError):
            score.mixture_crps(np.full((1, 2, 3), np.nan), [0.5, 0.5], np.zeros((1, 3)))

    def test_patient_weighting_is_not_row_weighting(self):
        ids = np.asarray(["a", "a", "b"])
        np.testing.assert_allclose(controls.patient_weights(ids), [0.25, 0.25, 0.5])
        rms = fit_persistence(np.zeros((3, 3)), np.asarray([[1]*3, [1]*3, [3]*3]), ids)
        np.testing.assert_allclose(rms, np.sqrt([5, 5, 5]))
        self.assertNotAlmostEqual(float(rms[0]), np.sqrt(11/3))
        np.testing.assert_allclose(score.patient_values(np.asarray([1, 1, 3]), ids), [1, 3])
        for bad_ids in (["a"], [], ["", "a", "b"]):
            with self.subTest(ids=bad_ids), self.assertRaises(ValueError):
                fit_persistence(np.zeros((3, 3)), np.zeros((3, 3)), bad_ids)

    def test_zero_intrinsic_scale_floor_and_tie_order(self):
        zero = np.zeros((2, 3))
        receipt = gaussian_calibration_receipt(zero, zero, zero, ["v0", "v1"])
        self.assertEqual(receipt["selected"], 0.5)
        np.testing.assert_allclose(receipt["grid_scores"], [0.011684748862755457]*3, atol=1e-15)
        self.assertEqual(calibrate_gaussian(zero, zero, zero, ["v0", "v1"]), 0.5)
        with self.assertRaises(ValueError):
            calibrate_gaussian(zero, zero-1, zero, ["v0", "v1"])

    def test_every_gaussian_candidate_includes_observation_floor(self):
        mean = np.zeros((3, 3)); scale = np.full((3, 3), 0.02)
        truth = np.asarray([[0.1]*3, [0.1]*3, [0.0]*3])
        receipt = gaussian_calibration_receipt(mean, scale, truth, ["a", "a", "b"])
        expected = [float(np.sum(np.asarray([0.25, 0.25, 0.5])[:, None]*score.gaussian_crps(
            mean, np.sqrt((k*scale)**2+0.0025), truth))/3) for k in (0.5, 1, 2)]
        np.testing.assert_allclose(receipt["grid_scores"], expected, atol=1e-15)
        self.assertEqual(receipt["selected"], (0.5, 1, 2)[int(np.argmin(expected))])

    def test_mixture_quantiles_and_gaussian_intervals(self):
        centers = np.zeros((1, 1, 3)); weights = np.ones(1)
        for level in score.LEVELS:
            lo, hi = score._intervals_mixture(centers, weights, level)
            np.testing.assert_allclose(lo, -ndtri((1+level)/2)*0.05, atol=1e-14)
            np.testing.assert_allclose(hi, ndtri((1+level)/2)*0.05, atol=1e-14)
            gl, gh = score._intervals_gaussian(np.zeros((1, 3)), np.full((1, 3), 0.05), level)
            np.testing.assert_allclose(lo, gl, atol=1e-14)
            np.testing.assert_allclose(hi, gh, atol=1e-14)


class FilterControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source, cls.validation = _fit_data()
        cls.fits = fit_controls(cls.source, cls.validation)

    def test_hand_computed_filter_rts_and_lag_covariance(self):
        mean, variance, lag = controls._smooth(np.asarray([0.0, 2.0]), np.asarray([0.0, 1.0]), 0, 1, 1)
        np.testing.assert_allclose(mean, [2/3, 4/3], atol=1e-14)
        np.testing.assert_allclose(variance, [2/3, 2/3], atol=1e-14)
        np.testing.assert_allclose(lag, [1/3], atol=1e-14)
        d = mean[1]-mean[0]
        expected_square = d*d+variance[1]+variance[0]-2*lag[0]
        self.assertAlmostEqual(expected_square, 10/9, places=14)
        self.assertAlmostEqual((0-mean[0])**2+variance[0], 10/9, places=14)
        history = FilterHistory(np.asarray([[0]*3, [2]*3], float), np.asarray([0, 1], float))
        parameters = [{"final": [0, 1, 1]} for _ in range(3)]
        post, var = controls._posterior(history, parameters)
        np.testing.assert_allclose(post, [4/3]*3, atol=1e-14)
        np.testing.assert_allclose(var, [2/3]*3, atol=1e-14)
        single, single_var = controls._posterior(FilterHistory(np.full((1, 3), 5.0), np.zeros(1)), parameters)
        np.testing.assert_array_equal(single, [5, 5, 5])
        np.testing.assert_array_equal(single_var, [1, 1, 1])

    def test_hand_computed_first_em_update_includes_initial_state(self):
        records = controls._chain_records((_chain("a", ("v0", "v1"), [0, 2], [0, 1]),), {"a"})
        fitted = controls._fit_filter(records)
        for parameter in fitted["parameters"]:
            np.testing.assert_allclose(parameter["initial"], [2, 0.001, 0.001], atol=1e-15)
            np.testing.assert_allclose(parameter["updates"][0], [2, 1/1500, 1/1500], atol=1e-14)
            self.assertEqual(len(parameter["updates"]), 30)
            np.testing.assert_allclose(parameter["final"], [2, 1e-6, 1e-6], atol=1e-14)

    def test_patient_interval_weights_and_distinct_node_edge_denominators(self):
        f = self.fits.state["filter"]
        self.assertEqual(f["patient_interval_counts"], {"a": 3, "b": 1})
        self.assertEqual(f["patient_node_counts"], {"a": 5, "b": 2})
        self.assertEqual(f["patient_weights"], {"a": 1/6, "b": 1/2})
        self.assertAlmostEqual(f["edge_denominator"], 1)
        self.assertAlmostEqual(f["node_denominator"], 11/6)
        self.assertEqual(f["singleton_exclusions"], 1)
        self.assertEqual(f["chain_membership"][0]["edges"], [["a0", "a1"], ["a1", "a2"]])
        self.assertEqual(f["chain_membership"][3]["visit_ids"], ["isolated"])
        self.assertFalse(f["chain_membership"][3]["included"])

    def test_em_m_step_uses_new_drift_and_all_node_residuals(self):
        f = self.fits.state["filter"]
        for c, p in enumerate(f["parameters"]):
            old_b, old_q, old_r = p["initial"]
            moments = []; r_sum = 0.0; time_sum = 0.0; d_sum = 0.0
            for ch in f["chain_membership"]:
                if not ch["included"]:
                    continue
                w = f["patient_weights"][ch["patient_id"]]
                y = np.asarray(ch["estimates"])[:, c]; dt = np.asarray(ch["gaps_weeks"])
                ms, vs, lag = controls._smooth(y, dt, old_b, old_q, old_r)
                d = np.diff(ms); v = d*d+vs[1:]+vs[:-1]-2*lag
                moments.append((w, d, v, dt[1:]))
                d_sum += w*d.sum(); time_sum += w*dt[1:].sum()
                r_sum += w*np.sum((y-ms)**2+vs)
            b = d_sum/time_sum
            q = max(1e-6, sum(w*np.sum((v-2*b*dt*d+b*b*dt*dt)/dt)
                              for w, d, v, dt in moments)/f["edge_denominator"])
            r = max(1e-6, r_sum/f["node_denominator"])
            np.testing.assert_allclose(p["updates"][0], [b, q, r], atol=1e-14)

    def test_em_is_finite_deterministic_and_portable(self):
        again = fit_controls(self.source, self.validation)
        self.assertEqual(again.to_dict(), self.fits.to_dict())
        portable = json.loads(json.dumps(self.fits.to_dict(), allow_nan=False))
        restored = ControlFits.from_dict(portable)
        self.assertEqual(restored.to_dict(), portable)
        query = _inputs(np.zeros((1, 3)))
        for name, value in predict_controls(restored, query).items():
            for field, original in predict_controls(self.fits, query)[name].items():
                if field != "kind":
                    np.testing.assert_array_equal(value[field], original)
        tampered = deepcopy(portable); tampered["state"]["e_changes"][0][0] += 1
        with self.assertRaises(ValueError):
            ControlFits.from_dict(tampered)
        tampered["state_sha256"] = controls._digest(tampered["state"])
        with self.assertRaises(ValueError):
            ControlFits.from_dict(tampered)

    def test_nonpositive_outgoing_and_singleton_preserve_all_p_e_rows(self):
        state = self.fits.state
        self.assertEqual(state["outgoing_nonpositive_exclusions"], 1)
        self.assertEqual(state["typical_outgoing_weeks"], 3)
        self.assertEqual(state["source"]["row_ids"], ["s0", "s1", "s2"])
        np.testing.assert_allclose(state["e_weights"], [0.25, 0.25, 0.5])
        np.testing.assert_array_equal(state["e_changes"], self.source.truth)
        np.testing.assert_allclose(state["p_rms"], np.sqrt([5, 5, 5]))
        with self.assertRaises(ValueError):
            fit_controls(replace(self.source, chains=(_chain("a", ("x", "y"), [0, 1], [0, 0]),)), self.validation)
        with self.assertRaises(ValueError):
            fit_controls(replace(self.source, outgoing_weeks=np.asarray([0, -2, 0])), self.validation)

    def test_empirical_centers_preserve_mean_and_use_selected_grid_noise(self):
        state = self.fits.state
        self.assertEqual(state["e_mean"], [2, 2, 2])
        receipt = state["calibration_receipts"]["E"]
        expected = []
        val = np.asarray(state["validation"]["estimates"])
        changes = np.asarray(state["e_changes"]); m = np.asarray(state["e_mean"])
        weights = np.asarray(state["e_weights"])
        for k in (0.5, 1, 2):
            centers = val[:, None, :]+m+k*(changes-m)
            crps = score.mixture_crps(centers, weights, self.validation.truth, 0.05)
            expected.append(float(crps.mean()))
            np.testing.assert_allclose(np.sum(centers*weights[None, :, None], axis=1), val+m)
        np.testing.assert_allclose(receipt["grid_scores"], expected)
        self.assertEqual(receipt["selected"], (0.5, 1, 2)[int(np.argmin(expected))])
        prediction = predict_controls(self.fits, _inputs(np.full((1, 3), 5.0)))["E"]
        np.testing.assert_allclose(prediction["centers"], 5+m+receipt["selected"]*(changes-m)[None, :, :])
        self.assertEqual(prediction["observation_sd"], 0.05)

    def test_fp_ssm_share_posterior_and_source_only_horizon(self):
        history = FilterHistory(np.asarray([[1]*3, [3]*3], float), np.asarray([0, 2], float))
        inputs = ControlInputs(np.full((1, 3), 3.0), (history,))
        prediction = predict_controls(self.fits, inputs)
        parameters = self.fits.state["filter"]["parameters"]
        mean, variance = controls._posterior(history, parameters)
        np.testing.assert_allclose(prediction["FP"]["mean"][0], mean)
        np.testing.assert_allclose(prediction["FP"]["intrinsic_scale"][0]**2, variance)
        drift = np.asarray([p["final"][0] for p in parameters])
        q = np.asarray([p["final"][1] for p in parameters])
        np.testing.assert_allclose(prediction["SSM"]["mean"][0], mean+drift*3)
        np.testing.assert_allclose(prediction["SSM"]["intrinsic_scale"][0]**2, variance+q*3)
        self.assertEqual([f.name for f in fields(ControlInputs)], ["estimates", "histories"])
        self.assertEqual([f.name for f in fields(FilterHistory)], ["estimates", "gaps_weeks"])
        with self.assertRaises(ValueError):
            predict_controls(self.fits, self.source)

    def test_bad_histories_source_membership_and_validation_boundaries(self):
        for history in (FilterHistory(np.zeros((5, 3)), np.asarray([0, 1, 1, 1, 1])),
                        FilterHistory(np.zeros((2, 3)), np.asarray([0, -1])),
                        FilterHistory(np.zeros((1, 3)), np.ones(1)),
                        FilterHistory(np.full((1, 3), np.nan), np.zeros(1))):
            with self.subTest(history=history), self.assertRaises(ValueError):
                predict_controls(self.fits, ControlInputs(np.zeros((1, 3)), (history,)))
        with self.assertRaises(ValueError):
            predict_controls(self.fits, ControlInputs(np.ones((1, 3)), (FilterHistory(np.zeros((1, 3)), np.zeros(1)),)))
        with self.assertRaises(ValueError):
            fit_controls(self.source, replace(self.validation, chains=self.source.chains))
        with self.assertRaises(ValueError):
            fit_controls(self.source, replace(self.validation, patient_ids=np.asarray(["a", "v1"])))
        with self.assertRaises(ValueError):
            fit_controls(replace(self.source, chains=self.source.chains+self.source.chains[:1]), self.validation)

    def test_zero_changes_negative_estimates_and_filter_floors(self):
        x = np.full((2, 3), -2.0)
        source = ControlFitData(_inputs(x), x.copy(), np.asarray(["a", "a"]), np.asarray(["r0", "r1"]),
                                (_chain("a", ("x", "y"), [-2, -2], [0, 1]),), np.ones(2))
        validation = ControlFitData(_inputs(x), x.copy(), np.asarray(["v", "v"]), np.asarray(["v0", "v1"]), (), np.zeros(0))
        fits = fit_controls(source, validation)
        np.testing.assert_array_equal(fits.state["p_rms"], [0, 0, 0])
        np.testing.assert_array_equal(fits.state["e_changes"], np.zeros((2, 3)))
        for control, prediction in predict_controls(fits, _inputs(x)).items():
            if control == "E":
                np.testing.assert_array_equal(prediction["centers"], np.full((2, 2, 3), -2.0))
            else:
                np.testing.assert_allclose(prediction["mean"], x)
        for p in fits.state["filter"]["parameters"]:
            self.assertGreaterEqual(p["final"][1], 1e-6)
            self.assertGreaterEqual(p["final"][2], 1e-6)


class ContrastAndGateTests(unittest.TestCase):
    def setUp(self):
        self.draws = np.random.default_rng(2026).integers(0, 18, size=(10000, 18))

    def test_all_ten_fixed_signs_and_relative_denominators(self):
        values = {"JH": 1, "OH": 3, "JL": 2, "OL": 5, "P": 6, "E": 7, "FP": 8, "SSM": 9,
                  "P_JH": 4, "P_OH": 4}
        contrasts = score._contrasts({k: np.full(18, v, float) for k, v in values.items()}, self.draws)
        expected = {"JH-OH": -2, "JL-OL": -3, "JH-JL": -1, "OH-OL": -2,
                    "JH-JL-OH+OL": 1, "JH-P": -5, "JH-E": -6,
                    "JH-FP": -7, "JH-SSM": -8, "(P_JH-JH)-(P_OH-OH)": 2}
        self.assertEqual(set(contrasts), set(expected))
        for name, difference in expected.items():
            self.assertEqual(contrasts[name]["difference"], difference)
            self.assertEqual(contrasts[name]["ci_99_5"], [difference, difference])
            self.assertEqual(contrasts[name]["ci_95"], [difference, difference])
        self.assertEqual(contrasts["JH-OH"]["denominator_crps"], 3)
        self.assertEqual(contrasts["JH-OH"]["relative_denominator"], "OH")
        self.assertEqual(contrasts["JH-OH"]["relative_reduction"], 2/3)
        self.assertEqual(contrasts["JH-JL"]["relative_reduction"], 0.5)
        self.assertIsNone(contrasts["JH-JL-OH+OL"]["relative_denominator"])
        self.assertEqual(contrasts["JH-JL-OH+OL"]["interpretation"], "larger_gradient_advantage_with_latest")
        self.assertEqual(contrasts["(P_JH-JH)-(P_OH-OH)"]["adjusted_result"], "favorable")
        values["OL"] = 3
        negative = score._contrasts({k: np.full(18, v, float) for k, v in values.items()}, self.draws)
        self.assertEqual(negative["JH-JL-OH+OL"]["difference"], -1)
        self.assertEqual(negative["JH-JL-OH+OL"]["interpretation"], "larger_gradient_advantage_with_history")
        values["OL"] = 4
        unresolved = score._contrasts({k: np.full(18, v, float) for k, v in values.items()}, self.draws)
        self.assertEqual(unresolved["JH-JL-OH+OL"]["interpretation"], "unresolved")

    def test_unresolved_is_not_equivalence_or_harm(self):
        item = score.paired_interval(np.asarray([-1, 1]*9, float), self.draws)
        self.assertEqual(score._direction(item), "unresolved")
        self.assertEqual(score._direction(score.paired_interval(np.ones(18), self.draws)), "harmful")
        self.assertEqual(score._direction(score.paired_interval(-np.ones(18), self.draws)), "favorable")
        self.assertEqual(score._direction(score.paired_interval(np.ones(18), self.draws), True), "favorable")
        exact_zero = score.paired_interval(np.zeros(18), self.draws)
        self.assertEqual(score._direction(exact_zero), "unresolved")

    def test_retention_both_joint_arms_all_compartments_and_loss_directions(self):
        ids = np.asarray([f"p{i}" for i in range(18)])
        metrics = {a: {"current_mae": np.full((18, 3), 1.0),
                       "current_dice": np.full((18, 3), 0.8)} for a in score.ARMS}
        metrics["JH"]["current_mae"][:, 0] += 0.04
        metrics["JH"]["current_dice"][:, 0] -= 0.01
        good = score._retention(metrics, ids, self.draws)
        self.assertTrue(good["passes"])
        self.assertAlmostEqual(good["JH-OH"]["current_mae"]["per_compartment"][0]["difference"], 0.04)
        self.assertAlmostEqual(good["JH-OH"]["current_dice"]["per_compartment"][0]["difference"], 0.01)
        metrics["JL"]["current_mae"][:, 2] += 0.06
        self.assertFalse(score._retention(metrics, ids, self.draws)["passes"])
        metrics["JL"]["current_mae"][:, 2] -= 0.06
        metrics["JL"]["current_dice"][:, 1] -= 0.03
        self.assertFalse(score._retention(metrics, ids, self.draws)["JL-OL"]["passes"])
        metrics["JL"]["current_dice"][:, 1] = 0.9
        self.assertTrue(score._retention(metrics, ids, self.draws)["JL-OL"]["passes"])

    def test_five_percent_margin_coverage_bounds_and_each_control_flag(self):
        contrasts = {name: {"ci_99_5": [-0.2, -0.1], "relative_reduction": 0.05}
                     for name in ("JH-OH", "JH-JL", "JH-P", "JH-E", "JH-FP", "JH-SSM")}
        retention = {"JH-OH": {"passes": True}, "JL-OL": {"passes": True}, "passes": True}
        flags = score._capability_flags(contrasts, retention, np.asarray([0.7, 0.8, 0.9]), {"JH_passes": True})
        self.assertTrue(all(flags.values()))
        contrasts["JH-OH"]["relative_reduction"] = 0.04999
        flags = score._capability_flags(contrasts, retention, np.asarray([0.7, 0.8, 0.9]), {"JH_passes": True})
        self.assertFalse(flags["joint_material_reduction"])
        self.assertTrue(flags["joint_beats_observation"])
        contrasts["JH-SSM"]["ci_99_5"][1] = 0.0
        flags = score._capability_flags(contrasts, retention, np.asarray([0.69, 0.8, 0.9]), {"JH_passes": True})
        self.assertFalse(flags["beats_SSM"])
        self.assertFalse(flags["beats_all_common_controls"])
        self.assertTrue(flags["beats_P"])
        self.assertFalse(flags["coverage_80"])
        self.assertFalse(score._capability_flags(contrasts, retention, np.asarray([0.7, 0.8, 0.91]),
                                                 {"JH_passes": True})["coverage_80"])

    def test_retention_margins_are_strict_not_less_equal(self):
        ids = np.asarray([f"p{i}" for i in range(18)])
        metrics = {a: {"current_mae": np.zeros((18, 3)),
                       "current_dice": np.zeros((18, 3))} for a in score.ARMS}
        metrics["JH"]["current_mae"][:, 0] = 0.05
        boundary = {"difference": 0.05, "ci_95": [0.05, 0.05], "ci_99_5": [0.05, 0.05]}
        with patch.object(score, "paired_interval", side_effect=lambda *args: deepcopy(boundary)):
            result = score._retention(metrics, ids, self.draws)
        self.assertFalse(result["JH-OH"]["current_mae"]["per_compartment"][0]["passes"])
        metrics["JH"]["current_mae"][:] = 0
        metrics["OL"]["current_dice"][:, 2] = 0.02
        boundary = {"difference": 0.02, "ci_95": [0.02, 0.02], "ci_99_5": [0.02, 0.02]}
        with patch.object(score, "paired_interval", side_effect=lambda *args: deepcopy(boundary)):
            result = score._retention(metrics, ids, self.draws)
        self.assertFalse(result["JL-OL"]["current_dice"]["per_compartment"][2]["passes"])

    def test_empty_subset_reports_counts_and_null_not_nan(self):
        metrics = {"crps": np.ones((2, 3)), "mae": np.zeros((2, 3))}
        summary = score._summary(metrics, np.ones((2, 3)), np.ones((2, 3)), np.asarray(["a", "b"]))
        for name in ("empty_target", "appearance", "disappearance", "both_empty", "empty_cutoff"):
            for compartment in summary["sparse_subsets"][name]:
                self.assertEqual(compartment["rows"], 0)
                self.assertEqual(compartment["patients"], 0)
                self.assertIsNone(compartment["crps"])
                self.assertIsNone(compartment["mae"])
        json.dumps(summary, allow_nan=False)

    def test_shared_bootstrap_draws_literal_constant_and_random_pairing(self):
        differences = np.arange(18, dtype=float)
        got = score.paired_interval(differences, self.draws)
        literal = differences[self.draws].mean(axis=1)
        np.testing.assert_array_equal(got["ci_99_5"], np.quantile(literal, [0.0025, 0.9975]))
        np.testing.assert_array_equal(got["ci_95"], np.quantile(literal, [0.025, 0.975]))
        self.assertEqual(got["difference"], 8.5)


class ProductionAssessmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source, validation = _fit_data()
        cls.fits = fit_controls(source, validation)
        cls.arrays, cls.identity = _production_bundle(cls.fits)

    def bundle(self):
        return {k: v.copy() for k, v in self.arrays.items()}, deepcopy(self.identity)

    def test_complete_staged_only_json_and_sparse_masks(self):
        arrays, identity = self.bundle()
        original_keys = set(arrays)
        with tempfile.TemporaryDirectory() as directory:
            with patch("builtins.open", side_effect=AssertionError("analyze must not write files")), \
                 patch.object(Path, "write_text", side_effect=AssertionError("no publishing")):
                result = score.analyze(arrays, identity)
            self.assertEqual(list(Path(directory).iterdir()), [])
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["status"], "staged_only")
        self.assertEqual(result["independent_review"], "not_performed")
        self.assertNotIn("completed_budget_journal_tip", result)
        self.assertEqual(set(arrays), original_keys)
        self.assertEqual(result["cohorts"]["primary"]["rows"], 75)
        self.assertEqual(result["cohorts"]["single"]["patients"], 23)
        self.assertEqual(len(result["primary_contrasts"]), 10)
        self.assertEqual(result["protocol"]["bootstrap_draws"], 10000)
        self.assertEqual(result["protocol"]["bootstrap_seed"], 2026)
        self.assertEqual(result["scoped_capability_outcome"], "INCONCLUSIVE_FOR_FULL_CAPABILITY")
        self.assertFalse(result["capability_flags"]["coverage_80"])
        m = result["cohorts"]["primary"]["scores"]["JH"]["seed_score_average"]
        expected_current = score.patient_values(arrays["primary_JH_42_current_dice"],
                                                 arrays["primary_patient_ids"]).mean()
        expected_future = score.patient_values(arrays["primary_JH_42_future_dice"],
                                                arrays["primary_patient_ids"]).mean()
        self.assertAlmostEqual(m["current_dice"]["mean"], expected_current)
        self.assertAlmostEqual(m["future_dice"]["mean"], expected_future)
        self.assertTrue(all(c["current_dice"] == 1.0 for c in m["sparse_subsets"]["empty_cutoff"]))
        for name in ("appearance", "disappearance", "both_empty", "empty_target", "empty_cutoff"):
            counts = m["sparse_subsets"][name]
            self.assertTrue(all(c["rows"] > 0 for c in counts))
            self.assertTrue(all(c["patients"] > 0 for c in counts))
            self.assertTrue(all(c["future_dice"] is not None for c in counts))

    def test_production_npz_without_pickle_and_distinct_mask_diagnostics(self):
        arrays, identity = self.bundle()
        arrays["nuisance_JH_identity_42_mask_diagnostic"] = np.zeros((18, 3, 1), dtype=bool)
        arrays["primary_JH_42_soft_dice_diagnostic"] = np.full((75, 3), 0.123)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"private_predictions.npz"
            np.savez(path, **arrays)
            with np.load(path, allow_pickle=False) as loaded:
                result = score.analyze(loaded, identity)
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ["private_predictions.npz"])
        current = result["cohorts"]["primary"]["scores"]["JH"]["seeds"]["42"]["current_dice"]["mean"]
        self.assertNotEqual(current, 0.123)
        self.assertEqual(result["status"], "staged_only")

    def test_zero_neural_intrinsic_scale_is_valid_in_all_conditions(self):
        arrays, identity = self.bundle()
        for cohort in ("primary", "single"):
            for arm in score.ARMS:
                for seed in score.SEEDS:
                    arrays[f"{cohort}_{arm}_{seed}_intrinsic_scale"][:] = 0
        for arm in score.ARMS:
            for condition in score.CONDITIONS:
                for seed in score.SEEDS:
                    arrays[f"nuisance_{arm}_{condition}_{seed}_intrinsic_scale"][:] = 0
        result = score.analyze(arrays, identity)
        self.assertEqual(result["status"], "staged_only")
        width = result["cohorts"]["primary"]["scores"]["JH"]["seed_score_average"]["width_0.8"]["mean"]
        self.assertAlmostEqual(width, 2*ndtri(0.9)*0.05, places=13)

    def test_scores_are_seed_averaged_not_predictions_and_patient_not_rows(self):
        arrays, identity = self.bundle()
        truth = arrays["primary_truth"]
        seed_scores = []
        for seed, error in zip(score.SEEDS, (-1.0, 0.0, 1.0)):
            prefix = f"primary_OH_{seed}"
            errors = np.full((75, 3), error)
            errors[arrays["primary_patient_ids"] == "p00"] += 1.0
            arrays[prefix+"_mean"] = truth+errors
            arrays[prefix+"_intrinsic_scale"][:] = 0.5
            identity["calibration"][f"OH_{seed}"]["forecast"] = 1.0
            arrays[prefix+"_calibration"] = np.asarray(1.0)
            for nuisance_condition in score.CONDITIONS:
                arrays[f"nuisance_OH_{nuisance_condition}_{seed}_calibration"] = np.asarray(1.0)
                arrays[f"nuisance_OH_{nuisance_condition}_{seed}_mean"] = arrays[prefix+"_mean"][:18].copy()
            seed_scores.append(score.gaussian_crps(arrays[prefix+"_mean"], np.hypot(0.5, 0.05), truth))
        result = score.analyze(arrays, identity)
        got = result["cohorts"]["primary"]["scores"]["OH"]["seed_score_average"]["crps"]["mean"]
        by_row = np.mean(seed_scores, axis=0)
        expected = score.patient_values(by_row, arrays["primary_patient_ids"]).mean()
        self.assertAlmostEqual(got, expected, places=13)
        self.assertNotAlmostEqual(got, by_row.mean(), places=6)
        ensemble = np.mean([arrays[f"primary_OH_{s}_mean"] for s in score.SEEDS], axis=0)
        ensemble_score = score.patient_values(score.gaussian_crps(ensemble, np.hypot(0.5, 0.05), truth),
                                               arrays["primary_patient_ids"]).mean()
        self.assertNotAlmostEqual(got, ensemble_score, places=6)

    def test_measured_targets_cutoff_and_identity_swaps_raise(self):
        for key in ("primary_truth", "single_cutoff_measured", "nuisance_truth", "nuisance_cutoff_measured"):
            arrays, identity = self.bundle(); arrays[key][0, 0] += 0.01
            with self.subTest(key=key), self.assertRaises(ValueError):
                score.analyze(arrays, identity)
        arrays, identity = self.bundle()
        arrays["primary_row_ids"][[0, 1]] = arrays["primary_row_ids"][[1, 0]]
        with self.assertRaises(ValueError):
            score.analyze(arrays, identity)
        arrays, identity = self.bundle(); arrays["nuisance_patient_ids"][0] = "wrong"
        with self.assertRaises(ValueError):
            score.analyze(arrays, identity)

    def test_production_cardinality_not_weakened_by_identity(self):
        for cohort in score.COHORTS:
            arrays, identity = self.bundle()
            identity["cohort_rows"][cohort] = identity["cohort_rows"][cohort][:-1]
            with self.subTest(cohort=cohort), self.assertRaises(ValueError):
                score.analyze(arrays, identity)
        arrays, identity = self.bundle()
        identity["cohort_rows"]["single"][-1]["patient_id"] = "p00"
        identity["cohort_rows"]["single"][22]["patient_id"] = "p00"
        with self.assertRaises(ValueError):
            score.analyze(arrays, identity)

    def test_missing_or_mislabelled_seeds_arms_conditions_and_rows_raise(self):
        for old, new in (("primary_OH_44_mean", "primary_OH_45_mean"),
                         ("single_JL_42_future_dice", "single_XX_42_future_dice"),
                         ("nuisance_JH_resolution_43_mean", "nuisance_JH_other_43_mean")):
            arrays, identity = self.bundle(); arrays[new] = arrays.pop(old)
            with self.subTest(old=old), self.assertRaises(ValueError):
                score.analyze(arrays, identity)
        for key in ("primary_JL_44_current_dice", "single_JH_42_future_dice",
                    "nuisance_OH_intensity_43_current_mean", "primary_FP_intrinsic_scale"):
            arrays, identity = self.bundle(); del arrays[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                score.analyze(arrays, identity)
        for seeds in ([42, 43, 43], [44, 43, 42], [42, 43, 45]):
            arrays, identity = self.bundle(); arrays["seeds"] = np.asarray(seeds)
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                score.analyze(arrays, identity)

    def test_calibration_and_rms_swaps_rejected_in_all_namespaces(self):
        for key in ("primary_OH_42_calibration", "single_JL_44_persistence_calibration",
                    "nuisance_JH_resolution_43_calibration", "primary_P_calibration"):
            arrays, identity = self.bundle()
            old = float(arrays[key]); arrays[key] = np.asarray(2.0 if old != 2 else 0.5)
            with self.subTest(key=key), self.assertRaises(ValueError):
                score.analyze(arrays, identity)
        arrays, identity = self.bundle(); arrays["single_JH_43_persistence_rms"][1] += 0.01
        with self.assertRaises(ValueError):
            score.analyze(arrays, identity)
        arrays, identity = self.bundle(); arrays["primary_OH_42_calibration"] = np.asarray([0.5])
        with self.assertRaises(ValueError):
            score.analyze(arrays, identity)

    def test_e_portable_fit_centers_weights_noise_and_coherent_fp_are_checked(self):
        for key, index in (("primary_E_centers", (0, 0, 0)), ("single_E_weights", (0,)),
                           ("primary_P_intrinsic_scale", (0, 1)), ("single_SSM_mean", (0, 1)),
                           ("primary_SSM_intrinsic_scale", (0, 1))):
            arrays, identity = self.bundle(); arrays[key][index] += 0.01
            with self.subTest(key=key), self.assertRaises(ValueError):
                score.analyze(arrays, identity)
        arrays, identity = self.bundle(); arrays["primary_E_observation_sd"] = np.asarray(0.1)
        with self.assertRaises(ValueError):
            score.analyze(arrays, identity)
        arrays, identity = self.bundle()
        arrays["primary_E_centers"] += 0.25
        with self.assertRaises(ValueError):
            score.analyze(arrays, identity)

    def test_nonfinite_negative_scales_dice_and_pickle_arrays_raise(self):
        for key, value in (("primary_JH_42_mean", np.nan), ("single_FP_intrinsic_scale", -1),
                           ("single_OH_44_current_dice", 1.01), ("nuisance_OL_intensity_42_mean", np.inf)):
            arrays, identity = self.bundle(); arrays[key][0, 0] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                score.analyze(arrays, identity)
        arrays, identity = self.bundle(); arrays["primary_row_ids"] = arrays["primary_row_ids"].astype(object)
        with self.assertRaises(ValueError):
            score.analyze(arrays, identity)
        arrays, identity = self.bundle(); arrays["primary_JH_42_future_dice"][0, 0] = 0.5
        self.assertEqual(arrays["primary_truth"][0, 0], 0.0)
        with self.assertRaises(ValueError):
            score.analyze(arrays, identity)

    def test_nuisance_per_compartment_gate_and_seedwise_drift(self):
        arrays, identity = self.bundle()
        for seed_index, seed in enumerate(score.SEEDS):
            base = arrays[f"nuisance_JH_identity_{seed}_mean"]
            arrays[f"nuisance_JH_intensity_{seed}_mean"] = base.copy()
            arrays[f"nuisance_JH_intensity_{seed}_mean"][:, 0] += 0.20
            arrays[f"nuisance_JH_resolution_{seed}_mean"] = base.copy()
            arrays[f"nuisance_JH_resolution_{seed}_mean"][:, 1] += (-0.3, 0.0, 0.3)[seed_index]
        result = score.analyze(arrays, identity)
        intensity = result["nuisance"]["JH"]["changes"]["intensity"]["per_compartment"]
        self.assertFalse(intensity[0]["passes"])
        self.assertTrue(intensity[1]["passes"])
        self.assertTrue(intensity[2]["passes"])
        self.assertFalse(result["capability_flags"]["nuisance"])
        drift = result["nuisance"]["JH"]["changes"]["resolution"]["per_compartment"][1]
        self.assertAlmostEqual(drift["mean_absolute_predictive_drift"]["difference"], 0.2, places=13)
        self.assertEqual(drift["relative_denominator"], "identity_compartment_crps")

    def test_nuisance_identity_cannot_be_refitted_or_substituted(self):
        arrays, identity = self.bundle(); arrays["nuisance_JH_identity_42_mean"][0, 0] += 0.01
        with self.assertRaises(ValueError):
            score.analyze(arrays, identity)
        arrays, identity = self.bundle()
        identity["cohort_rows"]["nuisance"] = identity["cohort_rows"]["nuisance"][::-1]
        for key in ("row_ids", "patient_ids", "truth", "cutoff_measured"):
            arrays["nuisance_"+key] = arrays["nuisance_"+key][::-1].copy()
        with self.assertRaises(ValueError):
            score.analyze(arrays, identity)


if __name__ == "__main__":
    unittest.main()
