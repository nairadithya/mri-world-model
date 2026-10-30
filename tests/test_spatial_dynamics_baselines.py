import unittest
import numpy as np

from src.harness.analysis.spatial_dynamics_baselines import (
    patient_weights, patient_uniform_mean, fit_persistence, fit_empirical_change,
    fit_uncertainty_scale, central_interval, OBSERVATION_SD,
    fit_empirical_condition_receipts,
)


class SpatialDynamicsBaselineTests(unittest.TestCase):
    def test_patient_uniform_compartment_aggregation(self):
        ids = ["a", "a", "a", "b"]
        np.testing.assert_allclose(patient_weights(ids), [1/6, 1/6, 1/6, 1/2])
        scores = np.array([[0, 3, 6], [3, 6, 9], [6, 9, 12], [9, 12, 15]])
        # Compartment mean per example, then equal-patient mean.
        self.assertEqual(patient_uniform_mean(scores, ids), 9.0)

    def test_persistence_component_rms_and_unchanged_query_mean(self):
        decoded = np.zeros((4, 3))
        measured = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7], [4, 8, 12]])
        p = fit_persistence(decoded, measured, ["a", "a", "a", "b"])
        expected = np.sqrt(np.array([(1+9+25)/6+16/2, (4+16+36)/6+64/2, (9+25+49)/6+144/2]))
        np.testing.assert_allclose(p.residual_rms, expected)
        means, scale = p.distribution([[2, 3, 4]], 2)
        np.testing.assert_array_equal(means, [[2, 3, 4]])
        np.testing.assert_allclose(scale, np.sqrt((2*expected)**2 + OBSERVATION_SD**2))

    def test_joint_empirical_rows_weights_and_crps_direct_reference(self):
        ids = ["a", "a", "b", "c"]
        changes = np.array([[1, -1, 2], [3, -3, 4], [-2, 2, -1], [6, -6, 8]], float)
        fitted = fit_empirical_change(np.zeros_like(changes), changes, ids, condition="gaussian_observation")
        weights = np.array([1/6, 1/6, 1/3, 1/3])
        mean = weights @ changes
        # Shared scalar calibration preserves every empirical vector's correlation structure.
        out = fitted.distribution([[10, 20, 30]], 2)[0] - np.array([10, 20, 30])
        np.testing.assert_allclose(out, mean + 2*(changes-mean))
        np.testing.assert_allclose(out[:, 0] - mean[0], -(out[:, 1] - mean[1]))
        target = np.array([[11.5, 18.5, 32.]])
        got = fitted.crps([[10, 20, 30]], target, 1)[0]
        # Independent unsmoothed weighted empirical CRPS formula, componentwise.
        samples = changes + np.array([10, 20, 30])
        direct = np.array([np.sum(weights*np.abs(target[0,c]-samples[:,c])) - .5*np.sum(weights[:,None]*weights[None,:]*np.abs(samples[:,c,None]-samples[None,:,c])) for c in range(3)])
        # Account for the protocol's Gaussian observation smoothing using a direct normal expectation formula.
        from statistics import NormalDist
        def abs_normal(mu, sd):
            z = mu/sd
            return sd*np.sqrt(2/np.pi)*np.exp(-z*z/2)+mu*(2*np.vectorize(NormalDist().cdf)(z)-1)
        smooth = np.array([np.sum(weights*abs_normal(target[0,c]-samples[:,c], OBSERVATION_SD)) - .5*np.sum(weights[:,None]*weights[None,:]*abs_normal(samples[:,c,None]-samples[None,:,c], np.sqrt(2)*OBSERVATION_SD)) for c in range(3)])
        np.testing.assert_allclose(got, smooth, atol=1e-12)
        self.assertEqual(direct.shape, (3,))

    def test_empirical_single_component_floor_is_not_calibrated(self):
        fitted = fit_empirical_change([[0., 0., 0.]], [[1., 2., 3.]], ["a"], condition="gaussian_observation")
        expected = OBSERVATION_SD * (np.sqrt(2 / np.pi) - 1 / np.sqrt(np.pi))
        for calibration in (.5, 1., 2.):
            np.testing.assert_allclose(fitted.crps([[0., 0., 0.]], [[1., 2., 3.]], calibration),
                                       np.full((1, 3), expected), rtol=0, atol=1e-12)

    def test_discrete_crps_hand_calculation_and_condition_rejection(self):
        control = fit_empirical_change([[0., 0., 0.], [0., 0., 0.]],
                                       [[-1., 2., 0.], [3., -2., 4.]], ["a", "b"],
                                       condition="discrete")
        got = control.crps([[0., 0., 0.]], [[1., 0., 2.]])[0]
        samples = np.array([[-1., 2., 0.], [3., -2., 4.]])
        direct = np.array([np.mean(np.abs(samples[:, c] - [1., 0., 2.][c])) -
                           .5*np.mean(np.abs(samples[:, c, None] - samples[None, :, c]))
                           for c in range(3)])
        np.testing.assert_allclose(got, direct)
        self.assertEqual(control.observation_sd, 0)
        with self.assertRaises(ValueError):
            fit_empirical_change([[0, 0, 0]], [[1, 1, 1]], ["a"], condition="unknown")

    def test_noise_conditions_share_source_law_and_validation_calibration(self):
        decoded = np.zeros((3, 3))
        measured = np.array([[1., -1., 0.], [3., -3., 2.], [-2., 2., -1.]])
        ids = ["a", "a", "b"]
        discrete = fit_empirical_change(decoded, measured, ids, condition="discrete")
        smooth = fit_empirical_change(decoded, measured, ids, condition="gaussian_observation")
        np.testing.assert_array_equal(discrete.changes, smooth.changes)
        np.testing.assert_array_equal(discrete.distribution([[4., 5., 6.]], 2),
                                      smooth.distribution([[4., 5., 6.]], 2))
        receipts = fit_empirical_condition_receipts(decoded, measured, ids,
                                                    [[0., 0., 0.], [1., 2., 3.]],
                                                    [[1., 0., 2.], [2., 1., 4.]], ["v", "w"])
        self.assertEqual(set(receipts), {"discrete", "gaussian_observation"})
        for receipt in receipts.values():
            self.assertIn(receipt["calibration"], (.5, 1., 2.))
            self.assertTrue(np.isfinite(receipt["scores_by_scale"]).all())
            self.assertTrue((receipt["scores_by_scale"] >= 0).all())

    def test_nondegenerate_receipts_match_all_calibrated_references(self):
        import math
        from statistics import NormalDist
        source = np.array([[1., -1., 2.], [3., -3., 4.], [-2., 2., -1.]])
        weights = np.array([.25, .25, .5])
        center = weights @ source
        latest = np.array([[0., 1., 2.], [1., 2., 3.], [2., 3., 4.]])
        target = np.array([[1., 0., 2.], [2., 1., 4.], [0., 2., 3.]])
        receipts = fit_empirical_condition_receipts(np.zeros_like(source), source, ["a", "a", "b"],
                                                   latest, target, ["v", "v", "w"])
        def absolute_moment(value, sd):
            if sd == 0:
                return abs(value)
            z = value / sd
            return sd * math.sqrt(2 / math.pi) * math.exp(-z*z/2) + value * (2*NormalDist().cdf(z)-1)
        for condition, sd in (("discrete", 0.), ("gaussian_observation", .05)):
            expected = np.zeros((3, 3, 3))
            for ki, k in enumerate((.5, 1., 2.)):
                shifts = center + k * (source - center)
                for row in range(3):
                    for c in range(3):
                        samples = latest[row, c] + shifts[:, c]
                        first = sum(weights[i] * absolute_moment(target[row, c]-samples[i], sd) for i in range(3))
                        second = sum(weights[i]*weights[j] * absolute_moment(samples[i]-samples[j], math.sqrt(2)*sd)
                                     for i in range(3) for j in range(3))
                        expected[ki, row, c] = first - .5*second
            np.testing.assert_allclose(receipts[condition]["scores_by_scale"], expected, rtol=0, atol=1e-12)
            patient_means = expected.mean(axis=2) @ np.array([.25, .25, .5])
            self.assertEqual(receipts[condition]["calibration"], (.5, 1., 2.)[int(np.argmin(patient_means))])
            self.assertFalse(np.allclose(expected[0], expected[1]))

    def test_calibration_ties_and_validation_only(self):
        ids = ["a", "a", "b"]
        scores = np.ones((3, 3, 3))
        self.assertEqual(fit_uncertainty_scale(scores, ids), .5)
        scores = np.repeat(np.array([[0., 0., 3.], [1.2, 1.2, 1.2], [4., 4., 4.]])[:, :, None], 3, axis=2)
        self.assertEqual(fit_uncertainty_scale(scores, ids), 1.)

    def test_intervals_and_reject_malformed_inputs(self):
        lo, hi = central_interval([0], [1], .5)
        self.assertAlmostEqual(lo[0], -.67448975, places=6)
        self.assertAlmostEqual(hi[0], .67448975, places=6)
        for fn in (lambda: fit_persistence([[1, np.nan, 2]], [[2, 3, 4]], ["a"]),
                   lambda: fit_empirical_change([[1, 2]], [[2, 3]], ["a"], condition="discrete"),
                   lambda: patient_uniform_mean([[1, np.inf, 2]], ["a"]),
                   lambda: fit_empirical_change(np.empty((0, 3)), np.empty((0, 3)), [], condition="discrete"),
                   lambda: fit_empirical_change([[1, 2, 3]], [[2, 3, np.nan]], ["a"], condition="discrete"),
                   lambda: fit_empirical_change([[1, 2, 3]], [[2, 3, 4]], [""], condition="discrete")):
            with self.assertRaises(ValueError):
                fn()


if __name__ == "__main__":
    unittest.main()
