import unittest

import torch

from src.model.spatial_dynamics import SpatialDynamics, FrozenAffineVolumeDecoder, gaussian_crps


class SpatialDynamicsTests(unittest.TestCase):
    def assert_prediction_equal(self, left, right):
        for left_field, right_field in zip(left, right):
            torch.testing.assert_close(left_field, right_field, rtol=0, atol=0)

    def assert_prediction_close(self, left, right):
        for left_field, right_field in zip(left, right):
            torch.testing.assert_close(left_field, right_field, rtol=1e-6, atol=1e-7)

    def inputs(self):
        torch.manual_seed(4)
        maps = torch.randn(2, 3, 128, 2, 2, 2)
        avail = torch.ones(2, 3, 4)
        gaps = torch.tensor([[0., .5, .8], [0., .3, .4]])
        mask = torch.ones(2, 3)
        return maps, avail, gaps, mask

    def test_crps_zero_residual(self):
        got = gaussian_crps(torch.tensor([0.]), torch.tensor([2.]), torch.tensor([0.]))
        self.assertAlmostEqual(got.item(), 2 * (2 / (2 * 3.141592653589793) ** .5 - 1 / 3.141592653589793 ** .5), places=6)

    def test_forward_backward_finite_and_decoder_frozen_with_gradient(self):
        model = SpatialDynamics()
        args = self.inputs()
        pred = model(*args)
        weight = torch.ones(3, 128, requires_grad=True)
        bias = torch.zeros(3, requires_grad=True)
        decoder = FrozenAffineVolumeDecoder(weight, bias,
                 torch.zeros(128), torch.ones(128), torch.zeros(3), torch.ones(3))
        mean, scale = decoder(pred.feature_mean, pred.feature_scale)
        loss = gaussian_crps(mean, scale, torch.zeros_like(mean)).mean() + .1 * pred.feature_mean.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNone(weight.grad)
        self.assertIsNone(bias.grad)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_decoder_analytic_shared_channel_noise_and_floor(self):
        w = torch.zeros(3, 128); w[:, 0] = 2
        d = FrozenAffineVolumeDecoder(w, torch.tensor([3., 4., 5.]), torch.full((128,), 7.), torch.full((128,), 2.),
                                     torch.tensor([11., 13., 17.]), torch.tensor([3., 4., 5.]))
        mu = torch.zeros(1, 128, 2, 2, 2); mu[:, 0] = 3
        sigma = torch.ones(1, 128); sigma[:, 0] = 4
        mean, scale = d(mu, sigma, calibration=2)
        expected = torch.tensor([[((3 * 2 + 7) * 2 + 3) * 3 + 11,
                                  ((3 * 2 + 7) * 2 + 4) * 4 + 13,
                                  ((3 * 2 + 7) * 2 + 5) * 5 + 17]])
        self.assertTrue(torch.equal(mean, expected))
        # One channel is shared over all voxels: variance is not divided by voxel count.
        self.assertTrue(torch.allclose(scale, torch.tensor([[9216.0025 ** .5, 16384.0025 ** .5, 25600.0025 ** .5]]), atol=1e-5))

    def test_decoder_observation_floor_for_each_calibration(self):
        decoder = FrozenAffineVolumeDecoder(torch.zeros(2, 128), torch.zeros(2),
            torch.zeros(128), torch.ones(128), torch.zeros(2), torch.ones(2))
        mu, sigma = torch.zeros(1, 128, 2, 2, 2), torch.ones(1, 128)
        for calibration in (0.5, 1.0, 2.0):
            _, scale = decoder(mu, sigma, calibration)
            torch.testing.assert_close(scale, torch.full((1, 2), 0.05), rtol=0, atol=1e-7)

    def test_decoder_adds_opposite_channel_variances(self):
        weight = torch.zeros(1, 128); weight[0, :2] = torch.tensor([2., -3.])
        decoder = FrozenAffineVolumeDecoder(weight, torch.zeros(1), torch.zeros(128),
            torch.ones(128), torch.zeros(1), torch.ones(1))
        mu = torch.zeros(1, 128, 1, 1, 1)
        sigma = torch.ones(1, 128)
        _, scale = decoder(mu, sigma, calibration=2.0)
        expected = torch.sqrt(torch.tensor(2.0 ** 2 * 2.0 ** 2 + 3.0 ** 2 * 2.0 ** 2 + 0.05 ** 2))
        torch.testing.assert_close(scale, expected.reshape(1, 1), rtol=0, atol=1e-6)

    def test_decoder_mean_and_scale_gradients_are_independent(self):
        decoder = FrozenAffineVolumeDecoder(torch.ones(1, 2), torch.zeros(1),
            torch.zeros(2), torch.ones(2), torch.zeros(1), torch.ones(1))
        mu = torch.ones(1, 2, 1, 1, 1, requires_grad=True)
        sigma = torch.ones(1, 2, requires_grad=True)
        mean, _ = decoder(mu, sigma)
        mean.sum().backward()
        self.assertIsNotNone(mu.grad); self.assertGreater(mu.grad.abs().sum().item(), 0)
        self.assertIsNone(sigma.grad)
        mu2 = torch.ones(1, 2, 1, 1, 1, requires_grad=True)
        sigma2 = torch.ones(1, 2, requires_grad=True)
        _, scale = decoder(mu2, sigma2)
        scale.sum().backward()
        self.assertIsNone(mu2.grad)
        self.assertIsNotNone(sigma2.grad); self.assertGreater(sigma2.grad.abs().sum().item(), 0)

    def test_interventions_and_current_spatial_path(self):
        model = SpatialDynamics().eval(); maps, a, g, m = self.inputs()
        h = model(maps, a, g, m, mode="H")
        l = model(maps, a, g, m, mode="L")
        zeroed = maps.clone(); zeroed[:, :-1] = 0
        self.assert_prediction_equal(l, model(zeroed, a, g, m, mode="H"))
        pooled = maps.clone(); pooled[:, :-1] = maps[:, :-1].mean((-3,-2,-1), keepdim=True)
        gg = model(maps, a, g, m, mode="G")
        self.assert_prediction_close(gg, model(pooled, a, g, m, mode="H"))
        one = (maps[:, -1:], a[:, -1:], torch.zeros_like(g[:, -1:]), torch.ones_like(m[:, -1:]))
        single = [model(*one, mode=mode) for mode in ("H", "L", "G")]
        self.assert_prediction_equal(single[0], single[1])
        self.assert_prediction_equal(single[0], single[2])
        self.assertFalse(torch.allclose(l.feature_mean, h.feature_mean))
        changed_past = maps.clone(); changed_past[:, 0] += 9
        self.assertTrue(torch.equal(model(changed_past,a,g,m,mode="L").feature_mean, l.feature_mean))
        rearranged = maps.clone()
        rearranged[:, 0] = rearranged[:, 0].flip(-1)
        self.assertTrue(torch.allclose(model(rearranged,a,g,m,mode="G").feature_mean, gg.feature_mean))
        self.assertEqual(h.feature_mean.shape, (2,128,2,2,2))
        current_changed = maps.clone(); current_changed[:, -1, :, 0, 0, 0] += 3
        self.assertFalse(torch.allclose(model(current_changed,a,g,m,mode="G").feature_mean, gg.feature_mean))
        self.assertFalse(torch.equal(model(current_changed,a,g,m,mode="L").feature_mean,
                                     model(maps,a,g,m,mode="L").feature_mean))
        self.assertTrue(torch.isfinite(gg.feature_scale).all())

    def test_mask_padding_invariance_and_no_mutation(self):
        model = SpatialDynamics().eval(); maps, a, g, mask = self.inputs()
        original = maps.clone()
        one_gap = torch.zeros_like(g[:, -1:])
        one = model(maps[:, -1:], a[:, -1:], one_gap, torch.ones(2,1))
        padded_maps = torch.cat((torch.full_like(maps[:, :2], 99), maps[:, -1:]), 1)
        padded_a = torch.cat((torch.zeros_like(a[:, :2]), a[:, -1:]), 1)
        padded_g = torch.zeros_like(g)
        padded_mask = torch.tensor([[0.,0.,1.]]).expand(2,-1)
        padded = model(padded_maps,padded_a,padded_g,padded_mask)
        self.assertTrue(torch.allclose(one.feature_mean,padded.feature_mean))
        self.assertTrue(torch.equal(maps, original))

    def test_reject_invalid_gap_sequence(self):
        model = SpatialDynamics()
        maps, a, g, m = self.inputs()
        with self.assertRaises(ValueError):
            model(maps, a, torch.tensor([[.1, .5, .8], [0., .3, .4]]), m)
        with self.assertRaises(ValueError):
            model(maps, a, torch.tensor([[0., 0., .8], [0., .3, .4]]), m)

    def test_reject_target_keyword(self):
        with self.assertRaises(TypeError):
            SpatialDynamics()( *self.inputs(), target=torch.zeros(1))


if __name__ == "__main__":
    unittest.main()
