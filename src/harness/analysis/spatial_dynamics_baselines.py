"""Patient-uniform source-fitted anatomical controls for R37."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import NormalDist
from typing import Sequence

import numpy as np

OBSERVATION_SD = 0.05
CALIBRATION_GRID = (0.5, 1.0, 2.0)
N_COMPARTMENTS = 3


def _matrix(values, patient_ids, name):
    x = np.asarray(values, dtype=float)
    ids = np.asarray(patient_ids, dtype=object)
    if x.ndim != 2 or x.shape[1] != N_COMPARTMENTS or ids.ndim != 1 or x.shape[0] != len(ids) or not len(ids):
        raise ValueError(f"{name} values must be nonempty [N,3] with matching patient IDs")
    if not np.isfinite(x).all() or any(not isinstance(p, str) or not p for p in ids):
        raise ValueError(f"{name} values and patient IDs must be finite and present")
    return x, ids


def patient_weights(patient_ids: Sequence[str]) -> np.ndarray:
    ids = np.asarray(patient_ids, dtype=object)
    if ids.ndim != 1 or not len(ids) or any(not isinstance(p, str) or not p for p in ids):
        raise ValueError("patient IDs must be a nonempty vector of present strings")
    counts = defaultdict(int)
    for p in ids:
        counts[p] += 1
    return np.asarray([1.0 / (len(counts) * counts[p]) for p in ids])


def patient_uniform_mean(values, patient_ids) -> float:
    x, ids = _matrix(values, patient_ids, "scoring")
    return float(np.dot(patient_weights(ids), x.mean(axis=1)))


def patient_uniform_crps(scores, patient_ids) -> float:
    return patient_uniform_mean(scores, patient_ids)


def _abs_normal_expectation(mu, sd):
    if sd == 0:
        return np.abs(mu)
    z = mu / sd
    return sd * math.sqrt(2 / math.pi) * np.exp(-0.5 * z * z) + mu * (2 * np.vectorize(NormalDist().cdf)(z) - 1)


def _check_observation_sd(sd):
    if not math.isfinite(sd) or sd < 0:
        raise ValueError("observation SD must be finite and nonnegative")


@dataclass(frozen=True)
class PersistenceControl:
    residual_rms: np.ndarray
    observation_sd: float = OBSERVATION_SD

    def distribution(self, latest_decoded_mean, calibration=1.0):
        _check_calibration(calibration)
        mean = np.asarray(latest_decoded_mean, float)
        if mean.ndim != 2 or mean.shape[1] != N_COMPARTMENTS or not np.isfinite(mean).all():
            raise ValueError("query means must be finite [N,3]")
        _check_observation_sd(self.observation_sd)
        return mean, np.sqrt((self.residual_rms * calibration) ** 2 + self.observation_sd ** 2)


def fit_persistence(source_decoded, source_measured, source_patient_ids) -> PersistenceControl:
    decoded, ids = _matrix(source_decoded, source_patient_ids, "source decoded")
    measured, ids2 = _matrix(source_measured, source_patient_ids, "source measured")
    if not np.array_equal(ids, ids2) or decoded.shape != measured.shape:
        raise ValueError("source decoded and measured rows must match")
    weights = patient_weights(ids)
    rms = np.sqrt(np.sum(weights[:, None] * (measured - decoded) ** 2, axis=0))
    return PersistenceControl(rms)


def _weighted_joint_crps(samples, sample_weights, target, observation_sd):
    """Vector CRPS; one source row remains a joint three-compartment sample."""
    result = np.zeros(N_COMPARTMENTS)
    for c in range(N_COMPARTMENTS):
        x = samples[:, c]
        first = np.sum(sample_weights * _abs_normal_expectation(target[c] - x, observation_sd))
        pair = _abs_normal_expectation(x[:, None] - x[None, :], math.sqrt(2) * observation_sd)
        result[c] = first - 0.5 * np.sum(sample_weights[:, None] * sample_weights[None, :] * pair)
    return result


@dataclass(frozen=True)
class ChangeControl:
    changes: np.ndarray
    patient_ids: tuple[str, ...]
    condition: str

    @property
    def observation_sd(self):
        return {"discrete": 0.0, "gaussian_observation": OBSERVATION_SD}[self.condition]

    def distribution(self, latest_decoded_mean, calibration=1.0):
        _check_calibration(calibration)
        mean = np.asarray(latest_decoded_mean, float)
        if mean.ndim != 2 or mean.shape[1] != N_COMPARTMENTS or not np.isfinite(mean).all():
            raise ValueError("query means must be finite [N,3]")
        _check_observation_sd(self.observation_sd)
        weights = patient_weights(self.patient_ids)
        weights /= weights.sum()
        empirical_mean = weights @ self.changes
        # Transform each joint row with one scalar; never resample coordinates separately.
        shifts = empirical_mean + calibration * (self.changes - empirical_mean)
        return mean[:, None, :] + shifts[None, :, :]

    def crps(self, latest_decoded_mean, target, calibration=1.0):
        means = np.asarray(latest_decoded_mean, float)
        y = np.asarray(target, float)
        if means.ndim != 2 or means.shape[1] != N_COMPARTMENTS or y.shape != means.shape or not np.isfinite(means).all() or not np.isfinite(y).all():
            raise ValueError("targets and query means must be finite matching [N,3] arrays")
        # Avoid constructing [Nquery,Nsource,3] when only scores are needed.
        shifts = self.distribution(np.zeros((1, N_COMPARTMENTS)), calibration)[0]
        weights = patient_weights(self.patient_ids); weights /= weights.sum()
        return np.stack([_weighted_joint_crps(means[i] + shifts, weights, y[i], self.observation_sd) for i in range(len(y))])


def fit_empirical_change(source_decoded, target_measured, patient_ids, *, condition: str) -> ChangeControl:
    if condition not in ("discrete", "gaussian_observation"):
        raise ValueError("condition must be 'discrete' or 'gaussian_observation'")
    decoded, ids = _matrix(source_decoded, patient_ids, "source decoded")
    measured, ids2 = _matrix(target_measured, patient_ids, "source measured")
    if not np.array_equal(ids, ids2) or decoded.shape != measured.shape:
        raise ValueError("source decoded and measured rows must match")
    return ChangeControl(measured - decoded, tuple(ids), condition)


def fit_empirical_condition_receipts(source_decoded, source_measured, source_patient_ids,
                                     validation_latest, validation_target, validation_patient_ids):
    """Fit both fixed noise conditions and calibrate each on validation only."""
    receipts = {}
    for condition in ("discrete", "gaussian_observation"):
        control = fit_empirical_change(source_decoded, source_measured, source_patient_ids,
                                       condition=condition)
        scores = np.stack([control.crps(validation_latest, validation_target, scale)
                           for scale in CALIBRATION_GRID])
        calibration = fit_uncertainty_scale(scores, validation_patient_ids)
        receipts[condition] = {"control": control, "scores_by_scale": scores,
                               "calibration": calibration}
    return receipts


def fit_uncertainty_scale(validation_crps_by_scale, validation_patient_ids):
    """Select only from supplied validation [scale, example, compartment] CRPS."""
    scores = np.asarray(validation_crps_by_scale, float)
    ids = np.asarray(validation_patient_ids, object)
    if scores.shape != (len(CALIBRATION_GRID), len(ids), N_COMPARTMENTS) or not np.isfinite(scores).all():
        raise ValueError("validation scores must be finite [3, examples, 3]")
    means = [patient_uniform_crps(row, ids) for row in scores]
    return CALIBRATION_GRID[min(range(3), key=lambda i: (means[i], i))]


def central_interval(mean, scale, level):
    if not 0 < level < 1:
        raise ValueError("level must be between zero and one")
    m, s = np.broadcast_arrays(np.asarray(mean, float), np.asarray(scale, float))
    if not np.isfinite(m).all() or not np.isfinite(s).all() or (s <= 0).any():
        raise ValueError("interval parameters must be finite with positive scales")
    z = NormalDist().inv_cdf((1 + level) / 2)
    return m - z * s, m + z * s


def _check_calibration(value):
    if value not in CALIBRATION_GRID:
        raise ValueError("calibration must be selected from [0.5, 1, 2]")
