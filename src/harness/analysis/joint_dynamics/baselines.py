from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

import numpy as np
from scipy.special import ndtr

GRID = (0.5, 1.0, 2.0)
OBSERVATION_SD = 0.05


@dataclass(frozen=True)
class FilterHistory:
    estimates: np.ndarray
    gaps_weeks: np.ndarray


@dataclass(frozen=True)
class SourceChain:
    """Already split at unusable visits/nonpositive intervals; never bridges a break."""
    patient_id: str
    visit_ids: tuple[str, ...]
    estimates: np.ndarray
    gaps_weeks: np.ndarray


@dataclass(frozen=True)
class ControlInputs:
    estimates: np.ndarray
    histories: tuple[FilterHistory, ...]


@dataclass(frozen=True)
class ControlFitData:
    inputs: ControlInputs
    truth: np.ndarray
    patient_ids: np.ndarray
    row_ids: np.ndarray
    chains: tuple[SourceChain, ...]
    outgoing_weeks: np.ndarray


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.shape != shape:
        raise ValueError(f"{name} must be numeric {shape}")
    x = raw.astype(np.float64)
    if not np.isfinite(x).all():
        raise ValueError(f"nonfinite {name}")
    return x


def _ids(value: Any, n: int, name: str, unique: bool = False) -> np.ndarray:
    x = np.asarray(value)
    if x.shape != (n,) or not n or x.dtype.kind not in "US" or any(not str(v) for v in x):
        raise ValueError(f"invalid {name}")
    x = x.astype(str)
    if unique and len(set(x)) != n:
        raise ValueError(f"duplicate {name}")
    return x


def patient_weights(patient_ids: Any) -> np.ndarray:
    ids = _ids(patient_ids, len(patient_ids), "patient IDs")
    counts = Counter(ids)
    return np.asarray([1.0 / (len(counts) * counts[p]) for p in ids])


def gaussian_crps(mean: Any, sd: Any, truth: Any) -> np.ndarray:
    m, s, y = np.broadcast_arrays(np.asarray(mean, float), np.asarray(sd, float),
                                  np.asarray(truth, float))
    if not all(np.isfinite(v).all() for v in (m, s, y)) or (s <= 0).any():
        raise ValueError("invalid Gaussian parameters")
    z = (y - m) / s
    result = s * (z * (2 * ndtr(z) - 1) + 2 * np.exp(-z*z/2) / np.sqrt(2*np.pi)
                  - 1 / np.sqrt(np.pi))
    if not np.isfinite(result).all():
        raise ValueError("nonfinite Gaussian score")
    return result


def _abs_normal(delta: np.ndarray, sd: float) -> np.ndarray:
    z = delta / sd
    return sd * np.sqrt(2/np.pi) * np.exp(-z*z/2) + delta * (2*ndtr(z)-1)


def mixture_crps(centers: Any, weights: Any, truth: Any,
                 observation_sd: float = OBSERVATION_SD) -> np.ndarray:
    c = np.asarray(centers, float)
    y = np.asarray(truth, float)
    if c.ndim != 3 or not c.shape[1] or y.shape != (len(c), 3) or c.shape[2] != 3:
        raise ValueError("invalid mixture shapes")
    w = _array(weights, (c.shape[1],), "mixture weights")
    if (w < 0).any() or not np.isclose(w.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("mixture weights must sum to one")
    if not np.isfinite(observation_sd) or observation_sd <= 0:
        raise ValueError("invalid observation noise")
    if not np.isfinite(c).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite mixture")
    first = np.sum(w[None, :, None] * _abs_normal(c-y[:, None, :], observation_sd), axis=1)
    pair_weights = w[:, None, None] * w[None, :, None]
    second = np.stack([np.sum(pair_weights * _abs_normal(row[:, None, :]-row[None, :, :],
                                                       np.sqrt(2)*observation_sd), axis=(0, 1))
                       for row in c])
    result = first - 0.5*second
    if not np.isfinite(result).all():
        raise ValueError("nonfinite mixture score")
    return result


def fit_persistence(estimates: Any, truth: Any, patient_ids: Any) -> np.ndarray:
    n = len(patient_ids)
    x = _array(estimates, (n, 3), "persistence estimates")
    y = _array(truth, (n, 3), "persistence truth")
    return np.sqrt(np.sum(patient_weights(patient_ids)[:, None] * (y-x)**2, axis=0))


def gaussian_calibration_receipt(mean: Any, intrinsic_scale: Any, truth: Any,
                                 patient_ids: Any) -> dict[str, Any]:
    """Raw validation parameters and all floor-inclusive grid scores, tie by grid order."""
    n = len(patient_ids)
    m = _array(mean, (n, 3), "calibration mean")
    s = _array(intrinsic_scale, (n, 3), "calibration intrinsic scale")
    y = _array(truth, (n, 3), "calibration truth")
    if (s < 0).any():
        raise ValueError("negative intrinsic scale")
    w = patient_weights(patient_ids)
    scores = [float(np.sum(w[:, None]*gaussian_crps(m, np.hypot(k*s, OBSERVATION_SD), y))/3)
              for k in GRID]
    return _plain({"mean": m, "intrinsic_scale": s, "truth": y,
                   "patient_ids": list(map(str, patient_ids)), "grid": GRID,
                   "grid_scores": scores, "selected": GRID[int(np.argmin(scores))],
                   "observation_sd": OBSERVATION_SD})


def calibrate_gaussian(mean: Any, intrinsic_scale: Any, truth: Any, patient_ids: Any) -> float:
    return gaussian_calibration_receipt(mean, intrinsic_scale, truth, patient_ids)["selected"]


def _history(history: FilterHistory, query: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(history, FilterHistory):
        raise ValueError("expected label-free FilterHistory")
    n = len(history.estimates)
    y = _array(history.estimates, (n, 3), "history estimates")
    dt = _array(history.gaps_weeks, (n,), "history gaps")
    if not n or (query and n > 4) or dt[0] != 0 or (dt[1:] <= 0).any():
        raise ValueError("history must begin at zero and have positive subsequent gaps")
    return y, dt


def _inputs(inputs: ControlInputs) -> np.ndarray:
    if not isinstance(inputs, ControlInputs):
        raise ValueError("prediction requires label-free ControlInputs")
    x = _array(inputs.estimates, (len(inputs.estimates), 3), "input estimates")
    if not len(x) or len(inputs.histories) != len(x):
        raise ValueError("history count mismatch")
    for i, h in enumerate(inputs.histories):
        y, _ = _history(h)
        if not np.array_equal(y[-1], x[i]):
            raise ValueError("history cutoff differs from canonical estimate")
    return x


def _data(data: ControlFitData, validation: bool = False) -> dict[str, Any]:
    if not isinstance(data, ControlFitData):
        raise ValueError("expected ControlFitData")
    x = _inputs(data.inputs)
    n = len(x)
    y = _array(data.truth, (n, 3), "control truth")
    ids = _ids(data.patient_ids, n, "patient IDs")
    rows = _ids(data.row_ids, n, "row IDs", unique=True)
    if validation and (len(data.chains) or len(data.outgoing_weeks)):
        raise ValueError("validation cannot contain source chains or outgoing intervals")
    outgoing = _array(data.outgoing_weeks, (0 if validation else n,), "source outgoing gaps")
    return _plain({"estimates": x, "truth": y, "patient_ids": ids, "row_ids": rows,
                   "histories": [{"estimates": h.estimates, "gaps_weeks": h.gaps_weeks}
                                 for h in data.inputs.histories], "outgoing_weeks": outgoing})


def _chain_records(chains: tuple[SourceChain, ...], source_patients: set[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chain in chains:
        if not isinstance(chain, SourceChain) or chain.patient_id not in source_patients:
            raise ValueError("chain outside source membership")
        y, dt = _history(FilterHistory(chain.estimates, chain.gaps_weeks), query=False)
        ids = _ids(chain.visit_ids, len(y), "chain visit IDs", unique=True)
        for visit in ids:
            if str(visit) in seen:
                raise ValueError("source chains must deduplicate visits and edges")
            seen.add(str(visit))
        records.append(_plain({"patient_id": chain.patient_id, "visit_ids": ids,
                               "estimates": y, "gaps_weeks": dt,
                               "edges": [[str(a), str(b)] for a, b in zip(ids[:-1], ids[1:])],
                               "included": len(y) >= 2}))
    if not any(r["included"] for r in records):
        raise ValueError("no positive source intervals for EM")
    return records


def _smooth(y: np.ndarray, dt: np.ndarray, b: float, q: float,
            r: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(y)
    mf = np.empty(n); vf = np.empty(n)
    mp = np.empty(n); vp = np.empty(n)
    mf[0], vf[0] = y[0], r
    mp[0], vp[0] = y[0], r
    for i in range(1, n):
        mp[i] = mf[i-1] + b*dt[i]
        vp[i] = vf[i-1] + q*dt[i]
        gain = vp[i] / (vp[i]+r)
        mf[i] = mp[i] + gain*(y[i]-mp[i])
        vf[i] = (1-gain)*vp[i]
    ms, vs = mf.copy(), vf.copy()
    lag = np.empty(n-1)
    for i in range(n-2, -1, -1):
        gain = vf[i]/vp[i+1]
        ms[i] = mf[i] + gain*(ms[i+1]-mp[i+1])
        vs[i] = vf[i] + gain*gain*(vs[i+1]-vp[i+1])
        lag[i] = gain*vs[i+1]
    return ms, vs, lag


def _fit_filter(records: list[dict[str, Any]]) -> dict[str, Any]:
    included = [ch for ch in records if ch["included"]]
    intervals: Counter[str] = Counter()
    nodes: Counter[str] = Counter()
    for ch in included:
        intervals[ch["patient_id"]] += len(ch["visit_ids"])-1
        nodes[ch["patient_id"]] += len(ch["visit_ids"])
    weights = {p: 1/(len(intervals)*count) for p, count in sorted(intervals.items())}
    edge_den = sum(weights[p]*intervals[p] for p in weights)
    node_den = sum(weights[p]*nodes[p] for p in weights)
    parameters = []
    for c in range(3):
        seq = [(np.asarray(ch["estimates"])[:, c], np.asarray(ch["gaps_weeks"]),
                weights[ch["patient_id"]]) for ch in included]
        time_den = sum(w*dt[1:].sum() for _, dt, w in seq)
        b = sum(w*np.diff(y).sum() for y, _, w in seq)/time_den
        energy = sum(w*np.sum((np.diff(y)-b*dt[1:])**2) for y, dt, w in seq)
        r = max(1e-3, 0.5*energy/edge_den)
        q = max(1e-3, sum(w*np.sum((np.diff(y)-b*dt[1:])**2/dt[1:])
                          for y, dt, w in seq)/edge_den)
        initial = [float(b), float(q), float(r)]
        updates = []
        for _ in range(30):
            moments = []
            drift_sum = 0.0
            observation_sum = 0.0
            for y, dt, w in seq:
                ms, vs, lag = _smooth(y, dt, b, q, r)
                d = np.diff(ms)
                v = d*d + vs[1:]+vs[:-1]-2*lag
                moments.append((d, v, dt[1:], w))
                drift_sum += w*d.sum()
                observation_sum += w*np.sum((y-ms)**2+vs)
            b = drift_sum/time_den
            q = max(1e-6, sum(w*np.sum((v-2*b*dt*d+b*b*dt*dt)/dt)
                              for d, v, dt, w in moments)/edge_den)
            r = max(1e-6, observation_sum/node_den)
            updates.append([float(b), float(q), float(r)])
        parameters.append({"initial": initial, "updates": updates, "final": updates[-1]})
    return {"parameters": parameters, "patient_interval_counts": dict(sorted(intervals.items())),
            "patient_node_counts": dict(sorted(nodes.items())), "patient_weights": weights,
            "edge_denominator": edge_den, "node_denominator": node_den,
            "singleton_exclusions": sum(not ch["included"] for ch in records),
            "chain_membership": records, "chain_membership_sha256": _digest(records),
            "iterations": 30}


def _posterior(history: FilterHistory, parameters: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    y, dt = _history(history)
    mean, variance = y[0].copy(), np.asarray([p["final"][2] for p in parameters], dtype=np.float64)
    b = np.asarray([p["final"][0] for p in parameters], dtype=np.float64)
    q = np.asarray([p["final"][1] for p in parameters], dtype=np.float64)
    r = variance.copy()
    for i in range(1, len(y)):
        mean += b*dt[i]
        variance += q*dt[i]
        gain = variance/(variance+r)
        mean += gain*(y[i]-mean)
        variance *= 1-gain
    return mean, variance


def _raw_predictions(state: dict[str, Any], inputs: ControlInputs) -> dict[str, dict[str, Any]]:
    x = _inputs(inputs)
    rms = np.asarray(state["p_rms"])
    parameters = state["filter"]["parameters"]
    posterior = [_posterior(h, parameters) for h in inputs.histories]
    pm = np.stack([p[0] for p in posterior])
    pv = np.stack([p[1] for p in posterior])
    b = np.asarray([p["final"][0] for p in parameters])
    q = np.asarray([p["final"][1] for p in parameters])
    horizon = state["typical_outgoing_weeks"]
    return {"P": {"kind": "gaussian", "mean": x.copy(),
                  "intrinsic_scale": np.broadcast_to(rms, x.shape).copy()},
            "FP": {"kind": "gaussian", "mean": pm, "intrinsic_scale": np.sqrt(pv)},
            "SSM": {"kind": "gaussian", "mean": pm+b*horizon,
                    "intrinsic_scale": np.sqrt(pv+q*horizon)}}


def _e_receipt(state: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    changes = np.asarray(state["e_changes"])
    weights = np.asarray(state["e_weights"])
    mean = np.asarray(state["e_mean"])
    x = np.asarray(validation["estimates"])
    y = np.asarray(validation["truth"])
    w = patient_weights(validation["patient_ids"])
    scores = []
    for k in GRID:
        centers = x[:, None, :] + mean + k*(changes-mean)
        scores.append(float(np.sum(w[:, None]*mixture_crps(centers, weights, y))/3))
    return {"grid": list(GRID), "grid_scores": scores,
            "selected": GRID[int(np.argmin(scores))], "observation_sd": OBSERVATION_SD}


@dataclass(frozen=True)
class ControlFits:
    state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        body = _plain(self.state)
        return {"schema": "r38-control-fits-v3", "state": body, "state_sha256": _digest(body)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ControlFits:
        """Validate portable receipts without importing any historical runtime or pickle class."""
        if not isinstance(value, dict) or value.get("schema") != "r38-control-fits-v3":
            raise ValueError("unsupported control receipt")
        try:
            state = json.loads(json.dumps(value["state"], allow_nan=False))
            if _digest(state) != value["state_sha256"]:
                raise ValueError("control fit digest mismatch")
            for name in ("source", "validation"):
                raw = state[name]
                if _digest(raw) != state[name+"_sha256"]:
                    raise ValueError(f"{name} membership hash mismatch")
                data = _restore_data(raw, state["filter"]["chain_membership"] if name == "source" else [])
                if _data(data, name == "validation") != raw:
                    raise ValueError("invalid portable input receipt")
            source, val = state["source"], state["validation"]
            if set(source["patient_ids"]) & set(val["patient_ids"]):
                raise ValueError("source/validation overlap")
            rms = fit_persistence(source["estimates"], source["truth"], source["patient_ids"])
            changes = np.asarray(source["truth"])-np.asarray(source["estimates"])
            weights = patient_weights(source["patient_ids"])
            for actual, expected in ((state["p_rms"], rms), (state["e_changes"], changes),
                                     (state["e_weights"], weights), (state["e_mean"], weights@changes)):
                if not np.array_equal(np.asarray(actual), expected):
                    raise ValueError("source-derived control parameter mismatch")
            records = _chain_records(_restore_data(source, state["filter"]["chain_membership"]).chains,
                                     set(source["patient_ids"]))
            filt = state["filter"]
            if records != filt["chain_membership"] or _digest(records) != filt["chain_membership_sha256"]:
                raise ValueError("chain membership mismatch")
            if filt["iterations"] != 30 or len(filt["parameters"]) != 3:
                raise ValueError("incomplete EM receipt")
            intervals: Counter[str] = Counter(); nodes: Counter[str] = Counter()
            for ch in records:
                if ch["included"]:
                    intervals[ch["patient_id"]] += len(ch["visit_ids"])-1
                    nodes[ch["patient_id"]] += len(ch["visit_ids"])
            pw = {p: 1/(len(intervals)*n) for p, n in sorted(intervals.items())}
            if (filt["patient_interval_counts"] != dict(intervals)
                    or filt["patient_node_counts"] != dict(nodes) or filt["patient_weights"] != pw
                    or filt["edge_denominator"] != sum(pw[p]*intervals[p] for p in pw)
                    or filt["node_denominator"] != sum(pw[p]*nodes[p] for p in pw)
                    or filt["singleton_exclusions"] != sum(not r["included"] for r in records)):
                raise ValueError("EM weight/denominator receipt mismatch")
            for p in filt["parameters"]:
                initial = _array(p["initial"], (3,), "EM initialization")
                updates = _array(p["updates"], (30, 3), "EM updates")
                if ((initial[1:] < 1e-3).any() or (updates[:, 1:] < 1e-6).any()
                        or p["final"] != p["updates"][-1]):
                    raise ValueError("invalid EM parameters")
            outgoing = np.asarray(source["outgoing_weeks"])
            positive = outgoing[outgoing > 0]
            if (not len(positive) or state["typical_outgoing_weeks"] != float(np.median(positive))
                    or state["outgoing_nonpositive_exclusions"] != int(np.sum(outgoing <= 0))):
                raise ValueError("horizon receipt mismatch")
            raw_predictions = _raw_predictions(state, _restore_data(val, []).inputs)
            for name, pred in raw_predictions.items():
                expected = gaussian_calibration_receipt(pred["mean"], pred["intrinsic_scale"],
                                                        val["truth"], val["patient_ids"])
                if expected != state["calibration_receipts"][name]:
                    raise ValueError("Gaussian calibration receipt mismatch")
            if _e_receipt(state, val) != state["calibration_receipts"]["E"]:
                raise ValueError("E calibration receipt mismatch")
            if state["calibration"] != {k: v["selected"] for k, v in state["calibration_receipts"].items()}:
                raise ValueError("selected calibration mismatch")
        except (KeyError, TypeError, IndexError, OverflowError) as exc:
            raise ValueError("incomplete control receipt") from exc
        return cls(state)


def _restore_data(raw: dict[str, Any], chains: list[dict[str, Any]]) -> ControlFitData:
    inputs = ControlInputs(np.asarray(raw["estimates"]), tuple(
        FilterHistory(np.asarray(h["estimates"]), np.asarray(h["gaps_weeks"])) for h in raw["histories"]))
    return ControlFitData(inputs, np.asarray(raw["truth"]), np.asarray(raw["patient_ids"]),
                          np.asarray(raw["row_ids"]), tuple(SourceChain(
                              ch["patient_id"], tuple(ch["visit_ids"]), np.asarray(ch["estimates"]),
                              np.asarray(ch["gaps_weeks"])) for ch in chains), np.asarray(raw["outgoing_weeks"]))


def fit_controls(source: ControlFitData, validation: ControlFitData) -> ControlFits:
    sr = _data(source); vr = _data(validation, validation=True)
    if set(sr["patient_ids"]) & set(vr["patient_ids"]):
        raise ValueError("source/validation patients overlap")
    records = _chain_records(source.chains, set(sr["patient_ids"]))
    outgoing = np.asarray(sr["outgoing_weeks"])
    positive = outgoing[outgoing > 0]
    if not len(positive):
        raise ValueError("no positive source outgoing horizon")
    changes = np.asarray(sr["truth"])-np.asarray(sr["estimates"])
    weights = patient_weights(sr["patient_ids"])
    state = _plain({"source": sr, "validation": vr, "source_sha256": _digest(sr),
                    "validation_sha256": _digest(vr),
                    "p_rms": fit_persistence(sr["estimates"], sr["truth"], sr["patient_ids"]),
                    "e_changes": changes, "e_weights": weights, "e_mean": weights@changes,
                    "filter": _fit_filter(records), "typical_outgoing_weeks": float(np.median(positive)),
                    "outgoing_nonpositive_exclusions": int(np.sum(outgoing <= 0))})
    receipts = {}
    for name, pred in _raw_predictions(state, validation.inputs).items():
        receipts[name] = gaussian_calibration_receipt(pred["mean"], pred["intrinsic_scale"],
                                                     validation.truth, validation.patient_ids)
    receipts["E"] = _e_receipt(state, vr)
    state["calibration_receipts"] = receipts
    state["calibration"] = {name: receipt["selected"] for name, receipt in receipts.items()}
    return ControlFits(_plain(state))


def predict_controls(fits: ControlFits, inputs: ControlInputs) -> dict[str, dict[str, Any]]:
    if not isinstance(fits, ControlFits):
        raise ValueError("expected portable ControlFits")
    x = _inputs(inputs)
    state = fits.state
    result = _raw_predictions(state, inputs)
    for name, pred in result.items():
        pred["calibration"] = float(state["calibration"][name])
    changes, mean = np.asarray(state["e_changes"]), np.asarray(state["e_mean"])
    k = state["calibration"]["E"]
    result["E"] = {"kind": "mixture", "centers": x[:, None, :] + mean + k*(changes-mean),
                   "weights": np.asarray(state["e_weights"]), "observation_sd": OBSERVATION_SD}
    for name, prediction in result.items():
        for field in (("centers", "weights") if name == "E" else ("mean", "intrinsic_scale")):
            if not np.isfinite(prediction[field]).all():
                raise ValueError("nonfinite control prediction")
    return result
