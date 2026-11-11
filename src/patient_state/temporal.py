"""Frozen-observation spatial recurrence and horizon-conditioned feature changes.

This module does not label an untrained predictor as useful. Its checkpoint must
identify the fixed observation checkpoint, comparison arm and trained horizons.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .observation import FEATURES
from .state import PatientState, check_update


class SpatialPredictor(nn.Module):
    def __init__(self, arm="history", memory_channels=32):
        super().__init__()
        if arm not in ("history", "current"):
            raise ValueError("Use history or current; persistence has no fitted model")
        self.arm = arm
        self.input_projection = nn.Conv3d(FEATURES[-1]+4, memory_channels, 1)
        self.gates = nn.Conv3d(2*memory_channels+1, 2*memory_channels, 3, padding=1)
        self.candidate = nn.Conv3d(2*memory_channels+1, memory_channels, 3, padding=1)
        self.horizon = nn.Sequential(nn.Linear(1, memory_channels), nn.SiLU(), nn.Linear(memory_channels, memory_channels))
        self.local = nn.ModuleList([nn.Conv3d(c, 2*c, 1) for c in FEATURES])
        self.context = nn.ModuleList([nn.Conv3d(memory_channels, 2*c, 1) for c in FEATURES])
        for module in (*self.local, *self.context):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        for module, c in zip(self.local, FEATURES):
            with torch.no_grad():
                module.bias[c:] = -2.

    def update(self, features, memory=None, previous_gap=0., availability=None):
        if float(previous_gap) < 0:
            raise ValueError("Observed elapsed time cannot be negative")
        f = features[-1].float()
        a = f.new_ones((f.shape[0], 4)) if availability is None else availability.to(f)
        if a.shape != (f.shape[0], 4) or not a.bool().any(1).all():
            raise ValueError("Invalid observed sequence availability")
        current = self.input_projection(torch.cat((f, a[..., None, None, None].expand(-1, -1, *f.shape[2:])), 1))
        if memory is None or self.arm == "current":
            memory = torch.zeros_like(current)
            previous_gap = 0.
        if memory.shape != current.shape:
            raise ValueError("Temporal memory must use the same registered patch grid")
        gap = torch.full_like(current[:, :1], math.log1p(float(previous_gap))/math.log(53))
        reset, update = self.gates(torch.cat((current, memory, gap), 1)).sigmoid().chunk(2, 1)
        candidate = self.candidate(torch.cat((current, reset*memory, gap), 1)).tanh()
        return (1-update)*memory + update*candidate

    def predict(self, current_features, memory, horizon):
        if horizon not in (12, 24):
            raise ValueError("Only the declared nominal horizons are supported")
        h = memory.new_full((memory.shape[0], 1), float(horizon)/24)
        conditioned = memory + self.horizon(h)[..., None, None, None]
        means, logvars = [], []
        for feature, local, context in zip(current_features, self.local, self.context):
            parameters = local(feature.float()) + F.interpolate(context(conditioned), size=feature.shape[2:], mode="trilinear", align_corners=False)
            delta, logvar = parameters.chunk(2, 1)
            means.append(feature.float()+delta)
            logvars.append(logvar.clamp(-8, 4))
        return means, logvars

    def assimilate_state(self, current, previous=None):
        if previous is not None:
            check_update(previous, current)
        memory = {}
        patches = current.metadata.get("patches", [{"prefix": ""}])
        device = next(self.parameters()).device
        for patch in patches:
            prefix = patch["prefix"]
            features = [current.features[prefix+f"feature{i}"].to(device) for i in range(6)]
            old = previous.memory.get(prefix+"recurrent") if previous is not None else None
            if previous is not None and self.arm == "history" and old is None:
                raise ValueError("Assimilate the previous observation before updating history")
            old = old.to(device) if old is not None else None
            gap = current.metadata["nominal_week"] - previous.metadata["nominal_week"] if previous is not None else 0
            memory[prefix+"recurrent"] = self.update(features, old, gap, torch.tensor([current.metadata["availability"]], device=device))
        return PatientState(current.features, dict(current.metadata, temporal_arm=self.arm), memory)

    def forecast_state(self, state, horizon, sample=False, generator=None):
        if 'patches' in state.metadata:
            from .forecast_inference import forecast_distribution, sample_state
            predicted, logvars = forecast_distribution(self, state, horizon)
            if sample:
                seed = int(torch.randint(2**31, (), generator=generator, device=generator.device if generator is not None else 'cpu'))
                predicted = sample_state(predicted, logvars, seed)
            return predicted, logvars
        features, variances = {}, {}
        device = next(self.parameters()).device
        for patch in state.metadata.get("patches", [{"prefix": ""}]):
            prefix = patch["prefix"]
            current = [state.features[prefix+f"feature{i}"].to(device) for i in range(6)]
            memory = state.memory[prefix+"recurrent"].to(device)
            means, logvars = self.predict(current, memory, horizon)
            for i, (mean, logvar) in enumerate(zip(means, logvars)):
                key = prefix+f"feature{i}"
                features[key] = mean + (torch.randn(mean.shape, device=device, generator=generator)*(.5*logvar).exp() if sample else 0)
                variances[key] = logvar
        if "brain_support" in state.features:
            features["brain_support"] = state.features["brain_support"]
        meta = dict(state.metadata, horizon_nominal_weeks=horizon,
                    forecast_week=state.metadata["nominal_week"]+horizon,
                    forecast_method=self.arm, uncertainty_status="learned feature Gaussian; decoded calibration requires evaluation")
        return PatientState(features, meta, {**state.memory, **{"logvar:"+k: v for k, v in variances.items()}}), variances


def feature_nll(means, logvars, targets):
    return torch.stack([.5*((target.float()-mean).square()*(-lv).exp()+lv+math.log(2*math.pi)).mean()
                        for mean, lv, target in zip(means, logvars, targets)]).mean()


def ensemble_crps(samples, target):
    """Proper empirical CRPS; first axis is predictive samples."""
    return (samples-target).abs().mean(0) - .5*(samples[:, None]-samples[None, :]).abs().mean((0, 1))


def forecast_pairs(rows, horizons=(12, 24), tolerance=4):
    """Target matching is bookkeeping only; realized future gaps are never inputs."""
    patients = {}
    for row in rows:
        if row.get("dataset", "LUMIERE") != "LUMIERE":
            continue
        patients.setdefault(row["patient"], []).append(row)
    pairs = []
    for patient, visits in patients.items():
        visits.sort(key=lambda r: r["visit_order"])
        for i, current in enumerate(visits):
            if i == 0:
                continue  # common history-eligible comparison population
            for horizon in horizons:
                candidates = [(abs(r["visit_order"][0]-current["visit_order"][0]-horizon), r) for r in visits[i+1:]
                              if r["visit_order"][0] > current["visit_order"][0]]
                if not candidates:
                    continue
                distance, target = min(candidates, key=lambda item: (item[0], item[1]["visit_order"]))
                if distance <= tolerance:
                    pairs.append({"patient": patient, "history": visits[:i+1], "target": target, "horizon": horizon,
                                  "target_offset": target["visit_order"][0]-current["visit_order"][0]-horizon})
    return pairs
