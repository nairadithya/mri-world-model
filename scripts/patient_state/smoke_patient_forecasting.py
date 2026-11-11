#!/usr/bin/env python3
"""Two-real-pair wiring check; this does not train or validate the final forecaster."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from src.patient_state.data import catalogue, load_visit
from src.patient_state.observation import JointObservation
from src.patient_state.temporal import SpatialPredictor, forecast_pairs, feature_nll, ensemble_crps
from src.patient_state.losses import joint_loss, segmentation_metrics
from src.patient_state.state import write_json, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(17)
    observation = JointObservation.from_checkpoint(args.observation)
    initial_observation = digest(args.observation)
    pairs = forecast_pairs(catalogue(args.catalogue)["groups"]["fitting"])[:2]
    if len(pairs) != 2:
        raise ValueError("Need two history-eligible real pairs")
    records = []
    for arm in ("current", "history"):
        torch.manual_seed(17)
        predictor = SpatialPredictor(arm).cuda()
        opt = torch.optim.AdamW(predictor.parameters(), lr=1e-3)
        for pair in pairs:
            # Fixed brain-centered patch, chosen without looking at the future.
            history_features = []
            history = pair["history"][-3:]
            with torch.no_grad():
                for row in history:
                    case = load_visit(row)
                    x = case["images"][:, :, 27:155, 45:173, 27:155].cuda()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        fs = observation.features(x, torch.ones((1, 4), device="cuda", dtype=torch.bool))
                    history_features.append([v.float() for v in fs])
                target = load_visit(pair["target"], labels=True)
                tx = target["images"][:, :, 27:155, 45:173, 27:155].cuda()
                ty = target["labels"][:, 27:155, 45:173, 27:155].cuda()
                tb = target["support"][:, :, 27:155, 45:173, 27:155].cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    target_features = [v.float() for v in observation.features(tx, torch.ones((1, 4), device="cuda", dtype=torch.bool))]
            opt.zero_grad(set_to_none=True)
            memory = None
            for i, fs in enumerate(history_features):
                gap = history[i]["visit_order"][0]-history[i-1]["visit_order"][0] if i else 0
                memory = predictor.update(fs, memory, gap)
            means, logvars = predictor.predict(history_features[-1], memory, pair["horizon"])
            nll = feature_nll(means, logvars, target_features)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, images = observation.decode_features(means)
                anatomy, _ = joint_loss(logits, images, tx, ty, tb, torch.ones_like(tx, dtype=torch.bool))
            loss = nll+.1*anatomy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 12., error_if_nonfinite=True)
            opt.step()
            records.append({"arm": arm, "horizon": pair["horizon"], "target_offset": pair["target_offset"],
                            "feature_nll": float(nll.detach()), "decoded_loss": float(anatomy.detach())})
            del means, logvars, logits, images, loss, nll, anatomy, memory
        path = args.out / f"{arm}.pt"
        torch.save({"model": predictor.state_dict(), "arm": arm, "observation_sha256": initial_observation,
                    "scope": "two-pair instrument check only; not a trained forecast checkpoint"}, path)
        reload = SpatialPredictor(arm).cuda()
        reload.load_state_dict(torch.load(path, weights_only=True)["model"])
        with torch.no_grad():
            def prediction(net):
                memory = None
                for i, fs in enumerate(history_features):
                    gap = history[i]["visit_order"][0]-history[i-1]["visit_order"][0] if i else 0
                    memory = net.update(fs, memory, gap)
                return net.predict(history_features[-1], memory, pair["horizon"])
            a, b = prediction(predictor), prediction(reload)
            difference = max(float((u-v).abs().max()) for u, v in zip(a[0]+a[1], b[0]+b[1]))
            if difference != 0:
                raise AssertionError("Temporal checkpoint replay differs")
            samples = []
            for _ in range(4):
                drawn = [mean+torch.randn_like(mean)*(.5*lv).exp() for mean, lv in zip(*a)]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = observation.decode_features(drawn)
                pred = logits.argmax(1)
                samples.append(torch.stack([((pred == c) & tb[:, 0]).sum()/1000 for c in (1, 2, 3)]))
            reference = torch.stack([((ty == c) & tb[:, 0]).sum()/1000 for c in (1, 2, 3)])
            crps = ensemble_crps(torch.stack(samples), reference)
            records.append({"arm": arm, "reload_max_difference": difference, "decoded_volume_crps_ml": crps.cpu().tolist(),
                            "score": segmentation_metrics(pred, ty, tb[:, 0])})
        if any(p.grad is not None or p.requires_grad for p in observation.parameters()):
            raise AssertionError("Observation model was not frozen")
        del predictor, reload, history_features, target_features, a, b, drawn
    if digest(args.observation) != initial_observation:
        raise AssertionError("Observation checkpoint changed")
    write_json(args.out / "smoke.json", {"passed": True, "scope": "wiring only; final observation and forecasting training pending",
               "observation_unchanged": True, "arms": records})


if __name__ == "__main__":
    main()
