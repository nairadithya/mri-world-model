#!/usr/bin/env python3
"""Integrated patient-state inference, inventory and observation training."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory", help="Metadata-only follow-up coverage")
    inv.add_argument("--catalogue", type=Path, required=True)
    inv.add_argument("--out", type=Path, required=True)
    timeline = sub.add_parser("timeline", help="Append available visits through one saved-state interface")
    timeline.add_argument("--catalogue", type=Path, required=True)
    timeline.add_argument("--patient", required=True)
    timeline.add_argument("--through", help="Last available visit; later images are never loaded")
    timeline.add_argument("--out", type=Path, required=True)
    timeline.add_argument("--r69", type=Path)
    timeline.add_argument("--r71", type=Path)
    timeline.add_argument("--checkpoint", type=Path, help="Joint observation checkpoint")
    timeline.add_argument("--current-forecast", type=Path, help="Selected current-only forecast checkpoint")
    timeline.add_argument("--history-forecast", type=Path, help="Selected history forecast checkpoint")
    timeline.add_argument("--structured-current",type=Path,help="Structured current-only anatomical transition")
    timeline.add_argument("--structured-history",type=Path,help="Structured history-conditioned anatomical transition")
    timeline.add_argument("--amount-first",type=Path,help="Fixed terminal amount-first quantity checkpoint; failed gates remain diagnostic")
    timeline.add_argument("--amount-location",type=Path,help="Eligible quantity-preserving location checkpoint")
    timeline.add_argument("--forecast-samples", type=int, default=16)
    timeline.add_argument("--seed", type=int, default=17)
    timeline.add_argument("--device", default="cuda")
    timeline.add_argument("--hide", nargs="*", choices=("CT1", "T1", "T2", "FLAIR"), default=[], help="Deliberately unavailable sequences; used for completion")
    args = parser.parse_args()
    from src.patient_state.data import catalogue, horizon_inventory
    from src.patient_state.state import write_json
    value = catalogue(args.catalogue)
    if args.command == "inventory":
        result = horizon_inventory(value)
        write_json(args.out, result)
        print(json.dumps(result, indent=2))
        return
    import torch
    torch.set_num_threads(4)
    from src.patient_state.pipeline import append_visit
    predictors = None
    structured_predictors = None
    amount_predictor = amount_location = None
    if args.checkpoint:
        from src.patient_state.observation import JointObservation
        model = JointObservation.from_checkpoint(args.checkpoint, args.device)
    else:
        if not args.r69 or not args.r71:
            parser.error("timeline requires --checkpoint or both --r69 and --r71")
        from src.patient_state.bridge import LegacyObservation
        model = LegacyObservation(args.r69, args.r71, args.seed, args.device)
    if args.current_forecast or args.history_forecast:
        if not args.checkpoint or not args.current_forecast or not args.history_forecast:
            parser.error('Learned timelines require observation, current-only and history checkpoints')
        if 'FLAIR' in args.hide:
            parser.error('The trained forecast cutoff requires current FLAIR')
        if args.device != 'cuda' or args.forecast_samples < 2:
            parser.error('Learned forecasting requires CUDA and at least two samples')
        from src.patient_state.forecast_inference import load_predictor
        predictors={arm:load_predictor(path,model.model_id) for arm,path in
                    [('current',args.current_forecast),('history',args.history_forecast)]}
        if any(net.arm!=arm for arm,net in predictors.items()):
            parser.error('Current-only and history checkpoint roles must match their arguments')
    if args.structured_current or args.structured_history:
        if not args.checkpoint or not args.structured_current or predictors:
            parser.error('Structured timelines require observation and current-only transition; history is optional')
        if args.hide or args.device!='cuda':
            parser.error('Structured first version requires CUDA and all four available sequences')
        from src.patient_state.structured_inference import load_transition
        structured_predictors={arm:load_transition(path,model.model_id) for arm,path in
                               [('current',args.structured_current),('history',args.structured_history)] if path is not None}
        if any(net.arm!=arm for arm,net in structured_predictors.items()):
            parser.error('Structured checkpoint roles must match')
    if args.amount_first:
        if not args.checkpoint or predictors or structured_predictors or args.hide or args.device!='cuda':
            parser.error('Amount-first requires a frozen joint observation checkpoint, CUDA and all sequences')
        from src.patient_state.amount_first_inference import load_quantity
        amount_predictor=load_quantity(args.amount_first,model.model_id)
        if args.amount_location:
            from src.patient_state.amount_first import LocationReadout
            checkpoint=torch.load(args.amount_location,map_location='cpu',weights_only=True)
            if checkpoint['quantity_sha256']!=amount_predictor.model_id:
                parser.error('Location checkpoint uses different quantities')
            amount_location=LocationReadout().cuda().eval();amount_location.load_state_dict(checkpoint['model'])
    elif args.amount_location:
        parser.error('--amount-location requires --amount-first')
    rows = sorted([r for group in value["groups"].values() for r in group if r["patient"] == args.patient], key=lambda r: r["visit_order"])
    if not rows:
        parser.error("Patient not in development catalogue")
    if any(r.get("dataset") == "MU" for r in rows) and (predictors or any(r.get('time_unit')!='diagnosis_relative_days' for r in rows)):
        parser.error("MU timeline requires verified elapsed times and persistence or structured transitions")
    if args.through:
        names = [r["visit"] for r in rows]
        if args.through not in names:
            parser.error("Unknown cutoff visit")
        rows = rows[:names.index(args.through)+1]
    existing = args.out / args.patient / "timeline.json"
    done = {r["visit"] for r in json.loads(existing.read_text())["visits"]} if existing.exists() else set()
    for row in rows:
        if row["visit"] not in done:
            result = append_visit(model, row, args.out, args.device, hidden_sequences=args.hide,
                                  predictors=predictors, forecast_samples=args.forecast_samples,structured_predictors=structured_predictors,
                                  amount_predictor=amount_predictor,amount_location=amount_location)
            print(json.dumps({k: result[k] for k in ("visit", "current_volumes_ml", "pipeline_seconds", "state_payload_bytes")}), flush=True)


if __name__ == "__main__":
    main()
