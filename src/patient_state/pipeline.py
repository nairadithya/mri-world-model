"""Append observations and preserve issued forecasts in the existing viewer tree."""
from pathlib import Path
import json
import time
import uuid

import nibabel as nib
import numpy as np
import torch

from .anatomy_readouts import anatomy_readouts, readout_contract
from .data import load_visit, metadata
from .state import LABELS, SEQUENCES, PatientState, check_update, persistence, write_json


def volumes(segmentation, affine, labels=LABELS):
    voxel_ml = abs(float(np.linalg.det(np.asarray(affine)[:3, :3]))) / 1000
    return {name: float((segmentation == label).sum()) * voxel_ml for label, name in labels.items()}


def save_nifti(path, array, affine):
    nib.save(nib.Nifti1Image(np.asarray(array), np.asarray(affine)), str(path))


def append_visit(model, row, root, device="cuda", horizons=(12, 24), hidden_sequences=(), predictors=None, forecast_samples=16, structured_predictors=None, amount_predictor=None, amount_location=None):
    from .timing import elapsed_weeks
    if amount_predictor is not None and (predictors or structured_predictors or hidden_sequences or row.get('dataset')!='MU'):
        raise ValueError('Amount-first v1 requires MU current-only full-sequence context and a separate forecast formulation')
    if amount_location is not None and amount_predictor is None:
        raise ValueError('Location requires explicit quantities')
    if predictors and structured_predictors:
        raise ValueError('Choose one transition formulation per timeline')
    if structured_predictors and hidden_sequences:
        raise ValueError('Structured first version was fitted with all four sequences available')
    root = Path(root)
    patient_root = root / row["patient"]
    timeline_path = patient_root / "timeline.json"
    timeline = json.loads(timeline_path.read_text()) if timeline_path.exists() else {
        "version": 1, "patient": row["patient"], "visits": [],
        "scope": "Development model outputs; automated labels; nominal week bins"}
    if getattr(model, "is_smoke", False):
        timeline["scope"] = "INSTRUMENT CHECK: untrained smoke model, not useful anatomical predictions."
    destination = patient_root / row["visit"]
    if destination.exists():
        raise FileExistsError("Visit already exists; choose a new output root for another model")
    start = time.perf_counter()
    case = load_visit(row)
    meta = dict(metadata(row), dataset=row.get("dataset", "LUMIERE"))
    for name in hidden_sequences:
        meta["availability"][SEQUENCES.index(name)] = False
    observation = model.encode(case["images"].to(device), meta)
    # A hidden sequence must not supply even an inference-time foreground mask.
    available = observation.metadata["availability"]
    brain_support = case["support"] if available[0] else (case["images"][:, available] != 0).any(1, keepdim=True)
    observation.features["brain_support"] = brain_support.to(device)
    if structured_predictors or amount_predictor is not None:
        observation.features['observed_images'] = case['images']
    observation.metadata["support_source"] = "available CT1 foreground" if available[0] else "union of available-sequence foregrounds"
    if predictors:
        # Rebuild the bounded four-visit memory from declared saved states. Older
        # observations cannot silently accumulate beyond the trained window.
        history = predictors['history']
        previous = None
        with torch.inference_mode():
            for prior in timeline['visits'][-3:]:
                loaded = PatientState.load(patient_root / prior['state'])
                if loaded.metadata.get('temporal_model_id') != history.model_id:
                    raise ValueError('Timeline belongs to another temporal model; choose a new output root')
                previous = history.assimilate_state(loaded, previous)
                previous.memory = {k:v.cpu() for k,v in previous.memory.items()}
            observation = history.assimilate_state(observation, previous)
            observation.memory = {k:v.cpu() for k,v in observation.memory.items()}
        observation.metadata.update(temporal_model_id=history.model_id, history_limit=4)
        timeline.update(version=2, scope='Development predictions from observations available at each cutoff. Nominal weeks. Current volumes are model estimates. Later comparisons use model-derived anatomy, not reference labels.',
                        forecast_uncertainty=f'90% empirical volume intervals from {forecast_samples} draws; calibration is not assumed. Persistence has no fitted uncertainty.')
        if any(net.is_smoke for net in predictors.values()):
            timeline['scope']='INSTRUMENT CHECK: smoke checkpoints, not trained forecasts. '+timeline['scope']
    elif timeline["visits"]:
        previous = PatientState.load(patient_root / timeline["visits"][-1]["state"])
        check_update(previous, observation)
        # Baseline memory retains the previous compact/coarse representation and time.
        key = "compact" if "compact" in previous.features else sorted(k for k in previous.features if "feature" in k)[-1]
        observation.memory = {"previous_coarse": previous.features[key].to(device),
                              "previous_week": torch.tensor([previous.metadata["nominal_week"] if previous.metadata['time_unit']=='nominal_week' else previous.metadata['diagnosis_relative_day']/7], device=device)}
        del previous
    final_destination = destination
    destination = patient_root / (".pending-" + row["visit"] + "-" + uuid.uuid4().hex[:8])
    destination.mkdir(parents=True)
    state_path = destination / "state.pt"
    observation.save(state_path)
    payload = observation.payload_bytes
    del observation
    # Both applications run exclusively from the serialized state.
    state = PatientState.load(state_path, "cpu" if model.backend.startswith("resenc-l-joint") else device)
    out = model.read(state)
    support = state.features["brain_support"][0, 0].cpu().numpy()
    maps = anatomy_readouts(out["logits"], torch.as_tensor(support, device=out["logits"].device)[None])
    seg = maps["native"][0].cpu().numpy().astype(np.uint8)
    affine = state.metadata["affine"]
    labels = dict(LABELS)
    if getattr(model, "num_classes", 4) == 5:
        labels[4] = "resection_cavity"
        timeline["anatomy_note"] = "Five anatomical categories include cavity. LUMIERE has no cavity reference annotation; its cavity predictions remain unvalidated."
    current_volumes = volumes(seg, affine, labels)
    save_nifti(destination / "segmentation.nii.gz", seg, affine)
    anatomy = readout_contract(out["logits"].shape[1], row.get("dataset", "LUMIERE"))
    for name, tensor in maps.items():
        array = tensor[0].cpu().numpy().astype(np.uint8)
        filename = "segmentation.nii.gz" if name == "native" else "conditional_legacy_segmentation.nii.gz"
        if name != "native":
            save_nifti(destination / filename, array, affine)
        anatomy[name].update(segmentation=f"{row['visit']}/{filename}",
                             volumes_ml=volumes(array, affine, labels if name == "native" else LABELS),
                             model_id=model.model_id, state=f"{row['visit']}/state.pt")
    for i, name in enumerate(SEQUENCES):
        save_nifti(destination / f"{name}.nii.gz", case["images"][0, i].numpy(), affine)
    for i, name in enumerate(out["image_sequences"]):
        save_nifti(destination / f"predicted_{name}.nii.gz", out["images"][0, i].float().cpu().numpy(), affine)
    forecasts = []
    structured_context_path = None
    for horizon in horizons:
        prediction = persistence(state, horizon)
        # A persistence state is an immutable reference to the issued observation state.
        # Duplicating the hundreds-of-MiB features would add no information.
        forecasts.append({"method": "persistence", "horizon_nominal_weeks": horizon,
                          "forecast_week": prediction.metadata["forecast_week"],
                          "state": f"{row['visit']}/state.pt", "volumes_ml": current_volumes,
                          "segmentation": f"{row['visit']}/segmentation.nii.gz",
                          "uncertainty": None, "uncertainty_status": prediction.metadata["uncertainty_status"]})
    if predictors:
        from .forecast_inference import forecast_distribution, sample_state
        with torch.inference_mode():
            for arm, predictor in predictors.items():
                for horizon in horizons:
                    mean_state, logvars = forecast_distribution(predictor, state, horizon)
                    decoded = model.read(mean_state)
                    forecast_seg = decoded['logits'].argmax(1)[0].cpu().numpy().astype(np.uint8)
                    forecast_seg[~support] = 0
                    name = f'{arm}_h{horizon}'
                    save_nifti(destination/(name+'_segmentation.nii.gz'), forecast_seg, affine)
                    save_nifti(destination/(name+'_FLAIR.nii.gz'), decoded['images'][0,3].float().cpu().numpy(), affine)
                    draws = []
                    for sample in range(forecast_samples):
                        drawn = sample_state(mean_state, logvars, 17000+sample)
                        sample_seg = model.read(drawn)['logits'].argmax(1)[0].cpu().numpy().astype(np.uint8)
                        sample_seg[~support] = 0
                        draws.append(list(volumes(sample_seg, affine, labels).values()))
                    draws = np.asarray(draws)
                    lo, hi = np.quantile(draws, [.05,.95], axis=0)
                    forecasts.append({'method':arm, 'horizon_nominal_weeks':horizon,
                        'forecast_week':row['visit_order'][0]+horizon,
                        'segmentation':f"{row['visit']}/{name}_segmentation.nii.gz",
                        'image':f"{row['visit']}/{name}_FLAIR.nii.gz", 'predictor_id':predictor.model_id,
                        'history_used':([v['visit'] for v in timeline['visits'][-3:]]+[row['visit']]) if arm=='history' else [row['visit']],
                        'volumes_ml':volumes(forecast_seg,affine,labels),
                        'distribution':{'samples_ml':draws.tolist(),'mean_ml':draws.mean(0).tolist(),
                                        'lower90_ml':lo.tolist(),'upper90_ml':hi.tolist(),'width90_ml':(hi-lo).tolist()},
                        'uncertainty_status':'learned feature marginal scales; decoded empirical intervals, calibration not assumed'})
    if structured_predictors:
        from .structured_inference import make_context_state, predict_context
        context_state=make_context_state(state,row,seg,state.features['observed_images'],state.features['brain_support'])
        context_state.metadata['transition_ids']={arm:net.model_id for arm,net in structured_predictors.items()}
        context_state.save(destination/'forecast_context.pt')
        context_state=PatientState.load(destination/'forecast_context.pt')
        structured_context_path=f"{row['visit']}/forecast_context.pt"
        earlier=[v for v in timeline['visits'] if v['nominal_week']<elapsed_weeks(row)]
        previous_context=PatientState.load(patient_root/earlier[-1]['forecast_context']) if earlier else None
        if previous_context is not None and previous_context.metadata.get('transition_ids')!=context_state.metadata['transition_ids']:
            raise ValueError('Use a new timeline for different transition checkpoints')
        for arm,predictor in structured_predictors.items():
            for horizon in horizons:
                result=predict_context(predictor,context_state,previous_context if arm=='history' else None,horizon)
                name=f'{arm}_h{horizon}'
                save_nifti(destination/(name+'_segmentation.nii.gz'),result['segmentation'],affine)
                save_nifti(destination/(name+'_FLAIR.nii.gz'),result['FLAIR'],affine)
                # Compact issued parameters plus an immutable, explicit context reference.
                torch.save({'parameters':result['parameters'],'context_state':structured_context_path,
                            'predictor_id':predictor.model_id,'horizon':horizon},destination/(name+'_distribution.pt'))
                forecasts.append({'method':arm,'horizon_nominal_weeks':horizon,'forecast_week':elapsed_weeks(row)+horizon,
                    'segmentation':f"{row['visit']}/{name}_segmentation.nii.gz",'image':f"{row['visit']}/{name}_FLAIR.nii.gz",
                    'state':f"{row['visit']}/{name}_distribution.pt",'predictor_id':predictor.model_id,
                    'history_used':([previous_context.metadata['visit']] if arm=='history' and previous_context is not None else [])+[row['visit']],
                    'volumes_ml':volumes(result['segmentation'],affine,labels),'distribution':result['distribution'],
                    'uncertainty_status':'Learned eight-scenario anatomical mixture; displayed map, image and point volumes share the highest-weight scenario. 90% volume intervals are uncalibrated.'})
        timeline.update(version=2,scope='Development structured forecasts. Current anatomy is frozen-model output. Later comparisons use newly observed model anatomy. MU timing is recorded diagnosis-relative days expressed in weeks; LUMIERE uses nominal week bins.',
                        forecast_uncertainty='90% weighted intervals from eight learned anatomical scenarios; calibration must be evaluated.')
        if any(net.is_smoke for net in structured_predictors.values()):
            timeline['scope']='INSTRUMENT CHECK: smoke checkpoints, not trained predictions. '+timeline['scope']
    if amount_predictor is not None:
        from .structured_inference import make_context_state
        from .amount_first_inference import predict_amount_state
        context_state=make_context_state(state,row,seg,state.features['observed_images'],state.features['brain_support'])
        context_state.save(destination/'forecast_context.pt')
        context_state=PatientState.load(destination/'forecast_context.pt')
        structured_context_path=f"{row['visit']}/forecast_context.pt"
        for horizon in horizons:
            result=predict_amount_state(amount_predictor,context_state,horizon,amount_location)
            name=f'amount_first_h{horizon}'
            save_nifti(destination/(name+'_sample0.nii.gz'),result['segmentation'],affine)
            torch.save({'sample_counts':result['sample_counts'],'representative_sample':0,
                        'context_state':structured_context_path,'quantity_id':amount_predictor.model_id,
                        'horizon':horizon},destination/(name+'_samples.pt'))
            forecasts.append({'method':'amount-first '+('learned location' if amount_location is not None else 'current-anatomy allocation'),
                'horizon_nominal_weeks':horizon,'forecast_week':elapsed_weeks(row)+horizon,
                'segmentation':f"{row['visit']}/{name}_sample0.nii.gz",'state':f"{row['visit']}/{name}_samples.pt",
                'volumes_ml':volumes(result['segmentation'],affine,labels),'distribution':result['distribution'],
                'representative_sample':0,'presence_probability':result['presence_probability'],
                'predictor_id':amount_predictor.model_id,'history_used':[row['visit']],
                'uncertainty_status':amount_predictor.status+'; 64 joint samples, empirical 90% intervals; calibration not established',
                'forecast_image_status':'Anatomy overlay on cutoff FLAIR; no future image synthesis'})
        timeline.update(version=2,scope='MU amount-first development forecasts. '+amount_predictor.status+'. Displayed sample0 volumes and distribution means are distinct. Observation and completion share the frozen saved state.',
                        forecast_uncertainty='64 joint presence/count samples; empirical 90% intervals; no calibration claim')
    comparisons = []
    for prior in timeline["visits"]:
        for forecast in prior["forecasts"]:
            error = elapsed_weeks(row) - forecast["forecast_week"]
            if abs(error) > 4 or elapsed_weeks(row) <= prior["nominal_week"]:
                continue
            old_seg = np.asarray(nib.load(patient_root / forecast["segmentation"]).dataobj)
            dice = {}
            for label, name in labels.items():
                a, b = old_seg == label, seg == label
                total = int(a.sum() + b.sum())
                dice[name] = float(2 * (a & b).sum() / total) if total else 1.
            comparisons.append({"issued_at": prior["visit"], "horizon_nominal_weeks": forecast["horizon_nominal_weeks"],
                                "target_offset_nominal_weeks": error, "method": forecast["method"],
                                "reference": "new scan model segmentation, not ground truth",
                                "forecast_segmentation": forecast['segmentation'],
                                "dice_against_current_prediction": dice,
                                "volume_error_ml": {k: forecast["volumes_ml"][k] - v for k, v in current_volumes.items()}})
            if forecast.get('distribution'):
                old=np.array(list(prior['current_volumes_ml'].values()))
                new=np.array(list(current_volumes.values()))
                predicted=np.array(list(forecast['volumes_ml'].values()))
                lo=np.array(forecast['distribution']['lower90_ml']);hi=np.array(forecast['distribution']['upper90_ml'])
                comparisons[-1].update(observed_change_ml=(new-old).tolist(),predicted_change_ml=(predicted-old).tolist(),
                    change_group='changed' if (np.abs(new-old)>np.maximum(2,.2*old)).any() else 'stable',
                    coverage90=((new>=lo)&(new<=hi)).tolist(),width90_ml=(hi-lo).tolist())
    entry = {"visit": row["visit"], "nominal_week": elapsed_weeks(row), "time_unit": state.metadata['time_unit'],
             "state": f"{row['visit']}/state.pt", "model_id": model.model_id,
             "backend": model.backend, "input_sequences": [s for s, a in zip(SEQUENCES, state.metadata["availability"]) if a],
             "current_volumes_ml": current_volumes, "anatomy_readouts": anatomy, "displayed_anatomy": "native",
             "image_readout_state": f"{row['visit']}/state.pt", "forecasts": forecasts,
             "previous_forecast_comparisons": comparisons, "state_payload_bytes": payload,
             "state_file_bytes": state_path.stat().st_size, "pipeline_seconds": time.perf_counter() - start}
    if structured_context_path:
        entry['forecast_context']=structured_context_path
    timeline["visits"].append(entry)
    write_json(destination / "outputs.json", entry)
    destination.rename(final_destination)
    write_json(timeline_path, timeline)
    return entry
