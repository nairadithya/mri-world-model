"""One development seed, resumable full schedule, application-aware selection."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from .data import catalogue, load_visit, metadata
from .losses import joint_loss, segmentation_metrics
from .observation import JointObservation
from .state import PatientState, write_json, digest
from .observation_selection import select_full, bounded_rows


class Patches(Dataset):
    def __init__(self, rows, size=128, seed=17, length=100000, offset=0, fixed=False):
        self.rows, self.size, self.seed, self.length, self.offset, self.fixed = rows, size, seed, length, offset, fixed
        groups = defaultdict(list)
        for row in rows:
            groups[row["patient"]].append(row)
        self.groups = list(groups.values())
        datasets = defaultdict(list)
        for group in self.groups:
            datasets[group[0].get("dataset", "LUMIERE")].append(group)
        self.datasets = list(datasets.values())

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + index + self.offset)
        dataset = self.datasets[int(rng.integers(len(self.datasets)))]
        group = dataset[int(rng.integers(len(dataset)))]
        row = self.rows[index // 2 % len(self.rows)] if self.fixed else group[int(rng.integers(len(group)))]
        images = np.load(row["images"], mmap_mode="r")
        mask = np.load(row["mask"], mmap_mode="r")
        support = np.load(row["support"], mmap_mode="r")
        # Sample in ZXY storage, then restore XYZ for the network.
        lesion = (index % 2 == 0) if self.fixed else (rng.random() < .5)
        coarse = (mask[::4, ::4, ::4] > 0) if lesion and row.get("supervised_classes") == 5 else ((mask[::4, ::4, ::4] > 0) & (support[::4, ::4, ::4] > 0) if lesion else support[::4, ::4, ::4] > 0)
        coords = np.argwhere(coarse)
        if not len(coords):
            coords = np.argwhere(support[::4, ::4, ::4] > 0)
        if not len(coords):
            raise ValueError("No supported patch")
        center = coords[int(rng.integers(len(coords)))] * 4
        start = [int(np.clip(c-self.size//2, 0, s-self.size)) for c, s in zip(center, mask.shape)]
        sl = tuple(slice(s, s+self.size) for s in start)
        x = np.array(images[(slice(None), *sl)].transpose(0, 2, 3, 1), dtype=np.float32, order="C")
        y = np.array(mask[sl].transpose(1, 2, 0), dtype=np.int64, order="C")
        b = np.array(support[sl].transpose(1, 2, 0), dtype=bool, order="C")
        if not np.isfinite(x).all() or not np.isin(y, (-1, 0, 1, 2, 3, 4) if row.get("supervised_classes") == 5 else (-1, 0, 1, 2, 3)).all() or not b.any():
            raise ValueError("Invalid patch")
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(b)[None], row["patient"], row.get("supervised_classes", 4)


def augment(x, y, b):
    for axis in (2, 3, 4):
        if random.random() < .5:
            x, y, b = x.flip(axis), y.flip(axis-1), b.flip(axis)
    if random.random() < .5:
        angle = random.uniform(-.17, .17)
        scale = random.uniform(.9, 1.1)
        transform = x.new_tensor([[math.cos(angle), -math.sin(angle), 0, 0],
                                  [math.sin(angle), math.cos(angle), 0, 0], [0, 0, 1, 0]])[None]
        transform[:, :, :3] *= scale
        grid = F.affine_grid(transform.expand(x.shape[0], -1, -1), x.shape, align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        y = F.grid_sample((y[:, None]+1).float(), grid, mode="nearest", align_corners=False)[:, 0].long()-1
        b = F.grid_sample(b.float(), grid, mode="nearest", align_corners=False) > .5
    return x, y, b


def hide(x, forced=None, support=None):
    availability = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    hidden = torch.zeros_like(x, dtype=torch.bool)
    for i in range(len(x)):
        # Half of updates preserve all sequences, but hide an image region.
        if forced is not None or random.random() < .5:
            channel = forced if forced is not None else random.randrange(4)
            availability[i, channel] = False
            hidden[i, channel] = True
            if forced is None and random.random() < .2:
                other = (channel + random.randrange(1, 4)) % 4
                availability[i, other] = False
                hidden[i, other] = True
        if forced is None:
            width = max(8, x.shape[-1]//4)
            if support is not None:
                coordinates = torch.nonzero(support[i, 0, ::4, ::4, ::4], as_tuple=False)*4
                if not len(coordinates):
                    coordinates = torch.nonzero(support[i, 0], as_tuple=False)
                if not len(coordinates):
                    raise ValueError("Cannot hide a brain region in empty support")
                center = coordinates[random.randrange(len(coordinates))].tolist()
                start = [max(0, min(int(c)-width//2, s-width)) for c, s in zip(center, x.shape[2:])]
            else:
                start = [random.randrange(0, s-width+1) for s in x.shape[2:]]
            hidden[(i, slice(None), *(slice(s, s+width) for s in start))] = True
    return x.masked_fill(hidden, 0), availability, hidden


def atomic_checkpoint(path, value):
    tmp = Path(path).with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


@torch.inference_mode()
def select(model, rows, size):
    # Fixed first/last visit per selection patient; two deterministic patches each.
    groups = defaultdict(list)
    patient_datasets = {}
    for row in rows:
        groups[row["patient"]].append(row)
        patient_datasets[row["patient"]] = row.get("dataset", "LUMIERE")
    selected = []
    for visits in groups.values():
        visits.sort(key=lambda r: r["visit_order"])
        selected += [visits[0]] + ([visits[-1]] if len(visits) > 1 else [])
    dataset = Patches(selected, size, seed=7100, length=len(selected)*2, fixed=True)
    loader = DataLoader(dataset, batch_size=1, num_workers=2)
    model.eval()
    records = []
    for x, y, b, patient, supervised in loader:
        x, y, b = x.cuda(), y.cuda(), b.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x, torch.ones(x.shape[:2], device="cuda", dtype=torch.bool))
        metrics = segmentation_metrics(logits[:, :int(supervised[0])].argmax(1), y, b[:, 0], range(1, int(supervised[0])))
        dice = np.mean([v["dice"] for v in metrics.values()])
        hidden_mae, lesion_mae, gradients = [], [], []
        for channel in range(4):
            corrupted, availability, hidden = hide(x, channel)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits_hidden, predicted = model(corrupted, availability)
            mask = b[:, 0]
            error = (predicted[:, channel].float()-x[:, channel]).abs()
            hidden_mae.append(float(error[mask].mean()))
            lesion = (y > 0) & mask
            lesion_mae.append(float(error[lesion].mean()) if lesion.any() else None)
            gradient_values = []
            for axis in (1, 2, 3):
                a, z = [slice(None)]*4, [slice(None)]*4
                a[axis], z[axis] = slice(1, None), slice(None, -1)
                a, z = tuple(a), tuple(z)
                valid = mask[a] & mask[z]
                residual = predicted[:, channel].float()-x[:, channel]
                gradient_values.append(float((residual[a]-residual[z]).abs()[valid].mean()))
            gradients.append(float(np.mean(gradient_values)))
        records.append({"patient": patient[0], "dataset": patient_datasets[patient[0]], "dice": float(dice), "compartments": metrics,
                        "missing_sequence_mae": hidden_mae, "missing_sequence_lesion_mae": lesion_mae,
                        "missing_sequence_gradient_mae": gradients})
    per_patient = []
    for patient in sorted({r["patient"] for r in records}):
        rs = [r for r in records if r["patient"] == patient]
        lesions = [v for r in rs for v in r["missing_sequence_lesion_mae"] if v is not None]
        per_patient.append({"dataset": patient_datasets[patient], "dice": float(np.mean([r["dice"] for r in rs])),
                            "mae": float(np.mean([r["missing_sequence_mae"] for r in rs])),
                            "lesion_mae": float(np.mean(lesions)) if lesions else None,
                            "gradient_mae": float(np.mean([r["missing_sequence_gradient_mae"] for r in rs]))})
    metric_keys = ("dice", "mae", "lesion_mae", "gradient_mae")
    by_dataset = {dataset: {k: float(np.mean([r[k] for r in per_patient if r["dataset"] == dataset and r[k] is not None])) for k in metric_keys}
                  for dataset in sorted({r["dataset"] for r in per_patient})}
    aggregate = {k: float(np.mean([r[k] for r in by_dataset.values()])) for k in metric_keys}
    aggregate["selection_score"] = 1-aggregate["dice"] + .25*aggregate["lesion_mae"] + .1*aggregate["mae"] + .1*aggregate["gradient_mae"]
    aggregate["scope"] = "fixed selection patches; full-volume evaluation required before usefulness claim"
    aggregate["by_dataset"] = by_dataset
    aggregate["records"] = records
    model.train()
    return aggregate


def train(args):
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    value = catalogue(args.catalogue)
    catalogue_sha256 = digest(args.catalogue)
    args.out.mkdir(parents=True, exist_ok=True)
    model = JointObservation(cavity=bool(args.initialize_observation)).cuda()
    initialization = model.initialize_observation(args.initialize_observation) if args.initialize_observation else model.initialize_pretrained(args.pretrained)
    parameters = list(model.parameters())
    opt = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    start_step, best, curve = 0, float("inf"), []
    if args.resume:
        saved = torch.load(args.out / "last.pt", map_location="cpu", weights_only=False)
        if saved["config"].get("catalogue_sha256") != catalogue_sha256 or saved["initialization"]["checkpoint_sha256"] != initialization["checkpoint_sha256"]:
            raise ValueError("Resume data or pretrained initialization changed")
        for key in ("steps", "patch_size", "seed", "accumulate", "lr", "select_every", "full_volume_selection"):
            if saved["config"][key] != getattr(args, key):
                raise ValueError("Resume schedule changed: " + key)
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        start_step, best = saved["step"], saved["best"]
        random.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        curve = json.loads((args.out / "curve.json").read_text()) if (args.out / "curve.json").exists() else []
    elif (args.out / "last.pt").exists():
        raise FileExistsError("Use --resume or a fresh output directory")
    write_json(args.out / "initialization.json", initialization)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["catalogue_sha256"] = catalogue_sha256
    config["source_sha256"] = {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    write_json(args.out / "config.json", config)
    if args.full_volume_selection:
        write_json(args.out / "selection-population.json", bounded_rows(value["groups"]["selection"]))
    rows = value["groups"]["fitting"]
    if args.smoke:
        rows = [next(r for r in rows if r.get("dataset", "LUMIERE") == dataset)
                for dataset in sorted({r.get("dataset", "LUMIERE") for r in rows})]
    dataset = Patches(rows, args.patch_size, args.seed, (args.steps-start_step)*args.accumulate,
                      offset=start_step*args.accumulate, fixed=args.smoke)
    loader = DataLoader(dataset, batch_size=1, num_workers=2, pin_memory=True)
    iterator = iter(loader)
    model.train()
    started = time.perf_counter()
    elapsed_updates = []
    first_weight = model.encoder.stem.convs[0].conv.weight.detach().clone() if args.smoke else None
    for step in range(start_step+1, args.steps+1):
        update_start = time.perf_counter()
        # Linear warmup followed by a complete cosine decay, not an update-budget gate.
        warmup = min(1000, max(1, args.steps//10))
        fraction = min(1., step/warmup) if step <= warmup else .5*(1+math.cos(math.pi*(step-warmup)/(args.steps-warmup)))
        for group in opt.param_groups:
            group["lr"] = args.lr * max(.01, fraction)
        opt.zero_grad(set_to_none=True)
        logs = []
        for _ in range(args.accumulate):
            x, y, b, _, supervised = next(iterator)
            x, y, b = x.cuda(), y.cuda(), b.cuda()
            x, y, b = augment(x, y, b)
            corrupted, availability, hidden = hide(x, support=b)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, reconstructed = model(corrupted, availability)
                loss, metrics = joint_loss(logits, reconstructed, x, y, b, hidden, int(supervised[0]))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            (loss/args.accumulate).backward()
            logs.append(metrics)
        torch.nn.utils.clip_grad_norm_(parameters, 12., error_if_nonfinite=True)
        opt.step()
        torch.cuda.synchronize()
        elapsed_updates.append(time.perf_counter()-update_start)
        if step % 25 == 0 or args.smoke:
            print(json.dumps({"step": step, "lr": opt.param_groups[0]["lr"], "loss": float(loss.detach()),
                              "seconds_per_update": float(np.mean(elapsed_updates[-25:])), **metrics}), flush=True)
        if step % args.select_every == 0 or step == args.steps:
            result = (select_full(model, value["groups"]["selection"]) if args.full_volume_selection else select(model, value["groups"]["selection"], args.patch_size)) if not args.smoke else {"selection_score": float(loss.detach()), "scope": "smoke only, not selection"}
            improved = result["selection_score"] < best
            best = min(best, result["selection_score"])
            curve.append({"step": step, "elapsed_seconds": time.perf_counter()-started, **result})
            write_json(args.out / "curve.json", curve)
            saved = {"backend": model.backend, "model": model.state_dict(), "optimizer": opt.state_dict(),
                     "step": step, "best": best, "config": config, "initialization": initialization,
                     "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}
            atomic_checkpoint(args.out / "last.pt", saved)
            if improved:
                atomic_checkpoint(args.out / "best.pt", saved)
            print(json.dumps({"step": step, "selection": {k: v for k, v in result.items() if k != "records"}, "best": best}), flush=True)
        elif step % 250 == 0:
            atomic_checkpoint(args.out / "last.pt", {"backend": model.backend, "model": model.state_dict(),
                "optimizer": opt.state_dict(), "step": step, "best": best, "config": config, "initialization": initialization,
                "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()})
        if args.smoke and step == args.steps:
            model.eval()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                expected = model(corrupted, availability)
            reload = JointObservation.from_checkpoint(args.out / "last.pt")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                actual = reload(corrupted, availability)
            differences = [float((a.float()-b.float()).abs().max()) for a, b in zip(expected, actual)]
            if max(differences) != 0 or torch.equal(first_weight, model.encoder.stem.convs[0].conv.weight):
                raise AssertionError("Checkpoint replay or encoder training failed")
            meta = metadata(rows[0])
            meta["availability"] = availability[0].cpu().tolist()
            state = reload.encode(corrupted[:1], meta)
            state.save(args.out / "smoke-state.pt")
            from_state = reload.read(PatientState.load(args.out / "smoke-state.pt", "cuda"))
            state_difference = float((from_state["logits"].float()-actual[0][:1].float()).abs().max())
            if state_difference != 0:
                raise AssertionError("Saved-state readout differs")
            score = segmentation_metrics(actual[0][:, :int(supervised[0])].argmax(1), y, b[:, 0], range(1, int(supervised[0])))
            write_json(args.out / "smoke.json", {"passed": True, "updates": args.steps,
                       "checkpoint_max_difference": differences, "state_max_difference": state_difference,
                       "encoder_changed": True, "score": score, "mean_seconds_per_update": float(np.mean(elapsed_updates)),
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "state_payload_bytes": state.payload_bytes})
    write_json(args.out / "completed.json", {"steps": args.steps, "best_selection_score": best,
               "elapsed_seconds": time.perf_counter()-started, "mean_update_seconds": float(np.mean(elapsed_updates)),
               "boundary_review_required": not args.smoke, "scope": "development fitting complete; usefulness not established by training completion"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, required=True)
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument("--pretrained", type=Path)
    initialization.add_argument("--initialize-observation", type=Path)
    parser.add_argument("--full-volume-selection", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=37500)
    parser.add_argument("--select-every", type=int, default=1250)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.patch_size not in (64, 96, 128, 160) or args.steps < 2:
        parser.error("Use a supported 32-divisible patch size and at least two updates")
    train(args)


if __name__ == "__main__":
    main()
