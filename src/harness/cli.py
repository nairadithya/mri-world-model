"""Anatomy-first harness CLI (v2) with explicit legacy compatibility.

Subcommands:
  anatomy manifest           build the frozen lesion-state manifest
  anatomy baseline           run the continuous anatomy gate
  checkpoint inspect PATH    audit old/current checkpoint containers
  legacy <command> ...       reproduce the historical RANO harness

Train methods live under :mod:`src.harness.train`; the CLI preserves every
artifact path in :mod:`src.harness.paths`.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

from . import METHODS, METRICS, PROTOCOLS, TASKS
from .checkpoints import inspect as inspect_checkpoint
from .config import load_config
from .data.protocols import HeroSplit, LockedProtocol
from .encode.cache import load as load_cache
from .encode.encoders import ALL_VIEWS, encode_lumiere
from .eval.aggregate import patient_bootstrap, percentile_ci
from .eval.evaluator import ReadoutEvaluator
from .eval.latent import PREDICTIONS, LatentPairEvaluator
from .eval.report import print_latent, print_provenance, print_result
from .paths import CHAMPION, DEFAULT_CONFIG, EVAL_FOLDS, PROBE_CACHE, PROBE_HEAD
from .runrecord import make_record

# Train methods that live under src/harness/train but are imported lazily (they
# pull heavy deps: MONAI/peft/CNN). Entries are "module:function".
TRAIN_MODULES = {
    "field": "src.harness.train.field:main",
    "head": "src.harness.train.head:main",
    "lora": "src.harness.train.lora:main",
    "cnn3d": "src.harness.train.cnn3d:main",
    "cnn2d": "src.harness.train.cnn2d:main",
    "lesion": "src.harness.train.lesion:main",
}

# Active anatomy workflows.
ANATOMY_MODULES = {
    "manifest": "src.harness.data.anatomy_manifest:main",
    "features": "src.harness.encode.anatomy:main",
    "baseline": "src.harness.analysis.anatomy_baselines:main",
    "residual": "src.harness.analysis.anatomy_residual:main",
    "fusion": "src.harness.analysis.anatomy_fusion:main",
    "coverage": "src.harness.analysis.lesion_coverage:main",
    "lesion-encode": "src.harness.encode.lesion:main",
    "lesion-eval": "src.harness.analysis.lesion_fusion:main",
    "lesion-decode": "src.harness.analysis.lesion_decode:main",
    "lesion-transfer": "src.harness.analysis.lesion_transfer:main",
}

# Historical bespoke analyses / feature builders, imported lazily.
LEGACY_RUN_MODULES = {
    "interface": "src.harness.encode.interface:main",
    "radiomics-features": "src.harness.encode.radiomics:main",
    "surprise": "src.harness.analysis.surprise:main",
    "leadtime": "src.harness.analysis.leadtime:main",
    "persistence": "src.harness.analysis.persistence:main",
    "split-gate": "src.harness.analysis.split_gate:main",
    "horizon-eval": "src.harness.analysis.horizon_eval:main",
    "horizon-probe": "src.harness.analysis.horizon_probe:main",
    "pred-latent": "src.harness.analysis.pred_latent:main",
    "volume": "src.harness.analysis.volume:main",
    "forecast-baselines": "src.harness.analysis.forecast_baselines:main",
    "temporal-baselines": "src.harness.analysis.temporal_baselines:main",
    "tabular-baseline": "src.harness.analysis.tabular_baseline:main",
    "radiomics": "src.harness.analysis.radiomics:main",
    "concept": "src.harness.analysis.concept:main",
    "cross-site": "src.harness.analysis.cross_site:main",
    "sailor": "src.harness.analysis.sailor:main",
    "sailor-gap": "src.harness.analysis.sailor_gap:main",
    "sailor-interval": "src.harness.analysis.sailor_interval:main",
    "freeze": "src.harness.analysis.freeze:main",
    "lock": "src.harness.data.lock:main",
    "manifest": "src.harness.data.manifest:main",
}

# Temporary top-level ``run`` alias accepts both namespaces.
RUN_MODULES = {**LEGACY_RUN_MODULES,
               "anatomy-manifest": ANATOMY_MODULES["manifest"],
               "anatomy-baselines": ANATOMY_MODULES["baseline"]}

USAGE = """\
usage: harness.py <command> [options]

  anatomy manifest [options]    build/audit lesion-state-v1 rows
  anatomy baseline [options]    run the continuous prospective floor
  checkpoint inspect PATH       inspect checkpoint format and legacy contents
  list                          show active anatomy workflows
  legacy <command> [options]    old list/encode/eval/train/run interface

Compatibility aliases remain temporarily available: encode, eval, train, run.
"""


def _log(msg: str) -> None:
    print(msg, flush=True)


def cmd_list(_args) -> None:
    _log("harness:    anatomy-v2")
    _log("anatomy:    manifest, features, baseline, residual, fusion, coverage, "
         "lesion-encode, lesion-eval, lesion-decode, lesion-transfer")
    _log("checkpoint: inspect")
    _log("legacy:     list, encode, eval, train, run")


def cmd_legacy_list(_args=None) -> None:
    _log("legacy metrics:   " + ", ".join(METRICS.names()))
    _log("legacy tasks:     " + ", ".join(TASKS.names()))
    _log("legacy protocols: " + ", ".join(PROTOCOLS.names()))
    _log("legacy methods:   " + ", ".join(
        sorted(set(METHODS.names()) | set(TRAIN_MODULES))))
    _log("legacy run:       " + ", ".join(sorted(LEGACY_RUN_MODULES)))


def _dispatch_entry(spec: str, argv: list[str]) -> None:
    mod_name, fn_name = spec.split(":")
    getattr(importlib.import_module(mod_name), fn_name)(argv)


def cmd_anatomy(rest: list[str]) -> None:
    if not rest or rest[0] in ("-h", "--help", "help"):
        print("usage: harness.py anatomy <" + "|".join(ANATOMY_MODULES) + "> [options]")
        return
    name, argv = rest[0], rest[1:]
    if name not in ANATOMY_MODULES:
        raise SystemExit(
            f"unknown anatomy command {name!r}; known: {sorted(ANATOMY_MODULES)}")
    _dispatch_entry(ANATOMY_MODULES[name], argv)


def cmd_checkpoint(rest: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="harness.py checkpoint")
    sub = ap.add_subparsers(dest="operation", required=True)
    inspect_p = sub.add_parser("inspect")
    inspect_p.add_argument("path")
    inspect_p.add_argument("--json", action="store_true")
    args = ap.parse_args(rest)
    result = inspect_checkpoint(args.path)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")


def _resolve_protocol(spec: str, cfg: dict):
    if spec.endswith(".json") or os.path.sep in spec:
        return LockedProtocol(spec)
    if spec == "hero_split":
        return HeroSplit(cfg)
    if spec == "locked_unseen":
        return LockedProtocol(EVAL_FOLDS)
    return PROTOCOLS.get(spec)(cfg)


def cmd_encode(args) -> None:
    cfg = load_config(args.config)
    encode_lumiere(cfg, args.champion, args.cache,
                   views=tuple(args.views), patients=args.patients,
                   config_path=args.config)


def _run_eval(cfg, cache, args, view):
    task_cls = TASKS.get(args.task)
    task = task_cls() if isinstance(task_cls, type) else task_cls
    proto = _resolve_protocol(args.protocol, cfg)
    ev = ReadoutEvaluator(
        task, proto, view=view, readout=args.readout, hidden=args.hidden,
        seed=args.seed, standardize=args.standardize,
        train_pool=args.train_pool, cohort=args.cohort,
        metrics=args.metrics, boot=args.boot, metric_seed=args.seed)
    return ev.run(cache["patients"])


def cmd_eval(args) -> None:
    cfg = load_config(args.config)
    cache = load_cache(args.cache, warn_provenance=True, require_current=True)
    patients = cache["patients"]
    print_provenance(cache.get("provenance"))
    task_cls = TASKS.get(args.task)
    task = task_cls() if isinstance(task_cls, type) else task_cls
    proto = _resolve_protocol(args.protocol, cfg)
    if task.frame == "latent":
        _cmd_eval_latent(args, cache, task, proto)
        return
    cohort = args.cohort
    try:
        pool = proto.cohort(cohort)
    except (KeyError, TypeError):
        pool = list(patients)
    n_lab = sum(int((patients[p]["labels"] >= 0).sum())
                for p in pool if p in patients and "labels" in patients[p])
    version = getattr(proto, "protocol", {}).get("version", "?")
    _log(f"eval task={task.name} view={args.view or task.default_view} "
         f"protocol={args.protocol} v{version} cohort={cohort} "
         f"{len(pool)} patients, {n_lab} labelled visits; "
         f"train_pool={args.train_pool}, readout={args.readout}")

    result = _run_eval(cfg, cache, args, args.view)
    print_result(result)

    record = make_record("eval", provenance=cache.get("provenance"), config=cfg)
    record.add(result)

    if args.compare and args.compare != (args.view or task.default_view):
        other = _run_eval(cfg, cache, args, args.compare)
        primary = args.metrics[0]
        n_cls = result.meta.get("n_cls", 4)
        metric = METRICS.get(primary)
        metric = metric() if isinstance(metric, type) else metric
        score_metric = getattr(metric, "requires", "labels") == "scores"
        left = result.score_oof if score_metric else result.oof
        right = other.score_oof if score_metric else other.oof
        _, d_vals = patient_bootstrap(
            left, right, boot=args.boot, seed=args.seed,
            metric_fn=lambda p, t: _metric_fn(
                primary, p, t, n_cls, scores=p if score_metric else None))
        dlo, dhi = percentile_ci(d_vals)
        base = result.metrics[primary] - other.metrics[primary]
        sig = "SIG" if dhi < 0 or dlo > 0 else "n.s."
        _log(f"  compare {args.compare}: pooled {other.metrics[primary]:.4f}  "
             f"paired diff {base:+.4f} [{dlo:+.4f},{dhi:+.4f}] {sig}")
        record.add(other)

    if args.out:
        record.write(args.out)
        _log(f"wrote {args.out}")


def _cmd_eval_latent(args, cache, task, proto) -> None:
    """Representation-error path: score latent (t, t+n) predictions."""
    patients = cache["patients"]
    try:
        pool = proto.cohort(args.cohort)
    except (KeyError, TypeError):
        pool = None
    predictions = args.prediction or ["persistence", "champ"]
    _log(f"eval task={task.name} protocol={args.protocol} "
         f"cohort={args.cohort} predictions={predictions}")
    ev = LatentPairEvaluator(
        metrics=args.metrics, predictions=predictions,
        champion_path=args.champion, gap_head_path=args.gap_head,
        per_horizon=True, boot=args.boot, seed=args.seed)
    result = ev.run(patients, pids=pool)
    print_latent(result)
    if args.out:
        record = make_record("eval", provenance=cache.get("provenance"),
                             config=None)
        record.results.append({
            "name": task.name, "methods": result.methods,
            "overall": result.overall,
            "per_horizon": {f"{n}:{m}": v
                            for (n, m), v in result.per_horizon.items()},
            "pairwise": {m: list(v) for m, v in result.pairwise.items()},
            "n_pairs": result.n_pairs, "n_patients": result.n_patients,
        })
        record.write(args.out)
        _log(f"wrote {args.out}")


def _metric_fn(name, pred, y, n_cls=4, scores=None):
    from .eval.metrics import compute_metrics
    metric_pred = pred if scores is None else pred.argmax(dim=-1)
    return compute_metrics([name], y, metric_pred, scores=scores,
                           n_cls=n_cls)[name]


def cmd_train(args) -> None:
    if not args.rest:
        raise SystemExit("usage: harness.py train <method> [args...]")
    method, rest = args.rest[0], args.rest[1:]
    if method == "readout":
        ns = _eval_parser().parse_args(rest)
        cmd_eval(ns)
        return
    if method in METHODS:
        entry = getattr(METHODS.get(method), "entry", None)
        if entry is not None:
            entry(rest)
            return
    if method in TRAIN_MODULES:
        mod_name, fn_name = TRAIN_MODULES[method].split(":")
        getattr(importlib.import_module(mod_name), fn_name)(rest)
        return
    known = sorted(set(METHODS.names()) | set(TRAIN_MODULES))
    raise SystemExit(f"unknown train method {method!r}; known: {known}")


def cmd_run(args) -> None:
    """Dispatch ``run <name>`` to a bespoke analysis / feature-builder module."""
    if not args.rest:
        raise SystemExit("usage: harness.py run <name> [args...]")
    name, rest = args.rest[0], args.rest[1:]
    if name not in RUN_MODULES:
        raise SystemExit(f"unknown run target {name!r}; known: {sorted(RUN_MODULES)}")
    mod_name, fn_name = RUN_MODULES[name].split(":")
    getattr(importlib.import_module(mod_name), fn_name)(rest)


def cmd_legacy(rest: list[str]) -> None:
    if not rest or rest[0] in ("-h", "--help", "help"):
        print("usage: harness.py legacy <list|encode|eval|train|run> [options]")
        return
    command, argv = rest[0], rest[1:]
    if command == "list":
        cmd_legacy_list(argv)
    elif command == "encode":
        cmd_encode(_encode_parser().parse_args(argv))
    elif command == "eval":
        cmd_eval(_eval_parser().parse_args(argv))
    elif command == "train":
        cmd_train(argparse.Namespace(rest=argv))
    elif command == "run":
        if not argv:
            raise SystemExit("usage: harness.py legacy run <name> [args...]")
        name, module_argv = argv[0], argv[1:]
        if name not in LEGACY_RUN_MODULES:
            raise SystemExit(
                f"unknown legacy run target {name!r}; "
                f"known: {sorted(LEGACY_RUN_MODULES)}")
        _dispatch_entry(LEGACY_RUN_MODULES[name], module_argv)
    else:
        raise SystemExit(f"unknown legacy command {command!r}")


# ---------------------------------------------------------------- parsers --
def _encode_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="harness.py encode")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--champion", default=CHAMPION)
    p.add_argument("--cache", default=PROBE_CACHE)
    p.add_argument("--views", nargs="*", default=list(ALL_VIEWS))
    p.add_argument("--patients", nargs="*", default=None)
    return p


def _eval_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="harness.py eval")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--cache", default=PROBE_CACHE)
    p.add_argument("--task", default="rano4_forecast", choices=TASKS.names())
    p.add_argument("--view", default=None)
    p.add_argument("--protocol", default="locked_unseen")
    p.add_argument("--cohort", default="unseen")
    p.add_argument("--train-pool", default="unseen", choices=["unseen", "train"])
    p.add_argument("--readout", default="mlp", choices=["linear", "mlp", "ridge"])
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--standardize", action="store_true")
    p.add_argument("--metrics", nargs="*", default=["macro_f1", "accuracy"])
    p.add_argument("--compare", default=None)
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    # representation-error (frame="latent") options
    p.add_argument("--prediction", nargs="*", default=None,
                   choices=list(PREDICTIONS))
    p.add_argument("--champion", default=CHAMPION)
    p.add_argument("--gap-head", default=PROBE_HEAD)
    return p


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "list":
        cmd_list(rest)
    elif cmd == "anatomy":
        cmd_anatomy(rest)
    elif cmd == "checkpoint":
        cmd_checkpoint(rest)
    elif cmd == "legacy":
        cmd_legacy(rest)
    elif cmd == "encode":
        _log("DEPRECATED: use `harness.py legacy encode`; compatibility alias active")
        cmd_encode(_encode_parser().parse_args(rest))
    elif cmd == "eval":
        _log("DEPRECATED: use `harness.py legacy eval`; compatibility alias active")
        cmd_eval(_eval_parser().parse_args(rest))
    elif cmd == "train":
        _log("DEPRECATED: use `harness.py legacy train`; compatibility alias active")
        cmd_train(argparse.Namespace(rest=rest))
    elif cmd == "run":
        _log("DEPRECATED: prefer `harness.py anatomy ...` or `harness.py legacy run ...`")
        cmd_run(argparse.Namespace(rest=rest))
    else:
        raise SystemExit(f"unknown command {cmd!r}\n\n{USAGE}")
