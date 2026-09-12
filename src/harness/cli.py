"""Unified harness CLI.

Subcommands:
  list                       show registered metrics/tasks/protocols/methods
  encode                     frozen representation pass -> feature cache
  eval                       task x protocol x readout x metrics (pluggable)
  train <method> [args...]   train a representation method (native or legacy)

Old ``scripts/*.py`` entry points remain until each method is migrated; this
CLI is the forward target and preserves every artifact path in
:mod:`src.harness.paths`.
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys

from . import METHODS, METRICS, PROTOCOLS, TASKS
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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LEGACY_TRAIN = {
    "jepa": "scripts/run_train.py",
    "field": "scripts/train_field.py",
    "head": "scripts/task_train.py",
    "lora": "scripts/finetune_lora.py",
    "cnn3d": "scripts/train_supervised_cnn.py",
    "cnn2d": "scripts/train_supervised_cnn2d.py",
}

USAGE = """\
usage: harness.py <command> [options]

  list                          registered metrics / tasks / protocols / methods
  encode --champion C --cache P [--views ...] [--patients ...]
  eval --cache P --task rano4_forecast --view states_forecast \\
       [--protocol locked_unseen|hero_split|path.json] [--cohort unseen] \\
       [--train-pool unseen|train] [--readout linear|mlp|ridge] [--metrics ...] \\
       [--compare VIEW] [--out results.json]
  train <jepa|field|head|lora|cnn3d|cnn2d|readout> [native args ...]
"""


def _log(msg: str) -> None:
    print(msg, flush=True)


def cmd_list(_args) -> None:
    _log("metrics:   " + ", ".join(METRICS.names()))
    _log("tasks:     " + ", ".join(TASKS.names()))
    _log("protocols: " + ", ".join(PROTOCOLS.names()))
    _log("methods:   " + ", ".join(sorted(set(METHODS.names()) | set(LEGACY_TRAIN))))


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
    cache = load_cache(args.cache, warn_provenance=False)
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
        _, d_vals = patient_bootstrap(
            result.oof, other.oof, boot=args.boot, seed=args.seed,
            metric_fn=lambda p, t: _metric_fn(primary, p, t, n_cls))
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


def _metric_fn(name, pred, y, n_cls=4):
    from .eval.metrics import compute_metrics
    # bootstrap passes argmax labels; only label metrics are supported here
    return compute_metrics([name], y, pred, n_cls=n_cls)[name]


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
    if method not in LEGACY_TRAIN:
        raise SystemExit(f"unknown train method {method!r}; "
                         f"known: {sorted(set(METHODS.names()) | set(LEGACY_TRAIN))}")
    script = os.path.join(ROOT, LEGACY_TRAIN[method])
    print(f"[harness] delegating train {method} -> {LEGACY_TRAIN[method]} "
          f"(migrated in a later phase)", flush=True)
    old_argv = sys.argv
    sys.argv = [script, *rest]
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        sys.argv = old_argv


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
    elif cmd == "encode":
        cmd_encode(_encode_parser().parse_args(rest))
    elif cmd == "eval":
        cmd_eval(_eval_parser().parse_args(rest))
    elif cmd == "train":
        cmd_train(argparse.Namespace(rest=rest))
    else:
        raise SystemExit(f"unknown command {cmd!r}\n\n{USAGE}")
