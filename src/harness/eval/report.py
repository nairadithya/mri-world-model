"""Human-readable report formatters.

The default locked-probe lines are byte-compatible with the legacy
``probe_rano.run_locked`` output so Kaggle verdict regexes and ``info/``
records keep parsing.
"""
from __future__ import annotations

from .evaluator import EvalResult
from ..data.tasks import RANO_PROBE_NAMES


def print_result(result: EvalResult) -> None:
    """Generic one-block report (metric, CI, majority baseline)."""
    primary = next(iter(result.metrics), None)
    if primary is None:
        print(f"{result.name}: no rows")
        return
    ci = result.ci.get(primary)
    ci_str = f" [{ci[0]:.4f},{ci[1]:.4f}]" if ci else ""
    print(f"  pooled {primary} {result.metrics[primary]:.4f}{ci_str}  "
          f"(n_pat={result.n_pat}, n_rows={result.n_rows})")
    if result.majority is not None:
        k = result.majority["class"]
        names = result.meta.get("class_names") or RANO_PROBE_NAMES
        name = names[k] if k < len(names) else str(k)
        print(f"  majority ({name}) acc={result.majority['accuracy']:.4f}")


def print_header(protocol: dict, cohort: str, eval_pool, n_lab: int,
                 train_pool: str, feat: str, hidden) -> None:
    print(f"locked protocol v{protocol['version']}: cohort={cohort} "
          f"{len(eval_pool)} patients, {n_lab} labelled visits; "
          f"train_pool={train_pool}, feat={feat}-"
          f"{'mlp' if hidden else 'linear'}")


def print_provenance(prov: dict | None) -> None:
    import os
    if prov:
        print(f"cache provenance: champion={os.path.basename(prov['champion'])} "
              f"git={str(prov.get('git_sha'))[:10]} date={prov.get('date')}")
    else:
        print("WARNING: cache has no provenance block (pre-2026-09-10 legacy)")


def print_latent(result) -> None:
    """Representation-error table (overall + per-horizon + paired vs persist)."""
    if not result.n_pairs:
        print("representation error: no valid pairs")
        return
    print(f"representation error ({result.n_pairs} pairs, "
          f"{result.n_patients} patients)")
    print(f"  {'method':>12} {'cos_err':>8}")
    for m in result.methods:
        print(f"  {m:>12} {result.overall[m]:>8.4f}")
    for m, (d, lo, hi, sig) in result.pairwise.items():
        ci = "" if lo != lo else f" [{lo:+.4f},{hi:+.4f}]"
        print(f"  vs persistence {m}: {d:+.4f}{ci} {sig}")
    if result.per_horizon:
        ns = sorted({n for n, _ in result.per_horizon})
        print("  " + f"{'n':>4} {'pairs':>6} " +
              " ".join(f"{m:>9}" for m in result.methods))
        for n in ns:
            vals = " ".join(f"{result.per_horizon[(n, m)]:>9.4f}"
                            for m in result.methods)
            print(f"  {n:>4} {result.horizon_counts.get(n, 0):>6} {vals}")
