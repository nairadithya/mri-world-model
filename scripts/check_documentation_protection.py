"""Compare protected checkout state read-only; never inspect experiment outputs."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from documentation_sources import PACKAGE, ROOT, sha


def git(repo, *args):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    return subprocess.check_output(["git", *args], cwd=repo, env=env).decode()


def compare(baseline, parent):
    result = []
    for row in baseline["worktrees"]:
        name = row["worktree"]
        if name not in ("world-model", "world-model-r36", "world-model-representations"):
            raise ValueError(f"Unexpected protected checkout {name}")
        tree = parent / name
        changed = []
        for path, digest in row["tracked_sha256"].items():
            target = tree / path
            actual = sha(target.read_bytes()) if target.is_file() else None
            if actual != digest:
                changed.append({"path": path, "baseline_sha256": digest, "current_sha256": actual})
        head = git(tree, "rev-parse", "HEAD").strip()
        status = git(tree, "status", "--short")
        result.append({"worktree": name, "baseline_head": row["head"], "current_head": head,
                       "head_equal": head == row["head"], "status_equal": status == row["status"],
                       "current_status": status, "tracked_files_checked": len(row["tracked_sha256"]),
                       "changed_files": changed})
    return result


def main():
    baseline = json.loads((PACKAGE / "protection_baseline.json").read_text())
    rows = compare(baseline, ROOT.parent)
    catalog = json.loads((PACKAGE / "catalog.json").read_text())
    live_checks = []
    for source in catalog["sources"]:
        for origin in source["origins"]:
            if origin["kind"] == "worktree":
                path = ROOT.parent / origin["worktree"] / source["path"]
                actual = sha(path.read_bytes()) if path.is_file() else None
                live_checks.append({"id": source["id"], "worktree": origin["worktree"],
                                    "matches_capture": actual == source["sha256"]})
    refs = []
    for line in git(ROOT, "for-each-ref", "--format=%(refname) %(objectname) %(symref)", "refs/heads", "refs/remotes").splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] != "refs/heads/docs/portable-research-records":
            refs.append({"ref": fields[0], "commit": fields[1]})
    unchanged = all(row["head_equal"] and row["status_equal"] and not row["changed_files"] for row in rows)
    unchanged = unchanged and all(row["matches_capture"] for row in live_checks)
    report = {"recorded_utc": datetime.now(timezone.utc).isoformat(),
              "status": "UNCHANGED" if unchanged else "CHANGES_REQUIRE_ATTRIBUTION",
              "baseline_utc": baseline["captured_utc"], "worktrees": rows,
              "live_markdown_checks": live_checks, "branch_refs_match_capture": refs == catalog["refs"],
              "current_refs": refs,
              "scope": "Protected checkout HEAD, Git status, baseline tracked-file bytes, and captured live Markdown. Git optional locks disabled. No process controls or experiment-output traversal. Changes may belong to the concurrent owner; this comparison does not attribute them. Shared outputs and process state are not certified by file equality."}
    (PACKAGE / "protection_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "branch_refs_match_capture": report["branch_refs_match_capture"],
                      "tracked_files_checked": sum(row["tracked_files_checked"] for row in rows),
                      "live_markdown_checks": len(live_checks),
                      "changes": [row["worktree"] for row in rows if row["changed_files"] or not row["head_equal"] or not row["status_equal"]]}, indent=2))


if __name__ == "__main__":
    main()
