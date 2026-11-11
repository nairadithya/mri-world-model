"""Bind research draft links to selected frozen sources without changing source bytes."""

import argparse
import json
import os
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from documentation_sources import DRAFTS, NAMES, PACKAGE, ROOT

SELECTED = {
    "TODO.md": "S016",
    "info/ablations.md": "S020",
    "info/decisions.md": "S026",
    "info/directions.md": "S035",
    "info/eval_protocol.md": "S041",
    "info/frontier_lumiere.md": "S043",
    "info/methodology_validity.md": "S047",
    "info/p0_audit.md": "S051",
    "info/p2_lesion_architecture.md": "S058",
    "info/r36_delta_representation_results.md": "S064",
    "info/results.md": "S068",
}
SPECIAL = {
    "../../../world-model-representations/info/r35_representation_comparison.md": "S062",
    "../../../world-model-r36/info/r36_delta_representation_results.md": "S065",
}
SUPPLEMENTARY = {
    "/tmp/world-model-doc-draft/r36-snapshot-evidence.json": "info/documentation_sources/supplementary/r36-runtime-snapshot.json",
    "/home/adithya/.pi/agent/sessions/--home-adithya-projects-world-model--/subagent-artifacts/outputs/cdb58c14-6da4-4b99-a19e-1b44828a69f6/review/r36-protocol-recovery.md": "info/documentation_sources/supplementary/r36-protocol-recovery.md",
}


def selected_sources(catalog):
    rows = {row["id"]: row for row in catalog["sources"]}
    selected = {}
    for row in catalog["sources"]:
        if row["role"] in ("research", "retained-procedure"):
            if row["path"] in selected and row["path"] not in SELECTED:
                raise ValueError(f"Unspecified source priority for {row['path']}")
            selected[row["path"]] = row
    for path, source_id in SELECTED.items():
        if rows[source_id]["path"] != path:
            raise ValueError(f"Source priority mismatch for {source_id}")
        selected[path] = rows[source_id]
    return rows, selected


def portable_target(raw, path, root, rows, selected):
    url = urlsplit(raw)
    if url.path in SUPPLEMENTARY:
        target = SUPPLEMENTARY[url.path]
    elif url.path in SPECIAL:
        target = rows[SPECIAL[url.path]]["snapshot"]
    elif url.scheme or url.netloc or raw.startswith("/"):
        return raw
    else:
        resolved = (path.parent / unquote(url.path)).resolve()
        if not resolved.is_relative_to(root.resolve()):
            return raw
        original = resolved.relative_to(root.resolve()).as_posix()
        if original not in selected:
            return raw
        target = selected[original]["snapshot"]
    relative = Path(os.path.relpath(root / target, path.parent)).as_posix()
    return relative + ("#" + url.fragment if url.fragment else "")


def rewrite(text, path, root, rows, selected):
    def inline(match):
        return match[1] + portable_target(match[2], path, root, rows, selected)

    text = re.sub(r"(\]\()([^\s)]+)", inline, text)
    return re.sub(r"(^\[[^\]]+\]:\s*)(\S+)", inline, text, flags=re.M)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    catalog = json.loads((PACKAGE / "catalog.json").read_text())
    rows, selected = selected_sources(catalog)
    changes = []
    for name in NAMES:
        path = DRAFTS / name
        old = path.read_text()
        new = rewrite(old, path, ROOT, rows, selected)
        if new != old:
            changes.append(name)
            if args.apply:
                path.write_text(new)
    print(json.dumps({"changed_drafts": changes, "applied": args.apply,
                      "scope": "Research-source targets only. No source snapshot or protected worktree writes."}, indent=2))
    if changes and not args.apply:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
