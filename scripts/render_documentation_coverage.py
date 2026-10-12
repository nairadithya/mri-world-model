"""Render the curated per-source ledger without inferring semantic coverage."""

import json
import os
from pathlib import Path

from documentation_sources import DRAFTS, PACKAGE, ROOT


def render(package=PACKAGE, drafts=DRAFTS, root=ROOT):
    catalog = json.loads((package / "catalog.json").read_text())
    ledger = json.loads((package / "coverage.json").read_text())
    sources = {row["id"]: row for row in catalog["sources"]}
    if len(ledger) != len(sources) or {row["id"] for row in ledger} != set(sources):
        raise ValueError("Coverage must name every source exactly once")
    prefix = Path(os.path.relpath(package, drafts)).as_posix()
    text = ["# Source coverage", "",
            "This ledger covers all 72 captured Markdown versions across 48 paths, ten branch refs, and three preexisting worktrees. It records parent comparisons, not scientific acceptance. The catalog fixes the capture scope and origins. No network fetch or research job was run.", "",
            "The notes and experiments child reports were incomplete. The parent completed direct full reads or complete variant diffs before recording these dispositions. The contracts report was checked against frozen sources and its omissions were integrated. The first reviewer assessed the inventory only and returned BLOCKED before integration. A later independent review is a separate gate.", "",
            "Archived sources retain their exact bytes, including historical claims and embedded links that may no longer work. The consolidated research documents give the interpretation. Operational guides and CLI skills remain separate and discoverable.", "",
            "Token extraction in the catalog is a search aid, not a global identifier ontology. A42, A43, and A44 in S065 denote arm A at seeds 42, 43, and 44. They are not new ablation entries. D183 in S042 is part of a DOI, not decision D183.", "",
            "## Per-source dispositions", "",
            "| Source | Original path | Read mode | Destinations | Disposition |", "|---|---|---|---|---|"]
    for row in sorted(ledger, key=lambda row: row["id"]):
        source = sources[row["id"]]
        link = prefix + "/versions/" + Path(source["snapshot"]).name
        dest = ", ".join(f"[{name}]({name})" for name in row["destinations"])
        text.append(f"| [{row['id']}]({link}) | `{source['path']}` | {row['read_mode']} | {dest} | {row['disposition']} |")
    text += ["", "## Evidence and origins", "",
             "The [curated JSON ledger](../documentation_sources/coverage.json) records exact frozen source ranges, draft sections, read modes, and the person responsible. The [catalog](../documentation_sources/catalog.json) records every branch commit, blob, and live-worktree origin. The [supplementary manifest](../documentation_sources/supplementary/manifest.json) binds the recovered runtime receipt, protocol review, and worktree-creation evidence.", "",
             "Integrated comparisons are [notes](../documentation_sources/comparisons/notes.json), [experiments](../documentation_sources/comparisons/experiments.json), and [contracts](../documentation_sources/comparisons/contracts.json). Received child reports remain separately archived with their original limitations.", ""]
    (drafts / "SOURCE_COVERAGE.md").write_text("\n".join(text).replace("../documentation_sources/", prefix + "/"))
    slices = json.loads((package / "slices.json").read_text())
    output = package / "comparisons"
    output.mkdir(exist_ok=True)
    for name in ("notes", "experiments", "contracts"):
        ids = set(slices[name])
        report = {"slice": name, "coverage": [row for row in ledger if row["id"] in ids],
                  "missing_material": [], "unread": [], "verdict": "INTEGRATED_FOR_REVIEW",
                  "scope": "Parent dispositions and integration. Not independent semantic review or scientific acceptance."}
        (output / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    render()
