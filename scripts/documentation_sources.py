"""Capture branch documentation and verify a portable research draft package."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "info/documentation_sources"
DRAFTS = ROOT / "info"
NAMES = ("TODO.md", "EXPERIMENTS.md", "METHODS.md", "NOTES_AND_INSIGHTS.md")


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def refs(repo):
    result = []
    for line in git(repo, "for-each-ref", "--format=%(refname) %(objectname) %(symref)", "refs/heads", "refs/remotes").decode().splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] != "refs/heads/docs/portable-research-records":
            result.append({"ref": fields[0], "commit": fields[1]})
    return result


def branch_markdown(repo, ref):
    for entry in git(repo, "ls-tree", "-rz", ref).split(b"\0"):
        if not entry:
            continue
        desc, path = entry.split(b"\t", 1)
        mode, kind, oid = desc.decode().split()
        name = path.decode()
        if kind == "blob" and name.lower().endswith(".md"):
            yield name, oid


def live_markdown(repo):
    paths = git(repo, "ls-files", "-co", "--exclude-standard", "-z").decode().split("\0")
    return sorted({p for p in paths if p.lower().endswith(".md") and (Path(repo) / p).is_file()})


def role(path):
    if path.startswith("info/drafts/"):
        return "draft-baseline", list(NAMES)
    if path == "AGENTS.md" or path.startswith(".agents/") or path.startswith(".pi/"):
        return "retained-agent-contract", ["METHODS.md"]
    if path in ("README.md", "info/README.md", "scripts/README.md") or path.startswith(".audit/"):
        return "retained-operations", ["TODO.md", "METHODS.md"]
    if path in ("info/preprocessing.md", "src/preprocessing/reprocess_sailor.md"):
        return "retained-procedure", ["METHODS.md", "EXPERIMENTS.md"]
    if path in ("TODO.md", "info/r36_delta_representation_handoff.md", "info/action_conditioned_representation_brief.md"):
        return "research", ["TODO.md", "METHODS.md", "NOTES_AND_INSIGHTS.md"]
    if path in ("info/ablations.md", "info/results.md", "info/pilot.md", "info/clean_rescues.md", "info/p0_refresh.md", "info/p1_baselines.md", "info/p1_anatomy_baselines.md", "info/p2_seed_sensitivity.md", "info/r35_representation_comparison.md", "info/r36_delta_representation_results.md"):
        return "research", ["EXPERIMENTS.md", "NOTES_AND_INSIGHTS.md"]
    return "research", ["METHODS.md", "NOTES_AND_INSIGHTS.md"]


def capture(repo, worktrees):
    if (PACKAGE / "catalog.json").exists():
        raise ValueError("Refuse to overwrite a frozen catalog")
    versions = {}
    branch_refs = refs(repo)
    for item in branch_refs:
        for path, oid in branch_markdown(repo, item["ref"]):
            data = git(repo, "cat-file", "blob", oid)
            key = (path, sha(data))
            versions.setdefault(key, {"data": data, "origins": []})["origins"].append({"kind": "branch", **item, "blob": oid})
    preserved = []
    for tree in worktrees:
        tree = Path(tree).resolve()
        commit = git(tree, "rev-parse", "HEAD").decode().strip()
        branch = git(tree, "branch", "--show-current").decode().strip()
        live = {}
        for path in live_markdown(tree):
            data = (tree / path).read_bytes()
            key = (path, sha(data))
            versions.setdefault(key, {"data": data, "origins": []})["origins"].append({"kind": "worktree", "worktree": tree.name, "branch": branch, "commit": commit})
            live[path] = sha(data)
        tracked = [p for p in git(tree, "ls-files", "-z").decode().split("\0") if p.lower().endswith(".md")]
        preserved.append({"worktree": tree.name, "commit": commit, "branch": branch, "markdown_sha256": live, "absent_tracked_markdown": [p for p in tracked if not (tree / p).exists()]})
    rows = []
    for number, ((path, digest), version) in enumerate(sorted(versions.items()), 1):
        content = version.pop("data")
        text = content.decode()
        target = PACKAGE / "versions" / (digest + ".md")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != content:
            raise ValueError("Snapshot collision")
        target.write_bytes(content)
        source_role, destinations = role(path)
        rows.append({"id": f"S{number:03}", "path": path, "sha256": digest, "snapshot": str(target.relative_to(ROOT)), "origins": version["origins"], "role": source_role, "destinations": destinations, "headings": re.findall(r"^#{1,6}\s+(.+)$", text, re.M), "identifiers": sorted(set(re.findall(r"\b(?:D\d+b?|A\d+|I[1-4]|R\d+|G\d+|K3-\d+|V[1-5]|E[0-4])\b", text)))})
    result = {"schema": 1, "captured_utc": datetime.now(timezone.utc).isoformat(), "scope": "All existing local and remote-tracking branch tips and all three preexisting live worktrees. No network fetch. Ignored datasets, checkpoints, outputs, dependencies, and session artifacts are outside the Markdown corpus.", "refs": branch_refs, "worktrees": preserved, "sources": rows}
    (PACKAGE / "catalog.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"refs": len(branch_refs), "versions": len(rows), "paths": len({r['path'] for r in rows}), "unique_bytes": len({r['sha256'] for r in rows})}))


def heading_ids(text):
    ids = set(re.findall(r'<a\s+id=[\"\']([^\"\']+)', text))
    used = {}
    for heading in re.findall(r"^#{1,6}\s+(.+)$", text, re.M):
        heading = re.sub(r"<[^>]+>", "", heading).strip().rstrip("#").strip().lower()
        anchor = re.sub(r"[^\w\- ]", "", heading).replace(" ", "-")
        count = used.get(anchor, 0)
        used[anchor] = count + 1
        ids.add(anchor if count == 0 else f"{anchor}-{count}")
    return ids


def check_links(path, root):
    errors = []
    text = path.read_text()
    targets = re.findall(r"(?<!!)\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)", text)
    targets += re.findall(r"^\[[^\]]+\]:\s*(\S+)", text, re.M)
    for raw in targets:
        url = urlsplit(raw)
        if url.scheme in ("https", "http", "mailto"):
            continue
        if url.scheme or url.netloc or raw.startswith("/"):
            errors.append(f"{path.name}: machine-local target {raw}")
            continue
        target = (path.parent / unquote(url.path)).resolve() if url.path else path.resolve()
        if not target.is_relative_to(root.resolve()):
            errors.append(f"{path.name}: target outside repository {raw}")
        elif not target.is_file():
            errors.append(f"{path.name}: missing target {raw}")
        elif url.fragment and target.suffix == ".md" and unquote(url.fragment) not in heading_ids(target.read_text()):
            errors.append(f"{path.name}: missing fragment {raw}")
    if re.search(r"/tmp/|/home/adithya/|\.pi/agent/sessions/|\.\./\.\./\.\./world-model", text):
        errors.append(f"{path.name}: temporary or machine-local provenance remains")
    return len(targets), errors


def verify(package=PACKAGE, drafts=DRAFTS, root=ROOT):
    catalog = json.loads((package / "catalog.json").read_text())
    errors = []
    coverage_path = package / "coverage.json"
    if coverage_path.exists():
        coverage = json.loads(coverage_path.read_text())
    else:
        coverage = []
        errors.append("Missing coverage.json")
    sources = {r["id"]: r for r in catalog["sources"]}
    source_ids = set(sources)
    coverage_ids = [r["id"] for r in coverage]
    if set(coverage_ids) != source_ids or len(coverage_ids) != len(source_ids):
        errors.append("Coverage is not a one-to-one mapping of all source versions")
    for row in catalog["sources"]:
        source = root / row["snapshot"]
        if not source.exists() or sha(source.read_bytes()) != row["sha256"]:
            errors.append(f"{row['id']}: source bytes changed or missing")
    for row in coverage:
        if not row.get("disposition") or not row.get("checked_by") or not row.get("evidence"):
            errors.append(f"{row['id']}: missing checked disposition or evidence")
        status = row.get("comparison_status")
        if status not in ("checked", "unread", "missing_material"):
            errors.append(f"{row['id']}: missing or invalid comparison_status")
        elif status != "checked":
            errors.append(f"{row['id']}: comparison incomplete ({status})")
        if not row.get("read_mode"):
            errors.append(f"{row['id']}: missing read_mode")
        destinations = row.get("destinations", [])
        for destination in destinations:
            if destination not in NAMES:
                errors.append(f"{row['id']}: unexpected destination {destination}")
        if len(destinations) != len(set(destinations)):
            errors.append(f"{row['id']}: duplicate destinations")
        source = sources.get(row["id"])
        if source and set(destinations) != set(source["destinations"]) and not row.get("destination_override"):
            errors.append(f"{row['id']}: destinations differ from catalog without a destination_override rationale")
    supplementary_count = 0
    supplementary = package / "supplementary"
    manifest = supplementary / "manifest.json"
    if supplementary.is_dir():
        if not manifest.is_file():
            errors.append("Missing supplementary manifest")
        else:
            for item in json.loads(manifest.read_text()):
                target = (root / item["path"]).resolve()
                supplementary_count += 1
                if not target.is_relative_to(supplementary.resolve()):
                    errors.append(f"Supplementary target outside evidence directory: {item['path']}")
                elif not target.is_file() or sha(target.read_bytes()) != item["sha256"]:
                    errors.append(f"Supplementary bytes changed or missing: {item['path']}")
    archive_count = 0
    archive_root = root / "info/archive"
    if archive_root.is_dir():
        for directory in sorted(path for path in archive_root.iterdir() if path.is_dir()):
            manifest = directory / "manifest.json"
            if not manifest.is_file():
                errors.append(f"Missing archive manifest: {directory.name}")
                continue
            records = json.loads(manifest.read_text())["records"]
            originals = [row["original_path"] for row in records]
            if len(originals) != len(set(originals)):
                errors.append(f"{manifest.parent.name}: duplicate archive origins")
            for row in records:
                target = (root / row["archive_path"]).resolve()
                archive_count += 1
                if not target.is_relative_to(manifest.parent.resolve()):
                    errors.append(f"Archive target outside archive: {row['archive_path']}")
                elif not target.is_file() or sha(target.read_bytes()) != row["sha256"]:
                    errors.append(f"Archive bytes changed or missing: {row['archive_path']}")
    count = 0
    combined = ""
    draft_hashes = {}
    for name in NAMES:
        path = drafts / name
        if not path.is_file():
            errors.append(f"Missing draft {name}")
            continue
        text = path.read_text()
        combined += text + "\n"
        draft_hashes[name] = sha(path.read_bytes())
        links, failures = check_links(path, root)
        count += links
        errors.extend(failures)
        if len(re.findall(r"^# ", text, re.M)) != 1:
            errors.append(f"{name}: expected one title")
    if not (drafts / "SOURCE_COVERAGE.md").exists():
        errors.append("Missing human-readable source coverage")
    else:
        links, failures = check_links(drafts / "SOURCE_COVERAGE.md", root)
        count += links
        errors.extend(failures)
    baselines = [r for r in catalog["sources"] if r["path"] in {f"info/drafts/{name}" for name in NAMES}]
    for name in NAMES:
        if not any(r["path"] == f"info/drafts/{name}" for r in baselines):
            errors.append(f"Missing frozen draft baseline {name}")
    for baseline in baselines:
        for identifier in baseline["identifiers"]:
            if not re.search(r"\b" + re.escape(identifier) + r"\b", combined):
                errors.append(f"{baseline['id']}: lost baseline identifier {identifier}")
    report = {"status": "FAIL" if errors else "PASS", "source_versions": len(source_ids), "branch_refs": len(catalog["refs"]), "local_links_checked": count, "supplementary_files_checked": supplementary_count, "archive_records_checked": archive_count, "draft_sha256": draft_hashes, "errors": errors, "scope": "Frozen source, supplementary and archive bytes, checked coverage records, baseline identifier presence, and draft link portability. This is not scientific acceptance or a semantic completeness proof."}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("capture", "verify"))
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--worktrees", type=Path, nargs="*")
    args = parser.parse_args()
    if args.command == "capture":
        capture(args.repo, args.worktrees)
    else:
        report = verify()
        (PACKAGE / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        raise SystemExit(bool(report["errors"]))


if __name__ == "__main__":
    main()
