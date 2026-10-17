import json
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from documentation_sources import NAMES, sha, verify


class DocumentationSourcesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root / "info/documentation_sources"
        self.drafts = self.root / "info/drafts"
        self.package.mkdir(parents=True)
        self.drafts.mkdir(parents=True)
        self.sources = []
        self.coverage = []
        for number, name in enumerate(NAMES, 1):
            text = f"# {name}\n\nA{number}\n"
            (self.drafts / name).write_text(text)
            self.add_source(number, f"info/drafts/{name}", text, list(NAMES), [f"A{number}"])
        self.add_source(5, "info/research.md", "# Source\n\n## Evidence\n", ["METHODS.md"], [])
        (self.drafts / "SOURCE_COVERAGE.md").write_text("# Source coverage\n")
        self.save()

    def add_source(self, number, path, text, destinations, identifiers):
        digest = sha(text.encode())
        snapshot = f"info/documentation_sources/versions/{digest}.md"
        target = self.root / snapshot
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        source_id = f"S{number:03}"
        self.sources.append({"id": source_id, "path": path, "snapshot": snapshot,
                             "sha256": digest, "destinations": destinations,
                             "identifiers": identifiers})
        self.coverage.append({"id": source_id, "disposition": "Compared and included.",
                              "checked_by": "fixture", "evidence": ["Source lines 1-3."],
                              "read_mode": "full read", "comparison_status": "checked",
                              "destinations": destinations[:]})

    def save(self):
        catalog = {"sources": self.sources, "refs": []}
        (self.package / "catalog.json").write_text(json.dumps(catalog))
        (self.package / "coverage.json").write_text(json.dumps(self.coverage))

    def report(self):
        return verify(self.package, self.drafts, self.root)

    def assert_failure(self, message):
        report = self.report()
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any(message in error for error in report["errors"]), report)

    def test_complete_fixture_passes(self):
        self.assertEqual(self.report()["errors"], [])

    def test_cli_verify_prints_report_without_writing_receipt(self):
        import shutil
        script_dir = self.root / "scripts"
        script_dir.mkdir()
        script = script_dir / "documentation_sources.py"
        shutil.copy(ROOT / "scripts/documentation_sources.py", script)
        for name in (*NAMES, "SOURCE_COVERAGE.md"):
            shutil.copy(self.drafts / name, self.root / "info" / name)
        package = self.root / "info/documentation_sources"
        result = subprocess.run([sys.executable, "-B", str(script), "verify"],
                                capture_output=True, text=True)
        report = json.loads(result.stdout)
        receipt = package / "verification.json"
        self.assertEqual(result.returncode, 0, report)
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(receipt.exists())

    def test_cli_verify_failure_preserves_existing_receipt(self):
        import shutil
        script_dir = self.root / "scripts"
        script_dir.mkdir()
        script = script_dir / "documentation_sources.py"
        shutil.copy(ROOT / "scripts/documentation_sources.py", script)
        for name in (*NAMES, "SOURCE_COVERAGE.md"):
            shutil.copy(self.drafts / name, self.root / "info" / name)
        receipt = self.package / "verification.json"
        sentinel = b"historical receipt\\x00bytes\\n"
        receipt.write_bytes(sentinel)
        (self.root / "info/TODO.md").unlink()
        result = subprocess.run([sys.executable, "-B", str(script), "verify"],
                                capture_output=True, text=True)
        report = json.loads(result.stdout)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(receipt.read_bytes(), sentinel)

    def test_renderer_keeps_later_updates_separate_and_is_idempotent(self):
        from render_documentation_coverage import render
        (self.package / "slices.json").write_text(json.dumps({"notes": [], "experiments": [], "contracts": []}))
        catalog = (self.package / "catalog.json").read_bytes()
        update = self.package / "updates/2026-10-13/coverage.json"
        update.parent.mkdir(parents=True)
        update.write_text(json.dumps({"records": []}))
        for destination, link in ((self.drafts, "../documentation_sources/updates/2026-10-13/coverage.json"),
                                  (self.root / "info", "documentation_sources/updates/2026-10-13/coverage.json")):
            with self.subTest(destination=destination):
                render(self.package, destination, self.root)
                page = destination / "SOURCE_COVERAGE.md"
                self.assertIn(f"[2026-10-13 source dispositions]({link})", page.read_text())
                self.assertIn("separate from the original frozen capture", page.read_text())
                first = page.read_bytes()
                render(self.package, destination, self.root)
                self.assertEqual(page.read_bytes(), first)
        self.assertEqual((self.package / "catalog.json").read_bytes(), catalog)

    def test_missing_coverage_file_fails_without_crashing(self):
        (self.package / "coverage.json").unlink()
        self.assert_failure("Missing coverage.json")

    def test_missing_and_duplicate_source_coverage_fail(self):
        self.coverage[-1] = self.coverage[0].copy()
        self.save()
        self.assert_failure("one-to-one mapping")

    def test_missing_or_modified_snapshot_fails(self):
        target = self.root / self.sources[-1]["snapshot"]
        target.unlink()
        self.assert_failure("source bytes changed or missing")
        target.write_text("# Changed\n")
        self.assert_failure("source bytes changed or missing")

    def test_all_four_baseline_identifiers_are_checked(self):
        for number, name in enumerate(NAMES, 1):
            with self.subTest(name=name):
                path = self.drafts / name
                old = path.read_text()
                path.write_text(f"# {name}\n")
                self.assert_failure(f"lost baseline identifier A{number}")
                path.write_text(old)

    def test_missing_baseline_fails(self):
        self.sources.pop(0)
        self.coverage.pop(0)
        self.save()
        self.assert_failure("Missing frozen draft baseline TODO.md")

    def test_destinations_must_match_catalog_or_explain_override(self):
        self.coverage[-1]["destinations"] = ["TODO.md"]
        self.save()
        self.assert_failure("destinations differ from catalog")
        self.coverage[-1]["destination_override"] = "The task decision belongs in TODO.md."
        self.save()
        self.assertEqual(self.report()["errors"], [])

    def test_incomplete_comparisons_cannot_pass(self):
        for status in (None, "unread", "missing_material"):
            with self.subTest(status=status):
                self.coverage[-1]["comparison_status"] = status
                self.save()
                self.assert_failure("comparison")

    def test_read_mode_and_evidence_are_required(self):
        for field, error in (("read_mode", "missing read_mode"), ("evidence", "missing checked disposition")):
            with self.subTest(field=field):
                old = self.coverage[-1].pop(field)
                self.save()
                self.assert_failure(error)
                self.coverage[-1][field] = old

    def test_broken_targets_and_fragments_fail(self):
        path = self.drafts / "METHODS.md"
        old = path.read_text()
        for link, error in (("missing.md", "missing target"),
                            ("TODO.md#absent", "missing fragment")):
            with self.subTest(link=link):
                path.write_text(old + f"\n[Source]({link})\n")
                self.assert_failure(error)
        path.write_text(old + "\n[Source](TODO.md#todomd)\n")
        self.assertEqual(self.report()["errors"], [])

    def test_machine_local_and_sibling_targets_fail(self):
        path = self.drafts / "TODO.md"
        old = path.read_text()
        for link in ("file:///tmp/report.md", "/home/adithya/report.md",
                     "../../../world-model-r36/info/results.md"):
            with self.subTest(link=link):
                path.write_text(old + f"\n[Source]({link})\n")
                self.assert_failure("target")

    def test_missing_draft_fails_without_crashing(self):
        (self.drafts / "EXPERIMENTS.md").unlink()
        self.assert_failure("Missing draft EXPERIMENTS.md")

    def add_supplement(self):
        directory = self.package / "supplementary"
        directory.mkdir()
        target = directory / "receipt.json"
        target.write_text('{"status": "historical"}\n')
        manifest = directory / "manifest.json"
        manifest.write_text(json.dumps([{"path": str(target.relative_to(self.root)),
                                         "sha256": sha(target.read_bytes())}]))
        return target, manifest

    def test_supplementary_bytes_are_verified(self):
        target, _ = self.add_supplement()
        self.assertEqual(self.report()["supplementary_files_checked"], 1)
        self.assertEqual(self.report()["errors"], [])
        target.write_text("changed\n")
        self.assert_failure("Supplementary bytes changed or missing")
        target.unlink()
        self.assert_failure("Supplementary bytes changed or missing")

    def test_missing_supplementary_manifest_fails(self):
        _, manifest = self.add_supplement()
        manifest.unlink()
        self.assert_failure("Missing supplementary manifest")

    def test_supplementary_path_must_stay_in_evidence_directory(self):
        _, manifest = self.add_supplement()
        source = self.sources[-1]
        manifest.write_text(json.dumps([{"path": source["snapshot"], "sha256": source["sha256"]}]))
        self.assert_failure("Supplementary target outside evidence directory")

    def test_portable_links_follow_the_document_location(self):
        from portable_documentation_links import portable_target
        source = self.sources[-1]
        selected = {"info/research.md": source}
        for parent, original, prefix in ((self.drafts, "../research.md", "../"),
                                         (self.root / "info", "research.md", "")):
            with self.subTest(parent=parent):
                actual = portable_target(original + "#evidence", parent / "METHODS.md",
                                         self.root, {}, selected)
                expected = prefix + "documentation_sources/versions/" + Path(source["snapshot"]).name + "#evidence"
                self.assertEqual(actual, expected)

    def test_archived_originals_are_verified(self):
        archive = self.root / "info/archive/2026-10-12"
        archive.mkdir(parents=True)
        target = archive / "old.md"
        target.write_text("# Original\n")
        row = {"original_path": "old.md", "archive_path": str(target.relative_to(self.root)),
               "sha256": sha(target.read_bytes())}
        manifest = archive / "manifest.json"
        manifest.write_text(json.dumps({"records": [row]}))
        self.assertEqual(self.report()["archive_records_checked"], 1)
        self.assertEqual(self.report()["errors"], [])
        target.write_text("changed\n")
        self.assert_failure("Archive bytes changed or missing")
        manifest.write_text(json.dumps({"records": [row, row]}))
        self.assert_failure("duplicate archive origins")
        row["archive_path"] = self.sources[-1]["snapshot"]
        manifest.write_text(json.dumps({"records": [row]}))
        self.assert_failure("Archive target outside archive")
        manifest.unlink()
        self.assert_failure("Missing archive manifest")

    def test_missing_human_coverage_fails(self):
        (self.drafts / "SOURCE_COVERAGE.md").unlink()
        self.assert_failure("Missing human-readable source coverage")


if __name__ == "__main__":
    unittest.main()
