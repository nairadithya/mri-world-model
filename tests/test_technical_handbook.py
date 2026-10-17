from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build_technical_handbook.py"
PYTHON = sys.executable


class TechnicalHandbookTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        (self.root / "info/handbook/chapters").mkdir(parents=True)
        (self.root / "scripts/documentation_sources.py").write_bytes((ROOT / "scripts/documentation_sources.py").read_bytes())
        (self.root / "scripts/build_technical_handbook.py").write_bytes(BUILDER.read_bytes())
        self.layout = '<!doctype html><nav>{{NAV}}</nav><main>{{CONTENT}}</main><i>{{IDENTITY}}</i><b>{{CHAPTER_COUNT}}</b>'
        self.set_layout()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def set_layout(self) -> None:
        (self.root / "info/handbook/layout.html").write_text(self.layout, encoding="utf-8")

    def chapter(self, filename: str, ident: str, title: str, extra: str = "", body: str = "") -> None:
        (self.root / "info/handbook/chapters" / filename).write_text(
            f'<section class="chapter" id="{ident}" data-part="Primer" {extra}><h2>{title}</h2>{body}</section>', encoding="utf-8"
        )

    def run_builder(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([PYTHON, str(BUILDER), "--root", str(self.root), *args], text=True, capture_output=True)

    def valid_sources(self) -> None:
        self.chapter("01-first.html", "first", "First idea")
        self.chapter("02-second.html", "second", "Second idea", 'data-requires="first"')

    def test_build_navigation_identity_and_check(self) -> None:
        self.valid_sources()
        built = self.run_builder()
        self.assertEqual(built.returncode, 0, built.stderr)
        output_path = self.root / "info/technical_handbook.html"
        first = output_path.read_bytes()
        self.assertIn(b'href="#first"><span class="toc-number">1</span> First idea', first)
        self.assertLess(first.index(b">First idea"), first.index(b">Second idea"))
        self.assertIn(b"<b>2</b>", first)
        self.assertEqual(self.run_builder("--check").returncode, 0)
        self.assertEqual(self.run_builder().returncode, 0)
        self.assertEqual(output_path.read_bytes(), first)

    def test_stale_check_is_read_only(self) -> None:
        self.valid_sources()
        self.assertEqual(self.run_builder().returncode, 0)
        path = self.root / "info/technical_handbook.html"
        path.write_text("stale", encoding="utf-8")
        before = path.read_bytes()
        result = self.run_builder("--check")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(path.read_bytes(), before)

    def test_broken_local_file_and_markdown_fragments(self) -> None:
        self.chapter("01.html", "first", "First", body='<a href="missing.html">bad</a>')
        self.assertNotEqual(self.run_builder().returncode, 0)
        (self.root / "info/guide.md").write_text("# Heading\n", encoding="utf-8")
        (self.root / "info/handbook/chapters/01.html").write_text(
            '<section class="chapter" id="first" data-part="Primer"><h2>First</h2><a href="guide.md#heading">ok</a></section>', encoding="utf-8"
        )
        self.assertEqual(self.run_builder().returncode, 0)
        changed = (self.root / "info/handbook/chapters/01.html").read_text().replace("#heading", "#absent")
        (self.root / "info/handbook/chapters/01.html").write_text(changed)
        self.assertNotEqual(self.run_builder().returncode, 0)

    def test_duplicate_ids_and_forward_prerequisite(self) -> None:
        self.chapter("01.html", "first", "First", body='<svg><g id="first"></g></svg>')
        self.assertNotEqual(self.run_builder().returncode, 0)
        self.chapter("01.html", "first", "First")
        self.chapter("02.html", "second", "Second", 'data-requires="later"')
        self.assertNotEqual(self.run_builder().returncode, 0)

    def test_repository_links_and_html_fragments(self) -> None:
        (self.root / "source.py").write_text("pass\n")
        self.chapter("01.html", "first", "First", body='<a href="../source.py">source</a>')
        self.assertEqual(self.run_builder().returncode, 0)
        (self.root / "info/other.html").write_text('<h1 id="known">Other</h1>')
        self.chapter("01.html", "first", "First", body='<a href="other.html#absent">bad</a>')
        self.assertNotEqual(self.run_builder().returncode, 0)
        self.chapter("01.html", "first", "First", body='<a href="other.html#known">good</a>')
        self.assertEqual(self.run_builder().returncode, 0)
        self.chapter("01.html", "first", "First", body='<a href="../../outside.txt">bad</a>')
        result = self.run_builder()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("link escapes repository", result.stderr)

    def test_navigation_escapes_authored_titles(self) -> None:
        self.chapter("01.html", "first", "A &amp; B")
        self.assertEqual(self.run_builder().returncode, 0)
        output = (self.root / "info/technical_handbook.html").read_text()
        self.assertIn('<span class="toc-number">1</span> A &amp; B</a>', output)

    def test_remote_assets_rejected_external_citation_allowed(self) -> None:
        self.chapter("01.html", "first", "First", body='<a href="https://example.org/paper">citation</a>')
        self.assertEqual(self.run_builder().returncode, 0)
        for body in ('<img src="https://example.org/image.png">',
                     '<svg><image href="https://example.org/image.svg"/></svg>'):
            with self.subTest(body=body):
                self.chapter("01.html", "first", "First", body=body)
                self.assertNotEqual(self.run_builder().returncode, 0)


if __name__ == "__main__":
    unittest.main()
