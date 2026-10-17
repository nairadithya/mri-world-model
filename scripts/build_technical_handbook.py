from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


@dataclass(frozen=True)
class Chapter:
    id: str
    title: str
    part: str
    requires: tuple[str, ...]
    html: str


class HandbookError(ValueError):
    pass


class FragmentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.elements: list[tuple[str, dict[str, str | None]]] = []
        self.stack: list[str] = []
        self.roots: list[tuple[str, dict[str, str | None]]] = []
        self.h2_depth = 0
        self.h2_count = 0
        self.title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        self.elements.append((tag, values))
        if not self.stack:
            self.roots.append((tag, values))
        if "id" in values:
            self.ids.append(values["id"] or "")
        if tag == "h2":
            self.h2_count += 1
            self.h2_depth = len(self.stack) + 1
        if tag not in {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == "h2" and self.h2_depth:
            self.h2_depth = 0
        if tag in self.stack:
            self.stack = self.stack[: len(self.stack) - 1 - self.stack[::-1].index(tag)]

    def handle_data(self, data: str) -> None:
        if self.h2_depth:
            self.title_parts.append(data)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)


def _parse_chapters(root: Path) -> list[Chapter]:
    paths = sorted((root / "info/handbook/chapters").glob("*.html"))
    chapters: list[Chapter] = []
    seen: set[str] = set()
    for path in paths:
        source = path.read_text(encoding="utf-8")
        parser = FragmentParser()
        parser.feed(source)
        if len(parser.roots) != 1:
            raise HandbookError(f"{path}: expected exactly one chapter root")
        tag, attrs = parser.roots[0]
        classes = (attrs.get("class") or "").split()
        chapter_id, part = attrs.get("id"), attrs.get("data-part")
        if tag != "section" or "chapter" not in classes or not chapter_id or not part:
            raise HandbookError(f"{path}: malformed chapter root metadata")
        if chapter_id in seen:
            raise HandbookError(f"{path}: duplicate chapter id {chapter_id}")
        seen.add(chapter_id)
        if parser.h2_count != 1 or not "".join(parser.title_parts).strip():
            raise HandbookError(f"{path}: expected exactly one titled h2")
        requires = tuple((attrs.get("data-requires") or "").split())
        chapters.append(Chapter(chapter_id, " ".join("".join(parser.title_parts).split()), part, requires, source))
    if not chapters:
        raise HandbookError("no handbook chapter sources found")
    prior: set[str] = set()
    for chapter in chapters:
        if len(chapter.requires) != len(set(chapter.requires)):
            raise HandbookError(f"{chapter.id}: duplicate prerequisite")
        missing = [item for item in chapter.requires if item not in prior]
        if missing:
            raise HandbookError(f"{chapter.id}: missing or forward prerequisite {missing[0]}")
        prior.add(chapter.id)
    return chapters


def _navigation(chapters: list[Chapter]) -> str:
    parts: list[tuple[str, list[tuple[int, Chapter]]]] = []
    for number, chapter in enumerate(chapters, 1):
        if not parts or parts[-1][0] != chapter.part:
            parts.append((chapter.part, []))
        parts[-1][1].append((number, chapter))
    groups = []
    for part, entries in parts:
        items = "".join(
            f'<li><a href="#{chapter.id}"><span class="toc-number">{number}</span> {chapter.title}</a></li>'
            for number, chapter in entries
        )
        groups.append(f'<section class="toc-part"><h2>{part}</h2><ol class="toc-list">{items}</ol></section>')
    return '<nav aria-label="Contents">' + "".join(groups) + "</nav>"


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.links: list[tuple[str, str]] = []

    def handle_data(self, data: str) -> None:
        if re.search(r"(?:url\s*\(\s*['\"]?|@import\s+['\"]?)(?:https?:)?//", data, re.I):
            raise HandbookError("remote styles or fonts are forbidden")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        for key, value in attrs:
            if key == "style" and value and re.search(r"(?:url\s*\(\s*['\"]?|@import\s+['\"]?)(?:https?:)?//", value, re.I):
                raise HandbookError("remote styles or fonts are forbidden")
        if values.get("id") is not None:
            self.ids.append(values["id"] or "")
        for key in ("href", "src", "action", "poster"):
            value = values.get(key)
            if value:
                if key != "href" and urlsplit(value).scheme in {"http", "https"}:
                    raise HandbookError(f"remote asset is forbidden: {value}")
                self.links.append((key, value))
        if tag in {"script", "style", "link"}:
            value = values.get("src") or values.get("href") or ""
            if value and (urlsplit(value).scheme in {"http", "https"} or value.startswith("//")):
                raise HandbookError(f"remote executable/style resource is forbidden: {value}")


def _validate_links(root: Path, rendered: str) -> None:
    parser = LinkParser()
    parser.feed(rendered)
    if len(parser.ids) != len(set(parser.ids)):
        raise HandbookError("duplicate id in rendered handbook")
    ids = set(parser.ids)
    for _, raw in parser.links:
        parsed = urlsplit(raw)
        if parsed.scheme in {"javascript", "data", "file"} or parsed.netloc and parsed.scheme not in {"http", "https"}:
            raise HandbookError(f"unsafe link target: {raw}")
        if parsed.scheme in {"http", "https", "mailto", "tel"}:
            continue
        if parsed.scheme:
            raise HandbookError(f"unsupported link scheme: {raw}")
        if raw.startswith("//"):
            raise HandbookError(f"remote asset or link is forbidden: {raw}")
        fragment = unquote(parsed.fragment)
        if parsed.path:
            target = (root / "info" / unquote(parsed.path)).resolve()
            base = (root / "info").resolve()
            if not target.is_relative_to(base):
                raise HandbookError(f"link escapes repository documentation: {raw}")
            if not target.is_file():
                raise HandbookError(f"missing local link target: {raw}")
            if fragment and target.suffix.lower() in {".md", ".markdown"}:
                from documentation_sources import heading_ids

                headings = heading_ids(target.read_text(encoding="utf-8"))
                if fragment not in headings:
                    raise HandbookError(f"missing Markdown fragment: {raw}")
            elif fragment and target.resolve() == (root / "info/technical_handbook.html").resolve() and fragment not in ids:
                raise HandbookError(f"missing HTML fragment: {raw}")
        elif fragment and fragment not in ids:
            raise HandbookError(f"missing HTML fragment: {raw}")


def render(root: Path) -> bytes:
    chapters = _parse_chapters(root)
    layout_path = root / "info/handbook/layout.html"
    layout = layout_path.read_bytes()
    layout_text = layout.decode("utf-8")
    tokens = ("{{NAV}}", "{{CONTENT}}", "{{IDENTITY}}", "{{CHAPTER_COUNT}}")
    if any(layout_text.count(token) != 1 for token in tokens):
        raise HandbookError("layout must contain each template slot exactly once")
    digest = hashlib.sha256()
    digest.update(Path(__file__).read_bytes())
    digest.update(layout)
    for path in sorted((root / "info/handbook/chapters").glob("*.html")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    output = layout_text.replace("{{NAV}}", _navigation(chapters))
    output = output.replace("{{CONTENT}}", "\n".join(chapter.html for chapter in chapters))
    output = output.replace("{{IDENTITY}}", digest.hexdigest())
    output = output.replace("{{CHAPTER_COUNT}}", str(len(chapters)))
    if re.search(r"\{\{[A-Z_]+\}\}", output):
        raise HandbookError("unresolved template slot")
    _validate_links(root, output)
    return output.encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    output = args.root / "info/technical_handbook.html"
    try:
        expected = render(args.root.resolve())
        if args.check:
            if not output.is_file() or output.read_bytes() != expected:
                raise HandbookError("generated handbook is stale; rebuild it")
            print("Technical handbook: PASS")
        else:
            output.write_bytes(expected)
            print(f"Wrote {output}")
        return 0
    except (HandbookError, OSError, UnicodeError) as exc:
        print(f"Technical handbook: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
