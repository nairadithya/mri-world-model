from html.parser import HTMLParser
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit


class ExplainerAudit(HTMLParser):
    def __init__(self):
        super().__init__()
        self.experiment_ids = set()
        self.local_links = []
        self.anchor_depth = 0
        self.errors = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a":
            self.anchor_depth += 1
            if self.anchor_depth > 1:
                self.errors.append("nested anchor")
            href = attrs.get("href", "")
            if href and not href.startswith(("http:", "https:", "#", "mailto:")):
                self.local_links.append(href)
        if tag == "td" and "id" in attrs.get("class", "").split():
            self._reading_id = True

    def handle_endtag(self, tag):
        if tag == "a":
            self.anchor_depth -= 1
        if tag == "td":
            self._reading_id = False

    def handle_data(self, data):
        if getattr(self, "_reading_id", False):
            match = re.fullmatch(r"A(\d+)", data.strip())
            if match:
                self.experiment_ids.add(int(match.group(1)))


page = Path(__file__).resolve().parents[1] / "info" / "research_run.html"
source = page.read_text(encoding="utf-8")
audit = ExplainerAudit()
audit.feed(source)
audit.close()
expected_ids = set(range(1, 42))
missing_ids = sorted(expected_ids - audit.experiment_ids)
extra_ids = sorted(audit.experiment_ids - expected_ids)
missing_links = [
    href for href in audit.local_links
    if not (page.parent / urlsplit(href).path).exists()
]
required_claims = ["0bed48b", "067ed6c", "768cd7e", "236352d", "A40", "A41", "0.334 ± 0.060", "0.309", "Readout seed and width alone", "pooled anatomy MAE", "No causal replacement score exists"]
missing_claims = [claim for claim in required_claims if claim not in source]
if audit.anchor_depth != 0:
    audit.errors.append("unclosed anchor")
if missing_ids or extra_ids or missing_links or missing_claims or audit.errors:
    print(f"missing experiment IDs: {missing_ids}")
    print(f"unexpected experiment IDs: {extra_ids}")
    print(f"broken local links: {missing_links}")
    print(f"missing required claims: {missing_claims}")
    print(f"HTML structure errors: {audit.errors}")
    sys.exit(1)
print(f"PASS: all {len(expected_ids)} experiment IDs, {len(audit.local_links)} local links, and cutoff/branch claims checked")
