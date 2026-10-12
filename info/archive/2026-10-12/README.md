# Research documentation archive

This archive preserves 32 original research records and five supporting presentation assets from the documentation checkout. The operator approved their archival on October 12, 2026. Every moved file retains its original bytes, including dirty or untracked documentation copied earlier from the source checkout.

The directory mirrors the old repository layout. Root research pages are at this level. Former `info/` pages are under `info/`. The [manifest](manifest.json) records each original path, archive path, SHA256 hash, and whether the file was tracked at the migration's base commit.

## Current reading order

Start with the consolidated [Notes and insights](../../NOTES_AND_INSIGHTS.md), [Experiments](../../EXPERIMENTS.md), [Methods](../../METHODS.md), and [Research TODO](../../TODO.md). [Source coverage](../../SOURCE_COVERAGE.md) preserves branch-version differences and links to the immutable captured sources.

## Historical limits

Archived plans do not authorize work. Archived scores may be corrected, withdrawn, incomplete, or based on retired protocols. In particular, A40 withdraws the learned P2.4 and adaptation forecasting claims. R36 statements remain dated execution snapshots, not live status.

The archived `info/notes.md` and `info/explainer.html` were already intentionally deleted in the source checkout. Their committed historical copies are preserved here without restoring them as current guidance. The HTML page includes withdrawn claims and references plot images absent from this documentation checkout. It is preserved as source history, not a validated or repaired presentation.

Files retain their original embedded links. Some links refer to old locations or unavailable artifacts. Do not use archived pages as current operating instructions. Operating guides and research guardrails remain at their documented locations outside this archive.

## Integrity checks

From the repository root, compare the archived bytes with the manifest:

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path('info/archive/2026-10-12/manifest.json').read_text())
for row in manifest['records']:
	assert hashlib.sha256(Path(row['archive_path']).read_bytes()).hexdigest() == row['sha256'], row['archive_path']
print(f"Checked {len(manifest['records'])} unchanged archived files")
PY
```

No dataset, checkpoint, experiment output, or protected research worktree was moved. The archive is append-only history. Do not rewrite its original files.
