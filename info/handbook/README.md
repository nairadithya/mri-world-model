# Maintain the technical handbook

The generated reader is [technical_handbook.html](../technical_handbook.html). It is an educational guide to the research record, not a fifth canonical result ledger. The four canonical pages retain their ownership. Publication assessments in the handbook are editorial judgments, not new scientific approvals or permission to run experiments.

## Build and check

Edit the numbered HTML fragments in `chapters/` or the page design in `layout.html`. Then run:

```bash
python3 -B scripts/build_technical_handbook.py
python3 -B scripts/build_technical_handbook.py --check
```

The builder writes only `info/technical_handbook.html`. The check mode writes nothing. Run these commands from the repository root. No third-party package or network service is needed to build or read the page.

Each fragment contains one `section` with class `chapter`, a stable `id`, a `data-part`, optional space-separated `data-requires`, and one `h2`. Filenames define the reading order. The builder derives navigation from that same order and rejects missing or forward prerequisites. Author citations relative to the generated page in `info/`, not relative to the source fragment.

The layout supplies `{{NAV}}`, `{{CONTENT}}`, `{{IDENTITY}}`, and `{{CHAPTER_COUNT}}` once each. The builder fills them deterministically. Local links and fragments must resolve. External references are ordinary links; the handbook loads no remote fonts, scripts, styles, or images.

## Keep the explanation subordinate to the evidence

Teach a term before using it to explain a result. State the task, cohort, fitting budget, units, aggregation, uncertainty, and correction status beside selected numbers. Link the canonical source. Keep invalid historical scores visibly withdrawn. Do not turn an inconclusive contrast into equivalence or a decoder improvement into encoder superiority.

Conceptual diagrams must identify themselves as schematics. They contain no participant images or measured plot data. Do not reuse historical figures that contain withdrawn claims. Numerical plot additions require the project's existing aggregate-input and figure-review procedure.

After an evidence change, review the relevant explanation and publication judgment. A fresh build or valid link does not prove semantic accuracy. Obtain independent read-only source comparison after substantive edits. Render the actual page at desktop and mobile widths and inspect the pixels. Check print output separately.

## Design decision

Two independent sketches were compared on October 17, 2026. GPT-6 Luna proposed an ordered chapter compiler with derived navigation. MiniMax M3 proposed explicit HTML fragments and a separate chapter manifest.

The ordered compiler is the base. HTML fragments replace the proposed custom Markdown subset. This preserves control over tables, semantic figures, and accessible diagrams without adding a parser dependency. The separate manifest and separate verifier are rejected because chapter identity and validation would have two owners. One builder with a read-only check mode owns both.

Accepted tradeoffs are explicit HTML authoring and a generated file committed for readers. The handbook is one offline page, not an application or a general documentation framework. The source-preservation baseline and review scope are recorded in the [dated update](../documentation_sources/updates/2026-10-17/README.md).
