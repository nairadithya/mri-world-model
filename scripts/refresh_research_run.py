"""Refresh the current R36 section and archive links from aggregate-only metrics."""
import argparse
import html
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def section(metrics, reviewed):
    expected = {"C_vs_B_frozen_accessibility": "INCONCLUSIVE", "state_retention": "VERIFIED",
                "D_nuisance_tradeoff": "INCONCLUSIVE", "incremental_temporal_information": "INCONCLUSIVE"}
    if metrics.get("hypotheses") != expected:
        raise ValueError("R36 narrative requires revision for changed hypothesis classifications")
    contrast = metrics["comparisons"]["C-B"]
    lower, upper = contrast["family_adjusted_ci"]
    status = ("Independent result review completed for the scoped report. Artifact verification and its limits are recorded in the review receipt."
              if reviewed else "Independent result review and artifact verification are pending. No quality verdict is accepted.")
    tables = []
    for cohort, arms in metrics["cohorts"].items():
        rows = []
        for arm, values in arms.items():
            local = values["local"]
            direct = local.get("direct_change", {}).get("patient_uniform_balanced_mae")
            cells = [arm, f'{values["state"]["patient_uniform_mae"]:.4f}']
            cells.extend(f'{local[method]["patient_uniform_balanced_mae"]:.4f}'
                         for method in ("frozen_probe", "independent_state", "target_only", "zero"))
            cells.append("not graded" if direct is None else f"{direct:.4f}")
            rows.append("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in cells) + "</tr>")
        tables.append(f'<h3>{cohort.upper()} development assessment</h3><div class="wide"><table>'
                      '<thead><tr><th>Arm</th><th>Direct state</th><th>Frozen difference</th>'
                      '<th>Independent state difference</th><th>Target-only</th><th>Zero change</th>'
                      '<th>Nonlinear change head</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>')
    state = metrics["state_C-A"]
    dstate = metrics["nuisance_comparisons"]["state_D-C"]
    return f'''<h2>R36 Phase 1, observed-change assessment</h2>
<div class="box"><b>Current result.</b> All twelve final-epoch training arms, thirty source-only probes, and both development-cohort assessments are present. The primary balanced-objective encoder contrast is <b>INCONCLUSIVE</b>. C minus B frozen-local MAE is {contrast["difference"]:+.6f}, with family-adjusted paired-patient interval [{lower:+.6f}, {upper:+.6f}]. This is not a verified encoder improvement.</div>
<p>{status} Phase 2 remains unauthorized.</p>
<p>R36 holds the backbone, seed-matched R35 initialization, training exposures, and frozen-probe recipe fixed. A continues state-only training. B adds whole-grid signed-change supervision. C balances changed and unchanged regions. D adds synthetic nuisance constraints. The original R35 encoder is the initialization control. Both observed scans are inputs, not future forecasts.</p>
{"".join(tables)}
<p class="small">Three-seed mean patient-uniform errors, not ensemble predictions. State is native log1p-volume MAE. The other columns are balanced signed-occupancy MAE. LUMIERE includes 149 visits from 26 patients and 117 pairs from 23 patients. SAILOR includes 269 visits and 242 pairs from 27 subjects. All readouts were fitted on source52 only. Both cohorts are developmental. The nonlinear heads and frozen probes answer different questions.</p>
<p>C meets the locked direct-state noninferiority margin against A. Its adjusted interval is [{state["family_adjusted_ci"][0]:+.6f}, {state["family_adjusted_ci"][1]:+.6f}], below the upper allowance of +0.05. This narrow result does not establish universal state retention or a state improvement. Neither C nor D establishes superiority over independent decoded-state differences.</p>
<p>D reduces average trained-perturbation excess false change, but its intended tradeoff remains inconclusive. D minus C direct-state error is {dstate["difference"]:+.6f}, with adjusted interval [{dstate["family_adjusted_ci"][0]:+.6f}, {dstate["family_adjusted_ci"][1]:+.6f}]. State error and noise sensitivity worsen. The state interval overlaps the +0.05 allowance. The intensity transform is largely canceled by per-sequence standardization, so its near-zero drift is not evidence of learned intensity invariance. Held-out resolution remains a separate stress test.</p>
<p>C and D show a descriptive nonlinear-decoder benefit over B, not the primary frozen-encoder gain. SAILOR frozen-difference errors exceed the independent-state controls in every arm. All SAILOR ridge-delta means exceed zero change. Empty compartments still have large errors; the complete aggregate companions include empty/nonempty visits, changed/unchanged partitions, appearance/disappearance, source-fixed change-size strata, scalar and ridge readouts, reversal, and nuisance responses.</p>
<p>The target audit preserved 599 visits and 496 pairs without exclusions. Sparse changed occupancy motivates the experiment but does not prove background dilution caused R35's behavior. One native enhancing disagreement vanished after resizing. A suspect extracranial contour remains in the primary analysis. Anatomical correspondence, segmentation accuracy, repeatability, and biological attribution remain unknown.</p>
<p class="source">Read the <a href="EXPERIMENTS.md#r36-completed">completed Phase 1 findings</a>, <a href="reviews/r36_closure_receipt.json">review completion receipt</a>, <a href="r36_phase1_metrics.json">aggregate companions and input hashes</a>, and <a href="METHODS.md#r36-phase1">locked experiment protocol</a>. Implementation and records are combined on <code>integration/r36-report-closure</code>. No new forecast run or method family is authorized by this report.</p>
'''


def refresh(page, metrics, reviewed=False):
    source = page.read_text()
    start = source.index("<h2>R36")
    end = source.index('<div class="timeline"', start)
    source = source[:start] + section(metrics, reviewed) + "\n" + source[end:]
    def archive(match):
        target = match[1]
        resolved = (page.parent / target).resolve()
        if resolved.exists() or not target.endswith(".md"):
            return match[0]
        relative = resolved.relative_to(ROOT).as_posix()
        archived = ROOT / "info/archive/2026-10-12" / relative
        if not archived.exists():
            raise ValueError(f"unresolved research link: {target}")
        return 'href="' + str(archived.relative_to(page.parent)) + '"'
    source = re.sub(r'href="([^"#:]+)"', archive, source)
    page.write_text(source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=ROOT / "info/r36_phase1_metrics.json")
    parser.add_argument("--reviewed", action="store_true", help="Use only after the durable independent review receipt is recorded")
    args = parser.parse_args()
    if args.reviewed:
        directory = ROOT / "info/reviews"
        receipt = json.loads((directory / "r36_closure_receipt.json").read_text())
        review, audit = receipt["scientific_result_review"], receipt["artifact_verification"]
        if (review["native_status"] != "complete" or review["verdict"] != "APPROVED_FOR_SCOPED_REPORT"
                or audit["native_status"] != "complete" or audit["verdict"] != "PASS" or audit["failed_assertions"]):
            raise ValueError("independent review and artifact audit must complete before marking the page reviewed")
        for filename, expected in receipt["evidence_sha256"].items():
            path = (directory / filename).resolve()
            if not path.is_relative_to(directory.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("closure evidence bytes differ from the recorded receipt")
    refresh(ROOT / "info/research_run.html", json.loads(args.metrics.read_text()), args.reviewed)
