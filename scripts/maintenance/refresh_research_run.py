"""Refresh current R36/R37 sections and archive links from aggregate-only evidence."""
import argparse
import html
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


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


def r37_evidence(root=ROOT):
    closure = json.loads((root / "info/documentation_sources/updates/2026-10-26/closure.json").read_text())
    if (closure["result_review"]["verdict"] != "APPROVED_FOR_SCOPED_R37_REPORTING"
            or closure["scientific_classification"] != "INCONCLUSIVE_FOR_FULL_CAPABILITY"
            or closure["experiment_success"] is not False):
        raise ValueError("R37 narrative requires the scoped inconclusive result approval")
    evidence = [closure[key] for key in ("aggregate", "implementation_review", "result_review")]
    replay = closure["independent_verification"]
    evidence += [{"path": replay[f"{key}_path"], "sha256": replay[f"{key}_sha256"]} for key in ("checker", "receipt")]
    for item in evidence:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("R37 evidence bytes differ from the closure receipt")
    metrics = json.loads((root / closure["aggregate"]["path"]).read_text())
    if metrics["decision"] != closure["scientific_classification"] or metrics["contract_digest"] != closure["contract_digest"]:
        raise ValueError("R37 aggregate/closure identity mismatch")
    expected = {"primary_contrasts": False, "mae_noninferiority": False, "coverage": False, "nuisance": False}
    if metrics["criteria"] != expected or any(not (c["ci"][0] <= 0 <= c["ci"][1]) for c in metrics["primary_comparisons"].values()):
        raise ValueError("R37 narrative requires revision for changed gates or contrasts")
    return metrics, closure


def r37_section(metrics, closure):
    def interval(value):
        return f'[{value["ci"][0]:+.6f}, {value["ci"][1]:+.6f}]'
    labels = {"H": "Spatial MRI history", "L": "Latest MRI, matched observation history", "P": "Decoded-state persistence",
              "E": "Source joint changes, Gaussian observation", "R": "Voxelwise ridge", "G": "Pooled earlier MRI maps"}
    rows = "".join(f'<tr><td>{arm}: {labels[arm]}</td><td>{metrics["arm_scores"][arm]["crps"]:.6f}</td>'
                   f'<td>{metrics["arm_scores"][arm]["mae"]:.6f}</td></tr>' for arm in labels)
    comparisons = metrics["primary_comparisons"]
    contrasts = "".join(f'<tr><td>H minus {arm}</td><td>{comparisons[f"H-{arm}"]["difference"]:+.6f}</td>'
                        f'<td>{interval(comparisons[f"H-{arm}"])}</td></tr>' for arm in ("L", "P", "E", "R", "G"))
    stress = ""
    for arm in ("H", "L"):
        value = metrics["nuisance"][arm]["resolution"]
        stress += (f'<tr><td>{arm}</td><td>{100*value["relative_crps_worsening"]:+.2f}%</td>'
                   f'<td>{value["crps_difference"]["difference"]:+.6f} {interval(value["crps_difference"])}</td>'
                   f'<td>{value["mae_difference"]["difference"]:+.6f} {interval(value["mae_difference"])}</td></tr>')
    cohort = metrics["cohort"]
    single = metrics["single_visit_companion"]["H"][0]
    coverage = ", ".join(f"{100*x:.2f}%" for x in metrics["H_80_coverage_by_compartment"])
    margins = metrics["mae_margins"]
    execution = closure["execution"]
    return f'''<h2 id="r37">R37, next-observed anatomical prediction</h2>
<div class="box"><b>Inconclusive for the full capability.</b> The fixed pilot completed on {closure["date"]} with scoped independent reporting approval. Added-history, earlier-spatial-detail and secondary latest-state benefits were not established. This formulation stops after one analysis; no rescue training follows.</div>
<p>The original R35 encoder, volume head and target normalizer stayed frozen. H, L and G used {len(execution["seeds"])} matched predictor seeds and {execution["epochs_per_fit"]} epochs each. Source52 supplied {execution["source_pairs"]} fitting pairs; internal13 supplied {execution["validation_pairs"]} selection/calibration pairs. L removes earlier MRI maps but retains H's gaps, masks and sequence-availability flags. G pools only earlier maps. P copies the latest MRI-decoded volume, not a measured cutoff mask. R fits linear spatial dynamics. E models source changes with fixed Gaussian observation noise; its condition was locked after the earlier noise ablation was known.</p>
<div class="wide"><table aria-label="R37 primary anatomical errors"><thead><tr><th>Arm</th><th>CRPS</th><th>Mean-prediction MAE</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="small">Native log1p(mm3) coordinates; lower is better. CRPS is the continuous ranked probability score. Average seed scores, not predictions, then examples within patient and patients equally. Primary assessment has {cohort["examples"]} pairs from {cohort["patients"]} reused development patients. Single-visit companions, {single["examples"]} pairs from {single["patients"]} patients, stay separate and can overlap those patients. E's lowest point CRPS is not a significance claim against every arm.</p>
<div class="wide"><table aria-label="R37 five primary paired comparisons"><thead><tr><th>Contrast</th><th>CRPS difference</th><th>Adjusted 99% interval</th></tr></thead><tbody>{contrasts}</tbody></table></div>
<p class="small">Negative favors H. All five intervals span zero. These are 10,000 paired patient-bootstrap resamples, seed 2026, with the five-contrast Bonferroni correction; unadjusted companions also span zero. H does not meet the required 5% point improvement over L. This is not equivalence or proof that MRI history has no useful information.</p>
<p>All four capability requirements failed. H minus L MAE is {margins["H-L"]["difference"]:+.6f}, 95% interval {interval(margins["H-L"])}; H minus P is {margins["H-P"]["difference"]:+.6f}, {interval(margins["H-P"])}. Both upper bounds exceed +0.05, which fails noninferiority without proving significant inferiority. H's 80% coverage is {coverage}, in necrotic/non-enhancing, enhancing and edema/FLAIR order. Enhancing coverage misses the point requirement of 70% to 90%; no population-calibration significance claim follows.</p>
<h3>Resolution nuisance is a separate harm companion</h3>
<div class="wide"><table aria-label="R37 half-resolution nuisance"><thead><tr><th>Arm</th><th>Relative CRPS worsening</th><th>CRPS increase, 95% interval</th><th>MAE increase, 95% interval</th></tr></thead><tbody>{stress}</tbody></table></div>
<p class="small">One earliest eligible primary cutoff per patient, {execution["nuisance_patients"]} patients and examples. H exceeds the +10% CRPS and +0.05 MAE nuisance margins. These paired 95% companions concern a synthetic half-resolution transform, not clinical scanner effects or the adjusted primary family. No adjusted primary contrast establishes H degradation. Intensity changes are near numerical noise because standardization largely cancels the affine transform, not because learned intensity invariance was demonstrated. Reported volume drift is in log-volume coordinates, not mm3.</p>
<p>Automated targets, sparse empty/appearance/disappearance subsets, reused development patients, approximate historical week-bin gaps, unknown interventions and frozen-decoder bias limit interpretation. No expert biological validation, independent generalization, fixed-day/multistep prediction or treatment-effect claim follows. Reporting approval is not novelty or publication acceptance.</p>
<p class="source">Read the <a href="EXPERIMENTS.md#r37-full-pilot">complete findings and companions</a>, <a href="r37_full_metrics.json">unchanged raw aggregate</a>, <a href="documentation_sources/updates/2026-10-26/closure.json">closure receipt</a> and <a href="documentation_sources/updates/2026-10-26/reviews/result-v1.md">independent result review</a>. The raw generator-time pending-review marker remains unchanged; the separate bound review supplies scoped approval. The checker independently reconstructed numerical reporting and bounded fixed-weight forwards, not the full training trajectory.</p>
'''


def refresh(page, metrics, reviewed=False, r37=None):
    source = page.read_text()
    start = source.index("<h2>R36")
    end = source.index('<div class="timeline"', start)
    current = section(metrics, reviewed) + (r37_section(*r37) if r37 is not None else "")
    source = source[:start] + current + "\n" + source[end:]
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
    refresh(ROOT / "info/research_run.html", json.loads(args.metrics.read_text()), args.reviewed, r37_evidence())
