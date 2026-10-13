"""Render aggregate R35 results without patient identifiers or scan images."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def table(headers: list[str], rows: list[list]) -> str:
    def cell(value):
        return html.escape(f"{value:.4f}" if isinstance(value, float) else str(value))
    return ('<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{cell(h)}</th>' for h in headers)
            + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{cell(v)}</td>' for v in row) + '</tr>' for row in rows)
            + '</tbody></table></div>')


def render(source: str, out: str) -> None:
    r = json.loads(Path(source).read_text())
    sections = []
    for cohort in ("LUMIERE", "SAILOR"):
        d = r[f"{cohort}_direct"]
        rows = []
        for arm, scored in d["arms"].items():
            rows.append([arm, scored["state"]["seed_mean_patient_uniform_mae"],
                         scored["independent_decoder_difference"]["seed_mean_patient_uniform_mae"],
                         scored.get("pair_head", {}).get("seed_mean_patient_uniform_mae", "Not applicable")])
        rows.append(["Source-trained mean and zero change", d["arms"]["state"]["state_mean"]["seed_mean_patient_uniform_mae"],
                     d["arms"]["state"]["zero"]["seed_mean_patient_uniform_mae"], d["arms"]["state"]["zero"]["seed_mean_patient_uniform_mae"]])
        rows.append(["Frozen scalar-source52 reference", d["frozen_source52"]["seed_mean_patient_uniform_mae"], "See source-readout table", "Not applicable"])
        sections.append(f'<section><h2>{cohort} fixed source-trained heads</h2>'
                        + table(["Model arm", "Current state MAE", "Independent state difference MAE", "Pair head MAE"], rows))
        comparisons = [[name.replace('_', ' '), value["difference"],
                        f'[{value["paired_patient_95ci"][0]:.4f}, {value["paired_patient_95ci"][1]:.4f}]',
                        f'[{value["family12_adjusted_ci"][0]:.4f}, {value["family12_adjusted_ci"][1]:.4f}]']
                       for name, value in d["comparisons"].items()]
        sections.append(table(["Candidate minus control", "MAE difference", "Patient 95% interval", "Family-12 interval"], comparisons) + '</section>')
    sections.append('<section><h2>Fixed LUMIERE ridge readouts on SAILOR</h2><p>Each readout is fitted only on the named source patient budget. All saved feature views, independent-state differences, training-mean change, and zero-change controls are reported. No SAILOR labels select a regularizer, epoch, or model.</p>')
    for budget, arms in r["SAILOR_source_readouts"].items():
        rows = []
        for arm, scored in arms.items():
            changes = scored["observed_change"]
            rows.append([arm, scored["state"]["seed_mean_patient_uniform_mae"],
                         changes["difference"]["seed_mean_patient_uniform_mae"],
                         changes["concatenation"]["seed_mean_patient_uniform_mae"],
                         changes["later_only"]["seed_mean_patient_uniform_mae"],
                         changes.get("learned_pair", {}).get("seed_mean_patient_uniform_mae", "Not applicable"),
                         changes["independent_state_difference"]["seed_mean_patient_uniform_mae"],
                         changes["train_mean_change"]["seed_mean_patient_uniform_mae"]])
        baseline = arms["state"]
        zero = baseline["observed_change"]["zero"]["seed_mean_patient_uniform_mae"]
        rows.append(["Source mean and zero change", baseline["state_mean"]["seed_mean_patient_uniform_mae"], zero, zero, zero, zero, zero, zero])
        sections.append(f'<h3>{budget} source patients</h3>' + table(["Model arm", "State", "Feature difference", "Concatenation", "Later only", "Learned pair", "Independent state difference", "Train-mean change"], rows))
    sections.append('</section><section><h2>All declared acquisition perturbations</h2><p>One fixed visit per LUMIERE evaluation patient. Ground-truth physical change is zero. Readouts are the same patient-held-out ridge models used for real observed change. Affine intensity invariance follows input standardization, not a learned robustness result.</p>')
    rows = []
    for arm, conditions in r["LUMIERE_nuisance"].items():
        for kind, value in conditions.items():
            rows.append([arm, kind, value["seed_mean_state_readout_drift"], value["seed_mean_false_change_mae"],
                         value["false_change_excess_over_identity"]])
    sections.append(table(["Model arm", "Perturbation", "State readout drift", "False-change MAE", "Excess over identity"], rows) + '</section>')
    sections.append('<section><h2>Visit-specific state diagnostic</h2><p>Post-hoc within-patient permutation changes visit assignment, not patient identity. A lower original error supports visit-specific information. It cannot distinguish biological variation from acquisition variation.</p>')
    rows = []
    for cohort in ("LUMIERE", "SAILOR"):
        for arm in ("state", "paired", "later_only", "frozen"):
            values = [r["state_audits"][f'{cohort}/seed{s}/{arm}'] for s in r["seeds"]]
            rows.append([cohort, arm, sum(v["state_patient_uniform_mae"] for v in values) / len(values),
                         sum(v["within_patient_shuffled_mae"] for v in values) / len(values)])
    sections.append(table(["Cohort", "Arm", "Original MAE", "Within-patient shuffled MAE"], rows) + '</section>')
    document = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>R35 MRI state and observed-change comparison</title>
<style>
body{margin:0;color:#20242b;background:#fff;font:16px/1.5 system-ui,sans-serif}main{max-width:1160px;margin:auto;padding:32px 24px 64px}h1{font-size:32px;line-height:1.15}h2{margin:30px 0 10px;font-size:24px}h3{font-size:19px}.scope{background:#f3f6f9;border-left:4px solid #2864a5;padding:14px 18px}.warning{border-left:4px solid #b75d17;padding:10px 18px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;margin:14px 0 22px;font-size:14px}td,th{text-align:right;padding:10px;border-bottom:1px solid #d8dde5;vertical-align:top}td:first-child,th:first-child{text-align:left;min-width:145px}th{background:#f3f6f9}section{border-top:1px solid #d8dde5;margin-top:26px}a{color:#2864a5}.foot{color:#58616e;font-size:14px}
</style></head><body><main>
<h1>R35 MRI state and observed-change comparison</h1>
<div class="scope">All twelve trained arms completed. The three seeds are 42, 43, and 44. Results below are means of errors across seeds, not errors from an ensemble prediction. Patients, not seeds or visits, are bootstrap units.</div>
<p>The targets are three native automated compartment log1p volumes. State assessment includes 149 LUMIERE visits from 26 patients and 269 SAILOR visits from 27 patients. Observed change includes 117 LUMIERE pairs from 23 patients and 242 SAILOR pairs from 27 patients. Both visits are inputs to observed-change assessment.</p>
<div class="warning">These results do not predict unobserved visits or validate biological change. SAILOR is developmental transfer, not untouched external validation. Its measurement source and the LUMIERE source differ. The spatial models receive additional dense-mask supervision. Differences from the pretrained frozen reference do not isolate architecture.</div>
<p>The state arm is the supervised spatial encoder. The paired, later-only, and frozen arms start from its selected checkpoint and receive twenty more training epochs. The frozen arm keeps that spatial encoder fixed. The frozen reference is the separate clean BRAINIAC model. No model receives lesion masks at MRI inference.</p>
<p>Negative MAE differences favor the candidate. Summary intervals are descriptive. The family-12 intervals show a multiplicity correction but do not create a new prespecified success gate. Empty and nonempty target strata and per-compartment errors remain in the machine-readable report.</p>
'''
    document += ''.join(sections)
    document += '<section><h2>Remaining evidence limits</h2><ul>' + ''.join(f'<li>{html.escape(x)}.</li>' for x in r["blocked"]) + '</ul></section>'
    document += '<p class="foot">Generated by <code>scripts/render_representation_report.py</code> from the verified R35 summary. Saved model weights, row predictions, and input digests remain local. <a href="EXPERIMENTS.md#r35">Consolidated findings and historical record</a>.</p></main></body></html>\n'
    Path(out).write_text(document)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    render(args.source, args.out)
