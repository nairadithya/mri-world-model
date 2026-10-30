# Workshop manuscript literature check

Checked October 24, 2026. This is source evidence for the [canonical manuscript assessment](../../../NOTES_AND_INSIGHTS.md#workshop-manuscript-feasibility), not a second research result narrative. The operator requires an accepted paper at ICWM, ICLR, or ICML and permits a workshop paper. The operator authorized this bounded feasibility check, not new training, a new scientific target, submission, or data acquisition.

## Search boundary and stopping rule

The decision was whether the completed R35/R36 comparisons support a specific short-paper contribution after checking the closest literature. The search stopped after nine targeted queries, inspection of seven retrieved primary full-text sources at the scopes below, one additional primary abstract with blocked full text, and two adjacent abstracts. There was no systematic-review or exhaustive-priority claim. No publication scores were treated as matched performance targets.

Queries used on October 24:

1. `longitudinal brain MRI change representation learning independent scans subtraction siamese change detection baseline`
2. `glioma MRI "change" "representation" "balanced" longitudinal`
3. `longitudinal MRI "change" "linear probe" representation learning`
4. `"Learning-based inference of longitudinal image changes" LILAC paper arxiv`
5. `"Automated assessment" "glioma" "longitudinal" "2022" Rudie`
6. `"LSOR" "Longitudinally-Consistent" arxiv`
7. `"Longitudinal Assessment of Posttreatment Diffuse Glioma" "volume" "segmentation"`
8. `"Learning-based inference of longitudinal image changes" "independently"`
9. `"longitudinal" "change" "frozen" "decoder" MRI representation`

Search results identified leads only. The comparisons below use the retrieved primary text, not the search provider's summary. BrainDiff was followed through its direct arXiv page rather than the third-party proxy returned by search. No additional search expansion followed that source.

## Closest primary sources

### PaIRNet, Kim and Sabuncu, Learning to Compare Longitudinal Images

- Source: [PMLR record](https://proceedings.mlr.press/v227/kim24a.html), [full paper](https://proceedings.mlr.press/v227/kim24a/kim24a.pdf).
- Read scope: complete extracted 18-page PDF, including Sections 3, 4.1.2, 4.2 and Appendices D and G. The PMLR record gives volume 227 and year 2024; the PDF carries MIDL 2023 headers. Use the publisher's bibliographic record rather than silently reconciling those dates.
- Inputs and task: two observed images, supervised target differences or self-supervised temporal ordering. Evaluations include synthetic tumor images and aging brain MRI, not the R36 glioma occupancy endpoint.
- Relevant control: cross-sectional supervised regression uses the same backbone to estimate each image's target separately, then subtracts predictions. Appendix G also compares a randomly initialized frozen feature extractor with trained models.
- Exact passage, Section 4.2: "computing two separate predictions and then taking the difference."
- Consequence: neither the independent-state-difference control nor checking whether pair learning earns its complexity is new here. Its random-feature ablation is not the same as R36's matched objective-only comparison of trained encoders with a newly fitted fixed-capacity probe. R35/R36 are not PaIRNet reproductions or a refutation of its results.

### LILAC, Kim and colleagues, Learning-based inference of longitudinal image changes

- Source: [PNAS article in PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11873959/), [primary full-text XML mirror](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11873959/fullTextXML).
- Read scope: article front matter and complete main-body text from the XML, including the brain-aging and mild cognitive impairment experiments, Discussion, and Materials and Methods. The supplement was not retrieved. The article identifies LILAC as an extension of PaIRNet; these are one method family, not independent replications.
- Inputs and task: two observed images. A shared CNN and bias-free linear readout learn temporal ordering, elapsed-time differences, or clinical-score differences. Optional metadata controls specified covariates. These are assessments of observed change, not past-only forecasts.
- Relevant control: the single-image regression baseline estimates age or clinical score at each visit and takes differences. The brain experiments directly compare this baseline with pair-trained models.
- Exact passage, brain-aging experiment: "calculates the difference in predicted age between a pair of longitudinal images."
- Consequence: the broad proposed state-versus-change paper question has direct prior art, including 3D brain MRI. A new manuscript must describe a scoped glioma/objective comparison rather than claim to introduce this distinction or show that paired learning generally fails.

### BrainDiff, Patel and Liu, Longitudinal Report Generation for Multimodal Brain MRI

- Source: [version 2 record](https://arxiv.org/abs/2609.00593v2), [version 2 full text](https://arxiv.org/html/2609.00593v2).
- Read scope: abstract and Introduction, Sections 3.1 and 3.2, Sections 5.1 through 6, and the relevant difference-pathway appendix passages. Tables 7 and 8 and their surrounding analysis were inspected. The arXiv record dates version 2 to September 3, 2026, before this check. Conference acceptance was not verified.
- Inputs and task: two observed brain MRI studies and a prior report, producing an interval-comparison report. Its evaluation task, encoder, targets, training scale, and data differ from R35/R36.
- Relevant controls: a difference-pathway ablation, a frozen-feature linear-probe test of interval change versus single-study pathology, a within-pair control against between-patient magnitude shortcuts, and an augmentation-only nuisance control.
- Exact passage, Section 5.2: "Interval change is much less decodable than single-study pathology."
- Consequence: frozen-feature change decodability and separating encoder information from the benefit of a difference module are already explicit in closely related brain MRI work. They cannot be the manuscript's first-of-kind claim. R36's remaining candidate contribution is its matched change-supervision objective comparison with independent decoded-state and nuisance companions. The absence of that exact combination in these inspected sections is not proof of global novelty. Do not adopt BrainDiff's proposed resolution explanation as an established cause of R36's outcome.

### STRIDE, Spatial-Temporal Representation for Interval-conditioned Disease Evolution in Longitudinal Glioblastoma MRI

- Source: [version 1 record](https://arxiv.org/abs/2610.08848v1), [version 1 full text](https://arxiv.org/html/2610.08848v1).
- Read scope: abstract and Introduction, problem formulation and pair-context construction, Sections 3 through 6. The record dates version 1 to October 2, 2026. A journal name in the manuscript header is not independent verification of journal acceptance.
- Inputs and task: both observed post-treatment MRI scans and their actual inter-scan interval, with LUMIERE pretraining followed by Burdenko adaptation for clinical-state classification. Pair context includes the observed follow-up representation.
- Relevant controls: single-timepoint versus paired inputs, different longitudinal fusion strategies, and cumulative component ablations. The authors explicitly note that sequential component additions do not identify independent component effects.
- Exact passage, Section 3.3: "Whenever applicable, the same visit-level spatial encoder, lesion-prior guidance, data augmentation, and optimization settings were maintained"
- Consequence: neither spatial-temporal glioma representations, timing-aware pair processing, nor a current-scan/pair contrast is new. This is observed-pair assessment, so its clinical classification scores are not a target for R37's past-only forecast. No reproduction or head-to-head superiority is claimed.

### TRACE, A Concept Bottleneck Model for Longitudinal 3D Glioblastoma Response Assessment

- Source: [version 2 full text](https://arxiv.org/html/2606.30313v2).
- Read scope: abstract and Introduction, Sections 3.5 through 3.7, Sections 4 and 5.1, and the frozen-feature concept-bottleneck baseline description in Appendix A.8. This is a partial primary-method read, not a complete supplement or code audit.
- Inputs and task: paired observed LUMIERE scans, segmentation inputs and clinical concepts for RANO classification. The method predicts root measurements and derives dependent RANO concepts deterministically.
- Relevant controls: frozen-embedding concept discovery with a regularized classifier, ground-truth/predicted-concept rule-based references, and graph/training ablations.
- Exact passage, Section 4.2: "isolating the behaviour of threshold-only decision logic and its sensitivity to concept-estimation noise."
- Consequence: explicit anatomical state measurements feeding change assessment and a frozen-feature baseline are not new. R36 differs in its matched encoder-objective and signed-occupancy endpoint; clinical label scores are not comparable. This read does not establish that TRACE lacks every independent-state or probe variant outside the checked scope.

### LSOR, Ouyang and colleagues, Longitudinally-Consistent Self-Organized Representation Learning

- Source: [version 1 record](https://arxiv.org/abs/2310.00213v1), [full text](https://arxiv.org/html/2310.00213v1). The raw abstract-page citation metadata identifies Jiahong Ouyang as first author; the readable full-text extraction omitted that name.
- Read scope: abstract and Introduction, the Evaluation description in Section 3.1, and the frozen-encoder Downstream Tasks passage in Section 3.2. This comparison does not depend on a full supplement audit.
- Inputs and task: observed longitudinal MRI from the Alzheimer's Disease Neuroimaging Initiative, with an autoencoder and longitudinal consistency objective for interpretable representations. Downstream tasks assess clinical-state discrimination and cognitive scores.
- Relevant control: encoders trained by different representation objectives are frozen, then evaluated using matched downstream MLP architectures. These are nonlinear downstream heads, not R36's local linear change probe.
- Exact passage, downstream evaluation: "we froze encoders trained by each method without fine-tuning"
- Consequence: matched frozen-encoder evaluation of longitudinal MRI objectives is established practice. The proposed contribution must be the particular controlled empirical finding, not the act of freezing an encoder or fitting a probe.

### Matoso and colleagues, Towards a deep learning approach for classifying treatment response in glioblastomas

- Source: [version 2 record](https://arxiv.org/abs/2504.18268v2), [version 2 full text](https://arxiv.org/html/2504.18268v2).
- Read scope: author list, abstract, and selected input-subtraction/result/discussion passages. No full implementation audit or new performance comparison was performed.
- Inputs and task: two observed consecutive LUMIERE MRI acquisitions for RANO classification; the paper compares subtracted-image inputs with individual-timepoint inputs among other choices.
- Exact passage, abstract: "inputting subtraction images instead of individual timepoints"
- Consequence: subtraction controls in glioblastoma assessment have prior art. Input-image subtraction is not identical to R35's independent decoded-state subtraction.
- Bibliographic correction: arXiv:2504.18268v2 belongs to Ana Matoso and colleagues. The old P1 audit's association of that identifier with Maurya is incorrect. The separate Maurya numerical claims have not been verified or reassigned by this check.

## Relevant lead with only a primary abstract available

Rudie and colleagues, [Longitudinal Assessment of Posttreatment Diffuse Glioma Tissue Volumes with Three-dimensional Convolutional Neural Networks](https://doi.org/10.1148/ryai.210243), 2022. The [indexed primary abstract](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:36204543%20AND%20SRC:MED&format=json&resultType=core) describes separate longitudinal-change networks trained on registered and subtracted scans and explicitly compares their change-classification performance with segmentation-based assessment. This is particularly direct prior art for broad state-versus-change claims in post-treatment glioma. The PMC page did not yield readable content, the full-text XML API did not supply the paper, and an institutional PDF returned HTTP 403. No detailed assertion about its full ablation set is made.

Two adjacent abstracts were also inspected: [Tikhonov's hybrid response classifier](https://arxiv.org/abs/2509.06511) and [TaDiff's treatment-aware future MRI generation](https://arxiv.org/abs/2309.05406). They distinguish observed response assessment from future generation. Their complete methods, exact comparison budgets, and scores were not verified for this decision.

## What remains to write, and what is not a contribution

The proposed question is now narrower: **Do gains from change-balanced supervision remain detectable under a matched fixed-capacity readout of the frozen encoder?** R36 B averages signed-occupancy error over the whole grid; C averages changed and unchanged voxel groups separately and weights the available group means equally. Their architecture, starting weights, state and volume-difference terms, training pairs and duration remain shared. The manuscript can report the descriptive decoder change, the unresolved fixed-probe primary contrast, and the required independent-state controls together. It cannot turn an unresolved interval into proof of equal representations or proof that only the decoder changed.

Rejected novelty claims include first use of MRI pairs, first state-versus-change comparison, first independent-state subtraction control, first frozen change probe, first longitudinal objective evaluation, and demonstrated future dynamics. None is needed for a carefully positioned empirical extension, but the exact experimental combination alone is not sufficient proof of publication-level novelty.

The assistant's assessment is **CONDITIONAL_GO_FOR_SHORT_WORKSHOP_DRAFT**, not approval for submission or a prediction of acceptance. A short draft is justified as a test of whether the controlled objective finding is useful beyond these predecessors. It must cite both LILAC and BrainDiff prominently. Its claims remain development-only and about automated targets. No faithful reproduction of these published systems exists in R35/R36. The full Rudie comparison remains a clearly bounded related-work verification item, not authorization to acquire or train on a new cohort.

## Main result figure sketch

The [machine-readable figure specification](figure-spec.json) refers directly to the existing public R36 aggregate. It does not copy a selected new set of measurements into the plot ledger. It is a layout and data-binding sketch, not a rendered or pixel-reviewed figure.

```text
A. Trained decoder scores           B. Frozen-probe objective contrasts
   B and C, both development sites     C-B primary; C-A and D-C companions
   Means only, descriptive             Paired patient intervals and zero line

C. Incremental-information controls D. Development-transfer check
   C/D minus target-only and           Every arm's frozen difference probe
   independent decoded-state probes    versus independent states and zero
   Paired patient intervals            Means only, descriptive
```

Use separate axes for absolute errors and error differences. Do not infer significance from the descriptive panels. All R36 figure panels use balanced signed-occupancy MAE, with lower error better. Pair counts, patient-uniform aggregation, three fitted seeds, probe fitting on the 52 source-training patients, final-epoch budgets, and development status belong in the caption. The interval panels use the already locked family-three adjustment, not new post-hoc tests. R35 belongs in a separate context table with its native log-volume units, dense-supervision confound, independent-state contrasts, and own uncertainty rules. It must not share an unlabeled error axis with R36.

Required manuscript companions remain the complete all-arm state table, C-versus-A state-retention result, D-versus-C measured state/noise harms and unresolved tradeoff, empty-compartment errors, both development sites, and target/provenance limits. The sketch does not authorize dropping these because the main figure is narrower. The nonlinear-decoder improvements and the linear-probe outcomes answer different questions; neither panel proves that the encoder contains no useful change information.

## Venue and review boundary

The official [ICLR 2027 workshop guidance](https://iclr.cc/Conferences/2027/WorkshopGuidelines) encourages short tracks with modest self-contained evaluations, but each workshop chooses its scope and acceptance rules. The [workshop call](https://iclr.cc/Conferences/2027/CallForWorkshops) lists workshop selection notification on November 29, 2026, a suggested contribution deadline of February 1, 2027, and mandatory author notification by February 26, 2027. A matching individual workshop call was not verified. The call page also contains an inconsistent 2026 event-year sentence; do not use that sentence as a verified event date. The [ICWM call](https://icwm.cc/call-for-papers) remains a working draft with a provisional November 30, 2026 paper deadline, not a confirmed workshop route. No ICML 2027 workshop call was verified.

Whether a non-archival workshop acceptance satisfies the operator's obligation remains unanswered. The specific call, archival status, authorship/disclosure rules, registration costs, and submission date must be confirmed before submission. ICLR's current short-track guidance requires primarily human-authored work while permitting AI assistance. This record and any assistant-produced draft are preparation aids, not a substitute for human scientific authorship.

The assistant performed this literature check and manuscript assessment. No independent reviewer assessed the literature, novelty or manuscript in this task, and no new scientific result review occurred. Existing R35/R36 approvals cover their stated result-reporting scope, not this publication judgment. No training, MRI access, private participant export, benchmark reproduction, or plot rendering ran for this task.
