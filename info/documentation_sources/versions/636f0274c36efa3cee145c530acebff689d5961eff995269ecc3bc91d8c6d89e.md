# P2 external supervision audit

Status: audited 2026-09-22; no external cohort is locally available. The host
has 75 GB free, so acquisition must be planned rather than started implicitly.

## UCSF-ALPTDG — preferred lesion/change supervision

- 298 adults, exactly two consecutive post-treatment timepoints each (596
  exams); median inter-scan interval 65 days.
- Expert voxelwise masks include enhancing tumor (ET), nonenhancing tumor core
  (NETC), surrounding nonenhancing FLAIR hyperintensity (SNFH), and resection
  cavity (RC), plus longitudinal change annotations.
- This is the closest match to P2's missing supervision: it supplies cavity,
  compartment-specific anatomy, and explicitly paired post-treatment change.
- Access is public for noncommercial research only after accepting the UCSF
  data-use agreement. No accepted DUA or local copy is present.

**Assigned role:** pretrain/fix a lesion and change teacher, with patients split
into teacher-train and teacher-validation before inspecting downstream LUMIERE
results. It cannot be the final external test if used for teacher training.

**Blocker:** user/institution must accept the DUA and provide an acquisition
location with sufficient space.

Primary sources: UCSF dataset article and data-availability statement,
https://pmc.ncbi.nlm.nih.gov/articles/PMC11294954/ ; UCSF dataset portal,
https://imagingdatasets.ucsf.edu/dataset/2 .

## Burdenko-GBM-Progression — long-trajectory/treatment cohort

- 180 primary-GBM patients, 645 studies, with 1–8 follow-ups per patient.
- Planning data include T1/T1C/T2/FLAIR, CT, RTSTRUCT, RTPLAN, and RTDOSE;
  follow-ups have at least T1C and FLAIR. Response status distinguishes tumor
  progression, pseudoprogression, and treatment response.
- Longitudinal follow-up GTV annotations exist only for a subset. MR images are
  in acquisition space while RT objects are aligned to CT; MRI-to-CT transforms
  are not shipped, although the authors provide a registration container.
- Imaging is 131.23 GB and subject to TCIA/NIH controlled access. The small
  clinical/genomic table is CC BY 4.0.

**Assigned role:** longer-trajectory and treatment-phase validation after
registration/label harmonization, not immediate lesion-teacher pretraining.

**Blockers:** restricted-license approval; dataset is larger than current free
space; subset annotation coverage must be measured before assigning splits.

Primary source: TCIA collection DOI 10.7937/E1QP-D183,
https://www.cancerimagingarchive.net/collection/burdenko-gbm-progression/ .

## UCSD-PTGBM — treatment-effect auxiliary cohort

- Version 3 was updated 2026-03-13 to correct misassigned patient IDs in the
  clinical spreadsheet; older metadata must not be used.
- 178 subjects and 243 timepoints with post-treatment structural, diffusion,
  and perfusion MRI, neuroradiologist-approved voxelwise segmentations, and
  diagnosis/treatment/follow-up metadata.
- The current release is 44.98 GB: 33.1 GB for 136 subjects/184 studies plus
  11.88 GB of BraTS-GLI test data. License is CC BY 4.0.
- The subject/timepoint counts imply that much of the cohort is cross-sectional
  or only sparsely longitudinal. Its strongest unique value is separating
  tumor from treatment-related change using advanced imaging, not serving as
  a natural-prevalence next-visit forecast cohort.

**Assigned role:** auxiliary tumor-versus-treatment-effect learning with
enriched sampling kept separate from forecast calibration.

**Blockers:** a full download would leave only about 30 GB free before
extraction/preprocessing; an external volume or selective official downloader
is required. Version-3 clinical-to-image joins must be audited before use.

Primary source: TCIA collection DOI 10.7937/fwv2-dt74,
https://www.cancerimagingarchive.net/collection/ucsd-ptgbm/ .

## Decision

Do not start a 45–131 GB acquisition on the current filesystem and do not use
SAILOR outcomes to invent the missing supervision. UCSF-ALPTDG is the first
choice if its DUA and storage are resolved. Until then, the reproducible local
P2 branch is closed negative at A39; treatment-effect, fixed-horizon,
uncertainty, and lesion-aware retraining remain blocked on new authorized data
or a prespecified new mechanism.
