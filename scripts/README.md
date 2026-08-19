# Acquisition toolkit (`scripts/`)

Tools to fetch Tier-0 datasets programmatically. Everything here was tested
live from this machine unless noted. See `../DATASET_ACQUISITION_PLAN.md` for
the full per-dataset runbook.

## What already works, right now (no accounts, no sign-up)

### `lumiere_fetch.py` — LUMIERE (91 GBM patients, weekly MRI during chemo-RT)

The entire dataset is **one 32.6 GB zip on Figshare, CC0, no auth**:
`https://ndownloader.figshare.com/files/38249697` (it's a 302 → S3 presigned
link; HEAD requests get 403, ranged GET works).

This tool reads the zip's central directory with tiny ranged fetches, then
pulls **only the entries you ask for** — so 5 patients cost ~1–2 GB, not 32 GB.

```bash
python3 scripts/lumiere_fetch.py --list                              # archive contents
python3 scripts/lumiere_fetch.py --n 5 --out data/lumiere            # first 5 patients
python3 scripts/lumiere_fetch.py --patients Patient-001 Patient-002 --out data/lumiere
python3 scripts/lumiere_fetch.py --full --out data/lumiere           # entire 32.6 GB (resumable)
```

Verified on the real archive: 91 patients (`Imaging/Patient-XXX/week-NNN/
{T1,CT1,T2,FLAIR}.nii.gz` + `DeepBraTumIA-segmentation/` + `HD-GLIO-AUTO-segmentation/`),
ZIP64-aware, CRC-checked. `data/lumiere_sample/` already holds Patient-001's
full longitudinal set (48 MB, 2 visits, segmentations with labels {0,1,2}).

### `data/lumiere_meta/` — already fetched (616 KB)

| file | contents |
|---|---|
| `readme__LUMIERE-readme.pdf` | dataset documentation |
| `rano_rating__LUMIERE-ExpertRating-v202211.csv` | **616 expert RANO ratings** (PD/SD/PR/CR per patient+week) — the ground truth for Path C/F |
| `demographics__LUMIERE-Demographics_Pathology.csv` | survival (weeks), sex, age, IDH, MGMT |
| `mri_params__LUMIERE-MRinfo.csv` | per-image acquisition parameters (vendor, TR/TE, flip…) |
| `completeness__LUMIERE-datacompleteness.csv` | which sequences/segmentations each visit has |

**Instant project seed:** join RANO ratings + demographics + completeness, and
compute per-visit tumor volumes from the segmentations → a real graph-node
trajectory table for toy Paths A/C. That's a genuine "Δgraph" from real data
with zero further downloads.

## Needs your account/network (documented, not blocked here)

| Tool / step | What's needed |
|---|---|
| TCIA downloads | Works from normal networks; `pip install tcia` (PyPI ✓) then `tcia get --collection "<name>" --destination ./data`. From this sandbox the API host (`services.cancerimagingarchive.net`) is unreachable — likely fine on campus. |
| BraTS 2021 | CBICA IPP (`ipp.cbica.upenn.edu`) registration + agreement; or Kaggle mirror (`kaggle datasets download`). |
| PhysioNet (MIMIC-IV, eICU) | CITI "Data or Specimens Only" certificate + PhysioNet DUA (1–5 days). Challenge-2019 is also gated behind login. |
| CT-RATE | HuggingFace account, accept gated terms, `huggingface-cli download ibrahimhamamci/CT-RATE`. |
| OASIS-3 / LUMIERE-XNAT | Free portal accounts (LUMIERE's Figshare zip above skips XNAT entirely). |

## Network reality check for this machine

Reachable from here: Figshare API + downloads ✓, GitHub ✓, HuggingFace ✓,
Nature/arXiv ✓, TCIA website ✓ (but **not** its API/GCS subdomains), PhysioNet
page ✓ (downloads need login), Zenodo ✗ (bot-blocked 403; fine in a browser).

Run `python3 scripts/access_probe.py` on *your* network for a fresh table.
