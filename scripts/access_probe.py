#!/usr/bin/env python3
"""Reachability probe for every Tier-0 dataset portal.

Run on YOUR network (campus/lab/home) to see which Tier-0 endpoints are
reachable before you start a big download. Prints a table with HTTP codes.

    python3 scripts/access_probe.py
"""
from __future__ import annotations

import subprocess

TARGETS = [
    # (label, url)
    ("TCIA portal",              "https://www.cancerimagingarchive.net/"),
    ("TCIA API",                 "https://services.cancerimagingarchive.net/services/v4/TCIA/query/getCollectionValues"),
    ("TCIA GCS bucket",          "https://gcs.cancerimagingarchive.net/"),
    ("LUMIERE zip (Figshare)",   "https://ndownloader.figshare.com/files/38249697"),
    ("LUMIERE XNAT",             "https://xnat.bmia.nl/"),
    ("OASIS/XNAT Central",       "https://xnatcentral.org/"),
    ("CBICA IPP (BraTS)",        "https://ipp.cbica.upenn.edu/"),
    ("ADNI (LONI IDA)",          "https://ida.loni.usc.edu/"),
    ("PPMI",                     "https://www.ppmi-info.org/"),
    ("NDA (ABCD/OAI)",           "https://nda.nih.gov/"),
    ("PhysioNet",                "https://physionet.org/"),
    ("PhysioNet Challenge 2019", "https://physionet.org/content/challenge-2019/"),
    ("fastMRI",                  "https://fastmri.med.nyu.edu/"),
    ("PI-CAI",                   "https://pi-cai.grand-challenge.org/"),
    ("CHAOS",                    "https://chaos.grand-challenge.org/"),
    ("HECKTOR",                  "https://hecktor.grand-challenge.org/"),
    ("autoPET",                  "https://autopet.grand-challenge.org/"),
    ("HuggingFace (CT-RATE)",    "https://huggingface.co/datasets/ibrahimhamamci/CT-RATE"),
    ("Zenodo",                   "https://zenodo.org/"),
    ("Figshare API",             "https://api.figshare.com/v2/articles"),
    ("BIMCV",                    "https://bimcv.cipf.es/"),
    ("IvyGAP",                   "https://ivygap.org/"),
    ("GitHub raw",               "https://raw.githubusercontent.com/"),
    ("MIMIC docs",               "https://mimic.mit.edu/"),
    ("Kaggle",                   "https://www.kaggle.com/"),
    ("UK Biobank",               "https://www.ukbiobank.ac.uk/"),
]

def probe(url: str) -> str:
    try:
        r = subprocess.run(
            ["curl", "-sL", "-m", "12", "-A", "Mozilla/5.0", "-o", "/dev/null",
             "-w", "%{http_code}", url],
            capture_output=True, text=True, timeout=20)
        code = r.stdout.strip()
        if code == "403":
            return "403 bot-block (browser OK)"
        if code == "000":
            return "000 unreachable here"
        return code
    except Exception:
        return "ERR/timeout"

def main():
    print(f"{'endpoint':28s}  status")
    print("-" * 60)
    for label, url in TARGETS:
        print(f"{label:28s}  {probe(url)}")

if __name__ == "__main__":
    main()
