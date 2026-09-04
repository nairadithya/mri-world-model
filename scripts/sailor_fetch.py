#!/usr/bin/env python3
"""SAILOR (v1) EBRAINS dataset fetcher, powered by the KG-core SDK.

SAILOR ("Serial Assessments in Longitudinal Oncological Research") is a
longitudinal high-grade-glioma MRI dataset: 27 patients, 3-19 time points of
multi-sequence MRI (T1/T1c/T2/FLAIR/DTI/DCE/DSC + expert + ONCOHabitats
masks), released on EBRAINS as a set of ``.tar.bz2`` archives plus metadata TSVs.

EBRAINS gates the bytes behind authentication, so this tool:

  1. Authenticates to EBRAINS with the official ``kg-core`` SDK. Auth precedence
     (all handled for you by the SDK):
        - ``KG_TOKEN`` env var            -> reuse an existing bearer token
        - ``KG_CLIENT_ID``+``KG_CLIENT_SECRET`` env vars -> service account
        - otherwise                       -> interactive OAuth device flow
                                            (prints a URL you visit to log in)
  2. Uses the EBRAINS *data-proxy* REST API to list the dataset's files and
     stream the ones you select to disk, with curl resume + progress.

The data-proxy requires the same bearer token, which we pull out of the SDK's
token handler (``TokenHandler.get_token()``) and pass as an ``Authorization``
header on the download requests.

Examples
--------
python3 scripts/sailor_fetch.py --list                      # show file tree + sizes
python3 scripts/sailor_fetch.py --all --out data/sailor     # whole dataset
python3 scripts/sailor_fetch.py --match 'rawdata_BIDS*' --out data/sailor
python3 scripts/sailor_fetch.py --match '*.tar.bz2' --out data/sailor

Auth shortcuts (set before running):
    export KG_TOKEN=eyJ...                  # a token you already have
    export KG_CLIENT_ID=... KG_CLIENT_SECRET=...   # unattended service account
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
from typing import Optional

DATASET_ID = "cae85bcb-8526-442d-b0d8-a866425efff8"
DATA_PROXY = "https://data-proxy.ebrains.eu/api/v1"
DP_DATASET = f"{DATA_PROXY}/datasets/{DATASET_ID}"


# --------------------------------------------------------------------------- #
# Authentication (shared module: .env cache + refresh + device flow)
# --------------------------------------------------------------------------- #
from ebrains_auth import get_token  # noqa: E402


# --------------------------------------------------------------------------- #
# data-proxy REST helpers (v1 object API: list via GET dataset, bytes via
# GET dataset/{object}). Tokens are refreshed before every attempt so multi-
# hour downloads survive the ~5 min access-token lifetime.
# --------------------------------------------------------------------------- #
def _hdr_file(token: str) -> str:
    hdr = tempfile.NamedTemporaryFile("w", suffix=".hdr", delete=False)
    hdr.write(f"Authorization: Bearer {token}\n")
    hdr.close()
    return hdr.name


def list_files(token: str, limit: int = 1000) -> list:
    """List all objects (S3-style marker pagination)."""
    import requests

    items, marker = [], None
    while True:
        params = {"limit": limit}
        if marker:
            params["marker"] = marker
        r = requests.get(DP_DATASET, headers={"Authorization": f"Bearer {token}"},
                         params=params, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"file listing failed (HTTP {r.status_code}): {r.text[:300]}")
        body = r.json()
        objs = body.get("objects", [])
        items.extend(objs)
        if len(objs) < limit:
            break
        marker = objs[-1]["name"]
    return items


def _file_fields(item: dict) -> tuple[str, int, Optional[str]]:
    """Normalize one object entry to (name, size_bytes, sha_hash)."""
    return (item.get("name", ""),
            int(item.get("bytes", 0) or 0),
            item.get("hash"))


def download_file(get_token_fn, item: dict, dest: str, retries: int = 5) -> None:
    """Stream one object to *dest*, resumable, refreshing the token per attempt."""
    from ebrains_auth import get_token as _refresh

    name, size, _ = _file_fields(item)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    url = f"{DP_DATASET}/{urllib.parse.quote(name, safe='')}"
    for attempt in range(1, retries + 1):
        try:
            token, _method = (get_token_fn() if get_token_fn else _refresh())
        except Exception as e:
            print(f"  [{attempt}/{retries}] token refresh failed ({e}); retrying...")
            import time as _time

            _time.sleep(5)
            continue
        hdr = _hdr_file(token)
        cmd = ["curl", "-sL", "-C", "-", "-H", f"@{hdr}", url, "-o", dest, "-#"]
        print(f"  [{attempt}/{retries}] {name} ({size/1e9:.2f} GB) -> {dest}")
        r = subprocess.run(cmd)
        os.unlink(hdr)
        if r.returncode == 0:
            have = os.path.getsize(dest) if os.path.exists(dest) else -1
            if size and have == size:
                print(f"    complete ({have/1e9:.2f} GB, size ok)")
                return
            if not size:
                print(f"    complete ({have/1e9:.2f} GB)")
                return
            print(f"    incomplete ({have}/{size} bytes); resuming...")
        else:
            print(f"    curl rc={r.returncode}; retrying...")
    raise RuntimeError(f"download failed after {retries} tries: {name}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true", help="list dataset files + sizes")
    g.add_argument("--all", action="store_true", help="download every file")
    g.add_argument("--match", metavar="GLOB", help="download files whose path matches glob")
    ap.add_argument("--out", default="data/sailor", help="destination directory")
    args = ap.parse_args()

    print("Authenticating to EBRAINS (cached token or device flow)...")
    try:
        token, method = get_token()
    except RuntimeError:
        # no cache: fall back to interactive device flow
        from ebrains_auth import issue_device_code, poll_and_cache

        body = issue_device_code()
        print("************************************************************************")
        print(f"To continue, authenticate at {body['verification_uri_complete']}")
        print("*************************************************************************")
        token = poll_and_cache(body["device_code"], body.get("interval", 4))
        method = "device-flow"
    print(f"  auth method: {method}")
    print(f"  token: {token[:8]}... ({len(token)} chars)")

    if args.list:
        items = list_files(token)
        print(f"\nSAILOR (v1) - {len(items)} entries under dataset {DATASET_ID}:")
        total = 0
        for it in items:
            p, s, _ = _file_fields(it)
            total += s
            print(f"  {s/1e6:10.1f} MB  {p}")
        print(f"\n  total: {total/1e9:.2f} GB")
        return

    items = list_files(token)
    if args.match:
        pat = re.compile(args.match.replace("*", ".*").replace("?", ".") + "$")
        chosen = [it for it in items if pat.match(_file_fields(it)[0])]
    else:
        chosen = items
    if not chosen:
        print("no files matched")
        return

    print(f"\nDownloading {len(chosen)} file(s) -> {args.out}/")
    from ebrains_auth import get_token as refresh_token

    for it in chosen:
        p, _s, _ = _file_fields(it)
        dest = os.path.join(args.out, p.lstrip("/"))
        if os.path.exists(dest) and os.path.getsize(dest) == _s and _s:
            print(f"  skip (complete): {p}")
            continue
        download_file(refresh_token, it, dest)
    print("done")


if __name__ == "__main__":
    main()
