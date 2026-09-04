#!/usr/bin/env python3
"""File a controlled-access request for the SAILOR dataset via data-proxy.

POST /v1/datasets/{id} = "Request access for a dataset". Prints the raw
response (may include a ToS acceptance URL the user must visit).

Usage:
    python scripts/sailor_request_access.py [--redirect-uri https://example.com/done]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from ebrains_auth import get_token, issue_device_code, poll_and_cache
from sailor_fetch import DP_DATASET


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--redirect-uri", default=None)
    args = ap.parse_args()

    import requests

    try:
        token, method = get_token()
    except RuntimeError:
        body = issue_device_code()
        print("************************************************************************")
        print(f"To continue, authenticate at {body['verification_uri_complete']}")
        print("*************************************************************************")
        token = poll_and_cache(body["device_code"], body.get("interval", 4))
        method = "device-flow"
    print(f"auth method: {method}", flush=True)
    print(f"token: {token[:8]}... ({len(token)} chars)", flush=True)

    params = {}
    if args.redirect_uri:
        params["redirect_uri"] = args.redirect_uri
    r = requests.post(
        DP_DATASET,
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        timeout=30,
    )
    print(f"HTTP {r.status_code}", flush=True)
    print(r.text[:2000], flush=True)


if __name__ == "__main__":
    main()
