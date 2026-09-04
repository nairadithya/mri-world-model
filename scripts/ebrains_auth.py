#!/usr/bin/env python3
"""Shared EBRAINS auth: .env token cache + refresh grant + scoped device flow.

.env variables (all optional, gitignored):
    KG_TOKEN           bearer access token (short-lived; used as-is)
    KG_REFRESH_TOKEN   long-lived refresh token (auto-exchanged, re-cached)
    KG_CLIENT_ID / KG_CLIENT_SECRET   service-account credentials

Precedence: KG_TOKEN > KG_REFRESH_TOKEN > client credentials > device flow.
Successful grants persist KG_REFRESH_TOKEN (+ fresh KG_TOKEN) to .env so the
browser dance happens at most once.
"""
from __future__ import annotations

import os
import time

CLIENT_ID = "kg-core-python"
WELL_KNOWN = "https://iam.ebrains.eu/auth/realms/hbp/.well-known/openid-configuration"
ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
SCOPES = "openid roles email team profile"


def load_env(path: str = ENV_PATH) -> None:
    """Minimal .env loader (no extra dependency). Does not override real env."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip("'\"")
            if k and k not in os.environ:
                os.environ[k] = v


def save_env(updates: dict, path: str = ENV_PATH) -> None:
    """Upsert keys in .env (mode 0600 — secrets)."""
    lines: dict[str, str] = {}
    order: list[str] = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.rstrip("\n").split("=", 1)
                    lines[k] = v
                    order.append(k)
    for k, v in updates.items():
        if k not in lines:
            order.append(k)
        lines[k] = v
    with open(path, "w") as f:
        for k in order:
            f.write(f"{k}={lines[k]}\n")
    os.chmod(path, 0o600)


def _token_endpoint() -> str:
    import requests

    wk = requests.get(WELL_KNOWN, timeout=30).json()
    return wk["token_endpoint"]


def refresh_access_token(refresh_token: str) -> dict:
    """Exchange a refresh token for a fresh token set (Keycloak, public client)."""
    import requests

    r = requests.post(
        _token_endpoint(),
        data={
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "refresh_token": refresh_token,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def cache_token_set(body: dict) -> str:
    """Persist refresh + access tokens to .env AND os.environ.

    Syncing os.environ matters: a long-lived process that cached a new grant
    must not keep refreshing with the stale pre-grant token (Keycloak rotates
    refresh tokens, so reuse of the old one 400s).
    """
    updates = {"KG_TOKEN": body["access_token"]}
    if body.get("refresh_token"):
        updates["KG_REFRESH_TOKEN"] = body["refresh_token"]
    save_env(updates)
    os.environ.update(updates)
    return body["access_token"]


def get_token() -> tuple[str, str]:
    """Return (access_token, method), using cache/refresh before device flow."""
    load_env()
    if os.environ.get("KG_TOKEN") and not os.environ.get("KG_REFRESH_TOKEN"):
        return os.environ["KG_TOKEN"], "token"  # user-supplied, don't touch
    if os.environ.get("KG_REFRESH_TOKEN"):
        try:
            body = refresh_access_token(os.environ["KG_REFRESH_TOKEN"])
            return cache_token_set(body), "refresh"
        except Exception as e:
            print(f"  refresh failed ({e}); falling through to device flow")
    cid, secret = os.environ.get("KG_CLIENT_ID"), os.environ.get("KG_CLIENT_SECRET")
    if cid and secret:
        from kg_core.oauth import ClientCredentials

        tok = ClientCredentials(cid, secret).get_token()
        if not tok:
            raise RuntimeError("client-credentials produced no token")
        return tok, "client-credentials"
    raise RuntimeError("no cached credentials; use device flow (issue_device_code)")


def issue_device_code(scopes: str = SCOPES) -> dict:
    """Start a scoped device flow. Returns Keycloak's device-authorization body."""
    import requests

    wk = requests.get(WELL_KNOWN, timeout=30).json()
    r = requests.post(
        wk["device_authorization_endpoint"],
        data={"client_id": CLIENT_ID, "scope": scopes},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def poll_and_cache(device_code: str, interval: int = 4, timeout: int = 290) -> str:
    """Poll until approved; cache the token set to .env. Network-error tolerant."""
    import requests

    ep = _token_endpoint()
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.post(
                ep,
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "client_id": CLIENT_ID,
                    "device_code": device_code,
                },
                timeout=30,
            )
        except Exception as e:
            print(f"poll: network error ({type(e).__name__}); retrying...", flush=True)
            time.sleep(interval)
            continue
        if r.status_code == 200:
            body = r.json()
            print(f"granted; scope={body.get('scope')!r}", flush=True)
            return cache_token_set(body)
        try:
            err = r.json().get("error", r.text[:100])
        except Exception:
            err = r.text[:100]
        if err == "expired_token":
            raise RuntimeError("device code expired before approval")
        print(f"poll: {r.status_code} {err} (retrying...)", flush=True)
        time.sleep(interval)
    raise RuntimeError("device code expired before approval")
