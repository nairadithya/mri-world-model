#!/usr/bin/env python3
"""Headless screenshot validator for scripts/view_scans.py.

Renders the local scan explorer in a headless Chromium (WebGL via SwiftShader)
and saves a PNG — so viewer changes can be verified without a display. Also
prints the loaded volume list + status, which is how the NiiVue API gotchas in
AGENTS.md were found.

This is a DEV tool, not a runtime dependency: it needs playwright, which is
NOT in requirements.txt. Install once:

    pip install playwright && playwright install chromium

Usage (viewer running in another shell):
    python scripts/view_scans.py --no-browser &          # serve on :8765
    python scripts/shot_viewer.py \
      "http://127.0.0.1:8765/?p=sub-01/ses-05/T1c.nii.gz&ov=sub-01/ses-05/EdemaMask-ONCO.nii.gz" \
      /tmp/opencode/shot.png

Exit codes: 0 ok, 2 playwright missing, 3 navigation/render failed.
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="viewer URL, deep-link params supported")
    ap.add_argument("out", help="output PNG path")
    ap.add_argument("--wait", type=float, default=90.0,
                    help="max seconds to wait for volumes to load")
    ap.add_argument("--settle", type=float, default=6.0,
                    help="extra seconds after volumes load before capture")
    ap.add_argument("--size", type=int, nargs=2, default=[1500, 950])
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed — run:\n"
              "  pip install playwright && playwright install chromium")
        return 2

    args_list = ["--no-sandbox", "--enable-unsafe-swiftshader",
                 "--use-gl=angle", "--use-angle=swiftshader",
                 "--ignore-gpu-blocklist", "--disable-dev-shm-usage"]
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=args_list, channel="chromium")
        pg = b.new_page(viewport={"width": args.size[0], "height": args.size[1]})
        pg.on("pageerror", lambda e: print("PAGEERR:", (e.stack or str(e))[:400]))
        pg.goto(args.url, wait_until="load", timeout=60000)

        vols, status = 0, ""
        waited = 0.0
        while waited < args.wait:
            pg.wait_for_timeout(1000)
            waited += 1
            vols = pg.evaluate("()=>nv.volumes.length")
            if vols > 0:
                break
        pg.wait_for_timeout(int(args.settle * 1000))
        vols = pg.evaluate("()=>nv.volumes.length")
        status = pg.evaluate("()=>document.getElementById('status').textContent")
        pg.screenshot(path=args.out)
        b.close()

    print(f"volumes: {vols} | status: {status} | {args.out}")
    if vols == 0:
        print("WARNING: no volumes loaded — check the deep-link rel paths")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
