#!/usr/bin/env python3
"""LUMIERE ranged zip downloader.

The LUMIERE dataset (91 GBM patients, weekly MRI during chemo-RT) is a single
32.6 GB zip on Figshare, CC0, no authentication:

    https://ndownloader.figshare.com/files/38249697

A normal download needs 32 GB of bandwidth. This tool instead reads the zip's
CENTRAL DIRECTORY via HTTP Range requests (tens of KB), then fetches ONLY the
byte ranges of the entries you want (e.g. 5 patients' NIfTI = ~1-2 GB), so you
can start working long before - or without ever - pulling the whole archive.

Requires: python3 + curl (uses `curl -sL -r` under the hood; handles the
ndownloader redirect + resume naturally).

Examples
--------
python3 lumiere_fetch.py --list                      # print archive contents
python3 lumiere_fetch.py --n 5 --out data/lumiere    # 5 patients' images
python3 lumiere_fetch.py --patients Patient-001 Patient-002 --out data/lumiere
python3 lumiere_fetch.py --full --out data/lumiere   # entire 32.6 GB archive
"""
from __future__ import annotations

import argparse
import binascii
import io
import os
import re
import subprocess
import sys
import zlib

URL = "https://ndownloader.figshare.com/files/38249697"
EOCD_SIG = b"PK\x05\x06"
CD_SIG = b"PK\x01\x02"
LH_SIG = b"PK\x03\x04"


def fetch_range(start: int, end: int, retries: int = 3) -> bytes:
    """Fetch bytes [start, end] inclusive. Uses curl -L (the ndownloader URL
    is a 302 to an S3 presigned link) and plain ranged GET, which S3 honours
    (HEAD gets 403 - never use -I here)."""
    for attempt in range(retries):
        r = subprocess.run(
            ["curl", "-sL", "-r", f"{start}-{end}", URL], capture_output=True)
        if r.returncode == 0 and r.stdout:
            return r.stdout
    raise RuntimeError(f"range fetch failed {start}-{end} after {retries} tries")


def total_size() -> int:
    """Learn the archive size from the Content-Range header of a suffix range
    request (bytes=-65536 fetches the last 64 KiB of the file)."""
    r = subprocess.run(["curl", "-sL", "-r", "-65536", "-D", "-", "-o", "/dev/null",
                        URL], capture_output=True, text=True)
    m = re.search(r"(?i)content-range:\s*bytes\s*\d+-\d+/(\d+)", r.stdout)
    if not m:
        raise RuntimeError(f"could not learn size; headers:\n{r.stdout[:300]}")
    return int(m.group(1))


CHUNK = 8 * 1024 * 1024  # each ranged fetch stays small vs the 10 s presign expiry


def fetch_long(start: int, length: int) -> bytes:
    """Fetch `length` bytes starting at `start`, chunked, so no single curl call
    outlives the presigned URL's short lifetime."""
    out = bytearray()
    pos = start
    while len(out) < length:
        n = min(CHUNK, length - len(out))
        out += fetch_range(pos, pos + n - 1)
        pos += n
    return bytes(out)


class RemoteZip:
    """Minimal reader for a ZIP file that lives on an HTTP server."""

    def __init__(self, url=URL):
        self.url = url
        size = total_size()
        tail = fetch_range(max(0, size - 65_536), size - 1)
        idx = tail.rfind(EOCD_SIG)
        if idx < 0:
            raise RuntimeError("EOCD not found")
        eocd = tail[idx:]
        n_entries = int.from_bytes(eocd[10:12], "little")
        cd_size = int.from_bytes(eocd[12:16], "little")
        cd_off = int.from_bytes(eocd[16:20], "little")
        # ZIP64: values marked 0xFFFFFFFF/0xFFFF live in the ZIP64 EOCD record,
        # whose absolute offset is stored in the locator (PK\x06\x07) that sits
        # immediately before the classic EOCD.
        if cd_off == 0xFFFFFFFF or n_entries == 0xFFFF:
            loc = tail[idx - 20:idx]
            if loc[:4] != b"PK\x06\x07":
                raise RuntimeError("ZIP64 locator not found")
            z64_off = int.from_bytes(loc[8:16], "little")
            z64 = fetch_range(z64_off, z64_off + 55)
            n_entries = int.from_bytes(z64[24:32], "little")
            cd_size = int.from_bytes(z64[40:48], "little")
            cd_off = int.from_bytes(z64[48:56], "little")
        # --- central directory ---------------------------------------------
        cd = fetch_long(cd_off, cd_size)
        self.entries, pos = [], 0
        for _ in range(n_entries):
            if cd[pos:pos + 4] != CD_SIG:
                raise RuntimeError("central directory parse error")
            method = int.from_bytes(cd[pos + 10:pos + 12], "little")
            crc = int.from_bytes(cd[pos + 16:pos + 20], "little")
            comp = int.from_bytes(cd[pos + 20:pos + 24], "little")
            orig = int.from_bytes(cd[pos + 24:pos + 28], "little")
            name_len = int.from_bytes(cd[pos + 28:pos + 30], "little")
            extra_len = int.from_bytes(cd[pos + 30:pos + 32], "little")
            comment_len = int.from_bytes(cd[pos + 32:pos + 34], "little")
            lho = int.from_bytes(cd[pos + 42:pos + 46], "little")
            name = cd[pos + 46:pos + 46 + name_len].decode("utf-8", "replace")
            # ZIP64 extra field (id 0x0001): 8-byte values, present only for
            # the 32-bit fields that were capped at 0xFFFFFFFF / 0xFFFF.
            extra = cd[pos + 46 + name_len: pos + 46 + name_len + extra_len]
            epos = 0
            while epos + 4 <= len(extra):
                eid = int.from_bytes(extra[epos:epos + 2], "little")
                elen = int.from_bytes(extra[epos + 2:epos + 4], "little")
                if eid == 0x0001:
                    ev = extra[epos + 4: epos + 4 + elen]
                    vi = 0
                    if comp == 0xFFFFFFFF and vi + 8 <= len(ev):
                        comp = int.from_bytes(ev[vi:vi + 8], "little"); vi += 8
                    if orig == 0xFFFFFFFF and vi + 8 <= len(ev):
                        orig = int.from_bytes(ev[vi:vi + 8], "little"); vi += 8
                    if lho == 0xFFFFFFFF and vi + 8 <= len(ev):
                        lho = int.from_bytes(ev[vi:vi + 8], "little"); vi += 8
                epos += 4 + elen
            self.entries.append(dict(name=name, method=method, crc=crc,
                                     comp=comp, lho=lho))
            pos += 46 + name_len + extra_len + comment_len

    def _find_size(self) -> int:
        return total_size()

    @property
    def roots(self) -> set:
        return {e["name"].split("/")[0] for e in self.entries if not e["name"].endswith("/")}

    def patients(self) -> list:
        pats = set()
        for e in self.entries:
            parts = e["name"].split("/")
            if len(parts) >= 2 and parts[1].startswith("Patient-"):
                pats.add(parts[1])
        return sorted(pats)

    def patient_entries(self, patient: str) -> list:
        return [e for e in self.entries if f"/{patient}/" in f"/{e['name']}"
                and not e["name"].endswith("/")]

    def extract(self, entry: dict, dest: str) -> bytes:
        """Fetch one entry's compressed bytes and decompress (stored/deflate)."""
        lo = entry["lho"]
        # read local header to locate data start
        lh = fetch_range(lo, lo + 30)
        if lh[:4] != LH_SIG:
            raise RuntimeError(f"bad local header {entry['name']}")
        nl = int.from_bytes(lh[26:28], "little")
        el = int.from_bytes(lh[28:30], "little")
        data_start = lo + 30 + nl + el
        raw = fetch_long(data_start, entry["comp"])
        if entry["method"] == 0:            # stored
            data = raw
        elif entry["method"] == 8:          # deflate
            data = zlib.decompressobj(-15).decompress(raw)
        else:
            raise RuntimeError(f"unsupported method {entry['method']}")
        if binascii.crc32(data) & 0xFFFFFFFF != entry["crc"]:
            raise RuntimeError(f"CRC mismatch {entry['name']}")
        out = os.path.join(dest, entry["name"].replace("\\", os.sep))
        os.makedirs(os.path.dirname(out) or dest, exist_ok=True)
        with open(out, "wb") as f:
            f.write(data)
        return data


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--list", action="store_true", help="print archive contents")
    g.add_argument("--patients", nargs="+", help="patient IDs, e.g. Patient-001")
    g.add_argument("--n", type=int, help="first N patients")
    g.add_argument("--full", action="store_true", help="download entire archive")
    ap.add_argument("--out", default="data/lumiere", help="destination dir")
    args = ap.parse_args()

    print("Reading remote zip central directory (a few hundred KB)...")
    z = RemoteZip(URL)
    print(f"  root: {sorted(z.roots)}")
    pats = z.patients()
    print(f"  patients available: {len(pats)}  ({pats[0]} .. {pats[-1]})")

    if args.list:
        for e in z.entries:
            print(f"  {e['comp']/1e6:8.1f} MB  {e['name']}")
        return

    if args.patients:
        want = args.patients
    elif args.n:
        want = pats[:args.n]
    elif args.full:
        # plain resumable download of the whole file
        os.makedirs(args.out, exist_ok=True)
        dest = os.path.join(args.out, "LUMIERE_Imaging.zip")
        print(f"Downloading 32.6 GB to {dest} (Ctrl+C-safe: curl resumes with -C -)")
        r = subprocess.run(["curl", "-sL", "-C", "-", "-o", dest, URL])
        print("done" if r.returncode == 0 else "download interrupted")
        return
    else:
        ap.error("one of --list/--patients/--n/--full required")

    total = 0
    for p in want:
        entries = z.patient_entries(p)
        if not entries:
            print(f"  !! no entries for {p}")
            continue
        for e in entries:
            data = z.extract(e, args.out)
            total += len(data)
            print(f"  {e['name']:60s} {len(data)/1e6:7.1f} MB  crc ok")
    print(f"\nExtracted {len(want)} patients, {total/1e9:.2f} GB -> {args.out}/")
    print("Tip: the readme/CSVs already fetched live in data/lumiere_meta/.")


if __name__ == "__main__":
    main()
