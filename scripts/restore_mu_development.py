#!/usr/bin/env python3
"""Restore only the fixed MU development patients with the official Aspera SDK."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--helpers", type=Path, required=True, help="Existing R62 package62.py helpers")
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    content = args.membership.read_bytes()
    sha = hashlib.sha256(content).hexdigest()
    if sha != "4df898761ecdb299bc87f99749fbebc8faab294043d454c4055850fcc676e2d6":
        raise ValueError("Membership differs from fixed R62 allocation")
    members = json.loads(content)
    patients = sorted(p for group in ("fitting", "selection", "pilot_scoring") for p in members[group])
    if len(patients) != 163 or set(patients) & set(members["reserved"]):
        raise ValueError("Invalid development membership")
    sys.path.insert(0, str(args.helpers.resolve()))
    from package62 import request, token
    data, _ = request(token(), "packages/1030/transfer_spec/download?type=received&transfer_type=connect",
                      {"paths": [{"path": "/MU-Glioma-Post/" + p} for p in patients]})
    sources = [p["source"] for p in data["paths"]]
    if len(sources) != len(patients) or {Path(p).name for p in sources} != set(patients):
        raise ValueError("Server returned paths outside development allowlist")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "membership.json").write_bytes(content)
    key = args.sdk / "etc/aspera_bypass_rsa.pem"
    if not key.exists():
        der = urllib.request.urlopen("https://raw.githubusercontent.com/IBM/aspera-cli/main/lib/aspera/data/2", timeout=60).read()
        subprocess.run(["openssl", "rsa", "-inform", "DER", "-out", str(key)], input=der, check=True)
        key.chmod(0o600)
    path_list = args.out / "development-transfer-paths.txt"
    path_list.write_text("".join(p + "\n" for p in sources))
    images = args.out / "images"
    images.mkdir(exist_ok=True)
    command = [str(args.sdk / "bin/ascp"), "-i", str(key), "-P", str(data["ssh_port"]),
               "-O", str(data["fasp_port"]), "--mode=recv", "--host="+data["remote_host"],
               "--user="+data["remote_user"], "--policy=fair", "-l", "100m", "-k", "2",
               "--file-list="+str(path_list), str(images)]
    print("Restoring 163 development patients; 40 reserved paths excluded", flush=True)
    result = subprocess.run(command, env=dict(os.environ, ASPERA_SCP_TOKEN=data["token"]))
    (args.out / "transfer-receipt.json").write_text(json.dumps({"returncode": result.returncode,
        "membership_sha256": sha, "development_patients": 163, "reserved_patients_requested": 0}, indent=2)+"\n")
    result.check_returncode()


if __name__ == "__main__":
    main()
