#!/usr/bin/env python3
"""Copy exact named opaque archive members to stdout; never decode MRI pixels."""
import argparse
from pathlib import Path
import sys
import tarfile

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--files0',type=Path,required=True);a=p.parse_args()
requested=set(x.decode() for x in a.files0.read_bytes().split(b'\0') if x)
seen=set();total=0
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as source,tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as destination:
    for member in source:
        if member.name not in requested:continue
        if not member.isfile() or member.name in seen or '..' in Path(member.name).parts:raise ValueError('Unexpected archived member')
        destination.addfile(member,source.extractfile(member));seen.add(member.name);total+=member.size
        print(f'TRANSFER {len(seen)}/{len(requested)} {total} bytes',file=sys.stderr,flush=True)
if seen!=requested:raise ValueError(f'Missing requested archive members: {sorted(requested-seen)}')
