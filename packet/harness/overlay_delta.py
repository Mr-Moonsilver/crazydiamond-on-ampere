#!/usr/bin/env python3
"""Require exact base + one pinned source change; hashes both complete manifests."""
import hashlib
from pathlib import Path
import sys

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(base,candidate,rel,digest):
    def files(root):
        return {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    assert files(base)==files(candidate),'overlay inventory differs'
    for root in (base,candidate):
        for m in (root/'SHA256SUMS',root/'core/SHA256SUMS'):
            for line in m.read_text().splitlines():
                h,n=line.split(maxsplit=1);p=m.parent/n
                assert p.resolve().is_relative_to(m.parent.resolve()) and sha(p)==h,'overlay checksum mismatch'
    for f in files(base)-{'SHA256SUMS','core/SHA256SUMS',rel}:
        assert sha(base/f)==sha(candidate/f),'extra changed file: '+f
    assert sha(candidate/rel)==digest,'candidate source mismatch'
if __name__=='__main__':
    try:check(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],sys.argv[4]);print('PASS exact overlay delta')
    except Exception as e:print(type(e).__name__+': invalid overlay delta');sys.exit(1)
