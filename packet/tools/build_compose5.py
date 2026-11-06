#!/usr/bin/env python3
"""The q18a-compose-throughput-b24 held bundle and owner runner: tools/build_inwait.py's configuration, unchanged (build_native43 with alpha-next-c1b's READY build as parent), for this candidate. Discovery mode only:
the two logs are the parent tree's and the records say so, no freeze proof, no parent reference. No GPU, no box action."""
import argparse
import json
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import build_inwait as parent  # noqa: E402

b, PINS, TOOLS_REPLACED, NEW_OVERLAY = parent.b, parent.PINS, parent.TOOLS_REPLACED, parent.NEW_OVERLAY
CANDIDATE = 'q18a-compose-throughput-b24'
parent.CANDIDATE = CANDIDATE    # the one name build_inwait.load_spec holds a spec to

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--selftest-log', type=Path, required=True)
    p.add_argument('--focused-log', type=Path, required=True)
    p.add_argument('--prepare-review', action='store_true')
    a = p.parse_args()
    parent.reuse_parent_logs()
    if a.prepare_review:
        b.dump(b.ROOT / b.REVIEW_PATH, b.prepare_review(b.load_spec(a.spec), a.selftest_log, a.focused_log))
    else:
        assert a.out is not None
        parent.build_discovery(a.spec, a.out.resolve(), a.selftest_log.resolve(), a.focused_log.resolve())
