#!/usr/bin/env python3
"""Negative control for the alpha-next-c1 freeze's overlay check: a one-byte change in an overlay file must fail overlay_level.

C0's control (tools/alpha_c0_overlay_tamper.py) imported unchanged with its freeze module set to tools/alpha_next_c1_freeze.py: the
same cases (T0 untampered, T1 a comment byte in every changed or new runtime module, T2 an integer constant in the new module
e3/stage_overlap.py, T3 a stale checksum list, T4-T7 a byte outside the declared change set, T8 restored), each naming the
refusal it expects. It works on a temporary copy of the built overlay and leaves the build untouched. (tools/alpha_c1_overlay_tamper.py
with the freeze module and record schema renamed; the parent is the regenerated alpha-next overlay.)

Usage: alpha_next_c1_overlay_tamper.py --parent PARENT/native30_overlay --candidate BUNDLE/native30_overlay --out JSON
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alpha_c0_overlay_tamper as c0  # noqa: E402
import alpha_next_c1_freeze as freeze  # noqa: E402

c0.freeze = freeze


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    rows = c0.run(a.parent, a.candidate, sys.stdout)
    names = {r['case'] for r in rows}
    ok = all(r['ok'] for r in rows) and any(r['case'] == 'T2-constant-in-a-new-module' and r['path'] == 'e3/stage_overlap.py'
                                            for r in rows) and {'T0-untampered-copy-accepted', 'T8-restored-copy-accepted'} <= names
    result = dict(schema='alpha-next-c1-overlay-tamper/1', verdict='PASS' if ok else 'FAIL', cases=rows,
                  derived_from='tools/alpha_c0_overlay_tamper.py over tools/alpha_next_c1_freeze.py',
                  candidate_manifest=c0.sha(a.candidate / 'SHA256SUMS'), parent_manifest=c0.sha(a.parent / 'SHA256SUMS'))
    with a.out.open('x') as handle:
        json.dump(result, handle, indent=1, sort_keys=True)
        handle.write('\n')
    print('OVERLAY_TAMPER %s %d/%d cases as expected' % (result['verdict'], sum(r['ok'] for r in rows), len(rows)))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
