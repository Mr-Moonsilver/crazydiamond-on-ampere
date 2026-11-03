#!/usr/bin/env python3
"""Negative control for the alpha-next-c1b freeze's overlay check: a one-byte change in an overlay file must be refused.

c1b's overlay check is build_alpha_next_c1b.metadata_review_only (the swapped build_native43 assertion: no non-METADATA row moved,
moved METADATA only in review-bound keys). On temporary copies of the built candidate overlay (manifest rewritten each time, so
only the content check can catch it): the untouched copy is accepted; one appended byte in each of TARGETS (an engine module, C1's
stage_overlap.py, a compiled library, the launcher, an E3 source, the readiness record's non-review key) is refused. The build is
never touched. usage: alpha_next_c1b_overlay_tamper.py --parent PARENT/native30_overlay --candidate BUNDLE/native30_overlay --out JSON
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_alpha_next_c1b as c1b  # noqa: E402

TARGETS = ('e3/engine_driver.py', 'e3/stage_overlap.py', 'e3/native/flash_s5_green.so', 'run-flash-s5-test.sh',
           'e3_sources/model.py')


def tamper(parent, candidate):
    parent, candidate = Path(parent).resolve(), Path(candidate).resolve()
    rows, ok = {}, True
    with tempfile.TemporaryDirectory(prefix='c1b-tamper-') as tmp:
        def copy(tag):
            place = Path(tmp) / tag
            shutil.copytree(candidate, place)
            c1b._writable(place)
            return place
        clean = copy('clean')
        c1b.b.manifest(clean)
        problems = c1b.metadata_review_only(parent, clean)
        rows['untouched'] = dict(expected='ACCEPTED', got='REFUSED' if problems else 'ACCEPTED', problems=problems)
        ok &= not problems
        for i, rel in enumerate(TARGETS):
            place = copy('t%d' % i)
            (place / rel).write_bytes((place / rel).read_bytes() + b'\0')
            c1b.b.manifest(place)
            problems = c1b.metadata_review_only(parent, place)
            rows[rel] = dict(expected='REFUSED', got='REFUSED' if problems else 'ACCEPTED', problems=problems)
            ok &= bool(problems)
        place = copy('readiness-key')
        path = place / 'e3/native-overlap-readiness.json'
        data = json.loads(path.read_text())
        data['c1b_undeclared'] = 1
        path.write_text(json.dumps(data, sort_keys=True, indent=2) + '\n')
        c1b.b.manifest(place)
        problems = c1b.metadata_review_only(parent, place)
        rows['e3/native-overlap-readiness.json:non-review-key'] = dict(expected='REFUSED', got='REFUSED' if problems else 'ACCEPTED',
                                                                       problems=problems)
        ok &= bool(problems)
    return dict(schema='alpha-next-c1b-overlay-tamper/1', verdict='PASS' if ok else 'FAIL', cases=rows, parent=str(parent),
                candidate=str(candidate), derived_from='build_alpha_next_c1b.metadata_review_only (the swapped overlay check)')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = tamper(a.parent, a.candidate)
    with a.out.open('x') as handle:
        handle.write(json.dumps(result, sort_keys=True, indent=1) + '\n')
    print(result['verdict'], {k: v['got'] for k, v in result['cases'].items()})
    raise SystemExit(0 if result['verdict'] == 'PASS' else 1)
