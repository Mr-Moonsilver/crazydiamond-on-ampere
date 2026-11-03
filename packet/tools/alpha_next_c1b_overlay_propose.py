#!/usr/bin/env python3
"""alpha-next-c1b spec aid: regenerate this tree's overlay under OUT and report what moved against alpha-next-c1's overlay.

tools/alpha_next_c1_overlay_propose.py's steps through tools/build_alpha_next_c1b.py (alpha-next-c1's delivered build-b as the
parent). For c1b the expected answer is: nothing added or removed, no engine row moved (engine_changed [], overlay_definitions {}),
and any moved row a METADATA row moved only in review-bound keys (build_alpha_next_c1b.metadata_review_only). The scratch review is
tools/alpha_next_c1b_review_pins.refreshed() (this tree's harness pins); --simulate-prepare also adds a scratch alpha_next_c1b key
to it, as the chain's prepare-review will, so the metadata movement the build will see is exercised before the chain. Scratch only:
OUT must not exist; the held build is never touched. No GPU, no box action.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_alpha_next_c1b as c1b  # noqa: E402
from build_alpha_next_c1b import b  # noqa: E402


def propose(out, simulate_prepare=False):
    out = Path(out).resolve()
    out.mkdir()
    bundle = out / 'bundle'
    b.stage(dict(tools=[]), bundle)
    review_path = bundle / 'e3_builder/native_candidate_review.json'
    import alpha_next_c1b_review_pins as pins
    before, review = pins.refreshed()
    assert json.loads(review_path.read_text()) == before, 'staged review is not the committed one'
    if simulate_prepare:
        review = dict(review, alpha_next_c1b=dict(scratch='alpha_next_c1b_overlay_propose --simulate-prepare'))
    b.dump(review_path, review)
    b.regenerate_overlay(out, bundle)
    overlay = bundle / 'native30_overlay'
    rows, parent = b.verify(overlay), b.verify(b.PARENT / 'native30_overlay')
    moved = sorted(n for n in parent if rows.get(n) != parent[n])
    engine = sorted(n for n in moved if n not in b.METADATA)
    definitions = {n: b.changed_definitions((b.PARENT / 'native30_overlay' / n).read_text(), (overlay / n).read_text())
                   for n in engine}
    return dict(moved=moved, engine_changed=engine, overlay_definitions=definitions,
                added=sorted(set(rows) - set(parent)), removed=sorted(set(parent) - set(rows)),
                metadata_review_only_problems=c1b.metadata_review_only(b.PARENT / 'native30_overlay', overlay, review),
                simulate_prepare=simulate_prepare)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--simulate-prepare', action='store_true')
    a = p.parse_args()
    print(json.dumps(propose(a.out, a.simulate_prepare), indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
