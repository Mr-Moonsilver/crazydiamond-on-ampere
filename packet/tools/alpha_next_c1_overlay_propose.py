#!/usr/bin/env python3
"""alpha-next-c1 spec aid: regenerate this tree's overlay under OUT and print the changed definitions per moved file.

tools/alpha_next_overlay_propose.py's steps, unchanged, through tools/build_alpha_next_c1.py (alpha-next's pinned pre-VM build as
the parent, its staging wrapper for the one added overlay module). The proposal names the moved engine files against alpha-next's
overlay, their changed definitions, and the files the regeneration adds (C1's e3/stage_overlap.py) or removes. A reviewer commits it
into the spec through tools/alpha_next_c1_spec.py; nothing is applied by this
tool. Scratch only: OUT must not exist; the held build is never touched. No GPU, no box action.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_alpha_next_c1 import b  # noqa: E402  (alpha-next's pinned build as the parent)


def propose(out):
    out = Path(out).resolve()
    out.mkdir()
    bundle = out / 'bundle'
    b.stage(dict(tools=[]), bundle)
    # Scratch only: the source pins of the review the gate chain prepares later (prepare_review's rule), so the generator's
    # review check sees this tree's harness. Its receipts are not read here, and the moved engine files do not depend on it.
    review_path = bundle / 'e3_builder/native_candidate_review.json'
    # tools/alpha_next_c1_review_pins.refreshed(): also the modules the committed review lacks (C1's stage_overlap.py, until the
    # pins commit before the chain adds it), within its declared ADDED.
    import alpha_next_c1_review_pins as pins
    assert json.loads(review_path.read_text()) == pins.refreshed()[0], 'staged review is not the committed one'
    b.dump(review_path, pins.refreshed()[1])
    b.regenerate_overlay(out, bundle)
    overlay = bundle / 'native30_overlay'
    rows, parent = b.verify(overlay), b.verify(b.PARENT / 'native30_overlay')
    moved = sorted(n for n in parent if rows.get(n) != parent[n])
    engine = sorted(n for n in moved if n not in b.METADATA)
    definitions = {n: b.changed_definitions((b.PARENT / 'native30_overlay' / n).read_text(), (overlay / n).read_text())
                   for n in engine}
    return dict(moved=moved, engine_changed=engine, overlay_definitions=definitions,
                added=sorted(set(rows) - set(parent)), removed=sorted(set(parent) - set(rows)))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(propose(a.out), indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
