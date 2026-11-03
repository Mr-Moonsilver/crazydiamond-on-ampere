#!/usr/bin/env python3
"""q18a-stage1-step spec aid: regenerate this tree's overlay under OUT and print the changed definitions per moved file.

tools/attn_meta_overlay_propose_b.py's steps by name, through tools/build_stage1_step.py (c1b's pinned READY build as the parent) and
tools/stage1_step_review_pins.refreshed() for the scratch review. The proposal names the moved engine files against c1b's overlay, their changed definitions, and the files
the regeneration adds or removes (nothing added, nothing removed). Nothing is applied. Scratch only: OUT must not exist. No GPU, no box.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_stage1_step import b  # noqa: E402  (c1b's pinned build as the parent)


def propose(out):
    out = Path(out).resolve()
    out.mkdir()
    bundle = out / 'bundle'
    b.stage(dict(tools=[]), bundle)
    # Scratch only: the source pins of the review the gate chain prepares later (prepare_review's rule), so the generator's
    # review check sees this tree's harness. Its receipts are not read here, and the moved engine files do not depend on it.
    review_path = bundle / 'e3_builder/native_candidate_review.json'
    # tools/stage1_step_review_pins.refreshed(): its ADDED is empty here.
    import stage1_step_review_pins as pins
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
