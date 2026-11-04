#!/usr/bin/env python3
"""q18a-attn-meta spec aid: regenerate this tree's overlay under OUT and print the changed definitions per moved file.

tools/alpha_next_c1_overlay_propose.py's steps, unchanged, through tools/build_attn_meta.py (c1's pinned delivered build as the
parent, its staging wrapper for the two added overlay modules) and tools/attn_meta_review_pins.refreshed() for the scratch review.
The proposal names the moved engine files against c1's overlay, their changed definitions, and the files the regeneration adds
(e3/attn_meta_batch.py, e3/attn_meta_select.py) or removes. Nothing is applied. Scratch only: OUT must not exist. No GPU, no box.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_attn_meta import b  # noqa: E402  (c1's pinned build as the parent)


def propose(out):
    out = Path(out).resolve()
    out.mkdir()
    bundle = out / 'bundle'
    b.stage(dict(tools=[]), bundle)
    # Scratch only: the source pins of the review the gate chain prepares later (prepare_review's rule), so the generator's
    # review check sees this tree's harness. Its receipts are not read here, and the moved engine files do not depend on it.
    review_path = bundle / 'e3_builder/native_candidate_review.json'
    # tools/attn_meta_review_pins.refreshed(): also the modules the committed review lacks (attn-meta's two, until the pins
    # commit before the chain adds them), within its declared ADDED.
    import attn_meta_review_pins as pins
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
