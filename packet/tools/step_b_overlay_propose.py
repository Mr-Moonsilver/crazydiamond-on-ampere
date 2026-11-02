#!/usr/bin/env python3
"""Step B spec aid: regenerate this tree's overlay under OUT and print the changed definitions per moved file.

tools/step_a_overlay_propose.py's steps against the frozen Step A parent overlay. A reviewer commits the proposal into the spec;
nothing is applied by this tool. No GPU, no box action.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_step_b import b  # noqa: E402  (Step A's frozen build as the parent)


def propose(out):
    out = Path(out).resolve()
    out.mkdir()
    bundle = out / 'bundle'
    b.stage(dict(tools=[]), bundle)
    # Scratch only: the source pins of the review the gate chain prepares later (prepare_review's rule), so the generator's
    # review check sees this tree's harness. Its receipts are not read here, and the moved engine files do not depend on it.
    review_path = bundle / 'e3_builder/native_candidate_review.json'
    review = json.loads(review_path.read_text())
    review['source_shas'] = {n: b.sha(b.ROOT / 'patches/s5_engine' / n) for n in review['source_shas']}
    review['harness_source_shas'] = {str(Path(n).relative_to('harness')): b.sha(b.ROOT / n)
                                     for n in b.git('ls-files', 'harness').splitlines() if (b.ROOT / n).is_file()}
    b.dump(review_path, review)
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
