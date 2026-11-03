#!/usr/bin/env python3
"""Refresh the source and harness pins of patches/s5_engine/native_candidate_review.json to the committed tree, before the chain's strict.

tools/alpha_next_c1_review_pins.py for alpha-next-c1b, imported and called unchanged with ADDED = set(): c1b adds no engine module,
so the review must already pin every make_s5_overlap.E3_MODULES module (stage_overlap.py included, from c1) and only the harness
pins (and any source pin of a moved engine file, of which c1b has none) refresh. Why it is needed: build_native43.prepare_review
refreshes only at the chain's prepare-review step, after strict; the c1b harness changes move harness_source_shas, which the
generator checks (make_s5_overlap._native_candidate_closure: 'native harness source changed after review'), and the bundle
self-run's review check reads the committed review. --check verifies the file. Run on a clean, committed tree, as the last commit
before the gate chain. tools/alpha_next_c1b_overlay_propose.py uses refreshed() for its scratch review only. No GPU, no box action.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alpha_next_c1_review_pins as base  # noqa: E402

ADDED = set()
REVIEW = base.REVIEW


def refreshed():
    saved, base.ADDED = base.ADDED, ADDED
    try:
        return base.refreshed()
    finally:
        base.ADDED = saved


def main(argv):
    assert not base.git('status', '--porcelain', '--', 'harness', 'patches', 'tools').strip(), 'commit the tree first'
    before, after = refreshed()
    moved = sorted(n for n in after['source_shas'] if before['source_shas'].get(n) != after['source_shas'][n])
    assert not moved, ('engine source pins move in a harness-only package', moved)
    text = json.dumps(after, indent=2, sort_keys=True) + '\n'
    if '--check' in argv:
        assert REVIEW.read_text() == text, 'review pins differ from the tree'
        print('review pins verified')
        return 0
    REVIEW.write_text(text)
    hmoved = sorted(n for n in after['harness_source_shas'] if before['harness_source_shas'].get(n) != after['harness_source_shas'][n])
    print(json.dumps(dict(source_moved=moved, harness_moved=len(hmoved), harness_dropped=sorted(
        set(before['harness_source_shas']) - set(after['harness_source_shas'])))))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
