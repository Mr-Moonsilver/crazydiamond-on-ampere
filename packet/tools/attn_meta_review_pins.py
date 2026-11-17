#!/usr/bin/env python3
"""Refresh the source and harness pins of patches/s5_engine/native_candidate_review.json to the committed tree, before the chain's strict.

tools/alpha_next_c1_review_pins.py for q18a-attn-meta, ADDED = attn-meta's two modules. Why: the tree keeps c1's review (generated,
never merged), which does not pin e3/attn_meta_batch.py or e3/attn_meta_select.py, now in make_s5_overlap.E3_MODULES; the generator
requires them pinned, and the attn-meta rows moved eight pinned engine sources. The rule is c1's tool's, unchanged in its body:
source_shas = every key the review holds plus the E3_MODULES it lacks (asserted within ADDED), each the committed file's sha256;
harness_source_shas = every tracked harness file. No other field moves. --check verifies the file. Run on a clean, committed tree, as
the last commit before the gate chain. tools/attn_meta_overlay_propose.py uses refreshed() for its scratch review only.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'patches/s5_engine/native_candidate_review.json'
ADDED = {'attn_meta_batch.py', 'attn_meta_select.py'}
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True)


def refreshed():
    import make_s5_overlap
    review = json.loads(REVIEW.read_text())
    required = {name + '.py' for name in make_s5_overlap.E3_MODULES}
    missing = required - set(review['source_shas'])
    assert missing <= ADDED and ADDED <= set(review['source_shas']) | missing, ('modules the review lacks', sorted(missing))
    out = dict(review)
    out['source_shas'] = {n: sha(ROOT / 'patches/s5_engine' / n) for n in sorted(set(review['source_shas']) | missing)}
    out['harness_source_shas'] = {str(Path(n).relative_to('harness')): sha(ROOT / n)
                                  for n in git('ls-files', 'harness').splitlines() if (ROOT / n).is_file()}
    assert set(out) == set(review)
    return review, out


if __name__ == '__main__':
    assert not git('status', '--porcelain', '--', 'harness', 'patches', 'tools').strip(), 'commit the tree first'
    before, after = refreshed()
    text = json.dumps(after, indent=2, sort_keys=True) + '\n'
    if '--check' in sys.argv:
        assert REVIEW.read_text() == text, 'review pins differ from the tree'
        print('review pins verified')
    else:
        REVIEW.write_text(text)
        moved = sorted(n for n in after['source_shas'] if before['source_shas'].get(n) != after['source_shas'][n])
        hmoved = sorted(n for n in after['harness_source_shas'] if before['harness_source_shas'].get(n) != after['harness_source_shas'][n])
        print(json.dumps(dict(source_moved=moved, harness_moved=len(hmoved), harness_dropped=sorted(
            set(before['harness_source_shas']) - set(after['harness_source_shas'])))))
