#!/usr/bin/env python3
"""The alpha-perf held bundle, freeze proof and owner runner: build_native43.py, unchanged, with held-f build-b as parent.

build_native43's text is never edited (a past queue ran it). Declared configuration, as module constants, because held-b is
the wrong parent here: the composed engine carries e3/native_auto_arm.py, an overlay file held-b does not have, and the
runner of this queue is held-f's. Parent = held-f build-b (bundle 15dffe62e5d4, runner d57375eb, source 6e36495). Candidate
name q18a-alpha-perf is admitted by a load_spec with the wider name rule; every other assertion, the freeze, the overlay
regeneration, the negative controls and the runner normalisation are build_native43's.
No GPU, no box action.
"""
import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_native43 as b  # noqa: E402

HELD = Path('<hub-dir>/agents/flash-next/tmp/native41-progress-held-f/build-b')
SOURCE = '6e36495a19b81e2cad7919cc0a6a6123cabdb04e'
b.HELD = HELD
b.PARENT = HELD / 'q18a-native41-progress-bundle-15dffe62e5d4'
b.PARENT_MANIFEST = '15dffe62e5d425c66c06f90a3387f624ebe618ab181d71141f60c881b192c7c8'
b.PARENT_OVERLAY_MANIFEST = '262d3a05a185d733299a6888dc4b6d20ce58ba892c568ff263e9fd17d1d338bb'
b.PARENT_REVIEW = '0ee53fd11e138c84104a9804f0a05f491c7e95b499294cb7b457ed74599f6d41'
b.PARENT_BUNDLE_SOURCE = SOURCE
b.PARENT_COMMIT = SOURCE
b.PARENT_CANDIDATE = 'q18a-native41-progress'
b.PARENT_STEM = 'q18a_native41_progress'
b.PARENT_RUNNER = HELD / 'q18a-native41-progress-owner-run-15dffe62e5d4.py'
b.PARENT_RUNNER_SHA = 'd57375eb9407ff114a0e0a6cdf39d474099f913cd2bc7a4355690c5f91602ea6'
b.PARENT_TESTS = 263


def load_spec(path):
    spec = json.loads(Path(path).read_text())
    assert spec.get('schema') == 'native43-packaging-spec/1', 'spec schema'
    assert set(spec) == b.SPEC_KEYS, sorted(set(spec) ^ b.SPEC_KEYS)
    assert spec['candidate'] == 'q18a-alpha-perf' and spec['review'] == 'prepare' and spec['dry_run'] is False
    assert re.fullmatch(r'[0-9a-f]{40}', spec['source_commit'])
    return spec


b.load_spec = load_spec


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--selftest-log', type=Path)
    p.add_argument('--focused-log', type=Path)
    p.add_argument('--prepare-review', action='store_true')
    p.add_argument('--propose', action='store_true')
    a = p.parse_args()
    if a.propose:
        print(json.dumps(b.propose(b.load_spec(a.spec)), indent=2, sort_keys=True))
    elif a.prepare_review:
        b.dump(b.ROOT / b.REVIEW_PATH, b.prepare_review(b.load_spec(a.spec), a.selftest_log, a.focused_log))
    else:
        assert a.out is not None and a.selftest_log is not None and a.focused_log is not None
        b.build(a.spec, a.out.resolve(), a.selftest_log.resolve(), a.focused_log.resolve())
