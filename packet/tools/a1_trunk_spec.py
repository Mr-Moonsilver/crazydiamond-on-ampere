#!/usr/bin/env python3
"""Write tools/native43_spec.a1-trunk.json for HEAD: tools/inwait_spec.py's generation (c1b's READY build as the parent, source_paths, definitions, the overlay proposal, the runner's case list) with this family's names over it. No GPU, no box action."""
import argparse
import json
from pathlib import Path
import re
import sys

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import build_a1_trunk  # noqa: E402,F401  (sets the candidate first)
import a1_trunk_engine_selfrun as selfrun  # noqa: E402
import inwait_spec as parent  # noqa: E402

NAME = 'native43_spec.a1-trunk.json'
QUEUE = ['q18a_a1_trunk.draft.sh', 'q18a_a1_trunk_declare.sh', 'q18a_a1_trunk_settings.sh']
BOUNDS = ('FLASH_S5_APPEND_YIELD_MAX_NEW', 'FLASH_S5_APPEND_YIELD_PER_BULK', 'FLASH_S5_APPEND_YIELD_HANDBACK_STEPS', 'FLASH_S5_APPEND_YIELD_REENTRY_RETRIES')
HEADER = [
    "q18a-a1-trunk packaged by tools/build_a1_trunk.py (spec a1-trunk); PREV is the alpha-next-c1b READY build bundle and runner. Discovery mode: strict and focused logs are the parent tree's.",
    'Engine: claude/a1-trunk, in-wait build-4c with the returning-agent yield and the draining phase (FLASH_S5_APPEND_YIELD_*, default off, selected per arm by typed control).',
    "Queue: in-wait's startup and cleanup with the four bounds for the whole run; the exact-path smoke, one decode-only baseline, matched off/on live pairs with returning agents, one solo; no fixed-schedule check.",
]
parent.NAME = NAME
parent.QUEUE = QUEUE
parent.SETTING_KEYS = parent.SETTING_KEYS + BOUNDS
parent.HEADER = HEADER
parent.selfrun = selfrun
parent.RUNNER_SWAPS = [[parent._parent_swap("engine.get('schema')=="), "engine.get('schema')==" + repr(selfrun.SCHEMA)],
                       [parent.PARENT_QUEUES_TEXT, parent.PARENT_QUEUES_TEXT + " and result.get('a1_trunk_queue')=='PASS'"]]


def expected_tests_from_sources():
    import a1_trunk_tools_derive as derive
    want = derive.expected()
    pinned = re.findall(r'^EXPECTED_TESTS=(\d+) ', (TOOLS / 'a1_trunk_cpu_probe.py').read_text(), re.M)
    assert pinned == [str(want)], ('the committed probe pins another count; regenerate it', pinned, want)
    return want


parent.expected_tests_from_sources = expected_tests_from_sources


def generate(scratch):
    spec = parent.generate(scratch, None)
    spec.update(label='a1-trunk', candidate='q18a-a1-trunk', review_key='a1_trunk', bundle_selfrun=dict(arm='TRUNK', probe='a1_trunk_cpu_probe.py'),
                checks=[['tools/a1_trunk_queue.py', '--check'], ['tools/a1_trunk_tools_derive.py', '--check']])
    details = spec['details']
    details['a1_trunk_settings'] = details.pop('inwait_settings')
    assert all(details['a1_trunk_settings'][k] is not None for k in BOUNDS), details['a1_trunk_settings']
    details.update(queue="q18a-a1-trunk: in-wait's queue with the four FLASH_S5_APPEND_YIELD_* bounds for the whole run; the exact-path smoke, one decode-only baseline, off/on live pairs with returning agents, one solo; its own small grade",
                   scope="the returning-agent yield on in-wait's engine against c1b's READY build, discovery mode; not a release assessment and not a numerics result")
    details['engine_delta'] = dict(focused_suites=selfrun.ENGINE_SUITES + ['test_a1_trunk_box', 'test_a1_trunk_queue'], lineage_rows=[])
    return spec


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--scratch', type=Path, required=True, help='a directory that does not exist yet (the overlay propose scratch)')
    a = p.parse_args()
    (TOOLS / NAME).write_text(json.dumps(generate(a.scratch), indent=1, sort_keys=True) + '\n')
    print('spec written for ' + parent.b.git('rev-parse', 'HEAD').strip())
