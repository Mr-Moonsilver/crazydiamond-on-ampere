#!/usr/bin/env python3
"""Write tools/native43_spec.compose-4.json for HEAD: tools/inwait_spec.py's generation (c1b's READY build as the parent, source_paths, definitions, the overlay proposal, the runner's case list) with this family's names over it, as a1-trunk's. Discovery mode. No GPU, no box action."""
import argparse
import json
from pathlib import Path
import re
import sys

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import build_compose4  # noqa: E402,F401  (sets the candidate first)
import compose4_engine_selfrun as selfrun  # noqa: E402
import inwait_spec as parent  # noqa: E402

NAME = 'native43_spec.compose-4.json'
QUEUE = ['q18a_compose_4.draft.sh', 'q18a_compose_4_declare.sh', 'q18a_compose_4_settings.sh']
BOUNDS = ('FLASH_S5_APPEND_YIELD_MAX_NEW', 'FLASH_S5_APPEND_YIELD_PER_BULK', 'FLASH_S5_APPEND_YIELD_HANDBACK_STEPS', 'FLASH_S5_APPEND_YIELD_REENTRY_RETRIES', 'CUDA_DEVICE_MAX_CONNECTIONS', 'FLASH_S5_PLE_PRIVATE', 'FLASH_S5_PLE_WITNESS')
HEADER = [
    "q18a-compose-4 packaged by tools/build_compose4.py (spec compose-4); PREV is the alpha-next-c1b READY build bundle and runner. Discovery mode: strict and focused logs are the parent tree's.",
    'Engine: claude/compose-4, inwait build-5d with the returning-agent yield and the private PLE buffer (FLASH_S5_INWAIT, FLASH_S5_APPEND_YIELD_*, FLASH_S5_PLE_*, default off, selected per arm by typed control).',
    "Queue: a1-trunk's startup and cleanup with connections 32 and the PLE switches for the whole run; the exact-path smoke, one decode-only baseline, the one-stream arm, the composed replay arm and its control; no fixed-schedule check.",
]
parent.NAME = NAME
parent.QUEUE = QUEUE
parent.SETTING_KEYS = parent.SETTING_KEYS + BOUNDS
parent.HEADER = HEADER
parent.selfrun = selfrun
parent.RUNNER_SWAPS = [[parent._parent_swap("engine.get('schema')=="), "engine.get('schema')==" + repr(selfrun.SCHEMA)],
                       [parent.PARENT_QUEUES_TEXT, parent.PARENT_QUEUES_TEXT + " and result.get('compose4_queue')=='PASS'"]]


def expected_tests_from_sources():
    import compose4_tools_derive as derive
    want = derive.expected()
    pinned = re.findall(r'^EXPECTED_TESTS=(\d+) ', (TOOLS / 'compose4_cpu_probe.py').read_text(), re.M)
    assert pinned == [str(want)], ('the committed probe pins another count; regenerate it', pinned, want)
    return want


parent.expected_tests_from_sources = expected_tests_from_sources


def generate(scratch):
    spec = parent.generate(scratch, None)
    spec.update(label='compose-4', candidate='q18a-compose-4', review_key='compose4', bundle_selfrun=dict(arm='TRUNK', probe='compose4_cpu_probe.py'),
                checks=[['tools/compose4_queue.py', '--check'], ['tools/compose4_tools_derive.py', '--check']])
    details = spec['details']
    details['compose4_settings'] = details.pop('inwait_settings')
    assert all(details['compose4_settings'][k] is not None for k in BOUNDS), details['compose4_settings']
    details.update(queue="q18a-compose-4: a1-trunk's queue with connections 32 and the PLE switches for the whole run; the exact-path smoke, one decode-only baseline, the one-stream arm, the composed replay arm and its control; no fixed-schedule check",
                   scope="the four lever tracks on one engine under the closed-loop replay, against c1b's READY build, discovery mode; not a release assessment and not a numerics result")
    details['engine_delta'] = dict(focused_suites=selfrun.ENGINE_SUITES + ['test_compose4_box', 'test_compose4_queue'], lineage_rows=[])
    return spec


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--scratch', type=Path, required=True, help='a directory that does not exist yet (the overlay propose scratch)')
    a = p.parse_args()
    (TOOLS / NAME).write_text(json.dumps(generate(a.scratch), indent=1, sort_keys=True) + '\n')
    print('spec written for ' + parent.b.git('rev-parse', 'HEAD').strip())
