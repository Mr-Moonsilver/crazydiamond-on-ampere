#!/usr/bin/env python3
"""Write tools/native43_spec.compose-5.json for HEAD: tools/inwait_spec.py's generation (c1b's READY build as the parent, source_paths, definitions, the overlay proposal, the runner's case list) with this family's names over it, as a1-trunk's. Discovery mode. No GPU, no box action."""
import argparse
import json
from pathlib import Path
import re
import sys

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import build_compose5  # noqa: E402,F401  (sets the candidate first)
import compose5_engine_selfrun as selfrun  # noqa: E402
import inwait_spec as parent  # noqa: E402

NAME = 'native43_spec.compose-5.json'
QUEUE = ['q18a_compose_soak_fix.draft.sh', 'q18a_compose_soak_fix_declare.sh', 'q18a_compose_soak_fix_settings.sh']
BOUNDS = ('FLASH_S5_CHUNK_TOKENS', 'FLASH_S5_APPEND_YIELD_MAX_NEW', 'FLASH_S5_APPEND_YIELD_PER_BULK', 'FLASH_S5_APPEND_YIELD_HANDBACK_STEPS', 'FLASH_S5_APPEND_YIELD_REENTRY_RETRIES', 'CUDA_DEVICE_MAX_CONNECTIONS', 'FLASH_S5_PLE_PRIVATE', 'FLASH_S5_PLE_WITNESS')
HEADER = [
    "q18a-compose-soak-fix packaged by tools/build_compose5.py (spec compose-5); PREV is the alpha-next-c1b READY build bundle and runner. Discovery mode: strict and focused logs are the parent tree's.",
    'Engine: claude/compose-5-build, compose-4 with the decline counters, chunk_bound and the builder at 2048 rows from dp8d (FLASH_S5_CHUNK_TOKENS=2048 for the whole run), default off, selected per arm by typed control. No 5f, no 1g, no witness.',
    "Queue: compose-4's startup and cleanup with connections 32, the PLE switches and the 2048 chunk for the whole run; the exact-path smoke, one decode-only baseline, the one-stream arm, frozen pairs on and off for the yield, the replay arm and its control without the yield; no fixed-schedule check.",
]
parent.NAME = NAME
parent.QUEUE = QUEUE
parent.SETTING_KEYS = parent.SETTING_KEYS + BOUNDS
parent.HEADER = HEADER
parent.selfrun = selfrun
parent.RUNNER_SWAPS = [[parent._parent_swap("engine.get('schema')=="), "engine.get('schema')==" + repr(selfrun.SCHEMA)],
                       [parent.PARENT_QUEUES_TEXT, parent.PARENT_QUEUES_TEXT + " and result.get('compose5_queue')=='PASS'"]]


def expected_tests_from_sources():
    import compose5_tools_derive as derive
    want = derive.expected()
    pinned = re.findall(r'^EXPECTED_TESTS=(\d+) ', (TOOLS / 'compose5_cpu_probe.py').read_text(), re.M)
    assert pinned == [str(want)], ('the committed probe pins another count; regenerate it', pinned, want)
    return want


parent.expected_tests_from_sources = expected_tests_from_sources


def generate(scratch):
    spec = parent.generate(scratch, None)
    spec.update(label='compose-5', candidate='q18a-compose-soak-fix', review_key='compose5', bundle_selfrun=dict(arm='TRUNK', probe='compose5_cpu_probe.py'),
                checks=[['tools/compose5_queue.py', '--check'], ['tools/compose5_tools_derive.py', '--check']])
    details = spec['details']
    details['compose5_settings'] = details.pop('inwait_settings')
    assert all(details['compose5_settings'][k] is not None for k in BOUNDS), details['compose5_settings']
    details.update(queue="q18a-compose-soak-fix: compose-4's queue with connections 32, the PLE switches and the 2048 chunk for the whole run; the exact-path smoke, one decode-only baseline, the one-stream arm, frozen yield pairs on two indexes, the replay arm and its control without the yield; no fixed-schedule check",
                   scope="b13 and hf at 32 with the four lever tracks under the closed-loop replay, the yield priced by frozen pairs, against c1b's READY build, discovery mode; not a release assessment and not a numerics result")
    details['engine_delta'] = dict(focused_suites=selfrun.ENGINE_SUITES + ['test_compose5_box', 'test_compose5_queue'], lineage_rows=[])
    return spec


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--scratch', type=Path, required=True, help='a directory that does not exist yet (the overlay propose scratch)')
    a = p.parse_args()
    (TOOLS / NAME).write_text(json.dumps(generate(a.scratch), indent=1, sort_keys=True) + '\n')
    print('spec written for ' + parent.b.git('rev-parse', 'HEAD').strip())
