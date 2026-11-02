#!/usr/bin/env python3
"""Write tools/native43_spec.step-b.json for HEAD: declarations computed from the tree, for a reviewer to commit.

source_paths and definitions come from build_native43.propose against Step A's build source; engine_changed and overlay_definitions from
a scratch overlay regeneration against Step A's frozen overlay. The source commit is HEAD, so run it with harness/ and patches/ complete
and committed. Only tools/ may change after it. No GPU, no box action.
"""
import argparse
import json
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from build_step_b import b  # noqa: E402
import step_b_engine_selfrun as selfrun  # noqa: E402
import step_b_overlay_propose as overlay  # noqa: E402

NAME = 'native43_spec.step-b.json'
STEP_A = json.loads((TOOLS / 'native43_spec.step-a.json').read_text())
def _step_a_swap(marker):
    """The text Step A's runner holds for one of its swaps: the new side of its spec's row, the one source."""
    rows = [row[1] for row in STEP_A['runner_swaps'] if marker in row[1]]
    assert len(rows) == 1, marker
    return rows[0]


STEP_A_CASES_TEXT = _step_a_swap("sorted(engine.get('cases') or {})==")
assert STEP_A_CASES_TEXT == "sorted(engine.get('cases') or {})==" + repr(selfrun.step_a_cases()), 'Step A runner case list differs from its sources'
RUNNER_SWAPS = [
    [_step_a_swap('import window_ok'), "    from step_b_gate import window_ok\n"],
    [_step_a_swap("engine.get('schema')=="), "engine.get('schema')=='step-b-selfrun/1'"],
    [_step_a_swap("result.get('step_a_queue')=='PASS'"),
     _step_a_swap("result.get('step_a_queue')=='PASS'") + " and result.get('step_b_queue')=='PASS'"],
    [STEP_A_CASES_TEXT, "sorted(engine.get('cases') or {})==" + repr(selfrun.expected_cases())],
    # Removed, not replaced: box timing is governed by the lead at launch (operator, 2 Nov); see harness/step_b_clock_bound.py.
    ["result.get('native35_p_c_continuation')=='PASS' and result.get('native39_trunk_queue')=='PASS'",
     "result.get('native35_p_c_continuation')=='PASS'"],
]
HEADER = [
    'Step B packaged by tools/build_step_b.py (spec step-b); PREV is the Step A bundle and runner.',
    'Engine: claude/step-b, consecutive LAYER entries of one stage as one bounded worker command, default off; the Step A engine plus '
    'the declared engine modules; default-off projections in step-b-freeze.json; the overlay files moved vs PREV are recomputed by '
    'the conf tool.',
    'Queue: Step A\'s startup and cleanup with FLASH_S5_LAYER_GROUPS=1 added; band 1 twelve arms at sizes 1, 2 and 4 and band 2 four arms '
    'at sizes 1 and 2, matched by the typed native_layer_group selection, fused selected in every arm; no production start.',
]
EXTRA_COMPILED = ['harness/step_b_gate.py', 'harness/step_b_clock_bound.py', 'harness/e3_step_b_box.py', 'harness/e3_step_b_probe.py', 'harness/test_step_b_queue.py',
                  'harness/test_e3_native_layer_groups.py', 'tools/build_step_b.py', 'tools/step_b_clock_scan.py', 'tools/step_b_cpu_probe.py',
                  'tools/step_b_delivery.py', 'tools/step_b_engine_selfrun.py', 'tools/step_b_freeze.py', 'tools/step_b_overlay_propose.py',
                  'tools/step_b_overlay_tamper.py', 'tools/step_b_precheck_conf.py', 'tools/step_b_queue.py', 'tools/step_b_spec.py',
                  'tools/step_b_tools_derive.py', 'tools/step_b_parent_pin.py', 'tools/step_b_staging_check.py', 'tools/step_b_vm_input.py', 'tools/step_b_vm_compare.py', 'tools/step_b_wait_probe.py']


def expected_tests_from_sources():
    """The bundle self-run's test count: the derived CPU probe's EXPECTED_TESTS: Step A's spec's count less the declared clock-bound tests, plus Step B's."""
    import step_b_tools_derive as derive
    assert derive.PARENT_TESTS == STEP_A['expected_tests'], (derive.PARENT_TESTS, STEP_A['expected_tests'])
    sys.path.insert(0, str(TOOLS.parent / 'harness'))
    import step_b_clock_bound as cb
    assert derive.CLOCK_BOUND_IN_PROBE == len(cb.IN_PROBE), (derive.CLOCK_BOUND_IN_PROBE, cb.IN_PROBE)
    return derive.PARENT_TESTS - derive.CLOCK_BOUND_IN_PROBE + derive.STEP_B_TESTS


def generate(scratch, expected_tests):
    assert expected_tests == expected_tests_from_sources(), (expected_tests, expected_tests_from_sources())
    head = b.git('rev-parse', 'HEAD').strip()
    spec = dict(source_commit=head, tools=[])
    proposed = b.propose(spec)
    paths = proposed['source_paths']
    me = 'A\ttools/' + NAME
    if me not in paths:
        paths = sorted(paths + [me])
    # The gate chain's prepare-review commit always moves the review pin after this spec is written; alpha-perf's and the
    # lifecycle specs declare it, Step A's build d did not and stopped at check_paths (tmp/step-a/pack/final-d/evidence/build-d.log).
    review = 'M\t' + b.REVIEW_PATH
    if review not in paths:
        paths = sorted(paths + [review])
    tools = sorted(row.split('\t', 1)[1][len('tools/'):] for row in paths
                   if row.startswith('A\ttools/') and row.count('/') == 1 and row.endswith(('.py', '.json')))
    moved = overlay.propose(scratch)
    assert not moved['added'] and not moved['removed'], moved
    dense = STEP_A['engine_selfrun']
    return dict(
        schema='native43-packaging-spec/1', label='step-b', dry_run=False, source_commit=head,
        candidate='q18a-step-b', arm='TRUNK', review='prepare', review_key='step_b',
        details=dict(
            model_settings_changed=False,
            step_b_settings={'FLASH_S5_AUTO_ARM': '1', 'FLASH_S5_NATIVE_PROGRESS': 'burst', 'FLASH_S5_NATIVE_QSA_POLICY': 'dense',
                             'FLASH_S5_QSA_FUSED': '1', 'FLASH_S5_LAYER_GROUPS': '1', 'FLASH_S5_UNION_RETIRE': None,
                             'FLASH_S5_NATIVE_QSA_TRACE': None, 'FLASH_S5_NATIVE_QSA_STOP_AFTER': None},
            queue='q18a-step-b: Step A\'s queue with the group switch and sixteen matched arms over its two bands, fused selected in all; '
                  'validity and engagement gated, speed, decode and memory reported; 9,000 s box budget',
            scope='consecutive layers of one stage as one bounded worker command against size 1; not a release assessment',
            engine_delta=dict(focused_suites=['test_e3_native_layer_groups'], lineage_rows=[])),
        source_paths=paths, definitions=proposed['definitions'], engine_changed=moved['engine_changed'],
        overlay_definitions=moved['overlay_definitions'], tools=tools,
        queue=['q18a_step_b.draft.sh', 'q18a_step_b_declare.sh', 'q18a_step_b_settings.sh'],
        expected_tests=expected_tests, runner_swaps=RUNNER_SWAPS, rehearsal=None,
        engine_selfrun=dict(schema='step-b-selfrun-dense-part/1', compiled=dense['compiled'] + EXTRA_COMPILED,
                            suites=dense['suites'], probes=dense['probes']),
        bundle_selfrun=dict(arm='TRUNK', probe='step_b_cpu_probe.py'), clock_split=STEP_A['clock_split'],
        precheck=dict(seams='', fixture_rows=[], header=HEADER),
        checks=[['tools/step_b_queue.py', '--check'], ['tools/step_b_tools_derive.py', '--check']])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--scratch', type=Path, required=True)
    p.add_argument('--expected-tests', type=int, required=True)
    a = p.parse_args()
    (TOOLS / NAME).write_text(json.dumps(generate(a.scratch, a.expected_tests), indent=1, sort_keys=True) + '\n')
    print('spec written for ' + b.git('rev-parse', 'HEAD').strip())
