#!/usr/bin/env python3
"""Write tools/native43_spec.step-a.json for HEAD: declarations computed from the tree, for a reviewer to commit.

source_paths and definitions come from build_native43.propose against alpha-perf's source 435dba6; engine_changed and
overlay_definitions from a scratch overlay regeneration against the alpha-perf build-d overlay. The source commit is HEAD, so
run it with harness/ and patches/ complete and committed. Only tools/ may change after it. No GPU, no box action.
"""
import argparse
import json
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from build_step_a import b  # noqa: E402
import step_a_engine_selfrun as selfrun  # noqa: E402
import step_a_overlay_propose as overlay  # noqa: E402

NAME = 'native43_spec.step-a.json'
ALPHA = json.loads((TOOLS / 'native43_spec.alpha-perf.json').read_text())
PARENT_CASES = ['abort-before', 'abort-live', 'auto', 'bad-graph', 'below-threshold', 'foreign', 'long-id', 'off', 'pending-full',
                'pending-split', 'progress-burst', 'progress-burst-current', 'progress-burst-skew', 'progress-burst-solo',
                'progress-current-solo', 'progress-trace', 'progress-trace-skew', 'progress-unset', 'receipts-auto',
                'receipts-below-threshold', 'receipts-latched', 'receipts-off', 'refuse', 'reuse', 'running-prefill', 'tp-skew']
RUNNER_SWAPS = [
    ["    from alpha_perf_gate import window_ok\n", "    from step_a_gate import window_ok\n"],
    ["engine.get('schema')=='alpha-perf-selfrun/1'", "engine.get('schema')=='step-a-selfrun/1'"],
    ["result.get('alpha_perf_queue')=='PASS'", "result.get('alpha_perf_queue')=='PASS' and result.get('step_a_queue')=='PASS'"],
    ["sorted(engine.get('cases') or {})==" + repr(PARENT_CASES),
     "sorted(engine.get('cases') or {})==" + repr(sorted(PARENT_CASES + list(selfrun.CASES)))],
]
HEADER = [
    'Step A packaged by tools/build_step_a.py (spec step-a); PREV is the alpha-perf build-d bundle and runner.',
    'Engine: claude/step-a, each dense QSA layer as one bounded worker command, default off; c3f1f64 (the alpha-perf engine) plus '
    'the eleven engine modules; default-off projections in step-a-freeze.json; the overlay files moved vs PREV are recomputed by '
    'the conf tool.',
    'Queue: alpha-perf\'s startup and cleanup with FLASH_S5_QSA_FUSED=1 added; band 1 eight arms and band 2 four arms, matched '
    'fused and stepped by the typed native_qsa_fused selection; no production start.',
]
EXTRA_COMPILED = ['harness/step_a_gate.py', 'harness/e3_step_a_box.py', 'harness/e3_step_a_probe.py', 'harness/test_step_a_queue.py',
                  'harness/test_step_a_report.py', 'harness/step_a_torn_lines.py', 'harness/test_step_a_torn_lines.py', 'harness/test_e3_native_qsa_fused.py', 'harness/e3_native_qsa_fused_device_probe.py',
                  'tools/build_step_a.py', 'tools/step_a_clock_scan.py', 'tools/step_a_cpu_probe.py', 'tools/step_a_delivery.py',
                  'tools/step_a_engine_selfrun.py', 'tools/step_a_freeze.py', 'tools/step_a_overlay_propose.py',
                  'tools/step_a_overlay_tamper.py', 'tools/step_a_precheck_conf.py', 'tools/step_a_queue.py', 'tools/step_a_report.py',
                  'tools/step_a_spec.py', 'tools/step_a_tools_derive.py']


def generate(scratch, expected_tests):
    head = b.git('rev-parse', 'HEAD').strip()
    spec = dict(source_commit=head, tools=[])
    proposed = b.propose(spec)
    paths = proposed['source_paths']
    me = 'A\ttools/' + NAME
    if me not in paths:
        paths = sorted(paths + [me])
    tools = sorted(row.split('\t', 1)[1][len('tools/'):] for row in paths
                   if row.startswith('A\ttools/') and row.count('/') == 1 and row.endswith(('.py', '.json')))
    moved = overlay.propose(scratch)
    assert not moved['added'] and not moved['removed'], moved
    dense = ALPHA['engine_selfrun']
    return dict(
        schema='native43-packaging-spec/1', label='step-a', dry_run=False, source_commit=head,
        candidate='q18a-step-a', arm='TRUNK', review='prepare', review_key='step_a',
        details=dict(
            model_settings_changed=False,
            step_a_settings={'FLASH_S5_AUTO_ARM': '1', 'FLASH_S5_NATIVE_PROGRESS': 'burst', 'FLASH_S5_NATIVE_QSA_POLICY': 'dense',
                             'FLASH_S5_QSA_FUSED': '1', 'FLASH_S5_UNION_RETIRE': None, 'FLASH_S5_NATIVE_QSA_TRACE': None,
                             'FLASH_S5_NATIVE_QSA_STOP_AFTER': None},
            queue='q18a-step-a: alpha-perf\'s queue with the fused switch and twelve matched arms over its two bands; validity and '
                  'engagement gated, speed and memory reported; 9,000 s box budget',
            scope='one bounded worker command per dense QSA layer against the stepped protocol; not a release assessment',
            engine_delta=dict(focused_suites=['test_e3_native_qsa_fused'], lineage_rows=[])),
        source_paths=paths, definitions=proposed['definitions'], engine_changed=moved['engine_changed'],
        overlay_definitions=moved['overlay_definitions'], tools=tools,
        queue=['q18a_step_a.draft.sh', 'q18a_step_a_declare.sh', 'q18a_step_a_settings.sh'],
        expected_tests=expected_tests, runner_swaps=RUNNER_SWAPS, rehearsal=None,
        engine_selfrun=dict(schema='step-a-selfrun-dense-part/1', compiled=dense['compiled'] + EXTRA_COMPILED,
                            suites=dense['suites'], probes=dense['probes']),
        bundle_selfrun=dict(arm='TRUNK', probe='step_a_cpu_probe.py'), clock_split=ALPHA['clock_split'],
        precheck=dict(seams='', fixture_rows=[], header=HEADER),
        checks=[['tools/step_a_queue.py', '--check'], ['tools/step_a_tools_derive.py', '--check']])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--scratch', type=Path, required=True)
    p.add_argument('--expected-tests', type=int, required=True)
    a = p.parse_args()
    (TOOLS / NAME).write_text(json.dumps(generate(a.scratch, a.expected_tests), indent=1, sort_keys=True) + '\n')
    print('spec written for ' + b.git('rev-parse', 'HEAD').strip())
