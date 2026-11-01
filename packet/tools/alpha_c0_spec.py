#!/usr/bin/env python3
"""Write tools/native43_spec.alpha-c0.json for HEAD: declarations computed from the tree, for a reviewer to commit.

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
from build_alpha_c0 import NEW_OVERLAY, b  # noqa: E402
import alpha_c0_engine_selfrun as selfrun  # noqa: E402
import alpha_c0_overlay_propose as overlay  # noqa: E402

NAME = 'native43_spec.alpha-c0.json'
ALPHA = json.loads((TOOLS / 'native43_spec.alpha-perf.json').read_text())
PARENT_CASES = ['abort-before', 'abort-live', 'auto', 'bad-graph', 'below-threshold', 'foreign', 'long-id', 'off', 'pending-full',
                'pending-split', 'progress-burst', 'progress-burst-current', 'progress-burst-skew', 'progress-burst-solo',
                'progress-current-solo', 'progress-trace', 'progress-trace-skew', 'progress-unset', 'receipts-auto',
                'receipts-below-threshold', 'receipts-latched', 'receipts-off', 'refuse', 'reuse', 'running-prefill', 'tp-skew']
RUNNER_SWAPS = [
    ["    from alpha_perf_gate import window_ok\n", "    from alpha_c0_gate import window_ok\n"],
    ["engine.get('schema')=='alpha-perf-selfrun/1'", "engine.get('schema')=='alpha-c0-selfrun/1'"],
    ["result.get('alpha_perf_queue')=='PASS'", "result.get('alpha_perf_queue')=='PASS' and result.get('alpha_c0_queue')=='PASS'"],
    ["sorted(engine.get('cases') or {})==" + repr(PARENT_CASES),
     "sorted(engine.get('cases') or {})==" + repr(sorted(PARENT_CASES + list(selfrun.CASES)))],
]
HEADER = [
    'C0 packaged by tools/build_alpha_c0.py (spec alpha-c0); PREV is the alpha-perf build-d bundle and runner.',
    'Engine: claude/c0-c1, stage-local prefill progress with overlap off, default off; c3f1f64 (the alpha-perf engine) plus the '
    'two new modules and the declared seams; default-off projections in alpha-c0-freeze.json; the overlay files moved vs PREV are '
    'recomputed by the conf tool.',
    'Queue: alpha-perf\'s startup and cleanup with FLASH_S5_STAGE_LOCAL=1 added; eight band 1 arms, matched serial and local by '
    'the typed native_stage_local selection; no production start.',
]
EXTRA_COMPILED = ['harness/alpha_c0_gate.py', 'harness/e3_alpha_c0_box.py', 'harness/e3_alpha_c0_probe.py', 'harness/test_alpha_c0_queue.py',
                  'harness/test_alpha_c0_report.py', 'harness/test_e3_stage_local.py', 'harness/test_e3_stage_local_control.py',
                  'harness/test_e3_stage_local_qsa.py', 'harness/stage_net.py', 'tools/build_alpha_c0.py', 'tools/alpha_c0_clock_scan.py',
                  'tools/alpha_c0_cpu_probe.py', 'tools/alpha_c0_delivery.py', 'tools/alpha_c0_engine_selfrun.py', 'tools/alpha_c0_freeze.py',
                  'tools/alpha_c0_overlay_propose.py', 'tools/alpha_c0_overlay_tamper.py', 'tools/alpha_c0_precheck_conf.py',
                  'tools/alpha_c0_queue.py', 'tools/alpha_c0_report.py', 'tools/alpha_c0_vm_compare.py', 'tools/alpha_c0_vm_input.py', 'tools/alpha_c0_spec.py', 'tools/alpha_c0_tools_derive.py',
                  'tools/stage_default_off.py', 'tools/c0_bit_identity.py']


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
    assert moved['added'] == sorted(NEW_OVERLAY) and not moved['removed'], moved
    dense = ALPHA['engine_selfrun']
    return dict(
        schema='native43-packaging-spec/1', label='alpha-c0', dry_run=False, source_commit=head,
        candidate='q18a-alpha-c0', arm='TRUNK', review='prepare', review_key='alpha_c0',
        details=dict(
            model_settings_changed=False,
            alpha_c0_settings={'FLASH_S5_AUTO_ARM': '1', 'FLASH_S5_NATIVE_PROGRESS': 'burst', 'FLASH_S5_NATIVE_QSA_POLICY': 'dense',
                             'FLASH_S5_STAGE_LOCAL': '1', 'FLASH_S5_UNION_RETIRE': None, 'FLASH_S5_NATIVE_QSA_TRACE': None,
                             'FLASH_S5_NATIVE_QSA_STOP_AFTER': None},
            queue='q18a-alpha-c0: alpha-perf\'s queue with the stage-local switch and eight matched arms over band 1; validity, '
                  'engagement and generated tokens gated, speed and memory reported; 5,400 s box budget',
            scope='stage-local native progress with overlap off, against the frozen serial path; not a release assessment',
            engine_delta=dict(focused_suites=['test_e3_stage_local', 'test_e3_stage_local_control', 'test_e3_stage_local_qsa'],
                              lineage_rows=[])),
        source_paths=paths, definitions=proposed['definitions'], engine_changed=moved['engine_changed'],
        overlay_definitions=moved['overlay_definitions'], tools=tools,
        queue=['q18a_alpha_c0.draft.sh', 'q18a_alpha_c0_declare.sh', 'q18a_alpha_c0_settings.sh'],
        expected_tests=expected_tests, runner_swaps=RUNNER_SWAPS, rehearsal=None,
        engine_selfrun=dict(schema='alpha-c0-selfrun-dense-part/1', compiled=dense['compiled'] + EXTRA_COMPILED,
                            suites=dense['suites'], probes=dense['probes']),
        bundle_selfrun=dict(arm='TRUNK', probe='alpha_c0_cpu_probe.py'), clock_split=ALPHA['clock_split'],
        precheck=dict(seams='', fixture_rows=[], header=HEADER),
        checks=[['tools/alpha_c0_queue.py', '--check'], ['tools/alpha_c0_tools_derive.py', '--check']])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--scratch', type=Path, required=True)
    p.add_argument('--expected-tests', type=int, required=True)
    a = p.parse_args()
    (TOOLS / NAME).write_text(json.dumps(generate(a.scratch, a.expected_tests), indent=1, sort_keys=True) + '\n')
    print('spec written for ' + b.git('rev-parse', 'HEAD').strip())
