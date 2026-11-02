#!/usr/bin/env python3
"""Write tools/native43_spec.alpha-c1.json for HEAD: declarations computed from the tree, for a reviewer to commit.

tools/alpha_c0_spec.py's form with C0 l as the parent: source_paths and definitions come from build_native43.propose against
C0 l's build commit cf0e301 (build_alpha_c1's PARENT_COMMIT); engine_changed and overlay_definitions from a scratch overlay
regeneration against the build-l overlay. The dense engine self-run part (compiled list, suites, probes) and the clock split are
C0 l's committed spec's (tools/native43_spec.alpha-c0.json), with C1's files added to the compiled list. The source commit is
HEAD, so run it with harness/ and patches/ complete and committed. Only tools/ may change after it. No GPU, no box action.
"""
import argparse
import json
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from build_alpha_c1 import NEW_OVERLAY, b  # noqa: E402
import alpha_c1_engine_selfrun as selfrun  # noqa: E402
import alpha_c1_overlay_propose as overlay  # noqa: E402
import alpha_c0_spec as c0_spec  # noqa: E402

NAME = 'native43_spec.alpha-c1.json'
C0 = json.loads((TOOLS / 'native43_spec.alpha-c0.json').read_text())
C0_CASES = sorted(c0_spec.PARENT_CASES + list(selfrun.c0.CASES))
RUNNER_SWAPS = [
    # C0 l's runner already has the window-free check_slot (operator, 2 Nov): nothing to swap there.
    ["engine.get('schema')=='alpha-c0-selfrun/1'", "engine.get('schema')=='alpha-c1-selfrun/1'"],
    ["result.get('alpha_c0_queue')=='PASS'", "result.get('alpha_c0_queue')=='PASS' and result.get('alpha_c1_queue')=='PASS'"],
    ["sorted(engine.get('cases') or {})==" + repr(C0_CASES),
     "sorted(engine.get('cases') or {})==" + repr(sorted(C0_CASES + list(selfrun.CASES)))],
]
HEADER = [
    'C1 packaged by tools/build_alpha_c1.py (spec alpha-c1); PREV is the C0 l build-l bundle (a1526e066c6a) and runner.',
    'Engine: claude/c1, two chunks in flight (FLASH_S5_STAGE_OVERLAP=1 beside FLASH_S5_STAGE_LOCAL=1), default off; C0 l\'s engine '
    '(cf0e301) plus e3/stage_overlap.py and the declared seams; default-off projections in alpha-c1-freeze.json; the overlay files '
    'moved vs PREV are recomputed by the conf tool.',
    'Queue: C0 l\'s startup and cleanup with FLASH_S5_STAGE_OVERLAP=1 added; twelve mirrored band 1 arms (serial, local, overlap; '
    'solo and live) and four band 2 arms; no production start.',
]
EXTRA_COMPILED = ['harness/alpha_c1_gate.py', 'harness/e3_alpha_c1_box.py', 'harness/e3_alpha_c1_probe.py',
                  'harness/test_alpha_c1_queue.py', 'harness/test_alpha_c1_report.py', 'harness/test_alpha_c1_box.py',
                  'harness/test_e3_stage_overlap.py', 'harness/test_e3_stage_overlap_control.py',
                  'tools/build_alpha_c1.py', 'tools/alpha_c1_clock_scan.py', 'tools/alpha_c1_cpu_probe.py', 'tools/alpha_c1_delivery.py',
                  'tools/alpha_c1_engine_selfrun.py', 'tools/alpha_c1_flaky.py', 'tools/alpha_c1_freeze.py',
                  'tools/alpha_c1_overlay_propose.py',
                  'tools/alpha_c1_overlay_tamper.py', 'tools/alpha_c1_precheck_conf.py', 'tools/alpha_c1_queue.py',
                  'tools/alpha_c1_report.py', 'tools/alpha_c1_vm_compare.py', 'tools/alpha_c1_vm_input.py', 'tools/alpha_c1_spec.py',
                  'tools/alpha_c1_tools_derive.py', 'tools/stage_default_off_c1.py']


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
    dense = C0['engine_selfrun']
    compiled = dense['compiled'] + [x for x in EXTRA_COMPILED if x not in dense['compiled']]
    return dict(
        schema='native43-packaging-spec/1', label='alpha-c1', dry_run=False, source_commit=head,
        candidate='q18a-alpha-c1', arm='TRUNK', review='prepare', review_key='alpha_c1',
        details=dict(
            model_settings_changed=False,
            alpha_c1_settings={'FLASH_S5_AUTO_ARM': '1', 'FLASH_S5_NATIVE_PROGRESS': 'burst', 'FLASH_S5_NATIVE_QSA_POLICY': 'dense',
                               'FLASH_S5_STAGE_LOCAL': '1', 'FLASH_S5_STAGE_OVERLAP': '1', 'FLASH_S5_UNION_RETIRE': None,
                               'FLASH_S5_NATIVE_QSA_TRACE': None, 'FLASH_S5_NATIVE_QSA_STOP_AFTER': None},
            queue='q18a-alpha-c1: C0 l\'s queue with the stage-overlap switch; twelve mirrored band 1 arms (serial, local and '
                  'overlap, solo and live) and four band 2 arms under the native41 caps; validity, engagement and generated tokens '
                  'gated, speed and memory reported; 9,000 s box budget, no absolute-time window',
            scope='two chunks in flight per request (stage overlap) on C0\'s stage-local progress, against the frozen serial path '
                  'and C0\'s local path; not a release assessment',
            engine_delta=dict(focused_suites=['test_e3_stage_overlap', 'test_e3_stage_overlap_control'], lineage_rows=[])),
        source_paths=paths, definitions=proposed['definitions'], engine_changed=moved['engine_changed'],
        overlay_definitions=moved['overlay_definitions'], tools=tools,
        queue=['q18a_alpha_c1.draft.sh', 'q18a_alpha_c1_declare.sh', 'q18a_alpha_c1_settings.sh'],
        expected_tests=expected_tests, runner_swaps=RUNNER_SWAPS, rehearsal=None,
        engine_selfrun=dict(schema='alpha-c1-selfrun-dense-part/1', compiled=compiled, suites=dense['suites'], probes=dense['probes']),
        bundle_selfrun=dict(arm='TRUNK', probe='alpha_c1_cpu_probe.py'), clock_split=C0['clock_split'],
        precheck=dict(seams='', fixture_rows=[], header=HEADER),
        checks=[['tools/alpha_c1_queue.py', '--check'], ['tools/alpha_c1_tools_derive.py', '--check']])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--scratch', type=Path, required=True)
    p.add_argument('--expected-tests', type=int, required=True)
    p.add_argument('--out', type=Path, help='write here instead of tools/' + NAME + ' (a dry run)')
    a = p.parse_args()
    (a.out or TOOLS / NAME).write_text(json.dumps(generate(a.scratch, a.expected_tests), indent=1, sort_keys=True) + '\n')
    print('spec written for ' + b.git('rev-parse', 'HEAD').strip())
