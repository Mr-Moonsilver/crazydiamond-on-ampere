#!/usr/bin/env python3
"""Write tools/native43_spec.alpha-perf.json for HEAD: declarations computed from the tree, for a reviewer to commit.

source_paths and definitions come from build_native43.propose against held-f's source 6e36495; engine_changed and
overlay_definitions from a scratch overlay regeneration against the held-f build-b overlay. The source commit is HEAD, so
run it with harness/ and patches/ complete and committed. Only tools/ may change after it. No GPU, no box action.
"""
import argparse
import json
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from build_alpha_perf import b  # noqa: E402
import alpha_perf_overlay_propose as overlay  # noqa: E402

NAME = 'native43_spec.alpha-perf.json'
LF = json.loads((TOOLS / 'native43_spec.native43-lifecycle-f.json').read_text())
RUNNER_SWAPS = [
    ["    from native41_progress_gate import window_ok\n", "    from alpha_perf_gate import window_ok\n"],
    ["engine.get('schema')=='native41-progress-selfrun/1'", "engine.get('schema')=='alpha-perf-selfrun/1'"],
    ["result.get('draft_window_slots')=='PASS'", "result.get('draft_window_slots')=='PASS' and result.get('alpha_perf_queue')=='PASS'"],
    ["and engine.get('image')==IMAGE and engine.get('box_action') is False,'exact-engine-selfrun-required')",
     "and engine.get('image')==IMAGE and engine.get('box_action') is False\n"
     "                and (engine.get('clock_split') or {}).get('verdict')=='PASS'\n"
     "                and (engine.get('scheduler_tick_probe') or {}).get('verdict')=='PASS' and (engine.get('scheduler_tick_probe') or {}).get('required_ok') is True,'exact-engine-selfrun-required')"],
]
HEADER = [
    'Alpha-perf packaged by tools/build_alpha_perf.py (spec alpha-perf); PREV is the frozen held-f build-b bundle and runner.',
    'Engine: the composed alpha trunk (held-f main c607333 merged with the dense line), byte-identical to claude/alpha-trunk '
    'at the merged commit except the review pin; the overlay files moved vs PREV are recomputed by the conf tool.',
    'Queue: held-f\'s startup and cleanup with FLASH_S5_NATIVE_QSA_POLICY=dense added, the le-fix death scan and watch, band 1 '
    '(held-f\'s eight arms) and band 2 (burst, 131,094 tokens, past the arena); no production start.',
]
EXTRA_COMPILED = ['harness/alpha_perf_gate.py', 'harness/e3_alpha_perf_box.py', 'harness/test_alpha_perf_queue.py',
                  'harness/native43_allocator_phase_deltas.py', 'harness/test_alpha_m3_receipts.py',
                  'tools/alpha_perf_queue.py', 'tools/alpha_perf_tools_derive.py', 'tools/alpha_perf_cpu_probe.py',
                  'tools/alpha_perf_clock_scan.py', 'tools/alpha_perf_precheck_conf.py', 'tools/alpha_perf_engine_selfrun.py',
                  'tools/alpha_perf_spec.py', 'tools/alpha_perf_overlay_propose.py', 'tools/build_alpha_perf.py',
                  'tools/alpha_perf_delivery.py', 'tools/alpha_m3_queue_deltas.py']
EXTRA_SUITES = ['test_alpha_m3_receipts']


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
    lf_selfrun = LF['engine_selfrun']
    return dict(
        schema='native43-packaging-spec/1', label='alpha-perf', dry_run=False, source_commit=head,
        candidate='q18a-alpha-perf', arm='TRUNK', review='prepare', review_key='alpha_perf',
        details=dict(
            model_settings_changed=False,
            alpha_perf_settings={'FLASH_S5_AUTO_ARM': '1', 'FLASH_S5_NATIVE_PROGRESS': 'burst',
                                 'FLASH_S5_NATIVE_QSA_POLICY': 'dense', 'FLASH_S5_NATIVE_QSA_TRACE': None,
                                 'FLASH_S5_NATIVE_QSA_STOP_AFTER': None},
            queue='q18a-alpha-perf: held-f\'s queue with the dense policy, band 1 unchanged and band 2 past the arena; speed '
                  'reported, validity gated; 2.5 h box budget',
            scope='prefill speed and decode-during-prefill on the composed alpha engine; no engine source change',
            engine_delta=dict(focused_suites=[], lineage_rows=[])),
        source_paths=paths, definitions=proposed['definitions'], engine_changed=moved['engine_changed'],
        overlay_definitions=moved['overlay_definitions'], tools=tools,
        queue=['q18a_alpha_perf.draft.sh', 'q18a_alpha_perf_declare.sh', 'q18a_alpha_perf_settings.sh'],
        expected_tests=expected_tests, runner_swaps=RUNNER_SWAPS, rehearsal=None,
        engine_selfrun=dict(schema='alpha-perf-selfrun-dense-part/1', compiled=lf_selfrun['compiled'] + EXTRA_COMPILED,
                            suites=lf_selfrun['suites'] + EXTRA_SUITES, probes=lf_selfrun['probes']),
        bundle_selfrun=dict(arm='TRUNK', probe='alpha_perf_cpu_probe.py'), clock_split=LF['clock_split'],
        precheck=dict(seams='', fixture_rows=[], header=HEADER),
        checks=[['tools/alpha_perf_queue.py', '--check'], ['tools/alpha_perf_tools_derive.py', '--check']])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--scratch', type=Path, required=True)
    p.add_argument('--expected-tests', type=int, required=True)
    a = p.parse_args()
    (TOOLS / NAME).write_text(json.dumps(generate(a.scratch, a.expected_tests), indent=1, sort_keys=True) + '\n')
    print('spec written for ' + b.git('rev-parse', 'HEAD').strip())
