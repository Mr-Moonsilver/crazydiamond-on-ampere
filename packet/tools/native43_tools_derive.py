#!/usr/bin/env python3
"""Derive the native43 bundle CPU probe from the frozen native39 trunk one.

native43_cpu_probe.py = native39_trunk_cpu_probe.py with:
  * FLASH_NATIVE35_ARM=NATIVE43, candidate q18a-native43-dense-boundary;
  * test_native43_boundary_queue added to the discovered suites (the trunk queue
    suite stays), EXPECTED_TESTS 218 -> 218 + NATIVE43_QUEUE_TESTS;
  * the frozen runner's check_slot negatives at the native43 pair window
    (native43_boundary_gate PAIR_ACTIVE_S + PAIR_KILL_MARGIN_S), with the gate's
    NOT_BEFORE/HARD_END asserted equal to the weekend gate; the trunk window value
    stays recorded under its own name;
  * the LOCALIZE=0 inert check also run on both native43 settings files
    (current, dense), each held to the trunk negative's exact expectations;
  * the native43 queue cases bound by test id: every case discovered, none failed,
    errored or skipped; recorded as native43_queue='PASS'.
The bundle self-run is tools/native43_bundle_selfrun.py with the spec's arm and probe.
Every replacement must match exactly once.  --check verifies the committed file;
--rewrite regenerates it after a declared change of this derivation.
"""
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
PARENT_TESTS = 218
NATIVE43_QUEUE_TESTS = 11  # test_native43_boundary_queue: 5 QueueTests + 6 GradeTests
CANDIDATE = 'q18a-native43-dense-boundary'

PROBE_SWAPS = (
    ('"""Native39 trunk CPU-only plugin discovery/source, wrapper, grade, builder-view, speed-arm, L2 row4, LOCALIZE=0 and trunk queue checks; never used by box workers.\n\n'
     'Derived from native35_p_c_cpu_probe.py by tools/native39_trunk_tools_derive.py.\n'
     'FLASH_NATIVE35_ARM must be TRUNK: it selects the frozen trunk owner runner, the\n'
     'trunk window (native39_trunk_gate) and the provenance binding; the trunk queue\n'
     'suite (test_native39_trunk_queue) runs with the inherited suites.',
     '"""Native43 CPU-only plugin discovery/source, wrapper, grade, builder-view, speed-arm, L2 row4, LOCALIZE=0, trunk queue and native43 queue checks; never used by box workers.\n\n'
     'Derived from native39_trunk_cpu_probe.py by tools/native43_tools_derive.py.\n'
     'FLASH_NATIVE35_ARM must be NATIVE43: it selects the frozen native43 owner runner, the\n'
     'pair window (native43_boundary_gate) and the provenance binding; the trunk queue\n'
     'suite (test_native39_trunk_queue) and the native43 queue suite\n'
     '(test_native43_boundary_queue) run with the inherited suites.'),
    ("suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native39_trunk_queue.py'))\n"
     'EXPECTED_TESTS=218  # the 183 P-c probe cases plus the 35 native39 trunk queue cases; pinned into the frozen runner gate\n',
     "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native39_trunk_queue.py'))\n"
     "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native43_boundary_queue.py'))\n"
     'EXPECTED_TESTS=%d  # the %d trunk probe cases plus the %d native43 boundary queue cases; pinned into the frozen runner gate\n'
     '# Native43 queue case ids are taken before the run (TestSuite.run releases its tests).\n'
     'def _cases(tests):\n'
     '    for test in tests:\n'
     '        yield from (_cases(test) if isinstance(test,unittest.TestSuite) else (test,))\n'
     "native43_cases=sorted(t.id() for t in _cases(suite) if type(t).__module__=='test_native43_boundary_queue')\n"
     % (PARENT_TESTS + NATIVE43_QUEUE_TESTS, PARENT_TESTS, NATIVE43_QUEUE_TESTS)),
    ("assert role=='TRUNK' and os.environ['FLASH_NATIVE34_ARM']=='O'\n",
     "assert role=='NATIVE43' and os.environ['FLASH_NATIVE34_ARM']=='O'\n"),
    ("owner.QNAME=='q18a-native39-trunk'\n",
     "owner.QNAME==%r\n" % CANDIDATE),
    ('active_and_cleanup=native39_trunk_gate.TRUNK_ACTIVE_AND_CLEANUP\n',
     'trunk_active_and_cleanup=native39_trunk_gate.TRUNK_ACTIVE_AND_CLEANUP\n'
     'import native43_boundary_gate\n'
     'assert (native43_boundary_gate.NOT_BEFORE,native43_boundary_gate.HARD_END)==(weekend_gate.NOT_BEFORE,weekend_gate.HARD_END)\n'
     '# The frozen native43 runner gates the whole pair: window_ok = pair active + pair kill margin.\n'
     'active_and_cleanup=native43_boundary_gate.PAIR_ACTIVE_S+native43_boundary_gate.PAIR_KILL_MARGIN_S\n'),
    ("negative=inert.run('/source/harness/runs/q18a_native39_trunk_settings.sh','TRUNK settings (LOCALIZE=0)')\n"
     'for row in (positive,negative):\n',
     "negative=inert.run('/source/harness/runs/q18a_native39_trunk_settings.sh','TRUNK settings (LOCALIZE=0)')\n"
     "native43_inert={p:inert.run('/source/harness/runs/q18a_native43_%s_settings.sh'%p,'NATIVE43 %s settings (LOCALIZE=0)'%p) for p in ('current','dense')}\n"
     'for row in (positive,negative,*native43_inert.values()):\n'),
    ("assert negative['installed']=={k:False for k in inert.PLUGINS} and negative['meta_path_finders']==[] and negative['wrapped']==[] and negative['native30_stderr_lines']==0\n",
     "assert negative['installed']=={k:False for k in inert.PLUGINS} and negative['meta_path_finders']==[] and negative['wrapped']==[] and negative['native30_stderr_lines']==0\n"
     'for row in native43_inert.values():\n'
     "    assert row['switches']['FLASH_NATIVE30_LOCALIZE']=='0' and row['callables_scanned']==negative['callables_scanned'] and row['target_modules']==negative['target_modules']\n"
     "    assert row['installed']=={k:False for k in inert.PLUGINS} and row['meta_path_finders']==[] and row['wrapped']==[] and row['native30_stderr_lines']==0\n"),
    ('inert_receipt=dict(verdict=\'PASS\',positive_control=positive,localize_off=negative,',
     'inert_receipt=dict(verdict=\'PASS\',positive_control=positive,localize_off=negative,native43_localize_off=native43_inert,'),
    ("Path('/out/native39-trunk-queue-check.json').write_text(json.dumps(dict(cases=trunk_cases,slots=trunk_slots),sort_keys=True,indent=2)+'\\n')\n",
     "Path('/out/native39-trunk-queue-check.json').write_text(json.dumps(dict(cases=trunk_cases,slots=trunk_slots),sort_keys=True,indent=2)+'\\n')\n"
     '# Native43: the boundary queue suite (exact drafts, real CLI/tokenize seam, bracket across clock views,\n'
     '# strict pair-grade negatives, real window budgets), bound by test id.\n'
     'native43_bad=sorted({t.id() for t,_ in result.failures+result.errors+result.skipped}&set(native43_cases))\n'
     'assert len(native43_cases)==%d and not native43_bad,(native43_cases,native43_bad)\n'
     "Path('/out/native43-queue-check.json').write_text(json.dumps(dict(cases=native43_cases,window_active_and_cleanup_s=active_and_cleanup),sort_keys=True,indent=2)+'\\n')\n"
     % NATIVE43_QUEUE_TESTS),
    ("record=dict(native39_trunk_queue='PASS',native39_trunk_queue_receipts=trunk_cases,native39_trunk_window_active_and_cleanup_s=active_and_cleanup,",
     "record=dict(native43_queue='PASS',native43_queue_cases=native43_cases,native43_window_active_and_cleanup_s=active_and_cleanup,"
     "native39_trunk_queue='PASS',native39_trunk_queue_receipts=trunk_cases,native39_trunk_window_active_and_cleanup_s=trunk_active_and_cleanup,"),
)

OUTPUTS = {'native43_cpu_probe.py': ('native39_trunk_cpu_probe.py', PROBE_SWAPS)}


def apply(text, swaps):
    for old, new in swaps:
        assert text.count(old) == 1, (old[:80], text.count(old))
        text = text.replace(old, new)
    return text


def generate(tools=TOOLS):
    return {name: apply((tools / source).read_text(), swaps) for name, (source, swaps) in OUTPUTS.items()}


if __name__ == '__main__':
    check, rewrite = '--check' in sys.argv, '--rewrite' in sys.argv
    for name, text in generate().items():
        path = TOOLS / name
        if check:
            assert path.read_text() == text, name
        elif rewrite:
            assert path.exists(), name
            path.write_text(text)
        else:
            assert not path.exists(), name + ' exists; use --check'
            path.write_text(text)
            path.chmod(0o775)
    print('native43 bundle tools ' + ('verified' if check else 'rewritten' if rewrite else 'written'))
