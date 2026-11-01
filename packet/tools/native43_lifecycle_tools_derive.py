#!/usr/bin/env python3
"""Derive the native43 lifecycle bundle CPU probe from the frozen native43 (n43-f) one.

native43_lifecycle_cpu_probe.py = native43_cpu_probe.py (the probe the n43-f bundle ran) with:
  * FLASH_NATIVE35_ARM=NATIVE43L, candidate q18a-native43-lifecycle;
  * test_native43_lifecycle_queue added to the discovered suites (the trunk and native43
    boundary queue suites stay), EXPECTED_TESTS 231 -> 231 + LIFECYCLE_TESTS;
  * the frozen runner's check_slot negatives at the lifecycle window
    (native43_lifecycle_gate LIFECYCLE_ACTIVE_AND_CLEANUP), with the gate's NOT_BEFORE/HARD_END
    asserted equal to the weekend gate; the native43 pair window value stays recorded under its
    own name;
  * the LOCALIZE=0 inert check also run on the lifecycle settings, held to the same expectations
    as the two native43 arms;
  * the lifecycle draft's real-clock slot receipt appended to the inherited ones;
  * the lifecycle queue receipts asserted and recorded (native43_lifecycle_queue='PASS'), beside
    the unchanged native43 and trunk queue receipts.
The bundle self-run is tools/native43_bundle_selfrun.py with the spec's arm and probe.
Every replacement must match exactly once.  --check verifies the committed file.
"""
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
PARENT_TESTS = 231
LIFECYCLE_TESTS = 16  # test_native43_lifecycle_queue
LIFECYCLE_CASES = sorted(['tail-short-lifecycle-connected-pass', 'tail-reuse-incomplete-inconclusive',
                          'tail-row1-not-pass-withholds', 'tail-engine-crash-withholds', 'tail-observer-mapped-withholds',
                          'tail-hygiene-refused-inconclusive', 'grade-short-scope-reading',
                          'workload-lifecycle-cli-native39-manifest', 'settings-dense-policy-no-trace-no-stop'])
CANDIDATE = 'q18a-native43-lifecycle'

PROBE_SWAPS = (
    ('"""Native43 CPU-only plugin discovery/source, wrapper, grade, builder-view, speed-arm, L2 row4, LOCALIZE=0, trunk queue and native43 queue checks; never used by box workers.\n\n'
     'Derived from native39_trunk_cpu_probe.py by tools/native43_tools_derive.py.\n'
     'FLASH_NATIVE35_ARM must be NATIVE43: it selects the frozen native43 owner runner, the\n'
     'pair window (native43_boundary_gate) and the provenance binding; the trunk queue\n'
     'suite (test_native39_trunk_queue) and the native43 queue suite\n'
     '(test_native43_boundary_queue) run with the inherited suites.',
     '"""Native43 lifecycle CPU-only plugin discovery/source, wrapper, grade, builder-view, speed-arm, L2 row4, LOCALIZE=0, trunk, native43 and lifecycle queue checks; never used by box workers.\n\n'
     'Derived from native43_cpu_probe.py (n43-f) by tools/native43_lifecycle_tools_derive.py.\n'
     'FLASH_NATIVE35_ARM must be NATIVE43L: it selects the frozen native43 lifecycle owner runner,\n'
     'the lifecycle window (native43_lifecycle_gate) and the provenance binding; the trunk queue\n'
     'suite (test_native39_trunk_queue), the native43 queue suite (test_native43_boundary_queue)\n'
     'and the lifecycle queue suite (test_native43_lifecycle_queue) run with the inherited suites.'),
    ("suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native43_boundary_queue.py'))\n"
     'EXPECTED_TESTS=231  # the 218 trunk probe cases plus the 13 native43 boundary queue cases; pinned into the frozen runner gate\n',
     "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native43_boundary_queue.py'))\n"
     "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native43_lifecycle_queue.py'))\n"
     'EXPECTED_TESTS=%d  # the %d native43 probe cases plus the %d native43 lifecycle queue cases; pinned into the frozen runner gate\n'
     % (PARENT_TESTS + LIFECYCLE_TESTS, PARENT_TESTS, LIFECYCLE_TESTS)),
    ("assert role=='NATIVE43' and os.environ['FLASH_NATIVE34_ARM']=='O'\n",
     "assert role=='NATIVE43L' and os.environ['FLASH_NATIVE34_ARM']=='O'\n"),
    ("owner.QNAME=='q18a-native43-dense-boundary'\n",
     "owner.QNAME==%r\n" % CANDIDATE),
    ('# The frozen native43 runner gates the whole pair: window_ok = pair active + pair kill margin.\n'
     'active_and_cleanup=native43_boundary_gate.PAIR_ACTIVE_S+native43_boundary_gate.PAIR_KILL_MARGIN_S\n',
     'pair_active_and_cleanup=native43_boundary_gate.PAIR_ACTIVE_S+native43_boundary_gate.PAIR_KILL_MARGIN_S\n'
     'import native43_lifecycle_gate\n'
     'assert (native43_lifecycle_gate.NOT_BEFORE,native43_lifecycle_gate.HARD_END)==(weekend_gate.NOT_BEFORE,weekend_gate.HARD_END)\n'
     '# The frozen native43 lifecycle runner gates on lifecycle_window_ok: lifecycle active + kill margin.\n'
     'active_and_cleanup=native43_lifecycle_gate.LIFECYCLE_ACTIVE_AND_CLEANUP\n'),
    ('slot_receipts=slot_receipts+trunk_slots\n',
     'slot_receipts=slot_receipts+trunk_slots\n'
     "lifecycle_slots=sys.modules['test_native43_lifecycle_queue'].SLOT_RECEIPTS\n"
     "assert len(lifecycle_slots)==1 and lifecycle_slots[0]['draft']=='q18a_native43_lifecycle.draft.sh' and lifecycle_slots[0]['real_wall_clock'] "
     "and lifecycle_slots[0]['active_s']==native43_lifecycle_gate.LIFECYCLE_ACTIVE_S\n"
     "assert lifecycle_slots[0]['latest_entry_utc']==native43_lifecycle_gate.LATEST_ENTRY_UTC=='2026-11-02T04:21:40Z'\n"
     'slot_receipts=slot_receipts+lifecycle_slots\n'),
    ("native43_inert={p:inert.run('/source/harness/runs/q18a_native43_%s_settings.sh'%p,'NATIVE43 %s settings (LOCALIZE=0)'%p) for p in ('current','dense')}\n",
     "native43_inert={p:inert.run('/source/harness/runs/q18a_native43_%s_settings.sh'%p,'NATIVE43 %s settings (LOCALIZE=0)'%p) for p in ('current','dense','lifecycle')}\n"),
    ("Path('/out/native43-queue-check.json').write_text(json.dumps(dict(cases=native43_cases,window_active_and_cleanup_s=active_and_cleanup),sort_keys=True,indent=2)+'\\n')\n",
     "Path('/out/native43-queue-check.json').write_text(json.dumps(dict(cases=native43_cases,window_active_and_cleanup_s=pair_active_and_cleanup),sort_keys=True,indent=2)+'\\n')\n"
     '# Native43 lifecycle (short scope): the lifecycle region under bash (instruments, real session client, native43\n'
     '# and native35 lifetime grades; no leak or pressure tool), the grade over record shapes, the lifecycle CLI on the\n'
     '# trunk W2 manifest and the sourced lifecycle settings through the real launcher render and engine policy parser.\n'
     "life_cases=sys.modules['test_native43_lifecycle_queue'].QUEUE_RECEIPTS\n"
     'assert sorted(r[\'case\'] for r in life_cases)==' + repr(LIFECYCLE_CASES) + " and all(r['verdict']=='PASS' for r in life_cases)\n"
     "assert {r['case']:r['lifecycle_grade'] for r in life_cases if 'lifecycle_grade' in r}=={'tail-short-lifecycle-connected-pass':'PASS',"
     "'tail-reuse-incomplete-inconclusive':'INCONCLUSIVE','tail-row1-not-pass-withholds':'INCONCLUSIVE','tail-engine-crash-withholds':'FAIL',"
     "'tail-hygiene-refused-inconclusive':'INCONCLUSIVE'}\n"
     "assert [(r['calls'],r['session_calls'],r['not_run']) for r in life_cases if r['case']=='tail-short-lifecycle-connected-pass']=="
     "[(['e3_q18a_lifecycle'],['pause','resume','prepare','resume','pause','resume'],['cross-session-leak','partial-leak','pressure-completed','pressure-lifetime'])]\n"
     "assert [r['instruments'] for r in life_cases if r['case']=='tail-observer-mapped-withholds']==['FAIL']\n"
     "assert [(r['hosts'],r['dense_arm_refused'],r['native39_lifecycle_refused'],r['engine_policy']) for r in life_cases "
     "if r['case']=='settings-dense-policy-no-trace-no-stop']==[(['0.0.0.0','127.0.0.1'],True,True,['dense',False])]\n"
     "Path('/out/native43-lifecycle-queue-check.json').write_text(json.dumps(dict(cases=life_cases,slots=lifecycle_slots),sort_keys=True,indent=2)+'\\n')\n"),
    ('record=dict(native43_queue=\'PASS\',native43_queue_cases=native43_cases,native43_window_active_and_cleanup_s=active_and_cleanup,',
     "record=dict(native43_lifecycle_queue='PASS',native43_lifecycle_queue_receipts=life_cases,native43_lifecycle_window_active_and_cleanup_s=active_and_cleanup,"
     "native43_queue='PASS',native43_queue_cases=native43_cases,native43_window_active_and_cleanup_s=pair_active_and_cleanup,"),
)

OUTPUTS = {'native43_lifecycle_cpu_probe.py': ('native43_cpu_probe.py', PROBE_SWAPS)}


def apply(text, swaps):
    for old, new in swaps:
        assert text.count(old) == 1, (old[:80], text.count(old))
        text = text.replace(old, new)
    return text


def generate(tools=TOOLS):
    return {name: apply((tools / source).read_text(), swaps) for name, (source, swaps) in OUTPUTS.items()}


if __name__ == '__main__':
    check = '--check' in sys.argv
    for name, text in generate().items():
        path = TOOLS / name
        if check:
            assert path.read_text() == text, name
        else:
            assert not path.exists(), name + ' exists; use --check'
            path.write_text(text)
            path.chmod(0o775)
    print('native43 lifecycle bundle tools ' + ('verified' if check else 'written'))
