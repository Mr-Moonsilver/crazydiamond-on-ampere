#!/usr/bin/env python3
"""Derive the native39 lifecycle bundle self-run tools from the trunk held-b ones.

native39_lifecycle_cpu_probe.py = native39_trunk_cpu_probe.py with:
  * FLASH_NATIVE35_ARM=LIFECYCLE, candidate q18a-native39-lifecycle;
  * test_native39_lifecycle_queue added to the discovered suites, EXPECTED_TESTS
    218 -> 233 (its 15 cases);
  * the owner window boundaries of native39_lifecycle_gate (9600 s + 900 s) for the
    frozen runner's check_slot negatives;
  * the lifecycle draft's real-clock slot receipt appended to the 21 inherited ones;
  * the LOCALIZE=0 inert check run on the lifecycle settings (the P-b positive
    control unchanged);
  * the lifecycle queue receipts asserted and recorded (native39_lifecycle_queue='PASS'),
    beside the unchanged trunk queue receipts.
native39_lifecycle_bundle_selfrun.py = native39_trunk_bundle_selfrun.py with the
lifecycle probe path, schema and arm; image, mounts and PROBE_ENV are unchanged.
Every replacement must match exactly once.  --check verifies the committed files.
"""
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
LIFECYCLE_CASES = sorted(['settings-trunk-engine-observer-offers-off', 'tail-engine-crash-withholds',
                          'tail-hygiene-refused-stops-leak-and-pressure', 'tail-lifecycle-connected-pool-telemetry-absent',
                          'tail-observer-mapped-withholds', 'tail-real-leak-detected-fails', 'tail-row1-not-pass-withholds',
                          'workload-lifecycle-cli-native39-manifest'])
LIFECYCLE_TESTS = 15

PROBE_SWAPS = (
    ('"""Native39 trunk CPU-only plugin discovery/source, wrapper, grade, builder-view, speed-arm, L2 row4, LOCALIZE=0 and trunk queue checks; never used by box workers.\n\n'
     'Derived from native35_p_c_cpu_probe.py by tools/native39_trunk_tools_derive.py.\n'
     'FLASH_NATIVE35_ARM must be TRUNK: it selects the frozen trunk owner runner, the\n'
     'trunk window (native39_trunk_gate) and the provenance binding; the trunk queue\n'
     'suite (test_native39_trunk_queue) runs with the inherited suites.',
     '"""Native39 lifecycle CPU-only plugin discovery/source, wrapper, grade, builder-view, speed-arm, L2 row4, LOCALIZE=0, trunk and lifecycle queue checks; never used by box workers.\n\n'
     'Derived from native39_trunk_cpu_probe.py by tools/native39_lifecycle_tools_derive.py.\n'
     'FLASH_NATIVE35_ARM must be LIFECYCLE: it selects the frozen lifecycle owner runner,\n'
     'the lifecycle window (native39_lifecycle_gate) and the provenance binding; the\n'
     'lifecycle queue suite (test_native39_lifecycle_queue) runs with the inherited suites.'),
    ("suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native39_trunk_queue.py'))\n",
     "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native39_trunk_queue.py'))\n"
     "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native39_lifecycle_queue.py'))\n"),
    ('EXPECTED_TESTS=218  # the 183 P-c probe cases plus the 35 native39 trunk queue cases; pinned into the frozen runner gate',
     'EXPECTED_TESTS=%d  # the 218 trunk probe cases plus the %d native39 lifecycle queue cases; pinned into the frozen runner gate'
     % (218 + LIFECYCLE_TESTS, LIFECYCLE_TESTS)),
    ("role=os.environ['FLASH_NATIVE35_ARM'];assert role=='TRUNK' and os.environ['FLASH_NATIVE34_ARM']=='O'",
     "role=os.environ['FLASH_NATIVE35_ARM'];assert role=='LIFECYCLE' and os.environ['FLASH_NATIVE34_ARM']=='O'"),
    ("provenance['candidate']==owner.QNAME=='q18a-native39-trunk'",
     "provenance['candidate']==owner.QNAME=='q18a-native39-lifecycle'"),
    ('active_and_cleanup=native39_trunk_gate.TRUNK_ACTIVE_AND_CLEANUP\n',
     'import native39_lifecycle_gate\n'
     'assert (native39_lifecycle_gate.NOT_BEFORE,native39_lifecycle_gate.HARD_END)==(weekend_gate.NOT_BEFORE,weekend_gate.HARD_END)\n'
     'active_and_cleanup=native39_lifecycle_gate.LIFECYCLE_ACTIVE_AND_CLEANUP\n'),
    ('slot_receipts=slot_receipts+trunk_slots\n',
     'slot_receipts=slot_receipts+trunk_slots\n'
     "lifecycle_slots=sys.modules['test_native39_lifecycle_queue'].SLOT_RECEIPTS\n"
     "assert len(lifecycle_slots)==1 and lifecycle_slots[0]['draft']=='q18a_native39_lifecycle.draft.sh' and lifecycle_slots[0]['real_wall_clock'] and lifecycle_slots[0]['active_s']==native39_lifecycle_gate.LIFECYCLE_ACTIVE_S\n"
     "assert lifecycle_slots[0]['latest_entry_utc']==native39_lifecycle_gate.LATEST_ENTRY_UTC=='2026-11-02T03:05:00Z'\n"
     'slot_receipts=slot_receipts+lifecycle_slots\n'),
    ("negative=inert.run('/source/harness/runs/q18a_native39_trunk_settings.sh','TRUNK settings (LOCALIZE=0)')",
     "negative=inert.run('/source/harness/runs/q18a_native39_lifecycle_settings.sh','LIFECYCLE settings (LOCALIZE=0)')"),
    ('record=dict(native39_trunk_queue=',
     '# Native39 lifecycle: the lifecycle region under bash (instruments, real session client, auth hook, leaktest,\n'
     '# pressure wait, pp_release and native35 lifetime grades), the lifecycle CLI on the trunk W2 manifest and the\n'
     '# sourced lifecycle settings through the real launcher render.\n'
     "life_cases=sys.modules['test_native39_lifecycle_queue'].QUEUE_RECEIPTS\n"
     'assert sorted(r[\'case\'] for r in life_cases)==' + repr(LIFECYCLE_CASES) + " and all(r['verdict']=='PASS' for r in life_cases)\n"
     "assert {r['case']:r['lifetime_grade'] for r in life_cases if 'lifetime_grade' in r}=={'tail-lifecycle-connected-pool-telemetry-absent':'INCONCLUSIVE','tail-row1-not-pass-withholds':'INCONCLUSIVE','tail-engine-crash-withholds':'FAIL','tail-real-leak-detected-fails':'FAIL'}\n"
     "assert [(r['leak_requests'],r['pressure_lifetime'],r['lifetime_missing']) for r in life_cases if r['case']=='tail-lifecycle-connected-pool-telemetry-absent']==[(24,'INCONCLUSIVE',['pressure-lifetime:INCONCLUSIVE'])]\n"
     "assert [r['instruments'] for r in life_cases if r['case']=='tail-observer-mapped-withholds']==['FAIL']\n"
     "assert [(r['hosts'],r['same_as_trunk'],r['pool_stats']) for r in life_cases if r['case']=='settings-trunk-engine-observer-offers-off']==[(['0.0.0.0','127.0.0.1'],True,False)]\n"
     "Path('/out/native39-lifecycle-queue-check.json').write_text(json.dumps(dict(cases=life_cases,slots=lifecycle_slots),sort_keys=True,indent=2)+'\\n')\n"
     "record=dict(native39_lifecycle_queue='PASS',native39_lifecycle_queue_receipts=life_cases,native39_lifecycle_window_active_and_cleanup_s=active_and_cleanup,"
     'native39_trunk_queue='),
)
SELFRUN_SWAPS = (
    ('"""Run exact frozen native39 trunk bundle bytes in the pinned image, CPU only (native35_p_c_selfrun.py with the trunk probe)."""',
     '"""Run exact frozen native39 lifecycle bundle bytes in the pinned image, CPU only (native39_trunk_bundle_selfrun.py with the lifecycle probe)."""'),
    ('# processes under the exact trunk settings env (native35_p_c_inert_check). FLASH_NATIVE35_ARM=TRUNK\n'
     '# selects the frozen trunk owner runner.\n',
     '# processes under the exact lifecycle settings env (native35_p_c_inert_check). FLASH_NATIVE35_ARM=LIFECYCLE\n'
     '# selects the frozen lifecycle owner runner.\n'),
    ("'/source/tools/native39_trunk_cpu_probe.py'", "'/source/tools/native39_lifecycle_cpu_probe.py'"),
    ("schema='native39-trunk-bundle-selfrun/1'", "schema='native39-lifecycle-bundle-selfrun/1'"),
    ("p.add_argument('--arm',choices=('TRUNK',),required=True)", "p.add_argument('--arm',choices=('LIFECYCLE',),required=True)"),
)
OUTPUTS = {'native39_lifecycle_cpu_probe.py': ('native39_trunk_cpu_probe.py', PROBE_SWAPS),
           'native39_lifecycle_bundle_selfrun.py': ('native39_trunk_bundle_selfrun.py', SELFRUN_SWAPS)}


def apply(text, swaps):
    for old, new in swaps:
        assert text.count(old) == 1, (old[:80], text.count(old))
        text = text.replace(old, new)
    return text


def generate():
    return {name: apply((TOOLS / source).read_text(), swaps) for name, (source, swaps) in OUTPUTS.items()}


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
    print('native39 lifecycle bundle tools ' + ('verified' if check else 'written'))
