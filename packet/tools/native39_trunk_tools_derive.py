#!/usr/bin/env python3
"""Derive the native39 trunk bundle self-run tools from the frozen native35 P-c ones.

native39_trunk_cpu_probe.py = native35_p_c_cpu_probe.py with:
  * FLASH_NATIVE35_ARM=TRUNK, candidate q18a-native39-trunk;
  * test_native39_trunk_queue added to the discovered suites, EXPECTED_TESTS
    183 -> 217 (its 34 cases);
  * the owner window boundaries of native39_trunk_gate (16200 s + 900 s) for the
    frozen runner's check_slot negatives;
  * the trunk draft's real-clock slot receipt appended to the 20 inherited ones;
  * the LOCALIZE=0 inert check run on the trunk settings (the P-b positive
    control unchanged);
  * the trunk queue receipts asserted and recorded (native39_trunk_queue='PASS').
native39_trunk_bundle_selfrun.py = native35_p_c_selfrun.py with the trunk probe
path, schema and arm; image, mounts and PROBE_ENV are unchanged.
Every replacement must match exactly once.  --check verifies the committed files;
--rewrite regenerates them after a declared change of this derivation.
"""
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
TRUNK_CASES = sorted(['bind-absent', 'bind-after-first-200', 'bind-before-first-200', 'bind-other-container', 'box-access-lines',
                      'guard-count-differs', 'guard-declared-counts', 'guard-no-engine', 'guard-refused-band',
                      'guard-tampered-declaration', 'health-deadline', 'launcher-loopback-host',
                      'round-wrapper-declared-not-run-only', 'tail-foreign-at-entry-no-split', 'tail-hidden-arrival-invalid',
                      'tail-maps-unreadable-withholds', 'tail-not-idle-withholds-w2', 'tail-observer-mapped-withholds',
                      'tail-observer-off-sealed-w1-plain-on-full-rounds', 'tail-transient-foreign-ingress-invalid',
                      'tail-unqualified-withholds-all', 'tail-unrestored-split-withholds-plain'])

PROBE_SWAPS = (
    ('"""Native35 P-c CPU-only plugin discovery/source, wrapper, grade, builder-view, speed-arm, L2 row4 and LOCALIZE=0 checks; never used by box workers.\n\n'
     'Derived from native35_p_b_cpu_probe.py (P-b). FLASH_NATIVE35_ARM must be P-C:\n'
     'it selects the frozen P-c owner runner, the P speed window and the provenance\n'
     'binding.',
     '"""Native39 trunk CPU-only plugin discovery/source, wrapper, grade, builder-view, speed-arm, L2 row4, LOCALIZE=0 and trunk queue checks; never used by box workers.\n\n'
     'Derived from native35_p_c_cpu_probe.py by tools/native39_trunk_tools_derive.py.\n'
     'FLASH_NATIVE35_ARM must be TRUNK: it selects the frozen trunk owner runner, the\n'
     'trunk window (native39_trunk_gate) and the provenance binding; the trunk queue\n'
     'suite (test_native39_trunk_queue) runs with the inherited suites.'),
    ("suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native35_*.py'))\n",
     "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native35_*.py'))\n"
     "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native39_trunk_queue.py'))\n"),
    ('EXPECTED_TESTS=183  # 98 native30-34, 27 native35 R/O/L, 17 native35 P, 11 native35 L2, 3 native35 P-b, 5 native35 P-c, 12 native35 P-c row4 and 10 P-c capacity continuation cases; pinned into the frozen runner gate',
     'EXPECTED_TESTS=217  # the 183 P-c probe cases plus the 34 native39 trunk queue cases; pinned into the frozen runner gate'),
    ("role=os.environ['FLASH_NATIVE35_ARM'];assert role=='P-C' and os.environ['FLASH_NATIVE34_ARM']=='O'",
     "role=os.environ['FLASH_NATIVE35_ARM'];assert role=='TRUNK' and os.environ['FLASH_NATIVE34_ARM']=='O'"),
    ("provenance['candidate']==owner.QNAME=='q18a-native35-'+role.lower()",
     "provenance['candidate']==owner.QNAME=='q18a-native39-trunk'"),
    ('active_and_cleanup=native35p_gate.SPEED_ACTIVE_AND_CLEANUP\n',
     'import native39_trunk_gate\n'
     'assert (native39_trunk_gate.NOT_BEFORE,native39_trunk_gate.HARD_END)==(weekend_gate.NOT_BEFORE,weekend_gate.HARD_END)\n'
     'active_and_cleanup=native39_trunk_gate.TRUNK_ACTIVE_AND_CLEANUP\n'),
    ("assert [r['active_s'] for r in slot_receipts if r['draft']=='q18a_native35_p_c.draft.sh']==[native35p_gate.SPEED_ACTIVE_S]\n",
     "assert [r['active_s'] for r in slot_receipts if r['draft']=='q18a_native35_p_c.draft.sh']==[native35p_gate.SPEED_ACTIVE_S]\n"
     "trunk_slots=sys.modules['test_native39_trunk_queue'].SLOT_RECEIPTS\n"
     "assert len(trunk_slots)==1 and trunk_slots[0]['draft']=='q18a_native39_trunk.draft.sh' and trunk_slots[0]['real_wall_clock'] and trunk_slots[0]['active_s']==native39_trunk_gate.TRUNK_ACTIVE_S\n"
     "slot_receipts=slot_receipts+trunk_slots\n"),
    ("negative=inert.run('/source/harness/runs/q18a_native35_p_c_settings.sh','P-C settings (LOCALIZE=0)')",
     "negative=inert.run('/source/harness/runs/q18a_native39_trunk_settings.sh','TRUNK settings (LOCALIZE=0)')"),
    ('record=dict(native35_p_c_continuation=',
     '# Native39 trunk: the queue\'s own segments under bash (bind receipt with the real host observer; row4 declared\n'
     '# NOT_RUN, per-rank instruments, sealed native W1, ingress, typed FULL + idle before the plain W1, W2 rounds,\n'
     '# returns and summary with the real checks, client and grade; W2 length guard; sourced settings through the real\n'
     '# launcher render; real box access lines).\n'
     "trunk_cases=sys.modules['test_native39_trunk_queue'].QUEUE_RECEIPTS\n"
     'assert sorted(r[\'case\'] for r in trunk_cases)==' + repr(TRUNK_CASES) + " and all(r['verdict']=='PASS' for r in trunk_cases)\n"
     "assert {r['case']:r['bind'] for r in trunk_cases if r['case'].startswith('bind-')}=={'bind-absent':'INCONCLUSIVE','bind-after-first-200':'FAIL','bind-before-first-200':'PASS','bind-other-container':'INCONCLUSIVE'}\n"
     "assert [r['trunk_hosts'] for r in trunk_cases if r['case']=='launcher-loopback-host']==[['0.0.0.0','127.0.0.1']]\n"
     "assert [r['plain_requests'] for r in trunk_cases if r['case']=='tail-observer-off-sealed-w1-plain-on-full-rounds']==[8]\n"
     "assert {r['case']:r['instruments'] for r in trunk_cases if 'instruments' in r}=={'tail-observer-mapped-withholds':'FAIL','tail-maps-unreadable-withholds':'INCONCLUSIVE'}\n"
     "assert [(r['full_idle'],r['plain_requests']) for r in trunk_cases if r['case']=='tail-unrestored-split-withholds-plain']==[('INCONCLUSIVE',0)]\n"
     "assert [r['split_entries'] for r in trunk_cases if r['case']=='tail-foreign-at-entry-no-split']==[0]\n"
     "assert [r['accepted'] for r in trunk_cases if r['case']=='round-wrapper-declared-not-run-only']==[['declared']]\n"
     "Path('/out/native39-trunk-queue-check.json').write_text(json.dumps(dict(cases=trunk_cases,slots=trunk_slots),sort_keys=True,indent=2)+'\\n')\n"
     "record=dict(native39_trunk_queue='PASS',native39_trunk_queue_receipts=trunk_cases,native39_trunk_window_active_and_cleanup_s=active_and_cleanup,"
     'native35_p_c_continuation='),
)
SELFRUN_SWAPS = (
    ('"""Run exact frozen native35 P-c bytes in the pinned image, CPU only (derived from native35_p_b_selfrun.py)."""',
     '"""Run exact frozen native39 trunk bundle bytes in the pinned image, CPU only (native35_p_c_selfrun.py with the trunk probe)."""'),
    ('# FLASH_NATIVE30_LOCALIZE=0 and FLASH_NATIVE34_DIAG=0, which the probe checks in fresh\n'
     '# processes under the exact settings env (native35_p_c_inert_check). FLASH_NATIVE35_ARM=P-C\n'
     '# selects the frozen owner runner.\n',
     '# FLASH_NATIVE30_LOCALIZE=0 and FLASH_NATIVE34_DIAG=0, which the probe checks in fresh\n'
     '# processes under the exact trunk settings env (native35_p_c_inert_check). FLASH_NATIVE35_ARM=TRUNK\n'
     '# selects the frozen trunk owner runner.\n'),
    ("'/source/tools/native35_p_c_cpu_probe.py'", "'/source/tools/native39_trunk_cpu_probe.py'"),
    ("schema='native35-p-c-selfrun/1'", "schema='native39-trunk-bundle-selfrun/1'"),
    ("p.add_argument('--arm',choices=('P-C',),required=True)", "p.add_argument('--arm',choices=('TRUNK',),required=True)"),
)
OUTPUTS = {'native39_trunk_cpu_probe.py': ('native35_p_c_cpu_probe.py', PROBE_SWAPS),
           'native39_trunk_bundle_selfrun.py': ('native35_p_c_selfrun.py', SELFRUN_SWAPS)}


def apply(text, swaps):
    for old, new in swaps:
        assert text.count(old) == 1, (old[:80], text.count(old))
        text = text.replace(old, new)
    return text


def generate():
    return {name: apply((TOOLS / source).read_text(), swaps) for name, (source, swaps) in OUTPUTS.items()}


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
    print('native39 trunk bundle tools ' + ('verified' if check else 'rewritten' if rewrite else 'written'))
