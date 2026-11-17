#!/usr/bin/env python3
"""Derive the native43 l-f bundle CPU probe from the frozen native43 lifecycle (l-e) one.

native43_lf_cpu_probe.py = native43_lifecycle_cpu_probe.py (the probe the l-e bundle ran) with:
  * candidate q18a-native43-lifecycle-f (the frozen l-f owner runner's QNAME); FLASH_NATIVE35_ARM stays NATIVE43L
    (the lifecycle runner, the lifecycle window native43_lifecycle_gate);
  * test_native43_lf_queue added to the discovered suites (every inherited suite stays, the l-e lifecycle queue
    suite included), EXPECTED_TESTS 247 -> 247 + LF_TESTS;
  * the l-f draft's real-clock slot receipt appended to the inherited ones (same window as l-e);
  * the LOCALIZE=0 inert check also run on the l-f settings, held to the same expectations;
  * the l-f queue receipts asserted and recorded (native43_lf_queue='PASS'), beside the unchanged lifecycle,
    native43 and trunk queue receipts.
The bundle self-run is tools/native43_bundle_selfrun.py with the spec's arm and probe.
Every replacement must match exactly once.  --check verifies the committed file.
"""
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
PARENT_TESTS = 247
LF_TESTS = 11  # test_native43_lf_queue
LF_CASES = ['lf-tail-hung-lifecycle-dead', 'lf-tail-watched-pass']
CANDIDATE = 'q18a-native43-lifecycle-f'

PROBE_SWAPS = (
    ('"""Native43 lifecycle CPU-only plugin discovery/source, wrapper, grade, builder-view, speed-arm, L2 row4, LOCALIZE=0, trunk, native43 and lifecycle queue checks; never used by box workers.\n\n'
     'Derived from native43_cpu_probe.py (n43-f) by tools/native43_lifecycle_tools_derive.py.\n',
     '"""Native43 l-f CPU-only plugin discovery/source, wrapper, grade, builder-view, speed-arm, L2 row4, LOCALIZE=0, trunk, native43, lifecycle and l-f queue checks; never used by box workers.\n\n'
     'Derived from native43_lifecycle_cpu_probe.py (l-e) by tools/native43_lf_tools_derive.py, which was derived\n'
     'from native43_cpu_probe.py (n43-f) by tools/native43_lifecycle_tools_derive.py.  The l-f queue suite\n'
     '(test_native43_lf_queue) runs with the inherited suites; the candidate is q18a-native43-lifecycle-f.\n'),
    ("suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native43_lifecycle_queue.py'))\n"
     'EXPECTED_TESTS=247  # the 231 native43 probe cases plus the 16 native43 lifecycle queue cases; pinned into the frozen runner gate\n',
     "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native43_lifecycle_queue.py'))\n"
     "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native43_lf_queue.py'))\n"
     'EXPECTED_TESTS=%d  # the %d native43 lifecycle probe cases plus the %d native43 l-f queue cases; pinned into the frozen runner gate\n'
     % (PARENT_TESTS + LF_TESTS, PARENT_TESTS, LF_TESTS)),
    ("owner.QNAME=='q18a-native43-lifecycle'\n",
     "owner.QNAME==%r\n" % CANDIDATE),
    ('slot_receipts=slot_receipts+lifecycle_slots\n',
     'slot_receipts=slot_receipts+lifecycle_slots\n'
     "lf_slots=sys.modules['test_native43_lf_queue'].SLOT_RECEIPTS\n"
     "assert len(lf_slots)==1 and lf_slots[0]['draft']=='q18a_native43_lifecycle_f.draft.sh' and lf_slots[0]['real_wall_clock'] "
     "and lf_slots[0]['active_s']==native43_lifecycle_gate.LIFECYCLE_ACTIVE_S\n"
     "assert lf_slots[0]['latest_entry_utc']==native43_lifecycle_gate.LATEST_ENTRY_UTC=='2026-11-02T04:21:40Z'\n"
     'slot_receipts=slot_receipts+lf_slots\n'),
    ("for p in ('current','dense','lifecycle')}\n",
     "for p in ('current','dense','lifecycle','lifecycle_f')}\n"),
    ("Path('/out/native43-lifecycle-queue-check.json').write_text(json.dumps(dict(cases=life_cases,slots=lifecycle_slots),sort_keys=True,indent=2)+'\\n')\n",
     "Path('/out/native43-lifecycle-queue-check.json').write_text(json.dumps(dict(cases=life_cases,slots=lifecycle_slots),sort_keys=True,indent=2)+'\\n')\n"
     '# Native43 l-f: the l-f lifecycle region under bash with the real engine watch around the lifecycle client and the\n'
     '# typed FULL hygiene return (healthy pass-through; on the l-e teardown line a hung lifecycle client and the hygiene\n'
     '# client after it are both stopped, the FULL prepare never sent; the lifecycle grade reads INCONCLUSIVE).\n'
     "lf_cases=sys.modules['test_native43_lf_queue'].QUEUE_RECEIPTS\n"
     "assert sorted(r['case'] for r in lf_cases)==" + repr(LF_CASES) + " and all(r['verdict']=='PASS' for r in lf_cases)\n"
     "assert {r['case']:r['lifecycle_grade'] for r in lf_cases}=={'lf-tail-watched-pass':'PASS','lf-tail-hung-lifecycle-dead':'INCONCLUSIVE'}\n"
     "assert {r['case']:r['watches'] for r in lf_cases}=={'lf-tail-watched-pass':['CLIENT_EXIT','CLIENT_EXIT'],'lf-tail-hung-lifecycle-dead':['DEAD','DEAD']}\n"
     "Path('/out/native43-lf-queue-check.json').write_text(json.dumps(dict(cases=lf_cases,slots=lf_slots),sort_keys=True,indent=2)+'\\n')\n"),
    ("record=dict(native43_lifecycle_queue='PASS',",
     "record=dict(native43_lf_queue='PASS',native43_lf_queue_receipts=lf_cases,native43_lifecycle_queue='PASS',"),
)

OUTPUTS = {'native43_lf_cpu_probe.py': ('native43_lifecycle_cpu_probe.py', PROBE_SWAPS)}


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
    print('native43 l-f bundle tools ' + ('verified' if check else 'written'))
