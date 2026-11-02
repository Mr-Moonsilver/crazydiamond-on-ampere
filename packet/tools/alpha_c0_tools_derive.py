#!/usr/bin/env python3
"""Derive the C0 bundle CPU probe from the alpha-perf one (alpha_perf_cpu_probe.py).

alpha_c0_cpu_probe.py = alpha_perf_cpu_probe.py with:
  * candidate q18a-alpha-c0 (the frozen C0 owner runner's QNAME); FLASH_NATIVE35_ARM stays TRUNK;
  * the alpha_c0_gate window instead of alpha_perf_gate;
  * test_alpha_c0_queue added to the discovered suites (every inherited suite stays), EXPECTED_TESTS 287 + the suite's;
  * the C0 draft's real-clock slot receipt appended to the inherited ones;
  * the LOCALIZE=0 inert check run on the C0 settings, held to the same expectations;
  * the C0 queue receipts asserted and recorded (alpha_c0_queue='PASS');
  * the declared real-clock window-slot exclusion (harness/clock_bound.py; operator, 2 Nov: box timing is governed by the lead at
    launch): the listed ids are stripped from the discovered suites, EXPECTED_TESTS drops by the 10 present here, and the slot
    receipt asserts read what still runs; the excluded ids are recorded;
  * the frozen owner runner's check_slot negatives at C0's own boundaries (its window is alpha_c0_gate's own hard end).
The bundle self-run is tools/native43_bundle_selfrun.py with the spec's arm and probe.
Every replacement must match exactly once.  --check verifies the committed file.
"""
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
PARENT_TESTS = 287
ALPHA_C0_TESTS = 64  # test_alpha_c0_queue
CLOCK_BOUND_HERE = 10  # harness/clock_bound.py ids among the suites this probe discovers (9 inherited + C0's own)
CANDIDATE = 'q18a-alpha-c0'
PARENT_CANDIDATE = 'q18a-alpha-perf'

PROBE_SWAPS = (
    ("suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_alpha_perf_queue.py'))\n"
     'EXPECTED_TESTS=%d  # the 263 held-f probe cases plus the 24 alpha-perf queue cases; pinned into the frozen runner gate\n'
     % PARENT_TESTS,
     "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_alpha_perf_queue.py'))\n"
     "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_alpha_c0_queue.py'))\n"
     "# Declared exclusion (harness/clock_bound.py; operator, 2 Nov: box timing is governed by the lead at launch). Test files unchanged.\n"
     "import clock_bound\n"
     "suite,clock_excluded=clock_bound.strip(suite)\n"
     "assert len(clock_excluded)==%d and set(clock_excluded)<=set(clock_bound.CLOCK_BOUND_FROZEN),('clock-bound-exclusion-differs',sorted(clock_excluded))\n"
     'EXPECTED_TESTS=%d  # the %d alpha-perf probe cases plus the %d C0 queue cases, less %d declared real-clock window-slot tests; pinned into the frozen runner gate\n'
     % (CLOCK_BOUND_HERE, PARENT_TESTS + ALPHA_C0_TESTS - CLOCK_BOUND_HERE, PARENT_TESTS, ALPHA_C0_TESTS, CLOCK_BOUND_HERE)),
    ("for instant in (weekend_gate.NOT_BEFORE-1,weekend_gate.HARD_END-active_and_cleanup+1):\n",
     "# The frozen runner's check_slot reads alpha_c0_gate.window_ok (C0's own hard end), so its negatives are C0's boundaries.\n"
     "for instant in (alpha_c0_gate.NOT_BEFORE-1,alpha_c0_gate.HARD_END-active_and_cleanup+1):\n"),
    ("assert len(slot_receipts)==20 and all(r['real_wall_clock'] for r in slot_receipts)\n"
     "assert {r['draft'] for r in slot_receipts}>={'q18a_native35_r.draft.sh','q18a_native35_o.draft.sh','q18a_native35_l.draft.sh','q18a_native35_p.draft.sh','q18a_native35_p_b.draft.sh','q18a_native35_p_c.draft.sh','q18a_native35_l2.draft.sh'}\n"
     "assert [r['active_s'] for r in slot_receipts if r['draft']=='q18a_native35_p_c.draft.sh']==[native35p_gate.SPEED_ACTIVE_S]\n",
     "assert all(r['real_wall_clock'] for r in slot_receipts)  # the drafts' own slot tests are excluded (clock_bound); others' receipts kept\n"),
    ("assert len(trunk_slots)==1 and trunk_slots[0]['draft']=='q18a_native39_trunk.draft.sh' and trunk_slots[0]['real_wall_clock'] and trunk_slots[0]['active_s']==native39_trunk_gate.TRUNK_ACTIVE_S\n",
     "assert trunk_slots==[]  # its one slot test is excluded (clock_bound)\n"),
    ("assert len(alpha_slots)==1 and alpha_slots[0]['draft']=='q18a_alpha_perf.draft.sh' and alpha_slots[0]['real_wall_clock'] and alpha_slots[0]['active_s']==alpha_perf_gate.ACTIVE_S\n",
     "assert alpha_slots==[]  # its one slot test is excluded (clock_bound)\n"),
    ("owner.QNAME==%r\n" % PARENT_CANDIDATE, "owner.QNAME==%r\n" % CANDIDATE),
    ('import alpha_perf_gate\nactive_and_cleanup=alpha_perf_gate.ACTIVE_S+alpha_perf_gate.CLEANUP_S\n',
     'import alpha_perf_gate\nimport alpha_c0_gate\nactive_and_cleanup=alpha_c0_gate.ACTIVE_S+alpha_c0_gate.CLEANUP_S\n'),
    ("+sys.modules['test_alpha_perf_queue'].SLOT_RECEIPTS",
     "+sys.modules['test_alpha_perf_queue'].SLOT_RECEIPTS+sys.modules['test_alpha_c0_queue'].SLOT_RECEIPTS"),
    ("negative=inert.run('/source/harness/runs/q18a_alpha_perf_settings.sh','TRUNK settings (LOCALIZE=0)')",
     "negative=inert.run('/source/harness/runs/q18a_alpha_c0_settings.sh','TRUNK settings (LOCALIZE=0)')"),
    ("record=dict(alpha_perf_queue='PASS',",
     "alpha_c0_slots=sys.modules['test_alpha_c0_queue'].SLOT_RECEIPTS\n"
     "assert alpha_c0_slots==[]  # C0's own slot test is excluded too (clock_bound)\n"
     "alpha_c0_calls=sys.modules['test_alpha_c0_queue'].ARGPARSE_RECEIPTS\n"
     "assert alpha_c0_calls and all(r['verdict']=='parsed' for r in alpha_c0_calls)\n"
     "Path('/out/alpha-c0-queue-check.json').write_text(json.dumps(dict(slots=alpha_c0_slots,argparse=alpha_c0_calls),sort_keys=True,indent=2)+'\\n')\n"
     "record=dict(clock_bound_excluded=sorted(clock_excluded),clock_bound_reason=clock_bound.REASON,alpha_c0_queue='PASS',alpha_c0_queue_slots=alpha_c0_slots,alpha_c0_argparse_calls=len(alpha_c0_calls),alpha_perf_queue='PASS',"),
)

OUTPUTS = {'alpha_c0_cpu_probe.py': ('alpha_perf_cpu_probe.py', PROBE_SWAPS)}


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
            path.write_text(text)
            path.chmod(0o775)
    print('C0 bundle tools ' + ('verified' if check else 'written'))
