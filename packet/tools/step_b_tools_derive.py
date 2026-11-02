#!/usr/bin/env python3
"""Derive the Step B bundle CPU probe from the Step A one (step_a_cpu_probe.py).

step_b_cpu_probe.py = step_a_cpu_probe.py with:
  * candidate q18a-step-b (the frozen Step B owner runner's QNAME); FLASH_NATIVE35_ARM stays TRUNK;
  * the step_b_gate window instead of step_a_gate;
  * test_step_b_queue added to the discovered suites (every inherited suite stays), EXPECTED_TESTS 333 + the suite's - 11;
  * the Step B draft's slot receipts appended to the inherited ones (empty: its slot test is declared out);
  * the LOCALIZE=0 inert check run on the Step B settings, held to the same expectations;
  * the Step B queue receipts asserted and recorded (step_b_queue='PASS');
  * the declared real-clock window-slot tests (harness/step_b_clock_bound.py, Step B's own included) left out of the suite before
    it runs and recorded with their reason, their slot receipts asserted empty, and the frozen runner's negative instants taken
    from step_b_gate (its HARD_END is Step B's 07:00Z).
The bundle self-run is tools/native43_bundle_selfrun.py with the spec's arm and probe.
Every replacement must match exactly once.  --check verifies the committed file.
"""
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
PARENT_TESTS = 333
STEP_B_TESTS = 52  # test_step_b_queue
CLOCK_BOUND_IN_PROBE = 11  # harness/step_b_clock_bound.IN_PROBE: the declared real-clock window-slot tests in the probe's suites
EXPECTED = PARENT_TESTS - CLOCK_BOUND_IN_PROBE + STEP_B_TESTS
CANDIDATE = 'q18a-step-b'
PARENT_CANDIDATE = 'q18a-step-a'

PROBE_SWAPS = (
    ("suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_step_a_queue.py'))\n"
     'EXPECTED_TESTS=%d  # the 287 alpha-perf probe cases plus the 46 Step A queue cases; pinned into the frozen runner gate\n'
     % PARENT_TESTS,
     "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_step_a_queue.py'))\n"
     "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_step_b_queue.py'))\n"
     'EXPECTED_TESTS=%d  # the %d Step A probe cases plus the %d Step B queue cases, less %d declared real-clock window-slot '
     'tests; pinned into the frozen runner gate\n' % (EXPECTED, PARENT_TESTS, STEP_B_TESTS, CLOCK_BOUND_IN_PROBE)),
    ('result=unittest.TextTestRunner(verbosity=2).run(suite)\n',
     'import step_b_clock_bound\n'
     'suite,clock_bound_removed=step_b_clock_bound.filtered(suite)\n'
     'assert clock_bound_removed==step_b_clock_bound.IN_PROBE,clock_bound_removed\n'
     'result=unittest.TextTestRunner(verbosity=2).run(suite)\n'),
    ('for instant in (weekend_gate.NOT_BEFORE-1,weekend_gate.HARD_END-active_and_cleanup+1):\n',
     'for instant in (step_b_gate.NOT_BEFORE-1,step_b_gate.HARD_END-active_and_cleanup+1):\n'),
    ("assert len(slot_receipts)==20 and all(r['real_wall_clock'] for r in slot_receipts)\n",
     "assert slot_receipts==[]  # their real-clock slot tests are declared (step_b_clock_bound) and left out\n"),
    ("assert {r['draft'] for r in slot_receipts}>={'q18a_native35_r.draft.sh',", "assert not {r['draft'] for r in slot_receipts}&{'q18a_native35_r.draft.sh',"),
    ("assert [r['active_s'] for r in slot_receipts if r['draft']=='q18a_native35_p_c.draft.sh']==[native35p_gate.SPEED_ACTIVE_S]\n",
     "assert [r['active_s'] for r in slot_receipts if r['draft']=='q18a_native35_p_c.draft.sh']==[]\n"),
    ("assert len(trunk_slots)==1 and trunk_slots[0]['draft']=='q18a_native39_trunk.draft.sh' and trunk_slots[0]['real_wall_clock'] and trunk_slots[0]['active_s']==native39_trunk_gate.TRUNK_ACTIVE_S\n",
     "assert trunk_slots==[]\n"),
    ("assert len(alpha_slots)==1 and alpha_slots[0]['draft']=='q18a_alpha_perf.draft.sh' and alpha_slots[0]['real_wall_clock'] and alpha_slots[0]['active_s']==alpha_perf_gate.ACTIVE_S\n",
     "assert alpha_slots==[]\n"),
    ("assert len(step_a_slots)==1 and step_a_slots[0]['draft']=='q18a_step_a.draft.sh' and step_a_slots[0]['real_wall_clock'] and step_a_slots[0]['active_s']==step_a_gate.ACTIVE_S\n",
     "assert step_a_slots==[]\n"),
    ("owner.QNAME==%r\n" % PARENT_CANDIDATE, "owner.QNAME==%r\n" % CANDIDATE),
    ('import step_a_gate\nactive_and_cleanup=step_a_gate.ACTIVE_S+step_a_gate.CLEANUP_S\n',
     'import step_a_gate\nimport step_b_gate\nactive_and_cleanup=step_b_gate.ACTIVE_S+step_b_gate.CLEANUP_S\n'),
    ("+sys.modules['test_step_a_queue'].SLOT_RECEIPTS",
     "+sys.modules['test_step_a_queue'].SLOT_RECEIPTS+sys.modules['test_step_b_queue'].SLOT_RECEIPTS"),
    ("negative=inert.run('/source/harness/runs/q18a_step_a_settings.sh','TRUNK settings (LOCALIZE=0)')",
     "negative=inert.run('/source/harness/runs/q18a_step_b_settings.sh','TRUNK settings (LOCALIZE=0)')"),
    ("record=dict(step_a_queue='PASS',",
     "step_b_slots=sys.modules['test_step_b_queue'].SLOT_RECEIPTS\n"
     "assert step_b_slots==[]\n"
     "step_b_calls=sys.modules['test_step_b_queue'].ARGPARSE_RECEIPTS\n"
     "assert step_b_calls and all(r['verdict']=='parsed' for r in step_b_calls)\n"
     "Path('/out/step-b-queue-check.json').write_text(json.dumps(dict(slots=step_b_slots,argparse=step_b_calls),sort_keys=True,indent=2)+'\\n')\n"
     "record=dict(step_b_queue='PASS',step_b_queue_slots=step_b_slots,step_b_argparse_calls=len(step_b_calls),"
     "clock_bound_frozen_excluded=clock_bound_removed,"
     "clock_bound_frozen_reason=step_b_clock_bound.REASON,step_a_queue='PASS',"),
)

OUTPUTS = {'step_b_cpu_probe.py': ('step_a_cpu_probe.py', PROBE_SWAPS)}


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
    print('Step B bundle tools ' + ('verified' if check else 'written'))
