#!/usr/bin/env python3
"""Derive the Step A bundle CPU probe from the alpha-perf one (alpha_perf_cpu_probe.py).

step_a_cpu_probe.py = alpha_perf_cpu_probe.py with:
  * candidate q18a-step-a (the frozen Step A owner runner's QNAME); FLASH_NATIVE35_ARM stays TRUNK;
  * the step_a_gate window instead of alpha_perf_gate;
  * test_step_a_queue added to the discovered suites (every inherited suite stays), EXPECTED_TESTS 287 + the suite's;
  * the Step A draft's real-clock slot receipt appended to the inherited ones;
  * the LOCALIZE=0 inert check run on the Step A settings, held to the same expectations;
  * the Step A queue receipts asserted and recorded (step_a_queue='PASS').
The bundle self-run is tools/native43_bundle_selfrun.py with the spec's arm and probe.
Every replacement must match exactly once.  --check verifies the committed file.
"""
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
PARENT_TESTS = 287
STEP_A_TESTS = 46  # test_step_a_queue
CANDIDATE = 'q18a-step-a'
PARENT_CANDIDATE = 'q18a-alpha-perf'

PROBE_SWAPS = (
    ("suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_alpha_perf_queue.py'))\n"
     'EXPECTED_TESTS=%d  # the 263 held-f probe cases plus the 24 alpha-perf queue cases; pinned into the frozen runner gate\n'
     % PARENT_TESTS,
     "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_alpha_perf_queue.py'))\n"
     "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_step_a_queue.py'))\n"
     'EXPECTED_TESTS=%d  # the %d alpha-perf probe cases plus the %d Step A queue cases; pinned into the frozen runner gate\n'
     % (PARENT_TESTS + STEP_A_TESTS, PARENT_TESTS, STEP_A_TESTS)),
    ("owner.QNAME==%r\n" % PARENT_CANDIDATE, "owner.QNAME==%r\n" % CANDIDATE),
    ('import alpha_perf_gate\nactive_and_cleanup=alpha_perf_gate.ACTIVE_S+alpha_perf_gate.CLEANUP_S\n',
     'import alpha_perf_gate\nimport step_a_gate\nactive_and_cleanup=step_a_gate.ACTIVE_S+step_a_gate.CLEANUP_S\n'),
    ("+sys.modules['test_alpha_perf_queue'].SLOT_RECEIPTS",
     "+sys.modules['test_alpha_perf_queue'].SLOT_RECEIPTS+sys.modules['test_step_a_queue'].SLOT_RECEIPTS"),
    ("negative=inert.run('/source/harness/runs/q18a_alpha_perf_settings.sh','TRUNK settings (LOCALIZE=0)')",
     "negative=inert.run('/source/harness/runs/q18a_step_a_settings.sh','TRUNK settings (LOCALIZE=0)')"),
    ("record=dict(alpha_perf_queue='PASS',",
     "step_a_slots=sys.modules['test_step_a_queue'].SLOT_RECEIPTS\n"
     "assert len(step_a_slots)==1 and step_a_slots[0]['draft']=='q18a_step_a.draft.sh' and step_a_slots[0]['real_wall_clock'] "
     "and step_a_slots[0]['active_s']==step_a_gate.ACTIVE_S\n"
     "step_a_calls=sys.modules['test_step_a_queue'].ARGPARSE_RECEIPTS\n"
     "assert step_a_calls and all(r['verdict']=='parsed' for r in step_a_calls)\n"
     "Path('/out/step-a-queue-check.json').write_text(json.dumps(dict(slots=step_a_slots,argparse=step_a_calls),sort_keys=True,indent=2)+'\\n')\n"
     "record=dict(step_a_queue='PASS',step_a_queue_slots=step_a_slots,step_a_argparse_calls=len(step_a_calls),alpha_perf_queue='PASS',"),
)

OUTPUTS = {'step_a_cpu_probe.py': ('alpha_perf_cpu_probe.py', PROBE_SWAPS)}


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
    print('Step A bundle tools ' + ('verified' if check else 'written'))
