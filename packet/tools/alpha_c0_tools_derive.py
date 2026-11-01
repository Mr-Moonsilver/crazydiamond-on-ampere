#!/usr/bin/env python3
"""Derive the C0 bundle CPU probe from the alpha-perf one (alpha_perf_cpu_probe.py).

alpha_c0_cpu_probe.py = alpha_perf_cpu_probe.py with:
  * candidate q18a-alpha-c0 (the frozen C0 owner runner's QNAME); FLASH_NATIVE35_ARM stays TRUNK;
  * the alpha_c0_gate window instead of alpha_perf_gate;
  * test_alpha_c0_queue added to the discovered suites (every inherited suite stays), EXPECTED_TESTS 287 + the suite's;
  * the C0 draft's real-clock slot receipt appended to the inherited ones;
  * the LOCALIZE=0 inert check run on the C0 settings, held to the same expectations;
  * the C0 queue receipts asserted and recorded (alpha_c0_queue='PASS').
The bundle self-run is tools/native43_bundle_selfrun.py with the spec's arm and probe.
Every replacement must match exactly once.  --check verifies the committed file.
"""
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
PARENT_TESTS = 287
ALPHA_C0_TESTS = 45  # test_alpha_c0_queue
CANDIDATE = 'q18a-alpha-c0'
PARENT_CANDIDATE = 'q18a-alpha-perf'

PROBE_SWAPS = (
    ("suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_alpha_perf_queue.py'))\n"
     'EXPECTED_TESTS=%d  # the 263 held-f probe cases plus the 24 alpha-perf queue cases; pinned into the frozen runner gate\n'
     % PARENT_TESTS,
     "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_alpha_perf_queue.py'))\n"
     "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_alpha_c0_queue.py'))\n"
     'EXPECTED_TESTS=%d  # the %d alpha-perf probe cases plus the %d C0 queue cases; pinned into the frozen runner gate\n'
     % (PARENT_TESTS + ALPHA_C0_TESTS, PARENT_TESTS, ALPHA_C0_TESTS)),
    ("owner.QNAME==%r\n" % PARENT_CANDIDATE, "owner.QNAME==%r\n" % CANDIDATE),
    ('import alpha_perf_gate\nactive_and_cleanup=alpha_perf_gate.ACTIVE_S+alpha_perf_gate.CLEANUP_S\n',
     'import alpha_perf_gate\nimport alpha_c0_gate\nactive_and_cleanup=alpha_c0_gate.ACTIVE_S+alpha_c0_gate.CLEANUP_S\n'),
    ("+sys.modules['test_alpha_perf_queue'].SLOT_RECEIPTS",
     "+sys.modules['test_alpha_perf_queue'].SLOT_RECEIPTS+sys.modules['test_alpha_c0_queue'].SLOT_RECEIPTS"),
    ("negative=inert.run('/source/harness/runs/q18a_alpha_perf_settings.sh','TRUNK settings (LOCALIZE=0)')",
     "negative=inert.run('/source/harness/runs/q18a_alpha_c0_settings.sh','TRUNK settings (LOCALIZE=0)')"),
    ("record=dict(alpha_perf_queue='PASS',",
     "alpha_c0_slots=sys.modules['test_alpha_c0_queue'].SLOT_RECEIPTS\n"
     "assert len(alpha_c0_slots)==1 and alpha_c0_slots[0]['draft']=='q18a_alpha_c0.draft.sh' and alpha_c0_slots[0]['real_wall_clock'] "
     "and alpha_c0_slots[0]['active_s']==alpha_c0_gate.ACTIVE_S\n"
     "alpha_c0_calls=sys.modules['test_alpha_c0_queue'].ARGPARSE_RECEIPTS\n"
     "assert alpha_c0_calls and all(r['verdict']=='parsed' for r in alpha_c0_calls)\n"
     "Path('/out/alpha-c0-queue-check.json').write_text(json.dumps(dict(slots=alpha_c0_slots,argparse=alpha_c0_calls),sort_keys=True,indent=2)+'\\n')\n"
     "record=dict(alpha_c0_queue='PASS',alpha_c0_queue_slots=alpha_c0_slots,alpha_c0_argparse_calls=len(alpha_c0_calls),alpha_perf_queue='PASS',"),
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
