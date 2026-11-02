#!/usr/bin/env python3
"""Derive the alpha-perf bundle CPU probe from the frozen held-f one (native41_progress_cpu_probe.py).

alpha_perf_cpu_probe.py = native41_progress_cpu_probe.py with:
  * candidate q18a-alpha-perf (the frozen alpha-perf owner runner's QNAME); FLASH_NATIVE35_ARM stays TRUNK;
  * the alpha_perf_gate window instead of native41_progress_gate;
  * test_alpha_perf_queue added to the discovered suites (every inherited suite stays), EXPECTED_TESTS 263 + the suite's;
  * the alpha-perf draft's real-clock slot receipt appended to the inherited ones;
  * the LOCALIZE=0 inert check run on the alpha-perf settings, held to the same expectations;
  * the alpha-perf queue receipts asserted and recorded (alpha_perf_queue='PASS').
The bundle self-run is tools/native43_bundle_selfrun.py with the spec's arm and probe.
Every replacement must match exactly once.  --check verifies the committed file.
"""
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
PARENT_TESTS = 263
ALPHA_TESTS = 24  # test_alpha_perf_queue
CANDIDATE = 'q18a-alpha-perf'

PROBE_SWAPS = (
    ("suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native41_progress_queue.py'))\n"
     'EXPECTED_TESTS=263  # held-e cases plus 19 held-f progress queue cases\n',
     "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native41_progress_queue.py'))\n"
     "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_alpha_perf_queue.py'))\n"
     'EXPECTED_TESTS=%d  # the %d held-f probe cases plus the %d alpha-perf queue cases; pinned into the frozen runner gate\n'
     % (PARENT_TESTS + ALPHA_TESTS, PARENT_TESTS, ALPHA_TESTS)),
    ("owner.QNAME=='q18a-native41-progress'\n", "owner.QNAME==%r\n" % CANDIDATE),
    ('import native41_progress_gate\nactive_and_cleanup=native41_progress_gate.ACTIVE_S+native41_progress_gate.CLEANUP_S\n',
     'import alpha_perf_gate\nactive_and_cleanup=alpha_perf_gate.ACTIVE_S+alpha_perf_gate.CLEANUP_S\n'),
    ("+sys.modules['test_native41_progress_queue'].SLOT_RECEIPTS",
     "+sys.modules['test_native41_progress_queue'].SLOT_RECEIPTS+sys.modules['test_alpha_perf_queue'].SLOT_RECEIPTS"),
    ("negative=inert.run('/source/harness/runs/q18a_native41_progress_settings.sh','TRUNK settings (LOCALIZE=0)')",
     "negative=inert.run('/source/harness/runs/q18a_alpha_perf_settings.sh','TRUNK settings (LOCALIZE=0)')"),
    ("record=dict(native39_trunk_queue='PASS',",
     "alpha_slots=sys.modules['test_alpha_perf_queue'].SLOT_RECEIPTS\n"
     "assert len(alpha_slots)==1 and alpha_slots[0]['draft']=='q18a_alpha_perf.draft.sh' and alpha_slots[0]['real_wall_clock'] "
     "and alpha_slots[0]['active_s']==alpha_perf_gate.ACTIVE_S\n"
     "alpha_calls=sys.modules['test_alpha_perf_queue'].ARGPARSE_RECEIPTS\n"
     "assert alpha_calls and all(r['verdict']=='parsed' for r in alpha_calls)\n"
     "Path('/out/alpha-perf-queue-check.json').write_text(json.dumps(dict(slots=alpha_slots,argparse=alpha_calls),sort_keys=True,indent=2)+'\\n')\n"
     "record=dict(alpha_perf_queue='PASS',alpha_perf_queue_slots=alpha_slots,alpha_perf_argparse_calls=len(alpha_calls),native39_trunk_queue='PASS',"),
)

OUTPUTS = {'alpha_perf_cpu_probe.py': ('native41_progress_cpu_probe.py', PROBE_SWAPS)}


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
    print('alpha-perf bundle tools ' + ('verified' if check else 'written'))
