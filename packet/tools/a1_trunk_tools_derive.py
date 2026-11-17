#!/usr/bin/env python3
"""Derive the q18a-a1-trunk bundle CPU probe from inwait's (inwait_cpu_probe.py), as inwait derived its own from c1b's.

a1_trunk_cpu_probe.py = inwait_cpu_probe.py with: candidate q18a-a1-trunk (the owner runner's QNAME); the a1_trunk_gate budget; test_a1_trunk_queue added to the discovered suites (every inherited suite stays),
EXPECTED_TESTS = inwait's + the suite's; its SLOT_RECEIPTS appended (empty); the LOCALIZE=0 inert check on this queue's settings; its queue receipts asserted and recorded (a1_trunk_queue='PASS'), inwait's kept.
Every replacement must match exactly once. --check verifies the committed file.
"""
from pathlib import Path
import re
import subprocess
import sys

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import alpha_next_tools_derive as parent_derive  # noqa: E402

QUEUE_SUITE = 'test_a1_trunk_queue'
CANDIDATE = 'q18a-a1-trunk'
PARENT_CANDIDATE = 'q18a-inwait'
PARENT_PROBE = 'inwait_cpu_probe.py'
EXPECTED_LINE = re.compile(r'^EXPECTED_TESTS=(\d+)  # [^\n]*\n', re.M)


def parent_tests():
    (found,) = EXPECTED_LINE.findall((TOOLS / PARENT_PROBE).read_text())
    return int(found)


def queue_tests():
    code = ('import sys,unittest,importlib; sys.path.insert(0,"."); import step_b_clock_bound as c; '
            's=unittest.defaultTestLoader.loadTestsFromName(%r); assert not c.expected_removed([%r]); '
            'm=importlib.import_module(%r); assert hasattr(m,"SLOT_RECEIPTS") and hasattr(m,"ARGPARSE_RECEIPTS"), "receipt lists"; '
            'print(s.countTestCases())' % (QUEUE_SUITE, QUEUE_SUITE, QUEUE_SUITE))
    count = int(subprocess.check_output([sys.executable, '-B', '-c', code], cwd=TOOLS.parent / 'harness', text=True).strip().splitlines()[-1])
    assert count > 0, count
    return count


def expected():
    return parent_tests() + queue_tests()


def probe_swaps():
    parent, ours = parent_tests(), queue_tests()
    text = (TOOLS / PARENT_PROBE).read_text()
    line = EXPECTED_LINE.search(text).group(0)
    discover = "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_inwait_queue.py'))\n"
    return (
        (discover + line,
         discover + "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_a1_trunk_queue.py'))\n"
         'EXPECTED_TESTS=%d  # the %d inwait probe cases plus the %d a1-trunk queue cases; pinned into the frozen runner gate\n' % (parent + ours, parent, ours)),
        ("owner.QNAME==%r\n" % PARENT_CANDIDATE, "owner.QNAME==%r\n" % CANDIDATE),
        ('import inwait_gate\nactive_and_cleanup=inwait_gate.ACTIVE_S+inwait_gate.CLEANUP_S\n',
         'import inwait_gate\nimport a1_trunk_gate\nactive_and_cleanup=a1_trunk_gate.ACTIVE_S+a1_trunk_gate.CLEANUP_S\n'),
        ("+sys.modules['test_inwait_queue'].SLOT_RECEIPTS\n", "+sys.modules['test_inwait_queue'].SLOT_RECEIPTS+sys.modules['test_a1_trunk_queue'].SLOT_RECEIPTS\n"),
        ("negative=inert.run('/source/harness/runs/q18a_inwait_settings.sh','TRUNK settings (LOCALIZE=0)')", "negative=inert.run('/source/harness/runs/q18a_a1_trunk_settings.sh','TRUNK settings (LOCALIZE=0)')"),
        ("record=dict(inwait_queue='PASS',",
         "a1_slots=sys.modules['test_a1_trunk_queue'].SLOT_RECEIPTS\n"
         "assert a1_slots==[]  # no absolute-time window here either\n"
         "a1_calls=sys.modules['test_a1_trunk_queue'].ARGPARSE_RECEIPTS\n"
         "assert a1_calls and all(r['verdict']=='parsed' for r in a1_calls)\n"
         "Path('/out/a1-trunk-queue-check.json').write_text(json.dumps(dict(slots=a1_slots,argparse=a1_calls),sort_keys=True,indent=2)+'\\n')\n"
         "record=dict(a1_trunk_queue='PASS',a1_trunk_argparse_calls=len(a1_calls),inwait_queue='PASS',"),
    )


OUTPUTS = {'a1_trunk_cpu_probe.py': (PARENT_PROBE, probe_swaps)}


def generate(tools=TOOLS):
    out = {}
    for name, (source, swaps) in OUTPUTS.items():
        text = parent_derive.apply((tools / source).read_text(), swaps())
        parent_derive.no_swapped_module(text)
        out[name] = text
    return out


if __name__ == '__main__':
    check = '--check' in sys.argv
    for name, text in generate().items():
        path = TOOLS / name
        if check:
            assert path.read_text() == text, name
        else:
            path.write_text(text)
            path.chmod(0o775)
    print('a1-trunk bundle tools ' + ('verified' if check else 'written'))
