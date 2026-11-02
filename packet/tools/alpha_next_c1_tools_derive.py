#!/usr/bin/env python3
"""Derive the alpha-next-c1 bundle CPU probe from alpha-next's (alpha_next_cpu_probe.py), as alpha-next derived its own from Step B's.

alpha_next_c1_cpu_probe.py = alpha_next_cpu_probe.py with:
  * candidate q18a-alpha-next-c1 (the frozen alpha-next-c1 owner runner's QNAME); FLASH_NATIVE35_ARM stays TRUNK;
  * the alpha_next_c1_gate budget beside alpha-next's (alpha_next_c1_gate.ACTIVE_S + CLEANUP_S; no absolute-time window);
  * test_alpha_next_c1_queue added to the discovered suites (every inherited suite stays, Step B's declared real-clock exclusion
    unchanged), EXPECTED_TESTS = the parent's (the pinned alpha-next build's expected_selfrun_tests, asserted equal to the literal in
    alpha-next's probe) + the suite's. The suite's count is read by loading it in a fresh process (none of its ids may be a declared
    clock-bound id, and it must expose SLOT_RECEIPTS and ARGPARSE_RECEIPTS as alpha-next's queue suite does); the probe file
    carries the literal, so a changed count fails --check until the probe is regenerated and committed;
  * test_alpha_next_c1_queue's SLOT_RECEIPTS appended to the inherited ones (empty: the draft has no slot);
  * the LOCALIZE=0 inert check run on the alpha-next-c1 settings, held to the same expectations;
  * the alpha-next-c1 queue receipts asserted and recorded (alpha_next_c1_queue='PASS'), alpha-next's kept (alpha_next_queue='PASS').
No probe suite holds a load-flaky or superseded id of harness/alpha_next_robust_swap.py (--check asserts it). Every replacement must
match exactly once. --check verifies the committed file. Before the queue suite lands, nothing can be generated (the count is the
queue's): tools/alpha_next_c1_spec_regen.sh --pre-queue skips this tool.
"""
from pathlib import Path
import re
import subprocess
import sys

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import alpha_next_tools_derive as parent_derive  # noqa: E402  (its no_swapped_module and apply, unchanged)

QUEUE_SUITE = 'test_alpha_next_c1_queue'
CANDIDATE = 'q18a-alpha-next-c1'
PARENT_CANDIDATE = 'q18a-alpha-next'
PARENT_PROBE = 'alpha_next_cpu_probe.py'
EXPECTED_LINE = re.compile(r'^EXPECTED_TESTS=(\d+)  # [^\n]*\n', re.M)


def parent_tests():
    """The parent's bundle self-run count: the pinned alpha-next build's, equal to the literal its probe carries."""
    import alpha_next_c1_parent_pin as parent_pin
    pinned = parent_pin.load()['expected_selfrun_tests']
    found = EXPECTED_LINE.findall((TOOLS / PARENT_PROBE).read_text())
    assert found == [str(pinned)], ('alpha-next probe literal differs from the pinned build', found, pinned)
    return pinned


PARENT_TESTS = None   # read through parent_tests() (the pinned build); a module constant would hide the pin


def queue_tests():
    """test_alpha_next_c1_queue's test count, loaded by name in a fresh process; none of its ids a declared clock-bound id."""
    assert (TOOLS.parent / 'harness' / (QUEUE_SUITE + '.py')).is_file(), 'the queue suite has not landed: ' + QUEUE_SUITE
    code = ('import sys,unittest,importlib; sys.path.insert(0,"harness"); import step_b_clock_bound as c; '
            's=unittest.defaultTestLoader.loadTestsFromName(%r); assert not c.expected_removed([%r]); '
            'm=importlib.import_module(%r); assert hasattr(m,"SLOT_RECEIPTS") and hasattr(m,"ARGPARSE_RECEIPTS"), "receipt lists"; '
            'print(s.countTestCases())' % (QUEUE_SUITE, QUEUE_SUITE, QUEUE_SUITE))
    count = int(subprocess.check_output([sys.executable, '-B', '-c', code], cwd=TOOLS.parent, text=True).strip())
    assert count > 0, count
    return count


def expected():
    return parent_tests() + queue_tests()


def probe_swaps():
    parent, ours = parent_tests(), queue_tests()
    text = (TOOLS / PARENT_PROBE).read_text()
    line = EXPECTED_LINE.search(text).group(0)
    discover = "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_alpha_next_queue.py'))\n"
    return (
        (discover + line,
         discover + "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_alpha_next_c1_queue.py'))\n"
         'EXPECTED_TESTS=%d  # the %d alpha-next probe cases plus the %d alpha-next-c1 queue cases; pinned into the frozen runner '
         'gate\n' % (parent + ours, parent, ours)),
        ("owner.QNAME==%r\n" % PARENT_CANDIDATE, "owner.QNAME==%r\n" % CANDIDATE),
        ('import alpha_next_gate\nactive_and_cleanup=alpha_next_gate.ACTIVE_S+alpha_next_gate.CLEANUP_S\n',
         'import alpha_next_gate\nimport alpha_next_c1_gate\n'
         'active_and_cleanup=alpha_next_c1_gate.ACTIVE_S+alpha_next_c1_gate.CLEANUP_S\n'),
        ("+sys.modules['test_alpha_next_queue'].SLOT_RECEIPTS\n",
         "+sys.modules['test_alpha_next_queue'].SLOT_RECEIPTS+sys.modules['test_alpha_next_c1_queue'].SLOT_RECEIPTS\n"),
        ("negative=inert.run('/source/harness/runs/q18a_alpha_next_settings.sh','TRUNK settings (LOCALIZE=0)')",
         "negative=inert.run('/source/harness/runs/q18a_alpha_next_c1_settings.sh','TRUNK settings (LOCALIZE=0)')"),
        ("record=dict(alpha_next_queue='PASS',",
         "c1_slots=sys.modules['test_alpha_next_c1_queue'].SLOT_RECEIPTS\n"
         "assert c1_slots==[]  # the alpha-next-c1 draft has no slot (no absolute-time window), so its queue records no slot receipt\n"
         "c1_calls=sys.modules['test_alpha_next_c1_queue'].ARGPARSE_RECEIPTS\n"
         "assert c1_calls and all(r['verdict']=='parsed' for r in c1_calls)\n"
         "Path('/out/alpha-next-c1-queue-check.json').write_text(json.dumps(dict(slots=c1_slots,argparse=c1_calls),sort_keys=True,indent=2)+'\\n')\n"
         "record=dict(alpha_next_c1_queue='PASS',alpha_next_c1_queue_slots=c1_slots,alpha_next_c1_argparse_calls=len(c1_calls),"
         "alpha_next_queue='PASS',"),
    )


OUTPUTS = {'alpha_next_c1_cpu_probe.py': (PARENT_PROBE, probe_swaps)}


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
    print('alpha-next-c1 bundle tools ' + ('verified' if check else 'written'))
