#!/usr/bin/env python3
"""Derive the alpha-next-c1b bundle CPU probe from alpha-next-c1's (alpha_next_c1_cpu_probe.py), as c1 derived its own from alpha-next's.

alpha_next_c1b_cpu_probe.py = alpha_next_c1_cpu_probe.py with:
  * candidate q18a-alpha-next-c1b (the frozen c1b owner runner's QNAME); FLASH_NATIVE35_ARM stays TRUNK;
  * the alpha_next_c1b_gate budget beside c1's (alpha_next_c1b_gate.ACTIVE_S + CLEANUP_S; no absolute-time window);
  * test_alpha_next_c1b_queue added to the discovered suites (every inherited suite stays), EXPECTED_TESTS = the parent's (the
    pinned alpha-next-c1 build's expected_selfrun_tests, asserted equal to the literal in c1's probe) + the suite's. The suite's
    count is read by loading it in a fresh process (none of its ids a declared clock-bound id; it must expose SLOT_RECEIPTS and
    ARGPARSE_RECEIPTS); the probe file carries the literal, so a changed count fails --check until it is regenerated;
  * test_alpha_next_c1b_queue's SLOT_RECEIPTS appended to the inherited ones (asserted empty: the c1b draft has no slot);
  * the LOCALIZE=0 inert check run on the alpha-next-c1b settings, held to the same expectations;
  * the c1b queue receipts asserted and recorded (alpha_next_c1b_queue='PASS', its argparse receipts all 'parsed', its check file
    /out/alpha-next-c1b-queue-check.json), c1's and alpha-next's kept.
Every replacement must match exactly once (alpha_next_tools_derive.apply, unchanged); no probe suite may be a robust-swap module.
--check verifies the committed file. Before the c1b queue suite lands nothing can be generated (the count is the queue's).
"""
from pathlib import Path
import re
import subprocess
import sys

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import alpha_next_tools_derive as parent_derive  # noqa: E402  (its no_swapped_module and apply, unchanged)

QUEUE_SUITE = 'test_alpha_next_c1b_queue'
CANDIDATE = 'q18a-alpha-next-c1b'
PARENT_CANDIDATE = 'q18a-alpha-next-c1'
PARENT_PROBE = 'alpha_next_c1_cpu_probe.py'
EXPECTED_LINE = re.compile(r'^EXPECTED_TESTS=(\d+)  # [^\n]*\n', re.M)


def parent_tests():
    """The parent's bundle self-run count: the pinned alpha-next-c1 build's, equal to the literal its probe carries."""
    import alpha_next_c1b_parent_pin as parent_pin
    pinned = parent_pin.load()['expected_selfrun_tests']
    found = EXPECTED_LINE.findall((TOOLS / PARENT_PROBE).read_text())
    assert found == [str(pinned)], ('alpha-next-c1 probe literal differs from the pinned build', found, pinned)
    return pinned


def queue_tests(root=TOOLS.parent):
    """test_alpha_next_c1b_queue's test count, loaded by name in a fresh process; none of its ids a declared clock-bound id."""
    assert (Path(root) / 'harness' / (QUEUE_SUITE + '.py')).is_file(), 'the c1b queue suite has not landed: ' + QUEUE_SUITE
    code = ('import sys,unittest,importlib; sys.path.insert(0,"harness"); import step_b_clock_bound as c; '
            's=unittest.defaultTestLoader.loadTestsFromName(%r); assert not c.expected_removed([%r]); '
            'm=importlib.import_module(%r); assert hasattr(m,"SLOT_RECEIPTS") and hasattr(m,"ARGPARSE_RECEIPTS"), "receipt lists"; '
            'print(s.countTestCases())' % (QUEUE_SUITE, QUEUE_SUITE, QUEUE_SUITE))
    count = int(subprocess.check_output([sys.executable, '-B', '-c', code], cwd=root, text=True).strip())
    assert count > 0, count
    return count


def expected():
    return parent_tests() + queue_tests()


def probe_swaps(parent=None, ours=None):
    parent = parent_tests() if parent is None else parent
    ours = queue_tests() if ours is None else ours
    text = (TOOLS / PARENT_PROBE).read_text()
    line = EXPECTED_LINE.search(text).group(0)
    discover = "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_alpha_next_c1_queue.py'))\n"
    return (
        (discover + line,
         discover + "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_alpha_next_c1b_queue.py'))\n"
         'EXPECTED_TESTS=%d  # the %d alpha-next-c1 probe cases plus the %d alpha-next-c1b queue cases; pinned into the frozen '
         'runner gate\n' % (parent + ours, parent, ours)),
        ("owner.QNAME==%r\n" % PARENT_CANDIDATE, "owner.QNAME==%r\n" % CANDIDATE),
        ('import alpha_next_c1_gate\nactive_and_cleanup=alpha_next_c1_gate.ACTIVE_S+alpha_next_c1_gate.CLEANUP_S\n',
         'import alpha_next_c1_gate\nimport alpha_next_c1b_gate\n'
         'active_and_cleanup=alpha_next_c1b_gate.ACTIVE_S+alpha_next_c1b_gate.CLEANUP_S\n'),
        ("+sys.modules['test_alpha_next_c1_queue'].SLOT_RECEIPTS\n",
         "+sys.modules['test_alpha_next_c1_queue'].SLOT_RECEIPTS+sys.modules['test_alpha_next_c1b_queue'].SLOT_RECEIPTS\n"),
        ("negative=inert.run('/source/harness/runs/q18a_alpha_next_c1_settings.sh','TRUNK settings (LOCALIZE=0)')",
         "negative=inert.run('/source/harness/runs/q18a_alpha_next_c1b_settings.sh','TRUNK settings (LOCALIZE=0)')"),
        ("record=dict(alpha_next_c1_queue='PASS',",
         "c1b_slots=sys.modules['test_alpha_next_c1b_queue'].SLOT_RECEIPTS\n"
         "assert c1b_slots==[]  # the alpha-next-c1b draft has no slot (no absolute-time window), so its queue records no slot receipt\n"
         "c1b_calls=sys.modules['test_alpha_next_c1b_queue'].ARGPARSE_RECEIPTS\n"
         "assert c1b_calls and all(r['verdict']=='parsed' for r in c1b_calls)\n"
         "Path('/out/alpha-next-c1b-queue-check.json').write_text(json.dumps(dict(slots=c1b_slots,argparse=c1b_calls),sort_keys=True,indent=2)+'\\n')\n"
         "record=dict(alpha_next_c1b_queue='PASS',alpha_next_c1b_queue_slots=c1b_slots,alpha_next_c1b_argparse_calls=len(c1b_calls),"
         "alpha_next_c1_queue='PASS',"),
    )


OUTPUTS = {'alpha_next_c1b_cpu_probe.py': (PARENT_PROBE, probe_swaps)}


def generate(tools=TOOLS, **counts):
    out = {}
    for name, (source, swaps) in OUTPUTS.items():
        text = parent_derive.apply((tools / source).read_text(), swaps(**counts))
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
    print('alpha-next-c1b bundle tools ' + ('verified' if check else 'written'))
