#!/usr/bin/env python3
"""Derive the alpha-next bundle CPU probe from the Step B one (step_b_cpu_probe.py).

alpha_next_cpu_probe.py = step_b_cpu_probe.py with:
  * candidate q18a-alpha-next (the frozen alpha-next owner runner's QNAME); FLASH_NATIVE35_ARM stays TRUNK;
  * the alpha_next_gate budget instead of step_b_gate's (no absolute-time window: the relative budget only);
  * test_alpha_next_queue added to the discovered suites (every inherited suite stays, Step B's declared real-clock exclusion
    unchanged), EXPECTED_TESTS Step B's 374 + the suite's. The suite's count is read by loading it in a fresh process (none of its
    ids may be a declared clock-bound id); the probe file carries the literal, so a changed count fails --check (the build's
    checks step) until the probe is regenerated and committed;
  * test_alpha_next_queue's SLOT_RECEIPTS appended to the inherited ones (empty: alpha-next's draft has no slot);
  * the LOCALIZE=0 inert check run on the alpha-next settings, held to the same expectations;
  * the alpha-next queue receipts asserted and recorded (alpha_next_queue='PASS'), Step B's kept (step_b_queue='PASS').
No probe suite holds a load-flaky or superseded id of harness/alpha_next_robust_swap.py, so the probe does not filter through it
(--check asserts that against the swap's declared modules). The bundle self-run is tools/native43_bundle_selfrun.py with the
spec's arm and probe. Every replacement must match exactly once. --check verifies the committed file.
"""
from pathlib import Path
import re
import subprocess
import sys

TOOLS = Path(__file__).resolve().parent
PARENT_TESTS = 374  # Step B's spec expected_tests (step_b_cpu_probe EXPECTED_TESTS)
QUEUE_SUITE = 'test_alpha_next_queue'
CANDIDATE = 'q18a-alpha-next'
PARENT_CANDIDATE = 'q18a-step-b'


def queue_tests():
    """test_alpha_next_queue's test count, loaded by name in a fresh process; none of its ids a declared clock-bound id."""
    assert (TOOLS.parent / 'harness' / (QUEUE_SUITE + '.py')).is_file(), 'the queue suite has not landed: ' + QUEUE_SUITE
    code = ('import sys,unittest; sys.path.insert(0,"harness"); import step_b_clock_bound as c; '
            's=unittest.defaultTestLoader.loadTestsFromName(%r); assert not c.expected_removed([%r]); '
            'print(s.countTestCases())' % (QUEUE_SUITE, QUEUE_SUITE))
    count = int(subprocess.check_output([sys.executable, '-B', '-c', code], cwd=TOOLS.parent, text=True).strip())
    assert count > 0, count
    return count


def expected():
    return PARENT_TESTS + queue_tests()


def probe_swaps():
    alpha_next_tests = queue_tests()
    return (
        ("suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_step_b_queue.py'))\n"
         'EXPECTED_TESTS=%d  # the 333 Step A probe cases plus the 51 Step B queue cases, less 10 declared real-clock window-slot '
         'tests; pinned into the frozen runner gate\n' % PARENT_TESTS,
         "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_step_b_queue.py'))\n"
         "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_alpha_next_queue.py'))\n"
         'EXPECTED_TESTS=%d  # the %d Step B probe cases plus the %d alpha-next queue cases; pinned into the frozen runner gate\n'
         % (PARENT_TESTS + alpha_next_tests, PARENT_TESTS, alpha_next_tests)),
        ("owner.QNAME==%r\n" % PARENT_CANDIDATE, "owner.QNAME==%r\n" % CANDIDATE),
        ('import step_b_gate\nactive_and_cleanup=step_b_gate.ACTIVE_S+step_b_gate.CLEANUP_S\n',
         'import step_b_gate\nimport alpha_next_gate\nactive_and_cleanup=alpha_next_gate.ACTIVE_S+alpha_next_gate.CLEANUP_S\n'),
        ("+sys.modules['test_step_b_queue'].SLOT_RECEIPTS\n",
         "+sys.modules['test_step_b_queue'].SLOT_RECEIPTS+sys.modules['test_alpha_next_queue'].SLOT_RECEIPTS\n"),
        ("negative=inert.run('/source/harness/runs/q18a_step_b_settings.sh','TRUNK settings (LOCALIZE=0)')",
         "negative=inert.run('/source/harness/runs/q18a_alpha_next_settings.sh','TRUNK settings (LOCALIZE=0)')"),
        ("record=dict(step_b_queue='PASS',",
         "alpha_next_slots=sys.modules['test_alpha_next_queue'].SLOT_RECEIPTS\n"
         "assert alpha_next_slots==[]  # alpha-next's draft has no slot (no absolute-time window), so its queue records no slot receipt\n"
         "alpha_next_calls=sys.modules['test_alpha_next_queue'].ARGPARSE_RECEIPTS\n"
         "assert alpha_next_calls and all(r['verdict']=='parsed' for r in alpha_next_calls)\n"
         "Path('/out/alpha-next-queue-check.json').write_text(json.dumps(dict(slots=alpha_next_slots,argparse=alpha_next_calls),sort_keys=True,indent=2)+'\\n')\n"
         "record=dict(alpha_next_queue='PASS',alpha_next_queue_slots=alpha_next_slots,alpha_next_argparse_calls=len(alpha_next_calls),"
         "step_b_queue='PASS',"),
    )


OUTPUTS = {'alpha_next_cpu_probe.py': ('step_b_cpu_probe.py', probe_swaps)}


def apply(text, swaps):
    for old, new in swaps:
        assert text.count(old) == 1, (old[:80], text.count(old))
        text = text.replace(old, new)
    return text


def probe_modules(text):
    """The test module patterns the probe discovers."""
    return re.findall(r"pattern='([^']+)'", text)


def no_swapped_module(text):
    """No discovered pattern matches a module named by harness/alpha_next_robust_swap.py (swapped or superseded)."""
    import fnmatch
    sys.path.insert(0, str(TOOLS.parent / 'harness'))
    import alpha_next_robust_swap as swap
    hits = sorted({m for m in swap.MODULES for p in probe_modules(text) if fnmatch.fnmatch(m + '.py', p)})
    assert not hits, ('probe discovers a robust-swap module; filter through alpha_next_robust_swap', hits)


def generate(tools=TOOLS):
    out = {}
    for name, (source, swaps) in OUTPUTS.items():
        text = apply((tools / source).read_text(), swaps())
        no_swapped_module(text)
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
    print('alpha-next bundle tools ' + ('verified' if check else 'written'))
