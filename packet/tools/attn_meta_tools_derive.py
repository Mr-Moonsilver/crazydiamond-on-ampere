#!/usr/bin/env python3
"""Derive the q18a-attn-meta bundle CPU probe from c1's (alpha_next_c1_cpu_probe.py), as c1 derived its own from alpha-next's.

attn_meta_cpu_probe.py = alpha_next_c1_cpu_probe.py with:
  * candidate q18a-attn-meta (the frozen attn-meta owner runner's QNAME); FLASH_NATIVE35_ARM stays TRUNK;
  * the attn_meta_gate budget beside c1's (attn_meta_gate.ACTIVE_S + CLEANUP_S; no absolute-time window);
  * test_attn_meta_queue added to the discovered suites (every inherited suite stays), EXPECTED_TESTS = the parent's (the pinned c1
    build's expected_selfrun_tests, asserted equal to the literal in c1's probe) + the suite's, read by loading it in a fresh
    process (none of its ids a declared clock-bound id; it must expose SLOT_RECEIPTS and ARGPARSE_RECEIPTS); the probe file
    carries the literal, so a changed count fails --check until the probe is regenerated and committed;
  * test_attn_meta_queue's SLOT_RECEIPTS appended (empty: the draft has no slot);
  * the LOCALIZE=0 inert check run on the attn-meta settings, held to the same expectations;
  * the attn-meta queue receipts asserted and recorded (attn_meta_queue='PASS'), c1's kept (alpha_next_c1_queue='PASS').
Every replacement must match exactly once (alpha_next_tools_derive.apply). --check verifies the committed file.
"""
from pathlib import Path
import re
import subprocess
import sys

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import alpha_next_tools_derive as parent_derive  # noqa: E402  (its no_swapped_module and apply, unchanged)

QUEUE_SUITE = 'test_attn_meta_queue'
CANDIDATE = 'q18a-attn-meta'
PARENT_CANDIDATE = 'q18a-alpha-next-c1'
PARENT_PROBE = 'alpha_next_c1_cpu_probe.py'
EXPECTED_LINE = re.compile(r'^EXPECTED_TESTS=(\d+)  # [^\n]*\n', re.M)


def parent_tests():
    """The parent's bundle self-run count: the pinned c1 build's, equal to the literal its probe carries."""
    import attn_meta_parent_pin as parent_pin
    pinned = parent_pin.load()['expected_selfrun_tests']
    found = EXPECTED_LINE.findall((TOOLS / PARENT_PROBE).read_text())
    assert found == [str(pinned)], ('c1 probe literal differs from the pinned build', found, pinned)
    return pinned


def queue_tests():
    """test_attn_meta_queue's test count, loaded by name in a fresh process (cwd harness/, as selftest and the bundle run it)."""
    assert (TOOLS.parent / 'harness' / (QUEUE_SUITE + '.py')).is_file(), 'the queue suite has not landed: ' + QUEUE_SUITE
    code = ('import sys,unittest,importlib; sys.path.insert(0,"."); import step_b_clock_bound as c; '
            's=unittest.defaultTestLoader.loadTestsFromName(%r); assert not c.expected_removed([%r]); '
            'm=importlib.import_module(%r); assert hasattr(m,"SLOT_RECEIPTS") and hasattr(m,"ARGPARSE_RECEIPTS"), "receipt lists"; '
            'print(s.countTestCases())' % (QUEUE_SUITE, QUEUE_SUITE, QUEUE_SUITE))
    count = int(subprocess.check_output([sys.executable, '-B', '-c', code], cwd=TOOLS.parent / 'harness', text=True).strip())
    assert count > 0, count
    return count


def expected():
    return parent_tests() + queue_tests()


def probe_swaps():
    parent, ours = parent_tests(), queue_tests()
    text = (TOOLS / PARENT_PROBE).read_text()
    line = EXPECTED_LINE.search(text).group(0)
    discover = "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_alpha_next_c1_queue.py'))\n"
    return (
        (discover + line,
         discover + "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_attn_meta_queue.py'))\n"
         'EXPECTED_TESTS=%d  # the %d alpha-next-c1 probe cases plus the %d attn-meta queue cases; pinned into the frozen runner '
         'gate\n' % (parent + ours, parent, ours)),
        ("owner.QNAME==%r\n" % PARENT_CANDIDATE, "owner.QNAME==%r\n" % CANDIDATE),
        ('import alpha_next_c1_gate\nactive_and_cleanup=alpha_next_c1_gate.ACTIVE_S+alpha_next_c1_gate.CLEANUP_S\n',
         'import alpha_next_c1_gate\nimport attn_meta_gate\n'
         'active_and_cleanup=attn_meta_gate.ACTIVE_S+attn_meta_gate.CLEANUP_S\n'),
        ("+sys.modules['test_alpha_next_c1_queue'].SLOT_RECEIPTS\n",
         "+sys.modules['test_alpha_next_c1_queue'].SLOT_RECEIPTS+sys.modules['test_attn_meta_queue'].SLOT_RECEIPTS\n"),
        ("negative=inert.run('/source/harness/runs/q18a_alpha_next_c1_settings.sh','TRUNK settings (LOCALIZE=0)')",
         "negative=inert.run('/source/harness/runs/q18a_attn_meta_settings.sh','TRUNK settings (LOCALIZE=0)')"),
        ("record=dict(alpha_next_c1_queue='PASS',",
         "am_slots=sys.modules['test_attn_meta_queue'].SLOT_RECEIPTS\n"
         "assert am_slots==[]  # the attn-meta draft has no slot (no absolute-time window), so its queue records no slot receipt\n"
         "am_calls=sys.modules['test_attn_meta_queue'].ARGPARSE_RECEIPTS\n"
         "assert am_calls and all(r['verdict']=='parsed' for r in am_calls)\n"
         "Path('/out/attn-meta-queue-check.json').write_text(json.dumps(dict(slots=am_slots,argparse=am_calls),sort_keys=True,indent=2)+'\\n')\n"
         "record=dict(attn_meta_queue='PASS',attn_meta_queue_slots=am_slots,attn_meta_argparse_calls=len(am_calls),"
         "alpha_next_c1_queue='PASS',"),
    )


OUTPUTS = {'attn_meta_cpu_probe.py': (PARENT_PROBE, probe_swaps)}


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
    print('attn-meta bundle tools ' + ('verified' if check else 'written'))
