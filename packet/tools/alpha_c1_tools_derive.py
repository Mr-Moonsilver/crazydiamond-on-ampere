#!/usr/bin/env python3
"""Derive the C1 bundle CPU probe from C0 l's (alpha_c0_cpu_probe.py, frozen in C0's bundle and unchanged here).

alpha_c1_cpu_probe.py = alpha_c0_cpu_probe.py with:
  * candidate q18a-alpha-c1 (the frozen C1 owner runner's QNAME); FLASH_NATIVE35_ARM stays TRUNK;
  * the alpha_c1_gate budget instead of alpha_c0_gate's (C1 has no absolute-time window either);
  * test_alpha_c1_queue added to the discovered suites (every inherited suite stays), EXPECTED_TESTS C0's 342 + the suite's;
    harness/clock_bound.py lists nothing in it, so the declared exclusion count stays C0's 9;
  * test_alpha_c1_queue's SLOT_RECEIPTS appended to the inherited ones (empty: C1's draft has no slot);
  * the LOCALIZE=0 inert check run on the C1 settings, held to the same expectations;
  * the C1 queue receipts asserted and recorded (alpha_c1_queue='PASS'), beside C0's;
  * C0's check_slot execution (two fixed instants, both accepted) kept: C1's runner inherits C0's no-op check_slot.
The bundle self-run is tools/native43_bundle_selfrun.py with the spec's arm and probe.

alpha_c1_vm_input.py = alpha_c0_vm_input.py (C0's one-card VM tool) with:
  * names alpha-c1-* / ALPHA_C1_* / alpha_c1_* for alpha-c0-* / ALPHA_C0_* / alpha_c0_* (counts asserted);
  * both C1 switches written into the GPU composition: FLASH_S5_STAGE_LOCAL and FLASH_S5_STAGE_OVERLAP 0 on the plain run, 1 on
    the --stage run (the CPU-shape bootstrap still sees neither, as C0's auto-arm switch);
  * the returned record carries alpha_c1_selection: the overlay's own stage_progress.stage_local_enabled and
    stage_overlap.stage_overlap_enabled over the composed environment, and the OverlapSegmentDriver class imported on the image;
  * SCOPE names what the one card cannot reach for C1: the overlap is engine-side (admission and two chunks in flight).
Every replacement must match exactly once.  --check verifies the committed file.
"""
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
PARENT_TESTS = 342   # C0 l's EXPECTED_TESTS
ALPHA_C1_TESTS = 14  # test_alpha_c1_queue
CLOCK_BOUND_HERE = 9  # unchanged: harness/clock_bound.py lists none of C1's tests
CANDIDATE = 'q18a-alpha-c1'
PARENT_CANDIDATE = 'q18a-alpha-c0'

PROBE_SWAPS = (
    ("suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_alpha_c0_queue.py'))\n",
     "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_alpha_c0_queue.py'))\n"
     "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_alpha_c1_queue.py'))\n"),
    ('EXPECTED_TESTS=%d  # the 287 alpha-perf probe cases plus the 64 C0 queue cases, less 9 declared real-clock window-slot tests; '
     'pinned into the frozen runner gate\n' % PARENT_TESTS,
     'EXPECTED_TESTS=%d  # C0\'s %d (the 287 alpha-perf probe cases plus the 64 C0 queue cases, less 9 declared real-clock '
     'window-slot tests) plus the %d C1 queue cases; pinned into the frozen runner gate\n'
     % (PARENT_TESTS + ALPHA_C1_TESTS, PARENT_TESTS, ALPHA_C1_TESTS)),
    ("owner.QNAME==%r\n" % PARENT_CANDIDATE, "owner.QNAME==%r\n" % CANDIDATE),
    ('import alpha_c0_gate\nactive_and_cleanup=alpha_c0_gate.ACTIVE_S+alpha_c0_gate.CLEANUP_S\n',
     'import alpha_c0_gate\nimport alpha_c1_gate\nactive_and_cleanup=alpha_c1_gate.ACTIVE_S+alpha_c1_gate.CLEANUP_S\n'),
    ("+sys.modules['test_alpha_c0_queue'].SLOT_RECEIPTS\n",
     "+sys.modules['test_alpha_c0_queue'].SLOT_RECEIPTS+sys.modules['test_alpha_c1_queue'].SLOT_RECEIPTS\n"),
    ("negative=inert.run('/source/harness/runs/q18a_alpha_c0_settings.sh','TRUNK settings (LOCALIZE=0)')",
     "negative=inert.run('/source/harness/runs/q18a_alpha_c1_settings.sh','TRUNK settings (LOCALIZE=0)')"),
    ("record=dict(clock_bound_excluded=sorted(clock_excluded),clock_bound_reason=clock_bound.REASON,alpha_c0_queue='PASS',",
     "alpha_c1_slots=sys.modules['test_alpha_c1_queue'].SLOT_RECEIPTS\n"
     "assert alpha_c1_slots==[]  # C1's draft has no slot (no absolute-time window), so its queue records no slot receipt\n"
     "alpha_c1_calls=sys.modules['test_alpha_c1_queue'].ARGPARSE_RECEIPTS\n"
     "assert alpha_c1_calls and all(r['verdict']=='parsed' for r in alpha_c1_calls)\n"
     "Path('/out/alpha-c1-queue-check.json').write_text(json.dumps(dict(slots=alpha_c1_slots,argparse=alpha_c1_calls),sort_keys=True,indent=2)+'\\n')\n"
     "record=dict(clock_bound_excluded=sorted(clock_excluded),clock_bound_reason=clock_bound.REASON,alpha_c1_queue='PASS',"
     "alpha_c1_queue_slots=alpha_c1_slots,alpha_c1_argparse_calls=len(alpha_c1_calls),alpha_c0_queue='PASS',"),
)

VM_SWAPS = (
    ('"""C0 one-card VM input: the B5 scenario on an overlay, on the plain path or through the stage wire.\n\n'
     "Derived from alpha-perf's tool (tools/alpha_perf_vm_input.py on claude/alpha-perf).\n",
     '"""C1 one-card VM input: the B5 scenario on an overlay, plain (both C1 switches off) or through the stage wire (both on).\n\n'
     "Generated by tools/alpha_c1_tools_derive.py from C0's one-card VM tool, itself derived from alpha-perf's tool.\n"
     "C1's overlap (two chunks in flight) is engine-side; on one card the stage run records that the overlay selects it.\n"),
    ('COMPOSE_ENV = "    os.environ.update(native_env())\\n    os.environ[\'FLASH_S5_AUTO_ARM\'] = \'1\'\\n    mark(\'compose-env-installed\'"\n',
     'COMPOSE_ENV = ("    os.environ.update(native_env())\\n    os.environ[\'FLASH_S5_AUTO_ARM\'] = \'1\'\\n"\n'
     '               "    os.environ.update(ALPHA_C1_SWITCHES[ALPHA_C1_STAGE])\\n    mark(\'compose-env-installed\'")\n'),
    ("    not_exercised=['StagedSegmentDriver (no engine driver)",
     "    not_exercised=['OverlapSegmentDriver and StagedAdmissionController (C1, engine-side): provisional admission of chunk n+1, "
     "two chunks in flight, expected-lane binding, overlap refusals; only the switch selection and the class import are recorded',\n"
     "                   'StagedSegmentDriver (no engine driver)"),
    ("ALPHA_C0_CALLS = dict(skips=0, scoped=0, local_releases=0)\n",
     "ALPHA_C0_CALLS = dict(skips=0, scoped=0, local_releases=0)\n"
     "# C1: both switches, off on the plain run and on on the stage run, set in the GPU composition only.\n"
     "ALPHA_C0_SWITCHES = {False: {'FLASH_S5_STAGE_LOCAL': '0', 'FLASH_S5_STAGE_OVERLAP': '0'},\n"
     "                     True: {'FLASH_S5_STAGE_LOCAL': '1', 'FLASH_S5_STAGE_OVERLAP': '1'}}\n\n\n"
     "def alpha_selection():\n"
     "    # The overlay's own selection functions over the composed environment, on the real image.\n"
     "    from vllm.v1.flash_e3.stage_overlap import OverlapSegmentDriver, stage_overlap_enabled\n"
     "    from vllm.v1.flash_e3.stage_progress import stage_local_enabled\n"
     "    return dict(local=stage_local_enabled(os.environ), overlap=stage_overlap_enabled(os.environ),\n"
     "                switches={k: os.environ.get(k) for k in sorted(ALPHA_C0_SWITCHES[True])},\n"
     "                driver_class=OverlapSegmentDriver.__name__)\n"),
    ('                        "        return dict(alpha_c0_stage=ALPHA_C0_STAGE, alpha_c0_stage_calls=dict(ALPHA_C0_CALLS),\\n"\n',
     '                        "        return dict(alpha_c0_stage=ALPHA_C0_STAGE, alpha_c0_stage_calls=dict(ALPHA_C0_CALLS),\\n"\n'
     '                        "                    alpha_c0_selection=alpha_selection(),\\n"\n'),
    ('    # C0: the plain or stage-scoped path,', '    # C0\'s plain or stage-scoped path,'),
)
# Whole-text renames after the swaps, each with its count (C0's file plus what the swaps add).
VM_RENAMES = (('ALPHA_C0', 'ALPHA_C1', 10), ('alpha_c0', 'alpha_c1', 4), ('alpha-c0', 'alpha-c1', 4))


def vm_input(text):
    text = apply(text, VM_SWAPS)
    for old, new, count in VM_RENAMES:
        assert text.count(old) == count, (old, text.count(old))
        text = text.replace(old, new)
    return text


OUTPUTS = {'alpha_c1_cpu_probe.py': ('alpha_c0_cpu_probe.py', PROBE_SWAPS),
           'alpha_c1_vm_input.py': ('alpha_c0_vm_input.py', vm_input)}


def apply(text, swaps):
    for old, new in swaps:
        assert text.count(old) == 1, (old[:80], text.count(old))
        text = text.replace(old, new)
    return text


def generate(tools=TOOLS):
    return {name: swaps((tools / source).read_text()) if callable(swaps) else apply((tools / source).read_text(), swaps)
            for name, (source, swaps) in OUTPUTS.items()}


if __name__ == '__main__':
    check = '--check' in sys.argv
    for name, text in generate().items():
        path = TOOLS / name
        if check:
            assert path.read_text() == text, name
        else:
            path.write_text(text)
            path.chmod(0o775)
    print('C1 bundle tools ' + ('verified' if check else 'written'))
