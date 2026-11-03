#!/usr/bin/env python3
"""Owner precheck configuration for q18a-alpha-next-c1b against the exact alpha-next-c1 parent (tools/alpha_next_c1b_parent.json).

tools/alpha_next_c1_precheck_conf.generate, imported and exec'd from its own source with c1b's names re-bound (never copy-edited):
PARENT_BUILD is c1's build (through alpha_next_c1b_parent_pin.load(), rc 3 while not pinned), STEM q18a_alpha_next_c1b, ADDED empty
(c1b adds no overlay file; every overlay row that moved is excluded from the VM comparison together with c1's METADATA), SEAMS the
c1b queue's and the delta suites' seam tests followed by c1's. Swapped exactly once each: the candidate/parent names, the self-run
container prefix, the work root and the header line. No GPU, no box action.
usage: alpha_next_c1b_precheck_conf.py --build OUT/build.json --out CONF [--selfrun OUT/selfrun/result.json]
"""
import argparse
import inspect
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import alpha_next_c1_precheck_conf as c1  # noqa: E402
import alpha_next_c1b_parent_pin as parent_pin  # noqa: E402

PARENT_BUILD = Path(parent_pin.load()['build'])
STEM = 'q18a_alpha_next_c1b'
ADDED = set()
SEAMS = ('test_alpha_next_c1b_queue.py e3_alpha_next_c1b_box.py alpha_next_c1b_arms.py alpha_next_c1b_gate.py '
         'test_e3_c1_open_rows.py test_alpha_next_c1_send_gate.py test_alpha_next_c1_defer_control_b.py ' + c1.SEAMS)
assert SEAMS.endswith(' ') and len(SEAMS.split()) == len(set(SEAMS.split())), 'seam list'
SWAPS = (("b['name'] == 'q18a-alpha-next-c1' and prev['name'] == 'q18a-alpha-next'",
          "b['name'] == 'q18a-alpha-next-c1b' and prev['name'] == 'q18a-alpha-next-c1'"),
         ("'alpha-next-c1-hub-cpu-'", "'alpha-next-c1b-hub-cpu-'"),
         ("tmp/alpha-next-c1/pack/precheck-work", "tmp/alpha-next-c1b/pack/precheck-work"),
         ("'# alpha-next-c1 on the composed engine. PREV = exact alpha-next (%s); VM_OVERLAY is source comparison only.\\n%s'",
          "'# alpha-next-c1b (harness-only over alpha-next-c1). PREV = exact alpha-next-c1 (%s); VM_OVERLAY is source comparison "
          "only.\\n%s'"))
TEXT = inspect.getsource(c1.generate)
for _old, _new in SWAPS:
    assert TEXT.count(_old) == 1, ('c1 precheck conf changed', _old)
    TEXT = TEXT.replace(_old, _new)
NS = dict(c1.__dict__, PARENT_BUILD=PARENT_BUILD, STEM=STEM, ADDED=ADDED, SEAMS=SEAMS)
exec(compile(TEXT, 'alpha_next_c1b_precheck_conf:generate', 'exec'), NS)
generate = NS['generate']

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--selfrun', type=Path)
    a = p.parse_args()
    generate(a.build, a.out, a.selfrun)
