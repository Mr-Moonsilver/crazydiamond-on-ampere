#!/usr/bin/env python3
"""alpha-next-c1 client-clock gate: tools/alpha_next_clock_scan.py's configuration (Step A's scan and controls, unchanged) over the
alpha-next-c1 queue.

  * shell texts: q18a_alpha_next_c1{.draft,_declare,_settings}.sh (plus qlib.sh), the queue helper's;
  * the arm root is e3_alpha_next_c1_box (the queue helper's), which imports e3_alpha_next_box; ARM_MODULES is that root followed by
    alpha-next's ARM_MODULES, read from tools/alpha_next_clock_scan.py's text, so a client clock in any inherited arm still refuses.
    Each listed module must be in the closure (held-f's rule): if e3_alpha_next_c1_box stops importing one of them, the scan fails
    rather than silently scanning less. e3_alpha_c1_box (C1 l's arm) is not imported by this arm, so it is not listed.
The record is step_a_clock_scan's with its own schema name and the C1-style verdict re-check (arm modules all in the closure, every
shell text scanned). No GPU, no box action.
"""
import argparse
import ast
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step_a_clock_scan as step_a  # noqa: E402



def _alpha_next_arm_modules():
    for node in ast.parse((Path(__file__).resolve().with_name('alpha_next_clock_scan.py')).read_text()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', None) == 'ARM_MODULES':
            return ast.literal_eval(node.value)
    raise AssertionError('alpha_next_clock_scan.ARM_MODULES not found')


TEXTS = ('q18a_alpha_next_c1.draft.sh', 'q18a_alpha_next_c1_declare.sh', 'q18a_alpha_next_c1_settings.sh')
ARM_MODULES = ('e3_alpha_next_c1_box',) + _alpha_next_arm_modules()
step_a.TEXTS = TEXTS
step_a.base.DRAFT = TEXTS[0]
step_a.base.SHELL = TEXTS
step_a.held_f.ARM = ARM_MODULES[0]
step_a.held_f.ARM_MODULES = ARM_MODULES


def main(a):
    rc = step_a.main(a)
    record = json.loads(a.out.read_text())
    ok = rc == 0 and record['verdict'] == 'PASS' and record['arm_modules'] == list(ARM_MODULES) \
        and all(m in record['scan']['closure'] for m in ARM_MODULES) and sorted(record['shell_shas']) >= sorted(TEXTS)
    record.update(schema='alpha-next-c1-clock-scan/1',
                  derived_from='tools/alpha_next_clock_scan.py over tools/step_a_clock_scan.py (held-f\'s scan)',
                  verdict='PASS' if ok else 'FAIL', classification=None if ok else 'HARNESS')
    a.out.write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(verdict=record['verdict'], arm_modules=record['arm_modules'])))
    return 0 if ok else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--bundle', type=Path)
    g.add_argument('--tree', type=Path)
    p.add_argument('--runner', type=Path)
    p.add_argument('--setcfg-reference', type=Path)
    p.add_argument('--out', type=Path, required=True)
    raise SystemExit(main(p.parse_args()))
