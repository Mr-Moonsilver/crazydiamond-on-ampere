#!/usr/bin/env python3
"""alpha-next-c1b client-clock gate: tools/alpha_next_c1_clock_scan.py's configuration (Step A's scan and controls, unchanged) over the
alpha-next-c1b queue.

  * shell texts: q18a_alpha_next_c1b{.draft,_declare,_settings}.sh (plus qlib.sh), the coordinator's queue texts;
  * the arm root is e3_alpha_next_c1b_box, followed by c1's ARM_MODULES (alpha_next_c1_clock_scan.ARM_MODULES, imported), so a client
    clock in any inherited arm still refuses. Each listed module must be in the closure (held-f's rule): if e3_alpha_next_c1b_box
    does not import e3_alpha_next_c1_box (and through it the rest), the scan FAILS rather than silently scanning less.
The record is step_a_clock_scan's with its own schema name and c1's verdict re-check. No GPU, no box action.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alpha_next_c1_clock_scan as c1  # noqa: E402  (configures step_a for c1; re-configured below)

step_a = c1.step_a
TEXTS = ('q18a_alpha_next_c1b.draft.sh', 'q18a_alpha_next_c1b_declare.sh', 'q18a_alpha_next_c1b_settings.sh')
ARM_MODULES = ('e3_alpha_next_c1b_box',) + tuple(c1.ARM_MODULES)
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
    record.update(schema='alpha-next-c1b-clock-scan/1',
                  derived_from='tools/alpha_next_c1_clock_scan.py over tools/step_a_clock_scan.py (held-f\'s scan)',
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
