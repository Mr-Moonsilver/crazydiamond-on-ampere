#!/usr/bin/env python3
"""C1 client-clock gate: C0's scan (tools/alpha_c0_clock_scan.py, itself held-f's) over the C1 queue.

C0's module is imported unchanged and reconfigured, as C0 reconfigures held-f's: the scanner, rules, allowlists, the le-fix
death-scan resolution and every negative control (held-e's, held-f's and C0's two) are C0's. Declared configuration:
  * shell texts: q18a_alpha_c1{.draft,_declare,_settings}.sh (plus qlib.sh);
  * the arm root is e3_alpha_c1_box, which imports C0's arm, then alpha-perf's, held-f's and held-e's: the window/evidence rule is
    applied to all five, and the arm control (progress-arm-sends-overlap-evidence) sends overlap_evidence from this root.
The record's schema becomes alpha-c1-clock-scan/1; nothing else in it is rewritten. Static, no box action.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alpha_c0_clock_scan as c0  # noqa: E402

TEXTS = ('q18a_alpha_c1.draft.sh', 'q18a_alpha_c1_declare.sh', 'q18a_alpha_c1_settings.sh')
ARM_MODULES = ('e3_alpha_c1_box',) + c0.held_f.ARM_MODULES
c0.TEXTS = TEXTS
c0.base.DRAFT = TEXTS[0]
c0.base.SHELL = TEXTS
c0.held_f.ARM = ARM_MODULES[0]
c0.held_f.ARM_MODULES = ARM_MODULES


def main(a):
    rc = c0.main(a)
    record = json.loads(a.out.read_text())
    ok = rc == 0 and record['verdict'] == 'PASS' and record['arm_modules'] == list(ARM_MODULES) \
        and all(m in record['scan']['closure'] for m in ARM_MODULES) and sorted(record['shell_shas']) >= sorted(TEXTS)
    record.update(schema='alpha-c1-clock-scan/1', derived_from='tools/alpha_c0_clock_scan.py (C0, over held-f\'s)',
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
