#!/usr/bin/env python3
"""Step B client-clock gate: tools/step_a_clock_scan.py's scan and controls, unchanged, over the Step B queue.

  * shell texts: q18a_step_b{.draft,_declare,_settings}.sh (plus qlib.sh);
  * the arm root is e3_step_b_box, which imports Step A's arm, then alpha-perf's and held-f's: the scan follows them all, so a client
    clock in any inherited arm still refuses.
The record is step_a_clock_scan's with its own schema name. No GPU, no box action.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step_a_clock_scan as step_a  # noqa: E402

TEXTS = ('q18a_step_b.draft.sh', 'q18a_step_b_declare.sh', 'q18a_step_b_settings.sh')
step_a.TEXTS = TEXTS
step_a.base.DRAFT = TEXTS[0]
step_a.base.SHELL = TEXTS
step_a.held_f.ARM = 'e3_step_b_box'
step_a.held_f.ARM_MODULES = ('e3_step_b_box', 'e3_step_a_box', 'e3_alpha_perf_box', 'e3_native41_progress_box', 'e3_native41_autoarm_box')


def main(a):
    rc = step_a.main(a)
    record = json.loads(a.out.read_text())
    record.update(schema='step-b-clock-scan/1', derived_from='tools/step_a_clock_scan.py (held-f\'s scan)')
    a.out.write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    return rc


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--bundle', type=Path)
    g.add_argument('--tree', type=Path)
    p.add_argument('--runner', type=Path)
    p.add_argument('--setcfg-reference', type=Path)
    p.add_argument('--out', type=Path, required=True)
    raise SystemExit(main(p.parse_args()))
