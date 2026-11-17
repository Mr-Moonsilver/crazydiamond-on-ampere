#!/usr/bin/env python3
"""alpha-next client-clock gate: tools/step_b_clock_scan.py's configuration of Step A's scan and controls, unchanged, over the
alpha-next queue.

  * shell texts: q18a_alpha_next{.draft,_declare,_settings}.sh (plus qlib.sh);
  * the arm root is e3_alpha_next_box. ARM_MODULES names every arm module the scan must find in its closure: alpha-next's root,
    the Step B arm it imports, and Step B's inherited chain (Step A, alpha-perf, held-f, held-e), so a client clock in any inherited
    arm still refuses. e3_alpha_c0_box is not imported by the alpha-next arm (it restates C0's declarations; importing C0's arm
    would wrap progress.chunk_breakdown for the whole process), so it is not listed. Each listed module must be in the closure (held-f's rule), so this list is the arm's import
    declaration: if e3_alpha_next_box stops importing one of them, the scan fails rather than silently scanning less.
The record is step_a_clock_scan's with its own schema name. No GPU, no box action.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step_a_clock_scan as step_a  # noqa: E402

TEXTS = ('q18a_alpha_next.draft.sh', 'q18a_alpha_next_declare.sh', 'q18a_alpha_next_settings.sh')
ARM_MODULES = ('e3_alpha_next_box', 'e3_step_b_box', 'e3_step_a_box', 'e3_alpha_perf_box',
               'e3_native41_progress_box', 'e3_native41_autoarm_box')
step_a.TEXTS = TEXTS
step_a.base.DRAFT = TEXTS[0]
step_a.base.SHELL = TEXTS
step_a.held_f.ARM = ARM_MODULES[0]
step_a.held_f.ARM_MODULES = ARM_MODULES


def main(a):
    rc = step_a.main(a)
    record = json.loads(a.out.read_text())
    record.update(schema='alpha-next-clock-scan/1',
                  derived_from='tools/step_b_clock_scan.py over tools/step_a_clock_scan.py (held-f\'s scan)')
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
