#!/usr/bin/env python3
"""alpha-next engine self-run, b: tools/alpha_next_engine_selfrun.py (which chains g1-g4 ran; superseded here, never edited) with
every probe case of every container launched through harness/alpha_next_probe_launch.py.

Why (g4 engine self-run, HARNESS): PLAN 2 sends every receipt through native_measurement's process ReceiptWriter, not the logger;
the frozen probes collect receipt lines with a logging.Handler, so 39 cases saw none ("progress-lines-bounded {longest: 0}") in
the progress (7), fused (6), groups (7), stage (6) and composed (13) containers. The launcher installs a REAL ReceiptWriter whose
write seam is an in-memory sink, reads every record back with receipt_lines.parse_line (the box grader's parser) and hands the
bare text to the probe's capture handlers; framing defects exit 3 (see the launcher's docstring and the leaf
probe-launch-receipt-sink in harness/alpha_next_seams_fixture_leaves.json). The probes, the engine and the base tool are unchanged.

The command-line builders wrapped: native41_progress_engine_selfrun.command_lines (progress container),
step_a_engine_selfrun.command_lines (fused, groups and composed containers and the negative, already Step B's and alpha-next's),
alpha_c0_engine_selfrun.command_lines (stage container, and alpha-c0 cases in the composed container). Same CLI as the base.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alpha_next_engine_selfrun as base  # noqa: E402

PROBE_LAUNCH = 'python3 -B alpha_next_probe_launch.py '


def probe_launch(lines):
    """Each probe line's interpreter call goes through the launcher (idempotent: a wrapped line is left as it is)."""
    out = []
    for line in lines:
        if '_probe.py ' in line and PROBE_LAUNCH not in line:
            assert line.count(' python3 -B e3_') == 1, line
            line = line.replace(' python3 -B e3_', ' ' + PROBE_LAUNCH + 'e3_')
        out.append(line)
    return out


def _wrap(module):
    inner = module.command_lines

    def command_lines(modes, suites, _inner=inner):
        return probe_launch(_inner(modes, suites))
    command_lines.alpha_next_probe_launch = True
    module.command_lines = command_lines


WRAPPED = (base.held_f, base.step_a, base.c0)
for _module in WRAPPED:
    _wrap(_module)
command_lines = base.step_a.command_lines
expected_cases = base.expected_cases


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--spec', type=Path)
    p.add_argument('--tree', type=Path)
    p.add_argument('--bundle', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--name')
    p.add_argument('--overlay', type=Path)
    p.add_argument('--sources', type=Path)
    p.add_argument('--negative', action='store_true')
    p.add_argument('--observe', action='store_true', help='negative only: record the outcomes, verdict OBSERVED, rc 1')
    p.add_argument('--compare', type=Path, help='candidate engine self-run directory')
    a = p.parse_args()
    if a.compare:
        raise SystemExit(base.compare(a.compare, a.out))
    if a.negative:
        raise SystemExit(base.negative(a.bundle, a.overlay, a.sources, a.out, a.name, a.observe))
    raise SystemExit(base.run(a.spec, a.tree, a.bundle, a.out, a.name))
