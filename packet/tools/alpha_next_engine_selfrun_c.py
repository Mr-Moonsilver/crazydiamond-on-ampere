#!/usr/bin/env python3
"""alpha-next engine self-run, c: tools/alpha_next_engine_selfrun_b.py (which chain g5 ran; superseded here, never edited) with
the negative control bounded and judged fail-closed (harness/alpha_next_wall_guard.py; lead 2 Nov ~23:35 CET).

Why (g5 negative observation, HARNESS): a candidate witness hung over Step B's overlay (fixed in 780b09ef) and nothing bounded the
container; after it was stopped, the self-run still wrote verdict OBSERVED (container rc 137, suites {}). Here, for --negative
(with or without --observe):
  * the container client runs under wall_guard.run_bounded with WALL_LIMIT_S; at the limit the container is stopped
    (docker stop -t 30) and the run is recorded TIMEOUT with its rc, the limit, the wall and what was in flight;
  * wall_guard.inc_reasons decides first: OBSERVED / PASS / FAIL only with container rc 0 (inner.sh records every step's rc in a
    file and exits 0), every negative suite reported and skips 0; anything else is verdict INC (classification INC, exit 4),
    never OBSERVED and never an expected FAIL.
Everything else is the base negative (alpha_next_engine_selfrun.negative) unchanged; the engine self-run and the comparison are
_b's (every probe case through harness/alpha_next_probe_launch.py). Same CLI.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alpha_next_engine_selfrun_b as b  # noqa: E402  (installs the probe-launcher wrapping)

base = b.base
step_a = base.step_a
sys.path.insert(0, str(base.ROOT / 'harness'))
import alpha_next_wall_guard as wall_guard  # noqa: E402

command_lines = b.command_lines
expected_cases = b.expected_cases
INC_RC = 4


class _GuardedSubprocess:
    """step_a.container's view of subprocess while the negative runs: call() is bounded; everything else is the module's."""
    STDOUT = subprocess.STDOUT

    def __init__(self, limit_s):
        self.limit_s = limit_s
        self.guard = None

    def __getattr__(self, name):
        return getattr(subprocess, name)

    def call(self, cmd, stdout=None, stderr=None):
        assert self.guard is None and stderr == subprocess.STDOUT, 'one container per negative'
        name = cmd[cmd.index('--name') + 1]

        def stop():
            subprocess.call(['sudo', '-n', 'docker', 'stop', '-t', '30', name], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
        self.guard = wall_guard.run_bounded(cmd, stdout, self.limit_s, stop)
        return self.guard['rc']


def guarded_container(*args, **kwargs):
    shim = _GuardedSubprocess(wall_guard.WALL_LIMIT_S)
    original = step_a.subprocess
    step_a.subprocess = shim
    try:
        got = base.filtered_container(*args, **kwargs)
    finally:
        step_a.subprocess = original
    return got, shim.guard


def negative(bundle, overlay, sources, out, name, observe=False):
    """base.negative with the bounded container and the fail-closed judge (see the module docstring)."""
    sha = base.sha
    bundle, overlay, sources, out = bundle.resolve(), overlay.resolve(), sources.resolve(), out.resolve()
    pins = json.loads(base.PINS.read_text())
    if observe:
        pins = dict(pins, tests={})
    place = base.frozen()
    assert sha(place / 'result.json') == pins['frozen_record_sha256'], 'pins were taken against another frozen record'
    modes = base.PARENT_PASS + base.c0.CASES + base.HELD_G_CASES
    (rc, result, cases, c, wall), guard = guarded_container(bundle, overlay, sources, out, name, modes, base.NEGATIVE_SUITES,
                                                            hide_candidate=True)
    outcomes = result.get('outcomes') or {}
    left_out = base.left_out_of(out / 'suites.log')
    reasons = base.judge(pins, outcomes, cases, left_out, place, observe)
    inc = wall_guard.inc_reasons(rc, result, list(base.NEGATIVE_SUITES), guard)
    ok = not inc and not reasons and result.get('skips') == 0 and not observe
    verdict = wall_guard.verdict(inc, observe, ok)
    record = dict(schema='alpha-next-negative/1', verdict=verdict, negative_control=True,
                  negative_kind='step-b-parent',
                  classification='INC' if inc else None if ok or observe else 'HARNESS-OR-ENGINE-UNCLASSIFIED',
                  inc_reasons=inc, wall_guard=guard, expected_container_rc=list(wall_guard.EXPECTED_CONTAINER_RC),
                  in_flight=wall_guard.in_flight(out) if inc else None,
                  container_rc=rc, reasons=reasons, suites={k: v for k, v in result.items() if k != 'outcomes'},
                  suite_names=list(base.NEGATIVE_SUITES),
                  outcomes_sha256=hashlib.sha256(json.dumps(outcomes, sort_keys=True).encode()).hexdigest(),
                  left_out={x: base.CONTAINER_EXCLUDED[x] for x in left_out if x in base.CONTAINER_EXCLUDED},
                  pinned_tests=len(pins['tests']), receipt_writer_resets=base.writer_resets(result),
                  cases={m: step_a.case_row(out, m, r) for m, r in cases.items()},
                  bundle=str(bundle), manifest=sha(bundle / 'SHA256SUMS'), overlay=str(overlay),
                  overlay_manifest=sha(overlay / 'SHA256SUMS'), sources=str(sources),
                  pins_sha256=None if observe else sha(base.PINS), image=base.IMAGE,
                  network='none', gpu_device_requests=0, gpu_execution=False, box_action=False, container=name, wall_s=wall,
                  command=c, log_shas={p.name: sha(p) for p in sorted(out.glob('*.log'))})
    if observe:
        record['observed'] = {k: v for k, v in sorted(outcomes.items()) if v[0] != 'ok'}
    (out / 'result.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print(record['verdict'], sha(out / 'result.json'), len(reasons), len(inc))
    if inc:
        return INC_RC
    return 0 if ok else 1


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
    p.add_argument('--observe', action='store_true', help='negative only: record the outcomes, verdict OBSERVED, rc 1 (INC: rc 4)')
    p.add_argument('--compare', type=Path, help='candidate engine self-run directory')
    a = p.parse_args()
    if a.compare:
        raise SystemExit(base.compare(a.compare, a.out))
    if a.negative:
        raise SystemExit(negative(a.bundle, a.overlay, a.sources, a.out, a.name, a.observe))
    raise SystemExit(base.run(a.spec, a.tree, a.bundle, a.out, a.name))
