#!/usr/bin/env python3
"""alpha-next-c1 engine self-run: alpha-next's six containers, unchanged, plus C1's overlap container. CPU only, no network.

Seven containers over one frozen bundle, the composed engine in each, all results kept under OUT:
  * progress/, dense/, fused/, groups/, stage/, composed/: tools/alpha_next_engine_selfrun.py's parts, called unchanged (imported, so
    its command lines, its receipt-writer reset in every shared-process runner and its dense line through step_b_clock_bound and
    alpha_next_robust_swap are installed exactly as in its own run); every C1 switch unset in all of them;
  * overlap/: C1's connected cases (tools/alpha_c1_engine_selfrun.py's: one case per mode of harness/e3_alpha_c1_probe.py, cases
    alpha-c1-<mode>, C1's exact command line form: FLASH_S5_STAGE_OVERLAP=1 installed by the probe beside FLASH_S5_STAGE_LOCAL=1,
    overlap-burst-slow1 included) and the suites C1, the composition fix and this queue add (OVERLAP_SUITES: test_e3_stage_overlap,
    test_alpha_next_c1_composed_fields (382ed704: alpha-next's per-chunk fields kept per frame, lean successors; leaves in
    alpha_next_c1_engine_fixture_leaves.json), test_e3_stage_overlap_control, the C1 rebind's test_c1_rebind_witness and
    test_c1_rebind_on (c1_rebind_fixture_leaves.json), the per-arm defer_bind control test_alpha_next_c1_defer_control, and
    test_alpha_next_c1_queue), in one unittest process with per-test outcomes (held-f's runner with alpha-next's writer reset).
    Over alpha-next's overlay (the negative) every one of these but the queue suite imports stage_overlap (directly or through
    test_e3_stage_overlap), so each fails to import there: pinned HARNESS rows at the one observation, the control biting.
OUT/result.json is the merged record (schema alpha-next-c1-selfrun/1) the frozen runner reads; its connected cases are exactly
expected_cases() (alpha-next's and C1's). Verdict PASS only when all seven are PASS, C1's pairs hold and the slow-stage-1 case opened
early. A failure keeps all records, classified HARNESS-OR-ENGINE-UNCLASSIFIED.

C1's pairs (part of the verdict, tools/alpha_c1_engine_selfrun.C1_PAIRS unchanged): each overlap mode and C1's local mode against C1's
serial mode with the same request tokens, native chunks and per-rank accepted segment commands; the solo overlap mode against C0's
serial solo mode (stage/). early_open: alpha-c1-overlap-burst-slow1 has early_opens > 0 (the lead's gate).

Comparison (--compare): against alpha-next's frozen engine record (tools/alpha_next_c1_parent.json engine_selfrun, sha-pinned), byte
for byte on the canonical trace: every connected case alpha-next's record holds, in its own part (C1 off == alpha-next exactly: with
the overlap switch unset nothing moves), and C1's serial and local modes (switch installed, overlap not selected) against alpha-next's
alpha-c0 case of the same mode (C1's SAME_AS_C0 rule). Overlap on is the mechanism above (pairs, early open), not an equality.

Negative control (--negative): the candidate harness over the regenerated alpha-next overlay and sources (build_alpha_next_c1's
OUT/parent, byte-identical to the parent). OVERLAP_SUITES run in full: exactly the pinned tests in alpha_next_c1_negative_pins.json do
not pass, each with its pinned outcome and reason, and every other test passes. Each alpha-c1 probe mode fails with its pinned error;
alpha-next's PARENT_PASS cases, C0's stage cases and held-g's probe cases pass with alpha-next's frozen traces. --negative --observe
records the outcomes once (verdict OBSERVED, rc 1, never a pass) for the pins file (two-phase, as alpha-next's); the pins are written
from that one observation, after review, by tools/alpha_next_c1_negative_pins_write.py.
Bounded and fail-closed (lead 2 Nov ~23:50 CET, after alpha-next's g5 negative hung; HARNESS): the negative's container client runs
under alpha-next's shared guard (harness/alpha_next_wall_guard.run_bounded, NEGATIVE_WALL_LIMIT_S): at the limit the container is
stopped (docker stop -t 30), the client killed after the grace, and the run recorded TIMEOUT with rc, limit, wall and what was in
flight. wall_guard.inc_reasons judges first: OBSERVED / PASS / FAIL only with container rc 0, every NEGATIVE_SUITES suite reported
and skips 0; otherwise (timeout, killed, rc 137 with suites {}, a missing suite, skips, an unreadable container record) the verdict
is INC, classification INC, exit INC_RC 4: never OBSERVED and never an expected FAIL.

Declared limit, as alpha-next and C1: the connected cases use declared device leaves below the worker; real collectives on two PP
stages and the anatomy switch on have no CPU composition. The box is the first place the whole stack is live.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alpha_next_engine_selfrun as an  # noqa: E402  (installs alpha-next's lines, writer resets and dense line)
# alpha-next's g4 fix (HARNESS, PLAN 2 receipts): every probe line through harness/alpha_next_probe_launch.py. Importing the _b
# tool wraps the base tool's command-line builders in place; C1's own probe lines go through the same launcher below.
import alpha_next_engine_selfrun_b as anb  # noqa: E402
import alpha_c1_engine_selfrun as c1  # noqa: E402  (C1's constants and helpers; installs nothing)

ROOT = an.ROOT
SCHEMA = 'alpha-next-c1-selfrun/1'
held_f, dense, step_a, step_b, c0 = an.held_f, an.dense, an.step_a, an.step_b, an.c0
OVERLAP_SUITES = ('test_e3_stage_overlap', 'test_alpha_next_c1_composed_fields', 'test_e3_stage_overlap_control',
                  'test_c1_rebind_witness', 'test_c1_rebind_on', 'test_alpha_next_c1_defer_control', 'test_alpha_next_c1_queue')
NEGATIVE_SUITES = OVERLAP_SUITES
CASES = c1.CASES
C1_PAIRS = c1.C1_PAIRS
SAME_AS_C0 = c1.SAME_AS_C0
EARLY_OPEN_CASE = c1.EARLY_OPEN_CASE
PARENT_PASS = an.PARENT_PASS + c0.CASES + an.HELD_G_CASES
PARTS = an.PARTS + ('overlap',)
PINS = Path(__file__).resolve().with_name('alpha_next_c1_negative_pins.json')
sha = an.sha
canonical = an.canonical
DECLARED_LIMIT = an.DECLARED_LIMIT + '; ' + c1.DECLARED_LIMIT
sys.path.insert(0, str(ROOT / 'harness'))
import alpha_next_wall_guard as wall_guard  # noqa: E402  (alpha-next's shared guard, 55b21ae5: one implementation for both)
INC_RC = 4
NEGATIVE_WALL_LIMIT_S = wall_guard.WALL_LIMIT_S      # 2,700 s, alpha-next's (its largest container wall ~820 s at load ~40)
NEGATIVE_STOP_GRACE_S = wall_guard.STOP_GRACE_S

# ---------------------------------------------------------------- command lines: alpha-next's, plus C1's cases in C1's form
_an_lines = step_a.command_lines
assert _an_lines is anb.command_lines and getattr(_an_lines, 'alpha_next_probe_launch', False), \
    'alpha-next command lines (with the probe launcher) not installed'


def command_lines(modes, suites):
    """alpha-next's lines (installed in step_a) for every non-C1 mode; alpha-c1 cases in C1's exact form (its own lines without
    their leading cd, which alpha-next's first line already holds)."""
    lines = _an_lines([m for m in modes if not m.startswith('alpha-c1-')], suites)
    for mode in modes:
        if mode.startswith('alpha-c1-'):
            own = c1.command_lines((mode,), ())
            assert own[0] == lines[0] == 'cd /source/harness', (own[0], lines[0])
            lines += anb.probe_launch(own[1:])
    return lines


step_a.command_lines = command_lines


def expected_cases():
    """The connected cases the merged record holds: alpha-next's and C1's. The one source for the spec's runner swap."""
    return sorted(an.expected_cases() + list(CASES))


def _pin():
    import alpha_next_c1_parent_pin as parent_pin
    return parent_pin.load()


def frozen():
    """alpha-next's frozen engine record, read from the parent file and pinned there."""
    pins = _pin()
    assert 'engine_selfrun' in pins, 'the parent pin has no engine self-run (a development parent cannot be compared)'
    place = Path(pins['engine_selfrun'])
    assert sha(place / 'result.json') == pins['engine_selfrun_sha256'], 'alpha-next engine record changed'
    return place


def part_of(mode):
    """The part of alpha-next's record (and of this one) a connected case lives in."""
    if mode in CASES:
        return 'overlap'
    if mode in held_f.CASES:
        return 'progress'
    if mode in step_a.CASES:
        return 'fused'
    if mode in step_b.CASES:
        return 'groups'
    if mode in c0.CASES:
        return 'stage'
    if mode in an.HELD_G_CASES:
        return 'composed'
    raise KeyError(mode)


def c1_pairs(cases):
    """C1's pairs over the merged cases (the solo overlap mode's partner is C0's serial solo case in stage/)."""
    rows, problems = {}, []
    for mode, (against, fields) in sorted(C1_PAIRS.items()):
        ours, theirs = cases.get(mode) or {}, cases.get(against) or {}
        same = {f: f in ours and f in theirs and ours[f] == theirs[f] for f in fields}
        ok = ours.get('verdict') == theirs.get('verdict') == 'PASS' and all(same.values())
        rows[mode] = dict(against=against, same=ok, fields=same,
                          canonical_equal=bool(ours and theirs and canonical(ours) == canonical(theirs)))
        if not ok:
            problems.append(mode)
    return rows, problems


def run_overlap(bundle, out, name):
    """The seventh container: C1's probe cases and OVERLAP_SUITES; its own record in OUT/overlap/result.json."""
    rc, result, cases, c, wall = step_a.container(bundle, bundle / 'native30_overlay', bundle / 'e3_builder', out, name, CASES,
                                                  OVERLAP_SUITES)
    ok = rc == 0 and result.get('ok') is True and result.get('skips') == 0 and (result.get('tests') or 0) > 0 \
        and an.writer_resets(result) is not None and all(v.get('verdict') == 'PASS' for v in cases.values()) \
        and sorted(cases) == sorted(CASES)
    record = dict(schema='alpha-next-c1-overlap-selfrun/1', verdict='PASS' if ok else 'FAIL', container_rc=rc,
                  suites={k: v for k, v in result.items() if k != 'outcomes'}, suite_names=list(OVERLAP_SUITES),
                  outcomes_sha256=hashlib.sha256(json.dumps(result.get('outcomes'), sort_keys=True).encode()).hexdigest(),
                  receipt_writer_resets=an.writer_resets(result),
                  cases={m: step_a.case_row(out, m, r) for m, r in cases.items()},
                  manifest=sha(bundle / 'SHA256SUMS'), image=an.IMAGE, network='none', gpu_device_requests=0,
                  gpu_execution=False, box_action=False, container=name, wall_s=wall, command=c,
                  log_shas={p.name: sha(p) for p in sorted(out.glob('*.log'))})
    (out / 'result.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print(record['verdict'], sha(out / 'result.json'))
    return 0 if ok else 1


def run(spec, tree, bundle, out, name):
    spec, bundle, out = Path(spec).resolve(), Path(bundle).resolve(), Path(out).resolve()
    out.mkdir()
    out.chmod(0o777)
    rcs = [held_f.run(bundle, out / 'progress', name + '-progress'),
           dense.run(spec, tree, bundle, out / 'dense', name + '-dense'),
           step_a.run_fused(bundle, out / 'fused', name + '-fused'),
           step_b.run_groups(bundle, out / 'groups', name + '-groups'),
           c0.run_stage(bundle, out / 'stage', name + '-stage'),
           an.run_composed(bundle, out / 'composed', name + '-composed'),
           run_overlap(bundle, out / 'overlap', name + '-overlap')]
    parts = [json.loads((out / x / 'result.json').read_text()) for x in PARTS]
    p, d, f, g, s, n, o = parts
    left = an.dense_left_out(spec, out)
    ok = rcs == [0] * len(PARTS) and all(x['verdict'] == 'PASS' for x in parts) and left['ok']
    pieces = [x['suites'] for x in parts]
    # The dense record keeps unittest's own line in 'ran'; every other part counts in 'tests'.
    ran = re.fullmatch(r'Ran (\d+) tests? in [0-9.]+s', pieces[1].get('ran') or '')
    tests = [pieces[0].get('tests') or 0, int(ran.group(1)) if ran else 0] + [x.get('tests') or 0 for x in pieces[2:]]
    ok = ok and ran is not None
    skips = [(x.get('skips') if x.get('skips') is not None else x.get('skipped')) or 0 for x in pieces]
    connected = dict(p['cases'], **f['cases'], **g['cases'], **s['cases'], **n['cases'], **o['cases'])
    ok = ok and sorted(connected) == expected_cases()
    pairs, differ = c1_pairs(connected)
    early = c1.early_open(connected)
    ok = ok and not differ and early['ok']
    resets = dict({x: an.writer_resets(r['suites']) for x, r in zip(PARTS, parts) if x != 'dense'},
                  dense=left['receipt_writer_resets'])
    ok = ok and all(v is not None for v in resets.values())
    record = dict(schema=SCHEMA, verdict='PASS' if ok else 'FAIL',
                  classification=None if ok else 'HARNESS-OR-ENGINE-UNCLASSIFIED', negative_control=False,
                  container_rc=max(x['container_rc'] for x in parts),
                  manifest=p['manifest'], bundle_manifest=p['manifest'], bundle=p['bundle'],
                  overlay_manifest=p['overlay_manifest'], image=p['image'], network=p['network'],
                  gpu_device_requests=max([d['clock_split'].get('gpu_device_requests') or 0]
                                          + [x['gpu_device_requests'] for x in parts if x is not d]),
                  gpu_execution=False, box_action=False, candidate=d.get('label') or spec.name,
                  tests=sum(tests) + len(connected), skips=sum(skips), connected_checks=sum(v['checks'] for v in connected.values()),
                  suites=dict(ok=all(x.get('ok') is True for x in pieces), skips=sum(skips), tests=sum(tests),
                              **dict(zip(PARTS, pieces))),
                  cases=connected, pairs=dict(f['pairs'], **g['pairs'], **s['pairs'], **n['pairs'], **pairs),
                  pair_fields=g['pair_fields'], held_g_pair_fields=n['pair_fields'],
                  c1_pair_fields={m: list(fl) for m, (_a, fl) in sorted(C1_PAIRS.items())}, c1_differs=differ, early_open=early,
                  probe39=d['probe39'], probe37=d['probe37'], py_compile=d['py_compile'],
                  scheduler_tick_probe=d.get('scheduler_tick_probe'), clock_split=d['clock_split'], dense_left_out=left,
                  composed_suites=list(an.SUITES), composed_left_out=n['left_out'], overlap_suites=list(OVERLAP_SUITES),
                  receipt_writer_resets=resets,
                  **{x + '_record_sha256': sha(out / x / 'result.json') for x in PARTS},
                  spec=str(spec), spec_sha256=sha(spec), declared_limit=DECLARED_LIMIT)
    (out / 'result.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(verdict=record['verdict'], tests=record['tests'], skips=record['skips'], cases=len(connected),
                          result_sha256=sha(out / 'result.json')), sort_keys=True))
    return 0 if ok else 1


def judge(pins, outcomes, cases, place, observe=False):
    """The negative's reasons from its record alone: a pure function, so it can be dry-run over a kept record."""
    reasons = []
    for test, (kind, text) in pins['tests'].items():
        got = outcomes.get(test)
        if not got or got[0] != kind or text not in got[1]:
            reasons.append('negative-test-differs:%s:%s' % (test, got))
    for test, (kind, text) in outcomes.items():
        if test not in pins['tests'] and kind != 'ok' and not observe:
            reasons.append('undeclared-negative-outcome:%s:%s:%s' % (test, kind, text))
    if not outcomes or (not pins['tests'] and not observe):
        reasons.append('negative-suites-not-run')
    for mode in CASES:
        got = [(cases.get(mode) or {}).get(k) for k in ('verdict', 'error_type', 'error')]
        if got != pins['cases'].get(mode):
            reasons.append('negative-case-differs:%s:%s' % (mode, got))
    for mode in PARENT_PASS:
        if (cases.get(mode) or {}).get('verdict') != 'PASS':
            reasons.append('parent-case-not-pass:' + mode)
        frozen_case = place / part_of(mode) / (mode + '.json')
        theirs = canonical(json.loads(frozen_case.read_text())) if frozen_case.is_file() else None
        if not cases.get(mode) or theirs is None or canonical(cases[mode]) != theirs:
            reasons.append('negative-trace-differs-from-frozen:' + mode)
    return reasons


def _stop_container(name):
    subprocess.call(['sudo', '-n', 'docker', 'stop', '-t', '30', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class _GuardedSubprocess:
    """step_a.container's view of subprocess while the negative runs: call() is bounded; everything else is the module's."""
    STDOUT = subprocess.STDOUT
    DEVNULL = subprocess.DEVNULL

    def __init__(self, limit_s):
        self.limit_s = limit_s
        self.guard = None

    def __getattr__(self, name):
        return getattr(subprocess, name)

    def call(self, cmd, stdout=None, stderr=None):
        assert self.guard is None and stderr == subprocess.STDOUT, 'one container per negative'
        name = cmd[cmd.index('--name') + 1]
        self.guard = wall_guard.run_bounded(cmd, stdout, self.limit_s, lambda: _stop_container(name), NEGATIVE_STOP_GRACE_S)
        return self.guard['rc']


def guarded_container(*args, **kwargs):
    """step_a.container under the wall guard: ((rc, result, cases, command, wall) or None, guard, error)."""
    shim = _GuardedSubprocess(NEGATIVE_WALL_LIMIT_S)
    original = step_a.subprocess
    step_a.subprocess = shim
    try:
        return step_a.container(*args, **kwargs), shim.guard, None
    except Exception as exc:   # e.g. the stopped container's record cannot be read: INC with the reason, never a traceback only
        return None, shim.guard, '%s: %s' % (type(exc).__name__, exc)
    finally:
        step_a.subprocess = original


def negative_judgement(rc, result, guard, reasons, observe, error=None):
    """(verdict, classification, inc_reasons, ok): wall_guard.inc_reasons first (fail closed), then OBSERVED / PASS / FAIL."""
    inc = wall_guard.inc_reasons(rc, result, list(NEGATIVE_SUITES), guard)
    if guard is None:
        inc.append('container-not-run-under-the-wall-guard')
    if error:
        inc.append('container-record-unreadable:' + error)
    ok = not inc and not reasons and (result or {}).get('skips') == 0 and not observe
    verdict = wall_guard.verdict(inc, observe, ok)
    return verdict, 'INC' if inc else None if ok or observe else 'HARNESS-OR-ENGINE-UNCLASSIFIED', inc, ok


def negative(bundle, overlay, sources, out, name, observe=False):
    """The candidate harness over the regenerated alpha-next overlay and sources; exactly the pinned rows differ."""
    bundle, overlay, sources, out = bundle.resolve(), overlay.resolve(), sources.resolve(), out.resolve()
    pins = json.loads(PINS.read_text())
    place = frozen()
    if observe:
        pins = dict(pins, tests={})
        assert pins['frozen_record_sha256'] in (None, sha(place / 'result.json')), 'stub taken against another frozen record'
    else:
        assert sha(place / 'result.json') == pins['frozen_record_sha256'], 'pins were taken against another frozen record'
    modes = PARENT_PASS + CASES
    got, guard, error = guarded_container(bundle, overlay, sources, out, name, modes, NEGATIVE_SUITES, hide_candidate=True)
    rc, result, cases, c, wall = got if got is not None else ((guard or {}).get('rc'), {}, {}, None, (guard or {}).get('wall_s'))
    outcomes = result.get('outcomes') or {}
    reasons = judge(pins, outcomes, cases, place, observe)
    verdict, classification, inc, ok = negative_judgement(rc, result, guard, reasons, observe, error)
    record = dict(schema='alpha-next-c1-negative/1', verdict=verdict,
                  negative_control=True, negative_kind='alpha-next-parent',
                  classification=classification, inc_reasons=inc, wall_guard=guard,
                  expected_container_rc=list(wall_guard.EXPECTED_CONTAINER_RC),
                  in_flight=wall_guard.in_flight(out) if inc else None,
                  container_rc=rc, reasons=reasons, suites={k: v for k, v in result.items() if k != 'outcomes'},
                  suite_names=list(NEGATIVE_SUITES),
                  outcomes_sha256=hashlib.sha256(json.dumps(outcomes, sort_keys=True).encode()).hexdigest(),
                  pinned_tests=len(pins['tests']), receipt_writer_resets=an.writer_resets(result),
                  cases={m: step_a.case_row(out, m, r) for m, r in cases.items()},
                  frozen_record_sha256=sha(place / 'result.json'),
                  bundle=str(bundle), manifest=sha(bundle / 'SHA256SUMS'), overlay=str(overlay),
                  overlay_manifest=sha(overlay / 'SHA256SUMS'), sources=str(sources),
                  pins_sha256=None if observe else sha(PINS), image=an.IMAGE,
                  network='none', gpu_device_requests=0, gpu_execution=False, box_action=False, container=name, wall_s=wall,
                  command=c, log_shas={p.name: sha(p) for p in sorted(out.glob('*.log'))})
    if observe and verdict == 'OBSERVED':       # an INC record carries no observation (the pin writer can never read one)
        record['observed'] = {k: v for k, v in sorted(outcomes.items()) if v[0] != 'ok'}
    (out / 'result.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print(record['verdict'], sha(out / 'result.json'), len(reasons), len(inc))
    if inc:
        return INC_RC
    return 0 if ok else 1


def comparison_rules():
    """(part, mode, their part, against): every alpha-next connected case in its own part, and C1's SAME_AS_C0 modes."""
    rules = [(part_of(m), m, part_of(m), m) for m in an.expected_cases()]
    rules += [('overlap', m, 'stage', against) for m, against in sorted(SAME_AS_C0.items())]
    return rules


def compare(candidate, out):
    """Canonical traces equal alpha-next's frozen engine record: every alpha-next case, and C1's serial and local modes."""
    candidate, place = Path(candidate).resolve(), frozen()
    rows, problems = {}, []
    for part, mode, their_part, against in comparison_rules():
        ours = json.loads((candidate / part / (mode + '.json')).read_text())
        theirs = json.loads((place / their_part / (against + '.json')).read_text())
        a, b = canonical(ours), canonical(theirs)
        if b is None:
            rows[mode] = dict(compared=False, why='no canonical trace in the frozen record')
            continue
        same = ours.get('verdict') == theirs.get('verdict') == 'PASS' and a == b
        rows[mode] = dict(compared=True, same=same, against='%s/%s' % (their_part, against),
                          canonical_sha256=hashlib.sha256(b).hexdigest() if same else None)
        if not same:
            problems.append(mode)
    result = dict(schema='alpha-next-c1-default-off-comparison/1', verdict='FAIL' if problems else 'PASS', differs=problems,
                  fields=held_f.CANONICAL, modes=rows, compared=sum(1 for v in rows.values() if v['compared']),
                  parent_record=str(place / 'result.json'), parent_record_sha256=sha(place / 'result.json'),
                  candidate=str(candidate), candidate_record_sha256=sha(candidate / 'result.json'),
                  meaning='with the C1 switch unset every alpha-next connected case has alpha-next\'s exact canonical trace; with the '
                          'switch installed and the arm on serial or local, each of those modes equals alpha-next\'s alpha-c0 case of '
                          'the same mode')
    with Path(out).open('x') as f:
        json.dump(result, f, sort_keys=True, indent=2)
        f.write('\n')
    print(result['verdict'], problems)
    return 0 if not problems else 1


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
        raise SystemExit(compare(a.compare, a.out))
    if a.negative:
        raise SystemExit(negative(a.bundle, a.overlay, a.sources, a.out, a.name, a.observe))
    raise SystemExit(run(a.spec, a.tree, a.bundle, a.out, a.name))
