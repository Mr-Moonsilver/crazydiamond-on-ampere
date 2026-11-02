#!/usr/bin/env python3
"""C1 engine self-run: C0 l's engine self-run (tools/alpha_c0_engine_selfrun.py, unchanged) plus connected overlap cases. CPU only.

Three containers over one frozen bundle, the composed engine in each, all results kept under OUT, exactly as C0's:
  * progress/: held-f's run() unchanged (18 connected auto-arm and receipt cases, the eight connected progress cases, its suites),
    every C0 and C1 switch unset;
  * dense/: native43_selfrun run() with its suites line through the bundle's harness/clock_bound.py -v -m (C0's wrapper, taken
    from alpha_c0_engine_selfrun unchanged); the ids left out and the reason are in the merged record (dense_clock_bound), and
    the verdict needs them equal to the listed ids of those suites;
  * stage/: C0's six alpha-c0 probe cases (C0's command lines, unchanged) and one case per mode of harness/e3_alpha_c1_probe.py
    (cases alpha-c1-<mode>, FLASH_S5_STAGE_OVERLAP=1 installed by the probe beside FLASH_S5_STAGE_LOCAL=1), with the C0 and C1
    suites in one unittest process with per-test outcomes.
OUT/result.json is the merged record (schema alpha-c1-selfrun/1) the frozen runner reads; verdict PASS only when all three are PASS,
every pair below holds and the slow-stage-1 case opened early. A failure keeps all records, classified HARNESS-OR-ENGINE-UNCLASSIFIED.

Pairs (part of the verdict): C0's (each local mode against the same serial mode: request tokens and native chunks), and C1's:
each overlap mode and C1's local mode against C1's serial mode with the same request tokens, native chunks and per-rank accepted
segment commands (per_rank_segments_sha256: each rank received the same device commands in the same order); the solo overlap
mode against C0's serial solo mode (request tokens, native chunks). early_open: the overlap-burst-slow1 case has early_opens > 0
(stage 0 opened n+1 before stage 1 finished n, in the slow-stage-1 mode of the real composition; the lead's gate).

Comparison (--compare): against C0 l's frozen engine record (tmp/c0c1/pack/build-l/engine-selfrun): every parent progress case
and every alpha-c0 case has C0 l's canonical trace byte for byte (C1's switch unset moves nothing), and C1's serial and local
modes (the overlap switch installed, overlap not selected) each equal C0's case of the same mode.

Negative control (--negative): the candidate harness over the regenerated C0 l overlay and sources (build_alpha_c1's OUT/parent,
byte-identical to the parent). The C0 and C1 suites run in full: exactly the pinned tests in alpha_c1_negative_pins.json do not
pass, each with its pinned outcome and reason, and every other test passes. Each alpha-c1 probe mode has its pinned outcome; the
parent's held-f cases and C0's six alpha-c0 cases pass with C0 l's frozen traces.

Declared limit, as C0: the connected cases run the parent's connected composition with a device leaf below the worker; real
collectives on two PP stages have no CPU composition. The box is the first place the whole stack is live.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alpha_c0_engine_selfrun as c0  # noqa: E402

held_f, dense = c0.held_f, c0.dense
C0_COMMAND_LINES = c0.command_lines   # C0's own, taken before any binding
SCHEMA = 'alpha-c1-selfrun/1'
C1_SUITES = ('test_e3_stage_overlap', 'test_e3_stage_overlap_control', 'test_alpha_c1_box', 'test_alpha_c1_report',
             'test_alpha_c1_queue')
SUITES = c0.SUITES + C1_SUITES
PROBE_MODES = ('overlap-burst', 'overlap-burst-solo', 'overlap-burst-skew', 'overlap-burst-slow1', 'local-burst', 'serial-burst')
CASES = tuple('alpha-c1-' + m for m in PROBE_MODES)
STAGE_CASES = c0.CASES + CASES
SEGMENTS = ('request_tokens', 'native', 'per_rank_segments_sha256')
C1_PAIRS = {'alpha-c1-overlap-burst': ('alpha-c1-serial-burst', SEGMENTS),
            'alpha-c1-overlap-burst-skew': ('alpha-c1-serial-burst', SEGMENTS),
            'alpha-c1-overlap-burst-slow1': ('alpha-c1-serial-burst', SEGMENTS),
            'alpha-c1-local-burst': ('alpha-c1-serial-burst', SEGMENTS),
            'alpha-c1-overlap-burst-solo': ('alpha-c0-serial-burst-solo', c0.PAIR_FIELDS)}
EARLY_OPEN_CASE = 'alpha-c1-overlap-burst-slow1'
# C1's serial and local modes against C0's case of the same mode (C0 l's frozen stage record).
SAME_AS_C0 = {'alpha-c1-serial-burst': 'alpha-c0-serial-burst', 'alpha-c1-local-burst': 'alpha-c0-local-burst'}
CANONICAL = held_f.CANONICAL
# C0 l's frozen engine record (its delivered build-l engine self-run): progress/ and stage/.
FROZEN = Path('<hub-dir>/agents/flash-next/tmp/c0c1/pack/build-l/engine-selfrun')
PARENT_PASS = c0.PARENT_PASS
PINS = Path(__file__).resolve().with_name('alpha_c1_negative_pins.json')
sha = held_f.sha
canonical = held_f.canonical
DECLARED_LIMIT = c0.DECLARED_LIMIT


def command_lines(modes, suites):
    """C0's lines (held-f's, and the alpha-c0 cases in C0's form); alpha-c1 cases in the same form with C1's switch unset too.
    The probe installs its settings through install_engine's env argument."""
    lines = C0_COMMAND_LINES([m for m in modes if not m.startswith('alpha-c1-')], suites)
    for mode in modes:
        if mode.startswith('alpha-c1-'):
            arg = mode[len('alpha-c1-'):]
            lines += ['env -u FLASH_S5_NATIVE_PROGRESS -u FLASH_S5_NATIVE_QSA_POLICY -u FLASH_S5_STAGE_LOCAL -u FLASH_S5_STAGE_OVERLAP '
                      'FLASH_S5_AUTO_ARM=1 python3 -B e3_alpha_c1_probe.py --mode %s --out /out/%s.json > /out/%s.log 2>&1'
                      % (arg, mode, mode), 'echo $? > /out/%s.rc' % mode]
    return lines


@contextlib.contextmanager
def _c1_lines():
    """C0's container() with these command lines for one call; C0's restored on exit."""
    saved = c0.command_lines
    c0.command_lines = command_lines
    try:
        yield
    finally:
        c0.command_lines = saved


def container(*args, **kwargs):
    with _c1_lines():
        return c0.container(*args, **kwargs)


def pairs(cases):
    rows, problems = c0.pairs(cases)
    for mode, (against, fields) in sorted(C1_PAIRS.items()):
        ours, theirs = cases.get(mode) or {}, cases.get(against) or {}
        same = {f: f in ours and f in theirs and ours[f] == theirs[f] for f in fields}
        ok = ours.get('verdict') == theirs.get('verdict') == 'PASS' and all(same.values())
        rows[mode] = dict(against=against, same=ok, fields=same,
                          canonical_equal=bool(ours and theirs and canonical(ours) == canonical(theirs)))
        if not ok:
            problems.append(mode)
    return rows, problems


def early_open(cases):
    counters = (((cases.get(EARLY_OPEN_CASE) or {}).get('overlap') or {}).get('counters') or {})
    opens = counters.get('early_opens')
    return dict(case=EARLY_OPEN_CASE, early_opens=opens, successor_frames=counters.get('successor_frames'),
                ok=type(opens) is int and opens > 0)


def run_stage(bundle, out, name):
    """The third container: the alpha-c0 and alpha-c1 cases and suites; its own record in OUT/stage/result.json."""
    rc, result, cases, c, wall = container(bundle, bundle / 'native30_overlay', bundle / 'e3_builder', out, name, STAGE_CASES,
                                           SUITES)
    rows, differ = pairs(cases)
    early = early_open(cases)
    ok = rc == 0 and result.get('ok') is True and result.get('skips') == 0 and not differ and early['ok'] and all(
        v.get('verdict') == 'PASS' for v in cases.values())
    record = dict(schema='alpha-c1-stage-selfrun/1', verdict='PASS' if ok else 'FAIL', container_rc=rc,
                  suites={k: v for k, v in result.items() if k != 'outcomes'},
                  outcomes_sha256=hashlib.sha256(json.dumps(result.get('outcomes'), sort_keys=True).encode()).hexdigest(),
                  cases={m: c0.case_row(out, m, r) for m, r in cases.items()}, pairs=rows, differs=differ, early_open=early,
                  pair_fields=dict(c0=list(c0.PAIR_FIELDS), c1={m: list(f) for m, (_a, f) in sorted(C1_PAIRS.items())}),
                  manifest=sha(bundle / 'SHA256SUMS'), image=c0.IMAGE, network='none', gpu_device_requests=0,
                  gpu_execution=False, box_action=False, container=name, wall_s=wall, command=c,
                  log_shas={p.name: sha(p) for p in sorted(out.glob('*.log'))})
    (out / 'result.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print(record['verdict'], sha(out / 'result.json'))
    return 0 if ok else 1


def run(spec, tree, bundle, out, name):
    spec, bundle, out = Path(spec).resolve(), Path(bundle).resolve(), Path(out).resolve()
    out.mkdir()
    out.chmod(0o777)
    rc_progress = held_f.run(bundle, out / 'progress', name + '-progress')
    original = dense.base.inner_script
    dense.base.inner_script = c0._dense_suites_through_clock_bound(original)
    try:
        rc_dense = dense.run(spec, tree, bundle, out / 'dense', name + '-dense')
    finally:
        dense.base.inner_script = original
    dense_clock_bound = c0._dense_clock_bound_record(spec, bundle, out)
    rc_stage = run_stage(bundle, out / 'stage', name + '-stage')
    p = json.loads((out / 'progress/result.json').read_text())
    d = json.loads((out / 'dense/result.json').read_text())
    f = json.loads((out / 'stage/result.json').read_text())
    ok = rc_progress == rc_dense == rc_stage == 0 and p['verdict'] == d['verdict'] == f['verdict'] == 'PASS'
    pieces = p['suites'], d['suites'], f['suites']
    # The progress and stage records count in 'tests'; the dense one keeps unittest's own line in 'ran'.
    ran = re.fullmatch(r'Ran (\d+) tests? in [0-9.]+s', pieces[1].get('ran') or '')
    tests = [pieces[0].get('tests') or 0, int(ran.group(1)) if ran else 0, pieces[2].get('tests') or 0]
    ok = ok and ran is not None and dense_clock_bound['ok']
    skips = [(x.get('skips') if x.get('skips') is not None else x.get('skipped')) or 0 for x in pieces]
    connected = dict(p['cases'], **f['cases'])
    record = dict(schema=SCHEMA, verdict='PASS' if ok else 'FAIL',
                  classification=None if ok else 'HARNESS-OR-ENGINE-UNCLASSIFIED', negative_control=False,
                  container_rc=max(p['container_rc'], d['container_rc'], f['container_rc']),
                  manifest=p['manifest'], bundle_manifest=p['manifest'], bundle=p['bundle'],
                  overlay_manifest=p['overlay_manifest'], image=p['image'], network=p['network'],
                  gpu_device_requests=max(p['gpu_device_requests'], d['clock_split'].get('gpu_device_requests') or 0,
                                          f['gpu_device_requests']),
                  gpu_execution=False, box_action=False, candidate=d.get('label') or spec.name,
                  tests=sum(tests) + len(connected), skips=sum(skips), connected_checks=sum(v['checks'] for v in connected.values()),
                  suites=dict(ok=all(x.get('ok') is True for x in pieces), skips=sum(skips), tests=sum(tests),
                              progress=pieces[0], dense=pieces[1], stage=pieces[2]),
                  cases=connected, pairs=f['pairs'], pair_fields=f['pair_fields'], early_open=f['early_open'],
                  probe39=d['probe39'], probe37=d['probe37'],
                  py_compile=d['py_compile'], scheduler_tick_probe=d.get('scheduler_tick_probe'), clock_split=d['clock_split'],
                  progress_record_sha256=sha(out / 'progress/result.json'), dense_record_sha256=sha(out / 'dense/result.json'),
                  stage_record_sha256=sha(out / 'stage/result.json'), spec=str(spec), spec_sha256=sha(spec),
                  dense_clock_bound=dense_clock_bound, declared_limit=DECLARED_LIMIT)
    (out / 'result.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(verdict=record['verdict'], tests=record['tests'], skips=record['skips'], cases=len(connected),
                          result_sha256=sha(out / 'result.json')), sort_keys=True))
    return 0 if ok else 1


def _frozen(mode):
    """C0 l's frozen case record: held-f cases under progress/, alpha-c0 cases under stage/."""
    return json.loads((FROZEN / ('stage' if mode.startswith('alpha-c0-') else 'progress') / (mode + '.json')).read_text())


def negative(bundle, overlay, sources, out, name):
    """The candidate harness over the regenerated C0 l overlay and sources; exactly the pinned rows differ."""
    bundle, overlay, sources, out = bundle.resolve(), overlay.resolve(), sources.resolve(), out.resolve()
    pins = json.loads(PINS.read_text())
    assert sha(FROZEN / 'result.json') == pins['frozen_record_sha256'], 'frozen engine record changed'
    modes = PARENT_PASS + c0.CASES + CASES
    rc, result, cases, c, wall = container(bundle, overlay, sources, out, name, modes, SUITES, hide_candidate=True)
    reasons = []
    outcomes = result.get('outcomes') or {}
    for test, (kind, text) in pins['tests'].items():
        got = outcomes.get(test)
        if not got or got[0] != kind or text not in got[1]:
            reasons.append('negative-test-differs:%s:%s' % (test, got))
    for test, (kind, text) in outcomes.items():
        if test not in pins['tests'] and kind != 'ok':
            reasons.append('undeclared-negative-outcome:%s:%s:%s' % (test, kind, text))
    if not outcomes or not pins['tests']:
        reasons.append('negative-suites-not-run')
    for mode in CASES:
        got = [cases[mode].get('verdict'), cases[mode].get('error_type'), cases[mode].get('error')]
        if got != pins['cases'][mode]:
            reasons.append('negative-case-differs:%s:%s' % (mode, got))
    for mode in PARENT_PASS + c0.CASES:
        if cases[mode].get('verdict') != 'PASS':
            reasons.append('parent-case-not-pass:' + mode)
        theirs = canonical(_frozen(mode))
        if not cases[mode] or theirs is None or canonical(cases[mode]) != theirs:
            reasons.append('negative-trace-differs-from-frozen:' + mode)
    ok = not reasons and result.get('skips') == 0
    record = dict(schema='alpha-c1-negative/1', verdict='PASS' if ok else 'FAIL', negative_control=True,
                  negative_kind='c0-l-parent', classification=None if ok else 'HARNESS-OR-ENGINE-UNCLASSIFIED', container_rc=rc,
                  reasons=reasons, suites={k: v for k, v in result.items() if k != 'outcomes'},
                  outcomes_sha256=hashlib.sha256(json.dumps(outcomes, sort_keys=True).encode()).hexdigest(),
                  pinned_tests=len(pins['tests']), cases={m: c0.case_row(out, m, r) for m, r in cases.items()},
                  bundle=str(bundle), manifest=sha(bundle / 'SHA256SUMS'), overlay=str(overlay),
                  overlay_manifest=sha(overlay / 'SHA256SUMS'), sources=str(sources), pins_sha256=sha(PINS), image=c0.IMAGE,
                  network='none', gpu_device_requests=0, gpu_execution=False, box_action=False, container=name, wall_s=wall,
                  command=c, log_shas={p.name: sha(p) for p in sorted(out.glob('*.log'))})
    (out / 'result.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print(record['verdict'], sha(out / 'result.json'), len(reasons))
    return 0 if ok else 1


def compare(candidate, out, frozen=FROZEN):
    """Canonical traces equal C0 l's frozen engine record: the parent's cases, C0's cases, and C1's serial and local modes."""
    candidate, pins = Path(candidate).resolve(), json.loads(PINS.read_text())
    assert sha(frozen / 'result.json') == pins['frozen_record_sha256'], 'frozen engine record changed'
    rows, problems = {}, []
    for mode in held_f.CASES + c0.CASES + tuple(SAME_AS_C0):
        place = candidate / ('progress' if mode in held_f.CASES else 'stage') / (mode + '.json')
        against = SAME_AS_C0.get(mode, mode)
        ours = json.loads(place.read_text())
        theirs = json.loads((frozen / ('stage' if against.startswith('alpha-c0-') else 'progress') / (against + '.json')).read_text())
        a, b = canonical(ours), canonical(theirs)
        if b is None:
            rows[mode] = dict(compared=False, why='no canonical trace in the frozen record')
            continue
        same = ours.get('verdict') == theirs.get('verdict') == 'PASS' and a == b
        rows[mode] = dict(compared=True, same=same, against=against,
                          canonical_sha256=hashlib.sha256(b).hexdigest() if same else None)
        if not same:
            problems.append(mode)
    result = dict(schema='alpha-c1-default-off-comparison/1', verdict='FAIL' if problems else 'PASS', differs=problems,
                  fields=CANONICAL, modes=rows, compared=sum(1 for v in rows.values() if v['compared']),
                  parent_record=str(frozen / 'result.json'), parent_record_sha256=pins['frozen_record_sha256'],
                  candidate=str(candidate), candidate_record_sha256=sha(candidate / 'result.json'),
                  meaning='with the C1 switch unset every parent connected case and every C0 case has C0 l\'s exact canonical '
                          'trace; with the switch installed and the arm on serial or local, each of those modes equals C0\'s case')
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
    p.add_argument('--compare', type=Path, help='candidate engine self-run directory')
    a = p.parse_args()
    if a.compare:
        raise SystemExit(compare(a.compare, a.out))
    if a.negative:
        raise SystemExit(negative(a.bundle, a.overlay, a.sources, a.out, a.name))
    raise SystemExit(run(a.spec, a.tree, a.bundle, a.out, a.name))
