#!/usr/bin/env python3
"""Step B engine self-run: Step A's self-run, its containers and records unchanged, plus connected group cases. CPU only, no network.

Four containers over one frozen bundle, the composed engine in each, all results kept under OUT:
  * progress/: tools/native41_progress_engine_selfrun.py run() unchanged (held-f), every switch unset;
  * dense/: tools/native43_selfrun.py run() unchanged (the dense line);
  * fused/: tools/step_a_engine_selfrun.py run_fused() unchanged: Step A's probe modes and suites, FLASH_S5_LAYER_GROUPS unset;
  * groups/: one case per mode of harness/e3_step_b_probe.py (cases step-b-<mode>), with both switches installed and each arm's size
    selected by the typed control, and the Step B suites in one unittest process with per-test outcomes.
OUT/result.json is the merged record (schema step-b-selfrun/1) the frozen runner reads; verdict PASS only when all four are PASS and
every pair below holds. A failure keeps all records, classified HARNESS-OR-ENGINE-UNCLASSIFIED.

Pairs (part of the verdict): each size 2 and size 4 mode against size 1 on the same load has the same request tokens and native chunks.
That the commands differ is what groups are, so the whole canonical trace is recorded per pair and never required.

Comparison (--compare): against Step A's frozen engine record, byte for byte on the canonical trace: every parent connected case; every
Step A fused and stepped mode with the Step B switch unset. Size 1 with the switch installed is compared on tokens and native chunks to
Step A's fused mode: its leaf driver differs from Step A's, so the trace field is recorded and not required.

Negative control (--negative): the candidate harness over the regenerated Step A overlay and sources (build_step_b's OUT/parent,
byte-identical to the parent). The Step B suites run in full: exactly the pinned tests in step_b_negative_pins.json do not pass, each
with its pinned outcome and reason, and every other test passes. Each step-b probe mode fails with its pinned error; Step A's fused
modes pass with the frozen traces.

Declared limit: the connected group cases put a declared layer-driver leaf under the real generated executor path. The real
QsaLayerDriver and QsaPrefillOperation in a group have no CPU composition with the real Core step; they run in
test_e3_native_layer_groups (GroupNetwork, real driver, bridge, dispatcher, QSA driver and operation) and on the VM's one real card. The
box is the first place the whole stack is live, and the delivery says so.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native37_robust_selfrun import IMAGE  # noqa: E402
import native41_progress_engine_selfrun as held_f  # noqa: E402
import native43_selfrun as dense  # noqa: E402
import step_a_engine_selfrun as step_a  # noqa: E402
import native39_trunk_selfrun as dense_base  # noqa: E402

# The dense container's suites run through the declared real-clock window-slot list (harness/step_b_clock_bound.py, reason "box timing
# is governed by the lead at launch (operator, 2 Nov)"), as strict and the bundle self-run do: the same suites, same verbosity, minus
# those ids. Chain d's first engine self-run failed only on test_native43_boundary_queue's window test past its 03:57Z close
# (HARNESS; the lead's pre-approval, 2 Nov). Tools only; the bundle and its harness are unchanged. The unfiltered suite is dropped
# before the run (del s), so unittest frees each test after it runs as it does under -m unittest (attempt 2 was killed without it).
_DENSE_UNITTEST = 'python3 -B -m unittest -v '
_DENSE_FILTERED = ('python3 -B -c "import sys,unittest; sys.path.insert(0,\'.\'); import step_b_clock_bound as c; names=sys.argv[1:]; '
                   's=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in names); k,r=c.filtered(s); del s; '
                   'assert r==c.expected_removed(names),r; [print(\'clock-bound, excluded (declared):\',x,file=sys.stderr) for x in r]; '
                   'sys.exit(0 if unittest.TextTestRunner(verbosity=2).run(k).wasSuccessful() else 1)" ')
_dense_original = dense_base.inner_script


def _dense_inner_script(label, suites, compiled):
    text = _dense_original(label, suites, compiled)
    if suites:
        assert text.count(_DENSE_UNITTEST) == 1, 'dense suite line'
        text = text.replace(_DENSE_UNITTEST, _DENSE_FILTERED)
    return text


dense_base.inner_script = _dense_inner_script

ROOT = Path(__file__).resolve().parents[1]
PARENT_FILE = ROOT / 'tools/step_b_parent.json'
SCHEMA = 'step-b-selfrun/1'
SUITES = ('test_e3_native_layer_groups', 'test_step_b_queue')
PROBE_MODES = ('g1-burst', 'g1-burst-solo', 'g1-burst-skew', 'g2-burst', 'g2-burst-solo', 'g2-burst-skew', 'g4-burst')
CASES = tuple('step-b-' + m for m in PROBE_MODES)
PAIRS = {'step-b-' + m: 'step-b-g1-' + m[len('g2-'):] for m in PROBE_MODES if not m.startswith('g1-')}
PAIR_FIELDS = ('request_tokens', 'native')
SAME_AS_STEP_A = {'step-b-g1-burst': 'step-a-fused-burst', 'step-b-g1-burst-solo': 'step-a-fused-burst-solo',
                  'step-b-g1-burst-skew': 'step-a-fused-burst-skew'}
PARENT_PASS = ('step-a-fused-burst', 'step-a-fused-burst-solo', 'step-a-stepped-burst')
PINS = Path(__file__).resolve().with_name('step_b_negative_pins.json')
sha = held_f.sha
canonical = held_f.canonical
DECLARED_LIMIT = ('connected group cases use a declared layer-driver leaf under the real generated executor path; the real QSA driver '
                  'and operation in a group run in test_e3_native_layer_groups and on the VM; see the tool docstring')
_step_a_lines = step_a.command_lines


def step_a_cases():
    """The connected cases Step A's engine record holds: held-f's and Step A's probe modes."""
    return sorted(list(held_f.CASES) + list(step_a.CASES))


def expected_cases():
    """The connected cases this record holds, the one source for the spec's runner swap and the record's own check."""
    return sorted(step_a_cases() + list(CASES))


def frozen():
    """Step A's frozen engine record, read from the parent file and pinned there."""
    pins = json.loads(PARENT_FILE.read_text())
    place = Path(pins['engine_selfrun'])
    assert sha(place / 'result.json') == pins['engine_selfrun_sha256'], 'Step A engine record changed'
    return place


def command_lines(modes, suites):
    """Step A's lines; step-b cases in the same form, with the group switch unset on the way in as well."""
    theirs = [m for m in modes if not m.startswith('step-b-')]
    lines = _step_a_lines(theirs, suites)
    for mode in modes:
        if mode.startswith('step-b-'):
            arg = mode[len('step-b-'):]
            lines += ['env -u FLASH_S5_NATIVE_PROGRESS -u FLASH_S5_NATIVE_QSA_POLICY -u FLASH_S5_QSA_FUSED -u FLASH_S5_LAYER_GROUPS '
                      'FLASH_S5_AUTO_ARM=1 python3 -B e3_step_b_probe.py --mode %s --out /out/%s.json > /out/%s.log 2>&1' % (arg, mode, mode),
                      'echo $? > /out/%s.rc' % mode]
    return lines


step_a.command_lines = command_lines


def pairs(cases):
    rows, problems = {}, []
    for mode, against in sorted(PAIRS.items()):
        ours, theirs = cases.get(mode) or {}, cases.get(against) or {}
        same = {f: f in ours and f in theirs and ours[f] == theirs[f] for f in PAIR_FIELDS}
        ok = ours.get('verdict') == theirs.get('verdict') == 'PASS' and all(same.values())
        rows[mode] = dict(against=against, same=ok, fields=same,
                          canonical_equal=bool(ours and theirs and canonical(ours) == canonical(theirs)))
        if not ok:
            problems.append(mode)
    return rows, problems


def run_groups(bundle, out, name):
    """The fourth container: the step-b cases and suites; its own record in OUT/groups/result.json."""
    rc, result, cases, c, wall = step_a.container(bundle, bundle / 'native30_overlay', bundle / 'e3_builder', out, name, CASES, SUITES)
    rows, differ = pairs(cases)
    ok = rc == 0 and result.get('ok') is True and result.get('skips') == 0 and not differ and all(
        v.get('verdict') == 'PASS' for v in cases.values())
    record = dict(schema='step-b-groups-selfrun/1', verdict='PASS' if ok else 'FAIL', container_rc=rc,
                  suites={k: v for k, v in result.items() if k != 'outcomes'},
                  outcomes_sha256=hashlib.sha256(json.dumps(result.get('outcomes'), sort_keys=True).encode()).hexdigest(),
                  cases={m: step_a.case_row(out, m, r) for m, r in cases.items()}, pairs=rows, pair_fields=list(PAIR_FIELDS),
                  differs=differ, manifest=sha(bundle / 'SHA256SUMS'), image=IMAGE, network='none', gpu_device_requests=0,
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
    rc_dense = dense.run(spec, tree, bundle, out / 'dense', name + '-dense')
    rc_fused = step_a.run_fused(bundle, out / 'fused', name + '-fused')
    rc_groups = run_groups(bundle, out / 'groups', name + '-groups')
    p = json.loads((out / 'progress/result.json').read_text())
    d = json.loads((out / 'dense/result.json').read_text())
    f = json.loads((out / 'fused/result.json').read_text())
    g = json.loads((out / 'groups/result.json').read_text())
    ok = rc_progress == rc_dense == rc_fused == rc_groups == 0 and p['verdict'] == d['verdict'] == f['verdict'] == g['verdict'] == 'PASS'
    pieces = p['suites'], d['suites'], f['suites'], g['suites']
    # The progress, fused and group records count in 'tests'; the dense one keeps unittest's own line in 'ran'.
    ran = re.fullmatch(r'Ran (\d+) tests? in [0-9.]+s', pieces[1].get('ran') or '')
    tests = [pieces[0].get('tests') or 0, int(ran.group(1)) if ran else 0, pieces[2].get('tests') or 0, pieces[3].get('tests') or 0]
    ok = ok and ran is not None
    skips = [(x.get('skips') if x.get('skips') is not None else x.get('skipped')) or 0 for x in pieces]
    connected = dict(p['cases'], **f['cases'], **g['cases'])
    ok = ok and sorted(connected) == expected_cases()
    record = dict(schema=SCHEMA, verdict='PASS' if ok else 'FAIL',
                  classification=None if ok else 'HARNESS-OR-ENGINE-UNCLASSIFIED', negative_control=False,
                  container_rc=max(p['container_rc'], d['container_rc'], f['container_rc'], g['container_rc']),
                  manifest=p['manifest'], bundle_manifest=p['manifest'], bundle=p['bundle'],
                  overlay_manifest=p['overlay_manifest'], image=p['image'], network=p['network'],
                  gpu_device_requests=max(p['gpu_device_requests'], d['clock_split'].get('gpu_device_requests') or 0,
                                          f['gpu_device_requests'], g['gpu_device_requests']),
                  gpu_execution=False, box_action=False, candidate=d.get('label') or spec.name,
                  tests=sum(tests) + len(connected), skips=sum(skips), connected_checks=sum(v['checks'] for v in connected.values()),
                  suites=dict(ok=all(x.get('ok') is True for x in pieces), skips=sum(skips), tests=sum(tests),
                              progress=pieces[0], dense=pieces[1], fused=pieces[2], groups=pieces[3]),
                  cases=connected, pairs=dict(f['pairs'], **g['pairs']), pair_fields=g['pair_fields'], probe39=d['probe39'],
                  probe37=d['probe37'], py_compile=d['py_compile'], scheduler_tick_probe=d.get('scheduler_tick_probe'),
                  clock_split=d['clock_split'], progress_record_sha256=sha(out / 'progress/result.json'),
                  dense_record_sha256=sha(out / 'dense/result.json'), fused_record_sha256=sha(out / 'fused/result.json'),
                  groups_record_sha256=sha(out / 'groups/result.json'), spec=str(spec), spec_sha256=sha(spec),
                  declared_limit=DECLARED_LIMIT)
    (out / 'result.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(verdict=record['verdict'], tests=record['tests'], skips=record['skips'], cases=len(connected),
                          result_sha256=sha(out / 'result.json')), sort_keys=True))
    return 0 if ok else 1


def negative(bundle, overlay, sources, out, name):
    """The candidate harness over the regenerated Step A overlay and sources; exactly the pinned rows differ."""
    bundle, overlay, sources, out = bundle.resolve(), overlay.resolve(), sources.resolve(), out.resolve()
    pins = json.loads(PINS.read_text())
    place = frozen()
    modes = PARENT_PASS + CASES
    rc, result, cases, c, wall = step_a.container(bundle, overlay, sources, out, name, modes, SUITES, hide_candidate=True)
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
    for mode in PARENT_PASS:
        if cases[mode].get('verdict') != 'PASS':
            reasons.append('parent-case-not-pass:' + mode)
        theirs = canonical(json.loads((place / 'fused' / (mode + '.json')).read_text()))
        if not cases[mode] or theirs is None or canonical(cases[mode]) != theirs:
            reasons.append('negative-trace-differs-from-frozen:' + mode)
    ok = not reasons and result.get('skips') == 0
    record = dict(schema='step-b-negative/1', verdict='PASS' if ok else 'FAIL', negative_control=True,
                  negative_kind='step-a-parent', classification=None if ok else 'HARNESS-OR-ENGINE-UNCLASSIFIED', container_rc=rc,
                  reasons=reasons, suites={k: v for k, v in result.items() if k != 'outcomes'},
                  outcomes_sha256=hashlib.sha256(json.dumps(outcomes, sort_keys=True).encode()).hexdigest(),
                  pinned_tests=len(pins['tests']), cases={m: step_a.case_row(out, m, r) for m, r in cases.items()},
                  bundle=str(bundle), manifest=sha(bundle / 'SHA256SUMS'), overlay=str(overlay),
                  overlay_manifest=sha(overlay / 'SHA256SUMS'), sources=str(sources), pins_sha256=sha(PINS), image=IMAGE,
                  network='none', gpu_device_requests=0, gpu_execution=False, box_action=False, container=name, wall_s=wall,
                  command=c, log_shas={p.name: sha(p) for p in sorted(out.glob('*.log'))})
    (out / 'result.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print(record['verdict'], sha(out / 'result.json'), len(reasons))
    return 0 if ok else 1


def compare(candidate, out):
    """Canonical traces equal Step A's frozen engine record; size 1 matches Step A's fused mode on tokens and chunks."""
    candidate, place = Path(candidate).resolve(), frozen()
    rows, problems = {}, []
    inherited = [('progress', m) for m in held_f.CASES] + [('fused', m) for m in step_a.CASES]
    for part, mode in inherited:
        ours = json.loads((candidate / part / (mode + '.json')).read_text())
        theirs = json.loads((place / part / (mode + '.json')).read_text())
        a, b = canonical(ours), canonical(theirs)
        if b is None:
            rows[mode] = dict(compared=False, why='no canonical trace in the frozen record')
            continue
        same = ours.get('verdict') == theirs.get('verdict') == 'PASS' and a == b
        rows[mode] = dict(compared=True, same=same, against=mode, canonical_sha256=hashlib.sha256(b).hexdigest() if same else None)
        if not same:
            problems.append(mode)
    for mode, against in sorted(SAME_AS_STEP_A.items()):
        ours = json.loads((candidate / 'groups' / (mode + '.json')).read_text())
        theirs = json.loads((place / 'fused' / (against + '.json')).read_text())
        fields = {f: f in ours and f in theirs and ours[f] == theirs[f] for f in PAIR_FIELDS}
        same = ours.get('verdict') == theirs.get('verdict') == 'PASS' and all(fields.values())
        rows[mode] = dict(compared=True, same=same, against=against, fields=fields, canonical_equal=canonical(ours) == canonical(theirs))
        if not same:
            problems.append(mode)
    result = dict(schema='step-b-default-off-comparison/1', verdict='FAIL' if problems else 'PASS', differs=problems,
                  fields=held_f.CANONICAL, modes=rows, compared=sum(1 for v in rows.values() if v['compared']),
                  parent_record=str(place / 'result.json'), parent_record_sha256=sha(place / 'result.json'),
                  candidate=str(candidate), candidate_record_sha256=sha(candidate / 'result.json'),
                  meaning='with the Step B switch unset every Step A connected case has its exact canonical trace; with the switch installed '
                          'and the arm on size 1, tokens and native chunks equal Step A\'s fused mode')
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
