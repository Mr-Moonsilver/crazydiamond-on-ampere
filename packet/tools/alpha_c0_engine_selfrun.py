#!/usr/bin/env python3
"""C0 engine self-run: the parent's exact connected self-runs plus connected stage cases. CPU only, no network.

Three containers over one frozen bundle, the composed engine in each, all results kept under OUT:
  * progress/: tools/native41_progress_engine_selfrun.py run() unchanged (held-f): 18 connected auto-arm and receipt cases and
    the eight connected native progress cases, burst included, plus its suites, with every C0 switch unset;
  * dense/: tools/native43_selfrun.py run() unchanged (the dense line): probe39, probe37, the spec's suites, the real-scheduler
    tick probe, the clock split;
  * stage/: one case per mode of harness/e3_alpha_c0_probe.py (cases alpha-c0-<mode>), local and serial selections with the switch
    installed, and the C0 suites in one unittest process with per-test outcomes.
OUT/result.json is the merged record (schema alpha-c0-selfrun/1) the frozen runner reads; verdict PASS only when all three are PASS
and every local-against-serial pair below holds. A failure keeps all records, classified HARNESS-OR-ENGINE-UNCLASSIFIED.

Pairs (part of the verdict): each local mode against the same serial mode has the same request tokens and native chunks.

Comparison (--compare): every parent case's canonical trace (trace, request_tokens, rpc_methods, cohort_tokens, ticks, geometry)
equals alpha-perf's frozen engine record byte for byte, and each serial mode equals the same parent progress mode: with the C0
switch unset nothing moves, and with it installed and the arm on serial nothing moves either.

Negative control (--negative): the candidate harness over the regenerated c3f1f64 overlay and sources (build_alpha_c0's OUT/parent,
byte-identical to the parent). The C0 suites run in full: exactly the pinned tests in alpha_c0_negative_pins.json do not pass, each
with its pinned outcome and reason, and every other test passes. Each alpha-c0 probe mode fails with its pinned error; the
parent's own connected cases auto, off, progress-unset, trace, burst and burst-solo pass with the frozen traces.

Declared limit: the connected stage cases run the parent's connected composition, which has a device leaf below the worker. A stage
command on two real PP stages with real collectives has no CPU composition: test_e3_stage_local* run the real driver, bridges,
dispatchers and QSA source with declared device leaves. The box is the first place the whole stack is live, and the delivery says so.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native37_robust_selfrun import IMAGE, overlay_mounts  # noqa: E402
import native41_progress_engine_selfrun as held_f  # noqa: E402
import native43_selfrun as dense  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'alpha-c0-selfrun/1'
SUITES = ('test_e3_stage_local', 'test_e3_stage_local_control', 'test_e3_stage_local_qsa', 'test_alpha_c0_queue', 'test_alpha_c0_report')
PROBE_MODES = ('local-burst', 'local-burst-solo', 'local-burst-skew', 'serial-burst', 'serial-burst-solo', 'serial-burst-skew')
CASES = tuple('alpha-c0-' + m for m in PROBE_MODES)
PAIRS = {'alpha-c0-' + m: 'alpha-c0-serial-' + m[len('local-'):] for m in PROBE_MODES if m.startswith('local-')}
PAIR_FIELDS = ('request_tokens', 'native')
SAME_AS = {'alpha-c0-' + m: 'progress-' + m[len('serial-'):] for m in PROBE_MODES if m.startswith('serial-')}
CANONICAL = held_f.CANONICAL
# Alpha-perf's frozen engine record for the composed trunk: its build-c engine self-run's held-f part.
FROZEN = Path('<hub-dir>/agents/flash-next/tmp/alpha-perf/build-d/engine-selfrun/progress')
PARENT_PASS = ('auto', 'off', 'progress-unset', 'progress-trace', 'progress-burst', 'progress-burst-solo')
PINS = Path(__file__).resolve().with_name('alpha_c0_negative_pins.json')
sha = held_f.sha
canonical = held_f.canonical
DECLARED_LIMIT = ('connected stage cases use the parent composition, with a device leaf below the worker; real collectives on two '
                  'PP stages have no CPU composition; see the tool docstring')


def _suites_py():
    """held-f's suite runner, loading each suite on its own (held-g's deviation, same text): a suite whose module fails at
    import on the parent is recorded under its own name instead of ending the process before any outcome exists."""
    text = held_f.SUITES_PY
    old_run = ("    r=unittest.TextTestRunner(verbosity=2,resultclass=Record).run("
               "unittest.defaultTestLoader.loadTestsFromNames(names))\n")
    new_run = ("    suite,imports=unittest.TestSuite(),{}\n"
               "    for n in names:\n"
               "        try:suite.addTest(unittest.defaultTestLoader.loadTestsFromName(n))\n"
               "        except Exception as exc:\n"
               "            last=''.join(traceback.format_exception_only(type(exc),exc)).strip().splitlines()\n"
               "            imports[n]=('error',(last[-1] if last else '')[:400])\n"
               "    r=unittest.TextTestRunner(verbosity=2,resultclass=Record).run(suite)\n"
               "    r.rows.update(imports)\n")
    for old, new in ((old_run, new_run),
                     ("ok=r.wasSuccessful(),", "ok=r.wasSuccessful() and not imports,import_errors=len(imports),"),
                     ("    raise SystemExit(0 if r.wasSuccessful() and not r.skipped else 1)\n",
                      "    raise SystemExit(0 if r.wasSuccessful() and not r.skipped and not imports else 1)\n")):
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    return text


def command_lines(modes, suites):
    """held-f's lines; alpha-c0 cases in the same form. The probe passes its settings to install_engine through its env argument."""
    lines = held_f.command_lines((), suites)
    for mode in modes:
        if mode.startswith('alpha-c0-'):
            arg = mode[len('alpha-c0-'):]
            lines += ['env -u FLASH_S5_NATIVE_PROGRESS -u FLASH_S5_NATIVE_QSA_POLICY -u FLASH_S5_STAGE_LOCAL FLASH_S5_AUTO_ARM=1 '
                      'python3 -B e3_alpha_c0_probe.py --mode %s --out /out/%s.json > /out/%s.log 2>&1' % (arg, mode, mode),
                      'echo $? > /out/%s.rc' % mode]
        else:
            lines += held_f.command_lines((mode,), ())[1:]
    return lines


def container(bundle, overlay, sources, out, name, modes, suites, hide_candidate=False):
    from build_native30_held import verify
    verify(bundle)
    verify(overlay)
    out.mkdir()
    out.chmod(0o777)
    (out / 'suites.py').write_text(_suites_py())
    (out / 'inner.sh').write_text('\n'.join(command_lines(modes, suites)) + '\n')
    c = ['sudo', '-n', 'docker', 'run', '--pull=never', '--name', name, '--runtime', 'runc', '--network', 'none',
         '--cpus', '4', '--memory', '6g', '--memory-swap', '6g', '--read-only',
         '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=1g', '--ulimit', 'core=0',
         '-e', 'VLLM_CACHE_ROOT=/tmp/vllm-cache', '-e', 'NVIDIA_VISIBLE_DEVICES=none', '-e', 'CUDA_VISIBLE_DEVICES=',
         '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'PYTHONPATH=/source/harness:/source/harness/e3_builder']
    mounts = dict(overlay_mounts(overlay))
    mounts.update({bundle: '/source/harness', sources: '/source/patches/s5_engine', bundle / 'native35_tools': '/source/tools',
                   bundle / 'sparse_stage': '/source/patches/sparse_stage',
                   (ROOT / 'reference/vllm-src').resolve(): '/source/reference/vllm-src'})
    for s, t in mounts.items():
        c += ['-v', '%s:%s:ro' % (s, t)]
    if hide_candidate:
        # The negative: the candidate's engine sources are on PYTHONPATH through the bundle; the parent's take their place.
        c += ['-v', '%s:/source/harness/e3_builder:ro' % sources]
    c += ['-v', '%s:/out' % out, '--entrypoint', 'bash', IMAGE, '/out/inner.sh']
    start = time.monotonic()
    with (out / 'container.log').open('x') as f:
        rc = subprocess.call(c, stdout=f, stderr=subprocess.STDOUT)
    inspected = json.loads(subprocess.check_output(['sudo', '-n', 'docker', 'inspect', name], text=True))[0]['HostConfig']
    assert not inspected.get('DeviceRequests') and not inspected.get('Devices') and inspected['NetworkMode'] == 'none'

    def read(n):
        return json.loads((out / n).read_text()) if (out / n).exists() else {}
    return rc, read('suites.json'), {mode: read(mode + '.json') for mode in modes}, c, round(time.monotonic() - start, 3)


def case_row(out, mode, r):
    return dict(verdict=r.get('verdict'), checks=len(r.get('checks', [])), error_type=r.get('error_type'),
                error=(r.get('error') or '')[:400] or None,
                canonical_sha256=hashlib.sha256(canonical(r)).hexdigest() if r and canonical(r) else None,
                sha256=sha(out / (mode + '.json')) if r else None)


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


def run_stage(bundle, out, name):
    """The third container: the alpha-c0 cases and suites; its own record in OUT/stage/result.json."""
    rc, result, cases, c, wall = container(bundle, bundle / 'native30_overlay', bundle / 'e3_builder', out, name, CASES, SUITES)
    rows, differ = pairs(cases)
    ok = rc == 0 and result.get('ok') is True and result.get('skips') == 0 and not differ and all(
        v.get('verdict') == 'PASS' for v in cases.values())
    record = dict(schema='alpha-c0-stage-selfrun/1', verdict='PASS' if ok else 'FAIL', container_rc=rc,
                  suites={k: v for k, v in result.items() if k != 'outcomes'},
                  outcomes_sha256=hashlib.sha256(json.dumps(result.get('outcomes'), sort_keys=True).encode()).hexdigest(),
                  cases={m: case_row(out, m, r) for m, r in cases.items()}, pairs=rows, pair_fields=list(PAIR_FIELDS),
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
    rc_stage = run_stage(bundle, out / 'stage', name + '-stage')
    p = json.loads((out / 'progress/result.json').read_text())
    d = json.loads((out / 'dense/result.json').read_text())
    f = json.loads((out / 'stage/result.json').read_text())
    ok = rc_progress == rc_dense == rc_stage == 0 and p['verdict'] == d['verdict'] == f['verdict'] == 'PASS'
    pieces = p['suites'], d['suites'], f['suites']
    # The progress and stage records count in 'tests'; the dense one keeps unittest's own line in 'ran'.
    ran = re.fullmatch(r'Ran (\d+) tests? in [0-9.]+s', pieces[1].get('ran') or '')
    tests = [pieces[0].get('tests') or 0, int(ran.group(1)) if ran else 0, pieces[2].get('tests') or 0]
    ok = ok and ran is not None
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
                  cases=connected, pairs=f['pairs'], pair_fields=f['pair_fields'], probe39=d['probe39'], probe37=d['probe37'],
                  py_compile=d['py_compile'], scheduler_tick_probe=d.get('scheduler_tick_probe'), clock_split=d['clock_split'],
                  progress_record_sha256=sha(out / 'progress/result.json'), dense_record_sha256=sha(out / 'dense/result.json'),
                  stage_record_sha256=sha(out / 'stage/result.json'), spec=str(spec), spec_sha256=sha(spec),
                  declared_limit=DECLARED_LIMIT)
    (out / 'result.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(verdict=record['verdict'], tests=record['tests'], skips=record['skips'], cases=len(connected),
                          result_sha256=sha(out / 'result.json')), sort_keys=True))
    return 0 if ok else 1


def negative(bundle, overlay, sources, out, name):
    """The candidate harness over the regenerated c3f1f64 overlay and sources; exactly the pinned rows differ."""
    bundle, overlay, sources, out = bundle.resolve(), overlay.resolve(), sources.resolve(), out.resolve()
    pins = json.loads(PINS.read_text())
    assert sha(FROZEN / 'result.json') == pins['frozen_record_sha256'], 'frozen engine record changed'
    modes = PARENT_PASS + CASES
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
    for mode in PARENT_PASS:
        if cases[mode].get('verdict') != 'PASS':
            reasons.append('parent-case-not-pass:' + mode)
        theirs = canonical(json.loads((FROZEN / (mode + '.json')).read_text()))
        if not cases[mode] or theirs is None or canonical(cases[mode]) != theirs:
            reasons.append('negative-trace-differs-from-frozen:' + mode)
    ok = not reasons and result.get('skips') == 0
    record = dict(schema='alpha-c0-negative/1', verdict='PASS' if ok else 'FAIL', negative_control=True,
                  negative_kind='c3f1f64-parent', classification=None if ok else 'HARNESS-OR-ENGINE-UNCLASSIFIED', container_rc=rc,
                  reasons=reasons, suites={k: v for k, v in result.items() if k != 'outcomes'},
                  outcomes_sha256=hashlib.sha256(json.dumps(outcomes, sort_keys=True).encode()).hexdigest(),
                  pinned_tests=len(pins['tests']), cases={m: case_row(out, m, r) for m, r in cases.items()},
                  bundle=str(bundle), manifest=sha(bundle / 'SHA256SUMS'), overlay=str(overlay),
                  overlay_manifest=sha(overlay / 'SHA256SUMS'), sources=str(sources), pins_sha256=sha(PINS), image=IMAGE,
                  network='none', gpu_device_requests=0, gpu_execution=False, box_action=False, container=name, wall_s=wall,
                  command=c, log_shas={p.name: sha(p) for p in sorted(out.glob('*.log'))})
    (out / 'result.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print(record['verdict'], sha(out / 'result.json'), len(reasons))
    return 0 if ok else 1


def compare(candidate, out, frozen=FROZEN):
    """Canonical traces equal alpha-perf's frozen engine record: the parent's cases and each serial mode."""
    candidate, pins = Path(candidate).resolve(), json.loads(PINS.read_text())
    assert sha(frozen / 'result.json') == pins['frozen_record_sha256'], 'frozen engine record changed'
    rows, problems = {}, []
    for mode in held_f.CASES + tuple(SAME_AS):
        place = candidate / ('stage' if mode in SAME_AS else 'progress') / (mode + '.json')
        ours = json.loads(place.read_text())
        theirs = json.loads((frozen / (SAME_AS.get(mode, mode) + '.json')).read_text())
        a, b = canonical(ours), canonical(theirs)
        if b is None:
            rows[mode] = dict(compared=False, why='no canonical trace in the frozen record')
            continue
        same = ours.get('verdict') == theirs.get('verdict') == 'PASS' and a == b
        rows[mode] = dict(compared=True, same=same, against=SAME_AS.get(mode, mode),
                          canonical_sha256=hashlib.sha256(b).hexdigest() if same else None)
        if not same:
            problems.append(mode)
    result = dict(schema='alpha-c0-default-off-comparison/1', verdict='FAIL' if problems else 'PASS', differs=problems,
                  fields=CANONICAL, modes=rows, compared=sum(1 for v in rows.values() if v['compared']),
                  parent_record=str(frozen / 'result.json'), parent_record_sha256=pins['frozen_record_sha256'],
                  candidate=str(candidate), candidate_record_sha256=sha(candidate / 'result.json'),
                  meaning='with the C0 switch unset every parent connected case has alpha-perf\'s exact canonical trace; '
                          'with the switch installed and the arm on serial, each serial mode equals the same parent progress mode')
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
