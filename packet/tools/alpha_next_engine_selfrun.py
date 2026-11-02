#!/usr/bin/env python3
"""alpha-next engine self-run: Step B's four containers, C0's stage container and the composed suites container. CPU only, no network.

Six containers over one frozen bundle, the composed engine in each, all results kept under OUT:
  * progress/, fused/, groups/: tools/step_b_engine_selfrun.py's parts, called unchanged (held-f's connected cases and suites,
    Step A's fused and stepped modes, Step B's connected group cases and suites);
  * dense/: tools/native43_selfrun.py run() unchanged (the dense line). One change, the same kind as Step B's and C0's: the suites
    line runs through harness/step_b_clock_bound.py's declared real-clock ids AND harness/alpha_next_robust_swap.py's declared ids
    (the load-flaky frozen method in test_native43_le_bounded_enqueue, swapped for its twin, which runs in composed/), each
    removed exactly once, the unfiltered suite dropped before the run as Step B's line does. The ids left out are read back from
    the dense suites.log and must equal the declared ids of the spec's suites (dense_left_out in the merged record);
  * stage/: tools/alpha_c0_engine_selfrun.py run_stage(), called unchanged: one case per mode of harness/e3_alpha_c0_probe.py
    (local and serial selections with the switch installed) and the C0 suites, its local-against-serial pairs;
  * composed/: held-g's connected probe cases (tools/native41_held_g_engine_selfrun.py's, claude/native41-held-g-pack: one case per
    mode of harness/e3_native41_held_g_probe.py, cases held-g-<mode>: held-f's eight progress modes with idle step and replies full
    and the R5 anatomy checks, the three L3 -wait and three L4 -lean modes, read from the probe's MODES/WAIT_MODES/LEAN_MODES; the
    same command line form) with held-g's pairs (each -wait mode against its step mode, each -lean mode against its full mode, on
    scheduled_nonempty_sha256, request tokens and native chunks), and the suites the composition adds and that no inherited
    container runs (stage-1 anatomy, held-g anatomy and lean
    replies, the stop-fix trunk's two suites, the alpha-next stop-fix matrix, C0 ring, peak receipt, receipt writer (PLAN 2) and
    stage-1 slow-rank suites, the 4C.5 path qualification suite, the two load-robust twins, the alpha-next queue
    suite, and test_step_a_report for the replaced tools/step_a_report.py), in one unittest process with per-test outcomes. No
    connected case runs FLASH_S5_HANDLER_ANATOMY on (default off; its CUDA event pairs have no CPU composition). Declared exclusions (CONTAINER_EXCLUDED, tools/stage1_sweep_git_bound.py's rule on
    claude/stage1-sweep): the stage-1 FreezeProof method (superseded by tools/alpha_next_freeze.py, alpha_next_robust_swap
    .SUPERSEDED) and the stage-1 Off byte-identity test, which reads its git base by `git show`; the containers mount no
    repository. Both run on the hub (the focused list through alpha_next_robust_swap; strict). test_alpha_next_receipts_mutants
    is not here: it copies the writer and reruns test_alpha_next_receipts in subprocesses (about 3 min); it runs on the hub in the
    focused list and strict.
OUT/result.json is the merged record (schema alpha-next-selfrun/1) the frozen runner reads; its connected cases are exactly
expected_cases() (Step B's, C0's stage cases and held-g's probe cases). Verdict PASS only when all six are PASS. A failure keeps all records,
classified HARNESS-OR-ENGINE-UNCLASSIFIED.

Comparison (--compare): against Step B's frozen engine record (tools/alpha_next_parent.json), byte for byte on the canonical trace:
every connected case Step B's record holds (progress, fused and groups), each C0 serial mode against Step B's progress mode of
the same name (C0's SAME_AS rule), and each held-g step/full probe mode against Step B's progress mode of the same name (held-g's
SAME_AS rule): with the C0 and anatomy switches unset nothing moves, with C0 installed and the arm on serial nothing moves, and
with the held-g selections at step/full the R1/R5 receipts change no scheduling, tick or token.

Negative control (--negative): the candidate harness over the regenerated Step B overlay and sources (build_alpha_next's OUT/parent,
byte-identical to the parent). C0's suites and the composed suites run in full: exactly the pinned tests in
alpha_next_negative_pins.json do not pass, each with its pinned outcome and reason, and every other test passes. Each alpha-c0 and
held-g probe mode fails with its pinned error; Step B's group cases and Step A's parent modes pass with Step B's frozen traces.
--negative --observe records the outcomes once (verdict OBSERVED, rc 1, never a pass) for the pins file, Step B's and stage-1
sweep's precedent; the pins are written from that one observation, after review, by tools/alpha_next_negative_pins_write.py.

Declared limit: the stage and group connected cases use declared device leaves (see Step B's and C0's tools); the anatomy switch on
runs in test_stage1_handler_anatomy and on the VM, never in a connected CPU case. The box is the first place the whole stack is live.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native37_robust_selfrun import IMAGE  # noqa: E402
import native41_progress_engine_selfrun as held_f  # noqa: E402
import native43_selfrun as dense  # noqa: E402
import native39_trunk_selfrun as dense_base  # noqa: E402
import step_a_engine_selfrun as step_a  # noqa: E402
import step_b_engine_selfrun as step_b  # noqa: E402  (installs Step B's command lines in step_a and its dense line)
import alpha_c0_engine_selfrun as c0  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PARENT_FILE = ROOT / 'tools/alpha_next_parent.json'
SCHEMA = 'alpha-next-selfrun/1'
SUITES = ('test_stage1_handler_anatomy', 'test_stage1_anatomy_report', 'test_step_a_report', 'test_e3_native_anatomy',
          'test_e3_native_lean_replies', 'test_alpha_next_stopfix_matrix', 'test_alpha_next_c0_ring', 'test_alpha_next_peak_receipt',
          'test_e3_stopfix_partial_segment', 'test_e3_stopfix_refused_open', 'test_alpha_next_receipts',
          'test_alpha_next_stage1_slow_rank', 'test_alpha_next_path_qualification', 'test_e3_w1_decode_driver_ordered',
          'test_native43_le_bounded_enqueue_stepped', 'test_alpha_next_queue')
NEGATIVE_SUITES = c0.SUITES + SUITES
PARENT_PASS = step_b.PARENT_PASS + step_b.CASES
SAME_AS = c0.SAME_AS


def _held_g_modes():
    """held-f's progress MODES and held-g's WAIT_MODES and LEAN_MODES, read from the probes' text (held-g's _probe_modes)."""
    def literals(path, *names):
        values = {}
        for node in ast.parse((ROOT / 'harness' / path).read_text()).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], 'id', None) in names:
                values[node.targets[0].id] = ast.literal_eval(node.value)
        assert set(values) == set(names), (path, names)
        return values
    base = literals('e3_native41_progress_probe.py', 'MODES')['MODES']
    held_g = literals('e3_native41_held_g_probe.py', 'WAIT_MODES', 'LEAN_MODES')
    assert base == held_f.PROGRESS_MODES, 'held-f progress probe modes changed'
    return base, held_g['WAIT_MODES'], held_g['LEAN_MODES']


HELD_G_STEP, HELD_G_WAIT, HELD_G_LEAN = _held_g_modes()
HELD_G_CASES = tuple('held-g-' + m for m in HELD_G_STEP + HELD_G_WAIT + HELD_G_LEAN)
HELD_G_PAIRS = dict([('held-g-' + m, 'held-g-' + m[:-len('-wait')]) for m in HELD_G_WAIT]
                    + [('held-g-' + m, 'held-g-' + m[:-len('-lean')]) for m in HELD_G_LEAN])
assert all(v in HELD_G_CASES for v in HELD_G_PAIRS.values()), HELD_G_PAIRS
HELD_G_PAIR_FIELDS = ('scheduled_nonempty_sha256', 'request_tokens', 'native')
HELD_G_SAME_AS = {'held-g-' + m: 'progress-' + m for m in HELD_G_STEP}
PINS = Path(__file__).resolve().with_name('alpha_next_negative_pins.json')
sha = held_f.sha
canonical = held_f.canonical
DECLARED_LIMIT = ('the anatomy switch on runs in test_stage1_handler_anatomy and on the VM, never in a connected CPU case; '
                  + step_b.DECLARED_LIMIT + '; ' + c0.DECLARED_LIMIT + '; see the tool docstring')

# ---------------------------------------------------------------- composed container: declared exclusions
_SWAP = None


def _robust_swap():
    """harness/alpha_next_robust_swap.py, the one declaration of the swapped and superseded ids."""
    global _SWAP
    if _SWAP is None:
        sys.path.insert(0, str(ROOT / 'harness'))
        import alpha_next_robust_swap
        _SWAP = alpha_next_robust_swap
    return _SWAP


GIT_BOUND_REASON = 'reads the anatomy git base 5f46c07 by git show; the container mounts no repository; runs on the hub tree'
CONTAINER_EXCLUDED = dict(
    {'test_stage1_handler_anatomy.Off.test_progress_lines_byte_identical_to_step_b_off_and_on': GIT_BOUND_REASON},
    **{name: 'superseded (alpha_next_robust_swap.SUPERSEDED): ' + why for name, why in _robust_swap().SUPERSEDED.items()
       if name.split('.', 1)[0] in SUITES})
EXCLUDED_LINE = 'composed, excluded (declared): '
_RUN_LINE = "    r=unittest.TextTestRunner(verbosity=2,resultclass=Record).run(suite)\n"
_RUN_FILTERED = ("    def flat(s):\n"
                 "        for x in s:\n"
                 "            yield from (flat(x) if isinstance(x,unittest.TestSuite) else [x])\n"
                 "    tests=list(flat(suite));drop=set(%r)\n"
                 "    removed=sorted(x.id() for x in tests if x.id() in drop)\n"
                 "    for x in removed:print(%r+x,file=sys.stderr)\n"
                 "    suite=unittest.TestSuite(x for x in tests if x.id() not in drop);tests=None\n" + _RUN_LINE)


def composed_suites_py(base):
    """step_a's suite runner with CONTAINER_EXCLUDED removed before the run (exact-once swap)."""
    text = base()
    assert text.count(_RUN_LINE) == 1, 'suite run line'
    return text.replace(_RUN_LINE, _RUN_FILTERED % (sorted(CONTAINER_EXCLUDED), EXCLUDED_LINE))


def applied(left_out, outcomes, candidate):
    """No declared id may have an outcome. Over the candidate every declared id was removed; over the parent each was removed or
    its module's import error is the recorded outcome (the stage-1 suite cannot import over Step B's overlay)."""
    if set(CONTAINER_EXCLUDED) & set(outcomes) or not set(left_out) <= set(CONTAINER_EXCLUDED):
        return False
    if candidate:
        return sorted(left_out) == sorted(CONTAINER_EXCLUDED)
    return all(import_failed(outcomes, i.split('.')[0]) for i in set(CONTAINER_EXCLUDED) - set(left_out))


def import_failed(outcomes, module):
    return any((outcomes.get(k) or ('',))[0] == 'error' for k in (module, 'unittest.loader._FailedTest.' + module))


def left_out_of(log):
    return sorted(re.findall('^' + re.escape(EXCLUDED_LINE) + r'(\S+)$', log.read_text(), re.M)) if log.is_file() else []


# ---------------------------------------------------------------- command lines: Step B's, plus C0's stage cases
_step_b_lines = step_a.command_lines


def command_lines(modes, suites):
    """Step B's lines (installed in step_a); alpha-c0 cases in C0's exact form."""
    lines = _step_b_lines([m for m in modes if not m.startswith(('alpha-c0-', 'held-g-'))], suites)
    for mode in modes:
        if mode.startswith('alpha-c0-'):
            lines += c0.command_lines((mode,), ())[1:]
        elif mode.startswith('held-g-'):
            # held-g's form (tools/native41_held_g_engine_selfrun.py): the probe passes its setting to install_engine itself.
            arg = mode[len('held-g-'):]
            lines += ['env -u FLASH_S5_NATIVE_PROGRESS FLASH_S5_AUTO_ARM=1 python3 -B e3_native41_held_g_probe.py '
                      '--mode %s --out /out/%s.json > /out/%s.log 2>&1' % (arg, mode, mode), 'echo $? > /out/%s.rc' % mode]
    return lines


step_a.command_lines = command_lines


def filtered_container(*args, **kwargs):
    """step_a.container with the composed suite runner (the composed container and the negative)."""
    original = step_a._suites_py
    step_a._suites_py = lambda: composed_suites_py(original)
    try:
        return step_a.container(*args, **kwargs)
    finally:
        step_a._suites_py = original


# ---------------------------------------------------------------- dense line: clock-bound and robust-swap ids
_DENSE_UNITTEST = 'python3 -B -m unittest -v '
DENSE_CLOCK_LINE = 'clock-bound, excluded (declared): '
DENSE_SWAP_LINE = 'robust-swap, excluded (declared): '
_DENSE_FILTERED = ('python3 -B -c "import sys,unittest; sys.path.insert(0,\'.\'); import step_b_clock_bound as c; '
                   'import alpha_next_robust_swap as w; names=sys.argv[1:]; '
                   's=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in names); k,r=c.filtered(s); del s; '
                   'assert r==c.expected_removed(names),r; k,q=w.filtered(k); assert q==w.expected_removed(names),q; '
                   '[print(\'%s\'+x,file=sys.stderr) for x in r]; [print(\'%s\'+x,file=sys.stderr) for x in q]; '
                   'sys.exit(0 if unittest.TextTestRunner(verbosity=2).run(k).wasSuccessful() else 1)" '
                   % (DENSE_CLOCK_LINE, DENSE_SWAP_LINE))
_dense_original = step_b._dense_original


def _dense_inner_script(label, suites, compiled):
    text = _dense_original(label, suites, compiled)
    if suites:
        assert text.count(_DENSE_UNITTEST) == 1, 'dense suite line'
        text = text.replace(_DENSE_UNITTEST, _DENSE_FILTERED)
    return text


dense_base.inner_script = _dense_inner_script


def dense_left_out(spec, out):
    """The ids the dense container removed (from its suites.log, sha-bound in the dense record) against the declared ones."""
    sys.path.insert(0, str(ROOT / 'harness'))
    import step_b_clock_bound as cb
    swap = _robust_swap()
    suites = json.loads(Path(spec).read_text())['engine_selfrun']['suites']
    log = out / 'dense/engine/suites.log'
    text = log.read_text() if log.is_file() else ''
    clock = sorted(re.findall('^' + re.escape(DENSE_CLOCK_LINE) + r'(\S+)$', text, re.M))
    swapped = sorted(re.findall('^' + re.escape(DENSE_SWAP_LINE) + r'(\S+)$', text, re.M))
    want_clock, want_swap = cb.expected_removed(suites), swap.expected_removed(suites)
    return dict(clock_bound=clock, clock_bound_expected=want_clock, robust_swap=swapped, robust_swap_expected=want_swap,
                ok=clock == want_clock and swapped == want_swap, clock_bound_reason=cb.REASON,
                robust_swap_reasons={x: swap.reason(x) for x in swapped if x in swap.ROBUST_SWAP or x in swap.SUPERSEDED},
                suites_log=str(log), suites_log_sha256=sha(log) if log.is_file() else None)


# ---------------------------------------------------------------- records
def expected_cases():
    """The connected cases this record holds: Step B's, C0's stage cases and held-g's probe cases. The one source for the spec's
    runner swap."""
    return sorted(step_b.expected_cases() + list(c0.CASES) + list(HELD_G_CASES))


def frozen():
    """Step B's frozen engine record, read from the parent file and pinned there."""
    pins = json.loads(PARENT_FILE.read_text())
    place = Path(pins['engine_selfrun'])
    assert sha(place / 'result.json') == pins['engine_selfrun_sha256'], 'Step B engine record changed'
    return place


def held_g_pairs(cases):
    """Every L3/L4 mode against its step/full counterpart on held-g's pair fields (held-g's pairs())."""
    rows, problems = {}, []
    for mode, against in sorted(HELD_G_PAIRS.items()):
        ours, theirs = cases.get(mode) or {}, cases.get(against) or {}
        same = {f: f in ours and f in theirs and ours[f] == theirs[f] for f in HELD_G_PAIR_FIELDS}
        ok = ours.get('verdict') == theirs.get('verdict') == 'PASS' and all(same.values())
        rows[mode] = dict(against=against, same=ok, fields=same,
                          scheduled_nonempty_sha256=ours.get('scheduled_nonempty_sha256') if ok else None)
        if not ok:
            problems.append(mode)
    return rows, problems


def run_composed(bundle, out, name):
    """The sixth container: held-g's probe cases and the composed suites; its own record in OUT/composed/result.json."""
    rc, result, cases, c, wall = filtered_container(bundle, bundle / 'native30_overlay', bundle / 'e3_builder', out, name,
                                                    HELD_G_CASES, SUITES)
    log = out / 'suites.log'
    left_out = left_out_of(log)
    rows, differ = held_g_pairs(cases)
    ok = rc == 0 and result.get('ok') is True and result.get('skips') == 0 and (result.get('tests') or 0) > 0 \
        and applied(left_out, result.get('outcomes') or {}, candidate=True) and not differ \
        and all(v.get('verdict') == 'PASS' for v in cases.values())
    record = dict(schema='alpha-next-composed-selfrun/1', verdict='PASS' if ok else 'FAIL', container_rc=rc,
                  suites={k: v for k, v in result.items() if k != 'outcomes'}, suite_names=list(SUITES),
                  outcomes_sha256=hashlib.sha256(json.dumps(result.get('outcomes'), sort_keys=True).encode()).hexdigest(),
                  left_out={x: CONTAINER_EXCLUDED[x] for x in left_out}, suites_log_sha256=sha(log) if log.is_file() else None,
                  cases={m: step_a.case_row(out, m, r) for m, r in cases.items()}, pairs=rows,
                  pair_fields=list(HELD_G_PAIR_FIELDS), differs=differ,
                  manifest=sha(bundle / 'SHA256SUMS'), image=IMAGE, network='none', gpu_device_requests=0,
                  gpu_execution=False, box_action=False, container=name, wall_s=wall, command=c,
                  log_shas={p.name: sha(p) for p in sorted(out.glob('*.log'))})
    (out / 'result.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print(record['verdict'], sha(out / 'result.json'))
    return 0 if ok else 1


PARTS = ('progress', 'dense', 'fused', 'groups', 'stage', 'composed')


def run(spec, tree, bundle, out, name):
    spec, bundle, out = Path(spec).resolve(), Path(bundle).resolve(), Path(out).resolve()
    out.mkdir()
    out.chmod(0o777)
    rcs = [held_f.run(bundle, out / 'progress', name + '-progress'),
           dense.run(spec, tree, bundle, out / 'dense', name + '-dense'),
           step_a.run_fused(bundle, out / 'fused', name + '-fused'),
           step_b.run_groups(bundle, out / 'groups', name + '-groups'),
           c0.run_stage(bundle, out / 'stage', name + '-stage'),
           run_composed(bundle, out / 'composed', name + '-composed')]
    p, d, f, g, s, n = (json.loads((out / x / 'result.json').read_text()) for x in PARTS)
    left = dense_left_out(spec, out)
    ok = rcs == [0] * len(PARTS) and all(x['verdict'] == 'PASS' for x in (p, d, f, g, s, n)) and left['ok']
    pieces = p['suites'], d['suites'], f['suites'], g['suites'], s['suites'], n['suites']
    # The dense record keeps unittest's own line in 'ran'; every other part counts in 'tests'.
    ran = re.fullmatch(r'Ran (\d+) tests? in [0-9.]+s', pieces[1].get('ran') or '')
    tests = [pieces[0].get('tests') or 0, int(ran.group(1)) if ran else 0] + [x.get('tests') or 0 for x in pieces[2:]]
    ok = ok and ran is not None
    skips = [(x.get('skips') if x.get('skips') is not None else x.get('skipped')) or 0 for x in pieces]
    connected = dict(p['cases'], **f['cases'], **g['cases'], **s['cases'], **n['cases'])
    ok = ok and sorted(connected) == expected_cases()
    record = dict(schema=SCHEMA, verdict='PASS' if ok else 'FAIL',
                  classification=None if ok else 'HARNESS-OR-ENGINE-UNCLASSIFIED', negative_control=False,
                  container_rc=max(x['container_rc'] for x in (p, d, f, g, s, n)),
                  manifest=p['manifest'], bundle_manifest=p['manifest'], bundle=p['bundle'],
                  overlay_manifest=p['overlay_manifest'], image=p['image'], network=p['network'],
                  gpu_device_requests=max(p['gpu_device_requests'], d['clock_split'].get('gpu_device_requests') or 0,
                                          f['gpu_device_requests'], g['gpu_device_requests'], s['gpu_device_requests'],
                                          n['gpu_device_requests']),
                  gpu_execution=False, box_action=False, candidate=d.get('label') or spec.name,
                  tests=sum(tests) + len(connected), skips=sum(skips), connected_checks=sum(v['checks'] for v in connected.values()),
                  suites=dict(ok=all(x.get('ok') is True for x in pieces), skips=sum(skips), tests=sum(tests),
                              **dict(zip(PARTS, pieces))),
                  cases=connected, pairs=dict(f['pairs'], **g['pairs'], **s['pairs'], **n['pairs']), pair_fields=g['pair_fields'],
                  held_g_pair_fields=n['pair_fields'],
                  probe39=d['probe39'], probe37=d['probe37'], py_compile=d['py_compile'],
                  scheduler_tick_probe=d.get('scheduler_tick_probe'), clock_split=d['clock_split'], dense_left_out=left,
                  composed_suites=list(SUITES), composed_left_out=n['left_out'],
                  **{x + '_record_sha256': sha(out / x / 'result.json') for x in PARTS},
                  spec=str(spec), spec_sha256=sha(spec), declared_limit=DECLARED_LIMIT)
    (out / 'result.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(verdict=record['verdict'], tests=record['tests'], skips=record['skips'], cases=len(connected),
                          result_sha256=sha(out / 'result.json')), sort_keys=True))
    return 0 if ok else 1


def judge(pins, outcomes, cases, left_out, place, observe=False):
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
    if not applied(left_out, outcomes, candidate=False):
        reasons.append('composed-exclusions-not-applied')
    for mode in c0.CASES + HELD_G_CASES:
        got = [(cases.get(mode) or {}).get(k) for k in ('verdict', 'error_type', 'error')]
        if got != pins['cases'].get(mode):
            reasons.append('negative-case-differs:%s:%s' % (mode, got))
    for mode in PARENT_PASS:
        part = 'groups' if mode in step_b.CASES else 'fused'
        if (cases.get(mode) or {}).get('verdict') != 'PASS':
            reasons.append('parent-case-not-pass:' + mode)
        theirs = canonical(json.loads((place / part / (mode + '.json')).read_text()))
        if not cases.get(mode) or theirs is None or canonical(cases[mode]) != theirs:
            reasons.append('negative-trace-differs-from-frozen:' + mode)
    return reasons


def negative(bundle, overlay, sources, out, name, observe=False):
    """The candidate harness over the regenerated Step B overlay and sources; exactly the pinned rows differ."""
    bundle, overlay, sources, out = bundle.resolve(), overlay.resolve(), sources.resolve(), out.resolve()
    pins = json.loads(PINS.read_text())
    if observe:
        pins = dict(pins, tests={})
    place = frozen()
    assert sha(place / 'result.json') == pins['frozen_record_sha256'], 'pins were taken against another frozen record'
    modes = PARENT_PASS + c0.CASES + HELD_G_CASES
    rc, result, cases, c, wall = filtered_container(bundle, overlay, sources, out, name, modes, NEGATIVE_SUITES, hide_candidate=True)
    outcomes = result.get('outcomes') or {}
    left_out = left_out_of(out / 'suites.log')
    reasons = judge(pins, outcomes, cases, left_out, place, observe)
    ok = not reasons and result.get('skips') == 0 and not observe
    record = dict(schema='alpha-next-negative/1', verdict='OBSERVED' if observe else 'PASS' if ok else 'FAIL', negative_control=True,
                  negative_kind='step-b-parent', classification=None if ok or observe else 'HARNESS-OR-ENGINE-UNCLASSIFIED',
                  container_rc=rc, reasons=reasons, suites={k: v for k, v in result.items() if k != 'outcomes'},
                  suite_names=list(NEGATIVE_SUITES),
                  outcomes_sha256=hashlib.sha256(json.dumps(outcomes, sort_keys=True).encode()).hexdigest(),
                  left_out={x: CONTAINER_EXCLUDED[x] for x in left_out if x in CONTAINER_EXCLUDED},
                  pinned_tests=len(pins['tests']), cases={m: step_a.case_row(out, m, r) for m, r in cases.items()},
                  bundle=str(bundle), manifest=sha(bundle / 'SHA256SUMS'), overlay=str(overlay),
                  overlay_manifest=sha(overlay / 'SHA256SUMS'), sources=str(sources),
                  pins_sha256=None if observe else sha(PINS), image=IMAGE,
                  network='none', gpu_device_requests=0, gpu_execution=False, box_action=False, container=name, wall_s=wall,
                  command=c, log_shas={p.name: sha(p) for p in sorted(out.glob('*.log'))})
    if observe:
        record['observed'] = {k: v for k, v in sorted(outcomes.items()) if v[0] != 'ok'}
    (out / 'result.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print(record['verdict'], sha(out / 'result.json'), len(reasons))
    return 0 if ok else 1


def compare(candidate, out):
    """Every Step B connected case's canonical trace equals Step B's frozen record; each C0 serial and held-g step/full mode equals
    Step B's progress mode of the same name."""
    candidate, place = Path(candidate).resolve(), frozen()
    rows, problems = {}, []
    inherited = ([('progress', m, m) for m in held_f.CASES] + [('fused', m, m) for m in step_a.CASES]
                 + [('groups', m, m) for m in step_b.CASES])
    assert sorted(m for _, m, _ in inherited) == step_b.expected_cases()
    rules = [(part, mode, part, against) for part, mode, against in inherited] + \
            [('stage', mode, 'progress', against) for mode, against in sorted(SAME_AS.items())] + \
            [('composed', mode, 'progress', against) for mode, against in sorted(HELD_G_SAME_AS.items())]
    for part, mode, their_part, against in rules:
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
    result = dict(schema='alpha-next-default-off-comparison/1', verdict='FAIL' if problems else 'PASS', differs=problems,
                  fields=held_f.CANONICAL, modes=rows, compared=sum(1 for v in rows.values() if v['compared']),
                  parent_record=str(place / 'result.json'), parent_record_sha256=sha(place / 'result.json'),
                  candidate=str(candidate), candidate_record_sha256=sha(candidate / 'result.json'),
                  meaning='with the C0 and anatomy switches unset every Step B connected case has its exact canonical trace; with C0 '
                          'installed and the arm on serial, each serial mode equals Step B\'s progress mode of the same name; each held-g '
                          'step/full probe mode equals Step B\'s progress mode of the same name')
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
    p.add_argument('--observe', action='store_true', help='negative only: record the outcomes, verdict OBSERVED, rc 1')
    p.add_argument('--compare', type=Path, help='candidate engine self-run directory')
    a = p.parse_args()
    if a.compare:
        raise SystemExit(compare(a.compare, a.out))
    if a.negative:
        raise SystemExit(negative(a.bundle, a.overlay, a.sources, a.out, a.name, a.observe))
    raise SystemExit(run(a.spec, a.tree, a.bundle, a.out, a.name))
