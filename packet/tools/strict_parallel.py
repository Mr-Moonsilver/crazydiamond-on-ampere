#!/usr/bin/env python3
"""Parallel strict: harness/selftest.py itself, unchanged, with its suite subprocesses run ahead of it in N worker processes.

What runs: the tree's own selftest.py source, exec'd in this process as __main__ (same check(), audit_suite(), summary lines, exit code,
same order, same inline cases). Only one thing differs: subprocess.run of a planned suite row returns the result a worker already
produced (or waits for it) instead of starting it there. Every other subprocess.run goes through untouched. So the log has the serial
run's lines in the serial run's order, and the build's regression_from() and tools/alpha_c1_flaky.py read it as they read a serial log.

The plan is read from selftest.py by AST (plan()): the (suite, label) table, and every subprocess.run([sys.executable, join(HERE, 'test_x.py')]).
Anything it does not recognize stops the run (rc 3); nothing is hard-coded. Every planned row must be claimed exactly once by selftest.py,
and every claim must carry the env the plan read; otherwise rc 3 even when all cases passed.

Isolation: every suite gets its own process and its own TMPDIR under one dated run folder, RUNS_ROOT/YYYYMMDD/HHMMSS-PID. Nothing is removed
(the lead, 3 Nov, on the operator's no-delete rule): the folder stays and the record reports its bytes, per suite and in total. A suite's own
tempfile cleanup is its own, as in the serial run. 25 suites on c1b do not use TMPDIR at all (they write unique mkdtemp names under the repo's
parent tmp/), which the state audit shows and does not count as sharing.

Lanes: the state audit (tools/strict_state_audit.py) decides which suites share state, and a timing-sensitive suite is declared by name. Both
sit in the lanes file's serial_lane. The serial lane runs after every parallel row has finished, one suite at a time, so a lane suite overlaps
no other suite of this run. A suite the lanes file has no current audit row for (new, or its file's sha moved) joins the serial lane: unknown
means serial, it never means parallel. The worker count is set at launch (--workers is required); this tool has no default and no clock rule.

rc: selftest.py's own exit code, or 3 for a plan/claim problem. The per-suite walls, workers, lane decisions, bytes and the load at start and
end go to --record, not the log.

usage (cwd TREE/harness, as the chain runs selftest.py; this file need not be in TREE):
  python3 -u strict_parallel.py --workers N --runs-root DIR --lanes LANES.json --record NEW.json [REVIEW_FIXTURE_DIR]
  python3 strict_parallel.py --check-plan --lanes LANES.json [--record NEW.json]    the plan and the lane decisions, nothing runs
  python3 strict_parallel.py --compare SERIAL.log PARALLEL.log --out NEW.json
  python3 strict_parallel.py --workers 4 --runs-root DIR --stub-suites ...          the inline part alone, suites answered rc 0 (the state
                                                                                    audit only: exits rc 4 whatever selftest.py says)
"""
import argparse
import ast
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback

SCHEMA = 'strict-parallel/1'
COMPARE_SCHEMA = 'strict-parallel-compare/1'
PLAN_ERROR_RC = 3
STUB_RC = 4


class PlanError(Exception):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- the plan, from selftest.py's AST
def _is_run(node):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'run'
            and isinstance(node.func.value, ast.Name) and node.func.value.id == 'subprocess')


def _static_suite(call):
    """[sys.executable, os.path.join(HERE, 'test_x.py')] exactly -> 'test_x.py', else None (a tool call of the inline part)."""
    if not call.args or not isinstance(call.args[0], ast.List) or len(call.args[0].elts) != 2:
        return None
    join = call.args[0].elts[1]
    if (isinstance(join, ast.Call) and isinstance(join.func, ast.Attribute) and join.func.attr == 'join' and len(join.args) == 2
            and isinstance(join.args[1], ast.Constant) and str(join.args[1].value).startswith('test_')):
        return join.args[1].value
    return None


def _extras(call):
    """The env rule of one subprocess.run: none, or dict(os.environ, K='v', ...) with constants; any other form stops the plan."""
    env = [k for k in call.keywords if k.arg == 'env']
    if not env:
        return {}
    value = env[0].value
    if (isinstance(value, ast.Call) and getattr(value.func, 'id', None) == 'dict' and len(value.args) == 1
            and getattr(value.args[0], 'attr', None) == 'environ' and all(isinstance(k.value, ast.Constant) for k in value.keywords)):
        return {k.arg: k.value.value for k in value.keywords}
    raise PlanError('selftest.py line %d: unrecognized env form' % call.lineno)


def plan(harness):
    """[(suite, argv, env extras)] in selftest.py's order, argv exactly what its code computes for the row."""
    harness = os.path.abspath(harness)
    tree = ast.parse(Path(harness, 'selftest.py').read_text())
    rows = []
    for node in tree.body:
        is_table = (isinstance(node, ast.For) and isinstance(node.target, ast.Tuple)
                    and [getattr(e, 'id', '') for e in node.target.elts] == ['suite', 'label'])
        if not is_table:
            for call in ast.walk(node):
                if _is_run(call):
                    suite = _static_suite(call)
                    if suite:
                        rows.append((suite, [sys.executable, os.path.join(harness, suite)], _extras(call)))
            continue
        assign = next((n for n in node.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'command'), None)
        call = next((c for n in node.body for c in ast.walk(n) if _is_run(c)), None)
        if assign is None or call is None or not isinstance(node.iter, ast.List):
            raise PlanError('selftest.py suite table: shape not recognized (line %d)' % node.lineno)
        code = compile(ast.Expression(assign.value), 'selftest.py:command', 'eval')
        names = {n.id for n in ast.walk(assign.value) if isinstance(n, ast.Name)} - {'sys', 'os', 'HERE', 'suite'}
        scope = dict(sys=sys, os=os, HERE=harness)
        # The module sets the command expression reads (ROBUST_SWAP_MODULES, ...), from selftest.py's own imports.
        for stmt in tree.body:
            if isinstance(stmt, ast.ImportFrom):
                for alias in stmt.names:
                    if (alias.asname or alias.name) in names:
                        scope[alias.asname or alias.name] = getattr(__import__(stmt.module), alias.name)
        missing = names - set(scope)
        if missing:
            raise PlanError('selftest.py command expression reads unknown names: %s' % sorted(missing))
        extras = _extras(call)
        for element in node.iter.elts:
            if not (isinstance(element, ast.Tuple) and isinstance(element.elts[0], ast.Constant)):
                raise PlanError('selftest.py suite table: row not a literal (line %d)' % element.lineno)
            suite = element.elts[0].value
            rows.append((suite, eval(code, dict(scope, suite=suite)), extras))  # noqa: S307  (selftest.py's own expression)
    seen = set()
    for suite, _, _ in rows:
        if suite in seen:
            raise PlanError('suite listed twice: ' + suite)
        seen.add(suite)
    return rows


# ---------------------------------------------------------------- lanes
def load_lanes(path, harness, rows):
    """{suite: reason} for the serial lane: declared rows, and every row without a current audit row (unknown means serial)."""
    if path is None:
        return {s: 'no lanes file: every row is unknown' for s, _, _ in rows}
    lanes = json.loads(Path(path).read_text())
    if lanes.get('schema') != 'strict-lanes/1':
        raise PlanError('lanes file schema: %r' % lanes.get('schema'))
    serial = {row['suite']: row['reason'] for row in lanes['serial_lane']}
    audited = lanes['audited']
    for suite, _, _ in rows:
        if suite not in serial and audited.get(suite) != sha(Path(harness, suite)):
            serial[suite] = 'no current audit row (unknown means serial)'
    return {s: r for s, r in serial.items() if s in {row[0] for row in rows}}


def check_required_dirs(path, harness):
    """A directory the lanes file names (relative to the harness) must exist before any row starts. Rows mkdtemp in it, some row makes it, and the serial order hides that: a parallel start does not."""
    if path is None:
        return []
    required = json.loads(Path(path).read_text()).get('required_dirs') or []
    missing = [d for d in required if not Path(harness, d).is_dir()]
    if missing:
        raise PlanError('required directory missing, make it before strict (lanes file required_dirs): %s' % ', '.join(os.path.normpath(os.path.join(harness, d)) for d in missing))
    return required


# ---------------------------------------------------------------- run
class Pool:
    """N workers over the free rows; when they have all finished, one thread runs the lane rows one at a time."""

    def __init__(self, rows, lane, workers, root, cwd, real_run):
        self.real_run, self.root, self.cwd, self.t0 = real_run, root, cwd, time.monotonic()
        self.futures = {row[0]: cf.Future() for row in rows}
        self.rows, self.lane_names, self.stats = {row[0]: row for row in rows}, set(lane), {}
        self.free = queue.SimpleQueue()
        for row in rows:
            if row[0] not in self.lane_names:
                self.free.put(row[0])
        self.lane_order = [row[0] for row in rows if row[0] in self.lane_names]
        self.threads = [threading.Thread(target=self.work, args=(i,), daemon=True) for i in range(workers)]
        self.lane_thread = threading.Thread(target=self.lane_work, daemon=True)
        for thread in self.threads:
            thread.start()
        self.lane_thread.start()

    def work(self, index):
        while True:
            try:
                suite = self.free.get_nowait()
            except queue.Empty:
                return
            self.one(suite, index)

    def lane_work(self):
        for thread in self.threads:
            thread.join()
        for suite in self.lane_order:
            self.one(suite, 'lane')

    def one(self, suite, worker):
        future = self.futures[suite]
        tmp = None
        try:
            tmp = tempfile.mkdtemp(prefix=suite[:-3] + '.', dir=self.root)
            _, argv, extras = self.rows[suite]
            start = time.monotonic()
            result = self.real_run(argv, capture_output=True, text=True, cwd=self.cwd, env=dict(os.environ, TMPDIR=tmp, **extras))
            self.stats[suite] = dict(worker=worker, start_s=round(start - self.t0, 2), wall_s=round(time.monotonic() - start, 2),
                                     rc=result.returncode, lane=suite in self.lane_names, tmp=tmp, tmp_bytes=tree_bytes(tmp))
            future.set_result(result)
        except BaseException as error:  # a worker's own failure must reach the claim, never hang it
            self.stats[suite] = dict(worker=worker, error=repr(error)[:300], tmp=tmp)
            future.set_exception(error)

    def join(self):
        self.lane_thread.join()


def tree_bytes(root):
    total = 0
    for folder, _, files in os.walk(root):
        for name in files:
            try:
                total += os.lstat(os.path.join(folder, name)).st_size
            except OSError:
                pass
    return total


class Stub:
    """--stub-suites: the state audit traces the inline part alone. Every claim answers rc 0 at once; nothing runs."""

    def __init__(self, rows):
        self.futures = {}
        for suite, argv, _ in rows:
            future = cf.Future()
            future.set_result(subprocess.CompletedProcess(argv, 0, 'Ran 0 tests in 0.000s\n\nOK\n', ''))
            self.futures[suite] = future
        self.stats = {}

    def join(self):
        pass


def run_folder(runs_root):
    """A new dated folder for this run's temp dirs: RUNS_ROOT/YYYYMMDD/HHMMSS-PID. A naming date, never a window."""
    now = time.localtime()
    folder = Path(runs_root).resolve() / time.strftime('%Y%m%d', now) / ('%s-%d' % (time.strftime('%H%M%S', now), os.getpid()))
    folder.mkdir(parents=True)
    return str(folder)


def run_selftest(harness, args, workers, lanes_path, record_path, runs_root, stub=False):
    harness = os.path.abspath(harness)
    selftest = os.path.join(harness, 'selftest.py')
    os.chdir(harness)
    sys.path[0] = harness            # as `python3 selftest.py` has it: this tool's directory is not on the path
    started, load_start = time.time(), os.getloadavg()
    rows = plan(harness)
    lane = load_lanes(lanes_path, harness, rows)
    check_required_dirs(lanes_path, harness)
    root = run_folder(runs_root)
    real_run = subprocess.run
    pool = Stub(rows) if stub else Pool(rows, lane, workers, root, harness, real_run)
    table = {tuple(argv): (suite, extras) for suite, argv, extras in rows}
    claims, problems = {}, []

    def claiming_run(args_, *a, **k):
        key = tuple(str(x) for x in args_) if isinstance(args_, (list, tuple)) else None
        if key not in table:
            return real_run(args_, *a, **k)
        suite, extras = table[key]
        claims[suite] = claims.get(suite, 0) + 1
        want_env = dict(os.environ, **extras) if extras else None
        if a or set(k) - {'capture_output', 'text', 'env'} or not k.get('capture_output') or not k.get('text') or k.get('env') != want_env:
            problems.append('claim-differs-from-plan:' + suite)
        return pool.futures[suite].result()

    subprocess.run = claiming_run
    sys.argv = [selftest] + list(args)
    scope = dict(__name__='__main__', __file__=selftest, __builtins__=__builtins__, __doc__=None)
    rc = 1
    try:
        exec(compile(Path(selftest).read_text(), selftest, 'exec'), scope)  # noqa: S102  (the tree's own selftest.py)
    except SystemExit as stop:
        rc = stop.code if isinstance(stop.code, int) else (0 if stop.code is None else 1)
    except BaseException:  # selftest.py died, or a worker's failure reached a claim: never a clean exit
        traceback.print_exc()
        problems.append('selftest-raised')
    finally:
        subprocess.run = real_run
    for suite, _, _ in rows:
        if claims.get(suite) != 1:
            problems.append('claims-not-one:%s:%s' % (suite, claims.get(suite)))
    pool.join()
    if record_path:
        record = dict(schema=SCHEMA, selftest_sha256=sha(selftest), head=_head(harness), workers=workers, wall_s=round(time.time() - started, 2),
                      rc=rc, problems=problems, suites=len(rows), serial_lane=lane, nice=os.nice(0), stub=stub,
                      lanes_sha256=sha(lanes_path) if lanes_path else None, loadavg_start=load_start, loadavg_end=os.getloadavg(),
                      run_folder=root, run_folder_bytes=tree_bytes(root), stats=pool.stats,
                      suite_wall_sum_s=round(sum(s.get('wall_s', 0) for s in pool.stats.values()), 1))
        with Path(record_path).open('x') as f:
            json.dump(record, f, sort_keys=True, indent=1)
            f.write('\n')
    if problems:
        print('strict-parallel: plan/claim problems: ' + json.dumps(problems), file=sys.stderr, flush=True)
        return PLAN_ERROR_RC
    return STUB_RC if stub else rc


def check_plan(harness, lanes_path, record_path):
    """Phase A: read the plan and decide the lanes; nothing runs. rc 3 on any unknown shape."""
    harness = os.path.abspath(harness)
    os.chdir(harness)
    sys.path[0] = harness
    rows = plan(harness)
    lane = load_lanes(lanes_path, harness, rows)
    required = check_required_dirs(lanes_path, harness)
    result = dict(schema='strict-parallel-plan/1', required_dirs=required, selftest_sha256=sha(os.path.join(harness, 'selftest.py')), suites=len(rows),
                  lanes_sha256=sha(lanes_path) if lanes_path else None, serial_lane=lane)
    if record_path:
        with Path(record_path).open('x') as f:
            json.dump(result, f, sort_keys=True, indent=1)
            f.write('\n')
    print('strict plan: %d suites, serial lane %d' % (len(rows), len(lane)))
    return 0


def _head(harness):
    done = subprocess.run(['git', '-C', harness, 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else None


# ---------------------------------------------------------------- compare two logs by case
CASE = re.compile(r'^(ok  |FAIL) (.*): got (.*), want (.*)$')
SUITE = re.compile(r'^selftest suite: (\{.*\})\s*$')
UNIT = re.compile(r'^(FAIL|ERROR): (\S+) \(([\w.]+)\)')
TAIL = re.compile(r'^(selftest coverage: .*|selftest: \d+/\d+ cases as required|rc=\d+)$')


def parse(text):
    """Ordered case rows, suite rows (without the output shas: those carry timings), unittest failure ids, and the summary lines."""
    cases, suites, units, tail = [], [], [], []
    for line in text.splitlines():
        m = CASE.match(line)
        if m:
            cases.append(dict(outcome=m.group(1).strip(), name=m.group(2), got=m.group(3), want=m.group(4)))
            continue
        m = SUITE.match(line)
        if m:
            row = json.loads(m.group(1))
            suites.append({k: row[k] for k in ('suite', 'returncode', 'skipped', 'tests_run')})
            continue
        m = UNIT.match(line)
        if m:
            units.append(m.group(1) + ' ' + m.group(3))
            continue
        if TAIL.match(line):
            tail.append(line)
    return dict(cases=cases, suites=suites, units=units, tail=tail)


def compare(serial, parallel, out):
    a, b = parse(Path(serial).read_text(errors='replace')), parse(Path(parallel).read_text(errors='replace'))
    differs = {k: [dict(serial=x, parallel=y) for x, y in zip(a[k], b[k]) if x != y][:20] for k in a}
    same = {k: a[k] == b[k] for k in a}
    result = dict(schema=COMPARE_SCHEMA, verdict='PASS' if all(same.values()) and a['cases'] and a['suites'] else 'FAIL', same=same,
                  counts=dict(serial={k: len(a[k]) for k in a}, parallel={k: len(b[k]) for k in b}), differs=differs,
                  serial_log=str(serial), serial_sha256=sha(serial), parallel_log=str(parallel), parallel_sha256=sha(parallel),
                  outcomes=dict(serial_fail=sum(c['outcome'] == 'FAIL' for c in a['cases']),
                                parallel_fail=sum(c['outcome'] == 'FAIL' for c in b['cases'])),
                  compared='every case in order (name, outcome, got, want); every suite row (suite, returncode, skipped, tests_run); every '
                           'unittest FAIL:/ERROR: id; the coverage, count and rc lines. Not compared: the suites\' output shas (they '
                           'carry timings) and the failed-suite dump text.')
    with Path(out).open('x') as f:
        json.dump(result, f, sort_keys=True, indent=1)
        f.write('\n')
    print(result['verdict'], json.dumps(result['counts'], sort_keys=True))
    return 0 if result['verdict'] == 'PASS' else 1


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--workers', type=int)
    p.add_argument('--runs-root')
    p.add_argument('--lanes')
    p.add_argument('--record')
    p.add_argument('--compare', nargs=2)
    p.add_argument('--check-plan', action='store_true')
    p.add_argument('--stub-suites', action='store_true')
    p.add_argument('--out')
    p.add_argument('--harness', default='.')
    p.add_argument('review', nargs='*')
    a = p.parse_args()
    if a.compare:
        if not a.out:
            p.error('--compare needs --out')
        return compare(a.compare[0], a.compare[1], a.out)
    lanes, record = a.lanes and os.path.abspath(a.lanes), a.record and os.path.abspath(a.record)
    try:
        if a.check_plan:
            return check_plan(a.harness, lanes, record)
        if a.workers is None or a.workers < 1 or not a.runs_root:
            p.error('--workers (at least 1) and --runs-root are required: both are set at launch')
        return run_selftest(a.harness, a.review, a.workers, lanes, record, os.path.abspath(a.runs_root), a.stub_suites)
    except PlanError as error:
        print('strict-parallel: ' + str(error), file=sys.stderr, flush=True)
        return PLAN_ERROR_RC


if __name__ == '__main__':
    raise SystemExit(main())
