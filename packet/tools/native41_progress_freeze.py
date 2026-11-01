#!/usr/bin/env python3
"""Fail-closed path and AST freeze of held-f vs held-e2 (818a4a7), source and generated overlay.

Default-off (``FLASH_S5_NATIVE_PROGRESS`` unset): every pre-existing function
the candidate changes projects back to held-e's exact AST once the declared
progress-only statements are removed, the generated executor is held-e2's
byte for byte once the four held-f worker-seam rows are reverted, and the new
worker hooks reduce to the stock call they wrap plus a declared per-worker
reply lock.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from native37_robust_freeze import prove
from build_native30_held import verify
ROOT = Path(__file__).resolve().parents[1]
PARENT = '818a4a7876751fcbb677f7b5994f6a1d44e2a185'
DECLARATION = ROOT / 'tools/native41_progress_declarations.json'
RUNTIME = {'e3/engine_driver.py', 'e3/executor_hooks.py', 'e3/native_measurement.py', 'e3/native_overlap_control.py',
           'e3/native_overlap_session.py', 'e3/probe_control.py', 'e3/reply_transport.py', 'e3/s5_startup.py',
           # Generated from executor_hooks._EXECUTOR_FORWARD (the held-f worker reply seam rows).
           'e3_sources/executor.py'}
ADDED = set()
META = {'SHA256SUMS', 'S5-OVERLAP-GENERATION.json', 'e3/native-implementation-closure.json',
        'e3/native-overlap-readiness.json', 'e3/decode-probe-evidence.json'}


def show(path):
    return subprocess.check_output(['git', 'show', PARENT + ':' + path], cwd=ROOT, text=True)


def paths():
    return subprocess.check_output(['git', 'diff', '--name-status', PARENT, '--', 'patches/s5_engine', 'harness', 'tools'],
                                   cwd=ROOT, text=True).splitlines()


def check_paths(actual, d):
    assert set(actual) == set(d['paths']), sorted(set(actual) ^ set(d['paths']))


def source_level():
    d = json.loads(DECLARATION.read_text())
    rows = paths()
    check_paths(rows, d)
    all_rows = subprocess.check_output(['git', 'diff', '--name-status', PARENT], cwd=ROOT, text=True).splitlines()
    ancillary = set(all_rows) - set(rows)
    assert ancillary <= set(d['ancillary_paths']), sorted(ancillary - set(d['ancillary_paths']))
    modified = {row.split('\t', 1)[1] for row in rows if row.startswith('M\t') and row.endswith('.py')}
    assert modified == set(d['definitions'])
    defs = {p: prove(show(p), (ROOT / p).read_text(), p, names) for p, names in d['definitions'].items()}
    compiled = []
    for row in rows:
        name = row.split('\t', 1)[1]
        if name.endswith('.py'):
            compile((ROOT / name).read_bytes(), name, 'exec')
            compiled.append(name)
    return dict(default_off=default_off(), parent=PARENT, paths=rows, ancillary_paths=sorted(ancillary),
                definitions=defs, compiled=compiled, skips=0,
                declaration_sha256=hashlib.sha256(DECLARATION.read_bytes()).hexdigest())


def overlay_level(parent, candidate):
    before, after = verify(parent), verify(candidate)
    assert set(after) == set(before) | ADDED, 'overlay file set changed'
    changed = {n for n in before if before[n] != after[n]} | {'SHA256SUMS'} | ADDED
    assert changed == RUNTIME | META | ADDED, sorted(changed ^ (RUNTIME | META | ADDED))
    d = json.loads(DECLARATION.read_text())['overlay_definitions']
    assert set(d) == RUNTIME
    defs = {p: prove((parent / p).read_text(), (candidate / p).read_text(), p, d[p]) for p in sorted(d)}
    assert (candidate / 'run-flash-s5-test.sh').read_bytes() == (parent / 'run-flash-s5-test.sh').read_bytes(), 'launcher changed'
    native = [p for p in before if p.endswith('.so') or '/native30_plugins/' in p or p == 'qsa.py']
    assert native and all(before[n] == after[n] for n in native)
    return dict(parent=str(parent), changed=sorted(changed), definitions=defs, executor_off=executor_off(parent, candidate),
                model_and_plugin_byte_identical=True, native_libraries_byte_identical=True)


def executor_off(parent, candidate):
    """The generated WorkerProc is held-e2's exactly once the held-f seam rows are reverted."""
    import sys
    sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
    import executor_hooks as eh
    rows = [row for row in eh._EXECUTOR_FORWARD if '_flash_s5_rpc' in row[1] or 'worker_enqueue_reply' in row[1]]
    assert len(rows) == 4, len(rows)
    text = (candidate / 'e3_sources/executor.py').read_text()
    for old, new in rows:
        assert text.count(new) == 1, new
        text = text.replace(new, old)
    assert text == (parent / 'e3_sources/executor.py').read_text(), 'generated executor differs beyond the seam rows'
    return dict(verdict='PASS', reverted_rows=4, parent_sha256=hashlib.sha256(text.encode()).hexdigest())


def negative_controls():
    d = json.loads(DECLARATION.read_text())
    p = 'patches/s5_engine/reply_transport.py'

    def refused(fn):
        try:
            fn()
        except AssertionError:
            return 'REFUSED'
        raise AssertionError('undeclared change accepted')
    old = function(show('patches/s5_engine/engine_driver.py'), 'NativeSegmentDriver._reset')
    new = function((ROOT / 'patches/s5_engine/engine_driver.py').read_text(), 'NativeSegmentDriver._reset')
    new.body.insert(0, ast.parse('self.phase = None').body[0])
    return {'undeclared-function': refused(lambda: prove(show(p), (ROOT / p).read_text() + '\ndef held_f_undeclared():\n pass\n',
                                                          p, d['definitions'][p])),
            'undeclared-path': refused(lambda: check_paths(paths() + ['A\tharness/held_f_undeclared.py'], d)),
            'undeclared-overlay': refused(lambda: prove('x=1\n', 'x=2\n', 'core/scheduler.py', [])),
            'undeclared-default-off-statement': refused(lambda: project(new, old, PROJECTIONS['engine_driver.py']['NativeSegmentDriver._reset']))}


def function(text, name):
    tree = ast.parse(text)
    if '.' in name:
        cls, name = name.split('.')
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return copy.deepcopy(next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name))


def _strip_docstrings(node):
    for n in ast.walk(node):
        body = getattr(n, 'body', None)
        if (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and body
                and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            n.body = body[1:] or [ast.Pass()]
    return node


def _text(node):
    node = _strip_docstrings(copy.deepcopy(node))
    node.name = 'f'
    return ast.unparse(node)


# Default-off projections, FLASH_S5_NATIVE_PROGRESS unset (so driver.progress,
# StockLocalFuture._progress and executor._flash_s5_progress_recording are None,
# direct_replies/direct are False and progress_mode(env) returns None):
#   off:  an ``if`` whose test is false then; it becomes its else branch (usually nothing).
#   on:   an ``if`` whose test is true then; it becomes its body.
#   drop: a statement binding a progress-only local, or unreachable after an ``on`` return.
#   subs: exact text substitutions on the unparsed result, each matching once.
# Each listed statement must match the declared number of times; the result
# must be the parent function's exact unparsed AST (docstrings ignored).
PROJECTIONS = {
    'engine_driver.py': {
        'NativeSegmentDriver.__init__': dict(drop={"self.progress = None": 1, "self._progress_origin = 'tick'": 1,
                                                   "self._in_wait = False": 1}),
        'NativeSegmentDriver._begin': dict(off={"if self.progress is not None and self._progress_owned():": 1}),
        'NativeSegmentDriver._drain_observed': dict(
            drop={"progress = self.progress": 1,
                  "row = None if progress is None else progress.native.pop(sequence, None)": 1,
                  "registered = None if row is None else getattr(call, 'registered_s', None)": 1},
            off={"if row is not None:": 2}),
        'NativeSegmentDriver._reset': dict(off={"if self.progress is not None:": 1}),
        'NativeSegmentDriver._status_record': dict(off={"if self.progress is not None:": 1}),
        'NativeSegmentDriver._submit_control': dict(off={"if action == 'native_progress' and self.progress is not None:": 1}),
        'NativeSegmentDriver._submit_observed': dict(
            drop={"progress = self.progress if self.progress is not None and self.progress.recording else None": 1,
                  "direct = progress is not None and progress.burst() and (kind in progress.DIRECT_KINDS)": 1,
                  "extra = {'direct': True} if direct else {}": 1,
                  "checked = envelope": 1,
                  "return sender(NativeProgress.with_meta(envelope, progress.meta(_sequence, direct)))": 1},
            off={"if direct and envelope[0] not in PROGRESS_DIRECT_METHODS:": 1, "if progress is not None:": 2},
            on={"if progress is None:": 1},
            subs=(("return check(checked)", "return check(envelope)"),
                  (", precheck=precheck if callable(check) else None, **extra)",
                   ", precheck=precheck if callable(check) else None)"))),
    },
    'executor_hooks.py': {
        'StockLocalFuture.__init__': dict(drop={"self._progress = None": 1}),
        'StockLocalFuture.take': dict(drop={"progress = self._progress": 1, "registered = None": 1},
                                      off={"if progress is not None:": 2}),
        'collective_rpc': dict(drop={"progress = getattr(executor, '_flash_s5_progress_recording', None)": 1,
                                     "wire_kwargs = kwargs": 1},
                               off={"if progress is not None:": 1},
                               subs=(("(send_method, args, wire_kwargs, broadcast_rank)",
                                      "(send_method, args, kwargs, broadcast_rank)"),)),
        'install': dict(subs=((", capacity=None, direct_replies=False):", ", capacity=None):"),
                              (", capacity=capacity, direct_replies=direct_replies)", ", capacity=capacity)"))),
        '_install_owned': dict(drop={"extra = {'direct_replies': True} if direct_replies else {}": 1},
                               subs=((", capacity=None, direct_replies=False):", ", capacity=None):"),
                                     (", clock=clock, **extra)", ", clock=clock)"))),
    },
    'reply_transport.py': {
        'ReplyTransport.__init__': dict(
            drop={"self._direct = {r: deque() for r in queues} if direct_replies else None": 1,
                  "loop = self._receive_loop if self._direct is None else self._receive_demux_loop": 1},
            off={"if direct_replies not in (False, True):": 1},
            subs=((", clock=time.monotonic, direct_replies=False):", ", clock=time.monotonic):"),
                  ("target=loop,", "target=self._receive_loop,"))),
        'ReplyTransport.submit': dict(
            drop={"fifo = self._direct if direct else self._todo": 1},
            off={"if direct is not False and (direct is not True or self._direct is None):": 1},
            subs=((", precheck=None, direct=False):", ", precheck=None):"),
                  ("fifo[rank].append(", "self._todo[rank].append("))),
    },
    's5_startup.py': {
        'install_engine': dict(
            drop={"from native_measurement import progress_mode": 1,
                  "try:\n    progress = progress_mode(env)\nexcept ValueError as exc:\n    raise StartupError(str(exc)) from None": 1,
                  "extra = {'direct_replies': True} if progress == 'burst' else {}": 1},
            off={"if progress is not None:": 1},
            subs=((", clock=clock, **extra)", ", clock=clock)"),)),
    },
    'native_overlap_control.py': {
        # _progress_burst() is False with FLASH_S5_NATIVE_PROGRESS unset: not (A and B or False) == (not A or not B).
        'NativeControlClient._call': dict(subs=((
            "if not (action == 'release_auto_arm' and os.environ.get('FLASH_S5_AUTO_ARM', '0') == '1' or "
            "(action == 'native_progress' and _progress_burst())):",
            "if action != 'release_auto_arm' or os.environ.get('FLASH_S5_AUTO_ARM', '0') != '1':"),)),
    },
    'native_overlap_session.py': {
        # The parent has no _action_native_progress, so its getattr gives None; the
        # candidate's getattr finds the method and this block sets None when not burst.
        'NativeOverlapSession.execute': dict(drop={
            "if action == 'native_progress' and (not (os.environ.get('FLASH_S5_NATIVE_PROGRESS', '') == 'burst' "
            "and os.environ.get('FLASH_S5_AUTO_ARM', '0') == '1')):": 1}),
    },
    'probe_control.py': {
        'attach_native_router': dict(off={"if auto_actions and env.get('FLASH_S5_NATIVE_PROGRESS', '') == 'burst':": 1}),
    },
}

# The new worker reply hooks with no envelope meta (rpc None) and progress unset
# (_worker_progress returns None): the stock call each wraps, plus the declared
# per-worker reply lock (uncontended: no direct writer exists then).
HOOKS_OFF = {
    'worker_rpc_begin': (dict(on={"if type(kwargs) is not dict or PROGRESS_META_KEY not in kwargs:": 1},
                              drop={"sequence, direct = decode_progress_meta(kwargs.pop(PROGRESS_META_KEY))": 1,
                                    "name = method if type(method) is str else 'callable'": 1,
                                    "state = _worker_progress(proc.rank)": 1,
                                    "return [sequence, name, time.monotonic_ns(), None, direct]": 1},
                              off={"if direct and name not in PROGRESS_DIRECT_METHODS:": 1,
                                   "if state is not None and name == 'flash_s5_release':": 1}),
                         "def f(proc, method, kwargs):\n    return None"),
    'worker_rpc_returned': (dict(on={"if rpc is None:": 1},
                                 drop={"rpc[3] = time.monotonic_ns()": 1, "state = _worker_progress(proc.rank)": 1},
                                 off={"if output_rank is None or proc.rank == output_rank:": 1, "if state is not None:": 1}),
                            "def f(proc, rpc, output_rank):\n    return"),
    'worker_handle_output': (dict(off={"if rpc is not None and rpc[4]:": 1, "if state is not None:": 1},
                                  drop={"state = _worker_progress(proc.rank)": 1}),
                             "def f(proc, rpc, output):\n    proc.handle_output(output)"),
    'worker_enqueue_reply': (dict(on={"if state is None or getattr(_DIRECT_WRITE, 'active', False):": 1},
                                  drop={"state = _worker_progress(proc.rank)": 1,
                                        "try:\n    rpc = state[1].popleft()\nexcept IndexError:\n    return": 1,
                                        "if rpc is not None:\n    state[0].record('r', rpc[0], rpc[1], rpc[2], rpc[3], time.monotonic_ns(), 0, 1)": 1}),
                             "def f(proc, result):\n    with _reply_lock(proc):\n        if (response_mq := proc.worker_response_mq) is not None:\n"
                             "            response_mq.enqueue(result)\n    return"),
}


def project(new, old, spec):
    """Apply one declared projection to ``new``; its text must equal ``old``'s."""
    counts = {}
    for kind in ('drop', 'off', 'on'):
        for head in spec.get(kind, {}):
            counts[(kind, head)] = 0

    def match(stmt):
        text = ast.unparse(stmt)
        first = text.splitlines()[0]
        for kind in ('drop', 'off', 'on'):
            table = spec.get(kind, {})
            if text in table:
                return kind, text
            # An ``if`` is named by its test line; every other statement by its full text.
            if isinstance(stmt, ast.If) and first in table:
                return kind, first
        return None

    def body(stmts):
        out = []
        for stmt in stmts:
            hit = match(stmt)
            if hit is not None:
                counts[hit] += 1
                kind = hit[0]
                if kind == 'off':
                    assert isinstance(stmt, ast.If), hit
                    out += body(stmt.orelse)
                elif kind == 'on':
                    assert isinstance(stmt, ast.If) and not stmt.orelse, hit
                    out += body(stmt.body)
                continue
            for field in ('body', 'orelse', 'finalbody'):
                if isinstance(getattr(stmt, field, None), list) and getattr(stmt, field):
                    setattr(stmt, field, body(getattr(stmt, field)) or [ast.Pass()])
            for handler in getattr(stmt, 'handlers', ()):
                handler.body = body(handler.body) or [ast.Pass()]
            out.append(stmt)
        return out
    node = _strip_docstrings(copy.deepcopy(new))
    node.body = body(node.body)
    for (kind, head), seen in counts.items():
        assert seen == spec[kind][head], 'default-off statement count differs: %s %r %d' % (kind, head, seen)
    text = _text(node)
    for a, b in spec.get('subs', ()):
        assert text.count(a) == 1, 'default-off substitution not unique: %r' % a
        text = text.replace(a, b)
    target = old if isinstance(old, str) else _text(old)
    assert text == target, 'default-off body differs'
    return hashlib.sha256(target.encode()).hexdigest()


def default_off():
    result = {}
    for file, specs in PROJECTIONS.items():
        old = show('patches/s5_engine/' + file)
        new = (ROOT / 'patches/s5_engine' / file).read_text()
        for name, spec in specs.items():
            result[file + ':' + name] = project(function(new, name), function(old, name), spec)
    hooks = (ROOT / 'patches/s5_engine/executor_hooks.py').read_text()
    old_hooks = show('patches/s5_engine/executor_hooks.py')
    # StockLocalFuture.result: progress None returns _stock_result, which is held-e's result body.
    result_fn = _strip_docstrings(function(hooks, 'StockLocalFuture.result'))
    assert [ast.unparse(s) for s in result_fn.body[:2]] == [
        'progress = self._progress', 'if progress is None:\n    return self._stock_result(timeout)'], 'result off branch changed'
    assert _text(function(hooks, 'StockLocalFuture._stock_result')) == _text(function(old_hooks, 'StockLocalFuture.result')), \
        'stock result body changed'
    result['executor_hooks.py:StockLocalFuture.result'] = hashlib.sha256(
        _text(function(old_hooks, 'StockLocalFuture.result')).encode()).hexdigest()
    for name, (spec, target) in HOOKS_OFF.items():
        result['executor_hooks.py:' + name + ' (off path)'] = project(function(hooks, name), target, spec)
    # _worker_progress gives None when progress_mode(os.environ) is None (one env read per process).
    worker = ast.unparse(function(hooks, '_worker_progress'))
    assert "if mode is not None else False" in worker and 'return state or None' in worker
    return dict(verdict='PASS', legacy_function_bodies=result, functions=len(result),
                meaning='source projection with FLASH_S5_NATIVE_PROGRESS unset: old receive loop and wait body '
                        'preserved (AST-identical), as are the submit and router bodies; new wrapper and shared '
                        'worker-output lock overhead declared (the worker reply hooks reduce to the stock call plus '
                        'the per-worker reply lock); canonical connected traces are checked separately against held-e2')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path)
    p.add_argument('--overlay', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    r = dict(verdict='PASS', source=source_level(), negative_controls=negative_controls())
    if a.overlay:
        r['overlay'] = overlay_level(a.parent, a.overlay)
    with a.out.open('x') as f:
        json.dump(r, f, sort_keys=True, indent=2)
        f.write('\n')
