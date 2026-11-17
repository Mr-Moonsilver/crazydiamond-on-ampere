#!/usr/bin/env python3
"""Default-off identity of the alpha trunk merge to both parents (source level, CPU only).

H  FLASH_S5_NATIVE_PROGRESS and FLASH_S5_AUTO_ARM unset: the trunk reduces to the dense line
   (f1e5820) plus held-f's one unswitched stage, the native40 unarmed hold (f6d1db9 over held-b).
D  FLASH_S5_NATIVE_QSA_POLICY, _TRACE and _STOP_AFTER unset: the trunk reduces to held-f (c607333)
   plus the dense line's declared unswitched deltas (failure path, row1 qualification only, one
   status key that is None when off, the bounded broadcast write).

Held-f's stages are reduced with held-f's own projection code (native41_progress_freeze and
native41_autoarm_freeze), re-run on the trunk.  Where the dense line changed the same file, the
target is that stage's file composed with the dense line by a clean ``git merge-file`` over
held-b (rc 0 asserted).  A projected body must therefore equal the parent's body with the dense
change in place.  The dense direction uses DENSE_OFF below, with the same project() semantics.
At every step, each definition that differs must be projected, an import, a pinned addition or
declared here.  Anything else is refused.  Negative controls mutate the trunk text in memory and
must be refused.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import native41_progress_freeze as P  # noqa: E402
import native41_autoarm_freeze as A  # noqa: E402
from native37_robust_freeze import definitions  # noqa: E402

REV = dict(base='1ffc394783d3d67e140056868b50ca2f74c88b3b',    # held-b, the merge base
           n40='f6d1db9c90e7bffa5423945a7148a5bb800e247c',     # native40 final (unarmed hold)
           e2='818a4a7876751fcbb677f7b5994f6a1d44e2a185',      # held-e2 (auto-arm)
           heldf='c607333834cf51d366329a8ca764449d4bbcca40',   # main: held-f merged
           dense='f1e5820ec949d65d395992c1d6b3530f05aa1bf2',   # l-f3 delivery (le-fix line)
           merge='dd6d977f1a4a08a6fb24048f7ecc3cf7afdd9e48')   # the alpha trunk merge
ENGINE = 'patches/s5_engine/'
SWITCHES = dict(H=('FLASH_S5_NATIVE_PROGRESS', 'FLASH_S5_AUTO_ARM'),
                D=('FLASH_S5_NATIVE_QSA_POLICY', 'FLASH_S5_NATIVE_QSA_TRACE', 'FLASH_S5_NATIVE_QSA_STOP_AFTER'))

# ---------------------------------------------------------------- H, auto-arm stage extras
# Held-e auto-arm receipts and the two auto-arm wrappers, with FLASH_S5_AUTO_ARM unset
# (bridge._memory_receipts False; driver.auto_arm None, the startup projection drops its install).
AUTO_OFF = {
    'worker_bridge.py': {
        'NativeWorkerBridge.__init__': dict(drop={
            "self._memory_receipts = os.environ.get('FLASH_S5_AUTO_ARM', '0') == '1'": 1,
            'self._memory_cuda = None': 1, 'self.memory_rows = []': 1, 'self.memory_receipt_errors = 0': 1}),
        'NativeWorkerBridge.open': dict(off={'if self._memory_receipts:': 1}),
        'NativeWorkerBridge._release_lane': dict(off={'if self._memory_receipts:': 1}),
    },
}
# The wrappers delegate to the renamed bodies that A.default_off proves equal to the parent's
# submit_control/_status_record.
AUTO_WRAPPERS = {
    'NativeSegmentDriver.submit_control': (
        dict(drop={"policy = getattr(self, 'auto_arm', None)": 1}, off={'if policy is not None:': 1}),
        'def f(self, command):\n    return self._submit_control(command)'),
    'NativeSegmentDriver._status_record': (
        dict(drop={"policy = getattr(self, 'auto_arm', None)": 1}, off={'if policy is not None:': 1}),
        'def f(self, now=None):\n    result = self._status_record_base(now)\n    return result'),
}

# ---------------------------------------------------------------- D, the dense line off
# Policy unset -> 'current', trace unset -> NativeQsaTrace never built (operation.trace None),
# stop-after unset -> driver._native43 None.  'unswitched' rows are substitutions that hold with the
# switches on or off; their reason is in UNSWITCHED_D.
DENSE_OFF = {
    'engine_driver.py': {
        'NativeSegmentDriver.__init__': dict(
            drop={'import os': 1,
                  "stop_at = os.environ.get('FLASH_S5_NATIVE_QSA_STOP_AFTER', '0')": 1,
                  "self._native43 = None if stop_at == '0' else dict(stop_at=int(stop_at), checkpoint=None, "
                  "cancel_started=False)": 1},
            off={"if stop_at not in ('0', '101200'):": 1,
                 "if stop_at != '0' and os.environ.get('FLASH_S5_NATIVE_QSA_TRACE', '0') != '1':": 1}),
        'NativeSegmentDriver._apply_release': dict(
            off={"if self._native43 is not None and transaction.chunk_end >= self._native43['stop_at']:": 1}),
        'NativeSegmentDriver.advance': dict(
            off={"if self._native43 is not None and self._native43['checkpoint'] is not None and "
                 "(not self._native43['cancel_started']):": 1,
                 "if self._native43 is not None and self._native43['checkpoint'] is not None:": 1}),
        'NativeSegmentDriver._status_record_base': dict(
            unswitched='status-key', subs=(("'native43_boundary': copy.deepcopy(self._native43), ", ''),)),
    },
    'executor_hooks.py': {
        'collective_rpc': dict(unswitched='bounded-enqueue', subs=((
            'bounded_enqueue(executor, executor.rpc_broadcast_mq, (send_method, args, wire_kwargs, broadcast_rank))',
            'executor.rpc_broadcast_mq.enqueue((send_method, args, wire_kwargs, broadcast_rank))'),)),
    },
    'native_resources.py': {
        'build_worker_runtime': dict(subs=((
            'NativeQsaDevice(_runner_device(runner), clock=clock, env=env)',
            'NativeQsaDevice(_runner_device(runner), clock=clock)'),)),
    },
    'qsa_driver.py': {
        'QsaLayerDriver._pre': dict(drop={'operation.boundary_evidence(selected, selection=True)': 1}),
        'QsaLayerDriver._tail': dict(drop={"self.operation.boundary_evidence(phase.attn_state['attn_output'])": 1}),
    },
    'qsa_operation.py': {
        'NativeQsaDevice.__init__': dict(
            drop={'self.native_qsa_policy, self.native_qsa_trace = native_qsa_settings(env)': 1,
                  'declared = os.environ if env is None else env': 1,
                  "self.native_qsa_boundary = declared.get('FLASH_S5_NATIVE_QSA_STOP_AFTER', '0') == '101200'": 1},
            subs=(('(self, device, *, clock=time.monotonic, env=None):', '(self, device, *, clock=time.monotonic):'),)),
        'QsaPrefillOperation.__init__': dict(
            drop={"self.transport_policy = getattr(device, 'native_qsa_policy', 'current')": 1,
                  "self.trace = NativeQsaTrace(self) if getattr(device, 'native_qsa_trace', False) else None": 1},
            off={"if self.transport_policy not in ('current', 'dense'):": 1}),
        'QsaPrefillOperation.bind_plan': dict(decorators={"trace_cpu('cpu_plan_ns')": 1}),
        'QsaPrefillOperation.sparse_selection_ready': dict(decorators={"trace_cpu('cpu_plan_ns')": 1}),
        'QsaPrefillOperation.fill_sparse_bucket': dict(
            decorators={"trace_cpu('cpu_submit_ns')": 1},
            drop={"timing = self.trace.begin('copy', self.device.copy) if self.trace is not None else None": 1},
            off={'if self.trace is not None:': 2, 'if self.trace is not None and sparse.wave_copy is None:': 1}),
        'QsaPrefillOperation.sparse_selection_eligible': dict(subs=((
            "return not (self.transport_policy == 'dense' and num_pages > pages_in_arena)", 'return True'),)),
        'QsaPrefillOperation.sparse_union_scratch': dict(off={'if self.trace is not None:': 1}),
        'QsaPrefillOperation.stage_sparse_selection': dict(off={'if self.trace is not None:': 1}),
        'QsaPrefillOperation.fill_bucket': dict(
            unswitched='hold-validate', decorators={"trace_cpu('cpu_submit_ns')": 1},
            drop={"timing = self.trace.begin('copy', self.device.copy) if self.trace is not None else None": 1,
                  'if run.sparse_hold is not None:\n    run.sparse_hold.validate(run.sparse_hold.owner, '
                  'run.layer_name, run.pages)': 1},
            off={'if self.trace is not None:': 2}),
        'QsaPrefillOperation._fill_sparse_wave': dict(
            off={'if self.trace is not None and sparse.wave_rows == 0:': 1, 'if self.trace is not None:': 3}),
        'QsaPrefillOperation.after_attention_bucket': dict(off={'if self.trace is not None:': 1}),
        'QsaPrefillOperation.release': dict(off={'if self.trace is not None:': 1}),
    },
}
UNSWITCHED_D = {
    'bounded-enqueue': 'native43 le-fix c9d099a: with the fatal channel absent the hook is the stock unbounded '
                       'call; with it (overlap on) the write waits in 1 s slices and raises only once the '
                       'executor is failed or shutting down',
    'hold-validate': 'native43 bd51de8: a dense bucket re-checks an attached sparse source hold '
                     '(SourceHold.validate, CPU only, raises SourceRefused on a stale or foreign hold)',
    'status-key': "one status-record key, 'native43_boundary', that is None when FLASH_S5_NATIVE_QSA_STOP_AFTER "
                  'is unset (copy.deepcopy(None))',
}
# Differing definitions without a projection, each with its reason.  Keys are file:definition.
DECLARED = {
    'H.P': {
        'executor_hooks.py:module:__all__': 'export list (held-f worker reply hooks)',
    },
    'H.A': {
        'decode_probe.py:module:GENERATED_SOURCES': 'build-time module list: packages native_auto_arm',
        'make_s5_overlap.py:module:E3_MODULES': 'build-time module list: packages native_auto_arm',
        'make_s5_overlap.py:_s5_launcher': 'build-time launcher text: native_auto_arm exempt from the name-only '
                                           'loop, launcher byte-identical (native41_autoarm_freeze overlay check)',
    },
    'D': {
        'executor_hooks.py:module:_EXECUTOR_FORWARD': 'unswitched (le-fix c9d099a): the generated stock broadcast '
                                                      'write goes through bounded_enqueue (see bounded-enqueue)',
        'executor_hooks.py:module:__all__': 'export list (le-fix log_worker_failure)',
        'executor_hooks.py:monitor_fatal_workers': 'unswitched, worker-death path only (le-fix 77d2771): bounded '
                                                   'optional fatal metadata and one ERROR line before teardown',
        'executor_hooks.py:report_worker_failure': 'unswitched, worker-death path only (le-fix 77d2771): the fatal '
                                                   'record carries bounded cause metadata',
        'qsa_operation.py:QsaPrefillOperation.row1_output': 'row1 qualification only (02585c7): unreachable '
                                                           'without a qualification session (first statement checked)',
        'row1_oracle.py:module:__all__': 'export list (row1 dense branch, qualification only)',
    },
}
# Unnamed module statements one side added: unparsed prefix -> sha256 of the whole unparsed statement.
# H.P: held-f's four worker reply seam rows; the hooks they call reduce to the stock call (HOOKS_OFF).
UNNAMED = {'H.P': {'executor_hooks.py': {'_EXECUTOR_FORWARD += (': 'f7051e492b4b269f0e19f9bb219332fa1c12a4b8e2228bc3d94578758164ae30'}},
           'H.A': {}, 'D': {}}
# native40 (held-c line): the unarmed hold is unswitched; it is part of held-f at every switch setting.
NATIVE40 = {
    'engine_driver.py': ['NativeSegmentDriver._split_hold_timeout'],
    'scheduler_hooks.py': [
        'PrefillAdmissionController.__init__', 'PrefillAdmissionController._admit_observed',
        'PrefillAdmissionController._note_split_hold', 'PrefillAdmissionController.expired_split_hold',
        'PrefillAdmissionController.native_admission_record', 'PrefillAdmissionController.note_split_hold_timeout',
        '_defer_stock_while_split_armed', 'defer_stock_positive_frontier', 'defer_stock_prefix_hit',
        'defer_stock_running_prefill', 'flash_s5_admit', 'module:_SCHEDULER_FORWARD', 'module:_SCHEDULER_REVERSE'],
}
# Pinned additions and import changes per step: (count, sha256 of the sorted 'file:name' list).
# The record lists every name; a new or dropped one changes the pin and is refused.
PINS = {
    'H.P': dict(added=(60, '304284da3e73b1f0df7c7d87cf80f69fea8f1f0a66b954f71bc74a12606d621e'),
                imports=(5, 'c908570562b9fdd7eb5e8ea122d392ac1d0c798549926b620a1fb328a245b5b0')),
    'H.A': dict(added=(9, '10e42c8672b9b5d9c3c54c340da431fa1e550ac4e779dae32078efc652b6adfd'),
                imports=(3, '532432773b2cbedf7154d810365a47968c57c325beb8772f85bd5aaddf3f6229')),
    'D': dict(added=(32, 'f0e41899e1eafcf46a878f279b95a516f4ffebba9121248ef247bd44cac4a759'),
              imports=(11, 'f980d148b5dc9eaa6d8651b695fa84c5ec53d046b7ff252044c8876710bca72a')),
}


# ---------------------------------------------------------------- sources
def show(rev, path):
    r = subprocess.run(['git', 'show', '%s:%s' % (REV[rev], path)], cwd=ROOT, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def engine_files():
    names = set()
    for rev in REV:
        out = subprocess.check_output(['git', 'ls-tree', '--name-only', REV[rev], ENGINE], cwd=ROOT, text=True)
        names |= {n for n in out.split() if n.endswith('.py')}
    return sorted(names)


def merge3(ours, base, theirs, label):
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for name, text in (('dense', ours), ('base', base), ('side', theirs)):
            Path(d, name).write_text(text or '')
            paths.append(str(Path(d, name)))
        r = subprocess.run(['git', 'merge-file', '-p', '-L', 'dense', '-L', 'held-b', '-L', 'stage'] + paths,
                           capture_output=True, text=True)
    assert r.returncode == 0, 'composed target does not merge cleanly: ' + label
    return r.stdout


def composed(stage, files):
    """The stage's files with the dense line's changes in place (clean merge over held-b)."""
    out, merged = {}, []
    for f in files:
        s, b, d = show(stage, f), show('base', f), show('dense', f)
        if s == b:
            out[f] = d
        elif d == b:
            out[f] = s
        else:
            out[f] = merge3(d, b, s, stage + ':' + f)
            merged.append(f[len(ENGINE):])
    return out, merged


def materialize(texts, work):
    root = Path(tempfile.mkdtemp(dir=work))
    for f, t in texts.items():
        if t is not None:
            (root / f).parent.mkdir(parents=True, exist_ok=True)
            (root / f).write_text(t)
    return root


# ---------------------------------------------------------------- projections
def project_table(table, current, target):
    """P.project over each row; 'decorators' are removed (with counts) before it."""
    out = {}
    for file, rows in table.items():
        for name, spec in rows.items():
            node = P.function(current[ENGINE + file], name)
            spec = dict(spec)
            spec.pop('unswitched', None)
            for deco, count in spec.pop('decorators', {}).items():
                kept = [d for d in node.decorator_list if ast.unparse(d) != deco]
                assert len(node.decorator_list) - len(kept) == count, 'decorator count differs: %s %s' % (name, deco)
                node.decorator_list = kept
            out[file + ':' + name] = labelled(file + ':' + name, P.project, node,
                                              P.function(target[ENGINE + file], name), spec)
    return out


def labelled(label, fn, *args):
    try:
        return fn(*args)
    except AssertionError as exc:
        raise AssertionError('%s: %s' % (label, exc)) from None


def wrappers(current):
    out = {}
    for name, (spec, body) in AUTO_WRAPPERS.items():
        out['engine_driver.py:' + name] = labelled('engine_driver.py:' + name, P.project,
                                                   P.function(current[ENGINE + 'engine_driver.py'], name), body, spec)
    return out


def dense_preconditions(current):
    """The switch-off facts the D table relies on, read from the trunk text itself."""
    op = current[ENGINE + 'qsa_operation.py']
    deco = ast.unparse(next(n for n in ast.parse(op).body if isinstance(n, ast.FunctionDef) and n.name == 'trace_cpu'))
    assert 'if self.trace is None:\n                return method(self, *args, **kwargs)' in deco, 'trace_cpu off path'
    ev = P._strip_docstrings(P.function(op, 'QsaPrefillOperation.boundary_evidence'))
    assert ast.unparse(ev.body[0]).startswith('if self.trace is None or') and \
        ast.unparse(ev.body[0]).endswith('\n    return'), 'boundary_evidence off path'
    row1 = P._strip_docstrings(P.function(op, 'QsaPrefillOperation.row1_output'))
    assert ast.unparse(row1.body[0]) == 'if self.qualification is None:\n    return None', 'row1_output guard'
    settings = ast.unparse(next(n for n in ast.parse(op).body
                                if isinstance(n, ast.FunctionDef) and n.name == 'native_qsa_settings'))
    assert "policy = env.get(NATIVE_QSA_POLICY_ENV, 'current')" in settings and \
        "trace = env.get(NATIVE_QSA_TRACE_ENV, '0')" in settings, 'policy/trace defaults'
    hooks = current[ENGINE + 'executor_hooks.py']
    bounded = ast.unparse(P.function(hooks, 'bounded_enqueue'))
    assert "if getattr(executor, '_flash_s5_fatal_lock', None) is None:\n        return queue.enqueue(envelope)" \
        in bounded, 'bounded_enqueue stock path'
    return ['trace_cpu: trace None calls the method directly',
            'boundary_evidence: returns at once with trace None',
            'row1_output: returns None without a qualification session',
            "native_qsa_settings: unset gives ('current', False)",
            'bounded_enqueue: no fatal channel gives the stock queue.enqueue(envelope)']


# ---------------------------------------------------------------- coverage
def unnamed(text):
    rows = definitions(text or '')
    tree = {k: v for k, v in rows.items() if k.startswith('module:') and '#' in k}
    return sorted(tree.values())


def coverage(step, current, target, projected, files):
    """Every definition differing between current and target is explained, or the step is refused."""
    added, imports, explained = [], [], {}
    for f in files:
        a, b = current.get(f), target.get(f)
        if (a or '') == (b or ''):
            continue
        short = f[len(ENGINE):]
        assert a is not None, '%s: %s removed' % (step, short)
        if b is None:
            added.append(short + ':file sha256 ' + hashlib.sha256(a.encode()).hexdigest())
            continue
        da, db = definitions(a), definitions(b)
        for name in sorted(set(da) | set(db)):
            if da.get(name) == db.get(name):
                continue
            key = short + ':' + name
            if name.startswith('module:import ') or name.startswith('module:from '):
                imports.append(key)
            elif name.startswith('module:') and '#' in name:
                continue  # compared as a multiset below
            elif name not in db:
                added.append(key)
            elif key in projected:
                explained[key] = 'projected'
            elif key in DECLARED[step]:
                explained[key] = DECLARED[step][key]
            else:
                raise AssertionError('%s: undeclared changed definition %s' % (step, key))
        extra = list(unnamed(a))
        for row in unnamed(b):
            assert row in extra, '%s: %s lost an unnamed module statement' % (step, short)
            extra.remove(row)
        declared = UNNAMED[step].get(short, {})
        texts = [_unparsed(a, row) for row in extra]
        got = {}
        for text in texts:
            heads = [h for h in declared if text.startswith(h)]
            assert len(heads) == 1, '%s: %s undeclared unnamed statement %r' % (step, short, text[:80])
            got[heads[0]] = hashlib.sha256(text.encode()).hexdigest()
        assert set(got) == set(declared) and all(declared[h] in (None, got[h]) for h in got), \
            '%s: %s unnamed statements differ: %r' % (step, short, got)
        explained.update({short + ':' + h: 'unnamed, declared sha256 ' + s for h, s in got.items()})
    missing = [k for k in projected if k.split(' (')[0] not in explained and k.split(' (')[0] not in added]
    pins = {k: (len(v), hashlib.sha256('\n'.join(sorted(v)).encode()).hexdigest())
            for k, v in (('added', added), ('imports', imports))}
    for k, pin in pins.items():
        assert PINS[step][k] in (None, pin), '%s: %s differ from the pin %r' % (step, k, pin)
    assert not missing, '%s: projected bodies that do not differ %r' % (step, missing)
    return dict(explained=explained, added=sorted(added), imports=sorted(imports),
                pins={k: list(v) for k, v in pins.items()})


def _unparsed(text, dumped):
    for node in ast.parse(text).body:
        if ast.dump(node) == dumped:
            return ast.unparse(node)
    raise AssertionError('unnamed statement not found')


# ---------------------------------------------------------------- the two directions
def direction_h(merged, files, work):
    m_e2, merged_e2 = composed('e2', files)
    m_n40, merged_n40 = composed('n40', files)
    dense = {f: show('dense', f) for f in files}
    saved = (P.ROOT, P.show, A.ROOT, A.show)
    roots = []
    try:
        roots.append(materialize(merged, work))
        P.ROOT, P.show = roots[-1], (lambda p: m_e2[p])
        progress = P.default_off()
        roots.append(materialize(m_e2, work))
        A.ROOT, A.show = roots[-1], (lambda p: m_n40[p])
        autoarm = A.default_off()
    finally:
        P.ROOT, P.show, A.ROOT, A.show = saved
        for r in roots:
            shutil.rmtree(r, ignore_errors=True)
    proj_p = set(k.split(' (')[0] for k in progress['legacy_function_bodies'])
    extras = project_table(AUTO_OFF, m_e2, m_n40)
    extras.update(wrappers(m_e2))
    proj_a = set(autoarm['legacy_function_bodies']) | set(extras)
    # A.default_off names the current (renamed) bodies; the wrappers keep the parent's names.
    step_p = coverage('H.P', merged, m_e2, proj_p, files)
    step_a = coverage('H.A', m_e2, m_n40, proj_a, files)
    n40 = {}
    for f in files:
        if (m_n40[f] or '') != (dense[f] or ''):
            da, db = definitions(m_n40[f] or ''), definitions(dense[f] or '')
            n40[f[len(ENGINE):]] = sorted(n for n in set(da) | set(db) if da.get(n) != db.get(n))
    declared = {f: sorted(v + [n for n in definitions(m_n40[ENGINE + f]) if n not in definitions(dense[ENGINE + f])
                               and n not in v]) for f, v in NATIVE40.items()}
    assert n40 == declared, ('native40 step differs from the declaration', n40, declared)
    return dict(switches_unset=list(SWITCHES['H']), target='dense line ' + REV['dense'],
                composed_targets=dict(e2=merged_e2, n40=merged_n40),
                progress=dict(verdict=progress['verdict'], functions=progress['functions'],
                              bodies=progress['legacy_function_bodies'], coverage=step_p),
                autoarm=dict(verdict=autoarm['verdict'], functions=autoarm['functions'] + len(extras),
                             bodies=dict(autoarm['legacy_function_bodies'], **extras), coverage=step_a),
                native40_unswitched=dict(definitions=n40, reason='native40 unarmed hold (b7ab6d7 line, f6d1db9): '
                                         'unswitched; held-f carries it at every switch setting'))


def direction_d(merged, files):
    heldf = {f: show('heldf', f) for f in files}
    pre = dense_preconditions(merged)
    bodies = project_table(DENSE_OFF, merged, heldf)
    step = coverage('D', merged, heldf, set(bodies), files)
    unswitched = {f + ':' + n: UNSWITCHED_D[spec['unswitched']] for f, rows in DENSE_OFF.items()
                  for n, spec in rows.items() if 'unswitched' in spec}
    return dict(switches_unset=list(SWITCHES['D']), target='held-f ' + REV['heldf'], preconditions=pre,
                bodies=bodies, functions=len(bodies), unswitched_substitutions=unswitched, coverage=step)


# ---------------------------------------------------------------- negatives
NEGATIVES = (
    ('H', 'progress-branch-forced-on', 'executor_hooks.py', 'collective_rpc',
     'if progress is not None:', 'if progress is not None or True:'),
    ('D', 'dense-policy-return-flipped', 'qsa_operation.py', 'QsaPrefillOperation.sparse_selection_eligible',
     "return not (self.transport_policy == 'dense'", "return not (self.transport_policy == 'current'"),
    ('D', 'undeclared-change-in-dense-only-function', 'qsa_operation.py', 'QsaPrefillOperation.end_layer',
     None, None),
    ('H', 'undeclared-change-in-heldf-only-function', 'native_auto_arm.py', None, None, None),
)


def mutate(text, name, old, new):
    if old is not None:
        node = P.function(text, name)
        seg = ast.get_source_segment(text, node)
        assert seg.count(old) >= 1, 'negative anchor missing: ' + old
        return text.replace(seg, seg.replace(old, new, 1), 1)
    if name is not None:
        node = P.function(text, name)
        line = text.splitlines(keepends=True)
        body = node.body[-1]
        indent = ' ' * body.col_offset
        line.insert(body.lineno - 1, indent + 'self.stats.setdefault("negative", 0)\n')
        return ''.join(line)
    return text + '\n_NEGATIVE_SWITCH = 1\n'


def negatives(merged, files, work):
    out = {}
    for direction, label, file, name, old, new in NEGATIVES:
        texts = dict(merged)
        texts[ENGINE + file] = mutate(merged[ENGINE + file], name, old, new)
        try:
            direction_h(texts, files, work) if direction == 'H' else direction_d(texts, files)
        except AssertionError as exc:
            out[label] = 'REFUSED: ' + str(exc)[:160]
            continue
        raise AssertionError('negative control accepted: ' + label)
    return out


def run(work):
    files = engine_files()
    merged = {f: show('merge', f) for f in files}
    tree = {f: ((ROOT / f).read_text() if (ROOT / f).is_file() else None) for f in files}
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    same = [f for f in files if tree[f] != merged[f]]
    assert not same, ('engine sources in this tree differ from the merge commit', same)
    return dict(schema='alpha-trunk-default-off/1', verdict='PASS', merge=REV['merge'], parents=dict(
        dense=REV['dense'], heldf=REV['heldf']), base=REV['base'], tree_head=head,
        engine_files=len(files), progress_unset_is_dense_line=direction_h(merged, files, work),
        policy_unset_is_heldf=direction_d(merged, files), negative_controls=negatives(merged, files, work),
        level='source (patches/s5_engine); overlays are regenerated from these sources '
              '(alpha_trunk_composition.py overlay level)',
        gpu_execution=False, box_action=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True, help='scratch dir for temporary roots')
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    record = run(a.work)
    with a.out.open('x') as f:
        json.dump(record, f, sort_keys=True, indent=1)
        f.write('\n')
    print(json.dumps(dict(verdict=record['verdict'],
                          H=dict(progress=record['progress_unset_is_dense_line']['progress']['functions'],
                                 autoarm=record['progress_unset_is_dense_line']['autoarm']['functions']),
                          D=record['policy_unset_is_heldf']['functions'],
                          negatives=sorted(record['negative_controls']))))
