#!/usr/bin/env python3
"""Fail-closed default-off freeze of Step B's engine against the final claude/step-a engine, source and generated overlay.

tools/step_a_freeze.py's machinery (project, removals, class_fields, _units, _references), its text kept, over Step B's declarations.
With ``FLASH_S5_LAYER_GROUPS`` unset, so ``driver.layer_groups`` is None, ``_tx_group`` 1, every OPEN and descriptor carries layer_group 1,
``dispatcher.layer_group`` 1 and no plan entry has a ``layer_end`` or ``group`` key:

* every pre-existing function Step B changes projects back to Step A's exact AST once the declared statements are removed; each removal
  is named and counted, and an undeclared statement makes the proof refuse (negative controls);
* every new definition is listed in NEW_HELPERS with its off form; the one the default-off path can call reduces to its value, and the
  rest are referenced only from code that cannot run default-off (reach);
* class fields, class rows, module statements and what cannot be projected are named in CLASS_FIELDS, CLASS_ROWS, MODULE_STATEMENTS and
  NOT_PROJECTED;
* the overlay holds each module exactly as make_s5_overlap.package(committed source) writes it, the parent overlay's as package(Step A
  source), and the generated vllm sources, launcher, plugins and native libraries are byte-identical.

This holds with FLASH_S5_QSA_FUSED set or unset: Step A's fused path is the parent's code here, and it stays unprojected.
Step B with the switch on is not claimed equal to Step A here: the suite, the connected probe, the VM and the box compare those.
CPU only, no box action.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from native37_robust_freeze import changed, prove  # noqa: E402
from build_native30_held import verify  # noqa: E402
import step_a_freeze as base  # noqa: E402
from step_a_freeze import _packaged, _references, _text, _units, function, project, removals  # noqa: E402

PARENT = 'a6426e2'
DECLARATION = ROOT / 'tools/step_b_declarations.json'
ENGINE = ('engine_driver.py', 'lane_dispatch.py', 'native_measurement.py', 'native_overlap_control.py', 'native_overlap_session.py',
          'probe_control.py', 's5_startup.py', 'wire_v1.py', 'worker_bridge.py')
RUNTIME = {'e3/' + name for name in ENGINE}
META = base.META
REVIEW = base.REVIEW


def show(path, commit=PARENT):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=ROOT, text=True)


def paths():
    return subprocess.check_output(['git', 'diff', '--name-status', PARENT, '--', 'patches/s5_engine'], cwd=ROOT, text=True).splitlines()


def check_paths(actual, d):
    assert set(actual) == set(d['paths']), sorted(set(actual) ^ set(d['paths']))


def declaration_draft():
    """The declaration the tree implies; a reviewer compares it to the committed one."""
    rows = paths()
    definitions = {}
    for row in rows:
        status, name = row.split('\t', 1)
        if status == 'M' and name.endswith('.py'):
            definitions[name] = changed(show(name), (ROOT / name).read_text())
    overlay = {'e3/' + name: changed(_packaged(show('patches/s5_engine/' + name)),
                                     _packaged((ROOT / 'patches/s5_engine' / name).read_text())) for name in ENGINE}
    return dict(schema='step-b-freeze-declaration/1', parent=PARENT, paths=sorted(rows), definitions=definitions, overlay_definitions=overlay)


def new_definitions(d):
    out = []
    for path, names in d['definitions'].items():
        file = path.rsplit('/', 1)[1]
        old = {q for q, _ in _units(show(path), file)}
        for name in names:
            if name.startswith('module:'):
                continue
            if name not in old:
                out.append((file, name))
    return out


def class_body_rows():
    return ({(f, 'class:' + c) for f, c, _ in CLASS_FIELDS} | {(f, 'class:' + c) for f, c, _, _ in CLASS_ROWS}
            | {('engine_driver.py', 'class:LayerGroups')})


def source_level():
    d = json.loads(DECLARATION.read_text())
    rows = paths()
    check_paths(rows, d)
    modified = {row.split('\t', 1)[1] for row in rows if row.startswith('M\t') and row.endswith('.py')}
    assert modified == set(d['definitions']), sorted(modified ^ set(d['definitions']))
    assert {'patches/s5_engine/' + n for n in ENGINE} == modified, sorted(modified)
    assert ['M\t' + REVIEW] == [r for r in rows if not r.endswith('.py')], 'only the review pin may move beside the sources'
    defs = {p: prove(show(p), (ROOT / p).read_text(), p, names) for p, names in d['definitions'].items()}
    compiled = []
    for name in sorted(modified):
        compile((ROOT / name).read_bytes(), name, 'exec')
        compiled.append(name)
    helpers = new_definitions(d)
    declared = {(f, n) for f, names in NEW_HELPERS.items() for n in names}
    assert set(helpers) - class_body_rows() == declared, sorted((set(helpers) - class_body_rows()) ^ declared)
    modules = {(p.rsplit('/', 1)[1], n) for p, names in d['definitions'].items() for n in names if n.startswith('module:')}
    assert modules == {(f, n) for f, names in MODULE_STATEMENTS.items() for n in names}, sorted(modules)
    return dict(default_off=default_off(), reach=reach(), parent=rev(PARENT), paths=rows, definitions=defs, compiled=compiled,
                skips=0, new_helpers={f + ':' + n: form for f, names in NEW_HELPERS.items() for n, form in names.items()},
                module_statements={f + ':' + n: why for f, names in MODULE_STATEMENTS.items() for n, why in names.items()},
                not_projected=NOT_PROJECTED, declaration_sha256=hashlib.sha256(DECLARATION.read_bytes()).hexdigest())


def rev(name):
    return subprocess.check_output(['git', 'rev-parse', name], cwd=ROOT, text=True).strip()


def overlay_level(parent, candidate):
    before, after = verify(parent), verify(candidate)
    assert set(after) == set(before), 'overlay file set changed'
    changed_files = {n for n in before if before[n] != after[n]} | {'SHA256SUMS'}
    assert RUNTIME <= changed_files <= RUNTIME | META, sorted(changed_files ^ (RUNTIME | META))
    d = json.loads(DECLARATION.read_text())['overlay_definitions']
    assert set(d) == RUNTIME
    defs = {p: prove((parent / p).read_text(), (candidate / p).read_text(), p, d[p]) for p in sorted(d)}
    for p in sorted(RUNTIME):
        name = p.split('/', 1)[1]
        assert (candidate / p).read_text() == _packaged((ROOT / 'patches/s5_engine' / name).read_text()), \
            'overlay module differs from the packaged committed source: ' + p
        assert (parent / p).read_text() == _packaged(show('patches/s5_engine/' + name)), \
            'parent overlay module differs from the packaged Step A source: ' + p
    assert (candidate / 'run-flash-s5-test.sh').read_bytes() == (parent / 'run-flash-s5-test.sh').read_bytes(), 'launcher changed'
    generated = sorted(n for n in before if n.startswith('e3_sources/'))
    assert generated and all(before[n] == after[n] for n in generated), 'generated vllm sources changed'
    native = [p for p in before if p.endswith('.so') or '/native30_plugins/' in p or p == 'qsa.py']
    assert native and all(before[n] == after[n] for n in native), 'native library or plugin changed'
    return dict(parent=str(parent), changed=sorted(changed_files), definitions=defs,
                generated_sources_byte_identical=len(generated), model_and_plugin_byte_identical=True,
                native_libraries_byte_identical=True, modules_equal_packaged_committed_source=sorted(RUNTIME),
                parent_modules_equal_packaged_step_a_source=sorted(RUNTIME))


# Default-off projections, step_a_freeze's kinds (off, drop, subs). The result must be Step A's exact unparsed AST (docstrings ignored).
TX_GROUP = 'if self._tx_group != 1:'
PROJECTIONS = {
    'engine_driver.py': {
        'NativeSegmentDriver.__init__': dict(drop={'self.layer_groups = None': 1, 'self._tx_group = 1': 1}),
        'NativeSegmentDriver._fused_control': dict(off={
            'if not fused and self.layer_groups is not None and (self.layer_groups.size != 1):': 1}),
        # _tx_group is 1: the group branches never run, and the plan is the parent's.
        'NativeSegmentDriver._fused_agree': dict(
            off={'if self._tx_group != 1 and descriptor.kind is SegmentKind.LAYER:': 1, TX_GROUP: 1},
            drop={'rows = profiles.pop()': 1},
            subs=(('for layer, shape in rows:', 'for layer, shape in profiles.pop():'),)),
        'NativeSegmentDriver._fused_hold': dict(off={TX_GROUP: 1}),
        'NativeSegmentDriver._apply_segments': dict(subs=((", **{'layer_group': descriptor.layer_group} if self._tx_group != 1 else {}", ''),)),
        # layer_groups None: _tx_group is 1; OpenRequest's layer_group field defaults to 1.
        'NativeSegmentDriver._begin': dict(
            drop={'self._tx_group = self.layer_groups.size if self._tx_fused and self.layer_groups is not None else 1': 1},
            off={TX_GROUP: 1}, subs=((', qsa_fused=self._tx_fused, layer_group=self._tx_group)', ', qsa_fused=self._tx_fused)'),)),
        # No plan entry has either key default-off (asserted in default_off), so both get() read their defaults. A layer descriptor
        # with layer_group 1 has layer_end == layer_start + 1 (wire_v1), so the two grant lookups are the same key.
        'NativeSegmentDriver._send_plan_entry': dict(
            off={'if self._tx_group != 1 and kind is SegmentKind.LAYER:': 1},
            subs=(("layer_end=entry.get('layer_end', None if layer is None else layer + 1), seq=entry.get('seq'), layer_group=entry.get('group', 1))",
                   "layer_end=None if layer is None else layer + 1, seq=entry.get('seq'))"),
                  ('self._profile_grants.get((stage, descriptor.layer_end - 1))', 'self._profile_grants.get((stage, layer))'),)),
        'NativeSegmentDriver._reset': dict(drop={'self._tx_group = 1': 1}),
        'NativeSegmentDriver._submit_control': dict(off={"if action == 'native_layer_group' and self.layer_groups is not None:": 1}),
        'NativeSegmentDriver._status_record': dict(off={'if self.layer_groups is not None:': 1}),
    },
    'lane_dispatch.py': {
        'SegmentCommand.__init__': dict(
            off={'if layer_group != 1:': 1}, drop={'self.layer_group = layer_group': 1},
            subs=((', profile=None, layer_group=1):', ', profile=None):'),)),
        'SegmentCommand.from_descriptor': dict(subs=((', profile=command.profile, layer_group=d.layer_group)', ', profile=command.profile)'),)),
        'LaneDispatcher._reset_transaction': dict(drop={'self.layer_group = 1': 1}),
        'LaneDispatcher._do_layer': dict(off={'if command.layer_group != self.layer_group:': 1, 'if self.layer_group != 1:': 1}),
    },
    'native_measurement.py': {
        # No journal row has layer_group default-off (_apply_segments adds it only when _tx_group != 1), so get() reads 1; for ints
        # already checked by _is_int, start < end <= start + 1 is end == start + 1.
        'NativeMeasurementCensus._ingest_segment': dict(subs=((
            "(not layer_start < layer_end <= layer_start + entry.get('layer_group', 1))", '(layer_end != layer_start + 1)'),)),
    },
    'native_overlap_control.py': {
        # _layer_groups() is False with the switch unset (NEW_OFF), so that disjunct adds nothing.
        'NativeControlClient._call': dict(subs=((" or (action == 'native_layer_group' and _layer_groups())", ''),)),
    },
    'native_overlap_session.py': {
        # With the switch unset the test holds for that action and the body sets handler None. Step A has no
        # _action_native_layer_group, so its getattr gave None for it too (asserted in default_off): the same refusal.
        'NativeOverlapSession.execute': dict(drop={
            "if action == 'native_layer_group' and (not (os.environ.get('FLASH_S5_LAYER_GROUPS', '0') == '1' and "
            "os.environ.get('FLASH_S5_QSA_FUSED', '0') == '1' and (os.environ.get('FLASH_S5_AUTO_ARM', '0') == '1'))):\n    handler = None": 1}),
    },
    'probe_control.py': {
        'attach_native_router': dict(off={
            "if auto_actions and env.get('FLASH_S5_QSA_FUSED', '0') == '1' and (env.get('FLASH_S5_LAYER_GROUPS', '0') == '1'):": 1}),
    },
    's5_startup.py': {
        'install_engine': dict(
            drop={"groups = env.get('FLASH_S5_LAYER_GROUPS', '0')": 1},
            off={"if groups not in ('0', '1'):": 1, "if groups == '1' and fused != '1':": 1, "if groups == '1':": 1}),
    },
    'wire_v1.py': {
        'SegmentDescriptor.__post_init__': dict(off={
            'if self.layer_group != 1:': 1,
            'if self.layer_group != 1 and (self.kind is not SegmentKind.LAYER or type(self.layer_group) is not int '
            'or self.layer_group not in LAYER_GROUP_SIZES):': 1}),
        # layer_group 1: __post_init__ holds layer_end == layer_start + 1, so the two tuples are equal.
        'SegmentDescriptor.layers': dict(subs=(('tuple(range(self.layer_start, self.layer_end))', '(self.layer_start,)'),)),
        'OpenRequest.__post_init__': dict(off={
            'if self.layer_group != 1 and (type(self.layer_group) is not int or not self.qsa_fused '
            'or self.layer_group not in LAYER_GROUP_SIZES):': 1}),
        # The same premise: the grant's query names the layer the descriptor holds.
        'SegmentCommand.__post_init__': dict(subs=(('query.layer != self.descriptor.layer_end - 1', 'query.layer != self.descriptor.layer_start'),)),
    },
    'worker_bridge.py': {
        'NativeWorkerBridge.open': dict(off={'if request.layer_group != 1:': 1}),
    },
}

NEW_OFF = {
    ('native_overlap_control.py', '_layer_groups'): (
        dict(subs=(("return os.environ.get('FLASH_S5_LAYER_GROUPS', '0') == '1' and _qsa_fused()", 'return False'),)),
        'def f():\n    return False'),
}

UNREACHED_FORM = 'not reachable default-off (reach)'
NEW_HELPERS = {
    'engine_driver.py': {
        'LayerGroups.__init__': 'never constructed: only install_layer_groups constructs it, called only under groups == 1',
        'LayerGroups.snapshot': UNREACHED_FORM,
        'layer_group_plan': UNREACHED_FORM,
        'NativeSegmentDriver.install_layer_groups': 'called only under groups == 1 in install_engine',
        'NativeSegmentDriver._layer_group_control': 'guarded by layer_groups is not None in _submit_control',
        'NativeSegmentDriver._group_stage_plan': 'guarded by _tx_group != 1 in _fused_agree',
    },
    'lane_dispatch.py': {
        'LaneDispatcher.select_layer_group': 'called only under request.layer_group != 1 in NativeWorkerBridge.open',
        'LaneDispatcher._do_layer_group': 'guarded by layer_group != 1 in _do_layer',
    },
    'native_overlap_control.py': {
        '_layer_groups': 'returns False (NEW_OFF)',
        'NativeControlClient.native_layer_group': 'reached only from _action_native_layer_group',
    },
    'native_overlap_session.py': {
        'NativeOverlapSession._action_native_layer_group': 'execute sets handler None for this action',
    },
}

UNREACHED = {
    'engine_driver.py': ('class:LayerGroups', 'layer_group_plan', 'NativeSegmentDriver.install_layer_groups',
                         'NativeSegmentDriver._layer_group_control', 'NativeSegmentDriver._group_stage_plan'),
    'lane_dispatch.py': ('LaneDispatcher.select_layer_group', 'LaneDispatcher._do_layer_group'),
    'native_overlap_control.py': ('NativeControlClient.native_layer_group',),
    'native_overlap_session.py': ('NativeOverlapSession._action_native_layer_group',),
}
GUARDS = {}

CLASS_FIELDS = (('wire_v1.py', 'SegmentDescriptor', ['layer_group: int = 1']),
                ('wire_v1.py', 'OpenRequest', ['layer_group: int = 1']))
# (file, class, the parent's row, this row): a row that changes rather than one added. __slots__ names the one field; its default is 1.
CLASS_ROWS = (('lane_dispatch.py', 'SegmentCommand',
               "__slots__ = ('transaction', 'stage', 'layer_start', 'layer_end', 'op_seq', 'kind', 'seq', 'source', 'profile')",
               "__slots__ = ('transaction', 'stage', 'layer_start', 'layer_end', 'op_seq', 'kind', 'seq', 'source', 'profile', 'layer_group')"),)

_WIRE_HEAD = ('module:from wire_v1 import BootstrapAck, BootstrapCommand, CancelAck, CancelRequest, PPReceiptAck, PPReceiptDelivery, PeerBinding, '
              'PeerBindingAck, PollRequest, QsaProfileGrant, QsaProfileQuery, QsaScheduleShape, ')
_WIRE_TAIL = ('qsa_profile_agreement, ReleaseRequest, ReleaseAck, OpenAck, OpenRequest, OpSeqTracker, ROUTE_BANK_FULL, ROUTE_BANK_SPLIT, SegmentAcceptance, '
              'SegmentCommand, SegmentDescriptor, SegmentKind, TransitionAck, TransitionCommand, TransitionPending, WorkerCompletionReport')
MODULE_STATEMENTS = {
    'engine_driver.py': {
        _WIRE_HEAD + 'LAYER_GROUP_SIZES, ' + _WIRE_TAIL: 'adds one name to an existing import, read only by _layer_group_control',
        _WIRE_HEAD + _WIRE_TAIL: 'the parent\'s row of the same import',
    },
    'wire_v1.py': {'module:LAYER_GROUP_SIZES': 'a constant read only by the group guards'},
}

NOT_PROJECTED = [
    dict(what='wire dataclass instances', detail='OpenRequest and SegmentDescriptor instances carry a defaulted layer_group, and the dispatcher\'s '
         'SegmentCommand a slot, so pickles and __eq__ inputs differ from Step A in bytes though not in meaning. Wire compatibility between a Step A '
         'engine and these workers is not claimed: one bundle installs both sides.'),
    dict(what='module imports', detail='engine_driver imports one more name from wire_v1, read only by _layer_group_control.'),
    dict(what='session handler lookup', detail='NativeOverlapSession.execute finds the new method through getattr, then clears it. The drop rests '
         'on Step A having no such method, asserted in default_off, and on the same exception following.'),
    dict(what='plan and descriptor premises', detail='_send_plan_entry reads layer_end and group from the plan entry, and its grant lookup and the '
         'wire grant check read layer_end - 1. The projection removes these because no plan entry has those keys default-off and a layer_group 1 '
         'descriptor has layer_end == layer_start + 1. Whole-program facts, not AST equalities; default_off asserts the only writer of those keys is '
         '_group_stage_plan, which reach proves unreachable default-off, and wire_v1 keeps its refusal of any other shape.'),
    dict(what='journal shape check', detail='native_measurement reads layer_group from the row; only _apply_segments writes it, under _tx_group != 1.'),
    dict(what='declared overhead', detail='Default-off adds attribute and dict reads, none of them a host wait or device call: _tx_group at each driver '
         'site named above, command.layer_group and self.layer_group per dispatcher layer, entry.get per plan entry and journal row, an env read per '
         'unknown native action. Not measured.'),
    dict(what='FLASH_S5_LAYER_GROUPS set to other values', detail='The proof is for the variable unset, as briefed. 0 behaves the same. Any value other '
         'than 0 or 1 now refuses at startup; Step A ignored the variable. Set to 1 without FLASH_S5_QSA_FUSED=1 also refuses.'),
    dict(what='native_candidate_review.json', detail='Not code: the review pin moves by design. The build checks it against the tree.'),
]


def plan_key_writers():
    """Every engine dict display or call naming layer_end= and group=, and every journal writer of layer_group: the plan premises."""
    rows = []
    for file in ENGINE:
        for qual, node in _units((ROOT / 'patches/s5_engine' / file).read_text(), file):
            for n in ast.walk(node):
                if isinstance(n, ast.Call) and {'layer_end', 'group'} <= {k.arg for k in n.keywords}:
                    rows.append((file, qual))
    return rows


def class_rows(new, old, name, was, now):
    """The class's non-method body is the parent's with exactly that one row replaced (docstrings ignored)."""
    def rows(text):
        node = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == name)
        out = [ast.unparse(item) for item in node.body if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if out and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            out = out[1:]
        return out
    body_new, body_old = rows(new), rows(old)
    assert body_old.count(was) == 1 and body_new.count(now) == 1, 'declared class row missing: ' + name
    assert [now if r == was else r for r in body_old] == body_new, 'class body differs beyond the declared row: ' + name
    return hashlib.sha256('\n'.join(body_old).encode()).hexdigest()


def default_off():
    result, table = {}, {}
    for file, specs in PROJECTIONS.items():
        old = show('patches/s5_engine/' + file)
        new = (ROOT / 'patches/s5_engine' / file).read_text()
        for name, spec in specs.items():
            try:
                result[file + ':' + name] = project(function(new, name), function(old, name), spec)
            except AssertionError as exc:
                raise AssertionError('%s:%s: %s' % (file, name, exc)) from None
            table[file + ':' + name] = removals(spec)
    for (file, name), (spec, target) in NEW_OFF.items():
        new = (ROOT / 'patches/s5_engine' / file).read_text()
        result[file + ':' + name + ' (off path)'] = project(function(new, name), target, spec)
        table[file + ':' + name + ' (off path)'] = removals(spec)
    d = json.loads(DECLARATION.read_text())
    wanted = set()
    for path, names in d['definitions'].items():
        file = path.rsplit('/', 1)[1]
        old_names = {q for q, _ in _units(show(path), file)}
        wanted |= {(file, n) for n in names if n in old_names and not n.startswith(('class:', 'module:'))}
    projected = {(f, n) for f, specs in PROJECTIONS.items() for n in specs}
    assert wanted == projected, sorted(wanted ^ projected)
    session = ast.parse(show('patches/s5_engine/native_overlap_session.py'))
    cls = next(n for n in session.body if isinstance(n, ast.ClassDef) and n.name == 'NativeOverlapSession')
    assert not any(getattr(n, 'name', None) == '_action_native_layer_group' for n in cls.body), 'parent has the session action'
    writers = plan_key_writers()
    assert writers == [('engine_driver.py', 'NativeSegmentDriver._group_stage_plan')], writers
    for file, name, added in CLASS_FIELDS:
        new, old = (ROOT / 'patches/s5_engine' / file).read_text(), show('patches/s5_engine/' + file)
        result[file + ':class:' + name] = base.class_fields(new, old, name, added)
    for file, name, was, now in CLASS_ROWS:
        new, old = (ROOT / 'patches/s5_engine' / file).read_text(), show('patches/s5_engine/' + file)
        result[file + ':class:' + name] = class_rows(new, old, name, was, now)
    return dict(verdict='PASS', legacy_function_bodies=result, functions=len(result), removals=table,
                premises=dict(session_parent_has_no_action=True, plan_keys_written_only_in=writers),
                meaning='source projection with FLASH_S5_LAYER_GROUPS unset: every Step A function Step B changes is its AST once the declared '
                        'statements are removed; the new helper on that path returns its parent value; class fields default to the parent '
                        'behaviour. See not_projected for what this cannot show.')


def _unreached_names():
    out = {}
    for file, names in UNREACHED.items():
        for name in names:
            out[name.split(':', 1)[-1].split('.')[-1]] = (file, name)
    return out


def reach(texts=None):
    """Each UNREACHED definition is referenced only from UNREACHED code or under its declared guard (step_a_freeze's rule)."""
    texts = texts or {}
    names = _unreached_names()
    projected = {(f, n) for f, specs in PROJECTIONS.items() for n in specs} | set(NEW_OFF)
    skipped = {(f, n) for f, ns in UNREACHED.items() for n in ns}
    checked, guarded = 0, []
    for file in ENGINE:
        new = texts.get(file) or (ROOT / 'patches/s5_engine' / file).read_text()
        for qual, node in _units(new, file):
            cls = qual.split('.')[0] if '.' in qual else None
            if (file, qual) in skipped or (cls and (file, 'class:' + cls) in skipped) or (file, qual) in projected:
                continue
            for name, guards in _references(node, set(names)):
                allowed = GUARDS.get(name, set())
                assert guards & allowed, 'Step B only definition reachable default-off: %s:%s -> %s %s' % (file, qual, name, sorted(guards))
                guarded.append('%s:%s -> %s under %s' % (file, qual, name, sorted(guards & allowed)[0]))
            checked += 1
    return dict(verdict='PASS', unreached=sorted('%s:%s' % v for v in names.values()), units_checked=checked, guarded_references=sorted(guarded))


def negative_controls():
    d = json.loads(DECLARATION.read_text())
    p = 'patches/s5_engine/lane_dispatch.py'

    def refused(fn):
        try:
            fn()
        except AssertionError:
            return 'REFUSED'
        raise AssertionError('undeclared change accepted')

    engine = (ROOT / 'patches/s5_engine/engine_driver.py').read_text()
    old_engine = show('patches/s5_engine/engine_driver.py')
    begin = function(engine, 'NativeSegmentDriver._begin')
    begin.body.insert(0, ast.parse('self.phase = None').body[0])
    spec = PROJECTIONS['engine_driver.py']['NativeSegmentDriver._begin']
    short = dict(spec, off={})
    twice = dict(spec, off={TX_GROUP: 2})
    tree = ast.parse(engine)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NativeSegmentDriver')
    tick = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'control_tick')
    tick.body.insert(0, ast.parse('self._group_stage_plan(0, set())').body[0])
    reached = ast.unparse(tree)
    own = ('native_overlap_control.py', '_layer_groups')
    # A second writer of the plan keys, outside the guarded method: the _send_plan_entry premise.
    writer = ast.parse(engine)
    wcls = next(n for n in writer.body if isinstance(n, ast.ClassDef) and n.name == 'NativeSegmentDriver')
    wtick = next(n for n in wcls.body if isinstance(n, ast.FunctionDef) and n.name == 'control_tick')
    wtick.body.insert(0, ast.parse('self.plan.append(dict(layer_end=1, group=2))').body[0])

    def extra_writer():
        rows = []
        for qual, node in _units(ast.unparse(writer), 'engine_driver.py'):
            for n in ast.walk(node):
                if isinstance(n, ast.Call) and {'layer_end', 'group'} <= {k.arg for k in n.keywords}:
                    rows.append(('engine_driver.py', qual))
        assert rows == [('engine_driver.py', 'NativeSegmentDriver._group_stage_plan')], rows

    def undeclared_helper():
        rows = set(new_definitions(dict(d, definitions=dict(d['definitions'], **{
            p: d['definitions'][p] + ['LaneDispatcher.step_b_undeclared']})))) - class_body_rows()
        assert rows == {(f, n) for f, names in NEW_HELPERS.items() for n in names}, 'undeclared new helper'

    def undeclared_class_row():
        new = (ROOT / 'patches/s5_engine/lane_dispatch.py').read_text()
        _, name, was, now = CLASS_ROWS[0]
        class_rows(new.replace("__slots__ = ('transaction',", "__slots__ = ('undeclared', 'transaction',", 1), show(p), name, was, now)
    return {'undeclared-function': refused(lambda: prove(show(p), (ROOT / p).read_text() + '\ndef step_b_undeclared():\n pass\n',
                                                          p, d['definitions'][p])),
            'undeclared-path': refused(lambda: check_paths(paths() + ['A\tpatches/s5_engine/step_b_undeclared.py'], d)),
            'undeclared-overlay': refused(lambda: prove('x=1\n', 'x=2\n', 'core/scheduler.py', [])),
            'undeclared-default-off-statement': refused(lambda: project(
                begin, function(old_engine, 'NativeSegmentDriver._begin'), spec)),
            'missing-default-off-declaration': refused(lambda: project(
                function(engine, 'NativeSegmentDriver._begin'), function(old_engine, 'NativeSegmentDriver._begin'), short)),
            'miscounted-default-off-statement': refused(lambda: project(
                function(engine, 'NativeSegmentDriver._begin'), function(old_engine, 'NativeSegmentDriver._begin'), twice)),
            'unguarded-step-b-reference': refused(lambda: reach({'engine_driver.py': reached})),
            'wrong-off-value': refused(lambda: project(
                function((ROOT / 'patches/s5_engine' / own[0]).read_text(), own[1]), 'def f():\n    return True', NEW_OFF[own][0])),
            'undeclared-new-helper': refused(undeclared_helper),
            'second-plan-key-writer': refused(extra_writer),
            'undeclared-class-row': refused(undeclared_class_row)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--overlay', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--draft-declaration', action='store_true', help='print the declaration the tree implies (for review), and exit')
    a = parser.parse_args()
    if a.draft_declaration:
        print(json.dumps(declaration_draft(), indent=1, sort_keys=True))
        raise SystemExit(0)
    if a.out is None:
        parser.error('--out is required')
    r = dict(verdict='PASS', source=source_level(), negative_controls=negative_controls())
    if a.overlay:
        r['overlay'] = overlay_level(a.parent, a.overlay)
    with a.out.open('x') as f:
        json.dump(r, f, sort_keys=True, indent=2)
        f.write('\n')
    print('STEP_B_FREEZE ' + r['verdict'])
