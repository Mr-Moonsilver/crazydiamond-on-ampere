#!/usr/bin/env python3
"""Fail-closed default-off freeze of Step A's engine against the composed trunk c3f1f64, source and generated overlay.

tools/native41_held_g_freeze.py's machinery (project, class_fields, reach, the negative controls), held-g's text kept, over
Step A's declarations. With ``FLASH_S5_QSA_FUSED`` unset, so ``driver.qsa_fused`` is None, ``_tx_fused`` False, every OPEN carries
qsa_fused False, ``dispatcher.qsa_fused`` False, no communicator has a ledger, ``_fused_grant`` None and no grant is fused:

* every pre-existing function Step A changes projects back to c3f1f64's exact AST once the declared statements are removed; each
  removal is named and counted, and an undeclared statement makes the proof refuse (negative controls);
* every new definition is listed in NEW_HELPERS with its off form; the one the default-off path can call reduces to its value,
  and the rest are referenced only from code that cannot run default-off (reach);
* class fields, module statements and what cannot be projected are named in CLASS_FIELDS, MODULE_STATEMENTS and NOT_PROJECTED;
* the overlay holds each module exactly as make_s5_overlap.package(committed source) writes it, the parent overlay's as
  package(c3f1f64 source), and the generated vllm sources, launcher, plugins and native libraries are byte-identical.

Step A with the switch on is not claimed equal to c3f1f64 here: the connected probe, the fused suites and the box compare those.
The harness and tools paths are the builder's (tools/build_native43.py source_level over the spec, against alpha-perf 435dba6);
this tool declares patches/s5_engine against c3f1f64. CPU only, no box action.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from native37_robust_freeze import changed, prove  # noqa: E402
from build_native30_held import verify  # noqa: E402

PARENT = 'c3f1f64998abcbd9f337e6a27eed7bd13fd512c1'
PARENT_SHORT = 'c3f1f64'
DECLARATION = ROOT / 'tools/step_a_declarations.json'
ENGINE = ('engine_driver.py', 'lane_comm.py', 'lane_dispatch.py', 'native_overlap_control.py', 'native_overlap_session.py',
          'probe_control.py', 'qsa_driver.py', 'qsa_operation.py', 's5_startup.py', 'wire_v1.py', 'worker_bridge.py')
RUNTIME = {'e3/' + name for name in ENGINE}
META = {'SHA256SUMS', 'S5-OVERLAP-GENERATION.json', 'e3/native-implementation-closure.json',
        'e3/native-overlap-readiness.json', 'e3/decode-probe-evidence.json', 'core/SHA256SUMS'}
REVIEW = 'patches/s5_engine/native_candidate_review.json'


def show(path, commit=PARENT_SHORT):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=ROOT, text=True)


def paths():
    return subprocess.check_output(['git', 'diff', '--name-status', PARENT_SHORT, '--', 'patches/s5_engine'], cwd=ROOT,
                                   text=True).splitlines()


def check_paths(actual, d):
    assert set(actual) == set(d['paths']), sorted(set(actual) ^ set(d['paths']))


def _packaged(source):
    """``source`` as the overlay holds it: make_s5_overlap.package(), which binds E3 imports to their packages."""
    engine = str(ROOT / 'patches/s5_engine')
    if engine not in sys.path:
        sys.path.insert(0, engine)
    import make_s5_overlap
    return make_s5_overlap.package(source)


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
    return dict(schema='step-a-freeze-declaration/1', parent=PARENT_SHORT, paths=sorted(rows), definitions=definitions,
                overlay_definitions=overlay)


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
    assert set(helpers) - class_body_rows() == declared, sorted(set(helpers) ^ declared)
    modules = {(p.rsplit('/', 1)[1], n) for p, names in d['definitions'].items() for n in names if n.startswith('module:')}
    assert modules == {(f, n) for f, names in MODULE_STATEMENTS.items() for n in names}, sorted(modules)
    return dict(default_off=default_off(), reach=reach(), parent=PARENT, paths=rows, definitions=defs, compiled=compiled,
                skips=0, new_helpers={f + ':' + n: form for f, names in NEW_HELPERS.items() for n, form in names.items()},
                module_statements={f + ':' + n: why for f, names in MODULE_STATEMENTS.items() for n, why in names.items()},
                not_projected=NOT_PROJECTED, declaration_sha256=hashlib.sha256(DECLARATION.read_bytes()).hexdigest())


def class_body_rows():
    return {(f, 'class:' + c) for f, c, _ in CLASS_FIELDS} | {('engine_driver.py', 'class:FusedQsa'),
                                                              ('lane_comm.py', 'class:CollectiveLedger')}


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
            'parent overlay module differs from the packaged c3f1f64 source: ' + p
    assert (candidate / 'run-flash-s5-test.sh').read_bytes() == (parent / 'run-flash-s5-test.sh').read_bytes(), 'launcher changed'
    generated = sorted(n for n in before if n.startswith('e3_sources/'))
    assert generated and all(before[n] == after[n] for n in generated), 'generated vllm sources changed'
    native = [p for p in before if p.endswith('.so') or '/native30_plugins/' in p or p == 'qsa.py']
    assert native and all(before[n] == after[n] for n in native), 'native library or plugin changed'
    return dict(parent=str(parent), changed=sorted(changed_files), definitions=defs,
                generated_sources_byte_identical=len(generated), model_and_plugin_byte_identical=True,
                native_libraries_byte_identical=True, modules_equal_packaged_committed_source=sorted(RUNTIME),
                parent_modules_equal_packaged_c3f1f64_source=sorted(RUNTIME))


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


# Default-off projections. Kinds (held-g's):
#   off:  an ``if`` whose test is false then; it becomes its else branch (an empty ``else`` stays empty).
#   drop: a statement binding a Step A only name (or the projection's premise read), or one whose effect the note proves.
#   subs: exact text substitutions on the unparsed result, (old, new) matching once or (old, new, n) matching n times.
# Each listed statement must match the declared number of times; the result must be the parent function's exact unparsed AST
# (docstrings ignored).
TX_FUSED = 'if self._tx_fused:'
PROJECTIONS = {
    'engine_driver.py': {
        'NativeSegmentDriver.__init__': dict(drop={'self.qsa_fused = None': 1, 'self._tx_fused = False': 1,
                                                   'self._fused_mark = None': 1, 'self._fused_mark_at = None': 1}),
        'NativeSegmentDriver._wait_permit': dict(off={"if self._tx_fused and inflight == ('poll',):": 1}),
        'NativeSegmentDriver._apply_segments': dict(off={TX_FUSED: 1}),
        # qsa_fused None: _tx_fused is False, _progress_owned is not called; OpenRequest's qsa_fused field defaults to False.
        'NativeSegmentDriver._begin': dict(
            drop={'self._tx_fused = self.qsa_fused is not None and self.qsa_fused.selected and self._progress_owned()': 1,
                  'self._fused_mark = self._fused_mark_at = None': 1},
            off={TX_FUSED: 1}, subs=(('admission.state_epoch, 1, qsa_fused=self._tx_fused)', 'admission.state_epoch, 1)'),)),
        'NativeSegmentDriver._drive': dict(off={TX_FUSED: 1, 'if self._tx_fused and self._fused_hold(self.plan[self.index]):': 1}),
        'NativeSegmentDriver._send_plan_entry': dict(off={'if self._tx_fused and profile is not None:': 1}),
        'NativeSegmentDriver._reset': dict(drop={'self._tx_fused = False': 1}),
        'NativeSegmentDriver._submit_control': dict(off={"if action == 'native_qsa_fused' and self.qsa_fused is not None:": 1}),
        'NativeSegmentDriver._status_record': dict(off={'if self.qsa_fused is not None:': 1}),
    },
    'lane_comm.py': {
        'PrefillTpCommunicator.__init__': dict(drop={'self.ledger = None': 1}),
        'PrefillTpCommunicator.all_reduce': dict(off={'if self.ledger is not None:': 1}),
        # A communicator's ledger is None unless select_fused set it.
        'canonical_selection': dict(drop={"ledger = getattr(communicator, 'ledger', None)": 1},
                                    off={'if ledger is not None:': 1}),
    },
    'lane_dispatch.py': {
        # dispatcher.qsa_fused is False unless select_fused ran; an OPEN with qsa_fused False never runs it.
        'LaneDispatcher._reset_transaction': dict(off={"if getattr(self, 'qsa_fused', False):": 1},
                                                  drop={'self.qsa_fused = False': 1, 'self.fused_profiles = None': 1}),
        'LaneDispatcher.submit': dict(drop={'ledger = self._ledger() if self.qsa_fused else None': 1},
                                      off={'if ledger is not None:': 1, 'if self.qsa_fused:': 1}),
        'LaneDispatcher._do_prepare': dict(off={'if self.qsa_fused:': 1}),
        'LaneDispatcher._do_layer': dict(off={'if self.qsa_fused:': 1}),
    },
    'native_overlap_control.py': {
        # _qsa_fused() is False with the switch unset (NEW_OFF), so that disjunct adds nothing.
        'NativeControlClient._call': dict(subs=((" or (action == 'native_qsa_fused' and _qsa_fused())", ''),)),
    },
    'native_overlap_session.py': {
        # With the switch unset the test holds for that action and the body sets handler None. c3f1f64 has no
        # _action_native_qsa_fused, so its getattr gave None for it too (asserted in default_off): the same refusal.
        'NativeOverlapSession.execute': dict(drop={
            "if action == 'native_qsa_fused' and (not (os.environ.get('FLASH_S5_QSA_FUSED', '0') == '1' and "
            "os.environ.get('FLASH_S5_AUTO_ARM', '0') == '1')):\n    handler = None": 1}),
    },
    'probe_control.py': {
        'attach_native_router': dict(off={"if auto_actions and env.get('FLASH_S5_QSA_FUSED', '0') == '1':": 1}),
    },
    'qsa_driver.py': {
        'QsaLayerDriver.__init__': dict(drop={'self._fused_grant = None': 1}),
        'QsaLayerDriver.run_layer': dict(off={'if self._fused_grant is not None:': 1}),
    },
    'qsa_operation.py': {
        # No grant is fused default-off (asserted in default_off: fused=True is built only inside _fused_agree).
        'QsaPrefillOperation.fill_bucket': dict(subs=((
            "if self.pending_buckets >= MAX_OUTSTANDING_BUCKETS and (not getattr(run.schedule_grant, 'fused', False)):",
            'if self.pending_buckets >= MAX_OUTSTANDING_BUCKETS:'),)),
    },
    's5_startup.py': {
        'install_engine': dict(
            drop={"fused = env.get('FLASH_S5_QSA_FUSED', '0')": 1},
            off={"if fused not in ('0', '1'):": 1,
                 "if fused == '1' and (auto != '1' or env.get('FLASH_S5_NATIVE_QSA_POLICY', 'current') != 'dense'):": 1,
                 "if fused == '1':": 1}),
    },
    'wire_v1.py': {
        'OpenRequest.__post_init__': dict(off={'if type(self.qsa_fused) is not bool:': 1}),
        'QsaProfileGrant.__post_init__': dict(off={
            "if type(self.fused) is not bool or (self.fused and self.shape.route != 'dense_dma'):": 1}),
        'SegmentAcceptance.__post_init__': dict(off={'if self.dense_profiles is not None:': 1,
                                                     'if self.collectives is not None:': 1}),
    },
    'worker_bridge.py': {
        'NativeWorkerBridge.open': dict(off={'if request.qsa_fused:': 1}),
        # profile.fused False takes the else branch, c3f1f64's call. The stepped dispatcher's result and acknowledgement hold
        # neither key, so both get() read None, the fields' defaults.
        'NativeWorkerBridge.submit': dict(
            off={'if command.profile.fused:': 1},
            subs=((", dense_profiles=result.get('dense_profiles'), collectives=acknowledgement.get('collectives')", ''),)),
    },
}

# The one new function the default-off path can call, by the name _call reaches it through.
NEW_OFF = {
    ('native_overlap_control.py', '_qsa_fused'): (
        dict(subs=(("return os.environ.get('FLASH_S5_QSA_FUSED', '0') == '1' and os.environ.get('FLASH_S5_AUTO_ARM', '0') == '1'",
                    'return False'),)),
        'def f():\n    return False'),
}

# Every new definition, by file, with its off form.
UNREACHED_FORM = 'not reachable default-off (reach)'
NEW_HELPERS = {
    'engine_driver.py': {
        'FusedQsa.__init__': 'never constructed: only install_qsa_fused constructs it, called only under fused == 1',
        'FusedQsa.snapshot': UNREACHED_FORM,
        'fused_expected_shape': UNREACHED_FORM,
        'NativeSegmentDriver.install_qsa_fused': 'called only under fused == 1 in install_engine',
        'NativeSegmentDriver._fused_control': 'guarded by qsa_fused is not None in _submit_control',
        'NativeSegmentDriver._fused_agree': 'guarded by _tx_fused in _apply_segments',
        'NativeSegmentDriver._fused_hold': 'guarded by _tx_fused in _drive',
        'NativeSegmentDriver._fused_watchdog': 'guarded by _tx_fused in _drive',
    },
    'lane_comm.py': {
        'CollectiveLedger.__init__': 'never constructed: only select_fused constructs it',
        'CollectiveLedger.begin': UNREACHED_FORM, 'CollectiveLedger.note': UNREACHED_FORM, 'CollectiveLedger.take': UNREACHED_FORM,
    },
    'lane_dispatch.py': {
        'LaneDispatcher.select_fused': 'called only under request.qsa_fused in NativeWorkerBridge.open',
        'LaneDispatcher._set_ledger': 'called only from select_fused and under qsa_fused in _reset_transaction',
        'LaneDispatcher._ledger': 'called only under qsa_fused in submit',
        'LaneDispatcher._do_layer_fused': 'guarded by qsa_fused in _do_layer',
    },
    'native_overlap_control.py': {
        '_qsa_fused': 'returns False (NEW_OFF)',
        'NativeControlClient.native_qsa_fused': 'reached only from _action_native_qsa_fused',
    },
    'native_overlap_session.py': {
        'NativeOverlapSession._action_native_qsa_fused': 'execute sets handler None for this action',
    },
    'qsa_driver.py': {
        'QsaLayerDriver.fused_profiles': 'called only under qsa_fused in _do_prepare',
        'QsaLayerDriver.arm_fused': 'called only from _do_layer_fused',
        'QsaLayerDriver._fused': 'guarded by _fused_grant in run_layer; arm_fused alone sets it',
    },
    'qsa_operation.py': {
        'QsaPrefillOperation.fused_refusal': 'called only from fused_profiles',
        'QsaPrefillOperation.fused_shape': 'called only from fused_profiles',
        'QsaPrefillOperation.advance_fused': 'called only from QsaLayerDriver._fused',
    },
    'worker_bridge.py': {
        'NativeWorkerBridge._fused_prepare_command': 'guarded by profile.fused in submit',
    },
}

# Definitions that cannot run default-off, by the name code reaches them through. Every reference from code outside this set must
# sit in the body of an ``if`` whose test is one of the declared guards; the projected functions are proven separately.
UNREACHED = {
    'engine_driver.py': ('class:FusedQsa', 'fused_expected_shape', 'NativeSegmentDriver.install_qsa_fused',
                         'NativeSegmentDriver._fused_control', 'NativeSegmentDriver._fused_agree',
                         'NativeSegmentDriver._fused_hold', 'NativeSegmentDriver._fused_watchdog'),
    'lane_comm.py': ('class:CollectiveLedger',),
    'lane_dispatch.py': ('LaneDispatcher.select_fused', 'LaneDispatcher._set_ledger', 'LaneDispatcher._ledger',
                         'LaneDispatcher._do_layer_fused'),
    'native_overlap_control.py': ('NativeControlClient.native_qsa_fused',),
    'native_overlap_session.py': ('NativeOverlapSession._action_native_qsa_fused',),
    'qsa_driver.py': ('QsaLayerDriver.fused_profiles', 'QsaLayerDriver.arm_fused', 'QsaLayerDriver._fused'),
    'qsa_operation.py': ('QsaPrefillOperation.fused_refusal', 'QsaPrefillOperation.fused_shape',
                         'QsaPrefillOperation.advance_fused'),
    'worker_bridge.py': ('NativeWorkerBridge._fused_prepare_command',),
}
GUARDS = {}

# (file, class, the added field rows). Each default leaves the parent's behaviour: see NOT_PROJECTED for the object's bytes.
CLASS_FIELDS = (('wire_v1.py', 'OpenRequest', ['qsa_fused: bool = False']),
                ('wire_v1.py', 'QsaProfileGrant', ['fused: bool = False']),
                ('wire_v1.py', 'SegmentAcceptance', ['dense_profiles: tuple | None = field(default=None, repr=False)',
                                                     'collectives: tuple | None = field(default=None, repr=False)']))

MODULE_STATEMENTS = {
    'engine_driver.py': {
        'module:from e3_types import ARENA_PAGES, CompletionKind, CompletionProof, DependencyRole, E3ContractError, E3Pending, '
        'LaneRole, OperationTicket, PAGE_TOKENS, TransactionId': 'adds two constants to an existing import, read only by '
        'fused_expected_shape',
        'module:from e3_types import CompletionKind, CompletionProof, DependencyRole, E3ContractError, E3Pending, LaneRole, '
        'OperationTicket, TransactionId': 'the parent\'s row of the same import',
        'module:from wire_v1 import BootstrapAck, BootstrapCommand, CancelAck, CancelRequest, PPReceiptAck, PPReceiptDelivery, '
        'PeerBinding, PeerBindingAck, PollRequest, QsaProfileGrant, QsaProfileQuery, QsaScheduleShape, qsa_profile_agreement, '
        'ReleaseRequest, ReleaseAck, OpenAck, OpenRequest, OpSeqTracker, ROUTE_BANK_FULL, ROUTE_BANK_SPLIT, SegmentAcceptance, '
        'SegmentCommand, SegmentDescriptor, SegmentKind, TransitionAck, TransitionCommand, TransitionPending, '
        'WorkerCompletionReport': 'adds two names to an existing import, read only by the fused code',
        'module:from wire_v1 import BootstrapAck, BootstrapCommand, CancelAck, CancelRequest, PPReceiptAck, PPReceiptDelivery, '
        'PeerBinding, PeerBindingAck, PollRequest, QsaProfileQuery, qsa_profile_agreement, ReleaseRequest, ReleaseAck, OpenAck, '
        'OpenRequest, OpSeqTracker, ROUTE_BANK_FULL, ROUTE_BANK_SPLIT, SegmentAcceptance, SegmentCommand, SegmentDescriptor, '
        'SegmentKind, TransitionAck, TransitionCommand, TransitionPending, WorkerCompletionReport': 'the parent\'s row',
    },
    'lane_comm.py': {
        'module:__all__': 'names CollectiveLedger',
        'module:import hashlib': 'stdlib module, read only by CollectiveLedger',
    },
    'lane_dispatch.py': {
        'module:from lane_comm import CollectiveLedger, LaneCommError': 'adds one name, read only by select_fused',
        'module:from lane_comm import LaneCommError': 'the parent\'s row',
    },
}

NOT_PROJECTED = [
    dict(what='wire dataclass instances', detail='OpenRequest, QsaProfileGrant and SegmentAcceptance instances carry the new '
         'defaulted fields, so their pickles and __eq__ inputs differ from c3f1f64 in bytes though not in meaning. No AST projection '
         'can show this; the default-off comparison shows the same scheduling and tokens, not the same wire bytes. Wire compatibility '
         'between a c3f1f64 engine and these workers is not claimed: one bundle installs both sides.'),
    dict(what='module imports', detail='hashlib and the added imported names load at module import. See module_statements.'),
    dict(what='session handler lookup', detail='NativeOverlapSession.execute finds the new method through getattr, then clears it. '
         'The drop rests on c3f1f64 having no such method, asserted in default_off, and on the same exception following.'),
    dict(what='fill_bucket substitution premise', detail='The projection removes getattr(run.schedule_grant, "fused", False) because '
         'no grant is fused default-off. That is a whole-program fact, not an AST equality; default_off asserts the only fused=True '
         'construction in the engine sources is inside _fused_agree, which reach proves unreachable default-off.'),
    dict(what='declared overhead', detail='Default-off adds attribute reads, none of them a host wait or device call: _tx_fused at '
         'each driver site named above, self.ledger per all_reduce, getattr(communicator, "ledger") per canonical selection, '
         'self.qsa_fused per dispatcher submit, env reads per native action when an unknown action arrives. Not measured.'),
    dict(what='FLASH_S5_QSA_FUSED set to other values', detail='The proof is for the variable unset, as briefed. 0 behaves the same '
         '(same branches). Any value other than 0 or 1 now refuses at startup; c3f1f64 ignored the variable.'),
    dict(what='native_candidate_review.json', detail='Not code: the review pin moves by design (source and harness shas, CPU receipts). '
         'The build checks it against the tree.'),
]


def project(new, old, spec):
    """Apply one declared projection to ``new``; its text must equal ``old``'s."""
    counts = {}
    for kind in ('drop', 'off', 'on', 'drop_prefix'):
        for head in spec.get(kind, {}):
            counts[(kind, head)] = 0
    unreachable = [0]

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
        # A loop over a Step A only empty sequence, by its first line.
        if isinstance(stmt, ast.For) and first in spec.get('drop_prefix', {}):
            return 'drop_prefix', first
        return None

    def body(stmts, keep_empty=False):
        out = []
        for index, stmt in enumerate(stmts):
            hit = match(stmt)
            if hit is not None:
                counts[hit] += 1
                kind = hit[0]
                if kind == 'off':
                    assert isinstance(stmt, ast.If), hit
                    out += body(stmt.orelse, keep_empty=True)
                elif kind == 'on':
                    assert isinstance(stmt, ast.If) and not stmt.orelse, hit
                    inner = body(stmt.body)
                    out += inner
                    if inner and isinstance(inner[-1], (ast.Return, ast.Raise)):
                        unreachable[0] += len(stmts) - index - 1
                        break
                continue
            for field in ('body', 'orelse', 'finalbody'):
                if isinstance(getattr(stmt, field, None), list) and getattr(stmt, field):
                    kept = body(getattr(stmt, field))
                    if field == 'body':
                        kept = kept or [ast.Pass()]
                    setattr(stmt, field, kept)
            for handler in getattr(stmt, 'handlers', ()):
                handler.body = body(handler.body) or [ast.Pass()]
            out.append(stmt)
        return out
    node = _strip_docstrings(copy.deepcopy(new))
    node.body = body(node.body) or [ast.Pass()]
    for (kind, head), seen in counts.items():
        assert seen == spec[kind][head], 'default-off statement count differs: %s %r %d' % (kind, head, seen)
    assert unreachable[0] == spec.get('unreachable', 0), 'unreachable statement count differs: %d' % unreachable[0]
    text = _text(node)
    for sub in spec.get('subs', ()):
        a, b, n = sub if len(sub) == 3 else (sub[0], sub[1], 1)
        assert text.count(a) == n, 'default-off substitution count differs: %r %d' % (a, text.count(a))
        text = text.replace(a, b)
    target = old if isinstance(old, str) else _text(old)
    assert text == target, 'default-off body differs'
    return hashlib.sha256(target.encode()).hexdigest()


def removals(spec):
    """Each declared removal, named and counted, for the record."""
    rows = []
    for kind in ('drop', 'off', 'on', 'drop_prefix'):
        rows += [dict(kind=kind, statement=head, count=n) for head, n in spec.get(kind, {}).items()]
    rows += [dict(kind='subs', statement=sub[0], replacement=sub[1], count=sub[2] if len(sub) == 3 else 1)
             for sub in spec.get('subs', ())]
    return rows


def fused_true_sites():
    """Every engine call passing fused=True: the qsa_operation premise. Returns (file, enclosing function) rows."""
    rows = []
    for file in ENGINE:
        for qual, node in _units((ROOT / 'patches/s5_engine' / file).read_text(), file):
            for n in ast.walk(node):
                if isinstance(n, ast.Call) and any(k.arg == 'fused' and isinstance(k.value, ast.Constant) and k.value.value is True
                                                   for k in n.keywords):
                    rows.append((file, qual))
    return rows


def default_off():
    result, table = {}, {}
    for file, specs in PROJECTIONS.items():
        old = show('patches/s5_engine/' + file)
        new = (ROOT / 'patches/s5_engine' / file).read_text()
        for name, spec in specs.items():
            result[file + ':' + name] = project(function(new, name), function(old, name), spec)
            table[file + ':' + name] = removals(spec)
    for (file, name), (spec, target) in NEW_OFF.items():
        new = (ROOT / 'patches/s5_engine' / file).read_text()
        result[file + ':' + name + ' (off path)'] = project(function(new, name), target, spec)
        table[file + ':' + name + ' (off path)'] = removals(spec)
    # Every changed pre-existing function has a projection: nothing is left to the reader.
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
    assert not any(getattr(n, 'name', None) == '_action_native_qsa_fused' for n in cls.body), 'parent has the session action'
    sites = fused_true_sites()
    assert sites == [('engine_driver.py', 'NativeSegmentDriver._fused_agree')], sites
    for file, name, added in CLASS_FIELDS:
        new, old = (ROOT / 'patches/s5_engine' / file).read_text(), show('patches/s5_engine/' + file)
        result[file + ':class:' + name] = class_fields(new, old, name, added)
    return dict(verdict='PASS', legacy_function_bodies=result, functions=len(result), removals=table,
                premises=dict(session_parent_has_no_action=True, fused_true_only_in=sites),
                meaning='source projection with FLASH_S5_QSA_FUSED unset: every c3f1f64 function Step A changes is its AST once the '
                        'declared statements are removed; the new helper on that path returns its parent value; class fields '
                        'default to the parent behaviour. See not_projected for what this cannot show.')


def class_fields(new, old, name, added):
    """The class's non-method body is the parent's plus exactly ``added`` (docstrings ignored)."""
    def rows(text):
        node = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == name)
        out = [ast.unparse(item) for item in node.body if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if out and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            out = out[1:]
        return ast.unparse(ast.ClassDef(name=node.name, bases=node.bases, keywords=node.keywords, body=[ast.Pass()],
                                        decorator_list=node.decorator_list, type_params=[])), out
    head_new, body_new = rows(new)
    head_old, body_old = rows(old)
    assert head_new == head_old, 'class header changed: ' + name
    assert [r for r in body_new if r not in added] == body_old, 'class body differs beyond the declared fields: ' + name
    assert sorted(r for r in body_new if r in added) == sorted(added), 'declared class fields missing: ' + name
    return hashlib.sha256('\n'.join(body_old).encode()).hexdigest()


def _unreached_names():
    out = {}
    for file, names in UNREACHED.items():
        for name in names:
            out[name.split(':', 1)[-1].split('.')[-1]] = (file, name)
    return out


def _units(text, file):
    """(qualified name, node) for every function and module statement, classes split into methods and body."""
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name, node
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield node.name + '.' + item.name, item
                else:
                    yield 'class:' + node.name, item
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        elif isinstance(node, ast.Assign) and [ast.unparse(t) for t in node.targets] == ['__all__']:
            continue
        else:
            yield 'module:' + type(node).__name__, node


def _references(node, names):
    """(name, guards) for each reference to ``names`` (Name, Attribute or an exact string constant).

    ``guards`` are the tests that must hold for the reference to run: the test of each enclosing ``if`` whose body holds it, and
    ``after-return <test>`` for each earlier ``if <test>`` in an enclosing block whose body ends in return/raise and has no else.
    """
    found = []

    def block(stmts, guards):
        for stmt in stmts:
            walk(stmt, guards)
            if (isinstance(stmt, ast.If) and not stmt.orelse and stmt.body
                    and isinstance(stmt.body[-1], (ast.Return, ast.Raise))):
                guards = guards | {'after-return ' + ast.unparse(stmt.test)}

    def walk(n, guards):
        if isinstance(n, ast.Name) and n.id in names:
            found.append((n.id, guards))
        elif isinstance(n, ast.Attribute) and n.attr in names:
            found.append((n.attr, guards))
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in names:
            found.append((n.value, guards))
        if isinstance(n, ast.If):
            walk(n.test, guards)
            block(n.body, guards | {ast.unparse(n.test)})
            block(n.orelse, guards)
            return
        for field, value in ast.iter_fields(n):
            if isinstance(value, list) and value and all(isinstance(v, ast.stmt) for v in value):
                block(value, guards)
            elif isinstance(value, list):
                for v in value:
                    if isinstance(v, ast.AST):
                        walk(v, guards)
            elif isinstance(value, ast.AST):
                walk(value, guards)
    walk(node, frozenset())
    return found


def reach(texts=None):
    """Each UNREACHED definition is referenced only from UNREACHED code or under its declared guard.

    Changed functions with a projection are proven separately (their default-off body is the parent's text, which names no Step A
    definition); every other function and module statement of the changed modules is checked as it is.
    """
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
                assert guards & allowed, 'Step A only definition reachable default-off: %s:%s -> %s %s' % (
                    file, qual, name, sorted(guards))
                guarded.append('%s:%s -> %s under %s' % (file, qual, name, sorted(guards & allowed)[0]))
            checked += 1
    return dict(verdict='PASS', unreached=sorted('%s:%s' % v for v in names.values()), units_checked=checked,
                guarded_references=sorted(guarded))


def negative_controls():
    d = json.loads(DECLARATION.read_text())
    p = 'patches/s5_engine/qsa_driver.py'

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
    twice = dict(spec, off={TX_FUSED: 2})
    # An unchanged method that calls a Step A only method without a guard.
    tree = ast.parse(engine)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NativeSegmentDriver')
    tick = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'control_tick')
    tick.body.insert(0, ast.parse('self._fused_watchdog()').body[0])
    reached = ast.unparse(tree)
    own = ('native_overlap_control.py', '_qsa_fused')

    def undeclared_helper():
        rows = set(new_definitions(dict(d, definitions=dict(d['definitions'], **{
            p: d['definitions'][p] + ['QsaLayerDriver.step_a_undeclared']})))) - class_body_rows()
        assert rows == {(f, n) for f, names in NEW_HELPERS.items() for n in names}, 'undeclared new helper'
    return {'undeclared-function': refused(lambda: prove(show(p), (ROOT / p).read_text() + '\ndef step_a_undeclared():\n pass\n',
                                                          p, d['definitions'][p])),
            'undeclared-path': refused(lambda: check_paths(paths() + ['A\tpatches/s5_engine/step_a_undeclared.py'], d)),
            'undeclared-overlay': refused(lambda: prove('x=1\n', 'x=2\n', 'core/scheduler.py', [])),
            'undeclared-default-off-statement': refused(lambda: project(
                begin, function(old_engine, 'NativeSegmentDriver._begin'), spec)),
            'missing-default-off-declaration': refused(lambda: project(
                function(engine, 'NativeSegmentDriver._begin'), function(old_engine, 'NativeSegmentDriver._begin'), short)),
            'miscounted-default-off-statement': refused(lambda: project(
                function(engine, 'NativeSegmentDriver._begin'), function(old_engine, 'NativeSegmentDriver._begin'), twice)),
            'unguarded-step-a-reference': refused(lambda: reach({'engine_driver.py': reached})),
            'wrong-off-value': refused(lambda: project(
                function((ROOT / 'patches/s5_engine' / own[0]).read_text(), own[1]), 'def f():\n    return True', NEW_OFF[own][0])),
            'undeclared-new-helper': refused(undeclared_helper)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--overlay', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--draft-declaration', action='store_true',
                        help='print the declaration the tree implies (for review), and exit')
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
    print('STEP_A_FREEZE ' + r['verdict'])
