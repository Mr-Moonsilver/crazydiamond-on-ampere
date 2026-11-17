#!/usr/bin/env python3
"""alpha-next freeze proof: composition, the features' own freeze records, the projection of alpha-next's own engine changes,
negative controls and the generated overlay (CPU, git; no box action).

The alpha trunk method, in three layers:

  composition  tools/alpha_next_composition.py prove(): every composed path is a parent's bytes, a clean merge or a declared
               resolution, from the stop-fix tip b82b832a to the stage-1 anatomy merge 5990af80.
  features     each composed feature's own freeze record, pinned by sha256 and verdict PASS (held-g, Step B, C0, stage-1 anatomy).
               Each proved its feature default-off against its own parent. The engine delta from the commit each record proved to
               the commit composed here is listed and may only be the review pin or a module in NEW (unreached default-off: C0's
               and stage-1's own reach claims); anything else refuses.
  own          alpha-next's commits after 5990af80. Every changed definition is one of:
                 PROJECTIONS   a pre-existing function; step_a_freeze.project removes the named statements and the result must be
                               5990af80's exact AST (docstrings ignored);
                 RECEIPTS      a new always-on receipt helper (stop receipts, item 1). It is not default-off: the shape check
                               allows it only its own counters, the journal, its log line, and reads; every body that can fail
                               catches and counts;
                 AUTO_ARM      a new helper reached only under the bridge's ``self._memory_receipts`` (FLASH_S5_AUTO_ARM=1; reach);
                 STAGE_LOCAL   a definition in stage_progress.py, reached only through StagedSegmentDriver, which s5_startup builds
                               only with FLASH_S5_STAGE_LOCAL=1 (reach over s5_startup);
                 CLASS_FIELDS / MODULE_STATEMENTS  named rows.
               Anything else refuses (negative controls).

The overlay tamper controls are tools/alpha_next_overlay_tamper.py's (over the built overlay). overlay_level proves the built overlay against Step B's: each changed and each new module is make_s5_overlap.package(committed
source) byte for byte, the parent's modules are package(Step B source 1a29d84f), and the launcher, generated vllm sources,
plugins and native libraries are byte-identical. With any switch on, alpha-next is not claimed equal to anything here.

    python3 tools/alpha_next_freeze.py --out FILE [--parent OVERLAY --overlay OVERLAY]
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
from native37_robust_freeze import changed  # noqa: E402
from build_native30_held import verify  # noqa: E402
import alpha_next_composition as composition  # noqa: E402
from step_a_freeze import _packaged, _references, _units, class_fields, function, project, removals  # noqa: E402

ENGINE = 'patches/s5_engine/'
REVIEW = ENGINE + 'native_candidate_review.json'
COMPOSED = '5990af80'                     # the last composition step (stage-1 anatomy merge)
STEP_B_SOURCE = '1a29d84f'                # the source of the Step B build that is this build's parent
FEATURES = Path('<hub-dir>/agents/flash-next/tmp')
NEW = {'e3/handler_anatomy.py', 'e3/stage_progress.py', 'e3/stage_wire.py'}
META = {'SHA256SUMS', 'S5-OVERLAP-GENERATION.json', 'e3/native-implementation-closure.json',
        'e3/native-overlap-readiness.json', 'e3/decode-probe-evidence.json', 'core/SHA256SUMS'}

# (name, record, sha256, proved commit, composed commit): the record proved `proved`; `composed` is what was merged or picked.
FEATURE_RECORDS = (
    ('held-g', 'native41-progress-held-g/final/build/freeze-proof.json',
     'f805cdec746d', '0efd7b183063fd61f1ac931d728403bf96e5e4e6', 'a9a3e8e6'),
    ('step-b', 'step-b/pack/e/build/step-b-freeze.json',
     '6a0aaf3e7518', '1a29d84f64fa2e2cccec5d79de2500fee5f4df89', '66b5780e'),
    ('c0', 'c0c1/pack/build-l/alpha-c0-freeze.json',
     'ca6912048a32', 'cf0e3016825e4a96d8a09166fdd982cd7429f976', 'f37ee41a'),
    ('stage1-anatomy', 'stage1-anatomy/freeze/freeze-f708f082.json',
     '88c3b8c813f4', 'f708f0828fd089c247e3631f77e3064fcef6cafb', 'c90a7f8e'),
)

_KV_ROUTE = (("emit_receipt_kv('flash-s5 ", "logging.getLogger(__name__).info('flash-s5 "), ("' % (", "', "),
             ('), logging.getLogger(__name__))', ')'))
PROJECTIONS = {
    'engine_driver.py': {
        'NativeProgress.end': dict(subs=(
            ('emit_receipt_lines(self.trace.lines(**self.identity or {}, mode=self.mode, open_rows=len(self.native)), logger)',
             'for text in self.trace.lines(**self.identity or {}, mode=self.mode, open_rows=len(self.native)):\n            logger.info(text)'),
            ('emit_receipt_lines(self.steps.lines(**self.identity or {}, mode=self.mode, idle=self.idle, replies=self.replies), logger)',
             'for text in self.steps.lines(**self.identity or {}, mode=self.mode, idle=self.idle, replies=self.replies):\n            logger.info(text)'))),
        'NativeSegmentDriver.__init__': dict(drop={'self._stop_partial_replays = 0': 1, 'self._stop_bound_exhausted = 0': 1}),
        'NativeSegmentDriver._drive_partial_stop_submission': dict(drop={'self._note_stop_partial_replay()': 1}),
        'NativeSegmentDriver._stop_round': dict(drop={'self._note_stop_bound_exhausted()': 1}),
        'NativeSegmentDriver._fused_stop_round': dict(drop={'self._note_stop_bound_exhausted()': 1}),
        'NativeSegmentDriver._status_record': dict(drop={"result['stop_receipts'] = self.stop_receipts()": 1}),
        'NativeSegmentDriver.overlap_evidence': dict(drop={
            "record['measurement']['stop_receipts'] = {'partial_replays': self._stop_partial_replays, "
            "'bound_exhausted': self._stop_bound_exhausted, 'basis': 'engine-lifetime-monotonic-count; recorded, never consulted'}": 1}),
    },
    'worker_bridge.py': {
        'NativeWorkerBridge.open': dict(drop={'self._peak_open()': 1}),
        'NativeWorkerBridge._release_lane': dict(drop={'self._peak_receipt(transaction)': 1}),
        'NativeWorkerBridge._memory_receipt': dict(drop={'from native_measurement import emit_receipt_kv': 1}, subs=_KV_ROUTE),
    },
    # PLAN 2 routing (writer not consulted): each emit call is the parent's logger call(s) with the same text and arguments.
    'qsa_operation.py': {
        'QsaPrefillOperation._alloc_receipt': dict(drop={'from native_measurement import emit_receipt_kv': 1}, subs=_KV_ROUTE),
    },
    'executor_hooks.py': {
        'worker_rpc_begin': dict(subs=(('emit_receipt_lines(state[0].lines(), _progress_logger())',
                                        'for text in state[0].lines():\n                _progress_logger().info(text)'),)),
    },
    's5_startup.py': {
        # install_receipt_writer() constructs the process writer and opens or writes nothing (WRITER_SHAPE); AUTO_ARM unset.
        'install_engine': dict(drop={'from native_measurement import install_receipt_writer, receipt_path': 1,
                                     'install_receipt_writer()': 1},
                            subs=(("\n    if auto == '1':\n        try:\n            from vllm.logger import init_logger\n"
                                   "        except ImportError:\n            import logging\n            init_logger = logging.getLogger\n"
                                   "        init_logger(__name__).info('flash-s5 startup receipt_path=%s', receipt_path())", ''),)),
        'install_native_lane': dict(drop={'from native_measurement import install_receipt_writer': 1, 'install_receipt_writer()': 1}),
        # DeviceSpans is built only with FLASH_S5_NATIVE_PROGRESS (held-g's reach); emit_lines is the routed group emit.
        '_device_spans': dict(subs=(('from native_measurement import DeviceSpans, emit_receipt_lines', 'from native_measurement import DeviceSpans'),
                                    (', emit_lines=lambda texts: emit_receipt_lines(texts, _progress_logger()))', ')'))),
    },
    'native_measurement.py': {
        # emit_lines None (the parent's construction): the parent's per-line emit.
        'DeviceSpans.__init__': dict(drop={'self.emit_lines = emit_lines': 1},
                                     off={'if emit_lines is not None and (not callable(emit_lines)):': 1},
                                     subs=((', emit_lines=None):', '):'),)),
        'DeviceSpans.release': dict(off={'if self.emit_lines is not None:': 1}, subs=(
            ('texts = self.trace.lines(txn=transaction.txn_seq, request_generation=transaction.request_generation, '
             'chunk_start=transaction.chunk_start, chunk_end=transaction.chunk_end, stage=self.stage)\n        for text in texts:',
             'for text in self.trace.lines(txn=transaction.txn_seq, request_generation=transaction.request_generation, '
             'chunk_start=transaction.chunk_start, chunk_end=transaction.chunk_end, stage=self.stage):'),)),
        # Not equal: the part budget shrinks (PROGRESS_LINE_BYTES 12000 -> 3968, minus the writer's identity); NOT_PROJECTED.
        'ProgressTrace.lines': dict(subs=(('PROGRESS_LINE_BYTES - RECEIPT_IDENTITY_BYTES - ', 'PROGRESS_LINE_BYTES - '),)),
    },
}

# Always-on receipt helpers: the self attributes each may write, and the calls each may make (unparsed callee text).
_RECEIPT_CALLS = {'self._record', 'self._stop_receipt', 'self._partial_stop_state', 'self._monotonic_ns', 'json.dumps',
                  '_progress_logger().info', '_progress_logger', 'emit_receipt_lines', 'dict', 'sorted', 'set', 'getattr', 'str', 'bool', 'int', 'entry.get'}
RECEIPTS = {
    'engine_driver.py': {
        'NativeSegmentDriver._note_stop_partial_replay': {'_stop_partial_replays'},
        'NativeSegmentDriver._note_stop_bound_exhausted': {'_stop_bound_exhausted'},
        'NativeSegmentDriver._stop_receipt': set(),
        'NativeSegmentDriver._partial_stop_state': set(),
        'NativeSegmentDriver.stop_receipts': set(),
    },
}
AUTO_ARM_GUARD = 'self._memory_receipts'
AUTO_ARM = {
    'worker_bridge.py': ('NativeWorkerBridge._peak_open', 'NativeWorkerBridge._peak_receipt', 'peak_reset', 'peak_receipt'),
}
STAGE_LOCAL_FILE = 'stage_progress.py'
ANATOMY_FILE = 'handler_anatomy.py'
# PLAN 2 writer: installed in every engine and worker process; opens and writes nothing until the first receipt (WRITER_SHAPE), and
# every receipt is emitted only by a site its own feature's switch guards (progress, auto-arm, anatomy, stage-local).
WRITER = {
    'native_measurement.py': ('class:ReceiptWriter', 'ReceiptWriter.__init__', 'ReceiptWriter._emit_kv', 'ReceiptWriter._next',
                              'ReceiptWriter._put', 'ReceiptWriter._start', 'ReceiptWriter.stats', 'ReceiptWriter.write_kv',
                              'ReceiptWriter.write_lines', 'emit_receipt_kv', 'emit_receipt_lines', 'install_receipt_writer',
                              'installed_receipt_writer', 'receipt_path', 'receipt_sha'),
}
_ND_IMPORT = ('module:from native_measurement import PROGRESS_N, %sNativeMeasurementCensus, encode_report_vector, decode_report_vector, '
              'ControlPathCounters, QsaControlCounters, PROGRESS_DIRECT_METHODS, PROGRESS_META_BYTES, PROGRESS_META_KEY, PROGRESS_MODES, '
              'ProgressTrace, StepAnatomy, encode_progress_meta')
_EH_IMPORT = ('module:from native_measurement import PROGRESS_DIRECT_METHODS, PROGRESS_META_KEY, ProgressTrace, decode_progress_meta, '
              '%sencode_progress_meta, progress_mode')
CLASS_FIELDS = (('engine_driver.py', 'NativeSegmentDriver',
                 ["_STOP_ROW_KEYS = ('transaction', 'op_seq', 'segment_kind', 'accepted_ranks', 'pending_ranks', 'reason')"]),)
MODULE_STATEMENTS = {
    'engine_driver.py': {
        'module:import json': 'the stop receipt line is JSON',
        'module:STOP_RECEIPT_LINE': "constant 'flash-s5 stop ', read only by _stop_receipt",
        'module:STOP_RECEIPT_SCHEMA': "constant 's5-stop-receipt/1', read only by _stop_receipt",
    },
    'worker_bridge.py': {'module:PEAK_LINE': "constant 'flash-s5 memory-peak ', read only by _peak_receipt"},
    'executor_hooks.py': {_EH_IMPORT % 'emit_receipt_lines, ': 'adds emit_receipt_lines to an existing import (PLAN 2 routing)',
                          _EH_IMPORT % '': "the parent's row of the same import"},
    'native_measurement.py': {
        'module:import os': 'the writer duplicates fd 1 and writes with os.write',
        'module:PROGRESS_LINE_BYTES': 'part budget 12000 -> 3968 (one record within PIPE_BUF); NOT_PROJECTED',
        'module:RECEIPT_IDENTITY_BYTES': 'writer constant', 'module:RECEIPT_JSON_PREFIXES': 'writer constant',
        'module:RECEIPT_KV_PREFIXES': 'writer constant', 'module:RECEIPT_PATH_LINE': 'writer constant',
        'module:RECEIPT_PIPE_BUF_MIN': 'writer constant', 'module:RECEIPT_RECORD_BYTES': 'writer constant',
        'module:_RECEIPT_WRITER': 'the process writer slot, None until install_receipt_writer',
    },
}
MODULE_STATEMENTS['engine_driver.py'].update({
    _ND_IMPORT % 'emit_receipt_lines, ': 'adds emit_receipt_lines to an existing import (PLAN 2 routing)',
    _ND_IMPORT % '': "the parent's row of the same import"})

NOT_PROJECTED = [
    dict(what='stop receipts (item 1) are always on', detail='Not default-off. Every stop partial replay and every exhausted stop '
         'bound adds a counter increment, a journal row (stop_bound_exhausted is new; the census counts non-measurement kinds and '
         'ignores them) and, with progress receipts installed, one `flash-s5 stop ` line. _status_record and overlap_evidence carry '
         'the counts under a new key. The shape check bounds what the helpers may touch; the stop-fix matrix proves the stop paths.'),
    dict(what='peak receipt under FLASH_S5_AUTO_ARM=1', detail='Off unless FLASH_S5_AUTO_ARM=1, which box arms set: then each chunk resets '
         'the caching allocator peak at open (a host counter reset, no sync) and writes one `flash-s5 memory-peak ` line at release. '
         'Stock QSA_STAGE_TIMING also resets the peak; the line flags it. Not measured for cost.'),
    dict(what='feature records are pinned, not re-run', detail='Each feature freeze proved its feature against its own parent, not '
         'against this trunk. The composition proof shows the composed tree is those commits merged, with declared resolutions; '
         'it does not re-prove each feature default-off over the merged tree. held-g was proved on its own branch and is picked '
         'here with equal patch-ids.'),
    dict(what='C0 stop fix 2 and ring reading', detail='stage_progress.py changes run only with FLASH_S5_STAGE_LOCAL=1; with it on they '
         'change C0 behaviour (partial PREPARE replay on stop) and are proved by the stop-fix matrix, not here.'),
    dict(what='receipt writer (PLAN 2) is installed always', detail='install_engine and install_native_lane install the process '
         'ReceiptWriter in every engine and worker process. Installing does no I/O (writer_shape); every receipt site stays behind its own '
         "feature switch, and with a writer installed a receipt goes out as one os.write on a private dup(1) instead of the logger, so "
         'the line carries pid/seq/sha and not the logger prefix. Whether box stdout is a FIFO, and that every process installs it, is '
         'not proved on CPU (receipt_path line under AUTO_ARM; graders read unsafe paths as INC).'),
    dict(what='progress part budget', detail='PROGRESS_LINE_BYTES 12000 -> 3968 and handler_anatomy LINE_BYTES likewise: with progress or '
         'anatomy on, a chunk splits into more parts (one record within PIPE_BUF). Off, no part is written.'),
    dict(what='native_candidate_review.json', detail='Not code: the review pin moves by design. The build checks it against the tree.'),
]


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True)


def rev(name):
    return git('rev-parse', name).strip()


def show(path, commit=COMPOSED):
    return git('show', commit + ':' + path)


def own_paths():
    return git('diff', '--name-status', COMPOSED, 'HEAD', '--', ENGINE).splitlines()


def changed_modules():
    """e3/<name>.py for every engine source that differs from Step B's build source."""
    rows = git('diff', '--name-only', STEP_B_SOURCE, 'HEAD', '--', ENGINE).split()
    return {'e3/' + p[len(ENGINE):] for p in rows if p.endswith('.py') and p != REVIEW} - NEW


CHANGED = changed_modules()
RUNTIME = CHANGED


def features():
    rows = {}
    for name, path, sha, proved, composed in FEATURE_RECORDS:
        record = FEATURES / path
        data = record.read_bytes()
        assert hashlib.sha256(data).hexdigest().startswith(sha), 'feature freeze record changed: ' + name
        assert json.loads(data)['verdict'] == 'PASS', 'feature freeze record not PASS: ' + name
        assert subprocess.run(['git', 'merge-base', '--is-ancestor', composed, 'HEAD'], cwd=ROOT).returncode == 0 or \
            name == 'held-g', 'composed commit not in this history: ' + name
        delta = git('diff', '--name-only', proved, composed, '--', ENGINE).split()
        extra = [p for p in delta if p != REVIEW and 'e3/' + p[len(ENGINE):] not in NEW]
        assert not extra, ('engine changed after the feature freeze outside its new modules', name, extra)
        rows[name] = dict(record=str(record), sha256=hashlib.sha256(data).hexdigest(), proved=proved, composed=rev(composed),
                          post_freeze_engine_delta=delta)
    return rows


def receipt_shape(node, allowed_attrs):
    """A receipt helper writes only its own self attributes and calls only the receipt allowlist."""
    for n in ast.walk(node):
        targets = []
        if isinstance(n, ast.Assign):
            targets = n.targets
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
            targets = [n.target]
        for t in targets:
            if isinstance(t, ast.Attribute) and ast.unparse(t.value) == 'self':
                assert t.attr in allowed_attrs, 'receipt writes driver state: self.' + t.attr
            elif isinstance(t, ast.Subscript):
                assert ast.unparse(t) == "progress.counters['line_errors']", 'receipt writes a subscript: ' + ast.unparse(t)
        if isinstance(n, ast.Call):
            assert ast.unparse(n.func) in _RECEIPT_CALLS, 'receipt calls outside its allowlist: ' + ast.unparse(n.func)
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and ast.unparse(n.func) in ('self._record', 'json.dumps', '_progress_logger().info',
                                                                'emit_receipt_lines'):
            inside = any(isinstance(t, ast.Try) and any(c is n for c in ast.walk(t)) for t in ast.walk(node))
            assert inside, 'receipt call that can fail is not caught: ' + ast.unparse(n.func)
    return hashlib.sha256(ast.unparse(node).encode()).hexdigest()


def stage_local_reach(texts=None):
    """StagedSegmentDriver is named only in s5_startup under the FLASH_S5_STAGE_LOCAL guard, and nowhere else outside stage_progress."""
    texts = texts or {}
    found = []
    for path in git('ls-files', ENGINE).split():
        if not path.endswith('.py') or path.endswith('/' + STAGE_LOCAL_FILE):
            continue
        file = path[len(ENGINE):]
        text = texts.get(file) or (ROOT / path).read_text()
        if 'StagedSegmentDriver' not in text and 'stage_progress' not in text and 'ring_limiting_readers' not in text:
            continue
        for qual, node in _units(text, file):
            for name, guards in _references(node, {'StagedSegmentDriver', 'ring_limiting_readers'}):
                assert name == 'StagedSegmentDriver', 'a stage_progress helper named outside it: %s:%s' % (file, qual)
                found.append((file, qual, sorted(guards)))
    need = {"env.get('FLASH_S5_STAGE_LOCAL', '0') != '0'", 'stage_local_enabled(env)'}
    assert [(f, q) for f, q, _ in found] == [('s5_startup.py', 'install_engine')], ('StagedSegmentDriver named elsewhere', found)
    assert all(need <= set(g) for _, _, g in found), ('StagedSegmentDriver reachable outside its guard', found)
    return dict(verdict='PASS', references=[[f, q, sorted(need)] for f, q, _ in found], other_guards=len(found[0][2]) - len(need))


def anatomy_reach(texts=None):
    """handler_anatomy is named only in executor_hooks._anatomy, inside the else of the FLASH_S5_HANDLER_ANATOMY != '1' test."""
    texts = texts or {}
    found = []
    for path in git('ls-files', ENGINE).split():
        if not path.endswith('.py') or path.endswith('/' + ANATOMY_FILE):
            continue
        file = path[len(ENGINE):]
        text = texts.get(file) or (ROOT / path).read_text()
        if 'handler_anatomy' not in text:
            continue
        for qual, node in _units(text, file):
            for n in ast.walk(node):
                if (isinstance(n, ast.Name) and n.id == 'handler_anatomy') or (
                        isinstance(n, ast.Import) and any(a.name == 'handler_anatomy' for a in n.names)):
                    found.append((file, qual))
    assert {f for f in found} == {('executor_hooks.py', '_anatomy')}, ('handler_anatomy named elsewhere', sorted(set(found)))
    fn = function(texts.get('executor_hooks.py') or (ROOT / ENGINE / 'executor_hooks.py').read_text(), '_anatomy')
    gates = [n for n in ast.walk(fn) if isinstance(n, ast.If)
             and ast.unparse(n.test) == "os.environ.get('FLASH_S5_HANDLER_ANATOMY', '') != '1'"]
    assert len(gates) == 1, 'anatomy gate missing'
    inside = {id(m) for s in gates[0].orelse for m in ast.walk(s)}
    outside = [n for n in ast.walk(fn) if ((isinstance(n, ast.Name) and n.id == 'handler_anatomy') or isinstance(n, ast.Import))
               and id(n) not in inside]
    assert not outside, 'handler_anatomy reached outside the switch'
    return dict(verdict='PASS', site='executor_hooks.py:_anatomy', gate="FLASH_S5_HANDLER_ANATOMY == '1'")


_WRITER_IO = {'os.dup', 'os.write', 'open', 'os.open', 'os.fstat', 'self._write', 'self._start', 'self._put', 'self._emit_kv',
              'receipt_path'}


def writer_shape(texts=None):
    """Installing the writer does no I/O: install_receipt_writer constructs only, and ReceiptWriter.__init__ calls none of _WRITER_IO;
    the emit functions consult the writer only when installed, else the parent's logger call."""
    texts = texts or {}
    text = texts.get('native_measurement.py') or (ROOT / ENGINE / 'native_measurement.py').read_text()
    for name in ('ReceiptWriter.__init__', 'install_receipt_writer'):
        calls = {ast.unparse(n.func) for n in ast.walk(function(text, name)) if isinstance(n, ast.Call)}
        assert not calls & _WRITER_IO, ('writer install does I/O', name, sorted(calls & _WRITER_IO))
    for name, logged in (('emit_receipt_lines', 'logger.info'), ('emit_receipt_kv', 'logger.info')):
        calls = {ast.unparse(n.func) for n in ast.walk(function(text, name)) if isinstance(n, ast.Call)}
        assert logged in calls, ('emit without the parent logger fallback', name)
    return dict(verdict='PASS', install_io_free=True, emit_fallback='logger.info when no writer is installed')


def auto_arm_reach(texts=None):
    texts = texts or {}
    rows = []
    for file, names in AUTO_ARM.items():
        text = texts.get(file) or (ROOT / ENGINE / file).read_text()
        short = {n.split('.')[-1] for n in names}
        for qual, node in _units(text, file):
            if qual in names:
                continue
            for name, guards in _references(node, short):
                if name in ('peak_reset', 'peak_receipt') and qual in names:
                    continue
                assert AUTO_ARM_GUARD in guards, 'auto-arm receipt reachable outside %s: %s:%s -> %s' % (AUTO_ARM_GUARD, file, qual, name)
                rows.append('%s:%s -> %s under %s' % (file, qual, name, AUTO_ARM_GUARD))
    return dict(verdict='PASS', guarded_references=sorted(rows))


def own(texts=None):
    """alpha-next's own engine changes after the last composition step, every changed definition accounted for."""
    texts = texts or {}
    rows = own_paths()
    for row in rows:
        status, path = row.split('\t', 1)
        assert status == 'M' and (path.endswith('.py') or path == REVIEW), 'undeclared engine path: ' + row
    result, table, accounted = {}, {}, {}
    for row in rows:
        path = row.split('\t', 1)[1]
        if path == REVIEW:
            continue
        file = path[len(ENGINE):]
        new_text, old_text = texts.get(file) or (ROOT / path).read_text(), show(path)
        names = changed(old_text, new_text)
        old_names = {q for q, _ in _units(old_text, file)}
        for name in names:
            key = file + ':' + name
            if file == STAGE_LOCAL_FILE:
                accounted[key] = 'stage-local (FLASH_S5_STAGE_LOCAL=1 only; stage_local_reach)'
            elif file == ANATOMY_FILE:
                accounted[key] = 'handler anatomy (FLASH_S5_HANDLER_ANATOMY=1 only; anatomy_reach)'
            elif name in WRITER.get(file, ()):
                assert name not in old_names, 'writer definition existed in the parent: ' + key
                accounted[key] = 'receipt writer (writer_shape)'
            elif name in PROJECTIONS.get(file, {}):
                spec = PROJECTIONS[file][name]
                try:
                    result[key] = project(function(new_text, name), function(old_text, name), spec)
                except AssertionError as exc:
                    raise AssertionError('%s: %s' % (key, exc)) from None
                table[key] = removals(spec)
                accounted[key] = 'projected'
            elif name in RECEIPTS.get(file, {}):
                assert name not in old_names, 'receipt helper existed in the parent: ' + key
                result[key] = receipt_shape(function(new_text, name), RECEIPTS[file][name])
                accounted[key] = 'always-on receipt (shape)'
            elif name in AUTO_ARM.get(file, ()):
                assert name not in old_names, 'auto-arm helper existed in the parent: ' + key
                accounted[key] = 'auto-arm only (auto_arm_reach)'
            elif name in MODULE_STATEMENTS.get(file, {}):
                accounted[key] = MODULE_STATEMENTS[file][name]
            elif name.startswith('class:') and any(f == file and 'class:' + c == name for f, c, _ in CLASS_FIELDS):
                fields = next(a for f, c, a in CLASS_FIELDS if f == file and 'class:' + c == name)
                result[key] = class_fields(new_text, old_text, name[len('class:'):], fields)
                accounted[key] = 'class fields'
            else:
                raise AssertionError('undeclared own change: ' + key)
    declared = ({f + ':' + n for f, s in PROJECTIONS.items() for n in s} | {f + ':' + n for f, s in RECEIPTS.items() for n in s}
                | {f + ':' + n for f, s in AUTO_ARM.items() for n in s} | {f + ':' + n for f, s in MODULE_STATEMENTS.items() for n in s}
                | {f + ':class:' + c for f, c, _ in CLASS_FIELDS} | {f + ':' + n for f, s in WRITER.items() for n in s})
    unused = sorted(declared - set(accounted))
    assert not unused, ('declared but not changed', unused)
    return dict(verdict='PASS', base=rev(COMPOSED), paths=rows, accounted=accounted, bodies=result, removals=table)


def source_level():
    proof = composition.prove()
    assert proof['verdict'] == 'PASS' and proof['composed'] == rev(COMPOSED)
    return dict(schema='alpha-next-source/1', verdict='PASS', head=rev('HEAD'), composition=proof, features=features(),
                own=own(), stage_local_reach=stage_local_reach(), auto_arm_reach=auto_arm_reach(),
                anatomy_reach=anatomy_reach(), writer_shape=writer_shape(),
                compiled=compiled(), not_projected=NOT_PROJECTED,
                declaration_sha256=hashlib.sha256((ROOT / 'tools/alpha_next_composition.json').read_bytes()).hexdigest())


def compiled():
    out = []
    for path in sorted(git('ls-files', ENGINE).split()):
        if path.endswith('.py'):
            compile((ROOT / path).read_bytes(), path, 'exec')
            out.append(path)
    return len(out)


def negative_controls():
    def refused(fn):
        try:
            fn()
        except AssertionError:
            return 'REFUSED'
        raise AssertionError('undeclared change accepted')

    driver = (ROOT / ENGINE / 'engine_driver.py').read_text()
    bridge = (ROOT / ENGINE / 'worker_bridge.py').read_text()
    startup = (ROOT / ENGINE / 's5_startup.py').read_text()

    def swap(text, a, b):
        assert text.count(a) == 1, a
        return text.replace(a, b)
    rows = {
        'undeclared-statement-in-a-projected-function': refused(lambda: own({'engine_driver.py': swap(
            driver, '        self._stop_partial_replays = 0\n', '        self._stop_partial_replays = 0\n        self.phase = None\n')})),
        'undeclared-new-function': refused(lambda: own({'engine_driver.py': driver + '\n\ndef alpha_next_undeclared():\n    pass\n'})),
        'receipt-writes-driver-state': refused(lambda: own({'engine_driver.py': swap(
            driver, '            self._stop_partial_replays += 1\n', '            self._stop_partial_replays += 1\n            self.active = False\n')})),
        'receipt-calls-the-driver': refused(lambda: own({'engine_driver.py': swap(
            driver, '            self._stop_partial_replays += 1\n', '            self._stop_partial_replays += 1\n            self._send_segment(None, ())\n')})),
        'receipt-log-uncaught': refused(lambda: receipt_shape(ast.parse(
            'def f(self):\n    json.dumps({})\n').body[0], set())),
        'peak-receipt-unguarded': refused(lambda: auto_arm_reach({'worker_bridge.py': swap(
            bridge, "        self._retained_cancel_ack = None\n", "        self._retained_cancel_ack = None\n        self._peak_open()\n")})),
        'staged-driver-unguarded': refused(lambda: stage_local_reach({'s5_startup.py': swap(
            startup, '    driver_class = NativeSegmentDriver\n',
            '    from stage_progress import StagedSegmentDriver\n    driver_class = StagedSegmentDriver\n')})),
        'feature-record-tampered': refused(lambda: _feature_tamper()),
        'writer-install-does-io': refused(lambda: writer_shape({'native_measurement.py': swap(
            (ROOT / ENGINE / 'native_measurement.py').read_text(), '        self._lock = threading.Lock()\n',
            '        self._lock = threading.Lock()\n        self.fd = os.dup(1)\n')})),
        'anatomy-imported-unguarded': refused(lambda: anatomy_reach({'executor_hooks.py': swap(
            (ROOT / ENGINE / 'executor_hooks.py').read_text(), '    global _ANATOMY\n',
            '    global _ANATOMY\n    import handler_anatomy\n')})),
        'routed-emit-with-other-text': refused(lambda: own({'qsa_operation.py': swap(
            (ROOT / ENGINE / 'qsa_operation.py').read_text(), "'flash-s5 native-alloc rank=%d ", "'flash-s5 native-alloc  rank=%d ")})),
    }
    assert set(rows.values()) == {'REFUSED'}, rows
    return rows


def _feature_tamper():
    global FEATURE_RECORDS
    saved = FEATURE_RECORDS
    try:
        FEATURE_RECORDS = ((saved[0][0], saved[0][1], '000000000000') + saved[0][3:],) + saved[1:]
        features()
    finally:
        FEATURE_RECORDS = saved


def overlay_level(parent, candidate):
    parent, candidate = Path(parent), Path(candidate)
    before, after = verify(parent), verify(candidate)
    assert set(after) - set(before) == NEW, ('overlay files added', sorted(set(after) - set(before)))
    assert not set(before) - set(after), 'overlay file removed'
    moved = {n for n in before if before[n] != after[n]} | {'SHA256SUMS'}
    runtime = {n for n in CHANGED if n in after}
    assert moved <= runtime | META, sorted(moved - (runtime | META))
    for path in sorted(runtime | NEW):
        name = path.split('/', 1)[1]
        assert (candidate / path).read_text() == _packaged((ROOT / ENGINE / name).read_text()), \
            'overlay module differs from the packaged committed source: ' + path
    for path in sorted(runtime):
        name = path.split('/', 1)[1]
        assert (parent / path).read_text() == _packaged(show(ENGINE + name, STEP_B_SOURCE)), \
            'parent overlay module differs from the packaged Step B source: ' + path
    assert (candidate / 'run-flash-s5-test.sh').read_bytes() == (parent / 'run-flash-s5-test.sh').read_bytes(), 'launcher changed'
    generated = sorted(n for n in before if n.startswith('e3_sources/'))
    assert generated and all(before[n] == after[n] for n in generated), 'generated vllm sources changed'
    native = [p for p in before if p.endswith('.so') or '/native30_plugins/' in p or p == 'qsa.py']
    assert native and all(before[n] == after[n] for n in native), 'native library or plugin changed'
    return dict(parent=str(parent), moved=sorted(moved), added=sorted(NEW), generated_sources_byte_identical=len(generated),
                native_libraries_byte_identical=True, launcher_byte_identical=True,
                launcher_sha256=hashlib.sha256((candidate / 'run-flash-s5-test.sh').read_bytes()).hexdigest(),
                modules_equal_packaged_committed_source=sorted(runtime | NEW),
                parent_modules_equal_packaged_step_b_source=sorted(runtime), unpackaged_changed=sorted(CHANGED - runtime))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--parent', type=Path)
    ap.add_argument('--overlay', type=Path)
    a = ap.parse_args()
    r = dict(schema='alpha-next-freeze/1', verdict='PASS', source=source_level(), negative_controls=negative_controls())
    if a.overlay:
        r['overlay'] = overlay_level(a.parent, a.overlay)
    with a.out.open('x') as f:
        json.dump(r, f, sort_keys=True, indent=1)
        f.write('\n')
    print('ALPHA_NEXT_FREEZE PASS')
