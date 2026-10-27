#!/usr/bin/env python3
"""Qualification orchestration driver (Flash-Next E3).

This module builds the controlled experiment that L's raw graders consume.  It
does **not** grade: every phase returns the raw evidence in the named schema
(``s5-qsa-direct-bytes/1``, ``s5-teacher-decode/2``,
``s5ro-replay-observer/1`` + ``s5-engine-accounting/1``) plus the retained
artifact digest.  Grading stays in L's ``e3_q18a_qualification`` /
``e3_teacher_decode_rule4`` / ``engine_offers``.

Phases are one-shot and non-overlapping:

``row1``
    Hand one request to the explicit native-source provider seam and retain the
    raw records it returns.  Absent provider -> INCONCLUSIVE, never PASS.

``rule4_controls``
    One actual teacher sequence on the reviewed instrument and the frozen
    teacher workload: A1 (full), A2 (full), B (split), C (full), 8 streams,
    768 continuation with measured positions 256..511, original ordered top20.
    The raw schema/2 artifact is stored exactly once.  No retry, no picking.

``rule4_candidate``
    Read the *retained* controls artifact by canonical digest and return it as
    the candidate.  It never sends a teacher request and refuses a missing,
    ambiguous or mismatched digest and any request to recollect.

``row4``
    A separate declared **short** placement window (40 replays / 40 tokens) over
    a live cohort, driven through the real all-rank pause/drain/transition
    actions plus ``observer_begin``/``observer_end``.  Raw graph receipts and
    engine-accounting receipts are retained separately, per rank; native-read
    and native-prefill are explicit provider seams (absent -> unavailable).

Every refusal and partial artifact is written once under the supplied immutable
output directory.  Ownership is never dropped: cleanup uses only the engine's
typed all-rank drain/transition acknowledgements, never a client-side resume or
a fabricated success.
"""
import copy
import hashlib
import json
import os
import sys
import threading
import time
import traceback
from dataclasses import replace

import e3_decode_correctness as correctness
import e3_teacher_decode_client as teacher_client
import e3_w1_decode_driver as protocol

_PATCHES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'patches', 's5_engine')
if _PATCHES not in sys.path:
    sys.path.insert(0, _PATCHES)
from native_qualification_frontend import (  # noqa: E402
    FRONTEND_PATH, WORKER_PATH)

PHASES = ('row1', 'rule4_controls', 'rule4_candidate', 'row4')
PHASE_ARTIFACTS = {'row1': 'row1.json', 'rule4_controls': 'rule4_controls.json',
                   'rule4_candidate': 'rule4_candidate.json',
                   'row4': 'row4.json'}
TEACHER_SEQUENCE = ['full-A1', 'full-A2', 'split-B', 'full-C']
TEACHER_ROUTES = {'full-A1': 'full', 'full-A2': 'full',
                  'split-B': 'split', 'full-C': 'full'}
TEACHER_SCHEMA = 's5-teacher-decode/2'
ROW1_SCHEMA = 's5-native-row1-store/1'
ROW1_PHASE = 'row1-native'
ROW1_ACTIONS = ('row1_configure', 'row1_evidence')
ROW1_PACKET_FIELDS = frozenset({
    'schema', 'run_id', 'rank', 'worker_incarnation', 'engine_epoch',
    'route_generation', 'disabled_route_generation', 'local_layers', 'enabled', 'raw'})
ROW_MAPPING_SCHEMA = 's5-teacher-row-mapping/1'
ROW_BASIS = 'sampler-invocation/1'
ROW1_PLAN_SCHEMA = 's5-native-row1-request-plan/1'
ROW1_PLAN_FIELDS = frozenset({'schema', 'workload', 'streams', 'run_index',
                              'request_sha256', 'phase', 'prompts'})
PREFILL_PLAN_SCHEMA = 's5-native-prefill-request-plan/1'
PREFILL_PLAN_FIELDS = frozenset({'schema', 'workload', 'streams', 'run_index',
                                 'request_sha256'})
STREAMS = 8
MEASURED = 256
NATIVE_READ_BYTES = 64 * 1024 * 1024
PLACEMENT_STREAMS = 8
PLACEMENT_REPLAYS = 40
PLACEMENT_WINDOW_TOKENS = 40
PLACEMENT_GENERATION_BUDGET = 64
# The placement split is its own declared native phase; stock preparation and
# ordinary decode phases keep the admission intent off.
PLACEMENT_PHASE = 'w2-overlap'
STOCK_PHASE = 'full-prefill-solo'
RANKS = (0, 1, 2, 3)
SNAPSHOT_MAX_BYTES = 2 * 1024 ** 3
MODEL_SETTING_KEYS = ('max_model_len', 'max_num_seqs', 'gpu_memory_utilization',
                      'enforce_eager', 'kv_cache_dtype')


def exception_details(exc, control=None):
    """Bounded collector diagnostics, with the known credential redacted."""
    message = str(exc).replace('\n', ' ').replace('\r', ' ')
    for value in (getattr(control, 'key', None), os.environ.get('VLLM_API_KEY')):
        if type(value) is str and value:
            message = message.replace(value, '***')
    return {'error_type': type(exc).__name__, 'error_message': message[:512],
            'error_stack': [{'file': os.path.basename(frame.filename),
                             'line': frame.lineno, 'function': frame.name}
                            for frame in traceback.extract_tb(exc.__traceback__)[-16:]]}


class QualificationError(RuntimeError):
    """Base class for a driver refusal (never graded as PASS)."""


class QualificationRefused(QualificationError):
    """A phase was attempted and refused; the refusal record is retained."""

    def __init__(self, code, *, record=None, cleanup=None, evidence=None):
        super().__init__(code)
        self.code = code
        self.record = record
        self.cleanup = cleanup
        self.evidence = evidence


class PhaseUnavailable(QualificationRefused):
    """A required provider/construction is absent (INCONCLUSIVE)."""


class PhaseConflict(QualificationRefused):
    """Another phase owns the engine/window (no concurrent tracing)."""


def canonical_sha(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _text(value):
    return type(value) is str and bool(value.strip())


def _digest(value):
    return (_text(value) and len(value) == 64
            and all(char in '0123456789abcdef' for char in value))


def _positive(value):
    return type(value) is int and value >= 1


def _require(condition, code):
    if not condition:
        raise QualificationRefused(code)


def freeze_row1_plan(plan, *, run_id, model):
    """Validate and freeze the queue-owned minimal request plan.

    The plan is the *only* source of the row1 request configuration: a caller
    that does not supply one gets a named INCONCLUSIVE, never a built-in
    workload or a relabelled stock/VM fixture.  ``workload`` is the exact
    correctness manifest object from the queue plan; prompts are derived from
    it with the reviewed cohort/salt helpers and never persisted.
    """
    if plan is None:
        raise PhaseUnavailable('missing-row1-request-plan')
    if type(plan) is not dict:
        raise QualificationRefused('row1-request-plan-not-an-object')
    unknown = set(plan) - ROW1_PLAN_FIELDS
    if unknown:
        raise QualificationRefused(
            'row1-request-plan-unknown-fields',
            evidence={'unknown': sorted(unknown)})
    if plan.get('schema') != ROW1_PLAN_SCHEMA:
        raise QualificationRefused('row1-request-plan-schema-differs')
    if plan.get('phase', ROW1_PHASE) != ROW1_PHASE:
        raise QualificationRefused('row1-request-plan-phase-differs')
    data = plan.get('workload')
    problems, workload, manifest_sha256 = correctness.validate_manifest(data)
    if problems or workload is None:
        raise QualificationRefused('row1-request-plan-workload-invalid',
                                   evidence={'problems': list(problems)})
    streams = plan.get('streams', workload.streams)
    if type(streams) is not int or streams != workload.streams:
        raise QualificationRefused('row1-request-plan-streams-differ')
    run_index = plan.get('run_index', 0)
    if type(run_index) is not int or run_index < 0:
        raise QualificationRefused('row1-request-plan-run-index-invalid')
    if not _text(model):
        raise QualificationError('model-required')
    cohort = correctness.build_cohort(workload)
    explicit = plan.get('prompts')
    if explicit is not None:
        if (type(explicit) is not list or len(explicit) != workload.streams
                or any(type(ids) is not list or len(ids) != workload.prompt_tokens
                       or len(ids) > 200000
                       or any(type(token) is not int or token < 0 for token in ids)
                       for ids in explicit)):
            raise QualificationRefused('row1-exact-prompt-shape-invalid')
        cohort = [dict(stream=index, prompt=list(ids), prompt_chars=0,
                       prompt_sha256=canonical_sha(ids)) for index,ids in enumerate(explicit)]
    salt = correctness.build_cache_salt(run_id, run_index)
    bodies = [correctness.request_body(model, row['prompt'], workload, salt)
              for row in cohort]
    digests = [correctness.request_sha256(body) for body in bodies]
    declared = plan.get('request_sha256')
    if declared is not None and declared != digests:
        raise QualificationRefused('row1-request-plan-requests-differ')
    identity = {'schema': ROW1_PLAN_SCHEMA, 'phase': ROW1_PHASE,
                'manifest_sha256': manifest_sha256, 'run_index': run_index,
                'streams': len(bodies), 'run_id': run_id,
                'request_sha256': digests}
    return {'schema': ROW1_PLAN_SCHEMA, 'phase': ROW1_PHASE,
            'workload': workload, 'manifest_sha256': manifest_sha256,
            'run_index': run_index, 'cohort': cohort, 'bodies': bodies,
            'request_sha256': digests, 'cache_salt': salt,
            'plan_sha256': canonical_sha(identity), 'identity': identity}


def freeze_prefill_plan(plan, *, run_id, model):
    """Freeze the queue-declared native prefill request for a placement window.

    No built-in prompt/token policy: the whole request comes from the plan.
    The engine-side offer attribution is a separate seam; this only proves a
    real prefill request was admitted and advanced inside the same window.
    """
    if plan is None:
        raise PhaseUnavailable('missing-native-prefill-request-plan')
    if type(plan) is not dict:
        raise QualificationRefused('native-prefill-plan-not-an-object')
    unknown = set(plan) - PREFILL_PLAN_FIELDS
    if unknown:
        raise QualificationRefused('native-prefill-plan-unknown-fields',
                                   evidence={'unknown': sorted(unknown)})
    if plan.get('schema') != PREFILL_PLAN_SCHEMA:
        raise QualificationRefused('native-prefill-plan-schema-differs')
    data = plan.get('workload')
    problems, workload, manifest_sha256 = correctness.validate_manifest(data)
    if problems or workload is None:
        raise QualificationRefused('native-prefill-plan-workload-invalid',
                                   evidence={'problems': list(problems)})
    if workload.streams < 1:
        raise QualificationRefused('native-prefill-plan-no-streams')
    streams = plan.get('streams', workload.streams)
    if type(streams) is not int or streams != workload.streams:
        raise QualificationRefused('native-prefill-plan-streams-differ')
    run_index = plan.get('run_index', 0)
    if type(run_index) is not int or run_index < 0:
        raise QualificationRefused('native-prefill-plan-run-index-invalid')
    if not _text(model):
        raise QualificationError('model-required')
    cohort = correctness.build_cohort(workload)
    salt = correctness.build_cache_salt(run_id, run_index)
    bodies = [correctness.request_body(model, row['prompt'], workload, salt)
              for row in cohort]
    digests = [correctness.request_sha256(body) for body in bodies]
    declared = plan.get('request_sha256')
    if declared is not None and declared != digests:
        raise QualificationRefused('native-prefill-plan-requests-differ')
    identity = {'schema': PREFILL_PLAN_SCHEMA,
                'manifest_sha256': manifest_sha256, 'run_index': run_index,
                'streams': len(bodies), 'run_id': run_id,
                'request_sha256': digests}
    return {'schema': PREFILL_PLAN_SCHEMA, 'workload': workload,
            'manifest_sha256': manifest_sha256, 'run_index': run_index,
            'cohort': cohort, 'bodies': bodies, 'request_sha256': digests,
            'cache_salt': salt, 'plan_sha256': canonical_sha(identity),
            'identity': identity}


class ImmutableStore:
    """Write-once artifact directory.  Files are 0600 and never replaced."""

    def __init__(self, root):
        root = os.path.abspath(os.fspath(root))
        if os.path.islink(root):
            raise QualificationError('artifact-store-must-not-be-a-symlink')
        if os.path.exists(root) and not os.path.isdir(root):
            raise QualificationError('artifact-store-not-a-directory')
        os.makedirs(root, mode=0o700, exist_ok=True)
        self.root = root

    @staticmethod
    def _name(name):
        if (not _text(name) or os.path.basename(name) != name
                or name in ('.', '..')):
            raise QualificationError('artifact-name-invalid')
        return name

    def path(self, name):
        return os.path.join(self.root, self._name(name))

    def write_once(self, name, record):
        """Write ``record`` once; an existing name is never overwritten."""
        path = self.path(name)
        data = (json.dumps(record, sort_keys=True, separators=(',', ':'))
                + '\n').encode()
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                                 0o600)
        except FileExistsError:
            raise QualificationRefused('artifact-already-exists:' + name) from None
        try:
            with os.fdopen(descriptor, 'wb') as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            # Keep the partial file: the brief forbids deleting evidence.
            raise
        return {'name': name, 'bytes': len(data),
                'sha256': canonical_sha(record),
                'file_sha256': hashlib.sha256(data).hexdigest()}

    def read(self, name, expected_sha256=None):
        path = self.path(name)
        try:
            with open(path, 'rb') as handle:
                data = handle.read()
        except OSError:
            raise PhaseUnavailable('artifact-missing:' + name) from None
        try:
            record = json.loads(data)
        except ValueError:
            raise QualificationRefused('artifact-not-json:' + name) from None
        sha = canonical_sha(record)
        if expected_sha256 is not None and sha != expected_sha256:
            raise QualificationRefused('artifact-digest-mismatch')
        return record, {'name': name, 'sha256': sha,
                        'file_sha256': hashlib.sha256(data).hexdigest()}

    def find_teacher_artifact(self, expected_sha256):
        """Find exactly one retained controls artifact by canonical digest."""
        if not _digest(expected_sha256):
            raise PhaseUnavailable('missing-retained-teacher-artifact-sha')
        matches = []
        for name in sorted(os.listdir(self.root)):
            if not name.endswith('.json'):
                continue
            try:
                record, meta = self.read(name)
            except QualificationError:
                continue
            if (record.get('phase') == 'rule4_controls'
                    and record.get('kind') == 'teacher'
                    and type(record.get('artifact')) is dict
                    and canonical_sha(record['artifact']) == expected_sha256):
                matches.append((name, record, meta))
        if not matches:
            raise PhaseUnavailable('retained-teacher-artifact-not-found')
        if len(matches) > 1:
            # Two byte-identical retained artifacts are still an ambiguous
            # selection; never pick.
            raise QualificationRefused('retained-teacher-artifact-ambiguous')
        return matches[0]

    def names(self):
        return sorted(os.listdir(self.root))


class LoopbackControl:
    """The existing authenticated loopback transport, nothing broader.

    ``session`` posts to the reviewed probe session path, ``worker`` and
    ``frontend`` post to the two narrow native-qualification adapter paths.
    The key is read from memory/environment and never placed in a command,
    argv or log.
    """

    def __init__(self, url, *, key=None, timeout_s=120.0):
        import e3_probe_rpc as rpc
        scheme, host, port = rpc.target(url)
        self.scheme, self.host, self.port = scheme, host, port
        key = key if key is not None else os.environ.get('VLLM_API_KEY')
        if (type(key) is not str or not key.strip()
                or '\n' in key or '\r' in key):
            raise QualificationError('native-qualification-requires-existing-auth')
        self.key = key
        if type(timeout_s) not in (int, float) or not float(timeout_s) > 0:
            raise QualificationError('native-qualification-timeout-invalid')
        self.timeout_s: float = float(timeout_s)

    @property
    def controller(self):
        """Sync controller adapter for the reviewed correctness client."""
        return _SessionController(self)

    def _post(self, path, command, timeout_s=None):
        # Same bounded loopback transport for the session and the two adapter
        # paths.  The native session answers with schema/action/run_id but does
        # not echo ``command_seq`` (a proven peer-input difference); the
        # sequence is still checked whenever the session does echo it, and the
        # adapter paths must always echo it.
        import http.client
        import math
        import e3_probe_rpc as rpc
        session = path == 'session'
        request_path = '/v1/flash_s5_native_overlap' if session else path
        response_max = rpc.MAX_RESPONSE if session or path == WORKER_PATH else 65536
        data = json.dumps(command, allow_nan=False).encode()
        if len(data) > rpc.MAX_COMMAND:
            raise QualificationRefused('command-too-large')
        wait = timeout_s if timeout_s is not None else self.timeout_s
        if type(wait) not in (int, float) or not math.isfinite(float(wait)) or wait <= 0:
            raise QualificationRefused('timeout-invalid')
        connection = (http.client.HTTPSConnection if self.scheme == 'https'
                      else http.client.HTTPConnection)(
                          self.host, self.port, timeout=float(wait))
        try:
            connection.request('POST', request_path, body=data,
                               headers={'Content-Type': 'application/json',
                                        'Authorization': 'Bearer ' + self.key})
            response = connection.getresponse()
            body = response.read(response_max + 1)
            if len(body) > response_max:
                raise QualificationRefused('response-too-large')
            if response.status == 409:
                raise QualificationRefused('controller-refusal')
            if response.status != 200:
                raise QualificationRefused('http-status-' + str(response.status))
            result = json.loads(body)
        except QualificationRefused:
            raise
        except Exception as exc:  # noqa: BLE001 - transport, never a verdict
            raise QualificationRefused('transport-failed:' + type(exc).__name__) from None
        finally:
            connection.close()
        if (type(result) is not dict or result.get('schema') != 1
                or result.get('action') != command.get('action')
                or result.get('run_id') != command.get('run_id')):
            raise QualificationRefused('response-identity-differs')
        # The native session omits the sequence echo; it is checked whenever
        # present, and always required from the two narrow adapter actions.
        if 'command_seq' in result and result['command_seq'] != command.get('command_seq'):
            raise QualificationRefused('response-identity-differs')
        if 'client_id' in result and result['client_id'] != command.get('client_id'):
            raise QualificationRefused('response-identity-differs')
        if not session and ('command_seq' not in result or 'client_id' not in result):
            raise QualificationRefused('response-identity-differs')
        return result

    def _base_url(self):
        return '%s://%s:%d' % (self.scheme, self.host, self.port)

    @property
    def base_url(self):
        return self._base_url()

    def session(self, command, timeout_s=None):
        return self._post('session', command, timeout_s)

    def frontend(self, command, timeout_s=None):
        return self._post(FRONTEND_PATH, command, timeout_s)

    def worker(self, command, timeout_s=None):
        return self._post(WORKER_PATH, command, timeout_s)


class _SessionController:
    """The existing ``HttpController.execute`` shape over this transport."""

    def __init__(self, control):
        self.control = control

    def execute(self, command, timeout_s):
        start = time.monotonic()
        try:
            reply = self.control.session(command, timeout_s)
        except QualificationRefused as exc:
            raise protocol.ControllerTransportError(
                'http_transport',
                {'kind': 'http', 'error': exc.code, 'timeout': False}) from None
        return reply, {'kind': 'http', 'rc': 0,
                       'seconds': time.monotonic() - start,
                       'timeout': False, 'stdout_len': 0, 'stderr_len': 0}


class QualificationDriver:
    """One-shot phase driver over a control seam and explicit providers."""

    def __init__(self, store, *, run_id, model, engine_epoch, control=None,
                 url=None, key=None, teacher_baseline=None,
                 teacher_workload=None, teacher_instrument=None,
                 row1_provider=None, row1_producer=None, placement=None,
                 native_read_provider=None, native_prefill_provider=None,
                 native_source_provider=None, clock=time.monotonic_ns,
                 timeout_s=120.0, route_banks=None):
        if not _text(run_id):
            raise QualificationError('run-id-required')
        if not _text(model):
            raise QualificationError('model-required')
        if not _positive(engine_epoch):
            raise QualificationError('engine-epoch-required')
        self.store = (store if isinstance(store, ImmutableStore)
                      else ImmutableStore(store))
        self.run_id = run_id
        self.model = model
        self.engine_epoch = engine_epoch
        self.control = control
        self.url = url
        if self.control is None:
            if url is None:
                raise QualificationError('control-or-loopback-url-required')
            self.control = LoopbackControl(url, key=key, timeout_s=timeout_s)
        elif self.url is None:
            self.url = getattr(self.control, 'base_url', None)
        self.teacher_baseline = teacher_baseline
        self.teacher_workload = teacher_workload
        self.teacher_instrument = teacher_instrument
        self.row1_provider = row1_provider
        self.row1_producer = row1_producer
        self.placement = placement
        self.native_read_provider = native_read_provider
        self.native_prefill_provider = native_prefill_provider
        self.native_source_provider = native_source_provider
        self.clock = clock
        self.timeout_s = timeout_s
        # Wire-v1 route-bank identities.  Supplied by the composition root, or
        # imported from the pinned engine module; never invented locally.
        self._route_banks = route_banks
        self._phase = threading.Lock()
        self._teacher_window = None
        self._commands = 0
        self._frontend_control = getattr(self.control, 'frontend', None)
        self.history = []

    # -- command helpers -------------------------------------------------
    def _client_id(self, phase):
        return 'q18a-%s-%s-%s' % (phase, self.run_id.replace('/', '-'),
                                  secrets_hex(8))

    def _command(self, phase_name, action, client_id, **fields):
        self._commands += 1
        command = {'schema': 1, 'run_id': self.run_id, 'client_id': client_id,
                   'command_seq': self._commands, 'action': action,
                   'engine_epoch': self.engine_epoch}
        command.update(fields)
        # Metadata only: never command values, payloads or credentials.
        self.history.append({'phase': phase_name, 'action': action,
                             'seq': command['command_seq'],
                             'keys': sorted(command)})
        return command

    def command_keys(self, phase=None, action=None):
        return [row for row in self.history
                if (phase is None or row['phase'] == phase)
                and (action is None or row['action'] == action)]

    def route_banks(self):
        """Return ``(full, split)`` wire-v1 bank identities, or refuse."""
        banks = self._route_banks
        if banks is None:
            try:
                from wire_v1 import ROUTE_BANK_FULL, ROUTE_BANK_SPLIT
            except ImportError:
                raise PhaseUnavailable('missing-route-bank-identity') from None
            banks = (ROUTE_BANK_FULL, ROUTE_BANK_SPLIT)
        if (type(banks) not in (tuple, list) or len(banks) != 2
                or type(banks[0]) is not int or type(banks[1]) is not int
                or banks[0] == banks[1]):
            raise QualificationError('route-bank-identities-invalid')
        return banks[0], banks[1]

    def _session(self, phase_name, action, *, require_epoch=True, **fields):
        command = self._command(phase_name, action,
                                self._client_id(phase_name), **fields)
        control = self.control
        if control is None:
            raise PhaseUnavailable('missing-control-transport')
        reply = control.session(command, self.timeout_s)
        if require_epoch:
            self._require_epoch(reply)
        return reply

    def _worker(self, phase, worker_action, payload):
        command = self._command(phase, 'qualification_worker',
                                self._client_id(phase),
                                worker_action=worker_action,
                                payload=dict(payload))
        control = self.control
        if control is None:
            raise PhaseUnavailable('missing-control-transport')
        reply = control.worker(command, self.timeout_s)
        self._require_epoch(reply)
        worker_reply = reply.get('worker_reply')
        if type(worker_reply) is not dict:
            raise QualificationRefused('worker-returned-no-record')
        if worker_reply.get('engine_epoch') != self.engine_epoch:
            raise QualificationRefused('worker-engine-epoch-differs')
        return worker_reply

    def _frontend(self, phase, enabled, plan):
        command = self._command(phase, 'qualification_frontend',
                                self._client_id(phase), enabled=enabled,
                                plan=plan)
        control = self.control
        if control is None:
            raise PhaseUnavailable('missing-control-transport')
        reply = control.frontend(command, self.timeout_s)
        self._require_epoch(reply)
        if reply.get('enabled') is not enabled:
            raise QualificationRefused('frontend-enabled-differs')
        return reply

    def _require_epoch(self, reply):
        if reply.get('engine_epoch') != self.engine_epoch:
            raise QualificationRefused('foreign-engine-epoch')

    # -- public phase interface -----------------------------------------
    def collect_phase(self, phase, **kwargs):
        """Collect one phase exactly once; return raw evidence, never a verdict."""
        if phase not in PHASES:
            raise QualificationError('phase-invalid')
        if not self._phase.acquire(blocking=False):
            record = self._refusal(phase, 'phase-overlap-refused',
                                   missing=['phase-overlap-refused'])
            raise PhaseConflict('phase-overlap-refused', record=record)
        try:
            if os.path.exists(self.store.path(PHASE_ARTIFACTS[phase])):
                # One collection per phase per store: never re-query the oracle,
                # re-run the teacher sequence or re-open a placement window.
                raise QualificationRefused('phase-already-recorded')
            if phase == 'row1':
                return self._collect_row1(**kwargs)
            if phase == 'rule4_controls':
                return self._collect_controls(**kwargs)
            if phase == 'rule4_candidate':
                return self._collect_candidate(**kwargs)
            return self._collect_row4(**kwargs)
        except QualificationRefused as exc:
            if exc.record is None:
                record = self._refusal(phase, exc.code, missing=[exc.code],
                                       cleanup=exc.cleanup,
                                       evidence=exc.evidence)
                exc.record = record
            raise
        except QualificationError:
            raise
        except BaseException as exc:  # noqa: BLE001 - keep a partial record
            record = self._refusal(phase, 'driver-exception:' + type(exc).__name__,
                                   missing=['driver-exception'])
            raise QualificationRefused('driver-exception:' + type(exc).__name__,
                                       record=record) from None
        finally:
            self._phase.release()

    def _refusal(self, phase, code, *, missing=None, evidence=None,
                 window=None, cleanup=None):
        record = {
            'schema': 1, 'phase': phase, 'run_id': self.run_id,
            'engine_epoch': self.engine_epoch,
            'refusal': code, 'missing': list(missing or []),
            'evidence': evidence, 'window': window, 'cleanup': cleanup,
            'retained': True, 'retries': 0, 'picked': False,
            'recorded_ns': int(self.clock()),
        }
        try:
            stored = self.store.write_once('refusal.' + phase + '.json', record)
        except QualificationRefused:
            # The first refusal is the retained evidence; never replace it.
            stored = None
        record['record'] = stored
        return record

    def _retain(self, phase, kind, evidence, *, extra=None):
        record = {
            'schema': 1, 'phase': phase, 'kind': kind, 'run_id': self.run_id,
            'model': self.model, 'engine_epoch': self.engine_epoch,
            'evidence': evidence, 'retries': 0, 'picked': False,
            'recorded_ns': int(self.clock()),
        }
        if extra:
            record.update(extra)
        stored = self.store.write_once(phase + '.json', record)
        record['record'] = stored
        return record

    # -- row1 -------------------------------------------------------------
    def _collect_row1(self, *, request_plan=None):
        """One scoped native row1 window over a real bounded request.

        The producer owns the exact lifecycle: prepare full -> paused split ->
        arm ``row1-native`` -> configure -> resume -> actual HTTP requests ->
        paused disable -> retained evidence -> disarm -> typed full return.
        The provider seam, when supplied, is an in-process alternative that
        performs configure/query/disable and returns the same packet list; it
        never replaces the admission actions.  Neither seam is allowed to run
        without an explicit request plan: a missing plan is a named
        ``missing-row1-request-plan`` INCONCLUSIVE before anything is armed.
        """
        control = self.control
        if control is None:
            raise PhaseUnavailable('missing-native-source-provider')
        if not callable(getattr(control, 'session', None)):
            raise PhaseUnavailable('missing-native-admission-transport')
        provider = self.row1_provider or self.native_source_provider
        producer = self.row1_producer
        if producer is None and request_plan is not None:
            producer = NativeRow1Producer(
                plan=freeze_row1_plan(request_plan, run_id=self.run_id,
                                      model=self.model),
                url=self.url, model=self.model)
        if producer is not None:
            result = producer.run(self, run_id=self.run_id,
                                  engine_epoch=self.engine_epoch)
            validated = self._row1_provider_result(result)
            arm = result.get('arm')
            disarm = result.get('disarm')
            configure = validated['configure_acks']
            disable = validated['disable_acks']
            packets = validated['workers']
            evidence, missing = self._row1_evidence(
                arm, configure, disable, packets, disarm,
                lifecycle=result.get('lifecycle'),
                requests=result.get('requests'),
                terminal=result.get('terminal'),
                plan_sha256=result.get('plan_sha256'),
                cleanup=result.get('cleanup'))
        elif provider is not None:
            arm = None
            configure = []
            disable = []
            packets = []
            disarm = None
            first = None
            try:
                arm = self._row1_arm()
                result = self._row1_provider(provider)
                configure = result['configure_acks']
                disable = result['disable_acks']
                packets = result['workers']
            except BaseException as exc:  # noqa: BLE001 - preserve first failure
                first = exc
            finally:
                # The oracle must be off before any teacher/W1 work even when
                # the evidence query failed; a provider owns its own disable.
                try:
                    disarm = self._row1_disarm()
                except BaseException as exc:  # noqa: BLE001
                    if first is None:
                        first = exc
            cleanup = {
                'admission_armed': arm is not None,
                'oracle_disabled': len(disable) == len(RANKS),
                'admission_disarmed': disarm is not None,
            }
            if first is not None:
                evidence = {
                    'phase': 'row1', 'run_id': self.run_id,
                    'engine_epoch': self.engine_epoch,
                    'native_phase': ROW1_PHASE,
                    'workers': copy.deepcopy(packets),
                    'configure_acks': copy.deepcopy(configure),
                    'disable_acks': copy.deepcopy(disable),
                    'admission': {'arm': arm, 'disarm': disarm},
                }
                if isinstance(first, QualificationRefused):
                    if first.evidence is None:
                        first.evidence = evidence
                    else:
                        first.evidence = {'partial': first.evidence,
                                          'phase_evidence': evidence}
                    first.cleanup = cleanup
                    raise first
                raise QualificationRefused(
                    'driver-exception:' + type(first).__name__,
                    evidence=evidence, cleanup=cleanup) from None
            evidence, missing = self._row1_evidence(
                arm, configure, disable, packets, disarm)
        else:
            raise PhaseUnavailable('missing-row1-request-plan')
        record = self._retain('row1', 'row1', evidence)
        return {
            'schema': 1, 'phase': 'row1', 'run_id': self.run_id,
            'engine_epoch': self.engine_epoch, 'evidence': evidence,
            # The whole canonical artifact, not a digest-only substitute.
            'row1_sha256': canonical_sha(evidence),
            'complete': not missing, 'missing': missing,
            'record': record['record'],
        }

    def _row1_arm(self):
        reply = self._session('row1', 'arm_admission', require_epoch=False,
                              phase=ROW1_PHASE)
        if (type(reply) is not dict or reply.get('phase') != ROW1_PHASE
                or reply.get('armed') is not True):
            raise QualificationRefused('row1-admission-not-armed')
        intent = reply.get('admission_intent')
        if (type(intent) is not dict or intent.get('phase') != ROW1_PHASE
                or intent.get('armed') is not True
                or intent.get('engine_epoch') != self.engine_epoch
                or intent.get('owner') != 'engine-core'):
            raise QualificationRefused('row1-admission-intent-differs')
        if (reply.get('route') != 'split'
                or type(reply.get('route_generation')) is not int
                or reply['route_generation'] < 0):
            raise QualificationRefused('row1-admission-route-differs')
        return copy.deepcopy(reply)

    def _row1_disarm(self):
        reply = self._session('row1', 'disarm_admission', require_epoch=False)
        if type(reply) is not dict or reply.get('armed') is not False:
            raise QualificationRefused('row1-admission-not-disarmed')
        return copy.deepcopy(reply)

    def _row1_rows(self, action, payload):
        if action not in ROW1_ACTIONS:
            raise QualificationError('row1-action-invalid')
        reply = self._worker('row1', action, copy.deepcopy(payload))
        rows = reply.get('workers')
        if type(rows) is not list or len(rows) != len(RANKS):
            raise QualificationRefused('row1-worker-ranks-incomplete')
        out = {}
        for row in rows:
            if type(row) is not dict:
                raise QualificationRefused('row1-worker-row-not-an-object')
            rank = row.get('worker_rank')
            if rank not in RANKS or rank in out:
                raise QualificationRefused('row1-worker-rank-invalid')
            if row.get('engine_epoch') != self.engine_epoch:
                raise QualificationRefused('row1-worker-epoch-differs')
            generation = row.get('route_generation')
            if type(generation) is not int or generation < 0:
                raise QualificationRefused('row1-worker-route-generation-invalid')
            inner = row.get('qualification')
            if type(inner) is not dict:
                raise QualificationRefused('row1-worker-returned-no-record')
            out[rank] = (row, inner)
        if set(out) != set(RANKS):
            raise QualificationRefused('row1-worker-ranks-incomplete')
        return out, reply

    def _row1_configure(self, enabled):
        rows, _ = self._row1_rows('row1_configure',
                                  {'enabled': enabled, 'run_id': self.run_id})
        acks = []
        for rank in RANKS:
            _, inner = rows[rank]
            if (inner.get('enabled') is not enabled
                    or inner.get('run_id') != self.run_id):
                raise QualificationRefused(
                    'row1-configure-ack-differs:rank-%d' % rank)
            if inner.get('engine_epoch') != self.engine_epoch:
                raise QualificationRefused(
                    'row1-configure-epoch-differs:rank-%d' % rank)
            generation = inner.get('route_generation')
            if type(generation) is not int or generation < 0:
                raise QualificationRefused(
                    'row1-configure-route-generation-invalid:rank-%d' % rank)
            acks.append({'rank': rank, 'ack': copy.deepcopy(inner)})
        return acks

    def _row1_packets(self):
        rows, reply = self._row1_rows('row1_evidence',
                                      {'run_id': self.run_id})
        packets = []
        try:
            for rank in RANKS:
                row, inner = rows[rank]
                packets.append(self._row1_packet(rank, row, inner))
        except QualificationRefused as exc:
            exc.evidence = {'workers': copy.deepcopy(packets),
                            'worker_reply': copy.deepcopy(reply)}
            raise
        generations = {packet['route_generation'] for packet in packets}
        if len(generations) != 1:
            raise QualificationRefused('row1-packet-route-generation-differs')
        return packets, reply

    def _row1_packet(self, rank, row, inner):
        if type(inner) is not dict:
            raise QualificationRefused('row1-packet-fields-differ:rank-%d' % rank)
        wrapper = {'graded_by', 'engine_supplied_verdict'}
        if wrapper & set(inner):
            if (set(inner) != ROW1_PACKET_FIELDS | wrapper
                    or inner.get('graded_by') != 'native_qualification.handle_worker'
                    or inner.get('engine_supplied_verdict') is not False):
                raise QualificationRefused('row1-worker-envelope-invalid:rank-%d' % rank)
            inner = {key: inner[key] for key in ROW1_PACKET_FIELDS}
        if set(inner) != ROW1_PACKET_FIELDS:
            raise QualificationRefused('row1-packet-fields-differ:rank-%d' % rank)
        if inner.get('schema') != ROW1_SCHEMA:
            raise QualificationRefused('row1-packet-schema-differs:rank-%d' % rank)
        if inner.get('run_id') != self.run_id:
            raise QualificationRefused('row1-packet-run-id-differs:rank-%d' % rank)
        if inner.get('rank') != rank:
            raise QualificationRefused('row1-packet-rank-differs:rank-%d' % rank)
        if inner.get('engine_epoch') != self.engine_epoch:
            raise QualificationRefused('row1-packet-epoch-differs:rank-%d' % rank)
        generation = inner.get('route_generation')
        disabled = inner.get('disabled_route_generation')
        if (type(generation) is not int or generation < 0
                or type(disabled) is not int or disabled < generation
                or disabled != row.get('route_generation')):
            raise QualificationRefused(
                'row1-packet-route-generation-differs:rank-%d' % rank)
        incarnation = inner.get('worker_incarnation')
        if not (_text(incarnation)
                or (type(incarnation) is int and incarnation >= 0)):
            raise QualificationRefused(
                'row1-packet-incarnation-invalid:rank-%d' % rank)
        layers = inner.get('local_layers')
        if (type(layers) is not list
                or any(not _text(name) for name in layers)
                or len(set(layers)) != len(layers)):
            raise QualificationRefused('row1-packet-layers-invalid:rank-%d' % rank)
        if inner.get('enabled') is not False:
            raise QualificationRefused('row1-packet-oracle-not-disabled:rank-%d' % rank)
        raw = inner.get('raw')
        if type(raw) is not dict or raw.get('enabled') is not False:
            raise QualificationRefused('row1-packet-raw-not-disabled:rank-%d' % rank)
        for items_key, count_key in (('records', 'comparisons'),
                                     ('layer_records', 'layers'),
                                     ('failure_records', 'failures')):
            items = raw.get(items_key)
            count = raw.get(count_key)
            if (not isinstance(items, (list, tuple))
                    or type(count) is not int or count < 0
                    or len(items) != count):
                raise QualificationRefused(
                    'row1-packet-raw-count-differs:rank-%d:%s' % (rank, items_key))
        capacity = raw.get('capacity')
        if (type(capacity) is not int or capacity < 0
                or capacity < max(raw['comparisons'], raw['layers'],
                                  raw['failures'])):
            raise QualificationRefused(
                'row1-packet-raw-capacity-invalid:rank-%d' % rank)
        return copy.deepcopy(inner)

    def _row1_provider(self, provider):
        request = {
            'phase': 'row1', 'run_id': self.run_id, 'model': self.model,
            'engine_epoch': self.engine_epoch, 'ranks': list(RANKS),
            'schema': ROW1_SCHEMA, 'transport': 'controlled-native-source',
        }
        result = provider(copy.deepcopy(request))
        if type(result) is not dict:
            raise QualificationRefused('row1-provider-returned-no-record')
        if result.get('vm_receipt_used_as_engine_evidence') is True:
            raise QualificationRefused('row1-vm-receipt-as-engine-evidence')
        try:
            return self._row1_provider_result(result)
        except QualificationRefused as exc:
            # Keep the provider's raw return even when one packet is refused.
            if exc.evidence is None:
                exc.evidence = {'provider_result': copy.deepcopy(result)}
            raise

    def _row1_provider_result(self, result):
        workers = result.get('workers')
        configure = result.get('configure_acks')
        disable = result.get('disable_acks')
        if not (isinstance(workers, list) and isinstance(configure, list)
                and isinstance(disable, list)):
            raise PhaseUnavailable('row1-packets-unavailable')
        workers = list(workers)
        configure = list(configure)
        disable = list(disable)
        if len(workers) != len(RANKS) or len(configure) != len(RANKS) \
                or len(disable) != len(RANKS):
            raise QualificationRefused('row1-provider-ranks-incomplete')
        packets = []
        for packet in workers:
            if type(packet) is not dict or packet.get('rank') not in RANKS:
                raise QualificationRefused('row1-provider-packet-invalid')
            # The disabled packet retains the qualified generation in
            # ``route_generation``; its disabled generation is the route the
            # engine actually returned to (possibly a later one after FULL).
            row = {'route_generation': packet.get('disabled_route_generation')}
            packets.append(self._row1_packet(packet['rank'], row, packet))
        if {packet['rank'] for packet in packets} != set(RANKS):
            raise QualificationRefused('row1-provider-ranks-incomplete')
        generations = {packet['route_generation'] for packet in packets}
        if len(generations) != 1:
            raise QualificationRefused('row1-packet-route-generation-differs')
        return {'workers': packets,
                'configure_acks': copy.deepcopy(configure),
                'disable_acks': copy.deepcopy(disable)}

    def _row1_evidence(self, arm, configure, disable, packets, disarm,
                       *, lifecycle=None, requests=None, terminal=None,
                       plan_sha256=None, cleanup=None):
        packets = sorted(copy.deepcopy(packets), key=lambda row: row['rank'])
        missing = []
        for packet in packets:
            rank = packet['rank']
            if not packet['raw']['records']:
                missing.append('row1-rank-comparisons-empty:%d' % rank)
            if not packet['local_layers']:
                missing.append('row1-rank-local-layers-empty:%d' % rank)
        if not any(packet['raw']['records'] for packet in packets):
            missing.append('row1-records-empty')
        evidence = {
            'phase': 'row1', 'run_id': self.run_id,
            'engine_epoch': self.engine_epoch,
            'native_phase': ROW1_PHASE,
            'source': 'native-source-oracle',
            'admission': {'arm': arm, 'disarm': disarm},
            'configure_acks': copy.deepcopy(configure),
            'disable_acks': copy.deepcopy(disable),
            'workers': packets,
            'model_settings_unchanged': True,
            'enabled_on_every_rank': all(
                row['ack'].get('enabled') is True for row in configure),
            'disabled_on_every_rank': all(
                packet['enabled'] is False for packet in packets),
            'comparisons': sum(len(packet['raw']['records'])
                               for packet in packets),
            'layer_records': sum(len(packet['raw']['layer_records'])
                                 for packet in packets),
            'failures': sum(len(packet['raw']['failure_records'])
                            for packet in packets),
            'missing': list(missing),
        }
        # Producer-only provenance.  These are the actual receipts, retained
        # verbatim; the provider seam cannot fabricate them.
        if plan_sha256 is not None:
            evidence['request_plan_sha256'] = plan_sha256
        if lifecycle is not None:
            evidence['lifecycle'] = copy.deepcopy(lifecycle)
        if requests is not None:
            evidence['requests'] = copy.deepcopy(requests)
        if terminal is not None:
            evidence['terminal'] = copy.deepcopy(terminal)
        if cleanup is not None:
            evidence['cleanup'] = copy.deepcopy(cleanup)
        return evidence, missing

    # -- rule4 controls ---------------------------------------------------
    def _teacher_workload(self):
        if self.teacher_workload is None:
            raise PhaseUnavailable('missing-teacher-workload')
        workload = self.teacher_workload
        if type(workload) is not correctness.Workload:
            raise QualificationError('teacher-workload-type-invalid')
        if (workload.streams != STREAMS
                or workload.generation_budget != teacher_client.TOTAL):
            raise QualificationRefused('teacher-workload-shape-differs')
        return workload

    def _instrument(self):
        if self.teacher_instrument is not None:
            instrument = self.teacher_instrument
        else:
            if self.teacher_baseline is None:
                raise PhaseUnavailable('missing-teacher-baseline')
            instrument = teacher_client.Instrument(
                self.teacher_baseline, snapshot_max_bytes=SNAPSHOT_MAX_BYTES,
                numerical_rule=4)
        return instrument

    def _collect_controls(self):
        if self.control is None:
            raise PhaseUnavailable('missing-teacher-transport')
        workload = self._teacher_workload()
        instrument = self._instrument()
        if not hasattr(instrument, 'finish') or not hasattr(instrument, 'cohort'):
            raise QualificationError('teacher-instrument-shape-invalid')
        if not callable(self._frontend_control):
            raise PhaseUnavailable('native-teacher-frontend-owner-unavailable')
        if not _text(self.url):
            raise PhaseUnavailable('missing-teacher-loopback-url')
        frontend_missing = []
        frontend_status = None
        # No source pin is supplied here: the source artifact is a readiness
        # input owned by the caller, and an unverified source must not turn a
        # raw collection into a PASS.  The grader decides.
        try:
            from e3_native_teacher_sequence import NativeTeacherSequence
            result = NativeTeacherSequence(self, instrument, workload).run()
        finally:
            self._teacher_window = {
                'kind': 'teacher', 'run_id': self.run_id,
                'ended_ns': int(self.clock()),
            }
            if self._frontend_control is not None:
                try:
                    frontend_status = self._frontend('rule4_controls', False, None)
                except QualificationRefused as exc:
                    frontend_missing.append('frontend-clear-refused:' + exc.code)
        teacher = result.get('teacher_forced') if type(result) is dict else None
        if type(teacher) is not dict:
            raise QualificationRefused('teacher-sequence-returned-no-record')
        evidence = teacher.get('evidence')
        if type(evidence) is not dict:
            raise PhaseUnavailable('teacher-evidence-unavailable')
        if evidence.get('schema') != TEACHER_SCHEMA:
            raise QualificationRefused('teacher-evidence-schema-differs')
        runs = evidence.get('runs')
        if type(runs) is not dict or set(runs) != set(TEACHER_SEQUENCE):
            raise QualificationRefused('teacher-sequence-is-not-A1-A2-B-C')
        for name in TEACHER_SEQUENCE:
            run = runs.get(name)
            if type(run) is not dict or run.get('route') != TEACHER_ROUTES[name]:
                raise QualificationRefused('teacher-route-differs:' + name)
            mapping = run.get('row_mapping') or {}
            if (mapping.get('overflow') is not False
                    or type(mapping.get('defects')) is not list
                    or mapping['defects']):
                raise QualificationRefused('teacher-collector-overflow:' + name)
            if len(mapping.get('records') or []) != PLACEMENT_STREAMS * MEASURED * 2:
                raise PhaseUnavailable('teacher-row-coverage-incomplete:' + name)
        positions = evidence.get('fixed_positions')
        expected = list(range(teacher_client.ENTRY,
                              teacher_client.ENTRY + teacher_client.MEASURE))
        if positions != expected:
            raise QualificationRefused('teacher-measured-positions-differ')
        if self._frontend_control is None:
            frontend_missing.append('frontend-plan-owner-unavailable')
        elif frontend_status is not None and frontend_status.get('enabled') is not False:
            frontend_missing.append('frontend-plan-still-admitted')
        artifact = copy.deepcopy(evidence)
        artifact_sha256 = canonical_sha(artifact)
        plan = getattr(instrument, 'plan', None)
        extra = {
            'artifact': artifact, 'artifact_sha256': artifact_sha256,
            'teacher_plan': copy.deepcopy(plan),
            'teacher_plan_sha256': canonical_sha(plan) if type(plan) is dict else None,
            'frontend_plan_calls': copy.deepcopy(result.get('native_frontend_calls', [])),
            'frontend_status': frontend_status,
            'frontend_missing': list(frontend_missing),
            'teacher_window': self._teacher_window,
            'instrument_geometry': dict(
                entry=teacher_client.ENTRY, measured=teacher_client.MEASURE,
                tail=teacher_client.TAIL, total=teacher_client.TOTAL),
        }
        record = self._retain('rule4_controls', 'teacher', evidence, extra=extra)
        evidence_flags = {
            'phase': 'rule4_controls', 'kind': 'controls',
            'teacher_runs': 1, 'retries': 0, 'picked': False,
            'engine_epoch': self.engine_epoch,
            'model_settings_unchanged': True,
            'artifact': artifact, 'artifact_sha256': artifact_sha256,
        }
        return {
            'schema': 1, 'phase': 'rule4_controls', 'run_id': self.run_id,
            'engine_epoch': self.engine_epoch,
            'evidence': evidence, 'flags': evidence_flags,
            'artifact': artifact, 'artifact_sha256': artifact_sha256,
            'teacher_plan': extra['teacher_plan'],
            'teacher_plan_sha256': extra['teacher_plan_sha256'],
            'frontend_plan_calls': extra['frontend_plan_calls'],
            'frontend_status': extra['frontend_status'],
            'complete': not frontend_missing,
            'missing': list(frontend_missing), 'record': record['record'],
        }

    # -- rule4 candidate --------------------------------------------------
    def _collect_candidate(self, *, artifact_sha256=None, recollect=False):
        if recollect:
            raise QualificationRefused('candidate-recollect-refused')
        if not _digest(artifact_sha256):
            raise PhaseUnavailable('missing-retained-teacher-artifact-sha')
        _name, retained, meta = self.store.find_teacher_artifact(artifact_sha256)
        artifact = retained.get('artifact')
        if type(artifact) is not dict:
            raise PhaseUnavailable('retained-teacher-artifact-missing')
        if canonical_sha(artifact) != artifact_sha256:
            raise QualificationRefused('artifact-digest-mismatch')
        if artifact.get('schema') != TEACHER_SCHEMA:
            raise QualificationRefused('candidate-artifact-schema-differs')
        evidence = {
            'phase': 'rule4_candidate', 'kind': 'candidate',
            'teacher_runs': 0, 'retries': 0, 'picked': False,
            'selected_from_same_artifact': True,
            'post_c_before_c': False,
            'engine_epoch': self.engine_epoch,
            'model_settings_unchanged': True,
            'artifact': copy.deepcopy(artifact),
            'artifact_sha256': artifact_sha256,
        }
        record = self._retain('rule4_candidate', 'candidate', evidence,
                              extra={'artifact_sha256': artifact_sha256,
                                     'source_artifact': _name,
                                     'source_artifact_record': meta})
        return {
            'schema': 1, 'phase': 'rule4_candidate', 'run_id': self.run_id,
            'engine_epoch': self.engine_epoch, 'evidence': evidence,
            'artifact': artifact, 'artifact_sha256': artifact_sha256,
            'source_artifact': _name, 'complete': True, 'missing': [],
            'record': record['record'],
        }

    # -- row4 -------------------------------------------------------------
    def _collect_row4(self, *, artifact_sha256=None, row1_sha256=None,
                      window_id=None):
        if not _digest(artifact_sha256):
            raise PhaseUnavailable('missing-retained-teacher-artifact-sha')
        if not _digest(row1_sha256):
            raise PhaseUnavailable('missing-retained-row1-sha')
        _name, retained, _retained_meta = self.store.find_teacher_artifact(artifact_sha256)
        if type(retained.get('artifact')) is not dict:
            raise PhaseUnavailable('retained-teacher-artifact-missing')
        if self.placement is None:
            raise PhaseUnavailable('missing-placement-provider')
        if not _text(window_id):
            raise PhaseUnavailable('missing-placement-window-id')
        if self._teacher_window is not None and self._teacher_window.get('kind') == 'teacher':
            # The teacher window must have ended before this phase; a provider
            # may supply its own proof, but we never open an observer while the
            # teacher sequence is still running.
            _require(self._teacher_window.get('ended_ns') is not None,
                     'teacher-window-not-ended')
        try:
            raw = self.placement.run(self, window_id=window_id,
                                     run_id=self.run_id,
                                     engine_epoch=self.engine_epoch,
                                     model=self.model)
        except QualificationRefused:
            raise
        except BaseException as exc:  # noqa: BLE001 - keep partial evidence
            raise QualificationRefused(
                'placement-exception:' + type(exc).__name__,
                evidence={'collector_error': exception_details(exc, self.control)}) from None
        if type(raw) is not dict:
            raise QualificationRefused('placement-provider-returned-no-record')
        window = raw.get('window')
        self._validate_placement_window(window, window_id)
        if raw.get('window_route_generation') != window.get('route_generation'):
            raise QualificationRefused('placement-window-generation-differs')
        raw_rows = raw.get('raw_placement')
        engine_rows = raw.get('engine_accounting')
        self._require_rank_rows(raw_rows, 'raw-placement')
        self._require_rank_rows(engine_rows, 'engine-accounting')
        if type(raw_rows) is not dict or type(engine_rows) is not dict:
            raise PhaseUnavailable('placement-rank-rows-unavailable')
        stock_prefix = raw.get('stock_prefix') or {}
        if stock_prefix.get('native_admission_intent') is not False:
            raise QualificationRefused('stock-prefix-native-admission-intent')
        for keys in stock_prefix.get('prepare_keys') or []:
            if type(keys) is not list or any(
                    type(key) is not str or 'native' in key
                    or 'admission' in key or 'prefill' in key
                    for key in keys):
                raise QualificationRefused('stock-prefix-native-admission-intent')
        for rank in RANKS:
            row = raw_rows[rank]
            for key in ('engine_epoch', 'route_generation', 'storage_epoch'):
                if key in row and row[key] is not None and type(row[key]) is not int:
                    raise QualificationRefused('raw-placement-identity-invalid:%d' % rank)
            if 'engine_epoch' in row and row['engine_epoch'] != self.engine_epoch:
                raise QualificationRefused('foreign-placement-epoch')
        raw_shas = {rank: canonical_sha(raw_rows[rank]) for rank in RANKS}
        if type(raw.get('raw_sha256')) is dict:
            for rank in RANKS:
                if raw['raw_sha256'].get(str(rank), raw['raw_sha256'].get(rank)) != raw_shas[rank]:
                    raise QualificationRefused('raw-placement-digest-differs:%d' % rank)
        partial = {
            'phase': 'row4', 'kind': 'placement-partial',
            'run_id': self.run_id, 'window_id': window_id,
            'engine_epoch': self.engine_epoch, 'window': window,
            'raw_placement': raw_rows, 'engine_accounting': engine_rows,
            'raw_sha256': {str(rank): raw_shas[rank] for rank in RANKS},
        }
        try:
            native_read = self._native_read(window_id, raw)
            prefill = self._native_prefill(window_id, window=raw)
        except QualificationRefused as exc:
            if exc.evidence is None:
                exc.evidence = partial
            raise
        prefill_request = raw.get('native_prefill_request')
        missing = []
        if native_read.get('state') == 'unavailable':
            missing.append('native-report-read-unavailable')
        if (type(prefill_request) is not dict
                or prefill_request.get('complete') is not True):
            missing.append('native-prefill-request-unavailable')
        if prefill.get('state') != 'constructed':
            missing.append('native-prefill-ledger-unproved')
        if raw.get('teacher_rows_during_placement') not in (0, None):
            missing.append('placement-teacher-trace-overlapped')
        if raw.get('shipping_path_instrument_disabled') is not True:
            missing.append('shipping-path-instrument-not-proved-disabled')
        evidence = {
            'phase': 'row4', 'engine_epoch': self.engine_epoch,
            'model_settings_unchanged': True,
            'row1_sha256': row1_sha256,
            'teacher_artifact_sha256': artifact_sha256,
            'raw_placement': raw_rows, 'engine_accounting': engine_rows,
            'raw_sha256': {str(rank): raw_shas[rank] for rank in RANKS},
            'native_observer_packets': raw.get('native_observer_packets') or {},
            'window': window,
            'raw_graph_receipts': raw.get('raw_graph_receipts') or {},
            'engine_accounting_receipts': raw.get('engine_accounting_receipts') or {},
            'per_rank_bounds': raw.get('per_rank_bounds') or {},
            'replay_census': raw.get('replay_census') or {},
            'native_read': native_read,
            'native_prefill_request': prefill_request,
            'window_route_generation': raw.get('window_route_generation'),
            'native_prefill_attribution': prefill,
            'shipping_path_instrument_disabled':
                raw.get('shipping_path_instrument_disabled') is True,
            'teacher_trace_receipt': raw.get('teacher_trace_receipt'),
        }
        record = self._retain('row4', 'placement', evidence,
                              extra={'window_id': window_id,
                                     'missing': missing})
        return {
            'schema': 1, 'phase': 'row4', 'run_id': self.run_id,
            'engine_epoch': self.engine_epoch, 'evidence': evidence,
            'window': window, 'complete': not missing, 'missing': missing,
            'artifact_sha256': artifact_sha256, 'row1_sha256': row1_sha256,
            'record': record['record'],
        }

    def _validate_placement_window(self, window, window_id):
        if type(window) is not dict:
            raise PhaseUnavailable('placement-window-unavailable')
        if window.get('window_id') != window_id:
            raise QualificationRefused('foreign-placement-window')
        if window.get('kind') != 'short':
            raise QualificationRefused('placement-window-is-not-short')
        if window.get('replays') != PLACEMENT_REPLAYS:
            raise QualificationRefused('placement-replay-census-differs')
        if window.get('window_tokens') != PLACEMENT_WINDOW_TOKENS:
            raise QualificationRefused('placement-window-tokens-differ')
        if window.get('run_id') != self.run_id:
            raise QualificationRefused('foreign-placement-run')
        if window.get('engine_epoch') != self.engine_epoch:
            raise QualificationRefused('foreign-placement-epoch')
        if window.get('teacher_overlap') is not False:
            raise QualificationRefused('placement-overlaps-teacher-window')
        if window.get('native_phase') != PLACEMENT_PHASE:
            raise QualificationRefused('placement-native-phase-differs')
        admission = window.get('admission')
        if (type(admission) is not dict or admission.get('armed') is not True
                or admission.get('phase') != PLACEMENT_PHASE):
            raise QualificationRefused('placement-admission-not-proved')
        start, end = window.get('start_ns'), window.get('end_ns')
        if not (type(start) is int and type(end) is int and end > start):
            raise QualificationRefused('placement-window-interval-invalid')
        request = window.get('native_prefill_request')
        if type(request) is not dict:
            raise QualificationRefused('native-prefill-request-missing')
        if request.get('complete') is not True:
            raise QualificationRefused('native-prefill-request-incomplete')
        if (type(request.get('plan_sha256')) is not str
                or type(request.get('request_sha256')) is not list
                or not request['request_sha256']):
            raise QualificationRefused('native-prefill-request-identity-missing')
        generation = window.get('route_generation')
        if type(generation) is not int or generation < 0:
            raise QualificationRefused('placement-window-generation-missing')

    def _require_rank_rows(self, rows, label):
        if type(rows) is not dict:
            raise PhaseUnavailable(label + '-missing')
        if set(rows) != set(RANKS):
            raise QualificationRefused(label + '-missing-rank')
        for rank in RANKS:
            if type(rows[rank]) is not dict:
                raise QualificationRefused(label + '-not-an-object:%d' % rank)
            if rows[rank].get('rank', rank) != rank:
                raise QualificationRefused(label + '-rank-differs:%d' % rank)

    def _native_read(self, window_id, raw):
        # Read metadata belongs to each actual native report. No independent
        # provider, file read or byte-count declaration can replace it.
        packets = raw.get('native_observer_packets') or {}
        reads = {str(rank): copy.deepcopy((packets.get(rank) or
                 packets.get(str(rank)) or {}).get('native_report_read'))
                 for rank in RANKS}
        return {'state': 'retained' if all(type(v) is dict for v in reads.values())
                else 'unavailable', 'window_id': window_id,
                'limit_bytes': NATIVE_READ_BYTES, 'per_rank': reads}

    def _native_prefill(self, window_id, window=None):
        from e3_native_placement_evidence import regrade_packet
        packets = (window or {}).get('native_observer_packets') or {}
        receipts, proofs = {}, {}
        for rank in RANKS:
            packet = packets.get(rank, packets.get(str(rank)))
            proof = regrade_packet(packet, rank=rank,
                engine_epoch=self.engine_epoch, run_id=self.run_id,
                window_id=window_id,
                route_generation=((window or {}).get('window') or {}).get('route_generation'))
            proofs[str(rank)] = proof
            receipts[str(rank)] = (packet or {}).get('native_prefill_accounting')
        constructed = all(row['verdict'] == 'PASS' for row in proofs.values())
        return {'state': 'constructed' if constructed else 'unavailable',
                'basis': 'source-owned-native-worker-ledgers',
                'window_id': window_id, 'per_rank': copy.deepcopy(receipts),
                'proofs': proofs,
                'request': copy.deepcopy((window or {}).get('native_prefill_request'))}



class NativeRow1Producer:
    """Actual bounded HTTP producer for one ``row1-native`` admission window.

    Lifecycle (all acknowledgements retained verbatim)::

        prepare(target=full) -> pause_enter_split -> arm(row1-native)
        -> row1_configure(True) -> resume -> actual /v1/completions cohort
        -> bounded retirement poll -> pause -> row1_configure(False)
        -> row1_evidence -> status -> disarm_admission -> typed leave_full

    The request configuration is *only* the frozen queue plan.  There is no
    built-in workload, no retry, no partial-window success and no unbounded
    polling.  A failure leaves the first exception intact and attaches the
    actual lifecycle/request/cleanup evidence; cleanup never fabricates a
    return it could not prove.
    """

    def __init__(self, *, plan, url, model, request_runner=None):
        if type(plan) is not dict:
            raise QualificationError('row1-plan-not-frozen')
        if not _text(url):
            raise PhaseUnavailable('missing-row1-request-url')
        if not _text(model):
            raise QualificationError('model-required')
        self.plan = plan
        self.url = url
        self.model = model
        self.request_runner = request_runner

    # -- public ----------------------------------------------------------
    def run(self, driver, *, run_id, engine_epoch):
        workload = self.plan['workload']
        lifecycle = {}
        state = {'armed': False, 'configured': False, 'disarmed': False,
                 'disable_acknowledged': False, 'disable_attempted': False,
                 'leave_attempted': False}
        requests = None
        packets = []
        configure_acks = []
        disable_acks = []
        arm = None
        disarm = None
        terminal = None
        first = None
        cleanup = None
        try:
            prepared = driver._session('row1', 'prepare',
                                       target_route='full')
            self._require_route(prepared, 'full', 'row1-prepare')
            lifecycle['prepare'] = copy.deepcopy(prepared)

            paused = driver._session('row1', 'pause_enter_split',
                                     require_epoch=False)
            self._require_route(paused, 'split', 'row1-pause')
            if type(paused.get('phase')) is not None and paused.get('phase'):
                raise QualificationRefused('row1-split-admitted-before-arm')
            lifecycle['pause_enter_split'] = copy.deepcopy(paused)

            arm = driver._row1_arm()
            state['armed'] = True
            lifecycle['arm'] = copy.deepcopy(arm)
            qualified_generation = arm['route_generation']

            configure_acks = driver._row1_configure(True)
            state['configured'] = True
            for ack in configure_acks:
                if ack['ack'].get('route_generation') != qualified_generation:
                    raise QualificationRefused(
                        'row1-configure-route-generation-moved')
            lifecycle['configure'] = copy.deepcopy(configure_acks)

            resumed = driver._session('row1', 'resume', require_epoch=False)
            if not (resumed.get('resumed') is True
                    or resumed.get('already_running') is True):
                raise QualificationRefused('row1-split-not-resumed')
            lifecycle['resume'] = copy.deepcopy(resumed)

            requests = self._run_requests(run_id=run_id,
                                          engine_epoch=engine_epoch)
            lifecycle['requests'] = copy.deepcopy(requests)

            retired = self._await_retired(driver, workload)
            lifecycle['retired_status'] = copy.deepcopy(retired)

            paused_again = driver._session('row1', 'pause_enter_split',
                                           require_epoch=False)
            self._require_route(paused_again, 'split', 'row1-disable-pause')
            lifecycle['pause_before_disable'] = copy.deepcopy(paused_again)

            state['disable_attempted'] = True
            disable_acks = driver._row1_configure(False)
            state['disable_acknowledged'] = True
            lifecycle['disable'] = copy.deepcopy(disable_acks)

            packets, evidence_reply = driver._row1_packets()
            lifecycle['row1_evidence'] = copy.deepcopy(evidence_reply)

            after = driver._session('row1', 'status', require_epoch=False)
            terminal = self._terminal(driver, after, packets)

            disarm = driver._row1_disarm()
            state['disarmed'] = True
            lifecycle['disarm'] = copy.deepcopy(disarm)

            cleanup = self._return_full(driver, state, lifecycle, terminal)
            terminal['after_return_status'] = copy.deepcopy(
                lifecycle.get('final_status'))
        except BaseException as exc:  # noqa: BLE001 - preserve the first failure
            first = exc
        if first is not None:
            cleanup = self._cleanup(driver, lifecycle, state)
            evidence = {
                'lifecycle': copy.deepcopy(lifecycle),
                'requests': copy.deepcopy(requests),
                'workers': copy.deepcopy(packets),
                'configure_acks': copy.deepcopy(configure_acks),
                'disable_acks': copy.deepcopy(disable_acks),
                'terminal': copy.deepcopy(terminal),
                'cleanup': copy.deepcopy(cleanup),
                'plan_sha256': self.plan['plan_sha256'],
            }
            if isinstance(first, QualificationRefused):
                first.evidence = (evidence if first.evidence is None
                                  else {'partial': first.evidence,
                                        'producer_evidence': evidence})
                first.cleanup = cleanup
                raise first
            raise QualificationRefused(
                'row1-producer-exception:' + type(first).__name__,
                evidence=evidence, cleanup=cleanup) from None
        return {
            'workers': packets,
            'configure_acks': configure_acks,
            'disable_acks': disable_acks,
            'arm': arm, 'disarm': disarm,
            'lifecycle': lifecycle, 'requests': requests,
            'terminal': terminal, 'cleanup': cleanup,
            'plan_sha256': self.plan['plan_sha256'],
        }

    # -- actual requests -------------------------------------------------
    def _run_requests(self, *, run_id, engine_epoch):
        if self.request_runner is not None:
            result = self.request_runner({
                'phase': 'row1', 'run_id': run_id,
                'engine_epoch': engine_epoch, 'model': self.model,
                'url': self.url, 'plan': copy.deepcopy(self.plan['bodies']),
                'cohort': copy.deepcopy(self.plan['cohort']),
                'workload': self.plan['workload'],
            })
            if type(result) is not dict or result.get('complete') is not True:
                raise QualificationRefused(
                    'row1-request-runner-incomplete',
                    evidence={'request_result': copy.deepcopy(result)})
            return result
        workload = self.plan['workload']
        cohorts = self.plan['cohort']
        bodies = self.plan['bodies']
        record = {
            'schema': 's5-native-row1-requests/1',
            'phase': 'row1', 'run_id': run_id,
            'engine_epoch': engine_epoch,
            'manifest_sha256': self.plan['manifest_sha256'],
            'plan_sha256': self.plan['plan_sha256'],
            'run_index': self.plan['run_index'],
            'streams': len(bodies),
            'request_sha256': list(self.plan['request_sha256']),
            'complete': False,
        }
        states = []
        alive = []
        try:
            states = correctness._start_streams(
                self.url, self.model, bodies, cohorts, workload)
            record['started_ns'] = time.monotonic_ns()
            opened = correctness._wait_open(states, workload)
            record['open'] = opened
            if opened.get('complete') is not True:
                raise QualificationRefused('row1-requests-not-opened',
                                           evidence={'requests': record})
            timed_out = correctness._wait_finish(states, workload)
            record['finish_timeout_streams'] = list(timed_out)
            rows = self._request_rows(states, workload)
            record['streams_detail'] = rows
            if timed_out or not all(row['valid'] for row in rows):
                raise QualificationRefused('row1-requests-incomplete',
                                           evidence={'requests': record})
            record['complete'] = True
        finally:
            if states:
                alive = correctness._stop_streams(record, states, workload)
                record['readers_alive_after_join'] = list(alive)
            record['finished_ns'] = time.monotonic_ns()
        if alive:
            raise QualificationRefused('row1-request-readers-alive',
                                       evidence={'requests': record})
        return record

    def _request_rows(self, states, workload):
        rows = []
        for state in states:
            with state.lock:
                tokens = list(state.tokens)
                row = {
                    'stream': state.index,
                    'prompt_sha256': state.prompt_sha256,
                    'prompt_chars': state.prompt_chars,
                    'request_sha256': state.request_sha256,
                    'trajectory_sha256': correctness._sha256_json(tokens),
                    'length': len(tokens),
                    'finish_reason': state.finish_reason,
                    'done_marker': state.done_marker,
                    'usage_seen': state.usage_seen,
                    'usage_completion': state.usage_completion,
                    'first_advance_at_ns': state.first_advance_ns,
                    'finish_at_ns': state.finish_at_ns,
                    'error': copy.deepcopy(state.error),
                    'malformed': copy.deepcopy(state.malformed),
                    'thread_done': state.thread_done,
                }
            reasons = []
            if row['malformed'] is not None:
                reasons.append('malformed:' + str(row['malformed']))
            if row['error'] is not None:
                reasons.append('error:' + str(row['error']))
            if row['finish_reason'] != workload.expected_finish_reason:
                reasons.append('finish-mismatch')
            if row['length'] != workload.expected_output_tokens:
                reasons.append('length-mismatch')
            if row['usage_seen'] is not True:
                reasons.append('usage-missing')
            elif row['usage_completion'] != row['length']:
                reasons.append('usage-token-mismatch')
            if row['done_marker'] is not True:
                reasons.append('done-marker-missing')
            if row['thread_done'] is not True:
                reasons.append('reader-not-joined')
            row['valid'] = not reasons
            row['reasons'] = reasons
            rows.append(row)
        return rows

    # -- bounded retirement / terminal FULL ------------------------------
    def _await_retired(self, driver, workload):
        deadline = time.monotonic() + workload.drain_timeout_s
        status = None
        while True:
            status = driver._session('row1', 'status', require_epoch=False)
            owned = status.get('owned_work')
            if type(owned) is dict and owned.get('retained_request_row') is False:
                return status
            if time.monotonic() >= deadline:
                break
            time.sleep(min(0.05, max(0.01, workload.drain_timeout_s / 20.0)))
        raise QualificationRefused(
            'row1-request-row-not-retired',
            evidence={'status': copy.deepcopy(status)})

    def _terminal(self, driver, status, packets):
        """Validate the post-disable generation against actual status."""
        route = status.get('route')
        generation = status.get('route_generation')
        if route not in ('split', 'full'):
            raise QualificationRefused('row1-status-route-invalid')
        if type(generation) is not int or generation < 0:
            raise QualificationRefused('row1-status-generation-invalid')
        owned = status.get('owned_work')
        if type(owned) is not dict:
            raise QualificationRefused('row1-status-owned-work-missing')
        if owned.get('retained_request_row') is not False:
            raise QualificationRefused('row1-request-row-still-owned')
        for packet in packets:
            if packet['disabled_route_generation'] != generation:
                raise QualificationRefused(
                    'row1-disabled-generation-not-current')
            if packet['enabled'] is not False:
                raise QualificationRefused('row1-evidence-not-disabled')
        return {'route': route, 'route_generation': generation,
                'owned_work': copy.deepcopy(owned),
                'admission_intent': copy.deepcopy(
                    status.get('admission_intent')),
                'status': copy.deepcopy(status),
                'terminal_full': route == 'full'}

    # -- cleanup ---------------------------------------------------------
    def _return_full(self, driver, state, lifecycle, terminal):
        """Typed return after a proven disable; terminal FULL needs resume."""
        if terminal.get('terminal_full') is True:
            resumed = driver._session('row1', 'resume', require_epoch=False)
            if not (resumed.get('resumed') is True
                    or resumed.get('already_running') is True):
                raise QualificationRefused('row1-terminal-full-not-resumed')
            lifecycle['terminal_resume'] = copy.deepcopy(resumed)
            final = driver._session('row1', 'status', require_epoch=False)
            if (final.get('route') != 'full'
                    or final.get('admission_intent') is not None):
                raise QualificationRefused('row1-terminal-full-unproved')
            lifecycle['final_status'] = copy.deepcopy(final)
            return {'attempted': False, 'restored': True,
                    'route': 'full', 'terminal_full': True,
                    'owner': 'engine-terminal-handback'}
        state['leave_attempted'] = True
        left = driver._session('row1', 'leave_full', require_epoch=False)
        if left.get('route') != 'full' or left.get('typed_acks') is not True:
            raise QualificationRefused('row1-leave-full-unproved')
        admission = left.get('admission')
        if (type(admission) is not dict
                or admission.get('armed') is not False
                or admission.get('phase') != STOCK_PHASE):
            raise QualificationRefused('row1-admission-not-disarmed')
        drain = left.get('worker_drain')
        if type(drain) is not list or len(drain) != len(RANKS):
            raise QualificationRefused('row1-leave-full-drain-missing')
        lifecycle['leave_full'] = copy.deepcopy(left)
        final = driver._session('row1', 'status', require_epoch=False)
        if final.get('route') != 'full':
            raise QualificationRefused('row1-leave-full-not-applied')
        lifecycle['final_status'] = copy.deepcopy(final)
        return {'attempted': True, 'restored': True, 'route': 'full',
                'typed_acks': True, 'drain': copy.deepcopy(drain),
                'owner': 'engine-leave-split'}

    def _cleanup(self, driver, lifecycle, state):
        """Bounded best-effort disable/arm-off; records, never masks, failure."""
        record = {'attempted': True, 'steps': {}, 'errors': {},
                  'restored': False}
        if (state['configured'] and not state['disable_acknowledged']
                and not state['disable_attempted']):
            try:
                paused = driver._session('row1', 'pause_enter_split',
                                         require_epoch=False)
                record['steps']['pause'] = paused.get('route')
                if paused.get('route') == 'split':
                    disable = driver._row1_configure(False)
                    state['disable_acknowledged'] = True
                    record['steps']['disable'] = len(disable)
                    packets, _ = driver._row1_packets()
                    record['steps']['evidence'] = [
                        {'rank': packet['rank'],
                         'disabled_route_generation':
                             packet['disabled_route_generation']}
                        for packet in packets]
                    record['workers'] = copy.deepcopy(packets)
            except BaseException as exc:  # noqa: BLE001
                record['errors']['disable'] = self._code(exc)
        if state['armed'] and not state['disarmed']:
            try:
                disarm = driver._row1_disarm()
                state['disarmed'] = True
                record['steps']['disarm'] = copy.deepcopy(disarm)
            except BaseException as exc:  # noqa: BLE001
                record['errors']['disarm'] = self._code(exc)
        try:
            status = driver._session('row1', 'status', require_epoch=False)
            record['steps']['status'] = {'route': status.get('route'),
                                         'route_generation':
                                             status.get('route_generation')}
            if status.get('route') == 'full':
                if status.get('admission_intent') is not None:
                    disarm = driver._row1_disarm()
                    state['disarmed'] = True
                    record['steps']['disarm'] = copy.deepcopy(disarm)
                resumed = driver._session('row1', 'resume',
                                          require_epoch=False)
                if not (resumed.get('resumed') is True
                        or resumed.get('already_running') is True):
                    raise QualificationRefused('cleanup-full-not-resumed')
                record['steps']['resume'] = copy.deepcopy(resumed)
                final = driver._session('row1', 'status',
                                        require_epoch=False)
                record['steps']['final_status'] = copy.deepcopy(final)
                record['restored'] = (
                    final.get('route') == 'full'
                    and final.get('admission_intent') is None)
            elif not state['leave_attempted']:
                state['leave_attempted'] = True
                left = driver._session('row1', 'leave_full',
                                       require_epoch=False)
                record['steps']['leave_full'] = copy.deepcopy(left)
                record['restored'] = (
                    left.get('route') == 'full'
                    and left.get('typed_acks') is True
                    and isinstance(left.get('admission'), dict)
                    and left['admission'].get('armed') is False)
            else:
                record['errors']['return'] = 'leave-full-already-attempted'
            lifecycle['cleanup'] = copy.deepcopy(record)
        except BaseException as exc:  # noqa: BLE001
            record['errors']['return'] = self._code(exc)
        return record

    @staticmethod
    def _code(exc):
        return exc.code if isinstance(exc, QualificationRefused) \
            else type(exc).__name__

    @staticmethod
    def _require_route(reply, route, label):
        if type(reply) is not dict or reply.get('route') != route:
            raise QualificationRefused(label + '-route-differs')


def secrets_hex(n):
    import secrets
    return secrets.token_hex(n)


class NativePlacementRunner:
    """Real short-window placement over the native session + engine utility.

    The cohort is prepared on the stock full route, paused with an all-rank
    typed transition, traced by ``observer_begin`` while paused, resumed for the
    declared short window, stopped before ``observer_end``, and returned to full
    only through the engine's typed drain/transition acknowledgements.  The
    teacher plan is explicitly cleared on the frontend and the teacher trace is
    read back separately; neither is used as placement evidence.
    """

    def __init__(self, *, workload, url, prefill_plan=None,
                 snapshot_max_bytes=SNAPSHOT_MAX_BYTES,
                 replicates=PLACEMENT_REPLAYS,
                 window_tokens=PLACEMENT_WINDOW_TOKENS,
                 generation_budget=PLACEMENT_GENERATION_BUDGET,
                 window_s=10.0, clock=time.monotonic_ns):
        if type(workload) is not correctness.Workload:
            raise QualificationError('placement-workload-type-invalid')
        if snapshot_max_bytes < 64 * 1024 ** 2:
            raise QualificationError('placement-snapshot-too-small')
        self.base_workload = workload
        self.workload = replace(workload, generation_budget=generation_budget,
                                window_tokens=window_tokens, window_s=window_s)
        self.url = url
        # Raw queue plan for the actual native prefill request placed inside
        # this window; frozen against the run id/model at ``run`` time.  A
        # missing plan is a named INCONCLUSIVE; an empty decode window is never
        # relabelled as native prefill work.
        self.prefill_plan = prefill_plan
        self.snapshot_max_bytes = snapshot_max_bytes
        self.replicates = replicates
        self.clock = clock

    # -- typed-ack validation --------------------------------------------
    @staticmethod
    def _require_acks(reply, *, bank, label, epoch=None):
        if reply.get('typed_acks') is not True:
            raise QualificationRefused('partial-transition:' + label)
        acks = reply.get('transition_acks')
        if type(acks) is not list or len(acks) != len(RANKS):
            raise QualificationRefused('missing-worker:' + label)
        seen = set()
        for row in acks:
            if type(row) is not dict:
                raise QualificationRefused('partial-transition:' + label)
            rank = row.get('worker_rank')
            if type(rank) is not int or rank not in RANKS or rank in seen:
                raise QualificationRefused('missing-worker:' + label)
            seen.add(rank)
            if row.get('typed') != 'TransitionAck':
                raise QualificationRefused('partial-transition:' + label)
            if epoch is not None and (type(row.get('engine_epoch')) is not int
                                      or row['engine_epoch'] != epoch):
                raise QualificationRefused('foreign-transition-epoch:' + label)
            if type(row.get('applied_bank')) is not int or row['applied_bank'] != bank:
                raise QualificationRefused('partial-transition:' + label)
            if (type(row.get('drained_streams')) is not int
                    or row['drained_streams'] < 1):
                raise QualificationRefused('partial-transition:' + label)
            if (type(row.get('applied_route_generation')) is not int
                    or row['applied_route_generation'] < 1):
                raise QualificationRefused('partial-transition:' + label)
        if seen != set(RANKS):
            raise QualificationRefused('missing-worker:' + label)

    # -- engine aggregate -> per-rank action records ----------------------
    @staticmethod
    def _qualifications(reply, *, action, epoch):
        rows = reply.get('workers')
        if type(rows) is not list or len(rows) != len(RANKS):
            raise QualificationRefused('missing-worker:' + action)
        out = {}
        for row in rows:
            if type(row) is not dict:
                raise QualificationRefused('missing-worker:' + action)
            rank = row.get('worker_rank')
            if rank not in RANKS or rank in out:
                raise QualificationRefused('missing-worker:' + action)
            record = row.get('qualification')
            if type(record) is not dict:
                raise QualificationRefused('worker-returned-no-record:' + action)
            if record.get('engine_epoch') != epoch:
                raise QualificationRefused('foreign-worker-epoch:' + action)
            if (type(record.get('route_generation')) is not int
                    or record['route_generation'] < 0):
                raise QualificationRefused('worker-route-generation-missing:' + action)
            if (type(record.get('storage_epoch')) is not int
                    or record['storage_epoch'] < 1):
                raise QualificationRefused('worker-storage-epoch-missing:' + action)
            if not _text(record.get('worker_incarnation')):
                raise QualificationRefused('worker-incarnation-missing:' + action)
            if (type(record.get('drained_streams')) is not int
                    or record['drained_streams'] < 1):
                raise QualificationRefused('worker-drain-unproved:' + action)
            out[rank] = record
        return out

    # -- main -------------------------------------------------------------
    def run(self, driver, *, window_id, run_id, engine_epoch, model):
        if driver.control is None:
            raise PhaseUnavailable('missing-control-transport')
        if not _text(self.url):
            raise PhaseUnavailable('missing-placement-loopback-url')
        full_bank, split_bank = driver.route_banks()
        # Missing queue plan -> named INCONCLUSIVE before any engine action.
        prefill = freeze_prefill_plan(self.prefill_plan, run_id=run_id,
                                      model=model)
        record: dict = {'calls': [], 'streams': [],
                        'readers_alive_after_join': [], 'open': None,
                        'window': None, 'prefill_request': None}
        states = []
        prefill_states = []
        prefill_api = {'plan_sha256': prefill['plan_sha256'],
                       'manifest_sha256': prefill['manifest_sha256'],
                       'request_sha256': list(prefill['request_sha256']),
                       'streams': len(prefill['bodies']), 'open': None,
                       'streams_detail': [], 'complete': False}
        window_generation = None
        transition_attempted = False
        returned_full = False
        cleanup = None
        try:
            # 1. Stock full-route preparation.  No admission/native field is
            # ever part of this command (see driver.command_keys()).
            prep = driver._session('row4', 'prepare', target_route='full')
            if prep.get('route') != 'full':
                raise QualificationRefused('prepare-route-is-not-full')
            self._require_acks(prep, bank=full_bank, label='prepare-full',
                               epoch=engine_epoch)
            # 2. Frontend plan cleared: the shipping path must not admit a
            # native prefill or keep a teacher plan for the placement window.
            frontend_before = driver._frontend('row4', False, None)
            if frontend_before.get('enabled') is not False:
                raise QualificationRefused('shipping-path-instrument-enabled')
            # 3. Install the observer only on an empty, drained, full worker.
            configure = driver._worker('row4', 'teacher_configure', {
                'action': 'teacher_configure', 'enabled': True,
                'snapshot_max_bytes': self.snapshot_max_bytes})
            configure_rows = self._qualifications(
                configure, action='teacher_configure', epoch=engine_epoch)
            for rank in RANKS:
                if configure_rows[rank].get('route') != 'full':
                    raise QualificationRefused('teacher-configure-route-differs')
            # 4. Open the live cohort on the stock full route.
            cohort = correctness.build_cohort(self.workload)
            if (type(cohort) is not list or len(cohort) != PLACEMENT_STREAMS):
                raise QualificationRefused('placement-cohort-shape-differs')
            salt = correctness.build_cache_salt(run_id, 9000)
            bodies = [correctness.request_body(model, row['prompt'],
                                               self.workload, salt)
                      for row in cohort]
            states = correctness._start_streams(
                self.url, model, bodies, cohort, self.workload)
            opened = correctness._wait_open(states, self.workload)
            record['open'] = opened
            if opened.get('complete') is not True:
                raise QualificationRefused('placement-open-incomplete')
            # 5. Real all-rank pause + split transition, arming the declared
            # native placement phase only for this window.  The session enters
            # the split route first and then arms the Core-owned intent; the
            # stock preparation above carried no phase and never armed it.
            transition_attempted = True
            split = driver._session('row4', 'pause_enter_split',
                                    phase=PLACEMENT_PHASE)
            if split.get('route') != 'split':
                raise QualificationRefused('split-transition-not-applied')
            admission = self._placement_admission(split, engine_epoch)
            window_generation = split.get('route_generation_after')
            if type(window_generation) is not int or window_generation < 0:
                raise QualificationRefused('split-route-generation-missing')
            self._require_acks(split, bank=split_bank, label='enter-split',
                               epoch=engine_epoch)
            # 6. Engine-side measurement marker, then the worker observer.
            begin = driver._session('row4', 'overlap_begin',
                                    route='split',
                                    decode_streams=len(cohort))
            if begin.get('route') != 'split':
                raise QualificationRefused('overlap-begin-route-differs')
            start_ns = begin.get('measure_start_ns')
            proof = begin.get('clock_proof') or {}
            if proof.get('proved') is not True:
                raise QualificationRefused('overlap-begin-clock-unproved')
            observer_begin = driver._worker('row4', 'observer_begin', {
                'action': 'observer_begin', 'run_id': run_id,
                'window_id': window_id})
            begin_rows = self._qualifications(
                observer_begin, action='observer_begin', epoch=engine_epoch)
            for rank in RANKS:
                if begin_rows[rank].get('route') != 'split':
                    raise QualificationRefused('observer-begin-route-differs')
            # 7. Submit the actual native prefill request *inside* this window,
            # then resume.  Its prompts/config come only from the frozen plan;
            # the engine offer ledger must attribute it separately.
            prefill_states = correctness._start_streams(
                self.url, model, prefill['bodies'], prefill['cohort'],
                prefill['workload'])
            record['prefill_request'] = prefill_api
            resume = driver._session('row4', 'resume', require_epoch=False)
            if not (resume.get('resumed') is True
                    or resume.get('already_running') is True):
                raise QualificationRefused('resume-not-accepted')
            prefill_api['open'] = correctness._wait_open(
                prefill_states, prefill['workload'])
            if prefill_api['open'].get('complete') is not True:
                raise QualificationRefused('native-prefill-not-opened')
            starts = {state.index: state.count() for state in states}
            window = correctness._wait_window(states, starts, self.workload)
            record['window'] = window
            if window.get('complete') is not True:
                raise QualificationRefused('placement-window-incomplete')
            prefill_api['streams_detail'] = self._prefill_progress(
                prefill_states, prefill['workload'])
            prefill_api['advanced'] = any(
                row['first_advance_at_ns'] is not None
                for row in prefill_api['streams_detail'])
            if prefill_api['advanced'] is not True:
                raise QualificationRefused('native-prefill-not-started')
            prefill_api['token_count_proved'] = any(
                row['prompt_tokens_proven']
                for row in prefill_api['streams_detail'])
            if prefill_api['token_count_proved'] is not True:
                # No relabelling: an unproved prompt length is not a native
                # prefill request for the declared plan.
                raise QualificationRefused('native-prefill-token-count-unproved')
            end = driver._session('row4', 'overlap_end')
            if end.get('route') not in ('split', 'full'):
                raise QualificationRefused('overlap-end-route-differs')
            end_ns = end.get('measure_end_ns')
            if not (type(start_ns) is int and type(end_ns) is int
                    and end_ns > start_ns):
                raise QualificationRefused('placement-interval-unproved')
            # 8. Stop the owned cohort at the window boundary, then close the
            # observer on a quiet split worker.
            correctness._stop_streams(record, states, self.workload)
            correctness._stop_streams(prefill_api, prefill_states,
                                      prefill['workload'])
            prefill_api['complete'] = (
                prefill_api.get('readers_alive_after_join') == []
                and prefill_api.get('advanced') is True
                and prefill_api.get('token_count_proved') is True)
            if record.get('readers_alive_after_join'):
                raise QualificationRefused('placement-readers-alive')
            if prefill_api.get('complete') is not True:
                raise QualificationRefused('native-prefill-readers-alive')
            observer_end = driver._worker('row4', 'observer_end', {
                'action': 'observer_end', 'run_id': run_id,
                'window_id': window_id})
            end_rows = self._qualifications(
                observer_end, action='observer_end', epoch=engine_epoch)
            raw_rows, engine_rows, raw_shas = {}, {}, {}
            bounds, census = {}, {}
            graph_receipts, accounting_receipts, native_packets = {}, {}, {}
            for rank in RANKS:
                row = end_rows[rank]
                current = row.get('current_route_generation', row.get('route_generation'))
                route = row.get('current_route', row.get('route'))
                if not ((route == 'split' and current == window_generation)
                        or (route == 'full' and current == window_generation + 1
                            and row.get('state') == 'closed'
                            and row.get('observer_stopped') is True)):
                    raise QualificationRefused('observer-end-route-differs')
                if row.get('route_generation') != window_generation:
                    # A window that spans a route transition cannot be
                    # flattened into one split interval: retain the receipts and
                    # refuse instead of relabelling generations.
                    raise QualificationRefused(
                        'placement-window-spans-route-generation',
                        evidence={'window_generation': window_generation,
                                  'rank': rank,
                                  'observed_generation':
                                      row.get('route_generation')})
                if end.get('route') == 'full' and not (
                        row.get('current_route') == 'full'
                        and row.get('current_route_generation') == window_generation + 1):
                    raise QualificationRefused('full-return-worker-window-contradiction')
                receipt = row.get('receipt')
                accounting = row.get('engine_accounting')
                if type(receipt) is not dict:
                    raise PhaseUnavailable('raw-graph-receipt-unavailable:%d' % rank)
                if type(accounting) is not dict:
                    raise PhaseUnavailable('engine-accounting-unavailable:%d' % rank)
                raw = copy.deepcopy(receipt)
                raw_rows[rank] = raw
                engine_rows[rank] = copy.deepcopy(accounting)
                raw_shas[rank] = canonical_sha(raw)
                native_packets[rank] = copy.deepcopy(row)
                graph_receipts[rank] = copy.deepcopy(receipt)
                accounting_receipts[rank] = copy.deepcopy(accounting)
                bounds[rank] = {
                    'drained_streams': row.get('drained_streams'),
                    'route_generation': row.get('route_generation'),
                    'current_route_generation': row.get('current_route_generation'),
                    'storage_epoch': row.get('storage_epoch'),
                    'worker_incarnation': row.get('worker_incarnation'),
                    'replay_census': copy.deepcopy(row.get('replay_census')),
                    'window_bounds': copy.deepcopy(receipt.get('bounds')),
                }
                census[rank] = copy.deepcopy(row.get('replay_census'))
            # 9. Read the teacher trace state separately.  It is never used as
            # placement evidence and any retained row is reported, not hidden.
            trace = driver._worker('row4', 'teacher_evidence', {
                'action': 'teacher_evidence'})
            trace_rows = self._qualifications(
                trace, action='teacher_evidence', epoch=engine_epoch)
            teacher_rows = sum(len(row.get('positions') or [])
                               for row in trace_rows.values())
            teacher_trace_receipt = {str(rank): trace_rows[rank] for rank in RANKS}
            # 10. Return to full through the engine's own typed ACKs.
            left, cleanup = self._leave_full(driver, engine_epoch)
            returned_full = True
            # 11. Clear the observer/teacher instrumentation on the empty
            # worker and prove the shipping frontend is disabled.
            teacher_clear = driver._worker('row4', 'teacher_configure', {
                'action': 'teacher_configure', 'enabled': False,
                'snapshot_max_bytes': self.snapshot_max_bytes})
            clear_rows = self._qualifications(
                teacher_clear, action='teacher_configure', epoch=engine_epoch)
            frontend_after = driver._frontend('row4', False, None)
            disabled = (frontend_after.get('enabled') is False
                        and frontend_after.get('admitted') is not True
                        and frontend_after.get('pending') in (0, None)
                        and all(row.get('route') == 'full'
                                for row in clear_rows.values()))
            window = {
                'kind': 'short', 'run_id': run_id,
                'window_id': window_id, 'engine_epoch': engine_epoch,
                'start_ns': start_ns, 'end_ns': end_ns,
                'host_boundary_basis': 'engine control observations; per-rank native bounds retained separately',
                'window_tokens': PLACEMENT_WINDOW_TOKENS,
                'replays': self.replicates,
                'observed_replays': {
                    rank: _replay_delta(((native_packets[rank].get('decode_window') or {})
                        .get('binding') or {}).get('router_census')) for rank in RANKS},
                'replay_count_basis': 'actual native binding router delta, not the client token target',
                'teacher_overlap': False,
                'teacher_window_end_ns': (driver._teacher_window or {}).get('ended_ns'),
                'cohort_id': 'placement-' + canonical_sha(
                    {'run_id': run_id, 'window_id': window_id})[:24],
                'decode_progress': end.get('decode_progress'),
                'native_phase': PLACEMENT_PHASE,
                'admission': admission,
                'route_generation': window_generation,
                'native_prefill_request': copy.deepcopy(prefill_api),
            }
            return {
                'window': window,
                'native_prefill_request': copy.deepcopy(prefill_api),
                'window_route_generation': window_generation,
                'raw_placement': raw_rows, 'engine_accounting': engine_rows,
                'native_observer_packets': native_packets,
                'raw_sha256': {str(rank): raw_shas[rank] for rank in RANKS},
                'raw_graph_receipts': graph_receipts,
                'engine_accounting_receipts': accounting_receipts,
                'per_rank_bounds': bounds, 'replay_census': census,
                'teacher_rows_during_placement': teacher_rows,
                'teacher_trace_receipt': teacher_trace_receipt,
                'shipping_path_instrument_disabled': bool(disabled),
                'cleanup': cleanup,
                'stock_prefix': {
                    'prepare_keys': [row['keys'] for row in
                                     driver.command_keys('row4', 'prepare')],
                    'native_admission_intent': False,
                },
                'frontend': {'before': frontend_before, 'after': frontend_after},
                'overlap_end': end, 'leave_full': left,
                'calls': record.get('calls'),
            }
        except BaseException as exc:  # noqa: BLE001 - preserve the first failure
            if transition_attempted and not returned_full:
                cleanup = self._attempt_cleanup(driver, engine_epoch)
            partial = {'kind': 'placement-partial',
                       'run_id': run_id, 'window_id': window_id,
                       'engine_epoch': engine_epoch,
                       'collector_error': exception_details(exc, getattr(driver, 'control', None)),
                       'open': record.get('open'),
                       'client_window': record.get('window'),
                       'reads_alive_after_join':
                           record.get('readers_alive_after_join'),
                       'prefill_request': copy.deepcopy(prefill_api),
                       'calls': [call for call in record.get('calls') or []]}
            if isinstance(exc, QualificationRefused):
                exc.cleanup = cleanup
                if exc.evidence is None:
                    exc.evidence = partial
                raise
            if isinstance(exc, QualificationError):
                raise QualificationRefused(
                    type(exc).__name__, cleanup=cleanup,
                    evidence=partial) from None
            raise QualificationRefused(
                'placement-exception:' + type(exc).__name__,
                cleanup=cleanup, evidence=partial) from None
        finally:
            if states:
                correctness._stop_streams(record, states, self.workload)
            if prefill_states:
                correctness._stop_streams(prefill_api, prefill_states,
                                          prefill['workload'])

    @staticmethod
    def _prefill_progress(states, workload):
        rows = []
        for state in states:
            with state.lock:
                row = {
                    'stream': state.index,
                    'prompt_sha256': state.prompt_sha256,
                    'prompt_chars': state.prompt_chars,
                    'request_sha256': state.request_sha256,
                    'prompt_tokens': state.prompt_tokens,
                    'prompt_token_ids_len': state.prompt_token_ids_len,
                    'first_advance_at_ns': state.first_advance_ns,
                    'finish_at_ns': state.finish_at_ns,
                    'finish_reason': state.finish_reason,
                    'length': len(state.tokens),
                    'error': copy.deepcopy(state.error),
                    'malformed': copy.deepcopy(state.malformed),
                }
            row['nominal_prompt_tokens'] = workload.prompt_tokens
            row['prompt_tokens_proven'] = (
                type(row['prompt_tokens']) is int and row['prompt_tokens'] > 0
                and row['prompt_tokens'] == row['prompt_token_ids_len'])
            # agent_replay.text uses a calibrated words/token estimate. The
            # unchanged seeded text is the request; actual token geometry is
            # bound by returned prompt ids and usage, not that estimate.
            row['prompt_tokens_match_declared'] = (
                row['prompt_tokens'] is not None
                and row['prompt_tokens'] == workload.prompt_tokens)
            rows.append(row)
        return rows

    def _placement_admission(self, split, engine_epoch):
        """Require the declared native phase to be armed by the engine itself."""
        if split.get('phase') != PLACEMENT_PHASE:
            raise QualificationRefused('placement-admission-phase-differs')
        record = split.get('admission')
        if type(record) is not dict:
            raise QualificationRefused('placement-admission-not-proved')
        intent = record.get('admission_intent')
        if (type(intent) is not dict
                or intent.get('phase') != PLACEMENT_PHASE
                or intent.get('armed') is not True
                or intent.get('engine_epoch') != engine_epoch
                or intent.get('owner') != 'engine-core'):
            raise QualificationRefused('placement-admission-intent-differs')
        if (record.get('armed') is not True
                or record.get('phase') != PLACEMENT_PHASE):
            raise QualificationRefused('placement-admission-not-armed')
        return copy.deepcopy(record)

    def _leave_full(self, driver, engine_epoch):
        full_bank, _ = driver.route_banks()
        left = driver._session('row4', 'leave_full', require_epoch=False)
        if left.get('route') != 'full':
            raise QualificationRefused('leave-full-not-applied')
        if left.get('typed_acks') is not True:
            raise QualificationRefused('partial-transition:leave-full')
        admission = left.get('admission')
        if (type(admission) is not dict
                or admission.get('armed') is not False
                or admission.get('phase') != STOCK_PHASE):
            raise QualificationRefused('placement-admission-not-disarmed')
        drain = left.get('worker_drain')
        if type(drain) is not list or len(drain) != len(RANKS):
            raise QualificationRefused('missing-worker:leave-full')
        seen = set()
        for row in drain:
            rank = row.get('worker_rank') if type(row) is dict else None
            if rank not in RANKS or rank in seen:
                raise QualificationRefused('missing-worker:leave-full')
            seen.add(rank)
            if (type(row.get('drained_streams')) is not int
                    or row['drained_streams'] < 1):
                raise QualificationRefused('cleanup-drain-unproved')
        if seen != set(RANKS):
            raise QualificationRefused('missing-worker:leave-full')
        # Validate the transition ACKs that the session also returns.
        self._require_acks(left, bank=full_bank, label='leave-full',
                           epoch=engine_epoch)
        return left, {'attempted': True, 'restored': True,
                      'route': 'full', 'typed_acks': True,
                      'worker_drain': drain,
                      'admission': copy.deepcopy(admission),
                      'owner': 'engine-leave-split'}

    def _attempt_cleanup(self, driver, engine_epoch):
        """Try a real typed return to full; never resume or invent a drain."""
        try:
            _left, record = self._leave_full(driver, engine_epoch)
            return record
        except BaseException as exc:  # noqa: BLE001 - preserve the first failure
            code = exc.code if isinstance(exc, QualificationRefused) else type(exc).__name__
            return {'attempted': True, 'restored': False,
                    'route': None, 'error': code,
                    'reason': 'engine did not prove an all-rank drained return; '
                              'ownership stays with the engine'}


def _replay_delta(census):
    """Observed replay count for one rank, only from queried integer values."""
    if type(census) is not dict:
        return None
    for key in ('total',):
        value = census.get(key)
        if type(value) is int and value >= 0:
            return value
    return None


class RetainedBankReader:
    """Rejected legacy misinterpretation; a file cannot prove a native report read."""

    def __init__(self, path):
        self.path = os.fspath(path)

    def __call__(self, request):
        raise QualificationRefused('bank-file-is-not-native-correlation-report')


__all__ = ['ImmutableStore', 'LoopbackControl', 'NativePlacementRunner',
           'NativeRow1Producer', 'PhaseConflict', 'PhaseUnavailable',
           'QualificationDriver', 'QualificationError', 'QualificationRefused',
           'RetainedBankReader', 'canonical_sha', 'freeze_row1_plan']
