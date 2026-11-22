"""Source-pinned native prefill offer ledger (``s5-native-prefill-accounting/1``).

The stock engine-offer vocabulary describes the runner/sampler *decode* path;
``LaneForward`` and the lane dispatcher bypass it.  This module adds the
smallest separate, source-declared provider for work admitted through
``LaneDispatcher``:

* the vocabulary below is declared before measuring.  A suboperation's stream
  role is fixed by the table, so an auxiliary launch cannot be relabelled as
  main work (or the reverse) by construction;
* every offer is created by :class:`activity_accounting.CorrelationBridge`;
  this module never builds an offer row and never invents ``thread`` /
  ``system_thread``;
* offers are opened by reversible shadows over the *actual* lane objects
  (dispatcher handlers, QSA layer driver phases, the QSA operation's bucket
  fill, and the pinned SharedExperts / PLE owner methods).  Each shadow is
  installed only after its source file hash matches a pin supplied by the
  caller, and is removed with the exact inverse;
* completion is independently credited: every successful bounded operation
  records its own event on its role stream, and the seal host-queries it.
  A flag alone can never make the window PASS.

The receipt is separate from the raw graph-only placement receipt and from the
decode engine-accounting receipt.  This module performs no CUDA work, no
engine control, and no writes to native loss flags.
"""
import functools
import hashlib
import inspect
import os
import threading
import time
from collections import Counter
from contextlib import contextmanager, nullcontext

try:  # imported from the generated package when installed there
    from .activity_accounting import KINDS, api_kind as _api_kind
    from .activity_accounting import host_return_ok as _host_return_ok
except ImportError:  # pragma: no cover - package-relative import
    from .activity_accounting import KINDS, api_kind as _api_kind
    from .activity_accounting import host_return_ok as _host_return_ok

try:  # same exception object the lane raises; package() rewrites this import
    from .e3_types import E3Pending as _PENDING
except ImportError:  # pragma: no cover - standalone module use
    _PENDING = type('E3Pending', (Exception,), {})

SCHEMA = 's5-native-prefill-accounting/1'
NATIVE_SCHEMA = 's5ro-correlation/2'
ENABLE_ENV = 'FLASH_S5_NATIVE_PREFILL_OFFERS'
MAX_OFFERS = 32768
MAX_OWNERS = 4096
MAX_RECEIPT_BYTES = 16 * 1024 * 1024
THREAD_ID_TYPES = ('default-pthread-low32', 'system-gettid')

KINDS = tuple(KINDS)
SEGMENTS = ('prepare', 'pp_receive', 'prefix', 'layer', 'suffix', 'pp_send',
            'state_finalize')
QSA_PHASES = ('pre', 'bucket', 'tail')
ROLES = ('prefill_main', 'shared_experts_aux', 'ple_aux', 'copy')

# CUPTI_ACTIVITY_MEMCPY_KIND_* values from the pinned CUPTI 13 header.  Kept
# identical to engine_offers.COPY_KIND; asserted by the CPU suite.
COPY_KIND = {'unknown': 0, 'H2D': 1, 'D2H': 2, 'H2A': 3, 'A2H': 4,
             'A2A': 5, 'A2D': 6, 'D2A': 7, 'D2D': 8, 'H2H': 9, 'P2P': 10}
ALL_COPY_KINDS = tuple(sorted(set(COPY_KIND.values())))
_HISTORY_READ_COPY_KINDS = (COPY_KIND['H2D'], COPY_KIND['D2H'],
                            COPY_KIND['D2D'], COPY_KIND['A2A'])
_COPY_ONLY = ('memcpy',)
_ALL = KINDS
_ZERO = ()

# --- declared vocabulary (before measuring) ---------------------------------
# ``role`` is fixed per suboperation.  ``segment`` / ``phase`` bind the offer
# to the actual lane cursor position.  ``kinds``/``copy_kinds`` are explicit.
FAMILIES = {
    'prefill_segment': {
        'prepare': {'expectation': 'activity_or_zero', 'kinds': _ALL,
                    'copy_kinds': ALL_COPY_KINDS, 'role': 'prefill_main',
                    'segment': 'prepare'},
        'pp_receive': {'expectation': 'activity_or_zero', 'kinds': _ALL,
                       'copy_kinds': ALL_COPY_KINDS, 'role': 'prefill_main',
                       'segment': 'pp_receive'},
        'prefix': {'expectation': 'activity_or_zero', 'kinds': _ALL,
                   'copy_kinds': ALL_COPY_KINDS, 'role': 'prefill_main',
                   'segment': 'prefix'},
        # A direct (non-QSA) layer executes its own arithmetic in this offer.
        'layer': {'expectation': 'activity_or_zero', 'kinds': _ALL,
                  'copy_kinds': ALL_COPY_KINDS, 'role': 'prefill_main',
                  'segment': 'layer', 'branch': 'direct'},
        # A QSA layer pauses; every activity is owned by a nested qsa offer.
        'layer_qsa': {'expectation': 'container', 'kinds': _ZERO,
                      'copy_kinds': _ZERO, 'role': 'prefill_main',
                      'segment': 'layer', 'branch': 'driver'},
        'suffix': {'expectation': 'activity_or_zero', 'kinds': _ALL,
                   'copy_kinds': ALL_COPY_KINDS, 'role': 'prefill_main',
                   'segment': 'suffix'},
        'pp_send': {'expectation': 'activity_or_zero', 'kinds': _ALL,
                    'copy_kinds': ALL_COPY_KINDS, 'role': 'prefill_main',
                    'segment': 'pp_send'},
        'state_finalize': {'expectation': 'activity_or_zero', 'kinds': _ALL,
                           'copy_kinds': ALL_COPY_KINDS,
                           'role': 'prefill_main',
                           'segment': 'state_finalize'},
    },
    'qsa': {
        # Indexer, K/V store and the attention projections: main compute.
        'index_store': {'expectation': 'activity_or_zero', 'kinds': _ALL,
                        'copy_kinds': ALL_COPY_KINDS,
                        'role': 'prefill_main', 'phase': 'pre'},
        # The history bucket copies one bounded arena piece on the owned copy
        # stream.  A kernel or memset on this role is refused by the grader.
        'bucket_fill': {'expectation': 'activity_or_zero',
                        'kinds': _COPY_ONLY,
                        'copy_kinds': _HISTORY_READ_COPY_KINDS,
                        'role': 'copy', 'phase': 'bucket',
                        'copy_role': True},
        # Eager attention for one host-submitted bucket.
        'selection_readback': {'expectation': 'activity_or_zero',
                               'kinds': _COPY_ONLY,
                               'copy_kinds': (COPY_KIND['D2H'],),
                               'role': 'copy', 'phase': 'bucket',
                               'copy_role': True},
        'sparse_bucket_fill': {'expectation': 'activity_or_zero',
                               'kinds': _COPY_ONLY,
                               'copy_kinds': (COPY_KIND['H2D'],),
                               'role': 'copy', 'phase': 'bucket',
                               'copy_role': True},
        'attention': {'expectation': 'activity_or_zero', 'kinds': _ALL,
                      'copy_kinds': ALL_COPY_KINDS,
                      'role': 'prefill_main', 'phase': 'bucket'},
        'tail': {'expectation': 'activity_or_zero', 'kinds': _ALL,
                 'copy_kinds': ALL_COPY_KINDS,
                 'role': 'prefill_main', 'phase': 'tail'},
        # LanePrivateChunk.scope runs these exact initializations inside the
        # QSA layer container, before the QsaLayerDriver child: uid.fill_(-2)
        # plus one 4-byte D2D seed copy, and each retained private Marlin lock
        # array's first zero. They are declared leaves, never generic
        # activity_or_zero work and never a relaxation of the container.
        'uid_init': {'expectation': 'activity_or_zero',
                     'kinds': ('kernel', 'memcpy'),
                     'copy_kinds': (COPY_KIND['D2D'],),
                     'role': 'prefill_main'},
        'lock_init': {'expectation': 'activity_or_zero',
                      'kinds': ('kernel', 'memset'),
                      'copy_kinds': _ZERO,
                      'role': 'prefill_main'},
    },
    'shared_experts': {
        # Real pinned SharedExperts owner methods.  The async branch executes
        # ``_layer`` under the owned prefill-green auxiliary ``_stream``.
        'maybe_forward_async': {'expectation': 'activity_or_zero',
                                'kinds': _ALL,
                                'copy_kinds': ALL_COPY_KINDS,
                                'role': 'shared_experts_aux',
                                'owner_attr': '_stream'},
        'forward': {'expectation': 'activity_or_zero', 'kinds': _ALL,
                    'copy_kinds': ALL_COPY_KINDS, 'role': 'prefill_main'},
        'wait': {'expectation': 'zero_work', 'kinds': _ZERO,
                 'copy_kinds': _ZERO, 'role': 'prefill_main'},
    },
    'ple': {
        # The pinned UVA look-up runs under the owned prefetch stream.
        'start_prefetch': {'expectation': 'activity_or_zero', 'kinds': _ALL,
                           'copy_kinds': ALL_COPY_KINDS, 'role': 'ple_aux',
                           'owner_attr': '_prefetch_stream'},
        # The join/reduce/select path runs on the caller's prefill main stream.
        'finalize_prefetch': {'expectation': 'activity_or_zero',
                              'kinds': _ALL,
                              'copy_kinds': ALL_COPY_KINDS,
                              'role': 'prefill_main'},
    },
}


class NativePrefillError(RuntimeError):
    pass


class SourcePinError(NativePrefillError):
    pass


class ShadowDrift(NativePrefillError):
    pass


def _sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _file_sha(path):
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def live_pins(*objects):
    """Offline/test helper: pin the files that currently define ``objects``.

    Production wiring passes the immutable package manifest hashes instead.
    The pin still exists before any offer is measured.
    """
    pins = {}
    for obj in objects:
        path = inspect.getfile(type(obj) if not inspect.isclass(obj) else obj)
        pins[path] = _file_sha(path)
    return pins


def _pin_for(pins, target):
    path = inspect.getfile(type(target))
    row = pins.get(path)
    if row is None:
        row = pins.get(os.path.basename(path))
    if row is None:
        raise SourcePinError('source pin missing for ' + os.path.basename(path))
    if type(row) is not str or len(row) != 64 or any(
            c not in '0123456789abcdef' for c in row):
        raise SourcePinError('invalid source pin for ' + os.path.basename(path))
    actual = _file_sha(path)
    if actual != row:
        raise SourcePinError('source pin drift: ' + os.path.basename(path))
    return actual


def declared_suboperation(family, suboperation):
    body = FAMILIES.get(family)
    if body is None:
        raise NativePrefillError('undeclared family: ' + str(family))
    decl = body.get(suboperation)
    if decl is None:
        raise NativePrefillError('undeclared suboperation: ' +
                                 str(family) + '/' + str(suboperation))
    return dict(decl, family=family, suboperation=suboperation)


def wire(family, suboperation):
    return family + '/' + suboperation


def _bounded(value, reason, limit=128):
    if type(value) is not str or not value or len(value) > limit:
        raise NativePrefillError(reason)
    return value


def _transaction_identity(transaction):
    """Bounded identity of a canonical TransactionId (uid is hashed only)."""
    if isinstance(transaction, dict):
        fields = {}
        for name in ('engine_epoch', 'request_generation', 'chunk_start',
                     'chunk_end', 'txn_seq'):
            value = transaction.get(name)
            if type(value) is not int:
                raise NativePrefillError('transaction missing ' + name)
            fields[name] = value
        digest = transaction.get('request_uid_sha256')
        if type(digest) is not str or len(digest) != 64:
            raise NativePrefillError('transaction missing request uid')
        fields['request_uid_sha256'] = digest
        return fields
    fields = {}
    for name in ('engine_epoch', 'request_generation', 'chunk_start',
                 'chunk_end', 'txn_seq'):
        value = getattr(transaction, name, None)
        if type(value) is not int:
            raise NativePrefillError('transaction missing ' + name)
        fields[name] = value
    uid = getattr(transaction, 'request_uid', None)
    if type(uid) is not str or not uid:
        raise NativePrefillError('transaction missing request uid')
    fields['request_uid_sha256'] = _sha_text(uid)
    return fields


def _identity_value(stream_identity, stream, label):
    row = stream_identity(stream)
    if not isinstance(row, dict):
        raise NativePrefillError(label + ': stream identity shape')
    context_id = row.get('context_id')
    stream_id = row.get('stream_id')
    if type(context_id) is not int or context_id < 1:
        raise NativePrefillError(label + ': context id')
    if type(stream_id) is not int or stream_id < 1:
        raise NativePrefillError(label + ': stream id')
    return {'context_id': context_id, 'stream_id': stream_id}


class NativePrefillLedger:
    """One worker's source-declared prefill offer window.

    Creation is inert.  ``declare_streams`` queries the real prefill, copy and
    auxiliary stream identities once, before any offer.  Offers then exist
    only through :class:`CorrelationBridge`, and only from inside an installed
    shadow (or an explicit lane binding in a test).
    """

    def __init__(self, bridge, *, worker, rank, epoch, generation,
                 route_generation, storage_epoch, run_id, window_id,
                 stream_identity, torch_module=None, clock=time.monotonic_ns,
                 env=None):
        if not callable(stream_identity):
            raise NativePrefillError('queried stream identity is required')
        if type(rank) is not int or rank < 0:
            raise NativePrefillError('invalid rank')
        if type(epoch) is not int or epoch < 1:
            raise NativePrefillError('invalid epoch')
        if type(generation) is not int or generation < 1:
            raise NativePrefillError('invalid generation')
        if type(route_generation) is not int or route_generation < 1:
            raise NativePrefillError('invalid route generation')
        if generation != route_generation:
            raise NativePrefillError('engine and route generation differ')
        if type(storage_epoch) is not int or storage_epoch < 0:
            raise NativePrefillError('invalid storage epoch')
        if getattr(bridge, 'epoch', epoch) != epoch:
            raise NativePrefillError('bridge epoch differs')
        if getattr(bridge, 'generation', generation) != generation:
            raise NativePrefillError('bridge generation differs')
        if getattr(bridge, 'rank', rank) != rank:
            raise NativePrefillError('bridge rank differs')
        self._env = dict(os.environ if env is None else env)
        if self._env.get(ENABLE_ENV) != '1':
            raise NativePrefillError('native prefill offers are not enabled')
        self.bridge = bridge
        self.worker = _bounded(worker, 'invalid worker')
        self.rank = rank
        self.epoch = epoch
        self.generation = generation
        self.route_generation = route_generation
        self.storage_epoch = storage_epoch
        self.run_id = _bounded(run_id, 'invalid run id')
        self.window_id = _bounded(window_id, 'invalid window id')
        self.stream_identity = stream_identity
        self.torch = torch_module
        self.clock = clock
        self._lock = threading.RLock()
        self._local = threading.local()
        self._meta = {}
        self._owners = {}
        self._owner_order = []
        self._streams = {}
        self._invalid = []
        self._invalidated = False
        self._started = None
        self._finished = None
        self._sealed = None
        self._shadows = []

    # -- lifecycle ---------------------------------------------------------
    def declare_streams(self, *, prefill_stream, copy_stream=None,
                        aux_streams=None):
        """Query and declare every role identity before the first offer.

        ``aux_streams`` maps ``shared_experts_aux`` / ``ple_aux`` to the owned
        prefill-green replacement stream(s).  Every candidate for a role must
        share one queried identity; a stock/original auxiliary is refused
        because its context is not the prefill green context.
        """
        with self._lock:
            if self._started is not None or self._meta:
                raise NativePrefillError('streams already declared')
            main = _identity_value(self.stream_identity, prefill_stream,
                                   'prefill_main')
            green = main['context_id']
            streams = {'prefill_main': {'stream': prefill_stream,
                                        'identity': main,
                                        'non_green_control': False}}
            if copy_stream is not None:
                copy = _identity_value(self.stream_identity, copy_stream,
                                       'copy')
                streams['copy'] = {
                    'stream': copy_stream, 'identity': copy,
                    'non_green_control': copy['context_id'] != green}
            for role, supplied in dict(aux_streams or {}).items():
                if role not in ('shared_experts_aux', 'ple_aux'):
                    raise NativePrefillError('undeclared auxiliary role')
                candidates = (supplied if isinstance(supplied, (tuple, list))
                              else (supplied,))
                if not candidates:
                    raise NativePrefillError('empty auxiliary stream role')
                identities = []
                for stream in candidates:
                    if stream is None:
                        raise NativePrefillError('disabled auxiliary stream')
                    row = _identity_value(self.stream_identity, stream,
                                          role)
                    if row['context_id'] != green:
                        raise NativePrefillError(
                            'auxiliary stream is not the prefill green context')
                    identities.append(row)
                if any(row != identities[0] for row in identities[1:]):
                    raise NativePrefillError('ambiguous auxiliary identity')
                streams[role] = {'stream': candidates[0],
                                 'identity': identities[0],
                                 'non_green_control': False}
            native = self.bridge.window()
            start = native.get('start_ns')
            if type(start) is not int or start < 0:
                raise NativePrefillError('native window start missing')
            self._streams = streams
            self._green_context_id = green
            self._started = int(start)
            self._native_route = native.get('route_generation')
            if self._native_route != self.route_generation:
                raise NativePrefillError('native route generation differs')
            self.owner_start('lane', 'prefill_main')
            return self.window()

    def _require_started(self):
        if self._started is None:
            raise NativePrefillError('streams are not declared')
        if self._finished is not None:
            raise NativePrefillError('window is finished')

    def _invalidate(self, reason):
        reason = str(reason)
        with self._lock:
            self._invalidated = True
            if reason not in self._invalid:
                self._invalid.append(reason)

    # -- owners ------------------------------------------------------------
    def owner_start(self, name, role, *, transaction=None, stream=None):
        """Start one explicit per-window owner mark and query its stream."""
        self._require_started()
        name = _bounded(name, 'invalid owner name')
        if role not in ROLES:
            raise NativePrefillError('undeclared owner role')
        declared = self._streams.get(role)
        if declared is None:
            raise NativePrefillError('owner role has no declared stream')
        if stream is not None:
            got = _identity_value(self.stream_identity, stream, role)
            if got != declared['identity']:
                self._invalidate('owner_stream_drift:' + role)
                raise NativePrefillError('owner stream differs from its role')
        with self._lock:
            if name in self._owners:
                raise NativePrefillError('duplicate owner')
            if len(self._owners) >= MAX_OWNERS:
                raise NativePrefillError('owner capacity')
            row = {
                'name': name, 'role': role,
                'context_id': declared['identity']['context_id'],
                'stream_id': declared['identity']['stream_id'],
                'transaction': (None if transaction is None
                                else _transaction_identity(transaction)),
                'start_ns': int(self.clock()), 'finish_ns': None,
                'submitted': 0, 'completed': 0, 'events': 0,
                '_events': [],
            }
            self._owners[name] = row
            self._owner_order.append(name)
            return dict(row, _events=[])

    def owner_finish(self, name, *, events=()):
        """Finish one owner mark; ``events`` are host-queried credits."""
        name = _bounded(name, 'invalid owner name')
        with self._lock:
            row = self._owners.get(name)
            if row is None:
                raise NativePrefillError('unknown owner')
            if row['finish_ns'] is not None:
                raise NativePrefillError('owner already finished')
            row['finish_ns'] = int(self.clock())
            for event in events:
                try:
                    ready = bool(event.query())
                except BaseException:
                    self._invalidate('owner_completion_query:' + name)
                    ready = False
                row['submitted'] += 1
                row['completed'] += 1 if ready else 0
                row['events'] += 1
            if row['finish_ns'] < row['start_ns']:
                self._invalidate('owner_time_order:' + name)
            return self.owner_record(name)

    def owner_record(self, name):
        row = self._owners[name]
        return {k: v for k, v in row.items() if not k.startswith('_')}

    # -- offers ------------------------------------------------------------
    def _ctx_binding(self):
        stack = getattr(self._local, 'bindings', None)
        return stack[-1] if stack else None

    @contextmanager
    def _binding(self, binding):
        stack = getattr(self._local, 'bindings', None)
        if stack is None:
            stack = []
            self._local.bindings = stack
        stack.append(binding)
        try:
            yield
        finally:
            if not stack or stack[-1] is not binding:
                raise NativePrefillError('prefill binding stack')
            stack.pop()

    def _role_stream(self, role):
        declared = self._streams.get(role)
        if declared is None:
            return None
        return declared

    def _expected(self, decl, expected):
        expected = {} if expected is None else dict(expected)
        for kind, count in expected.items():
            if kind not in KINDS or type(count) is not int or count < 0:
                raise NativePrefillError('invalid expected counts')
            if kind not in decl['kinds']:
                raise NativePrefillError('expected kind is not declared')
        return expected

    @contextmanager
    def _offer(self, decl, *, expected=None, transaction=None, op_seq=None,
               segment=None, phase=None, cursor=None, owner=None,
               inherit=True, inherit_phase=True, wire_name=None):
        """One bridge-owned offer with bounded declared metadata."""
        self._require_started()
        if self._invalidated:
            raise NativePrefillError('prefill observation is invalidated')
        expected = self._expected(decl, expected)
        role = decl['role']
        declared = self._role_stream(role)
        if declared is None:
            raise NativePrefillError('role stream is not declared: ' + role)
        ctx = self._ctx_binding()
        if inherit and transaction is None and ctx is not None:
            transaction = ctx['transaction']
            op_seq = ctx['op_seq']
            segment = ctx['segment']
            if inherit_phase and phase is None:
                phase = ctx['phase']
            if cursor is None:
                cursor = ctx['cursor']
            if owner is None:
                owner = ctx.get('owner')
        if transaction is not None:
            transaction = _transaction_identity(transaction)
        if op_seq is not None and (type(op_seq) is not int or op_seq < 0):
            raise NativePrefillError('invalid op_seq')
        if segment is not None and segment not in SEGMENTS:
            raise NativePrefillError('invalid segment kind')
        if phase is not None and phase not in QSA_PHASES:
            raise NativePrefillError('invalid QSA phase')
        if decl.get('segment') is not None and segment != decl['segment']:
            raise NativePrefillError('offer segment differs from declaration')
        if decl.get('phase') is not None and phase != decl['phase']:
            raise NativePrefillError('offer phase differs from declaration')
        with self._lock:
            if len(self._meta) >= MAX_OFFERS:
                raise NativePrefillError('offer capacity')
        with self.bridge.offer(
                wire_name or wire(decl['family'], decl['suboperation']),
                expected,
                               context_id=declared['identity']['context_id'],
                               stream_id=declared['identity']['stream_id'],
                               graph=False,
                               copy_kinds=tuple(decl['copy_kinds'] or ())) as row:
            meta = {
                'id': row['id'], 'family': decl['family'],
                'suboperation': decl['suboperation'],
                'expectation': decl['expectation'],
                'allowed_kinds': list(decl['kinds']),
                'copy_kinds': list(decl['copy_kinds'] or ()),
                'role': role,
                'context_id': declared['identity']['context_id'],
                'stream_id': declared['identity']['stream_id'],
                'graph': False, 'branch': decl.get('branch'),
                'transaction': transaction, 'op_seq': op_seq,
                'segment': segment, 'phase': phase, 'cursor': cursor,
                'owner': owner,
                'thread': row.get('thread'),
                'system_thread': row.get('system_thread'),
                'parent': row.get('parent'),
                'rank': self.rank, 'epoch': self.epoch,
                'generation': self.generation,
                'storage_epoch': self.storage_epoch,
                'run_id': self.run_id, 'window_id': self.window_id,
                'submitted': 0, 'completed': 0,
                'body_exception': None, 'completion_error': None,
                '_events': [],
            }
            if type(meta['thread']) is not int or meta['thread'] < 1:
                raise NativePrefillError('native offer thread key')
            with self._lock:
                self._meta[row['id']] = meta
            binding = {'transaction': transaction, 'op_seq': op_seq,
                       'segment': segment, 'phase': phase, 'cursor': cursor,
                       'owner': owner, 'family': decl['family'],
                       'suboperation': decl['suboperation']}
            nested = self._binding(binding) if inherit else _null()
            with nested:
                try:
                    yield meta
                except _PENDING:
                    meta['body_exception'] = 'E3Pending'
                    raise
                except BaseException as exc:
                    meta['body_exception'] = type(exc).__name__[:64]
                    raise
                else:
                    meta['submitted'] = 1
                    event = self._completion_event(declared['stream'])
                    if event is None:
                        meta['completion_error'] = 'provider_missing'
                    else:
                        meta['_events'].append(event)

    def _completion_event(self, stream):
        """Record one host-queryable event for this bounded operation."""
        if self.torch is None:
            return None
        try:
            event = self.torch.cuda.Event(enable_timing=False)
            event.record(stream)
            return event
        except BaseException:
            self._invalidate('completion_event_record')
            return None

    def segment(self, kind, *, transaction, op_seq, cursor_before=None,
                branch=None):
        """Offer for one accepted ``LaneDispatcher`` segment command."""
        if kind not in SEGMENTS:
            raise NativePrefillError('unknown segment kind')
        sub = 'layer_qsa' if kind == 'layer' and branch == 'driver' else kind
        decl = declared_suboperation('prefill_segment', sub)
        return self._offer(decl, transaction=transaction, op_seq=op_seq,
                           segment=kind, cursor=cursor_before,
                           owner='lane')

    def qsa(self, phase, *, cursor=None):
        """Offer for one QSA driver phase inside the current segment."""
        if phase not in QSA_PHASES:
            raise NativePrefillError('unknown QSA phase')
        decl = declared_suboperation('qsa', {
            'pre': 'index_store', 'bucket': 'attention',
            'tail': 'tail'}[phase])
        return self._offer(decl, phase=phase, cursor=cursor)

    def qsa_fill(self, *, cursor=None, suboperation='bucket_fill'):
        """Offer for one history bucket copy on the owned copy stream."""
        decl = declared_suboperation('qsa', suboperation)
        return self._offer(decl, phase='bucket', cursor=cursor)

    def private_leaf(self, suboperation, *, cursor=None):
        """Offer for one exact lane-private initialization leaf.

        Opened by ``lane_private.LanePrivateChunk`` around the real
        ``uid.fill_``/D2D seed or the first ``lock.zero_`` while the QSA layer
        container is active. The caller's body is the original operation: this
        provider never retries, skips or duplicates it. A refused or failing
        observation latches the window incomplete (the body still runs once)
        and never counts as completion.
        """
        try:
            ctx = self._ctx_binding()
            if ctx is None or ctx.get('segment') != 'layer':
                # The private scope only runs inside an active lane layer
                # offer. Anything else is an invalid scope: latch incomplete,
                # still run the caller's body once, and never fake a leaf.
                raise NativePrefillError('private leaf outside a lane layer')
            decl = declared_suboperation('qsa', suboperation)
            scope = self._offer(decl, cursor=cursor, owner='private')
            enter, leave = scope.__enter__, scope.__exit__
        except Exception:
            self._invalidate('private_leaf_refused:' + str(suboperation))
            return nullcontext()

        @contextmanager
        def leaf():
            try:
                meta = enter()
            except Exception:
                self._invalidate('private_leaf_entry:' + str(suboperation))
                yield None
                return
            try:
                yield meta
            except BaseException as exc:
                try:
                    leave(type(exc), exc, exc.__traceback__)
                except Exception:
                    self._invalidate('private_leaf_close:' + str(suboperation))
                raise
            else:
                try:
                    leave(None, None, None)
                except Exception:
                    self._invalidate('private_leaf_close:' + str(suboperation))

        return leaf()

    def aux(self, family, suboperation, *, owner=None, stream=None):
        """Offer for one real source-owner method under the current segment."""
        decl = declared_suboperation(family, suboperation)
        attr = decl.get('owner_attr')
        if attr is not None:
            if stream is None:
                raise NativePrefillError('auxiliary owner stream is required')
            declared = self._streams.get(decl['role'])
            if declared is None:
                raise NativePrefillError('auxiliary role is not declared')
            got = _identity_value(self.stream_identity, stream, decl['role'])
            if got != declared['identity']:
                self._invalidate('aux_stream_drift:' + decl['role'])
                raise NativePrefillError('auxiliary stream differs from role')
        return self._offer(decl, owner=owner, inherit_phase=False)

    @contextmanager
    def lane_binding(self, *, transaction, op_seq, segment, phase=None,
                     cursor=None):
        """Explicit wire binding for a lane-level (dispatcher-free) wiring."""
        if segment not in SEGMENTS:
            raise NativePrefillError('unknown segment kind')
        binding = {'transaction': _transaction_identity(transaction),
                   'op_seq': op_seq, 'segment': segment, 'phase': phase,
                   'cursor': cursor, 'owner': 'lane'}
        with self._binding(binding):
            yield binding

    def snapshot(self):
        """The whole shared bridge report, untouched."""
        return self.bridge.report()

    def poll_completions(self):
        """Host-query every offer's own completion event (never waits)."""
        failed = []
        with self._lock:
            rows = list(self._meta.values())
        for meta in rows:
            events = meta['_events']
            if not events:
                continue
            try:
                ready = all(bool(event.query()) for event in events)
            except BaseException:
                self._invalidate('completion_query')
                ready = False
                failed.append(meta['id'])
            meta['completed'] = 1 if ready else 0
        return failed

    # -- receipt -----------------------------------------------------------
    def window(self):
        roles = {}
        for role, row in self._streams.items():
            roles[role] = {
                'context_id': row['identity']['context_id'],
                'stream_id': row['identity']['stream_id'],
                'non_green_control': row['non_green_control'],
            }
        return {
            'kind': 'native-prefill',
            'worker': self.worker,
            'rank': self.rank,
            'epoch': self.epoch,
            'generation': self.generation,
            'route_generation': self.route_generation,
            'storage_epoch': self.storage_epoch,
            'run_id': self.run_id,
            'window_id': self.window_id,
            'green_context_id': getattr(self, '_green_context_id', None),
            'stream_roles': roles,
            'native_start_ns': self._started,
            'native_end_ns': self._finished,
            'owners': [self.owner_record(name) for name in self._owner_order],
            'invalidated': bool(self._invalidated),
            'invalid': list(self._invalid),
        }

    def finish(self):
        """Fix the observed window end; every owner must already be closed."""
        if self._started is None or self._finished is not None:
            raise NativePrefillError('window is not open')
        open_names = [name for name in self._owner_order
                      if self._owners[name]['finish_ns'] is None]
        if open_names:
            raise NativePrefillError('owners still open: ' +
                                     ','.join(sorted(open_names)))
        native = self.bridge.window()
        end = native.get('end_ns')
        if type(end) is not int or end <= self._started:
            raise NativePrefillError('native window end not observed')
        if native.get('start_ns') != self._started:
            raise NativePrefillError('native window start changed')
        if native.get('route_generation') != self.route_generation:
            raise NativePrefillError('native route generation changed')
        self._finished = end
        return self.window()

    def seal(self):
        """Grade the prefill partition over the retained whole report."""
        if self._sealed is not None:
            return self._sealed
        if self._finished is None:
            raise NativePrefillError('window is not finished')
        self.poll_completions()
        snapshot = self.bridge.report()
        native = snapshot.get('native')
        offers = snapshot.get('offers')
        shared_ids = ({row.get('id') for row in offers}
                      if isinstance(offers, list) else set())
        declared = {oid: self._public(meta)
                    for oid, meta in self._meta.items()}
        if not isinstance(native, dict):
            self._sealed = _empty_receipt(
                self.window(), missing=['bridge_report_shape'])
            return self._sealed
        try:
            view = partition_view(native, offers, set(declared),
                                  allow_shared=True)
            receipt = grade_prefill(view, declared, self.window())
        except NativePrefillError as exc:
            receipt = _empty_receipt(self.window(),
                                     missing=['view_failed:' + str(exc)])
        whole = _whole_summary(native, offers)
        receipt['whole'] = whole
        receipt['partition'] = {'offers': len(declared),
                                'shared_report_offers': len(shared_ids),
                                'whole_activity_records':
                                    whole['activity_records']}
        self._sealed = receipt
        return receipt

    report = seal

    @staticmethod
    def _public(meta):
        return {k: v for k, v in meta.items() if not k.startswith('_')}

    def close_shadows(self):
        """Remove every installed shadow in reverse order."""
        errors = []
        while self._shadows:
            shadows = self._shadows.pop()
            try:
                shadows.close()
            except BaseException as exc:  # aggregate; never skip the rest
                errors.append(type(exc).__name__)
        if errors:
            raise ShadowDrift('shadow close failed: ' + ','.join(errors))
        return True


@contextmanager
def _null():
    yield


def _empty_receipt(window, *, missing=None):
    return {
        'schema': SCHEMA, 'verdict': 'INCONCLUSIVE', 'window': window,
        'native_window': None, 'offers': [], 'declared': {}, 'joins': [],
        'families': {}, 'counts': {}, 'roles': {}, 'owners': [],
        'completion': {}, 'thread_id_type': None, 'native': {},
        'joined_records': 0, 'eager_by_kind': {},
        'failures': [], 'missing': sorted(set(missing or [])),
        'notes': [],
    }


def _whole_summary(native, offers):
    by_kind = Counter()
    for row in native.get('activities', []) if isinstance(native, dict) else []:
        if isinstance(row, dict) and row.get('kind') in KINDS and type(
                row.get('count')) is int:
            by_kind[row['kind']] += row['count']
    return {
        'activity_records': native.get('activity_records')
        if isinstance(native, dict) else None,
        'by_kind': {kind: by_kind[kind] for kind in KINDS},
        'offers': len(offers) if isinstance(offers, list) else None,
        'complete': native.get('complete') if isinstance(native, dict) else None,
        'errors': native.get('errors') if isinstance(native, dict) else None,
        'overflow': native.get('overflow') if isinstance(native, dict) else None,
        'open_scopes': (native.get('open_scopes')
                        if isinstance(native, dict) else None),
        'boundary_records': (native.get('boundary_records')
                             if isinstance(native, dict) else None),
    }


# --- partition view ---------------------------------------------------------
def _external_map(native):
    result = {}
    for item in native.get('external', []):
        cid = item.get('correlation_id')
        result[cid] = item.get('external_id')
    return result


def partition_view(native, offers, ids, *, allow_shared=False):
    """Build the exact ``ids`` sub-report from a retained whole /2 report.

    Rows are selected by CUSTOM0 offer id, never by context/name/time.  The
    view is a strict subset; ``allow_shared`` marks that the whole report also
    contains offers owned by another partition (the paired composer proves
    conservation over the whole).
    """
    if not isinstance(native, dict) or native.get('schema') != NATIVE_SCHEMA:
        raise NativePrefillError('correlation schema')
    if not isinstance(offers, list):
        raise NativePrefillError('offers shape')
    ids = set(ids)
    offered = {}
    for row in offers:
        if not isinstance(row, dict):
            raise NativePrefillError('offer row')
        oid = row.get('id')
        if type(oid) is not int or oid < 1 or oid in offered:
            raise NativePrefillError('offer identity')
        offered[oid] = row
    if not ids:
        raise NativePrefillError('empty partition')
    if not ids <= set(offered):
        raise NativePrefillError('declared offer missing from report')
    if not allow_shared and set(offered) != ids:
        raise NativePrefillError('foreign offers in a single-partition report')
    external = _external_map(native)
    selected_external = {cid: oid for cid, oid in external.items()
                         if oid in ids}
    selected_apis = [row for row in native.get('apis', [])
                     if row.get('correlation_id') in selected_external]
    selected_activities = [row for row in native.get('activities', [])
                           if row.get('correlation_id') in selected_external]
    selected_scopes = [row for row in native.get('scopes', [])
                       if row.get('id') in ids]
    receipts = Counter()
    for row in selected_activities:
        count = row.get('count')
        if type(count) is not int or count <= 0:
            raise NativePrefillError('activity count')
        receipts[row.get('kind')] += count
    view = {k: native.get(k) for k in (
        'schema', 'thread_id_type', 'complete', 'capacity', 'reserved_bytes',
        'errors', 'overflow', 'open_scopes', 'boundary_records', 'window')}
    view.update(scopes=selected_scopes, apis=selected_apis,
                external=[row for row in native.get('external', [])
                          if row.get('correlation_id') in selected_external],
                activities=selected_activities,
                activity_records=sum(receipts.values()),
                partition_totals={kind: receipts[kind] for kind in KINDS})
    return view


# --- pure grade -------------------------------------------------------------
def _int(value, reason, minimum=0):
    if type(value) is not int or value < minimum:
        raise _Bad(reason)
    return value


def _need(condition, reason):
    if not condition:
        raise _Bad(reason)


class _Bad(Exception):
    pass


def _add(items, value):
    if value not in items:
        items.append(value)


def _grade_window(window, fail, miss):
    _need(isinstance(window, dict), 'window_shape')
    _need(isinstance(window.get('worker'), str) and window['worker'],
          'window_worker')
    bounds = {
        'rank': _int(window.get('rank'), 'window_rank'),
        'epoch': _int(window.get('epoch'), 'window_epoch', 1),
        'generation': _int(window.get('generation'), 'window_generation', 1),
        'route': _int(window.get('route_generation'), 'window_route', 1),
        'storage': _int(window.get('storage_epoch'), 'window_storage'),
        'green': _int(window.get('green_context_id'), 'window_green', 1),
        'start': _int(window.get('native_start_ns'), 'window_start'),
        'end': _int(window.get('native_end_ns'), 'window_end', 1),
        'run_id': window.get('run_id'),
        'window_id': window.get('window_id'),
    }
    _need(bounds['generation'] == bounds['route'],
          'window_generation_mismatch')
    _need(bounds['end'] > bounds['start'], 'window_bounds')
    _need(isinstance(window.get('stream_roles'), dict), 'window_stream_roles')
    if window.get('invalidated'):
        _add(miss, 'prefill_instrumentation_invalid')
    for reason in window.get('invalid', []):
        _add(miss, 'prefill_invalid:' + str(reason))
    roles = {}
    for role, row in window['stream_roles'].items():
        _need(role in ROLES and isinstance(row, dict), 'window_role')
        context_id = _int(row.get('context_id'), 'role_context', 1)
        stream_id = _int(row.get('stream_id'), 'role_stream', 1)
        non_green = bool(row.get('non_green_control', False))
        if role in ('prefill_main', 'shared_experts_aux', 'ple_aux'):
            if context_id != bounds['green']:
                fail('role_context_not_prefill_green')
            if non_green:
                fail('aux_role_marks_control')
        elif role == 'copy':
            if non_green != (context_id != bounds['green']):
                fail('copy_role_control_flag')
        roles[role] = {'context_id': context_id, 'stream_id': stream_id,
                       'non_green_control': non_green}
    if 'prefill_main' not in roles:
        fail('prefill_main_role_missing')
        raise _Bad('no_main_role')
    if roles['prefill_main']['context_id'] != bounds['green']:
        fail('prefill_main_not_green')
    return bounds, roles


def _grade_native_envelope(native, bounds, fail, miss):
    _need(isinstance(native, dict), 'native_shape')
    _need(native.get('schema') == NATIVE_SCHEMA, 'correlation_schema')
    thread_type = native.get('thread_id_type')
    if thread_type not in THREAD_ID_TYPES:
        _add(miss, 'thread_id_type_invalid')
    if native.get('complete') is not True:
        _add(miss, 'correlation_incomplete')
    for key, reason in (('overflow', 'correlation_overflow'),
                        ('errors', 'correlation_errors'),
                        ('open_scopes', 'correlation_open_scopes'),
                        ('boundary_records', 'correlation_boundary')):
        value = _int(native.get(key), key + '_shape')
        if value:
            _add(miss, reason)
    _int(native.get('capacity'), 'capacity_shape')
    _int(native.get('reserved_bytes'), 'reserved_shape')
    native_window = native.get('window')
    _need(isinstance(native_window, dict), 'native_window_missing')
    if (native_window.get('start_ns') != bounds['start']
            or native_window.get('end_ns') != bounds['end']):
        fail('native_window_bounds_mismatch')
    if native_window.get('route_generation') != bounds['route']:
        fail('native_window_route_mismatch')
    return thread_type


def _grade_declared(declared, window, bounds, roles, fail, miss, completion):
    _need(isinstance(declared, dict), 'declared_shape')
    meta = {}
    for raw_id, row in declared.items():
        try:
            oid = int(raw_id)
        except (TypeError, ValueError):
            raise _Bad('declared_identity')
        _need(type(oid) is int and oid > 0 and oid not in meta,
              'declared_identity')
        _need(isinstance(row, dict), 'declared_row')
        meta[oid] = row
    if not meta:
        _add(miss, 'no_offers')
    if len(meta) > MAX_OFFERS:
        _add(miss, 'offer_capacity_exceeded')
    for oid, row in meta.items():
        decl = declared_suboperation(row.get('family'), row.get('suboperation'))
        if row.get('expectation') != decl['expectation']:
            fail('offer_expectation_mismatch')
        if list(row.get('allowed_kinds', [])) != list(decl['kinds']):
            fail('offer_kinds_mismatch')
        if list(row.get('copy_kinds', [])) != list(decl['copy_kinds'] or ()):
            fail('offer_copy_kinds_mismatch')
        if row.get('role') != decl['role']:
            fail('offer_role_mismatch')
        if row.get('graph') is not False:
            fail('offer_graph_in_prefill')
        if row.get('rank') != bounds['rank']:
            fail('offer_rank')
        if (row.get('epoch') != bounds['epoch']
                or row.get('generation') != bounds['generation']):
            fail('offer_generation')
        if row.get('storage_epoch') != bounds['storage']:
            fail('offer_storage_epoch')
        if row.get('run_id') != window.get('run_id'):
            fail('offer_run_id')
        if row.get('window_id') != window.get('window_id'):
            fail('offer_window_id')
        role_row = roles.get(row.get('role'))
        if role_row is None:
            fail('declared_role_missing')
        elif (row.get('context_id') != role_row['context_id']
              or row.get('stream_id') != role_row['stream_id']):
            fail('offer_stream_identity_mismatch')
        if decl.get('segment') is not None and row.get(
                'segment') != decl['segment']:
            fail('offer_segment_mismatch')
        if decl.get('phase') is not None and row.get('phase') != decl['phase']:
            fail('offer_phase_mismatch')
        if row.get('segment') is not None and row['segment'] not in SEGMENTS:
            fail('offer_segment_shape')
        if row.get('phase') is not None and row['phase'] not in QSA_PHASES:
            fail('offer_phase_shape')
        transaction = row.get('transaction')
        if transaction is None:
            _add(miss, 'offer_transaction_missing')
        else:
            _need(isinstance(transaction, dict), 'offer_transaction_shape')
            for name in ('engine_epoch', 'request_generation', 'chunk_start',
                         'chunk_end', 'txn_seq'):
                _int(transaction.get(name), 'offer_transaction_field')
            uid = transaction.get('request_uid_sha256')
            if type(uid) is not str or len(uid) != 64:
                fail('offer_transaction_uid')
        if row.get('op_seq') is not None:
            _int(row['op_seq'], 'offer_op_seq')
        if row.get('cursor') is not None and not isinstance(
                row['cursor'], (list, tuple)):
            fail('offer_cursor_shape')
        if row.get('body_exception') not in (None, 'E3Pending'):
            _add(miss, 'offer_body_exception')
        if row.get('completion_error') is not None:
            _add(miss, 'completion_provider:' + str(row['completion_error']))
        _int(row.get('submitted'), 'offer_submitted')
        _int(row.get('completed'), 'offer_completed')
        if row['completed'] > row['submitted']:
            fail('offer_completion_exceeds_submission')
        if row['submitted'] and not row['completed']:
            completion['unproven'].append(oid)
        if type(row.get('thread')) is not int or row['thread'] < 1:
            _add(miss, 'offer_thread_missing')
    return meta


def _grade_scope_tree(native, meta, fail, miss):
    _need(isinstance(native.get('scopes'), list), 'scopes_shape')
    stacks = {}
    opened = set()
    closed = set()
    for event in native['scopes']:
        _need(isinstance(event, dict), 'scope_shape')
        oid = event.get('id')
        if oid not in meta:
            fail('native_scope_unknown_offer')
            continue
        thread = meta[oid].get('thread')
        stack = stacks.setdefault(thread, [])
        if event.get('thread') != thread:
            fail('native_scope_thread')
        if event.get('system_thread') != meta[oid].get('system_thread'):
            fail('native_scope_system_thread_mismatch')
        if type(meta[oid].get('system_thread')) is not int or meta[oid]['system_thread'] < 1:
            _add(miss, 'offer_system_thread_missing')
        if event.get('action') == 'push':
            if oid in opened:
                fail('native_scope_duplicate_push')
                continue
            want = stack[-1] if stack else 0
            if event.get('parent') != want:
                fail('native_scope_parent')
            declared_parent = meta[oid].get('parent')
            if declared_parent is not None and declared_parent != want:
                fail('native_scope_parent_declared')
            opened.add(oid)
            stack.append(oid)
        elif event.get('action') == 'pop':
            if not stack or stack[-1] != oid or oid in closed:
                fail('native_scope_pop')
            else:
                stack.pop()
                closed.add(oid)
        else:
            fail('native_scope_action')
    if opened != set(meta) or closed != set(meta) or any(stacks.values()):
        fail('native_scope_coverage')
    threads = {row.get('thread') for row in meta.values()}
    if len(threads) > 1:
        _add(miss, 'prefill_offers_on_multiple_threads')


def _grade_apis_and_external(native, meta, fail):
    _need(isinstance(native.get('apis'), list), 'apis_shape')
    apis = {}
    for row in native['apis']:
        _need(isinstance(row, dict), 'api_shape')
        cid = _int(row.get('correlation_id'), 'api_identity', 1)
        if cid in apis:
            fail('duplicate_api')
            continue
        _need(row.get('domain') in ('runtime', 'driver'), 'api_domain')
        _need(isinstance(row.get('name'), str) and row['name'], 'api_name')
        _need(type(row.get('return_value')) is int, 'api_return')
        _int(row.get('thread'), 'api_thread', 1)
        start = _int(row.get('start'), 'api_start', 1)
        end = _int(row.get('end'), 'api_end', 1)
        _need(start <= end, 'api_time')
        apis[cid] = row
    _need(isinstance(native.get('external'), list), 'external_shape')
    external = {}
    for row in native['external']:
        _need(isinstance(row, dict), 'external_row')
        cid = _int(row.get('correlation_id'), 'external_identity', 1)
        if cid in external:
            fail('duplicate_external_join')
            continue
        if row.get('kind') != 'custom0':
            fail('foreign_external_kind')
            continue
        if cid not in apis:
            fail('external_api_missing')
            continue
        oid = row.get('external_id')
        if oid not in meta:
            fail('unknown_offer')
            continue
        external[cid] = oid
        if apis[cid]['thread'] != meta[oid].get('thread'):
            fail('external_thread_mismatch')
    return apis, external


def _grade_activities(native, meta, apis, external, fail, miss, by_offer,
                      by_kind, joins, families, activity_cids):
    _need(isinstance(native.get('activities'), list), 'activities_shape')
    seen = set()
    records = 0
    for row in native['activities']:
        _need(isinstance(row, dict), 'activity_row')
        kind = row.get('kind')
        _need(kind in KINDS, 'activity_kind')
        count = _int(row.get('count'), 'activity_count', 1)
        cid = _int(row.get('correlation_id'), 'activity_identity')
        graph_id = _int(row.get('graph_id', 0), 'activity_graph')
        context_id = _int(row.get('context_id'), 'activity_context')
        stream_id = _int(row.get('stream_id'), 'activity_stream')
        signature = (cid, kind, graph_id, context_id, stream_id,
                     row.get('copy_kind'), row.get('name'))
        if signature in seen:
            fail('duplicate_activity_join')
            continue
        seen.add(signature)
        oid = external.get(cid)
        if oid is None:
            fail('unoffered_activity')
            continue
        api = apis.get(cid)
        if api is None:
            fail('activity_without_api')
            continue
        info = meta[oid]
        if api.get('return_value') != 0:
            fail('failed_api_has_activity')
        if graph_id:
            fail('graph_activity_in_prefill')
            continue
        if _api_kind(api.get('name')) != kind:
            fail('api_activity_kind')
        if context_id != info['context_id']:
            fail('wrong_context')
        if stream_id != info['stream_id']:
            fail('wrong_stream')
        if kind not in info.get('allowed_kinds', ()):
            fail('wrong_kind')
        if kind == 'memcpy' and row.get('copy_kind') not in info.get(
                'copy_kinds', ()):
            fail('copy_direction')
        if info.get('role') == 'copy' and kind != 'memcpy':
            fail('copy_role_kernel')
        activity_cids.add(cid)
        records += count
        by_kind[kind] += count
        by_offer[oid][kind] += count
        joins.append({'offer': oid, 'correlation_id': cid, 'kind': kind,
                      'count': count, 'context_id': context_id,
                      'stream_id': stream_id, 'role': info['role']})
        families.setdefault(info['family'], {}).setdefault(
            info['suboperation'], Counter())[kind] += count
    if records != native.get('activity_records'):
        _add(miss, 'activity_census')
    totals = native.get('partition_totals')
    if totals is None:
        _add(miss, 'partition_totals_missing')
    elif {k: v for k, v in totals.items() if v} != dict(by_kind):
        _add(miss, 'partition_totals_mismatch')
    for cid, api in apis.items():
        if _api_kind(api.get('name')) != 'host':
            continue
        # Every observed host control call, joined or not, must be a success
        # or the exact pending event query. A pending query with GPU activity
        # is never host-only work.
        if not _host_return_ok(api.get('name'), api.get('return_value')):
            fail('host_api_failed')
        if cid in activity_cids:
            fail('host_api_has_activity')
    for cid, oid in external.items():
        api = apis[cid]
        kind = _api_kind(api.get('name'))
        if kind is None:
            _add(miss, 'unsupported_api:' + api['name'])
        elif kind == 'host':
            # Host control joined inside an offer: the exact event query may
            # report NOT_READY (600) with zero GPU work; other nonzero host
            # returns still fail, and a pending query completes nothing.
            if not _host_return_ok(api.get('name'), api.get('return_value')):
                fail('host_api_failed')
        elif api.get('return_value') != 0:
            fail('api_submission_failed')
        elif kind in KINDS and cid not in activity_cids:
            fail('submission_without_activity')
    return records


def _grade_expectations(meta, by_offer, fail, miss, completion):
    for oid, info in meta.items():
        counts = by_offer[oid]
        expectation = info.get('expectation')
        if expectation == 'activity' and not counts:
            _add(miss, 'activity_missing')
        if expectation in ('zero_work', 'container') and counts:
            fail('zero_work_has_activity' if expectation == 'zero_work'
                 else 'container_has_activity')
        if info.get('body_exception') == 'E3Pending' and counts:
            fail('pending_offer_has_activity')
        if info.get('submitted'):
            completion['offers_submitted'] += 1
            if info.get('completed'):
                completion['offers_completed'] += 1


def _grade_owners(window, roles, meta, bounds, fail, miss,
                      completion):
    rows = window.get('owners')
    _need(isinstance(rows, list), 'owners_shape')
    names = set()
    owned_roles = set()
    for row in rows:
        _need(isinstance(row, dict), 'owner_row')
        name = row.get('name')
        if type(name) is not str or not name or name in names:
            fail('owner_identity')
            continue
        names.add(name)
        role_row = roles.get(row.get('role'))
        if role_row is None:
            fail('owner_role_missing')
        elif (row.get('context_id') != role_row['context_id']
              or row.get('stream_id') != role_row['stream_id']):
            fail('owner_stream_identity_mismatch')
        start = _int(row.get('start_ns'), 'owner_start')
        finish = _int(row.get('finish_ns'), 'owner_finish', 1)
        if finish < start:
            fail('owner_time_order')
        if not (bounds['start'] <= start <= finish <= bounds['end']):
            fail('owner_outside_window')
        submitted = _int(row.get('submitted'), 'owner_submitted')
        completed = _int(row.get('completed'), 'owner_completed')
        if completed > submitted:
            fail('owner_completion_exceeds_submission')
        if submitted and completed < submitted:
            _add(miss, 'owner_completion_unproven:' + name)
        owned_roles.add(row.get('role'))
    for role in {row.get('role') for row in meta.values()}:
        if role not in owned_roles:
            _add(miss, 'owner_role_unmarked:' + role)
    completion['owners'] = len(rows)
    completion['unproven_owners'] = [
        row.get('name') for row in rows
        if row.get('submitted') and row.get('completed', 0)
        < row.get('submitted')]
    completion['roles'] = sorted(role for role in owned_roles if role)


def grade_prefill(native, declared, window):
    """Grade one prefill partition; FAIL beats INCONCLUSIVE.

    ``declared`` maps offer id (or its decimal string) to the bounded metadata
    recorded at run time.  No predicate is satisfied by flags alone: PASS
    requires joined activity, exact source/stream identities, and independent
    submitted/queried completion credit.
    """
    failures = []
    missing = []
    notes = []
    joins = []
    families = {}
    owner_rows = []
    by_offer = {}
    by_kind = Counter()
    completion = {'offers_submitted': 0, 'offers_completed': 0,
                  'unproven': [], 'owners': 0}
    meta = {}
    roles = {}
    thread_type = None

    def fail(reason):
        _add(failures, reason)

    def miss(reason):
        _add(missing, reason)

    try:
        bounds, roles = _grade_window(window, fail, missing)
        thread_type = _grade_native_envelope(native, bounds, fail, missing)
        meta = _grade_declared(declared, window, bounds, roles, fail,
                               missing, completion)
        by_offer = {oid: Counter() for oid in meta}
        _grade_scope_tree(native, meta, fail, missing)
        apis, external = _grade_apis_and_external(native, meta, fail)
        activity_cids = set()
        _grade_activities(native, meta, apis, external, fail, missing, by_offer,
                          by_kind, joins, families, activity_cids)
        _grade_expectations(meta, by_offer, fail, missing, completion)
        _grade_owners(window, roles, meta, bounds, fail, missing,
                      completion)
        owner_rows = [dict(row) for row in window.get('owners', [])
                      if isinstance(row, dict)]
        if not sum(by_kind.values()):
            miss('no_prefill_activity')
        if not any(row.get('submitted') for row in meta.values()):
            miss('no_submitted_operation')
        if any(not row.get('completed') for row in meta.values()
               if row.get('submitted')):
            miss('completion_unproven')
        txn_seqs = {}
        for row in meta.values():
            txn = row.get('transaction')
            if txn is None:
                continue
            key = tuple(sorted(txn.items()))
            txn_seqs.setdefault(key, set()).add(row.get('op_seq'))
        if any(None in seqs for seqs in txn_seqs.values()):
            miss('offer_op_seq_missing')
    except _Bad as exc:
        miss('malformed_prefill_evidence:' + str(exc))
    except (NativePrefillError, KeyError, TypeError, ValueError) as exc:
        miss('malformed_prefill_evidence:' + type(exc).__name__)
    verdict = 'FAIL' if failures else ('INCONCLUSIVE' if missing else 'PASS')
    return {
        'schema': SCHEMA,
        'verdict': verdict,
        'window': window,
        'native_window': (native.get('window')
                          if isinstance(native, dict) else None),
        'offers': [dict(meta[oid]) for oid in sorted(meta)],
        'declared': {str(oid): dict(row) for oid, row in meta.items()},
        'joins': joins,
        'families': {family: {sub: dict(counts)
                              for sub, counts in subs.items()}
                     for family, subs in families.items()},
        'counts': {kind: by_kind[kind] for kind in KINDS},
        'roles': roles,
        'owners': owner_rows,
        'completion': completion,
        'thread_id_type': thread_type,
        'native': {k: native.get(k) for k in (
            'schema', 'thread_id_type', 'complete', 'capacity',
            'reserved_bytes', 'errors', 'overflow', 'open_scopes',
            'boundary_records', 'activity_records', 'window')}
        if isinstance(native, dict) else {},
        'joined_records': sum(row['count'] for row in joins),
        'eager_by_kind': {kind: by_kind[kind] for kind in KINDS},
        'failures': sorted(set(failures)),
        'missing': sorted(set(missing)),
        'notes': notes,
    }


def validate_receipt(receipt):
    """Reject a mutated prefill receipt (bounded structural check)."""
    if not isinstance(receipt, dict) or receipt.get('schema') != SCHEMA:
        raise NativePrefillError('receipt_schema')
    if receipt.get('verdict') not in ('PASS', 'INCONCLUSIVE', 'FAIL'):
        raise NativePrefillError('receipt_verdict')
    declared = receipt.get('declared')
    if not isinstance(declared, dict):
        raise NativePrefillError('receipt_declared')
    joins = receipt.get('joins')
    if not isinstance(joins, list):
        raise NativePrefillError('receipt_joins')
    joined = 0
    for row in joins:
        if not isinstance(row, dict):
            raise NativePrefillError('receipt_join')
        oid = row.get('offer')
        if str(oid) not in declared:
            raise NativePrefillError('receipt_join_offer')
        count = row.get('count')
        if type(count) is not int or count <= 0:
            raise NativePrefillError('receipt_join_count')
        joined += count
    if joined != receipt.get('joined_records'):
        raise NativePrefillError('receipt_join_total')
    counts = receipt.get('counts')
    if not isinstance(counts, dict) or sum(
            v for v in counts.values() if type(v) is int) != joined:
        raise NativePrefillError('receipt_counts')
    if receipt.get('verdict') == 'PASS':
        if not joined:
            raise NativePrefillError('receipt_pass_without_activity')
        if receipt.get('failures') or receipt.get('missing'):
            raise NativePrefillError('receipt_pass_reasons')
        if (receipt.get('completion', {}).get('offers_submitted', 0)
                != receipt.get('completion', {}).get('offers_completed', -1)):
            raise NativePrefillError('receipt_pass_without_completion')
    return True


# --- reversible source-pinned shadows --------------------------------------
_SEAMS = {
    'lane_dispatcher': {
        '_do_prepare': 'prepare',
        '_do_pp_receive': 'pp_receive',
        '_do_prefix': 'prefix',
        '_do_layer': 'layer',
        '_do_suffix': 'suffix',
        '_do_pp_send': 'pp_send',
        '_do_state_finalize': 'state_finalize',
    },
    'lane_forward': {'begin': 'prefix', 'layer': 'layer',
                     'finish': 'suffix'},
    'qsa_driver': {'_pre': 'pre', '_bucket': 'bucket', '_tail': 'tail'},
    'qsa_operation': {'fill_bucket': 'bucket_fill',
                      'fill_sparse_bucket': 'sparse_bucket_fill',
                      'stage_sparse_selection': 'selection_readback'},
    'shared_experts': {'maybe_forward_async': 'maybe_forward_async',
                       'forward': 'forward', 'wait': 'wait'},
    'ple_embedding': {'start_prefetch': 'start_prefetch',
                      '_finalize_prefetch': 'finalize_prefetch'},
}


class ShadowSet:
    """Exact inverse of one source-pinned shadow installation."""

    def __init__(self, ledger, target, kind, pins):
        self.ledger = ledger
        self.target = target
        self.kind = kind
        self.pins = pins
        self.installed = []
        self.closed = False
        self.pin = None

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, exc_type, *_rest):
        try:
            self.close()
        except ShadowDrift:
            if exc_type is None:
                raise
            self.ledger._invalidate('shadow_close_drift')
        return False

    def install(self):
        if self.installed:
            raise NativePrefillError('shadows already installed')
        if self.kind not in _SEAMS:
            raise NativePrefillError('undeclared seam kind')
        self.pin = _pin_for(self.pins, self.target)
        try:
            for attr in _SEAMS[self.kind]:
                original = getattr(self.target, attr, None)
                if not callable(original):
                    raise SourcePinError('seam missing: ' + attr)
                if getattr(original, '_flash_s5_prefill_shadow', False):
                    raise NativePrefillError('seam already shadowed: ' + attr)
                wrapper = _make_shadow(self.ledger, self.kind, attr,
                                       self.target, original)
                had_own = attr in getattr(self.target, '__dict__', {})
                setattr(self.target, attr, wrapper)
                self.installed.append((attr, original, wrapper, had_own))
        except BaseException:
            self._restore(raise_on_drift=False)
            raise
        self.ledger._shadows.append(self)
        return self

    def _restore(self, *, raise_on_drift=True):
        drift = []
        for attr, original, wrapper, had_own in reversed(self.installed):
            current = getattr(self.target, attr, None)
            if current is not wrapper:
                drift.append(attr)
                continue
            if had_own:
                setattr(self.target, attr, original)
            else:
                delattr(self.target, attr)
        self.installed = []
        if drift:
            self.ledger._invalidate('shadow_changed_outside:' +
                                    ','.join(drift))
            if raise_on_drift:
                raise ShadowDrift('shadow changed outside: ' +
                                  ','.join(drift))
        return drift

    def close(self):
        if self.closed:
            return True
        self._restore(raise_on_drift=True)
        self.closed = True
        if self.ledger._shadows and self.ledger._shadows[-1] is self:
            self.ledger._shadows.pop()
        elif self in self.ledger._shadows:
            self.ledger._shadows.remove(self)
        return True

    def record(self):
        return {'kind': self.kind, 'target': type(self.target).__name__,
                'seams': sorted(attr for attr, _, _, _ in self.installed),
                'source_sha256': self.pin}


def install_seams(target, ledger, kind, *, pins):
    """Install the declared shadows for one lane object (before measuring)."""
    if not isinstance(ledger, NativePrefillLedger):
        raise NativePrefillError('ledger required')
    if not isinstance(pins, dict) or not pins:
        raise SourcePinError('source pins are required')
    shadow = ShadowSet(ledger, target, kind, pins)
    shadow.install()
    return shadow


def _make_shadow(ledger, kind, attr, target, original):
    @functools.wraps(original)
    def shadow(*args, **kwargs):
        # Telemetry cannot retry GPU submission, or prevent it on a full ledger.
        # Latch observation failure while executing the original exactly once.
        called = completed = False
        value = None
        body_error = None

        def run_original(*body_args, **body_kwargs):
            nonlocal called, completed, value, body_error
            if called:
                raise NativePrefillError('original already invoked')
            called = True
            try:
                value = original(*body_args, **body_kwargs)
            except BaseException as exc:
                body_error = exc
                raise
            completed = True
            return value

        try:
            return _invoke(ledger, kind, attr, target, run_original, args,
                           kwargs)
        except Exception as exc:
            if body_error is not None:
                # Preserve even a body exception hidden by a failing pop.
                raise body_error
            ledger._invalidate('shadow_observation:' + type(exc).__name__)
            if completed:
                return value
            if called:
                raise
            return run_original(*args, **kwargs)
    setattr(shadow, '_flash_s5_prefill_shadow', True)
    return shadow


def _invalidate_passthrough(ledger, reason, original, args, kwargs):
    ledger._invalidate(reason)
    return original(*args, **kwargs)


def _invoke(ledger, kind, attr, target, original, args, kwargs):
    if kind == 'lane_dispatcher':
        command = args[0] if args else None
        transaction = getattr(target, 'transaction', None)
        if command is None or transaction is None:
            return original(*args, **kwargs)
        op_seq = getattr(command, 'op_seq', None)
        segment = getattr(command, 'kind', None)
        if op_seq is None or segment not in SEGMENTS:
            return _invalidate_passthrough(
                ledger, 'dispatcher_command_shape', original, args, kwargs)
        cursor = _safe_cursor(target)
        branch = None
        if segment == 'layer':
            branch = 'driver' if _qsa_owns_layer(target, command) else 'direct'
        try:
            scope = ledger.segment(segment, transaction=transaction,
                                   op_seq=op_seq, cursor_before=cursor,
                                   branch=branch)
        except NativePrefillError:
            return _invalidate_passthrough(
                ledger, 'segment_offer_refused', original, args, kwargs)
        with scope as meta:
            result = original(*args, **kwargs)
            meta['cursor'] = [cursor, _safe_cursor(target)]
            return result

    if kind == 'lane_forward':
        binding = ledger._ctx_binding()
        if binding is None:
            return _invalidate_passthrough(
                ledger, 'lane_forward_without_binding', original, args,
                kwargs)
        segment = _SEAMS[kind][attr]
        branch = None
        if segment == 'layer' and args:
            branch = ('driver' if _forward_driver_owns(target, args[0])
                      else 'direct')
        sub = 'layer_qsa' if segment == 'layer' and branch == 'driver' else segment
        decl = declared_suboperation('prefill_segment', sub)
        # LaneForward owns an integer cursor. Nested owner offers inherit this
        # binding, so retain the same bounded list shape as dispatcher/QSA
        # cursors rather than passing a scalar that the raw grader refuses.
        cursor = _safe_cursor(target)
        try:
            scope = ledger._offer(decl, transaction=binding['transaction'],
                                  op_seq=binding['op_seq'], segment=segment,
                                  cursor=cursor, owner='lane')
        except NativePrefillError:
            return _invalidate_passthrough(
                ledger, 'lane_offer_refused', original, args, kwargs)
        with scope as meta:
            result = original(*args, **kwargs)
            meta['cursor'] = [cursor, _safe_cursor(target)]
            return result

    if kind == 'qsa_driver':
        phase = _SEAMS[kind][attr]
        cursor = _safe_cursor(target)
        try:
            scope = ledger.qsa(phase, cursor=cursor)
        except NativePrefillError:
            return _invalidate_passthrough(
                ledger, 'qsa_offer_refused', original, args, kwargs)
        with scope:
            return original(*args, **kwargs)

    if kind == 'qsa_operation':
        device = getattr(target, 'device', None)
        copy = getattr(device, 'copy', None)
        if copy is None:
            return _invalidate_passthrough(
                ledger, 'copy_stream_missing', original, args, kwargs)
        declared = ledger._streams.get('copy')
        if declared is None:
            return _invalidate_passthrough(
                ledger, 'copy_role_undeclared', original, args, kwargs)
        try:
            got = _identity_value(ledger.stream_identity, copy, 'copy')
        except NativePrefillError:
            return _invalidate_passthrough(
                ledger, 'copy_identity_query', original, args, kwargs)
        if got != declared['identity']:
            return _invalidate_passthrough(
                ledger, 'copy_stream_drift', original, args, kwargs)
        try:
            scope = ledger.qsa_fill(cursor=_safe_cursor(target),
                                    suboperation=_SEAMS[kind][attr])
        except NativePrefillError:
            return _invalidate_passthrough(
                ledger, 'fill_offer_refused', original, args, kwargs)
        with scope:
            return original(*args, **kwargs)

    if kind in ('shared_experts', 'ple_embedding'):
        sub = _SEAMS[kind][attr]
        decl = declared_suboperation(
            'shared_experts' if kind == 'shared_experts' else 'ple', sub)
        if ledger._ctx_binding() is None:
            return _invalidate_passthrough(
                ledger, 'aux_without_binding', original, args, kwargs)
        stream = None
        owner_attr = decl.get('owner_attr')
        if owner_attr is not None:
            stream = getattr(target, owner_attr, None)
            if stream is None:
                return _invalidate_passthrough(
                    ledger, 'aux_owner_stream_missing', original, args, kwargs)
        try:
            scope = ledger.aux(
                'shared_experts' if kind == 'shared_experts' else 'ple', sub,
                owner=type(target).__name__, stream=stream)
        except NativePrefillError:
            return _invalidate_passthrough(
                ledger, 'aux_offer_refused', original, args, kwargs)
        with scope:
            return original(*args, **kwargs)

    return _invalidate_passthrough(ledger, 'unknown_seam', original, args,
                                    kwargs)


def _safe_cursor(target):
    cursor = getattr(target, 'cursor', None)
    if callable(cursor):
        try:
            cursor = cursor()
        except BaseException:
            return None
    if cursor is None:
        return None
    if isinstance(cursor, tuple):
        return [value if type(value) is int or type(value) is str else None
                for value in cursor]
    if type(cursor) in (int, str):
        return [cursor]
    return None


def _qsa_owns_layer(dispatcher, command):
    lane = getattr(dispatcher, 'lane', None)
    driver = getattr(dispatcher, 'driver', None)
    if driver is None and lane is not None:
        driver = getattr(lane, 'attention_driver', None)
    start = getattr(command, 'layer_start', None)
    if lane is None or driver is None or start is None:
        return False
    owns = getattr(driver, 'owns', None)
    model = getattr(lane, 'model', None)
    if not callable(owns) or model is None:
        return False
    try:
        return bool(owns(model, start))
    except BaseException:
        return False


def _forward_driver_owns(lane, layer_idx):
    driver = getattr(lane, 'attention_driver', None)
    model = getattr(lane, 'model', None)
    owns = getattr(driver, 'owns', None)
    if not callable(owns) or model is None:
        return False
    try:
        return bool(owns(model, layer_idx))
    except BaseException:
        return False


__all__ = ['COPY_KIND', 'ENABLE_ENV', 'FAMILIES', 'KINDS', 'MAX_OFFERS',
           'NATIVE_SCHEMA', 'NativePrefillError', 'NativePrefillLedger', 'ROLES',
           'SCHEMA', 'SEGMENTS', 'ShadowDrift', 'ShadowSet', 'SourcePinError',
           'declared_suboperation', 'grade_prefill', 'install_seams',
           'live_pins', 'partition_view', 'validate_receipt', 'wire']
