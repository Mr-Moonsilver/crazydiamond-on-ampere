"""Default-off actual-replay context observer wrapper (CPU-only import).

Importing this module loads no library, imports no torch/vLLM and starts no
CUPTI session.  A caller must construct :class:`ReplayBinding` from a trusted
local shared library, pin its SHA-256, the observer source SHA-256 and the
already-loaded CUPTI library SHA-256, then create an :class:`ObserverSession`.

The wrapper owns the session lifecycle.  It never accepts a native path,
pointer or arbitrary native action over RPC; the native handle stays private.
Native driver IDs are private too: receipts use only scoped labels such as
``graph:0`` and ``node:0:12``.  No kernel names, arguments, pointers, tensor
contents, request IDs or token IDs are serialized.

Evidence boundary: this module cannot certify the caller's scheduler pause,
drain proof or worker/engine incarnation.  ``DrainedProof`` is bound into the
receipt as caller evidence with ``verified_by_observer=False``.  CPU fixture
success does not establish CUPTI runtime support.  A short tracing window is
not a performance measurement.
"""
import hashlib
import json
import os
from dataclasses import dataclass
from types import MappingProxyType

from e3_types import E3ContractError

ABI_VERSION = 1
NATIVE_SCHEMA_STATIC = 's5ro-native-static/1'
NATIVE_SCHEMA_WINDOW = 's5ro-native-window/2'
RECEIPT_SCHEMA = 's5ro-replay-observer/1'
HELPER_SOURCE = 'native/flash_s5_replay_observer.cpp'
CUPTI_API = 130001
_MISSING = object()

_SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            HELPER_SOURCE)

# Native status codes are transported unchanged; the wrapper exposes only
# short codes, never a native payload.
_STATUS_NAMES = {
    1: 'truncated_report',
    2: 'bad_arguments',
    3: 'unknown_session',
    4: 'bad_state',
    5: 'version_query_failed',
    6: 'cupti_version_mismatch',
    7: 'activity_register_failed',
    8: 'subscribe_failed',
    9: 'subscriber_conflict',
    10: 'callback_enable_failed',
    11: 'activity_enable_failed',
    12: 'context_bind_failed',
    13: 'graph_bind_failed',
    14: 'limit_exceeded',
    15: 'observer_busy',
    16: 'flush_failed',
    17: 'unsubscribe_failed',
    18: 'graph_duplicate',
    19: 'context_conflict',
    20: 'timestamp_failed',
    21: 'token_mismatch',
    22: 'internal_error',
    23: 'not_bound',
    24: 'window_active',
    25: 'cleanup_failed',
}

# The exact native reason vocabulary.  A native document containing any other
# string is rejected as malformed rather than interpreted.
REASONS = frozenset({
    'drop_seen', 'drop_query_error', 'buffer_refused', 'ring_overflow',
    'parse_error', 'unknown_activity_record', 'unknown_resource_event',
    'flush_failed', 'expected_context_unbound', 'expected_context_missing',
    'expected_context_duplicate', 'expected_context_contradictory',
    'expected_context_not_green', 'expected_context_sm_mismatch',
    'context_destroyed', 'known_wrong_context', 'unknown_graph',
    'ambiguous_graph', 'unknown_node', 'ambiguous_node', 'kind_mismatch',
    'missing_node_id', 'boundary_overlap', 'unknown_timestamp',
    'incomplete_record', 'invalid_interval', 'unknown_resource_timestamp',
    'update_seen', 'destroy_seen', 'inventory_changed', 'unsupported_node',
    'unknown_node_type', 'mapping_unknown', 'no_exercised_graph',
    'unattributed_expected_context', 'link_overflow', 'node_overlap',
    'missing_execution', 'no_graphs', 'graph_id_unknown', 'node_id_unknown',
    'graph_changed', 'empty_graph', 'duplicate_graph',
    'buffer_outstanding',
})

NATIVE_FAIL_REASONS = frozenset({
    'known_wrong_context', 'expected_context_not_green',
    'expected_context_sm_mismatch', 'context_destroyed',
})

LIMITATIONS = (
    'observer cannot certify scheduler pause or a completed consumer drain',
    'caller worker/engine incarnation is unverified by the observer',
    'CPU fixture success does not establish CUPTI runtime support',
    'tracing overhead is declared only; trace wall time is not performance',
    'only exercised descriptors receive replay-qualified status',
    'driver handles, graph/node IDs and kernel names are never emitted',
    'descriptor labels are capture-order positions for this bank incarnation',
    'CUPTI activity enable/disable is process-global; a conflicting subscriber is refused',
    'CUPTI activity callback registration carries no subscriber identity; an activity-only user is not detectable and its callbacks are replaced',
    'activity buffers outstanding at window end force INCONCLUSIVE; a forced flush is not device completion',
    'a bind is valid only for its route generation; a split/route transition needs a fresh bind while paused',
)

_NODE_TYPE_KEYS = ('kernel', 'memcpy', 'memset', 'graph', 'empty', 'event',
                   'host', 'conditional', 'memory', 'other', 'unknown')
_REQUIRED_KEYS = ('kernel', 'memcpy', 'memset')
_OBSERVED_KEYS = ('kernel', 'memcpy', 'memset')
_MAPPING_KEYS = ('graph_ids', 'node_keys', 'node_aliases', 'alias_unknown',
                 'ambiguous_graph_ids', 'node_overlap')
_COUNT_KEYS = (
    'context_records', 'kernel_records', 'copy_records', 'resource_events',
    'unknown_activity_records', 'unknown_resource_events',
    'records_before_window', 'records_after_window', 'records_in_window',
    'outside_scope_records', 'outside_scope_graph_records',
    'unattributed_expected_context', 'unattributed_other_context',
    'unknown_graph_records', 'ambiguous_graph_records', 'unknown_node_records',
    'ambiguous_node_records', 'kind_mismatch_records',
    'missing_node_id_records', 'boundary_overlap_records',
    'unknown_timestamp_records', 'incomplete_records',
    'invalid_interval_records', 'known_wrong_context_records',
    'buffers_requested', 'buffers_issued', 'buffers_completed',
    'buffers_issued_in_window', 'buffers_completed_in_window',
    'buffers_refused', 'buffers_outstanding', 'dropped_records_total',
    'drop_query_errors', 'parse_errors', 'invalid_records', 'callback_errors',
    'callback_reentrant_refused', 'duplicate_buffer_completions',
    'dependency_events', 'outside_scope_updates', 'graph_passes',
)
_FLAG_KEYS = (
    'permanent_overflow', 'buffer_refused', 'buffer_outstanding', 'parse_error', 'unknown_activity',
    'unknown_resource', 'drop_seen', 'drop_query_error', 'flush_failed',
    'update_seen', 'destroy_seen', 'inventory_changed',
    'expected_context_missing', 'expected_context_duplicate',
    'expected_context_contradictory', 'known_wrong_context',
    'known_wrong_greenness', 'known_wrong_sm', 'context_destroyed',
    'outside_scope_graph_records', 'no_exercised_graph', 'link_overflow',
    'reentrant_refused',
)
_CONTEXT_KEYS = (
    'bound', 'context_known', 'expected_sms', 'records', 'green_true',
    'green_false', 'sm_known', 'sm_matching', 'sm_mismatched', 'is_green',
    'num_multiprocessors', 'parent_context_present', 'duplicate',
    'contradictory',
)
_LIMIT_KEYS = (
    'buffer_count', 'buffer_bytes', 'max_context_records',
    'max_kernel_records', 'max_copy_records', 'max_graph_events',
    'max_graphs', 'max_nodes_per_graph', 'max_nodes_total', 'max_links',
)


class ReplayObserverError(E3ContractError):
    """A refused binding, malformed report or invalid caller proof.

    The message is a short code; loader paths, native payloads and chained
    exceptions are deliberately suppressed.
    """

    def __init__(self, code):
        self.code = str(code)
        super().__init__(self.code)

    def __str__(self):
        return self.code


def _is_digest(value):
    return (type(value) is str and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def _hash_file(path):
    try:
        with open(path, 'rb') as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        raise ReplayObserverError('identity_read_failed') from None


def helper_source_sha256():
    return _hash_file(_SOURCE_PATH)


def _loaded_cupti_path():
    """Resolve the already-loaded CUPTI 13 library from this process."""
    candidates = set()
    try:
        with open('/proc/self/maps', 'r', encoding='ascii', errors='replace') as handle:
            for line in handle:
                parts = line.split()
                if len(parts) < 6:
                    continue
                path = parts[5]
                if not path.startswith('/'):
                    continue
                base = os.path.basename(path)
                if base == 'libcupti.so.13' or base.startswith('libcupti.so.13.'):
                    candidates.add(os.path.realpath(path))
    except OSError:
        raise ReplayObserverError('cupti_library_missing') from None
    if not candidates:
        raise ReplayObserverError('cupti_library_missing')
    if len(candidates) != 1:
        raise ReplayObserverError('cupti_library_ambiguous')
    return candidates.pop()


def _check_int(value, low, high, code):
    if type(value) is not int or not low <= value <= high:
        raise ReplayObserverError(code)
    return value


@dataclass(frozen=True)
class ObserverLimits:
    """Explicit bounded capacities; no opaque native growth is permitted."""

    # full8 exhausted four buffers before replay. Execution capture is now
    # window-only; 32 MiB also covers the declared ~60k-node replay window.
    # These are host allocations, reported by reserved_bytes, not free VRAM.
    buffer_count: int = 32
    buffer_bytes: int = 1024 * 1024
    max_context_records: int = 128
    max_kernel_records: int = 131072
    max_copy_records: int = 32768
    # full8 observed ~190k resource callbacks while capturing the seven banks.
    max_graph_events: int = 262144
    max_graphs: int = 64
    max_nodes_per_graph: int = 131072
    max_nodes_total: int = 131072
    max_links: int = 65536
    max_receipt_bytes: int = 8 * 1024 * 1024

    def __post_init__(self):
        _check_int(self.buffer_count, 1, 64, 'bad_limits')
        _check_int(self.buffer_bytes, 64 * 1024, 64 * 1024 * 1024, 'bad_limits')
        _check_int(self.max_context_records, 1, 1000000, 'bad_limits')
        _check_int(self.max_kernel_records, 1, 8000000, 'bad_limits')
        _check_int(self.max_copy_records, 1, 8000000, 'bad_limits')
        _check_int(self.max_graph_events, 1, 4000000, 'bad_limits')
        _check_int(self.max_graphs, 1, 1024, 'bad_limits')
        _check_int(self.max_nodes_per_graph, 1, 1000000, 'bad_limits')
        _check_int(self.max_nodes_total, 1, 2000000, 'bad_limits')
        _check_int(self.max_links, 1, 4000000, 'bad_limits')
        _check_int(self.max_receipt_bytes, 1024, 64 * 1024 * 1024, 'bad_limits')
        if self.max_nodes_total < self.max_nodes_per_graph:
            raise ReplayObserverError('bad_limits')

    def native_args(self):
        return (self.buffer_count, self.buffer_bytes, self.max_context_records,
                self.max_kernel_records, self.max_copy_records,
                self.max_graph_events, self.max_graphs,
                self.max_nodes_per_graph, self.max_nodes_total, self.max_links)

    def as_record(self):
        return {name: getattr(self, name) for name in _LIMIT_KEYS}


@dataclass(frozen=True)
class DrainedProof:
    """Caller-owned evidence of a completed relevant-consumer drain.

    There is no default ``drained=True``.  The observer binds this record into
    the receipt but does not verify it.
    """

    worker_rank: int
    worker_incarnation: str
    engine_incarnation: str
    route_generation: int
    storage_epoch: int
    relevant_consumers: int
    controller_drained: bool

    def __post_init__(self):
        _check_int(self.worker_rank, 0, 1 << 31, 'bad_proof')
        _check_int(self.route_generation, 0, 1 << 31, 'bad_proof')
        _check_int(self.storage_epoch, 1, 1 << 31, 'bad_proof')
        _check_int(self.relevant_consumers, 0, 1 << 20, 'bad_proof')
        if type(self.controller_drained) is not bool:
            raise ReplayObserverError('bad_proof')
        for name in ('worker_incarnation', 'engine_incarnation'):
            value = getattr(self, name)
            if (type(value) is not str or not value or len(value) > 128
                    or any(ord(c) < 32 for c in value)):
                raise ReplayObserverError('bad_proof')

    def as_record(self):
        return {
            'worker_rank': self.worker_rank,
            'worker_incarnation': self.worker_incarnation,
            'engine_incarnation': self.engine_incarnation,
            'route_generation': self.route_generation,
            'storage_epoch': self.storage_epoch,
            'relevant_consumers': self.relevant_consumers,
            'controller_drained': self.controller_drained,
            'verified_by_observer': False,
        }


class ReplayBinding:
    """Pin the observer library, its source and the loaded CUPTI library.

    ``library_path`` must be a trusted local immutable path.  It is never
    accepted from an RPC.  ``cupti_library_path`` is for CPU fixtures; when
    omitted the production path is resolved from ``/proc/self/maps``.
    """

    def __init__(self, library_path, expected_library_sha256,
                 expected_source_sha256, expected_cupti_library_sha256, *,
                 ctypes_module=None, cupti_library_path=None):
        if (type(library_path) is not str or not os.path.isabs(library_path)
                or not os.path.isfile(library_path)):
            raise ReplayObserverError('bad_binding')
        for digest in (expected_library_sha256, expected_source_sha256,
                       expected_cupti_library_sha256):
            if not _is_digest(digest):
                raise ReplayObserverError('bad_binding')
        actual_library = _hash_file(library_path)
        if actual_library != expected_library_sha256:
            raise ReplayObserverError('library_sha_mismatch')
        actual_source = helper_source_sha256()
        if actual_source != expected_source_sha256:
            raise ReplayObserverError('source_sha_mismatch')
        if cupti_library_path is None:
            cupti_path = _loaded_cupti_path()
        else:
            if (type(cupti_library_path) is not str
                    or not os.path.isabs(cupti_library_path)
                    or not os.path.isfile(cupti_library_path)):
                raise ReplayObserverError('cupti_library_missing')
            cupti_path = os.path.realpath(cupti_library_path)
        actual_cupti = _hash_file(cupti_path)
        if actual_cupti != expected_cupti_library_sha256:
            raise ReplayObserverError('cupti_sha_mismatch')

        if ctypes_module is None:
            import ctypes as ctypes_module
        c = ctypes_module
        try:
            lib = c.CDLL(library_path)
        except OSError:
            raise ReplayObserverError('library_load_failed') from None
        U64, U32, I, P = c.c_uint64, c.c_uint32, c.c_int, c.c_void_p
        try:
            lib.s5ro_abi.argtypes = []
            lib.s5ro_abi.restype = I
            lib.s5ro_create.argtypes = [U32, U64, U32, U32, U32, U32, U32,
                                        U32, U32, U32, c.POINTER(U64)]
            lib.s5ro_create.restype = I
            lib.s5ro_reserved_bytes.argtypes = [U64, c.POINTER(U64)]
            lib.s5ro_reserved_bytes.restype = I
            lib.s5ro_start.argtypes = [U64]
            lib.s5ro_start.restype = I
            lib.s5ro_bind_context.argtypes = [U64, U64, U32]
            lib.s5ro_bind_context.restype = I
            lib.s5ro_bind_graph.argtypes = [U64, U64, U32, c.POINTER(U32)]
            lib.s5ro_bind_graph.restype = I
            lib.s5ro_static.argtypes = [U64, P, U64, c.POINTER(U64)]
            lib.s5ro_static.restype = I
            lib.s5ro_begin_window.argtypes = [U64, U32, c.POINTER(U64)]
            lib.s5ro_begin_window.restype = I
            lib.s5ro_end_window.argtypes = [U64, U64, U32, P, U64,
                                            c.POINTER(U64)]
            lib.s5ro_end_window.restype = I
            lib.s5ro_launch_begin.argtypes = [U64]
            lib.s5ro_launch_begin.restype = I
            lib.s5ro_launch_report.argtypes = [U64,U32,P,U64,c.POINTER(U64)]
            lib.s5ro_launch_report.restype = I
            lib.s5ro_graph_ids.argtypes = [U64,P,U64,c.POINTER(U64)]
            lib.s5ro_graph_ids.restype = I
            lib.s5ro_stop.argtypes = [U64]
            lib.s5ro_stop.restype = I
            lib.s5ro_destroy.argtypes = [U64]
            lib.s5ro_destroy.restype = I
            abi = int(lib.s5ro_abi())
        except (AttributeError, TypeError):
            raise ReplayObserverError('bad_binding') from None
        if abi != ABI_VERSION:
            raise ReplayObserverError('abi_mismatch')

        self.library_path = library_path
        self.library_sha256 = actual_library
        self.source_sha256 = actual_source
        self.cupti_library_sha256 = actual_cupti
        self.abi = abi
        self._c = c
        self._lib = lib  # private; not part of the public surface

    def identity(self):
        return {
            'abi': self.abi,
            'library_sha256': self.library_sha256,
            'source_sha256': self.source_sha256,
            'cupti_library_sha256': self.cupti_library_sha256,
        }


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReplayObserverError('duplicate_json_key')
        result[key] = value
    return result


def _reject_constant(_value):
    raise ReplayObserverError('malformed_native_report')


def _parse_report(text):
    if type(text) is not str:
        raise ReplayObserverError('malformed_native_report')
    try:
        return json.loads(text, object_pairs_hook=_no_duplicate_keys,
                          parse_constant=_reject_constant)
    except ReplayObserverError:
        raise
    except (ValueError, TypeError):
        raise ReplayObserverError('malformed_native_report') from None


def _exact_obj(value, keys, code='malformed_native_report'):
    if type(value) is not dict or set(value) != set(keys):
        raise ReplayObserverError(code)
    return value


def _strict_bool(value):
    if type(value) is not bool:
        raise ReplayObserverError('malformed_native_report')
    return value


def _strict_int(value, low=0, high=(1 << 63) - 1):
    if type(value) is not int or not low <= value <= high:
        raise ReplayObserverError('malformed_native_report')
    return value


def _reason_list(value):
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ReplayObserverError('malformed_native_report')
    if len(set(value)) != len(value) or any(item not in REASONS for item in value):
        raise ReplayObserverError('malformed_native_report')
    return tuple(value)


def _type_counts(value):
    _exact_obj(value, _NODE_TYPE_KEYS)
    return {key: _strict_int(value[key]) for key in _NODE_TYPE_KEYS}


def _required(value):
    _exact_obj(value, _REQUIRED_KEYS)
    return {key: _strict_int(value[key]) for key in _REQUIRED_KEYS}


def _limits(value):
    _exact_obj(value, _LIMIT_KEYS)
    return {key: _strict_int(value[key], 1) for key in _LIMIT_KEYS}


def _validate_static(report):
    _exact_obj(report, ('schema', 'abi', 'compiled_api', 'runtime_api',
                        'limits', 'reserved_bytes', 'started',
                        'expected_context', 'graph_slots', 'complete',
                        'reasons'))
    if report['schema'] != NATIVE_SCHEMA_STATIC:
        raise ReplayObserverError('malformed_native_report')
    if _strict_int(report['abi']) != ABI_VERSION:
        raise ReplayObserverError('malformed_native_report')
    if _strict_int(report['compiled_api']) != CUPTI_API:
        raise ReplayObserverError('malformed_native_report')
    if _strict_int(report['runtime_api']) != CUPTI_API:
        raise ReplayObserverError('malformed_native_report')
    limits = _limits(report['limits'])
    _strict_int(report['reserved_bytes'], 1)
    if _strict_bool(report['started']) is not True:
        raise ReplayObserverError('malformed_native_report')
    context = _exact_obj(report['expected_context'], ('bound', 'known', 'expected_sms'))
    if _strict_bool(context['bound']) is not True or _strict_bool(context['known']) is not True:
        raise ReplayObserverError('malformed_native_report')
    _strict_int(context['expected_sms'], 1)
    slots = report['graph_slots']
    if type(slots) is not list or not slots:
        raise ReplayObserverError('malformed_native_report')
    seen = set()
    union = set()
    for slot in slots:
        _exact_obj(slot, ('slot', 'node_count', 'node_types', 'required',
                          'unsupported', 'unknown_type', 'graph_id_known',
                          'node_ids_known', 'inventory_changed',
                          'static_complete', 'reasons'))
        index = _strict_int(slot['slot'])
        if index in seen or index >= limits['max_graphs']:
            raise ReplayObserverError('malformed_native_report')
        seen.add(index)
        node_count = _strict_int(slot['node_count'])
        if node_count > limits['max_nodes_per_graph']:
            raise ReplayObserverError('malformed_native_report')
        counts = _type_counts(slot['node_types'])
        if sum(counts.values()) != node_count:
            raise ReplayObserverError('malformed_native_report')
        required = _required(slot['required'])
        _strict_int(slot['unsupported'])
        _strict_int(slot['unknown_type'])
        graph_known = _strict_bool(slot['graph_id_known'])
        nodes_known = _strict_bool(slot['node_ids_known'])
        _strict_bool(slot['inventory_changed'])
        reasons = _reason_list(slot['reasons'])
        complete = _strict_bool(slot['static_complete'])
        expected_complete = (not reasons and node_count > 0 and graph_known
                             and nodes_known)
        if complete != expected_complete:
            raise ReplayObserverError('malformed_native_report')
        if required['kernel'] != counts['kernel'] or required['memcpy'] != counts['memcpy'] or required['memset'] != counts['memset']:
            raise ReplayObserverError('malformed_native_report')
        union.update(reasons)
    root = set(_reason_list(report['reasons']))
    if root != union:
        raise ReplayObserverError('malformed_native_report')
    if _strict_bool(report['complete']) != (not root):
        raise ReplayObserverError('malformed_native_report')
    return limits


def _validate_context(value):
    _exact_obj(value, _CONTEXT_KEYS)
    bound = _strict_bool(value['bound'])
    known = _strict_bool(value['context_known'])
    _strict_int(value['expected_sms'])
    records = _strict_int(value['records'])
    green_true = _strict_int(value['green_true'])
    green_false = _strict_int(value['green_false'])
    sm_known = _strict_int(value['sm_known'])
    sm_matching = _strict_int(value['sm_matching'])
    sm_mismatched = _strict_int(value['sm_mismatched'])
    if bound is not True or known is not True:
        raise ReplayObserverError('malformed_native_report')
    if green_true + green_false != records or sm_matching + sm_mismatched != sm_known:
        raise ReplayObserverError('malformed_native_report')
    if sm_known > green_true:
        raise ReplayObserverError('malformed_native_report')
    if value['is_green'] is not None and type(value['is_green']) is not bool:
        raise ReplayObserverError('malformed_native_report')
    if value['num_multiprocessors'] is not None:
        _strict_int(value['num_multiprocessors'])
    _strict_bool(value['parent_context_present'])
    duplicate = _strict_bool(value['duplicate'])
    contradictory = _strict_bool(value['contradictory'])
    if duplicate and contradictory:
        raise ReplayObserverError('malformed_native_report')
    if records == 0 and (green_true or green_false or sm_known):
        raise ReplayObserverError('malformed_native_report')


def _validate_window(report, limits, route_generation):
    _exact_obj(report, ('schema', 'abi', 'compiled_api', 'runtime_api',
                        'limits', 'reserved_bytes', 'window',
                        'expected_context', 'counts', 'mapping', 'graph_slots',
                        'flags', 'complete', 'reasons'))
    if report['schema'] != NATIVE_SCHEMA_WINDOW:
        raise ReplayObserverError('malformed_native_report')
    if _strict_int(report['abi']) != ABI_VERSION:
        raise ReplayObserverError('malformed_native_report')
    if _strict_int(report['compiled_api']) != CUPTI_API:
        raise ReplayObserverError('malformed_native_report')
    if _strict_int(report['runtime_api']) != CUPTI_API:
        raise ReplayObserverError('malformed_native_report')
    native_limits = _limits(report['limits'])
    limit_record = limits.as_record()
    _strict_int(report['reserved_bytes'], 1)
    window = _exact_obj(report['window'], ('start_ns', 'end_ns',
                                           'route_generation', 'flushed',
                                           'flush_forced'))
    start = _strict_int(window['start_ns'], 1)
    end = _strict_int(window['end_ns'], 1)
    if end < start:
        raise ReplayObserverError('malformed_native_report')
    if _strict_int(window['route_generation']) != route_generation:
        raise ReplayObserverError('malformed_native_report')
    _strict_bool(window['flushed'])
    if _strict_bool(window['flush_forced']) is not False:
        raise ReplayObserverError('malformed_native_report')
    _validate_context(report['expected_context'])
    counts_obj = _exact_obj(report['counts'],
                            tuple(_COUNT_KEYS) + ('ring_overflows',))
    for key in _COUNT_KEYS:
        _strict_int(counts_obj[key])
    rings = _exact_obj(counts_obj['ring_overflows'],
                       ('context', 'kernel', 'copy', 'resource', 'aliases'))
    for key in ('context', 'kernel', 'copy', 'resource', 'aliases'):
        _strict_int(rings[key])
    mapping = _exact_obj(report['mapping'], _MAPPING_KEYS)
    for key in ('graph_ids', 'node_keys', 'node_aliases', 'alias_unknown'):
        _strict_int(mapping[key])
    for key in ('ambiguous_graph_ids', 'node_overlap'):
        _strict_bool(mapping[key])
    flags = _exact_obj(report['flags'], _FLAG_KEYS)
    for key in _FLAG_KEYS:
        _strict_bool(flags[key])
    if flags['known_wrong_context'] != (counts_obj['known_wrong_context_records'] > 0):
        raise ReplayObserverError('malformed_native_report')
    if flags['drop_seen'] != (counts_obj['dropped_records_total'] > 0):
        raise ReplayObserverError('malformed_native_report')
    if flags['buffer_refused'] != (counts_obj['buffers_refused'] > 0):
        raise ReplayObserverError('malformed_native_report')
    outstanding_expected = (
        counts_obj['buffers_outstanding'] > 0
        or counts_obj['buffers_issued_in_window']
        > counts_obj['buffers_completed_in_window'])
    if flags['buffer_outstanding'] != outstanding_expected:
        raise ReplayObserverError('malformed_native_report')
    if flags['parse_error'] != (counts_obj['parse_errors'] > 0):
        raise ReplayObserverError('malformed_native_report')
    if flags['drop_query_error'] != (counts_obj['drop_query_errors'] > 0):
        raise ReplayObserverError('malformed_native_report')
    if flags['unknown_activity'] != (counts_obj['unknown_activity_records'] > 0):
        raise ReplayObserverError('malformed_native_report')
    if flags['unknown_resource'] != (counts_obj['unknown_resource_events'] > 0):
        raise ReplayObserverError('malformed_native_report')
    if flags['outside_scope_graph_records'] != (
            counts_obj['outside_scope_graph_records'] > 0):
        raise ReplayObserverError('malformed_native_report')
    if flags['reentrant_refused'] != (
            counts_obj['callback_reentrant_refused'] > 0):
        raise ReplayObserverError('malformed_native_report')
    if flags['permanent_overflow'] != any(rings[key] > 0 for key in rings):
        raise ReplayObserverError('malformed_native_report')
    if rings['aliases'] > 0 and not flags['link_overflow']:
        raise ReplayObserverError('malformed_native_report')
    context = report['expected_context']
    if flags['expected_context_missing'] != (context['records'] == 0):
        raise ReplayObserverError('malformed_native_report')
    if flags['expected_context_duplicate'] != context['duplicate']:
        raise ReplayObserverError('malformed_native_report')
    if flags['expected_context_contradictory'] != context['contradictory']:
        raise ReplayObserverError('malformed_native_report')
    if flags['known_wrong_greenness'] != (
            context['records'] > 0 and context['green_true'] == 0):
        raise ReplayObserverError('malformed_native_report')
    if flags['known_wrong_sm'] != (
            context['green_true'] > 0 and context['sm_known'] > 0
            and context['sm_matching'] == 0 and context['sm_mismatched'] > 0):
        raise ReplayObserverError('malformed_native_report')
    slots = report['graph_slots']
    if type(slots) is not list:
        raise ReplayObserverError('malformed_native_report')
    seen = set()
    exercised_any = False
    exercised_reasons = set()
    for slot in slots:
        _exact_obj(slot, ('slot', 'node_count', 'node_types', 'required',
                          'observed', 'missing', 'missing_kernel_indices',
                          'missing_copy_indices', 'missing_memset_indices',
                          'context_mismatch_nodes', 'kernel_executions',
                          'copy_executions', 'memset_executions',
                          'node_execution_bounds',
                          'graph_id_known', 'node_ids_known',
                          'static_complete', 'unsupported', 'unknown_type',
                          'inventory_changed', 'update_seen', 'destroy_seen',
                          'mapping_ambiguous', 'exercised', 'complete',
                          'reasons'))
        index = _strict_int(slot['slot'])
        if index in seen or index >= limit_record['max_graphs']:
            raise ReplayObserverError('malformed_native_report')
        seen.add(index)
        node_count = _strict_int(slot['node_count'])
        if node_count > limit_record['max_nodes_per_graph']:
            raise ReplayObserverError('malformed_native_report')
        type_counts = _type_counts(slot['node_types'])
        if sum(type_counts.values()) != node_count:
            raise ReplayObserverError('malformed_native_report')
        required = _required(slot['required'])
        observed = _exact_obj(slot['observed'], _OBSERVED_KEYS)
        missing = _exact_obj(slot['missing'], _OBSERVED_KEYS)
        for key in _OBSERVED_KEYS:
            _strict_int(observed[key])
            _strict_int(missing[key])
            if observed[key] + missing[key] != required[key]:
                raise ReplayObserverError('malformed_native_report')
        for key, expected in (('missing_kernel_indices', missing['kernel']),
                              ('missing_copy_indices', missing['memcpy']),
                              ('missing_memset_indices', missing['memset'])):
            indices = slot[key]
            if type(indices) is not list:
                raise ReplayObserverError('malformed_native_report')
            if len(indices) > 64 or len(indices) > expected:
                raise ReplayObserverError('malformed_native_report')
            for item in indices:
                if _strict_int(item) >= node_count:
                    raise ReplayObserverError('malformed_native_report')
        mismatches = slot['context_mismatch_nodes']
        if type(mismatches) is not list or len(mismatches) > 64:
            raise ReplayObserverError('malformed_native_report')
        for item in mismatches:
            if _strict_int(item) >= node_count:
                raise ReplayObserverError('malformed_native_report')
        _strict_int(slot['kernel_executions'])
        _strict_int(slot['copy_executions'])
        _strict_int(slot['memset_executions'])
        bounds = _exact_obj(slot['node_execution_bounds'], _OBSERVED_KEYS)
        for kind in _OBSERVED_KEYS:
            pair = _exact_obj(bounds[kind], ('min', 'max'))
            low, high = _strict_int(pair['min']), _strict_int(pair['max'])
            if low > high or (required[kind] == 0 and high != 0):
                raise ReplayObserverError('malformed_native_report')
        _strict_bool(slot['graph_id_known'])
        _strict_bool(slot['node_ids_known'])
        static_complete = _strict_bool(slot['static_complete'])
        _strict_int(slot['unsupported'])
        _strict_int(slot['unknown_type'])
        _strict_bool(slot['inventory_changed'])
        _strict_bool(slot['update_seen'])
        _strict_bool(slot['destroy_seen'])
        _strict_bool(slot['mapping_ambiguous'])
        exercised = _strict_bool(slot['exercised'])
        complete = _strict_bool(slot['complete'])
        reasons = _reason_list(slot['reasons'])
        if complete and (not exercised or not static_complete or reasons):
            raise ReplayObserverError('malformed_native_report')
        if exercised:
            exercised_any = True
            exercised_reasons.update(reasons)
        if required['kernel'] != type_counts['kernel'] or required['memcpy'] != type_counts['memcpy'] or required['memset'] != type_counts['memset']:
            raise ReplayObserverError('malformed_native_report')
    root_list = _reason_list(report['reasons'])
    root = set(root_list)
    if not exercised_reasons <= root:
        raise ReplayObserverError('malformed_native_report')
    if flags['no_exercised_graph'] != (not exercised_any):
        raise ReplayObserverError('malformed_native_report')
    if _strict_bool(report['complete']) != (not root):
        raise ReplayObserverError('malformed_native_report')
    if flags['known_wrong_context'] and 'known_wrong_context' not in root:
        raise ReplayObserverError('malformed_native_report')
    if flags['known_wrong_greenness'] and 'expected_context_not_green' not in root:
        raise ReplayObserverError('malformed_native_report')
    if flags['known_wrong_sm'] and 'expected_context_sm_mismatch' not in root:
        raise ReplayObserverError('malformed_native_report')
    if flags['context_destroyed'] and 'context_destroyed' not in root:
        raise ReplayObserverError('malformed_native_report')
    if flags['flush_failed'] and 'flush_failed' not in root:
        raise ReplayObserverError('malformed_native_report')
    if flags['buffer_outstanding'] and 'buffer_outstanding' not in root:
        raise ReplayObserverError('malformed_native_report')
    if mapping['ambiguous_graph_ids'] and 'ambiguous_graph' not in root:
        raise ReplayObserverError('malformed_native_report')
    if mapping['node_overlap'] and 'node_overlap' not in root:
        raise ReplayObserverError('malformed_native_report')
    if flags['link_overflow'] and 'link_overflow' not in root:
        raise ReplayObserverError('malformed_native_report')
    if flags['update_seen'] and 'update_seen' not in root:
        raise ReplayObserverError('malformed_native_report')
    if flags['inventory_changed'] and 'inventory_changed' not in root:
        raise ReplayObserverError('malformed_native_report')
    if native_limits != limits.as_record():
        raise ReplayObserverError('malformed_native_report')
    return counts_obj, flags, root_list


def _freeze(value):
    if type(value) is dict:
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if type(value) in (list, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _receipt_dict(value):
    """Plain JSON-safe copy of a frozen receipt."""
    if isinstance(value, MappingProxyType):
        return {key: _receipt_dict(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_receipt_dict(item) for item in value]
    return value


def _same(a, b):
    if a is b:
        return True
    try:
        return type(a) is type(b) and a == b
    except Exception:
        return False


def _positive_handle(value):
    return type(value) is int and value > 0


class ObserverSession:
    """One default-off observer session.  Native handles stay private."""

    def __init__(self, binding, limits=None):
        if not isinstance(binding, ReplayBinding):
            raise ReplayObserverError('bad_binding')
        if limits is None:
            limits = ObserverLimits()
        if not isinstance(limits, ObserverLimits):
            raise ReplayObserverError('bad_limits')
        self.binding = binding
        self.limits = limits
        self.state = 'created'
        self._token = 0
        self._window_token = None
        self._window_route = None
        self._bank = None
        self._static = None
        self._static_report = None
        self._descriptor_labels = ()
        self._descriptor_objects = ()
        self._bank_meta = None
        self._probe = None
        self._router = None
        self._split = None
        self._manager = None
        self._graphs = None
        self._handles = None
        self._pairs = None
        self._capture_owner = None
        self._capture_context = None
        self._graph_objects = ()
        self._handle_objects = ()
        self._local_handles = ()
        self._graph_handles = ()
        self._storage_epoch = None
        self._route_generation = None
        self._router_epoch = None
        self._router_generation = None
        self._green = None
        self._measured_sms = None
        self._requested_sms = None
        self._probe_route = None
        self._probe_interval = None
        self._split_state = _MISSING
        self._native_reserved = 0
        c = binding._c
        out = c.c_uint64()
        values = (limits.buffer_count, limits.buffer_bytes,
                  limits.max_context_records, limits.max_kernel_records,
                  limits.max_copy_records, limits.max_graph_events,
                  limits.max_graphs, limits.max_nodes_per_graph,
                  limits.max_nodes_total, limits.max_links)
        rc = binding._lib.s5ro_create(*values, c.byref(out))
        if rc != 0:
            raise ReplayObserverError(_STATUS_NAMES.get(rc, 'native_error'))
        self._token = int(out.value)
        if self._token == 0:
            raise ReplayObserverError('native_error')
        reserved = c.c_uint64()
        rc = binding._lib.s5ro_reserved_bytes(self._token, c.byref(reserved))
        if rc != 0:
            binding._lib.s5ro_destroy(self._token)
            raise ReplayObserverError(_STATUS_NAMES.get(rc, 'native_error'))
        self._native_reserved = int(reserved.value)

    # -- private native plumbing -----------------------------------------
    def _require(self, *states):
        if self.state not in states:
            raise ReplayObserverError('bad_state')

    def _report_call(self, function, args, retry=True):
        c = self.binding._c
        capacity = self.limits.max_receipt_bytes
        for _attempt in range(2):
            buffer = c.create_string_buffer(capacity)
            length = c.c_uint64()
            rc = function(*args, c.cast(buffer, c.c_void_p), capacity,
                          c.byref(length))
            required = int(length.value)
            if rc == 0:
                if required > capacity:
                    raise ReplayObserverError('malformed_native_report')
                raw = bytes(buffer.raw[:required])
                try:
                    text = raw.decode('utf-8')
                except UnicodeDecodeError:
                    raise ReplayObserverError('malformed_native_report') from None
                return _parse_report(text)
            if rc == 1:
                if (not retry or required <= capacity
                        or required > self.limits.max_receipt_bytes):
                    raise ReplayObserverError('receipt_too_large')
                capacity = required
                continue
            raise ReplayObserverError(_STATUS_NAMES.get(rc, 'native_error'))
        raise ReplayObserverError('receipt_too_large')

    # -- lifecycle --------------------------------------------------------
    def start(self):
        self._require('created')
        rc = self.binding._lib.s5ro_start(self._token)
        if rc != 0:
            if rc == 25:
                # Native retained subscription/activity ownership for a
                # bounded stop retry; close must not issue destroy yet.
                self.state = 'start_failed'
            raise ReplayObserverError(_STATUS_NAMES.get(rc, 'native_error'))
        self.state = 'started'
        return True

    def bind_bank(self, probe):
        """Bind exactly the retained bank graphs and the owning green split.

        No graph is recreated or replayed.  The bank must already be captured
        and its capture owner/checks must still name the same split.
        """
        self._require('started')
        bank = getattr(probe, 'bank', None)
        router = getattr(probe, 'router', None)
        split = getattr(probe, 'split', None)
        if bank is None or router is None or split is None:
            raise ReplayObserverError('probe_contract')
        manager = getattr(bank, 'manager', None)
        graphs = getattr(manager, 'graphs', None)
        pairs = getattr(bank, 'pairs', None)
        handles = getattr(bank, 'handles', None)
        if (manager is None or type(graphs) is not dict
                or type(pairs) not in (tuple, list) or not pairs
                or type(handles) is not dict):
            raise ReplayObserverError('bank_contract')
        if getattr(bank, 'captured', None) is not True:
            raise ReplayObserverError('bank_not_captured')
        if getattr(manager, '_flash_s5_capture_owner', None) is not split:
            raise ReplayObserverError('capture_owner_mismatch')
        if not callable(getattr(manager, '_flash_s5_capture_context', None)):
            raise ReplayObserverError('capture_owner_mismatch')
        if getattr(probe, 'interval', None) == 'invalidated':
            raise ReplayObserverError('probe_invalidated')
        storage_epoch = getattr(probe, 'storage_epoch', None)
        route_generation = getattr(probe, 'route_generation', None)
        router_epoch = getattr(router, 'storage_epoch', None)
        router_generation = getattr(router, 'route_generation', None)
        if (type(storage_epoch) is not int or storage_epoch < 1
                or type(router_epoch) is not int or router_epoch < 1):
            raise ReplayObserverError('storage_epoch_missing')
        if storage_epoch != router_epoch:
            raise ReplayObserverError('storage_epoch_mismatch')
        if (type(route_generation) is not int or route_generation < 0
                or type(router_generation) is not int or router_generation < 0):
            raise ReplayObserverError('route_generation_missing')
        if route_generation != router_generation:
            raise ReplayObserverError('route_generation_mismatch')
        route = getattr(probe, 'route', None)
        if route not in ('full', 'split'):
            raise ReplayObserverError('probe_contract')
        green = getattr(split, 'decode_context_id', None)
        measured_sms = getattr(split, 'decode_sms', None)
        requested_sms = getattr(split, 'requested_decode_sms', None)
        if not _positive_handle(green):
            raise ReplayObserverError('green_context_missing')
        if type(measured_sms) is not int or measured_sms < 1:
            raise ReplayObserverError('measured_sms_missing')
        if type(requested_sms) is not int or requested_sms < 1:
            raise ReplayObserverError('requested_sms_missing')
        if len(pairs) > self.limits.max_graphs:
            raise ReplayObserverError('graph_limit')

        descriptors = []
        local_handles = []
        ordered = []
        for position, pair in enumerate(pairs):
            if type(pair) not in (tuple, list) or len(pair) != 2:
                raise ReplayObserverError('descriptor_inventory_missing')
            descriptor, local = pair
            if descriptor not in graphs:
                raise ReplayObserverError('descriptor_inventory_missing')
            if any(_same(local, seen) for seen in local_handles):
                raise ReplayObserverError('descriptor_duplicate')
            handle = handles.get(local)
            if handle is None:
                raise ReplayObserverError('descriptor_inventory_missing')
            if (getattr(handle, 'bank', None) is not bank
                    or getattr(handle, 'destroyed', None) is not False
                    or not _same(getattr(handle, 'local', None), local)):
                raise ReplayObserverError('handle_contract')
            graph_obj = graphs[descriptor]
            raw = getattr(graph_obj, 'raw_cuda_graph', None)
            if not callable(raw):
                raise ReplayObserverError('graph_handle_missing')
            graph_handle = raw()
            if not _positive_handle(graph_handle):
                raise ReplayObserverError('graph_handle_missing')
            if any(graph_handle == seen[1] for seen in ordered):
                raise ReplayObserverError('graph_duplicate')
            descriptors.append(descriptor)
            local_handles.append(local)
            ordered.append((position, graph_handle, local))
        if len(ordered) != len(graphs):
            raise ReplayObserverError('descriptor_inventory_mismatch')

        rc = self.binding._lib.s5ro_bind_context(self._token, green, measured_sms)
        if rc != 0:
            raise ReplayObserverError(_STATUS_NAMES.get(rc, 'native_error'))
        for position, graph_handle, _local in ordered:
            node_count = self.binding._c.c_uint32()
            rc = self.binding._lib.s5ro_bind_graph(
                self._token, graph_handle, position,
                self.binding._c.byref(node_count))
            if rc != 0:
                raise ReplayObserverError(_STATUS_NAMES.get(rc, 'native_error'))

        static_report = self._report_call(self.binding._lib.s5ro_static,
                                          (self._token,))
        native_limits = _validate_static(static_report)
        if native_limits != self.limits.as_record():
            raise ReplayObserverError('malformed_native_report')
        if static_report['reserved_bytes'] != self._native_reserved:
            raise ReplayObserverError('malformed_native_report')

        # Detect mutation while binding: same-size replacement is not enough.
        if len(graphs) != len(descriptors):
            raise ReplayObserverError('bank_mutated')
        for position, graph_handle, local in ordered:
            descriptor = descriptors[position]
            if descriptor not in graphs:
                raise ReplayObserverError('bank_mutated')
            handle = handles.get(local)
            if handle is None or getattr(handle, 'bank', None) is not bank:
                raise ReplayObserverError('bank_mutated')
            if getattr(handle, 'destroyed', None) is not False:
                raise ReplayObserverError('bank_mutated')
            raw = getattr(graphs[descriptor], 'raw_cuda_graph', None)
            if not callable(raw) or raw() != graph_handle:
                raise ReplayObserverError('bank_mutated')
        if (getattr(probe, 'storage_epoch', None) != storage_epoch
                or getattr(probe, 'route_generation', None) != route_generation
                or getattr(router, 'storage_epoch', None) != router_epoch
                or getattr(router, 'route_generation', None) != router_generation):
            raise ReplayObserverError('bank_mutated')

        self._bank = bank
        self._probe = probe
        self._router = router
        self._split = split
        self._manager = manager
        self._graphs = graphs
        self._handles = handles
        self._pairs = pairs
        self._capture_owner = getattr(manager, '_flash_s5_capture_owner', None)
        self._capture_context = getattr(manager,
                                        '_flash_s5_capture_context', None)
        self._local_handles = tuple(local_handles)
        self._graph_handles = tuple(item[1] for item in ordered)
        self._graph_objects = tuple(graphs[descriptor]
                                    for descriptor in descriptors)
        self._handle_objects = tuple(handles[local]
                                     for local in local_handles)
        self._storage_epoch = storage_epoch
        self._route_generation = route_generation
        self._router_epoch = router_epoch
        self._router_generation = router_generation
        self._green = green
        self._measured_sms = measured_sms
        self._requested_sms = requested_sms
        self._probe_route = route
        self._probe_interval = getattr(probe, 'interval', None)
        self._split_state = getattr(split, 'state', _MISSING)
        self._static = _freeze(static_report)
        self._static_report = static_report
        self._descriptor_labels = tuple(f'descriptor:{i}'
                                        for i in range(len(ordered)))
        self._descriptor_objects = tuple(descriptors)
        self._bank_meta = {
            'storage_epoch': storage_epoch,
            'route_generation': route_generation,
            'descriptor_count': len(ordered),
            'split_decode_sms': measured_sms,
            'split_requested_sms': requested_sms,
            'green_context_bound': True,
            'capture_order_labels': self._descriptor_labels,
        }
        self.state = 'bound'
        return _freeze(static_report)

    def _live_bank_problem(self):
        """Return a short mutation code, or None when the bound owner is live.

        Only object identity, the same graph/handle set captured at bind and
        scalar epochs are inspected.  No address or driver identity leaves
        this method; end_window binds the code into an INCONCLUSIVE receipt.
        """
        bank = self._bank
        try:
            if bank is None:
                return 'bank_mutated'
            if getattr(self._probe, 'bank', None) is not bank:
                return 'bank_mutated'
            if getattr(self._probe, 'router', None) is not self._router:
                return 'bank_mutated'
            if getattr(self._probe, 'split', None) is not self._split:
                return 'bank_mutated'
            if getattr(bank, 'manager', None) is not self._manager:
                return 'bank_mutated'
            if getattr(self._manager, 'graphs', None) is not self._graphs:
                return 'bank_mutated'
            if getattr(bank, 'handles', None) is not self._handles:
                return 'bank_mutated'
            if getattr(bank, 'pairs', None) is not self._pairs:
                return 'bank_mutated'
            if getattr(bank, 'captured', None) is not True:
                return 'bank_mutated'
            owner = getattr(self._manager, '_flash_s5_capture_owner', None)
            if owner is not self._capture_owner or owner is not self._split:
                return 'bank_mutated'
            context_hook = getattr(self._manager,
                                   '_flash_s5_capture_context', None)
            if (not callable(context_hook)
                    or not _same(context_hook, self._capture_context)):
                return 'bank_mutated'
            interval = getattr(self._probe, 'interval', None)
            if interval == 'invalidated' or interval != self._probe_interval:
                return 'bank_mutated'
            if getattr(self._probe, 'route', None) != self._probe_route:
                return 'bank_mutated'
            if getattr(self._probe, 'storage_epoch', None) != self._storage_epoch:
                return 'bank_mutated'
            if getattr(self._probe, 'route_generation', None) != self._route_generation:
                return 'bank_mutated'
            if getattr(self._router, 'storage_epoch', None) != self._router_epoch:
                return 'bank_mutated'
            if getattr(self._router, 'route_generation', None) != self._router_generation:
                return 'bank_mutated'
            if getattr(self._split, 'decode_context_id', None) != self._green:
                return 'bank_mutated'
            if getattr(self._split, 'decode_sms', None) != self._measured_sms:
                return 'bank_mutated'
            if (getattr(self._split, 'requested_decode_sms', None)
                    != self._requested_sms):
                return 'bank_mutated'
            if not _same(getattr(self._split, 'state', _MISSING),
                         self._split_state):
                return 'bank_mutated'
            if len(self._graphs) != len(self._descriptor_objects):
                return 'bank_mutated'
            if len(self._handles) != len(self._local_handles):
                return 'bank_mutated'
            pairs = getattr(bank, 'pairs', None)
            if type(pairs) not in (tuple, list):
                return 'bank_mutated'
            if len(pairs) != len(self._descriptor_objects):
                return 'bank_mutated'
            for index, pair in enumerate(pairs):
                if type(pair) not in (tuple, list) or len(pair) != 2:
                    return 'bank_mutated'
                descriptor, local = pair
                if (not _same(descriptor, self._descriptor_objects[index])
                        or not _same(local, self._local_handles[index])):
                    return 'bank_mutated'
                graph = self._graphs.get(descriptor, _MISSING)
                if graph is not self._graph_objects[index]:
                    return 'bank_mutated'
                handle = self._handles.get(local, _MISSING)
                if handle is not self._handle_objects[index]:
                    return 'bank_mutated'
                if (getattr(handle, 'bank', None) is not bank
                        or getattr(handle, 'destroyed', None) is not False
                        or not _same(getattr(handle, 'local', None), local)):
                    return 'bank_mutated'
                raw = getattr(graph, 'raw_cuda_graph', None)
                if not callable(raw) or raw() != self._graph_handles[index]:
                    return 'bank_mutated'
        except Exception:
            return 'bank_mutated'
        return None

    def replay_census(self):
        """Private bound descriptors -> public graph labels; no CUPTI inference.

        The owner calls this at drained boundaries. Missing/unrecognized
        counters produce unavailable evidence, never an invented zero.
        """
        counts = getattr(self._router, 'replays_by_descriptor', None)
        total = getattr(self._router, 'replays', None)
        if (type(counts) is not dict or type(total) is not int or total < 0
                or any(type(n) is not int or n < 0 for n in counts.values())
                or sum(counts.values()) != total
                or any(d not in self._local_handles for d in counts)):
            return None
        return {'total': total, 'graphs': {
            f'graph:{i}': counts.get(d, 0)
            for i, d in enumerate(self._local_handles)}}

    def begin_window(self, *, route_generation, descriptor=None):
        """Start a bounded ordinary-engine observation window.

        ``descriptor`` is optional.  When supplied it must be one of the bank's
        captured descriptors; the receipt then requires that descriptor to be
        exercised.  The observer cannot manufacture a scheduler pause or make a
        descriptor a member of the caller's workload.
        """
        self._require('bound')
        if type(route_generation) is not int or route_generation < 0:
            raise ReplayObserverError('bad_route_generation')
        if self._live_bank_problem() is not None:
            raise ReplayObserverError('bank_mutated')
        if route_generation != self._bank_meta['route_generation']:
            raise ReplayObserverError('route_generation_mismatch')
        target = None
        if descriptor is not None:
            for index, candidate in enumerate(self._descriptor_objects):
                if _same(candidate, descriptor):
                    target = self._descriptor_labels[index]
                    break
            if target is None:
                raise ReplayObserverError('descriptor_not_in_bank')
        token = self.binding._c.c_uint64()
        rc = self.binding._lib.s5ro_begin_window(self._token, route_generation,
                                                 self.binding._c.byref(token))
        if rc != 0:
            raise ReplayObserverError(_STATUS_NAMES.get(rc, 'native_error'))
        self._window_token = int(token.value)
        self._window_route = route_generation
        self._declared_target = target
        self.state = 'window'
        return True

    def end_window(self, drained_proof):
        """Flush and collect after the caller's completed drain proof."""
        self._require('window')
        if not isinstance(drained_proof, DrainedProof):
            raise ReplayObserverError('bad_proof')
        token = self._window_token
        if token is None:
            raise ReplayObserverError('bad_state')
        live_problem = self._live_bank_problem()
        try:
            report = self._report_call(
                self.binding._lib.s5ro_end_window,
                (self._token, token, self._window_route), retry=False)
        except ReplayObserverError as exc:
            # rc=0 with malformed content and rc=1 truncation both happen after
            # native consumed the token.  Other statuses (notably a transient
            # timestamp failure) leave the native window active, so retain the
            # token and let the caller retry the same end.
            if exc.code in ('receipt_too_large', 'malformed_native_report',
                            'duplicate_json_key'):
                self._window_token = None
                self.state = 'ended'
            raise
        self._window_token = None
        self.state = 'ended'
        counts, flags, native_reasons = _validate_window(
            report, self.limits, self._window_route)
        if report['reserved_bytes'] != self._native_reserved:
            raise ReplayObserverError('malformed_native_report')
        return self._build_receipt(report, counts, flags, native_reasons,
                                   drained_proof, live_problem)

    # -- receipt ----------------------------------------------------------
    def _build_receipt(self, report, counts, flags, native_reasons, proof,
                       live_problem=None):
        fail, inconclusive = [], []

        for reason in native_reasons:
            if reason in NATIVE_FAIL_REASONS:
                fail.append(reason)
        if flags['known_wrong_context']:
            fail.append('known_wrong_context')
        if flags['known_wrong_greenness']:
            fail.append('expected_context_not_green')
        if flags['known_wrong_sm']:
            fail.append('expected_context_sm_mismatch')
        if flags['context_destroyed']:
            fail.append('context_destroyed')
        if proof.route_generation != self._window_route:
            fail.append('caller_route_generation_mismatch')
        if proof.storage_epoch != self._bank_meta['storage_epoch']:
            fail.append('caller_storage_epoch_mismatch')
        if not proof.controller_drained:
            inconclusive.append('drain_not_completed')
        if proof.relevant_consumers == 0:
            inconclusive.append('drain_consumers_missing')

        for reason in native_reasons:
            if reason not in NATIVE_FAIL_REASONS:
                inconclusive.append(reason)
        if not report['complete']:
            inconclusive.append('replay_incomplete')
        if flags['outside_scope_graph_records']:
            inconclusive.append('outside_scope_graph_activity')
        if flags['no_exercised_graph']:
            inconclusive.append('no_exercised_graph')
        if live_problem is not None:
            inconclusive.append(live_problem)

        exercised = []
        qualified = []
        unexercised = []
        declared_target = getattr(self, '_declared_target', None)
        target_exercised = declared_target is None
        for slot in report['graph_slots']:
            index = slot['slot']
            if index >= len(self._descriptor_labels):
                raise ReplayObserverError('malformed_native_report')
            label = f'graph:{index}'
            if slot['exercised']:
                exercised.append(label)
                if declared_target == self._descriptor_labels[index]:
                    target_exercised = True
                if slot['complete'] and report['complete']:
                    qualified.append(label)
                if not slot['complete']:
                    inconclusive.append('exercised_slot_incomplete')
            else:
                unexercised.append(label)
        if not target_exercised:
            inconclusive.append('declared_target_unexercised')

        # FAIL outranks unrelated missing evidence.
        if fail:
            verdict = 'FAIL'
            reasons = tuple(dict.fromkeys(fail + inconclusive))
        elif inconclusive:
            verdict = 'INCONCLUSIVE'
            reasons = tuple(dict.fromkeys(inconclusive))
        else:
            verdict = 'PASS'
            reasons = ()
        if verdict != 'PASS':
            qualified = []

        static = self._static_report
        if static is None:
            raise ReplayObserverError('bad_state')
        static_union = tuple(static['reasons'])
        static_verdict = 'PASS' if static['complete'] else 'INCONCLUSIVE'
        static_slots = []
        for slot in static['graph_slots']:
            static_slots.append({
                'label': f'graph:{slot["slot"]}',
                'node_count': slot['node_count'],
                'node_types': dict(slot['node_types']),
                'required': dict(slot['required']),
                'unsupported': slot['unsupported'],
                'unknown_type': slot['unknown_type'],
                'graph_id_known': slot['graph_id_known'],
                'node_ids_known': slot['node_ids_known'],
                'static_complete': slot['static_complete'],
                'reasons': tuple(slot['reasons']),
            })

        window_slots = []
        for slot in report['graph_slots']:
            index = slot['slot']
            missing_nodes = []
            for kind, key in (('kernel', 'missing_kernel_indices'),
                              ('memcpy', 'missing_copy_indices'),
                              ('memset', 'missing_memset_indices')):
                for node_index in slot[key]:
                    missing_nodes.append(f'node:{index}:{node_index}')
            mismatch_nodes = [f'node:{index}:{item}'
                              for item in slot['context_mismatch_nodes']]
            window_slots.append({
                'label': f'graph:{index}',
                'descriptor': self._descriptor_labels[index],
                'node_count': slot['node_count'],
                'node_types': dict(slot['node_types']),
                'required': dict(slot['required']),
                'observed': dict(slot['observed']),
                'missing': dict(slot['missing']),
                'missing_nodes': tuple(missing_nodes),
                'context_mismatch_nodes': tuple(mismatch_nodes),
                'kernel_executions': slot['kernel_executions'],
                'copy_executions': slot['copy_executions'],
                'memset_executions': slot['memset_executions'],
                'node_execution_bounds': {k: dict(v) for k, v in
                                          slot['node_execution_bounds'].items()},
                'graph_id_known': slot['graph_id_known'],
                'node_ids_known': slot['node_ids_known'],
                'static_complete': slot['static_complete'],
                'unsupported': slot['unsupported'],
                'unknown_type': slot['unknown_type'],
                'inventory_changed': slot['inventory_changed'],
                'update_seen': slot['update_seen'],
                'destroy_seen': slot['destroy_seen'],
                'mapping_ambiguous': slot['mapping_ambiguous'],
                'exercised': slot['exercised'],
                'replay_qualified': f'graph:{index}' in qualified,
                'complete': slot['complete'],
                'reasons': tuple(slot['reasons']),
            })

        context = report['expected_context']
        receipt = {
            'schema': RECEIPT_SCHEMA,
            'verdict': verdict,
            'reasons': reasons,
            'complete': verdict == 'PASS',
            'static_structure': {
                'verdict': static_verdict,
                'complete': static['complete'],
                'graph_count': len(static['graph_slots']),
                'reasons': static_union,
                'graph_slots': tuple(static_slots),
            },
            'replay': {
                'verdict': verdict,
                'window': {
                    'start_ns': report['window']['start_ns'],
                    'end_ns': report['window']['end_ns'],
                    'route_generation': report['window']['route_generation'],
                    'flushed': report['window']['flushed'],
                    'flush_forced': report['window']['flush_forced'],
                },
                'context': {
                    'bound': context['bound'],
                    'expected_sms': context['expected_sms'],
                    'records': context['records'],
                    'green_true': context['green_true'],
                    'green_false': context['green_false'],
                    'sm_known': context['sm_known'],
                    'sm_matching': context['sm_matching'],
                    'sm_mismatched': context['sm_mismatched'],
                    'is_green': context['is_green'],
                    'num_multiprocessors': context['num_multiprocessors'],
                    'parent_context_present': context['parent_context_present'],
                    'duplicate': context['duplicate'],
                    'contradictory': context['contradictory'],
                },
                'counts': {key: counts[key] for key in _COUNT_KEYS},
                'ring_overflows': dict(counts['ring_overflows']),
                'mapping': dict(report['mapping']),
                'graph_slots': tuple(window_slots),
                'flags': {key: flags[key] for key in _FLAG_KEYS},
                'complete': report['complete'],
                'reasons': native_reasons,
            },
            'coverage': {
                'scope': 'exercised descriptors only',
                'descriptor_inventory': self._descriptor_labels,
                'observed_descriptors': tuple(exercised),
                'replay_qualified': tuple(qualified),
                'unexercised_descriptors': tuple(unexercised),
                'declared_target': declared_target,
            },
            'binding': self.binding.identity(),
            'limits': self.limits.as_record(),
            'reserved_bytes': self._native_reserved,
            'bank': dict(self._bank_meta),
            'caller_proof': dict(proof.as_record(),
                                 route_generation_matches=(
                                     proof.route_generation == self._window_route),
                                 storage_epoch_matches=(
                                     proof.storage_epoch == self._bank_meta['storage_epoch'])),
            'source_identities': {
                'native_static_schema': NATIVE_SCHEMA_STATIC,
                'native_window_schema': NATIVE_SCHEMA_WINDOW,
                'compiled_api': report['compiled_api'],
                'runtime_api': report['runtime_api'],
            },
            'limitations': LIMITATIONS,
        }
        self._last_receipt = receipt
        return _freeze(receipt)

    # -- shutdown ---------------------------------------------------------
    def stop(self):
        if self.state == 'window':
            raise ReplayObserverError('window_active')
        if self.state in ('created', 'closed'):
            raise ReplayObserverError('bad_state')
        if self.state == 'stopped':
            return True
        rc = self.binding._lib.s5ro_stop(self._token)
        if rc != 0:
            raise ReplayObserverError(_STATUS_NAMES.get(rc, 'native_error'))
        self.state = 'stopped'
        return True

    def close(self):
        if self.state == 'closed':
            return True
        if self.state == 'window':
            raise ReplayObserverError('window_active')
        if self.state == 'created':
            pass
        elif self.state != 'stopped':
            self.stop()
        rc = self.binding._lib.s5ro_destroy(self._token)
        if rc != 0:
            raise ReplayObserverError(_STATUS_NAMES.get(rc, 'native_error'))
        self._token = 0
        self.state = 'closed'
        return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except ReplayObserverError:
            if exc_type is None:
                raise
        return False


__all__ = ['ABI_VERSION', 'DrainedProof', 'ObserverLimits', 'ObserverSession',
           'RECEIPT_SCHEMA', 'ReplayBinding', 'ReplayObserverError',
           'helper_source_sha256', 'REASONS', 'LIMITATIONS']
