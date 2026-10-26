"""E3 raw retained-graph audit (offline contract, own C ABI 1).

``audit_bank`` inspects exactly the graphs retained by a decode-probe split
bank (``probe.bank.manager.graphs``), compares their descriptor inventory with
the bank/router declarations and capture observations, and asks the pinned
native helper for bounded per-node metadata. It never replays, synchronises,
repairs or recreates a graph, never writes a file, and never accepts a path
from an RPC.

Importing this module imports neither torch nor vLLM and loads no library:
ctypes and the shared object are touched only when :class:`AuditBinding` is
constructed. The helper is loaded by exact SHA-256 and its own ABI version is
checked before any inspection. Handles are internal only; no handle, address,
``CUfunction``, kernel argument, tensor value or raw driver error string is put
in a result, reason or exception.

Evidence boundary: this module can prove graph-node context/SM metadata for a
retained, paused bank. It cannot prove that the scheduler is paused, that no
graph is replaying, or which worker/engine incarnation it is looking at; the
caller supplies and proves those. Unknown/unsupported driver queries are
INCONCLUSIVE. Only a known context or SM mismatch is FAIL, and a FAIL is never
masked by an unrelated unknown. A kernel-name substring is a candidate
collective classifier, never confirmed collective proof.
"""
import hashlib
import json
import os
import types
from dataclasses import dataclass

from e3_types import E3ContractError

ABI_VERSION = 1
HELPER_SOURCE = 'native/flash_s5_graph_audit.cu'
SCHEMA = 's5ga-graph-audit/1'
NATIVE_SCHEMA = 's5ga-native-audit/1'

# Exported signature, fixed together with native/flash_s5_graph_audit.cu.
INSPECT_SIGNATURE = (
    's5ga_inspect',
    ('uint64', 'uint64', 'uint64', 'uint32', 'uint32', 'uint64',
     'char*', 'uint64', 'uint64*'),
)

NATIVE_OK = 0
NATIVE_TRUNCATED = 1
NATIVE_BAD_ARGUMENTS = 2
NATIVE_GREEN_QUERY_FAILED = 3
NATIVE_GREEN_MISSING = 4
NATIVE_GREEN_MISMATCH = 5
NATIVE_CONTEXT_CONVERT_FAILED = 6
NATIVE_GRAPH_QUERY_FAILED = 7

STATUS_BY_RC = {
    NATIVE_OK: 'ok',
    NATIVE_TRUNCATED: 'truncated',
    NATIVE_BAD_ARGUMENTS: 'bad_arguments',
    NATIVE_GREEN_QUERY_FAILED: 'green_query_failed',
    NATIVE_GREEN_MISSING: 'green_missing',
    NATIVE_GREEN_MISMATCH: 'green_context_mismatch',
    NATIVE_CONTEXT_CONVERT_FAILED: 'context_convert_failed',
    NATIVE_GRAPH_QUERY_FAILED: 'graph_query_failed',
}

DEFAULT_MAX_NODES = 200_000
DEFAULT_MAX_DEPTH = 32
DEFAULT_MAX_BYTES = 4 * 1024 * 1024
MAX_NODES = 1_000_000
MAX_DEPTH = 64
MAX_BYTES = 64 * 1024 * 1024
MIN_BYTES = 256
MAX_GRAPHS = 1024

CAPTURE_BANK_LABEL = 'decode-split'
CAPTURE_STREAM_LABEL = 'decode-split-stream'
CAPTURE_CONTEXT_LABEL = 'decode-split-context'

PASS = 'PASS'
FAIL = 'FAIL'
INCONCLUSIVE = 'INCONCLUSIVE'

# Root/nodes schema of the native JSON. Anything outside these allowlists is
# rejected as a malformed receipt rather than tolerated.
ROOT_KEYS = {'schema', 'root_graph_nodes', 'child_graphs', 'counts',
             'node_count', 'unknown_nodes', 'unsupported_nodes', 'max_depth',
             'limits_exceeded', 'stream_context', 'collective', 'reasons',
             'complete', 'nodes'}
COUNT_KEYS = {'kernel', 'memcpy', 'memset', 'graph', 'other', 'unknown'}
STREAM_KEYS = {'green_query', 'green_matches', 'context_convert', 'sm_query',
               'sm_query_rc', 'sm_count'}
COLLECTIVE_KEYS = {'basis', 'confirmed', 'complete', 'unknown_nodes',
                   'candidates'}
CANDIDATE_KEYS = {'allreduce', 'allgather', 'reducescatter', 'broadcast',
                  'other_collective', 'none', 'unavailable'}
NODE_COMMON = {'id', 'graph', 'parent', 'depth', 'type', 'type_id',
               'supported', 'unknown', 'params', 'query_rc'}
NODE_EXTRA = {
    'kernel': {'context_known', 'context_matches', 'sm_query', 'sm_query_rc',
               'sm_count', 'grid', 'block', 'shared_bytes',
               'collective_candidate', 'collective_basis',
               'collective_confirmed'},
    'memcpy': {'direction', 'width_bytes', 'height', 'depth_count', 'bytes',
               'src_pitch', 'dst_pitch'},
    'memset': {'element_bytes', 'width_elements', 'height', 'bytes', 'pitch'},
    'graph': {'child'},
    'other': set(),
    'unknown': set(),
}
NODE_TYPES = set(NODE_EXTRA)
TYPE_IDS = {'kernel': 0, 'memcpy': 1, 'memset': 2, 'graph': 4}
MEMORY_KINDS = {'host', 'device', 'array', 'unified', 'unknown'}
SM_QUERIES = {'ok', 'unsupported', 'error', 'unknown'}
NATIVE_REASONS = {
    'limits_nodes', 'limits_depth', 'limits_output', 'unsupported_node',
    'node_query_error', 'node_type_error', 'kernel_context_unknown',
    'kernel_sm_unknown', 'kernel_name_unavailable', 'child_graph_error',
    'child_graph_cycle', 'child_graph_not_expanded', 'stream_sm_unknown',
}
STATUSES = frozenset(STATUS_BY_RC.values())
CAPTURE_CHECK_KEYS = {'bank', 'stream_label', 'context_label',
                      'vllm_matches_owner', 'torch_matches_owner', 'descriptor',
                      'basis'}

_MISSING = object()


class GraphAuditError(E3ContractError):
    """Short enum-like code; never carries a handle, name or payload."""

    def __init__(self, code):
        if type(code) is not str or not code:
            code = 'invalid'
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class NativeInspection:
    """Normalized result of one helper call. ``report`` is deep-frozen."""

    status: str
    native_rc: int
    complete: bool
    report: object = None
    required_bytes: int | None = None


def _is_digest(value):
    return (type(value) is str and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def helper_source_sha256():
    """SHA-256 of the packaged helper source (build provenance is owner-pinned)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), HELPER_SOURCE)
    try:
        with open(path, 'rb') as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError as exc:
        raise GraphAuditError('helper_source_missing') from exc


def _validate_limits(max_nodes, max_depth, max_bytes):
    if type(max_nodes) is not int or not 1 <= max_nodes <= MAX_NODES:
        raise GraphAuditError('bad_limits')
    if type(max_depth) is not int or not 0 <= max_depth <= MAX_DEPTH:
        raise GraphAuditError('bad_limits')
    if type(max_bytes) is not int or not MIN_BYTES <= max_bytes <= MAX_BYTES:
        raise GraphAuditError('bad_limits')


def _freeze(value):
    """Deep, immutable copy of a JSON document with strict scalar types."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if type(key) is not str:
                raise GraphAuditError('native_report_invalid')
            out[key] = _freeze(item)
        return types.MappingProxyType(out)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or type(value) in (bool, int, str):
        return value
    raise GraphAuditError('native_report_invalid')


def _no_duplicate_keys(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise GraphAuditError('malformed_json')
        out[key] = value
    return out


class AuditBinding:
    """ctypes binding of the pinned helper; exact library hash and own ABI only.

    Construction is the first point at which ctypes or the library is touched.
    No CUDA structure is declared in Python: only this helper's fixed exported
    signature is typed, and all CUDA structs live in the native translation
    unit compiled against the pinned CUDA 13 headers.
    """

    def __init__(self, path, expected_sha256, *, ctypes_module=None):
        if type(path) is not str or not path or not os.path.isfile(path):
            raise GraphAuditError('library_missing')
        if not _is_digest(expected_sha256):
            raise GraphAuditError('bad_expected_sha256')
        try:
            with open(path, 'rb') as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
        except OSError as exc:
            raise GraphAuditError('library_unreadable') from exc
        if digest != expected_sha256:
            raise GraphAuditError('library_sha_mismatch')
        if ctypes_module is None:
            import ctypes as ctypes_module
        c = ctypes_module
        try:
            lib = c.CDLL(path)
        except OSError as exc:
            raise GraphAuditError('library_load_failed') from exc
        try:
            abi_fn = lib.s5ga_abi
            inspect_fn = lib.s5ga_inspect
        except AttributeError as exc:
            raise GraphAuditError('symbol_missing') from exc
        U, I = c.c_uint64, c.c_int
        abi_fn.argtypes = []
        abi_fn.restype = I
        inspect_fn.argtypes = [U, U, U, c.c_uint32, c.c_uint32, U,
                               c.POINTER(c.c_char), U, c.POINTER(U)]
        inspect_fn.restype = I
        try:
            abi = int(abi_fn())
        except Exception as exc:  # a broken library is not a usable binding
            raise GraphAuditError('abi_missing') from exc
        if abi != ABI_VERSION:
            raise GraphAuditError('abi_mismatch')
        self.library_path = path
        self.library_sha256 = digest
        self.abi = abi
        self.helper_source_sha256 = helper_source_sha256()
        self._c = c
        self._lib = lib
        self._inspect_fn = inspect_fn

    def inspect(self, graph_handle, stream_handle,
                expected_green_context_handle, *, max_nodes, max_depth,
                max_bytes):
        """Bounded metadata for one raw graph; no path, replay or allocation."""
        for value in (graph_handle, stream_handle, expected_green_context_handle):
            if type(value) is not int or value <= 0:
                raise GraphAuditError('bad_handle')
        _validate_limits(max_nodes, max_depth, max_bytes)
        c = self._c
        buffer = c.create_string_buffer(max_bytes)
        out_len = c.c_uint64(0)
        rc = int(self._inspect_fn(
            c.c_uint64(graph_handle), c.c_uint64(stream_handle),
            c.c_uint64(expected_green_context_handle),
            c.c_uint32(max_nodes), c.c_uint32(max_depth), c.c_uint64(max_bytes),
            buffer, c.c_uint64(max_bytes), c.byref(out_len)))
        if rc != NATIVE_OK:
            required = int(out_len.value) if rc == NATIVE_TRUNCATED else None
            return NativeInspection(STATUS_BY_RC.get(rc, 'unknown_status'), rc,
                                    False, None, required)
        length = int(out_len.value)
        if length <= 0 or length > max_bytes:
            raise GraphAuditError('native_length_invalid')
        try:
            text = bytes(buffer.raw[:length]).decode('utf-8')
            document = json.loads(text, object_pairs_hook=_no_duplicate_keys)
        except (UnicodeDecodeError, ValueError) as exc:
            raise GraphAuditError('malformed_json') from exc
        if type(document) is not dict:
            raise GraphAuditError('malformed_json')
        complete = document.get('complete')
        if type(complete) is not bool:
            complete = False
        return NativeInspection('ok', 0, complete, _freeze(document), None)


# --- native report validation ----------------------------------------------

def _invalid():
    raise GraphAuditError('native_report_invalid')


def _int(value, minimum=0):
    if type(value) is not int or value < minimum:
        _invalid()
    return value


def _bool(value):
    if type(value) is not bool:
        _invalid()
    return value


def _text(value, allowed=None):
    if type(value) is not str or not value:
        _invalid()
    if allowed is not None and value not in allowed:
        _invalid()
    return value


def _vec3(value):
    if type(value) not in (tuple, list) or len(value) != 3:
        _invalid()
    return tuple(_int(item) for item in value)


def _validate_kernel(node):
    params = node['params']
    if params not in ('ok', 'error'):
        _invalid()
    ctx_known = _bool(node['context_known'])
    ctx_matches = _bool(node['context_matches'])
    if ctx_matches and not ctx_known:
        _invalid()
    sm_query = _text(node['sm_query'], SM_QUERIES)
    _int(node['sm_query_rc'])
    sm_count = node['sm_count']
    if sm_query == 'ok':
        _int(sm_count, 1)
    elif sm_count is not None:
        _invalid()
    if not ctx_known and (ctx_matches or sm_query != 'unknown'):
        _invalid()
    if ctx_known and sm_query == 'unknown':
        _invalid()
    if params == 'error':
        if ctx_known or ctx_matches or sm_query != 'unknown':
            _invalid()
        if any(node[key] is not None for key in ('grid', 'block', 'shared_bytes')):
            _invalid()
    else:
        _vec3(node['grid'])
        _vec3(node['block'])
        _int(node['shared_bytes'])
    _text(node['collective_candidate'], CANDIDATE_KEYS)
    if node['collective_basis'] != 'function_name_substring':
        _invalid()
    if node['collective_confirmed'] is not False:
        _invalid()
    expected_unknown = (params != 'ok' or not ctx_known or sm_query != 'ok')
    if node['unknown'] is not expected_unknown:
        _invalid()


def _validate_memcpy(node):
    params = node['params']
    if params not in ('ok', 'error'):
        _invalid()
    if params == 'error':
        if node['direction'] != 'unknown':
            _invalid()
        if any(node[key] is not None for key in
               ('width_bytes', 'height', 'depth_count', 'bytes',
                'src_pitch', 'dst_pitch')):
            _invalid()
        if node['unknown'] is not True:
            _invalid()
        return
    direction = _text(node['direction'])
    parts = direction.split('-to-')
    if len(parts) != 2 or any(part not in MEMORY_KINDS for part in parts):
        _invalid()
    width = _int(node['width_bytes'])
    height = _int(node['height'])
    depth = _int(node['depth_count'])
    bytes_value = node['bytes']
    pitch_src = _int(node['src_pitch'])
    pitch_dst = _int(node['dst_pitch'])
    del pitch_src, pitch_dst
    expected = width * height * depth
    if bytes_value is None:
        if node['unknown'] is not True:
            _invalid()
    elif _int(bytes_value) != expected or node['unknown'] is not False:
        _invalid()


def _validate_memset(node):
    params = node['params']
    if params not in ('ok', 'error'):
        _invalid()
    if params == 'error':
        if any(node[key] is not None for key in
               ('element_bytes', 'width_elements', 'height', 'bytes', 'pitch')):
            _invalid()
        if node['unknown'] is not True:
            _invalid()
        return
    element = _int(node['element_bytes'])
    width = _int(node['width_elements'])
    height = _int(node['height'])
    bytes_value = node['bytes']
    _int(node['pitch'])
    expected = width * height * element
    if bytes_value is None:
        if node['unknown'] is not True:
            _invalid()
    elif _int(bytes_value) != expected or node['unknown'] is not False:
        _invalid()


def _validate_graph_node(node, child_graphs):
    params = node['params']
    if params not in ('ok', 'error'):
        _invalid()
    child = node['child']
    if type(child) is not int or child < -1 or child > child_graphs:
        _invalid()
    if params == 'error' and child != -1:
        _invalid()
    if child == -1 and node['unknown'] is not True:
        _invalid()
    if child >= 0 and node['unknown'] is not False:
        _invalid()


def _validate_native_report(report, limits):
    """Validate the helper's bounded receipt and derive independent coverage."""
    if not isinstance(report, types.MappingProxyType) and type(report) is not dict:
        _invalid()
    keys = set(report)
    if keys != ROOT_KEYS:
        _invalid()
    if report['schema'] != NATIVE_SCHEMA:
        _invalid()
    counts = report['counts']
    if type(counts) not in (dict, types.MappingProxyType) or set(counts) != COUNT_KEYS:
        _invalid()
    counts = {key: _int(value) for key, value in counts.items()}
    node_count = _int(report['node_count'])
    root_nodes = _int(report['root_graph_nodes'])
    child_graphs = _int(report['child_graphs'])
    unknown_nodes = _int(report['unknown_nodes'])
    unsupported_nodes = _int(report['unsupported_nodes'])
    max_depth = _int(report['max_depth'])
    limits_exceeded = _bool(report['limits_exceeded'])
    complete = _bool(report['complete'])
    reasons = report['reasons']
    if type(reasons) not in (tuple, list):
        _invalid()
    reason_list = []
    for reason in reasons:
        _text(reason, NATIVE_REASONS)
        reason_list.append(reason)
    if len(set(reason_list)) != len(reason_list):
        _invalid()

    stream = report['stream_context']
    if type(stream) not in (dict, types.MappingProxyType) or set(stream) != STREAM_KEYS:
        _invalid()
    if (stream['green_query'] != 'ok' or stream['green_matches'] is not True
            or stream['context_convert'] != 'ok'):
        _invalid()
    sm_query = _text(stream['sm_query'], {'ok', 'unsupported', 'error'})
    sm_query_rc = _int(stream['sm_query_rc'])
    del sm_query_rc
    stream_sm_count = stream['sm_count']
    if sm_query == 'ok':
        _int(stream_sm_count, 1)
    elif stream_sm_count is not None:
        _invalid()

    collective = report['collective']
    if (type(collective) not in (dict, types.MappingProxyType)
            or set(collective) != COLLECTIVE_KEYS):
        _invalid()
    if collective['basis'] != 'function_name_substring':
        _invalid()
    if collective['confirmed'] is not False:
        _invalid()
    collective_complete = _bool(collective['complete'])
    collective_unknown = _int(collective['unknown_nodes'])
    candidates = collective['candidates']
    if (type(candidates) not in (dict, types.MappingProxyType)
            or set(candidates) != CANDIDATE_KEYS):
        _invalid()
    candidates = {key: _int(value) for key, value in candidates.items()}

    nodes = report['nodes']
    if type(nodes) not in (tuple, list):
        _invalid()
    normalized = []
    seen_ids = set()
    for node in nodes:
        if type(node) not in (dict, types.MappingProxyType):
            _invalid()
        node_type = node.get('type')
        if node_type not in NODE_TYPES:
            _invalid()
        expected_keys = NODE_COMMON | NODE_EXTRA[node_type]
        if set(node) != expected_keys:
            _invalid()
        node_id = _int(node['id'])
        if node_id in seen_ids:
            _invalid()
        seen_ids.add(node_id)
        graph_index = _int(node['graph'])
        parent = node['parent']
        if type(parent) is not int or parent < -1:
            _invalid()
        depth = _int(node['depth'])
        _int(node['type_id'], -1)
        supported = _bool(node['supported'])
        _bool(node['unknown'])
        _int(node['query_rc'])
        if node_type in ('kernel', 'memcpy', 'memset', 'graph'):
            if supported is not True or node['type_id'] != TYPE_IDS[node_type]:
                _invalid()
        else:
            if supported is not False or node['unknown'] is not True:
                _invalid()
            if node_type == 'unknown':
                if node['type_id'] != -1 or node['params'] != 'not_applicable':
                    _invalid()
            else:
                if node['type_id'] in TYPE_IDS.values() or node['params'] != 'not_applicable':
                    _invalid()
        if node_type == 'kernel':
            _validate_kernel(node)
        elif node_type == 'memcpy':
            _validate_memcpy(node)
        elif node_type == 'memset':
            _validate_memset(node)
        elif node_type == 'graph':
            _validate_graph_node(node, child_graphs)
        normalized.append(dict(node))

    by_id = {node['id']: node for node in normalized}
    for node in normalized:
        graph_index = node['graph']
        parent = node['parent']
        depth = node['depth']
        if graph_index == 0:
            if parent != -1 or depth != 0:
                _invalid()
        else:
            if parent < 0 or parent not in by_id or parent >= node['id']:
                _invalid()
            parent_node = by_id[parent]
            if parent_node['type'] != 'graph' or parent_node['child'] != graph_index:
                _invalid()
            if depth != parent_node['depth'] + 1:
                _invalid()
        if depth > limits['max_depth']:
            _invalid()

    recomputed_counts = {key: 0 for key in COUNT_KEYS}
    for node in normalized:
        recomputed_counts[node['type']] += 1
    if recomputed_counts != counts:
        _invalid()
    if node_count != len(normalized):
        _invalid()
    if root_nodes != sum(1 for node in normalized if node['graph'] == 0):
        _invalid()
    child_indices = [node['child'] for node in normalized
                     if node['type'] == 'graph' and node['child'] >= 0]
    if len(set(child_indices)) != len(child_indices):
        _invalid()
    if child_graphs != (max(child_indices) if child_indices else 0):
        _invalid()
    if set(child_indices) != set(range(1, child_graphs + 1)):
        _invalid()
    graphs_with_nodes = {node['graph'] for node in normalized}
    empty_child_graphs = tuple(sorted(set(range(1, child_graphs + 1))
                                      - graphs_with_nodes))
    if max_depth != max((node['depth'] for node in normalized), default=0):
        _invalid()

    evidence_unknown = sum(1 for node in normalized if node['unknown'])
    evidence_unsupported = sum(1 for node in normalized if not node['supported'])
    if evidence_unknown != unknown_nodes or evidence_unsupported != unsupported_nodes:
        _invalid()

    candidate_counts = {key: 0 for key in CANDIDATE_KEYS}
    evidence_collective_unknown = 0
    context = {'known': 0, 'matched': 0, 'mismatched': 0, 'unknown': 0,
               'mismatched_ids': [], 'unknown_ids': []}
    sm = {'queried': 0, 'unknown': 0, 'mismatched_ids': []}
    for node in normalized:
        if node['type'] == 'kernel':
            if node['params'] == 'ok':
                candidate_counts[node['collective_candidate']] += 1
                if node['collective_candidate'] == 'unavailable':
                    evidence_collective_unknown += 1
            else:
                candidate_counts['unavailable'] += 1
                evidence_collective_unknown += 1
            if node['context_known']:
                context['known'] += 1
                if node['context_matches']:
                    context['matched'] += 1
                else:
                    context['mismatched'] += 1
                    context['mismatched_ids'].append(node['id'])
            else:
                context['unknown'] += 1
                context['unknown_ids'].append(node['id'])
            if node['sm_query'] == 'ok':
                sm['queried'] += 1
            else:
                sm['unknown'] += 1
        elif node['type'] in ('other', 'unknown'):
            evidence_collective_unknown += 1
    if candidate_counts != candidates:
        _invalid()
    if evidence_collective_unknown != collective_unknown:
        _invalid()
    expected_collective_complete = (
        evidence_collective_unknown == 0 and not limits_exceeded
        and evidence_unsupported == 0)
    if collective_complete is not expected_collective_complete:
        _invalid()

    computed_complete = (not limits_exceeded and evidence_unsupported == 0
                         and evidence_unknown == 0 and sm_query == 'ok'
                         and collective_complete)
    if complete is not computed_complete:
        _invalid()
    if complete and reason_list:
        _invalid()
    if not complete and not reason_list:
        _invalid()

    # SM counts stay per node; the caller compares them with the split's
    # independently measured decode_sms.
    return {
        'counts': counts,
        'node_count': node_count,
        'root_graph_nodes': root_nodes,
        'child_graphs': child_graphs,
        'unknown_nodes': unknown_nodes,
        'unsupported_nodes': unsupported_nodes,
        'max_depth': max_depth,
        'limits_exceeded': limits_exceeded,
        'stream': {'sm_query': sm_query, 'sm_count': stream_sm_count},
        'collective': {'complete': collective_complete,
                       'unknown_nodes': collective_unknown,
                       'candidates': candidates},
        'complete': complete,
        'reasons': tuple(reason_list),
        'nodes': tuple(normalized),
        'empty_child_graphs': empty_child_graphs,
        'evidence': {'context': context, 'sm': sm},
    }


# --- audit_bank -------------------------------------------------------------

def _probe_state(probe):
    for name in ('bank', 'router', 'split', 'controller'):
        if getattr(probe, name, _MISSING) is _MISSING:
            raise GraphAuditError('probe_contract')
    in_flight = getattr(probe, 'in_flight', _MISSING)
    if type(in_flight) is not int or in_flight < 0:
        raise GraphAuditError('probe_contract')
    if in_flight:
        raise GraphAuditError('unresolved_work')
    if getattr(probe, 'quarantined', None) is not None:
        raise GraphAuditError('quarantined')
    if getattr(probe, 'shutdown_record', None) is not None:
        raise GraphAuditError('shutdown')
    admitting = getattr(probe.controller, 'admitting', _MISSING)
    if type(admitting) is not bool:
        raise GraphAuditError('probe_contract')
    if admitting is not True:
        raise GraphAuditError('transition_in_progress')
    return in_flight


def _binding_metadata(binding):
    digest = getattr(binding, 'library_sha256', _MISSING)
    if not _is_digest(digest):
        raise GraphAuditError('binding_contract')
    abi = getattr(binding, 'abi', _MISSING)
    if type(abi) is not int or abi != ABI_VERSION:
        raise GraphAuditError('binding_contract')
    if not callable(getattr(binding, 'inspect', None)):
        raise GraphAuditError('binding_contract')
    source = getattr(binding, 'helper_source_sha256', None)
    if source is not None:
        if not _is_digest(source):
            raise GraphAuditError('binding_contract')
        if source != helper_source_sha256():
            raise GraphAuditError('helper_source_mismatch')
    return {'library_sha256': digest, 'abi': abi, 'source_sha256': source,
            'source_basis': 'packaged source; build provenance is owner-pinned'}


def _label(descriptor):
    from graph_router import descriptor_name

    try:
        return descriptor_name(descriptor)
    except Exception as exc:
        raise GraphAuditError('capture_descriptor_invalid') from exc


def _equals(left, right):
    try:
        value = left is right or left == right
    except Exception:
        return False
    # Strict bool contract: an equality that returns 1/truthy is not evidence.
    return type(value) is bool and value


def _base_result(meta, binding_meta, limits, fail_reasons, inconclusive_reasons,
                 graphs, probe_observation):
    fail_reasons = sorted(set(fail_reasons))
    inconclusive_reasons = sorted(set(inconclusive_reasons))
    if fail_reasons:
        verdict = FAIL
    elif inconclusive_reasons or not graphs:
        verdict = INCONCLUSIVE
    else:
        verdict = PASS
    candidates = {key: 0 for key in CANDIDATE_KEYS}
    collective_unknown = 0
    collective_complete = bool(graphs)
    for graph in graphs:
        for key, value in graph['collective']['candidates'].items():
            candidates[key] += value
        collective_unknown += graph['collective']['unknown_nodes']
        collective_complete = collective_complete and graph['collective']['complete']
    return {
        'schema': SCHEMA,
        'verdict': verdict,
        'reasons': sorted(set(fail_reasons) | set(inconclusive_reasons)),
        'fail_reasons': fail_reasons,
        'inconclusive_reasons': inconclusive_reasons,
        'helper': binding_meta,
        'identity': meta,
        'scope': {
            'worker_incarnation': 'caller-supplied',
            'engine_incarnation': 'caller-supplied',
            'scheduler_pause': 'caller-proven',
            'replay_absent': 'not-proven-by-this-function',
            'worker_and_pause_note': 'audit_bank cannot certify scheduler pause or an engine incarnation',
        },
        'limits': dict(limits),
        'graphs': graphs,
        'collective_coverage': {
            'basis': 'function_name_substring',
            'confirmed': False,
            'complete': collective_complete,
            'candidates': candidates,
            'unknown_nodes': collective_unknown,
            'claim': 'candidate classification only; not confirmed collective proof',
        },
        'probe_observation': probe_observation,
    }


def _graph_record(label, graph_index, status, native_rc, complete, report=None,
                  reasons=(), required_bytes=None):
    record = {
        'descriptor': label,
        'graph_index': graph_index,
        'native_status': status,
        'native_rc': native_rc,
        'required_bytes': required_bytes,
        'complete': bool(complete),
        'node_count': 0,
        'counts': {key: 0 for key in COUNT_KEYS},
        'root_graph_nodes': 0,
        'child_graphs': 0,
        'max_depth': 0,
        'limits_exceeded': False,
        'context': {'known': 0, 'matched': 0, 'mismatched': 0, 'unknown': 0,
                    'mismatched_node_ids': [], 'unknown_node_ids': []},
        'sm': {'queried': 0, 'unknown': 0, 'mismatched_node_ids': []},
        'collective': {'basis': 'function_name_substring', 'confirmed': False,
                       'complete': False, 'candidates':
                       {key: 0 for key in CANDIDATE_KEYS}, 'unknown_nodes': 0},
        'reasons': sorted(set(reasons)),
    }
    if report is not None:
        evidence = report['evidence']
        record.update({
            'node_count': report['node_count'],
            'counts': dict(report['counts']),
            'root_graph_nodes': report['root_graph_nodes'],
            'child_graphs': report['child_graphs'],
            'max_depth': report['max_depth'],
            'limits_exceeded': report['limits_exceeded'],
            'context': {
                'known': evidence['context']['known'],
                'matched': evidence['context']['matched'],
                'mismatched': evidence['context']['mismatched'],
                'unknown': evidence['context']['unknown'],
                'mismatched_node_ids': list(evidence['context']['mismatched_ids']),
                'unknown_node_ids': list(evidence['context']['unknown_ids']),
            },
            'sm': {
                'queried': evidence['sm']['queried'],
                'unknown': evidence['sm']['unknown'],
                'mismatched_node_ids': list(evidence['sm']['mismatched_ids']),
            },
            'collective': {'basis': 'function_name_substring', 'confirmed': False,
                           'complete': report['collective']['complete'],
                           'candidates': dict(report['collective']['candidates']),
                           'unknown_nodes': report['collective']['unknown_nodes']},
        })
    return record


def _native_reason(reason):
    if reason.startswith('limits_') or reason == 'child_graph_not_expanded':
        return 'limits_exceeded'
    if reason in ('kernel_context_unknown', 'kernel_sm_unknown'):
        return 'node_placement_unknown'
    if reason in ('child_graph_error', 'child_graph_cycle'):
        return 'child_graph_unknown'
    if reason in ('kernel_name_unavailable',):
        return 'collective_coverage_unknown'
    if reason in ('node_query_error', 'node_type_error'):
        return 'node_query_unknown'
    if reason == 'unsupported_node':
        return 'unsupported_node'
    if reason == 'stream_sm_unknown':
        return 'stream_sm_unknown'
    return 'native_incomplete'


def audit_bank(probe, binding, *, max_nodes=DEFAULT_MAX_NODES,
               max_depth=DEFAULT_MAX_DEPTH, max_bytes=DEFAULT_MAX_BYTES,
               max_graphs=MAX_GRAPHS):
    """Audit exactly ``probe.bank.manager.graphs`` without replaying anything.

    Raises :class:`GraphAuditError` for a refused/unresolved state or an
    invalid binding/limit before doing any native work. Missing evidence is
    returned as an INCONCLUSIVE result; a known context/SM mismatch is FAIL and
    wins over any unrelated unknown.
    """
    _validate_limits(max_nodes, max_depth, max_bytes)
    if type(max_graphs) is not int or not 1 <= max_graphs <= MAX_GRAPHS:
        raise GraphAuditError('bad_limits')
    binding_meta = _binding_metadata(binding)
    _probe_state(probe)
    bank = probe.bank
    router = probe.router
    split = probe.split
    if bank is None or router is None or split is None:
        raise GraphAuditError('probe_contract')
    limits = {'max_nodes': max_nodes, 'max_depth': max_depth,
              'max_bytes': max_bytes, 'max_graphs': max_graphs}

    bank_label = getattr(router, 'name', None)
    if type(bank_label) is not str or not bank_label:
        raise GraphAuditError('probe_contract')
    storage_epoch = getattr(probe, 'storage_epoch', None)
    route_generation = getattr(probe, 'route_generation', None)
    router_epoch = getattr(router, 'storage_epoch', None)
    router_generation = getattr(router, 'route_generation', None)
    requested_sms = getattr(split, 'requested_decode_sms', None)
    measured_sms = getattr(split, 'decode_sms', None)
    stream_handle = getattr(split, 'decode_stream_token', None)
    green_handle = getattr(split, 'decode_context_id', None)

    meta = {
        'bank': bank_label,
        'route': getattr(probe, 'route', None),
        'storage_epoch': storage_epoch if type(storage_epoch) is int else None,
        'route_generation': route_generation if type(route_generation) is int else None,
        'descriptor_inventory': [],
        'graph_count': 0,
        'capture_labels': {'bank': CAPTURE_BANK_LABEL,
                           'stream': CAPTURE_STREAM_LABEL,
                           'context': CAPTURE_CONTEXT_LABEL},
        'split': {
            'requested_decode_sms': requested_sms if type(requested_sms) is int else None,
            'decode_sms': measured_sms if type(measured_sms) is int else None,
        },
    }
    probe_observation = {
        'in_flight': getattr(probe, 'in_flight', None),
        'interval': getattr(probe, 'interval', None),
        'quarantined': getattr(probe, 'quarantined', None),
        'shutdown_recorded': getattr(probe, 'shutdown_record', None) is not None,
    }

    fail, inconclusive = [], []

    def finish(graphs=()):
        return _base_result(meta, binding_meta, limits, fail, inconclusive,
                            list(graphs), probe_observation)

    if (type(storage_epoch) is not int or type(router_epoch) is not int):
        inconclusive.append('storage_epoch_missing')
    elif storage_epoch != router_epoch:
        fail.append('storage_epoch_mismatch')
    if (type(route_generation) is not int or type(router_generation) is not int):
        inconclusive.append('route_generation_missing')
    elif route_generation != router_generation:
        fail.append('route_generation_mismatch')
    if (type(stream_handle) is not int or stream_handle <= 0
            or type(green_handle) is not int or green_handle <= 0):
        inconclusive.append('split_identity_missing')
    if type(measured_sms) is not int or measured_sms < 1:
        inconclusive.append('measured_sms_missing')
    if type(requested_sms) is not int or requested_sms < 1:
        inconclusive.append('requested_sms_missing')
    if getattr(router, 'state', None) != 'ready':
        inconclusive.append('router_not_ready')
    if getattr(probe, 'interval', None) == 'invalidated':
        inconclusive.append('interval_invalidated')

    manager = getattr(bank, 'manager', None)
    if manager is None:
        inconclusive.append('manager_missing')
        return finish()
    graphs_obj = getattr(manager, 'graphs', None)
    if type(graphs_obj) is not dict:
        inconclusive.append('graphs_missing')
        return finish()
    pairs = getattr(bank, 'pairs', None)
    if type(pairs) not in (tuple, list) or not pairs:
        inconclusive.append('descriptor_inventory_missing')
        return finish()
    if getattr(bank, 'captured', None) is not True:
        inconclusive.append('bank_not_captured')

    ordered = []
    local_labels = []
    seen_labels = set()
    for pair in pairs:
        if type(pair) not in (tuple, list) or len(pair) != 2:
            inconclusive.append('descriptor_inventory_missing')
            continue
        vllm_desc, local = pair
        from graph_router import ExecutionDescriptor

        if type(local) is not ExecutionDescriptor:
            inconclusive.append('descriptor_inventory_missing')
            continue
        try:
            label = _label(local)
        except GraphAuditError:
            inconclusive.append('capture_descriptor_invalid')
            continue
        if label in seen_labels:
            fail.append('descriptor_duplicate')
        seen_labels.add(label)
        local_labels.append(label)
        ordered.append((vllm_desc, local, label))
    meta['descriptor_inventory'] = sorted(local_labels)
    if not ordered:
        inconclusive.append('descriptor_inventory_missing')
        return finish()
    local_by_label = {entry[2]: entry[1] for entry in ordered}
    if len(graphs_obj) > max_graphs:
        inconclusive.append('graph_limit')
        return finish()
    meta['graph_count'] = len(graphs_obj)

    # --- capture observations ------------------------------------------------
    keep = getattr(manager, '_flash_s5_keep_graphs', _MISSING)
    if keep is _MISSING:
        inconclusive.append('keep_graph_missing')
    elif type(keep) is not bool:
        inconclusive.append('keep_graph_malformed')
    elif keep is not True:
        inconclusive.append('keep_graph_disabled')
    owner = getattr(manager, '_flash_s5_capture_owner', _MISSING)
    if owner is _MISSING:
        inconclusive.append('capture_owner_missing')
    elif owner is not split:
        fail.append('capture_owner_mismatch')
    checks = getattr(manager, '_flash_s5_capture_stream_checks', _MISSING)
    if checks is _MISSING:
        inconclusive.append('capture_checks_missing')
    elif type(checks) is not dict:
        inconclusive.append('capture_checks_malformed')
    else:
        from graph_router import ExecutionDescriptor

        expected_labels = set(local_labels)
        for key in checks:
            if type(key) is not str or key not in expected_labels:
                fail.append('capture_check_extra')
                break
        for label in local_labels:
            check = checks.get(label)
            if check is None:
                inconclusive.append('capture_check_missing')
                continue
            if type(check) is not dict or set(check) != CAPTURE_CHECK_KEYS:
                inconclusive.append('capture_check_malformed')
                continue
            if (check['bank'] != CAPTURE_BANK_LABEL
                    or check['stream_label'] != CAPTURE_STREAM_LABEL
                    or check['context_label'] != CAPTURE_CONTEXT_LABEL
                    or type(check['basis']) is not str or not check['basis']):
                inconclusive.append('capture_check_malformed')
                continue
            if type(check['descriptor']) is not ExecutionDescriptor:
                inconclusive.append('capture_check_malformed')
                continue
            if not _equals(check['descriptor'], local_by_label[label]):
                fail.append('capture_descriptor_mismatch')
            for flag in ('vllm_matches_owner', 'torch_matches_owner'):
                if type(check[flag]) is not bool:
                    inconclusive.append('capture_check_malformed')
                elif check[flag] is not True:
                    fail.append('capture_stream_mismatch')

    # --- exact graph inventory ----------------------------------------------
    items = list(graphs_obj.items())
    matched = {}
    for key, graph in items:
        hits = [entry for entry in ordered if _equals(key, entry[0])]
        if not hits:
            fail.append('graph_extra')
            continue
        if len(hits) > 1:
            fail.append('descriptor_duplicate')
            continue
        label = hits[0][2]
        if label in matched:
            fail.append('descriptor_duplicate')
            continue
        matched[label] = (hits[0][1], graph, label)
    for _, _, label in ordered:
        if label not in matched and 'descriptor_duplicate' not in fail:
            inconclusive.append('graph_missing')

    router_descriptors = getattr(router, 'descriptors', None)
    if not isinstance(router_descriptors, (tuple, list)):
        inconclusive.append('router_inventory_missing')
    else:
        router_locals = [item for item in router_descriptors]
        pair_locals = [entry[1] for entry in ordered]

        def _count(needle, values):
            return sum(1 for value in values if _equals(needle, value))

        extra = any(_count(item, router_locals) > _count(item, pair_locals)
                    for item in router_locals)
        missing = any(_count(item, pair_locals) > _count(item, router_locals)
                      for item in pair_locals)
        if extra:
            fail.append('descriptor_inventory_extra')
        if missing:
            inconclusive.append('descriptor_inventory_missing')
        if (not extra and not missing
                and any(_count(item, router_locals) != 1 for item in router_locals)):
            inconclusive.append('router_inventory_malformed')
        if (not extra and not missing
                and tuple(router_locals) != tuple(pair_locals)):
            inconclusive.append('descriptor_inventory_order')
    handles = getattr(router, 'handles', None)
    if type(handles) is dict:
        handle_keys = list(handles)
        for item in handle_keys:
            if not any(_equals(item, pair_local) for pair_local in
                       [entry[1] for entry in ordered]):
                fail.append('router_handle_extra')
                break
        if any(not any(_equals(pair_local, key) for key in handle_keys)
               for pair_local in [entry[1] for entry in ordered]):
            inconclusive.append('router_handle_missing')

    if (type(stream_handle) is not int or stream_handle <= 0
            or type(green_handle) is not int or green_handle <= 0
            or type(measured_sms) is not int or measured_sms < 1):
        return finish()

    current_graphs = getattr(manager, 'graphs', None)
    if type(current_graphs) is not dict or current_graphs is not graphs_obj:
        inconclusive.append('concurrent_mutation')
        return finish()
    if len(current_graphs) != len(items):
        inconclusive.append('concurrent_mutation')
        return finish()

    # --- one native inspection per actual retained graph --------------------
    graph_records = []
    for index, (_, local, label) in enumerate(ordered):
        entry = matched.get(label)
        if entry is None:
            continue
        _, graph_obj, _ = entry
        raw = getattr(graph_obj, 'raw_cuda_graph', None)
        if not callable(raw):
            inconclusive.append('graph_api_unsupported')
            graph_records.append(_graph_record(label, index, 'unsupported', 0,
                                               False, reasons=('graph_api_unsupported',)))
            continue
        try:
            handle = raw()
        except Exception:
            inconclusive.append('graph_api_failed')
            graph_records.append(_graph_record(label, index, 'unsupported', 0,
                                               False, reasons=('graph_api_failed',)))
            continue
        if type(handle) is not int or handle <= 0:
            inconclusive.append('graph_api_failed')
            graph_records.append(_graph_record(label, index, 'unsupported', 0,
                                               False, reasons=('graph_api_failed',)))
            continue
        try:
            inspection = binding.inspect(handle, stream_handle, green_handle,
                                         max_nodes=max_nodes, max_depth=max_depth,
                                         max_bytes=max_bytes)
        except GraphAuditError as exc:
            inconclusive.append('native_' + exc.code)
            graph_records.append(_graph_record(label, index, 'invalid', 0, False,
                                               reasons=('native_' + exc.code,)))
            continue
        except Exception:
            inconclusive.append('native_call_failed')
            graph_records.append(_graph_record(label, index, 'invalid', 0, False,
                                               reasons=('native_call_failed',)))
            continue
        if not isinstance(inspection, NativeInspection):
            inconclusive.append('native_contract')
            graph_records.append(_graph_record(label, index, 'invalid', 0, False,
                                               reasons=('native_contract',)))
            continue
        if (type(inspection.status) is not str
                or inspection.status not in STATUSES
                or type(inspection.native_rc) is not int
                or type(inspection.complete) is not bool
                or (inspection.required_bytes is not None
                    and (type(inspection.required_bytes) is not int
                         or inspection.required_bytes < 0))):
            inconclusive.append('native_contract')
            graph_records.append(_graph_record(label, index, 'invalid', 0, False,
                                               reasons=('native_contract',)))
            continue
        if inspection.status != 'ok':
            if inspection.status == 'green_context_mismatch':
                fail.append('green_context_mismatch')
            else:
                inconclusive.append('native_' + inspection.status)
            graph_records.append(_graph_record(
                label, index, inspection.status, inspection.native_rc, False,
                reasons=('native_' + inspection.status,),
                required_bytes=inspection.required_bytes))
            continue
        try:
            report = _validate_native_report(inspection.report, limits)
        except GraphAuditError as exc:
            inconclusive.append('native_' + exc.code)
            graph_records.append(_graph_record(label, index, 'invalid',
                                               inspection.native_rc, False,
                                               reasons=('native_' + exc.code,)))
            continue
        if report['complete'] != inspection.complete:
            inconclusive.append('native_contract')
            graph_records.append(_graph_record(label, index, 'invalid',
                                               inspection.native_rc, False,
                                               reasons=('native_contract',)))
            continue
        mapped = [_native_reason(reason) for reason in report['reasons']]
        for reason in mapped:
            if reason == 'unsupported_node':
                inconclusive.append('unsupported_node')
            else:
                inconclusive.append(reason)
        record = _graph_record(label, index, 'ok', inspection.native_rc,
                               report['complete'], report=report,
                               reasons=mapped)
        stream_report = report['stream']
        if stream_report['sm_query'] == 'ok':
            if stream_report['sm_count'] != measured_sms:
                fail.append('stream_sm_mismatch')
                record['reasons'] = sorted(set(record['reasons'])
                                           | {'stream_sm_mismatch'})
        else:
            inconclusive.append('stream_sm_unknown')
        if report['root_graph_nodes'] == 0:
            # A retained split graph with no top-level node is not evidence
            # that the decode graph was captured; never let it PASS.
            inconclusive.append('empty_root_graph')
        if report['empty_child_graphs']:
            inconclusive.append('empty_child_graph')
        for node in report['nodes']:
            if node['type'] != 'kernel':
                continue
            node_id = node['id']
            if node['context_known'] and not node['context_matches']:
                fail.append('node_context_mismatch')
                if node_id not in record['context']['mismatched_node_ids']:
                    record['context']['mismatched_node_ids'].append(node_id)
            if node['sm_query'] == 'ok' and node['sm_count'] != measured_sms:
                fail.append('node_sm_mismatch')
                record['sm']['mismatched_node_ids'].append(node_id)
            if node['sm_query'] != 'ok':
                inconclusive.append('node_sm_unknown')
        if not report['complete']:
            inconclusive.append('native_incomplete')
        graph_records.append(record)

    # A graph missing its retained object is missing proof, not a PASS.
    if len(graph_records) != len(ordered):
        inconclusive.append('graph_incomplete')
    return finish(graph_records)


__all__ = ['ABI_VERSION', 'AuditBinding', 'CAPTURE_BANK_LABEL',
           'CAPTURE_CONTEXT_LABEL', 'CAPTURE_STREAM_LABEL', 'DEFAULT_MAX_BYTES',
           'DEFAULT_MAX_DEPTH', 'DEFAULT_MAX_NODES', 'FAIL',
           'GraphAuditError', 'HELPER_SOURCE', 'INCONCLUSIVE',
           'INSPECT_SIGNATURE', 'MAX_BYTES', 'MAX_DEPTH', 'MAX_GRAPHS',
           'MAX_NODES', 'MIN_BYTES', 'NATIVE_SCHEMA', 'NativeInspection', 'PASS',
           'SCHEMA', 'STATUS_BY_RC', 'audit_bank', 'helper_source_sha256']
