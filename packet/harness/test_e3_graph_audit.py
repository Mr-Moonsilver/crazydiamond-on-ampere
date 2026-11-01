"""CPU-only tests for the q17b raw retained-graph audit.

No torch, no vLLM, no GPU, no engine. Fakes implement only the seam attributes
``graph_audit`` documents. The native helper is compiled here against a CPU
stub ``cuda.h`` and stub driver, which exercises the walker's control flow,
status mapping and JSON schema -- it is NOT CUDA ABI/placement validation and
does not establish any GPU behaviour.

The fake-binding cases inject receipts. The stub cases exercise the actual
``native/flash_s5_graph_audit.cu`` walker through :class:`AuditBinding`.
"""
import ctypes
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))

from graph_audit import (  # noqa: E402
    ABI_VERSION,
    AuditBinding,
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_NODES,
    FAIL,
    GraphAuditError,
    INCONCLUSIVE,
    NATIVE_SCHEMA,
    NativeInspection,
    PASS,
    SCHEMA,
    audit_bank,
    helper_source_sha256,
)
from graph_router import ExecutionDescriptor, descriptor_name  # noqa: E402

SMS = 34
REQUESTED = 44
STREAM = 111_111_111_111
GREEN = 222_222_222_222
DIGEST0 = '0' * 64


def desc(tokens):
    return ExecutionDescriptor('full', tokens, tokens, 1, None, 0, 1)


# --- native receipt fixture -------------------------------------------------

def kernel(node_id, *, graph=0, parent=-1, depth=0, ctx_known=True,
           ctx_matches=True, sm_query='ok', sm_count=SMS, candidate='allreduce',
           params='ok', grid=(4, 1, 1), block=(128, 1, 1), shared=0,
           query_rc=0):
    if params == 'error':
        ctx_known, ctx_matches = False, False
        sm_query, sm_count = 'unknown', None
        grid = block = shared = None
        candidate = 'unavailable'
    if not ctx_known:
        ctx_matches, sm_query, sm_count = False, 'unknown', None
    elif sm_query == 'unknown':
        raise AssertionError('a known context must report an SM query result')
    unknown = params != 'ok' or not ctx_known or sm_query != 'ok'
    return {
        'id': node_id, 'graph': graph, 'parent': parent, 'depth': depth,
        'type': 'kernel', 'type_id': 0, 'supported': True, 'unknown': unknown,
        'params': params, 'query_rc': query_rc if params == 'ok' else 400,
        'context_basis': 'kernel_ctx' if ctx_known else 'unavailable',
        'context_known': ctx_known, 'context_matches': ctx_matches,
        'sm_query': sm_query, 'sm_query_rc': 0 if sm_query in ('ok', 'unknown') else 801, 'sm_count': sm_count,
        'grid': None if grid is None else list(grid),
        'block': None if block is None else list(block),
        'shared_bytes': shared,
        'collective_candidate': candidate,
        'collective_basis': 'function_name_substring',
        'collective_confirmed': False,
    }


def memcpy(node_id, *, graph=0, parent=-1, depth=0, params='ok',
           width=16, height=2, depth_count=3, src='device', dst='device'):
    if params == 'error':
        return {
            'id': node_id, 'graph': graph, 'parent': parent, 'depth': depth,
            'context_basis': 'not_exposed', 'context_known': False,
            'type': 'memcpy', 'type_id': 1, 'supported': True, 'unknown': True,
            'params': 'error', 'query_rc': 400,
            'direction': 'unknown', 'width_bytes': None, 'height': None,
            'depth_count': None, 'bytes': None, 'src_pitch': None,
            'dst_pitch': None,
        }
    return {
        'id': node_id, 'graph': graph, 'parent': parent, 'depth': depth,
        'context_basis': 'not_exposed', 'context_known': False,
            'type': 'memcpy', 'type_id': 1, 'supported': True, 'unknown': True,
        'params': 'ok', 'query_rc': 0,
        'direction': f'{src}-to-{dst}', 'width_bytes': width,
        'height': height, 'depth_count': depth_count,
        'bytes': width * height * depth_count,
        'src_pitch': 0, 'dst_pitch': 0,
    }


def memset(node_id, *, graph=0, parent=-1, depth=0, params='ok',
           element=4, width=4, height=2):
    if params == 'error':
        return {
            'id': node_id, 'graph': graph, 'parent': parent, 'depth': depth,
            'context_basis': 'not_exposed', 'context_known': False,
            'type': 'memset', 'type_id': 2, 'supported': True, 'unknown': True,
            'params': 'error', 'query_rc': 400, 'element_bytes': None,
            'width_elements': None, 'height': None, 'bytes': None, 'pitch': None,
        }
    return {
        'id': node_id, 'graph': graph, 'parent': parent, 'depth': depth,
        'context_basis': 'not_exposed', 'context_known': False,
            'type': 'memset', 'type_id': 2, 'supported': True, 'unknown': True,
        'params': 'ok', 'query_rc': 0, 'element_bytes': element,
        'width_elements': width, 'height': height,
        'bytes': element * width * height, 'pitch': 0,
    }


def graph_node(node_id, child, *, graph=0, parent=-1, depth=0, params='ok'):
    if params == 'error':
        child = -1
        unknown = True
        query_rc = 400
    else:
        unknown = False
        query_rc = 0
    return {
        'id': node_id, 'graph': graph, 'parent': parent, 'depth': depth,
        'type': 'graph', 'type_id': 4, 'supported': True, 'unknown': unknown,
        'params': params if params in ('ok', 'error') else 'error',
        'query_rc': query_rc, 'child': child,
    }


def other_node(node_id, *, graph=0, parent=-1, depth=0, type_id=5):
    return {
        'id': node_id, 'graph': graph, 'parent': parent, 'depth': depth,
        'type': 'other', 'type_id': type_id, 'supported': False,
        'unknown': True, 'params': 'not_applicable', 'query_rc': 0,
    }


def build_report(nodes, *, stream_sm_query='ok', stream_sm_count=SMS,
                 limits=False, extra_reasons=()):
    """Construct a native-schema report the audit validator must accept."""
    counts = {key: 0 for key in ('kernel', 'memcpy', 'memset', 'graph',
                                 'other', 'unknown')}
    candidates = {key: 0 for key in ('allreduce', 'allgather', 'reducescatter',
                                     'broadcast', 'other_collective', 'none',
                                     'unavailable')}
    unsupported = unknown_nodes = collective_unknown = 0
    child_indices = []
    for node in nodes:
        counts[node['type']] += 1
        if not node['supported']:
            unsupported += 1
        if node['unknown']:
            unknown_nodes += 1
        if node['type'] == 'kernel':
            if node['params'] == 'ok':
                candidates[node['collective_candidate']] += 1
                if node['collective_candidate'] == 'unavailable':
                    collective_unknown += 1
            else:
                candidates['unavailable'] += 1
                collective_unknown += 1
        elif node['type'] in ('other', 'unknown'):
            collective_unknown += 1
        elif node['type'] == 'graph' and node['child'] >= 0:
            child_indices.append(node['child'])
    child_graphs = max(child_indices) if child_indices else 0
    collective_complete = (collective_unknown == 0 and not limits
                           and unsupported == 0)
    complete = (not limits and unsupported == 0 and unknown_nodes == 0
                and stream_sm_query == 'ok' and collective_complete)

    reasons = set(extra_reasons)
    if limits:
        reasons.add('limits_nodes')
    if unsupported:
        reasons.add('unsupported_node')
    for node in nodes:
        if node['type'] == 'kernel':
            if node['params'] != 'ok':
                reasons.add('node_query_error')
            elif not node['context_known']:
                reasons.add('kernel_context_unknown')
            elif node['sm_query'] != 'ok':
                reasons.add('kernel_sm_unknown')
            if node['collective_candidate'] == 'unavailable':
                reasons.add('kernel_name_unavailable')
        elif node['type'] in ('memcpy', 'memset') and node['unknown']:
            reasons.add('memory_context_unavailable')
        elif node['type'] == 'graph' and node['unknown']:
            reasons.add('child_graph_error')
    if stream_sm_query != 'ok':
        reasons.add('stream_sm_unknown')
    if not complete and not reasons:
        reasons.add('node_query_error')
    if complete and reasons:
        raise AssertionError('a complete fixture cannot carry reasons')

    return {
        'schema': NATIVE_SCHEMA,
        'root_graph_nodes': sum(1 for node in nodes if node['graph'] == 0),
        'child_graphs': child_graphs,
        'counts': counts,
        'node_count': len(nodes),
        'unknown_nodes': unknown_nodes,
        'unsupported_nodes': unsupported,
        'max_depth': max((node['depth'] for node in nodes), default=0),
        'limits_exceeded': limits,
        'stream_context': {
            'green_query': 'ok', 'green_matches': True,
            'context_convert': 'ok', 'sm_query': stream_sm_query,
            'sm_query_rc': 0 if stream_sm_query == 'ok' else 801, 'sm_count': stream_sm_count,
        },
        'collective': {
            'basis': 'function_name_substring', 'confirmed': False,
            'complete': collective_complete,
            'unknown_nodes': collective_unknown, 'candidates': candidates,
        },
        'reasons': sorted(reasons),
        'complete': complete,
        'nodes': list(nodes),
    }


# --- seam fakes -------------------------------------------------------------

class FakeGraph:
    def __init__(self, handle):
        self.handle = handle

    def raw_cuda_graph(self):
        return self.handle


class VllmDesc:
    """Hashable stand-in for the pinned descriptor used as the manager key."""

    def __init__(self, num_tokens):
        self.num_tokens = num_tokens
        self.cg_mode = 'FULL'
        self.num_reqs = num_tokens
        self.uniform_token_count = 1
        self.max_query_len = None
        self.num_active_loras = 0
        self.num_ubatches = 1


class FakeBinding:
    library_sha256 = DIGEST0
    helper_source_sha256 = helper_source_sha256()
    abi = ABI_VERSION

    def __init__(self, reports):
        self.reports = reports
        self.calls = []

    def inspect(self, graph, stream, green, *, max_nodes, max_depth, max_bytes):
        self.calls.append({'graph': graph, 'stream': stream, 'green': green,
                           'max_nodes': max_nodes, 'max_depth': max_depth,
                           'max_bytes': max_bytes})
        result = self.reports[graph]
        if isinstance(result, BaseException):
            raise result
        return result


def scene(descriptors=None, handles=None, report_factory=None):
    descriptors = descriptors if descriptors is not None else [desc(8), desc(16)]
    handles = handles if handles is not None else [10 ** 12 + i
                                                   for i in range(len(descriptors))]
    split = types.SimpleNamespace(
        decode_stream_token=STREAM, decode_context_id=GREEN,
        decode_sms=SMS, requested_decode_sms=REQUESTED)
    pairs, graphs, checks, router_handles = [], {}, {}, {}
    for index, local in enumerate(descriptors):
        vllm = VllmDesc(local.num_tokens)
        pairs.append((vllm, local))
        graphs[vllm] = FakeGraph(handles[index])
        router_handles[local] = types.SimpleNamespace(descriptor=vllm, local=local, bank=None, destroyed=False)
        checks[descriptor_name(local)] = {
            'bank': 'decode-split',
            'stream_label': 'decode-split-stream',
            'context_label': 'decode-split-context',
            'vllm_matches_owner': True,
            'torch_matches_owner': True,
            'descriptor': local,
            'basis': 'test fixture',
        }
    manager = types.SimpleNamespace(
        graphs=graphs, _flash_s5_keep_graphs=True,
        _flash_s5_capture_owner=split,
        _flash_s5_capture_stream_checks=checks)
    bank = types.SimpleNamespace(manager=manager, pairs=tuple(pairs), captured=True, handles=dict(router_handles))
    for owned in router_handles.values():
        owned.bank = bank
    router = types.SimpleNamespace(
        name='decode-split', descriptors=tuple(descriptors),
        handles=router_handles, storage_epoch=1, route_generation=3,
        state='ready')
    controller = types.SimpleNamespace(admitting=True)
    probe = types.SimpleNamespace(
        bank=bank, router=router, split=split, controller=controller,
        in_flight=0, interval='active', storage_epoch=1, route_generation=3,
        quarantined=None, shutdown_record=None, route='split')
    if report_factory is None:
        report_factory = lambda: build_report([kernel(0)])  # noqa: E731
    reports = {handle: NativeInspection('ok', 0, True, report_factory())
               for handle in handles}
    binding = FakeBinding(reports)
    return probe, binding, manager, graphs, checks, router_handles, list(handles)


def thaw(value):
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [thaw(item) for item in value]
    return value


class ImportTests(unittest.TestCase):
    def test_import_needs_no_torch_vllm_or_ctypes(self):
        code = (
            'import sys\n'
            f'sys.path.insert(0, {str(ROOT / "patches/s5_engine")!r})\n'
            'import graph_audit\n'
            "assert 'torch' not in sys.modules\n"
            "assert 'vllm' not in sys.modules\n"
            "assert 'ctypes' not in sys.modules\n"
            "print(graph_audit.ABI_VERSION)\n"
        )
        seen = subprocess.run([sys.executable, '-c', code],
                              capture_output=True, text=True)
        self.assertEqual(seen.returncode, 0, seen.stderr)
        self.assertEqual(seen.stdout.strip(), str(ABI_VERSION))


class AuditBankTests(unittest.TestCase):
    def test_pass_covers_every_retained_graph_with_expected_owner(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        result = audit_bank(probe, binding)

        self.assertEqual(result['verdict'], PASS)
        self.assertEqual(result['reasons'], [])
        self.assertEqual(result['schema'], SCHEMA)
        self.assertEqual(len(result['graphs']), len(raw))
        self.assertEqual([g['descriptor'] for g in result['graphs']],
                         [descriptor_name(d) for d in probe.router.descriptors])
        self.assertTrue(all(g['complete'] for g in result['graphs']))
        self.assertEqual(len(binding.calls), len(raw))
        for call, handle in zip(binding.calls, raw):
            self.assertEqual(call['graph'], handle)
            self.assertEqual(call['stream'], STREAM)
            self.assertEqual(call['green'], GREEN)
            self.assertEqual(call['max_nodes'], DEFAULT_MAX_NODES)
            self.assertEqual(call['max_depth'], DEFAULT_MAX_DEPTH)
            self.assertEqual(call['max_bytes'], DEFAULT_MAX_BYTES)
        self.assertEqual(result['identity']['split'],
                         {'requested_decode_sms': REQUESTED, 'decode_sms': SMS})
        self.assertFalse(result['collective_coverage']['confirmed'])
        self.assertEqual(result['collective_coverage']['basis'],
                         'function_name_substring')
        self.assertEqual(result['scope']['scheduler_pause'], 'caller-proven')
        text = json.dumps(result)
        for handle in raw:
            self.assertNotIn(str(handle), text)
        self.assertNotIn(str(STREAM), text)
        self.assertNotIn(str(GREEN), text)
        self.assertNotIn('0x', text)

    def test_missing_graph_is_inconclusive_not_pass(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        first = next(iter(graphs))
        del graphs[first]
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], INCONCLUSIVE)
        self.assertIn('graph_missing', result['reasons'])
        self.assertEqual(len(binding.calls), 1)

    def test_extra_graph_is_fail(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        graphs[object()] = FakeGraph(10 ** 12 + 99)
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], FAIL)
        self.assertIn('graph_extra', result['reasons'])

    def test_duplicate_descriptor_is_fail(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        probe.bank.pairs = tuple(probe.bank.pairs) + (probe.bank.pairs[0],)
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], FAIL)
        self.assertIn('descriptor_duplicate', result['reasons'])

    def test_missing_keep_graph_is_inconclusive(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        del manager._flash_s5_keep_graphs
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], INCONCLUSIVE)
        self.assertIn('keep_graph_missing', result['reasons'])

    def test_disabled_keep_graph_is_inconclusive(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        manager._flash_s5_keep_graphs = False
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], INCONCLUSIVE)
        self.assertIn('keep_graph_disabled', result['reasons'])

    def test_missing_capture_check_is_inconclusive(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        checks.pop(next(iter(checks)))
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], INCONCLUSIVE)
        self.assertIn('capture_check_missing', result['reasons'])

    def test_capture_stream_contradiction_is_fail(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        next(iter(checks.values()))['torch_matches_owner'] = False
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], FAIL)
        self.assertIn('capture_stream_mismatch', result['reasons'])

    def test_capture_owner_mismatch_is_fail(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        manager._flash_s5_capture_owner = object()
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], FAIL)
        self.assertIn('capture_owner_mismatch', result['reasons'])

    def test_missing_capture_owner_is_inconclusive(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        del manager._flash_s5_capture_owner
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], INCONCLUSIVE)
        self.assertIn('capture_owner_missing', result['reasons'])

    def test_green_context_mismatch_is_fail(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        binding.reports[raw[0]] = NativeInspection('green_context_mismatch', 5, False)
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], FAIL)
        self.assertIn('green_context_mismatch', result['reasons'])

    def test_green_context_missing_is_inconclusive(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        binding.reports[raw[0]] = NativeInspection('green_missing', 4, False)
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], INCONCLUSIVE)
        self.assertIn('native_green_missing', result['reasons'])

    def test_wrong_node_context_wins_over_unknown_node(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        nodes = [kernel(0, ctx_matches=False),
                 kernel(1, ctx_known=False, sm_query='unknown', sm_count=None)]
        binding.reports[raw[0]] = NativeInspection(
            'ok', 0, False, build_report(nodes))
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], FAIL)
        self.assertIn('node_context_mismatch', result['reasons'])
        record = result['graphs'][0]
        self.assertEqual(record['context']['mismatched_node_ids'], [0])
        self.assertEqual(record['context']['unknown_node_ids'], [1])

    def test_wrong_node_sm_is_fail(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        nodes = [kernel(0, sm_count=SMS + 1)]
        binding.reports[raw[0]] = NativeInspection('ok', 0, True, build_report(nodes))
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], FAIL)
        self.assertIn('node_sm_mismatch', result['reasons'])
        self.assertEqual(result['graphs'][0]['sm']['mismatched_node_ids'], [0])

    def test_unknown_node_sm_is_inconclusive(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        nodes = [kernel(0, sm_query='unsupported', sm_count=None)]
        binding.reports[raw[0]] = NativeInspection(
            'ok', 0, False, build_report(nodes))
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], INCONCLUSIVE)
        self.assertIn('node_sm_unknown', result['reasons'])
        self.assertNotIn('node_sm_mismatch', result['fail_reasons'])

    def test_missing_params_kernel_is_inconclusive(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        nodes = [kernel(0, params='error')]
        binding.reports[raw[0]] = NativeInspection(
            'ok', 0, False, build_report(nodes))
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], INCONCLUSIVE)
        self.assertIn('node_query_unknown', result['reasons'])
        self.assertIn('native_incomplete', result['reasons'])

    def test_nested_graph_passes_when_receipt_is_consistent(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        nodes = [graph_node(0, 1),
                 kernel(1, graph=1, parent=0, depth=1)]
        binding.reports[raw[0]] = NativeInspection('ok', 0, True, build_report(nodes))
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], PASS)
        record = result['graphs'][0]
        self.assertEqual(record['child_graphs'], 1)
        self.assertEqual(record['max_depth'], 1)

    def test_inconsistent_child_receipt_is_invalid(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        report = build_report([graph_node(0, 1),
                               kernel(1, graph=1, parent=0, depth=1)])
        report['child_graphs'] = 2
        binding.reports[raw[0]] = NativeInspection('ok', 0, True, report)
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], INCONCLUSIVE)
        self.assertIn('native_native_report_invalid', result['reasons'])

    def test_dangling_empty_child_graph_is_not_a_pass(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        report = build_report([graph_node(0, 1)])
        binding.reports[raw[0]] = NativeInspection('ok', 0, True, report)
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], INCONCLUSIVE)
        self.assertIn('empty_child_graph', result['reasons'])

    def test_unsupported_node_is_inconclusive(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        binding.reports[raw[0]] = NativeInspection(
            'ok', 0, False, build_report([other_node(0)]))
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], INCONCLUSIVE)
        self.assertIn('unsupported_node', result['reasons'])
        self.assertFalse(result['graphs'][0]['collective']['complete'])

    def test_limits_are_inconclusive(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        binding.reports[raw[0]] = NativeInspection(
            'ok', 0, False, build_report([kernel(0)], limits=True))
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], INCONCLUSIVE)
        self.assertIn('limits_exceeded', result['reasons'])

    def test_truncated_native_output_carries_required_bytes(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        binding.reports[raw[0]] = NativeInspection(
            'truncated', 1, False, None, required_bytes=9999)
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], INCONCLUSIVE)
        self.assertIn('native_truncated', result['reasons'])
        self.assertEqual(result['graphs'][0]['required_bytes'], 9999)

    def test_native_error_code_is_short_and_payload_free(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        binding.reports[raw[0]] = GraphAuditError('malformed_json')
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], INCONCLUSIVE)
        self.assertIn('native_call_invalid', result['reasons'])

    def test_random_status_string_is_not_echoed(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        binding.reports[raw[0]] = NativeInspection('secret payload', 1, False)
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], INCONCLUSIVE)
        self.assertIn('native_contract', result['reasons'])
        self.assertNotIn('secret', json.dumps(result))

    def test_duplicate_node_id_invalidates_receipt(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        nodes = [kernel(0), kernel(0)]
        binding.reports[raw[0]] = NativeInspection(
            'ok', 0, True, build_report(nodes))
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], INCONCLUSIVE)
        self.assertIn('native_native_report_invalid', result['reasons'])

    def test_bool_node_count_invalidates_receipt(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        report = build_report([kernel(0)])
        report['node_count'] = True
        binding.reports[raw[0]] = NativeInspection('ok', 0, True, report)
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], INCONCLUSIVE)
        self.assertIn('native_native_report_invalid', result['reasons'])

    def test_extra_root_key_invalidates_receipt(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        report = build_report([kernel(0)])
        report['address'] = '0xdeadbeef'
        binding.reports[raw[0]] = NativeInspection('ok', 0, True, report)
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], INCONCLUSIVE)
        self.assertIn('native_native_report_invalid', result['reasons'])
        self.assertNotIn('deadbeef', json.dumps(result))

    def test_conflicting_complete_receipt_is_invalid(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        report = build_report([kernel(0)])
        report['complete'] = False
        binding.reports[raw[0]] = NativeInspection('ok', 0, False, report)
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], INCONCLUSIVE)
        self.assertIn('native_native_report_invalid', result['reasons'])

    def test_stream_sm_mismatch_is_fail(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        nodes = [kernel(0)]
        binding.reports[raw[0]] = NativeInspection(
            'ok', 0, True, build_report(nodes, stream_sm_count=SMS + 1))
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], FAIL)
        self.assertIn('stream_sm_mismatch', result['reasons'])

    def test_storage_epoch_mismatch_is_fail(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        probe.router.storage_epoch = 2
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], FAIL)
        self.assertIn('storage_epoch_mismatch', result['reasons'])

    def test_route_generation_mismatch_is_fail(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        probe.router.route_generation = 2
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], FAIL)
        self.assertIn('route_generation_mismatch', result['reasons'])

    def test_missing_split_identity_is_inconclusive_before_native(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        probe.split.decode_sms = None
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], INCONCLUSIVE)
        self.assertIn('measured_sms_missing', result['reasons'])
        self.assertEqual(binding.calls, [])

    def test_unsupported_graph_api_is_inconclusive(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        bad = types.SimpleNamespace()
        key = next(iter(graphs))
        graphs[key] = bad
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], INCONCLUSIVE)
        self.assertIn('graph_api_unsupported', result['reasons'])

    def test_router_inventory_mismatch_is_fail(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        probe.router.descriptors = tuple(probe.router.descriptors) + (desc(999),)
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], FAIL)
        self.assertIn('descriptor_inventory_extra', result['reasons'])

    def test_router_handle_missing_is_inconclusive(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        probe.router.handles.pop(next(iter(probe.router.handles)))
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], INCONCLUSIVE)
        self.assertIn('router_handle_missing', result['reasons'])

    def test_unresolved_probe_state_refuses_before_native(self):
        for attribute, value, code in (
                ('in_flight', 1, 'unresolved_work'),
                ('quarantined', 'failed', 'quarantined'),
                ('shutdown_record', {}, 'shutdown')):
            probe, binding, manager, graphs, checks, handles, raw = scene()
            setattr(probe, attribute, value)
            with self.assertRaises(GraphAuditError) as caught:
                audit_bank(probe, binding)
            self.assertEqual(caught.exception.code, code)
            self.assertEqual(binding.calls, [])

    def test_transition_in_progress_refuses(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        probe.controller.admitting = False
        with self.assertRaises(GraphAuditError) as caught:
            audit_bank(probe, binding)
        self.assertEqual(caught.exception.code, 'transition_in_progress')
        self.assertEqual(binding.calls, [])

    def test_limits_are_strict(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        for kwargs in ({'max_nodes': True}, {'max_depth': 1000},
                       {'max_bytes': 8}, {'max_graphs': 0}):
            with self.assertRaises(GraphAuditError) as caught:
                audit_bank(probe, binding, **kwargs)
            self.assertEqual(caught.exception.code, 'bad_limits')
        self.assertEqual(binding.calls, [])

    def test_binding_contract_is_checked(self):
        probe, binding, manager, graphs, checks, handles, raw = scene()
        for mutation in ('abi', 'sha', 'source'):
            probe, binding, manager, graphs, checks, handles, raw = scene()
            if mutation == 'abi':
                binding.abi = 2
                code = 'binding_contract'
            elif mutation == 'sha':
                binding.library_sha256 = 'nope'
                code = 'binding_contract'
            else:
                binding.helper_source_sha256 = '1' * 64
                code = 'helper_source_mismatch'
            with self.assertRaises(GraphAuditError) as caught:
                audit_bank(probe, binding)
            self.assertEqual(caught.exception.code, code)
            self.assertEqual(binding.calls, [])



class ReviewRegressionTests(unittest.TestCase):
    def test_missing_source_identity_refused(self):
        probe, binding, *_ = scene()
        binding.helper_source_sha256 = None
        with self.assertRaisesRegex(GraphAuditError, '^binding_contract$'):
            audit_bank(probe, binding)
        self.assertEqual(binding.calls, [])

    def test_conflicting_native_status_and_rc_never_pass(self):
        for rc, status in ((5, 'ok'), (0, 'green_context_mismatch')):
            probe, binding, *_, handles = scene()
            binding.reports[handles[0]] = NativeInspection(status, rc, True, build_report([kernel(0)]))
            result = audit_bank(probe, binding)
            self.assertNotEqual(result['verdict'], PASS)
            self.assertIn('native_contract', result['reasons'])

    def test_missing_and_foreign_replay_handles_never_pass(self):
        for mutation in ('missing', 'foreign', 'destroyed'):
            probe, binding, *_ = scene()
            if mutation == 'missing':
                probe.router.handles = None
            elif mutation == 'foreign':
                next(iter(probe.router.handles.values())).bank = object()
            else:
                next(iter(probe.router.handles.values())).destroyed = True
            self.assertNotEqual(audit_bank(probe, binding)['verdict'], PASS)

    def test_duplicate_raw_graph_never_passes(self):
        probe, binding, *_ = scene(handles=[1234, 1234])
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], FAIL)
        self.assertIn('graph_duplicate', result['reasons'])

    def test_vllm_descriptor_must_match_declared_mirror(self):
        probe, binding, *_ = scene()
        probe.bank.pairs[0][0].num_tokens = 77
        result = audit_bank(probe, binding)
        self.assertEqual(result['verdict'], FAIL)
        self.assertIn('capture_descriptor_mismatch', result['reasons'])

    def test_callbacks_cannot_replace_graph_or_change_ownership(self):
        for what in ('graph', 'checks', 'generation', 'quarantine', 'owner', 'handles'):
            probe, binding, manager, *_ = scene()
            inspect = binding.inspect
            def mutate(*args, **kwargs):
                if what == 'graph':
                    key = next(iter(manager.graphs))
                    manager.graphs[key] = FakeGraph(123)
                elif what == 'checks':
                    next(iter(manager._flash_s5_capture_stream_checks.values()))['torch_matches_owner'] = False
                elif what == 'generation':
                    probe.route_generation += 1
                elif what == 'quarantine':
                    probe.quarantined = 'payload-hidden'
                elif what == 'owner':
                    manager._flash_s5_capture_owner = object()
                else:
                    next(iter(probe.router.handles.values())).destroyed = True
                return inspect(*args, **kwargs)
            binding.inspect = mutate
            result = audit_bank(probe, binding)
            self.assertNotEqual(result['verdict'], PASS, what)
            self.assertIn('concurrent_mutation', result['reasons'])
            self.assertNotIn('payload-hidden', json.dumps(result))

    def test_invalid_probe_labels_are_never_echoed(self):
        for name in ('route', 'interval'):
            probe, binding, *_ = scene()
            setattr(probe, name, '0xfeedface-secret-payload')
            result = audit_bank(probe, binding)
            self.assertNotEqual(result['verdict'], PASS)
            self.assertNotIn('0xfeedface', json.dumps(result))
        probe, binding, *_ = scene()
        probe.router.name = '0xfeedface-secret-payload'
        with self.assertRaisesRegex(GraphAuditError, '^probe_contract$'):
            audit_bank(probe, binding)

    def test_error_codes_from_untrusted_binding_are_not_echoed(self):
        probe, binding, *_, handles = scene()
        binding.reports[handles[0]] = GraphAuditError('0xfeedface-secret-payload')
        result = audit_bank(probe, binding)
        self.assertNotIn('0xfeedface', json.dumps(result))
        self.assertEqual(result['verdict'], INCONCLUSIVE)

    def test_report_cannot_exceed_requested_node_limit(self):
        probe, binding, *_, handles = scene()
        binding.reports[handles[0]] = NativeInspection('ok', 0, True, build_report([kernel(0), kernel(1)]))
        result = audit_bank(probe, binding, max_nodes=1)
        self.assertEqual(result['verdict'], INCONCLUSIVE)

    def test_query_status_contradictions_cannot_pass(self):
        for field in ('query_rc', 'sm_query_rc'):
            probe, binding, *_, handles = scene()
            report = build_report([kernel(0)])
            report['nodes'][0][field] = 801
            binding.reports[handles[0]] = NativeInspection('ok', 0, True, report)
            self.assertEqual(audit_bank(probe, binding)['verdict'], INCONCLUSIVE)

    def test_invalid_node_ids_and_unknown_copy_kind_cannot_pass(self):
        for nodes in ([kernel(9)], [memcpy(0, src='unknown')]):
            probe, binding, *_, handles = scene()
            report = build_report(nodes)
            binding.reports[handles[0]] = NativeInspection('ok', 0, True, report)
            self.assertEqual(audit_bank(probe, binding)['verdict'], INCONCLUSIVE)

class BindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if shutil.which('g++') is None:
            raise unittest.SkipTest('g++ is not installed')
        cls.tmp = tempfile.mkdtemp(prefix='q17b-graph-audit-')
        cls.addClassCleanup(shutil.rmtree, cls.tmp, ignore_errors=True)
        source = ROOT / 'patches/s5_engine/native/flash_s5_graph_audit.cu'
        stub_header = Path(cls.tmp) / 'cuda.h'
        stub_source = Path(cls.tmp) / 'driver_stub.cpp'
        stub_header.write_text(STUB_HEADER)
        stub_source.write_text(STUB_DRIVER)
        cls.library = str(Path(cls.tmp) / 'libstub.so')
        seen = subprocess.run(
            ['g++', '-std=c++17', '-O1', '-g', '-shared', '-fPIC',
             '-Wall', '-Wextra', '-Wno-unused-parameter', '-I', cls.tmp,
             '-x', 'c++', str(source), '-x', 'c++', str(stub_source),
             '-o', cls.library],
            capture_output=True, text=True)
        if seen.returncode != 0:
            raise AssertionError('stub helper build failed:\n' + seen.stderr)
        cls.digest = hashlib.sha256(Path(cls.library).read_bytes()).hexdigest()
        cls.abi2 = cls._toy_library('abi2.cpp', TOY_ABI2)
        cls.missing = cls._toy_library('missing.cpp', TOY_MISSING)
        cls.bad_json = cls._toy_library('bad_json.cpp', TOY_BAD_JSON)
        cls.lib = ctypes.CDLL(cls.library)
        cls._declare_stub(cls.lib)

    @staticmethod
    def _declare_stub(lib):
        U, I, N = ctypes.c_uint64, ctypes.c_int, None
        lib.stub_func_backed.argtypes = [U]
        lib.stub_func_backed.restype = N
        lib.stub_stream_ctx.argtypes = [U, U, I]
        lib.stub_stream_ctx.restype = N
        lib.stub_enumeration.argtypes = [U, I]
        lib.stub_enumeration.restype = N
        lib.stub_reset.argtypes = []
        lib.stub_reset.restype = N
        lib.stub_make_green.argtypes = [U, ctypes.c_uint, I, I]
        lib.stub_make_green.restype = U
        lib.stub_make_stream.argtypes = [U, I]
        lib.stub_make_stream.restype = U
        lib.stub_make_graph.argtypes = []
        lib.stub_make_graph.restype = U
        lib.stub_make_func.argtypes = [ctypes.c_char_p]
        lib.stub_make_func.restype = U
        lib.stub_add_kernel.argtypes = [U, U, U, I]
        lib.stub_add_kernel.restype = U
        lib.stub_add_memcpy.argtypes = [U, I, I, U, U, U, I]
        lib.stub_add_memcpy.restype = U
        lib.stub_add_memset.argtypes = [U, U, U, U, I]
        lib.stub_add_memset.restype = U
        lib.stub_add_graph_node.argtypes = [U, U, I]
        lib.stub_add_graph_node.restype = U
        lib.stub_add_other.argtypes = [U, I]
        lib.stub_add_other.restype = U
        lib.stub_set_graph_rc.argtypes = [U, I]
        lib.stub_set_graph_rc.restype = N

    @classmethod
    def tearDownClass(cls):
        # CPU compiler artifacts under cls.tmp are removed by the class cleanup registered in setUpClass.
        pass

    @classmethod
    def _toy_library(cls, name, text):
        source = Path(cls.tmp) / name
        source.write_text(text)
        output = str(Path(cls.tmp) / (name + '.so'))
        seen = subprocess.run(
            ['g++', '-std=c++17', '-O1', '-shared', '-fPIC', '-x', 'c++',
             str(source), '-o', output], capture_output=True, text=True)
        if seen.returncode != 0:
            raise AssertionError('toy library build failed:\n' + seen.stderr)
        return output

    def setUp(self):
        self.lib.stub_reset()
        self.binding = AuditBinding(self.library, self.digest)

    def green(self, ctx, sm=SMS, sm_rc=0, convert_rc=0):
        return self.lib.stub_make_green(ctx, sm, sm_rc, convert_rc)

    def stream(self, green, rc=0):
        return self.lib.stub_make_stream(green, rc)

    def test_func_backed_ignored_ctx_cannot_prove_placement(self):
        green = self.green(12345, SMS)
        stream = self.stream(green)
        graph = self.lib.stub_make_graph()
        node = self.lib.stub_add_kernel(graph, self.lib.stub_make_func(b'ncclAllReduce'), 12345, 0)
        self.lib.stub_func_backed(node)
        seen = self.binding.inspect(graph, stream, green, max_nodes=10, max_depth=4, max_bytes=8192)
        self.assertFalse(seen.complete)
        node = seen.report['nodes'][0]
        self.assertEqual(node['context_basis'], 'ignored_for_func')
        self.assertFalse(node['context_known'])
        self.assertIsNone(node['sm_count'])
        self.assertEqual(node['collective_candidate'], 'allreduce')

    def test_actual_stream_context_query_and_mismatch(self):
        green = self.green(12345, SMS)
        stream = self.stream(green)
        graph = self.lib.stub_make_graph()
        for ctx, rc, status in ((0, 801, 'stream_context_query_failed'),
                                (54321, 0, 'stream_context_mismatch')):
            self.lib.stub_stream_ctx(stream, ctx, rc)
            seen = self.binding.inspect(graph, stream, green, max_nodes=10, max_depth=4, max_bytes=8192)
            self.assertEqual(seen.status, status)
            self.assertFalse(seen.complete)

    def test_changed_or_duplicate_native_inventory_is_not_complete(self):
        green = self.green(12345, SMS)
        stream = self.stream(green)
        graph = self.lib.stub_make_graph()
        func = self.lib.stub_make_func(b'k')
        self.lib.stub_add_kernel(graph, func, 12345, 0)
        self.lib.stub_add_kernel(graph, func, 12345, 0)
        for mode, reason in ((1, 'graph_changed'), (2, 'node_inventory_invalid')):
            self.lib.stub_enumeration(graph, mode)
            seen = self.binding.inspect(graph, stream, green, max_nodes=10, max_depth=4, max_bytes=8192)
            self.assertFalse(seen.complete)
            self.assertIn(reason, seen.report['reasons'])

    def test_pinned_cuda13_header_syntax(self):
        header = ROOT / 'reference/cuda-13/cuda.h'
        if not header.is_file():
            self.skipTest('pinned CUDA 13 reference header absent')
        self.assertEqual(hashlib.sha256(header.read_bytes()).hexdigest(),
                         '64025f81a9d6d2b10e9e11654f3523e5b046d01b883f0b9ca474e463274b80d4')
        result = subprocess.run(['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                                 '-I', str(header.parent), '-x', 'c++', '-fsyntax-only',
                                 str(ROOT / 'patches/s5_engine/native/flash_s5_graph_audit.cu')],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_handle_integer_overflow_refused_before_native(self):
        for handle in (-1, True, 2**64, 2**64 + 1):
            with self.assertRaisesRegex(GraphAuditError, '^bad_handle$'):
                self.binding.inspect(handle, 1, 1, max_nodes=10, max_depth=2, max_bytes=4096)

    def test_binding_identity_and_source_digest(self):
        self.assertEqual(self.binding.abi, ABI_VERSION)
        self.assertEqual(self.binding.library_sha256, self.digest)
        self.assertEqual(self.binding.helper_source_sha256, helper_source_sha256())
        self.assertRegex(helper_source_sha256(), r'^[0-9a-f]{64}$')

    def test_exact_library_sha_is_required(self):
        with self.assertRaises(GraphAuditError) as caught:
            AuditBinding(self.library, 'f' * 64)
        self.assertEqual(caught.exception.code, 'library_sha_mismatch')
        with self.assertRaises(GraphAuditError) as caught:
            AuditBinding(str(Path(self.tmp) / 'absent.so'), self.digest)
        self.assertEqual(caught.exception.code, 'library_missing')
        with self.assertRaises(GraphAuditError) as caught:
            AuditBinding(self.library, 'not-a-digest')
        self.assertEqual(caught.exception.code, 'bad_expected_sha256')

    def test_own_abi_is_required(self):
        digest = hashlib.sha256(Path(self.abi2).read_bytes()).hexdigest()
        with self.assertRaises(GraphAuditError) as caught:
            AuditBinding(self.abi2, digest)
        self.assertEqual(caught.exception.code, 'abi_mismatch')

    def test_missing_export_is_refused(self):
        digest = hashlib.sha256(Path(self.missing).read_bytes()).hexdigest()
        with self.assertRaises(GraphAuditError) as caught:
            AuditBinding(self.missing, digest)
        self.assertEqual(caught.exception.code, 'symbol_missing')

    def test_malformed_json_is_a_short_error(self):
        digest = hashlib.sha256(Path(self.bad_json).read_bytes()).hexdigest()
        binding = AuditBinding(self.bad_json, digest)
        with self.assertRaises(GraphAuditError) as caught:
            binding.inspect(1, 2, 3, max_nodes=10, max_depth=1, max_bytes=4096)
        self.assertEqual(caught.exception.code, 'malformed_json')
        self.assertNotIn('schema', str(caught.exception))

    def test_matching_green_context_and_sanitised_nodes(self):
        green = self.green(0x400000001, SMS)
        stream = self.stream(green)
        graph = self.lib.stub_make_graph()
        func = self.lib.stub_make_func(b'ncclAllReduce')
        self.lib.stub_add_kernel(graph, func, 0x400000001, 0)
        seen = self.binding.inspect(graph, stream, green, max_nodes=100,
                                    max_depth=4, max_bytes=65536)
        self.assertEqual(seen.status, 'ok')
        self.assertTrue(seen.complete)
        report = seen.report
        self.assertEqual(report['schema'], NATIVE_SCHEMA)
        self.assertEqual(report['counts']['kernel'], 1)
        self.assertEqual(report['unknown_nodes'], 0)
        self.assertEqual(report['stream_context']['sm_count'], SMS)
        node = report['nodes'][0]
        self.assertTrue(node['context_matches'])
        self.assertEqual(node['sm_count'], SMS)
        self.assertEqual(node['collective_candidate'], 'allreduce')
        self.assertIs(node['collective_confirmed'], False)
        self.assertEqual(node['grid'], (4, 1, 1))
        text = json.dumps(thaw(report))
        self.assertNotIn('ncclAllReduce', text)
        for token in (graph, stream, green, 0x400000001):
            self.assertNotIn(str(token), text)

    def test_wrong_or_absent_green_context_never_falls_back(self):
        green = self.green(0x400000002, SMS)
        stream = self.stream(green)
        graph = self.lib.stub_make_graph()
        self.lib.stub_add_kernel(graph, self.lib.stub_make_func(b'k'),
                                 0x400000002, 0)
        wrong = self.binding.inspect(graph, stream, green + 7, max_nodes=10,
                                     max_depth=2, max_bytes=4096)
        self.assertEqual(wrong.status, 'green_context_mismatch')
        self.assertIsNone(wrong.report)
        self.assertFalse(wrong.complete)

        no_green = self.stream(0)
        absent = self.binding.inspect(graph, no_green, green, max_nodes=10,
                                      max_depth=2, max_bytes=4096)
        self.assertEqual(absent.status, 'green_missing')

    def test_missing_green_query_api_is_inconclusive(self):
        green = self.green(0x400000003, SMS)
        stream = self.stream(green, 801)  # CUDA_ERROR_NOT_SUPPORTED
        graph = self.lib.stub_make_graph()
        seen = self.binding.inspect(graph, stream, green, max_nodes=10,
                                    max_depth=2, max_bytes=4096)
        self.assertEqual(seen.status, 'green_query_failed')
        self.assertIsNone(seen.report)

    def test_context_convert_failure_is_reported(self):
        green = self.green(0x400000004, SMS, convert_rc=400)
        stream = self.stream(green)
        graph = self.lib.stub_make_graph()
        seen = self.binding.inspect(graph, stream, green, max_nodes=10,
                                    max_depth=2, max_bytes=4096)
        self.assertEqual(seen.status, 'context_convert_failed')

    def test_node_context_mismatch_and_sm_unknown_are_explicit(self):
        green = self.green(0x400000005, SMS)
        stream = self.stream(green)
        foreign = self.green(0x400000006, SMS)
        unsupported = self.green(0x400000007, SMS, sm_rc=801)
        graph = self.lib.stub_make_graph()
        self.lib.stub_add_kernel(graph, self.lib.stub_make_func(b'k1'),
                                 0x400000006, 0)
        self.lib.stub_add_kernel(graph, self.lib.stub_make_func(b'k2'),
                                 0x400000007, 0)
        seen = self.binding.inspect(graph, stream, green, max_nodes=10,
                                    max_depth=2, max_bytes=8192)
        self.assertEqual(seen.status, 'ok')
        self.assertFalse(seen.complete)
        first, second = seen.report['nodes']
        self.assertTrue(first['context_known'])
        self.assertFalse(first['context_matches'])
        self.assertEqual(second['sm_query'], 'unsupported')
        self.assertIsNone(second['sm_count'])
        self.assertTrue(second['unknown'])
        self.assertIn('kernel_sm_unknown', seen.report['reasons'])

    def test_child_graph_is_walked_with_bounded_depth(self):
        green = self.green(0x400000008, SMS)
        stream = self.stream(green)
        child = self.lib.stub_make_graph()
        self.lib.stub_add_kernel(child, self.lib.stub_make_func(b'k'),
                                 0x400000008, 0)
        parent = self.lib.stub_make_graph()
        self.lib.stub_add_graph_node(parent, child, 0)
        seen = self.binding.inspect(parent, stream, green, max_nodes=10,
                                    max_depth=4, max_bytes=8192)
        self.assertEqual(seen.status, 'ok')
        self.assertTrue(seen.complete)
        report = seen.report
        self.assertEqual(report['child_graphs'], 1)
        self.assertEqual(report['node_count'], 2)
        self.assertEqual(report['max_depth'], 1)
        self.assertEqual([node['graph'] for node in report['nodes']], [0, 1])

    def test_depth_limit_is_inconclusive_not_silent(self):
        green = self.green(0x400000009, SMS)
        stream = self.stream(green)
        child = self.lib.stub_make_graph()
        self.lib.stub_add_kernel(child, self.lib.stub_make_func(b'k'),
                                 0x400000009, 0)
        parent = self.lib.stub_make_graph()
        self.lib.stub_add_graph_node(parent, child, 0)
        seen = self.binding.inspect(parent, stream, green, max_nodes=10,
                                    max_depth=0, max_bytes=8192)
        self.assertEqual(seen.status, 'ok')
        self.assertFalse(seen.complete)
        self.assertIn('limits_depth', seen.report['reasons'])
        self.assertEqual(seen.report['nodes'][0]['child'], -1)
        self.assertTrue(seen.report['nodes'][0]['unknown'])

    def test_memcpy_and_memset_metadata(self):
        green = self.green(0x40000000a, SMS)
        stream = self.stream(green)
        graph = self.lib.stub_make_graph()
        self.lib.stub_add_memcpy(graph, 2, 1, 16, 2, 3, 0)
        self.lib.stub_add_memset(graph, 4, 4, 2, 0)
        seen = self.binding.inspect(graph, stream, green, max_nodes=10,
                                    max_depth=2, max_bytes=8192)
        self.assertFalse(seen.complete)
        first, second = seen.report['nodes']
        self.assertFalse(first['context_known'])
        self.assertEqual(second['context_basis'], 'not_exposed')
        self.assertEqual(first['direction'], 'device-to-host')
        self.assertEqual(first['bytes'], 16 * 2 * 3)
        self.assertEqual(second['bytes'], 4 * 4 * 2)
        self.assertEqual(seen.report['counts']['memcpy'], 1)
        self.assertEqual(seen.report['counts']['memset'], 1)

    def test_unsupported_node_type_is_never_omitted(self):
        green = self.green(0x40000000b, SMS)
        stream = self.stream(green)
        graph = self.lib.stub_make_graph()
        self.lib.stub_add_other(graph, 5)
        seen = self.binding.inspect(graph, stream, green, max_nodes=10,
                                    max_depth=2, max_bytes=8192)
        self.assertFalse(seen.complete)
        self.assertEqual(seen.report['counts']['other'], 1)
        self.assertEqual(seen.report['unsupported_nodes'], 1)
        self.assertIn('unsupported_node', seen.report['reasons'])

    def test_node_budget_is_reported_not_truncated_as_success(self):
        green = self.green(0x40000000c, SMS)
        stream = self.stream(green)
        graph = self.lib.stub_make_graph()
        for _ in range(2):
            self.lib.stub_add_kernel(graph, self.lib.stub_make_func(b'k'),
                                     0x40000000c, 0)
        seen = self.binding.inspect(graph, stream, green, max_nodes=1,
                                    max_depth=2, max_bytes=8192)
        self.assertEqual(seen.status, 'ok')
        self.assertFalse(seen.complete)
        self.assertTrue(seen.report['limits_exceeded'])
        self.assertIn('limits_nodes', seen.report['reasons'])

    def test_small_output_buffer_reports_exact_required_size(self):
        green = self.green(0x40000000d, SMS)
        stream = self.stream(green)
        graph = self.lib.stub_make_graph()
        self.lib.stub_add_kernel(graph, self.lib.stub_make_func(b'k'),
                                 0x40000000d, 0)
        seen = self.binding.inspect(graph, stream, green, max_nodes=10,
                                    max_depth=2, max_bytes=256)
        self.assertEqual(seen.status, 'truncated')
        self.assertEqual(seen.native_rc, 1)
        self.assertIsNone(seen.report)
        self.assertGreater(seen.required_bytes, 256)

    def test_empty_graph_is_valid_native_receipt(self):
        green = self.green(0x40000000e, SMS)
        stream = self.stream(green)
        graph = self.lib.stub_make_graph()
        seen = self.binding.inspect(graph, stream, green, max_nodes=10,
                                    max_depth=2, max_bytes=4096)
        self.assertEqual(seen.status, 'ok')
        self.assertTrue(seen.complete)
        self.assertEqual(seen.report['root_graph_nodes'], 0)

    def test_real_helper_and_audit_bank_agree(self):
        green = self.green(0x40000000f, SMS)
        stream = self.stream(green)
        graph = self.lib.stub_make_graph()
        self.lib.stub_add_kernel(graph, self.lib.stub_make_func(b'allreduce'),
                                 0x40000000f, 0)
        probe, _, manager, graphs, checks, handles, raw = scene(
            descriptors=[desc(8)], handles=[graph])
        probe.split.decode_stream_token = stream
        probe.split.decode_context_id = green
        result = audit_bank(probe, self.binding)
        self.assertEqual(result['verdict'], PASS)
        self.assertEqual(result['graphs'][0]['counts']['kernel'], 1)

    def test_real_helper_wrong_sm_fails_audit(self):
        green = self.green(0x400000010, SMS + 1)
        stream = self.stream(green)
        graph = self.lib.stub_make_graph()
        self.lib.stub_add_kernel(graph, self.lib.stub_make_func(b'k'),
                                 0x400000010, 0)
        probe, _, manager, graphs, checks, handles, raw = scene(
            descriptors=[desc(8)], handles=[graph])
        probe.split.decode_stream_token = stream
        probe.split.decode_context_id = green
        result = audit_bank(probe, self.binding)
        self.assertEqual(result['verdict'], FAIL)
        self.assertIn('stream_sm_mismatch', result['reasons'])
        self.assertIn('node_sm_mismatch', result['reasons'])

    def test_real_helper_empty_root_is_not_a_pass(self):
        green = self.green(0x400000011, SMS)
        stream = self.stream(green)
        graph = self.lib.stub_make_graph()
        probe, _, manager, graphs, checks, handles, raw = scene(
            descriptors=[desc(8)], handles=[graph])
        probe.split.decode_stream_token = stream
        probe.split.decode_context_id = green
        result = audit_bank(probe, self.binding)
        self.assertEqual(result['verdict'], INCONCLUSIVE)
        self.assertIn('empty_root_graph', result['reasons'])


# --- CPU stub CUDA surface --------------------------------------------------
# Compile-only stand-in for the pinned CUDA 13 headers. It models just enough
# of the driver API to drive the walker; it is not an ABI or placement proof.

STUB_HEADER = r'''
#ifndef FLASH_S5_STUB_CUDA_H
#define FLASH_S5_STUB_CUDA_H
#include <stddef.h>
#include <stdint.h>

typedef int CUresult;
typedef int CUgraphNodeType;
typedef int CUmemorytype;
typedef int CUdevResourceType;
typedef unsigned long long CUdeviceptr;

typedef struct CUgraph_st *CUgraph;
typedef struct CUgraphNode_st *CUgraphNode;
typedef struct CUstream_st *CUstream;
typedef struct CUgreenCtx_st *CUgreenCtx;
typedef struct CUcontext_st *CUcontext;
typedef struct CUfunc_st *CUfunction;

#define CUDA_SUCCESS 0
#define CUDA_ERROR_INVALID_HANDLE 400
#define CUDA_ERROR_NOT_SUPPORTED 801
#define CUDA_VERSION 13000

#define CU_GRAPH_NODE_TYPE_KERNEL 0
#define CU_GRAPH_NODE_TYPE_MEMCPY 1
#define CU_GRAPH_NODE_TYPE_MEMSET 2
#define CU_GRAPH_NODE_TYPE_GRAPH 4
#define CU_GRAPH_NODE_TYPE_EVENT 5

#define CU_MEMORYTYPE_HOST 1
#define CU_MEMORYTYPE_DEVICE 2
#define CU_MEMORYTYPE_ARRAY 3
#define CU_MEMORYTYPE_UNIFIED 4

#define CU_DEV_RESOURCE_TYPE_SM 1

typedef struct { unsigned smCount; } CUdevSmResource;
typedef struct CUdevResource_st {
  CUdevResourceType type;
  CUdevSmResource sm;
} CUdevResource;

typedef void *CUkernel;
typedef struct CUDA_KERNEL_NODE_PARAMS_st {
  CUfunction func;
  unsigned gridDimX, gridDimY, gridDimZ;
  unsigned blockDimX, blockDimY, blockDimZ;
  unsigned sharedMemBytes;
  void **kernelParams;
  void **extra;
  CUkernel kern;
  CUcontext ctx;
} CUDA_KERNEL_NODE_PARAMS;

typedef struct CUDA_MEMCPY3D_st {
  size_t srcXInBytes, srcY, srcZ, srcLOD;
  CUmemorytype srcMemoryType;
  const void *srcHost;
  CUdeviceptr srcDevice;
  void *srcArray;
  void *reserved0;
  size_t srcPitch, srcHeight;
  size_t dstXInBytes, dstY, dstZ, dstLOD;
  CUmemorytype dstMemoryType;
  void *dstHost;
  CUdeviceptr dstDevice;
  void *dstArray;
  void *reserved1;
  size_t dstPitch, dstHeight;
  size_t WidthInBytes, Height, Depth;
} CUDA_MEMCPY3D;

typedef struct CUDA_MEMSET_NODE_PARAMS_st {
  CUdeviceptr dst;
  size_t elementSize;
  size_t width, height, pitch;
  unsigned value;
} CUDA_MEMSET_NODE_PARAMS;

#ifdef __cplusplus
extern "C" {
#endif
CUresult cuStreamGetGreenCtx(CUstream stream, CUgreenCtx *green);
CUresult cuStreamGetCtx(CUstream stream, CUcontext *ctx);
CUresult cuKernelGetName(const char **name, CUkernel kernel);
CUresult cuCtxFromGreenCtx(CUcontext *ctx, CUgreenCtx green);
CUresult cuCtxGetDevResource(CUcontext ctx, CUdevResource *resource,
                             CUdevResourceType type);
CUresult cuGraphGetNodes(CUgraph graph, CUgraphNode *nodes, size_t *count);
CUresult cuGraphNodeGetType(CUgraphNode node, CUgraphNodeType *type);
CUresult cuGraphKernelNodeGetParams(CUgraphNode node,
                                    CUDA_KERNEL_NODE_PARAMS *params);
CUresult cuGraphMemcpyNodeGetParams(CUgraphNode node, CUDA_MEMCPY3D *params);
CUresult cuGraphMemsetNodeGetParams(CUgraphNode node,
                                    CUDA_MEMSET_NODE_PARAMS *params);
CUresult cuGraphChildGraphNodeGetGraph(CUgraphNode node, CUgraph *child);
CUresult cuFuncGetName(const char **name, CUfunction func);
#ifdef __cplusplus
}
#endif
#endif
'''


STUB_DRIVER = r'''
#include <cuda.h>
#include <map>
#include <string>
#include <vector>

struct CtxInfo { unsigned sm = 0; int rc = 0; bool convert_ok = true; };
struct StreamInfo { uint64_t green = 0; int rc = 0; };
struct NodeInfo {
  int type = 0;
  int params_rc = 0;
  bool func_backed = false;
  uint64_t func = 0;
  uint64_t ctx = 0;
  uint64_t child = 0;
  int src_kind = CU_MEMORYTYPE_DEVICE;
  int dst_kind = CU_MEMORYTYPE_DEVICE;
  size_t width = 0, height = 1, depth = 1;
  size_t element = 0, mwidth = 0, mheight = 1;
};
struct GraphInfo { int rc = 0; std::vector<uint64_t> nodes; };

static uint64_t g_counter = 0;
static std::map<uint64_t, CtxInfo> g_ctx;
static std::map<uint64_t, StreamInfo> g_stream;
static std::map<uint64_t, StreamInfo> g_stream_ctx;
static std::map<uint64_t, int> g_enumeration;
static std::map<uint64_t, GraphInfo> g_graph;
static std::map<uint64_t, NodeInfo> g_node;
static std::map<uint64_t, std::string> g_func;

static uint64_t next_token() { return ++g_counter; }

extern "C" void stub_reset() {
  g_counter = 0;
  g_ctx.clear(); g_stream.clear(); g_stream_ctx.clear(); g_enumeration.clear(); g_graph.clear(); g_node.clear(); g_func.clear();
}

extern "C" uint64_t stub_make_green(uint64_t ctx, unsigned sm, int sm_rc,
                                    int convert_rc) {
  uint64_t token = 0x900000000000ULL + next_token();
  g_ctx[ctx] = CtxInfo{sm, sm_rc, convert_rc == CUDA_SUCCESS};
  // green token -> ctx via a dedicated map entry keyed by the green token.
  g_ctx[token] = CtxInfo{sm, sm_rc, convert_rc == CUDA_SUCCESS};
  // Remember which ctx each green maps to in the stream table's value only.
  g_stream[token] = StreamInfo{ctx, convert_rc};
  return token;
}

extern "C" uint64_t stub_make_stream(uint64_t green, int rc) {
  uint64_t token = 0x910000000000ULL + next_token();
  g_stream[token] = StreamInfo{green, rc};
  return token;
}

extern "C" uint64_t stub_make_graph() {
  uint64_t token = 0x920000000000ULL + next_token();
  g_graph[token] = GraphInfo{};
  return token;
}

extern "C" uint64_t stub_make_func(const char *name) {
  uint64_t token = 0x930000000000ULL + next_token();
  g_func[token] = name == nullptr ? std::string() : std::string(name);
  return token;
}

extern "C" uint64_t stub_add_kernel(uint64_t graph, uint64_t func,
                                    uint64_t ctx, int params_rc) {
  uint64_t token = 0x940000000000ULL + next_token();
  NodeInfo node;
  node.type = CU_GRAPH_NODE_TYPE_KERNEL;
  node.func = func; node.ctx = ctx; node.params_rc = params_rc;
  g_node[token] = node;
  g_graph[graph].nodes.push_back(token);
  return token;
}

extern "C" uint64_t stub_add_memcpy(uint64_t graph, int src_kind, int dst_kind,
                                    uint64_t width, uint64_t height,
                                    uint64_t depth, int params_rc) {
  uint64_t token = 0x950000000000ULL + next_token();
  NodeInfo node;
  node.type = CU_GRAPH_NODE_TYPE_MEMCPY;
  node.src_kind = src_kind; node.dst_kind = dst_kind;
  node.width = (size_t)width; node.height = (size_t)height;
  node.depth = (size_t)depth; node.params_rc = params_rc;
  g_node[token] = node;
  g_graph[graph].nodes.push_back(token);
  return token;
}

extern "C" uint64_t stub_add_memset(uint64_t graph, uint64_t element,
                                    uint64_t width, uint64_t height,
                                    int params_rc) {
  uint64_t token = 0x960000000000ULL + next_token();
  NodeInfo node;
  node.type = CU_GRAPH_NODE_TYPE_MEMSET;
  node.element = (size_t)element; node.mwidth = (size_t)width;
  node.mheight = (size_t)height; node.params_rc = params_rc;
  g_node[token] = node;
  g_graph[graph].nodes.push_back(token);
  return token;
}

extern "C" uint64_t stub_add_graph_node(uint64_t graph, uint64_t child,
                                        int params_rc) {
  uint64_t token = 0x970000000000ULL + next_token();
  NodeInfo node;
  node.type = CU_GRAPH_NODE_TYPE_GRAPH;
  node.child = child; node.params_rc = params_rc;
  g_node[token] = node;
  g_graph[graph].nodes.push_back(token);
  return token;
}

extern "C" uint64_t stub_add_other(uint64_t graph, int type) {
  uint64_t token = 0x980000000000ULL + next_token();
  NodeInfo node;
  node.type = type;
  g_node[token] = node;
  g_graph[graph].nodes.push_back(token);
  return token;
}

extern "C" void stub_set_graph_rc(uint64_t graph, int rc) {
  g_graph[graph].rc = rc;
}

static NodeInfo *find_node(uint64_t token) {
  auto it = g_node.find(token);
  return it == g_node.end() ? nullptr : &it->second;
}

extern "C" CUresult cuStreamGetGreenCtx(CUstream stream, CUgreenCtx *green) {
  auto it = g_stream.find((uint64_t)(uintptr_t)stream);
  if (it == g_stream.end()) return CUDA_ERROR_INVALID_HANDLE;
  if (it->second.rc != CUDA_SUCCESS) return it->second.rc;
  *green = (CUgreenCtx)(uintptr_t)it->second.green;
  return CUDA_SUCCESS;
}

extern "C" void stub_func_backed(uint64_t node) { g_node[node].func_backed = true; }
extern "C" void stub_stream_ctx(uint64_t stream, uint64_t ctx, int rc) {
  g_stream_ctx[stream] = StreamInfo{ctx, rc};
}
extern "C" void stub_enumeration(uint64_t graph, int mode) { g_enumeration[graph] = mode; }
extern "C" CUresult cuStreamGetCtx(CUstream stream, CUcontext *ctx) {
  auto custom = g_stream_ctx.find((uint64_t)(uintptr_t)stream);
  if (custom != g_stream_ctx.end()) {
    *ctx = (CUcontext)(uintptr_t)custom->second.green;
    return custom->second.rc;
  }
  auto it = g_stream.find((uint64_t)(uintptr_t)stream);
  if (it == g_stream.end()) return CUDA_ERROR_INVALID_HANDLE;
  auto green = g_stream.find(it->second.green);
  if (green == g_stream.end()) return CUDA_ERROR_INVALID_HANDLE;
  *ctx = (CUcontext)(uintptr_t)green->second.green;
  return CUDA_SUCCESS;
}
extern "C" CUresult cuKernelGetName(const char **name, CUkernel kernel) {
  return cuFuncGetName(name, (CUfunction)kernel);
}
extern "C" CUresult cuCtxFromGreenCtx(CUcontext *ctx, CUgreenCtx green) {
  auto it = g_stream.find((uint64_t)(uintptr_t)green);
  if (it == g_stream.end()) return CUDA_ERROR_INVALID_HANDLE;
  if (it->second.rc != CUDA_SUCCESS) return it->second.rc;
  *ctx = (CUcontext)(uintptr_t)it->second.green;  // green -> context token
  return CUDA_SUCCESS;
}

extern "C" CUresult cuCtxGetDevResource(CUcontext ctx, CUdevResource *resource,
                                        CUdevResourceType type) {
  if (type != CU_DEV_RESOURCE_TYPE_SM) return CUDA_ERROR_NOT_SUPPORTED;
  auto it = g_ctx.find((uint64_t)(uintptr_t)ctx);
  if (it == g_ctx.end()) return CUDA_ERROR_INVALID_HANDLE;
  if (it->second.rc != CUDA_SUCCESS) return it->second.rc;
  resource->type = type;
  resource->sm.smCount = it->second.sm;
  return CUDA_SUCCESS;
}

extern "C" CUresult cuGraphGetNodes(CUgraph graph, CUgraphNode *nodes,
                                    size_t *count) {
  auto it = g_graph.find((uint64_t)(uintptr_t)graph);
  if (it == g_graph.end()) return CUDA_ERROR_INVALID_HANDLE;
  if (it->second.rc != CUDA_SUCCESS) return it->second.rc;
  size_t n = it->second.nodes.size();
  if (nodes == nullptr) { *count = n; return CUDA_SUCCESS; }
  size_t room = *count < n ? *count : n;
  for (size_t i = 0; i < room; ++i)
    nodes[i] = (CUgraphNode)(uintptr_t)it->second.nodes[g_enumeration[(uint64_t)(uintptr_t)graph] == 2 ? 0 : i];
  *count = n + (g_enumeration[(uint64_t)(uintptr_t)graph] == 1 ? 1 : 0);
  return CUDA_SUCCESS;
}

extern "C" CUresult cuGraphNodeGetType(CUgraphNode node, CUgraphNodeType *type) {
  NodeInfo *info = find_node((uint64_t)(uintptr_t)node);
  if (info == nullptr) return CUDA_ERROR_INVALID_HANDLE;
  *type = info->type;
  return CUDA_SUCCESS;
}

extern "C" CUresult cuGraphKernelNodeGetParams(CUgraphNode node,
                                               CUDA_KERNEL_NODE_PARAMS *p) {
  NodeInfo *info = find_node((uint64_t)(uintptr_t)node);
  if (info == nullptr) return CUDA_ERROR_INVALID_HANDLE;
  if (info->params_rc != CUDA_SUCCESS) return info->params_rc;
  p->func = info->func_backed ? (CUfunction)(uintptr_t)info->func : nullptr;
  p->kern = info->func_backed ? nullptr : (CUkernel)(uintptr_t)info->func;
  p->gridDimX = 4; p->gridDimY = 1; p->gridDimZ = 1;
  p->blockDimX = 128; p->blockDimY = 1; p->blockDimZ = 1;
  p->sharedMemBytes = 0;
  p->ctx = (CUcontext)(uintptr_t)info->ctx;
  return CUDA_SUCCESS;
}

extern "C" CUresult cuGraphMemcpyNodeGetParams(CUgraphNode node,
                                               CUDA_MEMCPY3D *p) {
  NodeInfo *info = find_node((uint64_t)(uintptr_t)node);
  if (info == nullptr) return CUDA_ERROR_INVALID_HANDLE;
  if (info->params_rc != CUDA_SUCCESS) return info->params_rc;
  p->srcMemoryType = (CUmemorytype)info->src_kind;
  p->dstMemoryType = (CUmemorytype)info->dst_kind;
  p->WidthInBytes = info->width;
  p->Height = info->height;
  p->Depth = info->depth;
  return CUDA_SUCCESS;
}

extern "C" CUresult cuGraphMemsetNodeGetParams(CUgraphNode node,
                                               CUDA_MEMSET_NODE_PARAMS *p) {
  NodeInfo *info = find_node((uint64_t)(uintptr_t)node);
  if (info == nullptr) return CUDA_ERROR_INVALID_HANDLE;
  if (info->params_rc != CUDA_SUCCESS) return info->params_rc;
  p->elementSize = info->element;
  p->width = info->mwidth;
  p->height = info->mheight;
  return CUDA_SUCCESS;
}

extern "C" CUresult cuGraphChildGraphNodeGetGraph(CUgraphNode node,
                                                  CUgraph *child) {
  NodeInfo *info = find_node((uint64_t)(uintptr_t)node);
  if (info == nullptr) return CUDA_ERROR_INVALID_HANDLE;
  if (info->params_rc != CUDA_SUCCESS) return info->params_rc;
  if (info->child == 0) return CUDA_ERROR_INVALID_HANDLE;
  *child = (CUgraph)(uintptr_t)info->child;
  return CUDA_SUCCESS;
}

extern "C" CUresult cuFuncGetName(const char **name, CUfunction func) {
  auto it = g_func.find((uint64_t)(uintptr_t)func);
  if (it == g_func.end()) return CUDA_ERROR_INVALID_HANDLE;
  *name = it->second.c_str();
  return CUDA_SUCCESS;
}
'''


TOY_ABI2 = r'''
#include <stdint.h>
extern "C" int s5ga_abi() { return 2; }
extern "C" int s5ga_inspect(uint64_t, uint64_t, uint64_t, uint32_t, uint32_t,
                            uint64_t, char *, uint64_t, uint64_t *) {
  return 2;
}
'''

TOY_MISSING = r'''
extern "C" int s5ga_abi() { return 1; }
'''

TOY_BAD_JSON = r'''
#include <stdint.h>
#include <string.h>
extern "C" int s5ga_abi() { return 1; }
extern "C" int s5ga_inspect(uint64_t, uint64_t, uint64_t, uint32_t, uint32_t,
                            uint64_t, char *out, uint64_t cap, uint64_t *len) {
  const char text[] = "{\"schema\":";
  *len = sizeof(text) - 1;
  if (*len > cap) return 1;
  memcpy(out, text, *len);
  return 0;
}
'''


if __name__ == '__main__':
    unittest.main()
