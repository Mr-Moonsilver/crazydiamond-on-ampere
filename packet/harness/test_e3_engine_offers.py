"""CPU-only tests for the source-pinned engine offer ledger.

The fake CorrelationBridge models the reviewed ``/2`` contract (offer rows carry
both the CUPTI ``thread`` and the OS ``system_thread``) without CUPTI, CUDA or
torch.  The raw placement receipt is a separate object and is never edited by
composition; every test compares the composed result against a fresh copy.
"""
import ast
import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches' / 's5_engine'))
import engine_offers as eo  # noqa: E402

REF = ROOT / 'reference' / 'vllm-src'
MODEL_RUNNER = 'v1/worker/gpu/model_runner.py'
SAMPLER = 'v1/worker/gpu/sample/sampler.py'
ASYNC_UTILS = 'v1/worker/gpu/async_utils.py'
CUDAGRAPH_UTILS = 'v1/worker/gpu/cudagraph_utils.py'

GREEN = 41
START, END = 1_000_000, 9_000_000
IDENT = {
    'main': {'context_id': GREEN, 'stream_id': 51},
    'pp': {'context_id': GREEN, 'stream_id': 52},
    'output_copy': {'context_id': GREEN, 'stream_id': 53},
}
# A reviewed non-green output/control role: no other role may be non-green.
NON_GREEN_OUTPUT = copy.deepcopy(IDENT)
NON_GREEN_OUTPUT['output_copy'] = {
    'context_id': 77, 'stream_id': 53, 'non_green_control': True}
ENV = {eo.ENABLE_ENV: '1'}
OBSERVER = {
    'abi': 1,
    'library_sha256': 'a' * 64,
    'source_sha256': 'b' * 64,
    'cupti_library_sha256': 'c' * 64,
}
SOURCE_IDENTITIES = {
    'native_static_schema': 's5ro-native-static/1',
    'native_window_schema': 's5ro-native-window/1',
    'compiled_api': 130001,
    'runtime_api': 130001,
}
BANK = {
    'capture_order_labels': ['descriptor:0'],
    'descriptor_count': 1,
    'green_context_bound': True,
    'route_generation': 7,
    'split_decode_sms': 44,
    'split_requested_sms': 44,
    'storage_epoch': 1,
}
DESCRIPTOR = {
    'num_tokens': 128,
    'num_reqs': 8,
    'uniform_token_count': 1,
    'max_query_len': 1,
    'num_active_loras': 0,
    'num_ubatches': 1,
}
DESCRIPTOR_KEY = eo.descriptor_key(DESCRIPTOR)
CENSUS = {'kernel': 2, 'memcpy': 1, 'memset': 0}
NODE_TYPES = {
    'kernel': 2, 'memcpy': 1, 'memset': 0, 'graph': 0, 'empty': 0,
    'event': 0, 'host': 0, 'conditional': 0, 'memory': 0, 'other': 0,
    'unknown': 0,
}
LIMITS = {
    'buffer_bytes': 1048576, 'buffer_count': 4, 'max_context_records': 128,
    'max_copy_records': 32768, 'max_graph_events': 32768, 'max_graphs': 64,
    'max_kernel_records': 131072, 'max_links': 65536,
    'max_nodes_per_graph': 131072, 'max_nodes_total': 131072,
}

D2H = eo.COPY_KIND['D2H']
H2D = eo.COPY_KIND['H2D']
D2D = eo.COPY_KIND['D2D']


def make_binding(**overrides):
    values = dict(
        run_id='run-1', window_id='win-1', worker='w0', rank=0,
        worker_incarnation='worker-inc', engine_incarnation='engine:7',
        engine_epoch=3, route_generation=7, storage_epoch=1,
        green_context_id=GREEN, native_start_ns=START, native_end_ns=END,
        observer_identity=dict(OBSERVER),
        helper_identity=dict(SOURCE_IDENTITIES),
        declared_target='graph:0',
        descriptor_labels={'graph:0': DESCRIPTOR_KEY},
        router_census={'total': 1, 'graphs': {'graph:0': 1}},
        bank_identity_sha256=eo.bank_identity(BANK),
        stream_roles=copy.deepcopy(IDENT), pp_stage=0)
    values.update(overrides)
    return eo.bind_window(**values)


class Owner:
    is_first_pp_rank = True
    is_last_pp_rank = True
    pp_rank = 0
    dp_size = 1


class PPOwner:
    is_first_pp_rank = False
    is_last_pp_rank = False
    pp_rank = 1
    dp_size = 1


class FakeBridge:
    """The reviewed /2 contract, with no CUPTI and no CUDA."""

    def __init__(self, *, epoch=3, generation=7, rank=0,
                 thread_mode='cupti', pop_wrong=None, pop_error=None,
                 push_error=None, report_error=None):
        self.epoch = epoch
        self.generation = generation
        self.rank = rank
        self.thread_mode = thread_mode
        self.pop_wrong = pop_wrong
        self.pop_error = pop_error
        self.push_error = push_error
        self.report_error = report_error
        self.offers = []
        self.events = []
        self.apis = []
        self.external = []
        self.activities = []
        self.native = None
        self.next_id = 1
        self.next_correlation = 1000
        self._local = threading.local()
        self._keys = {}
        self._lock = threading.Lock()

    def _thread_keys(self):
        ident = threading.get_ident()
        with self._lock:
            if ident not in self._keys:
                index = len(self._keys) + 1
                self._keys[ident] = (1000 + index, 2000 + index)
            return self._keys[ident]

    @contextmanager
    def offer(self, family, expected, *, context_id, stream_id, graph=False,
              copy_kinds=()):
        if self.push_error is not None:
            raise self.push_error
        cupti, system = self._thread_keys()
        thread = system if self.thread_mode == 'system' else cupti
        stack = getattr(self._local, 'stack', None)
        if stack is None:
            stack = []
            self._local.stack = stack
        with self._lock:
            oid = self.next_id
            self.next_id += 1
            row = dict(id=oid, epoch=self.epoch, generation=self.generation,
                       rank=self.rank, thread=thread, system_thread=system,
                       parent=stack[-1] if stack else 0, family=family,
                       expected=dict(expected), context_id=context_id,
                       stream_id=stream_id, graph=graph,
                       copy_kinds=list(copy_kinds), closed=False,
                       body_exception=False)
            self.offers.append(row)
            self.events.append(('push', oid, row['parent'], thread, system))
        stack.append(oid)
        try:
            yield row
        except BaseException:
            row['body_exception'] = True
            raise
        finally:
            stack.pop()
            row['closed'] = True
            popped = oid if self.pop_wrong is None else self.pop_wrong
            row['popped_id'] = popped
            row['pop_rc'] = 0
            with self._lock:
                self.events.append(('pop', oid, 0, thread, system))
            if self.pop_error is not None:
                raise self.pop_error

    def record(self, kind, copy_kind=0, count=1, name=None,
               graph_id=0):
        """Submit one fake GPU work record against the innermost offer."""
        stack = getattr(self._local, 'stack', None)
        if not stack:
            return None
        oid = stack[-1]
        offer = next(row for row in self.offers if row['id'] == oid)
        with self._lock:
            self.next_correlation += 1
            correlation = self.next_correlation
        if name is None:
            name = {'kernel': 'cudaLaunchKernel',
                    'memcpy': 'cudaMemcpyAsync',
                    'memset': 'cudaMemsetAsync'}[kind]
        start = 2000 + len(self.apis) * 10
        self.apis.append({
            'domain': 'runtime', 'name': name,
            'correlation_id': correlation, 'thread': offer['thread'],
            'return_value': 0, 'start': start, 'end': start + 5,
        })
        self.external.append({'kind': 'custom0',
                              'correlation_id': correlation,
                              'external_id': oid})
        self.activities.append({
            'correlation_id': correlation, 'kind': kind,
            'context_id': offer['context_id'],
            'stream_id': offer['stream_id'], 'graph_id': graph_id,
            'copy_kind': copy_kind,
            'name': name if graph_id else name, 'count': count,
        })
        return correlation

    def report(self):
        if self.report_error is not None:
            raise self.report_error
        return {'native': self.native, 'offers': self.offers}


class NativeBuilder:
    def __init__(self, bridge, ledger=None, *, thread_type='default-pthread-low32',
                 scope_thread=None, window=None):
        self.bridge = bridge
        self.ledger = ledger
        self.thread_type = thread_type
        self.scope_thread = scope_thread
        self.window = window
        self.apis = []
        self.external = []
        self.activities = []
        self.correlation = 5000

    def _next(self):
        self.correlation += 1
        return self.correlation

    def offer(self, suboperation, family=None):
        for offer in self.bridge.offers:
            meta = self.ledger._meta[offer['id']]
            if meta['suboperation'] == suboperation and (
                    family is None or meta['family'] == family):
                return offer
        raise AssertionError('offer not found: ' + suboperation)

    def _api(self, offer, name, correlation, return_value):
        start = 3000 + len(self.apis) * 10
        self.apis.append({
            'domain': 'runtime', 'name': name,
            'correlation_id': correlation, 'thread': offer['thread'],
            'return_value': return_value, 'start': start, 'end': start + 5,
        })

    def add(self, offer, kind='kernel', graph_id=0, copy_kind=None, count=1,
            context_id=None, stream_id=None, return_value=0, api_name=None):
        meta = self.ledger._meta[offer['id']]
        correlation = self._next()
        if api_name is None:
            api_name = ('cudaGraphLaunch' if graph_id
                        else {'kernel': 'cudaLaunchKernel',
                              'memcpy': 'cudaMemcpyAsync',
                              'memset': 'cudaMemsetAsync'}[kind])
        self._api(offer, api_name, correlation, return_value)
        self.external.append({'kind': 'custom0',
                              'correlation_id': correlation,
                              'external_id': offer['id']})
        if copy_kind is None:
            declared = meta.get('copy_kinds')
            copy_kind = declared[0] if declared else D2D
        self.activities.append({
            'correlation_id': correlation, 'kind': kind,
            'context_id': meta['context_id'] if context_id is None
            else context_id,
            'stream_id': meta['stream_id'] if stream_id is None else stream_id,
            'graph_id': graph_id, 'copy_kind': copy_kind, 'name': api_name,
            'count': count,
        })
        return correlation

    def launch_only(self, offer):
        correlation = self._next()
        self._api(offer, 'cudaGraphLaunch', correlation, 0)
        self.external.append({'kind': 'custom0',
                              'correlation_id': correlation,
                              'external_id': offer['id']})
        return correlation

    def graph_launch(self, offer, census, *, graph_id=7, context_id=None,
                     stream_id=None):
        meta = self.ledger._meta[offer['id']]
        correlation = self.launch_only(offer)
        for kind in eo.KINDS:
            count = census.get(kind, 0)
            if not count:
                continue
            self.activities.append({
                'correlation_id': correlation, 'kind': kind,
                'context_id': meta['context_id'] if context_id is None
                else context_id,
                'stream_id': meta['stream_id'] if stream_id is None else stream_id,
                'graph_id': graph_id,
                'copy_kind': D2D if kind == 'memcpy' else 0,
                'name': '', 'count': count,
            })
        return correlation

    def foreign_activity(self, *, thread=99999, kind='kernel', count=1,
                         context_id=GREEN, stream_id=60):
        correlation = self._next()
        self.apis.append({
            'domain': 'runtime', 'name': 'cudaLaunchKernel',
            'correlation_id': correlation, 'thread': thread,
            'return_value': 0, 'start': 9000, 'end': 9005})
        self.activities.append({
            'correlation_id': correlation, 'kind': kind,
            'context_id': context_id, 'stream_id': stream_id, 'graph_id': 0,
            'copy_kind': 0, 'name': 'library', 'count': count,
        })
        return correlation

    def window_record(self):
        if self.window is not None:
            return dict(self.window)
        if self.ledger is not None and self.ledger.binding is not None:
            binding = self.ledger.binding
            return {'start_ns': binding.native_start_ns,
                    'end_ns': binding.native_end_ns,
                    'route_generation': binding.route_generation}
        return {'start_ns': START, 'end_ns': END, 'route_generation': 7}

    def build(self):
        scopes = []
        for action, oid, parent, thread, system in self.bridge.events:
            scope_thread = self.scope_thread or self.thread_type
            value = system if scope_thread == 'system' else thread
            scopes.append({'id': oid, 'parent': parent, 'thread': value,
                           'system_thread': system, 'action': action})
        activities = list(self.activities)
        return {
            'schema': eo.NATIVE_SCHEMA,
            'thread_id_type': self.thread_type,
            'complete': True,
            'capacity': 131072,
            'reserved_bytes': 123456,
            'errors': 0,
            'overflow': 0,
            'open_scopes': 0,
            'boundary_records': 0,
            'activity_records': sum(a['count'] for a in activities),
            'window': self.window_record(),
            # The reviewed /2 report has no aggregate window_totals field; the
            # activity rows are the only source.
            'apis': list(self.apis),
            'external': list(self.external),
            'scopes': scopes,
            'activities': activities,
        }


def make_ledger(owner, bridge, *, binding=None, identities=None, pp_stage=0,
                max_offers=eo.MAX_OFFERS, env=None, census=None):
    prior = eo._ACTIVE
    if prior is not None and prior is not getattr(owner, '_flash_s5_engine_ledger',
                                                  None):
        prior.deactivate()
    ledger = eo.EngineOfferLedger(
        bridge, worker='w0', rank=0, epoch=bridge.epoch,
        generation=bridge.generation, route_generation=bridge.generation,
        green_context_id=GREEN,
        identities=IDENT if identities is None else identities,
        pp_stage=pp_stage, census=census,
        env=ENV if env is None else env, max_offers=max_offers,
        binding=binding)
    ledger.activate(owner)
    return ledger


#: One reviewed eager record for each suboperation in the positive window.
POSITIVE_RECORDS = {
    'prep_state': ('kernel', None),
    'prepare_inputs': ('memcpy', H2D),
    'prepare_attn': ('memcpy', D2D),
    'preprocess_state': ('memcpy', D2D),
    'prepare_model_inputs': ('kernel', None),
    'select_hidden': ('kernel', None),
    'compute_full': ('kernel', None),
    'select_inputs': ('kernel', None),
    'draw': ('kernel', None),
    'completion_counts': ('memset', None),
    'async_output': ('memcpy', D2H),
    'postprocess_sampled': ('kernel', None),
    'postprocess_state': ('kernel', None),
}
POSITIVE_EAGER = sum(1 for _ in POSITIVE_RECORDS)
POSITIVE_BY_KIND = {'kernel': 8, 'memcpy': 4, 'memset': 1}


def open_positive_window(owner, ledger, *, graph_offers=1,
                         graph_census=CENSUS, pp=False, include_eager=True):
    """Open a short window with nested containers and one or two replays."""
    with ledger.step('execute'):
        if include_eager:
            eo.run(owner, 'request_state_table', 'prep_state', lambda: None)
            eo.run(owner, 'request_state_table', 'gather_batch_req_state',
                   lambda: None, branch='ordinary')
            eo.mark_zero_work(owner, 'request_state_table', 'no_work_step')
            eo.run(owner, 'input_attention', 'prepare_inputs', lambda: None)
            eo.run(owner, 'input_attention', 'prepare_attn', lambda: None,
                   branch='ordinary')
            eo.run(owner, 'input_attention', 'preprocess_state', lambda: None)
            eo.run(owner, 'forward_dispatch', 'dp_dispatch', lambda: None,
                   branch='dp1')
            eo.run(owner, 'forward_dispatch', 'prepare_model_inputs',
                   lambda: None)
            eo.run(owner, 'forward_dispatch', 'eplb_prepare', lambda: None)
            eo.run(owner, 'kv_connector', 'no_forward', lambda: None,
                   branch='noop')
            with ledger.offer_scope('logits_sampler', 'runner_sample'):
                with ledger.offer_scope('logits_sampler', 'sampler_call'):
                    with ledger.offer_scope('logits_sampler',
                                            'sampler_entry'):
                        eo.run(owner, 'sampler', 'select_inputs', lambda: None)
                        with ledger.offer_scope('logits_sampler',
                                                'sampler_sample'):
                            eo.run(owner, 'sampler', 'apply_params',
                                   lambda: None, branch='none')
                            eo.run(owner, 'sampler', 'topk_topp_inputs',
                                   lambda: None)
                            eo.run(owner, 'sampler', 'draw', lambda: None,
                                   branch='gumbel')
                            eo.run(owner, 'logits', 'select_hidden',
                                   lambda: None)
                            eo.run(owner, 'logits', 'compute_full',
                                   lambda: None)
                        eo.run(owner, 'sampler', 'completion_counts',
                               lambda: None)
                        eo.run(owner, 'sampler', 'gather_output',
                               lambda: None)
        for _ in range(graph_offers):
            with ledger.offer_scope('forward_dispatch', 'graph_replay',
                                    descriptor=DESCRIPTOR,
                                    census=graph_census):
                pass
        if pp:
            eo.run(owner, 'pp', 'pp_receive', lambda: None)
            eo.run(owner, 'pp', 'postprocess_num_computed', lambda: None)
            eo.run(owner, 'pp', 'pp_broadcast', lambda: None)
    with ledger.step('sample'):
        if include_eager:
            eo.run(owner, 'postprocess', 'postprocess_state', lambda: None)
            eo.run(owner, 'output_copy', 'async_output', lambda: None)
            with ledger.offer_scope('postprocess', 'execute_postprocess'):
                eo.run(owner, 'postprocess', 'postprocess_sampled',
                       lambda: None)
            # Branch-not-taken host fields.  These are not offers.
            eo.mark_branch(owner, 'sampler', 'nan_check', 'disabled')
            eo.mark_branch(owner, 'sampler', 'trace_apply', 'absent')
            eo.mark_branch(owner, 'sampler', 'topk_logprobs',
                           'not_requested')
            eo.mark_branch(owner, 'sampler', 'sampling_mask', 'disabled')
            eo.mark_branch(owner, 'logits', 'grammar', 'absent')
            eo.mark_branch(owner, 'sampler', 'gather_output', 'unsharded')
            eo.mark_branch(owner, 'sampler', 'shard_inputs', 'no_sharder')
            eo.mark_branch(owner, 'logits_sampler', 'sampler_call',
                           'empty_owner')


def add_positive_records(builder, *, graph_offers=1, graph_census=CENSUS,
                         include_eager=True, record_graph=True):
    if include_eager:
        for offer in builder.bridge.offers:
            sub = builder.ledger._meta[offer['id']]['suboperation']
            record = POSITIVE_RECORDS.get(sub)
            if record is None:
                continue
            kind, copy_kind = record
            builder.add(offer, kind=kind, copy_kind=copy_kind)
    if not record_graph:
        return
    graph_offer_rows = [offer for offer in builder.bridge.offers
                        if builder.ledger._meta[offer['id']]['suboperation']
                        == 'graph_replay']
    for offer in graph_offer_rows[:graph_offers]:
        builder.graph_launch(offer, graph_census)


def positive(*, binding=True, graph_offers=1, graph_census=CENSUS,
             include_eager=True, record_graph=True, pp=False, pp_stage=0,
             thread_mode='cupti',
             thread_type='default-pthread-low32', scope_thread=None,
             max_offers=eo.MAX_OFFERS, identities=None):
    owner = PPOwner() if pp else Owner()
    bridge = FakeBridge(thread_mode=thread_mode)
    identities = IDENT if identities is None else identities
    window_binding = make_binding(
        router_census={'total': graph_offers,
                       'graphs': {'graph:0': graph_offers}},
        pp_stage=pp_stage, stream_roles=copy.deepcopy(identities)) \
        if binding else None
    ledger = make_ledger(owner, bridge, binding=window_binding,
                         pp_stage=pp_stage, max_offers=max_offers,
                         identities=identities)
    open_positive_window(owner, ledger, graph_offers=graph_offers,
                         graph_census=graph_census, pp=pp,
                         include_eager=include_eager)
    builder = NativeBuilder(bridge, ledger, thread_type=thread_type,
                            scope_thread=scope_thread)
    add_positive_records(builder, graph_offers=graph_offers,
                         graph_census=graph_census,
                         include_eager=include_eager,
                         record_graph=record_graph)
    bridge.native = builder.build()
    return owner, bridge, ledger, builder


def grade(ledger, bridge, builder=None):
    native = builder.build() if builder is not None else bridge.native
    return eo.grade_engine(native, bridge.offers, ledger._meta,
                           ledger.window())


# --- raw placement receipt --------------------------------------------------
def raw_counts(engine, *, expected=None, other=None, replays=1,
               graph_census=CENSUS):
    graph = {kind: graph_census.get(kind, 0) * replays
             for kind in eo.KINDS}
    if expected is None:
        expected = engine.get('eager_green_records', 0)
    if other is None:
        other = engine.get('eager_other_records', 0)
    # The engine receipt's own partition is the whole in-window multiset; the
    # raw fixture must describe the same records, not add a second copy.
    by_kind = engine.get('engine_by_kind', {})
    kernel = by_kind.get('kernel', 0)
    copies = by_kind.get('memcpy', 0) + by_kind.get('memset', 0)
    graph_records = sum(graph.values())
    records = engine.get('joined_records', 0)
    counts = {key: 0 for key in eo._RAW_COUNT_KEYS}
    counts.update({
        'context_records': 1,
        'kernel_records': kernel,
        'copy_records': copies,
        'records_in_window': records,
        'outside_scope_records': other,
        'unattributed_expected_context': expected,
        'unattributed_other_context': other,
        'buffers_requested': 4,
        'buffers_issued': 4,
        'buffers_completed': 4,
        'buffers_issued_in_window': 4,
        'buffers_completed_in_window': 4,
        # Native graph-mapping fixpoint passes, independent of replays
        # (job9b primary.json: graph_passes=2 with replays=1).
        'graph_passes': 2,
    })
    return counts


def raw_receipt(engine, *, expected=None, other=None, replays=1,
                verdict='PASS', native_reasons=None, top_reasons=None,
                graph_census=CENSUS, qualified=None, declared_target='graph:0'):
    native_reasons = list(native_reasons or ())
    if top_reasons is None:
        top_reasons = list(native_reasons)
    if native_reasons and 'replay_incomplete' not in top_reasons:
        top_reasons = list(top_reasons) + ['replay_incomplete']
    if qualified is None:
        qualified = verdict == 'PASS'
    counts = raw_counts(engine, expected=expected, other=other,
                        replays=replays, graph_census=graph_census)
    node_types = dict(NODE_TYPES)
    for kind in eo.KINDS:
        node_types[kind] = graph_census.get(kind, 0)
    required = {kind: node_types[kind] for kind in eo.KINDS}
    executions = {kind: node_types[kind] * replays for kind in eo.KINDS}
    # Native per-node execution bounds: min/max over the nodes of each kind
    # (job9b: kernel 1/1 for 1386 kernel nodes under one replay).
    bounds = {kind: ({'min': replays, 'max': replays} if node_types[kind]
                     else {'min': 0, 'max': 0})
              for kind in eo.KINDS}
    slot_common = {
        'label': 'graph:0', 'node_count': sum(node_types.values()),
        'node_types': node_types, 'required': required,
        'unsupported': 0, 'unknown_type': 0, 'graph_id_known': True,
        'node_ids_known': True, 'static_complete': True, 'reasons': [],
    }
    static_slot = dict(slot_common)
    window_slot = dict(
        slot_common,
        descriptor='descriptor:0',
        observed=dict(required), missing={kind: 0 for kind in eo.KINDS},
        missing_nodes=[], context_mismatch_nodes=[],
        kernel_executions=executions['kernel'],
        copy_executions=executions['memcpy'],
        memset_executions=executions['memset'],
        node_execution_bounds=bounds,
        inventory_changed=False, update_seen=False, destroy_seen=False,
        mapping_ambiguous=False, exercised=True,
        replay_qualified=bool(qualified and verdict == 'PASS'),
        complete=True,
    )
    flags = {key: False for key in eo._RAW_FLAG_KEYS}
    mapping = {'graph_ids': 1, 'node_keys': sum(node_types.values()),
               'node_aliases': 0, 'alias_unknown': 0,
               'ambiguous_graph_ids': False, 'node_overlap': False}
    context = {
        'bound': True, 'context_known': True, 'expected_sms': 44,
        'records': 1, 'green_true': 1, 'green_false': 0, 'sm_known': 1,
        'sm_matching': 1, 'sm_mismatched': 0, 'is_green': True,
        'num_multiprocessors': 44, 'parent_context_present': True,
        'duplicate': False, 'contradictory': False,
    }
    binding = make_binding(router_census={
        'total': replays, 'graphs': {'graph:0': replays}})
    receipt = {
        'schema': eo.PLACEMENT_SCHEMA,
        'verdict': verdict,
        'reasons': list(top_reasons),
        'complete': verdict == 'PASS',
        'static_structure': {
            'verdict': 'PASS', 'complete': True, 'graph_count': 1,
            'reasons': [], 'graph_slots': [static_slot],
        },
        'replay': {
            'verdict': verdict,
            'window': {'start_ns': START, 'end_ns': END,
                       'route_generation': 7, 'flushed': True,
                       'flush_forced': False},
            'context': context,
            'counts': counts,
            'ring_overflows': {key: 0 for key in eo._RAW_RING_KEYS},
            'mapping': mapping,
            'context_diagnostics': {},
            'graph_slots': [window_slot],
            'flags': flags,
            'complete': not native_reasons,
            'reasons': list(native_reasons),
        },
        'coverage': {
            'scope': 'exercised descriptors only',
            'descriptor_inventory': ['descriptor:0'],
            'observed_descriptors': ['graph:0'],
            'replay_qualified': (['graph:0'] if qualified
                                 and verdict == 'PASS' else []),
            'unexercised_descriptors': [],
            'declared_target': declared_target,
        },
        'binding': dict(OBSERVER),
        'limits': dict(LIMITS),
        'reserved_bytes': 21314924,
        'bank': dict(BANK),
        'caller_proof': {
            'controller_drained': True,
            'engine_incarnation': 'engine:7',
            'relevant_consumers': 3,
            'route_generation': 7,
            'route_generation_matches': True,
            'storage_epoch': 1,
            'storage_epoch_matches': True,
            'verified_by_observer': False,
            'worker_incarnation': 'worker-inc',
            'worker_rank': 0,
        },
        'source_identities': dict(SOURCE_IDENTITIES),
        'limitations': [],
    }
    return receipt


def raw_sha(receipt):
    return hashlib.sha256(json.dumps(
        receipt, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


# --- vocabulary / transforms ------------------------------------------------
class VocabularyTests(unittest.TestCase):
    def test_all_seams_are_declared_once(self):
        self.assertEqual(set(eo.FAMILIES), {
            'request_state_table', 'input_attention', 'forward_dispatch',
            'pp', 'logits_sampler', 'logits', 'sampler', 'speculator',
            'kv_connector', 'output_copy', 'postprocess', 'diagnostics'})
        for family, body in eo.FAMILIES.items():
            self.assertTrue(body['suboperations'], family)
            for sub, decl in body['suboperations'].items():
                self.assertIn(decl['expectation'],
                              ('activity', 'activity_or_zero', 'zero_work',
                               'census', 'container', 'unsupported'),
                              (family, sub))
                self.assertIn('kinds', decl, (family, sub))
                if 'memcpy' in decl.get('kinds', ()):
                    self.assertIsNotNone(decl.get('copy_kinds'),
                                         (family, sub))

    def test_no_catch_all_and_explicit_leaves(self):
        for family in eo.FAMILIES:
            self.assertNotIn(family, ('unknown', 'other', 'catch_all',
                                      'default', 'misc'))
        self.assertEqual(eo.FAMILIES['request_state_table'][
            'suboperations']['gather_batch_req_state']['kinds'], ())
        self.assertEqual(eo.FAMILIES['forward_dispatch'][
            'suboperations']['dp_dispatch']['kinds'], ())
        for wire in ('sampler.nan_check', 'sampler.apply_params',
                     'sampler.topk_topp_inputs', 'sampler.draw',
                     'sampler.trace_apply', 'sampler.topk_logprobs',
                     'sampler.completion_counts', 'sampler.sampling_mask',
                     'logits.select_hidden', 'logits.compute_local',
                     'logits.compute_full', 'logits.tp_all_to_all',
                     'logits.grammar'):
            family, sub = wire.split('.')
            self.assertIn(sub, eo.FAMILIES[family]['suboperations'], wire)
        self.assertNotIn('compute_logits', eo.FAMILIES['sampler'][
            'suboperations'])
        for container in ('runner_sample', 'sampler_call', 'sampler_entry',
                          'sampler_sample'):
            self.assertEqual(eo.FAMILIES['logits_sampler'][
                'suboperations'][container]['expectation'], 'container')

    def test_graph_replay_is_census_not_one_kernel(self):
        decl = eo.FAMILIES['forward_dispatch']['suboperations']['graph_replay']
        self.assertEqual(decl['expectation'], 'census')
        self.assertTrue(decl['graph'])

    def test_unsupported_seams_are_named(self):
        self.assertEqual(eo.FAMILIES['forward_dispatch'][
            'suboperations']['offloader_sync']['expectation'], 'unsupported')
        self.assertEqual(eo.FAMILIES['speculator'][
            'suboperations']['propose']['expectation'], 'unsupported')
        self.assertEqual(eo.FAMILIES['sampler'][
            'suboperations']['rejection']['expectation'], 'unsupported')
        self.assertEqual(eo.FAMILIES['kv_connector'][
            'suboperations']['no_forward']['noop_branch'], 'noop')


class TransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = {
            rel: (REF / rel).read_text() for rel in (
                MODEL_RUNNER, SAMPLER, ASYNC_UTILS, CUDAGRAPH_UTILS)
        }

    def test_pins_match_reference_bytes(self):
        for rel, text in self.sources.items():
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),
                             eo.PINS[rel], rel)

    def test_every_transform_is_unique_ast_valid_and_reversible(self):
        for rel, source in self.sources.items():
            generated = eo.generate_one(rel, source)
            ast.parse(generated)
            self.assertIn(eo._MARKER[rel], generated)
            transforms = eo._transforms(rel, source)[0]
            for old, new, count in eo._transform_rows(transforms):
                # Chained transforms anchor on text an earlier transform
                # produced; generate_one's count check and the exact inverse
                # below are the uniqueness proof.
                self.assertTrue(old and new)
                self.assertGreaterEqual(count, 1)
            self.assertEqual(eo.revert_one(rel, generated, expected=source),
                             source, rel)

    def test_drift_and_double_application_refused(self):
        source = self.sources[MODEL_RUNNER]
        broken = source.replace('def execute_model(', 'def execute_modelX(', 1)
        with self.assertRaises(ValueError):
            eo.generate_one(MODEL_RUNNER, broken)
        generated = eo.generate_one(MODEL_RUNNER, source)
        with self.assertRaises(ValueError):
            eo.generate_one(MODEL_RUNNER, generated)
        with self.assertRaises(ValueError):
            eo.revert_one(SAMPLER, generated)

    def test_generated_runner_declares_every_required_seam(self):
        generated = eo.generate_one(MODEL_RUNNER,
                                    self.sources[MODEL_RUNNER])
        for family, sub in (
                ('request_state_table', 'gather_batch_req_state'),
                ('forward_dispatch', 'dp_dispatch'),
                ('input_attention', 'prepare_attn'),
                ('input_attention', 'attn_metadata'),
                ('forward_dispatch', 'prepare_inputs_embeds'),
                ('forward_dispatch', 'prepare_model_inputs'),
                ('forward_dispatch', 'pp_intermediate_copy'),
                ('sampler', 'shard_inputs'),                ('sampler', 'prompt_logprobs'),
                ('sampler', 'gather_output'),
                ('speculator', 'propose'),
                ('speculator', 'draft_writeback'),
                ('speculator', 'set_draft_tokens'),
                ('kv_connector', 'no_forward'),
                ('kv_connector', 'ec_no_forward'),
                ('postprocess', 'postprocess_sampled'),
                ('postprocess', 'postprocess_state')):
            self.assertIn('"%s", "%s"' % (family, sub), generated,
                          (family, sub))
        self.assertIn('run_graph_replay(', generated)
        self.assertIn('draft_tokens = _flash_s5_engine.run(', generated)
        self.assertIn('"empty_owner"', generated)
        self.assertIn('"unsharded"', generated)
        self.assertIn('"absent"', generated)
        self.assertIn('connector_branch', generated)

    def test_generated_sampler_has_explicit_leaves_and_preserved_calls(self):
        generated = eo.generate_one(SAMPLER, self.sources[SAMPLER])
        for family, sub in (
                ('sampler', 'select_inputs'), ('sampler', 'nan_check'),
                ('sampler', 'apply_params'), ('sampler', 'topk_topp_inputs'),
                ('sampler', 'draw'), ('sampler', 'trace_apply'),
                ('sampler', 'topk_logprobs'), ('sampler', 'completion_counts'),
                ('sampler', 'sampling_mask')):
            self.assertIn('"%s", "%s"' % (family, sub), generated)
        for literal in (
                'skip_top_k_top_p=True',
                'logprob_token_ids_state=self.logprob_token_ids_state',
                'expanded_idx_mapping=input_batch.expanded_idx_mapping',
                'max_per_req_token_ids=max_per_req_token_ids',
                'logits_mode=self.logprobs_mode in ("raw_logits", '
                '"processed_logits")',
                'self.sampling_states.get_top_k_top_p',
                'self._sample_random'):
            self.assertIn(literal, generated)
        # The absent branches are host-only records; there is no empty offer.
        self.assertIn('mark_branch', generated)
        self.assertNotIn('with _flash_s5_engine.leaf(\n            self, '
                         '"sampler", "trace_apply"', generated)
        self.assertNotIn('with _flash_s5_engine.leaf(\n            self, '
                         '"sampler", "topk_logprobs"', generated)

    def test_generated_run_arguments_do_no_work_before_the_offer(self):
        # ``run(owner, family, sub, fn, *args)`` evaluates args before the
        # push, so an argument must not index or call into device work. Only
        # reviewed host helpers are allowed; device expressions use a lambda.
        host_calls = {'self.sampler.get_logprobs_dims'}
        for rel, source in self.sources.items():
            tree = ast.parse(eo.generate_one(rel, source))
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == 'run'
                        and getattr(node.func.value, 'id', None)
                        == '_flash_s5_engine'):
                    continue
                where = (rel, node.lineno, node.args[2].value)
                exprs = list(node.args[4:]) + [
                    kw.value for kw in node.keywords if kw.arg != 'branch']
                for expr in exprs:
                    if isinstance(expr, ast.Lambda):
                        continue
                    for inner in ast.walk(expr):
                        self.assertNotIsInstance(inner, ast.Subscript, where)
                        if isinstance(inner, ast.Call):
                            self.assertIn(ast.unparse(inner.func),
                                          host_calls, where)
        runner = eo.generate_one(MODEL_RUNNER, self.sources[MODEL_RUNNER])
        self.assertIn('lambda: self.draft_tokens_handler.set_draft_tokens(',
                      runner)

    def test_generated_async_and_cudagraph_seams(self):
        async_src = eo.generate_one(ASYNC_UTILS,
                                    self.sources[ASYNC_UTILS])
        self.assertIn('"output_copy", "async_output"', async_src)
        self.assertIn('"output_copy", "ep_fault_query"', async_src)
        self.assertIn('self.sampled_token_ids = async_copy_to_np', async_src)
        cg_src = eo.generate_one(CUDAGRAPH_UTILS,
                                 self.sources[CUDAGRAPH_UTILS])
        self.assertIn('"offloader_sync"', cg_src)
        self.assertIn('get_offloader().sync_prev_onload()', cg_src)
        self.assertIn('offloader_branch', cg_src)

    def test_generated_runner_sample_executes_full_sharded_and_empty_branches(self):
        from types import SimpleNamespace as NS
        generated = eo.generate_one(MODEL_RUNNER, self.sources[MODEL_RUNNER])
        tree = ast.parse(generated)
        method = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == 'sample')
        module = ast.Module(body=[ast.ImportFrom(module='__future__',
            names=[ast.alias(name='annotations')], level=0), method], type_ignores=[])
        ast.fix_missing_locations(module)
        for sharded, empty in ((False, False), (True, False), (True, True)):
            with self.subTest(sharded=sharded, empty=empty):
                owner, bridge = Owner(), FakeBridge()
                ledger = make_ledger(owner, bridge, binding=make_binding())
                output = NS(num_sampled=object(), num_rejected=object())
                calls = []

                class Tensor:
                    def __getitem__(self, index):
                        # Pure [:, :vocab] is a view; index tensor launches a gather.
                        if not isinstance(index, tuple):
                            emit('gather')
                        return self

                tensor = Tensor()
                def emit(name):
                    calls.append(name)
                    if bridge.record('kernel') is None:
                        raise AssertionError('GPU work outside any offer: ' + name)
                    return tensor

                batch = NS(num_reqs=0 if empty else 8, num_draft_tokens=0,
                           logits_indices=object(), idx_mapping_np=[0, 1])
                global_batch = NS(num_reqs=8, num_draft_tokens=0,
                                  logits_indices=object(), idx_mapping_np=[0, 1])
                def shard(*args):
                    emit('shard')
                    return batch, object(), grammar, object()

                def sampler(*args):
                    # Same inner seam as the separately executed real Sampler test.
                    with ledger.offer_scope('sampler', 'select_inputs'):
                        emit('sample')
                    return output

                sampler.compute_nans = False
                sampler.get_logprobs_dims = lambda *a, **k: None
                owner.sampler = sampler
                owner.batch_sharder = NS(shard_sampler_inputs=shard) if sharded else None
                owner.model = NS(compute_logits=lambda *a: emit('compute_full'),
                                 compute_logits_local=lambda *a: emit('compute_local'))
                owner.structured_outputs_worker = NS(
                    apply_grammar_bitmask=lambda *a: emit('grammar'))
                owner.vocab_size, owner.device = 128, 'fake'
                owner.rejection_sampler = None
                grammar = NS(structured_output_request_ids=(), grammar_bitmask=object())
                def gather(*args, **kwargs):
                    emit('gather_output')
                    return output
                namespace = {'_flash_s5_engine': eo,
                             'all_to_all_logits': lambda *a: emit('all_to_all'),
                             'gather_sampler_output': gather}
                exec(compile(module, '<pinned-runner-sample>', 'exec'), namespace)
                with ledger.step('sample'):
                    result = namespace['sample'](owner, tensor,
                                                 global_batch if sharded else batch,
                                                 grammar)
                self.assertEqual(result, (output, output.num_sampled, output.num_rejected))
                self.assertEqual(calls.count('sample'), 0 if empty else 1)
                self.assertEqual(calls.count('compute_local'), int(sharded))
                self.assertEqual(calls.count('compute_full'), int(not sharded))
                self.assertEqual(calls.count('all_to_all'), int(sharded))
                self.assertEqual(calls.count('gather_output'), int(sharded))
                builder = NativeBuilder(bridge, ledger)
                builder.apis, builder.external = bridge.apis, bridge.external
                builder.activities = bridge.activities
                receipt = grade(ledger, bridge, builder)
                self.assertEqual(receipt['verdict'], 'PASS', receipt)
                self.assertEqual(receipt['joined_records'], len(calls))
                ledger.deactivate()

    def test_generated_sampler_executes_and_preserves_return(self):
        # A fake module namespace is enough for the transformed class; torch is
        # deliberately not imported by the test process.
        generated = eo.generate_one(SAMPLER, self.sources[SAMPLER])
        tree = ast.parse(generated)
        klass = next(node for node in tree.body
                     if isinstance(node, ast.ClassDef) and node.name == 'Sampler')
        module = ast.Module(body=[klass], type_ignores=[])
        ast.fix_missing_locations(module)

        class Tensor:
            def __init__(self, name='tensor'):
                self.name = name

            def view(self, *args):
                return self

        class FakeTorch:
            int64 = 'int64'
            float32 = 'float32'

            class device:
                pass

            @staticmethod
            def empty_like(value, dtype=None):
                return Tensor('empty')

        FakeTorch.Tensor = Tensor

        class FakeEnvs:
            VLLM_COMPUTE_NANS_IN_LOGITS = False

        class FakeSamplerOutput:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class FakeMask:
            @staticmethod
            def from_logits(*args):
                return None

        class Values:
            def __init__(self, values):
                self.np = values
                self.gpu = Tensor()

        class States:
            def __init__(self):
                self.top_k = Values(__import__('numpy').array([4]))
                self.temperature = Values(__import__('numpy').array([1.0]))
                self.seeds = Values(__import__('numpy').array([0]))
                self.min_p = Values(__import__('numpy').array([0.0]))
                self.top_p = Values(__import__('numpy').array([1.0]))

            def get_top_k_top_p(self, *args):
                return None, None

            def any_greedy(self, *args):
                return False

            def any_explicit_seed(self, *args):
                return False

        bridge_holder = {}

        class Indexed:
            def __init__(self):
                self.bridge = bridge_holder.get('bridge')

            def __getitem__(self, index):
                if self.bridge is not None:
                    self.bridge.record('kernel')
                return Tensor('selected')

            @property
            def new_ones(self):
                def make(count):
                    if self.bridge is not None:
                        self.bridge.record('memset')
                    return Tensor('ones')
                return make

        def sampled(*args):
            return Tensor('num_sampled'), Tensor('num_rejected')

        def apply_top_k_top_p(logits, top_k, top_p):
            return logits

        def gumbel_sample(*args, **kwargs):
            bridge = bridge_holder.get('bridge')
            if bridge is not None:
                bridge.record('kernel')
            return Tensor('sampled')

        def flashinfer_sample(*args):
            return Tensor('flashinfer')

        def flashinfer_supported():
            return False

        def compute_topk(*args, **kwargs):
            return Tensor('logprobs')

        def get_num_nans(*args):
            return Tensor('nans')

        namespace = {
            '__name__': 'fake_sampler',
            '__builtins__': __builtins__,
            'np': __import__('numpy'),
            'torch': FakeTorch(),
            'envs': FakeEnvs(),
            'PROCESSED_LOGPROBS_MODES': (),
            'LogprobsMode': str,
            'ReasoningConfig': object,
            'SamplingParams': object,
            'apply_top_k_top_p': apply_top_k_top_p,
            'flashinfer_sample': flashinfer_sample,
            'flashinfer_sampler_supported': flashinfer_supported,
            'InputBatch': object,
            'get_num_sampled_and_rejected': sampled,
            'get_num_nans': get_num_nans,
            'BadWordsState': object,
            'gumbel_sample': gumbel_sample,
            'LogitBiasState': object,
            'LogprobTokenIdsState': object,
            'compute_topk_scores': compute_topk,
            'SamplerOutput': FakeSamplerOutput,
            'SamplingMaskTensors': FakeMask,
            'PenaltiesState': object,
            'NO_LOGPROBS': -1,
            'SamplingStates': object,
            'ThinkingBudgetState': object,
            'TraceReplayState': object,
            'RequestState': object,
            '_flash_s5_engine': eo,
        }
        exec(compile(module, '<generated-sampler>', 'exec'), namespace)
        sampler_class = namespace['Sampler']

        owner = sampler_class.__new__(sampler_class)
        bridge = FakeBridge()
        bridge_holder['bridge'] = bridge
        binding = make_binding()
        ledger = make_ledger(owner, bridge, binding=binding)
        owner.logprobs_mode = 'raw_logprobs'
        owner.compute_nans = False
        owner.trace_replay_state = None
        owner.return_sampling_mask = False
        owner.use_flashinfer = False
        owner.use_fp64_gumbel = False
        owner.needs_logits_processing = __import__('numpy').zeros(4,
                                                                  dtype=bool)
        owner.sampling_states = States()
        owner.req_states = type('Req', (), {
            'prefill_len': Values(__import__('numpy').array([0]))})()
        owner.get_logprobs_dims = lambda idx: None
        batch = type('Batch', (), {})()
        batch.expanded_idx_mapping = Tensor()
        batch.idx_mapping = Tensor()
        batch.idx_mapping_np = __import__('numpy').zeros(3, dtype=int)
        batch.cu_num_logits_np = __import__('numpy').array([0, 1, 2, 3])
        batch.expanded_local_pos = Tensor()
        batch.positions = Indexed()
        batch.input_ids = Indexed()
        batch.logits_indices = Tensor()
        batch.seq_lens = Indexed()
        batch.num_reqs = 3
        batch.cu_num_logits = Tensor()
        with ledger.step('sample'):
            result = sampler_class.__call__(owner, Tensor('logits'), batch)
        self.assertEqual(result.sampled_token_ids.name, 'sampled')
        self.assertIs(result.__dict__['logprobs_tensors'], None)
        bridge.native = NativeBuilder(bridge, ledger).build()
        # The fake work rows have no NativeBuilder-built scope rows; use the
        # real bridge rows by rebuilding through the ledger's offer events.
        builder = NativeBuilder(bridge, ledger)
        builder.apis = list(bridge.apis)
        builder.external = list(bridge.external)
        builder.activities = list(bridge.activities)
        receipt = eo.grade_engine(builder.build(), bridge.offers,
                                  ledger._meta, ledger.window())
        self.assertEqual(receipt['verdict'], 'PASS', receipt)
        self.assertGreater(receipt['joined_records'], 0)


class BranchTests(unittest.TestCase):
    def test_gather_and_params_branches_are_source_derived(self):
        owner = Owner()
        self.assertEqual(eo.gather_branch(owner, None, True), 'dummy')
        self.assertEqual(eo.gather_branch(owner, None, False), 'ordinary')
        owner.adaptive_verification = object()
        output = type('Output', (), {'scheduled_spec_decode_tokens': [1]})()
        self.assertEqual(eo.gather_branch(owner, output, False), 'adaptive')
        owner.adaptive_verification = None
        self.assertEqual(eo.gather_branch(owner, output, False), 'ordinary')
        import numpy as np
        owner.needs_logits_processing = np.array([False, False])
        self.assertEqual(eo.params_branch(owner, np.array([0, 1])), 'none')
        owner.needs_logits_processing = np.array([False, True])
        self.assertEqual(eo.params_branch(owner, np.array([0, 1])),
                         'processing')
        del owner.needs_logits_processing
        self.assertEqual(eo.params_branch(owner, np.array([0])), 'unknown')

    def test_dp_branches_are_distinct(self):
        owner = Owner()
        self.assertEqual(eo.dp_branch(owner), 'dp1')
        owner.dp_size = 4
        self.assertEqual(eo.dp_branch(owner), 'collective')
        self.assertEqual(eo.dp_branch(owner, object()), 'reuse')

    def test_connector_identity_is_the_installed_class(self):
        def made(module, name):
            cls = type(name, (), {})
            cls.__module__ = module
            return cls()
        self.assertEqual(eo.connector_branch(made(
            'vllm.v1.worker.gpu.kv_connector', 'KVConnector')), 'noop')
        self.assertEqual(eo.connector_branch(made(
            'vllm.v1.worker.gpu.ec_connector', 'ECConnector')), 'noop')
        self.assertEqual(eo.connector_branch(made(
            'vllm.v1.worker.gpu.kv_connector', 'ActiveKVConnector')),
            'active')
        self.assertEqual(eo.connector_branch(made(
            'other', 'ECConnector')), 'active')

    def test_offloader_branch_is_explicitly_unreviewed(self):
        self.assertEqual(eo.offloader_branch(), 'stock_unreviewed')
        self.assertEqual(eo.offloader_branch(object()), 'stock_unreviewed')


class DefaultOffTests(unittest.TestCase):
    def tearDown(self):
        active = eo._ACTIVE
        if active is not None:
            active.deactivate()
        eo._CENSUS = {}

    def test_flag_off_refuses_ledger_and_keeps_helpers_inert(self):
        bridge = FakeBridge()
        owner = Owner()
        with self.assertRaises(eo.OfferDisabled):
            eo.EngineOfferLedger(
                bridge, worker='w0', rank=0, epoch=3, generation=7,
                route_generation=7, green_context_id=GREEN,
                identities=IDENT, env={})
        calls = []
        self.assertEqual(eo.run(owner, 'request_state_table', 'prep_state',
                                lambda: calls.append(1) or 9), 9)
        eo.scoped_call(owner, 'input_attention', 'prepare_inputs', None,
                       lambda: calls.append(2))
        eo.mark_zero_work(owner, 'request_state_table', 'no_work_step')
        eo.mark_branch(owner, 'sampler', 'nan_check', 'disabled')

        class Sample:
            @eo.offer_method('logits_sampler', 'sampler_entry')
            def call(self):
                return 4

            @eo.step_method('execute')
            def step(self):
                return 5

        self.assertEqual(Sample().call(), 4)
        self.assertEqual(Sample().step(), 5)
        self.assertEqual(calls, [1, 2])
        self.assertEqual(bridge.offers, [])
        self.assertIsNone(eo.current_ledger(owner))

    def test_enabled_flag_is_exact(self):
        self.assertTrue(eo._enabled({eo.ENABLE_ENV: '1'}))
        self.assertFalse(eo._enabled({eo.ENABLE_ENV: 'true'}))
        self.assertFalse(eo._enabled({eo.ENABLE_ENV: '01'}))
        self.assertFalse(eo._enabled({}))


# --- lifetime / containment -------------------------------------------------
class ContainmentTests(unittest.TestCase):
    def tearDown(self):
        active = eo._ACTIVE
        if active is not None:
            active.deactivate()
        eo._CENSUS = {}

    def test_capacity_refusal_runs_body_once_and_latches(self):
        owner = Owner()
        bridge = FakeBridge()
        ledger = make_ledger(owner, bridge, max_offers=1)
        calls = []
        with ledger.step('execute'):
            with ledger.offer_scope('request_state_table', 'prep_state'):
                calls.append('first')
            with ledger.offer_scope('request_state_table', 'prep_state'):
                calls.append('second')
            with ledger.offer_scope('request_state_table', 'prep_state'):
                calls.append('third')
        self.assertEqual(calls, ['first', 'second', 'third'])
        self.assertEqual(len(bridge.offers), 1)
        self.assertTrue(ledger.window()['invalidated'])
        self.assertTrue(any(g['kind'] == 'offer_capacity_exceeded'
                            for g in ledger.window()['gaps']))
        # Admission is sticky: no later pair is created.
        with ledger.step('sample'):
            with ledger.offer_scope('request_state_table', 'prep_state'):
                calls.append('fourth')
        self.assertEqual(len(bridge.offers), 1)
        self.assertEqual(calls[-1], 'fourth')

    def test_push_failure_is_contained_and_body_runs_once(self):
        owner = Owner()
        bridge = FakeBridge(push_error=RuntimeError('native push'))
        ledger = make_ledger(owner, bridge)
        calls = []
        with ledger.step('execute'):
            with ledger.offer_scope('request_state_table', 'prep_state') as o:
                self.assertIsNone(o)
                calls.append('body')
        self.assertEqual(calls, ['body'])
        self.assertEqual(bridge.offers, [])
        self.assertTrue(ledger.window()['invalidated'])
        self.assertTrue(any(g['kind'] == 'offer_push_failed'
                            for g in ledger.window()['gaps']))

    def test_pop_failure_is_contained_and_body_runs_once(self):
        owner = Owner()
        bridge = FakeBridge(pop_error=RuntimeError('native pop'))
        ledger = make_ledger(owner, bridge)
        calls = []
        with ledger.step('execute'):
            with ledger.offer_scope('request_state_table', 'prep_state'):
                calls.append('body')
        self.assertEqual(calls, ['body'])
        self.assertEqual(len(bridge.offers), 1)
        receipt = eo.grade_engine(
            NativeBuilder(bridge, ledger).build(), bridge.offers,
            ledger._meta, ledger.window())
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE', receipt)
        self.assertTrue(any('offer_pop' in item for item in receipt['missing']),
                        receipt)

    def test_body_exception_propagates_unchanged_even_if_cleanup_fails(self):
        owner = Owner()
        bridge = FakeBridge(pop_error=RuntimeError('cleanup'))
        ledger = make_ledger(owner, bridge)
        original = ValueError('engine body')

        def explode():
            raise original

        with ledger.step('execute'):
            with self.assertRaises(ValueError) as caught:
                with ledger.offer_scope('request_state_table', 'prep_state'):
                    explode()
        self.assertIs(caught.exception, original)
        self.assertEqual(len(bridge.offers), 1)
        # The pop failure is latched even though the bridge row says closed
        # and the body raised: invalidity never depends on bridge row flags.
        window = ledger.window()
        self.assertTrue(window['invalidated'])
        self.assertTrue(any(g['kind'] == 'offer_pop_failed'
                            for g in window['gaps']), window['gaps'])
        calls = []
        with ledger.step('execute'):
            with ledger.offer_scope('request_state_table', 'prep_state'):
                calls.append('later')
        self.assertEqual(calls, ['later'])
        self.assertEqual(len(bridge.offers), 1)

    def test_pop_identity_mismatch_is_typed_not_raised(self):
        owner = Owner()
        bridge = FakeBridge(pop_wrong=-1)
        ledger = make_ledger(owner, bridge)
        with ledger.step('execute'):
            with ledger.offer_scope('request_state_table', 'prep_state'):
                pass
        self.assertFalse(ledger._meta[bridge.offers[0]['id']]['pop_verified'])
        self.assertTrue(ledger.window()['invalidated'])
        receipt = eo.grade_engine(
            NativeBuilder(bridge, ledger).build(), bridge.offers,
            ledger._meta, ledger.window())
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE', receipt)
        self.assertIn('instrumentation_invalid', receipt['missing'])

    def test_branch_markers_are_not_offers(self):
        owner = Owner()
        bridge = FakeBridge()
        ledger = make_ledger(owner, bridge)
        eo.mark_branch(owner, 'sampler', 'nan_check', 'disabled')
        eo.mark_branch(owner, 'sampler', 'trace_apply', 'absent')
        eo.mark_branch(owner, 'sampler', 'topk_logprobs', 'not_requested')
        eo.mark_branch(owner, 'sampler', 'sampling_mask', 'disabled')
        eo.mark_branch(owner, 'logits_sampler', 'sampler_call', 'empty_owner')
        self.assertEqual(bridge.offers, [])
        self.assertEqual(ledger.window()['branches']['sampler.nan_check'],
                         {'disabled': 1})

    def test_explicit_zero_work_submits_nothing(self):
        owner = Owner()
        bridge = FakeBridge()
        ledger = make_ledger(owner, bridge)
        with ledger.step('execute'):
            eo.mark_zero_work(owner, 'request_state_table', 'no_work_step')
        self.assertEqual(len(bridge.offers), 1)
        self.assertEqual(bridge.offers[0]['expected'], {})
        self.assertEqual(bridge.activities, [])

    def test_final_controller_shutdown_is_closed_and_clean(self):
        owner = Owner()
        bridge = FakeBridge()
        ledger = make_ledger(owner, bridge)
        with ledger.step('execute'):
            with ledger.offer_scope('request_state_table', 'prep_state'):
                pass
        self.assertEqual(ledger.open_offers(), [])
        ledger.deactivate()
        self.assertFalse(ledger.active)
        self.assertIsNone(eo.current_ledger(owner))
        self.assertTrue(all(row['closed'] for row in bridge.offers))


# --- positive accounting / binding -----------------------------------------
class PositiveWindowTests(unittest.TestCase):
    def tearDown(self):
        active = eo._ACTIVE
        if active is not None:
            active.deactivate()
        eo._CENSUS = {}

    def test_join_is_nonempty_and_reconciles(self):
        owner, bridge, ledger, builder = positive()
        receipt = ledger.seal()
        self.assertEqual(receipt['verdict'], 'PASS', receipt)
        self.assertTrue(eo.validate_receipt(receipt))
        self.assertGreater(receipt['joined_records'], 0)
        self.assertEqual(receipt['eager_records'], POSITIVE_EAGER)
        self.assertEqual(receipt['graph']['joined_records'],
                         sum(CENSUS.values()))
        self.assertEqual(receipt['graph']['outside_count'], 0)
        self.assertEqual(receipt['graph']['unjoined_count'], 0)
        self.assertEqual(receipt['graph']['offers'][
            str(builder.offer('graph_replay')['id'])]['launch_count'], 1)
        self.assertEqual(receipt['schema'], eo.SCHEMA)
        self.assertNotEqual(receipt['schema'], eo.PLACEMENT_SCHEMA)
        self.assertEqual(receipt['binding'], make_binding())
        self.assertEqual(receipt['native_window']['start_ns'], START)
        self.assertIn('census_verified', ' '.join(receipt['notes']))
        for meta in receipt['declared'].values():
            self.assertNotIn('thread', meta)

    def test_bridge_thread_fields_are_the_source(self):
        owner, bridge, ledger, _ = positive()
        receipt = ledger.seal()
        for offer in receipt['offers']:
            self.assertIsInstance(offer['thread'], int)
            self.assertIsInstance(offer['system_thread'], int)
            self.assertNotEqual(offer['thread'], 0)
        self.assertEqual(receipt['thread_id_type'],
                         'default-pthread-low32')

    def test_system_gettid_contract(self):
        owner, bridge, ledger, _ = positive(
            thread_mode='system', thread_type='system-gettid')
        receipt = ledger.seal()
        self.assertEqual(receipt['verdict'], 'PASS', receipt)
        for offer in receipt['offers']:
            self.assertEqual(offer['thread'], offer['system_thread'])

    def test_scope_thread_cannot_substitute_os_key_under_default_mode(self):
        owner, bridge, ledger, _ = positive(scope_thread='system')
        receipt = ledger.seal()
        self.assertEqual(receipt['verdict'], 'FAIL', receipt['failures'])
        self.assertIn('native_scope_thread_mismatch', receipt['failures'])

    def test_two_calls_two_offers_two_replays(self):
        owner, bridge, ledger, builder = positive(
            graph_offers=2, include_eager=False)
        receipt = ledger.seal()
        self.assertEqual(receipt['verdict'], 'PASS', receipt)
        rows = [row for row in receipt['graph']['offers'].values()]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row['launch_count'] == 1 for row in rows))
        self.assertEqual(receipt['graph']['joined_records'],
                         2 * sum(CENSUS.values()))

    def test_one_offer_two_launches_refused(self):
        owner, bridge, ledger, _ = positive(graph_offers=1,
                                            include_eager=False,
                                            record_graph=False)
        offer = next(offer for offer in bridge.offers
                     if ledger._meta[offer['id']]['suboperation']
                     == 'graph_replay')
        # Add a second actual launch under the same census offer.
        builder = NativeBuilder(bridge, ledger)
        builder.graph_launch(offer, CENSUS)
        builder.graph_launch(offer, CENSUS)
        receipt = eo.grade_engine(builder.build(), bridge.offers,
                                  ledger._meta, ledger.window())
        self.assertEqual(receipt['verdict'], 'FAIL', receipt)
        self.assertIn('census_launch_multiplicity', receipt['failures'])

    def test_census_launch_without_node_rows_is_inconclusive(self):
        owner, bridge, ledger, _ = positive(graph_offers=1,
                                            include_eager=False)
        offer = next(offer for offer in bridge.offers
                     if ledger._meta[offer['id']]['suboperation']
                     == 'graph_replay')
        builder = NativeBuilder(bridge, ledger)
        builder.launch_only(offer)
        receipt = eo.grade_engine(builder.build(), bridge.offers,
                                  ledger._meta, ledger.window())
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE', receipt)
        self.assertIn('census_records_missing', receipt['missing'])

    def test_graph_only_window_passes_without_eager_records(self):
        owner, bridge, ledger, _ = positive(include_eager=False)
        receipt = ledger.seal()
        self.assertEqual(receipt['verdict'], 'PASS', receipt)
        self.assertEqual(receipt['eager_records'], 0)
        self.assertEqual(receipt['graph']['joined_records'],
                         sum(CENSUS.values()))

    def test_runner_prepare_leaves_accept_their_source_kernels(self):
        # Pinned prepare_inputs launches arange/Triton prep/gather kernels and
        # prepare_attn launches the block-table and slot-mapping kernels.
        for sub in ('prepare_inputs', 'prepare_attn'):
            self.assertIn('kernel', eo.FAMILIES['input_attention'][
                'suboperations'][sub]['kinds'], sub)
        owner, bridge, ledger, builder = positive()
        for sub in ('prepare_inputs', 'prepare_attn'):
            builder.add(builder.offer(sub), kind='kernel')
        receipt = grade(ledger, bridge, builder)
        self.assertEqual(receipt['verdict'], 'PASS', receipt)
        self.assertEqual(receipt['families']['0']['input_attention'][
            'prepare_attn'], {'memcpy': 1, 'kernel': 1})

    def test_pp_postprocess_uses_main_stream(self):
        owner, bridge, ledger, _ = positive(pp=True, pp_stage=1)
        receipt = ledger.seal()
        self.assertEqual(receipt['verdict'], 'PASS', receipt)
        for meta in receipt['declared'].values():
            if meta['suboperation'] in ('postprocess_num_computed',
                                        'postprocess_state'):
                self.assertEqual(meta['stream_role'], 'main')
                self.assertEqual(meta['pp_stage'], 1)
                self.assertEqual(meta['pp_branch'], 'middle')

    def test_pp_roles_and_main_postprocess(self):
        owner = PPOwner()
        bridge = FakeBridge()
        binding = make_binding(router_census={'total': 1,
                                              'graphs': {'graph:0': 1}},
                               pp_stage=1)
        ledger = make_ledger(owner, bridge, binding=binding, pp_stage=1)
        with ledger.step('execute'):
            eo.run(owner, 'pp', 'pp_receive', lambda: None)
            eo.run(owner, 'pp', 'pp_broadcast', lambda: None)
            with ledger.offer_scope('forward_dispatch', 'graph_replay',
                                    descriptor=DESCRIPTOR, census=CENSUS):
                pass
        with ledger.step('sample'):
            eo.run(owner, 'pp', 'postprocess_num_computed', lambda: None)
            eo.run(owner, 'pp', 'postprocess_state', lambda: None)
        builder = NativeBuilder(bridge, ledger)
        for sub in ('pp_receive', 'pp_broadcast'):
            builder.add(builder.offer(sub, 'pp'), kind='kernel')
        # postprocess_num_computed/postprocess_state are main-stream calls.
        for sub in ('postprocess_num_computed', 'postprocess_state'):
            builder.add(builder.offer(sub, 'pp'), kind='kernel',
                        stream_id=IDENT['main']['stream_id'])
        builder.graph_launch(builder.offer('graph_replay'), CENSUS)
        receipt = eo.grade_engine(builder.build(), bridge.offers,
                                  ledger._meta, ledger.window())
        self.assertEqual(receipt['verdict'], 'PASS', receipt)
        for sub in ('postprocess_num_computed', 'postprocess_state'):
            meta = ledger._meta[builder.offer(sub, 'pp')['id']]
            self.assertEqual(meta['stream_role'], 'main')
            self.assertEqual(meta['pp_stage'], 1)
            self.assertEqual(meta['pp_branch'], 'middle')

    def test_missing_pp_stage_is_inconclusive(self):
        owner = PPOwner()
        bridge = FakeBridge()
        ledger = make_ledger(owner, bridge, binding=None, pp_stage=None)
        with ledger.step('execute'):
            eo.run(owner, 'pp', 'pp_receive', lambda: None)
        builder = NativeBuilder(bridge, ledger)
        builder.add(builder.offer('pp_receive'), kind='kernel')
        receipt = eo.grade_engine(builder.build(), bridge.offers,
                                  ledger._meta, ledger.window())
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE', receipt)
        self.assertIn('pp_stage_missing', receipt['missing'])
        self.assertEqual(receipt['failures'], [])

    def test_optional_leaves_and_branches_join_leaves(self):
        owner = Owner()
        bridge = FakeBridge()
        binding = make_binding()
        ledger = make_ledger(owner, bridge, binding=binding)
        with ledger.step('execute'):
            with ledger.offer_scope('logits_sampler', 'sampler_call',
                                    branch='empty_owner'):
                pass
            eo.run(owner, 'sampler', 'shard_inputs', lambda: None,
                   branch='active')
            eo.run(owner, 'logits', 'compute_local', lambda: None)
            eo.run(owner, 'logits', 'compute_full', lambda: None)
            eo.run(owner, 'logits', 'tp_all_to_all', lambda: None)
            eo.run(owner, 'logits', 'grammar', lambda: None, branch='active')
            eo.run(owner, 'sampler', 'nan_check', lambda: None,
                   branch='computed')
            eo.run(owner, 'sampler', 'sampling_mask', lambda: None,
                   branch='enabled')
            eo.run(owner, 'sampler', 'gather_output', lambda: None,
                   branch='sharded')
            with ledger.offer_scope('forward_dispatch', 'graph_replay',
                                    descriptor=DESCRIPTOR, census=CENSUS):
                pass
        with ledger.step('sample'):
            eo.run(owner, 'sampler', 'shard_inputs', lambda: None,
                   branch='no_sharder')
            eo.run(owner, 'sampler', 'gather_output', lambda: None,
                   branch='unsharded')
        builder = NativeBuilder(bridge, ledger)
        builder.add(builder.offer('shard_inputs'), kind='memcpy',
                    copy_kind=8)
        for sub in ('compute_local', 'compute_full', 'tp_all_to_all',
                    'grammar', 'nan_check', 'sampling_mask', 'gather_output'):
            builder.add(builder.offer(sub), kind='kernel')
        builder.graph_launch(builder.offer('graph_replay'), CENSUS)
        receipt = eo.grade_engine(builder.build(), bridge.offers,
                                  ledger._meta, ledger.window())
        self.assertEqual(receipt['verdict'], 'PASS', receipt)
        # Every fake submission joins one non-container leaf; the empty-owner
        # container and the no-sharder/unsharded branches submit nothing.
        joined = {}
        for join in receipt['joins']:
            if join['graph_id']:
                continue
            meta = ledger._meta[join['offer']]
            joined.setdefault(meta['suboperation'], set()).add(
                join['kind'])
            self.assertNotEqual(meta['expectation'], 'container')
        self.assertEqual(set(joined), {
            'shard_inputs', 'compute_local', 'compute_full',
            'tp_all_to_all', 'grammar', 'nan_check', 'sampling_mask',
            'gather_output'})
        observed = {(meta['family'] + '.' + meta['suboperation']):
                    meta['branch'] for meta in ledger._meta.values()}
        self.assertEqual(observed['logits_sampler.sampler_call'],
                         'empty_owner')
        self.assertIn('active', [
            meta['branch'] for meta in ledger._meta.values()
            if meta['suboperation'] == 'shard_inputs'])
        self.assertIn('no_sharder', [
            meta['branch'] for meta in ledger._meta.values()
            if meta['suboperation'] == 'shard_inputs'])

    def test_noop_branch_with_activity_fails(self):
        owner, bridge, ledger, builder = positive()
        builder.add(builder.offer('no_forward'), kind='kernel')
        receipt = grade(ledger, bridge, builder)
        self.assertEqual(receipt['verdict'], 'FAIL', receipt)
        self.assertIn('noop_branch_has_activity', receipt['failures'])

    def test_external_without_api_fails(self):
        owner, bridge, ledger, builder = positive()
        target = builder.offer('prep_state')
        correlation = next(row['correlation_id'] for row in builder.external
                           if row['external_id'] == target['id'])
        builder.apis = [row for row in builder.apis
                        if row['correlation_id'] != correlation]
        receipt = grade(ledger, bridge, builder)
        self.assertEqual(receipt['verdict'], 'FAIL', receipt)
        self.assertIn('external_api_missing', receipt['failures'])

    def test_graph_launch_runtime_driver_wrapper_is_one_launch(self):
        owner, bridge, ledger, builder = positive(include_eager=False)
        offer = builder.offer('graph_replay')
        launch = next(row for row in builder.apis
                      if row['name'] == 'cudaGraphLaunch')
        # The driver wrapper is a second API row for the same actual launch.
        builder.apis.append({
            'domain': 'driver', 'name': 'cuGraphLaunch',
            'correlation_id': builder._next(),
            'thread': launch['thread'], 'return_value': 0,
            'start': launch['start'] + 1, 'end': launch['end'] - 1})
        builder.external.append(dict(kind='custom0',correlation_id=builder.apis[-1]['correlation_id'],external_id=offer['id']))
        receipt = grade(ledger, bridge, builder)
        self.assertEqual(receipt['verdict'], 'PASS', receipt)
        self.assertEqual(receipt['graph']['offers'][str(offer['id'])][
            'launch_count'], 1)

    def test_run_graph_replay_reads_retained_census(self):
        owner, bridge = Owner(), FakeBridge()
        binding = make_binding(
            router_census={'total': 2, 'graphs': {'graph:0': 2}},
            declared_target='graph:0')
        ledger = make_ledger(owner, bridge, binding=binding)
        key = eo.descriptor_key(DESCRIPTOR)
        eo.install_census(owner, {key: {'kernel': 1}})

        class Scheduler:
            def __init__(self, reqs):
                self.num_scheduled_tokens = {str(i): 1 for i in range(reqs)}
                self.total_num_scheduled_tokens = reqs

        class Batch:
            num_tokens = 128
            num_reqs = 8
            uniform_token_count = 1
            max_query_len = 1
            num_active_loras = 0
            num_ubatches = 1

        with ledger.step('execute'):
            eo.run_graph_replay(owner, Scheduler(5), Batch(), lambda: None)
            eo.run_graph_replay(owner, Scheduler(7), Batch(), lambda: None)
        builder = NativeBuilder(bridge, ledger)
        for offer in bridge.offers:
            builder.graph_launch(offer, {'kernel': 1})
        receipt = eo.grade_engine(builder.build(), bridge.offers,
                                  ledger._meta, ledger.window())
        self.assertEqual(receipt['verdict'], 'PASS', receipt)
        for offer in bridge.offers:
            meta = ledger._meta[offer['id']]
            self.assertEqual(meta['census'], {'kernel': 1})
            self.assertEqual(meta['descriptor']['num_tokens'], 128)
            self.assertNotIn('num_toks', meta['descriptor'])

    def test_legacy_bridge_thread_contract_is_inconclusive(self):
        owner, bridge, ledger, builder = positive()
        native = builder.build()
        native['schema'] = eo.NATIVE_SCHEMA_LEGACY
        native.pop('thread_id_type', None)
        for offer in bridge.offers:
            offer.pop('system_thread', None)
        receipt = eo.grade_engine(native, bridge.offers, ledger._meta,
                                  ledger.window())
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE', receipt)
        self.assertEqual(receipt['failures'], [])
        self.assertIn('thread_contract_missing', receipt['missing'])

    def test_dual_thread_scope_mismatch_fails(self):
        owner, bridge, ledger, builder = positive()
        native = builder.build()
        native['scopes'][0]['system_thread'] += 1
        receipt = eo.grade_engine(native, bridge.offers, ledger._meta,
                                  ledger.window())
        self.assertEqual(receipt['verdict'], 'FAIL', receipt)
        self.assertIn('native_scope_system_thread_mismatch',
                      receipt['failures'])


# --- binding and receipts ---------------------------------------------------
class CodexReviewTests(unittest.TestCase):
    def tearDown(self):
        if eo._ACTIVE is not None:
            eo._ACTIVE.deactivate()
        eo._CENSUS = {}

    def pending(self):
        bridge = FakeBridge()
        bridge.bounds = dict(start_ns=START, end_ns=0, route_generation=7)
        bridge.window = lambda: dict(bridge.bounds)
        ledger = make_ledger(Owner(), bridge, binding=make_binding(
            native_end_ns=None, router_census=None))
        return bridge, ledger

    def test_pending_binding_finalizes_only_observed_bounds(self):
        bridge, ledger = self.pending()
        before = ledger.binding.as_record()
        open_positive_window(ledger._owner, ledger)
        builder = NativeBuilder(bridge, ledger)
        add_positive_records(builder)
        ledger.deactivate()
        bridge.bounds['end_ns'] = END
        final = ledger.finish_window({'total': 1, 'graphs': {'graph:0': 1}})
        self.assertEqual({k: v for k, v in final.items()
                          if k not in ('native_end_ns', 'router_census')},
                         {k: v for k, v in before.items()
                          if k not in ('native_end_ns', 'router_census')})
        bridge.native = builder.build()
        receipt = ledger.seal()
        self.assertEqual(receipt['verdict'], 'PASS', receipt)
        with self.assertRaises(eo.OfferError):
            ledger.finish_window(final['router_census'])

    def test_finalization_refuses_active_unended_or_changed_window(self):
        bridge, ledger = self.pending()
        census = {'total': 1, 'graphs': {'graph:0': 1}}
        with self.assertRaisesRegex(eo.OfferError, 'not_ready'):
            ledger.finish_window(census)
        ledger.deactivate()
        for bounds in (
            dict(start_ns=START, end_ns=0, route_generation=7),
            dict(start_ns=START+1, end_ns=END, route_generation=7),
            dict(start_ns=START, end_ns=END, route_generation=8)):
            bridge.bounds = bounds
            with self.assertRaisesRegex(eo.OfferError, 'not_observed'):
                ledger.finish_window(census)
            self.assertIsNone(ledger.binding.native_end_ns)

    def test_pending_start_requires_native_identity(self):
        bridge, ledger = self.pending()
        ledger.deactivate()
        bridge.bounds['route_generation'] += 1
        with self.assertRaisesRegex(eo.OfferError, 'start_not_observed'):
            ledger.activate()
        self.assertFalse(ledger.active)

    def test_pending_cannot_be_graded_pass(self):
        bridge, ledger = self.pending()
        open_positive_window(ledger._owner, ledger)
        builder = NativeBuilder(bridge, ledger)
        add_positive_records(builder)
        builder.window = dict(start_ns=START, end_ns=END, route_generation=7)
        receipt = grade(ledger, bridge, builder)
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')

    def test_bad_descriptor_preserves_body_once(self):
        bridge, ledger = self.pending()
        calls = []
        result = eo.run_graph_replay(ledger._owner, None,
                                     type('BadBatch', (), {'num_tokens': 'bad'})(),
                                     lambda: calls.append('body') or 42)
        self.assertEqual((result, calls), (42, ['body']))
        self.assertTrue(ledger.window()['invalidated'])
        self.assertEqual(bridge.offers, [])

    def test_bad_branch_preserves_body_exception(self):
        bridge, ledger = self.pending()
        calls = []
        with self.assertRaisesRegex(RuntimeError, 'original'):
            with ledger.offer_scope('request_state_table', 'gather_batch_req_state',
                                    branch={'not': 'a string'}):
                calls.append('body')
                raise RuntimeError('original')
        self.assertEqual(calls, ['body'])
        self.assertTrue(ledger.window()['invalidated'])
        self.assertEqual(bridge.offers, [])

    def test_runtime_branch_note_failure_is_contained(self):
        bridge, ledger = self.pending()
        eo.mark_branch(ledger._owner, 'sampler', 'nan_check', {'bad': 1})
        self.assertTrue(ledger.window()['invalidated'])
        self.assertEqual(bridge.offers, [])

    def test_graph_wrapper_requires_its_own_external_correlation(self):
        _, bridge, ledger, builder = positive(include_eager=False)
        launch = next(a for a in builder.apis if a['name'] == 'cudaGraphLaunch')
        builder.apis.append(dict(launch, domain='driver', name='cuGraphLaunch',
                                correlation_id=builder._next(),
                                start=launch['start']+1, end=launch['end']-1))
        receipt = grade(ledger, bridge, builder)
        self.assertEqual(receipt['verdict'], 'FAIL')
        self.assertIn('unoffered_graph_submission', receipt['failures'])

    def test_one_activity_cannot_discharge_two_wrappers(self):
        _, bridge, ledger, builder = positive()
        offer = builder.offer('prep_state')
        correlation = next(x['correlation_id'] for x in builder.external
                           if x['external_id'] == offer['id'])
        api = next(x for x in builder.apis if x['correlation_id'] == correlation)
        for _ in range(2):
            cid = builder._next()
            builder.apis.append(dict(api, correlation_id=cid, domain='driver',
                                    name='cuLaunchKernel', start=api['start']+1,
                                    end=api['end']-1))
            builder.external.append(dict(kind='custom0', correlation_id=cid,
                                         external_id=offer['id']))
        receipt = grade(ledger, bridge, builder)
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('wrapper_peer_reused', receipt['missing'])

    def test_failed_zero_activity_api_stays_failure(self):
        for name, reason in [('cuDevicePrimaryCtxGetState', 'host_api_failed'),
                             ('cudaLaunchKernel', 'api_submission_failed')]:
            with self.subTest(name=name):
                _, bridge, ledger, builder = positive()
                api = builder.apis[0]
                builder.apis.append(dict(api, name=name, domain='driver',
                                        correlation_id=builder._next(), return_value=1))
                receipt = grade(ledger, bridge, builder)
                self.assertEqual(receipt['verdict'], 'FAIL')
                self.assertIn(reason, receipt['failures'])

    def test_raw_failure_survives_missing_engine_receipt(self):
        _, _, ledger, _ = positive()
        raw = raw_receipt(ledger.seal(), verdict='FAIL')
        result = eo.compose_placement(raw, None)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('raw_placement_fail', result['reasons'])

    def test_mixed_sm_mismatch_is_not_promoted_to_failure(self):
        _, _, ledger, _ = positive()
        engine = ledger.seal()
        raw = raw_receipt(engine, verdict='INCONCLUSIVE',
                          native_reasons=['unattributed_expected_context', 'sm_count'])
        raw['replay']['context']['sm_matching'] = 1
        raw['replay']['context']['sm_mismatched'] = 1
        result = eo.compose_placement(raw, engine)
        self.assertEqual(result['verdict'], 'INCONCLUSIVE', result)

    def test_native_complete_inconsistent_with_reasons_refused(self):
        _, _, ledger, _ = positive()
        engine = ledger.seal()
        raw = raw_receipt(engine, verdict='INCONCLUSIVE',
                          native_reasons=['unattributed_expected_context'])
        raw['replay']['complete'] = True
        result = eo.compose_placement(raw, engine)
        self.assertEqual(result['verdict'], 'INCONCLUSIVE')
        self.assertIn('native_complete_reasons', str(result['reasons']))


class BindingTests(unittest.TestCase):
    def tearDown(self):
        active = eo._ACTIVE
        if active is not None:
            active.deactivate()
        eo._CENSUS = {}

    def test_binding_is_required_for_pass(self):
        owner, bridge, ledger, _ = positive(binding=False)
        receipt = ledger.seal()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE', receipt)
        self.assertIn('binding_missing', receipt['missing'])
        self.assertEqual(receipt['failures'], [])

    def test_binding_cannot_arrive_after_offers(self):
        owner = Owner()
        bridge = FakeBridge()
        ledger = make_ledger(owner, bridge)
        with ledger.step('execute'):
            with ledger.offer_scope('request_state_table', 'prep_state'):
                pass
        with self.assertRaises(eo.OfferError):
            ledger.bind(make_binding())

    def test_binding_rejects_bad_identity_and_main_control(self):
        with self.assertRaises(eo.OfferError):
            make_binding(observer_identity={'abi': 1})
        with self.assertRaises(eo.OfferError):
            make_binding(bank_identity_sha256='x')
        bad = copy.deepcopy(IDENT)
        bad['main']['non_green_control'] = True
        with self.assertRaises(eo.OfferError):
            make_binding(stream_roles=bad)
        with self.assertRaises(eo.OfferError):
            make_binding(router_census={'total': 2,
                                        'graphs': {'graph:0': 1}})

    def test_engine_and_native_window_mismatch_refused(self):
        owner, bridge, ledger, builder = positive()
        native = builder.build()
        native['window']['start_ns'] += 1
        receipt = eo.grade_engine(native, bridge.offers, ledger._meta,
                                  ledger.window())
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE', receipt)
        self.assertIn('malformed_engine_evidence:native_window_binding_mismatch',
                      receipt['missing'])

    def test_validate_receipt_rejects_tampering(self):
        owner, bridge, ledger, _ = positive()
        receipt = ledger.seal()
        self.assertTrue(eo.validate_receipt(receipt))
        for mutate in (
                lambda r: r.__setitem__('joined_records',
                                        r['joined_records'] + 1),
                lambda r: r.__setitem__('verdict', 'FAIL'),
                lambda r: r.__setitem__('schema', 's5-engine-accounting/x'),
                lambda r: r.__setitem__('failures', ['wrong_context']),
                lambda r: r.__setitem__('binding', None),
                lambda r: r['native_window'].__setitem__('start_ns', 1)):
            bad = copy.deepcopy(receipt)
            mutate(bad)
            with self.assertRaises(ValueError):
                eo.validate_receipt(bad)

    def test_main_role_cannot_be_non_green_control(self):
        bad = copy.deepcopy(IDENT)
        bad['main'] = {'context_id': 6, 'stream_id': 51,
                       'non_green_control': True}
        with self.assertRaises(eo.OfferError):
            make_ledger(Owner(), FakeBridge(), identities=bad)


# --- negative joins ---------------------------------------------------------
class DrainControlTests(unittest.TestCase):
    def test_tail_event_controls_are_zero_activity_and_fail_on_error_or_activity(self):
        import activity_accounting as accounting
        names=('cudaEventCreate','cudaEventCreateWithFlags','cudaEventDestroy',
               'cuEventCreate','cuEventDestroy_v2')
        for index,name in enumerate(names):
            with self.subTest(api=name):
                self.assertEqual(accounting.api_kind(name),'host')
                owner,bridge,ledger,builder=positive()
                try:
                    cid=900000+index
                    api=dict(builder.apis[0],correlation_id=cid,name=name,return_value=0)
                    builder.apis.append(api)
                    self.assertEqual(grade(ledger,bridge,builder)['verdict'],'PASS')
                    api['return_value']=1
                    self.assertIn('host_api_failed',grade(ledger,bridge,builder)['failures'])
                    api['return_value']=0
                    row=dict(builder.activities[0],correlation_id=cid,graph_id=0)
                    builder.activities.append(row)
                    self.assertIn('unoffered_eager_activity',grade(ledger,bridge,builder)['failures'])
                    builder.external.append(dict(builder.external[0],correlation_id=cid))
                    self.assertIn('host_api_has_activity',grade(ledger,bridge,builder)['failures'])
                finally:ledger.deactivate()

class NegativeJoinTests(unittest.TestCase):
    def tearDown(self):
        active = eo._ACTIVE
        if active is not None:
            active.deactivate()
        eo._CENSUS = {}

    def assert_fail(self, receipt, reason=None):
        self.assertEqual(receipt['verdict'], 'FAIL', receipt)
        if reason is not None:
            self.assertIn(reason, receipt['failures'])

    def test_missing_external_join(self):
        owner, bridge, ledger, builder = positive()
        target = builder.offer('prepare_attn')
        builder.external = [row for row in builder.external
                            if row['external_id'] != target['id']]
        self.assert_fail(grade(ledger, bridge, builder),
                         'unoffered_eager_activity')

    def test_duplicate_external_join(self):
        owner, bridge, ledger, builder = positive()
        builder.external.append(dict(builder.external[0]))
        self.assert_fail(grade(ledger, bridge, builder),
                         'duplicate_external_join')

    def test_duplicate_activity_join(self):
        owner, bridge, ledger, builder = positive()
        builder.activities.append(dict(builder.activities[0]))
        self.assert_fail(grade(ledger, bridge, builder),
                         'duplicate_activity_join')

    def test_wrong_context_and_stream(self):
        owner, bridge, ledger, builder = positive()
        builder.activities[0]['context_id'] = 99
        self.assert_fail(grade(ledger, bridge, builder), 'wrong_context')
        owner, bridge, ledger, builder = positive()
        builder.activities[0]['stream_id'] = 99
        self.assert_fail(grade(ledger, bridge, builder), 'wrong_stream')

    def test_wrong_family_kind_and_copy_direction(self):
        owner, bridge, ledger, builder = positive()
        row = next(row for row in builder.activities if row['kind'] == 'kernel')
        row['kind'] = 'memset'
        self.assert_fail(grade(ledger, bridge, builder), 'api_activity_kind')
        owner, bridge, ledger, builder = positive()
        row = next(row for row in builder.activities if row['kind'] == 'memcpy')
        row['copy_kind'] = D2H
        self.assert_fail(grade(ledger, bridge, builder), 'copy_direction')

    def test_wrong_kind_with_matching_api(self):
        owner, bridge, ledger, builder = positive()
        target = builder.offer('async_output')
        correlation = next(row['correlation_id'] for row in builder.external
                           if row['external_id'] == target['id'])
        activity = next(row for row in builder.activities
                        if row['correlation_id'] == correlation)
        activity['kind'] = 'kernel'
        activity['name'] = 'cudaLaunchKernel'
        api = next(row for row in builder.apis
                   if row['correlation_id'] == correlation)
        api['name'] = 'cudaLaunchKernel'
        self.assert_fail(grade(ledger, bridge, builder), 'wrong_kind')

    def test_api_mismatch_is_distinct_from_family_kind(self):
        owner, bridge, ledger, builder = positive()
        target = builder.offer('async_output')
        correlation = next(row['correlation_id'] for row in builder.external
                           if row['external_id'] == target['id'])
        api = next(row for row in builder.apis
                   if row['correlation_id'] == correlation)
        api['name'] = 'cudaLaunchKernel'
        self.assert_fail(grade(ledger, bridge, builder), 'api_activity_kind')

    def test_eager_on_graph_offer_and_graph_on_eager_offer(self):
        owner, bridge, ledger, builder = positive()
        builder.add(builder.offer('graph_replay'), kind='kernel',
                    graph_id=0)
        self.assert_fail(grade(ledger, bridge, builder), 'offer_graph_scope')
        owner, bridge, ledger, builder = positive()
        builder.add(builder.offer('prep_state'), kind='kernel', graph_id=7)
        self.assert_fail(grade(ledger, bridge, builder), 'offer_graph_scope')

    def test_external_to_unknown_offer(self):
        owner, bridge, ledger, builder = positive()
        builder.external[0]['external_id'] = 999999
        self.assert_fail(grade(ledger, bridge, builder), 'unknown_offer')

    def test_cross_thread_api_join(self):
        owner, bridge, ledger, builder = positive()
        builder.apis[0]['thread'] = 77777
        self.assert_fail(grade(ledger, bridge, builder),
                         'external_thread_mismatch')

    def test_stale_generation_and_rank(self):
        owner, bridge, ledger, builder = positive()
        bridge.offers[0]['generation'] = 99
        self.assert_fail(grade(ledger, bridge, builder), 'stale_generation')
        owner, bridge, ledger, builder = positive()
        bridge.offers[0]['rank'] = 2
        self.assert_fail(grade(ledger, bridge, builder), 'offer_rank')

    def test_failed_api_with_activity(self):
        owner, bridge, ledger, builder = positive()
        builder.apis[0]['return_value'] = 1
        self.assert_fail(grade(ledger, bridge, builder),
                         'failed_api_has_activity')

    def test_graph_row_without_join_is_missing_not_eager(self):
        owner, bridge, ledger, builder = positive(graph_offers=1,
                                                  include_eager=False)
        builder.activities.append({
            'correlation_id': 0, 'kind': 'kernel', 'context_id': GREEN,
            'stream_id': 51, 'graph_id': 7, 'copy_kind': 0,
            'name': '', 'count': 1})
        receipt = grade(ledger, bridge, builder)
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE', receipt)
        self.assertIn('graph_row_without_join', receipt['missing'])

    def test_unaccounted_foreign_thread_is_inconclusive(self):
        owner, bridge, ledger, builder = positive()
        builder.foreign_activity()
        receipt = grade(ledger, bridge, builder)
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE', receipt)
        self.assertIn('unaccounted_activity_thread', receipt['missing'])
        self.assertEqual(receipt['failures'], [])

    def test_zero_work_and_container_submissions(self):
        owner, bridge, ledger, builder = positive()
        builder.add(builder.offer('no_work_step'), kind='kernel')
        self.assert_fail(grade(ledger, bridge, builder),
                         'zero_work_has_activity')
        owner, bridge, ledger, builder = positive()
        builder.add(builder.offer('execute_postprocess'), kind='kernel')
        self.assert_fail(grade(ledger, bridge, builder),
                         'container_has_activity')
        owner, bridge, ledger, builder = positive()
        target = builder.offer('no_work_step')
        correlation = builder.add(target, kind='kernel')
        builder.activities = [row for row in builder.activities
                              if row['correlation_id'] != correlation]
        self.assert_fail(grade(ledger, bridge, builder),
                         'zero_work_has_submission')

    def test_census_mismatch_and_graph_rows(self):
        owner, bridge, ledger, builder = positive(include_eager=False,
                                                  record_graph=False)
        offer = builder.offer('graph_replay')
        builder.add(offer, kind='kernel', graph_id=7)
        self.assert_fail(grade(ledger, bridge, builder), 'census_mismatch')
        owner, bridge, ledger, builder = positive(include_eager=False,
                                                  record_graph=False)
        offer = builder.offer('graph_replay')
        builder.graph_launch(offer, CENSUS)
        receipt = grade(ledger, bridge, builder)
        self.assertEqual(receipt['verdict'], 'PASS', receipt)

    def test_activity_census_tamper_and_overflow(self):
        owner, bridge, ledger, builder = positive()
        native = builder.build()
        native['activity_records'] += 1
        receipt = eo.grade_engine(native, bridge.offers, ledger._meta,
                                  ledger.window())
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE', receipt)
        self.assertIn('activity_census', receipt['missing'])
        native = builder.build()
        native['overflow'] = 1
        receipt = eo.grade_engine(native, bridge.offers, ledger._meta,
                                  ledger.window())
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE', receipt)
        self.assertIn('correlation_overflow', receipt['missing'])

    def test_zero_work_has_no_kinds(self):
        for family, body in eo.FAMILIES.items():
            for sub, decl in body['suboperations'].items():
                if decl['expectation'] == 'zero_work':
                    self.assertEqual(decl['kinds'], (), (family, sub))

    def test_unsupported_branch_is_inconclusive_and_never_pass(self):
        owner = Owner()
        bridge = FakeBridge()
        ledger = make_ledger(owner, bridge, binding=make_binding())
        with ledger.step('sample'):
            eo.run(owner, 'speculator', 'propose', lambda: None)
            with ledger.offer_scope('forward_dispatch', 'graph_replay',
                                    descriptor=DESCRIPTOR, census=CENSUS):
                pass
        builder = NativeBuilder(bridge, ledger)
        builder.add(builder.offer('propose'), kind='kernel')
        builder.graph_launch(builder.offer('graph_replay'), CENSUS)
        receipt = eo.grade_engine(builder.build(), bridge.offers,
                                  ledger._meta, ledger.window())
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE', receipt)
        self.assertIn('unsupported_coverage:speculator.propose',
                      receipt['missing'])

    def test_active_connector_branch_is_unqualified(self):
        owner = Owner()
        bridge = FakeBridge()
        ledger = make_ledger(owner, bridge)
        with ledger.step('execute'):
            eo.run(owner, 'kv_connector', 'no_forward', lambda: None,
                   branch='active')
        builder = NativeBuilder(bridge, ledger)
        builder.add(builder.offer('no_forward'), kind='kernel')
        receipt = eo.grade_engine(builder.build(), bridge.offers,
                                  ledger._meta, ledger.window())
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE', receipt)
        self.assertIn('branch_unqualified:kv_connector.no_forward=active',
                      receipt['missing'])


# --- composition ------------------------------------------------------------
class CompositionTests(unittest.TestCase):
    def tearDown(self):
        active = eo._ACTIVE
        if active is not None:
            active.deactivate()
        eo._CENSUS = {}

    def graph_only(self, **kwargs):
        return positive(include_eager=False, **kwargs)

    def test_raw_pass_composes_pass_and_keeps_raw_unchanged(self):
        owner, bridge, ledger, _ = self.graph_only()
        engine = ledger.seal()
        self.assertEqual(engine['verdict'], 'PASS', engine)
        raw = raw_receipt(engine)
        before = copy.deepcopy(raw)
        composed = eo.compose_placement(raw, engine)
        self.assertEqual(composed['verdict'], 'PASS', composed)
        self.assertEqual(composed['discharge']['origin'], 'raw_pass')
        self.assertEqual(raw, before)
        self.assertEqual(composed['raw']['sha256'], raw_sha(raw))

    def test_raw_inconclusive_expected_is_discharged_only_with_proof(self):
        owner, bridge, ledger, _ = positive()
        engine = ledger.seal()
        self.assertEqual(engine['verdict'], 'PASS', engine)
        expected = engine['eager_green_records']
        self.assertEqual(expected, POSITIVE_EAGER)
        raw = raw_receipt(
            engine, expected=expected, verdict='INCONCLUSIVE',
            native_reasons=['unattributed_expected_context'],
            top_reasons=['unattributed_expected_context',
                         'replay_incomplete'],
            qualified=False)
        before = copy.deepcopy(raw)
        composed = eo.compose_placement(raw, engine)
        self.assertEqual(composed['verdict'], 'PASS', composed)
        self.assertEqual(composed['discharge']['origin'],
                         'discharged_eager_expected_context')
        self.assertEqual(composed['discharge']['expected_context_records'],
                         expected)
        self.assertEqual(composed['discharge']['derived_reasons'],
                         ['replay_incomplete'])
        self.assertEqual(raw, before)

    def test_raw_pass_with_expected_context_refused(self):
        owner, bridge, ledger, _ = positive()
        engine = ledger.seal()
        raw = raw_receipt(engine, expected=POSITIVE_EAGER, verdict='PASS')
        composed = eo.compose_placement(raw, engine)
        self.assertEqual(composed['verdict'], 'INCONCLUSIVE', composed)

    def test_raw_other_context_is_not_dischargeable(self):
        owner, bridge, ledger, _ = positive(identities=NON_GREEN_OUTPUT)
        engine = ledger.seal()
        self.assertEqual(engine['eager_other_records'], 1)
        raw = raw_receipt(
            engine, other=1, verdict='INCONCLUSIVE',
            native_reasons=['unattributed_other_context'],
            top_reasons=['unattributed_other_context'], qualified=False)
        composed = eo.compose_placement(raw, engine)
        self.assertEqual(composed['verdict'], 'INCONCLUSIVE', composed)
        self.assertIn('placement_reason_not_dischargeable',
                      composed['reasons'][0])

    def test_wrong_context_fail_dominates(self):
        owner, bridge, ledger, _ = positive()
        engine = ledger.seal()
        raw = raw_receipt(engine, expected=POSITIVE_EAGER,
                          verdict='INCONCLUSIVE',
                          native_reasons=['unattributed_expected_context'],
                          top_reasons=['unattributed_expected_context'])
        raw['replay']['flags']['known_wrong_context'] = True
        raw['replay']['counts']['known_wrong_context_records'] = 1
        composed = eo.compose_placement(raw, engine)
        self.assertEqual(composed['verdict'], 'FAIL', composed)
        self.assertIn('placement_wrong_context:known_wrong_context',
                      composed['reasons'])

    def test_forged_raw_pass_is_refused(self):
        owner, bridge, ledger, _ = positive()
        engine = ledger.seal()
        raw = raw_receipt(engine, expected=POSITIVE_EAGER,
                          verdict='INCONCLUSIVE',
                          native_reasons=['unattributed_expected_context'])
        raw['verdict'] = 'PASS'
        raw['complete'] = True
        raw['reasons'] = []
        raw['replay']['verdict'] = 'PASS'
        composed = eo.compose_placement(raw, engine)
        self.assertEqual(composed['verdict'], 'INCONCLUSIVE', composed)

    def test_raw_unknown_reason_refused(self):
        owner, bridge, ledger, _ = positive()
        engine = ledger.seal()
        raw = raw_receipt(
            engine, expected=POSITIVE_EAGER, verdict='INCONCLUSIVE',
            native_reasons=['unattributed_expected_context'],
            top_reasons=['unattributed_expected_context',
                         'brand_new_condition'], qualified=False)
        composed = eo.compose_placement(raw, engine)
        self.assertEqual(composed['verdict'], 'INCONCLUSIVE', composed)

    def test_raw_derived_condition_without_positive_expected_refused(self):
        owner, bridge, ledger, _ = positive(include_eager=False)
        engine = ledger.seal()
        self.assertEqual(engine['eager_records'], 0)
        raw = raw_receipt(
            engine, expected=0, verdict='INCONCLUSIVE',
            native_reasons=['unattributed_expected_context'],
            top_reasons=['unattributed_expected_context',
                         'replay_incomplete'], qualified=False)
        composed = eo.compose_placement(raw, engine)
        self.assertEqual(composed['verdict'], 'INCONCLUSIVE', composed)
        self.assertIn('placement_expected_count_not_positive',
                      composed['reasons'])

    def test_raw_missing_node_refused(self):
        owner, bridge, ledger, _ = positive()
        engine = ledger.seal()
        raw = raw_receipt(engine, expected=POSITIVE_EAGER,
                          verdict='INCONCLUSIVE',
                          native_reasons=['unattributed_expected_context'],
                          qualified=False)
        slot = raw['replay']['graph_slots'][0]
        slot['missing']['kernel'] = 1
        slot['observed']['kernel'] -= 1
        slot['missing_nodes'] = ['node:0:0']
        composed = eo.compose_placement(raw, engine)
        self.assertEqual(composed['verdict'], 'INCONCLUSIVE', composed)
        self.assertIn('slot_missing_node', composed['reasons'][0])

    def test_raw_stale_identity_refused(self):
        owner, bridge, ledger, _ = positive()
        engine = ledger.seal()
        raw = raw_receipt(engine, expected=POSITIVE_EAGER,
                          verdict='INCONCLUSIVE',
                          native_reasons=['unattributed_expected_context'],
                          qualified=False)
        raw['binding']['library_sha256'] = 'd' * 64
        composed = eo.compose_placement(raw, engine)
        self.assertEqual(composed['verdict'], 'INCONCLUSIVE', composed)
        self.assertIn('observer_identity_mismatch', composed['reasons'][0])

    def test_raw_changed_count_refused(self):
        owner, bridge, ledger, _ = positive()
        engine = ledger.seal()
        raw = raw_receipt(engine, expected=POSITIVE_EAGER,
                          verdict='INCONCLUSIVE',
                          native_reasons=['unattributed_expected_context'],
                          qualified=False)
        raw['replay']['counts']['kernel_records'] += 1
        composed = eo.compose_placement(raw, engine)
        self.assertEqual(composed['verdict'], 'INCONCLUSIVE', composed)
        self.assertIn('records_by_kind', composed['reasons'][0])

    def test_raw_incomplete_drain_refused(self):
        owner, bridge, ledger, _ = positive()
        engine = ledger.seal()
        raw = raw_receipt(engine, expected=POSITIVE_EAGER,
                          verdict='INCONCLUSIVE',
                          native_reasons=['unattributed_expected_context'],
                          qualified=False)
        raw['caller_proof']['controller_drained'] = False
        composed = eo.compose_placement(raw, engine)
        self.assertEqual(composed['verdict'], 'INCONCLUSIVE', composed)
        self.assertIn('drain_not_completed', composed['reasons'][0])

    def test_raw_declared_target_absent_uses_binding(self):
        owner, bridge, ledger, _ = self.graph_only()
        engine = ledger.seal()
        raw = raw_receipt(engine, declared_target=None)
        composed = eo.compose_placement(raw, engine)
        self.assertEqual(composed['verdict'], 'PASS', composed)

    def test_engine_graph_offer_missing_refused(self):
        owner, bridge, ledger, _ = self.graph_only()
        engine = copy.deepcopy(ledger.seal())
        engine['graph']['offers'] = {}
        composed = eo.compose_placement(raw_receipt(engine), engine)
        self.assertEqual(composed['verdict'], 'INCONCLUSIVE', composed)
        self.assertIn('engine_graph_offers', composed['reasons'][0])

    def test_two_offers_two_replays_compose(self):
        owner, bridge, ledger, builder = positive(
            include_eager=False, graph_offers=2)
        engine = ledger.seal()
        self.assertEqual(engine['verdict'], 'PASS', engine)
        raw = raw_receipt(engine, replays=2)
        composed = eo.compose_placement(raw, engine)
        self.assertEqual(composed['verdict'], 'PASS', composed)
        self.assertEqual(composed['graph']['offers_by_label'],
                         {'graph:0': 2})

    def test_engine_inconclusive_or_fail_never_composes_pass(self):
        owner, bridge, ledger, _ = positive()
        engine = ledger.seal()
        raw = raw_receipt(engine)
        engine_bad = copy.deepcopy(engine)
        engine_bad['verdict'] = 'INCONCLUSIVE'
        engine_bad['missing'] = ['x']
        self.assertEqual(
            eo.compose_placement(raw, engine_bad)['verdict'],
            'INCONCLUSIVE')
        engine_fail = copy.deepcopy(engine)
        engine_fail['verdict'] = 'FAIL'
        engine_fail['failures'] = ['wrong_context']
        self.assertEqual(eo.compose_placement(raw, engine_fail)['verdict'],
                         'FAIL')

    def test_engine_receipt_forgery_is_refused(self):
        owner, bridge, ledger, _ = positive()
        engine = ledger.seal()
        forged = copy.deepcopy(engine)
        forged['failures'] = ['wrong_context']
        composed = eo.compose_placement(raw_receipt(engine), forged)
        self.assertEqual(composed['verdict'], 'INCONCLUSIVE', composed)
        self.assertIn('engine_receipt_invalid', composed['reasons'])

    def test_wrong_raw_schema_and_malformed_refused(self):
        owner, bridge, ledger, _ = positive()
        engine = ledger.seal()
        raw = raw_receipt(engine)
        raw['schema'] = 'other'
        self.assertEqual(eo.compose_placement(raw, engine)['verdict'],
                         'INCONCLUSIVE')
        raw = raw_receipt(engine)
        raw['replay']['counts'].pop('graph_passes')
        composed = eo.compose_placement(raw, engine)
        self.assertEqual(composed['verdict'], 'INCONCLUSIVE', composed)

    def test_owned_other_context_output_is_reconciled_but_not_pass(self):
        owner, bridge, ledger, _ = positive(identities=NON_GREEN_OUTPUT)
        engine = ledger.seal()
        self.assertEqual(engine['verdict'], 'PASS', engine)
        self.assertEqual(engine['eager_other_records'], 1)
        self.assertEqual(engine['eager_green_records'], POSITIVE_EAGER - 1)
        raw = raw_receipt(
            engine, other=1, replays=1, verdict='INCONCLUSIVE',
            native_reasons=['unattributed_other_context'],
            top_reasons=['unattributed_other_context'], qualified=False)
        composed = eo.compose_placement(raw, engine)
        # Owned other-context output is reconciled, not discharged: only a sole
        # positive unattributed_expected_context may licence PASS.
        self.assertEqual(composed['verdict'], 'INCONCLUSIVE', composed)
        self.assertEqual(composed['reasons'], [
            'placement_reason_not_dischargeable:'
            'unattributed_other_context'])

    @staticmethod
    def eager_inc(engine, **kwargs):
        """Raw eager-only INCONCLUSIVE shaped like replay_observer's wrapper."""
        values = dict(
            verdict='INCONCLUSIVE',
            native_reasons=['unattributed_expected_context'],
            top_reasons=['unattributed_expected_context', 'replay_incomplete'],
            qualified=False)
        values.update(kwargs)
        raw = raw_receipt(engine, **values)
        raw['replay']['complete'] = False
        return raw

    def test_graph_counters_use_native_semantics(self):
        # graph_passes counts mapping passes, never replays.
        owner, bridge, ledger, _ = self.graph_only()
        engine = ledger.seal()
        raw = raw_receipt(engine)
        raw['replay']['counts']['graph_passes'] = 5
        self.assertEqual(eo.compose_placement(raw, engine)['verdict'], 'PASS')
        # Per-node bounds are replays per node: a duplicated node cannot
        # compensate for a missing one even when the totals still match.
        raw = raw_receipt(engine)
        raw['replay']['graph_slots'][0]['node_execution_bounds'][
            'kernel'] = {'min': 0, 'max': 2}
        composed = eo.compose_placement(raw, engine)
        self.assertEqual(composed['verdict'], 'INCONCLUSIVE', composed)
        self.assertIn('slot_multiplicity', composed['reasons'][0])
        # Two replays: every node twice, totals node_types x 2.
        owner, bridge, ledger, _ = positive(include_eager=False,
                                            graph_offers=2)
        engine = ledger.seal()
        raw = raw_receipt(engine, replays=2)
        self.assertEqual(raw['replay']['graph_slots'][0][
            'node_execution_bounds']['kernel'], {'min': 2, 'max': 2})
        self.assertEqual(eo.compose_placement(raw, engine)['verdict'], 'PASS')

    def test_eager_discharge_reconciles_exact_count_and_context(self):
        owner, bridge, ledger, _ = positive()
        engine = ledger.seal()
        n = engine['eager_green_records']
        raw = self.eager_inc(engine)
        self.assertEqual(eo.compose_placement(raw, engine)['verdict'], 'PASS')
        # Same raw (N), engine accounting one fewer record (N-1): refused.
        owner, bridge, ledger, builder = positive()
        target = builder.offer('prepare_inputs')
        corr = {row['correlation_id'] for row in builder.external
                if row['external_id'] == target['id']}
        # Remove the whole submission (API, join and record), so the engine
        # receipt itself is internally complete with N-1 eager records.
        for attr in ('activities', 'apis', 'external'):
            setattr(builder, attr, [row for row in getattr(builder, attr)
                                    if row['correlation_id'] not in corr])
        bridge.native = builder.build()
        short = ledger.seal()
        self.assertEqual(short['verdict'], 'PASS', short)
        self.assertEqual(short['eager_green_records'], n - 1)
        composed = eo.compose_placement(raw, short)
        self.assertEqual(composed['verdict'], 'INCONCLUSIVE', composed)
        # One eager record in the primary context: FAIL dominates raw INC.
        owner, bridge, ledger, builder = positive()
        target = builder.offer('prepare_model_inputs')
        corr = next(row['correlation_id'] for row in builder.external
                    if row['external_id'] == target['id'])
        next(row for row in builder.activities
             if row['correlation_id'] == corr)['context_id'] = 1
        bridge.native = builder.build()
        primary = ledger.seal()
        self.assertIn('wrong_context', primary['failures'])
        self.assertEqual(eo.compose_placement(raw, primary)['verdict'],
                         'FAIL')
        # A raw FAIL is never discharged, even with complete accounting.
        failed = self.eager_inc(engine)
        failed['verdict'] = failed['replay']['verdict'] = 'FAIL'
        failed['reasons'] = ['caller_route_generation_mismatch'] + \
            failed['reasons']
        self.assertEqual(eo.compose_placement(failed, engine)['verdict'],
                         'FAIL')
        # Any other native root reason refuses, replay_incomplete included.
        for native in (['unattributed_expected_context', 'missing_execution'],
                       ['replay_incomplete']):
            other = self.eager_inc(engine, native_reasons=native)
            other['reasons'] = list(dict.fromkeys(native + ['replay_incomplete']))
            composed = eo.compose_placement(other, engine)
            self.assertEqual(composed['verdict'], 'INCONCLUSIVE', native)
            self.assertIn('placement_reason_not_dischargeable',
                          composed['reasons'][0])

    def test_owned_other_context_with_eager_expected_discharges(self):
        # job9b shape: native root is only unattributed_expected_context while
        # unattributed_other_context is a counter, not a reason.
        owner, bridge, ledger, _ = positive(identities=NON_GREEN_OUTPUT)
        engine = ledger.seal()
        self.assertEqual(engine['eager_other_records'], 1)
        composed = eo.compose_placement(self.eager_inc(engine, other=1),
                                        engine)
        self.assertEqual(composed['verdict'], 'PASS', composed)
        self.assertEqual(composed['reconciliation']['eager_other_records'], 1)

    def test_binding_without_role_is_inconclusive_not_fail(self):
        owner = Owner()
        bridge = FakeBridge()
        binding = make_binding(stream_roles=None)
        ledger = make_ledger(owner, bridge, binding=binding,
                             identities=NON_GREEN_OUTPUT)
        open_positive_window(owner, ledger)
        builder = NativeBuilder(bridge, ledger)
        add_positive_records(builder)
        bridge.native = builder.build()
        engine = ledger.seal()
        self.assertEqual(engine['verdict'], 'PASS', engine)
        composed = eo.compose_placement(self.eager_inc(engine, other=1),
                                        engine)
        self.assertEqual(composed['verdict'], 'INCONCLUSIVE', composed)
        self.assertEqual(composed['reasons'],
                         ['placement_refused:other_context_role_undeclared'])

    def test_unowned_other_context_is_fail(self):
        owner, bridge, ledger, _ = positive(identities=NON_GREEN_OUTPUT)
        engine = ledger.seal()
        # Move one output-copy join to an undeclared context while leaving the
        # declared role unchanged: ownership evidence must reject it.
        join = next(row for row in engine['joins']
                    if row['stream_role'] == 'output_copy')
        join['context_id'] = 99
        raw = raw_receipt(
            engine, verdict='INCONCLUSIVE',
            native_reasons=['unattributed_expected_context'],
            top_reasons=['unattributed_expected_context',
                         'replay_incomplete'], qualified=False)
        composed = eo.compose_placement(raw, engine)
        self.assertEqual(composed['verdict'], 'FAIL', composed)
        self.assertIn('placement_refused:other_context_identity',
                      composed['reasons'])


if __name__ == '__main__':
    unittest.main()
