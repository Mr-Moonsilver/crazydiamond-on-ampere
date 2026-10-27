"""CPU-only support for the native prefill offer and overlap tests.

This is a controlled stand-in for the native CUPTI sink and the CUDA event
provider, not native evidence.  It records exactly the ``s5ro-correlation/2``
shape the graders consume and lets a test drive the real ``LaneDispatcher``,
``LaneForward`` and the installed source-owner seams.
"""
import contextlib
from collections import Counter
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))

from e3_types import TransactionId  # noqa: E402
import engine_offers  # noqa: E402
import model_hooks  # noqa: E402
import native_prefill_offers as npo  # noqa: E402
from lane_dispatch import LaneDispatcher  # noqa: E402

NATIVE_SCHEMA = 's5ro-correlation/2'
MAIN_CTX = 41
DECODE_CTX = 40
MAIN_STREAM = 51
COPY_STREAM = 52
SE_STREAM = 53
PLE_STREAM = 54
DECODE_STREAM = 61
THREAD = 1101
SYSTEM_THREAD = 7001
TXN = TransactionId(1, 'prefill-request', 0, 0, 8, 1)


class FakeEvent:
    def __init__(self, enable_timing=False, ready=True):
        self.enable_timing = enable_timing
        self.stream = None
        self.ready = ready

    def record(self, stream=None):
        self.stream = stream
        self.ready = True

    def query(self):
        return bool(self.ready)


class FakeStream:
    def __init__(self, tag, context_id, stream_id):
        self.tag = tag
        self.context_id = context_id
        self.stream_id = stream_id
        self.cuda_stream = stream_id
        self.waited = []

    def wait_event(self, event):
        self.waited.append(event)


class FakeCuda:
    def __init__(self):
        self.events = []
        self.current = None

    def Event(self, enable_timing=False):
        event = FakeEvent(enable_timing)
        self.events.append(event)
        return event

    @contextlib.contextmanager
    def stream(self, stream):
        previous = self.current
        self.current = stream
        try:
            yield stream
        finally:
            self.current = previous

    def current_stream(self):
        return self.current

    def set_events_ready(self, ready):
        for event in self.events:
            event.ready = bool(ready)


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()

    @contextlib.contextmanager
    def inference_mode(self):
        yield


class FakeBridge:
    """Offer creator and native record sink with the bridge's public shape."""

    def __init__(self, *, start_ns=None, route_generation=3, epoch=1,
                 generation=3, rank=0, capacity=65536,
                 reserved_bytes=1024, thread=THREAD,
                 system_thread=SYSTEM_THREAD):
        self.start_ns = time.monotonic_ns() if start_ns is None else start_ns
        self.end_ns = 0
        self.route_generation = route_generation
        self.epoch = epoch
        self.generation = generation
        self.rank = rank
        self.capacity = capacity
        self.reserved_bytes = reserved_bytes
        self.thread = thread
        self.system_thread = system_thread
        self.offers = []
        self.scopes = []
        self.apis = []
        self.external = []
        self.activities = []
        self.next_id = 1
        self.next_cid = 500
        self._stack = []
        self.overflow = 0
        self.errors = 0
        self.open_scopes = 0
        self.boundary_records = 0
        self.complete = True

    # -- offers ---------------------------------------------------------
    @contextlib.contextmanager
    def offer(self, family, expected, *, context_id, stream_id, graph=False,
              copy_kinds=()):
        oid = self.next_id
        self.next_id += 1
        parent = self._stack[-1] if self._stack else 0
        row = {
            'id': oid, 'epoch': self.epoch, 'generation': self.generation,
            'rank': self.rank, 'thread': self.thread,
            'system_thread': self.system_thread, 'parent': parent,
            'family': family, 'expected': dict(expected),
            'context_id': context_id, 'stream_id': stream_id,
            'graph': bool(graph), 'copy_kinds': list(copy_kinds),
            'closed': False, 'body_exception': None,
        }
        self.offers.append(row)
        self.scopes.append({'id': oid, 'parent': parent,
                            'thread': self.thread,
                            'system_thread': self.system_thread,
                            'action': 'push'})
        self._stack.append(oid)
        try:
            yield row
        except BaseException:
            row['body_exception'] = True
            raise
        finally:
            self._stack.pop()
            self.scopes.append({'id': oid, 'parent': parent,
                                'thread': self.thread,
                                'system_thread': self.system_thread,
                                'action': 'pop'})
            row['closed'] = True

    # -- activity -------------------------------------------------------
    def activity(self, kind, *, stream, count=1, copy_kind=0, offer=None,
                 name=None, context_id=None, stream_id=None,
                 return_value=0, thread=None, graph_id=0):
        names = {'kernel': 'cudaLaunchKernel',
                 'memcpy': 'cudaMemcpyAsync',
                 'memset': 'cudaMemsetAsync'}
        name = name or names[kind]
        context_id = stream.context_id if context_id is None else context_id
        stream_id = stream.stream_id if stream_id is None else stream_id
        thread = self.thread if thread is None else thread
        cid = self.next_cid
        self.next_cid += 1
        self.apis.append({'correlation_id': cid, 'domain': 'runtime',
                          'name': name, 'start': 200 + cid,
                          'end': 201 + cid, 'thread': thread,
                          'return_value': return_value})
        oid = offer if offer is not None else (
            self._stack[-1] if self._stack else None)
        if oid is not None:
            self.external.append({'kind': 'custom0',
                                  'correlation_id': cid,
                                  'external_id': oid})
        self.activities.append({'correlation_id': cid, 'kind': kind,
                                'count': count, 'graph_id': graph_id,
                                'context_id': context_id,
                                'stream_id': stream_id,
                                'copy_kind': copy_kind, 'name': name})
        return cid

    def unoffered(self, kind='kernel', *, stream=None, name=None):
        stream = stream or MAIN
        names = {'kernel': 'cudaLaunchKernel',
                 'memcpy': 'cudaMemcpyAsync',
                 'memset': 'cudaMemsetAsync'}
        cid = self.next_cid
        self.next_cid += 1
        self.apis.append({'correlation_id': cid, 'domain': 'runtime',
                          'name': name or names[kind], 'start': 200 + cid,
                          'end': 201 + cid, 'thread': self.thread,
                          'return_value': 0})
        self.activities.append({'correlation_id': cid, 'kind': kind,
                                'count': 1, 'graph_id': 0,
                                'context_id': stream.context_id,
                                'stream_id': stream.stream_id,
                                'copy_kind': 0,
                                'name': name or names[kind]})
        return cid

    def host_control(self, name='cudaEventRecord', return_value=0):
        cid = self.next_cid
        self.next_cid += 1
        self.apis.append({'correlation_id': cid, 'domain': 'runtime',
                          'name': name, 'start': 200 + cid,
                          'end': 200 + cid, 'thread': self.thread,
                          'return_value': return_value})
        return cid

    # -- report ---------------------------------------------------------
    def window(self):
        return {'start_ns': self.start_ns, 'end_ns': self.end_ns,
                'route_generation': self.route_generation}

    def close_window(self, end_ns=None):
        if end_ns is None:
            end_ns = time.monotonic_ns()
        if type(end_ns) is not int or end_ns <= self.start_ns:
            raise ValueError('bad window end')
        self.end_ns = end_ns

    def report(self):
        by_kind = Counter()
        for row in self.activities:
            by_kind[row['kind']] += row['count']
        native = {
            'schema': NATIVE_SCHEMA,
            'thread_id_type': 'default-pthread-low32',
            'complete': self.complete,
            'capacity': self.capacity,
            'reserved_bytes': self.reserved_bytes,
            'errors': self.errors,
            'overflow': self.overflow,
            'open_scopes': self.open_scopes,
            'boundary_records': self.boundary_records,
            'window': self.window(),
            'scopes': [dict(row) for row in self.scopes],
            'apis': [dict(row) for row in self.apis],
            'external': [dict(row) for row in self.external],
            'activities': [dict(row) for row in self.activities],
            'activity_records': sum(row['count'] for row in self.activities),
            'partition_totals': {kind: by_kind[kind]
                                 for kind in ('kernel', 'memcpy', 'memset')},
        }
        return {'native': native,
                'offers': [dict(row) for row in self.offers],
                'window': self.window()}


MAIN = FakeStream('main', MAIN_CTX, MAIN_STREAM)
COPY = FakeStream('copy', MAIN_CTX, COPY_STREAM)
SE_AUX = FakeStream('se', MAIN_CTX, SE_STREAM)
PLE_AUX = FakeStream('ple', MAIN_CTX, PLE_STREAM)
DECODE = FakeStream('decode', DECODE_CTX, DECODE_STREAM)
DECODE_COPY = FakeStream('decode-copy', DECODE_CTX, DECODE_STREAM + 1)


def stream_identity(stream):
    return {'context_id': stream.context_id, 'stream_id': stream.stream_id}


class FakeChunk:
    req_index = 0

    def lane_inputs(self):
        return {'positions': 'p', 'input_ids': 'i',
                'query_start_loc': 'q', 'ngram_context': 'n'}

    def forward_scope(self):
        return contextlib.nullcontext()

    def finish_state(self):
        return True


class FakeAdmission:
    transaction = TXN
    num_tokens = 8


class FakeFactory:
    def admit(self, admission):
        return FakeChunk()


class FakeComms:
    pp = None


class FakeNative:
    def __init__(self, lane, *, torch_module):
        self.split = types.SimpleNamespace(prefill_stream=MAIN)
        self.copy_stream = COPY
        self.comms = FakeComms()
        self._lane = lane
        self._torch = torch_module

    def lane_forward(self, inputs, *, forward_context=None,
                     attention_driver=None):
        self._lane.attention_driver = attention_driver
        self._lane.forward_context = forward_context
        return self._lane


class FakeModel:
    """The generated lane-forward seam, calling the owned source methods."""

    start_layer = 0
    end_layer = 2

    def __init__(self, bridge, driver, operation, shared_experts, ple,
                 *, last_stage=False):
        self.bridge = bridge
        self.driver = driver
        self.operation = operation
        self.shared_experts = shared_experts
        self.ple = ple
        self.last_stage = last_stage

    def _flash_e3_is_last_stage(self):
        return self.last_stage

    def _flash_e3_forward_prefix(self, input_ids, intermediate=None,
                                 inputs_embeds=None):
        if self.ple is not None:
            self.ple.start_prefetch('hidden', ['0'])
        self.bridge.activity('kernel', stream=MAIN)
        return 'hidden'

    def _flash_e3_forward_layer(self, idx, hidden, block, injection, positions,
                                input_ids, query_start_loc, ngram_context):
        if self.driver is not None and self.driver.owns(self, idx):
            return self.driver.run_layer(None, idx)
        self.bridge.activity('kernel', stream=MAIN)
        if self.shared_experts is not None:
            self.shared_experts.maybe_forward_async('x', 0)
            self.shared_experts.wait()
        return ('hidden', None, None)

    def _flash_e3_forward_suffix(self, hidden, block, injection, last):
        self.bridge.activity('kernel', stream=MAIN)
        return None


class FakeQsaDriver:
    """Shape of the real driver phases; no model arithmetic."""

    def __init__(self, bridge, operation):
        self.bridge = bridge
        self.operation = operation
        self.calls = []

    def owns(self, model, layer_idx):
        return layer_idx == 0

    def run_layer(self, lane, layer_idx):
        self._pre()
        self._bucket()
        self._tail()
        return ('hidden', None, None)

    def _pre(self):
        self.calls.append('pre')
        self.bridge.activity('kernel', stream=MAIN)

    def _bucket(self):
        self.calls.append('bucket')
        self.operation.fill_bucket('staged', 0, 1, 0)
        self.bridge.activity('kernel', stream=MAIN)

    def _tail(self):
        self.calls.append('tail')
        self.bridge.activity('kernel', stream=MAIN)

    def cursor(self):
        return len(self.calls)


class FakeOperation:
    def __init__(self, bridge, copy):
        self.bridge = bridge
        self.device = types.SimpleNamespace(copy=copy)
        self.buckets = 0

    def fill_bucket(self, staged, page_lo, page_hi, block_table_row):
        self.buckets += 1
        self.bridge.activity('memcpy', stream=self.device.copy, copy_kind=8)

    def fill_sparse_bucket(self, staged, selected, compact_start, block_table_row):
        self.buckets += 1
        self.bridge.activity('memcpy', stream=self.device.copy, copy_kind=1)

    def stage_sparse_selection(self, tokens, info):
        self.bridge.activity('memcpy', stream=self.device.copy, copy_kind=2)
        self.bridge.activity('memcpy', stream=self.device.copy, copy_kind=2)


class FakeSharedExperts:
    def __init__(self, bridge, stream):
        self.bridge = bridge
        self._stream = stream
        self.calls = []

    def maybe_forward_async(self, value, idx):
        self.calls.append('async')
        self.bridge.activity('kernel', stream=self._stream)
        return True

    def forward(self, value):
        self.calls.append('forward')
        self.bridge.activity('kernel', stream=MAIN)
        return value

    def wait(self):
        self.calls.append('wait')
        return True


class FakePleEmbedding:
    def __init__(self, bridge, stream):
        self.bridge = bridge
        self._prefetch_stream = stream
        self.calls = []

    def start_prefetch(self, hidden_states, ids):
        self.calls.append('start')
        self.bridge.activity('kernel', stream=self._prefetch_stream)

    def _finalize_prefetch(self, prefetch_output, output):
        self.calls.append('finalize')
        self.bridge.activity('kernel', stream=MAIN)
        return output


def make_torch():
    return FakeTorch()


def make_ledger(bridge, *, torch_module=None, worker='w0', rank=0,
                window_id='window-1', run_id='run-1'):
    if torch_module is None:
        torch_module = make_torch()
    return npo.NativePrefillLedger(
        bridge, worker=worker, rank=rank, epoch=bridge.epoch,
        generation=bridge.generation,
        route_generation=bridge.route_generation, storage_epoch=1,
        run_id=run_id, window_id=window_id,
        stream_identity=stream_identity, torch_module=torch_module,
        env={npo.ENABLE_ENV: '1'})


def declare(ledger):
    return ledger.declare_streams(
        prefill_stream=MAIN, copy_stream=COPY,
        aux_streams={'shared_experts_aux': SE_AUX, 'ple_aux': PLE_AUX})


def start_role_owners(ledger, torch_module=None):
    torch_module = torch_module or ledger.torch
    events = []
    events.append(torch_module.cuda.Event(enable_timing=False))
    events[-1].record(MAIN)
    ledger.owner_start('copy', 'copy', stream=COPY)
    ledger.owner_start('shared_experts', 'shared_experts_aux', stream=SE_AUX)
    ledger.owner_start('ple', 'ple_aux', stream=PLE_AUX)
    return events


def finish_role_owners(ledger, events=()):
    ledger.owner_finish('copy', events=list(events))
    ledger.owner_finish('shared_experts', events=list(events))
    ledger.owner_finish('ple', events=list(events))


def build_lane(bridge, *, with_qsa=True, torch_module=None, last_stage=False):
    torch_module = torch_module or make_torch()
    operation = FakeOperation(bridge, COPY) if with_qsa else None
    driver = FakeQsaDriver(bridge, operation) if with_qsa else None
    shared = FakeSharedExperts(bridge, SE_AUX)
    ple = FakePleEmbedding(bridge, PLE_AUX)
    model = FakeModel(bridge, driver, operation, shared, ple,
                      last_stage=last_stage)
    lane = model_hooks.LaneForward(
        model, {'positions': 'p', 'input_ids': 'i',
                'query_start_loc': 'q', 'ngram_context': 'n'},
        stream=MAIN, torch_module=torch_module, attention_driver=driver)
    native = FakeNative(lane, torch_module=torch_module)
    dispatcher = LaneDispatcher(
        runner=types.SimpleNamespace(name='runner'), native=native,
        factory=FakeFactory(), torch_module=torch_module,
        driver_factory=((lambda operation, qsa_module, getter: driver)
                        if driver is not None else None))
    dispatcher.open(TXN, FakeAdmission(), types.SimpleNamespace(
        drain=lambda: True, cancel=lambda reason='x': reason,
        release=lambda: True))
    return {
        'torch': torch_module, 'operation': operation, 'driver': driver,
        'shared': shared, 'ple': ple, 'model': model, 'lane': lane,
        'native': native, 'dispatcher': dispatcher,
    }


def submit_all(dispatcher):
    from lane_dispatch import SegmentCommand
    results = []
    plan = [('prepare', None, None), ('prefix', None, None),
            ('layer', 0, 1), ('layer', 1, 2), ('suffix', None, None),
            ('state_finalize', None, None)]
    for op_seq, (kind, layer_start, layer_end) in enumerate(plan):
        command = SegmentCommand(
            TXN, 0, kind, op_seq, layer_start=layer_start,
            layer_end=layer_end)
        results.append(dispatcher.submit(command))
    return results


def install_lane_seams(parts, ledger):
    shadows = []
    pins = npo.live_pins(parts['dispatcher'])
    shadows.append(npo.install_seams(parts['dispatcher'], ledger,
                                     'lane_dispatcher', pins=pins))
    if parts['driver'] is not None:
        shadows.append(npo.install_seams(parts['driver'], ledger,
                                         'qsa_driver',
                                         pins=npo.live_pins(parts['driver'])))
    if parts['operation'] is not None:
        shadows.append(npo.install_seams(parts['operation'], ledger,
                                         'qsa_operation',
                                         pins=npo.live_pins(
                                             parts['operation'])))
    shadows.append(npo.install_seams(parts['shared'], ledger,
                                     'shared_experts',
                                     pins=npo.live_pins(parts['shared'])))
    shadows.append(npo.install_seams(parts['ple'], ledger, 'ple_embedding',
                                     pins=npo.live_pins(parts['ple'])))
    return shadows


# --- decode partition over the real engine offer ledger --------------------
def engine_binding(bridge, *, green_context_id, identities, run_id='run-1',
                   window_id='window-1', rank=0, epoch=None, storage_epoch=1):
    return engine_offers.bind_window(
        run_id=run_id, window_id=window_id, worker='w0', rank=rank,
        worker_incarnation='worker-inc', engine_incarnation='engine-inc',
        engine_epoch=epoch or bridge.epoch,
        route_generation=bridge.route_generation,
        storage_epoch=storage_epoch, green_context_id=green_context_id,
        native_start_ns=bridge.start_ns, native_end_ns=None,
        observer_identity={'abi': 'test', 'library_sha256': '0' * 64,
                           'source_sha256': '0' * 64,
                           'cupti_library_sha256': '0' * 64},
        helper_identity={'kind': 'cpu-test'},
        declared_target='target', descriptor_labels={'target': 'num_tokens'},
        router_census=None, bank_identity_sha256='a' * 64,
        stream_roles=identities, pp_stage=0)


def decode_ledger(bridge, *, green_context_id=DECODE_CTX,
                  stream=DECODE, copy_stream=DECODE_COPY):
    identities = {
        'main': {'context_id': green_context_id,
                 'stream_id': stream.stream_id,
                 'non_green_control': False},
        'output_copy': {'context_id': copy_stream.context_id,
                        'stream_id': copy_stream.stream_id,
                        'non_green_control': False},
    }
    binding = engine_binding(bridge, green_context_id=green_context_id,
                             identities=identities)
    ledger = engine_offers.EngineOfferLedger(
        bridge, worker='w0', rank=0, epoch=bridge.epoch,
        generation=bridge.generation,
        route_generation=bridge.route_generation,
        green_context_id=green_context_id, identities=identities,
        pp_stage=0, binding=binding,
        env={engine_offers.ENABLE_ENV: '1'})
    ledger.start()
    return ledger, binding


def run_decode_offers(ledger, bridge, *, stream=DECODE):
    with ledger.step('sample'):
        with ledger.offer_scope('sampler', 'draw'):
            bridge.activity('kernel', stream=stream)
        with ledger.offer_scope('request_state_table', 'no_work_step'):
            pass
        with ledger.offer_scope('output_copy', 'async_output',
                                branch='ordinary') as offer:
            if offer is not None:
                bridge.activity('memcpy', stream=DECODE_COPY, copy_kind=2)


def finish_decode(ledger, bridge, *, end_ns=None, close=False):
    ledger.deactivate()
    if close:
        bridge.close_window(end_ns)
    ledger.finish_window({'total': 1, 'graphs': {'g0': 1}})
    return dict(ledger._meta), ledger.window()


__all__ = ['COPY', 'DECODE', 'DECODE_COPY', 'DECODE_CTX', 'MAIN', 'MAIN_CTX', 'PLE_AUX',
           'SE_AUX', 'TXN', 'FakeBridge', 'FakePleEmbedding',
           'FakeSharedExperts', 'build_lane', 'declare', 'decode_ledger',
           'engine_binding', 'finish_decode', 'finish_role_owners',
           'install_lane_seams', 'make_ledger', 'make_torch',
           'run_decode_offers', 'start_role_owners', 'stream_identity',
           'submit_all']
