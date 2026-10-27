"""CPU-only tests for the q17b actual-replay context observer.

The fixture compiles the exact observer source into the same shared library as
a fake CUDA driver and CUPTI.  Tests drive the same exported C ABI that the
Python wrapper drives; no driver, GPU, profiling run or model replay occurs.
Build artifacts are kept under a fresh temporary directory.
"""
import ctypes
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import MappingProxyType

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PATCHES = os.path.join(ROOT, 'patches', 's5_engine')
FIXTURES = os.path.join(HERE, 'fixtures', 'e3_replay_observer')
if PATCHES not in sys.path:
    sys.path.insert(0, PATCHES)

import replay_observer as ro  # noqa: E402

STATIC_GRAPH = {'id': 11, 'nodes': [(101, 0), (102, 0)]}
SECOND_GRAPH = {'id': 12, 'nodes': [(201, 0)]}
CONTEXT_ID = 7
EXPECTED_SMS = 8
STORAGE_EPOCH = 3
ROUTE_GENERATION = 4


class FakeGraph:
    def __init__(self, handle, mutate=False):
        self.handle = int(handle)
        self.mutate = mutate
        self.calls = 0

    def raw_cuda_graph(self):
        self.calls += 1
        if self.mutate and self.calls > 1:
            return self.handle + 1
        return self.handle


class FakeHandle:
    def __init__(self, local, bank):
        self.local = local
        self.bank = bank
        self.destroyed = False


class FakeManager:
    def __init__(self, owner):
        self.graphs = {}
        self._flash_s5_capture_owner = owner
        self._flash_s5_capture_context = lambda: None


class FakeBank:
    def __init__(self):
        self.manager = None
        self.pairs = []
        self.handles = {}
        self.captured = True
        self.graph_handles = []
        self.graph_ids = []
        self.node_handles = []
        self.descriptors = []


class FakeSplit:
    def __init__(self, green, sms, requested=None):
        self.decode_context_id = int(green)
        self.decode_sms = int(sms)
        self.requested_decode_sms = int(sms if requested is None else requested)


class FakeRouter:
    def __init__(self, storage_epoch, route_generation):
        self.storage_epoch = int(storage_epoch)
        self.route_generation = int(route_generation)
        self.replays = 0
        self.replays_by_descriptor = {}


class FakeProbe:
    def __init__(self, bank, router, split):
        self.bank = bank
        self.router = router
        self.split = split
        self.storage_epoch = router.storage_epoch
        self.route_generation = router.route_generation
        self.route = 'split'
        self.interval = 'active'


class ObserverTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('g++')
        if compiler is None:
            raise unittest.SkipTest('g++ is required for the CPU fixture')
        cls.build_dir = tempfile.mkdtemp(prefix='s5ro-build-')
        cls.library_path = os.path.join(cls.build_dir, 'libobserver.so')
        command = [
            compiler, '-std=c++17', '-O1', '-g', '-shared', '-fPIC',
            '-Wall', '-Wextra', '-Werror', '-Wno-unused-parameter',
            '-I', FIXTURES,
            '-x', 'c++', os.path.join(PATCHES, ro.HELPER_SOURCE),
            '-x', 'c++', os.path.join(FIXTURES, 'stub_cupti.cpp'),
            '-o', cls.library_path,
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            raise AssertionError('fixture build failed: %s\n%s' % (command,
                                                                  completed.stderr))
        cls.fake_cupti = os.path.join(cls.build_dir, 'libcupti.so.13')
        with open(cls.fake_cupti, 'wb') as handle:
            handle.write(b'fixture cupti 13 identity\n')
        cls.library_sha = _sha(cls.library_path)
        cls.cupti_sha = _sha(cls.fake_cupti)
        cls.binding = ro.ReplayBinding(
            cls.library_path, cls.library_sha, ro.helper_source_sha256(),
            cls.cupti_sha, cupti_library_path=cls.fake_cupti)
        cls.ctl = ctypes.CDLL(cls.library_path)
        _declare(cls.ctl)

    def setUp(self):
        self.ctl.stub_reset()
        self.session = None
        self.bank = None
        self.probe = None
        self.green = None

    def _cleanup_session(self):
        if self.session is None or self.session.state == 'closed':
            return
        if self.session.state == 'window':
            try:
                self.session.end_window(self.proof())
            except Exception:
                pass
        for _attempt in range(2):
            try:
                if self.session.state in ('started', 'bound', 'ended'):
                    self.session.stop()
                break
            except ro.ReplayObserverError:
                try:
                    self.ctl.stub_release_held()
                except Exception:
                    pass
        try:
            self.session.close()
        except ro.ReplayObserverError:
            pass

    # -- bank and session helpers ----------------------------------------
    def make_bank(self, specs=None, *, sms=EXPECTED_SMS, context_id=CONTEXT_ID,
                  storage_epoch=STORAGE_EPOCH, route_generation=ROUTE_GENERATION,
                  captured=True, mutate=False):
        if specs is None:
            specs = [STATIC_GRAPH, SECOND_GRAPH]
        bank = FakeBank()
        bank.captured = captured
        split = FakeSplit(self.ctl.stub_new_green(context_id, sms), sms)
        self.green = split.decode_context_id
        router = FakeRouter(storage_epoch, route_generation)
        manager = FakeManager(split)
        bank.manager = manager
        for spec in specs:
            graph_handle = int(self.ctl.stub_new_graph(spec['id']))
            graph = FakeGraph(graph_handle, mutate=spec.get('mutate', False))
            node_handles = []
            for node_id, node_type in spec['nodes']:
                node_handle = int(self.ctl.stub_new_node(node_id, node_type))
                self.ctl.stub_graph_add_node(graph_handle, node_handle)
                node_handles.append(node_handle)
            descriptor = object()
            local = object()
            bank.pairs.append((descriptor, local))
            bank.handles[local] = FakeHandle(local, bank)
            manager.graphs[descriptor] = graph
            bank.descriptors.append(descriptor)
            bank.graph_handles.append(graph_handle)
            bank.graph_ids.append(spec['id'])
            bank.node_handles.append(node_handles)
        self.bank = bank
        self.probe = FakeProbe(bank, router, split)
        return bank

    def new_session(self, limits=None, *, start=False, bind=False):
        self.session = ro.ObserverSession(self.binding, limits)
        self.addCleanup(self._cleanup_session)
        if start:
            self.session.start()
        if bind:
            self.session.bind_bank(self.probe)
        return self.session

    def open_clean(self, limits=None, specs=None):
        self.make_bank(specs)
        self.new_session(limits, start=True, bind=True)
        self.ctl.stub_emit_context(CONTEXT_ID, 1, EXPECTED_SMS, 0)
        return self.session

    def proof(self, **overrides):
        values = dict(worker_rank=0, worker_incarnation='worker-1',
                      engine_incarnation='engine-1',
                      route_generation=ROUTE_GENERATION,
                      storage_epoch=STORAGE_EPOCH, relevant_consumers=1,
                      controller_drained=True)
        values.update(overrides)
        return ro.DrainedProof(**values)

    def begin(self, descriptor=None, route=ROUTE_GENERATION):
        self.ctl.stub_set_timestamp(1000)
        self.session.begin_window(route_generation=route,
                                  descriptor=descriptor)
        self.ctl.stub_set_timestamp(2000)

    def end(self, proof=None):
        self.ctl.stub_set_timestamp(3000)
        return self.session.end_window(proof if proof is not None else self.proof())

    def emit_kernel(self, graph_id, node_id, *, context_id=CONTEXT_ID,
                    start=1100, end=1200, completed=1200):
        self.ctl.stub_emit_kernel(context_id, graph_id, node_id, start, end,
                                  completed)

    def emit_graph_kernels(self, graph_index=0, *, completed=None):
        spec = (self.bank.graph_ids[graph_index],)
        nodes = [node_id for node_id, node_type in
                 (STATIC_GRAPH if graph_index == 0 else SECOND_GRAPH)['nodes']]
        for offset, node_id in enumerate(nodes):
            self.emit_kernel(spec[0], node_id, start=1100 + offset * 100,
                             end=1200 + offset * 100,
                             completed=(1200 + offset * 100
                                        if completed is None else completed))

    def assert_code(self, code, callable_, *args, **kwargs):
        with self.assertRaises(ro.ReplayObserverError) as raised:
            callable_(*args, **kwargs)
        self.assertEqual(raised.exception.code, code)

    def text(self, receipt):
        return json.dumps(ro._receipt_dict(receipt), sort_keys=True)


def _sha(path):
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _declare(lib):
    U64, U32, I, P = (ctypes.c_uint64, ctypes.c_uint32, ctypes.c_int,
                      ctypes.c_size_t)
    specs = {
        'stub_reset': ([], None),
        'stub_set_version': ([U32], None),
        'stub_set_version_rc': ([I], None),
        'stub_set_subscribe_rc': ([I], None),
        'stub_set_unsubscribe_rc': ([I], None),
        'stub_set_enable_callback_rc': ([I], None),
        'stub_set_enable_activity_rc': ([I], None),
        'stub_activity_enabled': ([I], I),
        'stub_set_register_rc': ([I], None),
        'stub_set_flush_rc': ([I], None),
        'stub_set_timestamp_rc': ([I], None),
        'stub_set_drop_rc': ([I], None),
        'stub_set_deliver_on_flush': ([I], None),
        'stub_set_timestamp': ([U64], None),
        'stub_set_reenter': ([I], None),
        'stub_set_foreign_subscriber': ([I], None),
        'stub_has_subscriber': ([], I),
        'stub_set_dropped': ([P], None),
        'stub_set_hold_next': ([I], None),
        'stub_set_disable_activity_rc': ([I], None),
        'stub_set_threaded_flush': ([I], None),
        'stub_install_foreign_callbacks': ([], None),
        'stub_held_count': ([], P),
        'stub_foreign_completions': ([], P),
        'stub_refused_requests': ([], P),
        'stub_pending_count': ([], P),
        'stub_new_context': ([U32], U64),
        'stub_new_green': ([U32, U32], U64),
        'stub_context_of_green': ([U64], U64),
        'stub_new_graph': ([U32], U64),
        'stub_new_node': ([U64, I], U64),
        'stub_new_exec': ([U32], U64),
        'stub_graph_add_node': ([U64, U64], None),
        'stub_emit_context': ([U32, U32, U32, U32], None),
        'stub_emit_kernel': ([U32, U32, U64, U64, U64, U64], None),
        'stub_emit_copy': ([U32, U32, U64, U64, U64], None),
        'stub_emit_memset': ([U32, U32, U64, U64, U64], None),
        'stub_emit_unknown': ([U32], None),
        'stub_emit_corrupt': ([], None),
        'stub_deliver': ([], I),
        'stub_release_held': ([], None),
        'stub_resource_graph': ([U32, U64, U64, U64, U64, I, U64], None),
        'stub_resource_context': ([U32, U64], None),
        'stub_resource_missing_descriptor': ([U32], None),
    }
    for name, (args, result) in specs.items():
        function = getattr(lib, name)
        function.argtypes = args
        if result is not None:
            function.restype = result


class ObserverCleanTests(ObserverTestBase):
    def test_per_node_counts_expose_loss_hidden_by_total_and_presence(self):
        self.open_clean()
        self.begin()
        for _ in range(3):
            self.emit_kernel(11, 101)
        self.emit_kernel(11, 102)
        receipt = self.end()
        # Presence alone remains PASS; the owner additionally requires min and
        # max to equal the independent router's count (2 in this example).
        slot = receipt['replay']['graph_slots'][0]
        self.assertEqual(slot['kernel_executions'], 4)
        self.assertEqual(slot['node_execution_bounds']['kernel'], {'min':1,'max':3})
        self.assertEqual(slot['missing']['kernel'], 0)

    def test_replay_census_preserves_zeroes_and_refuses_unknown_or_inconsistent_counts(self):
        self.open_clean()
        self.assertEqual(self.session.replay_census(),
                         {'total':0,'graphs':{'graph:0':0,'graph:1':0}})
        self.probe.router.replays = 2
        self.probe.router.replays_by_descriptor[self.bank.pairs[0][1]] = 2
        self.assertEqual(self.session.replay_census(),
                         {'total':2,'graphs':{'graph:0':2,'graph:1':0}})
        self.probe.router.replays = 3
        self.assertIsNone(self.session.replay_census())
        self.probe.router.replays_by_descriptor[object()] = 1
        self.assertIsNone(self.session.replay_census())

    def test_execution_collection_is_enabled_only_for_replay_window(self):
        self.open_clean()
        self.assertEqual(self.ctl.stub_activity_enabled(9), 1)
        for kind in (1, 2, 10):
            self.assertEqual(self.ctl.stub_activity_enabled(kind), 0)
        # Context evidence queued at startup is flushed before the timestamp.
        self.assertGreater(self.ctl.stub_pending_count(), 0)
        self.ctl.stub_set_threaded_flush(1)
        self.begin()
        self.assertEqual(self.ctl.stub_pending_count(), 0)
        for kind in (1, 2, 10):
            self.assertEqual(self.ctl.stub_activity_enabled(kind), 1)
        self.emit_graph_kernels(0)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'PASS')
        for kind in (1, 2, 10):
            self.assertEqual(self.ctl.stub_activity_enabled(kind), 0)
        self.assertEqual(self.ctl.stub_activity_enabled(9), 1)

    def test_capture_event_volume_from_full8_fits_declared_resource_ring(self):
        self.make_bank()
        self.new_session(start=True)
        # Real full8 had ~190k callbacks before its bounded replay. Repeated
        # creation notifications exercise storage without inventing node data.
        for _ in range(190000):
            self.ctl.stub_resource_graph(9, self.bank.graph_handles[0],
                                         0, 0, 0, 0, 0)
        self.session.bind_bank(self.probe)
        self.ctl.stub_emit_context(CONTEXT_ID, 1, EXPECTED_SMS, 0)
        self.begin()
        self.emit_graph_kernels(0)
        receipt = self.end()
        self.assertEqual(receipt['replay']['ring_overflows']['resource'], 0)
        self.assertEqual(receipt['verdict'], 'PASS')

    def test_pre_window_held_credit_refuses_begin_until_returned(self):
        self.open_clean()
        self.ctl.stub_set_hold_next(1)
        self.ctl.stub_deliver()
        self.assert_code('observer_busy', self.begin)
        self.assertEqual(self.session.state, 'bound')
        self.assertEqual(self.ctl.stub_activity_enabled(10), 0)
        self.ctl.stub_release_held()
        self.begin()
        self.emit_graph_kernels(0)
        self.assertEqual(self.end()['verdict'], 'PASS')

    def test_pre_window_flush_failure_cannot_be_retried_into_a_pass(self):
        self.open_clean()
        self.ctl.stub_set_flush_rc(1)
        self.assert_code('flush_failed', self.begin)
        self.ctl.stub_set_flush_rc(0)
        self.assert_code('bad_state', self.begin)
        self.session.stop()

    def test_window_enable_failure_retains_cleanup_ownership(self):
        self.open_clean()
        self.ctl.stub_set_enable_activity_rc(1)
        self.assert_code('activity_enable_failed', self.begin)
        self.ctl.stub_set_enable_activity_rc(0)
        self.assert_code('bad_state', self.begin)
        self.session.stop()

    def test_window_disable_failure_invalidates_proof(self):
        self.open_clean()
        self.begin()
        self.emit_graph_kernels(0)
        self.ctl.stub_set_disable_activity_rc(1)
        receipt = self.end()
        self.ctl.stub_set_disable_activity_rc(0)
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('flush_failed', receipt['reasons'])
        self.session.stop()

    def test_clean_exercised_descriptor_passes(self):
        self.open_clean()
        self.begin(self.bank.descriptors[0])
        self.emit_graph_kernels(0)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'PASS')
        self.assertEqual(receipt['reasons'], ())
        self.assertTrue(receipt['complete'])
        self.assertEqual(receipt['coverage']['observed_descriptors'],
                         ('graph:0',))
        self.assertEqual(receipt['coverage']['replay_qualified'],
                         ('graph:0',))
        self.assertEqual(receipt['coverage']['unexercised_descriptors'],
                         ('graph:1',))
        self.assertEqual(receipt['static_structure']['verdict'], 'PASS')
        self.assertTrue(receipt['replay']['graph_slots'][0]['complete'])
        self.assertFalse(receipt['replay']['graph_slots'][1]['exercised'])

    def test_delayed_completion_within_window_passes(self):
        self.open_clean()
        self.begin(self.bank.descriptors[0])
        self.emit_graph_kernels(0, completed=2500)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'PASS')
        self.assertEqual(receipt['replay']['counts']['incomplete_records'], 0)

    def test_unexercised_inventory_does_not_poison_exercised_descriptor(self):
        self.open_clean()
        self.begin()
        self.emit_graph_kernels(0)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'PASS')
        self.assertEqual(receipt['coverage']['unexercised_descriptors'],
                         ('graph:1',))
        self.assertEqual(receipt['coverage']['replay_qualified'],
                         ('graph:0',))

    def test_declared_target_must_be_exercised(self):
        self.open_clean()
        self.begin(self.bank.descriptors[1])
        self.emit_graph_kernels(0)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('declared_target_unexercised', receipt['reasons'])
        self.assertEqual(receipt['coverage']['replay_qualified'], ())

    def test_all_kernel_nodes_are_required(self):
        self.open_clean()
        self.begin(self.bank.descriptors[0])
        self.emit_kernel(11, 101, start=1100, end=1200, completed=1200)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('missing_execution', receipt['reasons'])
        slot = receipt['replay']['graph_slots'][0]
        self.assertEqual(slot['missing']['kernel'], 1)
        self.assertIn('node:0:1', slot['missing_nodes'])
        self.assertFalse(slot['replay_qualified'])

    def test_static_incomplete_is_separate_from_replay_pass(self):
        self.open_clean(specs=[{'id': 11, 'nodes': []}, SECOND_GRAPH])
        self.begin(self.bank.descriptors[1])
        self.emit_kernel(12, 201, start=1100, end=1200, completed=1200)
        receipt = self.end()
        self.assertEqual(receipt['static_structure']['verdict'], 'INCONCLUSIVE')
        self.assertIn('empty_graph', receipt['static_structure']['reasons'])
        self.assertEqual(receipt['replay']['verdict'], 'PASS')
        self.assertEqual(receipt['verdict'], 'PASS')
        self.assertEqual(receipt['coverage']['replay_qualified'], ('graph:1',))

    def test_static_unsupported_node_is_explicit(self):
        self.make_bank([{'id': 11, 'nodes': [(101, 13)]}])
        self.new_session(start=True, bind=True)
        static = self.session._static_report
        self.assertFalse(static['complete'])
        self.assertTrue(static['graph_slots'][0]['unsupported'] >= 1)
        self.assertIn('unsupported_node', static['reasons'])
        self.session.stop()

    def test_no_exercised_graph_is_inconclusive(self):
        self.open_clean()
        self.begin()
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('no_exercised_graph', receipt['reasons'])
        self.assertEqual(receipt['coverage']['replay_qualified'], ())


    def test_receipt_is_a_copy_after_buffer_reuse(self):
        self.open_clean()
        self.begin()
        self.emit_graph_kernels(0, completed=1200)
        receipt = self.end()
        before = self.text(receipt)
        self.ctl.stub_reset()
        self.assertEqual(self.text(receipt), before)
        self.assertEqual(receipt['replay']['counts']['kernel_records'], 2)
        self.assertEqual(receipt['replay']['mapping']['node_keys'], 3)

class ObserverCorrelationTests(ObserverTestBase):
    def configure(self,mode=0):
        import activity_accounting as a
        self.ctl.stub_set_thread_id_mode.argtypes=[ctypes.c_int]
        self.ctl.stub_set_thread_id_mode(mode)
        self.open_clean()
        self.bridge=a.CorrelationBridge(self.session,epoch=1,generation=ROUTE_GENERATION)
        f=self.ctl.stub_emit_correlated
        f.argtypes=[ctypes.c_uint32,ctypes.c_uint64,ctypes.c_uint32,ctypes.c_uint32,
                    ctypes.c_uint64,ctypes.c_uint64,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_uint64,ctypes.c_uint32]
        f.restype=None
        self.begin()
        return self.bridge

    def test_actual_c_abi_deferred_ids_nested_exception_pop_and_readonly_report(self):
        import threading
        b=self.configure()
        with b.offer('outer',{},context_id=7,stream_id=9):
            try:
                with b.offer('inner',{'kernel':1},context_id=7,stream_id=9) as o:
                    self.ctl.stub_emit_correlated(10,o['id'],5,o['thread'],1100,1500,7,0,0,0)
                    self.ctl.stub_emit_correlated(11,o['id'],4,o['thread'],1200,1400,7,0,0,1)
                    raise ValueError('CPU fixture')
            except ValueError:pass
        self.end();r=b.report()
        self.assertEqual(r['grade']['verdict'],'PASS',r['grade'])
        self.assertEqual(r['native']['open_scopes'],0)
        self.assertEqual(r['native']['thread_id_type'],'default-pthread-low32')
        self.assertNotEqual(b.offers[0]['thread'],b.offers[0]['system_thread'])
        self.assertEqual(r,b.report())
        self.assertEqual(r['grade']['joined_records'],1)
        self.assertEqual(len(r['grade']['wrappers']),1)
        self.assertTrue(b.offers[1]['body_exception'])

    def test_native_window_identity_is_available_before_and_after_work(self):
        b = self.configure()
        start = b.window()
        self.assertEqual(start, dict(start_ns=1000, end_ns=0,
                                     route_generation=ROUTE_GENERATION))
        with b.offer('probe', {'kernel': 1}, context_id=7, stream_id=9) as offer:
            self.ctl.stub_emit_correlated(10, offer['id'], 4, offer['thread'],
                                         1100, 1400, 7, 0, 0, 1)
        receipt = self.end()
        final = b.window()
        self.assertEqual(final, dict(start_ns=1000, end_ns=3000,
                                     route_generation=ROUTE_GENERATION))
        self.assertEqual(b.report()['native']['window'], final)
        self.assertEqual(receipt['replay']['window']['start_ns'], final['start_ns'])
        self.assertEqual(receipt['replay']['window']['end_ns'], final['end_ns'])
        self.assertEqual(b.report()['grade']['verdict'], 'PASS')

    def test_real_native_receipts_compose_after_observed_window_finalization(self):
        import engine_offers as eo
        import activity_accounting as accounting
        self.open_clean(specs=[STATIC_GRAPH, SECOND_GRAPH])
        bridge = accounting.CorrelationBridge(self.session, epoch=1,
                                              generation=ROUTE_GENERATION, rank=0)
        descriptor = {'num_tokens': 8, 'num_reqs': 8}
        roles = {'main': {'context_id': CONTEXT_ID, 'stream_id': 9}}
        self.begin()
        start = bridge.window()
        binding = eo.bind_window(
            run_id='native-fixture', window_id='placement-1', worker='w0', rank=0,
            worker_incarnation='worker-1', engine_incarnation='engine-1',
            engine_epoch=1, route_generation=ROUTE_GENERATION,
            storage_epoch=STORAGE_EPOCH, green_context_id=CONTEXT_ID,
            native_start_ns=start['start_ns'], native_end_ns=None,
            observer_identity=self.binding.identity(),
            helper_identity={'native_static_schema': ro.NATIVE_SCHEMA_STATIC,
                             'native_window_schema': ro.NATIVE_SCHEMA_WINDOW,
                             'compiled_api': 130001, 'runtime_api': 130001},
            declared_target='graph:0',
            descriptor_labels={'graph:0': eo.descriptor_key(descriptor),
                               'graph:1': eo.descriptor_key(dict(descriptor, num_tokens=16))},
            router_census=None, bank_identity_sha256=eo.bank_identity(self.session._bank_meta),
            stream_roles=roles, pp_stage=0)
        ledger = eo.EngineOfferLedger(
            bridge, worker='w0', rank=0, epoch=1, generation=ROUTE_GENERATION,
            route_generation=ROUTE_GENERATION, green_context_id=CONTEXT_ID,
            identities=roles, pp_stage=0, env={eo.ENABLE_ENV: '1'}, binding=binding)
        self.addCleanup(ledger.deactivate)
        ledger.activate()
        eager = self.ctl.stub_emit_correlated
        eager.argtypes = [ctypes.c_uint32,ctypes.c_uint64,ctypes.c_uint32,ctypes.c_uint32,
                          ctypes.c_uint64,ctypes.c_uint64,ctypes.c_uint32,ctypes.c_uint32,
                          ctypes.c_uint64,ctypes.c_uint32]
        graph = self.ctl.stub_emit_correlated_graph
        graph.argtypes = [ctypes.c_uint32,ctypes.c_uint64,ctypes.c_uint32,
                          ctypes.c_uint32,ctypes.c_uint32,ctypes.c_uint64,ctypes.c_uint64]
        with ledger.step('execute'):
            with ledger.offer_scope('request_state_table', 'prep_state'):
                offer = bridge.offers[-1]
                eager(10,offer['id'],4,offer['thread'],1100,1400,CONTEXT_ID,0,0,1)
            with ledger.offer_scope('forward_dispatch', 'graph_replay',
                    descriptor=descriptor, census={'kernel':2,'memcpy':0,'memset':0}):
                offer = bridge.offers[-1]
                graph(11,offer['id'],offer['thread'],CONTEXT_ID,11,101,102)
        ledger.deactivate()
        raw = self.end()
        raw_copy = ro._receipt_dict(raw)
        self.assertEqual(raw['verdict'], 'INCONCLUSIVE')
        self.assertEqual(raw['replay']['counts']['unattributed_expected_context'], 1)
        ledger.finish_window({'total':1,'graphs':{'graph:0':1}})
        engine = ledger.seal()
        self.assertEqual(engine['verdict'], 'PASS', engine)
        composed = eo.compose_placement(raw_copy, engine)
        self.assertEqual(composed['verdict'], 'PASS', composed)
        self.assertEqual(ro._receipt_dict(raw), raw_copy)
        self.assertEqual(engine['joined_records'], 3)
        self.assertEqual(engine['eager_green_records'], 1)
        self.assertEqual(engine['native_window']['start_ns'], start['start_ns'])

    def test_native_two_threads_scopes_do_not_leak_between_threads(self):
        import threading
        b=self.configure();errors=[];barrier=threading.Barrier(2,timeout=5)
        def work(c):
            try:
                barrier.wait()
                with b.offer('thread',{'kernel':1},context_id=7,stream_id=9) as o:
                    # No concurrent stub buffer mutation; native TLS itself is
                    # tested on two distinct threads using a CPU lock.
                    with lock:self.ctl.stub_emit_correlated(c,o['id'],4,o['thread'],1100+c,1200+c,7,0,0,1)
                barrier.wait()
            except Exception as e:errors.append(type(e).__name__)
        lock=threading.Lock();ts=[threading.Thread(target=work,args=(i,)) for i in (10,20)]
        for t in ts:t.start()
        for t in ts:t.join()
        self.assertEqual(errors,[]);self.end();r=b.report()
        self.assertEqual(r['grade']['verdict'],'PASS',r['grade'])
        self.assertEqual(len({x['thread'] for x in r['offers']}),2)

    def test_wrong_pop_and_stale_generation_remain_poisoned_after_cleanup(self):
        b=self.configure()
        with b.offer('empty',{},context_id=7,stream_id=9) as o:
            self.assertNotEqual(b.lib.s5ro_offer_pop(b.token,o['id']+1),0)
            self.assertNotEqual(b.lib.s5ro_offer_push(b.token,99,ROUTE_GENERATION-1),0)
        self.end();r=b.report()
        self.assertEqual(r['grade']['verdict'],'INCONCLUSIVE')
        self.assertEqual(r['native']['errors'],2)
        self.assertEqual(r['native']['open_scopes'],0)


    def test_explicit_system_mode_and_mid_session_change_refused(self):
        import threading
        b=self.configure(mode=1)
        with b.offer('empty',{},context_id=7,stream_id=9) as o:
            self.assertEqual(o['thread'],threading.get_native_id())
            self.assertEqual(o['system_thread'],o['thread'])
        self.ctl.stub_set_thread_id_mode(0)
        with self.assertRaises(RuntimeError):
            with b.offer('foreign_mode',{},context_id=7,stream_id=9):pass
        self.end()
        self.assertEqual(b.report()['grade']['verdict'],'INCONCLUSIVE')

class ObserverContextTests(ObserverTestBase):
    def test_activity_census_counts_graph_zero_all_kinds_without_waiving_guard(self):
        self.open_clean();self.begin()
        self.emit_graph_kernels()
        self.emit_kernel(0,0,start=1500,end=1550,completed=1550)
        self.ctl.stub_emit_copy(CONTEXT_ID,0,0,1600,1650)
        self.ctl.stub_emit_memset(CONTEXT_ID,0,0,1700,1750)
        receipt=self.end()
        self.assertIn('unattributed_expected_context',receipt['replay']['reasons'])
        fn=self.binding._lib.s5ro_activity_census
        fn.argtypes=[ctypes.c_uint64]*3+[ctypes.c_void_p,ctypes.c_uint64,ctypes.POINTER(ctypes.c_uint64)]
        fn.restype=ctypes.c_int
        def query(a,b):return self.session._report_call(fn,(self.session._token,a,b))
        all_=query(1000,3000)
        self.assertTrue(all_['complete']);self.assertEqual(all_['matched_records'],5)
        zero=[x for x in all_['rows'] if x['graph_id']==0]
        self.assertEqual({x['kind'] for x in zero},{'kernel','memcpy','memset'})
        self.assertEqual(sum(x['count'] for x in zero),3)
        sub=query(1490,1560)
        self.assertEqual(sub['matched_records'],1)
        self.assertEqual(sub['rows'][0]['name'],'fixture_kernel')
        boundary=query(1520,1560)
        self.assertFalse(boundary['complete']);self.assertEqual(boundary['boundary_records'],1)
        again=query(1000,3000)
        self.assertEqual(all_,again)  # query never consumes records or clears loss

    def test_context_delivered_before_window_is_retained(self):
        self.open_clean()
        self.ctl.stub_deliver()
        self.begin()
        self.emit_graph_kernels(0)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'PASS')
        self.assertGreater(receipt['replay']['context']['records'], 0)

    def test_missing_context_is_inconclusive(self):
        self.make_bank()
        self.new_session(start=True, bind=True)
        self.begin()
        self.emit_graph_kernels(0)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('expected_context_missing', receipt['reasons'])
        self.assertEqual(receipt['replay']['context']['records'], 0)

    def test_wrong_known_context_fails(self):
        self.open_clean()
        self.begin()
        self.emit_kernel(11, 101, context_id=9, completed=1200)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'FAIL')
        self.assertIn('known_wrong_context', receipt['reasons'])
        self.assertGreater(receipt['replay']['counts']
                           ['known_wrong_context_records'], 0)
        self.assertEqual(receipt['coverage']['replay_qualified'], ())

    def test_wrong_known_sm_fails(self):
        self.make_bank()
        self.new_session(start=True, bind=True)
        self.ctl.stub_emit_context(CONTEXT_ID, 1, 4, 0)
        self.begin()
        self.emit_graph_kernels(0)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'FAIL')
        self.assertIn('expected_context_sm_mismatch', receipt['reasons'])
        self.assertTrue(receipt['replay']['flags']['known_wrong_sm'])

    def test_non_green_context_fails(self):
        self.make_bank()
        self.new_session(start=True, bind=True)
        self.ctl.stub_emit_context(CONTEXT_ID, 0, EXPECTED_SMS, 0)
        self.begin()
        self.emit_graph_kernels(0)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'FAIL')
        self.assertIn('expected_context_not_green', receipt['reasons'])
        self.assertTrue(receipt['replay']['flags']['known_wrong_greenness'])

    def test_duplicate_context_is_inconclusive(self):
        self.open_clean()
        self.ctl.stub_emit_context(CONTEXT_ID, 1, EXPECTED_SMS, 0)
        self.begin()
        self.emit_graph_kernels(0)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('expected_context_duplicate', receipt['reasons'])
        self.assertTrue(receipt['replay']['context']['duplicate'])

    def test_contradictory_context_is_inconclusive(self):
        self.open_clean()
        self.ctl.stub_emit_context(CONTEXT_ID, 1, 4, 0)
        self.begin()
        self.emit_graph_kernels(0)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('expected_context_contradictory', receipt['reasons'])
        self.assertTrue(receipt['replay']['context']['contradictory'])
        self.assertIsNone(receipt['replay']['context']['num_multiprocessors'])

    def test_context_destroy_during_window_fails(self):
        self.open_clean()
        self.begin()
        self.emit_graph_kernels(0)
        context_handle = int(self.ctl.stub_context_of_green(self.green))
        self.ctl.stub_resource_context(2, context_handle)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'FAIL')
        self.assertIn('context_destroyed', receipt['reasons'])
        self.assertTrue(receipt['replay']['flags']['context_destroyed'])

    def test_fail_outranks_missing_evidence(self):
        self.make_bank()
        self.new_session(start=True, bind=True)
        self.ctl.stub_emit_context(CONTEXT_ID, 0, EXPECTED_SMS, 0)
        self.begin()
        self.emit_kernel(11, 101, start=900, end=1100, completed=1100)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'FAIL')
        self.assertIn('expected_context_not_green', receipt['reasons'])
        self.assertIn('boundary_overlap', receipt['reasons'])


class ObserverRecordTests(ObserverTestBase):
    def test_unknown_graph_record_is_inconclusive(self):
        self.open_clean()
        self.begin()
        self.emit_kernel(999, 101)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('unknown_graph', receipt['reasons'])

    def test_unknown_node_is_inconclusive(self):
        self.open_clean()
        self.begin()
        self.emit_kernel(11, 9999)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('unknown_node', receipt['reasons'])

    def test_kind_mismatch_is_inconclusive(self):
        self.open_clean()
        self.begin()
        self.ctl.stub_emit_copy(CONTEXT_ID, 11, 101, 1100, 1200)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('kind_mismatch', receipt['reasons'])

    def test_missing_node_id_is_inconclusive(self):
        self.open_clean()
        self.begin()
        self.emit_kernel(11, 0)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('missing_node_id', receipt['reasons'])

    def test_boundary_overlap_is_inconclusive(self):
        self.open_clean()
        self.begin()
        self.emit_kernel(11, 101, start=900, end=1100, completed=1100)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('boundary_overlap', receipt['reasons'])

    def test_invalid_interval_is_inconclusive(self):
        self.open_clean()
        self.begin()
        self.emit_kernel(11, 101, start=1300, end=1200, completed=1200)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('invalid_interval', receipt['reasons'])

    def test_unknown_timestamp_is_inconclusive(self):
        self.open_clean()
        self.begin()
        self.emit_kernel(11, 101, start=0, end=0, completed=0)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('unknown_timestamp', receipt['reasons'])

    def test_incomplete_completion_is_inconclusive(self):
        self.open_clean()
        self.begin()
        self.emit_graph_kernels(0, completed=0)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('incomplete_record', receipt['reasons'])
        self.assertGreater(receipt['replay']['counts']['incomplete_records'], 0)

    def test_completed_before_kernel_end_is_invalid(self):
        self.open_clean()
        self.begin()
        self.emit_kernel(11, 101, start=1100, end=1200, completed=1100)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('invalid_interval', receipt['reasons'])
        self.assertGreater(
            receipt['replay']['counts']['invalid_interval_records'], 0)

    def test_wholly_pre_window_activity_is_not_boundary_overlap(self):
        self.open_clean()
        self.begin()
        self.emit_kernel(11, 101, start=100, end=200, completed=200)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertNotIn('boundary_overlap', receipt['reasons'])
        self.assertGreater(
            receipt['replay']['counts']['records_before_window'], 0)
        self.assertIn('no_exercised_graph', receipt['reasons'])

    def test_wholly_post_window_activity_is_not_boundary_overlap(self):
        self.open_clean()
        self.begin()
        self.emit_kernel(11, 101, start=4000, end=4100, completed=4100)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertNotIn('boundary_overlap', receipt['reasons'])
        self.assertGreater(
            receipt['replay']['counts']['records_after_window'], 0)


class ObserverMappingTests(ObserverTestBase):
    def test_graph_create_resource_envelope_does_not_query_descriptor_as_node(self):
        self.open_clean()
        self.ctl.stub_set_timestamp(500)
        self.ctl.stub_resource_graph(9, self.bank.graph_handles[0], 0, 0, 0, 0, 0)
        self.begin(self.bank.descriptors[0])
        self.emit_graph_kernels(0)
        receipt = self.end()
        self.assertNotIn('mapping_unknown', receipt['reasons'])
        # Creating an already bound graph is still an inventory mutation.
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')

    def test_missing_graph_descriptor_cannot_pass(self):
        self.open_clean()
        self.begin(self.bank.descriptors[0])
        self.emit_graph_kernels(0)
        self.ctl.stub_resource_missing_descriptor(9)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE', self.text(receipt))

    def test_cloned_graph_node_maps_to_retained_original(self):
        self.open_clean()
        cloned_graph = int(self.ctl.stub_new_graph(112))
        cloned_node = int(self.ctl.stub_new_node(202, 0))
        self.ctl.stub_set_timestamp(500)
        self.ctl.stub_resource_graph(11, cloned_graph, self.bank.graph_handles[0],
                                     0, 0, 0, 0)
        self.ctl.stub_resource_graph(20, cloned_graph, self.bank.graph_handles[0],
                                     cloned_node, self.bank.node_handles[0][0], 0, 0)
        self.begin(self.bank.descriptors[0])
        self.emit_kernel(112, 202, completed=1200)
        self.emit_kernel(112, 102, completed=1300)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'PASS')
        slot = receipt['replay']['graph_slots'][0]
        self.assertEqual(slot['observed']['kernel'], 2)
        self.assertEqual(receipt['coverage']['replay_qualified'], ('graph:0',))

    def mixed_clone(self, truthful=False):
        # Retained driver inventory is authoritative. Reproduce clone
        # callbacks advertising KERNEL for the memcpy/memset aliases.
        self.open_clean(specs=[{'id': 11, 'nodes': [(101, 0), (102, 1), (103, 2)]}])
        clone_graph = int(self.ctl.stub_new_graph(112))
        self.ctl.stub_set_timestamp(500)
        self.ctl.stub_resource_graph(11, clone_graph, self.bank.graph_handles[0],
                                     0, 0, 0, 0)
        for i, kind in enumerate((0, 1, 2)):
            node = int(self.ctl.stub_new_node(201 + i, kind))
            self.ctl.stub_resource_graph(20, clone_graph, self.bank.graph_handles[0],
                                         node, self.bank.node_handles[0][i], kind if truthful else 0, 0)

    def test_mixed_clone_type_hint_does_not_override_driver_inventory(self):
        self.mixed_clone()
        self.begin(self.bank.descriptors[0])
        for _ in range(3):
            self.emit_kernel(112, 201)
            self.ctl.stub_emit_copy(CONTEXT_ID, 112, 202, 1100, 1200)
            self.ctl.stub_emit_memset(CONTEXT_ID, 112, 203, 1100, 1200)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'PASS', self.text(receipt))
        for kind in ('kernel', 'memcpy', 'memset'):
            self.assertEqual(receipt['replay']['graph_slots'][0]
                             ['node_execution_bounds'][kind], {'min': 3, 'max': 3})
        counts = receipt['replay']['counts']
        self.assertEqual(counts['alias_type_disagreements'], 2)
        self.assertEqual(counts['alias_type_memcpy_disagreements'], 1)
        self.assertEqual(counts['alias_type_memset_disagreements'], 1)
        self.assertEqual(counts['alias_type_reported_kernel_disagreements'], 2)
        self.assertFalse(receipt['replay']['mapping']['node_overlap'])

    def test_mixed_clone_true_type_and_schema_roundtrip(self):
        self.mixed_clone(truthful=True)
        captured = []
        original = self.session._report_call
        def capture(*args, **kwargs):
            report = original(*args, **kwargs)
            captured.append(copy.deepcopy(report))
            return report
        self.session._report_call = capture
        self.begin(self.bank.descriptors[0])
        self.emit_kernel(112, 201)
        self.ctl.stub_emit_copy(CONTEXT_ID, 112, 202, 1100, 1200)
        self.ctl.stub_emit_memset(CONTEXT_ID, 112, 203, 1100, 1200)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'PASS')
        self.assertEqual(receipt['replay']['counts']['alias_type_disagreements'], 0)
        report = captured[-1]
        self.assertEqual(report['schema'], 's5ro-native-window/4')
        report['schema'] = 's5ro-native-window/2'
        with self.assertRaises(ro.ReplayObserverError):
            ro._validate_window(report, self.session.limits, ROUTE_GENERATION)
        report['schema'] = 's5ro-native-window/4'
        del report['counts']['alias_type_disagreements']
        with self.assertRaises(ro.ReplayObserverError):
            ro._validate_window(report, self.session.limits, ROUTE_GENERATION)

    def test_two_copy_aliases_cannot_hide_missing_original(self):
        self.open_clean(specs=[{'id': 11, 'nodes': [(101, 1), (102, 1)]}])
        graph = int(self.ctl.stub_new_graph(112))
        self.ctl.stub_set_timestamp(500)
        self.ctl.stub_resource_graph(11, graph, self.bank.graph_handles[0], 0, 0, 0, 0)
        for node_id in (201, 202):
            node = int(self.ctl.stub_new_node(node_id, 1))
            self.ctl.stub_resource_graph(20, graph, self.bank.graph_handles[0],
                                         node, self.bank.node_handles[0][0], 0, 0)
        self.begin(self.bank.descriptors[0])
        for node_id in (201, 202):
            self.ctl.stub_emit_copy(CONTEXT_ID, 112, node_id, 1100, 1200)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('missing_execution', receipt['reasons'])
        self.assertEqual(receipt['replay']['graph_slots'][0]
                         ['node_execution_bounds']['memcpy'], {'min': 0, 'max': 2})

    def test_one_copy_alias_to_two_originals_is_still_ambiguous(self):
        self.open_clean(specs=[{'id': 11, 'nodes': [(101, 1), (102, 1)]}])
        graph = int(self.ctl.stub_new_graph(112))
        node = int(self.ctl.stub_new_node(201, 1))
        self.ctl.stub_set_timestamp(500)
        self.ctl.stub_resource_graph(11, graph, self.bank.graph_handles[0], 0, 0, 0, 0)
        for original in self.bank.node_handles[0]:
            self.ctl.stub_resource_graph(20, graph, self.bank.graph_handles[0],
                                         node, original, 0, 0)
        self.begin(self.bank.descriptors[0])
        self.ctl.stub_emit_copy(CONTEXT_ID, 112, 201, 1100, 1200)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('ambiguous_node', receipt['reasons'])

    def test_clone_type_hint_cannot_turn_copy_into_kernel(self):
        self.mixed_clone()
        self.begin(self.bank.descriptors[0])
        self.emit_kernel(112, 201)
        self.emit_kernel(112, 202)
        self.ctl.stub_emit_memset(CONTEXT_ID, 112, 203, 1100, 1200)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('kind_mismatch', receipt['reasons'])
        self.assertEqual(receipt['replay']['graph_slots'][0]['copy_executions'], 0)

    def test_clone_type_hint_cannot_cover_a_missing_copy(self):
        self.mixed_clone()
        self.begin(self.bank.descriptors[0])
        self.emit_kernel(112, 201)
        self.ctl.stub_emit_memset(CONTEXT_ID, 112, 203, 1100, 1200)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('missing_execution', receipt['reasons'])

    def test_original_and_clone_both_count(self):
        self.open_clean()
        cloned_graph = int(self.ctl.stub_new_graph(112))
        cloned_node = int(self.ctl.stub_new_node(202, 0))
        self.ctl.stub_set_timestamp(500)
        self.ctl.stub_resource_graph(11, cloned_graph, self.bank.graph_handles[0],
                                     0, 0, 0, 0)
        cloned_node_2 = int(self.ctl.stub_new_node(203, 0))
        self.ctl.stub_resource_graph(20, cloned_graph, self.bank.graph_handles[0],
                                     cloned_node, self.bank.node_handles[0][0], 0, 0)
        self.ctl.stub_resource_graph(20, cloned_graph, self.bank.graph_handles[0],
                                     cloned_node_2, self.bank.node_handles[0][1], 0, 0)
        self.begin(self.bank.descriptors[0])
        self.emit_kernel(11, 101)
        self.emit_kernel(112, 202, start=1200, end=1300, completed=1300)
        self.emit_kernel(11, 102, start=1300, end=1400, completed=1400)
        self.emit_kernel(112, 203, start=1400, end=1500, completed=1500)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'PASS')
        self.assertEqual(receipt['replay']['graph_slots'][0]
                         ['kernel_executions'], 4)

    def test_node_update_during_window_is_inconclusive(self):
        self.open_clean()
        self.begin(self.bank.descriptors[0])
        self.emit_graph_kernels(0)
        self.ctl.stub_resource_graph(22, self.bank.graph_handles[0], 0,
                                     self.bank.node_handles[0][0], 0, 0, 0)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('update_seen', receipt['reasons'])
        self.assertTrue(receipt['replay']['flags']['update_seen'])

    def test_graph_destroy_during_window_is_inconclusive(self):
        self.open_clean()
        self.begin(self.bank.descriptors[0])
        self.emit_graph_kernels(0)
        self.ctl.stub_resource_graph(10, self.bank.graph_handles[0], 0, 0, 0, 0, 0)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('destroy_seen', receipt['reasons'])
        self.assertTrue(receipt['replay']['flags']['destroy_seen'])

    def test_inventory_change_during_window_is_inconclusive(self):
        self.open_clean()
        self.begin(self.bank.descriptors[0])
        self.emit_graph_kernels(0)
        new_node = int(self.ctl.stub_new_node(303, 0))
        self.ctl.stub_resource_graph(13, self.bank.graph_handles[0], 0,
                                     new_node, 0, 0, 0)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('inventory_changed', receipt['reasons'])
        self.assertTrue(receipt['replay']['flags']['inventory_changed'])

    def test_64bit_cloned_node_id_is_not_truncated(self):
        self.open_clean()
        cloned_graph = int(self.ctl.stub_new_graph(112))
        wide_node_id = (1 << 32) + 5
        cloned_node = int(self.ctl.stub_new_node(wide_node_id, 0))
        self.ctl.stub_set_timestamp(500)
        self.ctl.stub_resource_graph(11, cloned_graph, self.bank.graph_handles[0],
                                     0, 0, 0, 0)
        self.ctl.stub_resource_graph(20, cloned_graph, self.bank.graph_handles[0],
                                     cloned_node, self.bank.node_handles[0][0], 0, 0)
        self.begin(self.bank.descriptors[0])
        self.emit_kernel(112, wide_node_id)
        self.emit_kernel(112, 102)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'PASS')
        slot = receipt['replay']['graph_slots'][0]
        self.assertEqual(slot['observed']['kernel'], 2)
        self.assertEqual(receipt['replay']['mapping']['node_aliases'], 1)

    def test_ambiguous_cloned_node_is_inconclusive(self):
        self.open_clean()
        cloned_graph = int(self.ctl.stub_new_graph(112))
        clone = int(self.ctl.stub_new_node(202, 0))
        self.ctl.stub_set_timestamp(500)
        self.ctl.stub_resource_graph(11, cloned_graph, self.bank.graph_handles[0],
                                     0, 0, 0, 0)
        self.ctl.stub_resource_graph(20, cloned_graph, self.bank.graph_handles[0],
                                     clone, self.bank.node_handles[0][0], 0, 0)
        self.ctl.stub_resource_graph(20, cloned_graph, self.bank.graph_handles[0],
                                     clone, self.bank.node_handles[0][1], 0, 0)
        self.begin(self.bank.descriptors[0])
        self.emit_kernel(112, 202)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('ambiguous_node', receipt['reasons'])

    def test_update_between_bind_and_begin_invalidates_binding(self):
        self.open_clean()
        self.ctl.stub_set_timestamp(500)
        self.ctl.stub_resource_graph(22, self.bank.graph_handles[0], 0,
                                     self.bank.node_handles[0][0], 0, 0, 0)
        self.begin()
        self.emit_graph_kernels(0)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('update_seen', receipt['reasons'])
        self.assertTrue(receipt['replay']['graph_slots'][0]['update_seen'])

    def test_destroy_between_bind_and_begin_invalidates_binding(self):
        self.open_clean()
        self.ctl.stub_set_timestamp(500)
        self.ctl.stub_resource_graph(10, self.bank.graph_handles[0], 0,
                                     0, 0, 0, 0)
        self.begin()
        self.emit_graph_kernels(0)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('destroy_seen', receipt['reasons'])
        self.assertTrue(receipt['replay']['graph_slots'][0]['destroy_seen'])

    def test_registered_dependency_callbacks_invalidate_binding(self):
        for cbid in (15, 16):
            with self.subTest(cbid=cbid):
                self.open_clean()
                self.ctl.stub_set_timestamp(500)
                self.ctl.stub_resource_graph(
                    cbid, self.bank.graph_handles[0], 0,
                    self.bank.node_handles[0][0], 0, 0, 0)
                self.begin()
                self.emit_graph_kernels(0)
                receipt = self.end()
                self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
                self.assertIn('inventory_changed', receipt['reasons'])
                self.session.close()

    def test_pre_bind_creation_history_does_not_poison_binding(self):
        self.make_bank()
        self.new_session(start=True)
        cloned_graph = int(self.ctl.stub_new_graph(112))
        cloned_node = int(self.ctl.stub_new_node(202, 0))
        self.ctl.stub_set_timestamp(500)
        self.ctl.stub_resource_graph(11, cloned_graph,
                                     self.bank.graph_handles[0], 0, 0, 0, 0)
        self.ctl.stub_resource_graph(20, cloned_graph,
                                     self.bank.graph_handles[0], cloned_node,
                                     self.bank.node_handles[0][0], 0, 0)
        self.session.bind_bank(self.probe)
        self.ctl.stub_emit_context(CONTEXT_ID, 1, EXPECTED_SMS, 0)
        self.begin()
        self.emit_kernel(112, 202)
        self.emit_kernel(112, 102)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'PASS')
        self.assertEqual(receipt['replay']['mapping']['node_aliases'], 1)

    def test_recycled_graph_identity_is_ambiguous(self):
        self.open_clean()
        self.ctl.stub_set_timestamp(500)
        self.ctl.stub_resource_graph(10, self.bank.graph_handles[0], 0,
                                     0, 0, 0, 0)
        self.ctl.stub_resource_graph(9, self.bank.graph_handles[0], 0,
                                     0, 0, 0, 0)
        self.begin()
        self.emit_graph_kernels(0)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertTrue(receipt['replay']['mapping']['ambiguous_graph_ids'])
        self.assertIn('ambiguous_graph', receipt['reasons'])


class ObserverBoundsTests(ObserverTestBase):
    def test_pre_window_resource_overflow_is_not_cleared_by_begin(self):
        self.make_bank()
        self.new_session(ro.ObserverLimits(max_graph_events=1), start=True)
        for _ in range(2):
            self.ctl.stub_resource_graph(9, self.bank.graph_handles[0],
                                         0, 0, 0, 0, 0)
        self.session.bind_bank(self.probe)
        self.ctl.stub_emit_context(CONTEXT_ID, 1, EXPECTED_SMS, 0)
        self.begin()
        self.emit_graph_kernels(0)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('ring_overflow', receipt['reasons'])
        self.assertEqual(receipt['replay']['ring_overflows']['resource'], 1)

    def test_dropped_counts_accumulate_across_reset_on_read(self):
        self.make_bank()
        self.new_session()
        self.ctl.stub_set_dropped(2)
        self.session.start()
        self.session.bind_bank(self.probe)
        self.ctl.stub_emit_context(CONTEXT_ID, 1, EXPECTED_SMS, 0)
        self.begin()
        self.emit_graph_kernels(0)
        self.ctl.stub_set_dropped(4)
        receipt = self.end()
        self.assertEqual(receipt['replay']['counts']['dropped_records_total'], 6)
        self.assertTrue(receipt['replay']['flags']['drop_seen'])
        self.assertIn('drop_seen', receipt['reasons'])
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')

    def test_drop_query_error_is_inconclusive(self):
        self.open_clean()
        self.begin()
        self.emit_graph_kernels(0)
        self.ctl.stub_set_drop_rc(1)
        receipt = self.end()
        self.ctl.stub_set_drop_rc(0)
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('drop_query_error', receipt['reasons'])

    def test_kernel_ring_overflow_is_permanent(self):
        self.open_clean(ro.ObserverLimits(max_kernel_records=1))
        self.begin()
        self.emit_graph_kernels(0)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('ring_overflow', receipt['reasons'])
        self.assertEqual(receipt['replay']['ring_overflows']['kernel'], 1)
        self.assertTrue(receipt['replay']['flags']['permanent_overflow'])

    def test_refused_buffer_and_held_ownership(self):
        self.open_clean(ro.ObserverLimits(buffer_count=1))
        self.begin()
        self.ctl.stub_emit_unknown(77)
        self.ctl.stub_set_hold_next(1)
        self.ctl.stub_deliver()
        self.assertEqual(int(self.ctl.stub_held_count()), 1)
        self.emit_graph_kernels(0)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('buffer_refused', receipt['reasons'])
        self.assertGreater(receipt['replay']['counts']['buffers_refused'], 0)
        self.assert_code('observer_busy', self.session.stop)
        self.ctl.stub_release_held()
        self.assertEqual(int(self.ctl.stub_held_count()), 0)
        self.session.stop()

    def test_withheld_negative_record_after_positive_coverage_is_inconclusive(self):
        self.open_clean()
        self.begin()
        self.emit_graph_kernels(0)
        self.ctl.stub_deliver()
        self.emit_kernel(11, 101, context_id=999)
        self.ctl.stub_set_hold_next(1)
        self.ctl.stub_deliver()
        self.assertEqual(int(self.ctl.stub_held_count()), 1)
        self.ctl.stub_set_deliver_on_flush(0)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('buffer_outstanding', receipt['reasons'])
        self.assertTrue(receipt['replay']['flags']['buffer_outstanding'])
        self.assertGreater(
            receipt['replay']['counts']['buffers_issued_in_window'],
            receipt['replay']['counts']['buffers_completed_in_window'])
        before = self.text(receipt)
        self.assert_code('observer_busy', self.session.stop)
        self.ctl.stub_release_held()
        self.assertEqual(int(self.ctl.stub_held_count()), 0)
        # The earlier receipt is immutable even though its withheld buffer has
        # now completed with contradictory records.
        self.assertEqual(self.text(receipt), before)
        self.assertTrue(self.session.stop())
        self.assertTrue(self.session.close())

    def test_negative_record_released_before_end_fails(self):
        self.open_clean()
        self.begin()
        self.emit_graph_kernels(0)
        self.ctl.stub_deliver()
        self.emit_kernel(11, 101, context_id=999)
        self.ctl.stub_set_hold_next(1)
        self.ctl.stub_deliver()
        self.assertEqual(int(self.ctl.stub_held_count()), 1)
        self.ctl.stub_release_held()
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'FAIL')
        self.assertIn('known_wrong_context', receipt['reasons'])
        self.assertNotIn('buffer_outstanding', receipt['reasons'])

    def test_withheld_record_with_failed_flush_is_inconclusive(self):
        self.open_clean()
        self.begin()
        self.emit_graph_kernels(0)
        self.ctl.stub_deliver()
        self.emit_kernel(11, 101, context_id=999)
        self.ctl.stub_set_hold_next(1)
        self.ctl.stub_deliver()
        self.ctl.stub_set_deliver_on_flush(0)
        self.ctl.stub_set_flush_rc(1)
        receipt = self.end()
        self.ctl.stub_set_flush_rc(0)
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('buffer_outstanding', receipt['reasons'])
        self.assertIn('flush_failed', receipt['reasons'])
        self.ctl.stub_release_held()
        self.assertTrue(self.session.stop())
        self.assertTrue(self.session.close())

    def test_reentrant_completion_is_refused(self):
        self.open_clean()
        self.ctl.stub_set_reenter(1)
        self.begin()
        self.emit_graph_kernels(0)
        receipt = self.end()
        self.ctl.stub_set_reenter(0)
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('ring_overflow', receipt['reasons'])
        self.assertTrue(receipt['replay']['flags']['reentrant_refused'])
        self.assertGreater(receipt['replay']['counts']
                           ['callback_reentrant_refused'], 0)

    def test_corrupt_record_is_inconclusive(self):
        self.open_clean()
        self.begin()
        self.emit_graph_kernels(0)
        self.ctl.stub_emit_corrupt()
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('parse_error', receipt['reasons'])
        self.assertGreater(receipt['replay']['counts']['parse_errors'], 0)

    def test_unknown_activity_is_inconclusive(self):
        self.open_clean()
        self.begin()
        self.emit_graph_kernels(0)
        self.ctl.stub_emit_unknown(77)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('unknown_activity_record', receipt['reasons'])
        self.assertTrue(receipt['replay']['flags']['unknown_activity'])

    def test_flush_failure_is_inconclusive(self):
        self.open_clean()
        self.begin()
        self.emit_graph_kernels(0)
        self.ctl.stub_set_flush_rc(1)
        receipt = self.end()
        self.ctl.stub_set_flush_rc(0)
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('flush_failed', receipt['reasons'])
        self.assertFalse(receipt['replay']['window']['flushed'])


class ObserverLifecycleTests(ObserverTestBase):
    def test_repeated_start_refused(self):
        self.new_session(start=True)
        self.assert_code('bad_state', self.session.start)

    def test_bind_before_start_refused(self):
        self.make_bank()
        self.new_session()
        self.assert_code('bad_state', self.session.bind_bank, self.probe)

    def test_begin_before_bind_refused(self):
        self.new_session(start=True)
        self.assert_code('bad_state', self.session.begin_window,
                         route_generation=ROUTE_GENERATION)

    def test_end_without_window_refused(self):
        self.make_bank()
        self.new_session(start=True, bind=True)
        self.assert_code('bad_state', self.session.end_window, self.proof())

    def test_stop_during_window_refused(self):
        self.open_clean()
        self.begin()
        self.assert_code('window_active', self.session.stop)
        self.assert_code('window_active', self.session.close)
        self.emit_graph_kernels(0)
        self.end()

    def test_stop_and_close_are_ordered_and_idempotent(self):
        self.open_clean()
        self.begin()
        self.emit_graph_kernels(0)
        self.end()
        self.assertTrue(self.session.stop())
        self.assertTrue(self.session.stop())
        self.assertTrue(self.session.close())
        self.assertTrue(self.session.close())
        self.assertEqual(self.session.state, 'closed')

    def test_stop_before_start_refused_and_close_succeeds(self):
        self.new_session()
        self.assert_code('bad_state', self.session.stop)
        self.assertTrue(self.session.close())

    def test_held_buffer_preserves_ownership_at_stop(self):
        self.new_session(start=True)
        self.ctl.stub_emit_unknown(77)
        self.ctl.stub_set_hold_next(1)
        self.ctl.stub_deliver()
        self.assert_code('observer_busy', self.session.stop)
        self.ctl.stub_release_held()
        self.assertTrue(self.session.stop())
        self.assertTrue(self.session.close())

    def test_subscriber_conflict_refused(self):
        self.ctl.stub_set_foreign_subscriber(1)
        self.new_session()
        self.assert_code('subscriber_conflict', self.session.start)
        self.assertTrue(self.session.close())

    def test_subscriber_conflict_preserves_foreign_callbacks(self):
        self.ctl.stub_install_foreign_callbacks()
        self.ctl.stub_set_foreign_subscriber(1)
        self.new_session()
        self.assert_code('subscriber_conflict', self.session.start)
        self.ctl.stub_emit_unknown(77)
        self.assertEqual(int(self.ctl.stub_deliver()), 1)
        self.assertEqual(int(self.ctl.stub_foreign_completions()), 1)
        self.ctl.stub_emit_unknown(78)
        self.assertEqual(int(self.ctl.stub_deliver()), 1)
        self.assertEqual(int(self.ctl.stub_foreign_completions()), 2)
        self.assertTrue(self.session.close())

    def test_start_cleanup_unsubscribe_failure_quarantines_and_retries(self):
        self.ctl.stub_set_register_rc(1)
        self.ctl.stub_set_unsubscribe_rc(1)
        self.new_session()
        self.assert_code('cleanup_failed', self.session.start)
        self.assertEqual(self.session.state, 'start_failed')
        self.ctl.stub_set_register_rc(0)
        self.ctl.stub_set_unsubscribe_rc(0)
        self.assertTrue(self.session.stop())
        self.assertTrue(self.session.close())

    def test_start_cleanup_flush_failure_quarantines_and_retries(self):
        self.ctl.stub_set_enable_activity_rc(1)
        self.ctl.stub_set_flush_rc(1)
        self.new_session()
        self.assert_code('cleanup_failed', self.session.start)
        self.assertEqual(self.session.state, 'start_failed')
        self.assertEqual(self.ctl.stub_has_subscriber(), 1)
        self.ctl.stub_set_enable_activity_rc(0)
        self.ctl.stub_set_flush_rc(0)
        self.assertTrue(self.session.stop())
        self.assertTrue(self.session.close())

    def test_stop_disable_failure_quarantines_and_retries(self):
        self.new_session(start=True)
        self.ctl.stub_set_disable_activity_rc(1)
        self.assert_code('cleanup_failed', self.session.stop)
        self.ctl.stub_set_disable_activity_rc(0)
        self.assertTrue(self.session.stop())
        self.assertTrue(self.session.close())

    def test_stop_flush_failure_quarantines_and_retries(self):
        self.new_session(start=True)
        self.ctl.stub_set_flush_rc(1)
        self.assert_code('cleanup_failed', self.session.stop)
        self.ctl.stub_set_flush_rc(0)
        self.assertTrue(self.session.stop())
        self.assertTrue(self.session.close())

    def test_stop_unsubscribe_failure_quarantines_and_retries(self):
        self.new_session(start=True)
        self.ctl.stub_set_unsubscribe_rc(1)
        self.assert_code('unsubscribe_failed', self.session.stop)
        self.ctl.stub_set_unsubscribe_rc(0)
        self.assertTrue(self.session.stop())
        self.assertTrue(self.session.close())

    def test_runtime_version_mismatch_refused(self):
        self.ctl.stub_set_version(120000)
        self.new_session()
        self.assert_code('cupti_version_mismatch', self.session.start)
        self.assertTrue(self.session.close())

    def test_version_query_failure_refused(self):
        self.ctl.stub_set_version_rc(1)
        self.new_session()
        self.assert_code('version_query_failed', self.session.start)
        self.assertTrue(self.session.close())

    def test_register_failure_refused(self):
        self.ctl.stub_set_register_rc(1)
        self.new_session()
        self.assert_code('activity_register_failed', self.session.start)
        self.assertTrue(self.session.close())

    def test_callback_enable_failure_refused(self):
        self.ctl.stub_set_enable_callback_rc(1)
        self.new_session()
        self.assert_code('callback_enable_failed', self.session.start)
        self.assertTrue(self.session.close())

    def test_activity_enable_failure_refused(self):
        self.ctl.stub_set_enable_activity_rc(1)
        self.new_session()
        self.assert_code('activity_enable_failed', self.session.start)
        self.assertTrue(self.session.close())

    def test_bank_contract_refusals(self):
        self.make_bank()
        self.new_session(start=True)
        self.bank.manager._flash_s5_capture_owner = object()
        self.assert_code('capture_owner_mismatch', self.session.bind_bank,
                         self.probe)

    def test_bank_not_captured_refused(self):
        self.make_bank(captured=False)
        self.new_session(start=True)
        self.assert_code('bank_not_captured', self.session.bind_bank, self.probe)

    def test_descriptor_inventory_missing_refused(self):
        self.make_bank()
        self.new_session(start=True)
        local = self.bank.pairs[0][1]
        del self.bank.handles[local]
        self.assert_code('descriptor_inventory_missing',
                         self.session.bind_bank, self.probe)

    def test_duplicate_graph_handle_refused(self):
        self.make_bank()
        self.new_session(start=True)
        descriptor = self.bank.descriptors[1]
        self.bank.manager.graphs[descriptor] = self.bank.manager.graphs[
            self.bank.descriptors[0]]
        self.assert_code('graph_duplicate', self.session.bind_bank, self.probe)

    def test_bank_mutation_during_bind_refused(self):
        self.make_bank([{'id': 11, 'nodes': [(101, 0)], 'mutate': True}])
        self.new_session(start=True)
        self.assert_code('bank_mutated', self.session.bind_bank, self.probe)

    def test_route_generation_mismatch_refused(self):
        self.open_clean()
        self.assert_code('route_generation_mismatch',
                         self.session.begin_window,
                         route_generation=ROUTE_GENERATION + 1)
        self.begin()
        self.end()

    def test_unknown_descriptor_refused(self):
        self.open_clean()
        self.assert_code('descriptor_not_in_bank', self.session.begin_window,
                         route_generation=ROUTE_GENERATION, descriptor=object())

    def test_capture_owner_mutation_before_begin_refused(self):
        self.open_clean()
        self.bank.manager._flash_s5_capture_owner = object()
        self.assert_code('bank_mutated', self.session.begin_window,
                         route_generation=ROUTE_GENERATION)
        self.assertEqual(self.session.state, 'bound')
        self.session.stop()

    def test_capture_state_mutation_before_begin_refused(self):
        self.open_clean()
        self.bank.captured = False
        self.assert_code('bank_mutated', self.session.begin_window,
                         route_generation=ROUTE_GENERATION)
        self.session.stop()

    def test_graph_handle_mutation_before_begin_refused(self):
        self.open_clean()
        self.bank.manager.graphs[self.bank.descriptors[0]].handle = 999999
        self.assert_code('bank_mutated', self.session.begin_window,
                         route_generation=ROUTE_GENERATION)
        self.session.stop()

    def test_handle_destroyed_before_begin_refused(self):
        self.open_clean()
        local = self.bank.pairs[0][1]
        self.bank.handles[local].destroyed = True
        self.assert_code('bank_mutated', self.session.begin_window,
                         route_generation=ROUTE_GENERATION)
        self.session.stop()

    def test_storage_epoch_mutation_before_end_is_inconclusive(self):
        self.open_clean()
        self.begin()
        self.emit_graph_kernels(0)
        self.probe.storage_epoch = 999
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('bank_mutated', receipt['reasons'])
        self.assertFalse(receipt['complete'])
        self.session.stop()

    def test_live_route_mutation_before_end_is_inconclusive(self):
        self.open_clean()
        self.begin()
        self.emit_graph_kernels(0)
        self.probe.route_generation = ROUTE_GENERATION + 1
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('bank_mutated', receipt['reasons'])
        self.session.stop()

    def test_graph_object_mutation_before_end_is_inconclusive(self):
        self.open_clean()
        self.begin()
        self.emit_graph_kernels(0)
        self.bank.manager.graphs[self.bank.descriptors[0]].handle = 999999
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('bank_mutated', receipt['reasons'])
        self.session.stop()

    def test_session_limit_exceeded(self):
        limits = ro.ObserverLimits(buffer_count=64,
                                   buffer_bytes=64 * 1024 * 1024)
        self.assert_code('limit_exceeded', ro.ObserverSession,
                         self.binding, limits)


    def test_small_receipt_capacity_fails_closed(self):
        self.open_clean(ro.ObserverLimits(max_receipt_bytes=2048))
        self.begin()
        self.emit_graph_kernels(0)
        self.assert_code('receipt_too_large', self.session.end_window, self.proof())
        self.assertEqual(self.session.state, 'ended')
        self.session.stop()

class ObserverProofTests(ObserverTestBase):
    def test_timestamp_failure_retains_window_token_and_retries(self):
        self.open_clean()
        self.begin()
        self.emit_graph_kernels(0)
        self.ctl.stub_set_timestamp_rc(1)
        self.assert_code('timestamp_failed', self.session.end_window, self.proof())
        self.assertEqual(self.session.state, 'window')
        self.assert_code('window_active', self.session.stop)
        self.ctl.stub_set_timestamp_rc(0)
        receipt = self.end()
        self.assertEqual(receipt['verdict'], 'PASS')
        self.assertTrue(self.session.stop())

    def _clean_receipt(self, **proof):
        self.open_clean()
        self.begin()
        self.emit_graph_kernels(0)
        return self.end(self.proof(**proof))

    def test_route_generation_mismatch_fails(self):
        receipt = self._clean_receipt(route_generation=ROUTE_GENERATION + 1)
        self.assertEqual(receipt['verdict'], 'FAIL')
        self.assertIn('caller_route_generation_mismatch', receipt['reasons'])

    def test_storage_epoch_mismatch_fails(self):
        receipt = self._clean_receipt(storage_epoch=STORAGE_EPOCH + 1)
        self.assertEqual(receipt['verdict'], 'FAIL')
        self.assertIn('caller_storage_epoch_mismatch', receipt['reasons'])

    def test_drain_not_completed_inconclusive(self):
        receipt = self._clean_receipt(controller_drained=False)
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('drain_not_completed', receipt['reasons'])

    def test_no_relevant_consumers_inconclusive(self):
        receipt = self._clean_receipt(relevant_consumers=0)
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE')
        self.assertIn('drain_consumers_missing', receipt['reasons'])

    def test_caller_proof_is_unverified(self):
        receipt = self._clean_receipt()
        self.assertFalse(receipt['caller_proof']['verified_by_observer'])
        self.assertTrue(receipt['caller_proof']['route_generation_matches'])
        self.assertTrue(receipt['caller_proof']['storage_epoch_matches'])


class ObserverThreadingTests(ObserverTestBase):
    def test_threaded_callbacks_do_not_deadlock(self):
        code = r"""
import ctypes
import copy
import sys
sys.path.insert(0, %r)
import test_e3_replay_observer as t
import replay_observer as ro

lib = ctypes.CDLL(%r)
t._declare(lib)
binding = ro.ReplayBinding(%r, %r, ro.helper_source_sha256(), %r,
                           cupti_library_path=%r)
proof = ro.DrainedProof(worker_rank=0, worker_incarnation='worker-1',
                        engine_incarnation='engine-1',
                        route_generation=4, storage_epoch=3,
                        relevant_consumers=1, controller_drained=True)


def make_session():
    lib.stub_reset()
    bank = t.FakeBank()
    split = t.FakeSplit(lib.stub_new_green(7, 8), 8)
    router = t.FakeRouter(3, 4)
    manager = t.FakeManager(split)
    bank.manager = manager
    graph_handle = int(lib.stub_new_graph(11))
    graph = t.FakeGraph(graph_handle)
    node_one = int(lib.stub_new_node(101, 0))
    node_two = int(lib.stub_new_node(102, 0))
    lib.stub_graph_add_node(graph_handle, node_one)
    lib.stub_graph_add_node(graph_handle, node_two)
    descriptor = object()
    local = object()
    bank.pairs.append((descriptor, local))
    bank.handles[local] = t.FakeHandle(local, bank)
    manager.graphs[descriptor] = graph
    bank.descriptors.append(descriptor)
    bank.graph_handles.append(graph_handle)
    bank.graph_ids.append(11)
    bank.node_handles.append([node_one, node_two])
    probe = t.FakeProbe(bank, router, split)
    return ro.ObserverSession(binding), probe


session, probe = make_session()
session.start()
session.bind_bank(probe)
lib.stub_emit_context(7, 1, 8, 0)
lib.stub_set_timestamp(1000)
session.begin_window(route_generation=4)
lib.stub_set_timestamp(2000)
lib.stub_emit_kernel(7, 11, 101, 1100, 1200, 1200)
lib.stub_emit_kernel(7, 11, 102, 1200, 1300, 1300)
lib.stub_set_threaded_flush(1)
lib.stub_set_timestamp(3000)
receipt = session.end_window(proof)
assert receipt['verdict'] == 'PASS', receipt['reasons']
assert session.stop()
assert session.close()

# A cleanup failure after a callback-thread flush is quarantined and can be
# retried; close must not free state while callbacks may still run.
session, probe = make_session()
lib.stub_set_threaded_flush(1)
lib.stub_set_enable_activity_rc(1)
lib.stub_set_unsubscribe_rc(1)
try:
    session.start()
except ro.ReplayObserverError as exc:
    assert exc.code == 'cleanup_failed', exc.code
else:
    raise AssertionError('expected start cleanup failure')
lib.stub_set_enable_activity_rc(0)
lib.stub_set_unsubscribe_rc(0)
assert session.stop()
assert session.close()

session, probe = make_session()
lib.stub_set_threaded_flush(1)
lib.stub_set_disable_activity_rc(1)
session.start()
try:
    session.stop()
except ro.ReplayObserverError as exc:
    assert exc.code == 'cleanup_failed', exc.code
else:
    raise AssertionError('expected stop cleanup failure')
lib.stub_set_disable_activity_rc(0)
assert session.stop()
assert session.close()
print('threaded-ok')
""" % (HERE, self.library_path, self.library_path, self.library_sha,
       self.cupti_sha, self.fake_cupti)
        try:
            completed = subprocess.run(
                [sys.executable, '-I', '-c', code],
                capture_output=True, text=True, timeout=10)
        except subprocess.TimeoutExpired:
            self.fail('threaded callback path deadlocked past its deadline')
        if completed.returncode:
            self.fail('threaded child failed (%d): %s' % (
                completed.returncode, completed.stderr))
        self.assertIn('threaded-ok', completed.stdout)


class ObserverPythonTests(ObserverTestBase):
    def test_limits_reject_non_strict_integers(self):
        for value in (True, 1.0, '1', 0, 65):
            with self.subTest(value=value):
                with self.assertRaises(ro.ReplayObserverError) as raised:
                    ro.ObserverLimits(buffer_count=value)
                self.assertEqual(raised.exception.code, 'bad_limits')
        with self.assertRaises(ro.ReplayObserverError):
            ro.ObserverLimits(max_nodes_per_graph=10, max_nodes_total=9)
        with self.assertRaises(ro.ReplayObserverError):
            ro.ObserverLimits(max_receipt_bytes=1)

    def test_proof_has_no_default_drain(self):
        with self.assertRaises(TypeError):
            ro.DrainedProof(worker_rank=0, worker_incarnation='w',
                            engine_incarnation='e', route_generation=1,
                            storage_epoch=1, relevant_consumers=1)
        for value in (1, 0, None):
            with self.subTest(value=value):
                with self.assertRaises(ro.ReplayObserverError) as raised:
                    ro.DrainedProof(0, 'w', 'e', 1, 1, 1, value)
                self.assertEqual(raised.exception.code, 'bad_proof')
        with self.assertRaises(ro.ReplayObserverError):
            ro.DrainedProof(0, '', 'e', 1, 1, 1, True)
        with self.assertRaises(ro.ReplayObserverError):
            ro.DrainedProof(0, 'w\x01', 'e', 1, 1, 1, True)

    def test_session_rejects_foreign_limits(self):
        self.assert_code('bad_limits', ro.ObserverSession, self.binding, object())

    def test_binding_hash_and_source_mismatches(self):
        self.assert_code(
            'library_sha_mismatch', ro.ReplayBinding, self.library_path,
            '0' * 64, ro.helper_source_sha256(), self.cupti_sha,
            cupti_library_path=self.fake_cupti)
        self.assert_code(
            'source_sha_mismatch', ro.ReplayBinding, self.library_path,
            self.library_sha, '0' * 64, self.cupti_sha,
            cupti_library_path=self.fake_cupti)
        self.assert_code(
            'cupti_sha_mismatch', ro.ReplayBinding, self.library_path,
            self.library_sha, ro.helper_source_sha256(), '0' * 64,
            cupti_library_path=self.fake_cupti)
        self.assert_code(
            'bad_binding', ro.ReplayBinding, 'libobserver.so',
            self.library_sha, ro.helper_source_sha256(), self.cupti_sha,
            cupti_library_path=self.fake_cupti)
        self.assert_code(
            'bad_binding', ro.ReplayBinding, self.library_path,
            'not-a-digest', ro.helper_source_sha256(), self.cupti_sha,
            cupti_library_path=self.fake_cupti)
        self.assert_code(
            'cupti_library_missing', ro.ReplayBinding, self.library_path,
            self.library_sha, ro.helper_source_sha256(), self.cupti_sha,
            cupti_library_path=os.path.join(self.build_dir, 'missing.so'))

    def test_receipt_is_deeply_immutable(self):
        self.open_clean()
        self.begin(self.bank.descriptors[0])
        self.emit_graph_kernels(0)
        receipt = self.end()
        self.assertIsInstance(receipt, MappingProxyType)
        self.assertIsInstance(receipt['replay'], MappingProxyType)
        self.assertIsInstance(receipt['replay']['counts'], MappingProxyType)
        self.assertIsInstance(receipt['replay']['graph_slots'], tuple)
        with self.assertRaises(TypeError):
            receipt['verdict'] = 'FAIL'
        with self.assertRaises(TypeError):
            receipt['replay']['counts']['parse_errors'] = 99

    def test_receipt_has_no_raw_handles_names_or_paths(self):
        self.open_clean()
        self.begin(self.bank.descriptors[0])
        self.emit_graph_kernels(0)
        receipt = self.end()
        text = self.text(receipt)
        forbidden_keys = {'handle', 'pointer', 'address', 'name', 'kernel_name',
                          'node_id', 'exec_id',
                          'library_path', 'cupti_library_path'}
        seen = set()

        def walk(value):
            if isinstance(value, MappingProxyType):
                for key, item in value.items():
                    seen.add(key)
                    walk(item)
            elif isinstance(value, tuple):
                for item in value:
                    walk(item)

        walk(receipt)
        # Schema4 deliberately exposes CUPTI numeric IDs, never CUDA handles.
        self.assertEqual(receipt['replay']['context_diagnostics']['id_namespace'],'CUPTI')
        self.assertFalse(forbidden_keys & seen)
        self.assertNotIn('0x', text)
        for raw in (str(self.bank.graph_handles[0]),
                    str(self.green), str(self.bank.node_handles[0][0])):
            self.assertNotIn(raw, text)
        self.assertIn('"scope": "exercised descriptors only"', text)

    def test_receipt_binds_identities_and_limits(self):
        limits = ro.ObserverLimits(max_receipt_bytes=4 * 1024 * 1024)
        self.open_clean(limits)
        self.begin(self.bank.descriptors[0])
        self.emit_graph_kernels(0)
        receipt = self.end()
        self.assertEqual(receipt['binding']['library_sha256'],
                         self.library_sha)
        self.assertEqual(receipt['binding']['source_sha256'],
                         ro.helper_source_sha256())
        self.assertEqual(receipt['binding']['cupti_library_sha256'],
                         self.cupti_sha)
        self.assertEqual(receipt['binding']['abi'], 1)
        self.assertEqual(receipt['limits'], limits.as_record())
        self.assertGreater(receipt['reserved_bytes'], 0)
        self.assertEqual(receipt['source_identities']['compiled_api'], 130001)
        self.assertEqual(receipt['source_identities']['runtime_api'], 130001)
        self.assertIsInstance(receipt['limitations'], tuple)

    def test_import_has_no_runtime_side_effects(self):
        code = (
            "import sys\n"
            "sys.path.insert(0, %r)\n"
            "import replay_observer\n"
            "assert 'ctypes' not in sys.modules\n"
            "assert 'torch' not in sys.modules\n"
            "print('ok')\n" % PATCHES
        )
        completed = subprocess.run([sys.executable, '-I', '-c', code],
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), 'ok')

    def test_helper_source_hash_is_stable(self):
        first = ro.helper_source_sha256()
        second = ro.helper_source_sha256()
        self.assertEqual(first, second)
        self.assertRegex(first, r'^[0-9a-f]{64}$')


if __name__ == '__main__':
    unittest.main()
