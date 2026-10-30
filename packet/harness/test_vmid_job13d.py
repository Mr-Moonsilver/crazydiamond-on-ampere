"""CPU acceptance for VM13d real private initialization + pending query.

The positive constructor path below is the probe's own
``build_bindings``/``run_private_scope`` code, exercised with the real
``LaneRuntime``/``WorkerState``/``LanePrivateBindings``/pinned WorkspaceManager
and only the CUDA device leaves replaced by address-bearing tensors.  The
ledger is the real ``NativePrefillLedger`` over the controlled producer shape.
Receipt and composition grading use the actual producer dictionaries; the
strict negatives are CPU mutations, never injected into a CUDA call.
"""
import contextlib
import copy
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'patches/s5_engine'), str(ROOT / 'harness')]

import lane_private  # noqa: E402
import native_placement as np_  # noqa: E402
import native_prefill_support as support  # noqa: E402
import test_e3_lane_private as lp  # noqa: E402
import test_e3_native_placement as placement_fixture  # noqa: E402
import vmid_job13d as job  # noqa: E402

NS = types.SimpleNamespace
_PRODUCER = {'bridge': None}


def _emit(kind, **kwargs):
    bridge = _PRODUCER['bridge']
    if bridge is not None and bridge._stack:
        bridge.activity(kind, stream=support.MAIN, **kwargs)


class ObservedTensor(lp.Tensor):
    """A real address-bearing tensor that emits the real launch it stands for."""

    def fill_(self, value):
        _emit('kernel')
        return super().fill_(value)

    def zero_(self):
        _emit('memset')
        return super().zero_()

    def copy_(self, other, non_blocking=False):
        _emit('memcpy', copy_kind=8)
        return super().copy_(other, non_blocking)

    def __getitem__(self, item):
        out = super().__getitem__(item)
        if type(out) is lp.Tensor:
            return ObservedTensor(out.shape, out.dtype, storage=out._storage,
                                  offset=out.offset, device=out.device)
        return out


def cpu_torch():
    torch = lp.fake_torch()
    torch.empty = lambda shape, dtype, device=None: ObservedTensor(
        shape, dtype, device=device or 'cuda:0')
    torch.full = lambda shape, value, dtype, device=None: ObservedTensor(
        shape, dtype, device=device or 'cuda:0').fill_(value)
    torch.tensor = lambda values, dtype, device=None: ObservedTensor(
        (len(values),), dtype, device=device or 'cuda:0')
    torch.device = lambda value: value
    return torch


class RealConstructorPathTests(unittest.TestCase):
    def setUp(self):
        _PRODUCER['bridge'] = None
        self.torch = cpu_torch()
        self.ws = lp.FakeWorkspaceModule(lanes=2, torch=self.torch)
        self.stream = lp.FakeStream(700)
        self.bundle = job.build_bindings(
            self.torch, workspace_module=self.ws,
            prefill_stream=self.stream, chunk_tokens=job.CHUNK_TOKENS,
            width=job.WIDTH, req_index=job.REQ_INDEX, seed=job.SEED)

    def make_ledger(self):
        bridge = support.FakeBridge()
        ledger = support.make_ledger(bridge, torch_module=support.make_torch())
        support.declare(ledger)
        return bridge, ledger

    def run_scope(self, ledger, op_seq, chunk=None):
        _PRODUCER['bridge'] = ledger.bridge
        try:
            job.run_private_scope(
                self.bundle, ledger,
                chunk or job.new_chunk(self.bundle),
                self.bundle['transaction'], op_seq=op_seq, layer_idx=0,
                torch_module=self.torch)
        finally:
            _PRODUCER['bridge'] = None

    def seal(self, bridge, ledger):
        ledger.owner_finish('lane')
        bridge.close_window()
        ledger.finish()
        return ledger.seal()

    def test_discovery_is_the_real_binding_over_real_owned_buffers(self):
        bindings = self.bundle['bindings']
        self.assertIsInstance(bindings, lane_private.LanePrivateBindings)
        self.assertIs(bindings.uid, self.bundle['uid'])
        self.assertIs(bindings.decode_uid, self.bundle['decode_uid'])
        self.assertEqual(bindings.width, job.WIDTH)
        self.assertGreater(self.bundle['workspace_bytes'], 0)
        # The prefill lane was funded from the stock high-water mark.
        manager = self.ws.current_workspace_manager()
        self.assertIs(manager._current_workspaces[1],
                      bindings._workspace_buffer)
        self.assertTrue(manager._locked)
        self.assertIs(self.bundle['runtime'].prefill_workspace, 1)

    def test_sharing_is_preserved_across_the_two_layers(self):
        locks = self.bundle['bindings'].marlin_locks
        self.assertEqual(len(locks), 2)
        self.assertIs(locks[0][3], locks[1][3])
        self.assertIs(locks[0][4], locks[1][4])

    def test_first_scope_runs_the_actual_leaves_and_restores_owners(self):
        bridge, ledger = self.make_ledger()
        self.bundle['bindings'].attach_ledger(ledger)
        self.run_scope(ledger, op_seq=1)
        receipt = self.seal(bridge, ledger)
        self.assertEqual(receipt['verdict'], 'PASS', receipt)
        counts = job.uid_lock_counts(receipt)
        self.assertEqual(counts['uid_init'], {'kernel': 1, 'memcpy': 1})
        self.assertEqual(counts['lock_init'], {'memset': 1})
        self.assertEqual(self.bundle['uid'].read(0), job.SEED)
        self.assertTrue(all(self.bundle['uid'].read(i) == -2
                            for i in range(1, 4)))
        self.assertTrue(all(row[4].read(i) == 0
                            for row in self.bundle['bindings'].marlin_locks
                            for i in range(row[4].numel())))
        self.assertIs(self.bundle['fc'].QSA_REQ_UID,
                      self.bundle['decode_uid'])
        for owner, original in zip(self.bundle['attn'],
                                   self.bundle['original']):
            self.assertIs(owner.topk_indices_buffer, original)
        self.assertIs(self.bundle['lock_owner'].workspace,
                      self.bundle['lock_original'])
        containers = job._container_offers(receipt)
        self.assertEqual(len(containers), 1)
        leaves = [row for row in receipt['offers']
                  if row.get('suboperation') in ('uid_init', 'lock_init')]
        self.assertEqual({row['suboperation'] for row in leaves},
                         {'uid_init', 'lock_init'})
        self.assertTrue(all(row['parent'] == containers[0]['id']
                            for row in leaves))
        self.assertNotIn(containers[0]['id'], job._joined_offers(receipt))

    def test_repeated_scope_reoffers_uid_but_not_the_retained_lock(self):
        bridge, ledger = self.make_ledger()
        self.bundle['bindings'].attach_ledger(ledger)
        self.run_scope(ledger, op_seq=1)
        self.run_scope(ledger, op_seq=2)
        receipt = self.seal(bridge, ledger)
        self.assertEqual(receipt['verdict'], 'PASS', receipt)
        counts = job.uid_lock_counts(receipt)
        self.assertEqual(counts['uid_init'], {'kernel': 2, 'memcpy': 2})
        self.assertEqual(counts['lock_init'], {'memset': 1})
        self.assertEqual(len(self.bundle['bindings']._lock_initialized), 1)

    def test_missing_provider_still_runs_the_real_body(self):
        self.run_scope_untracked = None
        _PRODUCER['bridge'] = None
        job.run_private_scope(
            self.bundle, _NullLedger(), job.new_chunk(self.bundle),
            self.bundle['transaction'], op_seq=1, layer_idx=0,
            torch_module=self.torch)
        self.assertEqual(self.bundle['uid'].read(0), job.SEED)
        self.assertTrue(all(row[4].read(i) == 0
                            for row in self.bundle['bindings'].marlin_locks
                            for i in range(row[4].numel())))
        self.assertIs(self.bundle['fc'].QSA_REQ_UID,
                      self.bundle['decode_uid'])

    def test_attach_refuses_a_provider_without_private_leaf(self):
        with self.assertRaises(lane_private.LanePrivateError):
            self.bundle['bindings'].attach_ledger(object())


class _NullLedger:
    """A ledger with no private-leaf provider: the scope body must still run."""

    @contextlib.contextmanager
    def segment(self, *args, **kwargs):
        yield NS()


def real_private_receipt(*, snapshot=False):
    """The sealed receipt from the probe's own constructor/scope path."""
    _PRODUCER['bridge'] = None
    torch = cpu_torch()
    workspace = lp.FakeWorkspaceModule(lanes=2, torch=torch)
    bundle = job.build_bindings(torch, workspace_module=workspace,
                                prefill_stream=lp.FakeStream(700))
    bridge = support.FakeBridge()
    ledger = support.make_ledger(bridge, torch_module=support.make_torch())
    support.declare(ledger)
    bundle['bindings'].attach_ledger(ledger)
    _PRODUCER['bridge'] = bridge
    try:
        job.run_private_scopes(
            bundle, ledger,
            [(job.new_chunk(bundle), 0), (job.new_chunk(bundle), 0)],
            bundle['transaction'], op_seq=1, torch_module=torch)
    finally:
        _PRODUCER['bridge'] = None
    ledger.owner_finish('lane')
    decode, _ = support.decode_ledger(bridge)
    support.run_decode_offers(decode, bridge)
    bridge.host_control('cudaEventQuery', return_value=600)
    bridge.host_control('cudaEventQuery', return_value=0)
    bridge.close_window()
    declared, window = support.finish_decode(decode, bridge)
    ledger.finish()
    receipt = ledger.seal()
    if not snapshot:
        return receipt
    whole = ledger.snapshot()
    composed = np_.compose_overlap(whole['native'], whole['offers'],
        decode_declared=declared, decode_window=window,
        prefill_declared=receipt['declared'], prefill_window=receipt['window'],
        prefill_receipt=receipt)
    return dict(native_report=whole['native'], offer_report=whole['offers'],
                decode_declared=declared, decode_window=window,
                prefill_receipt=receipt, composed=composed)


class PendingEvidenceTests(unittest.TestCase):
    def test_exact_pending_query_is_host_only_and_completion_is_separate(self):
        bridge = support.FakeBridge()
        cid = bridge.host_control('cudaEventQuery', return_value=600)
        bridge.host_control('cudaEventQuery', return_value=0)
        evidence = job.pending_evidence(bridge.report()['native'])
        self.assertEqual(evidence['pending'], [cid])
        self.assertEqual(len(evidence['success']), 1)
        self.assertTrue(job.pending_not_joined(evidence))

    def test_pending_on_a_non_query_is_refused(self):
        bridge = support.FakeBridge()
        bridge.host_control('cudaEventRecord', return_value=600)
        evidence = job.pending_evidence(bridge.report()['native'])
        self.assertEqual(evidence['pending'], [])
        self.assertEqual(len(evidence['refused']), 1)
        self.assertFalse(job.pending_not_joined(evidence))

    def test_pending_query_with_joined_activity_is_not_host_only(self):
        bridge = support.FakeBridge()
        cid = bridge.host_control('cudaEventQuery', return_value=600)
        bridge.activity('kernel', stream=support.MAIN, offer=1,
                        name='cudaEventQuery', return_value=600)
        evidence = job.pending_evidence(bridge.report()['native'])
        self.assertIn(cid, evidence['pending'])
        self.assertTrue(set(evidence['pending'])
                        & set(evidence['with_activity']))


class CompositionMutationTests(unittest.TestCase):
    """Every negative is a CPU mutation of a real producer dictionary."""

    @classmethod
    def setUpClass(cls):
        raw = placement_fixture.positive()
        cls.fix = {key: copy.deepcopy(raw[key]) for key in (
            'native', 'offers', 'decode_declared', 'decode_window',
            'prefill_declared', 'prefill_window', 'prefill_receipt')}
        cls.base = placement_fixture.compose(cls.fix)

    def compose(self, mutate):
        fix = copy.deepcopy(self.fix)
        mutate(fix['native'], fix['prefill_declared'])
        return placement_fixture.compose(fix)

    def test_baseline_is_pass_with_exact_conservation(self):
        self.assertEqual(self.base['verdict'], 'PASS', self.base)
        self.assertTrue(self.base['conservation']['proved'])

    def test_pending_event_query_is_conserved_without_activity(self):
        def fn(native, _declared):
            native['apis'].append(dict(
                correlation_id=900001, domain='runtime',
                name='cudaEventQuery', start=native['window']['start_ns'] + 1,
                end=native['window']['start_ns'] + 2, thread=1101,
                return_value=600))
        out = self.compose(fn)
        self.assertEqual(out['verdict'], 'PASS', out)
        self.assertEqual(out['conservation']['host_apis'], 1)

    def test_six_hundred_on_non_query_is_refused(self):
        def fn(native, _declared):
            native['apis'].append(dict(
                correlation_id=900002, domain='runtime',
                name='cudaEventRecord', start=native['window']['start_ns'] + 1,
                end=native['window']['start_ns'] + 2, thread=1101,
                return_value=600))
        out = self.compose(fn)
        self.assertEqual(out['verdict'], 'FAIL', out)
        self.assertTrue(any(reason.startswith('conservation:host_api_failed')
                            for reason in out['failures']), out)

    def test_pending_query_with_gpu_activity_is_refused(self):
        def fn(native, declared):
            oid = int(next(iter(declared)))
            offer = declared[str(oid)]
            native['apis'].append(dict(
                correlation_id=900003, domain='runtime',
                name='cudaEventQuery', start=native['window']['start_ns'] + 1,
                end=native['window']['start_ns'] + 2, thread=offer['thread'],
                return_value=600))
            native['external'].append(dict(kind='custom0',
                                           correlation_id=900003,
                                           external_id=oid))
            native['activities'].append(dict(
                correlation_id=900003, kind='kernel', count=1, graph_id=0,
                context_id=offer['context_id'],
                stream_id=offer['stream_id'], copy_kind=0,
                name='cudaEventQuery'))
            native['activity_records'] += 1
            native['partition_totals']['kernel'] += 1
        out = self.compose(fn)
        self.assertEqual(out['verdict'], 'FAIL', out)
        self.assertIn('prefill_grade_fail:api_activity_kind',
                      out['failures'])

    def test_primary_context_work_is_rejected_with_a_named_reason(self):
        def fn(native, _declared):
            native['activities'][0]['context_id'] += 1
        out = self.compose(fn)
        self.assertEqual(out['verdict'], 'FAIL', out)
        self.assertIn('prefill_grade_fail:wrong_context', out['failures'])

    def test_unoffered_eager_work_is_rejected_with_a_named_reason(self):
        def fn(native, _declared):
            cid = 900004
            native['apis'].append(dict(
                correlation_id=cid, domain='runtime',
                name='cudaLaunchKernel', start=native['window']['start_ns'] + 1,
                end=native['window']['start_ns'] + 2, thread=1101,
                return_value=0))
            native['activities'].append(dict(
                correlation_id=cid, kind='kernel', count=1, graph_id=0,
                context_id=native['activities'][0]['context_id'],
                stream_id=native['activities'][0]['stream_id'], copy_kind=0))
            native['activity_records'] += 1
        out = self.compose(fn)
        self.assertEqual(out['verdict'], 'FAIL', out)
        self.assertTrue(any('unpartitioned_activity' in reason
                            for reason in out['failures']), out)

    def test_pending_query_never_discharges_offer_completion(self):
        def fn(_native, declared):
            for row in declared.values():
                if row.get('submitted'):
                    row['completed'] = 0
        out = self.compose(fn)
        self.assertEqual(out['verdict'], 'INCONCLUSIVE', out)
        self.assertIn('prefill_grade_inconclusive', out['missing'])

    def test_raw_loss_is_still_binding(self):
        def fn(native, _declared):
            native['overflow'] = 1
        out = self.compose(fn)
        self.assertNotEqual(out['verdict'], 'PASS', out)
        self.assertIn('whole_raw_loss', out['missing'])


class ProbeGradeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = real_private_receipt(snapshot=True)
        cls.receipt = cls.fixture['prefill_receipt']
        cls.composed = cls.fixture['composed']

    def record(self):
        record = {
            'source_pins_verified': True, 'graph_bytes_stable': True,
            'cleanup_ok': True, 'contexts_disjoint': True,
            'raw_unchanged': True, 'graph_output_exact': True,
            'private_scope_first': True, 'private_scope_repeat': True,
            'uid_bytes_exact': True, 'lock_zero': True,
            'owners_restored': True, 'decoder_contents_unchanged': True,
            'sharing_preserved': True, 'pending_query_observed': True,
            'pending_query_not_joined': True, 'queried_completion': True,
            'prefill_receipt': copy.deepcopy(self.receipt),
            'composed': copy.deepcopy(self.composed),
            'pending_evidence': {'pending': [1], 'success': [2], 'joined': [],
                                 'with_activity': []},
            'shadow_pins': [{'source_sha256': 'a' * 64}],
        }
        expected = job.s.sha(Path(lane_private.__file__))
        record['source_pins'] = {'lane_private.py': expected}
        record['private_method_pins'] = [dict(method=name, source_sha256=expected)
            for name in ('LanePrivateChunk._populate_uid_once', 'LanePrivateBindings._initialize_lock_once')]
        record.update(copy.deepcopy(self.fixture))
        job.collect_pending(record, record['native_report'])
        return record

    def test_real_producer_partition_passes_without_inventing_raw_graph_evidence(self):
        record = self.record()
        self.assertEqual(record['composed']['verdict'], 'PASS')
        result = job.case_grade(record)
        self.assertEqual(result['verdict'], 'INCONCLUSIVE')
        self.assertIn('raw_graph_receipt_missing', result['missing'])

    def test_missing_or_wrong_private_counts_never_pass(self):
        record = self.record()
        record['prefill_receipt'] = copy.deepcopy(record['prefill_receipt'])
        record['prefill_receipt']['families']['qsa']['uid_init'] = {
            'kernel': 1, 'memcpy': 1}
        self.assertEqual(job.case_grade(record)['verdict'], 'FAIL')
        record = self.record()
        record['prefill_receipt'] = copy.deepcopy(record['prefill_receipt'])
        record['prefill_receipt']['families']['qsa']['lock_init'] = {}
        self.assertEqual(job.case_grade(record)['verdict'], 'FAIL')

    def test_pending_and_completion_are_both_required(self):
        record = self.record()
        record['pending_evidence'] = {'pending': [], 'success': [2],
                                      'joined': [], 'with_activity': []}
        self.assertEqual(job.case_grade(record)['verdict'], 'INCONCLUSIVE')
        record = self.record()
        record['pending_evidence'] = {'pending': [1], 'success': [],
                                      'joined': [], 'with_activity': []}
        self.assertEqual(job.case_grade(record)['verdict'], 'INCONCLUSIVE')
        record = self.record()
        record['queried_completion'] = False
        self.assertEqual(job.case_grade(record)['verdict'], 'FAIL')

    def test_joined_pending_activity_is_a_failure(self):
        record = self.record()
        native = record['native_report']
        cid = record['pending_evidence']['pending'][0]
        oid = record['prefill_receipt']['offers'][0]['id']
        native['external'].append(dict(kind='custom0', correlation_id=cid, external_id=oid))
        job.collect_pending(record, native)
        self.assertEqual(job.case_grade(record)['verdict'], 'FAIL')

    def test_owner_completion_and_child_inc_are_enforced(self):
        record = self.record()
        record['composed'] = copy.deepcopy(record['composed'])
        record['composed']['prefill']['completion'] = {
            'unproven_owners': ['lane']}
        self.assertEqual(job.case_grade(record)['verdict'], 'INCONCLUSIVE')
        raw = {'cases': [], 'regressions': {
            'job2_child_scope': {'verdict': 'PASS'}},
            'cannot_execute': ['x'], 'build_verified': True}
        self.assertNotEqual(job.grade(raw)['verdict'], 'PASS')
        self.assertIn('job2_child_scope_not_declared_inc',
                      job.grade(raw)['missing'])

    def test_no_synthetic_green_dictionary(self):
        record = self.record()
        del record['prefill_receipt']
        self.assertEqual(job.case_grade(record)['verdict'], 'INCONCLUSIVE')


class SourceContractTests(unittest.TestCase):
    def test_probe_uses_real_objects_and_no_fake_provider(self):
        source = Path(job.__file__).read_text()
        for needle in ('LanePrivateBindings.discover', 'prepare_workspace',
                       'attach_ledger(prefill)',
                       'chunk_scope(chunk).scope(layer_idx)',
                       "set_per_process_memory_fraction(.125, 0)",
                       "s.monitor(args.out, 'flashnext-job13d')"):
            self.assertIn(needle, source)
        self.assertIn("torch.cuda, '_sleep'", source)
        self.assertNotIn('_lock_initialized.clear()', source)
        self.assertNotIn('synthetic-provider', source)

    def test_compose_import_is_the_pinned_source(self):
        self.assertIs(np_.compose_overlap,
                      native_prefill_support_compose())

    def test_probe_has_no_hard_coded_peer_revision(self):
        source = Path(job.__file__).read_text()
        # Pins come from the root bundle closure, never a literal digest.
        self.assertIn('e3_build_closure.json', source)
        self.assertNotIn("'builder_sources':", source)


def native_prefill_support_compose():
    import native_placement
    return native_placement.compose_overlap


class RootCollectorTests(unittest.TestCase):
    def test_pending_scanner_refuses_graph_receipt_and_consumes_actual_native_report(self):
        bridge = support.FakeBridge()
        pending = bridge.host_control('cudaEventQuery', return_value=600)
        bridge.host_control('cudaEventQuery', return_value=0)
        record = {}
        job.collect_pending(record, bridge.report()['native'])
        self.assertEqual(record['pending_evidence']['pending'], [pending])
        self.assertTrue(record['pending_query_observed'])
        with self.assertRaisesRegex(RuntimeError, 'pending_native_report_schema'):
            job.collect_pending({}, {'schema': 's5ro-replay-observer/1'})

    def test_actual_pinned_workspace_globals_are_initialized_before_private_discovery(self):
        import ast
        torch = cpu_torch()
        torch.device = lambda value: value
        backing = lp.FakeWorkspaceModule(lanes=2, torch=torch)
        source = (ROOT/'reference/vllm-src/v1/worker/workspace.py').read_text()
        names = {'current_workspace_manager', 'init_workspace_manager', 'reset_workspace_manager'}
        nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names]
        namespace = dict(torch=torch, _manager=None, WorkspaceManager=type(backing.manager),
                         logger=NS(warning=lambda *args: None))
        text = 'from __future__ import annotations\n' + '\n'.join(ast.get_source_segment(source, n) for n in nodes)
        exec(compile(text, '<pinned-workspace-globals>', 'exec'), namespace)
        module = NS(**{name: namespace[name] for name in names})
        with self.assertRaisesRegex(AssertionError, 'not initialized'):
            module.current_workspace_manager()
        manager = job.initialize_workspace(torch, module)
        self.assertIs(manager, module.current_workspace_manager())
        self.assertEqual(manager._num_lanes, 2)
        self.assertEqual(manager._num_ubatches, 1)

class RootDrainTests(unittest.TestCase):
    def test_both_actual_streams_drained_and_no_constant_claim(self):
        calls=[]
        streams=[NS(cuda_stream=n,synchronize=lambda n=n:calls.append(n)) for n in (11,12)]
        self.assertEqual(job.drain_split_streams(NS(decode_stream=streams[0],prefill_stream=streams[1])),2)
        self.assertEqual(calls,[11,12])
        with self.assertRaisesRegex(RuntimeError,'consumer_streams_unproved'):
            job.drain_split_streams(NS(decode_stream=streams[0],prefill_stream=streams[0]))

class RootDeviceAndSourceTests(unittest.TestCase):
    def test_constructor_uses_the_tensor_device_object_not_the_input_string(self):
        class Device:
            def __init__(self, value):self.value=str(value)
            def __str__(self):return self.value
            def __eq__(self, other):return isinstance(other, Device) and str(self)==str(other)
        torch=cpu_torch();torch.device=Device
        full,empty=torch.full,torch.empty
        def normalized(fn):
            def call(*args,**kwargs):
                kwargs['device']=Device(kwargs.get('device','cuda:0'))
                return fn(*args,**kwargs)
            return call
        torch.full=normalized(full);torch.empty=normalized(empty)
        workspace=lp.FakeWorkspaceModule(lanes=2,torch=torch)
        bundle=job.build_bindings(torch,workspace_module=workspace)
        self.assertIsInstance(bundle['runner'].device, Device)
        self.assertEqual(bundle['runner'].device,bundle['uid'].device)
    def test_actual_private_method_pin_and_drift(self):
        torch=cpu_torch();workspace=lp.FakeWorkspaceModule(lanes=2,torch=torch)
        bundle=job.build_bindings(torch,workspace_module=workspace)
        pins=job.private_method_pins(bundle['bindings'],lane_private,job.s.sha(Path(lane_private.__file__)))
        self.assertEqual(len(pins),2)
        with self.assertRaisesRegex(RuntimeError,'private_method_source_pin'):
            job.private_method_pins(bundle['bindings'],lane_private,'0'*64)

if __name__ == '__main__':
    unittest.main()
