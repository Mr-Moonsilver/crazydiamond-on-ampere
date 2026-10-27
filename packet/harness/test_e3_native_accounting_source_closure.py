"""CPU acceptance for native-accounting source closure (E3).

Two source gaps, exercised with the real modules over a controlled native sink:

1. ``cuEventQuery``/``cudaEventQuery`` may report the documented
   ``CUDA_ERROR_NOT_READY`` (600) while captured work is pending. That is a
   zero-GPU-work host observation, never completion, and no other API may
   report 600. The shared classifier is consumed by every grading path.
2. ``LanePrivateChunk.scope`` initializes the private UID buffer and each
   retained Marlin lock *inside* the QSA layer container. Those exact launches
   are declared leaf offers; the container stays a zero-activity container.

No CUDA, engine, box or native evidence is used here; the native sink is a
controlled stand-in and passing is a contract check only.
"""
import contextlib
import copy
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'harness'))
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))

import activity_accounting as accounting  # noqa: E402
import lane_private  # noqa: E402
import native_prefill_offers as npo  # noqa: E402
import native_prefill_support as support  # noqa: E402
import test_e3_lane_private as lp  # noqa: E402
import test_e3_native_prefill_offers as base  # noqa: E402


# --- 1. the shared event-query classifier ----------------------------------

class ClassifierTests(unittest.TestCase):
    def test_exact_query_allows_success_or_not_ready_only(self):
        for name in ('cuEventQuery', 'cudaEventQuery', 'cuEventQuery_v2',
                     'cudaEventQuery_ptsz', 'cuEventQuery_v7000_ptds'):
            self.assertEqual(accounting.host_return_class(name, 0), 'success')
            self.assertEqual(accounting.host_return_class(name, 600), 'pending')
            self.assertTrue(accounting.host_return_ok(name, 600))
        # 600 is specific to the event-query family.
        for name in ('cuEventRecord', 'cudaEventRecord', 'cudaEventQueryEx',
                     'cuStreamQuery', 'cudaStreamSynchronize',
                     'cuEventSynchronize', 'cudaLaunchKernel'):
            self.assertEqual(accounting.host_return_class(name, 600), 'refused')
            self.assertFalse(accounting.host_return_ok(name, 600))
        # Other nonzero query statuses still refuse.
        for value in (1, 400, 700, 802, -1, None, True, 600.0):
            self.assertEqual(
                accounting.host_return_class('cudaEventQuery', value),
                'refused', value)
        self.assertEqual(accounting.CUDA_ERROR_NOT_READY, 600)

    def test_query_is_classified_as_host_control(self):
        for name in ('cuEventQuery', 'cudaEventQuery_v2'):
            self.assertEqual(accounting.api_kind(name), 'host')


# --- 2. every grading path that sees a host return -------------------------

def accounting_fixture():
    offers = [dict(id=1, epoch=4, generation=2, thread=1010,
                   system_thread=10, parent=0, family='pre', closed=True,
                   graph=False, expected={'kernel': 1}, context_id=7,
                   stream_id=9)]
    report = dict(
        schema=accounting.SCHEMA, thread_id_type='default-pthread-low32',
        complete=True, overflow=0, errors=0, open_scopes=0,
        boundary_records=0, activity_records=1,
        scopes=[dict(id=1, parent=0, thread=1010, system_thread=10,
                     action=action) for action in ('push', 'pop')],
        apis=[dict(correlation_id=10, domain='runtime',
                   name='cudaLaunchKernel_v7000', start=100, end=150,
                   thread=1010, return_value=0),
              dict(correlation_id=11, domain='driver', name='cuLaunchKernel',
                   start=110, end=140, thread=1010, return_value=0)],
        external=[dict(kind='custom0', correlation_id=cid, external_id=1)
                  for cid in (10, 11)],
        activities=[dict(correlation_id=11, kind='kernel', count=1,
                         graph_id=0, context_id=7, stream_id=9, copy_kind=0)])
    return report, offers


def add_accounting_api(report, name='cudaEventQuery', value=600, cid=12,
                       activity=None):
    report['apis'].append(dict(correlation_id=cid, domain='runtime',
                               name=name, start=200, end=201, thread=1010,
                               return_value=value))
    if activity is not None:
        report['external'].append(dict(kind='custom0',
                                       correlation_id=cid, external_id=1))
        report['activities'].append(dict(correlation_id=cid, kind=activity,
                                         count=1, graph_id=0, context_id=7,
                                         stream_id=9, copy_kind=0))
        report['activity_records'] += 1


class ActivityAccountingHostQueryTests(unittest.TestCase):
    def grade(self, report, offers):
        return accounting.grade(report, offers, epoch=4, generation=2)

    def test_pending_event_query_is_a_zero_work_host_observation(self):
        report, offers = accounting_fixture()
        self.assertEqual(self.grade(report, offers)['verdict'], 'PASS')
        add_accounting_api(report)
        result = self.grade(report, offers)
        self.assertEqual(result['verdict'], 'PASS', result)

    def test_not_ready_on_any_other_api_refused(self):
        for name in ('cudaEventRecord', 'cudaStreamSynchronize',
                     'cuEventSynchronize'):
            report, offers = accounting_fixture()
            add_accounting_api(report, name=name)
            result = self.grade(report, offers)
            self.assertEqual(result['verdict'], 'FAIL', result)
            self.assertIn('host_api_failed', result['failures'])

    def test_other_nonzero_event_query_refused(self):
        report, offers = accounting_fixture()
        add_accounting_api(report, value=1)
        result = self.grade(report, offers)
        self.assertIn('host_api_failed', result['failures'])

    def test_pending_query_with_activity_is_refused(self):
        report, offers = accounting_fixture()
        add_accounting_api(report, activity='kernel')
        result = self.grade(report, offers)
        self.assertEqual(result['verdict'], 'FAIL', result)
        self.assertIn('failed_api_has_activity', result['failures'])
        self.assertIn('api_activity_kind', result['failures'])

    def test_unknown_api_and_raw_loss_are_never_discharged(self):
        report, offers = accounting_fixture()
        add_accounting_api(report, name='cuMysteryQuery', value=600)
        self.assertIn('unsupported_api:cuMysteryQuery',
                      self.grade(report, offers)['missing'])
        report, offers = accounting_fixture()
        report['errors'] = 1
        self.assertNotEqual(self.grade(report, offers)['verdict'], 'PASS')


def prefill_query(native, receipt, name='cudaEventQuery', value=600,
                  join=None, activity=None):
    """Append one flat API row to the real producer dictionary."""
    cid = 900000
    thread = native['scopes'][0]['thread']
    offer = None
    if join is not None:
        offer = receipt['declared'][str(join)]
        thread = offer['thread']
    native['apis'].append(dict(correlation_id=cid, domain='runtime',
                               name=name, start=500000, end=500001,
                               thread=thread, return_value=value))
    if join is not None:
        native['external'].append(dict(kind='custom0', correlation_id=cid,
                                       external_id=join))
        if activity is not None:
            native['activities'].append(dict(
                correlation_id=cid, kind=activity, count=1, graph_id=0,
                context_id=offer['context_id'],
                stream_id=offer['stream_id'], copy_kind=0, name=name))
            native['activity_records'] += 1
            native['partition_totals'][activity] += 1
    return cid


class PrefillHostQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.bridge, cls.ledger, cls.parts, cls.shadows, cls.window,
         cls.receipt) = base.positive()
        cls.native = copy.deepcopy(cls.ledger.snapshot()['native'])

    def grade(self, mutate):
        native = copy.deepcopy(self.native)
        receipt = copy.deepcopy(self.receipt)
        mutate(native, receipt)
        return base.grade(native, receipt, receipt['window'])

    def test_pending_query_without_activity_is_accepted(self):
        def fn(native, receipt):
            prefill_query(native, receipt)
        out = self.grade(fn)
        self.assertEqual(out['verdict'], 'PASS', out)

    def test_pending_query_joined_inside_an_offer_is_accepted(self):
        def fn(native, receipt):
            oid = receipt['offers'][0]['id']
            prefill_query(native, receipt, join=oid)
        out = self.grade(fn)
        self.assertEqual(out['verdict'], 'PASS', out)

    def test_pending_on_other_api_and_other_nonzero_refused(self):
        for name, value in (('cudaEventRecord', 600),
                            ('cudaStreamSynchronize', 600),
                            ('cudaEventQuery', 1),
                            ('cuEventQuery', 700)):
            def fn(native, receipt, name=name, value=value):
                prefill_query(native, receipt, name=name, value=value)
            out = self.grade(fn)
            self.assertEqual(out['verdict'], 'FAIL', (name, value, out))
            self.assertIn('host_api_failed', out['failures'])

    def test_pending_query_with_gpu_activity_is_refused(self):
        def fn(native, receipt):
            oid = receipt['offers'][0]['id']
            prefill_query(native, receipt, join=oid, activity='kernel')
        out = self.grade(fn)
        self.assertEqual(out['verdict'], 'FAIL', out)
        self.assertIn('failed_api_has_activity', out['failures'])

    def test_pending_is_never_completion(self):
        # The event provider reports pending: no OFFER can discharge it, even
        # though the observed cudaEventQuery row reports 600.
        def fn(native, receipt):
            prefill_query(native, receipt)
            for row in receipt['declared'].values():
                if row.get('submitted'):
                    row['completed'] = 0
        out = self.grade(fn)
        self.assertEqual(out['verdict'], 'INCONCLUSIVE', out)
        self.assertIn('completion_unproven', out['missing'])

    def test_wrong_stream_context_and_raw_loss_still_bind(self):
        def wrong_stream(native, _receipt):
            native['activities'][0]['stream_id'] += 1
        self.assertIn('wrong_stream', self.grade(wrong_stream)['failures'])

        def wrong_context(native, _receipt):
            native['activities'][0]['context_id'] += 1
        self.assertIn('wrong_context', self.grade(wrong_context)['failures'])

        def raw_loss(native, _receipt):
            native['overflow'] = 1
        self.assertIn('correlation_overflow', self.grade(raw_loss)['missing'])

    def test_unknown_and_unoffered_api_not_discharged(self):
        def unknown(native, receipt):
            oid = receipt['offers'][0]['id']
            prefill_query(native, receipt, name='cuMystery', value=600,
                          join=oid, activity='kernel')
        out = self.grade(unknown)
        self.assertEqual(out['verdict'], 'FAIL', out)
        self.assertIn('api_activity_kind', out['failures'])
        self.assertIn('unsupported_api:cuMystery', out['missing'])


# --- 3. lane-private initialization leaves ---------------------------------

_PRODUCER = {'bridge': None}


class ObservedTensor(lp.Tensor):
    """The test tensor emits the activity the real initialization launches."""

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


def _emit(kind, **kwargs):
    bridge = _PRODUCER['bridge']
    if bridge is not None and bridge._stack:
        bridge.activity(kind, stream=support.MAIN, **kwargs)


class LockOwner:
    def __init__(self):
        self.workspace = ObservedTensor((4,), 'int32')


class LockLayer(lp.FakeLayer):
    def __init__(self, attn, tokens):
        super().__init__(attn, tokens)
        self.lock = LockOwner()

    def named_modules(self):
        return tuple(super().named_modules()) + (('lock', self.lock),)


class RefusingBridge(support.FakeBridge):
    """A bridge that refuses one exact wire name at enter or at pop."""

    def __init__(self, *, fail_wire=None, mode='enter'):
        super().__init__()
        self.fail_wire = fail_wire
        self.mode = mode

    @contextlib.contextmanager
    def offer(self, family, expected, *, context_id, stream_id, graph=False,
              copy_kinds=()):
        if family == self.fail_wire and self.mode == 'enter':
            raise RuntimeError('refused push')
        with super().offer(family, expected, context_id=context_id,
                           stream_id=stream_id, graph=graph,
                           copy_kinds=copy_kinds) as row:
            yield row
        if family == self.fail_wire and self.mode == 'exit':
            raise RuntimeError('refused pop')


class LanePrivateLeafTests(unittest.TestCase):
    chunk_tokens = lp.PrivateBindingsTests.chunk_tokens
    width = lp.PrivateBindingsTests.width
    req_index = lp.PrivateBindingsTests.req_index

    def setUp(self):
        _PRODUCER['bridge'] = None
        self.torch = lp.fake_torch()
        self.torch.empty = (lambda shape, dtype, device:
                            ObservedTensor(shape, dtype, device=device))
        self.attn = lp.FakeQsaAttention(self.width)
        self.original = self.attn.topk_indices_buffer
        self.original.fill_rows(900)
        self.model = lp.FakeModel(LockLayer(self.attn, self.chunk_tokens))
        self.decode_uid = lp.Tensor((4,), 'int32')
        self.decode_uid.fill_(-2)
        self.private_uid = ObservedTensor((4,), 'int32')
        self.private_uid.fill_(-2)
        self.scratch = lp.Tensor(
            (self.chunk_tokens * self.width * 4,), 'uint8')
        self.slots = lp.Tensor((4,), 'int32')
        self.slots.fill_(-2)
        self.slots.write(self.req_index, 7)
        self.fc = lp.FakeForwardContextModule(self.decode_uid)
        self.ws = lp.FakeWorkspaceModule(lanes=2, torch=self.torch)
        self.runtime = lp.LaneRuntime(
            lp.WorkerState(0, engine_epoch=3),
            forward_context=lp._ForwardContextAdapter(self.fc),
            workspace=lp._WorkspaceLaneAdapter(self.ws, 1),
            prefill_forward_context=object(), prefill_workspace=1,
            uid_namespace=self.fc, prefill_uid_buffer=self.private_uid)
        self.runtime._native_owned_buffers = (
            types.SimpleNamespace(name=lane_private.UID_BUFFER_NAME,
                                  tensor=self.private_uid),
            types.SimpleNamespace(name=lane_private.TOPK_SCRATCH_NAME,
                                  tensor=self.scratch))
        self.runner = types.SimpleNamespace(
            model=self.model, device='cuda:0', _fc_module=self.fc,
            max_num_tokens=self.chunk_tokens,
            _qsa_uid_slots=self.slots, _qsa_uid_next=7)
        tx, snapshot = lp.lane_metadata(self.runtime.state,
                                        end=self.chunk_tokens)
        self.runtime.open_prefill(tx, snapshot)
        self.runner._test_transaction = tx
        self.runner.req_states = types.SimpleNamespace(
            req_id_to_index={tx.request_uid: self.req_index})
        self.runtime.release = lambda transaction, proofs=(): None
        self.ws.manager._ensure_workspace_size(128)
        self.bindings = lane_private.LanePrivateBindings.discover(
            self.runner, self.runtime, chunk_tokens=self.chunk_tokens,
            torch_module=self.torch)
        self.bindings.prepare_workspace()
        self.ws.manager.lock()
        self.lock_original = self.bindings.marlin_locks[0][3]
        self.lock_private = self.bindings.marlin_locks[0][4]
        self.lock_module = self.bindings.marlin_locks[0][2]

    def make_chunk(self):
        return lp.FakeChunk(
            self.runner, self.req_index, self.chunk_tokens,
            lp.metadata_for(self.attn, self.chunk_tokens))

    def make_ledger(self, bridge=None):
        bridge = bridge if bridge is not None else support.FakeBridge()
        torch_module = support.make_torch()
        ledger = support.make_ledger(bridge, torch_module=torch_module)
        support.declare(ledger)
        self.bindings.attach_ledger(ledger)
        return bridge, ledger

    def run_scope(self, ledger, chunk=None):
        chunk = chunk if chunk is not None else self.make_chunk()
        with ledger.segment('layer',
                            transaction=self.runner._test_transaction,
                            op_seq=0, branch='driver'):
            with self.bindings.chunk_scope(chunk).scope(0):
                pass

    def seal(self, bridge, ledger):
        ledger.owner_finish('lane')
        bridge.close_window()
        ledger.finish()
        return ledger.seal()

    def test_uid_and_lock_initialization_are_nested_leaves(self):
        bridge, ledger = self.make_ledger()
        _PRODUCER['bridge'] = bridge
        try:
            self.run_scope(ledger)
        finally:
            _PRODUCER['bridge'] = None
        receipt = self.seal(bridge, ledger)
        self.assertEqual(receipt['verdict'], 'PASS', receipt)
        counts = receipt['families']['qsa']
        self.assertEqual(counts['uid_init'], {'kernel': 1, 'memcpy': 1})
        self.assertEqual(counts['lock_init'], {'memset': 1})
        self.assertGreater(receipt['joined_records'], 0)
        # Both leaves are children of the zero-activity QSA container.
        containers = [row for row in receipt['offers']
                      if row['expectation'] == 'container']
        self.assertEqual(len(containers), 1)
        leaves = [row for row in receipt['offers']
                  if row['suboperation'] in ('uid_init', 'lock_init')]
        self.assertEqual({row['suboperation'] for row in leaves},
                         {'uid_init', 'lock_init'})
        self.assertTrue(all(row['parent'] == containers[0]['id']
                            for row in leaves))
        self.assertEqual(
            bridge.report()['native']['partition_totals'],
            {'kernel': 1, 'memcpy': 1, 'memset': 1})
        # Owner restoration is exact.
        self.assertIs(self.fc.QSA_REQ_UID, self.decode_uid)
        self.assertIs(self.attn.topk_indices_buffer, self.original)
        self.assertIs(self.lock_module.workspace, self.lock_original)
        self.assertEqual(self.private_uid.read(0), 7)
        self.assertTrue(all(self.private_uid.read(i) == -2 for i in range(1, 4)))

    def test_first_use_zeroes_but_repeated_use_does_not_reoffer_lock(self):
        self.lock_private.write(0, 12345)
        bridge, ledger = self.make_ledger()
        _PRODUCER['bridge'] = bridge
        try:
            self.run_scope(ledger)
            self.lock_private.write(0, 999)
            self.run_scope(ledger, chunk=self.make_chunk())
        finally:
            _PRODUCER['bridge'] = None
        receipt = self.seal(bridge, ledger)
        self.assertEqual(receipt['verdict'], 'PASS', receipt)
        self.assertEqual(receipt['families']['qsa']['uid_init'],
                         {'kernel': 2, 'memcpy': 2})
        self.assertEqual(receipt['families']['qsa']['lock_init'],
                         {'memset': 1})
        # The second scope did not re-zero the retained lock.
        self.assertEqual(self.lock_private.read(0), 999)

    def test_invalid_scope_latches_incomplete_and_runs_the_body_once(self):
        self.lock_private.write(0, 12345)
        bridge, ledger = self.make_ledger()
        chunk = self.make_chunk()
        # No lane layer container is open: the private leaves have no exact
        # scope, so each offer is refused and the window latches incomplete.
        with self.bindings.chunk_scope(chunk).scope(0):
            pass
        receipt = self.seal(bridge, ledger)
        self.assertTrue(receipt['window']['invalidated'], receipt)
        self.assertEqual(receipt['verdict'], 'INCONCLUSIVE', receipt)
        self.assertIn('private_leaf_refused:uid_init',
                      receipt['window']['invalid'])
        self.assertIn('private_leaf_refused:lock_init',
                      receipt['window']['invalid'])
        self.assertEqual(self.private_uid.read(0), 7)
        self.assertEqual(self.lock_private.read(0), 0)
        self.assertIs(self.fc.QSA_REQ_UID, self.decode_uid)
        self.assertIs(self.attn.topk_indices_buffer, self.original)
        self.assertIs(self.lock_module.workspace, self.lock_original)

    def test_attach_refuses_a_provider_without_private_leaf(self):
        with self.assertRaises(lane_private.LanePrivateError):
            self.bindings.attach_ledger(object())
        self.bindings.attach_ledger(None)
        with self.bindings.chunk_scope(self.make_chunk()).scope(0):
            pass
        self.assertEqual(self.private_uid.read(0), 7)
        self.assertEqual(self.lock_private.read(0), 0)

    def test_provider_context_entry_and_exit_failure_never_skip_or_repeat_zero(self):
        for failed in ('enter', 'exit'):
            invalid = []
            calls = []
            class Scope:
                def __enter__(self):
                    if failed == 'enter':
                        raise RuntimeError('observer entry')
                def __exit__(self, *args):
                    if failed == 'exit':
                        raise RuntimeError('observer exit')
            provider = types.SimpleNamespace(
                private_leaf=lambda _: Scope(), _invalidate=invalid.append)
            lock = types.SimpleNamespace(zero_=lambda: calls.append('zero'))
            self.bindings.attach_ledger(provider)
            self.bindings._initialize_lock_once(lock)
            self.assertEqual(calls, ['zero'])
            self.assertIn(id(lock), self.bindings._lock_initialized)
            self.assertEqual(len(invalid), 1)

    def test_provider_cannot_suppress_original_zero_failure_or_set_initialized(self):
        calls = []
        class Scope:
            def __enter__(self):pass
            def __exit__(self, *args):return True
        provider = types.SimpleNamespace(private_leaf=lambda _: Scope(),
                                         _invalidate=lambda _: None)
        def zero():
            calls.append('zero')
            raise RuntimeError('original zero failed')
        lock = types.SimpleNamespace(zero_=zero)
        self.bindings.attach_ledger(provider)
        with self.assertRaisesRegex(RuntimeError, 'original zero failed'):
            self.bindings._initialize_lock_once(lock)
        self.assertEqual(calls, ['zero'])
        self.assertNotIn(id(lock), self.bindings._lock_initialized)

    def test_missing_provider_runs_the_original_once_without_latching(self):
        self.lock_private.write(0, 12345)
        chunk = self.make_chunk()
        with self.bindings.chunk_scope(chunk).scope(0):
            pass
        self.assertEqual(self.private_uid.read(0), 7)
        self.assertTrue(all(self.private_uid.read(i) == -2 for i in range(1, 4)))
        self.assertEqual(self.lock_private.read(0), 0)
        self.assertIs(self.fc.QSA_REQ_UID, self.decode_uid)
        self.assertIs(self.attn.topk_indices_buffer, self.original)

    def test_refused_push_still_runs_the_body_once_and_latches(self):
        self.lock_private.write(0, 12345)
        bridge = RefusingBridge(fail_wire='qsa/uid_init', mode='enter')
        _PRODUCER['bridge'] = bridge
        ledger = None
        try:
            bridge, ledger = self.make_ledger(bridge)
            self.run_scope(ledger)
        finally:
            _PRODUCER['bridge'] = None
        receipt = self.seal(bridge, ledger)
        # A refused push opens no leaf: the leaf activity necessarily lands in
        # the still-open container, so grading is FAIL (never PASS) on top of
        # the latched incompleteness. The body ran exactly once regardless.
        self.assertNotEqual(receipt['verdict'], 'PASS', receipt)
        self.assertTrue(receipt['window']['invalidated'], receipt)
        self.assertIn('prefill_instrumentation_invalid', receipt['missing'])
        self.assertIn('prefill_invalid:private_leaf_entry:uid_init',
                      receipt['missing'])
        self.assertEqual(self.private_uid.read(0), 7)
        self.assertEqual(self.lock_private.read(0), 0)

    def test_failed_pop_and_missing_mark_still_run_the_body_once(self):
        for mode in ('exit', 'mark'):
            # Each iteration is a fresh first use of the retained lock.
            self.bindings._lock_initialized.clear()
            self.lock_private.write(0, 12345)
            if mode == 'mark':
                bridge = support.FakeBridge()
                bridge.fail_wire = None
                torch_module = support.make_torch()

                def broken(*_a, **_k):
                    raise RuntimeError('event provider unavailable')
                torch_module.cuda.Event = broken
                ledger = support.make_ledger(bridge, torch_module=torch_module)
                support.declare(ledger)
                self.bindings.attach_ledger(ledger)
            else:
                bridge = RefusingBridge(fail_wire='qsa/uid_init', mode='exit')
                bridge, ledger = self.make_ledger(bridge)
            _PRODUCER['bridge'] = bridge
            try:
                self.run_scope(ledger)
            finally:
                _PRODUCER['bridge'] = None
            receipt = self.seal(bridge, ledger)
            # The failure latches the window invalid; once invalidated the
            # following leaf offer is refused too, so its activity lands in
            # the container. Grading is therefore FAIL (never PASS) with the
            # invalidity surfaced. The bodies still ran exactly once.
            self.assertNotEqual(receipt['verdict'], 'PASS', (mode, receipt))
            self.assertTrue(receipt['window']['invalidated'], (mode, receipt))
            self.assertIn('prefill_instrumentation_invalid',
                          receipt['missing'], mode)
            self.assertEqual(self.private_uid.read(0), 7, mode)
            self.assertEqual(self.lock_private.read(0), 0, mode)
            self.assertIs(self.fc.QSA_REQ_UID, self.decode_uid, mode)


if __name__ == '__main__':
    unittest.main()
