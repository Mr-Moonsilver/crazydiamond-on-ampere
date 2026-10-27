"""CPU contract tests for the source-declared native prefill offer ledger.

The real ``LaneDispatcher``, ``LaneForward`` and installed source seams run
over a controlled native sink. Passing here is a contract check, never native
CUDA evidence.
"""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'harness'))
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))

import native_prefill_offers as npo  # noqa: E402
import native_prefill_support as support  # noqa: E402


def positive():
    bridge = support.FakeBridge()
    torch_module = support.make_torch()
    ledger = support.make_ledger(bridge, torch_module=torch_module)
    window = support.declare(ledger)
    parts = support.build_lane(bridge, torch_module=torch_module)
    shadows = support.install_lane_seams(parts, ledger)
    events = support.start_role_owners(ledger, parts['torch'])
    support.submit_all(parts['dispatcher'])
    support.finish_role_owners(ledger, events=events[:1])
    ledger.owner_finish('lane', events=events[:1])
    for shadow in reversed(shadows):
        shadow.close()
    bridge.close_window()
    ledger.finish()
    receipt = ledger.seal()
    return bridge, ledger, parts, shadows, window, receipt


def report_and_window(ledger):
    snapshot = ledger.snapshot()
    return (copy.deepcopy(snapshot['native']),
            copy.deepcopy(ledger.window()))


def grade(native, receipt, window=None):
    window = copy.deepcopy(window) if window is not None else copy.deepcopy(
        receipt['window'])
    return npo.grade_prefill(native, copy.deepcopy(receipt['declared']), window)


class PositiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.bridge, cls.ledger, cls.parts, cls.shadows, cls.window,
         cls.receipt) = positive()
        cls.native = copy.deepcopy(cls.ledger.snapshot()['native'])

    def test_positive_receipt_is_pass_with_source_joins(self):
        r = self.receipt
        self.assertEqual(r['verdict'], 'PASS', r)
        self.assertEqual(r['failures'], [])
        self.assertEqual(r['missing'], [])
        self.assertEqual(r['counts'], {'kernel': 8, 'memcpy': 1, 'memset': 0})
        self.assertEqual(r['joined_records'], 9)
        self.assertEqual(len(r['declared']), 13)
        self.assertEqual(r['completion']['offers_submitted'], 13)
        self.assertEqual(r['completion']['offers_completed'], 13)
        self.assertTrue(npo.validate_receipt(r))

    def test_shared_experts_and_ple_are_their_own_aux_offers(self):
        fam = self.receipt['families']
        self.assertEqual(fam['shared_experts']['maybe_forward_async'],
                         {'kernel': 1})
        self.assertEqual(fam['ple']['start_prefetch'], {'kernel': 1})
        roles = self.receipt['roles']
        self.assertEqual(roles['shared_experts_aux']['context_id'],
                         support.MAIN_CTX)
        self.assertEqual(roles['ple_aux']['context_id'], support.MAIN_CTX)
        self.assertEqual(roles['copy']['non_green_control'], False)
        auxjoins = [j for j in self.receipt['joins']
                    if j['role'] in ('shared_experts_aux', 'ple_aux')]
        self.assertEqual(len(auxjoins), 2)
        self.assertTrue(all(j['context_id'] == support.MAIN_CTX
                            for j in auxjoins))

    def test_copy_role_carries_only_the_bucket_memcpy(self):
        joins = [j for j in self.receipt['joins'] if j['role'] == 'copy']
        self.assertEqual(len(joins), 1)
        self.assertEqual(joins[0]['kind'], 'memcpy')
        self.assertEqual(joins[0]['stream_id'], support.COPY_STREAM)
        qsa = self.receipt['families']['qsa']
        self.assertEqual(qsa['bucket_fill'], {'memcpy': 1})
        self.assertEqual(qsa['attention'], {'kernel': 1})

    def test_lane_container_and_direct_layers_are_distinct(self):
        seg = self.receipt['families']['prefill_segment']
        self.assertEqual(seg['prefix'], {'kernel': 1})
        self.assertEqual(seg['layer'], {'kernel': 1})
        self.assertEqual(seg['suffix'], {'kernel': 1})
        self.assertEqual(seg.get('prepare', {}), {})
        self.assertEqual(seg.get('state_finalize', {}), {})
        containers = [row for row in self.receipt['offers']
                      if row['expectation'] == 'container']
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0]['branch'], 'driver')
        self.assertEqual(containers[0]['segment'], 'layer')

    def test_shadows_restore_exactly(self):
        for shadow in self.shadows:
            self.assertTrue(shadow.closed)
        self.assertNotIn('_do_prefix', self.parts['dispatcher'].__dict__)
        self.assertNotIn('maybe_forward_async', self.parts['shared'].__dict__)
        self.assertNotIn('start_prefetch', self.parts['ple'].__dict__)
        self.assertNotIn('_pre', self.parts['driver'].__dict__)


class MutationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.bridge, cls.ledger, cls.parts, cls.shadows, cls.window,
         cls.receipt) = positive()
        cls.native = copy.deepcopy(cls.ledger.snapshot()['native'])
        cls.rec = copy.deepcopy(cls.receipt)

    def mutate(self, fn):
        native = copy.deepcopy(self.native)
        rec = copy.deepcopy(self.rec)
        fn(native, rec)
        return grade(native, rec, rec['window'])

    def test_lost_activity_row_against_census_is_inconclusive(self):
        def fn(native, _rec):
            row = native['activities'].pop()
            cid = row['correlation_id']
            native['apis'] = [api for api in native['apis']
                              if api['correlation_id'] != cid]
            native['external'] = [ext for ext in native['external']
                                  if ext['correlation_id'] != cid]
        out = self.mutate(fn)
        self.assertEqual(out['verdict'], 'INCONCLUSIVE')
        self.assertIn('activity_census', out['missing'])

    def test_duplicate_join_is_fail(self):
        def fn(native, _rec):
            native['activities'].append(dict(native['activities'][0]))
        out = self.mutate(fn)
        self.assertEqual(out['verdict'], 'FAIL')
        self.assertIn('duplicate_activity_join', out['failures'])

    def test_wrong_context_and_stream_are_fail(self):
        def context(native, _rec):
            native['activities'][0]['context_id'] += 1
        self.assertIn('wrong_context', self.mutate(context)['failures'])

        def stream(native, _rec):
            native['activities'][0]['stream_id'] += 1
        self.assertIn('wrong_stream', self.mutate(stream)['failures'])

    def test_foreign_external_and_unoffered_same_symbol_are_fail(self):
        def foreign(native, _rec):
            native['external'][0]['external_id'] = 9999
        out = self.mutate(foreign)
        self.assertEqual(out['verdict'], 'FAIL')
        self.assertIn('unknown_offer', out['failures'])

        def unoffered(native, _rec):
            cid = 99999
            native['apis'].append({
                'correlation_id': cid, 'domain': 'runtime',
                'name': 'cudaLaunchKernel', 'start': 1, 'end': 2,
                'thread': support.THREAD, 'return_value': 0})
            native['activities'].append({
                'correlation_id': cid, 'kind': 'kernel', 'count': 1,
                'graph_id': 0, 'context_id': support.MAIN_CTX,
                'stream_id': support.MAIN_STREAM, 'copy_kind': 0,
                'name': 'cudaLaunchKernel'})
        out = self.mutate(unoffered)
        self.assertEqual(out['verdict'], 'FAIL')
        self.assertIn('unoffered_activity', out['failures'])

    def test_wrong_thread_missing_pop_and_duplicate_external(self):
        def thread(native, _rec):
            native['scopes'][0]['thread'] += 1
        self.assertIn('native_scope_thread', self.mutate(thread)['failures'])

        def pop(native, _rec):
            native['scopes'] = [row for row in native['scopes']
                                if not (row['action'] == 'pop'
                                        and row['id'] == 1)]
        out = self.mutate(pop)
        self.assertIn('native_scope_coverage', out['failures'])

        def duplicate(native, _rec):
            native['external'].append(dict(native['external'][0]))
        out = self.mutate(duplicate)
        self.assertIn('duplicate_external_join', out['failures'])

    def test_raw_loss_is_never_discharged(self):
        for field in ('errors', 'overflow', 'open_scopes', 'boundary_records'):
            def fn(native, _rec, field=field):
                native[field] = 1
            out = self.mutate(fn)
            self.assertEqual(out['verdict'], 'INCONCLUSIVE', field)
            self.assertIn('correlation_' + field if field != 'boundary_records'
                          else 'correlation_boundary', out['missing'], field)

    def test_copy_role_kernel_is_fail(self):
        def fn(native, _rec):
            for row in native['activities']:
                if row['kind'] == 'memcpy' and row['context_id'] == \
                        support.MAIN_CTX and row['stream_id'] == \
                        support.COPY_STREAM:
                    row['kind'] = 'kernel'
                    row['name'] = 'cudaLaunchKernel'
                    for api in native['apis']:
                        if api['correlation_id'] == row['correlation_id']:
                            api['name'] = 'cudaLaunchKernel'
        out = self.mutate(fn)
        self.assertEqual(out['verdict'], 'FAIL')
        self.assertIn('copy_role_kernel', out['failures'])

    def test_foreign_window_and_missing_op_seq_are_refused(self):
        def window(_native, rec):
            for row in rec['declared'].values():
                row['window_id'] = 'other-window'
        out = self.mutate(window)
        self.assertIn('offer_window_id', out['failures'])

        def op_seq(_native, rec):
            first = sorted(rec['declared'])[0]
            rec['declared'][first]['op_seq'] = None
        out = self.mutate(op_seq)
        self.assertEqual(out['verdict'], 'INCONCLUSIVE')
        self.assertIn('offer_op_seq_missing', out['missing'])

    def test_wrong_stream_identity_in_declared_is_fail(self):
        def fn(_native, rec):
            next(iter(rec['declared'].values()))['stream_id'] += 1
        out = self.mutate(fn)
        self.assertIn('offer_stream_identity_mismatch', out['failures'])

    def test_missing_owner_role_blocks_pass(self):
        def fn(_native, rec):
            rec['window']['owners'] = [
                row for row in rec['window']['owners']
                if row['role'] != 'ple_aux']
        out = self.mutate(fn)
        self.assertEqual(out['verdict'], 'INCONCLUSIVE')
        self.assertIn('owner_role_unmarked:ple_aux', out['missing'])

    def test_pending_offer_with_activity_is_fail(self):
        def fn(_native, rec):
            joined = {row['offer'] for row in rec['joins']}
            pending = min(joined)
            rec['declared'][str(pending)]['body_exception'] = 'E3Pending'
        out = self.mutate(fn)
        self.assertIn('pending_offer_has_activity', out['failures'])

    def test_graph_activity_is_never_prefill_evidence(self):
        def fn(native, _rec):
            native['activities'][0]['graph_id'] = 7
        out = self.mutate(fn)
        self.assertEqual(out['verdict'], 'FAIL')
        self.assertIn('graph_activity_in_prefill', out['failures'])


class CompletionTests(unittest.TestCase):
    def test_missing_or_unqueried_completion_is_inconclusive(self):
        bridge = support.FakeBridge()
        torch_module = support.make_torch()
        ledger = support.make_ledger(bridge, torch_module=torch_module)
        support.declare(ledger)
        parts = support.build_lane(bridge, torch_module=torch_module)
        support.install_lane_seams(parts, ledger)
        events = support.start_role_owners(ledger, parts['torch'])
        support.submit_all(parts['dispatcher'])
        support.finish_role_owners(ledger, events=events[:1])
        ledger.owner_finish('lane', events=events[:1])
        torch_module.cuda.set_events_ready(False)
        bridge.close_window()
        ledger.finish()
        out = ledger.seal()
        self.assertEqual(out['verdict'], 'INCONCLUSIVE')
        self.assertIn('completion_unproven', out['missing'])
        self.assertEqual(out['completion']['offers_completed'], 0)

    def test_no_event_provider_never_passes(self):
        bridge = support.FakeBridge()
        ledger = support.make_ledger(bridge, torch_module=None)
        ledger.torch = None
        support.declare(ledger)
        parts = support.build_lane(bridge)
        support.install_lane_seams(parts, ledger)
        support.submit_all(parts['dispatcher'])
        ledger.owner_finish('lane')
        bridge.close_window()
        ledger.finish()
        out = ledger.seal()
        self.assertEqual(out['verdict'], 'INCONCLUSIVE')
        self.assertIn('completion_unproven', out['missing'])
        self.assertIn('completion_provider:provider_missing', out['missing'])


class LifecycleTests(unittest.TestCase):
    def test_stream_declaration_refuses_stock_auxiliary_and_uncopyable_roles(self):
        bridge = support.FakeBridge()
        ledger = support.make_ledger(bridge)
        stock = support.FakeStream('stock', 99, 99)
        with self.assertRaises(npo.NativePrefillError):
            ledger.declare_streams(prefill_stream=support.MAIN,
                                   copy_stream=support.COPY,
                                   aux_streams={'shared_experts_aux': stock})
        bridge = support.FakeBridge()
        ledger = support.make_ledger(bridge)
        support.declare(ledger)
        self.assertEqual(ledger.window()['green_context_id'],
                         support.MAIN_CTX)
        with self.assertRaises(npo.NativePrefillError):
            ledger.declare_streams(prefill_stream=support.MAIN)

    def test_enable_env_is_required(self):
        bridge = support.FakeBridge()
        with self.assertRaises(npo.NativePrefillError):
            npo.NativePrefillLedger(
                bridge, worker='w0', rank=0, epoch=1, generation=3,
                route_generation=3, storage_epoch=1, run_id='r',
                window_id='w', stream_identity=support.stream_identity,
                env={})

    def test_source_pin_missing_and_drift_refused(self):
        bridge = support.FakeBridge()
        ledger = support.make_ledger(bridge)
        support.declare(ledger)
        parts = support.build_lane(bridge)
        with self.assertRaises(npo.SourcePinError):
            npo.install_seams(parts['dispatcher'], ledger, 'lane_dispatcher',
                              pins={'not-the-file': '0' * 64})
        pins = npo.live_pins(parts['dispatcher'])
        shadow = npo.install_seams(parts['dispatcher'], ledger,
                                   'lane_dispatcher', pins=pins)
        with self.assertRaises(npo.NativePrefillError):
            npo.install_seams(parts['dispatcher'], ledger, 'lane_dispatcher',
                              pins=pins)
        parts['dispatcher']._do_prefix = lambda command: None
        with self.assertRaises(npo.ShadowDrift):
            shadow.close()
        self.assertTrue(ledger.window()['invalidated'])
        self.assertIn('shadow_changed_outside:_do_prefix',
                      ledger.window()['invalid'])

    def test_duplicate_submission_creates_no_second_offer(self):
        bridge = support.FakeBridge()
        ledger = support.make_ledger(bridge)
        support.declare(ledger)
        parts = support.build_lane(bridge)
        support.install_lane_seams(parts, ledger)
        support.start_role_owners(ledger, parts['torch'])
        support.submit_all(parts['dispatcher'])
        before = len(bridge.offers)
        from lane_dispatch import SegmentCommand
        ack = parts['dispatcher'].submit(SegmentCommand(
            support.TXN, 0, 'prefix', 1))
        self.assertEqual(len(bridge.offers), before)
        self.assertEqual(ack['op_seq'], 1)

    def test_cancelled_transaction_submits_nothing(self):
        bridge = support.FakeBridge()
        ledger = support.make_ledger(bridge)
        support.declare(ledger)
        parts = support.build_lane(bridge)
        support.install_lane_seams(parts, ledger)
        parts['dispatcher'].cancel('test')
        from lane_dispatch import LaneDispatchError, SegmentCommand
        before = len(bridge.offers)
        with self.assertRaises(LaneDispatchError):
            parts['dispatcher'].submit(SegmentCommand(
                support.TXN, 0, 'prefix', 0))
        self.assertEqual(len(bridge.offers), before)

    def test_aux_offer_outside_a_segment_is_invalidated_not_faked(self):
        bridge = support.FakeBridge()
        ledger = support.make_ledger(bridge)
        support.declare(ledger)
        parts = support.build_lane(bridge)
        npo.install_seams(parts['shared'], ledger, 'shared_experts',
                          pins=npo.live_pins(parts['shared']))
        parts['shared'].maybe_forward_async('x', 0)
        self.assertTrue(ledger.window()['invalidated'])
        self.assertIn('aux_without_binding', ledger.window()['invalid'])


if __name__ == '__main__':
    unittest.main()


class SourceShapedReviewTests(unittest.TestCase):
    def test_real_sparse_operation_copy_boundaries_have_copy_offers(self):
        # Actual generated sparse attention + operation; fake device boundary.
        import test_e3_sparse_history_dma as sparse
        sparse.GeneratedSparseTests.setUpClass()
        fixture = sparse.GeneratedSparseTests('test_compact_rows_and_remap_drive_the_arena_exactly')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        bridge = support.FakeBridge()
        ledger = support.make_ledger(bridge)
        support.declare(ledger)
        support.start_role_owners(ledger)
        device = fixture.device
        device.copy, device.compute = support.COPY, support.MAIN
        for method, direction in [('copy_indices_to_host', 'D2H'),
                                  ('copy_rows', 'H2D'), ('copy_pages', 'H2D')]:
            original = getattr(device, method)
            def observed(*args, _original=original, _direction=direction):
                result = _original(*args)
                bridge.activity('memcpy', stream=support.COPY,
                                copy_kind=npo.COPY_KIND[_direction])
                return result
            setattr(device, method, observed)
        original_prepare = fixture.prepare
        def prepare(*args, **kwargs):
            values = original_prepare(*args, **kwargs)
            operation = values[0]
            npo.install_seams(operation, ledger, 'qsa_operation',
                              pins=npo.live_pins(operation))
            return values
        fixture.prepare = prepare
        with ledger.segment('layer', transaction=support.TXN, op_seq=1):
            operation, _, buckets = fixture.sparse_case()
        self.assertGreater(buckets, 0)
        support.finish_role_owners(ledger)
        ledger.owner_finish('lane')
        bridge.close_window()
        ledger.finish()
        receipt = ledger.seal()
        self.assertEqual(receipt['verdict'], 'PASS', receipt)
        self.assertEqual(receipt['families']['qsa']['selection_readback']['memcpy'], 2)
        self.assertGreater(receipt['families']['qsa']['sparse_bucket_fill']['memcpy'], 0)
        self.assertTrue(all(row['role'] == 'copy' for row in receipt['joins']))
        self.assertTrue(operation.drain())
        ledger.close_shadows()

    def test_native_system_thread_cannot_disagree_with_bridge_offer(self):
        bridge, ledger, _, _, _, receipt = positive()
        native = copy.deepcopy(bridge.report()['native'])
        native['scopes'][0]['system_thread'] += 1
        out = grade(native, receipt)
        self.assertEqual(out['verdict'], 'FAIL', out)
        self.assertIn('native_scope_system_thread_mismatch', out['failures'])

    def test_full_ledger_observes_failure_without_blocking_original(self):
        from unittest.mock import patch
        bridge = support.FakeBridge()
        ledger = support.make_ledger(bridge)
        support.declare(ledger)
        parts = support.build_lane(bridge)
        operation = parts['operation']
        shadow = npo.install_seams(operation, ledger, 'qsa_operation',
                                   pins=npo.live_pins(operation))
        self.addCleanup(shadow.close)
        with patch.object(npo, 'MAX_OFFERS', 0):
            operation.fill_bucket(None, 0, 1, None)
        self.assertEqual(operation.buckets, 1)
        self.assertTrue(ledger.window()['invalidated'])

    def test_pop_failure_does_not_duplicate_original_or_hide_body_error(self):
        from contextlib import contextmanager
        bridge = support.FakeBridge()
        ledger = support.make_ledger(bridge)
        support.declare(ledger)
        parts = support.build_lane(bridge)
        operation = parts['operation']
        real_offer = bridge.offer
        @contextmanager
        def bad_pop(*args, **kwargs):
            try:
                with real_offer(*args, **kwargs) as row:
                    yield row
            finally:
                raise RuntimeError('test pop failed')
        bridge.offer = bad_pop
        shadow = npo.install_seams(operation, ledger, 'qsa_operation',
                                   pins=npo.live_pins(operation))
        self.addCleanup(shadow.close)
        operation.fill_bucket(None, 0, 1, None)
        self.assertEqual(operation.buckets, 1)
        self.assertTrue(ledger.window()['invalidated'])
        expected = ValueError('original body refused')
        def refused(*args):
            raise expected
        # Install a fresh owner after the previous exact shadows are removed.
        shadow.close()
        ledger = support.make_ledger(bridge)
        support.declare(ledger)
        operation.fill_sparse_bucket = refused
        shadow = npo.install_seams(operation, ledger, 'qsa_operation',
                                   pins=npo.live_pins(operation))
        self.addCleanup(shadow.close)
        with self.assertRaises(ValueError) as caught:
            operation.fill_sparse_bucket(None, None, 0, None)
        self.assertIs(caught.exception, expected)

    def test_unknown_family_is_missing_evidence_instead_of_grader_exception(self):
        bridge, ledger, _, _, _, receipt = positive()
        native = copy.deepcopy(bridge.report()['native'])
        rec = copy.deepcopy(receipt)
        next(iter(rec['declared'].values()))['family'] = 'unknown-family'
        out = grade(native, rec)
        self.assertEqual(out['verdict'], 'INCONCLUSIVE', out)
        self.assertIn('malformed_prefill_evidence:NativePrefillError', out['missing'])
