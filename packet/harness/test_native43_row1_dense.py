"""Native43 l-f row1 under the dense policy: the reference on its own dense branch.

l-e (q18a-native43-lifecycle run 1, 1 Nov 2026) failed row1 with route_mismatch: at 25 pages the dense
policy puts the native operation on dense_dma, while the original flash_e2_arena_attention, re-run as the
byte oracle, takes its sparse branch (sparse_uva).  The fix (lead decision (a), 1 Nov 2026) makes the
expected reference route a function of the policy (row1_oracle.expected_reference) and, where the policy
changed the native route, enters the reference on its own existing dense branch (row1_oracle.dense_branch):
the same code object, whose ``os.environ.get("VLLM_QSA_KVO_SPARSE_PREFILL", "1")`` reads '0'.  The process
environment is never written.

* ExpectedReferenceTests: the declared table, row by row, and its refusals.
* DenseBranchTests: the real generated flash_e2_arena_attention (arena_attention.generate over the pinned
  reference ops/qsa.py) under fake torch/triton/kernels.  Unforced it enters its sparse branch (the l-e
  path); forced it declares dense_dma, fills every bucket through e2.fill and launches one attention kernel
  per bucket; os.environ is unchanged; the code object, defaults and module globals are the original's.
* DensePolicyRow1Tests: the generated sparse helper and the real QsaPrefillOperation/Row1Qualifier, dense
  policy at 25 pages.  With the real reference the layer passes on dense_dma with two reference fills and two
  dense bucket comparisons.  With dense_branch disabled (the pre-fix composition) and the declared route
  stub, row1 fails with route_mismatch, the l-e failure.

Leaves are declared in harness/native43_row1_dense_fixture_leaves.json.  Nothing here is GPU evidence: the
attention kernel writes no output, so output byte equality is the oracle mechanism only.
"""
import os
import sys
import types
import unittest
from unittest import mock

import test_e3_native_source_oracle as so
import test_e3_sparse_history_dma as st
import qsa_operation as qo
import row1_oracle as ro
import sparse_history_dma as shd

SWITCH = ro.DENSE_SWITCH
PAGE_ROWS = shd.PAGE_ROWS
ARENA_PAGES = shd.ARENA_PAGES


class ExpectedReferenceTests(unittest.TestCase):
    def test_declared_table(self):
        rows = [
            # policy, native route, pages -> expected route, forced
            ('current', 'sparse_dma', 25, 'sparse_uva', False),
            ('current', 'dense_dma', 25, 'dense_dma', False),
            ('current', 'dense_dma', 24, 'dense_dma', False),
            ('current', 'empty', 25, 'empty', False),
            ('dense', 'dense_dma', 1, 'dense_dma', False),
            ('dense', 'dense_dma', 24, 'dense_dma', False),
            ('dense', 'dense_dma', 25, 'dense_dma', True),
            ('dense', 'dense_dma', 128, 'dense_dma', True),
        ]
        for policy, route, pages, expected, forced in rows:
            with self.subTest(policy=policy, route=route, pages=pages):
                self.assertEqual(ro.expected_reference(policy, route, pages, ARENA_PAGES), (expected, forced))

    def test_dense_beyond_the_arena_refuses_a_non_dense_native_route(self):
        for route in ('sparse_dma', 'empty'):
            with self.subTest(route=route), self.assertRaises(ro.Row1OracleError):
                ro.expected_reference('dense', route, 25, ARENA_PAGES)

    def test_below_the_threshold_the_dense_policy_keeps_the_pinned_mapping(self):
        for route in ('sparse_dma', 'dense_dma', 'empty'):
            self.assertEqual(ro.expected_reference('dense', route, 24, ARENA_PAGES),
                             ro.expected_reference('current', route, 24, ARENA_PAGES))

    def test_invalid_inputs_refuse(self):
        for args in (('DENSE', 'dense_dma', 25, 24), ('', 'dense_dma', 25, 24), ('dense', 'sparse_uva', 25, 24),
                     ('dense', 'dense_dma', 0, 24), ('dense', 'dense_dma', 25, 0), ('dense', 'dense_dma', 25.0, 24),
                     ('dense', 'dense_dma', True, 24)):
            with self.subTest(args=args), self.assertRaises(ro.Row1OracleError):
                ro.expected_reference(*args)


class SparseBranchEntered(Exception):
    pass


class SpyIndices(st.T):
    """logical_indices whose first sparse-branch call (index_select) raises: the branch was entered."""

    def index_select(self, *args, **kwargs):
        raise SparseBranchEntered


class Session:
    """The e2 surface the reference calls; records the calls (Row1ReferenceSession's interface)."""

    def __init__(self):
        self.calls = []

    def route(self, name, pages):
        self.calls.append(('route', name, pages))

    def fill(self, staged, lo, hi, table):
        self.calls.append(('fill', lo, hi, staged.shape[0]))

    def consume_begin(self):
        self.calls.append(('consume_begin',))

    def consume_end(self):
        self.calls.append(('consume_end',))


def compile_reference(test):
    """The real generated flash_e2_arena_attention under the harness fakes (module globals like ops/qsa.py)."""
    module = st.ast.Module(body=[st._function_node(test.ops_e2, 'flash_e2_arena_attention')], type_ignores=[])
    st.ast.fix_missing_locations(module)
    test.reference_kernel = st.FakeKernel('reference-gqa')
    test.reference_gather = st.FakeKernel('reference-gather')
    scope = {'__name__': 'ops.qsa', 'torch': test.torch, 'os': os, 'triton': st.FakeTriton(), '_STAGE_TIMER': None,
             '_qsa_sparse_staged_gqa_kernel': test.reference_kernel,
             '_qsa_gather_selected_host_rows': test.reference_gather,
             '_select_config': lambda rows, heads, prefill, width: (64, 4, 1, None)}
    exec(compile(module, '<generated e2 reference>', 'exec'), scope)
    return scope['flash_e2_arena_attention'], scope


class DenseBranchTests(st._GeneratedHarness, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.reference, self.scope = compile_reference(self)
        patcher = mock.patch.dict(os.environ, {SWITCH: '1'})  # the box identity-start value
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop('QSA_SPARSE_MIN_PAGES', None)  # unset on the box; restored by the patcher

    def call(self, function, pages, *, rows=8):
        logical = SpyIndices((rows, 5), device='cuda:0', dtype='torch.int32', values=[0, 1, 4048, 4049, 4] * rows)
        query = st.T((rows, 6, 256), device='cuda:0')
        output = st.T((rows, 6, 256), device='cuda:0')
        selected = st.T((rows,), device='cuda:0', dtype='torch.int64', values=list(range(rows)))
        session = Session()
        function(query, self.make_source(pages), logical, self.table, selected, pages - 1, self.arena, True, output,
                 e2=session)
        return session

    def test_unforced_the_reference_enters_its_sparse_branch_beyond_the_arena(self):
        with self.assertRaises(SparseBranchEntered):
            self.call(self.reference, 25)
        # Up to the arena it is dense without any force (both policies agree there).
        session = self.call(self.reference, 24)
        self.assertEqual(session.calls[0], ('route', 'dense_dma', 24))

    def test_forced_the_reference_takes_its_own_dense_branch(self):
        before = dict(os.environ)
        for pages in (25, 49, 128):
            with self.subTest(pages=pages):
                self.reference_kernel.calls.clear()
                forced = ro.dense_branch(self.reference)
                session = self.call(forced, pages)
                buckets = [(lo, min(lo + ARENA_PAGES, pages)) for lo in range(0, pages, ARENA_PAGES)]
                self.assertEqual(session.calls[0], ('route', 'dense_dma', pages))
                fills = [c for c in session.calls if c[0] == 'fill']
                self.assertEqual(fills, [('fill', lo, hi, hi - lo) for lo, hi in buckets])
                self.assertEqual(session.calls.count(('consume_begin',)), len(buckets))
                self.assertEqual(len(self.reference_kernel.calls), len(buckets))
                flags = [(c[2]['FIRST_BUCKET'], c[2]['LAST_BUCKET']) for c in self.reference_kernel.calls]
                self.assertEqual(flags, [(i == 0, i == len(buckets) - 1) for i in range(len(buckets))])
                self.assertEqual(self.reference_gather.calls, [])
        self.assertEqual(dict(os.environ), before)
        self.assertEqual(os.environ[SWITCH], '1')

    def test_the_forced_function_is_the_original_code_with_only_os_answered(self):
        forced = ro.dense_branch(self.reference)
        self.assertIs(forced.__code__, self.reference.__code__)
        self.assertEqual(forced.__defaults__, self.reference.__defaults__)
        self.assertEqual(forced.__kwdefaults__, self.reference.__kwdefaults__)
        self.assertIs(forced.__closure__, self.reference.__closure__)
        self.assertEqual(forced.__qualname__, self.reference.__qualname__)
        self.assertIsNot(forced.__globals__, self.scope)
        self.assertIs(self.scope['os'], os, 'the reference module globals are never changed')
        self.assertEqual({k for k in forced.__globals__ if forced.__globals__[k] is not self.scope.get(k)}, {'os'})
        self.assertEqual(forced.__globals__['os'].environ.get(SWITCH, '1'), '0')
        os.environ['QSA_PROBE_KEY'] = 'kept'
        self.assertEqual(forced.__globals__['os'].environ.get('QSA_PROBE_KEY'), 'kept')
        self.assertIsNone(forced.__globals__['os'].environ.get('QSA_ABSENT_KEY'))
        for name in ('getenv', 'path', 'putenv'):
            with self.subTest(name=name), self.assertRaises(AttributeError):
                getattr(forced.__globals__['os'], name)
        with self.assertRaises(AttributeError):
            forced.__globals__['os'].environ.__setitem__  # read-only view: no item assignment

    def test_only_the_pinned_plain_function_is_accepted(self):
        def no_switch(*args, e2):
            return os.environ.get('OTHER', '1')

        def own_os(*args, e2):
            return os.environ.get('VLLM_QSA_KVO_SPARSE_PREFILL', '1')
        own_os = types.FunctionType(own_os.__code__, dict(own_os.__globals__, os=types.SimpleNamespace()), 'own_os')
        bound = types.MethodType(self.reference, object())
        for candidate in (no_switch, own_os, bound, lambda *a, **k: None, len, None):
            with self.subTest(candidate=candidate), self.assertRaises(ro.Row1OracleError):
                ro.dense_branch(candidate)


class DenseRows(st.FakeSparseDevice):
    """FakeSparseDevice whose dense page copy moves the rows (the real copy_pages effect)."""

    def copy_pages(self, source, arena, source_page, arena_page, pages):
        super().copy_pages(source, arena, source_page, arena_page, pages)
        for page in range(pages):
            for offset in range(PAGE_ROWS):
                arena.storage.rows[arena.row_offset + (arena_page + page) * PAGE_ROWS + offset] = \
                    source.row_bytes((source_page + page) * PAGE_ROWS + offset)


def route_stub(q, kv_cache, logical_indices, block_table_row, rows, max_logical_page, arena, use_prefill_config,
               out, retained=None, retained_copied=0, *, e2):
    """Declared stand-in for the pinned reference's route decision and dense fill sequence (leaf route-stub)."""
    pages_in_arena = arena.shape[0]
    num_pages = max_logical_page + 1
    route_stub.process_switch.append(sys.modules['os'].environ.get(SWITCH))
    if (num_pages > int(os.environ.get('QSA_SPARSE_MIN_PAGES', pages_in_arena))
            and os.environ.get('VLLM_QSA_KVO_SPARSE_PREFILL', '1') == '1'):
        e2.route('sparse_uva', num_pages)  # its own UVA gather; no e2.fill
        return
    e2.route('dense_dma', num_pages)
    for lo in range(0, num_pages, pages_in_arena):
        hi = min(lo + pages_in_arena, num_pages)
        e2.fill(arena[:hi - lo], lo, hi, block_table_row)


route_stub.process_switch = []


class DensePolicyRow1Tests(st._GeneratedHarness, unittest.TestCase):
    for _name in ('compile_helper', 'make_operation', 'drive'):
        locals()[_name] = getattr(so.GeneratedRow1Tests, _name)

    def setUp(self):
        self.store = ro.Row1RecordStore(capacity=64)
        self.reference = route_stub
        st._GeneratedHarness.setUp(self)
        self.device = DenseRows(auto_ready=True)
        self.real_reference, _ = compile_reference(self)
        patcher = mock.patch.dict(os.environ, {SWITCH: '1'})
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop('QSA_SPARSE_MIN_PAGES', None)
        route_stub.process_switch.clear()

    def use(self, reference):
        self.reference = reference
        self.helper = self.compile_helper()

    def case(self, pages, policy):
        """GeneratedRow1Tests.case with an identity block table and the policy set before the operation latches it."""
        self.device.native_qsa_policy = policy
        tokens = (pages - 1) * PAGE_ROWS + 17
        mapping = tuple(range(pages))
        operation, lane, transaction = self.make_operation([('qsa.0', mapping)])
        source = self.make_source(pages)
        operation.register_source('qsa.0', source, versions=tuple(range(1, pages + 3)))
        operation.begin_forward(layers=['qsa.0'])
        self.device.auto_ready = False
        operation.begin_layer('qsa.0', source)
        operation.producer_submitted('qsa.0', source)
        hold = shd.SparseSourceHold(st.make_owner(lane, 'qsa.0', mapping, tokens, transaction=transaction),
                                    allocator=st.FakeAllocator())
        operation.attach_sparse_source('qsa.0', hold)
        operation.qualification = ro.Row1Qualifier(
            self.store, transaction=transaction, lane=lane, backend=so.RowBackend(),
            epochs=lambda: dict(worker_incarnation='generated-worker', engine_epoch=2, route_generation=1))
        layer = types.SimpleNamespace(layer_name='qsa.0', _qsa_kv_offload=True, _qsa_stage_min_rows=1,
                                      _qsa_stage_arena_bytes=so.ARENA_ROWS * so.ROW_BYTES)
        flat = []
        for row in range(tokens):
            flat += [row % 60000, -1]
        logical = st.T((tokens, 2), device='cuda:0', dtype='torch.int32', values=flat)
        query = st.T((tokens, so.QUERY_HEADS, so.HEAD_DIM), device='cuda:0', dtype='torch.bfloat16')
        out_rows = so.Rows(tokens)
        out_rows.rows = [b'OUT'] * tokens
        output = st.T((tokens, so.QUERY_HEADS, so.HEAD_DIM), device='cuda:0', dtype='torch.bfloat16',
                      storage=out_rows)
        metadata = types.SimpleNamespace(
            _qsa_host_meta=((0, tokens), (tokens,)),
            query_start_loc=st.T((2,), device='cpu', dtype='torch.int32'),
            seq_lens=st.T((1,), device='cpu', dtype='torch.int32'),
            block_table=st.T((1, 260), device='cuda:0', dtype='torch.int64', values=list(range(260))))
        operation.attach_continuation(self.helper(operation, layer, query, source, logical, metadata, tokens, True,
                                                  output))
        return operation, hold

    def finish(self, operation, hold):
        operation.end_layer('qsa.0')
        operation.end_forward()
        self.device.ready_all()
        self.assertTrue(operation.drain())
        operation.release()
        self.assertTrue(hold.released)

    def test_dense_policy_25_pages_passes_dense_against_the_real_reference(self):
        before = dict(os.environ)
        self.use(self.real_reference)
        operation, hold = self.case(25, 'dense')
        self.assertEqual(operation.transport_policy, 'dense')
        self.drive(operation)
        run = operation._layers['qsa.0']
        self.assertEqual(run.sparse.route, 'dense_dma')
        self.assertEqual([(b.lo, b.hi) for b in run.buckets], [(0, 24), (24, 25)])
        buckets = [r for r in self.store.records() if r.identity.operation == 'dense_page_bucket']
        outputs = [r for r in self.store.records() if r.identity.operation == 'qsa_output_before_moe']
        self.assertEqual([(r.bucket, r.rows, r.equal) for r in buckets],
                         [(0, 24 * PAGE_ROWS, True), (1, PAGE_ROWS, True)])
        self.assertEqual(len(outputs), 1)
        self.assertEqual((outputs[0].route, outputs[0].pieces, outputs[0].equal, outputs[0].finite),
                         ('dense_dma', 2, True, True))
        self.assertEqual(self.store.failures(), ())
        self.assertEqual(len(self.reference_kernel.calls), 2)
        self.assertEqual(self.reference_gather.calls, [])
        self.finish(operation, hold)
        layer, = self.store.layer_records()
        self.assertTrue(layer.passed)
        self.assertEqual((layer.route, layer.bucket_checks, layer.output_checks), ('dense_dma', 2, 1))
        self.assertEqual(dict(os.environ), before)

    def test_dense_policy_with_the_route_stub_passes_and_the_process_switch_stays_on(self):
        self.use(route_stub)
        operation, hold = self.case(25, 'dense')
        self.drive(operation)
        output = [r for r in self.store.records() if r.identity.operation == 'qsa_output_before_moe'][0]
        self.assertEqual((output.route, output.pieces, output.equal), ('dense_dma', 2, True))
        self.assertEqual(route_stub.process_switch, ['1'])
        self.finish(operation, hold)

    def test_pre_fix_composition_fails_with_the_l_e_route_mismatch(self):
        self.use(route_stub)
        operation, hold = self.case(25, 'dense')
        with mock.patch.object(qo, 'dense_branch', lambda reference: reference):
            with self.assertRaises(ro.Row1OracleError) as caught:
                self.drive(operation)
        self.assertIn('route_mismatch', str(caught.exception))
        failure, = self.store.failures()
        self.assertEqual(failure.kind, 'route_mismatch')
        self.assertEqual(dict(failure.evidence), {'reference_route': 'sparse_uva', 'expected_route': 'dense_dma'})
        self.assertFalse(hold.released)
        operation.cancel('pre-fix fixture')

    def test_dense_policy_at_the_arena_needs_no_force(self):
        self.use(route_stub)
        operation, hold = self.case(24, 'dense')
        with mock.patch.object(qo, 'dense_branch', side_effect=AssertionError('no force at 24 pages')):
            self.drive(operation)
        output = [r for r in self.store.records() if r.identity.operation == 'qsa_output_before_moe'][0]
        self.assertEqual((output.route, output.pieces, output.equal), ('dense_dma', 1, True))
        self.finish(operation, hold)

    def test_current_policy_is_unchanged_beyond_the_arena(self):
        self.use(route_stub)
        operation, hold = self.case(25, 'current')
        with mock.patch.object(qo, 'dense_branch', side_effect=AssertionError('no force under current')):
            self.drive(operation)
        output = [r for r in self.store.records() if r.identity.operation == 'qsa_output_before_moe'][0]
        self.assertEqual((output.route, output.pieces, output.equal), ('sparse_dma', 0, True))
        self.finish(operation, hold)


if __name__ == '__main__':
    unittest.main()
