"""CPU-only tests for the E3 QSA layer driver and its attention-module transform (D4/D2)."""
import ast
import hashlib
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))

import e3_types as et  # noqa: E402
import operation_source  # noqa: E402
import qsa_driver  # noqa: E402
import qsa_operation as qo  # noqa: E402
import runner_hooks  # noqa: E402
import sparse_history_dma as shd  # noqa: E402
from e3_types import E3Pending  # noqa: E402
from model_hooks import LaneForward  # noqa: E402
from qsa_driver import QsaDriverError, QsaLayerDriver  # noqa: E402

REFERENCE = ROOT / 'reference/vllm-src'


class TransformTests(unittest.TestCase):
    def setUp(self):
        if not REFERENCE.is_dir():
            self.skipTest('reference/vllm-src is not available on this host')
        self.path = 'models/qwen4_exp/nvidia/qsa.py'
        pinned = (REFERENCE / self.path).read_text()
        self.composed = operation_source.generate_layer(runner_hooks.qsa_staged(pinned))

    def test_pin_is_the_d2_composed_source(self):
        self.assertEqual(hashlib.sha256(self.composed.encode()).hexdigest(), qsa_driver.PINS[self.path])

    def test_generate_and_exact_inverse(self):
        out = qsa_driver.generate_one(self.path, self.composed)
        ast.parse(out)
        tree = ast.parse(out)
        cls = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.ClassDef) and node.name == 'Qwen4ExpQSAAttention')
        names = [node.name for node in cls.body if isinstance(node, ast.FunctionDef)]
        self.assertIn('_flash_e3_attention_pre', names)
        self.assertIn('_flash_e3_attention_post', names)
        self.assertIn('forward', names)
        self.assertEqual(qsa_driver.revert_one(self.path, out, expected=self.composed), self.composed)
        with self.assertRaises(ValueError):
            qsa_driver.generate_one(self.path, out)
        with self.assertRaises(ValueError):
            qsa_driver.generate_one(self.path, self.composed + '\n')
        with self.assertRaises(ValueError):
            qsa_driver.revert_one(self.path, self.composed)
        with self.assertRaises(ValueError):
            qsa_driver.generate_one('other.py', self.composed)


# --- runtime fakes --------------------------------------------------------------------

class Shape:
    def __init__(self, *dims):
        self.shape = dims

    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    def zero_(self):
        self.zeroed = True
        return self

    def __getitem__(self, item):
        return self


class FakeOperation:
    def __init__(self, buckets=2, pending_before=(1,)):
        self.calls = []
        self.buckets = buckets
        self.pending_before = set(pending_before)
        self.continuation = None
        self.advances = 0
        self.produced = 0

    def begin_layer(self, name, source):
        self.calls.append(('begin', name))

    def producer_submitted(self, name, *owners):
        self.calls.append(('producer', name, len(owners)))

    def attach_continuation(self, generator):
        self.calls.append(('attach',))
        self.continuation = generator

    def advance(self):
        self.advances += 1
        if self.advances in self.pending_before:
            raise E3Pending('consumer not queried')
        return next(self.continuation)

    def end_layer(self, name):
        self.calls.append(('end', name))

    def cancel(self, reason):
        self.calls.append(('cancel', reason))
        return reason


def staged_operation(operation, attn, query, kv_cache, logical, main, num_tokens, use_prefill, output):
    for bucket in range(operation.buckets):
        yield ('ticket', bucket)
    return (0, num_tokens)


def make_model(torch=None):
    def attn_layer(idx):
        indexer = types.SimpleNamespace(
            packed_output_width=7, raw_key_cache=types.SimpleNamespace(prefix=f'side{idx}'),
            index_qk_proj=lambda h: ('pqk', None))
        indexer_call = lambda pqk, pos, buf: Shape(8, 7)  # noqa: E731
        indexer.__call__ = indexer_call
        attn = types.SimpleNamespace(
            layer_name=f'qsa{idx}', _qsa_host_kv='host', kv_cache=Shape(4, 1, 4048, 512),
            topk_indices_buffer=Shape(8, 2052), _max_decode_query_len=1,
            impl=types.SimpleNamespace(do_kv_cache_update=lambda *a: None),
            indexer=types.SimpleNamespace(
                packed_output_width=7, raw_key_cache=types.SimpleNamespace(prefix=f'side{idx}'),
                index_qk_proj=lambda h: ('pqk', None)),
        )
        attn.indexer = _CallableIndexer(idx)
        attn._flash_e3_attention_pre = lambda positions, hidden: {
            'query': 'q', 'key': 'k', 'value': 'v', 'gate': None, 'attn_output': Shape(8, 12, 256),
            'projected_qk': 'pqk', 'num_tokens': 8}
        attn._flash_e3_attention_post = lambda state: ('o', state['num_tokens'])
        layer = types.SimpleNamespace(layer_type='full_attention', self_attn=attn, ple=None)
        layer._flash_e3_layer_pre = lambda h, b, i, **kw: (f'{h}+pre', 'block_in', 'inj')
        layer._flash_e3_layer_post = lambda h, a, i: (f'{h}|{a[0]}', 'mlp', 'inj2')
        return layer

    layers = [types.SimpleNamespace(layer_type='linear_attention', ple=None), attn_layer(1),
              types.SimpleNamespace(layer_type='linear_attention', ple=None)]
    model = types.SimpleNamespace(layers=layers, start_layer=0, end_layer=3, _mtp_hidden_buffer=None,
                                  _flash_e3_is_last_stage=lambda: False)
    model._flash_e3_forward_prefix = lambda *a: 'h0'
    model._flash_e3_forward_layer = lambda idx, h, *a: (f'{h}>{idx}', 'b', 'i')
    model._flash_e3_forward_suffix = lambda h, b, i, last: {'hidden_states': h}
    return model


class _CallableIndexer:
    def __init__(self, idx):
        self.packed_output_width = 7
        self.raw_key_cache = types.SimpleNamespace(prefix=f'side{idx}')

    def index_qk_proj(self, hidden):
        return 'pqk', None

    def __call__(self, pqk, positions, buffer):
        return Shape(8, 7)


def metadata():
    main = types.SimpleNamespace(num_actual_tokens=8, slot_mapping='slots', max_query_len=8)
    side = types.SimpleNamespace(num_actual_tokens=8)
    return types.SimpleNamespace(attn_metadata={'qsa1': main, 'side1': side})


def inputs():
    return dict(positions='pos', input_ids='ids', query_start_loc='qsl', ngram_context='ng')


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.model = make_model()
        self.operation = FakeOperation(buckets=2, pending_before={2})
        self.driver = QsaLayerDriver(self.operation, staged_operation=staged_operation,
                                     forward_context=metadata, clock=lambda: 0)
        self.lane = LaneForward(self.model, inputs(), stream=types.SimpleNamespace(cuda_stream=5),
                                attention_driver=self.driver, clock=lambda: 0)

    def test_owns_only_qsa_layers_with_lane_methods(self):
        self.assertFalse(self.driver.owns(self.model, 0))
        self.assertTrue(self.driver.owns(self.model, 1))
        del self.model.layers[1].self_attn._flash_e3_attention_pre
        self.assertFalse(self.driver.owns(self.model, 1))

    def test_phases_pre_buckets_pending_tail(self):
        lane = self.lane
        lane.begin()
        lane.layer(0)
        kinds = []
        # pre -> paused; advance#1 bucket0 -> paused; advance#2 pending -> paused;
        # advance#3 bucket1 -> paused; advance#4 StopIteration -> typed tail
        # boundary; the next layer call executes tail after peer agreement.
        for _ in range(6):
            record = lane.layer(1)
            kinds.append(record.kind)
        self.assertEqual(kinds, ['layer-paused'] * 5 + ['layer'])
        self.assertEqual(lane.cursor, 2)
        self.assertEqual(lane.hidden_states, 'h0>0+pre|o')
        self.assertEqual([c[0] for c in self.operation.calls], ['begin', 'producer', 'attach', 'end'])
        self.assertEqual(self.operation.calls[1], ('producer', 'qsa1', 3))
        tags = [r[0] for r in self.driver.records]
        self.assertEqual(tags, ['pre', 'bucket', 'pending', 'bucket', 'attended', 'tail'])
        self.assertEqual(self.driver.records[4][2], (0, 8))
        self.assertIsNone(self.driver.cursor())
        lane.layer(2)
        lane.finish()
        self.assertEqual(lane.output, {'hidden_states': 'h0>0+pre|o>2'})

    def test_cursor_and_abort(self):
        self.lane.begin()
        self.lane.layer(0)
        self.lane.layer(1)
        self.assertEqual(self.driver.cursor(), (1, 'bucket', 0))
        self.lane.layer(1)
        self.assertEqual(self.driver.cursor(), (1, 'bucket', 1))
        self.assertEqual(self.driver.abort('cancelled'), 'cancelled')
        self.assertIsNone(self.driver.cursor())
        self.assertIn(('cancel', 'cancelled'), self.operation.calls)

    def test_metadata_and_identity_refusals(self):
        bad = QsaLayerDriver(self.operation, staged_operation=staged_operation,
                             forward_context=lambda: types.SimpleNamespace(attn_metadata=None))
        lane = LaneForward(self.model, inputs(), stream=None, attention_driver=bad, clock=lambda: 0)
        lane.begin()
        lane.layer(0)
        with self.assertRaises(QsaDriverError):
            lane.layer(1)
        missing = QsaLayerDriver(self.operation, staged_operation=staged_operation,
                                 forward_context=lambda: types.SimpleNamespace(attn_metadata={'qsa1': None}))
        lane = LaneForward(self.model, inputs(), stream=None, attention_driver=missing, clock=lambda: 0)
        lane.begin()
        lane.layer(0)
        with self.assertRaises(QsaDriverError):
            lane.layer(1)
        mismatched = metadata()
        mismatched.attn_metadata['side1'].num_actual_tokens = 9
        wrong = QsaLayerDriver(self.operation, staged_operation=staged_operation,
                               forward_context=lambda: mismatched)
        lane = LaneForward(self.model, inputs(), stream=None, attention_driver=wrong, clock=lambda: 0)
        lane.begin()
        lane.layer(0)
        with self.assertRaises(QsaDriverError):
            lane.layer(1)
        with self.assertRaises(QsaDriverError):
            QsaLayerDriver(None, staged_operation=staged_operation, forward_context=metadata)
        with self.assertRaises(QsaDriverError):
            QsaLayerDriver(self.operation, staged_operation=None, forward_context=metadata)


# --- sparse-history driver seam -------------------------------------------------------

class LeaseAllocator:
    """Retains a reference object for the shared source-hold proof."""

    def __init__(self):
        self.refs = set()

    def retain(self, owner):
        reference = object()
        self.refs.add(id(reference))
        return reference

    def assert_retained(self, reference):
        if id(reference) not in self.refs:
            raise RuntimeError('lease lost')

    def release(self, reference):
        self.refs.discard(id(reference))


def make_sparse_hold(layer='qsa1'):
    inner_transaction = et.TransactionId(engine_epoch=1, request_uid='req',
                                         request_generation=0, chunk_start=0,
                                         chunk_end=8, txn_seq=1)
    inner_lane = et.LaneId(worker_rank=0, role=et.LaneRole.PREFILL,
                           storage_epoch=2, context_id=0)
    owner = shd.SourceOwner(inner_transaction, inner_lane, layer,
                            storage_epoch=inner_lane.storage_epoch,
                            source_epoch=0, source_incarnation=1,
                            allocation_incarnation=1,
                            physical_pages=(3,), valid_tokens=8,
                            mapping_version=1)
    return shd.SparseSourceHold(owner, allocator=LeaseAllocator())


class SparseFakeOperation(FakeOperation):
    def __init__(self, *, pending_selection=False, refuse_after=0, **kwargs):
        super().__init__(**kwargs)
        self.pending_selection = pending_selection
        self.refuse_after = refuse_after
        self.attached = []
        self.sparse_source = None
        self.next_sparse_ticket = 0

    def attach_sparse_source(self, name, hold):
        self.calls.append(('hold', name))
        self.attached.append(hold)
        self.sparse_source = (name, hold)

    def has_sparse_source(self, name):
        return self.sparse_source is not None and self.sparse_source[0] == name

    def sparse_selection_pending(self, token):
        return qo.SparseSelectionPending(token)

    def stage_ticket(self):
        self.next_sparse_ticket += 1
        return self.next_sparse_ticket


def sparse_staged_operation(operation, attn, query, kv_cache, logical, main,
                            num_tokens, use_prefill, output):
    if operation.pending_selection:
        yield operation.sparse_selection_pending(operation.stage_ticket())
    for bucket in range(operation.buckets):
        yield ('sparse-ticket', bucket)
    return (0, num_tokens)


def refusing_staged_operation(operation, attn, query, kv_cache, logical, main,
                              num_tokens, use_prefill, output):
    yield ('sparse-ticket', 0)
    raise shd.SparsePlanRefused(
        shd.SparseRefusal('descriptor_cap', 0, shd.MAX_DESCRIPTORS + 1,
                          num_tokens, 1, num_tokens * shd.ROW_BYTES, 1))


class SparseDriverTests(unittest.TestCase):
    def build(self, operation, **kwargs):
        driver = QsaLayerDriver(
            operation, staged_operation=staged_operation,
            sparse_staged_operation=kwargs.pop('sparse_staged_operation',
                                               sparse_staged_operation),
            source_holds=kwargs.pop('source_holds', None),
            forward_context=metadata, clock=lambda: 0)
        lane = LaneForward(make_model(), inputs(),
                           stream=types.SimpleNamespace(cuda_stream=5),
                           attention_driver=driver, clock=lambda: 0)
        lane.begin()
        lane.layer(0)
        return driver, lane

    def test_hold_attaches_before_the_compact_helper_runs(self):
        operation = SparseFakeOperation(buckets=1, pending_before=set(),
                                        pending_selection=True)
        hold = make_sparse_hold()
        driver, lane = self.build(operation, source_holds={'qsa1': hold})
        self.assertEqual(lane.layer(1).kind, 'layer-paused')   # pre + attach
        self.assertEqual(operation.calls[-2:], [('hold', 'qsa1'), ('attach',)])
        self.assertIs(operation.attached[0], hold)
        self.assertTrue(operation.has_sparse_source('qsa1'))
        self.assertEqual(lane.layer(1).kind, 'layer-paused')   # selection phase
        self.assertEqual(lane.layer(1).kind, 'layer-paused')   # compact bucket
        lane.layer(1)                                          # StopIteration/boundary
        self.assertTrue(driver.at_tail_boundary(1))
        lane.layer(1)                                          # tail after boundary
        tags = [record[0] for record in driver.records]
        self.assertEqual(tags, ['pre', 'selection', 'bucket', 'attended', 'tail'])
        self.assertEqual(driver.phases, {})

    def test_dense_helper_is_kept_when_no_hold_is_admitted(self):
        operation = SparseFakeOperation(buckets=1, pending_before=set())
        driver, lane = self.build(operation)
        lane.layer(1)
        self.assertFalse(operation.has_sparse_source('qsa1'))
        self.assertEqual(operation.attached, [])
        self.assertNotIn(('hold', 'qsa1'), operation.calls)
        lane.layer(1)
        lane.layer(1)
        lane.layer(1)
        self.assertEqual(driver.records[-1][0], 'tail')

    def test_typed_refusal_parks_the_layer_and_never_counts_overlap(self):
        operation = SparseFakeOperation(buckets=1, pending_before=set())
        driver, lane = self.build(
            operation, source_holds={'qsa1': make_sparse_hold()},
            sparse_staged_operation=refusing_staged_operation)
        lane.layer(1)                      # pre; the generator body is lazy
        lane.layer(1)                      # the first sparse bucket is admitted
        with self.assertRaises(QsaDriverError) as caught:
            lane.layer(1)                  # the generator then refuses
        self.assertIn('descriptor_cap', str(caught.exception))
        refusal = [record for record in driver.records
                   if record[0] == 'sparse_refusal']
        self.assertEqual(refusal, [('sparse_refusal', 1, 'descriptor_cap', 8, 8192)])
        self.assertEqual(driver.phases, {})
        self.assertIn(('cancel', 'sparse_refusal'), operation.calls)
        # Only buckets admitted before the typed refusal are reported; the
        # layer never reaches attention/tail and cannot pass as overlap success.
        self.assertEqual([record[0] for record in driver.records],
                         ['pre', 'bucket', 'sparse_refusal'])
        self.assertNotIn('attended', [record[0] for record in driver.records])

    def test_source_holds_require_the_sparse_helper_and_an_admitted_hold(self):
        hold = make_sparse_hold()
        operation = FakeOperation()
        with self.assertRaises(QsaDriverError):
            QsaLayerDriver(operation, staged_operation=staged_operation,
                           forward_context=metadata,
                           source_holds={'qsa1': hold})
        with self.assertRaises(QsaDriverError):
            QsaLayerDriver(operation, staged_operation=staged_operation,
                           sparse_staged_operation=sparse_staged_operation,
                           forward_context=metadata,
                           source_holds={'qsa1': object()})
        with self.assertRaises(QsaDriverError):
            QsaLayerDriver(operation, staged_operation=staged_operation,
                           sparse_staged_operation='not-callable',
                           forward_context=metadata)
        with self.assertRaises(QsaDriverError):
            QsaLayerDriver(operation, staged_operation=staged_operation,
                           sparse_staged_operation=sparse_staged_operation,
                           source_holds={'': hold}, forward_context=metadata)


if __name__ == '__main__':
    unittest.main()
