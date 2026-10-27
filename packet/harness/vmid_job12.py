#!/usr/bin/env python3
"""One serial QSA-only job: fixed-capacity native union/remap + sparse CPU DMA.

No model, engine, TP, GDN/HC/PLE/MoE, projection weights, speed claim or engine
control.  Reuses job4's bounded synthetic QSA setup: a pinned host KV pool, the
existing 24-page arena, the pinned staged-attention entry point and the same
public oracle.  The candidate path is the real E3 operation: fixed-capacity
sort/scan/pack/remap on the device, bounded D2H of the count and selection into
pinned scratch, a yielded pending sentinel, compact-row DMA and attention.

Only the harness reads host scalars, and only after an explicit ``synchronize``
of the whole probe.  The operation's selection path never synchronizes and the
pending branch never touches pinned bytes before the copy event is queried.

Run inside the pinned image (the queue does exactly this):

    python3 <H>/vmid_job12.py --mode probe --out <Q>
"""
import argparse
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
# Harness bundle layout mirrors patches/: vmid_sources/s5_engine and
# vmid_sources/sparse_stage.  A repository checkout falls back to patches/.
_BUNDLE = HERE / 'vmid_sources' / 's5_engine'
_S5 = _BUNDLE if _BUNDLE.is_dir() else HERE.parent / 'patches' / 's5_engine'
if str(_S5) not in sys.path:
    sys.path.insert(0, str(_S5))

import vmid_support as s
import vmid_qsa_dma as qd
import vmid_sparse_plan as sp
import e3_types as et
import operation_source
import arena_attention
import qsa_operation as qo
import sparse_history_dma as shd

LAYERS = 4
QUERY_ROWS = 80
QUERY_HEADS = 12
HEAD_DIM = 256
TOPK = 2051
WIDTH = TOPK + 1
MAX_LOGICAL_PAGE = 49
NUM_PAGES = MAX_LOGICAL_PAGE + 1
PAGE = shd.PAGE_ROWS
ARENA_PAGES = shd.ARENA_PAGES
ARENA_ROWS = shd.ARENA_ROWS
VALID_TOKENS = NUM_PAGES * PAGE
CAPACITY = NUM_PAGES * PAGE
MAP_PAGES = 52
VERSIONS = (1,) * MAP_PAGES
DELAY_CYCLES = 2_000_000_000
# Selection capacity is host-known: the pinned mark/scan buffers are sized by
# NUM_PAGES*PAGE, never by a device value.
SELECTION_BYTES = CAPACITY * 4

IDENTITY = tuple(range(MAP_PAGES))
FRAGMENTED = tuple((i * 17 + 3) % MAP_PAGES for i in range(MAP_PAGES))
ALIASED = (3, 3) + IDENTITY[2:]


def chunks(tokens, width=TOPK):
    """Split a sorted token run into query rows; the rest of each row pads."""
    tokens = tuple(tokens)
    if any(a >= b for a, b in zip(tokens, tokens[1:])):
        raise ValueError('chunk tokens must be sorted and unique')
    return tuple(tokens[lo:lo + width] for lo in range(0, len(tokens), width))


def case_specs():
    """Bounded selection/padding/duplicate/threshold matrix; no device import."""
    p = PAGE
    return (
        dict(name='all_padding', rows=(), mapping=IDENTITY),
        dict(name='interleaved_padding', mapping=FRAGMENTED, rows=(
            (0, -1, p - 1, -1, p),
            (p + 1, -1, 2 * p - 1, 2 * p, -1, 0),
        )),
        dict(name='duplicate_rows', mapping=IDENTITY, rows=(
            tuple(range(10)), tuple(range(10)), (0, 5, 9),
        )),
        dict(name='page_edges', mapping=FRAGMENTED, rows=(
            (0, p - 1, p, p + 1, 2 * p - 1, 2 * p, 3 * p - 1),
        )),
        dict(name='aliased_pages', mapping=ALIASED, rows=(
            (0, p - 1, p, p + 1, 2 * p - 1, 2 * p),
        )),
        # 40*p-1 distinct tokens: one token below the pinned 0.8 threshold.
        dict(name='threshold_sparse', mapping=IDENTITY,
             rows=chunks(range(40 * p - 1))),
        # Exactly 0.8 of the fixed capacity: the pinned dense fallback.
        dict(name='threshold_dense', mapping=IDENTITY,
             rows=chunks(range(40 * p))),
        dict(name='arena_tail_cancel', mapping=FRAGMENTED,
             rows=chunks(range(25 * p + 13))),
        dict(name='delayed_selection', mapping=IDENTITY, rows=(
            (0, p - 1, p, p + 1),)),
    )


CASES = case_specs()
# The case specs are immutable module constants; their derived union and plan
# are cached by identity so grading never rebuilds a 160k-row plan per layer.
_CASES_BY_NAME = {case['name']: case for case in CASES}
_UNION_CACHE = {}
_PLAN_CACHE = {}


def _cached(case, name, cache, build):
    if name is not None and _CASES_BY_NAME.get(name) is case:
        value = cache.get(name)
        if value is None:
            value = build()
            cache[name] = value
        return value
    return build()


def selection_union(case):
    """The actual selected-row union; duplicates collapse, padding is -1."""
    return _cached(case, case.get('name'), _UNION_CACHE, lambda: tuple(sorted(
        {value for row in case['rows'] for value in row if value != -1})))


def expected_route(case):
    count = len(selection_union(case))
    if count == 0:
        return 'empty'
    return 'sparse_dma' if count * 5 < CAPACITY * 4 else 'dense_dma'


def rows_matrix(rows, slots=QUERY_ROWS, width=TOPK, frontier=VALID_TOKENS):
    """Pure row layout: selected values first, then -1 padding, then counts."""
    if len(rows) > slots:
        raise ValueError('rows exceed the query slot count')
    flat, counts = [], []
    for index in range(slots):
        row = tuple(rows[index]) if index < len(rows) else ()
        if len(row) > width:
            raise ValueError('row exceeds the selection width')
        for value in row:
            if type(value) is not int or not -1 <= value < frontier:
                raise ValueError('selection value outside the valid frontier')
        flat.extend(row)
        flat.extend([-1] * (width - len(row)))
        counts.append(sum(1 for value in row if value != -1))
    return flat, counts


def expected_plan(case):
    """The real sparse transport plan for this case, built without a device."""
    return _cached(case, case.get('name'), _PLAN_CACHE, lambda: shd.make_sparse_plan(
        selection_union(case), tuple(case['mapping'][:NUM_PAGES]), VERSIONS, 1,
        valid_tokens=VALID_TOKENS))


def expected_buckets(case):
    route = expected_route(case)
    if route == 'empty':
        return 0
    if route == 'sparse_dma':
        return len(expected_plan(case).buckets)
    return len(qo.plan_buckets(tuple(case['mapping'][:NUM_PAGES])))


def expected_descriptors(case):
    return expected_plan(case).descriptors if expected_route(case) == 'sparse_dma' else None


# --- loading the pinned source ------------------------------------------------

def source_path():
    bundled = HERE / 'vmid_sources' / 'ops_qsa.py'
    return bundled if bundled.exists() else HERE.parent / 'patches' / 'retain' / 'ops_qsa.py.v8.1'


def load_attention():
    source = source_path()
    if s.sha(source) != qd.SOURCE_SHA:
        raise ValueError('unpinned_qsa_source')
    # Actual Triton definitions from the reviewed source, not image stock.
    from importlib.machinery import SourceFileLoader
    loader = SourceFileLoader('vmid_job12_qsa', str(source))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise ValueError('unpinned_qsa_source')
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    e2 = arena_attention.generate(source.read_text())
    if hashlib.sha256(e2.encode()).hexdigest() != operation_source.E2_OPS_SHA256:
        raise ValueError('unreviewed_e2_attention_source')
    namespace = dict(module.__dict__)
    exec(compile(operation_source.generate_ops(e2), '<vm<vm-id>-job12>', 'exec'),
         namespace)
    return module, namespace


# --- CPU-side fixtures used by the probe and its selftest ---------------------

class ProbeAllocator:
    """A real retained reference object; never a boolean readiness flag."""

    def __init__(self):
        self.refs = set()
        self.count = 0

    def retain(self, owner):
        self.count += 1
        reference = ('lease', self.count, id(owner))
        self.refs.add(reference)
        return reference

    def assert_retained(self, reference):
        if reference not in self.refs:
            raise RuntimeError('allocator lease lost')

    def release(self, reference):
        if reference not in self.refs:
            raise RuntimeError('allocator lease already dropped')
        self.refs.discard(reference)


def make_hold(lane, transaction, layer, mapping, allocator):
    owner = shd.SourceOwner(
        transaction, lane, layer, storage_epoch=lane.storage_epoch,
        source_epoch=0, source_incarnation=1, allocation_incarnation=1,
        physical_pages=tuple(mapping[:NUM_PAGES]), valid_tokens=VALID_TOKENS,
        mapping_version=1)
    return shd.SparseSourceHold(owner, allocator=allocator)


def transaction(index):
    return et.TransactionId(engine_epoch=1, request_uid='vm<vm-id>-job12',
                            request_generation=0, chunk_start=0,
                            chunk_end=QUERY_ROWS, txn_seq=index)


LANE = None


def lane():
    global LANE
    if LANE is None:
        LANE = et.LaneId(worker_rank=0, role=et.LaneRole.PREFILL,
                         storage_epoch=3, context_id=0)
    return LANE


class Probe:
    """GPU fixture owner.  Constructed only inside ``probe``."""

    def __init__(self, torch, module, namespace, outdir):
        self.torch = torch
        self.module = module
        self.namespace = namespace
        self.outdir = outdir
        torch.cuda.set_per_process_memory_fraction(.125, 0)
        torch.set_num_threads(4)
        torch.manual_seed(12012)
        if torch.cuda.device_count() != 1:
            raise RuntimeError('one_device_required')
        if torch.cuda.get_device_capability(0) != (8, 6):
            raise RuntimeError('one_sm86_device_required')
        self.device = qo.NativeQsaDevice('cuda:0')
        self.compute = torch.cuda.current_stream(0)
        self.copy = torch.cuda.Stream(device='cuda:0')
        self.device.bind_streams(self.compute, self.copy)
        self.hosts = [torch.empty((MAP_PAGES, 1, PAGE, 512),
                                  dtype=torch.bfloat16, device='cpu',
                                  pin_memory=True).normal_(0, .03)
                      for _ in range(LAYERS)]
        self.resident = torch.empty_like(self.hosts[0], device='cuda')
        self.arena = torch.empty((ARENA_PAGES, 1, PAGE, 512),
                                 dtype=torch.bfloat16, device='cuda')
        self.ref_arena = torch.empty_like(self.arena)
        self.readback = torch.empty((ARENA_ROWS, 512), dtype=torch.bfloat16,
                                    device='cpu', pin_memory=True)
        self.scratch = torch.empty((32,), dtype=torch.int32, device='cuda')
        self.scratch_pin = torch.empty((32,), dtype=torch.int32, device='cpu',
                                       pin_memory=True)
        self.rows = torch.arange(QUERY_ROWS, dtype=torch.int64, device='cuda')
        self.seed = torch.randn((QUERY_ROWS, QUERY_HEADS, HEAD_DIM),
                                dtype=torch.bfloat16, device='cuda') * .03
        self.logical = {}
        self.tables = {}
        self.layer_index = 0
        self.max_union_bytes = 0

    def selection(self, case):
        name = case['name']
        if name not in self.logical:
            flat, counts = rows_matrix(case['rows'])
            self.logical[name] = self.torch.tensor(
                flat, dtype=self.torch.int32,
                device='cuda').reshape(QUERY_ROWS, TOPK)
            counts = self.torch.tensor(
                counts, dtype=self.torch.int32,
                device='cuda').reshape(QUERY_ROWS, 1)
            self.logical[name] = self.torch.cat(
                (self.logical[name], counts), dim=1)
            self.tables[name] = self.torch.tensor(
                case['mapping'], dtype=self.torch.int32, device='cuda')
        return self.logical[name], self.tables[name]

    def operation(self, txn, mapping):
        provider = lambda _name, pages: tuple(mapping[:pages])  # noqa: E731
        return qo.QsaPrefillOperation(txn, lane(), device=self.device,
                                      plan_provider=provider)

    def flat(self, host):
        return host.view(-1, 512)

    def expected_sparse_rows(self, case, plan, bucket):
        host = self.flat(self.hosts[self.layer_index])
        tokens = plan.selected[bucket.compact_start:bucket.compact_end]
        ids = [case['mapping'][token // PAGE] * PAGE + token % PAGE
               for token in tokens]
        return self.torch.index_select(
            host, 0, self.torch.tensor(ids, dtype=self.torch.int64))

    def expected_dense_rows(self, case, bucket):
        host = self.flat(self.hosts[self.layer_index])
        ids = [case['mapping'][page] * PAGE + row
               for page in range(bucket.lo, bucket.hi)
               for row in range(PAGE)]
        return self.torch.index_select(
            host, 0, self.torch.tensor(ids, dtype=self.torch.int64))

    def verify_bucket(self, case, route, plan, bucket):
        rows = bucket.rows if route == 'sparse_dma' else (bucket.hi - bucket.lo) * PAGE
        expected = (self.expected_sparse_rows(case, plan, bucket)
                    if route == 'sparse_dma' else self.expected_dense_rows(case, bucket))
        self.readback[:rows].copy_(self.flat(self.arena)[:rows])
        return bool(self.torch.equal(self.readback[:rows].view(self.torch.uint8),
                                     expected.view(self.torch.uint8)))

    def reference(self, case, layer, out):
        host = self.hosts[layer]
        self.resident.copy_(host, non_blocking=True)
        logical, table = self.selection(case)
        self.module.qsa_sparse_paged_attention_staged(
            self.seed, self.resident, logical, table, self.rows,
            MAX_LOGICAL_PAGE, self.ref_arena, True, out)
        self.torch.cuda.synchronize()


def _delayed_checks(probe, operation, name, advance_s, stats):
    """Host-side proof that the selection submit did not block.

    ``advance_s`` is the time spent in the ``advance`` that staged the
    readback and yielded its pending sentinel.  Nothing here synchronizes and
    no pinned selection byte is read: the event is only queried.
    """
    torch = probe.torch
    stats['pending_selection_advances'] += 1
    if stats.get('first_advance_s') is None:
        stats['first_advance_s'] = advance_s
    sparse = operation._layers[name].sparse
    stats['event_pending'] = bool(
        stats['event_pending'] or not probe.device.query(sparse.selection.event))
    try:
        operation.sparse_selection_count(sparse.selection_token)
    except qo.OperationError:
        stats['selection_count_read_refused'] = True
    if not operation.sparse_selection_ready(sparse.selection_token):
        t0 = time.perf_counter()
        # Independent work on both bound streams while the D2H event is
        # pending: a submit must return, not wait for the selection.
        probe.scratch.zero_()
        with torch.cuda.stream(probe.copy):
            probe.scratch_pin[:8].copy_(probe.scratch[:8], non_blocking=True)
        if stats.get('independent_submit_s') is None:
            stats['independent_submit_s'] = time.perf_counter() - t0
        stats['independent_submitted'] = True


def run_layer(probe, case, layer, delay_cycles=0):
    """One serial layer: stock reference, then the real E3 operation."""
    torch = probe.torch
    name = 'qsa.%d' % layer
    probe.layer_index = layer
    host = probe.hosts[layer]
    mapping = case['mapping']
    logical, table = probe.selection(case)
    txn = transaction(layer)
    allocator = ProbeAllocator()
    hold = make_hold(lane(), txn, name, mapping, allocator)
    operation = probe.operation(txn, mapping)
    ref = torch.full_like(probe.seed, -7)
    actual = torch.full_like(probe.seed, -7)
    probe.reference(case, layer, ref)
    record: dict = dict(name=case['name'], layer=layer, route=expected_route(case),
                  selection_count=len(selection_union(case)),
                  selected_sorted=None, descriptors=expected_descriptors(case),
                  buckets=expected_buckets(case), reference_calls=1,
                  candidate_calls=1, bytes_exact=True, output_exact=False,
                  output_finite=False, completed=False, hold_released=False)
    delayed: dict = dict(case=case['name'], layer=layer,
                   pending_selection_advances=0, first_advance_s=None,
                   selection_ready_s=None, event_pending=False,
                   selection_count_read_refused=False,
                   independent_submitted=False, independent_submit_s=None)

    with torch.cuda.stream(probe.compute):
        probe.arena.fill_(-3)
        if delay_cycles:
            torch.cuda._sleep(delay_cycles)
        operation.register_source(name, host, versions=VERSIONS)
        operation.begin_forward()
        operation.begin_layer(name)
        operation.producer_submitted(name, host)
        operation.bind_plan(name, table, NUM_PAGES, VALID_TOKENS, probe.arena,
                            versions=VERSIONS)
        operation.attach_sparse_source(name, hold)
        operation.bind_sparse_context(name, probe.arena, table,
                                      min_pages=probe.arena.shape[0])
        generator = probe.namespace['flash_s5_sparse_arena_attention'](
            probe.seed, probe.resident, logical, table, probe.rows,
            MAX_LOGICAL_PAGE, probe.arena, True, actual, e3op=operation)
        operation.attach_continuation(generator)

        submitted = []
        route = None
        started = time.perf_counter()
        while True:
            began = time.perf_counter()
            try:
                item = operation.advance()
            except StopIteration:
                break
            except et.E3Pending:
                if time.perf_counter() - started > 300:
                    raise RuntimeError('selection_consumer_deadline')
                time.sleep(.0005)
                continue
            if isinstance(item, qo.SparseSelectionPending):
                if delay_cycles:
                    _delayed_checks(probe, operation, name,
                                    time.perf_counter() - began, delayed)
                continue
            submitted.append(item)
            run = operation._layers[name]
            if route is None:
                route = 'sparse_dma' if run.sparse.plan is not None else 'dense_dma'
                record['route'] = route
            if delay_cycles and delayed['selection_ready_s'] is None:
                delayed['selection_ready_s'] = time.perf_counter() - started
            torch.cuda.synchronize()
            plan = run.sparse.plan if route == 'sparse_dma' else None
            if route == 'sparse_dma':
                if plan is None:
                    raise AssertionError('sparse route without a loaded plan')
                bucket = plan.buckets[len(submitted) - 1]
            else:
                bucket = run.buckets[len(submitted) - 1]
            record['bytes_exact'] &= probe.verify_bucket(
                case, route, plan, bucket)
            record['buckets'] = len(plan.buckets) if plan is not None else len(run.buckets)
        operation.end_layer(name)
        operation.end_forward()
        while not operation.drain():
            if time.perf_counter() - started > 300:
                raise RuntimeError('operation_drain_deadline')
            time.sleep(.001)
        operation.release()

    run = operation._layers[name]
    if route == 'sparse_dma':
        record['selected_sorted'] = run.sparse.plan.selected == selection_union(case)
        record['descriptors'] = run.sparse.plan.descriptors
    record['completed'] = bool(operation.cancelled is None and not operation._failed)
    record['hold_released'] = bool(hold.released)
    record['union_scratch_bytes'] = operation.stats['union_scratch_bytes']
    record['output_exact'] = bool(torch.equal(ref.view(torch.uint8),
                                              actual.view(torch.uint8)))
    record['output_finite'] = bool(torch.isfinite(ref).all().item()
                                   and torch.isfinite(actual).all().item())
    return record, delayed, operation, hold, allocator, ref, actual


def run_cancellation(probe, case, layer, outdir):
    """Fill one bucket, cancel, drain, then reuse the same arena."""
    torch = probe.torch
    name = 'qsa.%d' % layer
    probe.layer_index = layer
    host = probe.hosts[layer]
    mapping = case['mapping']
    logical, table = probe.selection(case)
    plan = expected_plan(case)
    ref = torch.full_like(probe.seed, -7)
    probe.reference(case, layer, ref)
    record: dict = dict(layer=layer, planned_buckets=len(plan.buckets),
                  cancelled=False, bytes_exact=False, tail_poison=False,
                  no_partial_publish=False, advance_refused=False,
                  drain_ok=False, released=False, reuse_refused=False,
                  refill_output_exact=False, refill_completed=False)
    txn = transaction(100 + layer)
    allocator = ProbeAllocator()
    hold = make_hold(lane(), txn, name, mapping, allocator)
    operation = probe.operation(txn, mapping)
    actual = torch.full_like(probe.seed, -7)
    started = time.perf_counter()
    with torch.cuda.stream(probe.compute):
        probe.arena.fill_(-3)
        operation.register_source(name, host, versions=VERSIONS)
        operation.begin_forward()
        operation.begin_layer(name)
        operation.producer_submitted(name, host)
        operation.bind_plan(name, table, NUM_PAGES, VALID_TOKENS, probe.arena,
                            versions=VERSIONS)
        operation.attach_sparse_source(name, hold)
        operation.bind_sparse_context(name, probe.arena, table,
                                      min_pages=probe.arena.shape[0])
        generator = probe.namespace['flash_s5_sparse_arena_attention'](
            probe.seed, probe.resident, logical, table, probe.rows,
            MAX_LOGICAL_PAGE, probe.arena, True, actual, e3op=operation)
        operation.attach_continuation(generator)
        while True:
            try:
                item = operation.advance()
            except StopIteration:
                raise AssertionError('cancellation case finished before cancel')
            except et.E3Pending:
                time.sleep(.0005)
                continue
            if not isinstance(item, qo.SparseSelectionPending):
                break
        operation.cancel('vmid_job12_cancel')
        record['cancelled'] = True
        try:
            operation.advance()
        except qo.OperationCancelled:
            record['advance_refused'] = True
        torch.cuda.synchronize()
        record['bytes_exact'] = probe.verify_bucket(
            case, 'sparse_dma', plan, plan.buckets[0])
        probe.readback.copy_(probe.flat(probe.arena))
        poison = torch.full((ARENA_ROWS, 512), -3, dtype=torch.bfloat16)
        record['tail_poison'] = bool(torch.equal(
            probe.readback[plan.buckets[0].rows:].view(torch.uint8),
            poison[plan.buckets[0].rows:].view(torch.uint8)))
        while not operation.drain():
            if time.perf_counter() - started > 300:
                raise RuntimeError('cancel_drain_deadline')
            time.sleep(.001)
        record['drain_ok'] = True
        operation.release()
        record['released'] = bool(hold.released and not allocator.refs)
        try:
            hold.begin_bucket(99, 1, 1, shd.ROW_BYTES)
        except shd.SourceRefused:
            record['reuse_refused'] = True
    run = operation._layers[name]
    record['no_partial_publish'] = bool(
        operation.stats['sparse_buckets'] == 1
        and operation.stats['sparse_pieces'] == plan.buckets[0].descriptors
        and hold.stats['published'] == 1 and hold.stats['abandoned'] == 0
        and hold.stats['consumed'] == 1
        and run.sparse.next_bucket == 1)
    # Reuse the SAME arena with a fresh operation and hold after drain.
    refill = run_layer(probe, case, layer)
    record['refill_output_exact'] = refill[0]['output_exact']
    record['refill_completed'] = refill[0]['completed']
    s.write(outdir / ('cancel-layer%d.json' % layer), record)
    return record


def job4_regression(probe, outdir):
    """The serial gather-only job4 candidate still matches the stock oracle."""
    import vmid_job4 as job4
    torch = probe.torch
    case = next(item for item in CASES if item['name'] == 'page_edges')
    selected = selection_union(case)
    host = probe.hosts[0]
    probe.layer_index = 0
    probe.resident.copy_(host, non_blocking=True)
    logical, table = probe.selection(case)
    arena = torch.empty((ARENA_PAGES, 1, PAGE, 512), dtype=torch.bfloat16,
                        device='cuda')
    readback = torch.empty((ARENA_ROWS, 512), dtype=torch.bfloat16,
                           device='cpu', pin_memory=True)
    copy_stream = torch.cuda.Stream()
    plan = sp.make_plan(selected, tuple(case['mapping']), VERSIONS, 1)
    indices = job4.selection(torch, selected)
    ref = torch.full_like(probe.seed, -7)
    actual = torch.full_like(probe.seed, -7)
    probe.module.qsa_sparse_paged_attention_staged(
        probe.seed, probe.resident, indices, table, probe.rows,
        MAX_LOGICAL_PAGE, arena, True, ref)
    torch.cuda.synchronize()
    namespace = dict(probe.module.__dict__)
    exec(compile(qd.generate(source_path().read_text()),
                 '<vm<vm-id>-job12-job4-regression>', 'exec'), namespace)
    fill = job4.Fill(torch, host, arena, readback, plan, copy_stream)
    job4.attention(torch, probe.module, namespace, fill, probe.seed,
                   probe.resident, indices, table, probe.rows, arena, actual)
    torch.cuda.synchronize()
    record = dict(case=case['name'], reference_calls=1, candidate_calls=1,
                  pieces=fill.pieces, kernels=fill.kernels,
                  bytes_exact=bool(fill.bytes_exact),
                  tail_unchanged=bool(fill.tail),
                  completed=fill.lease.state == 'free',
                  output_exact=bool(torch.equal(ref.view(torch.uint8),
                                                actual.view(torch.uint8))))
    s.write(outdir / 'job4-regression.json', record)
    return record


# --- grading -----------------------------------------------------------------

def grade(r):
    fails = list(r.get('failures', []))
    missing = list(r.get('missing', []))
    if r.get('schema') != 'vm<vm-id>-job12/1':
        missing.append('schema')
    if r.get('source_sha256') != qd.SOURCE_SHA:
        missing.append('source_pin')
    if r.get('sparse_stage_sha256') != operation_source.SPARSE_STAGE_SHA256:
        missing.append('sparse_stage_pin')
    if r.get('union_source_sha256') != operation_source.UNION_SOURCE_SHA256:
        missing.append('union_source_pin')
    if r.get('memory_fraction') != .125:
        fails.append('memory_fraction')
    if r.get('geometry') != dict(
            layers=LAYERS, query_rows=QUERY_ROWS, query_heads=QUERY_HEADS,
            head_dim=HEAD_DIM, topk=TOPK, page_rows=PAGE,
            arena_pages=ARENA_PAGES, piece_pages=shd.MAX_PIECE_PAGES,
            descriptor_cap=shd.MAX_DESCRIPTORS, capacity=CAPACITY,
            map_pages=MAP_PAGES):
        missing.append('geometry')
    expected = [(case['name'], layer) for case in CASES for layer in range(LAYERS)]
    got = r.get('cases', [])
    if [(item.get('name'), item.get('layer')) for item in got] != expected:
        missing.append('case_layer_coverage')
    specs = {case['name']: case for case in CASES}
    for item in got:
        label = '%s:layer%s' % (item.get('name'), item.get('layer'))
        case = specs.get(item.get('name'))
        if case is None:
            fails.append(label + ':unknown_case')
            continue
        route = expected_route(case)
        plan = expected_plan(case)
        if item.get('route') != route:
            fails.append(label + ':route')
        if item.get('selection_count') != len(selection_union(case)):
            fails.append(label + ':selection_count')
        if item.get('buckets') != expected_buckets(case):
            fails.append(label + ':buckets')
        if item.get('reference_calls') != 1 or item.get('candidate_calls') != 1:
            fails.append(label + ':attention_calls')
        if route == 'sparse_dma':
            if item.get('selected_sorted') is not True:
                fails.append(label + ':sorted_union')
            if item.get('descriptors') != plan.descriptors:
                fails.append(label + ':descriptors')
        elif item.get('selected_sorted') is not None:
            fails.append(label + ':dense_claims_plan')
        for key in ('bytes_exact', 'output_exact', 'output_finite',
                    'completed', 'hold_released'):
            if item.get(key) is False:
                fails.append(label + ':' + key)
            elif item.get(key) is not True:
                missing.append(label + ':' + key)
    delayed = r.get('delayed')
    if not isinstance(delayed, dict):
        missing.append('delayed_selection')
    else:
        if delayed.get('case') != 'delayed_selection' or delayed.get('layer') != 0:
            missing.append('delayed_case')
        if delayed.get('pending_selection_advances', 0) < 1:
            missing.append('selection_pending_not_observed')
        if delayed.get('event_pending') is not True:
            fails.append('selection_event_not_pending')
        if delayed.get('selection_count_read_refused') is not True:
            fails.append('selection_count_read_early')
        if delayed.get('independent_submitted') is not True:
            fails.append('nonblocking_submit_missing')
        for key in ('first_advance_s', 'independent_submit_s'):
            value = delayed.get(key)
            if (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value >= 0):
                if value > .1:
                    fails.append('delayed_' + key + '_blocked')
            else:
                missing.append('delayed_' + key)
        value = delayed.get('selection_ready_s')
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value >= 0):
            if value < .1:
                fails.append('selection_not_delayed')
        else:
            missing.append('delayed_selection_ready_s')
        for key in ('output_exact', 'completed'):
            if delayed.get(key) is not True:
                missing.append('delayed_' + key)
    cancel = r.get('cancellation', [])
    if [item.get('layer') for item in cancel] != list(range(LAYERS)):
        missing.append('cancellation_layers')
    for item in cancel:
        label = 'cancel%s' % item.get('layer')
        if item.get('planned_buckets') != 2:
            fails.append(label + ':planned_buckets')
        for key in ('cancelled', 'bytes_exact', 'tail_poison',
                    'no_partial_publish', 'advance_refused', 'drain_ok',
                    'released', 'reuse_refused', 'refill_output_exact',
                    'refill_completed'):
            if item.get(key) is False:
                fails.append(label + ':' + key)
            elif item.get(key) is not True:
                missing.append(label + ':' + key)
    regression = r.get('job4_regression')
    if not isinstance(regression, dict):
        missing.append('job4_regression')
    else:
        if regression.get('case') != 'page_edges':
            missing.append('job4_regression_case')
        if regression.get('reference_calls') != 1 or regression.get('candidate_calls') != 1:
            fails.append('job4_regression_calls')
        for key in ('bytes_exact', 'tail_unchanged', 'completed', 'output_exact'):
            if regression.get(key) is False:
                fails.append('job4_regression:' + key)
            elif regression.get(key) is not True:
                missing.append('job4_regression:' + key)
    peaks = (r.get('allocator_peak_allocated'), r.get('allocator_peak_reserved'))
    if any(type(value) is not int or value < 0 for value in peaks):
        missing.append('allocator_peaks')
    elif not 0 < peaks[0] <= peaks[1] <= 1024 ** 3:
        fails.append('allocator_growth')
    if r.get('distinct_host_layers') is not True:
        missing.append('independent_layers')
    if r.get('selection_bytes') != SELECTION_BYTES:
        missing.append('selection_scratch_bound')
    union_bytes = r.get('union_scratch_bytes')
    if type(union_bytes) is not int or union_bytes <= 0:
        missing.append('union_scratch_bytes')
    elif union_bytes > qo.MAX_UNION_SCRATCH_BYTES:
        fails.append('union_scratch_bound')
    if r.get('cleanup_ok') is False:
        fails.append('cleanup')
    elif r.get('cleanup_ok') is not True:
        missing.append('cleanup')
    return s.verdict(fails, missing)


def probe(outdir):
    result: dict = dict(
        schema='vm<vm-id>-job12/1', image=s.IMAGE, source_sha256=qd.SOURCE_SHA,
        sparse_stage_sha256=operation_source.SPARSE_STAGE_SHA256,
        union_source_sha256=operation_source.UNION_SOURCE_SHA256,
        memory_fraction=.125, cases=[], cancellation=[], delayed=None,
        job4_regression=None, selection_bytes=SELECTION_BYTES,
        union_scratch_bytes=0,
        geometry=dict(layers=LAYERS, query_rows=QUERY_ROWS,
                      query_heads=QUERY_HEADS, head_dim=HEAD_DIM, topk=TOPK,
                      page_rows=PAGE, arena_pages=ARENA_PAGES,
                      piece_pages=shd.MAX_PIECE_PAGES,
                      descriptor_cap=shd.MAX_DESCRIPTORS,
                      capacity=CAPACITY, map_pages=MAP_PAGES),
        scope='one serial QSA-only synthetic lane: pinned staged attention vs '
              'fixed-capacity native union/remap plus compact-row CPU DMA; no '
              'engine, no model, no overlap claim, no speed claim',
        failures=[], missing=[], started_wall_s=time.time())
    torch = None
    try:
        import torch
        if (os.environ.get('VLLM_QSA_KVO_SPARSE_PREFILL') != '1'
                or os.environ.get('QSA_STAGE_TIMING') != '0'):
            raise RuntimeError('declared_sparse_mode_required')
        module, namespace = load_attention()
        probe_ = Probe(torch, module, namespace, outdir)
        result['pinned_host_bytes'] = (
            sum(host.numel() * host.element_size() for host in probe_.hosts)
            + probe_.readback.numel() * 2)
        result['arena_bytes'] = probe_.arena.numel() * 2
        result['reference_arena_bytes'] = probe_.ref_arena.numel() * 2
        result['resident_reference_bytes'] = probe_.resident.numel() * 2
        result['distinct_host_layers'] = (
            len({host.data_ptr() for host in probe_.hosts}) == LAYERS
            and all(host.is_pinned() for host in probe_.hosts))
        time.sleep(2)
        for case in CASES:
            for layer in range(LAYERS):
                delay = DELAY_CYCLES if (case['name'] == 'delayed_selection'
                                         and layer == 0) else 0
                record, delayed, _op, _hold, _alloc, _ref, _actual = run_layer(
                    probe_, case, layer, delay_cycles=delay)
                result['union_scratch_bytes'] = max(
                    result['union_scratch_bytes'],
                    record.get('union_scratch_bytes', 0))
                if delay:
                    delayed.update(output_exact=record['output_exact'],
                                   completed=record['completed'])
                    result['delayed'] = delayed
                result['cases'].append(record)
                s.write(outdir / ('%s-layer%d.json'
                                  % (case['name'], layer)), record)
        for layer in range(LAYERS):
            result['cancellation'].append(run_cancellation(
                probe_, CASES[7], layer, outdir))
        result['job4_regression'] = job4_regression(probe_, outdir)
        result['allocator_peak_allocated'] = torch.cuda.max_memory_allocated(0)
        result['allocator_peak_reserved'] = torch.cuda.max_memory_reserved(0)
        time.sleep(1)
    except Exception as exc:
        result['error_type'] = type(exc).__name__
        if isinstance(exc, AssertionError) or (torch is not None
                                               and isinstance(exc, torch.cuda.OutOfMemoryError)):
            result['failures'].append('oracle_or_allocation_failure')
        else:
            result['missing'].append('probe_exception')
        import traceback
        result['error_stack'] = [
            dict(file=Path(item.filename).name, line=item.lineno,
                 function=item.name)
            for item in traceback.extract_tb(exc.__traceback__)]
    finally:
        try:
            if torch is not None:
                torch.cuda.synchronize()
            result['cleanup_ok'] = True
        except Exception:
            result['cleanup_ok'] = False
        result['ended_wall_s'] = time.time()
        result['grade'] = grade(result)
        s.write(outdir / 'probe.json', result)
    return result['grade']['rc']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('probe', 'monitor', 'finalize'),
                        required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--process-rc', type=int, default=2)
    args = parser.parse_args()
    if args.mode == 'monitor':
        return s.monitor(args.out, 'flashnext-job12')
    if args.mode == 'finalize':
        return s.finalize(args.out, args.process_rc, grade, 'job12')
    return probe(args.out)


if __name__ == '__main__':
    sys.exit(main())
