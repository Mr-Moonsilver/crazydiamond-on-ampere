#!/usr/bin/env python3
"""VM<vm-id> job14b: real native-source binding + the actual row-1 byte oracle.

Offline scaffold for a single-GPU, serially driven qualification.  Nothing in
this file was executed while it was authored; every tensor, event and stream is
constructed only inside :func:`probe`.  The CPU acceptance tests exercise the
real ``NativeHostSourceAllocator`` / ``LocalSourceRegistry`` /
``SparseSourceHold`` objects and this module's immutable grading rules.

What the probe intentionally does NOT do:

* it never imports or constructs a vLLM engine, model, scheduler or LLM;
* it never uses ``ProbeAllocator`` or a ``released=True`` flag as source
  lifetime evidence -- the holds come from the real allocator and registry;
* it never allocates a second history buffer, never allocates a whole model
  history per rank and never treats a synthetic frame as scheduler proof.

Entry points::

    probe(modules, native, out)   # root supplies the loaded helper/native bundle
    grade(record)                 # pure, deterministic verdict over the record

``modules`` is what :func:`load_modules` returns (or the root's compiled
helper directory): the generated sparse *layer* helper, the generated ops
module and the pinned source module.  ``native`` is what :func:`load_native`
returns (or the root's checksummed bundle): the native source modules and, when
the platform can create it, ``make_uva_view(source, device)``.

The sparse layer helper is the only code that calls ``bind_row1_uva`` /
``bind_row1_reference`` / ``row1_output``; the probe therefore drives the
generated ``_flash_s5_sparse_staged_operation`` instead of the bare ops kernel.
"""
import argparse
import ast
from dataclasses import replace
import hashlib
import importlib.util
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
_BUNDLE = HERE / 'e3_builder'
_S5 = _BUNDLE if _BUNDLE.is_dir() else HERE.parent / 'patches' / 's5_engine'
for _path in (str(_S5), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import arena_attention  # noqa: E402
import e3_types as et  # noqa: E402
import native_resources  # noqa: E402
import native_source as ns  # noqa: E402
import operation_source  # noqa: E402
import qsa_operation as qo  # noqa: E402
import row1_oracle as ro  # noqa: E402
import exactness as ex  # noqa: E402
import runner_hooks  # noqa: E402
import sparse_history_dma as shd  # noqa: E402
import wire_v1 as wire  # noqa: E402
import worker_bridge as wb  # noqa: E402
import vmid_job12c as job12c  # noqa: E402  (real-file loader/plan helpers)
import vmid_support as s  # noqa: E402

SCHEMA = 'vm<vm-id>-job14b/1'

# Bounded synthetic geometry.  50 scheduler pages puts the native sparse branch
# above the 24-page arena threshold while keeping one query and one out tensor
# per layer and one 51-page pinned history per layer inside the VM12c envelope.
LAYERS = 4
PAGE = shd.PAGE_ROWS
ARENA_PAGES = shd.ARENA_PAGES
ARENA_ROWS = shd.ARENA_ROWS
NUM_PAGES = 50
MAP_PAGES = NUM_PAGES + 1
PAGES = tuple(range(1, NUM_PAGES + 1))
TOKENS = NUM_PAGES * PAGE
CAPACITY = TOKENS
QUERY_HEADS = 12
QUERY_ROWS = 100
TOPK = 2051
HEAD_DIM = 256
SPARSE_SELECTION = 1000
DENSE_SELECTION = TOKENS
MEMORY_FRACTION = .125
OWN_GPU_BOUND = 1 << 30
HOST_BOUND = 1 << 30

DENSE_CASE = 'dense_full_frontier'
SPARSE_CASE = 'sparse_natural'
CASES = (DENSE_CASE, SPARSE_CASE)
ROUTES = {DENSE_CASE: 'dense_dma', SPARSE_CASE: 'sparse_dma'}
SELECTION_COUNT = {DENSE_CASE: DENSE_SELECTION, SPARSE_CASE: SPARSE_SELECTION}
REFUSAL_CAUSES = {
    'stale_epoch': 'state epoch differs',
    'foreign_request': 'transaction differs',
    'changed_physical_page': 'physical source map differs',
    'premature_release': 'already released',
}
_REFERENCE = HERE / 'e3_reference'
if not _REFERENCE.is_dir():
    _REFERENCE = HERE.parent / 'reference' / 'vllm-src'
LAYER_SOURCE = _REFERENCE / 'models/qwen4_exp/nvidia/qsa.py'
TORCH_UTILS_SHA = 'eec8b1e3c7c673d3d171b92530f8b07acabdad5b0e3ab07cc0e717f827af8e5a'


# --- bounded helpers ---------------------------------------------------------

def bounded_error_message(error):
    """Retain the actionable message, never a tensor/address/credential."""
    return str(error)[:512]


def bounded_error_stack(error):
    import traceback
    return [dict(file=Path(item.filename).name, function=item.name,
                 line=item.lineno)
            for item in traceback.extract_tb(error.__traceback__)][-12:]


def geometry():
    return dict(layers=LAYERS, tokens=TOKENS, num_pages=NUM_PAGES,
                map_pages=MAP_PAGES, page_rows=PAGE, row_width=ns.ROW_WIDTH,
                arena_pages=ARENA_PAGES, arena_rows=ARENA_ROWS,
                query_heads=QUERY_HEADS, query_rows=QUERY_ROWS, topk=TOPK, head_dim=HEAD_DIM,
                selection_capacity=CAPACITY,
                sparse_selection=SPARSE_SELECTION)


def selection_tokens(case):
    """The exact per-row logical token ids for one case (width 1 + count col)."""
    if case not in CASES:
        raise ValueError('unknown job14b case')
    count = SELECTION_COUNT[case]
    return tuple(range(count)) + (-1,) * (TOKENS - count)


def selection_matrix(case):
    count = SELECTION_COUNT[case]
    if count > QUERY_ROWS * TOPK:
        raise ValueError('selection exceeds fixed query geometry')
    rows = [tuple(range(lo, min(lo + TOPK, count)))
            for lo in range(0, count, TOPK)]
    values, counts = job12c.rows_matrix(rows, slots=QUERY_ROWS, width=TOPK,
                                       frontier=TOKENS)
    packed = []
    for row, count in enumerate(counts):
        packed.extend(values[row * TOPK:(row + 1) * TOPK])
        packed.append(count)
    return packed, counts


def make_uva_view(source, device):
    """The pinned engine utility, proved against the runtime's mapped pointer."""
    import ctypes
    import vllm.utils.torch_utils as utils
    if s.sha(Path(utils.__file__)) != TORCH_UTILS_SHA:
        raise RuntimeError('torch_utils_source_pin')
    if not source.is_pinned() or not source.is_contiguous():
        raise RuntimeError('uva_source_not_pinned_contiguous')
    view = utils.get_accelerator_view_from_cpu_tensor(source)
    runtime = ctypes.CDLL('libcudart.so.13')
    fn = runtime.cudaHostGetDevicePointer
    fn.argtypes = (ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint)
    fn.restype = ctypes.c_int
    mapped = ctypes.c_void_p()
    rc = int(fn(ctypes.byref(mapped), ctypes.c_void_p(source.data_ptr()), 0))
    if rc != 0:
        raise RuntimeError('cudaHostGetDevicePointer rc=' + str(rc))
    if (not view.is_cuda or str(view.device) != str(device)
            or view.shape != source.shape or view.dtype != source.dtype
            or not view.is_contiguous() or view.data_ptr() != mapped.value):
        raise RuntimeError('uva_view_not_same_mapped_storage')
    return view


def expected_route(case):
    """The operation's own integer threshold, evaluated on host-known counts."""
    count = SELECTION_COUNT[case]
    return 'sparse_dma' if count * 5 < CAPACITY * 4 else 'dense_dma'


def expected_plan(case):
    """The real sparse transport plan, built without a device."""
    if expected_route(case) != 'sparse_dma':
        return None
    selected = tuple(range(SELECTION_COUNT[case]))
    return shd.make_sparse_plan(selected, PAGES, (1,) * MAP_PAGES, 1,
                                valid_tokens=TOKENS)


def expected_buckets(case):
    if expected_route(case) == 'dense_dma':
        return len(qo.plan_buckets(PAGES))
    return len(expected_plan(case).buckets)


def expected_fills(case):
    return expected_buckets(case) if expected_route(case) == 'dense_dma' else 0


_CACHED_PINS = {}


def expected_pins():
    """Regenerate both helper digests from the pinned sources (pure, no torch)."""
    if not _CACHED_PINS:
        _CACHED_PINS['ops'] = job12c.generated_ops_sha256()
        _CACHED_PINS['layer'] = hashlib.sha256(
            operation_source.generate_layer(
                runner_hooks.qsa_staged(LAYER_SOURCE.read_text())).encode()
        ).hexdigest()
        _CACHED_PINS['layer_source'] = s.sha(LAYER_SOURCE)
    return dict(_CACHED_PINS)


# --- loading the actual generated call sites --------------------------------

def _package_modules(names):
    parent = None
    for name in names:
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            module.__package__ = name.rpartition('.')[0]
            sys.modules[name] = module
        if parent is not None:
            setattr(parent, name.rpartition('.')[2], module)
        parent = module
    return parent


def _load_file(path, text, digest, name):
    """Write inspectable source once, verify it, then load it for real."""
    if path.exists():
        if hashlib.sha256(path.read_text().encode()).hexdigest() != digest:
            raise ValueError('compiled_helper_checksum_mismatch')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    from importlib.machinery import SourceFileLoader
    loader = SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise ValueError('generated_module_spec')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    if path.read_text() != text:
        raise ValueError('generated_module_changed')
    return module


def helper_definitions(text, filename):
    tree = ast.parse(text, filename=str(filename))
    names = {operation_source.LAYER_HELPER, operation_source.SPARSE_LAYER_HELPER}
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name in names]
    if {n.name for n in definitions} != names or len(definitions) != 2:
        raise ValueError('generated_layer_helper_definitions')
    return ast.Module(body=definitions, type_ignores=[])


def install_ops_module(module):
    # Alias the actual generated module. Copying its namespace would overwrite
    # the alias module's __name__/__package__ and register the wrong sys.modules
    # key, while functions would still keep a different globals dictionary.
    _package_modules(('models', 'models.qwen4_exp', 'models.qwen4_exp.nvidia',
                      'models.qwen4_exp.nvidia.ops'))
    canonical = 'models.qwen4_exp.nvidia.ops.qsa'
    sys.modules[canonical] = module
    setattr(sys.modules['models.qwen4_exp.nvidia.ops'], 'qsa', module)
    return module


def load_modules(outdir, helper=None):
    """Load the generated ops module and the generated sparse *layer* helper.

    Both are written as real files so Triton ``@jit`` sees inspectable source.
    Root may pre-populate ``helper`` with the same generated files; their
    SHA-256 is checked either way.
    """
    import torch
    source = job12c.source_path()
    if s.sha(source) != job12c.qd.SOURCE_SHA:
        raise ValueError('unpinned_qsa_source')
    e2_ops = arena_attention.generate(source.read_text())
    if hashlib.sha256(e2_ops.encode()).hexdigest() != operation_source.E2_OPS_SHA256:
        raise ValueError('unreviewed_e2_attention_source')
    generated_ops = operation_source.generate_ops(e2_ops)
    if not LAYER_SOURCE.is_file():
        raise ValueError('pinned layer source is unavailable')
    generated_layer = operation_source.generate_layer(
        runner_hooks.qsa_staged(LAYER_SOURCE.read_text()))

    directory = Path(helper) if helper is not None else Path(outdir) / 'helpers'
    directory.mkdir(parents=True, exist_ok=True)
    ops_sha = hashlib.sha256(generated_ops.encode()).hexdigest()
    layer_sha = hashlib.sha256(generated_layer.encode()).hexdigest()
    ops_mod = _load_file(directory / 'ops_qsa_native.py', generated_ops,
                         ops_sha, 'vmid_job14b_ops')
    layer_path = directory / 'qsa_native.py'

    ops_pkg = install_ops_module(ops_mod)

    _package_modules(('models', 'models.qwen4_exp', 'models.qwen4_exp.nvidia'))
    layer_pkg = types.ModuleType('models.qwen4_exp.nvidia.qsa')
    layer_pkg.__package__ = 'models.qwen4_exp.nvidia'
    layer_pkg.__dict__.update(__name__=layer_pkg.__name__,
                              __package__=layer_pkg.__package__,
                              torch=torch)
    if layer_path.exists():
        if hashlib.sha256(layer_path.read_text().encode()).hexdigest() != layer_sha:
            raise ValueError('compiled_helper_checksum_mismatch')
        text = layer_path.read_text()
    else:
        text = generated_layer
        layer_path.parent.mkdir(parents=True, exist_ok=True)
        layer_path.write_text(text)
    # Execute only the two exact generated helper definitions. Importing the
    # full owner module would construct its unrelated model import graph; this
    # slice tests these unchanged call sites, not the full model owner class.
    exec(compile(helper_definitions(text, layer_path),
                 str(layer_path), 'exec'), layer_pkg.__dict__)
    sys.modules[layer_pkg.__name__] = layer_pkg

    from importlib.machinery import SourceFileLoader
    loader = SourceFileLoader('vmid_job14b_source', str(source))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    source_module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = source_module
    loader.exec_module(source_module)

    return dict(torch=torch, ops=ops_pkg, ops_module=ops_mod, layer=layer_pkg,
                source=source_module,
                pins=dict(
                    source_sha256=job12c.qd.SOURCE_SHA,
                    layer_source_sha256=s.sha(LAYER_SOURCE),
                    sparse_stage_sha256=operation_source.SPARSE_STAGE_SHA256,
                    union_source_sha256=operation_source.UNION_SOURCE_SHA256,
                    generated_ops=dict(source_file='helpers/ops_qsa_native.py',
                                       source_sha256=ops_sha),
                    generated_layer=dict(source_file='helpers/qsa_native.py',
                                         source_sha256=layer_sha)))


def load_native(make_uva_view=None, uva_aliased=False):
    """The native source modules the probe is required to exercise."""
    return dict(ns=ns, shd=shd, ro=ro, qo=qo, et=et, wire=wire, wb=wb,
                nr=native_resources, osrc=operation_source,
                runner_hooks=runner_hooks, support=s, job12c=job12c,
                make_uva_view=make_uva_view, uva_aliased=uva_aliased)


# --- the synthetic-but-real runner ------------------------------------------

class _HostLayer:
    """One offloaded QSA layer: its actual pinned host KV (and optional UVA)."""

    def __init__(self, name, source, uva):
        self.layer_name = name
        self._qsa_kv_offload = True
        self._qsa_host_kv = source
        self.kv_cache = uva
        self._qsa_stage_min_rows = 1
        self._qsa_stage_arena_bytes = ARENA_ROWS * shd.ROW_BYTES


class NativeRunner:
    """The exact runner surface ``verify_kv_layout``/``_qsa_layers`` prove.

    A container of tensor owners only: no model, no engine, no config beyond
    the pinned block-table/KV-group vectors the native binder reads.  One
    unrelated circular group is kept first so only the real QSA group is
    treated as transport pages.
    """

    def __init__(self, layer_names, sources, uva_views, *, unrelated=('mamba.0',)):
        if not (len(layer_names) == len(sources) == len(uva_views)):
            raise ValueError('one source and one UVA view per local layer')
        self.layer_names = tuple(layer_names)
        self.sources = dict(zip(self.layer_names, sources))
        self.layers = [_HostLayer(name, source, uva)
                       for name, source, uva in zip(self.layer_names, sources,
                                                    uva_views)]
        self.model = types.SimpleNamespace(modules=lambda: iter(self.layers))
        self.kv_cache_config = types.SimpleNamespace(kv_cache_groups=(
            types.SimpleNamespace(
                layer_names=tuple(unrelated),
                kv_cache_spec=types.SimpleNamespace(block_size=8)),
            types.SimpleNamespace(
                layer_names=self.layer_names,
                kv_cache_spec=types.SimpleNamespace(block_size=PAGE))))
        self.block_tables = types.SimpleNamespace(
            blocks_per_kv_block=(1, 1),
            block_sizes=(8, PAGE),
            kernel_block_sizes=(8, PAGE))


def make_admission(transaction, state_epoch, *, pages=PAGES):
    """A synthetic frame: labelled synthetic, never scheduler-hold evidence."""
    return wire.AdmissionRecord(
        transaction=transaction, prompt_end=transaction.chunk_end + 1,
        state_epoch=state_epoch, committed_end=transaction.chunk_start,
        block_tables=((1,), tuple(pages)), slot_mappings=((0,),),
        new_block_ids_to_zero=(), cow_pairs=(),
        input_ids=tuple(range(transaction.chunk_end - transaction.chunk_start)))


def make_descriptor(admission, *, state_epoch=None, transaction=None,
                    pages=PAGES, ranks=(0,)):
    return wire.SourceLeaseDescriptor(
        transaction or admission.transaction,
        admission.state_epoch if state_epoch is None else state_epoch,
        tuple(pages), tuple(ranks))


def make_dispatcher(runner, lane):
    return types.SimpleNamespace(operation=types.SimpleNamespace(lane=lane),
                                 runner=runner, source_registry=None,
                                 source_allocator=None)


def make_lane(epoch):
    return et.LaneId(worker_rank=0, role=et.LaneRole.PREFILL,
                     storage_epoch=epoch, context_id=0)


def make_transaction(*, txn_seq=1, generation=0, epoch=1,
                     chunk_start=TOKENS-QUERY_ROWS, chunk_end=TOKENS):
    return et.TransactionId(engine_epoch=epoch, request_uid='vm<vm-id>-job14b',
                            request_generation=generation, chunk_start=chunk_start,
                            chunk_end=chunk_end, txn_seq=txn_seq)


# --- one serial layer through the generated sparse call sites ---------------

def _plan_provider(binding):
    table = dict(binding.layer_tables)

    def provider(layer_name, num_pages):
        pages = table[layer_name]
        if len(pages) != num_pages:
            raise qo.OperationError('plan provider page count mismatch')
        return tuple(pages)
    return provider


def _epochs(probe):
    def provider():
        return dict(worker_incarnation='vm<vm-id>-job14b-worker',
                    engine_epoch=1, route_generation=probe.route_generation)
    return provider


def binding_lane(binding):
    """The lane the real registry/hold bound (never a re-created copy)."""
    return binding.holds[binding.layers[0]].owner.lane


def _selection_tensor(torch, case):
    flat, _counts = selection_matrix(case)
    return torch.tensor(flat, dtype=torch.int32,
                        device='cuda:0').reshape(QUERY_ROWS, TOPK + 1)


def _metadata(probe, frontier):
    torch = probe.torch
    return types.SimpleNamespace(
        _qsa_host_meta=((0, QUERY_ROWS), (frontier,)),
        query_start_loc=torch.tensor([0, QUERY_ROWS], dtype=torch.int32, device='cpu'),
        seq_lens=torch.tensor([frontier], dtype=torch.int32, device='cpu'),
        block_table=probe.table_row.reshape(1, NUM_PAGES))


def drive_operation(operation, *, deadline_s=300):
    """One serial advance loop; a pending sentinel is retried, never waited on."""
    started = time.perf_counter()
    selections = 0
    buckets = 0
    while True:
        try:
            item = operation.advance()
        except StopIteration:
            return selections, buckets
        except et.E3Pending:
            if time.perf_counter() - started > deadline_s:
                raise RuntimeError('operation_advance_deadline')
            time.sleep(.0005)
            continue
        if isinstance(item, qo.SparseSelectionPending):
            selections += 1
            continue
        buckets += 1


def _make_operation(probe, binding, lane):
    operation = qo.QsaPrefillOperation(
        binding.transaction, lane, device=probe.device,
        plan_provider=_plan_provider(binding))
    operation.state_epoch = binding.state_epoch
    operation.qualification = ro.Row1Qualifier(
        ro.Row1RecordStore(capacity=64), transaction=binding.transaction,
        lane=lane, backend=ex.TorchExactBackend(), epochs=_epochs(probe))
    return operation


def _begin(probe, operation, binding, layer):
    name = probe.runner.layer_names[layer]
    source = binding.sources[name]
    operation.register_source(name, source,
                              source_epoch=binding.source_epochs[name],
                              versions=binding.versions[name])
    operation.begin_forward()
    operation.begin_layer(name, source)
    operation.producer_submitted(name, source)
    operation.attach_sparse_source(name, binding.holds[name])
    return name


def make_continuation(probe, operation, layer, case, uva, output):
    return probe.modules['layer']._flash_s5_sparse_staged_operation(
        operation, probe.runner.layers[layer], probe.query, uva,
        _selection_tensor(probe.torch, case),
        _metadata(probe, operation.transaction.chunk_end), QUERY_ROWS, True,
        output)


def collect_row1(store):
    """Consume the producer's actual flat as_dict schema, without re-keying."""
    records = [entry.as_dict() for entry in store.records()]
    layers = [entry.as_dict() for entry in store.layer_records()]
    failures = [entry.as_dict() for entry in store.failures()]
    buckets = [row for row in records if row.get('operation') in ro._BUCKET_OPERATIONS]
    outputs = [row for row in records if row.get('operation') == ro._OUTPUT_OPERATION]
    return records, layers, failures, buckets, outputs


def run_layer(probe, case, layer, binding, *, tag):
    """Drive one generated sparse layer and compare bytes/output bitwise."""
    torch = probe.torch
    lane = binding_lane(binding)
    uva = probe.uva_for(layer)
    output = torch.full_like(probe.query, -7)
    with torch.cuda.stream(probe.compute):
        operation = _make_operation(probe, binding, lane)
        name = _begin(probe, operation, binding, layer)
        operation.attach_continuation(
            make_continuation(probe, operation, layer, case, uva, output))
        selections, advanced = drive_operation(operation)
        operation.end_layer(name)
        operation.end_forward()
        started = time.perf_counter()
        while not operation.drain():
            if time.perf_counter() - started > 300:
                raise RuntimeError('operation_drain_deadline')
            time.sleep(.001)
        operation.release()
        torch.cuda.synchronize()
        run = operation._layers[name]
        store = operation.qualification.store
        records, layers, failures, bucket_records, output_records = collect_row1(store)
        hold = binding.holds[name]
        plan = run.sparse.plan if run.sparse is not None else None
        completed = bool(operation.cancelled is None and not operation._failed
                         and operation._released)
    del uva
    return dict(
        tag=tag, name=case, layer=layer,
        route=(run.sparse.route if run.sparse is not None else None),
        selection_count=(None if run.sparse is None
                         else run.sparse.selection_count),
        buckets=(len(plan.buckets) if plan is not None else len(run.buckets)),
        expected_fills=expected_fills(case),
        descriptors=(plan.descriptors if plan is not None else None),
        selection_sentinels=selections, advanced_buckets=advanced,
        reference_calls=1, candidate_calls=1,
        row1_records=records, row1_layers=layers, row1_failures=failures,
        row1_bucket_checks=len(bucket_records),
        row1_output_checks=len(output_records),
        arena_bytes_exact=bool(bucket_records) and all(
            r['equal'] is True and r['different_bytes'] is None
            for r in bucket_records),
        output_exact=bool(output_records) and all(
            r['equal'] is True for r in output_records),
        output_finite=bool(output_records) and all(
            r['finite'] is True for r in output_records),
        completed=completed,
        consumer_queried=sum(1 for entry in run.consumed
                             if entry.event is not None and entry.complete),
        hold_released=bool(hold.released),
        allocator_live=probe.dispatcher.source_allocator.inventory()['live'],
        uva_aliased=bool(probe.uva_aliased),
        allocator='NativeHostSourceAllocator',
        probe_allocator_used=False)


# --- case, two-epoch, cancellation and refusal phases -----------------------

def build_binding(probe, transaction, state_epoch, lane):
    admission = make_admission(transaction, state_epoch)
    descriptor = make_descriptor(admission)
    dispatcher = make_dispatcher(probe.runner, lane)
    binding = ns.bind_local_sources(
        dispatcher, wb.admission_to_chunk(admission), descriptor, rank=0)
    probe.dispatcher = dispatcher
    return dispatcher, binding, admission, descriptor


def close_binding(dispatcher, binding):
    registry = dispatcher.source_registry
    for opened in binding.opened:
        if not opened.hold.released:
            opened.hold.release()
    binding.close(registry)


def run_case(probe, case, epoch, txn_seq):
    lane = make_lane(epoch)
    transaction = make_transaction(txn_seq=txn_seq)
    probe.route_generation += 1
    dispatcher, binding, _admission, _descriptor = build_binding(
        probe, transaction, epoch, lane)
    layers = []
    try:
        for layer in range(LAYERS):
            layers.append(run_layer(probe, case, layer, binding,
                                    tag='%s:layer%d' % (case, layer)))
    finally:
        close_binding(dispatcher, binding)
    inventory = dispatcher.source_allocator.inventory()
    registry_active = len(dispatcher.source_registry._active)
    return dict(
        name=case, epoch=epoch, txn_seq=transaction.txn_seq, layers=layers,
        route=ROUTES[case], locks=len(layers),
        holds_released=all(layer['hold_released'] for layer in layers),
        source_closed=(registry_active == 0 and inventory['live'] == 0
                       and all(layer['hold_released'] for layer in layers)),
        registry_active=registry_active, allocator_live=inventory['live'],
        allocator_registrations=inventory['registrations'],
        allocator_allocations=inventory['allocations'],
        probe_allocator_used=False,
        distinct_host_layers=probe.distinct_host_layers,
        same_tensor_uva=bool(probe.uva_aliased),
        uva_aliased=bool(probe.uva_aliased))


def run_epochs(probe):
    """Two consecutive chunk/storage epochs of the SAME request and layers."""
    probe.route_generation += 1
    lane1 = make_lane(3)
    txn1 = make_transaction(txn_seq=1, chunk_start=TOKENS-2*QUERY_ROWS,
                            chunk_end=TOKENS-QUERY_ROWS)
    dispatcher, binding, _a, _descriptor = build_binding(probe, txn1, 2, lane1)
    registry = dispatcher.source_registry
    allocator = dispatcher.source_allocator
    name = probe.runner.layer_names[0]
    first = run_layer(probe, SPARSE_CASE, 0, binding, tag='epoch1')
    first_registration = binding.registrations[name]
    first_inventory = allocator.inventory()
    # An active hold must refuse a newer-epoch rebind in place.
    newer_lane = make_lane(4)
    txn2 = make_transaction(txn_seq=2, chunk_start=txn1.chunk_end)
    admission2 = make_admission(txn2, 3)
    rebind_refused, rebind_cause = False, None
    dispatcher.operation.lane = newer_lane
    try:
        ns.bind_local_sources(dispatcher, wb.admission_to_chunk(admission2),
                              make_descriptor(admission2), rank=0)
    except shd.SourceRefused as exc:
        rebind_refused, rebind_cause = True, bounded_error_message(exc)
    lane_before = registry.lane
    close_binding(dispatcher, binding)
    # Released holds permit the exact newer epoch rebind on the same registry.
    second_admission = make_admission(txn2, 3)
    second = ns.bind_local_sources(
        dispatcher, wb.admission_to_chunk(second_admission),
        make_descriptor(second_admission), rank=0)
    probe.dispatcher = dispatcher
    second_registration = second.registrations[name]
    second_run = run_layer(probe, SPARSE_CASE, 0, second, tag='epoch2')
    same_registry = dispatcher.source_registry is registry
    lane_after = binding_lane(second)
    mapping_advanced = (second_registration.mapping_version
                        > first_registration.mapping_version)
    epoch_stable = second.source_epochs == binding.source_epochs
    retained = (dispatcher.source_allocator.inventory()['allocations']
                == first_inventory['allocations'])
    close_binding(dispatcher, second)
    closed = (dispatcher.source_allocator.inventory()['live'] == 0
              and len(registry._active) == 0)
    return dict(
        request_same=txn1.same_request(txn2),
        generation_reset=txn1.request_generation != txn2.request_generation,
        contiguous_chunks=txn2.chunk_start == txn1.chunk_end,
        chunk_spans=[[txn1.chunk_start,txn1.chunk_end],[txn2.chunk_start,txn2.chunk_end]],
        chunk_epochs=(dict(storage_epoch=lane1.storage_epoch, state_epoch=2,
                           txn_seq=1),
                      dict(storage_epoch=lane_after.storage_epoch,
                           state_epoch=3, txn_seq=2)),
        active_hold_rebind_refused=rebind_refused,
        active_hold_rebind_cause=rebind_cause,
        registry_same=same_registry,
        lane_advanced=(lane_before.storage_epoch < lane_after.storage_epoch
                       and lane_after == newer_lane and registry.lane == lane_after),
        mapping_version_advanced=mapping_advanced,
        source_epoch_stable=epoch_stable,
        allocation_retained=retained,
        first_completed=first['completed'],
        first_output_exact=first['output_exact'],
        second_completed=second_run['completed'],
        second_output_exact=second_run['output_exact'],
        source_closed=closed)


def run_cancellation(probe):
    """Accept one partial bucket, cancel, drain, then refill after closure."""
    torch = probe.torch
    lane = make_lane(5)
    transaction = make_transaction(txn_seq=11)
    probe.route_generation += 1
    dispatcher, binding, _admission, _descriptor = build_binding(
        probe, transaction, 4, lane)
    name = probe.runner.layer_names[0]
    uva = probe.uva_for(0)
    output = torch.full_like(probe.query, -7)
    with torch.cuda.stream(probe.compute):
        operation = _make_operation(probe, binding, lane)
        _begin(probe, operation, binding, 0)
        operation.attach_continuation(
            make_continuation(probe, operation, 0, DENSE_CASE, uva, output))
        first = None
        deadline = time.perf_counter() + 300
        while first is None:
            if time.perf_counter() > deadline:
                raise RuntimeError('cancel_advance_deadline')
            try:
                item = operation.advance()
            except et.E3Pending:
                time.sleep(.0005)
                continue
            if isinstance(item, qo.SparseSelectionPending):
                continue
            first = item
        run = operation._layers[name]
        planned = len(run.buckets)
        hold = binding.holds[name]
        started_buckets = len(run.consumed)
        # The operation owns the dense path. SparseSourceHold has no dense
        # bucket ledger, so releasing that lower-level object directly would
        # bypass the real operation gate. Test the real public owner instead.
        release_refused, release_cause = False, None
        try:
            operation.release()
        except et.E3Pending as exc:
            release_refused, release_cause = True, bounded_error_message(exc)
        operation.cancel('vm<vm-id>-job14b_cancel')
        advance_refused = False
        try:
            operation.advance()
        except qo.OperationCancelled:
            advance_refused = True
        # The accepted operation still owns its source frame.
        newer = make_admission(make_transaction(txn_seq=12), 5)
        rebind_refused, rebind_cause = False, None
        try:
            ns.bind_local_sources(dispatcher, wb.admission_to_chunk(newer),
                                  make_descriptor(newer), rank=0)
        except shd.SourceRefused as exc:
            rebind_refused, rebind_cause = True, bounded_error_message(exc)
        filling_at_cancel = sum(1 for entry in hold._buckets.values()
                                if entry['state'] == 'filling')
        torch.cuda.synchronize()
        started = time.perf_counter()
        while not operation.drain():
            if time.perf_counter() - started > 300:
                raise RuntimeError('cancel_drain_deadline')
            time.sleep(.001)
        operation.release()
        released_after_drain = bool(hold.released)
        aborted = any(record['aborted'] is not None
                      for record in [entry.as_dict()
                                     for entry in operation.qualification.store
                                     .layer_records()])
    del uva
    close_binding(dispatcher, binding)
    # The operation released the exercised layer. This binding also owns the
    # other three, unused layers; count zero only after their explicit close.
    allocator_refs_zero = dispatcher.source_allocator.inventory()['live'] == 0
    # Same request, new transaction and lane epoch: refill must succeed.
    refill_lane = make_lane(6)
    refill_txn = make_transaction(txn_seq=13)
    probe.route_generation += 1
    dispatcher2, refill_binding, _a, _d = build_binding(
        probe, refill_txn, 6, refill_lane)
    refill = run_layer(probe, DENSE_CASE, 0, refill_binding, tag='refill')
    close_binding(dispatcher2, refill_binding)
    no_partial_publish = bool(
        all(entry.event is not None and entry.complete for entry in run.consumed)
        and len(run.consumed) == started_buckets
        and run.next_bucket == started_buckets
        and run.active_bucket is None and run._pending_consumer is None
        and filling_at_cancel == 0
        and not any(entry.identity.operation == ro._OUTPUT_OPERATION
                    for entry in operation.qualification.store.records()))
    return dict(
        name='cancel_partial_dense', planned_buckets=planned,
        started_buckets=started_buckets,
        partial_fill=0 < started_buckets < planned, cancelled=True,
        advance_refused=advance_refused,
        release_refused_while_owned=release_refused,
        release_refusal_cause=release_cause,
        rebind_refused_while_owned=rebind_refused,
        rebind_refusal_cause=rebind_cause,
        filling_at_cancel=filling_at_cancel,
        publish_requires_all_pieces=True,
        no_partial_publish=no_partial_publish,
        drain_ok=True, released_after_drain=released_after_drain,
        allocator_refs_zero=allocator_refs_zero, row1_aborted=aborted,
        refill_completed=refill['completed'],
        refill_output_exact=refill['output_exact'],
        refill_route=refill['route'])


def _refusal(case, action):
    try:
        action()
    except Exception as exc:
        return dict(case=case, refused=True,
                    cause=bounded_error_message(exc),
                    error_type=type(exc).__name__)
    return dict(case=case, refused=False, cause=None, error_type=None)


def run_refusals(probe):
    """Refusals through the real allocator/registry/hold objects."""
    lane = make_lane(7)
    transaction = make_transaction(txn_seq=21)
    dispatcher, binding, admission, descriptor = build_binding(
        probe, transaction, 6, lane)
    name = probe.runner.layer_names[0]
    group = dict((row[0], row[1]) for row in binding.group_rows)[name]
    registration = binding.registrations[name]
    registry = dispatcher.source_registry
    allocator = dispatcher.source_allocator
    hold = binding.holds[name]
    results = []
    results.append(_refusal(
        'stale_epoch', lambda: registry.open(
            replace(descriptor, state_epoch=7), admission, layer=name,
            group=group, allocator=allocator, registration=registration)))
    foreign = make_transaction(txn_seq=23, generation=1)
    results.append(_refusal(
        'foreign_request', lambda: registry.open(
            replace(descriptor, transaction=foreign), admission, layer=name,
            group=group, allocator=allocator, registration=registration)))
    other = PAGES[-1] if hold.owner.physical_pages[0] != PAGES[-1] else PAGES[0]
    results.append(_refusal(
        'changed_physical_page',
        lambda: hold.validate(hold.owner, name, (other,))))
    # Premature release: the released reference must not validate again.
    reference = hold.reference
    hold.release()
    results.append(_refusal(
        'premature_release',
        lambda: hold.validate(hold.owner, name, hold.owner.physical_pages)))
    results.append(_refusal(
        'premature_reference',
        lambda: allocator.assert_retained(reference)))
    close_binding(dispatcher, binding)
    return results


# --- grading -----------------------------------------------------------------

def _truth(item, key, label, fails, missing):
    value = item.get(key)
    if value is False:
        fails.append(label + ':' + key)
    elif value is not True:
        missing.append(label + ':' + key)


def _digest_pin(value, expected, label, fails, missing):
    if not (isinstance(value, str) and len(value) == 64):
        missing.append(label)
    elif expected is None:
        missing.append(label + '_unverifiable')
    elif value != expected:
        fails.append(label + '_mismatch')


def grade(record):
    """Pure grading over one probe record.  No tensor or device is touched."""
    r = record
    fails = list(r.get('failures', []))
    missing = list(r.get('missing', []))
    try:
        pins = expected_pins()
    except Exception:
        pins = {}
    if r.get('schema') != SCHEMA:
        missing.append('schema')
    if r.get('source_sha256') != job12c.qd.SOURCE_SHA:
        missing.append('source_pin')
    if r.get('sparse_stage_sha256') != operation_source.SPARSE_STAGE_SHA256:
        missing.append('sparse_stage_pin')
    if r.get('union_source_sha256') != operation_source.UNION_SOURCE_SHA256:
        missing.append('union_source_pin')
    if pins and r.get('layer_source_sha256') != pins.get('layer_source'):
        missing.append('layer_source_pin')
    generated = r.get('generated_ops')
    if not isinstance(generated, dict):
        missing.append('generated_ops')
    else:
        if generated.get('source_file') != 'helpers/ops_qsa_native.py':
            missing.append('generated_ops')
        _digest_pin(generated.get('source_sha256'), pins.get('ops'),
                    'generated_ops', fails, missing)
    layer = r.get('generated_layer')
    if not isinstance(layer, dict):
        missing.append('generated_layer')
    else:
        if layer.get('source_file') != 'helpers/qsa_native.py':
            missing.append('generated_layer')
        _digest_pin(layer.get('source_sha256'), pins.get('layer'),
                    'generated_layer', fails, missing)
    if r.get('memory_fraction') != MEMORY_FRACTION:
        fails.append('memory_fraction')
    if r.get('geometry') != geometry():
        missing.append('geometry')
    if r.get('allocator') != 'NativeHostSourceAllocator':
        fails.append('probe_allocator_substituted')
    if r.get('probe_allocator_used') is not False:
        fails.append('probe_allocator_used')
    if r.get('uva_aliased') is not True:
        missing.append('native_uva_view')
    if r.get('distinct_host_layers') is not True:
        missing.append('independent_host_layers')

    cases = r.get('cases', [])
    if [case.get('name') for case in cases] != list(CASES):
        missing.append('case_coverage')
    for case in cases:
        label = str(case.get('name'))
        if label not in CASES:
            fails.append(label + ':unknown_case')
            continue
        if case.get('route') != ROUTES[label]:
            fails.append(label + ':route')
        if case.get('locks') != LAYERS:
            missing.append(label + ':layer_coverage')
        if [entry.get('layer') for entry in case.get('layers', [])] != list(range(LAYERS)):
            missing.append(label + ':layer_order')
        _truth(case, 'holds_released', label, fails, missing)
        _truth(case, 'source_closed', label, fails, missing)
        if case.get('probe_allocator_used') is not False:
            fails.append(label + ':probe_allocator_used')
        if case.get('allocator_live') != 0 or case.get('registry_active') != 0:
            fails.append(label + ':source_not_closed')
        for item in case.get('layers', []):
            sub = '%s:layer%s' % (label, item.get('layer'))
            if item.get('route') != ROUTES[label]:
                fails.append(sub + ':route')
            if item.get('selection_count') != SELECTION_COUNT[label]:
                fails.append(sub + ':selection_count')
            if item.get('buckets') != expected_buckets(label):
                fails.append(sub + ':buckets')
            expect_plan = expected_plan(label)
            if item.get('descriptors') != (expect_plan.descriptors
                                           if expect_plan is not None else None):
                fails.append(sub + ':descriptors')
            if item.get('expected_fills') != expected_fills(label):
                fails.append(sub + ':expected_fills')
            if item.get('reference_calls') != 1 or item.get('candidate_calls') != 1:
                fails.append(sub + ':call_counts')
            for key in ('arena_bytes_exact', 'output_exact', 'output_finite',
                        'completed', 'hold_released'):
                _truth(item, key, sub, fails, missing)
            if item.get('uva_aliased') is not True:
                missing.append(sub + ':uva_aliased')
            if item.get('consumer_queried') != item.get('buckets'):
                fails.append(sub + ':consumer_not_queried')
            if item.get('row1_bucket_checks') != item.get('buckets'):
                fails.append(sub + ':bucket_check_count')
            if item.get('row1_output_checks') != 1:
                fails.append(sub + ':output_check_count')
            if item.get('row1_failures'):
                fails.append(sub + ':row1_failure_records')
            layer_records = item.get('row1_layers', [])
            if len(layer_records) != 1 or layer_records[0].get('passed') is not True:
                fails.append(sub + ':row1_layer')
            records = item.get('row1_records', [])
            outputs = [x for x in records
                       if x.get('operation') == ro._OUTPUT_OPERATION]
            bucket_records = [x for x in records
                              if x.get('operation')
                              in ro._BUCKET_OPERATIONS]
            if len(bucket_records) != item.get('buckets'):
                missing.append(sub + ':raw_bucket_records')
            if len(outputs) != 1:
                missing.append(sub + ':raw_output_record')
            for record in bucket_records + outputs:
                if record.get('equal') is not True:
                    fails.append(sub + ':raw_equal')
                if not (isinstance(record.get('compared_bytes'), int)
                        and record['compared_bytes'] > 0):
                    missing.append(sub + ':raw_compared_bytes')
                if record.get('equal') is True and record.get('different_bytes') is not None:
                    fails.append(sub + ':raw_different_bytes')
                if not (isinstance(record.get('digest'), str)
                        and len(record['digest']) == 64):
                    missing.append(sub + ':raw_digest')
            if outputs and outputs[0].get('pieces') != expected_fills(label):
                fails.append(sub + ':reference_fills')

    epochs = r.get('epochs')
    if not isinstance(epochs, dict):
        missing.append('epochs')
    else:
        for key in ('request_same', 'contiguous_chunks', 'active_hold_rebind_refused',
                    'registry_same', 'lane_advanced', 'mapping_version_advanced',
                    'source_epoch_stable', 'allocation_retained',
                    'first_completed', 'first_output_exact', 'second_completed',
                    'second_output_exact', 'source_closed'):
            _truth(epochs, key, 'epochs', fails, missing)
        if epochs.get('generation_reset') is not False:
            fails.append('epochs:generation_reset')
        cause = str(epochs.get('active_hold_rebind_cause') or '')
        if not any(word in cause for word in ('already open', 'lane changed',
                                              'storage epoch', 'lease is open')):
            missing.append('epochs:rebind_cause')

    cancel = r.get('cancellation')
    if not isinstance(cancel, dict):
        missing.append('cancellation')
    else:
        for key in ('partial_fill', 'cancelled', 'advance_refused',
                    'release_refused_while_owned', 'rebind_refused_while_owned',
                    'publish_requires_all_pieces', 'no_partial_publish',
                    'drain_ok', 'released_after_drain', 'allocator_refs_zero',
                    'row1_aborted', 'refill_completed', 'refill_output_exact'):
            _truth(cancel, key, 'cancellation', fails, missing)
        if cancel.get('filling_at_cancel') != 0:
            fails.append('cancellation:partial_bucket_published')
        planned = cancel.get('planned_buckets') or 0
        started = cancel.get('started_buckets') or 0
        if not 0 < started < planned:
            fails.append('cancellation:not_a_partial_fill')
        if cancel.get('refill_route') != 'dense_dma':
            fails.append('cancellation:refill_route')
        for key in ('release_refusal_cause', 'rebind_refusal_cause'):
            if not cancel.get(key):
                missing.append('cancellation:' + key)

    by_case = {item.get('case'): item for item in r.get('refusals', [])
               if isinstance(item, dict)}
    for case, cause in REFUSAL_CAUSES.items():
        item = by_case.get(case)
        if item is None:
            missing.append('refusal:' + case)
            continue
        if item.get('refused') is not True:
            fails.append('refusal:' + case + ':not_refused')
        if cause not in str(item.get('cause') or ''):
            fails.append('refusal:' + case + ':wrong_cause')
    if by_case.get('premature_reference', {}).get('refused') is not True:
        missing.append('refusal:premature_reference')

    peaks = (r.get('allocator_peak_allocated'), r.get('allocator_peak_reserved'))
    if any(type(value) is not int or value < 0 for value in peaks):
        missing.append('allocator_peaks')
    elif not 0 < peaks[0] <= peaks[1] <= OWN_GPU_BOUND:
        fails.append('allocator_growth')
    pinned = r.get('pinned_host_bytes')
    if type(pinned) is not int or pinned <= 0:
        missing.append('pinned_host_bytes')
    elif pinned > HOST_BOUND:
        fails.append('pinned_host_over_bound')
    own = r.get('own_gpu_bytes')
    if type(own) is not int or own <= 0:
        missing.append('own_gpu_bytes')
    elif own > OWN_GPU_BOUND:
        fails.append('own_gpu_over_bound')
    _truth(r, 'cleanup_ok', 'probe', fails, missing)
    return s.verdict(fails, missing)


# --- probe -------------------------------------------------------------------

class Probe:
    """GPU fixture owner.  Constructed only inside ``probe``."""

    def __init__(self, modules, native, out):
        self.modules = modules
        self.native = native
        self.torch = modules['torch']
        self.ns, self.shd = native['ns'], native['shd']
        self.ro, self.qo = native['ro'], native['qo']
        self.wire, self.et = native['wire'], native['et']
        torch = self.torch
        torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, 0)
        torch.set_num_threads(4)
        torch.manual_seed(14014)
        if torch.cuda.device_count() != 1:
            raise RuntimeError('one_device_required')
        if torch.cuda.get_device_capability(0) != (8, 6):
            raise RuntimeError('one_sm86_device_required')
        self.device = self.qo.NativeQsaDevice('cuda:0')
        self.compute = torch.cuda.current_stream(0)
        self.copy = torch.cuda.Stream(device='cuda:0')
        self.device.bind_streams(self.compute, self.copy)
        self.sources = [
            torch.empty((MAP_PAGES, 1, PAGE, self.ns.ROW_WIDTH),
                        dtype=torch.bfloat16, device='cpu',
                        pin_memory=True).normal_(0, .03)
            for _ in range(LAYERS)]
        self.distinct_host_layers = (
            len({src.data_ptr() for src in self.sources}) == LAYERS
            and all(src.is_pinned() and src.is_contiguous()
                    for src in self.sources))
        make_uva = native.get('make_uva_view')
        self.uva_aliased = bool(callable(make_uva)
                                and native.get('uva_aliased', False))
        self.uva = ([make_uva(source, 'cuda:0') for source in self.sources]
                    if self.uva_aliased else [None] * LAYERS)
        self.runner = NativeRunner(
            tuple('qsa.%d' % index for index in range(LAYERS)),
            self.sources, self.uva)
        self.arena = torch.empty((ARENA_PAGES, 1, PAGE, self.ns.ROW_WIDTH),
                                 dtype=torch.bfloat16, device='cuda:0')
        self.query = torch.randn((QUERY_ROWS, QUERY_HEADS, HEAD_DIM),
                                 dtype=torch.bfloat16, device='cuda:0') * .03
        self.table_row = torch.tensor(PAGES, dtype=torch.int32, device='cuda:0')
        self.route_generation = 0
        self.dispatcher = None
        hook = lambda kv_cache, nbytes: self.arena  # noqa: E731
        self.modules['ops'].qsa_get_staging_arena = hook
        self.modules['ops_module'].qsa_get_staging_arena = hook

    def uva_for(self, layer):
        if self.uva_aliased:
            return self.uva[layer]
        # A copied device history is not this qualification's reference.
        # The retained receipt names the missing alias; no replacement copy.
        raise RuntimeError('native_uva_view_unavailable')


def probe(modules, native, out):
    """Run the bounded serial qualification.  ``modules``/``native`` injected."""
    out = Path(out)
    pins = modules.get('pins', {})
    result = dict(
        schema=SCHEMA, image=s.IMAGE,
        source_sha256=pins.get('source_sha256', job12c.qd.SOURCE_SHA),
        layer_source_sha256=pins.get('layer_source_sha256'),
        sparse_stage_sha256=operation_source.SPARSE_STAGE_SHA256,
        union_source_sha256=operation_source.UNION_SOURCE_SHA256,
        generated_ops=pins.get('generated_ops'),
        generated_layer=pins.get('generated_layer'),
        memory_fraction=MEMORY_FRACTION, geometry=geometry(),
        allocator='NativeHostSourceAllocator', probe_allocator_used=False,
        uva_aliased=False, distinct_host_layers=False,
        uva_source_pin=TORCH_UTILS_SHA,
        uva_basis='pinned-engine-utility + cudaHostGetDevicePointer equality',
        cases=[], epochs=None, cancellation=None, refusals=[],
        scope=('one serial QSA-only synthetic lane: the actual '
               'NativeHostSourceAllocator/LocalSourceRegistry holds and the '
               'actual generated sparse/dense layer call sites; no model, no '
               'engine, no scheduler hold, no overlap or speed claim'),
        failures=[], missing=[], started_wall_s=time.time())
    fixture = None
    try:
        fixture = Probe(modules, native, out)
        result['uva_aliased'] = bool(fixture.uva_aliased)
        result['distinct_host_layers'] = bool(fixture.distinct_host_layers)
        result['pinned_host_bytes'] = int(sum(
            src.numel() * src.element_size() for src in fixture.sources))
        result['own_gpu_bytes'] = int(
            fixture.query.numel() * fixture.query.element_size()
            + fixture.arena.numel() * fixture.arena.element_size()
            + 2 * fixture.query.numel() * fixture.query.element_size()
            + (0 if fixture.uva_aliased else
               fixture.sources[0].numel() * fixture.sources[0].element_size()))
        time.sleep(2)
        for index, case in enumerate(CASES):
            result['cases'].append(run_case(
                fixture, case, epoch=2, txn_seq=1 + index))
        result['epochs'] = run_epochs(fixture)
        result['cancellation'] = run_cancellation(fixture)
        result['refusals'] = run_refusals(fixture)
        result['allocator_peak_allocated'] = int(
            fixture.torch.cuda.max_memory_allocated(0))
        result['allocator_peak_reserved'] = int(
            fixture.torch.cuda.max_memory_reserved(0))
        time.sleep(1)
    except Exception as exc:
        result['error_type'] = type(exc).__name__
        result['error_message'] = bounded_error_message(exc)
        result['error_stack'] = bounded_error_stack(exc)
        result['missing'].append('probe_exception')
    finally:
        try:
            if fixture is not None:
                fixture.torch.cuda.synchronize()
            result['cleanup_ok'] = True
        except Exception:
            result['cleanup_ok'] = False
        result['ended_wall_s'] = time.time()
        result['grade'] = grade(result)
        s.write(out / 'probe.json', result)
    return result['grade']['rc']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('probe', 'monitor', 'finalize'),
                        required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--helper', type=Path, default=None)
    parser.add_argument('--process-rc', type=int, default=2)
    args = parser.parse_args()
    if args.mode == 'monitor':
        # Root must add 'flashnext-job14b' to vmid_support.OWNED_CONTAINERS.
        return s.monitor(args.out, 'flashnext-job14b')
    if args.mode == 'finalize':
        return s.finalize(args.out, args.process_rc, grade, 'job14b')
    try:
        modules = load_modules(args.out, args.helper)
        native = load_native(make_uva_view=make_uva_view, uva_aliased=True)
        return probe(modules, native, args.out)
    except Exception as exc:
        record = dict(schema=SCHEMA, failures=[], missing=['probe_load_exception'],
                      error_type=type(exc).__name__,
                      error_message=bounded_error_message(exc),
                      error_stack=bounded_error_stack(exc))
        record['grade'] = s.verdict([], record['missing'])
        s.write(args.out / 'probe.json', record)
        return 2


if __name__ == '__main__':
    sys.exit(main())
