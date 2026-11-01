#!/usr/bin/env python3
"""Step A V3, VM only: the fused dense QSA path on one real GPU, against the stepped path.

Real objects: generated qsa helper and arena attention, compiled kernels, the shared
staging arena, QsaPrefillOperation, NativeQsaDevice, bind_local_sources and its holds,
two CUDA streams. Runtime methods are unchanged; the mutant devices are named negative
controls and never count toward a pass.

Each comparison starts from the same seeded sources, queries and selections. A fused
run takes no host wait: the only completion query is the drain after every layer has
been submitted, and the bytes are read after that drain, so the oracle cannot be the
sync that makes it pass. Delays hold one stream behind real GPU work so that any
missing stream dependency shows up as different bytes.

Declared limits: one rank, so no TP or PP collective runs here; no decode beside it;
synthetic history. No engine claim. It runs under the shared VM lock, under 3 GiB.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import e3_native_sparse_waves_device_probe as base
from e3_native_sparse_waves_device_probe import PAGE, WIDTH, TOPK, make_alignment, mapped_uva, bounded_poll

SCHEMA = 'step-a-fused-device/1'
SEED = 20261101
HEADS = 12
checks, numbers = {}, {}


def check(name, ok, **detail):
    checks[name] = dict(ok=bool(ok), **detail)
    print(('OK   ' if ok else 'FAIL ') + name, json.dumps(detail, default=str)[:300], flush=True)


class Sources:
    """The seeded pinned histories; reused across runs as the worker's histories are."""
    def __init__(self):
        self.rows = {}

    def get(self, pages, index):
        import torch
        key = (pages, index)
        if key not in self.rows:
            generator = torch.Generator().manual_seed(SEED + 1000 * pages + index)
            source = torch.empty((pages + 1, 1, PAGE, WIDTH), dtype=torch.bfloat16,
                                 device='cpu', pin_memory=True)
            source.copy_((torch.randn(source.shape, generator=generator) * .03).to(torch.bfloat16))
            self.rows[key] = (source, mapped_uva(source))
        return self.rows[key]


SOURCES = Sources()


def delay_compute(stream, seconds):
    """Real GPU work ahead of the first kernel on the compute stream."""
    import torch
    a = torch.randn((2500, 2500), device='cuda:0')
    b = a @ a
    torch.cuda.synchronize()  # Setup only, before the run's clock starts.
    start = time.monotonic()
    b = a @ a
    torch.cuda.synchronize()
    one = max(time.monotonic() - start, 1e-4)
    with torch.cuda.stream(stream):
        for _ in range(int(seconds / one) + 1):
            b = a @ a
    return a, b


def delay_copy(stream, seconds):
    """A real DMA backlog ahead of the first copy on the copy stream."""
    import torch
    host = torch.empty((256 * 1024 * 1024,), dtype=torch.uint8, device='cpu', pin_memory=True)
    target = torch.empty((256 * 1024 * 1024,), dtype=torch.uint8, device='cuda:0')
    target.copy_(host, non_blocking=True)
    torch.cuda.synchronize()  # Setup only, before the run's clock starts.
    start = time.monotonic()
    target.copy_(host, non_blocking=True)
    torch.cuda.synchronize()
    one = max(time.monotonic() - start, 1e-4)
    with torch.cuda.stream(stream):
        for _ in range(int(seconds / one) + 1):
            target.copy_(host, non_blocking=True)
    return target, host


def admitted(f, pages):
    """The admission and metadata fields of a LaneChunk. Leaf: its lane tables hold 32 pages
    in make_alignment's fixture, and this probe needs more, so the QSA row is written here."""
    import torch
    n = f.tokens
    tx = f.et.TransactionId(1, 'device-path', 1, f.frontier - n, f.frontier, 1)
    row = (3, 5, 1, 6, 2, 7, 4, 8) + tuple(range(9, 32))
    admission = f.lc.ChunkAdmission(tx, f.frontier + 1, (1,) * n, (0, 0),
                                    (row, row, tuple(range(1, pages + 1))), state_epoch=1)
    table = torch.zeros((1, pages + 8), dtype=torch.int32, device='cuda:0')
    table[0, :pages] = torch.arange(1, pages + 1, dtype=torch.int32, device='cuda:0')
    return SimpleNamespace(
        admission=admission, block_tables=(None, None, table),
        query_start_loc=torch.tensor([0, n], dtype=torch.int32, device='cuda:0'),
        query_start_loc_cpu=torch.tensor([0, n], dtype=torch.int32, device='cpu'),
        seq_lens=torch.tensor([f.frontier], dtype=torch.int32, device='cuda:0'),
        seq_lens_cpu=torch.tensor([f.frontier], dtype=torch.int32, device='cpu'))


def run(mode, *, pages, slack=0, rows=103, layers=2, delay=None, mutant=None, cancel_after=None):
    """mode: 'stepped' (advance, host credit) or 'fused' (advance_fused, no host wait)."""
    import torch
    # The model module first, as the engine imports it: qsa and model are circular.
    import vllm.models.qwen4_exp.nvidia.model  # noqa: F401
    from vllm.models.qwen4_exp.nvidia import qsa
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
    from vllm.v1.flash_e3 import lane_dispatch, native_source as ns, qsa_operation as qo, wire_v1 as wire

    frontier = PAGE * pages - slack
    base.PAGES, base.FRONTIER = pages, frontier
    modules = []
    for index in range(layers):
        source, uva = SOURCES.get(pages, index)
        layer = torch.nn.Module()
        layer.layer_name, layer._qsa_kv_offload = 'probe.qsa.%d' % index, True
        layer._qsa_host_kv, layer.kv_cache = source, uva
        layer._qsa_stage_min_rows, layer._qsa_stage_arena_bytes = qsa._QSA_STAGE_MIN_ROWS, 24 * PAGE * WIDTH * 2
        modules.append(layer)
    f = make_alignment(block_size=PAGE, frontier=frontier, tokens=rows, qsa_layer=modules[0])
    # The real engine's QSA layers share one KV group.
    for layer in modules[1:]:
        f.runner.model.append(layer)
        f.config.kv_cache_groups[-1].layer_names.append(layer.layer_name)
    chunk = admitted(f, pages)
    compute, copy = torch.cuda.current_stream(), torch.cuda.Stream()

    class Device(qo.NativeQsaDevice):
        def wait_event(self, token, event):
            if mutant == 'no-arena-order' and token == self.copy:
                return
            if mutant == 'no-copy-before-kernel' and token == self.compute:
                return
            super().wait_event(token, event)

    device = (Device if mutant else qo.NativeQsaDevice)('cuda:0')
    device.bind_streams(compute, copy)
    device.native_qsa_policy = 'dense'
    transaction = chunk.admission.transaction
    lane = f.et.LaneId(0, f.et.LaneRole.PREFILL, chunk.admission.state_epoch, 0)
    descriptor = wire.SourceLeaseDescriptor(transaction, chunk.admission.state_epoch,
                                            tuple(range(1, pages + 1)), (0,))

    def plan(layer_name, count):
        row = dict(binding.layer_tables)[layer_name]
        assert len(row) == count
        return row
    operation = qo.QsaPrefillOperation(transaction, lane, device=device, plan_provider=plan)
    operation.state_epoch = chunk.admission.state_epoch
    owner = lane_dispatch.LaneDispatcher(f.runner, None, None, torch_module=torch)
    owner.operation = operation
    binding = ns.bind_local_sources(owner, chunk.admission, descriptor, rank=0)
    assert operation.fused_refusal() is None
    helper = qsa._flash_s5_sparse_staged_operation
    for layer in modules:
        operation.register_source(layer.layer_name, binding.sources[layer.layer_name],
                                  source_epoch=binding.source_epochs[layer.layer_name],
                                  versions=binding.versions[layer.layer_name])
        # As bind_operation_sources: attached now, so a cancel releases every history.
        operation.attach_sparse_source(layer.layer_name, binding.holds[layer.layer_name])
    torch.cuda.reset_peak_memory_stats()
    # Every per-layer input is on the device before any delay: a blocking .to() would eat it.
    prepared = []
    for index, layer in enumerate(modules):
        name = layer.layer_name
        generator = torch.Generator().manual_seed(SEED + 7 * index + rows)
        query = (torch.randn((rows, HEADS, 256), generator=generator) * .03).to(torch.bfloat16).to('cuda:0')
        output = torch.full_like(query, -7)
        packed = torch.full((rows, TOPK + 1), -1, dtype=torch.int32, device='cpu')
        packed[:, -1] = 0
        for row, lo in enumerate(range(0, frontier, TOPK)):
            hi = min(lo + TOPK, frontier)
            packed[row, :hi - lo] = torch.arange(lo, hi, dtype=torch.int32)
            packed[row, -1] = hi - lo
        selection = packed.to('cuda:0')
        md = FlashAttentionMetadata(
            num_actual_tokens=rows, max_query_len=rows, query_start_loc=chunk.query_start_loc,
            max_seq_len=frontier, seq_lens=chunk.seq_lens, block_table=chunk.block_tables[2],
            slot_mapping=torch.arange(rows, dtype=torch.int64, device='cuda:0'), use_cascade=False,
            common_prefix_len=0, cu_prefix_query_lens=None, prefix_kv_lens=None, suffix_kv_lens=None)
        f.lc.attach_qsa_host_metadata({name: md}, (name,), chunk.query_start_loc_cpu, chunk.seq_lens_cpu,
                                      chunk.admission)
        prepared.append((query, output, selection, md))
    torch.cuda.synchronize()
    keep, delay_event = None, None
    if delay == 'compute':
        keep = delay_compute(compute, 1.5)
        delay_event = torch.cuda.Event()
        delay_event.record(compute)
    elif delay == 'copy':
        keep = delay_copy(copy, 1.5)
        delay_event = torch.cuda.Event()
        delay_event.record(copy)
    operation.begin_forward()
    outputs, arenas, submit_s, cancelled, buckets = [], [], 0., False, []
    last_event = None
    for index, layer in enumerate(modules):
        name = layer.layer_name
        source, uva = SOURCES.get(pages, index)
        query, output, selection, md = prepared[index]
        start = time.monotonic()
        operation.begin_layer(name, source)
        operation.producer_submitted(name, source)
        operation.attach_continuation(helper(operation, layer, query, uva, selection, md, rows, True, output))
        query_row = wire.QsaProfileQuery(transaction, chunk.admission.state_epoch, 0, index, 0, (0,))
        deadline = time.monotonic() + 180
        count = 0
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError('QSA advance deadline')
            try:
                marker = operation.advance_fused() if mode == 'fused' else operation.advance()
            except StopIteration:
                break
            except f.et.E3Pending:
                assert mode == 'stepped', 'the fused path must not need a host wait'
                time.sleep(.001)
                continue
            if getattr(marker, 'qsa_schedule_boundary', False):
                if mode == 'fused':
                    shape = operation.fused_shape(name, frontier, PAGE)
                    operation.authorize_schedule(wire.QsaProfileGrant(query_row, shape, fused=True))
                else:
                    while True:
                        shape = operation.schedule_profile()
                        if shape is not None:
                            break
                        time.sleep(.001)
                    operation.authorize_schedule(wire.QsaProfileGrant(query_row, shape))
                continue
            assert not getattr(marker, 'sparse_selection_pending', False), 'dense must not select'
            assert not getattr(marker, 'sparse_copy_pending', False), 'dense must not copy sparse waves'
            count += 1
            if cancel_after is not None and count == cancel_after:
                hold = binding.holds[name]
                consumed = operation._layers[name].consumed
                unresolved = not all(row.event.query() for row in consumed)
                delay_live = delay_event is not None and not delay_event.query()
                operation.cancel('step-a device proof')
                # The fused path refuses to continue after a cancel.
                refused = False
                try:
                    operation.advance_fused()
                except qo.OperationCancelled:
                    refused = True
                released_before_drain = hold.released
                drained_at_once = bool(operation.drain())
                bounded_poll(operation.drain, 60)
                operation.poll()
                operation.release()
                binding.close(owner.source_registry)
                return dict(cancelled=True, unresolved_at_cancel=unresolved, delay_live_at_cancel=delay_live,
                            advance_refused=refused,
                            released_before_drain=released_before_drain, drained_at_once=drained_at_once,
                            hold_released=hold.released, allocator_live=owner.source_allocator.inventory()['live'],
                            registry_active=len(owner.source_registry._active))
        submit_s += time.monotonic() - start
        run_row = operation._layers[name]
        arenas.append(id(run_row.arena) if hasattr(run_row, 'arena') else None)
        last_event = run_row.consumed[-1].event
        buckets.append(count)
        operation.end_layer(name)
        outputs.append(output)
    unresolved = not last_event.query()
    delay_live = delay_event is not None and not delay_event.query()
    submitted_at = time.monotonic()
    operation.end_forward()
    bounded_poll(operation.drain, 60)
    drain_s = time.monotonic() - submitted_at
    operation.poll()
    operation.release()
    released = [binding.holds[m.layer_name].released for m in modules]
    binding.close(owner.source_registry)
    stats = dict(operation.stats)
    cpu = [o.to(torch.float32).cpu() for o in outputs]
    del keep
    return dict(outputs=cpu, digests=[hashlib.sha256(o.numpy().tobytes()).hexdigest() for o in cpu],
                buckets=buckets, submit_s=submit_s, drain_s=drain_s, unresolved_at_submit_end=unresolved,
                delay_live_at_submit_end=delay_live,
                released=released, same_arena=len(set(arenas)) == 1, waits=stats['waits'],
                copies=stats['copy_events'], consumers=stats['consumer_events'], bytes=stats['bytes'],
                peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
                allocator_live=owner.source_allocator.inventory()['live'],
                registry_active=len(owner.source_registry._active))


def public(row):
    return {k: v for k, v in row.items() if k != 'outputs'}


def same(a, b):
    import torch
    return all(torch.equal(x, y) for x, y in zip(a['outputs'], b['outputs']))


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = dict(schema=SCHEMA, verdict='FAIL', engine_qualified=False, tp_collectives=False, box_evidence=False)
    try:
        # Geometry: exact buckets, partial final bucket, partial final page, more buckets.
        for tag, kwargs, want in (('25p', dict(pages=25), [2, 2]),
                                  ('48p', dict(pages=48), [2, 2]),
                                  ('50p-partial', dict(pages=50, slack=1000), [3, 3]),
                                  ('73p', dict(pages=73, rows=150), [4, 4])):
            old = run('stepped', **kwargs)
            new = run('fused', **kwargs)
            numbers[tag] = dict(stepped=public(old), fused=public(new))
            check('bytes-' + tag, same(old, new), buckets=new['buckets'], old_digests=old['digests'][0][:16],
                  new_digests=new['digests'][0][:16])
            check('buckets-' + tag, new['buckets'] == want and old['buckets'] == want, want=want, got=new['buckets'])
            check('same-device-work-' + tag, all(old[k] == new[k] for k in ('copies', 'consumers', 'bytes')),
                  copies=new['copies'], consumers=new['consumers'], bytes=new['bytes'])
            check('arena-shared-' + tag, new['same_arena'])
            check('holds-released-' + tag, all(new['released']) and new['allocator_live'] == 0
                  and new['registry_active'] == 0)
        # The stepped path against itself: the comparison's noise floor is exact bytes.
        reference = run('stepped', pages=50)
        again = run('stepped', pages=50)
        check('stepped-repeats-exactly', same(reference, again))
        # Delayed consumers and delayed copies, with no host wait in the candidate.
        for delay in ('compute', 'copy'):
            new = run('fused', pages=50, delay=delay)
            numbers['delay-' + delay] = public(new)
            check('bytes-under-%s-delay' % delay, same(reference, new), submit_s=new['submit_s'], drain_s=new['drain_s'])
            check('no-host-wait-under-%s-delay' % delay,
                  new['unresolved_at_submit_end'] and new['delay_live_at_submit_end'] and new['drain_s'] > .5,
                  unresolved=new['unresolved_at_submit_end'], delay_live=new['delay_live_at_submit_end'],
                  submit_s=new['submit_s'], drain_s=new['drain_s'])
        # Negative controls: without the stream dependency the same delay changes the bytes.
        for mutant, delay in (('no-arena-order', 'compute'), ('no-copy-before-kernel', 'copy')):
            new = run('fused', pages=50, delay=delay, mutant=mutant)
            numbers['mutant-' + mutant] = public(new)
            check('negative-' + mutant + '-differs', new['delay_live_at_submit_end'] and not same(reference, new),
                  digests=new['digests'][0][:16], delay_live=new['delay_live_at_submit_end'])
        # A cancel with the device still working keeps the hold until the drain.
        row = run('fused', pages=50, delay='compute', cancel_after=1)
        numbers['cancel'] = row
        check('cancel-with-unresolved-consumer', row['unresolved_at_cancel'] and row['delay_live_at_cancel']
              and row['advance_refused']
              and not row['released_before_drain'] and not row['drained_at_once'] and row['hold_released']
              and row['allocator_live'] == 0 and row['registry_active'] == 0, **row)
        new = run('fused', pages=50)
        check('after-cancel-bytes', same(reference, new))
        result['verdict'] = 'PASS' if all(row['ok'] for row in checks.values()) else 'FAIL'
    except BaseException as exc:
        result['exception'] = traceback.format_exc()
        print(result['exception'], flush=True)
    result.update(checks=checks, numbers=numbers, torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    print('STEP_A_DEVICE', result['verdict'], sum(r['ok'] for r in checks.values()), '/', len(checks), flush=True)
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
