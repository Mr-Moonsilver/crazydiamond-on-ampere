"""Event-backed CPU-source DMA into E2's EXISTING arena. GPU-unqualified.

All torch imports are lazy. Tests use delayed byte-buffer transfers. No tensor
allocations are made by the adapter: sources are the six loaded registered host
pools, destination is the baseline arena. Step 4 must install runner hooks.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import time

from retention_contract import ContractError, Pending
from serial_dma import TensorFacts


@dataclass(frozen=True, repr=False)
class Fence:
    device: str
    stream: int
    event: object


class TorchArenaBackend:
    def __init__(self, device):
        import torch
        self.torch = torch
        self.device = str(torch.device(device))
        if not self.device.startswith('cuda:'):
            raise ContractError('explicit CUDA device ordinal required')
        with torch.cuda.device(self.device):
            self.copy_stream = torch.cuda.Stream(device=self.device)

    def facts(self, tensor):
        return TensorFacts(str(tensor.device), tuple(tensor.shape), tuple(tensor.stride()),
                           str(tensor.dtype), tensor.element_size(), tensor.is_pinned(),
                           tensor.is_contiguous(), tensor.data_ptr())

    def execution_stream(self):
        if self.torch.cuda.current_device() != self.torch.device(self.device).index:
            raise ContractError('worker CUDA device changed')
        if self.torch.cuda.is_current_stream_capturing():
            raise ContractError('CPU arena DMA cannot run inside graph capture')
        stream = self.torch.cuda.current_stream(self.device)
        return int(stream.cuda_stream)

    def execution_fence(self):
        token = self.execution_stream()
        event = self.torch.cuda.Event()
        event.record(self.torch.cuda.current_stream(self.device))
        return Fence(self.device, token, event)

    def submit(self, host, arena, pieces):
        # CPU source views only. No index_select or UVA kernel in this path.
        # A single terminal event covers all <=4-page pieces of one bucket.
        with self.torch.cuda.device(self.device), self.torch.cuda.stream(self.copy_stream):
            for piece in pieces:
                dst = arena[piece.arena_page:piece.arena_page + piece.pages]
                src = host[piece.source_page:piece.source_page + piece.pages]
                dst.copy_(src, non_blocking=True)
            event = self.torch.cuda.Event()
            event.record(self.copy_stream)
        return Fence(self.device, int(self.copy_stream.cuda_stream), event)

    def wait(self, fence, stream):
        if (not isinstance(fence, Fence) or fence.device != self.device or fence.stream != stream or
                str(fence.event.device) != self.device):
            raise ContractError('unrecorded or foreign completion event')
        fence.event.synchronize()

    def copy_stream_token(self):
        return int(self.copy_stream.cuda_stream)


class ArenaDMA:
    def __init__(self, mirror, hosts, backend):
        if set(hosts) != set(mirror.layer_groups):
            raise ContractError('exactly the six loaded local QSA host pools required')
        self.mirror, self.backend = mirror, backend
        self.hosts = dict(hosts)
        self.host_facts = {layer: backend.facts(tensor) for layer, tensor in hosts.items()}
        ranges = []
        for facts in self.host_facts.values():
            self._layout(facts)
            if facts.device != 'cpu' or not facts.pinned or facts.shape[0] != mirror.num_blocks:
                raise ContractError('requires exact-size registered CPU pool, never CUDA UVA source')
            if any(max(lo, facts.address) < min(hi, facts.address + facts.nbytes) for lo,hi in ranges):
                raise ContractError('local host histories alias')
            ranges.append((facts.address, facts.address + facts.nbytes))
        self.arena = None
        self.arena_facts = None
        self.current = None
        self.failed_events = []
        self.stats = dict(dense_dma=0, sparse_uva=0, empty=0, buckets=0, pieces=0, bytes=0,
                          producer_wait_s=0.0, copy_wait_s=0.0, consumer_wait_s=0.0)
        self.routes_by_layer = {name:dict(dense_dma=0,sparse_uva=0,empty=0) for name in hosts}

    def _layout(self, facts):
        size = self.mirror.pool.page_tokens
        if (len(facts.shape) != 4 or facts.shape[1:] != (1,size,512) or
                facts.strides != (size*512,size*512,512,1) or not facts.contiguous or
                facts.dtype != 'torch.bfloat16' or facts.item_bytes != 2 or facts.address <= 0):
            raise ContractError('requires contiguous BF16 BLNHC H=1 width=512')

    def _binding(self, arena):
        facts = self.backend.facts(arena)
        self._layout(facts)
        if facts.device != self.backend.device or facts.shape[0] != self.mirror.arena.arena_pages:
            raise ContractError('existing arena layout/capacity/device changed')
        if self.arena is None:
            self.arena, self.arena_facts = arena, facts
        elif facts != self.arena_facts:
            raise ContractError('existing arena storage changed')

    def _check(self, layer):
        if self.mirror.failed or self.mirror.arena.state == 'quarantined':
            raise Pending('failed DMA adapter requires worker recovery')
        if self.backend.facts(self.hosts[layer]) != self.host_facts[layer]:
            raise ContractError('host allocation changed after binding')
        if self.backend.facts(self.arena) != self.arena_facts:
            raise ContractError('arena allocation changed after binding')

    def _wait(self, fence, stream, kind):
        start = time.perf_counter()
        try:
            self.backend.wait(fence, stream)
        except BaseException:
            self.failed_events.append(fence)
            self.mirror.failed = True
            raise
        self.stats[kind + '_wait_s'] += time.perf_counter() - start

    @contextmanager
    def session(self, rid, layer, arena, table, valid_tokens):
        if self.current is not None:
            raise Pending('one serial arena invocation per worker')
        if not self.mirror.active or rid not in self.mirror.current or layer not in self.hosts:
            raise ContractError('no matching worker request/layer frame')
        row = self.mirror.current[rid]
        if valid_tokens != row.computed + row.scheduled:
            raise ContractError('attention sequence length disagrees with CPU mirror')
        self._binding(arena)
        self._check(layer)
        session = Session(self, rid, layer, table)
        self.current = session
        try:
            yield session
            session.close()
        except BaseException:
            # Keep tensor bindings and any pending tickets on failure. Do not
            # pretend a failed submission/fence has released its source lease.
            self.mirror.failed = True
            raise
        finally:
            if not self.mirror.failed:
                self.current = None


class Session:
    def __init__(self, adapter, rid, layer, table):
        self.adapter, self.rid, self.layer = adapter, rid, layer
        self.table = table
        self.table_facts = adapter.backend.facts(table)
        if (self.table_facts.device != adapter.backend.device or len(self.table_facts.shape) != 1 or
                self.table_facts.dtype not in ('torch.int32', 'torch.int64') or
                self.table_facts.strides != (1,) or not self.table_facts.contiguous):
            raise ContractError('invalid worker GPU block-table row')
        self.stream = adapter.backend.execution_stream()
        self.route_name = None
        self.bucket = self.copy_ticket = self.consumer_ticket = None

    def _check(self):
        a = self.adapter
        a._check(self.layer)
        if a.current is not self or a.backend.execution_stream() != self.stream:
            raise ContractError('execution stream or session changed')
        if a.backend.facts(self.table) != self.table_facts:
            raise ContractError('GPU block table view changed')

    def route(self, name, pages):
        self._check()
        if self.route_name is not None or name not in ('empty','sparse_uva','dense_dma'):
            raise ContractError('invalid or repeated attention route')
        if type(pages) is not int or pages < 0 or (name != 'empty' and pages == 0):
            raise ContractError('invalid routed page count')
        self.route_name = name
        a = self.adapter
        a.stats[name] += 1
        a.routes_by_layer[self.layer][name] += 1
        if name != 'dense_dma':
            return  # Unchanged sparse-UVA path; NO extra producer/consumer wait.
        row = a.mirror.current[self.rid]
        expected = (row.computed + row.scheduled + a.mirror.pool.page_tokens - 1) // a.mirror.pool.page_tokens
        if pages != expected:
            raise ContractError('dense page count disagrees with source lease')
        fence = a.backend.execution_fence()
        a.mirror.producer_ready(self.layer, a.mirror.sequence,
                               lambda: a._wait(fence, self.stream, 'producer'))
        a.mirror.acquire(self.rid, self.layer)

    def fill(self, staged, lo, hi, table):
        self._check()
        a = self.adapter
        facts = a.backend.facts(staged)
        expected = a.arena_facts
        if (self.route_name != 'dense_dma' or a.backend.facts(table) != self.table_facts or
                facts != TensorFacts(expected.device,(hi-lo,*expected.shape[1:]),expected.strides,
                                     expected.dtype,expected.item_bytes,expected.pinned,
                                     expected.contiguous,expected.address)):
            raise ContractError('wrong dense bucket destination or mapping view')
        bucket, ticket = a.mirror.arena.begin_fill()
        self.bucket, self.copy_ticket = bucket, ticket
        if (bucket.lo,bucket.hi) != (lo,hi):
            raise ContractError('attention bucket order differs from original')
        if any(p.source_page + p.pages > a.host_facts[self.layer].shape[0] for p in bucket.pieces):
            raise ContractError('DMA reads beyond pinned source allocation')
        fence = a.backend.submit(a.hosts[self.layer], a.arena, bucket.pieces)
        a._wait(fence, a.backend.copy_stream_token(), 'copy')
        a.mirror.arena.complete_fill(ticket)
        self.copy_ticket = None
        a.stats['buckets'] += 1
        a.stats['pieces'] += len(bucket.pieces)
        a.stats['bytes'] += (hi-lo) * a.mirror.pool.page_tokens * 1024

    def consume_begin(self):
        self._check()
        if self.route_name != 'dense_dma' or self.bucket is None:
            raise ContractError('no filled dense bucket')
        self.consumer_ticket = self.adapter.mirror.arena.begin_attention(self.bucket)

    def consume_end(self):
        self._check()
        if self.consumer_ticket is None:
            raise ContractError('no attention consumption')
        a = self.adapter
        fence = a.backend.execution_fence()
        a._wait(fence, self.stream, 'consumer')
        a.mirror.arena.complete_attention(self.consumer_ticket)
        self.consumer_ticket = self.bucket = None

    def close(self):
        self._check()
        if self.route_name is None:
            raise ContractError('attention returned without a route')
        if self.route_name == 'dense_dma':
            self.adapter.mirror.arena.release()
