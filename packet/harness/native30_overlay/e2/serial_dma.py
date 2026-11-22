"""E2 serial DMA backend. Not yet wired to vLLM's block ownership hooks.

The torch backend is executable but GPU-unqualified. Tests inject a byte-buffer
backend. Source leases and allocation incarnations come from retention_contract;
that model is NOT a substitute for real engine block references.
"""
from dataclasses import dataclass

from .retention_contract import ContractError, Pending


@dataclass(frozen=True)
class TensorFacts:
    device: str
    shape: tuple
    strides: tuple
    dtype: str
    item_bytes: int
    pinned: bool
    contiguous: bool
    # Internal overlap checks only. Never put addresses in reports.
    address: int

    @property
    def nbytes(self):
        n = self.item_bytes
        for dimension in self.shape:
            n *= dimension
        return n


@dataclass(frozen=True)
class Binding:
    host: object
    resident: object


class TorchBackend:
    def __init__(self, device):
        import torch
        self.torch = torch
        self.device = str(torch.device(device))
        if not self.device.startswith('cuda:'):
            raise ContractError('explicit CUDA device ordinal required')
        with torch.cuda.device(self.device):
            self.stream = torch.cuda.Stream(device=self.device)

    def facts(self, tensor):
        return TensorFacts(str(tensor.device), tuple(tensor.shape),
                           tuple(tensor.stride()), str(tensor.dtype),
                           tensor.element_size(), tensor.is_pinned(),
                           tensor.is_contiguous(), tensor.data_ptr())

    def wait(self, event):
        if str(event.device) != self.device:
            raise ContractError('unrecorded or foreign-device completion event')
        event.synchronize()

    def copy_rows(self, binding, source_row, destination_row, count):
        # H=1 BLNHC is checked by SerialDMA. Views never allocate a CPU pack
        # buffer or convert a CUDA UVA view back into a host source.
        width = binding.host.shape[-1]
        src = binding.host.view(-1, width).narrow(0, source_row, count)
        dst = binding.resident.view(-1, width).narrow(0, destination_row, count)
        with self.torch.cuda.device(self.device), self.torch.cuda.stream(self.stream):
            dst.copy_(src, non_blocking=True)
            done = self.torch.cuda.Event()
            done.record(self.stream)
        return done


class SerialDMA:
    """Six fixed slabs, CPU sources, one bounded copy at a time per rank.

Caller must record producer events AFTER the actual KV store and CoW work;
consumer events AFTER attention. E2 waits on the CPU intentionally. E3 must
replace these waits with dispatcher-owned event progress, not reuse this class
on the decode-critical thread. No tensors are allocated by this adapter.
    """
    def __init__(self, bank, bindings, backend):
        if set(bindings) != set(bank.layers):
            raise ContractError('exactly the six loaded local layers required')
        self.bank, self.backend = bank, backend
        self.bindings = dict(bindings)
        self.facts = {}
        ranges = []
        for name in bank.layers:
            b = self.bindings[name]
            host, dst = backend.facts(b.host), backend.facts(b.resident)
            for f in (host, dst):
                if len(f.shape) != 4 or f.shape[1:] != (1, bank.pool.page_tokens, 512):
                    raise ContractError('requires production BLNHC H=1, width=512')
                if f.dtype != 'torch.bfloat16' or f.item_bytes != 2 or not f.contiguous:
                    raise ContractError('requires contiguous BF16 storage')
                if f.strides != (bank.pool.page_tokens*512, bank.pool.page_tokens*512, 512, 1):
                    raise ContractError('unexpected source or destination strides')
                if f.address <= 0 or f.shape[0] < 1:
                    raise ContractError('empty allocation')
                for device, lo, hi in ranges:
                    if f.device == device and max(lo, f.address) < min(hi, f.address+f.nbytes):
                        raise ContractError('history allocations overlap')
                ranges.append((f.device, f.address, f.address+f.nbytes))
            if host.device != 'cpu' or not host.pinned:
                raise ContractError('DMA requires pinned CPU source, never CUDA UVA')
            if dst.device != backend.device or not dst.device.startswith('cuda:'):
                raise ContractError('resident slab is on the wrong device')
            if dst.shape[0] != bank.capacity_pages:
                raise ContractError('resident capacity differs from reserved ledger')
            self.facts[name] = (host, dst)
        self.bytes_completed = 0
        self.copies_completed = 0
        self.failed_events = []  # Retain event objects if a CUDA wait fails.

    def _check(self, history, layer, draining=False):
        if history.bank is not self.bank or self.bank.active is not history:
            raise ContractError('foreign or retired transaction')
        if draining:
            if history.retired or history.quarantined:
                raise ContractError('cannot drain retired or quarantined transaction')
        else:
            history.available()
        if layer not in self.bindings:
            raise ContractError('unknown layer')
        b = self.bindings[layer]
        if (self.backend.facts(b.host), self.backend.facts(b.resident)) != self.facts[layer]:
            raise ContractError('bound tensor allocation or layout changed')

    def _wait_complete(self, history, ticket, event):
        try:
            self.backend.wait(event)
            history.complete(ticket)
        except BaseException:
            # A submit/wait failure cannot establish that DMA stopped. Keep the
            # ticket, tensor bindings and leases; recovery is an engine decision.
            history.quarantined = True
            self.failed_events.append(event)
            raise

    def producer_complete(self, history, ticket, event):
        self._check(history, ticket.layer, draining=True)
        if ticket.kind != 'producer' or ticket not in history.pending:
            raise ContractError('not an outstanding producer')
        self._wait_complete(history, ticket, event)

    def fill(self, history, layer):
        self._check(history, layer)
        host, _ = self.facts[layer]
        # Validate the entire plan before a first submission. No partial source
        # reads if a later physical index is outside this actual pinned tensor.
        plans = history.plan(layer)
        for plan in plans:
            if plan.row_bytes != 1024 or not 1 <= len(plan.sources) <= 4:
                raise ContractError('invalid DMA geometry')
            if any(not 0 < s.page.physical < host.shape[0] for s in plan.sources):
                raise ContractError('physical block outside host allocation')
        for plan in plans:
            ticket = history.copy(plan)
            try:
                event = self.backend.copy_rows(
                    self.bindings[layer], plan.sources[0].page.physical*self.bank.pool.page_tokens,
                    plan.logical_start*self.bank.pool.page_tokens,
                    sum(s.rows for s in plan.sources))
            except BaseException:
                history.quarantined = True
                raise
            self._wait_complete(history, ticket, event)
            self.bytes_completed += plan.bytes
            self.copies_completed += 1
        return {'copies': len(plans), 'bytes': sum(p.bytes for p in plans)}

    def begin_attention(self, history, layer):
        self._check(history, layer)
        ticket = history.attend(layer)
        return ticket, self.bindings[layer].resident

    def attention_complete(self, history, ticket, event):
        # Cancellation is allowed while a consumer is finishing: drain its
        # existing ticket rather than opening new work or dropping ownership.
        self._check(history, ticket.layer, draining=True)
        if ticket.kind != 'attention' or ticket not in history.pending:
            raise ContractError('not an outstanding consumer')
        self._wait_complete(history, ticket, event)

    def close(self):
        h = self.bank.active
        if h is not None and not h.retired:
            raise Pending('retire transaction before releasing tensor bindings')
        self.bindings.clear()
