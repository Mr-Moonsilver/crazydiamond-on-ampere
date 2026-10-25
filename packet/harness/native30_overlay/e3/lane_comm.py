"""E3 prefill-lane communication (D4, worker/native side).

Decode keeps the stock TP device communicator (custom all-reduce, captured in
its graphs) and the stock PP handler/sample ring. A prefill segment must never
share either: concurrent use of the custom all-reduce signal buffers from two
streams races, and NCCL operations on one communicator from two streams have no
cross-rank order. So the lane gets

* :class:`PrefillTpCommunicator`: a PyNccl-only stand-in installed on the TP
  group coordinator for the duration of one bounded host operation
  (:func:`tp_scope`); every other collective fails closed so a new call site
  surfaces in the smoke instead of silently routing through decode's path;
* :class:`PrefillPPChannel`: a separate PyNccl communicator on the PP group with
  owned ping-pong buffers and a logical header (transaction identity, stage,
  sequence, token count) sent before each payload. The receiver pre-posts the
  receive on the prefill stream and never host-blocks; its header is validated
  from a pinned mirror once the completion event is queried complete. Sender
  storage is released only after the receiver's consumer proof returns through
  the engine.

Both communicators are created and warmed in a fixed rank order at startup.
Nothing here is imported at module load from torch/vLLM; the constructors take
the communicator class and torch module so CPU tests inject fakes.
"""
from contextlib import contextmanager

from e3_types import E3ContractError, TransactionId

HEADER_MAGIC = 0x5E3_C0DE
HEADER_LEN = 8
PP_SLOTS = 2
WARMUP_ELEMENTS = 16


class LaneCommError(E3ContractError):
    pass


def _torch():
    import torch

    return torch


def _pynccl_class():
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    return PyNcclCommunicator


# --- TP ------------------------------------------------------------------------

class PrefillTpCommunicator:
    """TP device communicator for the prefill lane: PyNccl on the lane stream only."""

    ca_comm = None
    qr_comm = None
    symm_mem_comm = None
    fi_ar_comm = None
    fi_pcie_ipc_ar_comm = None

    def __init__(self, pynccl, device, *, torch_module=None):
        if pynccl is None or getattr(pynccl, 'disabled', True):
            raise LaneCommError('prefill TP communicator needs an enabled PyNccl communicator')
        self.torch = torch_module if torch_module is not None else _torch()
        self.pynccl_comm = pynccl
        self.device = device
        self.world_size = int(getattr(pynccl, 'world_size', 0))
        self.rank = int(getattr(pynccl, 'rank', 0))
        self.collectives = 0
        self.warmed = False

    def all_reduce(self, input_):
        stream = self.torch.cuda.current_stream(self.device)
        out = self.pynccl_comm.all_reduce(input_, stream=stream)
        if out is None:
            raise LaneCommError('prefill PyNccl all-reduce returned nothing')
        self.collectives += 1
        return out

    def _refuse(self, name):
        raise LaneCommError(f'prefill lane does not route {name}; add an explicit lane path')

    def all_gather(self, input_, dim=-1):
        self._refuse('all_gather')

    def reduce_scatter(self, input_, dim=-1):
        self._refuse('reduce_scatter')

    def all_gatherv(self, *args, **kwargs):
        self._refuse('all_gatherv')

    def reduce_scatterv(self, *args, **kwargs):
        self._refuse('reduce_scatterv')

    def gather(self, *args, **kwargs):
        self._refuse('gather')

    def broadcast(self, *args, **kwargs):
        self._refuse('broadcast')

    def send(self, *args, **kwargs):
        self._refuse('send')

    def recv(self, *args, **kwargs):
        self._refuse('recv')

    def dispatch(self, *args, **kwargs):
        self._refuse('dispatch')

    def combine(self, *args, **kwargs):
        self._refuse('combine')

    def warm(self, stream):
        """One small all-reduce on the lane stream; every rank calls it in order."""
        torch = self.torch
        ones = torch.ones((WARMUP_ELEMENTS,), dtype=torch.float32, device=self.device)
        with torch.cuda.stream(stream):
            out = self.all_reduce(ones)
        stream.synchronize()
        expected = float(self.world_size)
        values = out.cpu().tolist()
        if any(abs(v - expected) > 0 for v in values):
            raise LaneCommError('prefill TP warm-up all-reduce produced a wrong sum')
        self.warmed = True
        return True


class _ScopeState:
    active = False


@contextmanager
def tp_scope(group, communicator):
    """Install the lane communicator on the TP group for one host operation."""
    if not isinstance(communicator, PrefillTpCommunicator):
        raise LaneCommError('tp_scope needs a PrefillTpCommunicator')
    if _ScopeState.active:
        raise LaneCommError('prefill TP scope is not reentrant')
    if not hasattr(group, 'device_communicator'):
        raise LaneCommError('group has no device communicator slot')
    saved = group.device_communicator
    if saved is communicator:
        raise LaneCommError('prefill TP communicator is already installed')
    _ScopeState.active = True
    group.device_communicator = communicator
    try:
        yield communicator
    finally:
        group.device_communicator = saved
        _ScopeState.active = False


# --- PP ------------------------------------------------------------------------

def header_values(transaction, stage, seq, num_tokens):
    if type(transaction) is not TransactionId:
        raise LaneCommError('PP header needs a TransactionId')
    for name, value in (('stage', stage), ('seq', seq), ('num_tokens', num_tokens)):
        if type(value) is not int or value < 0:
            raise LaneCommError(f'PP header {name} must be a non-negative int')
    if num_tokens == 0 or num_tokens > transaction.chunk_end - transaction.chunk_start:
        raise LaneCommError('PP header token count does not fit the transaction chunk')
    return [HEADER_MAGIC, transaction.engine_epoch, transaction.txn_seq,
            transaction.request_generation, transaction.chunk_start, transaction.chunk_end,
            stage * 1_000_000 + seq, num_tokens]


class PPTicket:
    __slots__ = ('slot', 'kind', 'transaction', 'stage', 'seq', 'num_tokens', 'event',
                 'state', 'consumer_event')

    def __init__(self, slot, kind, transaction, stage, seq, num_tokens, event):
        self.slot = slot
        self.kind = kind
        self.transaction = transaction
        self.stage = stage
        self.seq = seq
        self.num_tokens = num_tokens
        self.event = event
        self.state = 'posted'
        self.consumer_event = None


class PrefillPPChannel:
    """Owned ping-pong PP transport for the lane's HC tensor between two stages."""

    def __init__(self, pynccl, device, stream, *, max_tokens, hidden_width, dtype,
                 is_sender, peer_rank, slots=PP_SLOTS, torch_module=None):
        if pynccl is None or getattr(pynccl, 'disabled', True):
            raise LaneCommError('prefill PP channel needs an enabled PyNccl communicator')
        for name, value in (('max_tokens', max_tokens), ('hidden_width', hidden_width),
                            ('slots', slots), ('peer_rank', peer_rank)):
            if type(value) is not int or value < (0 if name == 'peer_rank' else 1):
                raise LaneCommError(f'PP channel {name} is invalid')
        self.torch = torch_module if torch_module is not None else _torch()
        torch = self.torch
        self.pynccl = pynccl
        self.device = device
        self.stream = stream
        self.max_tokens = max_tokens
        self.hidden_width = hidden_width
        self.dtype = dtype
        self.is_sender = bool(is_sender)
        self.peer_rank = peer_rank
        self.slots = slots
        self.payload = [torch.empty((max_tokens, hidden_width), dtype=dtype, device=device)
                        for _ in range(slots)]
        self.header = [torch.empty((HEADER_LEN,), dtype=torch.int64, device=device)
                       for _ in range(slots)]
        self.header_host = [torch.empty((HEADER_LEN,), dtype=torch.int64, pin_memory=True)
                            for _ in range(slots)]
        self.free_slots = list(range(slots))
        self.tickets = {}
        self.sent = 0
        self.received = 0
        self.quarantined = None
        self.warmed = False

    # -- inventory ----------------------------------------------------------
    def bytes_owned(self):
        element = int(self.payload[0].element_size()) if self.payload else 0
        return self.slots * (self.max_tokens * self.hidden_width * element + HEADER_LEN * 8)

    def describe(self):
        return {'slots': self.slots, 'max_tokens': self.max_tokens,
                'hidden_width': self.hidden_width, 'bytes_owned': self.bytes_owned(),
                'role': 'sender' if self.is_sender else 'receiver', 'sent': self.sent,
                'received': self.received, 'free_slots': len(self.free_slots)}

    def _require_live(self):
        if self.quarantined is not None:
            raise LaneCommError('PP channel is quarantined: ' + self.quarantined)

    def _acquire(self):
        if not self.free_slots:
            return None
        return self.free_slots.pop(0)

    def _event(self):
        return self.torch.cuda.Event(enable_timing=False)

    # -- sender -------------------------------------------------------------
    def send(self, payload, transaction, stage, seq):
        """Copy the lane's HC tensor into an owned slot and enqueue header+payload."""
        self._require_live()
        if not self.is_sender:
            raise LaneCommError('receiver side cannot send')
        num_tokens = int(payload.shape[0])
        values = header_values(transaction, stage, seq, num_tokens)
        if num_tokens > self.max_tokens or int(payload.shape[1]) != self.hidden_width:
            raise LaneCommError('PP payload shape exceeds the channel buffers')
        slot = self._acquire()
        if slot is None:
            raise LaneCommError('PP channel has no free send slot (backpressure)')
        torch = self.torch
        with torch.cuda.stream(self.stream):
            self.header[slot].copy_(torch.tensor(values, dtype=torch.int64), non_blocking=True)
            self.payload[slot][:num_tokens].copy_(payload, non_blocking=True)
            self.pynccl.send(self.header[slot], self.peer_rank, stream=self.stream)
            self.pynccl.send(self.payload[slot][:num_tokens], self.peer_rank, stream=self.stream)
            event = self._event()
            event.record(self.stream)
        ticket = PPTicket(slot, 'send', transaction, stage, seq, num_tokens, event)
        self.tickets[id(ticket)] = ticket
        self.sent += 1
        return ticket

    def release_send(self, ticket, consumer_proof):
        """Free a send slot after local completion and the receiver's consumer proof."""
        self._check_ticket(ticket, 'send')
        if consumer_proof is None:
            raise LaneCommError('send release needs the receiver consumer proof')
        if not bool(ticket.event.query()):
            raise LaneCommError('send release before local send completion')
        self._retire(ticket)
        return True

    # -- receiver -----------------------------------------------------------
    def post_receive(self, transaction, stage, seq, num_tokens):
        """Pre-post header+payload receives; returns immediately."""
        self._require_live()
        if self.is_sender:
            raise LaneCommError('sender side cannot receive')
        header_values(transaction, stage, seq, num_tokens)
        if num_tokens > self.max_tokens:
            raise LaneCommError('PP receive exceeds the channel buffers')
        slot = self._acquire()
        if slot is None:
            raise LaneCommError('PP channel has no free receive slot (backpressure)')
        torch = self.torch
        with torch.cuda.stream(self.stream):
            self.pynccl.recv(self.header[slot], self.peer_rank, stream=self.stream)
            self.pynccl.recv(self.payload[slot][:num_tokens], self.peer_rank, stream=self.stream)
            self.header_host[slot].copy_(self.header[slot], non_blocking=True)
            event = self._event()
            event.record(self.stream)
        ticket = PPTicket(slot, 'recv', transaction, stage, seq, num_tokens, event)
        self.tickets[id(ticket)] = ticket
        return ticket

    def poll_receive(self, ticket):
        """None while pending; the payload view once the header validates."""
        self._check_ticket(ticket, 'recv')
        if ticket.state == 'ready':
            return self.payload[ticket.slot][:ticket.num_tokens]
        if not bool(ticket.event.query()):
            return None
        expected = header_values(ticket.transaction, ticket.stage, ticket.seq, ticket.num_tokens)
        actual = [int(v) for v in self.header_host[ticket.slot].tolist()]
        if actual != expected:
            self.quarantined = 'PP header mismatch on receive'
            ticket.state = 'quarantined'
            raise LaneCommError(self.quarantined)
        ticket.state = 'ready'
        self.received += 1
        return self.payload[ticket.slot][:ticket.num_tokens]

    def consumer_done(self, ticket, consumer_event):
        """Record the consumer's event; the slot frees once it is queried complete."""
        self._check_ticket(ticket, 'recv')
        if ticket.state != 'ready':
            raise LaneCommError('consumer completion before the receive was ready')
        ticket.consumer_event = consumer_event
        return ticket

    def release_receive(self, ticket):
        self._check_ticket(ticket, 'recv')
        if ticket.consumer_event is None or not bool(ticket.consumer_event.query()):
            raise LaneCommError('receive release before the consumer completed')
        self._retire(ticket)
        return True

    # -- common -------------------------------------------------------------
    def _check_ticket(self, ticket, kind):
        if not isinstance(ticket, PPTicket) or self.tickets.get(id(ticket)) is not ticket:
            raise LaneCommError('unknown PP ticket')
        if ticket.kind != kind:
            raise LaneCommError(f'PP ticket is not a {kind} ticket')
        if ticket.state == 'retired':
            raise LaneCommError('PP ticket already retired')

    def _retire(self, ticket):
        ticket.state = 'retired'
        del self.tickets[id(ticket)]
        self.free_slots.append(ticket.slot)

    def outstanding(self):
        return len(self.tickets)

    def warm(self):
        """Fixed-order warm-up: sender sends, receiver receives one header-sized tensor."""
        torch = self.torch
        probe = torch.full((HEADER_LEN,), HEADER_MAGIC if self.is_sender else 0,
                           dtype=torch.int64, device=self.device)
        with torch.cuda.stream(self.stream):
            if self.is_sender:
                self.pynccl.send(probe, self.peer_rank, stream=self.stream)
            else:
                self.pynccl.recv(probe, self.peer_rank, stream=self.stream)
        self.stream.synchronize()
        if not self.is_sender and any(int(v) != HEADER_MAGIC for v in probe.cpu().tolist()):
            raise LaneCommError('PP warm-up probe did not arrive intact')
        self.warmed = True
        return True


# --- startup ------------------------------------------------------------------

class PrefillCommunicators:
    """Both lane communicators, created and warmed in fixed order on one rank."""

    def __init__(self, tp_comm, pp_channel, *, record=None):
        self.tp = tp_comm
        self.pp = pp_channel
        self.record = dict(record or {})

    @classmethod
    def build(cls, tp_group, pp_group, device, stream, *, max_tokens, hidden_width, dtype,
              pynccl_class=None, torch_module=None):
        pynccl_class = pynccl_class or _pynccl_class()
        torch = torch_module if torch_module is not None else _torch()
        tp_nccl = pynccl_class(group=tp_group.cpu_group, device=device)
        tp_comm = PrefillTpCommunicator(tp_nccl, device, torch_module=torch)
        pp_channel = None
        if int(getattr(pp_group, 'world_size', 1)) > 1:
            pp_nccl = pynccl_class(group=pp_group.cpu_group, device=device)
            is_sender = not bool(pp_group.is_last_rank)
            if is_sender and bool(pp_group.is_first_rank):
                peer = int(pp_group.rank_in_group) + 1
            elif not is_sender:
                peer = int(pp_group.rank_in_group) - 1
            else:
                raise LaneCommError('prefill PP channel supports exactly two stages')
            pp_channel = PrefillPPChannel(pp_nccl, device, stream, max_tokens=max_tokens,
                                          hidden_width=hidden_width, dtype=dtype,
                                          is_sender=is_sender, peer_rank=peer,
                                          torch_module=torch)
        return cls(tp_comm, pp_channel)

    def warm(self, stream):
        """TP first, then PP; identical order on every rank."""
        self.tp.warm(stream)
        if self.pp is not None:
            self.pp.warm()
        self.record['warmed'] = True
        return True

    def describe(self):
        record = {'tp_world_size': self.tp.world_size, 'tp_warmed': self.tp.warmed,
                  'pp': self.pp.describe() if self.pp is not None else None}
        record.update(self.record)
        return record


__all__ = ['HEADER_LEN', 'HEADER_MAGIC', 'LaneCommError', 'PP_SLOTS', 'PPTicket',
           'PrefillCommunicators', 'PrefillPPChannel', 'PrefillTpCommunicator',
           'header_values', 'tp_scope']
