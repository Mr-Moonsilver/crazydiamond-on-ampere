"""Offline S4 protocol prototype. No torch, worker RPCs, allocations or KV-manager hooks.

An acknowledgement means the worker has observed its completion fence, NOT merely
returned from a Python forward. StageSlot demonstrates that local contract; the
distributed adapter that supplies private metadata and fences is intentionally absent.
"""
from dataclasses import dataclass
from enum import Enum


class ProtocolError(RuntimeError):
    pass


class Phase(Enum):
    ACTIVE = "active"
    READY = "ready_to_commit"
    COMMITTED = "committed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled_and_released"


@dataclass(frozen=True)
class ChunkKey:
    request_id: str
    generation: int
    chunk_id: int
    base_tokens: int
    chunk_tokens: int

    def __post_init__(self):
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be nonempty")
        for name, minimum in (("generation", 1), ("chunk_id", 0),
                              ("base_tokens", 0), ("chunk_tokens", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"invalid {name}")

    @property
    def model_owner(self):
        return (self.request_id, self.generation, self.chunk_id)


@dataclass(frozen=True)
class Ticket:
    key: ChunkKey
    stage: int
    ordinal: int
    lo: int
    hi: int


@dataclass(frozen=True)
class Completion:
    ticket: Ticket
    succeeded: bool


class ChunkLedger:
    """One chunk, TP acknowledgements, sequential PP handoff, commit exactly once.

    Reservations are logical here. A production KV adapter must enforce the same
    frontier against ALL allocation/publication/free paths. This class does not.
    The conservative first version permits only one segment in flight globally.
    Decode dispatches are outside this ledger and cannot advance its cursor.
    """

    def __init__(self, key, stage_ranges=((0, 26), (26, 48)), tp_size=2, layers=1):
        if type(layers) is not int or layers <= 0 or type(tp_size) is not int or tp_size <= 0:
            raise ValueError("positive integer segment length and TP size required")
        if not stage_ranges or stage_ranges[0][0] != 0:
            raise ValueError("stage ranges must begin at layer zero")
        previous = 0
        for lo, hi in stage_ranges:
            if type(lo) is not int or type(hi) is not int or lo != previous or hi <= lo:
                raise ValueError("stage ranges must be contiguous and nonempty")
            previous = hi
        self.key = key
        self.ranges = tuple(stage_ranges)
        self.tp_size, self.layers = tp_size, layers
        self.stage, self.cursor, self.ordinal = 0, 0, 0
        self.phase = Phase.ACTIVE
        self.inflight = None
        self.acks = set()
        self.handoff_stage = None
        self.handoff_acks = set()
        self.quiescent = set()
        self.computed_tokens = key.base_tokens
        self.reserved_until = key.base_tokens + key.chunk_tokens

    def issue(self):
        if self.phase is not Phase.ACTIVE or self.inflight is not None or self.handoff_stage is not None:
            raise ProtocolError("cannot issue: pending segment or inactive chunk")
        ticket = Ticket(self.key, self.stage, self.ordinal, self.cursor,
                        min(self.cursor + self.layers, self.ranges[self.stage][1]))
        self.inflight, self.acks = ticket, set()
        return ticket

    def acknowledge(self, completion, tp_rank):
        if not isinstance(completion, Completion) or type(completion.succeeded) is not bool:
            raise ProtocolError("completion must carry an explicit success/drain status")
        ticket = completion.ticket
        if (self.phase not in (Phase.ACTIVE, Phase.CANCELLING)
                or self.inflight is None or ticket != self.inflight):
            raise ProtocolError("stale or out-of-order segment completion")
        if type(tp_rank) is not int or not 0 <= tp_rank < self.tp_size or tp_rank in self.acks:
            raise ProtocolError("unknown or duplicate TP acknowledgement")
        if not completion.succeeded:
            self.cancel()
        self.acks.add(tp_rank)
        if len(self.acks) != self.tp_size:
            return
        self.inflight = None
        if self.phase is Phase.CANCELLING:
            return  # a late completion must never publish cancelled work
        self.cursor, self.ordinal = ticket.hi, self.ordinal + 1
        if self.cursor == self.ranges[self.stage][1]:
            if self.stage == len(self.ranges) - 1:
                self.phase = Phase.READY
            else:
                self.handoff_stage, self.handoff_acks = self.stage, set()
                self.stage += 1
                self.cursor = self.ranges[self.stage][0]

    def acknowledge_handoff(self, key, source_stage, tp_rank):
        """Both send and receiving private-buffer readiness have completed for this TP lane."""
        if (key != self.key or self.phase not in (Phase.ACTIVE, Phase.CANCELLING)
                or self.handoff_stage is None or source_stage != self.handoff_stage
                or type(tp_rank) is not int or not 0 <= tp_rank < self.tp_size
                or tp_rank in self.handoff_acks):
            raise ProtocolError("unknown, stale or duplicate PP handoff")
        self.handoff_acks.add(tp_rank)
        if len(self.handoff_acks) == self.tp_size:
            self.handoff_stage = None

    def commit(self):
        if self.phase is not Phase.READY:
            raise ProtocolError("commit requires every layer on every TP/PP rank")
        self.computed_tokens = self.reserved_until
        self.phase = Phase.COMMITTED
        return self.computed_tokens

    def cancel(self):
        if self.phase not in (Phase.ACTIVE, Phase.READY, Phase.CANCELLING):
            raise ProtocolError("cannot cancel a terminal chunk")
        self.phase = Phase.CANCELLING

    def acknowledge_quiescent(self, key, stage, tp_rank):
        if self.phase is not Phase.CANCELLING or key != self.key:
            raise ProtocolError("quiescence belongs to a different/cancelled generation")
        if (type(stage) is not int or type(tp_rank) is not int
                or not 0 <= stage < len(self.ranges) or not 0 <= tp_rank < self.tp_size):
            raise ProtocolError("unknown worker")
        if self.handoff_stage is not None:
            raise ProtocolError("pipeline transfer is still in flight")
        if self.inflight is not None and self.inflight.stage == stage and tp_rank not in self.acks:
            raise ProtocolError("worker still has an unacknowledged operation")
        worker = (stage, tp_rank)
        if worker in self.quiescent:
            raise ProtocolError("duplicate quiescence acknowledgement")
        self.quiescent.add(worker)

    def release_cancelled(self):
        if (self.phase is not Phase.CANCELLING or self.inflight is not None or self.handoff_stage is not None
                or len(self.quiescent) != len(self.ranges) * self.tp_size):
            raise ProtocolError("workspace/pages remain pinned until ALL workers drain")
        self.phase = Phase.CANCELLED
        return self.computed_tokens  # prefix publication remains at the old frontier


class StageSlot:
    """Own one model continuation and private bindings until completion or drain.

    `bind()` installs an immutable, request-owned forward context for ONE call;
    it must also restore QSA UID and any global scratch bindings. No implementation
    of that vLLM adapter is supplied. `record_fence()` must cover every stream that
    touched the operation, including PLE, TP communication and PP transfer.

    On a failed fence recording, the slot stays quarantined; it cannot be released.
    Losing a device requires worker recovery, not claiming its outstanding writes
    are finished. No block freeing happens in this class.
    """

    def __init__(self, key, stage, layer_range, iterator, resources):
        self.key, self.stage = key, stage
        self.cursor, self.end = layer_range
        if self.end <= self.cursor:
            raise ValueError("empty stage")
        self.iterator = iterator
        self.resources = resources  # strong ownership; caller must supply private copies
        self.pending = None
        self.fence = None
        self.result = None
        self.cancelled = self.failed = self.done = self.closed = False
        self._ordinal = None
        self.consumer_fences = []
        self.needs_consumer_fence = False

    def advance(self, ticket, bind, record_fence):
        if (self.closed or self.cancelled or self.failed or self.done or self.pending is not None
                or ticket.key != self.key or ticket.stage != self.stage
                or ticket.lo != self.cursor or not self.cursor < ticket.hi <= self.end
                or (self._ordinal is not None and ticket.ordinal != self._ordinal + 1)):
            raise ProtocolError("wrong owner, cursor, order, or uncompleted segment")
        self.pending = ticket
        try:
            with bind():
                result = next(self.iterator)
            lo, hi, output = result
            if (lo, hi) != (ticket.lo, ticket.hi) or (output is None) != (hi != self.end):
                raise ProtocolError("model segment did not match the dispatch ticket")
            self.result = result
        except BaseException:
            self.failed = True
            raise
        finally:
            try:
                self.fence = record_fence()
                if self.fence is None or not callable(getattr(self.fence, "query", None)):
                    raise ProtocolError("missing completion fence")
            except BaseException:
                self.failed = True
                self.fence = None
                raise

    def poll(self):
        if self.pending is None or self.fence is None or not self.fence.query():
            return None
        ticket = self.pending
        self.pending, self.fence = None, None
        if not self.cancelled and not self.failed:
            self.cursor, self._ordinal = ticket.hi, ticket.ordinal
            self.done = ticket.hi == self.end
        return Completion(ticket, succeeded=not self.cancelled and not self.failed)

    def cancel(self, key):
        if self.closed or key != self.key:
            raise ProtocolError("wrong workspace owner")
        self.cancelled = True

    def output(self, key):
        if (key != self.key or self.closed or not self.done or self.cancelled or self.failed
                or self.needs_consumer_fence):
            raise ProtocolError("no completed stage output for this owner")
        self.needs_consumer_fence = True
        return self.result[2]

    def hold_output(self, fence):
        """Register the PP transport/sampling consumer after taking stage output."""
        if not self.needs_consumer_fence or not callable(getattr(fence, "query", None)):
            raise ProtocolError("no output consumer or no consumer fence")
        self.consumer_fences.append(fence)
        self.needs_consumer_fence = False

    def close(self, key):
        if (key != self.key or self.closed or self.pending is not None or self.needs_consumer_fence
                or any(not fence.query() for fence in self.consumer_fences)
                or not (self.done or self.cancelled or self.failed)):
            raise ProtocolError("cannot free a live workspace")
        self.iterator.close()  # caller must have included output consumers in its fence
        self.resources = self.result = None
        self.closed = True
