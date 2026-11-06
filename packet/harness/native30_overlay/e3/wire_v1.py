"""Canonical wire-v1 records shared by the engine driver and the worker lanes.

This module is the single schema both sides of the E3 overlap speak. It is
frozen and validated at construction, imports neither torch nor vLLM, and does
no I/O, so the engine driver, the generated worker hooks and the CPU fixtures
all use exactly the same objects.

The records deliberately keep three things separate:

* **Admission** is a snapshot of what the scheduler already allocated for one
  chunk. ``block_tables`` are the FULL immutable per-group rows -- the worker
  discovers its own local QSA/Mamba groups, it is never handed a flat
  ``block_ids`` list as authoritative history.
* **Segments** describe one bounded worker operation. There is no whole-forward
  record and no implicit fallback: one kind, one stage, at most one layer.
* **Completion** is reported only from host-queried facts. An RPC reply is not
  a completion proof, and a report carries its own transaction/lane identity
  so a stale report cannot advance another transaction.

Sequence rules (binding, see ``docs/e3-briefs/contract-decisions.md``):

* The ``seq`` in a PP segment is the explicit PP transfer sequence, shared by
  sender and receiver. It is a different field from ``op_seq`` and is never
  defaulted from it.
* ``op_seq`` is not part of a descriptor. It is assigned when the worker
  ACCEPTS the command and is strictly increasing per lane. A retry of an
  identical command is idempotent; :class:`OpSeqTracker` enforces both rules
  and never consumes a sequence for pending work.

Request UIDs, token ids and physical block tables never appear in ``repr``.
"""
from dataclasses import dataclass, field
from enum import Enum
import hashlib

from e3_types import (
    CompletionKind,
    CompletionProof,
    DependencyRole,
    E3ContractError,
    LaneId,
    LaneRole,
    OperationTicket,
    TransactionId,
)
from lane_comm import PPConsumerReceipt
from sparse_history_dma import ARENA_PAGES, ARENA_ROWS, MAX_DESCRIPTORS, MAX_LAYER_PAGES

# coverage-soak-fix: the most prompt rows a terminal chunk leaves stock (a geometry tail; scheduler_hooks pins the same, the
# staged minimum qsa.py _QSA_STAGE_MIN_ROWS); 1 is the one-row handback.
GEOMETRY_TAIL_MAX = 64

WIRE_VERSION = 1

# The two route banks in the fixed decode-graph bank. They are wire values, not
# local enum names, so both sides compare the same integer.
ROUTE_BANK_FULL = 0
ROUTE_BANK_SPLIT = 1
ROUTE_BANKS = (ROUTE_BANK_FULL, ROUTE_BANK_SPLIT)

CURSOR_PHASES = ('prepare', 'pp_receive', 'prefix', 'layer', 'suffix',
                 'pp_send', 'state_finalize', 'commit', 'idle')

FAILED_CODES = ('worker_failure', 'cancelled', 'preempted', 'timeout',
                'state_epoch', 'pp_mismatch', 'quarantined', 'aborted')

# Why a bounded cancellation was requested. The values are the generated stop
# kinds; the worker records the reason but never infers a disposition from it.
CANCEL_REASONS = ('abort', 'cancel', 'disconnect', 'preempt', 'reset', 'pause',
                  'shutdown', 'timeout')

# What the worker actually owned when cancellation reached it. This is a
# disposition, not a completion proof: only ``released`` certifies that the
# lane storage and the request row are closed.
CANCEL_DISPOSITIONS = ('never_open', 'opened', 'accepted', 'quarantined',
                       'released')


class SegmentKind(str, Enum):
    """The only operations a worker may be asked to run in the first cut.

    ``layer`` carries exactly one half-open global layer id, so slicing a
    forward into arbitrary spans is refused here rather than at the worker.
    """
    PREPARE = 'prepare'
    PP_RECEIVE = 'pp_receive'
    PREFIX = 'prefix'
    LAYER = 'layer'
    SUFFIX = 'suffix'
    PP_SEND = 'pp_send'
    STATE_FINALIZE = 'state_finalize'


# Kind -> stages the first implementation accepts. LAYER/SUFFIX are common to
# both stages. Only the two PP transfer directions are stage-specific.
_KIND_STAGES = {
    SegmentKind.PREPARE: (0, 1),
    SegmentKind.PP_RECEIVE: (1,),
    SegmentKind.PREFIX: (0, 1),
    SegmentKind.LAYER: None,
    SegmentKind.SUFFIX: None,
    SegmentKind.PP_SEND: (0,),
    SegmentKind.STATE_FINALIZE: (0, 1),
}

_PP_KINDS = (SegmentKind.PP_SEND, SegmentKind.PP_RECEIVE)


def _int(name, value, minimum=0):
    if type(value) is not int:
        raise E3ContractError(f'{name} must be an int, not {type(value).__name__}')
    if value < minimum:
        raise E3ContractError(f'{name} must be >= {minimum}')
    return value


def _tuple(name, value):
    if type(value) is not tuple:
        raise E3ContractError(name + ' must be an immutable tuple')
    return value


def _ints(name, value, *, minimum=0, allow_empty=True, unique=True):
    _tuple(name, value)
    if not allow_empty and not value:
        raise E3ContractError(name + ' must not be empty')
    for item in value:
        _int(name, item, minimum)
    if unique and len(set(value)) != len(value):
        raise E3ContractError(name + ' must not repeat an entry')
    return value


def _kind(name, value, enum):
    if type(value) is not enum:
        raise E3ContractError(f'{name} must be {enum.__name__}')
    return value


def _text(name, value):
    if type(value) is not str or not value:
        raise E3ContractError(f'{name} must be a non-empty string')
    if len(value) > 256 or any(ord(c) < 32 for c in value):
        raise E3ContractError(f'{name} contains unsupported characters')
    return value


@dataclass(frozen=True, repr=False)
class AdmissionRecord:
    """What the scheduler allocated for one chunk, captured once.

    ``prompt_end`` is the number of prompt tokens; the chunk is
    ``[transaction.chunk_start, transaction.chunk_end)`` and must leave at
    least the final token for the ordinary handback path. ``committed_end`` is
    the committed frontier BEFORE this chunk (chunk_start); the worker's
    private align seed is derived from it, never sent.

    ``native_end`` (coverage-soak-fix) is where the native turn ends: the
    prompt's last row (``prompt_end - 1``, the default, 0 on the wire means it)
    or, on a geometry tail, the last Mamba block boundary, leaving stock at most
    ``GEOMETRY_TAIL_MAX`` rows. The chunk ending there is the terminal one.
    """
    transaction: TransactionId
    prompt_end: int
    state_epoch: int
    committed_end: int
    block_tables: tuple
    slot_mappings: tuple
    new_block_ids_to_zero: tuple
    cow_pairs: tuple
    input_ids: tuple = field(repr=False)
    ngram_context: tuple = field(default=(), repr=False)
    sparse_active: bool = False
    native_end: int = 0

    def __post_init__(self):
        transaction = self.transaction
        if type(transaction) is not TransactionId:
            raise E3ContractError('admission needs a TransactionId')
        _int('prompt_end', self.prompt_end, 2)
        if transaction.chunk_start >= transaction.chunk_end:
            raise E3ContractError('admission chunk is empty')
        if type(self.native_end) is not int:
            raise E3ContractError('native_end must be an int')
        if self.native_end == 0:
            object.__setattr__(self, 'native_end', self.prompt_end - 1)
        if not 1 <= self.prompt_end - self.native_end <= GEOMETRY_TAIL_MAX:
            raise E3ContractError('native_end leaves stock more than a geometry tail')
        if transaction.chunk_end > self.prompt_end - 1:
            raise E3ContractError(
                'admission must leave the last prompt token for handback')
        if transaction.chunk_end > self.native_end:
            raise E3ContractError('admission chunk ends past the native end')
        _int('state_epoch', self.state_epoch, 1)
        _int('committed_end', self.committed_end, 0)
        if self.committed_end != transaction.chunk_start:
            raise E3ContractError('committed_end must be the prior chunk frontier')

        _tuple('block_tables', self.block_tables)
        if not self.block_tables:
            raise E3ContractError('admission needs its per-group block rows')
        for row in self.block_tables:
            _ints('block table row', row, minimum=0, unique=False)
        _tuple('slot_mappings', self.slot_mappings)
        for row in self.slot_mappings:
            _ints('slot mapping row', row, minimum=0)

        _ints('new_block_ids_to_zero', self.new_block_ids_to_zero, minimum=1)
        _tuple('cow_pairs', self.cow_pairs)
        seen_dst = set()
        for pair in self.cow_pairs:
            _tuple('cow pair', pair)
            if len(pair) != 2:
                raise E3ContractError('a cow pair is (source, destination)')
            source, destination = pair
            _int('cow source', source, 1)
            _int('cow destination', destination, 1)
            if source == destination:
                raise E3ContractError('a cow pair must copy to another block')
            if destination in seen_dst:
                raise E3ContractError('duplicate cow destination')
            seen_dst.add(destination)

        expected = transaction.chunk_end - transaction.chunk_start
        _ints('input_ids', self.input_ids, minimum=0, unique=False)
        if len(self.input_ids) != expected:
            raise E3ContractError('input_ids must cover exactly the chunk')
        _ints('ngram_context', self.ngram_context, minimum=0, unique=False)
        if type(self.sparse_active) is not bool:
            raise E3ContractError('sparse_active must be bool')

    @property
    def chunk_tokens(self):
        return self.transaction.chunk_end - self.transaction.chunk_start

    @property
    def remaining_tokens(self):
        """Prompt tokens still unprocessed after this chunk (>= 1)."""
        return self.prompt_end - self.transaction.chunk_end

    @property
    def terminal(self):
        """This chunk hands the request to stock for its final prompt token
        (coverage-soak-fix: or a geometry tail's rows): it ends at ``native_end``.

        Only a terminal chunk releases the worker's request row; a chunk that
        is followed by another native chunk keeps the row for ``update()``.
        """
        return self.transaction.chunk_end == self.native_end

    @property
    def align_accepted(self):
        """The worker's private align precondition; always exactly 1."""
        return 1

    def align_seed(self, block_size):
        """``(committed_end - 1) // block_size``; private to the worker."""
        _int('block_size', block_size, 1)
        return (self.committed_end - 1) // block_size

    def flat_blocks(self):
        """All entries in group order, including nulls; never a lease-ID list."""
        return tuple(block for row in self.block_tables for block in row)

    def __repr__(self):
        groups = len(self.block_tables)
        blocks = len(self.flat_blocks())
        return (f'AdmissionRecord(txn_seq={self.transaction.txn_seq}, '
                f'prompt_end={self.prompt_end}, state_epoch={self.state_epoch}, '
                f'committed_end={self.committed_end}, groups={groups}, '
                f'blocks={blocks}, chunk_tokens={self.chunk_tokens}, '
                f'sparse_active={self.sparse_active}, native_end={self.native_end})')


@dataclass(frozen=True, repr=False)
class SourceLeaseDescriptor:
    """Immutable engine-owned source lease handed to a worker transport.

    The engine retains the real allocator block objects; this record is the
    serialisable half of that ownership. It is not a ``source_pinned=True``
    claim: the worker must call :meth:`validate_frame` with the admission it
    actually opened, which re-checks the transaction, the state epoch and every
    physical page against that frame's own block rows. Anything else is
    refused, so a stale or widened descriptor cannot authorise an attention
    read.
    """
    transaction: TransactionId
    state_epoch: int
    block_ids: tuple
    consumer_ranks: tuple

    def __post_init__(self):
        if type(self.transaction) is not TransactionId:
            raise E3ContractError('source lease needs a TransactionId')
        _int('state_epoch', self.state_epoch, 1)
        _ints('block_ids', self.block_ids, minimum=1)
        if not self.block_ids:
            raise E3ContractError('a source lease covers at least one page')
        if self.block_ids != tuple(sorted(set(self.block_ids))):
            raise E3ContractError('source lease pages must be unique and ordered')
        _ints('consumer_ranks', self.consumer_ranks, minimum=0)
        if not self.consumer_ranks:
            raise E3ContractError('a source lease names its consumer ranks')
        if self.consumer_ranks != tuple(sorted(set(self.consumer_ranks))):
            raise E3ContractError('source lease ranks must be unique and ordered')

    def validate_frame(self, admission, ranks):
        """Prove the descriptor against the worker's admitted chunk.

        ``admission`` is the frame the worker actually opened: the engine's
        :class:`AdmissionRecord` or the lane's frozen ``ChunkAdmission``. Both
        carry the same transaction, state epoch, per-group rows and CoW pairs.
        Every lease page must appear in those rows (a retained CoW source is
        named by ``cow_pairs`` after its destination replaced it in the row).
        Returns ``True``; any difference raises before a page can be read.
        """
        block_tables = getattr(admission, 'block_tables', None)
        cow_pairs = getattr(admission, 'cow_pairs', None)
        if (type(block_tables) is not tuple or type(cow_pairs) is not tuple
                or type(getattr(admission, 'transaction', None)) is not TransactionId
                or type(getattr(admission, 'state_epoch', None)) is not int):
            raise E3ContractError('source lease validation needs the opened admission')
        if admission.transaction != self.transaction:
            raise E3ContractError('source lease belongs to another transaction')
        if admission.state_epoch != self.state_epoch:
            raise E3ContractError('source lease state epoch differs from the admission')
        page_set = set(self.block_ids)
        frame_pages = {block for row in block_tables for block in row if block > 0}
        # A retained CoW source has already been replaced in the request row
        # by its destination; the admitted frame names it in ``cow_pairs``.
        frame_pages.update(block for pair in cow_pairs for block in pair)
        if not page_set <= frame_pages:
            raise E3ContractError('source lease page is outside the admitted rows')
        if not set(ranks) <= set(self.consumer_ranks):
            raise E3ContractError('source lease does not name every consumer rank')
        return True

    def __repr__(self):
        return (f'SourceLeaseDescriptor(txn_seq={self.transaction.txn_seq}, '
                f'state_epoch={self.state_epoch}, pages={len(self.block_ids)}, '
                f'ranks={len(self.consumer_ranks)})')


# Step B: the group sizes a transaction can latch; 1 is Step A's command stream.
LAYER_GROUP_SIZES = (1, 2, 4)


@dataclass(frozen=True, repr=False)
class SegmentDescriptor:
    """One bounded worker operation. No transaction-wide forward exists.

    ``depends_on`` names the tickets that must be complete before this segment
    may run. ``seq`` is the explicit PP transfer sequence and exists only for
    the two PP kinds; it is never derived from ``op_seq``.
    """
    transaction: TransactionId
    kind: SegmentKind
    stage: int
    depends_on: tuple = ()
    layer_start: int | None = None
    layer_end: int | None = None
    seq: int | None = None
    # Step B: the transaction's latched group size; 1 is Step A's one layer per segment.
    layer_group: int = 1

    def __post_init__(self):
        if type(self.transaction) is not TransactionId:
            raise E3ContractError('segment needs a TransactionId')
        _kind('kind', self.kind, SegmentKind)
        _int('stage', self.stage, 0)
        if self.stage not in (0, 1):
            raise E3ContractError('wire-v1 supports exactly two stages')
        allowed = _KIND_STAGES[self.kind]
        if allowed is not None and self.stage not in allowed:
            raise E3ContractError(
                f'{self.kind.value} is not valid on stage {self.stage}')
        _tuple('depends_on', self.depends_on)
        for ticket in self.depends_on:
            if type(ticket) is not OperationTicket:
                raise E3ContractError('depends_on needs OperationTickets')
            if ticket.transaction != self.transaction:
                raise E3ContractError('dependency belongs to another transaction')
        if len(set(self.depends_on)) != len(self.depends_on):
            raise E3ContractError('duplicate dependency ticket')
        if self.kind is SegmentKind.LAYER:
            start = _int('layer_start', self.layer_start, 0)
            end = _int('layer_end', self.layer_end, 1)
            if self.layer_group != 1:
                if not start < end <= start + self.layer_group:
                    raise E3ContractError('a layer group covers 1..size consecutive layers')
            elif end != start + 1:
                raise E3ContractError('one layer per segment in the first cut')
        else:
            if self.layer_start is not None or self.layer_end is not None:
                raise E3ContractError('only layer segments carry layer ids')
        if self.kind in _PP_KINDS:
            _int('seq', self.seq, 0)
        elif self.seq is not None:
            raise E3ContractError('only PP segments carry a transfer seq')
        if self.layer_group != 1 and (self.kind is not SegmentKind.LAYER
                                      or type(self.layer_group) is not int
                                      or self.layer_group not in LAYER_GROUP_SIZES):
            raise E3ContractError('only a layer segment carries a known group size')

    @property
    def is_pp(self):
        return self.kind in _PP_KINDS

    @property
    def layers(self):
        return () if self.kind is not SegmentKind.LAYER else tuple(range(self.layer_start, self.layer_end))

    def satisfied(self, completed_tickets):
        """True when every dependency is present in the completed set."""
        done = set(completed_tickets)
        return all(ticket in done for ticket in self.depends_on)

    def __repr__(self):
        fields = [f'txn_seq={self.transaction.txn_seq}', f'kind={self.kind.value}',
                  f'stage={self.stage}']
        if self.kind is SegmentKind.LAYER:
            fields.append(f'layer={self.layer_start}')
        if self.kind in _PP_KINDS:
            fields.append(f'seq={self.seq}')
        fields.append(f'deps={len(self.depends_on)}')
        return 'SegmentDescriptor(' + ', '.join(fields) + ')'


@dataclass(frozen=True)
class CompletionCursor:
    """Where the worker's bounded operation cursor actually stands."""
    phase: str
    layer: int
    bucket: int

    def __post_init__(self):
        if type(self.phase) is not str or self.phase not in CURSOR_PHASES:
            raise E3ContractError('unknown completion phase')
        _int('layer', self.layer, 0)
        _int('bucket', self.bucket, 0)


@dataclass(frozen=True, repr=False)
class WorkerCompletionReport:
    """One rank's host-queried completion state for one transaction.

    ``completed_sequence`` is the worker's queried watermark, not the highest
    submitted sequence. Proofs are real completion records; an operation that
    is merely acknowledged is in ``pending_op_seqs``.
    """
    transaction: TransactionId
    worker_rank: int
    completed_op_seqs: tuple
    pending_op_seqs: tuple
    completed_sequence: int
    proofs: tuple
    cursor: CompletionCursor
    state_epoch: int
    host_submit_ns: tuple = ()
    failed_code: str | None = None

    def __post_init__(self):
        transaction = self.transaction
        if type(transaction) is not TransactionId:
            raise E3ContractError('report needs a TransactionId')
        _int('worker_rank', self.worker_rank, 0)
        _ints('completed_op_seqs', self.completed_op_seqs, minimum=0)
        _ints('pending_op_seqs', self.pending_op_seqs, minimum=0)
        if self.completed_op_seqs != tuple(sorted(self.completed_op_seqs)):
            raise E3ContractError('completed_op_seqs must be ordered')
        if self.pending_op_seqs != tuple(sorted(self.pending_op_seqs)):
            raise E3ContractError('pending_op_seqs must be ordered')
        done = set(self.completed_op_seqs)
        if done.intersection(self.pending_op_seqs):
            raise E3ContractError('an op_seq cannot be both complete and pending')
        _int('completed_sequence', self.completed_sequence, 0)
        _tuple('proofs', self.proofs)
        tickets = set()
        for proof in self.proofs:
            if type(proof) is not CompletionProof:
                raise E3ContractError('proofs must be CompletionProof records')
            if proof.completion_seq > self.completed_sequence:
                raise E3ContractError('watermark is behind a completion proof')
            ticket = proof.ticket
            if ticket.transaction != transaction:
                raise E3ContractError('proof belongs to another transaction')
            if ticket.worker_rank != self.worker_rank:
                raise E3ContractError('proof belongs to another worker')
            if ticket.op_seq not in done:
                raise E3ContractError('proof without a completed op_seq')
            key = (ticket, proof.kind)
            if key in tickets:
                raise E3ContractError('duplicate completion proof')
            tickets.add(key)
        if type(self.cursor) is not CompletionCursor:
            raise E3ContractError('report needs a CompletionCursor')
        _int('state_epoch', self.state_epoch, 1)
        _tuple('host_submit_ns', self.host_submit_ns)
        accepted_ops = done.union(self.pending_op_seqs)
        last = -1
        for entry in self.host_submit_ns:
            _tuple('host_submit_ns entry', entry)
            if len(entry) != 2:
                raise E3ContractError('host_submit_ns entries are (op_seq, ns)')
            op_seq, ns = entry
            _int('host op_seq', op_seq, 0)
            _int('host_submit_ns', ns, 0)
            if op_seq not in accepted_ops:
                raise E3ContractError('submit time for an unaccepted operation')
            if op_seq <= last:
                raise E3ContractError('host submit times must be ordered')
            last = op_seq
        if self.failed_code is not None:
            if type(self.failed_code) is not str or self.failed_code not in FAILED_CODES:
                raise E3ContractError('unknown failed_code')

    @property
    def failed(self):
        return self.failed_code is not None

    def has_proof(self, kind, role=None):
        if type(kind) is not CompletionKind:
            raise E3ContractError('kind must be CompletionKind')
        return any(p.kind is kind and (role is None or p.ticket.dependency_role is role)
                   for p in self.proofs)

    def state_written(self):
        return self.has_proof(CompletionKind.STATE_WRITE, DependencyRole.COMMIT)

    def __repr__(self):
        return (f'WorkerCompletionReport(txn_seq={self.transaction.txn_seq}, '
                f'rank={self.worker_rank}, completed={len(self.completed_op_seqs)}, '
                f'pending={len(self.pending_op_seqs)}, '
                f'watermark={self.completed_sequence}, proofs={len(self.proofs)}, '
                f'phase={self.cursor.phase}, layer={self.cursor.layer}, '
                f'state_epoch={self.state_epoch}, failed_code={self.failed_code})')


# -- held-g L4: lean completion reports -------------------------------------
#: The per-transaction reply format, chosen by the engine at OPEN. ``full`` is
#: the earlier behaviour: every reply carries the rank's whole report.
REPLIES = ('full', 'lean')
LEAN_DIGEST_SIZE = 32
LEAN_DIGEST_START = bytes(LEAN_DIGEST_SIZE)


def lean_digest(prior, bases, completed_op_seqs, proofs, host_submit_ns):
    """The next link of one rank's lean report chain; both sides compute it.

    It covers the delta's bases (completed, proof and submit-time counts), its
    op_seqs, each proof's (op_seq, kind, completion_seq) and its submit times,
    in the order sent. The engine re-derives the chain from what it applied,
    so a gap, replay, reorder or substitution changes it.
    """
    if type(prior) is not bytes or len(prior) != LEAN_DIGEST_SIZE:
        raise E3ContractError('lean digest link must be 32 bytes')
    body = repr((tuple(bases), completed_op_seqs,
                 tuple((p.ticket.op_seq, p.kind.value, p.completion_seq) for p in proofs),
                 host_submit_ns))
    return hashlib.sha256(prior + body.encode()).digest()


@dataclass(frozen=True, repr=False)
class LeanCompletionReport:
    """One rank's completion delta in a ``replies='lean'`` transaction (held-g L4).

    ``completed_op_seqs``, ``proofs`` and ``host_submit_ns`` hold only what the
    rank had not reported before in this transaction. ``pending_op_seqs``,
    ``completed_sequence``, ``cursor`` and ``failed_code`` are its whole current
    values, as in a :class:`WorkerCompletionReport`. ``base_completed``,
    ``base_proofs`` and ``base_hosts`` count what it reported before, and
    ``digest`` is the chain end after this delta (:func:`lean_digest`). The
    engine extends its own accumulated report with :class:`LeanStream`.
    """
    transaction: TransactionId
    worker_rank: int
    base_completed: int
    base_proofs: int
    base_hosts: int
    completed_op_seqs: tuple
    pending_op_seqs: tuple
    completed_sequence: int
    proofs: tuple
    cursor: CompletionCursor
    state_epoch: int
    digest: bytes
    host_submit_ns: tuple = ()
    failed_code: str | None = None

    def __post_init__(self):
        transaction = self.transaction
        if type(transaction) is not TransactionId:
            raise E3ContractError('lean report needs a TransactionId')
        _int('worker_rank', self.worker_rank, 0)
        _int('base_completed', self.base_completed, 0)
        _int('base_proofs', self.base_proofs, 0)
        _int('base_hosts', self.base_hosts, 0)
        _ints('completed_op_seqs', self.completed_op_seqs, minimum=0)
        _ints('pending_op_seqs', self.pending_op_seqs, minimum=0)
        if self.completed_op_seqs != tuple(sorted(self.completed_op_seqs)):
            raise E3ContractError('completed_op_seqs must be ordered')
        if self.pending_op_seqs != tuple(sorted(self.pending_op_seqs)):
            raise E3ContractError('pending_op_seqs must be ordered')
        if set(self.completed_op_seqs).intersection(self.pending_op_seqs):
            raise E3ContractError('an op_seq cannot be both complete and pending')
        _int('completed_sequence', self.completed_sequence, 0)
        _tuple('proofs', self.proofs)
        keys = set()
        for proof in self.proofs:
            if type(proof) is not CompletionProof:
                raise E3ContractError('proofs must be CompletionProof records')
            if proof.completion_seq > self.completed_sequence:
                raise E3ContractError('watermark is behind a completion proof')
            ticket = proof.ticket
            if ticket.transaction != transaction:
                raise E3ContractError('proof belongs to another transaction')
            if ticket.worker_rank != self.worker_rank:
                raise E3ContractError('proof belongs to another worker')
            key = (ticket.op_seq, proof.kind)
            if key in keys:
                raise E3ContractError('duplicate completion proof')
            keys.add(key)
        if type(self.cursor) is not CompletionCursor:
            raise E3ContractError('report needs a CompletionCursor')
        _int('state_epoch', self.state_epoch, 1)
        if type(self.digest) is not bytes or len(self.digest) != LEAN_DIGEST_SIZE:
            raise E3ContractError('lean report digest must be 32 bytes')
        _tuple('host_submit_ns', self.host_submit_ns)
        last = -1
        for entry in self.host_submit_ns:
            _tuple('host_submit_ns entry', entry)
            if len(entry) != 2:
                raise E3ContractError('host_submit_ns entries are (op_seq, ns)')
            _int('host op_seq', entry[0], 0)
            _int('host_submit_ns', entry[1], 0)
            if entry[0] <= last:
                raise E3ContractError('host submit times must be ordered')
            last = entry[0]
        if self.failed_code is not None:
            if type(self.failed_code) is not str or self.failed_code not in FAILED_CODES:
                raise E3ContractError('unknown failed_code')

    def bases(self):
        return self.base_completed, self.base_proofs, self.base_hosts

    def empty(self):
        """No new completion, proof or submit time."""
        return not (self.completed_op_seqs or self.proofs or self.host_submit_ns)

    def __repr__(self):
        return (f'LeanCompletionReport(txn_seq={self.transaction.txn_seq}, '
                f'rank={self.worker_rank}, '
                f'base={self.base_completed}/{self.base_proofs}/{self.base_hosts}, '
                f'completed=+{len(self.completed_op_seqs)}, '
                f'pending={len(self.pending_op_seqs)}, '
                f'watermark={self.completed_sequence}, proofs=+{len(self.proofs)}, '
                f'state_epoch={self.state_epoch}, failed_code={self.failed_code})')


def _accumulated(**fields):
    """A WorkerCompletionReport built from fields LeanStream.extend has checked.

    ``__post_init__`` walks the whole report; the stream instead checked each
    new item against everything applied before, which keeps the same
    invariants (the held-g tests run the lean probes with the validating
    constructor in its place and compare).
    """
    report = object.__new__(WorkerCompletionReport)
    for name in WorkerCompletionReport.__dataclass_fields__:
        object.__setattr__(report, name, fields[name])
    return report


class LeanStream:
    """The engine's accumulated view of one rank's lean reports in one transaction.

    :meth:`extend` checks one delta in O(delta) against everything applied
    before and returns the cumulative :class:`WorkerCompletionReport`, which is
    what every consumer of a full report keeps reading. A refused delta leaves
    the stream unchanged.
    """
    __slots__ = ('report', 'completed', 'proof_keys', 'hosts', 'digest', 'state_written',
                 'closed')

    def __init__(self):
        self.report = None
        self.completed = set()
        self.proof_keys = set()
        self.hosts = 0
        self.digest = LEAN_DIGEST_START
        self.state_written = False
        self.closed = False

    def end(self):
        """(completed, proof and submit-time counts, chain end): what the release compares."""
        return len(self.completed), len(self.proof_keys), self.hosts, self.digest

    def extend(self, lean):
        if type(lean) is not LeanCompletionReport:
            raise E3ContractError('lean stream needs a LeanCompletionReport')
        if self.closed:
            raise E3ContractError('a full report closed this lean stream')
        # The delta's own invariants again on this side of the wire (O(delta)).
        lean.__post_init__()
        prior = self.report
        if prior is not None and (lean.transaction != prior.transaction
                                  or lean.worker_rank != prior.worker_rank
                                  or lean.state_epoch != prior.state_epoch):
            raise E3ContractError('lean report changes the stream identity')
        if lean.bases() != (len(self.completed), len(self.proof_keys), self.hosts):
            raise E3ContractError('lean report does not continue the applied stream')
        digest = lean_digest(self.digest, lean.bases(), lean.completed_op_seqs, lean.proofs,
                             lean.host_submit_ns)
        if digest != lean.digest:
            raise E3ContractError('lean report chain differs from the applied stream')
        completed = self.completed
        new = lean.completed_op_seqs
        if not completed.isdisjoint(new):
            raise E3ContractError('lean report repeats a reported completion')
        if prior is not None and lean.completed_sequence < prior.completed_sequence:
            raise E3ContractError('completion report regressed')
        pending = lean.pending_op_seqs
        new_set = set(new)
        if not completed.isdisjoint(pending):
            raise E3ContractError('an op_seq cannot be both complete and pending')
        if prior is not None:
            for op in prior.pending_op_seqs:
                if op not in completed and op not in new_set and op not in pending:
                    raise E3ContractError('a pending operation left the report')
        state = self.state_written
        keys = []
        for proof in lean.proofs:
            op = proof.ticket.op_seq
            if op not in completed and op not in new_set:
                raise E3ContractError('proof without a completed op_seq')
            key = (op, proof.kind)
            if key in self.proof_keys:
                raise E3ContractError('duplicate completion proof')
            keys.append(key)
            if (proof.kind is CompletionKind.STATE_WRITE
                    and proof.ticket.dependency_role is DependencyRole.COMMIT):
                state = True
        hosts = lean.host_submit_ns
        if hosts:
            if prior is not None and prior.host_submit_ns and hosts[0][0] <= prior.host_submit_ns[-1][0]:
                raise E3ContractError('host submit times must be ordered')
            for op, _ns in hosts:
                if op not in completed and op not in new_set and op not in pending:
                    raise E3ContractError('submit time for an unaccepted operation')
        base = () if prior is None else prior.completed_op_seqs
        merged = _accumulated(
            transaction=lean.transaction, worker_rank=lean.worker_rank,
            completed_op_seqs=tuple(sorted(base + new)) if new else base,
            pending_op_seqs=pending, completed_sequence=lean.completed_sequence,
            proofs=(() if prior is None else prior.proofs) + lean.proofs,
            cursor=lean.cursor, state_epoch=lean.state_epoch,
            host_submit_ns=(() if prior is None else prior.host_submit_ns) + hosts,
            failed_code=lean.failed_code)
        completed.update(new)
        self.proof_keys.update(keys)
        self.hosts += len(hosts)
        self.digest = digest
        self.state_written = state
        self.report = merged
        return merged


@dataclass(frozen=True, repr=False)
class BootstrapCommand:
    """Bind every worker incarnation to the engine owner, exactly once.

    The command carries no worker-local context/SM/pool identity: each worker
    answers from its own retained decode storage and its own route controller.
    It is idempotent for the same engine epoch; a worker already bound to a
    different epoch refuses instead of mutating a live transaction.
    """
    engine_epoch: int
    ranks: tuple

    def __post_init__(self):
        _int('engine_epoch', self.engine_epoch, 1)
        _ints('ranks', self.ranks, minimum=0, allow_empty=False)
        if self.ranks != tuple(sorted(self.ranks)):
            raise E3ContractError('bootstrap ranks must be ordered')

    def __repr__(self):
        return (f'BootstrapCommand(engine_epoch={self.engine_epoch}, '
                f'ranks={len(self.ranks)})')


@dataclass(frozen=True, repr=False)
class BootstrapAck:
    """One worker's engine-owner binding and retained decode storage identity.

    ``lane`` is the persistent decode storage retained at worker startup, not a
    request lane.  ``decode_replay_watermark`` is the worker's own queried
    replay census under that storage; the engine seeds route transitions with
    it rather than guessing.  ``bound=False`` is a refusal, never a success.
    """
    engine_epoch: int
    worker_rank: int
    incarnation: str = field(repr=False)
    lane: LaneId
    route_generation: int = 0
    decode_replay_watermark: int = 0
    bound: bool = False

    def __post_init__(self):
        _int('engine_epoch', self.engine_epoch, 1)
        _int('worker_rank', self.worker_rank, 0)
        _text('incarnation', self.incarnation)
        if type(self.lane) is not LaneId:
            raise E3ContractError('bootstrap ack needs the retained decode LaneId')
        if self.lane.role is not LaneRole.DECODE:
            raise E3ContractError('bootstrap lane must be decode storage')
        if self.lane.worker_rank != self.worker_rank:
            raise E3ContractError('bootstrap lane belongs to another worker')
        _int('route_generation', self.route_generation, 0)
        _int('decode_replay_watermark', self.decode_replay_watermark, 0)
        if type(self.bound) is not bool:
            raise E3ContractError('bound must be bool')

    def matches(self, command):
        if type(command) is not BootstrapCommand:
            raise E3ContractError('matches() needs a BootstrapCommand')
        return (self.engine_epoch == command.engine_epoch
                and self.worker_rank in command.ranks)

    def __repr__(self):
        return (f'BootstrapAck(engine_epoch={self.engine_epoch}, '
                f'rank={self.worker_rank}, incarnation=<len {len(self.incarnation)}>, '
                f'storage_epoch={self.lane.storage_epoch if self.lane else None}, '
                f'route_generation={self.route_generation}, '
                f'watermark={self.decode_replay_watermark}, bound={self.bound})')


@dataclass(frozen=True, repr=False)
class TransitionCommand:
    """Green/full route transition, fenced on the exact engine/route epoch."""
    engine_epoch: int
    expected_route_generation: int
    route_generation: int
    target_bank: int
    storage_epochs: tuple
    decode_replay_watermarks: tuple

    def __post_init__(self):
        _int('engine_epoch', self.engine_epoch, 0)
        _int('expected_route_generation', self.expected_route_generation, 0)
        _int('route_generation', self.route_generation, 1)
        if self.route_generation != self.expected_route_generation + 1:
            raise E3ContractError('a transition advances the route generation by one')
        _int('target_bank', self.target_bank, 0)
        if self.target_bank not in ROUTE_BANKS:
            raise E3ContractError('unknown route bank')
        _ints('storage_epochs', self.storage_epochs, minimum=1, allow_empty=False,
              unique=False)
        _ints('decode_replay_watermarks', self.decode_replay_watermarks,
              minimum=0, unique=False)
        if len(self.decode_replay_watermarks) != len(self.storage_epochs):
            raise E3ContractError('one replay watermark per rank is required')

    def __repr__(self):
        return (f'TransitionCommand(engine_epoch={self.engine_epoch}, '
                f'expected_route_generation={self.expected_route_generation}, '
                f'route_generation={self.route_generation}, '
                f'target_bank={self.target_bank}, ranks={len(self.storage_epochs)})')


@dataclass(frozen=True, repr=False)
class TransitionAck:
    """A worker's applied transition. Identity echoes the command exactly.

    The replay watermark is the one field that is *not* an echo: it is this
    rank's own queried census, so it may exceed the commanded floor.
    """
    engine_epoch: int
    expected_route_generation: int
    route_generation: int
    target_bank: int
    storage_epochs: tuple
    applied_route_generation: int
    applied_bank: int
    applied_storage_epochs: tuple
    worker_rank: int = 0
    # This rank's queried replay census under the persistent decode storage.
    decode_replay_watermark: int = 0
    drained_streams: int = 0

    def __post_init__(self):
        _int('engine_epoch', self.engine_epoch, 0)
        _int('expected_route_generation', self.expected_route_generation, 0)
        _int('route_generation', self.route_generation, 1)
        if self.route_generation != self.expected_route_generation + 1:
            raise E3ContractError('a transition advances the route generation by one')
        _int('target_bank', self.target_bank, 0)
        if self.target_bank not in ROUTE_BANKS:
            raise E3ContractError('unknown route bank')
        _ints('storage_epochs', self.storage_epochs, minimum=1, allow_empty=False,
              unique=False)
        _int('worker_rank', self.worker_rank, 0)
        _int('decode_replay_watermark', self.decode_replay_watermark, 0)
        _int('drained_streams', self.drained_streams, 0)
        _int('applied_route_generation', self.applied_route_generation, 1)
        _int('applied_bank', self.applied_bank, 0)
        _ints('applied_storage_epochs', self.applied_storage_epochs, minimum=1,
              allow_empty=False, unique=False)
        if self.applied_route_generation != self.route_generation:
            raise E3ContractError('ack did not apply the requested generation')
        if self.applied_bank != self.target_bank:
            raise E3ContractError('ack did not apply the requested bank')
        if self.applied_storage_epochs != self.storage_epochs:
            raise E3ContractError('ack did not apply the requested storage epochs')

    def matches(self, command):
        if type(command) is not TransitionCommand:
            raise E3ContractError('matches() needs a TransitionCommand')
        if self.worker_rank >= len(command.decode_replay_watermarks):
            return False
        # The commanded watermark is a floor: the worker answers with its own
        # queried census under the persistent storage, which may be ahead of
        # what the engine last saw.  A lower value would be a regression.
        commanded = command.decode_replay_watermarks[self.worker_rank]
        return (self.engine_epoch == command.engine_epoch
                and self.expected_route_generation == command.expected_route_generation
                and self.route_generation == command.route_generation
                and self.target_bank == command.target_bank
                and self.storage_epochs == command.storage_epochs
                and self.decode_replay_watermark >= commanded)

    def __repr__(self):
        return (f'TransitionAck(engine_epoch={self.engine_epoch}, '
                f'route_generation={self.route_generation}, '
                f'target_bank={self.target_bank}, '
                f'applied_bank={self.applied_bank}, '
                f'ranks={len(self.storage_epochs)})')


@dataclass(frozen=True)
class TransitionPending:
    """Successful RPC reply: this exact command has not switched this rank.

    A transient drain never travels as a worker exception over the owned FIFO
    transport. It is neither an applied transition nor a completion proof.
    """
    command: TransitionCommand
    worker_rank: int
    observed_route_generation: int

    def __post_init__(self):
        if type(self.command) is not TransitionCommand:
            raise E3ContractError('pending transition needs the exact command')
        _int('worker_rank', self.worker_rank, 0)
        _int('observed_route_generation', self.observed_route_generation, 0)
        if self.worker_rank >= len(self.command.storage_epochs):
            raise E3ContractError('pending rank is outside the transition cohort')
        if self.observed_route_generation != self.command.expected_route_generation:
            raise E3ContractError('pending rank has already changed its route')

    def matches(self, command):
        return type(command) is TransitionCommand and self.command == command


@dataclass(frozen=True, repr=False)
class OpenRequest:
    """Bind each worker lane to one admitted chunk before any segment runs."""
    admission: AdmissionRecord
    ranks: tuple
    expected_state_epoch: int
    max_outstanding: int
    # Held-g L4: the reply format of this transaction (REPLIES).
    replies: str = 'full'
    # Step A: this transaction runs each dense QSA layer as one bounded command.
    qsa_fused: bool = False
    # Step B: this transaction's group size; above 1 only with qsa_fused.
    layer_group: int = 1

    def __post_init__(self):
        if type(self.admission) is not AdmissionRecord:
            raise E3ContractError('open needs an AdmissionRecord')
        if type(self.replies) is not str or self.replies not in REPLIES:
            raise E3ContractError('unknown open reply format')
        if type(self.qsa_fused) is not bool:
            raise E3ContractError('open qsa_fused must be bool')
        if self.layer_group != 1 and (type(self.layer_group) is not int or not self.qsa_fused
                                      or self.layer_group not in LAYER_GROUP_SIZES):
            raise E3ContractError('open layer_group needs a known size and the fused selection')
        _ints('ranks', self.ranks, minimum=0, allow_empty=False)
        if self.ranks != tuple(sorted(self.ranks)):
            raise E3ContractError('open ranks must be ordered')
        _int('expected_state_epoch', self.expected_state_epoch, 1)
        if self.expected_state_epoch != self.admission.state_epoch:
            raise E3ContractError('open does not match the admitted state epoch')
        _int('max_outstanding', self.max_outstanding, 1)

    @property
    def transaction(self):
        return self.admission.transaction

    def __repr__(self):
        return (f'OpenRequest(txn_seq={self.admission.transaction.txn_seq}, '
                f'ranks={len(self.ranks)}, '
                f'state_epoch={self.expected_state_epoch}, '
                f'max_outstanding={self.max_outstanding})')


@dataclass(frozen=True, repr=False)
class OpenAck:
    """One worker's lane identity for the opened chunk; an ack is not work."""
    transaction: TransactionId
    lane: LaneId
    stage: int
    state_epoch: int
    pp_stages: int
    accepted: bool

    def __post_init__(self):
        if type(self.transaction) is not TransactionId:
            raise E3ContractError('open ack needs a TransactionId')
        if type(self.lane) is not LaneId:
            raise E3ContractError('open ack needs a LaneId')
        _int('stage', self.stage, 0)
        _int('state_epoch', self.state_epoch, 1)
        _int('pp_stages', self.pp_stages, 1)
        if self.stage >= self.pp_stages:
            raise E3ContractError('lane stage is outside the pipeline')
        if type(self.accepted) is not bool:
            raise E3ContractError('accepted must be bool')

    def matches(self, request):
        if type(request) is not OpenRequest:
            raise E3ContractError('matches() needs an OpenRequest')
        return (self.transaction == request.transaction
                and self.lane.worker_rank in request.ranks
                and self.state_epoch == request.expected_state_epoch)

    def __repr__(self):
        return (f'OpenAck(txn_seq={self.transaction.txn_seq}, '
                f'rank={self.lane.worker_rank}, stage={self.stage}, '
                f'state_epoch={self.state_epoch}, accepted={self.accepted})')


@dataclass(frozen=True, repr=False)
class PeerBinding:
    """Bind a stage-0 sender's PP channel to the actual receiver lane/route.

    The engine learns the receiver lane only from that rank's :class:`OpenAck`,
    so the binding is a separate bounded command. The worker calls
    ``PrefillPPChannel.bind_consumer`` with exactly these values; no lane id or
    route generation is inferred from local state, a rank number or a
    watermark.
    """
    transaction: TransactionId
    sender_rank: int
    receiver_lane: LaneId
    route_generation: int

    def __post_init__(self):
        if type(self.transaction) is not TransactionId:
            raise E3ContractError('peer binding needs a TransactionId')
        _int('sender_rank', self.sender_rank, 0)
        if type(self.receiver_lane) is not LaneId:
            raise E3ContractError('peer binding needs the receiver LaneId')
        if self.receiver_lane.role is not LaneRole.PREFILL:
            raise E3ContractError('PP peer must be a prefill lane')
        if self.receiver_lane.worker_rank == self.sender_rank:
            raise E3ContractError('a sender cannot be its own PP consumer')
        _int('route_generation', self.route_generation, 0)

    @property
    def receiver_rank(self):
        return self.receiver_lane.worker_rank

    def __repr__(self):
        return (f'PeerBinding(txn_seq={self.transaction.txn_seq}, '
                f'sender={self.sender_rank}, receiver={self.receiver_rank}, '
                f'storage_epoch={self.receiver_lane.storage_epoch}, '
                f'route_generation={self.route_generation})')


@dataclass(frozen=True, repr=False)
class PeerBindingAck:
    """The sender applied exactly the requested peer/route binding."""
    transaction: TransactionId
    sender_rank: int
    receiver_lane: LaneId
    route_generation: int
    bound: bool

    def __post_init__(self):
        if type(self.transaction) is not TransactionId:
            raise E3ContractError('peer binding ack needs a TransactionId')
        _int('sender_rank', self.sender_rank, 0)
        if type(self.receiver_lane) is not LaneId:
            raise E3ContractError('peer binding ack needs the receiver LaneId')
        _int('route_generation', self.route_generation, 0)
        if type(self.bound) is not bool:
            raise E3ContractError('bound must be bool')

    def matches(self, binding):
        if type(binding) is not PeerBinding:
            raise E3ContractError('matches() needs a PeerBinding')
        return (self.transaction == binding.transaction
                and self.sender_rank == binding.sender_rank
                and self.receiver_lane == binding.receiver_lane
                and self.route_generation == binding.route_generation)

    def __repr__(self):
        return (f'PeerBindingAck(txn_seq={self.transaction.txn_seq}, '
                f'sender={self.sender_rank}, receiver={self.receiver_lane.worker_rank}, '
                f'route_generation={self.route_generation}, bound={self.bound})')


@dataclass(frozen=True, repr=False)
class PPReceiptDelivery:
    """Engine-routed, already-validated consumer receipts for one sender.

    The engine forwards only receipts whose queried PP_CONSUMER proof came from
    the paired receiver lane; the sender revalidates transaction, stage,
    sequence and route through its bound channel before storing anything.
    """
    transaction: TransactionId
    sender_rank: int
    receipts: tuple

    def __post_init__(self):
        if type(self.transaction) is not TransactionId:
            raise E3ContractError('receipt delivery needs a TransactionId')
        _int('sender_rank', self.sender_rank, 0)
        _tuple('receipts', self.receipts)
        if not self.receipts:
            raise E3ContractError('receipt delivery must carry a receipt')
        seen = set()
        for receipt in self.receipts:
            if type(receipt) is not PPConsumerReceipt:
                raise E3ContractError('receipts must be PPConsumerReceipt records')
            if receipt.transaction != self.transaction:
                raise E3ContractError('receipt belongs to another transaction')
            if receipt.seq in seen:
                raise E3ContractError('duplicate receipt sequence')
            seen.add(receipt.seq)

    @property
    def seqs(self):
        return tuple(receipt.seq for receipt in self.receipts)

    def __repr__(self):
        return (f'PPReceiptDelivery(txn_seq={self.transaction.txn_seq}, '
                f'sender={self.sender_rank}, seqs={self.seqs})')


@dataclass(frozen=True, repr=False)
class PPReceiptAck:
    """The sender accepted the receipts; a duplicate receipt is idempotent."""
    transaction: TransactionId
    sender_rank: int
    seqs: tuple
    accepted: bool

    def __post_init__(self):
        if type(self.transaction) is not TransactionId:
            raise E3ContractError('receipt ack needs a TransactionId')
        _int('sender_rank', self.sender_rank, 0)
        _ints('seqs', self.seqs, minimum=0, allow_empty=False)
        if type(self.accepted) is not bool:
            raise E3ContractError('accepted must be bool')

    def matches(self, delivery):
        if type(delivery) is not PPReceiptDelivery:
            raise E3ContractError('matches() needs a PPReceiptDelivery')
        return (self.transaction == delivery.transaction
                and self.sender_rank == delivery.sender_rank
                and self.seqs == delivery.seqs)

    def __repr__(self):
        return (f'PPReceiptAck(txn_seq={self.transaction.txn_seq}, '
                f'sender={self.sender_rank}, seqs={self.seqs}, '
                f'accepted={self.accepted})')


@dataclass(frozen=True)
class QsaScheduleShape:
    """Queried local submission schedule; never token indices or completion.

    Counts describe bounded host submissions, not CUDA API/activity counts.
    Keeping these counts separate permits different local selections only
    when their actual lockstep submission profiles agree.
    """
    route: str
    bucket_submissions: tuple
    bucket_waves: tuple

    def __post_init__(self):
        if type(self.route) is not str or self.route not in ('dense_dma', 'sparse_dma', 'empty'):
            raise E3ContractError('unknown QSA schedule route')
        _ints('bucket submissions', self.bucket_submissions, minimum=1, unique=False)
        _ints('bucket waves', self.bucket_waves, minimum=1, unique=False)
        if len(self.bucket_submissions) != len(self.bucket_waves):
            raise E3ContractError('QSA schedule bucket vectors disagree')
        maximum_buckets = (MAX_LAYER_PAGES + ARENA_PAGES - 1) // ARENA_PAGES
        maximum_waves = (ARENA_ROWS + MAX_DESCRIPTORS - 1) // MAX_DESCRIPTORS
        if len(self.bucket_waves) > maximum_buckets:
            raise E3ContractError('QSA schedule exceeds existing layer page bound')
        if self.route == 'empty':
            if self.bucket_waves:
                raise E3ContractError('empty QSA schedule carries bucket work')
        elif not self.bucket_waves:
            raise E3ContractError('nonempty QSA schedule needs bucket work')
        if any(w > maximum_waves for w in self.bucket_waves):
            raise E3ContractError('QSA schedule exceeds existing per-bucket wave bound')
        # Intermediate streamed waves yield; the final wave and attention
        # always share one pull. There is no event-timing-dependent extra pull.
        if self.bucket_submissions != self.bucket_waves:
            raise E3ContractError('QSA schedule differs from copy-wave submissions')
        if self.route == 'dense_dma' and any(s != 1 or w != 1 for s, w in
                zip(self.bucket_submissions, self.bucket_waves)):
            raise E3ContractError('dense QSA bucket is one bounded submission')


@dataclass(frozen=True, repr=False)
class QsaProfileQuery:
    """Read-only query for the exact submitted layer boundary."""
    transaction: TransactionId
    state_epoch: int
    stage: int
    layer: int
    boundary_op_seq: int
    ranks: tuple

    def __post_init__(self):
        if type(self.transaction) is not TransactionId:
            raise E3ContractError('QSA profile query needs a TransactionId')
        _int('state_epoch', self.state_epoch, 1)
        _int('stage', self.stage, 0)
        if self.stage not in (0, 1):
            raise E3ContractError('QSA profile query names an unknown stage')
        _int('layer', self.layer, 0)
        _int('boundary_op_seq', self.boundary_op_seq, 0)
        _ints('profile ranks', self.ranks, minimum=0, allow_empty=False)
        if self.ranks != tuple(sorted(self.ranks)):
            raise E3ContractError('QSA profile ranks must be ordered')

    def __repr__(self):
        return (f'QsaProfileQuery(txn_seq={self.transaction.txn_seq}, '
                f'stage={self.stage}, layer={self.layer}, boundary={self.boundary_op_seq})')


@dataclass(frozen=True, repr=False)
class QsaProfileObservation:
    """Worker observation: pending is not agreement or copy completion."""
    query: QsaProfileQuery
    worker_rank: int
    status: str
    shape: QsaScheduleShape | None = None
    basis: str | None = None
    # Receipt-only fields: not part of profile equality/grant authorization.
    selected_count: int | None = field(default=None, compare=False)
    worker_host_ns: int = field(default=0, compare=False)

    def __post_init__(self):
        if type(self.query) is not QsaProfileQuery:
            raise E3ContractError('QSA profile observation needs its exact query')
        _int('worker_rank', self.worker_rank, 0)
        if type(self.status) is not str or self.status not in ('ready', 'pending', 'skipped'):
            raise E3ContractError('unknown QSA profile observation status')
        target = self.worker_rank in self.query.ranks
        if (self.status == 'skipped') == target:
            raise E3ContractError('QSA profile observation target status differs')
        if self.status == 'ready':
            if type(self.shape) is not QsaScheduleShape:
                raise E3ContractError('ready QSA profile needs a bounded shape')
            if self.basis not in ('queried-selection-copy', 'bound-dense-host-plan'):
                raise E3ContractError('ready QSA profile needs a queried plan basis')
            if self.basis == 'bound-dense-host-plan' and self.shape.route != 'dense_dma':
                raise E3ContractError('dense host plan cannot certify a sparse selection')
        elif self.shape is not None or self.basis is not None:
            raise E3ContractError('pending/skipped QSA profile cannot supply a shape')

    def __repr__(self):
        return (f'QsaProfileObservation(rank={self.worker_rank}, status={self.status}, '
                f'query={self.query!r})')


@dataclass(frozen=True)
class QsaProfileGrant:
    """Engine agreement binding; worker still checks its own live profile."""
    query: QsaProfileQuery
    shape: QsaScheduleShape
    # Step A: agreed at PREPARE (the query's boundary op) before the stage's
    # first collective; the layer runs as one command and never pauses.
    fused: bool = False

    def __post_init__(self):
        if type(self.query) is not QsaProfileQuery or type(self.shape) is not QsaScheduleShape:
            raise E3ContractError('QSA profile grant needs its exact query and shape')
        if type(self.fused) is not bool or (self.fused and self.shape.route != 'dense_dma'):
            raise E3ContractError('a fused QSA grant needs the dense route')


def qsa_profile_agreement(query, replies, ranks):
    """Return pending/agreed/unsupported without manufacturing completion.

    Malformed identities remain protocol errors. A valid profile mismatch is
    a named supported refusal for the driver's coordinated stop path.
    """
    if type(query) is not QsaProfileQuery:
        raise E3ContractError('QSA profile agreement needs a typed query')
    _ints('engine profile ranks', ranks, allow_empty=False)
    if ranks != tuple(sorted(ranks)):
        raise E3ContractError('engine profile ranks must be ordered')
    _tuple('profile replies', replies)
    if (len(replies) != len(ranks) or any(type(r) is not QsaProfileObservation for r in replies)
            or {r.worker_rank for r in replies} != set(ranks)):
        raise E3ContractError('QSA profile replies do not cover every rank')
    if any(r.query != query for r in replies):
        raise E3ContractError('QSA profile reply belongs to another boundary')
    if not set(query.ranks) <= set(ranks):
        raise E3ContractError('QSA profile targets are outside the engine cohort')
    targets = tuple(r for r in replies if r.worker_rank in query.ranks)
    if any(r.status == 'pending' for r in targets):
        return 'pending', None
    shapes = {r.shape for r in targets}
    if len(shapes) != 1:
        return 'unsupported-native-schedule-profile', None
    return 'agreed', QsaProfileGrant(query, next(iter(shapes)))


@dataclass(frozen=True, repr=False)
class PollRequest:
    """Typed bounded poll: it carries the epoch the query must still match."""
    transaction: TransactionId
    state_epoch: int
    budget: int

    def __post_init__(self):
        if type(self.transaction) is not TransactionId:
            raise E3ContractError('poll needs a TransactionId')
        _int('state_epoch', self.state_epoch, 1)
        _int('budget', self.budget, 1)

    def __repr__(self):
        return (f'PollRequest(txn_seq={self.transaction.txn_seq}, '
                f'state_epoch={self.state_epoch}, budget={self.budget})')


@dataclass(frozen=True, repr=False)
class SegmentCommand:
    """One wire submission: an immutable descriptor plus its accepted seq.

    ``ranks`` names the lanes that must execute. Every other worker returns a
    skipped acceptance without executing anything.

    ``source`` is the engine's actual :class:`SourceLeaseDescriptor`, attached
    only to a stage's first PREPARE command after open/bind. It is the transport
    half of the engine's real allocator hold; a worker refuses the command when
    it is missing, stale or foreign, before any GPU submission.
    """
    descriptor: SegmentDescriptor
    op_seq: int
    ranks: tuple
    source: SourceLeaseDescriptor | None = None
    profile: QsaProfileGrant | None = None

    def __post_init__(self):
        if type(self.descriptor) is not SegmentDescriptor:
            raise E3ContractError('command needs a SegmentDescriptor')
        _int('op_seq', self.op_seq, 0)
        _ints('ranks', self.ranks, minimum=0, allow_empty=False)
        if self.ranks != tuple(sorted(self.ranks)):
            raise E3ContractError('target ranks must be ordered')
        if self.source is not None:
            if type(self.source) is not SourceLeaseDescriptor:
                raise E3ContractError('command source must be a source lease')
            if self.kind is not SegmentKind.PREPARE:
                raise E3ContractError('only a PREPARE command carries a source lease')
            if self.source.transaction != self.transaction:
                raise E3ContractError('source lease belongs to another transaction')
            if not set(self.ranks) <= set(self.source.consumer_ranks):
                raise E3ContractError(
                    'source lease does not name every target rank')
        if self.profile is not None:
            if type(self.profile) is not QsaProfileGrant or self.kind is not SegmentKind.LAYER:
                raise E3ContractError('only a layer carries a typed QSA profile grant')
            query = self.profile.query
            if (query.transaction != self.transaction or query.stage != self.stage
                    or query.layer != self.descriptor.layer_end - 1
                    or query.ranks != self.ranks or query.boundary_op_seq >= self.op_seq):
                raise E3ContractError('QSA profile grant differs from the layer command')

    @property
    def transaction(self):
        return self.descriptor.transaction

    @property
    def kind(self):
        return self.descriptor.kind

    @property
    def stage(self):
        return self.descriptor.stage

    def __repr__(self):
        source = ('' if self.source is None
                  else f', source_pages={len(self.source.block_ids)}')
        return (f'SegmentCommand({self.descriptor!r}, op_seq={self.op_seq}, '
                f'ranks={len(self.ranks)}{source})')


@dataclass(frozen=True, repr=False)
class SegmentAcceptance:
    """One worker's answer to a command: executed, pending, or not addressed.

    ``pending`` means the bounded lane could not accept the work yet; the
    engine retries the identical command with the same ``op_seq``. ``skipped``
    means the lane was not a target and executed nothing.
    """
    transaction: TransactionId
    worker_rank: int
    kind: SegmentKind
    stage: int
    op_seq: int
    accepted: bool = False
    pending: bool = False
    skipped: bool = False
    cursor: CompletionCursor | None = None
    host_submit_ns: int = 0
    paused: bool = False
    # Host-queried completions ride along with the reply (piggybacked poll).
    report: WorkerCompletionReport | None = field(default=None, repr=False)
    profile_boundary: bool = False
    tail_boundary: bool = False
    # Step A: a PREPARE's per-layer dense schedule, ((layer, shape), ...).
    dense_profiles: tuple | None = field(default=None, repr=False)
    # Step A: this command's collectives on the lane TP communicator: (count, digest
    # over order, communicator and message shape). Peers must report the same value.
    collectives: tuple | None = field(default=None, repr=False)

    def __post_init__(self):
        if type(self.transaction) is not TransactionId:
            raise E3ContractError('acceptance needs a TransactionId')
        if self.dense_profiles is not None:
            if self.kind is not SegmentKind.PREPARE or not self.accepted:
                raise E3ContractError('only an accepted PREPARE carries dense profiles')
            _tuple('dense_profiles', self.dense_profiles)
            if (any(type(row) is not tuple or len(row) != 2 or type(row[0]) is not int
                    or type(row[1]) is not QsaScheduleShape for row in self.dense_profiles)
                    or [row[0] for row in self.dense_profiles]
                    != sorted({row[0] for row in self.dense_profiles})):
                raise E3ContractError('dense profiles must be ordered (layer, shape) rows')
        if self.collectives is not None:
            if (not self.accepted or type(self.collectives) is not tuple
                    or len(self.collectives) != 2 or type(self.collectives[0]) is not int
                    or self.collectives[0] < 0 or type(self.collectives[1]) is not str):
                raise E3ContractError('collective ledger must be an accepted (count, digest)')
        _int('worker_rank', self.worker_rank, 0)
        _kind('kind', self.kind, SegmentKind)
        _int('stage', self.stage, 0)
        _int('op_seq', self.op_seq, 0)
        if type(self.paused) is not bool or (self.paused and not self.accepted):
            raise E3ContractError('only an accepted operation may be paused')
        if self.paused and self.kind is not SegmentKind.LAYER:
            raise E3ContractError('only a QSA layer can pause at bucket credit')
        if (type(self.profile_boundary) is not bool or
                (self.profile_boundary and not self.paused)):
            raise E3ContractError('QSA profile boundary must be an accepted paused layer')
        if (type(self.tail_boundary) is not bool or
                (self.tail_boundary and (not self.paused or self.profile_boundary))):
            raise E3ContractError('QSA tail boundary must be a distinct accepted paused layer')
        flags = (self.accepted, self.pending, self.skipped)
        if any(type(flag) is not bool for flag in flags):
            raise E3ContractError('acceptance flags must be bool')
        if sum(flags) != 1:
            raise E3ContractError('exactly one of accepted/pending/skipped is set')
        if self.cursor is not None and type(self.cursor) is not CompletionCursor:
            raise E3ContractError('cursor must be a CompletionCursor')
        _int('host_submit_ns', self.host_submit_ns, 0)
        if self.report is not None:
            if type(self.report) not in (WorkerCompletionReport, LeanCompletionReport):
                raise E3ContractError('acceptance report must be a completion report')
            if self.report.transaction != self.transaction:
                raise E3ContractError('acceptance report belongs to another transaction')
            if self.report.worker_rank != self.worker_rank:
                raise E3ContractError('acceptance report comes from another rank')

    def matches(self, command):
        if type(command) is not SegmentCommand:
            raise E3ContractError('matches() needs a SegmentCommand')
        return (self.transaction == command.transaction
                and self.kind is command.kind
                and self.stage == command.stage
                and self.op_seq == command.op_seq
                and self.worker_rank in command.ranks)

    def __repr__(self):
        state = 'accepted' if self.accepted else ('pending' if self.pending else 'skipped')
        return (f'SegmentAcceptance(txn_seq={self.transaction.txn_seq}, '
                f'rank={self.worker_rank}, kind={self.kind.value}, '
                f'stage={self.stage}, op_seq={self.op_seq}, {state})')


@dataclass(frozen=True, repr=False)
class ReleaseRequest:
    """Release worker-owned chunk storage after the final queried report.

    ``terminal`` is the explicit request-row disposition and is not inferred by
    the worker:

    * nonterminal (``False``): the chunk's storage is released but the lane's
      request row is preserved, because the next chunk of the same request
      reaches ``LaneRequest.update()``;
    * terminal (``True``): the engine is handing the request to the ordinary
      one-token path, so the worker closes the request row before it
      acknowledges; stock must never see a row this lane still owns.

    No reply to this command is a GPU proof; reports were independently checked
    against accepted tickets before this bounded release request was issued.
    """
    transaction: TransactionId
    state_epoch: int
    completed_sequences: tuple
    terminal: bool
    # Held-g L4, a lean transaction only: per rank (rank, completed, proof and
    # submit-time counts, chain end) of the engine's applied stream. The worker
    # releases only when it equals what it sent and nothing is left unsent.
    lean_streams: tuple = ()

    def __post_init__(self):
        if type(self.transaction) is not TransactionId:
            raise E3ContractError('release needs a TransactionId')
        _tuple('lean_streams', self.lean_streams)
        last = -1
        for entry in self.lean_streams:
            _tuple('lean stream entry', entry)
            if len(entry) != 5:
                raise E3ContractError('lean stream entry is (rank, completed, proofs, hosts, digest)')
            _int('rank', entry[0])
            _int('lean completed count', entry[1])
            _int('lean proof count', entry[2])
            _int('lean submit-time count', entry[3])
            if type(entry[4]) is not bytes or len(entry[4]) != LEAN_DIGEST_SIZE:
                raise E3ContractError('lean stream digest must be 32 bytes')
            if entry[0] <= last:
                raise E3ContractError('lean stream ranks must be unique and ordered')
            last = entry[0]
        _int('state_epoch', self.state_epoch, 1)
        _tuple('completed_sequences', self.completed_sequences)
        if type(self.terminal) is not bool:
            raise E3ContractError('release needs an explicit boolean terminal flag')
        if not self.completed_sequences:
            raise E3ContractError('release needs all-rank watermarks')
        ranks = []
        for entry in self.completed_sequences:
            _tuple('completed sequence entry', entry)
            if len(entry) != 2:
                raise E3ContractError('release entry is (rank, watermark)')
            rank, sequence = entry
            _int('rank', rank)
            _int('watermark', sequence)
            ranks.append(rank)
        if ranks != sorted(set(ranks)):
            raise E3ContractError('release ranks must be unique and ordered')
        if self.lean_streams and [entry[0] for entry in self.lean_streams] != ranks:
            raise E3ContractError('lean streams must cover the release ranks')

    def __repr__(self):
        return (f'ReleaseRequest(txn_seq={self.transaction.txn_seq}, '
                f'state_epoch={self.state_epoch}, ranks={len(self.completed_sequences)}, '
                f'terminal={self.terminal})')


@dataclass(frozen=True, repr=False)
class ReleaseAck:
    """Storage/request-row closure after the correlated final report.

    ``released`` and ``request_closed`` are closure facts the worker derived
    from its drained dispatcher and freed request row. They are deliberately
    not a new kind of GPU proof: the engine only asks for this ack after its
    own accepted-ticket, state-write and PP-consumer checks passed.

    A nonterminal release answers ``released=True, request_closed=False``: the
    chunk storage is gone but the request row is retained for the next chunk.
    A terminal release must answer closed, and the engine refuses a terminal
    acknowledgment that still owns the row (or a nonterminal one that dropped
    it).
    """
    transaction: TransactionId
    worker_rank: int
    state_epoch: int
    completed_sequence: int
    released: bool = False
    request_closed: bool = False

    def __post_init__(self):
        if type(self.transaction) is not TransactionId:
            raise E3ContractError('release needs a TransactionId')
        _int('worker_rank', self.worker_rank)
        _int('state_epoch', self.state_epoch, 1)
        _int('completed_sequence', self.completed_sequence)
        if type(self.released) is not bool or type(self.request_closed) is not bool:
            raise E3ContractError('release closure flags must be bool')
        if self.request_closed and not self.released:
            raise E3ContractError('request closure requires released storage')

    @property
    def closed(self):
        return self.released and self.request_closed

    def __repr__(self):
        return (f'ReleaseAck(txn_seq={self.transaction.txn_seq}, '
                f'rank={self.worker_rank}, state_epoch={self.state_epoch}, '
                f'completed_sequence={self.completed_sequence}, '
                f'released={self.released}, request_closed={self.request_closed})')


@dataclass(frozen=True, repr=False)
class CancelRequest:
    """Bounded cancellation attempt for one opened transaction."""
    transaction: TransactionId
    state_epoch: int
    reason: str

    def __post_init__(self):
        if type(self.transaction) is not TransactionId:
            raise E3ContractError('cancel needs a TransactionId')
        _int('state_epoch', self.state_epoch, 1)
        if type(self.reason) is not str or self.reason not in CANCEL_REASONS:
            raise E3ContractError('unknown cancel reason')

    def __repr__(self):
        return (f'CancelRequest(txn_seq={self.transaction.txn_seq}, '
                f'state_epoch={self.state_epoch}, reason={self.reason})')


@dataclass(frozen=True, repr=False)
class CancelAck:
    """One bounded cancellation tick's disposition on one worker.

    ``released``/``request_closed`` are true only after the dispatcher drained
    and the request row was removed. A not-released ack keeps all ownership on
    the worker; the engine retries within its own bound and never treats a
    transport reply as a drain proof.
    """
    transaction: TransactionId
    worker_rank: int
    state_epoch: int
    disposition: str
    completed_op_seqs: tuple
    pending_op_seqs: tuple
    completed_sequence: int
    released: bool = False
    request_closed: bool = False
    # Host-queried completion state observed in this same bounded cancel tick.
    # Cancellation retirement uses this report instead of re-running the normal
    # state-written publication gate; it is not a publication signal.
    report: WorkerCompletionReport | None = None

    def __post_init__(self):
        if type(self.transaction) is not TransactionId:
            raise E3ContractError('cancel ack needs a TransactionId')
        _int('worker_rank', self.worker_rank, 0)
        _int('state_epoch', self.state_epoch, 1)
        if type(self.disposition) is not str or self.disposition not in CANCEL_DISPOSITIONS:
            raise E3ContractError('unknown cancellation disposition')
        _ints('completed_op_seqs', self.completed_op_seqs, minimum=0)
        _ints('pending_op_seqs', self.pending_op_seqs, minimum=0)
        if self.completed_op_seqs != tuple(sorted(self.completed_op_seqs)):
            raise E3ContractError('completed_op_seqs must be ordered')
        if self.pending_op_seqs != tuple(sorted(self.pending_op_seqs)):
            raise E3ContractError('pending_op_seqs must be ordered')
        if set(self.completed_op_seqs).intersection(self.pending_op_seqs):
            raise E3ContractError('an op_seq cannot be both complete and pending')
        _int('completed_sequence', self.completed_sequence, 0)
        if type(self.released) is not bool or type(self.request_closed) is not bool:
            raise E3ContractError('cancel closure flags must be bool')
        if self.request_closed and not self.released:
            raise E3ContractError('request closure requires released storage')
        if self.disposition == 'released' and not self.closed:
            raise E3ContractError('a released disposition must certify closure')
        if self.closed and self.disposition not in ('released', 'never_open'):
            raise E3ContractError('closure requires the released disposition')
        if self.disposition == 'never_open' and self.report is not None:
            raise E3ContractError('a never-open worker has no completion report')
        if (self.disposition == 'never_open'
                and (self.completed_op_seqs or self.pending_op_seqs
                     or self.completed_sequence != 0)):
            raise E3ContractError('a never-open worker has no operation history')
        if self.report is not None:
            if type(self.report) is not WorkerCompletionReport:
                raise E3ContractError('cancel report must be a completion report')
            if (self.report.transaction != self.transaction
                    or self.report.worker_rank != self.worker_rank
                    or self.report.completed_sequence != self.completed_sequence):
                raise E3ContractError('cancel report identity differs from the ack')

    @property
    def closed(self):
        return self.released and self.request_closed

    def release_ack(self, completed_sequence):
        """The controller's terminal row for an all-rank cancellation barrier."""
        return ReleaseAck(self.transaction, self.worker_rank, self.state_epoch,
                          completed_sequence, released=self.released,
                          request_closed=self.request_closed)

    def __repr__(self):
        return (f'CancelAck(txn_seq={self.transaction.txn_seq}, '
                f'rank={self.worker_rank}, epoch={self.state_epoch}, '
                f'disposition={self.disposition}, '
                f'completed={len(self.completed_op_seqs)}, '
                f'pending={len(self.pending_op_seqs)}, '
                f'released={self.released}, request_closed={self.request_closed})')


class OpSeqTracker:
    """Per-lane ``op_seq`` discipline for one bounded in-flight command.

    ``propose`` never consumes a sequence. ``accepted`` is the only consumer,
    and it accepts a strictly greater sequence or replays an already accepted,
    identical command. A pending/backpressured retry reuses its sequence.
    """

    def __init__(self, start=-1):
        if type(start) is not int or start < -1:
            raise E3ContractError('start must be -1 or an accepted op_seq')
        self._last = start
        self._pending = None
        self._accepted = {}
        self.replays = 0

    @property
    def last_accepted(self):
        return self._last

    @property
    def pending(self):
        """The outstanding (op_seq, descriptor) proposal, if any."""
        return self._pending

    def next_op_seq(self):
        return self._last + 1

    def propose(self, descriptor):
        if type(descriptor) is not SegmentDescriptor:
            raise E3ContractError('propose needs a SegmentDescriptor')
        if self._pending is not None:
            if self._pending[1] != descriptor:
                raise E3ContractError(
                    'a pending operation must be retried unchanged')
            return self._pending[0]
        op_seq = self._last + 1
        self._pending = (op_seq, descriptor)
        return op_seq

    def accepted(self, op_seq, descriptor):
        """Record acceptance; return True on first acceptance, False on replay."""
        if type(descriptor) is not SegmentDescriptor:
            raise E3ContractError('acceptance needs a SegmentDescriptor')
        _int('op_seq', op_seq, 0)
        prior = self._accepted.get(op_seq)
        if prior is not None:
            if prior != descriptor:
                raise E3ContractError('accepted op_seq reused by another command')
            self.replays += 1
            return False
        if self._pending is not None and self._pending[0] == op_seq:
            if self._pending[1] != descriptor:
                raise E3ContractError('pending operation changed before acceptance')
            self._pending = None
        if op_seq != self._last + 1:
            raise E3ContractError('accepted op_seq must strictly increase')
        self._accepted[op_seq] = descriptor
        self._last = op_seq
        return True

    def accepted_descriptor(self, op_seq):
        return self._accepted.get(op_seq)

    def __repr__(self):
        return (f'OpSeqTracker(last_accepted={self._last}, '
                f'pending={None if self._pending is None else self._pending[0]}, '
                f'replays={self.replays})')


__all__ = [
    'AdmissionRecord', 'BootstrapAck', 'BootstrapCommand',
    'CANCEL_DISPOSITIONS', 'CANCEL_REASONS', 'CURSOR_PHASES',
    'CancelAck', 'CancelRequest', 'CompletionCursor', 'FAILED_CODES',
    'OpenAck', 'OpenRequest', 'OpSeqTracker', 'PPReceiptAck', 'PPReceiptDelivery',
    'PeerBinding', 'PeerBindingAck', 'PollRequest', 'ReleaseRequest', 'ReleaseAck',
    'ROUTE_BANKS', 'ROUTE_BANK_FULL', 'ROUTE_BANK_SPLIT',
    'SegmentAcceptance', 'SegmentCommand', 'SegmentDescriptor', 'SegmentKind',
    'SourceLeaseDescriptor', 'QsaScheduleShape', 'QsaProfileQuery',
    'QsaProfileObservation', 'QsaProfileGrant', 'qsa_profile_agreement',
    'TransitionAck', 'TransitionCommand', 'TransitionPending', 'WIRE_VERSION', 'WorkerCompletionReport',
    'LEAN_DIGEST_START', 'LeanCompletionReport', 'LeanStream', 'REPLIES', 'lean_digest',
]
