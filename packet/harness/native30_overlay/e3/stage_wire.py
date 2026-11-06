"""C0 worker side: one stage-scoped command record on the one broadcast ring.

``FLASH_S5_STAGE_LOCAL=1`` makes the engine wrap every chunk command in a
:class:`StageScoped` record naming its target ranks.  It stays on the same ring
and the same generated ``flash_s5_*`` method:

* a non-target worker reads the slot (broadcast reclamation is unchanged),
  executes nothing, and writes no reply (:data:`STAGE_SKIP`);
* a target worker unwraps the record and runs the ordinary bridge handler.

The record exists only when the engine switch is on, so the worker has no
switch of its own and the default path cannot reach this module's handlers.

The local release also lives here, not in the frozen bridge or dispatcher.  A
stage may release its lane once its own work is queried complete.  The sender
may do so before the receiver's consumer proof exists: its completed PP send
ticket moves from the dispatcher to a retained-send table on this worker.  The
PP slot stays owned, and only the same typed receipt frees it, through the same
channel validation.  Nothing here frees a slot on an RPC or a timer.
"""
from dataclasses import dataclass

from e3_types import E3ContractError

RETAINED_ATTR = '_flash_s5_retained_sends'
RETIRED_ATTR = '_flash_s5_retired_sends'
RETIRED_LIMIT = 8


class _StageSkip:
    """A non-target's RPC result.  Process-local; it is never pickled."""
    __slots__ = ()

    def __repr__(self):
        return 'STAGE_SKIP'

    def __reduce__(self):
        raise E3ContractError('a stage skip never leaves its worker')


STAGE_SKIP = _StageSkip()


@dataclass(frozen=True, repr=False)
class StageScoped:
    """One chunk command and the only ranks that execute and answer it."""
    message: object
    ranks: tuple
    local_release: bool = False
    # coverage-throughput build-2 (a): the stage's successor AdmissionRecord, or None. Advisory: the worker may stage the
    # successor's PREPARE slices 1-2 after answering; nothing is decided by it and it never raises into the command.
    ahead: object = None

    def __post_init__(self):
        ranks = self.ranks
        if (type(ranks) is not tuple or not ranks
                or any(type(rank) is not int or rank < 0 for rank in ranks)
                or ranks != tuple(sorted(set(ranks)))):
            raise E3ContractError('stage scope needs ordered unique target ranks')
        if type(self.message) is StageScoped:
            raise E3ContractError('a stage scope does not nest')
        if type(self.local_release) is not bool:
            raise E3ContractError('local_release must be bool')
        if self.ahead is not None:
            from wire_v1 import AdmissionRecord
            if type(self.ahead) is not AdmissionRecord:
                raise E3ContractError('ahead must be an AdmissionRecord')

    def __repr__(self):
        return (f'StageScoped({type(self.message).__name__}, ranks={self.ranks}, '
                f'local_release={self.local_release}' + (', ahead' if self.ahead is not None else '') + ')')


def retained_sends(bridge):
    """This worker's completed sends that still wait for their consumer proof."""
    table = bridge.__dict__.get(RETAINED_ATTR)
    if table is None:
        table = bridge.__dict__[RETAINED_ATTR] = {}
    return table


def _retired_sends(bridge):
    table = bridge.__dict__.get(RETIRED_ATTR)
    if table is None:
        table = bridge.__dict__[RETIRED_ATTR] = {}
    return table


def _channel(bridge):
    comms = getattr(getattr(bridge.dispatcher, 'native', None), 'comms', None)
    return getattr(comms, 'pp', None)


def detach_pp_send(bridge):
    """Move a completed send with no proof yet from the dispatcher to the table.

    An incomplete send stays where it is, so the ordinary drain answers not
    drained and the engine retries.  A send whose proof already arrived also
    stays: the ordinary release path frees it.
    """
    dispatcher = bridge.dispatcher
    ticket = dispatcher.pp_send_ticket
    if ticket is None or dispatcher.pp_consumer_proof is not None:
        return False
    if not bool(ticket.event.query()):
        return False
    key = (ticket.transaction, ticket.seq)
    table = retained_sends(bridge)
    if key in table:
        raise E3ContractError('a retained PP send already holds this transfer')
    table[key] = ticket
    dispatcher.pp_send_ticket = None
    return True


def local_release(bridge, request):
    """The ordinary typed release, for this stage's ranks alone."""
    from wire_v1 import ReleaseRequest
    if type(request) is not ReleaseRequest:
        raise E3ContractError('a local release needs a typed ReleaseRequest')
    if getattr(bridge, 'transaction', None) == request.transaction:
        detach_pp_send(bridge)
    return bridge.release(request)


def receipt(bridge, delivery):
    """Free retained sends on their proof; anything else is the ordinary receipt."""
    from wire_v1 import PPReceiptAck
    if delivery.sender_rank != bridge.rank:
        raise E3ContractError('receipt delivery targets another sender rank')
    table = retained_sends(bridge)
    keys = [(delivery.transaction, item.seq) for item in delivery.receipts]
    if any(key in table for key in keys):
        if not all(key in table for key in keys):
            raise E3ContractError('a receipt delivery spans retained and open sends')
        channel = _channel(bridge)
        if channel is None:
            raise E3ContractError('a retained send has no PP channel')
        # Validate every entry before freeing any slot.
        for key, item in zip(keys, delivery.receipts):
            channel.validate_consumer_proof(table[key], item)
        retired = _retired_sends(bridge)
        for key, item in zip(keys, delivery.receipts):
            channel.release_send(table.pop(key), item)
            retired[key] = item
            while len(retired) > RETIRED_LIMIT:
                retired.pop(next(iter(retired)))
        return PPReceiptAck(delivery.transaction, bridge.rank, delivery.seqs, True)
    retired = _retired_sends(bridge)
    if all(key in retired and retired[key] == item
           for key, item in zip(keys, delivery.receipts)):
        # A duplicate of the exact accepted receipt is idempotent, as on an open
        # transaction.  A different proof for the same transfer falls through to
        # the ordinary path and is refused there.
        return PPReceiptAck(delivery.transaction, bridge.rank, delivery.seqs, True)
    return bridge.receipt(delivery)


def pending_send(bridge, command):
    """A typed pending answer when a stage's PP send has no free slot.

    Queried before the dispatcher's partial-enqueue boundary, so the identical
    command can be retried; a raised channel backpressure would quarantine.
    """
    from wire_v1 import SegmentAcceptance, SegmentKind, SegmentCommand
    if type(command) is not SegmentCommand or command.kind is not SegmentKind.PP_SEND:
        return None
    channel = _channel(bridge)
    free = getattr(channel, 'free_slots', None)
    if free is None or free:
        return None
    if command.op_seq in bridge.dispatcher.accepted:
        return None     # A replay of an accepted send takes no new slot.
    bridge._require_open(command.transaction)
    report = bridge._build_report()
    return SegmentAcceptance(command.transaction, bridge.rank, command.kind, command.stage,
                             command.op_seq, pending=True, cursor=report.cursor, report=report)


def stage_ahead(bridge, record):
    """Build-2 (a): stage the successor's PREPARE slices 1-2 on this worker; advisory, never raises."""
    try:
        import lane_chunk
        stage = getattr(getattr(bridge, 'dispatcher', None), 'stage_ahead', None)
        if stage is None:
            return None
        from worker_bridge import admission_to_chunk
        return stage(admission_to_chunk(record))
    except Exception:
        try:
            import lane_chunk
            lane_chunk.ahead_count('errors')
        except Exception:
            pass
        return None


def _scoped_answer(bridge, name, scoped, dispatch):
    message = scoped.message
    if scoped.local_release:
        if name != 'release':
            raise E3ContractError('only a release is stage-local')
        return local_release(bridge, message)
    if name == 'receipt':
        return receipt(bridge, message)
    if name == 'submit':
        answer = pending_send(bridge, message)
        if answer is not None:
            return answer
    return dispatch(message)


def stage_rpc(bridge, name, scoped, dispatch):
    """One scoped command on this worker; ``dispatch`` is the ordinary path. A carried ``ahead`` is staged after the answer."""
    if bridge.rank not in scoped.ranks:
        return STAGE_SKIP
    if type(scoped.message) is StageScoped:
        raise E3ContractError('a stage scope does not nest')
    answer = _scoped_answer(bridge, name, scoped, dispatch)
    if scoped.ahead is not None:
        stage_ahead(bridge, scoped.ahead)
    return answer


__all__ = ['RETAINED_ATTR', 'STAGE_SKIP', 'StageScoped', 'detach_pp_send', 'local_release',
           'pending_send', 'receipt', 'retained_sends', 'stage_ahead', 'stage_rpc']
