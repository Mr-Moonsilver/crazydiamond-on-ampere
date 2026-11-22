"""Host-gap batch for small prefill pieces (``FLASH_S5_SMALL_PIECE_BATCH``, default 0).

A piece of at most :data:`SMALL_PIECE_ROWS` rows runs each stage as about 12 short LAYER commands.  Between two
of them the worker idles for one engine round trip (reply, apply, next send: ~3.2 ms measured,
SIZING-parking-gaps-20261112 "Per-transaction fixed cost").  With the setting on, the engine sends a stage's
LAYER command together with the stage's next LAYER command as one :class:`SegmentBatch` on the same
``flash_s5_submit`` RPC, and the worker runs the second right after the first.

Rules (each one has a CPU test in ``harness/test_small_piece_batch.py``):

* Off (unset or ``0``): :func:`s5_startup` constructs today's driver class; nothing here is imported and no
  wire record changes.  ``1`` needs the stage-overlap driver.  Any other value refuses at startup.
* Only LAYER entries of a fused transaction (no credit pending, no profile or tail boundary), on a staged
  stage cursor, with no retry and no tail boundary, and only when the next plan entry is a LAYER as well.
  PREPARE, PREFIX, SUFFIX, PP, STATE_FINALIZE, polls, receipts, release and every stop replay stay single.
* Command 0 is built and booked by the frozen ``_send_plan_entry``.  Command 1 is predicted with the same
  descriptor rule; at apply time the frozen ``_send_plan_entry`` is replayed for it with the send captured and
  the rebuilt command must equal the one the worker ran (else fail closed).  Then the frozen
  ``_apply_segments`` applies it.  Trackers, tickets, TP-tail and PP obligations, fused ledger checks and
  journal rows are therefore the frozen code's, in the frozen order.
* Worker: command 0 is today's call (an exception propagates exactly as today).  Before command 1 the worker
  stops with a typed pending answer when a message waits in its ring (the yield point: a stock
  ``execute_model`` behind the batch waits for at most one native command, as today), or when its live count of
  accepted-not-complete commands reached the run-ahead bound the engine sent.  An exception in command 1 is
  answered as a typed :class:`BatchFailure`: command 0 keeps its normal accounting, command 1 is never
  applied, and the engine fails closed with ownership retained.
* At most :data:`BATCH_MAX` = 2 commands.  TP peers decide the yield on their own, so they may stop at
  different commands; with two commands they differ by at most one, which is the frozen partially-accepted
  retry (``_retry``, replayed before any cancel by ``_drive_partial_stop_submission``).  Three or more would
  let one peer run two commands ahead of the other, which no frozen owner covers (see STATUS.md).

Counters are host monotonic stamps and counts only (no device sync), in the driver status under
``small_piece_batch``.
"""
from dataclasses import dataclass
import logging
import time

from .e3_types import E3ContractError, OperationTicket
from .wire_v1 import SegmentAcceptance, SegmentCommand, SegmentDescriptor, SegmentKind

SMALL_PIECE_BATCH_ENV = 'FLASH_S5_SMALL_PIECE_BATCH'
STATUS_SCHEMA = 's5-small-piece-batch/1'
# The terminal Mamba-block remainder (= --prefix-match-unit 176): the measured tiny class.
SMALL_PIECE_ROWS = 176
# Two commands keep TP peers at most one command apart (module docstring).
BATCH_MAX = 2
STOP_REASONS = ('complete', 'ring', 'run_ahead', 'not_accepted', 'failed')
REFUSALS = ('not_staged', 'stopping', 'not_fused', 'rows', 'tail_boundary', 'tail_obligation',
            'last_layer', 'next_not_layer')
_MESSAGE_LIMIT = 200

log = logging.getLogger(__name__)


def small_piece_batch_enabled(env):
    """The launch setting; unset or 0 is today's path, 1 installs, anything else refuses."""
    value = env.get(SMALL_PIECE_BATCH_ENV, '0')
    if value not in ('0', '1'):
        raise E3ContractError(SMALL_PIECE_BATCH_ENV + ' must be 0 or 1')
    return value == '1'


# -- wire records --------------------------------------------------------------
@dataclass(frozen=True)
class SegmentBatch:
    """Consecutive LAYER commands of one stage, run in order by each target worker."""
    commands: tuple
    run_ahead: int

    def __post_init__(self):
        commands = self.commands
        if type(commands) is not tuple or not 2 <= len(commands) <= BATCH_MAX:
            raise E3ContractError('a segment batch holds 2..%d commands' % BATCH_MAX)
        first = commands[0]
        for index, command in enumerate(commands):
            if type(command) is not SegmentCommand or command.kind is not SegmentKind.LAYER:
                raise E3ContractError('a segment batch holds LAYER commands only')
            if (command.transaction != first.transaction or command.stage != first.stage
                    or command.ranks != first.ranks or command.op_seq != first.op_seq + index
                    or command.source is not None):
                raise E3ContractError('a segment batch is one stage, consecutive op_seq')
        if type(self.run_ahead) is not int or self.run_ahead < 1:
            raise E3ContractError('a segment batch needs a positive run-ahead bound')

    @property
    def transaction(self):
        return self.commands[0].transaction

    @property
    def ranks(self):
        return self.commands[0].ranks


@dataclass(frozen=True)
class BatchFailure:
    """Command ``index`` raised on this worker; nothing after it ran."""
    index: int
    error: str
    message: str

    def __post_init__(self):
        if type(self.index) is not int or self.index < 1:
            raise E3ContractError('a batch failure names a later command')
        if type(self.error) is not str or type(self.message) is not str:
            raise E3ContractError('a batch failure carries text')


@dataclass(frozen=True)
class SegmentBatchReply:
    """One worker's answers, in command order, and why it stopped."""
    worker_rank: int
    items: tuple
    stop: str
    failure: BatchFailure | None = None

    def __post_init__(self):
        if type(self.worker_rank) is not int or self.worker_rank < 0:
            raise E3ContractError('batch reply needs a worker rank')
        if (type(self.items) is not tuple or not self.items
                or any(type(item) is not SegmentAcceptance for item in self.items)):
            raise E3ContractError('batch reply items are typed acceptances')
        if self.stop not in STOP_REASONS:
            raise E3ContractError('batch reply stop reason is unknown')
        if (self.stop == 'failed') != (self.failure is not None):
            raise E3ContractError('a batch failure and its stop reason go together')
        if self.failure is not None and (type(self.failure) is not BatchFailure
                                         or self.failure.index != len(self.items)):
            raise E3ContractError('a batch failure follows the last answered command')


# -- worker side -----------------------------------------------------------------
def ring_waiting(bridge):
    """A message waits in this rank's ring (read-only peek; with no ring to read, yes)."""
    probe = getattr(getattr(bridge, 'dispatcher', None), 'ring_waiting', None)
    if not callable(probe):
        return True
    return bool(probe())


def live_pending(bridge):
    """This rank's accepted-not-complete commands, host-queried now (event queries, never a wait)."""
    return len(bridge.dispatcher.poll()['pending_op_seqs'])


def _yield_reason(bridge, batch):
    if ring_waiting(bridge):
        return 'ring'
    if live_pending(bridge) >= batch.run_ahead:
        return 'run_ahead'
    return None


def _not_run(bridge, command):
    """The typed pending answer for a command this worker did not start; it carries no report."""
    return SegmentAcceptance(command.transaction, bridge.rank, command.kind, command.stage,
                             command.op_seq, pending=True, cursor=bridge._cursor())


def run_batch(bridge, batch, dispatch):
    """Run ``batch`` on this worker; ``dispatch`` is the ordinary per-command submit path."""
    if type(batch) is not SegmentBatch:
        raise E3ContractError('run_batch needs a SegmentBatch')
    items = []
    stop = 'complete'
    failure = None
    for index, command in enumerate(batch.commands):
        if not index:
            # Exactly today's single command: its exceptions take today's failure path.
            items.append(dispatch(command))
            continue
        previous = items[-1]
        if (not previous.accepted or previous.pending or previous.paused
                or previous.profile_boundary or previous.tail_boundary):
            stop = 'not_accepted'
            break
        reason = _yield_reason(bridge, batch)
        if reason is not None:
            items.append(_not_run(bridge, command))
            stop = reason
            break
        try:
            items.append(dispatch(command))
        except Exception as exc:
            log.exception('small-piece batch: command %d (op_seq %d) failed on rank %s',
                          index, command.op_seq, getattr(bridge, 'rank', None))
            failure = BatchFailure(index, type(exc).__name__, str(exc)[:_MESSAGE_LIMIT])
            stop = 'failed'
            break
    return SegmentBatchReply(bridge.rank, tuple(items), stop, failure)


def scoped_submit(bridge, message, dispatch):
    """``stage_wire._scoped_answer``'s hook: a batch answer, or None for every other submit."""
    if type(message) is not SegmentBatch:
        return None
    return run_batch(bridge, message, dispatch)


# -- engine side -------------------------------------------------------------------
def _new_counters():
    return dict(sent=0, commands_sent=0, single_layer_commands=0, applied_commands=0,
                second_applied=0, second_not_run=0, partial=0, failures=0,
                worker_stop={reason: 0 for reason in STOP_REASONS},
                refused={reason: 0 for reason in REFUSALS},
                submit_ns=[0, 0], send_to_apply_ns=[0, 0, 0], worker_host_ns=[0, 0],
                pieces=dict(batched=0, unbatched=0),
                unbatched_reason={reason: 0 for reason in REFUSALS},
                piece_stage_ns=dict(batched=[0, 0], unbatched=[0, 0]))


def _driver_base():
    from .stage_overlap import OverlapSegmentDriver
    return OverlapSegmentDriver


def _make_driver_class():
    from .engine_driver import DriverError, LayerGroups, _SEGMENT_ROLE

    class SmallPieceBatchDriver(_driver_base()):
        """The stage-overlap driver with small-piece LAYER pairs; see the module docstring."""

        def __init__(self, controller, **kwargs):
            super().__init__(controller, **kwargs)
            self.small_batch = _new_counters()
            self._batches = {}
            self._batch_entry = None
            self._batch_capture = None
            self._batch_applying = None

        # -- send ------------------------------------------------------------
        def _send_plan_entry(self, entry):
            self._batch_entry = entry
            try:
                return super()._send_plan_entry(entry)
            finally:
                self._batch_entry = None

        def _send_segment(self, command, targets):
            capture = self._batch_capture
            if capture is not None:
                # Apply-time replay of the frozen send bookkeeping: the worker already ran ``capture``.
                self._batch_capture = None
                if command != capture:
                    raise DriverError('small-piece-batch-prediction-differs')
                self._command = command
                self._targets = tuple(targets)
                return
            entry, self._batch_entry = self._batch_entry, None
            batch = None if entry is None else self._batch_for(command, targets)
            if batch is None:
                return super()._send_segment(command, targets)
            self._command = command
            self._targets = tuple(targets)
            started = time.monotonic_ns()
            sequence = self._submit('segment', ('flash_s5_submit', (batch,), {}, None), self.ranks)
            now = time.monotonic_ns()
            self._batches[sequence] = (batch, now, self.index)
            counters = self.small_batch
            counters['sent'] += 1
            counters['commands_sent'] += len(batch.commands)
            counters['submit_ns'][0] += now - started
            counters['submit_ns'][1] += 1
            stats = self._scoped_frame.stats
            stats.setdefault('small_batches', [0, 0])[self._scoped_stage] += 1

        def _batch_refusal(self, command):
            if self._scoped_frame is None or self._scoped_stage is None:
                return 'not_staged'
            if self.stop_kind is not None or self.error is not None:
                return 'stopping'
            if not self._tx_fused:
                return 'not_fused'
            if self.tx.chunk_end - self.tx.chunk_start > SMALL_PIECE_ROWS:
                return 'rows'
            if self._tail_boundary is not None:
                return 'tail_boundary'
            if self._tail_submission is not None and self._tail_submission.command != command:
                return 'tail_obligation'
            if self.index + 1 >= len(self.plan):
                return 'last_layer'
            following = self.plan[self.index + 1]
            if following['kind'] is not SegmentKind.LAYER or following['stage'] != command.stage:
                return 'next_not_layer'
            return None

        def _batch_for(self, command, targets):
            if command.kind is not SegmentKind.LAYER:
                return None
            reason = self._batch_refusal(command)
            if reason is not None:
                # Counted per small-piece LAYER command; a large piece is not a candidate and is not counted.
                if reason not in ('not_staged', 'rows'):
                    self.small_batch['refused'][reason] += 1
                    self.small_batch['single_layer_commands'] += 1
                    first = self._scoped_frame.stats.setdefault('small_refusal', [None, None])
                    if first[self._scoped_stage] is None:
                        first[self._scoped_stage] = reason
                return None
            following = self.plan[self.index + 1]
            predicted = self._predict(following, command, targets)
            limit = (LayerGroups.RUN_AHEAD[self._tx_group] if self._tx_group != 1
                     else self.qsa_fused.run_ahead)
            return SegmentBatch((command, predicted), limit)

        def _predict(self, entry, previous, targets):
            """The command the frozen ``_send_plan_entry`` builds for ``entry`` once ``previous`` is accepted."""
            stage = entry['stage']
            depends = tuple(OperationTicket(self.tx, rank, self.lanes[rank], previous.op_seq,
                                            _SEGMENT_ROLE[previous.kind])
                            for rank in self.stage_ranks[stage])
            layer = entry.get('layer')
            descriptor = SegmentDescriptor(
                self.tx, entry['kind'], stage, depends_on=depends,
                layer_start=layer, layer_end=entry.get('layer_end', None if layer is None else layer + 1),
                seq=entry.get('seq'), layer_group=entry.get('group', 1))
            profile = self._profile_grants.get((stage, descriptor.layer_end - 1))
            return SegmentCommand(descriptor, previous.op_seq + 1, tuple(targets), profile=profile)

        # -- apply -----------------------------------------------------------
        def _apply(self, kind, values, sequence=None):
            record = self._batches.pop(sequence, None) if kind == 'segment' else None
            if record is None:
                return super()._apply(kind, values, sequence)
            self._batch_applying = record
            try:
                return super()._apply(kind, values, sequence)
            finally:
                self._batch_applying = None

        def _apply_segments(self, values):
            record, self._batch_applying = self._batch_applying, None
            if record is None:
                return super()._apply_segments(values)
            return self._apply_batch(record, values)

        def _apply_batch(self, record, values):
            batch, sent_ns, index = record
            counters = self.small_batch
            elapsed = time.monotonic_ns() - sent_ns
            timing = counters['send_to_apply_ns']
            timing[0] += elapsed
            timing[1] = max(timing[1], elapsed)
            timing[2] += 1
            replies = tuple(values)
            if (len(replies) != len(self.ranks)
                    or any(type(reply) is not SegmentBatchReply for reply in replies)
                    or {reply.worker_rank for reply in replies} != set(self.ranks)):
                raise DriverError('small-piece batch reply does not cover every stage rank')
            for reply in replies:
                if len(reply.items) > len(batch.commands):
                    raise DriverError('small-piece batch reply answers more commands than sent')
                counters['worker_stop'][reply.stop] += 1
            if self._command != batch.commands[0] or self.index != index:
                raise DriverError('small-piece batch reply differs from its stage cursor')
            # Command 0: exactly the frozen apply of a single LAYER reply.
            super()._apply_segments(tuple(reply.items[0] for reply in replies))
            counters['applied_commands'] += 1
            self._note_host(replies, 0)
            if any(reply.failure is not None for reply in replies):
                counters['failures'] += 1
                failed = sorted((reply.worker_rank, reply.failure.error) for reply in replies
                                if reply.failure is not None)
                self._record('small_piece_batch_failure', transaction=self.tx.txn_seq,
                             op_seq=batch.commands[1].op_seq, failed=[list(row) for row in failed])
                raise DriverError('small-piece-batch-command-failed: %s' % (failed,))
            second = [reply.items[1] if len(reply.items) > 1 else None for reply in replies]
            ran = [item for item in second if item is not None and item.accepted]
            done = self._retry is None and self.phase == 'segments' and self.index == index + 1
            if not done:
                if ran:
                    raise DriverError('small-piece-batch-diverged')
                return
            if not ran:
                # No rank started command 1: the cursor stays on it, as if it was never sent.
                counters['second_not_run'] += 1
                return
            if any(item is None for item in second):
                raise DriverError('small-piece batch reply lacks the second answer')
            # Command 1: the frozen send bookkeeping, then the frozen apply.
            self._batch_capture = batch.commands[1]
            try:
                self._send_plan_entry(self.plan[self.index])
            finally:
                leftover, self._batch_capture = self._batch_capture, None
            if leftover is not None:
                raise DriverError('small-piece-batch-replay-did-not-send')
            command = self._command
            tail_command = (self._tail_submission is not None
                            and self._tail_submission.command == command)
            self.qsa_control.add(command.stage, command.descriptor.layer_start, layer_commands=1,
                                 bucket_commands=int(command.profile is not None and not tail_command),
                                 tail_commands=int(tail_command))
            super()._apply_segments(tuple(second))
            counters['applied_commands'] += 1
            counters['second_applied'] += 1
            counters['partial'] += int(self._retry is not None)
            self._note_host(replies, 1)

        def _note_host(self, replies, index):
            host = self.small_batch['worker_host_ns']
            for reply in replies:
                if len(reply.items) > index and reply.items[index].accepted:
                    host[0] += reply.items[index].host_submit_ns
                    host[1] += 1

        # -- per-piece receipt totals -------------------------------------------
        def _stage_line(self, transaction, stats, both_ns):
            super()._stage_line(transaction, stats, both_ns)
            try:
                if transaction.chunk_end - transaction.chunk_start > SMALL_PIECE_ROWS:
                    return
                counters = self.small_batch
                batched = stats.get('small_batches', [0, 0])
                refusal = stats.get('small_refusal', [None, None])
                for stage in (0, 1):
                    first, last = stats['first_ns'][stage], stats['last_ns'][stage]
                    key = 'batched' if batched[stage] else 'unbatched'
                    counters['pieces'][key] += 1
                    if not batched[stage] and refusal[stage] is not None:
                        counters['unbatched_reason'][refusal[stage]] += 1
                    if type(first) is int and type(last) is int:
                        row = counters['piece_stage_ns'][key]
                        row[0] += last - first
                        row[1] += 1
            except Exception:
                log.debug('small-piece batch totals failed', exc_info=True)

        def _reset(self):
            super()._reset()
            self._batches = {sequence: record for sequence, record in self._batches.items()
                             if sequence in self._seq_route}

        def _status_record(self, now=None):
            result = super()._status_record(now)
            result['small_piece_batch'] = self.small_batch_status()
            return result

        def small_batch_status(self):
            counters = self.small_batch
            return dict(schema=STATUS_SCHEMA, installed=True, rows_max=SMALL_PIECE_ROWS, batch_max=BATCH_MAX,
                        outstanding=len(self._batches),
                        **{name: (dict(value) if type(value) is dict else
                                  list(value) if type(value) is list else value)
                           for name, value in counters.items()
                           if name not in ('pieces', 'piece_stage_ns')},
                        pieces=dict(counters['pieces']),
                        piece_stage_ns={key: list(row) for key, row in counters['piece_stage_ns'].items()})

    return SmallPieceBatchDriver


_CLASS = []


def driver_class():
    """The engine driver class with the batch installed (built once, on first use)."""
    if not _CLASS:
        _CLASS.append(_make_driver_class())
    return _CLASS[0]


__all__ = ['BATCH_MAX', 'BatchFailure', 'REFUSALS', 'SMALL_PIECE_BATCH_ENV', 'SMALL_PIECE_ROWS', 'STOP_REASONS',
           'SegmentBatch', 'SegmentBatchReply', 'driver_class', 'live_pending', 'ring_waiting', 'run_batch',
           'scoped_submit', 'small_piece_batch_enabled']
