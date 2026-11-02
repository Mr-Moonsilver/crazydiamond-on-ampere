"""C1: two chunks in flight, one command in flight per stage (``FLASH_S5_STAGE_OVERLAP=1``).

Rests on C0 (:mod:`stage_progress`): per-stage cursors, target-only replies and a
local release per stage.  C1 adds one thing: chunk n+1 of the same request is
admitted once stage 0 has locally released chunk n, so stage 0 runs n+1 while
stage 1 still runs n.  Never more than two chunks, never more than one command in
flight per stage, and each rank still receives its device commands in the serial
order (stage 0: all of n, then all of n+1; stage 1 likewise), so the numerics do
not change.

Allocation (provisional, chosen before admission).  n+1 is allocated by the real
admission path while n is still parked, with the request's
``num_computed_tokens`` set to n's end and ``num_in_flight_tokens`` raised by
n's width for the duration of the call.  The real KV manager then allocates
exactly what it would after commit(n) (``tmp/c0c1/c1/alloc_diff.py``), provided
``cache_blocks(n)`` is inert for n+1's allocation.  :func:`overlap_refusal` names
the chunks where it is not; such a chunk waits for commit(n) and is admitted the
ordinary way.  Nothing is decided after admission and nothing falls back after
partial execution.

Lanes.  Stage 0's PREPARE needs source owners for all four ranks and its peer
binding needs the receiver's lane, but stage 1 opens n+1 only after releasing n.
A worker's prefill lane epoch rises by exactly one per open
(``lane_state.LaneState.reserve``), so stage 0 binds n+1 against stage 1's
*expected* lanes (n's lanes, epoch + 1).  Stage 1's open must return exactly
those lanes, before any PP send of n+1; anything else is a DriverError, and the
real source hold also refuses a changed owner set.

Commit stays the frozen commit, in chunk order, once per chunk.  Stop: see
:meth:`OverlapSegmentDriver._stop_frame`.
"""
from contextlib import contextmanager
from dataclasses import replace
import json
import logging
import os
import time

from e3_types import E3ContractError
from engine_driver import DriverError, NativeSegmentDriver, _progress_logger
from scheduler_hooks import PrefillAdmissionController
from stage_progress import StagedSegmentDriver, stage_local_enabled
from wire_v1 import CancelRequest, PollRequest, ReleaseAck, SegmentKind

STAGE_OVERLAP_ENV = 'FLASH_S5_STAGE_OVERLAP'
REFUSALS = ('older-terminal', 'successor-may-be-terminal', 'older-at-last-hash-boundary',
            'older-at-shared-prefix-boundary', 'partial-tail-pending', 'manager-shape')
# alpha-next composition: per-chunk fields the composed base driver keeps on ``self`` (held-g L4 lean replies,
# Step A fused QSA, Step B layer groups), set at each chunk's begin and read by that chunk's handlers.  C0 runs one
# frame, so its swap (stage_progress._CHUNK_FIELDS) never needed them; with two frames each keeps its own here, or
# n+1's begin and n's reset would overwrite the other chunk's values (n's commit set _tx_group to 1 under a grouped
# n+1).  ``_fused_stop_at`` is not here: the stop bound runs over the whole stop, both frames, as the base driver's.
COMPOSED_CHUNK_FIELDS = ('_replies', '_lean_streams', '_tx_fused', '_tx_group', '_fused_mark', '_fused_mark_at')


class _ProgressNoBegin:
    """The engine progress recorder seen by the successor's frozen ``_begin``: every read is the real recorder's
    (so ``_open_replies`` picks the arm's reply format, lean included, and the lean chunk counter counts it), only
    ``begin`` does nothing (one engine progress line per commit; see OverlapSegmentDriver._reset)."""

    def __init__(self, progress):
        self.__dict__['_progress'] = progress

    def begin(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        return getattr(self.__dict__['_progress'], name)

    def __setattr__(self, name, value):
        setattr(self.__dict__['_progress'], name, value)


def stage_overlap_enabled(env):
    value = env.get(STAGE_OVERLAP_ENV, '0')
    if value not in ('0', '1'):
        raise DriverError(STAGE_OVERLAP_ENV + ' must be 0 or 1')
    if value == '1' and not stage_local_enabled(env):
        raise DriverError(STAGE_OVERLAP_ENV + ' needs FLASH_S5_STAGE_LOCAL=1')
    return value == '1'


# C1 rebind (review 2 Nov, section 2, option (a)): stage 0 binds n+1's PP consumer just before n+1's PP_SEND,
# not at its early open, so n+1's PREPARE/PREFIX/layers/SUFFIX need not wait for n's consumer proof.  Default off.
DEFER_BIND_ENV = 'FLASH_S5_STAGE_OVERLAP_DEFER_BIND'


def stage_overlap_defer_bind_enabled(env):
    value = env.get(DEFER_BIND_ENV, '0')
    if value not in ('0', '1'):
        raise DriverError(DEFER_BIND_ENV + ' must be 0 or 1')
    if value == '1' and not stage_overlap_enabled(env):
        raise DriverError(DEFER_BIND_ENV + ' needs FLASH_S5_STAGE_OVERLAP=1')
    return value == '1'


# C1 overlap observation (default off): per chunk command, device-completion bounds read from the reports the
# engine already ingests (no extra RPC, poll or event query), on the engine's CLOCK_MONOTONIC.  A read, never a gate.
OBSERVE_ENV = 'FLASH_S5_STAGE_OVERLAP_OBSERVE'
OBSERVE_LINE = 'flash-s5 overlap-observe '
OBSERVE_SCHEMA = 's5-overlap-observe/1'
OBSERVE_OPS_PER_LINE = 64
OBSERVE_SUBMITS_MAX = 16


def stage_overlap_observe_enabled(env):
    value = env.get(OBSERVE_ENV, '0')
    if value not in ('0', '1'):
        raise DriverError(OBSERVE_ENV + ' must be 0 or 1')
    if value == '1' and not stage_overlap_enabled(env):
        raise DriverError(OBSERVE_ENV + ' needs FLASH_S5_STAGE_OVERLAP=1')
    return value == '1'


class OverlapObserver:
    """Device-completion bounds per (rank, op_seq) of each chunk, from ingested reports only.

    For a reply applied at ``t_apply`` to a command submitted at ``t_submit``, the reply's
    report was built by the worker in between.  An op the report lists pending completed
    after ``t_submit`` (``lo`` keeps the latest such time); an op it lists completed first
    completed before ``t_apply`` (``hi``).  Both are host CLOCK_MONOTONIC ns, the clock of
    the 'n' progress rows.  The PP consumer marker names the reply (transport sequence) whose
    report first carried a sender's PP_CONSUMER proof.  Storage lives on the chunk's frame
    (``stats['obs']``) and leaves with it: ops are the chunk's accepted ops, at most two
    consumer marks.
    """

    def __init__(self, clock=time.monotonic_ns):
        self.clock = clock
        self.current = None     # (sequence, t_submit, t_apply) of the reply being applied
        self.submits = {}       # sequence -> t_submit, for chunk commands in flight only
        self.counters = dict(reports=0, ops=0, consumer_marks=0, lines=0, line_bytes=0,
                             submits_dropped=0, host_ns=0)

    def note_submit(self, sequence):
        self.submits[sequence] = self.clock()
        while len(self.submits) > OBSERVE_SUBMITS_MAX:
            # One command in flight per stage, two frames: more is a reply that never applied.
            self.submits.pop(next(iter(self.submits)))
            self.counters['submits_dropped'] += 1

    def begin_apply(self, sequence):
        self.current = (sequence, self.submits.pop(sequence, None), self.clock())

    def end_apply(self):
        self.current = None

    def note_report(self, frame, report, descriptors, consumer_before, consumer_after):
        started = time.perf_counter_ns()
        current = self.current
        if current is None or current[1] is None:
            return
        sequence, t_submit, t_apply = current
        obs = frame.stats.setdefault('obs', dict(ops={}, consumer=[]))
        ops = obs['ops']
        rank = report.worker_rank
        for op in report.pending_op_seqs:
            row = ops.get((rank, op))
            if row is None:
                descriptor = descriptors.get((rank, op))
                row = ops[(rank, op)] = [getattr(getattr(descriptor, 'kind', None), 'value', None),
                                         getattr(descriptor, 'layer_start', None), None, None]
                self.counters['ops'] += 1
            if row[3] is None and (row[2] is None or t_submit > row[2]):
                row[2] = t_submit
        for op in report.completed_op_seqs:
            row = ops.get((rank, op))
            if row is None:
                descriptor = descriptors.get((rank, op))
                row = ops[(rank, op)] = [getattr(getattr(descriptor, 'kind', None), 'value', None),
                                         getattr(descriptor, 'layer_start', None), None, None]
                self.counters['ops'] += 1
            if row[3] is None:
                row[3] = t_apply
        for sender in sorted(set(consumer_after) - set(consumer_before)):
            obs['consumer'].append([sequence, sender, rank, t_apply])
            self.counters['consumer_marks'] += 1
        self.counters['reports'] += 1
        self.counters['host_ns'] += time.perf_counter_ns() - started

    def lines(self, transaction, stats, frame_info):
        """The chunk's observation as bounded log lines (parts of at most OBSERVE_OPS_PER_LINE ops)."""
        obs = stats.get('obs') or dict(ops={}, consumer=[])
        ops = [[rank, op] + row for (rank, op), row in sorted(obs['ops'].items())]
        parts = [ops[i:i + OBSERVE_OPS_PER_LINE] for i in range(0, len(ops), OBSERVE_OPS_PER_LINE)] or [[]]
        out = []
        for index, part in enumerate(parts):
            body = dict(schema=OBSERVE_SCHEMA, txn=transaction.txn_seq,
                        chunk=[transaction.chunk_start, transaction.chunk_end],
                        part=index, parts=len(parts),
                        fields=['rank', 'op_seq', 'kind', 'layer', 'lo_ns', 'hi_ns'], ops=part)
            if index == 0:
                body.update(frame_info, seqs=_ranges(stats.get('seqs', ())), consumer=obs['consumer'])
            text = OBSERVE_LINE + json.dumps(body, separators=(',', ':'), sort_keys=True)
            self.counters['lines'] += 1
            self.counters['line_bytes'] += len(text)
            out.append(text)
        return out


def overlap_refusal(manager, request, older_end, older_terminal, cap):
    """Why n+1 must wait for commit(n), or None.  Pure; read before any allocation.

    ``cache_blocks(n)`` must not change what ``allocate_slots(n+1)`` does:
    * it publishes the partial prompt tail when n reaches the last hash boundary
      (eagle: one hash block lower), which forces a CoW in the next allocation;
    * the same at the request's shared prefix boundary;
    * a partial hit or an internal checkpoint still pending on the request.
    A successor that may be terminal also waits: ``remaining <= cap`` is the
    conservative stand-in for the exact plan.  At cap 1024 it also covers the
    hash boundary rule, which stays explicit.
    """
    if older_terminal:
        return 'older-terminal'
    coordinator = getattr(manager, 'coordinator', None)
    groups = getattr(coordinator, 'single_type_managers', None)
    hash_size = getattr(getattr(manager, 'block_pool', None), 'hash_block_size', None)
    prompt = getattr(request, 'num_prompt_tokens', None)
    total = getattr(request, 'num_tokens', None)
    if (groups is None or type(hash_size) is not int or hash_size < 1
            or type(prompt) is not int or type(total) is not int):
        return 'manager-shape'
    if total - 1 - older_end <= cap:
        return 'successor-may-be-terminal'
    boundary = prompt // hash_size * hash_size
    if any(getattr(group, 'use_eagle', False) for group in groups):
        boundary = max(boundary - hash_size, 0)
    if older_end >= boundary:
        return 'older-at-last-hash-boundary'
    if older_end == getattr(request, 'shared_prefix_boundary', None):
        return 'older-at-shared-prefix-boundary'
    uid = request.request_id
    for group in groups:
        if uid in getattr(group, '_partial_hit_reqs', ()) or uid in getattr(group, '_checkpoints', {}):
            return 'partial-tail-pending'
    return None


@contextmanager
def provisional_fields(request, older):
    """n+1 sees n as computed and still in flight, for one allocation call only."""
    held = request.num_computed_tokens, request.num_in_flight_tokens
    request.num_computed_tokens = older.transaction.chunk_end
    request.num_in_flight_tokens = held[1] + (older.transaction.chunk_end - older.transaction.chunk_start)
    try:
        yield
    finally:
        request.num_computed_tokens, request.num_in_flight_tokens = held


class _OneRequestQueue:
    """The waiting-queue view for a successor: the parked request, never requeued."""

    def __init__(self, request):
        self.request = request

    def peek_request(self):
        return self.request

    def remove_request(self, request):
        if request is not self.request:
            raise E3ContractError('successor admission removed another request')

    def prepend_request(self, request):
        raise E3ContractError('a request with a chunk in flight is not requeued')

    def __iter__(self):
        yield self.request

    def __len__(self):
        return 1


class _NoRequeue:
    """Commit(n) with n+1 in flight: the request stays parked, so nothing is requeued."""

    def __init__(self):
        self.prepended = 0

    def prepend_request(self, request):
        self.prepended += 1


class StagedAdmissionController(PrefillAdmissionController):
    """The frozen controller with one successor record beside ``parked``.

    ``parked`` stays the oldest chunk, so ``active``, ``parked_count`` and every
    frozen reader keep their meaning.  Lookups by transaction cover both records.
    """

    def __init__(self, scheduler, **kwargs):
        super().__init__(scheduler, **kwargs)
        self.successor = None
        self._view = None
        self._successor_queue = None
        self.overlap_counters = dict(successors=0, promoted=0, successor_cancels=0,
                                     partial_successor_retires=0, alloc_wait=0)
        self.overlap_refusals = dict.fromkeys(REFUSALS, 0)
        self.last_successor_refusal = None

    # -- successor admission ------------------------------------------------
    def _queue(self, scheduler):
        if self._successor_queue is not None:
            return self._successor_queue
        return super()._queue(scheduler)

    def _admit_observed(self, scheduler):
        parked = self.parked
        if parked is None:
            return super()._admit_observed(scheduler)
        if self.successor is not None or parked.failed or parked.admission is None:
            return None
        owner = getattr(scheduler, '_flash_s5_driver', None)
        query = getattr(owner, 'successor_admissible', None)
        if not callable(query) or not query(parked.transaction):
            return None
        request = parked.request
        cap = min(self.chunk_tokens, getattr(scheduler, 'max_num_scheduled_tokens', self.chunk_tokens))
        reason = overlap_refusal(scheduler.kv_cache_manager, request, parked.transaction.chunk_end,
                                 parked.admission.terminal, cap)
        if reason is not None:
            self.overlap_refusals[reason] += 1
            self.last_successor_refusal = dict(reason=reason, transaction=parked.transaction.txn_seq)
            return None
        receipts = (self.refused, self.alloc_failures, self.refusal_seq, self.last_refusal)
        self.parked = None
        self._successor_queue = _OneRequestQueue(request)
        try:
            with provisional_fields(request, parked):
                record = super()._admit_observed(scheduler)
        finally:
            successor = self.parked
            self.parked = parked
            self._successor_queue = None
            if successor is not None:
                self.successor = successor
        if record is None:
            if successor is not None:
                raise E3ContractError('successor admission parked a record without admitting it')
            # Allocation (or a plan refusal) at this moment is not a serial event:
            # the chunk waits for commit(n), and the serial receipts are kept as they were.
            self.refused, self.alloc_failures, self.refusal_seq, self.last_refusal = receipts
            self.overlap_counters['alloc_wait'] += 1
            return None
        self.overlap_counters['successors'] += 1
        return record

    # -- lookups by transaction ---------------------------------------------
    def _record_for(self, transaction):
        successor = self.successor
        if successor is not None and transaction is not None and transaction == successor.transaction:
            return successor
        return None

    @contextmanager
    def begin_view(self, record):
        """``admission()`` names ``record`` while the driver begins its frame."""
        if record is not self.successor or self._view is not None:
            raise E3ContractError('a begin view needs the current successor')
        self._view = record
        try:
            yield record
        finally:
            self._view = None

    def _parked_for(self, transaction):
        record = self._record_for(transaction)
        return record if record is not None else super()._parked_for(transaction)

    def admission(self, transaction=None):
        if transaction is None and self._view is not None:
            return self._view.admission
        record = self._record_for(transaction)
        return record.admission if record is not None else super().admission(transaction)

    def request(self, transaction=None):
        record = self._record_for(transaction)
        return record.request if record is not None else super().request(transaction)

    def source_hold(self, transaction=None):
        record = self._record_for(transaction)
        return record.source_hold if record is not None else super().source_hold(transaction)

    def _source_is_current(self, transaction):
        record = self._record_for(transaction)
        if record is None:
            return super()._source_is_current(transaction)
        return not record.failed and not record.cache_published

    # -- retirement -----------------------------------------------------------
    def commit(self, transaction, proof):
        successor = self.successor
        if successor is None:
            return super().commit(transaction, proof)
        parked = self.parked
        if parked is None or parked.transaction != transaction:
            raise E3ContractError('chunks commit in order: the oldest first')
        queue, parked.queue = parked.queue, _NoRequeue()
        try:
            request = super().commit(transaction, proof)
        finally:
            requeue, parked.queue = parked.queue, queue
        if self.parked is not None or requeue.prepended != 1:
            raise E3ContractError('commit with a successor left the request in an unexpected place')
        successor.queue = queue
        self.parked, self.successor = successor, None
        self.overlap_counters['promoted'] += 1
        return request

    def cancel(self, transaction, *, proof=None, never_submitted=False):
        successor = self.successor
        if successor is None:
            return super().cancel(transaction, proof=proof, never_submitted=never_submitted)
        if transaction != successor.transaction:
            raise E3ContractError('the successor is retired before the older chunk')
        older, identities = self.parked, dict(self._request_identities)
        self.parked = successor
        try:
            request = super().cancel(transaction, proof=proof, never_submitted=never_submitted)
        except BaseException:
            self.parked = older
            raise
        # The older chunk of the same request is still live and keeps its identity.
        self.parked, self.successor = older, None
        self._request_identities = identities
        self.overlap_counters['successor_cancels'] += 1
        return request

    def retire_successor(self, transaction, *, proof, releases, unopened):
        """Cancel a successor that some stage never opened; no publication.

        ``releases`` are the closed rows of every rank that opened it; ``unopened``
        the ranks the engine never sent its open to.  Those ranks hold no
        operation, report or source binding of this chunk, so the source lease is
        released over the opened ranks' barrier only.  The frozen source hold
        takes no partial barrier and is left unreleased, its record dropped with
        the chunk.
        """
        record = self.successor
        if record is None or record.transaction != transaction:
            raise E3ContractError('partial retirement is for the current successor only')
        unopened = frozenset(unopened)
        opened = tuple(rank for rank in self.ranks if rank not in unopened)
        if not unopened or not opened or not unopened <= set(self.ranks):
            raise E3ContractError('partial retirement needs opened and unopened ranks')
        if not record.pp_sent <= record.pp_consumed:
            raise E3ContractError('a cancelled chunk still owns PP transfers')
        if not record.bound or proof is None or proof.ticket != record.ticket:
            raise E3ContractError('partial retirement needs the bound producer proof')
        if set(record.reports) & unopened:
            raise E3ContractError('an unopened rank reported work for this chunk')
        if type(releases) is not dict or set(releases) != set(opened):
            raise E3ContractError('partial retirement needs every opened rank closed')
        for rank in opened:
            ack = releases[rank]
            if type(ack) is not ReleaseAck or not ack.request_closed:
                raise E3ContractError('partial retirement needs closed typed rows')
            self.note_worker_release(ack)
        self.leases.release_after_workers(record.ticket, releases, opened)
        self.frontier.cancel(transaction)
        if record.capture is not None and record.capture.release_retained(self._scheduler):
            self.retained_releases += 1
        self.successor = None
        self.cancels += 1
        self.overlap_counters['partial_successor_retires'] += 1
        return record.request

    def summary(self):
        result = super().summary()
        result['successor'] = 0 if self.successor is None else 1
        return result


class OverlapSegmentDriver(StagedSegmentDriver):
    """C0's staged driver with one successor frame; see the module docstring."""
    CONTROLLER_CLASS = StagedAdmissionController

    def __init__(self, controller, **kwargs):
        super().__init__(controller, **kwargs)
        self.overlap_selected = True
        self.overlap_counters = dict(successor_frames=0, overlapped_commits=0, early_opens=0,
                                     lane_checks=0, closed_by_successor=0, successor_retires=0,
                                     unbegun_successor_cancels=0, partial_replays=0)
        self.defer_bind = stage_overlap_defer_bind_enabled(os.environ)
        self.defer_counters = dict(deferred_opens=0, deferred_binds=0)
        self.overlap_observer = OverlapObserver() if stage_overlap_observe_enabled(os.environ) else None

    # -- the composed per-chunk fields, per frame -------------------------------
    def _new_frame(self):
        frame = super()._new_frame()
        attrs = self.__dict__
        frame.composed = {name: attrs[name] for name in COMPOSED_CHUNK_FIELDS if name in attrs}
        return frame

    @contextmanager
    def _scope(self, frame, stage=None):
        with super()._scope(frame, stage) as bound:
            attrs = self.__dict__
            composed = getattr(frame, 'composed', None)
            if composed is not None:
                attrs.update(composed)
            try:
                yield bound
            finally:
                if composed is not None and not frame.dead:
                    frame.composed = {name: attrs[name] for name in composed}

    def _show_rest(self):
        super()._show_rest()
        if self._frames:
            composed = getattr(self._frames[0], 'composed', None)
            if composed is not None:
                self.__dict__.update(composed)

    # -- admission query (scheduler side) --------------------------------------
    def successor_admissible(self, transaction):
        """True when n+1 may be admitted now: stage 0 locally released n, nothing else pending."""
        if not (self.stage_selected and self.overlap_selected):
            return False
        if (self.error is not None or self.stop_kind is not None or self._native43 is not None
                or self._control is not None or self._control_orphans
                or self._route_exchange() is not None or self._return_pending is not None
                or not self._split_ready or self._profile_failure is not None):
            return False
        if len(self._frames) != 1:
            return False
        frame = self._frames[0]
        return (not frame.dead and frame.chunk['tx'] == transaction and 0 in frame.released
                and frame.chunk['admission'] is not None and not frame.chunk['admission'].terminal)

    # -- successor frame ------------------------------------------------------
    def _begin_successor(self):
        controller = self.controller
        record = controller.successor
        older = self._frames[0]
        attrs = self.__dict__
        progress = attrs['progress']
        with controller.begin_view(record):
            # One progress line per commit (see _reset); the recorder is still read, so a lean arm opens n+1 lean.
            attrs['progress'] = None if progress is None else _ProgressNoBegin(progress)
            try:
                NativeSegmentDriver._begin(self)
            finally:
                attrs['progress'] = progress
        frame = self._new_frame()
        frame.successor_of = older
        frame.early = False
        if self.overlap_observer is not None:
            frame.stats['obs_successor_of'] = older.chunk['tx'].txn_seq
        frame.expected_lanes = {rank: replace(older.chunk['lanes'][rank],
                                              storage_epoch=older.chunk['lanes'][rank].storage_epoch + 1)
                                for rank in self.stage_ranks[1]}
        older.closed_by_successor = {}
        self._frames.append(frame)
        self.overlap_counters['successor_frames'] += 1
        self._show_rest()

    def _frame_ready(self, frame, stage):
        """A successor's stage starts only after that stage released the older chunk."""
        older = getattr(frame, 'successor_of', None)
        if older is None or older.dead or older not in self._frames:
            return True
        if stage not in older.released:
            return False
        if stage == 0 and frame.stage[0]['phase'] == 'bind':
            # The sender's PP channel may be rebound only after n's tickets retired.
            # With the deferred bind this is the same gate, reached just before n+1's PP_SEND.
            chunk = older.chunk
            return (chunk['_receipt_inflight'] is None
                    and set(chunk['_receipt_acked']) == set(self.pp_peers))
        return True

    def _drive(self):
        if not self._frames:
            return super()._drive()
        if self._chunk_command_blocked():
            return
        self._maybe_commit()
        controller = self.controller
        if (controller.successor is not None and self._frames
                and not any(frame.chunk['tx'] == controller.successor.transaction for frame in self._frames)):
            self._begin_successor()
        if self.defer_bind:
            for frame in tuple(self._frames):
                self._deferred_bind_due(frame)
        polls = []
        for frame in tuple(self._frames):
            for stage in (0, 1):
                if frame.dead or not self._frame_ready(frame, stage):
                    continue
                if self._drive_stage(frame, stage):
                    polls.append((frame, stage))
        for frame, stage in polls:
            if any(self._route_stage(sequence) == stage for sequence in self._inflight):
                continue   # another frame's command took this stage meanwhile
            with self._scope(frame, stage):
                self._submit('poll', ('flash_s5_poll',
                                      (PollRequest(self.tx, self.admission.state_epoch,
                                                   self.poll_budget),), {}, None), self.ranks)

    def _submit_observed(self, kind, envelope, ranks, *, timeout_s=None):
        sequence = super()._submit_observed(kind, envelope, ranks, timeout_s=timeout_s)
        frame = self._scoped_frame
        if frame is not None:
            frame.stats.setdefault('seqs', []).append(sequence)
            if self.overlap_observer is not None and self._scoped_stage is not None:
                self.overlap_observer.note_submit(sequence)
        return sequence

    # -- overlap observation (FLASH_S5_STAGE_OVERLAP_OBSERVE) ---------------------
    def _apply(self, kind, values, sequence=None):
        observer = self.overlap_observer
        if observer is None:
            return super()._apply(kind, values, sequence)
        observer.begin_apply(sequence)
        try:
            return super()._apply(kind, values, sequence)
        finally:
            observer.end_apply()

    def _ingest_report(self, report):
        observer = self.overlap_observer
        if observer is None or self._scoped_frame is None:
            return super()._ingest_report(report)
        before = tuple(self._consumer_proofs)
        super()._ingest_report(report)
        observer.note_report(self._scoped_frame, report, self._accepted_descriptors,
                             before, tuple(self._consumer_proofs))

    # -- open -----------------------------------------------------------------
    def _apply_stage_open(self, values):
        frame, stage = self._scoped_frame, self._scoped_stage
        expected = getattr(frame, 'expected_lanes', None)
        super()._apply_stage_open(values)
        if expected is None:
            return
        if stage == 1:
            self.overlap_counters['lane_checks'] += 1
            if any(self.lanes[rank] != expected[rank] for rank in self.stage_ranks[1]):
                raise DriverError('stage 1 opened another lane than the successor was bound to')
        elif frame.opened == {0}:
            self._early_open(frame)

    def _early_open(self, frame):
        """Stage 0 opened n+1 before stage 1: plan, owners and binding on the expected lanes."""
        lanes = dict(self.lanes)
        lanes.update(frame.expected_lanes)
        plans = {0: [], 1: []}
        for entry in self._build_plan():
            plans[entry['stage']].append(entry)
        self.controller.note_source_owners(self.tx, lanes, self.layer_count)
        self.lanes.update(frame.expected_lanes)
        self._bindings = {}
        self._bind_index = 0
        self._pending_binding = None
        self.plan, self.index, self.phase = plans[0], 0, 'bind'
        if self.defer_bind:
            # n+1's own compute needs no PP binding; the bind waits in front of its PP_SEND.
            self.phase = 'segments'
            frame.deferred_bind = True
            self.defer_counters['deferred_opens'] += 1
        other = frame.stage[1]
        other['plan'], other['index'] = plans[1], 0
        frame.early = True
        self.overlap_counters['early_opens'] += 1
        if self.overlap_observer is not None:
            frame.stats['obs_early'] = True
            frame.stats['obs_deferred'] = bool(getattr(frame, 'deferred_bind', False))
        self._record('chunk_open_stage0', transaction=self.tx.txn_seq,
                     chunk_start=self.tx.chunk_start, chunk_end=self.tx.chunk_end,
                     state_epoch=frame.open_state_epoch,
                     expected_lanes=[[rank, lane.storage_epoch, lane.context_id]
                                     for rank, lane in sorted(frame.expected_lanes.items())])

    # -- deferred PP binding (FLASH_S5_STAGE_OVERLAP_DEFER_BIND) -----------------
    def _deferred_send_next(self, cursor, chunk):
        """Stage 0's next command is the PP_SEND and this frame's senders are not all bound."""
        return (cursor['_retry'] is None and cursor['index'] < len(cursor['plan'])
                and cursor['plan'][cursor['index']]['kind'] is SegmentKind.PP_SEND
                and len(chunk['_bindings']) < len(self.stage_ranks[0]))

    def _segment_gate(self, frame, stage):
        """A deferred frame's PP_SEND never leaves before its own binding (tick or in-wait)."""
        if (stage == 0 and getattr(frame, 'deferred_bind', False)
                and self._deferred_send_next(self.__dict__, self.__dict__)):
            return True
        return super()._segment_gate(frame, stage)

    def _deferred_bind_due(self, frame):
        """Move a deferred frame's stage 0 into ``bind`` once its PP_SEND is next.

        From there the frozen bind path runs unchanged: ``_frame_ready`` holds it until n's
        receipts are all acked (n's retained send retired under n's own binding), then
        ``_next_binding`` binds n+1's receiver lanes and ``_apply_bind`` returns to segments.
        """
        if frame.dead or not getattr(frame, 'deferred_bind', False) or self._scoped_frame is not None:
            return False
        cursor = frame.stage[0]
        if cursor['phase'] != 'segments' or not self._deferred_send_next(cursor, frame.chunk):
            return False
        if any(route == (frame, 0) for route in self._seq_route.values()):
            return False
        cursor['phase'] = 'bind'
        self.defer_counters['deferred_binds'] += 1
        return True

    def _chunk_opened(self, frame):
        if not getattr(frame, 'early', False):
            return super()._chunk_opened(frame)
        # Plan, owners and binding were set at stage 0's open; stage 1 joins.  The
        # retained-row move is C0's, word for word.
        ranks = self._all_ranks
        if self._retained is not None:
            row = self._retained
            if (not row['transaction'].same_request(self.tx)
                    or row['transaction'] == self.tx):
                raise DriverError('open does not continue the retained request row')
            self._record('request_row_continued', transaction=self.tx.txn_seq,
                         previous_transaction=row['transaction'].txn_seq,
                         request_generation=self.tx.request_generation,
                         state_epoch=self.admission.state_epoch,
                         previous_state_epoch=row['state_epoch'],
                         chunk_start=self.tx.chunk_start, chunk_end=self.tx.chunk_end,
                         ranks=list(ranks))
            self._retained = None
            self._retained_cancel_request = None
            self._retained_cancel_acks = {}
        self._stage_fields(frame, 1)['phase'] = 'segments'
        self._record('chunk_open', transaction=self.tx.txn_seq, request=self.tx.request_uid,
                     request_generation=self.tx.request_generation,
                     chunk_start=self.tx.chunk_start, chunk_end=self.tx.chunk_end,
                     state_epoch=frame.open_state_epoch,
                     committed_end=self.admission.committed_end,
                     prompt_end=self.admission.prompt_end,
                     remaining_tokens=self.admission.remaining_tokens,
                     terminal=self.admission.terminal,
                     stages=[self.stage_of[rank] for rank in sorted(self.lanes)],
                     lane_storage_epochs=[self.lanes[rank].storage_epoch
                                          for rank in sorted(self.lanes)])

    # -- commit ---------------------------------------------------------------
    def _maybe_commit(self):
        oldest = self._frames[0] if self._frames else None
        super()._maybe_commit()
        if oldest is None or not oldest.dead or not self._frames:
            return
        successor = self._frames[0]
        self.overlap_counters['overlapped_commits'] += 1
        # C0 resets these at a chunk's begin; a successor began before this commit.
        self._both_ns = 0
        self._headroom_chunk = 0
        if self._retained is not None and successor.opened == {0, 1}:
            # Both stages already moved the row into n+1 before n committed.
            row = self._retained
            if (not row['transaction'].same_request(successor.chunk['tx'])
                    or row['transaction'] == successor.chunk['tx']):
                raise DriverError('a successor does not continue the committed row')
            self._record('request_row_continued', transaction=successor.chunk['tx'].txn_seq,
                         previous_transaction=row['transaction'].txn_seq,
                         request_generation=successor.chunk['tx'].request_generation,
                         state_epoch=successor.chunk['admission'].state_epoch,
                         previous_state_epoch=row['state_epoch'],
                         chunk_start=successor.chunk['tx'].chunk_start,
                         chunk_end=successor.chunk['tx'].chunk_end,
                         ranks=list(self._all_ranks), moved_before_commit=True)
            self._retained = None
            self._retained_cancel_request = None
            self._retained_cancel_acks = {}

    def _stage_line(self, transaction, stats, both_ns):
        super()._stage_line(transaction, stats, both_ns)
        observer = self.overlap_observer
        if observer is not None:
            info = dict(successor_of=stats.get('obs_successor_of'), early=stats.get('obs_early'),
                        deferred_bind=stats.get('obs_deferred'), stage_ranks=[list(self.stage_ranks[0]),
                                                                             list(self.stage_ranks[1])])
            try:
                logger = _progress_logger()
                for line in observer.lines(transaction, stats, info):
                    logger.info('%s', line)
            except Exception:
                logging.getLogger(__name__).debug('overlap observation line failed', exc_info=True)
        if self.progress is None or not self.overlap_selected:
            return   # a local or serial arm logs exactly C0's lines
        try:
            _progress_logger().info('flash-s5 stage-overlap txn=%d seqs=%s successor=%s',
                                    transaction.txn_seq, _ranges(stats.get('seqs', ())),
                                    'none' if not self._frames else self._frames[0].chunk['tx'].txn_seq)
        except Exception:
            logging.getLogger(__name__).debug('overlap line failed', exc_info=True)

    def _reset(self):
        frame = self._scoped_frame
        rest = [item for item in self._frames if item is not frame and not item.dead]
        progress = self.progress
        carry = None
        if rest and progress is not None and progress.recording:
            carry = dict(progress.native)
        stop = (self.stop_kind, self._stop_rounds, self._stop_ticks)
        super()._reset()
        if not rest:
            return
        self.stop_kind, self._stop_rounds, self._stop_ticks = stop
        if carry is not None and progress is not None and self._progress_owned():
            # One engine progress line per commit: the remaining chunk starts the next line,
            # keeping the rows of its commands still in flight.
            nxt = rest[0].chunk['tx']
            progress.begin(dict(txn=nxt.txn_seq, request_generation=nxt.request_generation,
                                chunk_start=nxt.chunk_start, chunk_end=nxt.chunk_end,
                                decode_live=self._progress_decode_live()))
            progress.native.update(carry)

    # -- stop -----------------------------------------------------------------
    def _drive_stop(self):
        controller = self.controller
        successor = controller.successor
        if (successor is not None and self._frames
                and not any(frame.chunk['tx'] == successor.transaction for frame in self._frames)):
            # Admitted, never begun: nothing was sent for it.
            controller.cancel(successor.transaction, never_submitted=True)
            self.overlap_counters['unbegun_successor_cancels'] += 1
            self._record('successor_cancel', transaction=successor.transaction.txn_seq,
                         never_begun=True, reason=self.stop_kind)
            return
        super()._drive_stop()

    def _stop_frame(self, frame):
        """Newest first.  One cancel per rank, for the chunk that rank holds.

        Per stage s, with frames [n, n+1]:
        * s still on n, or released n without opening n+1: n's cancel closes the row
          there (the worker's open or retained row); n+1 never reached s.
        * s opened n+1: n+1's cancel closes the row there; n's row on s had moved
          into n+1, so n's closed row on s is its local release bound to that cancel.
        n+1 retires first; n never commits during a stop.
        """
        # Every frame: finish a half-accepted command before any cancel (finding 2 on the
        # frozen trunk; the serial path, with no frame, keeps the frozen stop).
        if self._stop_obligations(frame) or self._stop_partial_segment(frame):
            return
        older = getattr(frame, 'successor_of', None)
        if older is None or older.dead:
            return super()._stop_frame(frame)
        closed = self._frame_closed_rows(frame)
        opened_ranks = {rank for stage in frame.opened for rank in self.stage_ranks[stage]}
        if opened_ranks <= set(closed):
            self._retire_successor(frame, closed, opened_ranks)
            return
        request = CancelRequest(frame.chunk['tx'], frame.chunk['admission'].state_epoch, self.stop_kind)
        self._stop_round()
        for stage in sorted(frame.opened):
            with self._scope(frame, stage):
                if set(self.ranks) <= set(closed) or self._inflight:
                    continue
                self._cancel_targets = frozenset(self.ranks)
                self._submit('cancel', ('flash_s5_cancel', (request,), {}, None), self.ranks)

    def _stop_partial_segment(self, frame):
        """Finish a frame's command one TP peer accepted and the other answered pending.

        The exact retained command (same transaction, op_seq, descriptor) is replayed to the
        stage's targets under the stop-round bound, as the frozen path does for tail and PP
        obligations, so both peers hold it before the cancel; never a new command.
        """
        for stage in sorted(frame.opened):
            with self._scope(frame, stage):
                command = self._retry
                if command is None:
                    continue
                # op_seq repeats across stages and accepted_tickets is the chunk's (both stages'):
                # only this command's own targets count (as the trunk fix, alpha-trunk-stopfix 5c7dbb5).
                targets = set(self._targets)
                accepted = {rank for rank, op_seq in self.accepted_tickets
                            if op_seq == command.op_seq and rank in targets}
                if not accepted or targets <= accepted:
                    continue
                if command.transaction != self.tx:
                    raise DriverError('stop found a foreign partially accepted command')
                self._stop_round()
                self._send_segment(command, self._targets)
                self.overlap_counters['partial_replays'] += 1
                return True
        return False

    def _frame_closed_rows(self, frame):
        with self._scope(frame):
            return self._closed_rows()

    def _retire_successor(self, frame, closed, opened_ranks):
        older = frame.successor_of
        with self._scope(frame):
            unopened = frozenset(self._all_ranks) - opened_ranks
            accepted = {rank for rank, _ in self.accepted_tickets}
            if accepted & unopened:
                raise DriverError('an unopened rank has accepted operations')
            rows = {}
            for rank in sorted(opened_ranks):
                ack = closed[rank]
                report = getattr(ack, 'report', None) or self.reports.get(rank)
                if report is None or report.pending_op_seqs or report.failed \
                        or report.completed_sequence != ack.completed_sequence:
                    raise DriverError('successor cancellation is not a clean terminal row')
                rows[rank] = ack if type(ack) is ReleaseAck else ack.release_ack(report.completed_sequence)
            if self.bound_ticket is None:
                if self.accepted_tickets:
                    raise DriverError('accepted operations exist without a bound producer')
                self.controller.cancel(self.tx, never_submitted=True)
                path = 'never-submitted'
            elif not unopened:
                for rank in self._all_ranks:
                    self.controller.note_worker_release(rows[rank])
                self.controller.cancel(self.tx, proof=self._producer_proof())
                path = 'all-ranks'
            else:
                self.controller.retire_successor(self.tx, proof=self._producer_proof(),
                                                 releases=rows, unopened=unopened)
                path = 'partial'
            transaction = self.tx
            for rank in opened_ranks:
                if self.stage_of[rank] not in older.released:
                    raise DriverError('a stage opened the successor before releasing the older chunk')
                # This rank's row of n had moved into n+1 and closed with it.
                older.closed_by_successor[rank] = rows[rank]
            # No closure memory here: that record is the all-rank close of the older chunk.
            self._record('cancel', transaction=transaction.txn_seq, never_submitted=path == 'never-submitted',
                         closed_ranks=sorted(opened_ranks), unopened_ranks=sorted(unopened),
                         successor=True, path=path, reason=self.stop_kind)
            self.overlap_counters['successor_retires'] += 1
            self._reset()

    def _closed_rows(self):
        rows = super()._closed_rows()
        # The older frame carries the rows its retired successor closed.
        moved = getattr(self._scoped_frame, 'closed_by_successor', None) or {}
        for rank, successor_row in moved.items():
            if rank in rows:
                continue
            local = self._release_acks.get(rank)
            if local is None or not local.released or local.request_closed:
                raise DriverError('a row closed by the successor lacks the local release')
            rows[rank] = replace(local, request_closed=True)
            self.overlap_counters['closed_by_successor'] += 1
        return rows

    # -- per-arm selection ------------------------------------------------------
    def _stage_control(self, command):
        """C0's control with one more field, ``overlap``.

        Overlap needs local: ``overlap=True`` with ``local`` not True is refused before anything
        is applied, and selecting serial selects no overlap.  Without the field, a local
        selection keeps the current overlap choice.
        """
        if type(command) is not dict:
            return super()._stage_control(command)
        overlap = command.get('overlap', None)
        if 'overlap' in command:
            if type(overlap) is not bool:
                raise DriverError('native-stage-overlap-invalid')
            if overlap and command.get('local') is not True:
                raise DriverError('native-stage-overlap-needs-local')
        reply = super()._stage_control({k: v for k, v in command.items() if k != 'overlap'})
        previous = self.overlap_selected
        if not reply['local']:
            overlap = False
        elif overlap is None:
            overlap = previous
        self.overlap_selected = overlap
        reply.update(overlap=overlap, previous_overlap=previous)
        return reply

    def stage_status(self):
        result = super().stage_status()
        result['overlap'] = dict(selected=self.overlap_selected, counters=dict(self.overlap_counters),
                                 controller=dict(self.controller.overlap_counters),
                                 refusals=dict(self.controller.overlap_refusals))
        if self.defer_bind:
            result['overlap']['defer_bind'] = dict(counters=dict(self.defer_counters))
        if self.overlap_observer is not None:
            result['overlap']['observe'] = dict(counters=dict(self.overlap_observer.counters))
        return result


def _ranges(values):
    """Sorted ints as 'a-b,c' runs; the grade attributes interleaved rows by sequence."""
    out, run = [], None
    for value in sorted(values):
        if run is not None and value == run[1] + 1:
            run[1] = value
            continue
        if run is not None:
            out.append(run)
        run = [value, value]
    if run is not None:
        out.append(run)
    return ','.join(str(a) if a == b else '%d-%d' % (a, b) for a, b in out) or '-'


__all__ = ['DEFER_BIND_ENV', 'OBSERVE_ENV', 'OBSERVE_LINE', 'OBSERVE_SCHEMA', 'OverlapObserver', 'OverlapSegmentDriver', 'REFUSALS', 'STAGE_OVERLAP_ENV', 'StagedAdmissionController',
           'overlap_refusal', 'provisional_fields', 'stage_overlap_defer_bind_enabled', 'stage_overlap_enabled',
           'stage_overlap_observe_enabled']
