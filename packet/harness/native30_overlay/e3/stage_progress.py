"""C0: stage-local prefill progress with overlap disabled (``FLASH_S5_STAGE_LOCAL=1``).

The frozen :class:`~engine_driver.NativeSegmentDriver` drives one chunk through
one all-rank command stream: every command waits for all four replies.  This
subclass keeps its command handlers and splits the stream by stage:

* each stage has its own cursor (phase, plan, index, retry, profile, tail);
* a command goes to its stage's ranks only, and only they answer
  (:mod:`stage_wire`), so applying it cannot depend on a non-target rank (D1);
* each stage releases its lane locally once its own work is queried complete;
  the sender's completed PP send stays owned on the worker until the receiver's
  consumer proof arrives;
* the scheduler commit is the frozen commit, once, after both stages released
  and every receipt was accepted.

Each rank receives the same commands in the same order as on the frozen path:

    stage 0: PREPARE, PREFIX, layers, SUFFIX, PP_SEND, STATE_FINALIZE
    stage 1: PREPARE, PP_RECEIVE, PREFIX, layers, SUFFIX, STATE_FINALIZE

so the numerics do not change.  Only one chunk is admitted at a time.

Selection: ``FLASH_S5_STAGE_LOCAL=1`` installs this class with staged progress
selected.  The typed ``native_stage_local`` control, accepted only while paused,
idle and FULL, selects per measurement arm on one engine incarnation.  The choice
is read once, before a chunk's open.  An unselected chunk has no frame and every
override below hands it to the frozen method, so it never sees a stage scope.

Form: the frozen handlers read their state from ``self``.  :meth:`_scope` binds
one chunk's fields and one stage's fields onto ``self`` around each call, so
those handlers run unchanged and later edits to them flow through inheritance.
Outside a scope ``self`` shows the oldest chunk, which is what the frozen
readers (``tx``, ``active``, ``stop``, abort and pause hooks) expect.
"""
from contextlib import contextmanager
from dataclasses import replace
import logging
import time

from e3_types import E3Pending
from engine_driver import (DriverError, NativeSegmentDriver, _WAIT_SEGMENT_KINDS,
                           _progress_logger)
from native_measurement import emit_receipt_kv
from reply_transport import Backpressure
from stage_wire import StageScoped
from wire_v1 import (CancelRequest, OpSeqTracker, OpenAck, PollRequest, ReleaseAck,
                     ReleaseRequest, SegmentKind)

STAGE_LOCAL_ENV = 'FLASH_S5_STAGE_LOCAL'
STAGE_CONTROL = 'native_stage_local'
# Slots a native command must leave writable for stock decode.  A stage's
# commands no longer wait for the other stage's ranks, so without this reserve
# they could fill the ring and block a stock enqueue on the owner thread.
RING_RESERVE = 3
# A finalize poll carries no work.  It also leaves slots for the other stage's next
# command: a stage polling at tick rate could otherwise take every slot a slow
# reader frees, and the peer whose command it is waiting for would never be sent.
POLL_RESERVE = RING_RESERVE + 2

_CHUNK_FIELDS = ('tx', 'admission', 'lanes', 'trackers', 'tickets', 'accepted_tickets',
                 '_accepted_descriptors', 'reports', 'bound_ticket', 'pp_send_ticket',
                 'receiver_ticket', 'last_ticket', '_pp_submission', 'pp_send_seq',
                 'pp_receive_seq', '_bindings', '_bind_index', '_pending_binding',
                 '_receipts', '_receipt_inflight', '_receipt_acked', '_receipt_last',
                 '_consumer_proofs', '_pp_noted', '_release_acks', '_stop_acks',
                 '_profile_grants')
_STAGE_FIELDS = ('phase', 'plan', 'index', '_retry', '_command', '_targets', 'open_request',
                 'release_request', '_release_attempts', '_profile_query',
                 '_profile_observed', '_profile_deadline', '_profile_poll_count',
                 '_tail_boundary', '_tail_submission', '_cancel_targets')
_WAIT_KINDS = ('segment', 'profile')


def stage_local_enabled(env):
    value = env.get(STAGE_LOCAL_ENV, '0')
    if value not in ('0', '1'):
        raise DriverError(STAGE_LOCAL_ENV + ' must be 0 or 1')
    return value == '1'


def ring_limiting_readers(sender, need):
    """Reader indices that still owe a read on a written slot of the next ``need``; a pure query.

    alpha-next C0 ring reading (assessment 4C.6): reported, never consulted.  Reader
    ``i`` is metadata column ``1 + i``, the ring's i-th local reader (declared: the
    workers attach in rank order, so reader i is the i-th rank of the executor).
    """
    queue = getattr(sender, 'queue', None)
    buffer = getattr(queue, 'buffer', None)
    metadata = getattr(buffer, 'get_metadata', None)
    index = getattr(queue, 'current_idx', None)
    size = getattr(buffer, 'max_chunks', None)
    if not callable(metadata) or type(index) is not int or type(size) is not int:
        return ()
    owing = set()
    for offset in range(min(need, size)):
        with metadata((index + offset) % size) as slots:
            if slots[0] and sum(slots[1:]) != buffer.n_reader:
                owing.update(i for i, flag in enumerate(slots[1:]) if not flag)
    return tuple(sorted(owing))


def ring_has_headroom(sender, need):
    """True when the next ``need`` broadcast slots are writable; a pure query.

    The same slot rule as ``BoundedSender._writable`` and the stock
    ``acquire_write``, over the same ring metadata.
    """
    queue = getattr(sender, 'queue', None)
    buffer = getattr(queue, 'buffer', None)
    metadata = getattr(buffer, 'get_metadata', None)
    index = getattr(queue, 'current_idx', None)
    size = getattr(buffer, 'max_chunks', None)
    if not callable(metadata) or type(index) is not int or type(size) is not int:
        raise DriverError('broadcast ring metadata is unavailable')
    for offset in range(min(need, size)):
        with metadata((index + offset) % size) as slots:
            if slots[0] and sum(slots[1:]) != buffer.n_reader:
                return False
    return True


class _ScopedSender:
    """The one bounded sender, with each envelope wrapped for its target ranks."""
    __slots__ = ('driver', 'real', 'ranks')

    def __init__(self, driver, real, ranks):
        self.driver, self.real, self.ranks = driver, real, ranks

    def _wrap(self, envelope):
        method, args, kwargs, rank = envelope
        scoped = StageScoped(args[0], self.ranks,
                             local_release=method == 'flash_s5_release', ahead=self._ahead(method, args[0]))
        return method, (scoped,) + tuple(args[1:]), kwargs, rank

    def _ahead(self, method, message):
        """Build-2 (a): the successor frame's admission for this stage, on this stage's STATE_FINALIZE submit and on its
        polls and release from the finalize phase on; None otherwise, or with the setting off, or once the successor opened
        on this stage."""
        driver = self.driver
        if not getattr(driver, 'prepare_ahead', False):
            return None
        frame, stage = getattr(driver, '_scoped_frame', None), getattr(driver, '_scoped_stage', None)
        frames = getattr(driver, '_frames', ())
        if frame is None or stage is None or frame not in frames:
            return None
        index = frames.index(frame)
        if index + 1 >= len(frames):
            return None
        successor = frames[index + 1]
        if successor.dead or stage in successor.opened:
            return None
        if method == 'flash_s5_submit':
            from wire_v1 import SegmentKind
            if getattr(message, 'kind', None) is not SegmentKind.STATE_FINALIZE:
                return None
        elif method in ('flash_s5_poll', 'flash_s5_release'):
            if driver.__dict__.get('phase') not in ('finalize', 'release'):
                return None
        else:
            return None
        request = successor.stage[stage].get('open_request')
        admission = getattr(request, 'admission', None)
        if admission is not None and driver.__dict__.get('ahead_nosync') is True:
            # parity-ahead: the arm's selection; the worker stages this record with pinned non-blocking copies.
            from stage_wire import AheadNoSync
            counters = driver.__dict__.get('ahead_nosync_counters')
            if counters is not None:
                counters['sent'] += 1
            return AheadNoSync(admission)
        return admission

    def check(self, envelope):
        self.driver._check_headroom(self.real, envelope)
        return self.real.check(self._wrap(envelope))

    def __call__(self, envelope):
        return self.real(self._wrap(envelope))


class ChunkFrame:
    """One admitted chunk: chunk-wide fields plus one cursor per stage."""

    def __init__(self, chunk, stages):
        self.chunk = chunk
        self.stage = stages
        self.opened = set()
        self.released = set()
        self.dead = False
        self.open_state_epoch = None
        self.stats = dict(commands=[0, 0], segment_ns=[0, 0], first_ns=[None, None],
                          last_ns=[None, None], local_release_ns=[None, None])


class StagedSegmentDriver(NativeSegmentDriver):
    """Per-stage command streams on the one ring; see the module docstring."""

    def __init__(self, controller, **kwargs):
        super().__init__(controller, **kwargs)
        self._all_ranks = self.ranks
        self.stage_selected = True
        self._frames = []
        self._scoped_frame = None
        self._scoped_stage = None
        self._submit_ranks = None
        # sequence -> (frame, stage); a sequence absent here is not a chunk command.
        self._seq_route = {}
        self._command = None
        self._targets = ()
        self.open_request = None
        self.release_request = None
        self.stage_counters = dict(chunks=0, headroom_refused=0, local_releases=0,
                                   both_stages_ns=0, selections=0)
        self._busy_since = {0: None, 1: None}
        self._both_since = None
        self._both_ns = 0
        self._headroom_chunk = 0
        # alpha-next ring reading: time unable to submit and the reader that limits
        # reclamation, beside headroom_refused (recorded, never consulted).
        self.stage_ring = dict(blocked_ns=0, episodes=0, blocked_max_ns=0,
                               limiting={}, limiting_sole={})
        self._ring_blocked_since = None
        self._ring_blocked_chunk_ns = 0

    # -- scope ------------------------------------------------------------
    @contextmanager
    def _scope(self, frame, stage=None):
        """Bind ``frame`` (and one stage cursor) onto ``self`` for one handler."""
        if self._scoped_frame is not None:
            raise DriverError('stage scopes do not nest')
        attrs = self.__dict__
        attrs.update(frame.chunk)
        rest_inflight = attrs['_inflight']
        if stage is not None:
            attrs.update(frame.stage[stage])
            attrs['ranks'] = self.stage_ranks[stage]
            attrs['_inflight'] = {sequence: kind for sequence, kind in rest_inflight.items()
                                  if self._route_stage(sequence) == stage}
        self._scoped_frame, self._scoped_stage = frame, stage
        try:
            yield frame
        finally:
            self._scoped_frame, self._scoped_stage = None, None
            if stage is not None:
                for sequence, kind in attrs['_inflight'].items():
                    rest_inflight.setdefault(sequence, kind)
                attrs['_inflight'] = rest_inflight
                attrs['ranks'] = self._all_ranks
            if not frame.dead:
                frame.chunk = {name: attrs[name] for name in _CHUNK_FIELDS}
                if stage is not None:
                    frame.stage[stage] = {name: attrs[name] for name in _STAGE_FIELDS}
            self._show_rest()

    def _route_stage(self, sequence):
        route = self._seq_route.get(sequence)
        return None if route is None else route[1]

    def _show_rest(self):
        """Outside a scope ``self`` shows the oldest live chunk."""
        if not self._frames:
            return
        oldest = self._frames[0]
        attrs = self.__dict__
        attrs.update(oldest.chunk)
        phases = {oldest.stage[stage]['phase'] for stage in (0, 1)}
        for name in ('segments', 'profile', 'bind', 'opened', 'open', 'finalize', 'release'):
            if name in phases:
                attrs['phase'] = name
                break
        else:
            attrs['phase'] = 'released'

    def _stage_fields(self, frame, stage):
        """A stage's cursor, wherever it is bound right now."""
        if self._scoped_frame is frame and self._scoped_stage == stage:
            return self.__dict__
        return frame.stage[stage]

    # -- bounded transport ------------------------------------------------
    def _bounded(self):
        real = super()._bounded()
        if self._scoped_stage is None:
            return real
        return _ScopedSender(self, real, self._submit_ranks)

    def _check_headroom(self, real, envelope):
        if self.stop_kind is not None:
            self._ring_unblocked()
            return    # A stop only needs the ordinary writable slot.
        poll = envelope[0] == 'flash_s5_poll' and type(envelope[1][0]) is PollRequest
        need = (POLL_RESERVE if poll else RING_RESERVE) + 1
        if not ring_has_headroom(real, need):
            self.stage_counters['headroom_refused'] += 1
            self._headroom_chunk += 1
            self._ring_blocked(real, need)
            raise Backpressure('ring_headroom')
        self._ring_unblocked()

    def _ring_blocked(self, real, need):
        """Receipt only: open a blocked episode and name the readers owing a read."""
        try:
            if self._ring_blocked_since is None:
                self._ring_blocked_since = time.monotonic_ns()
                self.stage_ring['episodes'] += 1
            owing = ring_limiting_readers(real, need)
            ranks = self._all_ranks
            names = [ranks[i] if i < len(ranks) else i for i in owing]
            for rank in names:
                self.stage_ring['limiting'][rank] = self.stage_ring['limiting'].get(rank, 0) + 1
            if len(names) == 1:
                rank = names[0]
                self.stage_ring['limiting_sole'][rank] = self.stage_ring['limiting_sole'].get(rank, 0) + 1
        except Exception:
            pass

    def _ring_unblocked(self):
        since, self._ring_blocked_since = self._ring_blocked_since, None
        if since is None:
            return
        elapsed = time.monotonic_ns() - since
        self.stage_ring['blocked_ns'] += elapsed
        self.stage_ring['blocked_max_ns'] = max(self.stage_ring['blocked_max_ns'], elapsed)
        self._ring_blocked_chunk_ns += elapsed

    def _submit_observed(self, kind, envelope, ranks, *, timeout_s=None):
        frame, stage = self._scoped_frame, self._scoped_stage
        if frame is None or stage is None:
            if frame is not None:
                raise DriverError('a chunk command needs a stage scope')
            return super()._submit_observed(kind, envelope, ranks, timeout_s=timeout_s)
        ranks = tuple(ranks)
        if not set(ranks) <= set(self.stage_ranks[stage]):
            raise DriverError('a stage command targets another stage')
        self._submit_ranks = ranks
        try:
            sequence = super()._submit_observed(kind, envelope, ranks, timeout_s=timeout_s)
        finally:
            self._submit_ranks = None
        self._seq_route[sequence] = (frame, stage)
        now = time.monotonic_ns()
        stats = frame.stats
        stats['commands'][stage] += 1
        if stats['first_ns'][stage] is None:
            stats['first_ns'][stage] = now
        if kind in _WAIT_KINDS:
            self._note_busy(stage, now, True)
        return sequence

    def _note_busy(self, stage, now, busy):
        """Seconds with a segment or profile command outstanding on both stages."""
        self._busy_since[stage] = now if busy else None
        both = all(value is not None for value in self._busy_since.values())
        if both and self._both_since is None:
            self._both_since = now
        elif not both and self._both_since is not None:
            self._both_ns += now - self._both_since
            self._both_since = None

    def _apply(self, kind, values, sequence=None):
        route = self._seq_route.pop(sequence, None)
        if route is None:
            return super()._apply(kind, values, sequence)
        frame, stage = route
        now = time.monotonic_ns()
        frame.stats['last_ns'][stage] = now
        if kind in _WAIT_KINDS:
            self._note_busy(stage, now, False)
        with self._scope(frame, stage):
            if kind == 'open':
                self._apply_stage_open(values)
            elif kind == 'release':
                self._apply_local_release(values)
            else:
                super()._apply(kind, values, sequence)

    # -- chunk begin and open ---------------------------------------------
    def _new_frame(self):
        attrs = self.__dict__
        stages = {}
        for stage in (0, 1):
            stages[stage] = dict(
                phase='open', plan=[], index=0, _retry=None, _command=None, _targets=(),
                open_request=replace(attrs['open_request'], ranks=self.stage_ranks[stage]),
                release_request=None, _release_attempts=0, _profile_query=None,
                _profile_observed={}, _profile_deadline=None, _profile_poll_count=0,
                _tail_boundary=None, _tail_submission=None, _cancel_targets=frozenset())
        chunk = {name: attrs[name] for name in _CHUNK_FIELDS}
        for name in ('lanes', 'trackers', 'tickets'):
            chunk[name] = {}
        return ChunkFrame(chunk, stages)

    def _begin(self):
        if self._frames:
            raise DriverError('stage-local progress runs one chunk at a time')
        super()._begin()
        if not self.stage_selected or self.tx is None:
            return
        frame = self._new_frame()
        self._frames.append(frame)
        self._both_ns = 0
        self._headroom_chunk = 0
        self._ring_blocked_chunk_ns = 0
        self._show_rest()

    def _apply_stage_open(self, values):
        """The frozen open checks, per stage; the chunk row moves once all opened."""
        frame, stage = self._scoped_frame, self._scoped_stage
        acks = tuple(values)
        if len(acks) != len(self.ranks):
            raise DriverError('open reply does not cover every stage rank')
        request = self.open_request
        seen = set()
        for ack in acks:
            if type(ack) is not OpenAck or not ack.matches(request):
                raise DriverError('open acknowledgment is stale or mistyped')
            if not ack.accepted:
                raise DriverError('worker refused the chunk open')
            rank = ack.lane.worker_rank
            if rank in seen:
                raise DriverError('duplicate open acknowledgment')
            seen.add(rank)
            if ack.pp_stages != 2 or ack.lane.role.value != 'prefill':
                raise DriverError('open acknowledged the wrong lane or PP topology')
            if ack.stage != stage or ack.stage != self.stage_of[rank]:
                raise DriverError('worker stage disagrees with the split')
            self.lanes[rank] = ack.lane
            prior_lane, last = self._last_lane_sequences.get(rank, (None, -1))
            self.trackers[rank] = OpSeqTracker(last if prior_lane == ack.lane else -1)
        frame.opened.add(stage)
        self.phase = 'opened'
        frame.open_state_epoch = acks[0].state_epoch
        if frame.opened == {0, 1}:
            self._chunk_opened(frame)

    def _chunk_opened(self, frame):
        """Every rank opened: the frozen row move, plan, owner binding and journal."""
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
        plans = {0: [], 1: []}
        for entry in self._build_plan():
            plans[entry['stage']].append(entry)
        self.controller.note_source_owners(self.tx, self.lanes, self.layer_count)
        self._bindings = {}
        self._bind_index = 0
        self._pending_binding = None
        for stage in (0, 1):
            fields = self._stage_fields(frame, stage)
            fields['plan'] = plans[stage]
            fields['index'] = 0
            fields['phase'] = 'bind' if stage == 0 else 'segments'
        self._record('chunk_open', transaction=self.tx.txn_seq, request=self.tx.request_uid,
                     request_generation=self.tx.request_generation,
                     chunk_start=self.tx.chunk_start, chunk_end=self.tx.chunk_end,
                     state_epoch=frame.open_state_epoch,
                     committed_end=self.admission.committed_end,
                     prompt_end=self.admission.prompt_end,
                     remaining_tokens=self.admission.remaining_tokens,
                     terminal=self.admission.terminal,
                     native_end=getattr(self.admission, 'native_end', self.admission.prompt_end - 1),
                     stages=[self.stage_of[rank] for rank in sorted(self.lanes)],
                     lane_storage_epochs=[self.lanes[rank].storage_epoch
                                          for rank in sorted(self.lanes)])

    def _prepare_source(self, targets):
        # The source hold names every opened rank; only the send is stage-local.
        stage_ranks = self.ranks
        self.ranks = self._all_ranks
        try:
            return super()._prepare_source(targets)
        finally:
            self.ranks = stage_ranks

    # -- per-stage drive --------------------------------------------------
    def _chunk_command_blocked(self):
        """A non-chunk command (route, control) still holds the stream, as on trunk."""
        return any(sequence not in self._seq_route for sequence in self._inflight)

    def _drive(self):
        if not self._frames:
            return super()._drive()
        if self._chunk_command_blocked():
            return
        self._maybe_commit()
        polls = []
        for frame in tuple(self._frames):
            for stage in (0, 1):
                if not frame.dead and self._drive_stage(frame, stage):
                    polls.append((frame, stage))
        # After every stage's real work, so a poll never takes a command's slot.
        for frame, stage in polls:
            with self._scope(frame, stage):
                self._submit('poll', ('flash_s5_poll',
                                      (PollRequest(self.tx, self.admission.state_epoch,
                                                   self.poll_budget),), {}, None), self.ranks)

    def _segment_gate(self, frame, stage):
        """True when this stage's next plan entry waits on the other stage."""
        if self.index >= len(self.plan) or self._retry is not None:
            return False
        kind = self.plan[self.index]['kind']
        if stage == 1 and kind is SegmentKind.PREPARE:
            # The controller binds the producer to stage 0's PREPARE.
            return self.bound_ticket is None
        if stage == 1 and kind is SegmentKind.PP_RECEIVE:
            obligation = self._pp_submission
            return (obligation is None
                    or obligation.send_accepted != set(self.stage_ranks[0]))
        if stage == 0 and kind is SegmentKind.PP_SEND:
            # An accepted send always has an opened, prepared peer.
            return frame.stage[1]['index'] == 0 or frame.stage[1]['phase'] in ('open', 'opened')
        return False

    def _drive_stage(self, frame, stage):
        """One command on this stage, if it has one; True when it only wants a poll."""
        with self._scope(frame, stage):
            if self._inflight:
                return False
            phase = self.phase
            if phase == 'opened':
                return False
            if phase == 'bind':
                self._next_binding()
                return False
            if phase == 'finalize':
                return self._drive_stage_finalize(stage)
            if phase == 'released':
                if stage == 0:
                    self._next_receipt()
                return False
            if phase == 'segments' and self._segment_gate(frame, stage):
                return False
            # open, profile, segments and release are the frozen handlers: the
            # scope makes their submissions target only this stage.
            super()._drive()
            return False

    def _stage_drained(self):
        """Every accepted op and state write on this stage's ranks, host-queried."""
        for rank in self.ranks:
            report = self.reports.get(rank)
            if report is None or report.failed or report.pending_op_seqs:
                return False
            expected = {op for r, op in self.accepted_tickets if r == rank}
            if not expected <= set(report.completed_op_seqs) or not report.state_written():
                return False
        return True

    def _drive_stage_finalize(self, stage):
        if stage == 0 and self._next_receipt():
            return False
        if self._stage_drained():
            self.phase = 'release'
            self._send_release()
            return False
        return True

    def _apply_local_release(self, values):
        """The frozen release checks for one stage; the commit is separate."""
        frame, stage = self._scoped_frame, self._scoped_stage
        acks = tuple(values)
        if (self.phase != 'release' or len(acks) != len(self.ranks)
                or any(type(ack) is not ReleaseAck for ack in acks)
                or {ack.worker_rank for ack in acks} != set(self.ranks)):
            raise DriverError('local release lacks the complete typed stage reply')
        expected = dict(self.release_request.completed_sequences)
        terminal = self.release_request.terminal
        for ack in acks:
            if (ack.transaction != self.tx or ack.state_epoch != self.admission.state_epoch
                    or ack.completed_sequence != expected[ack.worker_rank]):
                raise DriverError('release acknowledgment differs from the final report')
            if not terminal and ack.request_closed:
                raise DriverError('nonterminal release closed the request row')
        for ack in acks:
            prior = self._release_acks.get(ack.worker_rank)
            if prior is not None and ack != prior:
                raise DriverError('release retry contradicts an earlier accepted row')
            if ack.closed if terminal else ack.released:
                self._release_acks[ack.worker_rank] = ack
        if self.stop_kind is not None:
            return
        if set(self.ranks) <= set(self._release_acks):
            self.phase = 'released'
            frame.released.add(stage)
            frame.stats['local_release_ns'][stage] = time.monotonic_ns()
            self.stage_counters['local_releases'] += 1

    def _maybe_commit(self):
        """The frozen commit, once: both stages released and every receipt accepted."""
        if not self._frames:
            return
        frame = self._frames[0]
        if frame.released != {0, 1}:
            return
        chunk = frame.chunk
        if (not chunk['_pp_noted'] or chunk['_receipt_inflight'] is not None
                or set(chunk['_receipt_acked']) != set(self.pp_peers)):
            return
        if any(route[0] is frame for route in self._seq_route.values()):
            return
        stats = frame.stats
        both_ns = self._both_ns
        with self._scope(frame):
            if not self.controller.ready_to_release(self.tx):
                raise DriverError('stage-local commit without release readiness')
            self.phase = 'release'
            self.release_request = ReleaseRequest(
                self.tx, self.admission.state_epoch,
                tuple((rank, self.reports[rank].completed_sequence) for rank in self.ranks),
                self.admission.terminal)
            transaction = self.tx
            self._apply_release(tuple(self._release_acks[rank] for rank in self.ranks))
        self.stage_counters['chunks'] += 1
        self.stage_counters['both_stages_ns'] += both_ns
        self._stage_line(transaction, stats, both_ns)

    def _stage_line(self, transaction, stats, both_ns):
        """One per-chunk receipt line; a logging failure is never an engine failure."""
        if self.progress is None:
            return
        try:
            # Alpha-next PLAN 2: one atomic write through the process receipt writer.
            emit_receipt_kv(
                'flash-s5 stage-local txn=%d chunk=%d-%d commands=%d,%d both_stages_ns=%d '
                'first_ns=%s,%s last_ns=%s,%s local_release_ns=%s,%s headroom_refused=%d '
                'headroom_blocked_ns=%d'
                % (transaction.txn_seq, transaction.chunk_start, transaction.chunk_end,
                   stats['commands'][0], stats['commands'][1], both_ns,
                   stats['first_ns'][0], stats['first_ns'][1],
                   stats['last_ns'][0], stats['last_ns'][1],
                   stats['local_release_ns'][0], stats['local_release_ns'][1],
                   self._headroom_chunk, self._ring_blocked_chunk_ns),
                _progress_logger())
        except Exception:
            logging.getLogger(__name__).debug('stage line failed', exc_info=True)

    def _reset(self):
        frame = self._scoped_frame
        super()._reset()
        if frame is not None:
            frame.dead = True
            self._frames = [item for item in self._frames if item is not frame]
        self._seq_route = {sequence: route for sequence, route in self._seq_route.items()
                           if route[0] is not frame}

    # -- burst in-wait progress -------------------------------------------
    def _stage_wait_ok(self, frame, stage):
        """The frozen in-wait permit, for one stage's stream."""
        cursor = frame.stage[stage]
        inflight = tuple(kind for sequence, kind in self._inflight.items()
                         if self._route_stage(sequence) == stage)
        phase = cursor['phase']
        if phase == 'profile':
            return inflight in ((), ('profile',))
        if phase != 'segments' or len(inflight) > 1:
            return False
        if inflight:
            return (inflight == ('segment',) and cursor['_command'] is not None
                    and cursor['_command'].kind in _WAIT_SEGMENT_KINDS)
        if cursor['_retry'] is not None:
            return cursor['_retry'].kind in _WAIT_SEGMENT_KINDS
        return (cursor['index'] < len(cursor['plan'])
                and cursor['plan'][cursor['index']]['kind'] in _WAIT_SEGMENT_KINDS)

    def _wait_permit(self, nested=False):
        if not self._frames:
            return super()._wait_permit(nested)
        progress = self.progress
        if (progress is None or not progress.burst() or (self._in_wait and not nested)
                or self.error is not None or self.stop_kind is not None
                or self._profile_failure is not None or self._control is not None
                or self._control_orphans or self._route_exchange() is not None
                or self._return_pending is not None or not self._bootstrapped
                or not self._split_ready or self.tx is None or not self._progress_owned()
                or self._chunk_command_blocked()):
            return False
        return any(self._stage_wait_ok(frame, stage)
                   for frame in self._frames for stage in (0, 1))

    def _drain_observed(self):
        if not self._in_wait or not self._frames:
            return super()._drain_observed()
        # In a stock wait only an in-chunk command's reply is applied.  The rest
        # (PREPARE, PP, finalize, polls, receipts, release, open, bind) stays on
        # the tick, before schedule(), exactly as the frozen permit keeps it.
        held = {}
        for sequence, kind in self._inflight.items():
            route = self._seq_route.get(sequence)
            cursor = None if route is None else route[0].stage[route[1]]
            if (route is None or kind not in _WAIT_KINDS
                    or (kind == 'segment'
                        and cursor['_command'].kind not in _WAIT_SEGMENT_KINDS)):
                held[sequence] = kind
        for sequence in held:
            del self._inflight[sequence]
        try:
            return super()._drain_observed()
        finally:
            self._inflight.update(held)

    def progress_in_wait(self):
        """Per-stage ``_drain`` + ``_drive`` under the permit; returns commands sent."""
        if not self._frames:
            return super().progress_in_wait()
        if not self._wait_permit() or not self.progress.wait_allowed():
            return None
        self._in_wait = True
        self._progress_origin = 'wait'
        sent = 0
        try:
            self._drain()
            if not self._wait_permit(nested=True) or not self.progress.wait_allowed():
                return 0
            for frame in tuple(self._frames):
                for stage in (0, 1):
                    if frame.dead or not self._stage_wait_ok(frame, stage):
                        continue
                    if any(self._route_stage(sequence) == stage for sequence in self._inflight):
                        continue
                    before = len(self._inflight)
                    with self._scope(frame, stage):
                        if not (self.phase == 'segments' and self._segment_gate(frame, stage)):
                            super()._drive()
                    sent += len(self._inflight) - before
            return sent
        except Backpressure:
            return None if not sent else sent
        except E3Pending:
            raise
        except BaseException:
            self.error = 'native driver failed; ownership retained'
            raise
        finally:
            self._in_wait = False
            self._progress_origin = 'tick'

    # -- stop path --------------------------------------------------------
    def _drive_stop(self):
        if not self._frames:
            super()._drive_stop()
            return
        if self._inflight and any(kind != 'poll' for kind in self._inflight.values()):
            # A reply is owed; the transport's deadline bounds the wait.
            return
        if self._transition is not None:
            if self._transition_retry:
                self._send_transition()
            return
        self._stop_frame(self._frames[-1])

    def _stop_obligations(self, frame):
        """Replay an accepted tail or PP command to the peers still owing it."""
        for stage in (0, 1):
            with self._scope(frame, stage):
                if self._drive_tail_stop_submission():
                    return True
        obligation = frame.chunk['_pp_submission']
        if obligation is not None and obligation.send_accepted:
            stage = 0 if obligation.send_accepted != set(self.stage_ranks[0]) else 1
            with self._scope(frame, stage):
                if self._drive_pp_stop_submission():
                    return True
        with self._scope(frame, 0):
            if self._receipts_pending():
                self._stop_round()
                self._next_receipt()
                return True
        # Stop fix 2 on the staged driver (alpha-next): a stage segment one TP peer
        # accepted while the other answered pending is replayed, exact, to that
        # stage's targets before any cancel.  Each stage has its own retry cursor;
        # the frozen method filters accepted tickets by its targets (op_seq is per lane).
        for stage in (0, 1):
            with self._scope(frame, stage):
                if self._drive_partial_stop_submission():
                    return True
        return False

    def _stop_retained_rows(self, frame):
        """Close a retained row on a stage that never opened this frame."""
        row = self._retained
        if row is None:
            return False
        unopened = [stage for stage in (0, 1) if stage not in frame.opened]
        if not unopened:
            raise DriverError('a retained row outlived the chunk open')
        pending = [stage for stage in unopened
                   if not all(rank in self._retained_cancel_acks
                              and self._retained_cancel_acks[rank].closed
                              for rank in self.stage_ranks[stage])]
        if not pending:
            # Opened stages moved the row into this frame; unopened ones closed it.
            self._retained = None
            self._retained_cancel_request = None
            self._retained_cancel_acks = {}
            return False
        self._stop_round()
        if self._retained_cancel_request is None:
            self._retained_cancel_request = CancelRequest(
                row['transaction'], row['state_epoch'], self.stop_kind or 'cancel')
        for stage in pending:
            with self._scope(frame, stage):
                self._submit('retained_cancel',
                             ('flash_s5_cancel', (self._retained_cancel_request,), {}, None),
                             self.ranks)
        return True

    def _stop_frame(self, frame):
        if self._stop_obligations(frame):
            return
        if self._stop_retained_rows(frame):
            return
        with self._scope(frame):
            closed = self._closed_rows()
            if set(closed) == set(self.ranks):
                self._retire_cancel(closed)
                return
            request = CancelRequest(self.tx, self.admission.state_epoch, self.stop_kind)
        self._stop_round()
        for stage in (0, 1):
            with self._scope(frame, stage):
                if set(self.ranks) <= set(closed):
                    continue
                if self._inflight:
                    continue
                self._cancel_targets = frozenset(self.ranks)
                self._submit('cancel', ('flash_s5_cancel', (request,), {}, None), self.ranks)

    # -- per-arm selection and status --------------------------------------
    def _submit_control(self, command):
        if type(command) is dict and command.get('action') == STAGE_CONTROL:
            return self._stage_control(command)
        return super()._submit_control(command)

    def _stage_control(self, command):
        """Typed per-arm selection; the same paused idle FULL rule as the other arm switches."""
        if self.error is not None:
            raise DriverError(self.error)
        if set(command) - {'action', 'local', 'engine_epoch'}:
            raise DriverError('native-stage-local-command-fields-differ')
        epoch = command.get('engine_epoch')
        if epoch is not None and (type(epoch) is not int or epoch != self.engine_epoch):
            raise DriverError('native-control-engine-epoch-differs')
        local = command.get('local')
        if type(local) is not bool:
            raise DriverError('native-stage-local-invalid')
        policy = getattr(self, 'auto_arm', None)
        recording = self.progress is not None and self.progress.recording
        if (not self._engine_paused() or self._split_ready or self.tx is not None or self._frames
                or self.controller.active or self._inflight or self._control is not None
                or self._route_exchange() is not None or self._return_pending is not None
                or self._retained is not None or self._admission_intent is not None
                or (policy is not None and policy.active) or recording):
            raise DriverError('native-stage-local-requires-paused-idle-full')
        previous = self.stage_selected
        self.stage_selected = local
        self.stage_counters['selections'] += int(previous != local)
        now = int(self._monotonic_ns())
        return {'schema': 1, 'action': STAGE_CONTROL, 'local': local, 'previous': previous,
                'engine_epoch': self.engine_epoch, 'route': 'full',
                'route_generation': self.route_generation,
                'observed_clock_ns': now, 'clock': self._clock_record(now)}

    def _status_record(self, now=None):
        result = super()._status_record(now)
        result['native_stage_local'] = self.stage_status()
        return result

    def stage_status(self):
        """Bounded status row: the selection, counters and each live chunk's stage cursors."""
        return dict(schema='s5-stage-local/1', selected=self.stage_selected,
                    ring_reserve=RING_RESERVE, poll_reserve=POLL_RESERVE,
                    counters=dict(self.stage_counters),
                    ring=dict(self.stage_ring, limiting=dict(self.stage_ring['limiting']),
                              limiting_sole=dict(self.stage_ring['limiting_sole']),
                              blocked_open=self._ring_blocked_since is not None),
                    frames=[dict(transaction=frame.chunk['tx'].txn_seq,
                                 opened=sorted(frame.opened), released=sorted(frame.released),
                                 phases=[frame.stage[stage]['phase'] for stage in (0, 1)],
                                 commands=list(frame.stats['commands']))
                            for frame in self._frames])


__all__ = ['ChunkFrame', 'POLL_RESERVE', 'RING_RESERVE', 'STAGE_CONTROL', 'STAGE_LOCAL_ENV', 'StagedSegmentDriver',
           'ring_has_headroom', 'stage_local_enabled']
