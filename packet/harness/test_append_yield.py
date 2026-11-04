"""returning-append Option A (A1): the chunk-boundary yield of an auto-arm bulk to one short native request. CPU only.

Lead decision 3 Nov ~04:35 CET (Option A accepted, f3951ee0) and ~05:00 CET (A1). The yield, in five driver phases:

  closing   at a bulk chunk boundary the bulk's retained worker row closes through the existing typed all-rank
            CancelRequest (wire reason 'pause', journal reason 'yield'), no publication or frontier move, route kept split;
  append    the short held request (<= FLASH_S5_APPEND_YIELD_MAX_NEW new tokens) runs natively from its cached start;
  handback  its terminal commit returns FULL like every native request's; only its one-token handback runs there,
            every other stock prefill (the bulk first) stays held;
  reenter   auto-arm's own entry sequence (drain, barrier, identity, enter_split, arm) for the bulk, bound to its
            request, plan from its committed frontier, no 32,768 floor;
  resume    split and armed; only the bulk may be admitted; its first chunk from the frontier ends the yield.

Suites (each over the real modules of this tree; leaves in append_yield_fixture_leaves.json):
  Config            launch bounds (all four or none), the startup wiring, the per-arm control.
  Candidate         the yield candidate rule over the real controller.
  YieldAtBoundary   the connected cycle: real PrefillAdmissionController, NativeSegmentDriver, AutoArmPolicy and four
                    real bridge workers (test_e3_native_overlap.cold_stack); Codex items 1-4 (drained close, fresh
                    transaction and row identities, no block reuse, reopen from the committed frontier).
  OffIsFrozen       settings absent or control off: the frozen hold-through-FULL path, journal and counters.
  Holds             what the FULL window releases and holds.
  Stops             stop/abort at every yield state (the stop-matrix cells for the yield).
  RealBuilderState  the real LaneChunkFactory / LaneAlign over the reviewed fake runner with the pinned Mamba align
                    arithmetic: a yielded bulk's chunks see the same state as an uninterrupted bulk's, with row-slot
                    reuse and chunks crossing Mamba block boundaries.
  Mutants           each named engine mutation fails a test here.
Phase 2b (lead resume 5):
  *Staged           B1: YieldAtBoundary, Stops, Mutants, HandbackBound, ReentryRefused and ReentryNeverRaises rerun on
                    StagedSegmentDriver (the box driver); StagedIdentity: both drivers leave the same persistent state and
                    yield journal; StagedMutants: mutants of the staged overrides the yield crosses.
  HandbackBound     B2: every end of the FULL window, FLASH_S5_APPEND_YIELD_HANDBACK_STEPS counted in engine steps.
  ReentryRefused    B3: FLASH_S5_APPEND_YIELD_REENTRY_RETRIES refused passes, then the existing refused-admission rule.
  ReentryNeverRaises  the re-entry's other refusals close by the frozen restore / rollback rules.
Review of f76a31de (lead, fixed in this round; each has a *Staged twin):
  ReviewMajor1StaleYield      a profile refusal in the append runs the yield's abort rule; the stale-yield guard.
  ReviewMajor2AppendRefused   a refused append admission exits the append phase (resume; partial: the append row
                              closes, then resume); the refused append is not reselected for that bulk.
  ReviewMajor3ReentryBounded  drain / identity refusals in re-entry are bounded by REENTRY_RETRIES.
  Candidate / Mutants         MAJOR-3 slot bound (running + 2), contiguous decode counts, the free-block check.
  ReviewMinor1NoDeadlineAtResume  no stale control deadline at resume after append_left.
  Stops                       MINOR-2: a reset in any yield state ends it ('stop-reset'), FULL plus the stock tail.
  YieldExitInvariant          binding: EXIT_TABLE, phase (candidate, close, append, FULL, re-entry, resume) x event
                              (stop, abort, reset, refusal, client exit, none), each cell's exit within EXIT_STEPS
                              engine steps (no clock), plus mutants that each remove one exit; delta review rows:
                              resume entered via append_left / refused append / partial row close.
Delta review of ae865d44 (each has a *Staged twin):
  DeltaMajor1ResumeBounded    every path into resume exits within REENTRY_RETRIES passes counted from resume's entry
                              when the bulk's reopen is refused; mutant: the last arm's snapshot (hangs).
  DeltaMinor1HandbackSkipped  an append with more than its handback token left skips the FULL window ('handback-skipped').
  DeltaMinor2DrainRetrySpaced a drain refusal waits one scheduler pass, pause and running-prefill hold released; a retry
                              succeeds after the in-flight work clears.
No GPU, CUDA or box claim: chunk timing, decode stalls and numerics are the box's.
"""
import ast
from contextlib import contextmanager
import enum
from pathlib import Path
import sys
import types
import unittest
from unittest import mock
import weakref

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'patches/s5_engine')]
import engine_driver as ed  # noqa: E402
import native_auto_arm as auto  # noqa: E402
import scheduler_hooks as sh  # noqa: E402
from e3_types import E3ContractError  # noqa: E402
import test_e3_admission_native as adm  # noqa: E402
import test_e3_native_overlap as native  # noqa: E402
import native_overlap_worker as nw  # noqa: E402
import test_e3_r4_worker_bridge as bridge_suite  # noqa: E402
import test_native39_trunk as prior  # noqa: E402
import test_e3_lane_chunk as lc  # noqa: E402
from e3_types import TransactionId  # noqa: E402
from lane_chunk import ChunkAdmission, LaneChunkFactory  # noqa: E402

NS = types.SimpleNamespace
BULK_UID = bridge_suite.UID              # the rig's 12-token request: chunks 4, 4, 3
APPEND_UID = 'ra-append'
OTHER_UID = 'ra-other'


class Pause(str, enum.Enum):
    """Leaf: the pinned scheduler PauseState (an enum with PAUSED_ALL); compares equal to the rig's strings."""
    UNPAUSED = 'UNPAUSED'
    PAUSED_ALL = 'PAUSED_ALL'


class Req(adm.Request):
    """Leaf: the pinned vllm Request fields the candidate, re-entry and cohort read."""

    def __init__(self, uid, tokens, computed=0):
        super().__init__(uid=uid, tokens=tokens, computed=computed)
        self.status = NS(name='WAITING')
        self.num_output_tokens = self.num_preemptions = 0
        self.num_prompt_tokens = tokens
        self.resumable = False
        self.is_prefill_chunk = False
        self.lora_request = self.pooling_params = self.structured_output_request = self.ec_transfer_params = None
        self.sampling_params = NS(prompt_logprobs=None, logprobs=None)


from stage_progress import StagedSegmentDriver  # noqa: E402
from stage_wire import STAGE_SKIP, StageScoped  # noqa: E402
import stage_net  # noqa: E402

DRIVERS = {'native': ed.NativeSegmentDriver, 'staged': StagedSegmentDriver}


class StagedDirectTransport(bridge_suite.DirectTransport):
    """Leaf: the owned ReplyTransport plus the generated worker loop for a stage-scoped broadcast.

    bridge_suite.DirectTransport with the one rule FLASH_S5_STAGE_LOCAL adds (executor_hooks.worker_rpc ->
    stage_wire.stage_rpc, worker_handle_output): every rank executes the broadcast; a non-target rank returns
    STAGE_SKIP and writes no reply; only the scope's ranks answer and only they are expected.  An unscoped
    command keeps the base rule (every rank answers an untargeted broadcast).
    """

    def submit(self, kind, send, ranks=None, timeout_s=300.0, aggregate=None, precheck=None):
        if precheck is not None:
            precheck()
        sequence = self.next
        self.next += 1
        self._calls[sequence] = NS(ready=False)
        send(sequence)
        method, args, kwargs, output_rank = self.sender.envelopes[-1]
        message = args[0] if args else None
        everyone = tuple(sorted(self.network.workers))
        scoped = type(message) is StageScoped
        if scoped and output_rank is None:
            if tuple(ranks) != message.ranks:
                raise AssertionError('a stage-scoped broadcast must expect exactly its scope ranks')
            targets = message.ranks
        elif scoped:
            if tuple(ranks) != (output_rank,) or output_rank not in message.ranks:
                raise AssertionError('a stage-scoped targeted reply registered the wrong ranks')
            targets = (output_rank,)
        elif output_rank is None:
            if tuple(ranks) != everyone:
                raise AssertionError('untargeted broadcast must expect every rank')
            targets = everyone
        else:
            if tuple(ranks) != (output_rank,):
                raise AssertionError('targeted reply registered the wrong ranks')
            targets = (output_rank,)
        values, failure = [], None
        for rank in everyone:
            if rank in self.missing:
                continue
            try:
                value = getattr(self.network.workers[rank], method)(*args, **kwargs)
            except BaseException as exc:
                if rank not in targets:
                    self.silent_worker_errors.append((method, rank, type(exc).__name__))
                if rank in targets and failure is None:
                    failure = exc
                continue
            if scoped and rank not in message.ranks:
                if value is not STAGE_SKIP:
                    raise AssertionError('a non-target rank answered a stage-scoped command')
                continue
            if rank in targets:
                values.append(value)
        if failure is not None:
            raise failure
        self.ready[sequence] = values
        self._calls[sequence].ready = not self.hold
        return sequence


def staged_cold_stack():
    """native.cold_stack with the box's driver: FLASH_S5_STAGE_LOCAL=1 builds StagedSegmentDriver (s5_startup
    :378-381), staged progress selected.  The bounded sender carries the ring metadata the staged headroom check
    reads (stage_net.FakeRing, never written here: the reserve itself is test_e3_stage_local.RingHeadroom's)."""
    with mock.patch.object(bridge_suite, 'DirectTransport', StagedDirectTransport), \
            mock.patch.object(bridge_suite.ed, 'NativeSegmentDriver', StagedSegmentDriver):
        stack = native.cold_stack()
    stack.sender.queue = stage_net.FakeRing((0, 1, 2, 3))
    assert type(stack.driver) is StagedSegmentDriver and stack.driver.stage_selected
    return stack


STACKS = {'native': lambda: native.cold_stack(), 'staged': lambda: staged_cold_stack()}


class Rig:
    """The connected yield rig (see the module docstring); every step is the engine's own call.

    ``driver`` picks NativeSegmentDriver or StagedSegmentDriver (the box driver); the default is the class
    attribute a test class's setUp sets (StagedDriver mixin), so every rig a staged test builds is staged."""
    DRIVER = 'native'

    def __init__(self, *, max_new=8, per_bulk=4, handback_steps=8, reentry_retries=4, cached=None, enabled=True,
                 configure=True, driver=None):
        self.kind = driver or Rig.DRIVER
        self.stack = STACKS[self.kind]()
        assert type(self.stack.driver) is DRIVERS[self.kind], (self.kind, type(self.stack.driver))
        self.core, self.frontend = native.core_for(self.stack, paused=True)
        self.s, self.c, self.d = self.stack.scheduler, self.stack.controller, self.stack.driver
        self.bulk = self.s.waiting.peek_request()
        upgrade(self.bulk)
        self.s.requests = {self.bulk.request_id: self.bulk}
        self.s.waiting = prior.IterQueue([self.bulk])
        self.s.skipped_waiting = prior.IterQueue([])
        prior.fcfs(self.s)
        self.s.running = []
        self.core.batch_queue = []
        # Leaf: the pinned Scheduler/VllmConfig fields auto-arm's _select reads (FCFS, DP 1, no speculation)
        self.s.policy = NS(name='FCFS')
        self.s.vllm_config = NS(parallel_config=NS(data_parallel_size=1), speculative_config=None)
        manager = self.s.kv_cache_manager
        for worker in self.stack.workers.values():
            construction_leaf(worker.dispatcher.native)
        self.cached = dict(cached or {})
        manager.get_computed_blocks = self._computed
        manager.free = self._free
        # Leaf: the pinned coordinator.get_num_blocks_to_allocate / block_pool.get_num_free_blocks / watermark the
        # free-block check reads (auto-arm _plan, the yield candidate); the rig allocates one block per token.
        self.free_blocks = 1_000_000
        manager.coordinator = NS(get_num_blocks_to_allocate=self._blocks_needed)
        manager.watermark_blocks = 0
        manager.block_pool.get_num_free_blocks = lambda: self.free_blocks
        self.lookups = []
        if configure:
            self.c.append_yield = sh.AppendYieldConfig(max_new, per_bulk, handback_steps, reentry_retries,
                                                       enabled=enabled)
        self.control({'action': 'enter_split'})
        self.p = auto.AutoArmPolicy(self.core, self.d)
        self.d.auto_arm = self.p
        self.p.request_ref = weakref.ref(self.bulk)
        self.p.selection_seq = 9
        self.p.old_pause = Pause.UNPAUSED
        self.p.cohort = ()
        self.p.phase, self.p._internal = 'split', True
        try:
            self.d._arm_admission(auto.PHASE)
        finally:
            self.p._internal = False
        self.p.phase = 'native'
        self.core.paused = False
        self.s.pause_state = Pause.UNPAUSED

    def _computed(self, request):
        self.lookups.append(request.request_id)
        return NS(blocks=()), self.cached.get(request.request_id, 0), 0

    def _blocks_needed(self, *, request_id, num_tokens, total_computed_tokens, **_kwargs):
        return num_tokens - total_computed_tokens

    def _free(self, request):
        """Leaf: the pinned KVCacheManager.free (kv_cache_manager.py:603): the request's blocks return to the pool."""
        manager = self.s.kv_cache_manager
        manager.block_pool.freed.extend(b for b in manager.rows.pop(request.request_id, ()) if b)

    def stock_abort(self, request):
        """Leaf: the pinned Scheduler.finish_requests(ABORTED) after executor_hooks.before_abort returned: the
        request leaves its queue, ``requests`` and the worker rows; its blocks are freed."""
        for queue in (self.s.skipped_waiting, self.s.waiting):
            if request in queue.items:
                queue.items.remove(request)
        if request in self.s.running:
            self.s.running.remove(request)
        self.s.requests.pop(request.request_id, None)
        request.status = NS(name='FINISHED_ABORTED')
        self._free(request)
        for worker in self.stack.workers.values():
            worker.model_runner.req_states.req_id_to_index.pop(request.request_id, None)

    def control(self, command):
        result = self.d.submit_control(command)
        for _ in range(300):
            if not hasattr(result, 'done') or result.done():
                return result.result() if hasattr(result, 'result') else result
            self.d.control_tick()
            self.stack.complete_everything()
        raise AssertionError('control never settled')

    # -- engine steps --------------------------------------------------------
    def admit(self):
        """The engine's admission seam; returns the parked transaction (None when nothing was admitted)."""
        record = sh.flash_s5_admit(self.s)
        if record is None:
            return None
        return self.c.parked.transaction

    def finish_chunk(self):
        self.stack.drive_to_finalize()
        self.stack.complete_everything()
        self.stack.drive(lambda: self.d.tx is None)
        assert self.d.error is None

    def drive(self, until, limit=400):
        for _ in range(limit):
            if until():
                return
            self.d.advance()
            self.stack.complete_everything()
        raise AssertionError('driver did not reach the state; yield=%r' % (self.d.append_yield_phase(),))

    def tick(self, until, limit=400):
        """auto-arm's own engine tick (executor_hooks runs policy.tick each Core step) plus control settlement."""
        for _ in range(limit):
            if until():
                return
            with auto_arm_workers():
                self.p.tick()
                self.d.advance()
                self.stack.complete_everything()
        raise AssertionError('policy did not reach the state; phase=%s yield=%r'
                             % (self.p.phase, self.d.append_yield_phase()))

    def add_held(self, uid, tokens, cached=0, queue='skipped'):
        request = Req(uid, tokens)
        self.cached[uid] = cached
        self.s.requests[uid] = request
        (self.s.skipped_waiting if queue == 'skipped' else self.s.waiting).items.append(request)
        return request

    def journal(self, kind=None):
        return [e for e in self.d._journal if kind is None or e['kind'] == kind]

    def kinds(self):
        return [e['kind'] for e in self.d._journal]

    def stock_handback(self, request):
        """Leaf: the pinned stock step that schedules a request's last prompt token on FULL and samples one token.

        Scheduler side: the request leaves the waiting queue for ``running`` with one output token.  Worker side:
        stock's add_request gives it a req_states row whose prefill is complete (the census counts it as a live
        decode), on every rank.
        """
        for queue in (self.s.skipped_waiting, self.s.waiting):
            if request in queue.items:
                queue.items.remove(request)
        request.num_computed_tokens = request.num_tokens
        request.num_output_tokens = 1
        request.status = NS(name='RUNNING')
        self.s.running.append(request)
        for worker in self.stack.workers.values():
            states = worker.model_runner.req_states
            slot = len(states.prefill_len.np)
            states.req_id_to_index[request.request_id] = slot
            states.prefill_len.np.append(request.num_tokens)
            states.num_computed_prefill_tokens.append(request.num_tokens)

    def stock_finish(self, request):
        """Leaf: the pinned stock step that finishes a request (EOS/length): scheduler and worker rows leave."""
        self.s.running.remove(request)
        self.s.requests.pop(request.request_id, None)
        request.status = NS(name='FINISHED_STOPPED')
        for worker in self.stack.workers.values():
            worker.model_runner.req_states.req_id_to_index.pop(request.request_id, None)

    def rows(self, uid):
        return self.s.kv_cache_manager.rows.get(uid, ())


def construction_leaf(native_obj):
    """Leaf: native_startup.NativeStartup.inventory() and its decode owner (lane_state.DecodeStorageOwner).

    The bridge rig's worker native is a namespace over the real WorkerState, GraphRouter and controller; the
    construction inventory is the box's measured bank/communicator record.  Only the fields the identity row
    reads are given, each from the worker's own objects (route, generation, storage epoch, rank, incarnation).
    """
    router, state = native_obj.router, native_obj.state
    native_obj.inventory = lambda: {'route': native_obj.controller.route,
                                    'route_generation': native_obj.controller.route_generation}
    native_obj.decode_owner = NS(storage_epoch=router.handle.storage_epoch, incarnation=state.incarnation,
                                 worker_rank=state.worker_rank,
                                 lane_id=NS(context_id=0, storage_epoch=router.handle.storage_epoch,
                                            worker_rank=state.worker_rank))
    return native_obj


@contextmanager
def auto_arm_workers(counts=(1, 2, 3, 4)):
    """Leaf: native_auto_arm.worker_capability under FLASH_S5_AUTO_ARM=1.

    On the box it dispatches the stock cudagraph keys against the captured decode-split bank (CUDA objects this
    CPU rig does not have); here the supported cohort counts are given and every other field is the worker's own
    (router epoch, transition seq, storage epoch), so auto-arm's identity proof reads the same record shape.
    """
    def capability(native_obj):
        router = native_obj.router
        return dict(schema='s5-auto-decode-capability/1', supported_counts=list(counts),
                    engine_epoch=router.engine_epoch, route_generation=router._transition_seq,
                    storage_epoch=router.handle.storage_epoch)
    with mock.patch.dict('os.environ', {'FLASH_S5_AUTO_ARM': '1'}), \
            mock.patch.object(auto, 'worker_capability', capability):
        yield


def upgrade(request):
    """The rig's bulk becomes a full Req leaf in place (same object, same queue)."""
    fields = Req(request.request_id, request.num_tokens).__dict__
    for name, value in fields.items():
        if name not in ('all_token_ids', 'num_computed_tokens'):
            setattr(request, name, value)
    return request


def run_to_yield(rig, *, append_tokens=11, append_cached=8):
    """Bulk chunk 1 committed; an append held; the yield starts at that boundary."""
    txn = rig.admit()
    assert txn is not None and txn.request_uid == BULK_UID, txn
    append = rig.add_held(APPEND_UID, append_tokens, cached=append_cached)
    rig.finish_chunk()
    return append


def worker_rows(rig):
    """Every rank's bridge view of retained worker rows and open transactions (the barrier's own query)."""
    return {rank: nw._bridge_rows(worker.model_runner) for rank, worker in rig.stack.workers.items()}


def to_append(rig):
    rig.drive(lambda: rig.d.append_yield_phase() == 'append')


def run_append(rig):
    """The append's native chunks to its terminal commit (the yield then owes the FULL return)."""
    transactions = []
    while rig.d.append_yield_phase() == 'append':
        txn = rig.admit()
        assert txn is not None and txn.request_uid == rig.d._yield['append'].request_id, (txn, rig.c.last_refusal)
        transactions.append(txn)
        rig.finish_chunk()
    return transactions


def to_full(rig):
    rig.drive(lambda: not rig.d._split_ready and not rig.d.active)


def to_resume(rig):
    rig.tick(lambda: rig.d.append_yield_phase() == 'resume')


def finish_bulk(rig):
    """The bulk's remaining native chunks, then its FULL return (the policy's own tail)."""
    while rig.bulk.num_computed_tokens < rig.bulk.num_tokens - 1:
        txn = rig.admit()
        assert txn is not None and txn.request_uid == BULK_UID, (txn, rig.c.last_refusal,
                                                                  rig.d.append_yield_phase())
        rig.finish_chunk()
    to_full(rig)


def connected_cycle(tc, **kwargs):
    """The A1 cycle with every Codex 1-4 / A1 1-3 check; returns the rig (used by the mutants too)."""
    kwargs.setdefault('per_bulk', 1)
    rig = Rig(cached={}, **kwargs)
    d, c, s = rig.d, rig.c, rig.s
    append = run_to_yield(rig)
    bulk_row = max(c._committed, key=lambda t: t.txn_seq)
    frontier = bulk_row.chunk_end
    tc.assertEqual(d.append_yield_phase(), 'closing')
    generation = d.route_generation
    bulk_rows = rig.rows(BULK_UID)
    freed_before = list(s.kv_cache_manager.block_pool.freed)
    tc.assertTrue(all(row['retained_rows'] == 1 for row in worker_rows(rig).values()))
    # one generation: nothing is admitted while the bulk row closes; the bulk and the append stay held
    tc.assertIsNone(rig.admit())
    tc.assertTrue(sh.defer_stock_positive_frontier(s, rig.bulk))
    to_append(rig)
    # Codex 4: a drained close on every rank; Codex 1: the scheduler request and its blocks untouched
    closed = rig.journal('request_row_closed')
    tc.assertEqual(len(closed), 1)
    tc.assertEqual((closed[0]['reason'], closed[0]['dispositions']), ('yield', ['released']))
    tc.assertTrue(all(row['retained_rows'] == 0 and not row['active'] for row in worker_rows(rig).values()))
    tc.assertIsNone(d._retained)
    tc.assertIsNone(d.stop_kind)
    tc.assertTrue(d._split_ready)
    tc.assertEqual(d.route_generation, generation)
    tc.assertIs(s.requests[BULK_UID], rig.bulk)
    tc.assertEqual(rig.bulk.num_computed_tokens, frontier)
    tc.assertEqual(rig.rows(BULK_UID), bulk_rows)
    tc.assertEqual(s.kv_cache_manager.block_pool.freed, freed_before)
    tc.assertIn(rig.bulk, s.waiting.items)
    # the append runs natively from its cached start (8 of 11 tokens cached: one chunk 8..10)
    transactions = run_append(rig)
    tc.assertEqual([(t.chunk_start, t.chunk_end) for t in transactions], [(8, 10)])
    # Codex 4: fresh transaction and row identity for the append
    tc.assertNotEqual(transactions[0].txn_seq, bulk_row.txn_seq)
    tc.assertNotEqual(transactions[0].request_uid, bulk_row.request_uid)
    # Codex 2: the append never writes a block the bulk holds
    tc.assertFalse(set(rig.rows(APPEND_UID)) & set(bulk_rows))
    tc.assertEqual(rig.rows(BULK_UID), bulk_rows)
    tc.assertEqual(d.append_yield_phase(), 'handback')
    to_full(rig)
    tc.assertEqual(d.route_generation, generation + 1)
    tc.assertIsNone(d.admission_intent())
    # A1 1/2: the FULL window holds the bulk and every other stock prefill; only the append's handback runs
    other = rig.add_held(OTHER_UID, 6)
    tc.assertFalse(sh.defer_stock_positive_frontier(s, append))
    tc.assertTrue(sh.defer_stock_positive_frontier(s, rig.bulk))
    tc.assertTrue(sh.defer_stock_prefix_hit(s, NS(blocks=()), request=other))
    tc.assertEqual(c.yield_stock_deferrals_by_kind, {'fresh': 1, 'positive-frontier': 1, 'running-prefill': 0})
    with auto_arm_workers():
        rig.p.tick()
    tc.assertEqual(rig.p.phase, 'native', 'no re-entry before the append handback ran')
    rig.stock_handback(append)
    # A1 3: re-entry through auto-arm's entry sequence, bound to the bulk, plan from its frontier, no 32,768 floor
    to_resume(rig)
    tc.assertTrue(d._split_ready)
    tc.assertEqual(d.route_generation, generation + 2)
    tc.assertEqual(rig.p.plan['start'], frontier)
    tc.assertLess(rig.p.plan['end'] - rig.p.plan['start'], auto.MIN_NATIVE_ROWS)
    tc.assertEqual(rig.p.cohort, (APPEND_UID,))
    tc.assertEqual(d.admission_intent()['request_uid'], BULK_UID)
    tc.assertTrue(sh.defer_stock_prefix_hit(s, NS(blocks=()), request=other))
    lookups = list(rig.lookups)
    txn = rig.admit()
    tc.assertIsNotNone(txn)
    # Codex 3: reopen from the committed frontier, never from a fresh prefix lookup; Codex 4: fresh identity
    tc.assertEqual(rig.lookups, lookups)
    tc.assertEqual(txn.request_uid, BULK_UID)
    tc.assertEqual(txn.chunk_start, frontier)
    tc.assertNotEqual(txn.txn_seq, bulk_row.txn_seq)
    # the request generation names the scheduler request object, which is kept (never re-created); the row's
    # identity is the new transaction and its state epoch, so a late reply for the closed row cannot match it
    tc.assertEqual(txn.request_generation, bulk_row.request_generation)
    tc.assertEqual(rig.rows(BULK_UID)[:len(bulk_rows)], bulk_rows)
    reopened = txn
    rig.drive(lambda: d.tx is not None)
    tc.assertIsNone(d.append_yield_phase())
    tc.assertEqual(d.append_yield_counters, dict(selected=1, rows_closed=1, admitted=1, handbacks=1,
                                                 reentries=1, reopened=1, abandoned=0,
                                                 handback_expired=0, reentry_refused=0,
                                                 append_refused=0, handbacks_served=1,
                                                 handback_skipped=0))
    rig.finish_chunk()
    tc.assertGreater(c._committed[reopened]['state_epoch'], c._committed[bulk_row]['state_epoch'])
    tc.assertGreater(c._committed[reopened]['state_epoch'], c._committed[transactions[0]]['state_epoch'])
    txn = rig.admit()
    tc.assertIsNotNone(txn, (c.last_refusal, d.append_yield_phase(), rig.kinds()[-6:]))
    tc.assertEqual((txn.chunk_start, txn.chunk_end), (8, 11))
    rig.finish_chunk()
    to_full(rig)
    tc.assertEqual(rig.bulk.num_computed_tokens, rig.bulk.num_tokens - 1)
    tc.assertEqual([e['reason'] for e in rig.journal('append_yield_end')], ['reopened'])
    tc.assertEqual(len(rig.journal('append_yield_selected')), 1)
    tc.assertIn(other, s.skipped_waiting.items)
    return rig


class YieldAtBoundary(unittest.TestCase):
    def test_the_connected_cycle(self):
        connected_cycle(self)

    def test_a_second_held_request_gets_its_own_yield_up_to_per_bulk(self):
        rig = Rig(cached={}, per_bulk=2)
        append = run_to_yield(rig)
        to_append(rig)
        run_append(rig)
        to_full(rig)
        other = rig.add_held(OTHER_UID, 6)
        rig.stock_handback(append)
        to_resume(rig)
        self.assertIsNotNone(rig.admit())
        rig.finish_chunk()
        # the reopened bulk's chunk ends at 8: the second yield selects the other held request
        self.assertEqual(rig.d.append_yield_phase(), 'closing')
        to_append(rig)
        transactions = run_append(rig)
        self.assertEqual([(t.chunk_start, t.chunk_end) for t in transactions], [(0, 4), (4, 5)])
        self.assertFalse(set(rig.rows(OTHER_UID)) & set(rig.rows(BULK_UID)))
        to_full(rig)
        rig.stock_handback(other)
        to_resume(rig)
        self.assertEqual(rig.p.cohort, tuple(sorted((APPEND_UID, OTHER_UID))))
        finish_bulk(rig)
        self.assertEqual(rig.d.append_yield_counters['reopened'], 2)
        self.assertEqual(len(rig.journal('append_yield_selected')), 2)
        self.assertEqual(len(rig.journal('request_row_closed')), 2)

    def test_a_multi_chunk_append_from_a_short_cache_hit(self):
        rig = Rig(cached={}, max_new=10)
        run_to_yield(rig, append_tokens=11, append_cached=0)
        to_append(rig)
        transactions = run_append(rig)
        self.assertEqual([(t.chunk_start, t.chunk_end) for t in transactions], [(0, 4), (4, 8), (8, 10)])
        self.assertEqual(rig.d.append_yield_counters['admitted'], 1)
        self.assertEqual(len({t.txn_seq for t in transactions}), 3)
        self.assertEqual(rig.d.append_yield_phase(), 'handback')

    def test_the_append_open_waits_for_the_closed_row(self):
        """One generation: while the bulk row is retained (closing) no other row may open on any rank."""
        rig = Rig(cached={})
        run_to_yield(rig)
        for _ in range(3):
            self.assertIsNone(rig.admit())
            self.assertIsNotNone(rig.d._retained)
        self.assertEqual(rig.d.append_yield_counters['admitted'], 0)

    def test_journal_attribution(self):
        """Observer accounting at the engine journal: every yield record is attributed (seq, route generation)
        and carries no request UID; the row close and the append's chunks name their transactions."""
        rig = connected_cycle(self)
        records = [e for e in rig.d._journal if e['kind'].startswith('append_yield_')]
        self.assertEqual([e['kind'] for e in records],
                         ['append_yield_selected', 'append_yield_drained', 'append_yield_row_closed', 'append_yield_admitted',
                          'append_yield_handback', 'append_yield_full', 'append_yield_reenter',
                          'append_yield_resume', 'append_yield_bulk_reopened', 'append_yield_end'])
        self.assertTrue(all(e['seq'] == 1 and type(e['route_generation']) is int for e in records))
        uids = {BULK_UID, APPEND_UID, OTHER_UID}
        for record in records:
            self.assertFalse(uids & {v for v in record.values() if type(v) is str}, record)
        admitted = rig.journal('append_yield_admitted')[0]['transaction']
        reopened = rig.journal('append_yield_bulk_reopened')[0]['transaction']
        commits = [e.get('transaction') for e in rig.journal('chunk_commit')]
        self.assertIn(admitted, commits)
        self.assertIn(reopened, commits)
        self.assertEqual(rig.journal('request_row_closed')[0]['reason'], 'yield')

    def test_status_history_splits_each_yield_window(self):
        """A1 grade input: per yield, the engine times of every phase (FULL window = handback..full..reenter,
        re-entry = reenter..resume), in journal order, bounded and without request UIDs."""
        rig = connected_cycle(self)
        record = rig.d.submit_control({'action': 'status'})['append_yield']
        self.assertIsNone(record['current'])
        self.assertEqual(len(record['history']), 1)
        entry = record['history'][0]
        self.assertEqual((entry['seq'], entry['reason']), (1, 'reopened'))
        order = ['selected', 'drained', 'row_closed', 'admitted', 'handback', 'full', 'reenter', 'resume', 'bulk_reopened', 'end']
        self.assertEqual(list(entry['times']), order)
        times = [entry['times'][name] for name in order]
        self.assertEqual(times, sorted(times))
        journal = {e['kind'][len('append_yield_'):]: e['observed_ns'] for e in rig.d._journal
                   if e['kind'].startswith('append_yield_')}
        self.assertEqual(entry['times'], journal)
        self.assertEqual(entry['route_generation_end'] - entry['route_generation_selected'], 2)
        self.assertNotIn(BULK_UID, repr(record))
        self.assertNotIn(APPEND_UID, repr(record))
        # the hold counters ride in the same status row (lead: the status row, one engine snapshot)
        # the FULL window's holds (the other request, the bulk); once split again the split hold holds them
        self.assertEqual(record['holds'], {'fresh': 1, 'positive-frontier': 1, 'running-prefill': 0})
        self.assertEqual(record['holds'], rig.c.yield_stock_deferrals_by_kind)
        self.assertEqual(record['end_reasons'], {'reopened': 1})


class StagedDriver:
    """Mixin (lead B1): every rig the test class builds runs StagedSegmentDriver, the box driver
    (FLASH_S5_STAGE_LOCAL=1 in the alpha-next settings; s5_startup.py:378-381)."""

    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(Rig, 'DRIVER', 'staged')
        patcher.start()
        self.addCleanup(patcher.stop)


class YieldAtBoundaryStaged(StagedDriver, YieldAtBoundary):
    """YieldAtBoundary, every case on StagedSegmentDriver."""


def yield_records(rig):
    return [(e['kind'], e.get('reason'), e.get('phase')) for e in rig.d._journal
            if e['kind'].startswith('append_yield_') or e['kind'] in ('request_row_closed', 'chunk_commit')]


class StagedIdentity(unittest.TestCase):
    """B1: the staged yield is the native yield.  The connected cycle on both drivers leaves the same persistent
    state (scheduler rows, allocations, cache calls, commits, frontier rows, leases, transfer sequence, and per
    rank: factory registrations/updates/admissions/releases, PP channel sends and tickets, bridge, dispatcher and
    runtime releases; stage_net.RingStack.persistent_state, test_e3_stage_local's identity record) and the same
    yield journal.  The worker-side builder state is a function of those per-rank admissions and registrations,
    so RealBuilderState's proof (driver-free, the real LaneChunkFactory) holds for the staged driver too."""

    def run_both(self, scenario):
        out = {}
        for kind in DRIVERS:
            with mock.patch.object(Rig, 'DRIVER', kind):
                rig = scenario()
            self.assertIs(type(rig.d), DRIVERS[kind])
            out[kind] = rig
        return out

    def test_connected_cycle_same_persistent_state_and_journal(self):
        rigs = self.run_both(lambda: connected_cycle(self))
        native_rig, staged_rig = rigs['native'], rigs['staged']
        self.assertEqual(stage_net.RingStack.persistent_state(staged_rig.stack),
                         stage_net.RingStack.persistent_state(native_rig.stack))
        self.assertEqual(yield_records(staged_rig), yield_records(native_rig))
        counters = staged_rig.d.stage_counters
        # bulk chunk 1, the append's chunk, the reopened bulk chunk, the bulk's last chunk: each a staged frame
        self.assertEqual((counters['chunks'], counters['local_releases'], counters['headroom_refused']), (4, 8, 0))
        self.assertEqual(staged_rig.d._frames, [])
        self.assertEqual(staged_rig.d._seq_route, {})

    def test_two_yields_same_persistent_state(self):
        def two():
            rig = Rig(cached={}, per_bulk=2)
            append = run_to_yield(rig)
            to_append(rig)
            run_append(rig)
            to_full(rig)
            other = rig.add_held(OTHER_UID, 6)
            rig.stock_handback(append)
            to_resume(rig)
            rig.admit()
            rig.finish_chunk()
            to_append(rig)
            run_append(rig)
            to_full(rig)
            rig.stock_handback(other)
            to_resume(rig)
            finish_bulk(rig)
            return rig
        rigs = self.run_both(two)
        self.assertEqual(stage_net.RingStack.persistent_state(rigs['staged'].stack),
                         stage_net.RingStack.persistent_state(rigs['native'].stack))
        self.assertEqual(yield_records(rigs['staged']), yield_records(rigs['native']))
        self.assertEqual(rigs['staged'].d.append_yield_counters['reopened'], 2)

    def test_the_yield_close_is_an_unscoped_all_rank_command_outside_any_frame(self):
        """The close crosses no stage scope: no frame is live at the boundary, the CancelRequest goes to all four
        ranks on the ordinary bounded sender (its own writable-slot check), and its reply is the frozen apply."""
        rig = Rig(cached={}, driver='staged')
        run_to_yield(rig)
        self.assertEqual(rig.d._frames, [])
        self.assertIsNone(rig.d.tx)
        before = len(rig.stack.sender.envelopes)
        rig.d.advance()
        sent = rig.stack.sender.envelopes[before:]
        cancels = [e for e in sent if e[0] == 'flash_s5_cancel']
        self.assertEqual(len(cancels), 1)
        method, args, _kwargs, output_rank = cancels[0]
        self.assertIsNone(output_rank)
        self.assertNotIsInstance(args[0], StageScoped)
        self.assertEqual(args[0].reason, ed.YIELD_CANCEL_REASON)
        self.assertEqual(rig.d._seq_route, {})
        to_append(rig)
        self.assertEqual(rig.journal('request_row_closed')[-1]['ranks'], [0, 1, 2, 3])


# --- Config, control, candidate ----------------------------------------------------------------------------------

import s5_startup  # noqa: E402
import native_overlap_control as nc  # noqa: E402
import executor_hooks as eh  # noqa: E402


class Config(unittest.TestCase):
    BOTH = {'FLASH_S5_APPEND_YIELD_MAX_NEW': '2048', 'FLASH_S5_APPEND_YIELD_PER_BULK': '4',
            'FLASH_S5_APPEND_YIELD_HANDBACK_STEPS': '16', 'FLASH_S5_APPEND_YIELD_REENTRY_RETRIES': '4'}
    RECORD = {'max_new': 2048, 'per_bulk': 4, 'handback_steps': 16, 'reentry_retries': 4, 'enabled': True}

    def test_all_four_or_none(self):
        self.assertIsNone(sh.AppendYieldConfig.from_env({}))
        config = sh.AppendYieldConfig.from_env(self.BOTH)
        self.assertEqual((config.max_new, config.per_bulk, config.handback_steps, config.reentry_retries,
                          config.enabled), (2048, 4, 16, 4, True))
        self.assertEqual(config.record(), self.RECORD)
        self.assertEqual(sorted(self.BOTH), sorted(sh.APPEND_YIELD_ENV))
        self.assertEqual(sorted(self.BOTH), sorted(nc.APPEND_YIELD_ENV))
        for name in self.BOTH:
            env = dict(self.BOTH)
            del env[name]
            with self.subTest(missing=name), self.assertRaisesRegex(E3ContractError, 'all four launch bounds'):
                sh.AppendYieldConfig.from_env(env)

    def test_bounds_are_positive_decimal_ints(self):
        for value in ('0', '-1', '01', '+4', ' 4', '4.0', 'x', '', '1e3'):
            for name in self.BOTH:
                with self.subTest(value=value, name=name), self.assertRaises(E3ContractError):
                    sh.AppendYieldConfig.from_env(dict(self.BOTH, **{name: value}))
        for args in ((0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0), (True, 1, 1, 1), (1, 1, 1.0, 1),
                     (1, 1, 1, '4')):
            with self.subTest(args=args), self.assertRaises(E3ContractError):
                sh.AppendYieldConfig(*args)
        with self.assertRaises(TypeError):
            sh.AppendYieldConfig(1, 1)                   # the two step bounds have no default
        with self.assertRaises(E3ContractError):
            sh.AppendYieldConfig(1, 1, 1, 1, enabled=1)

    def refused(self, env, text):
        # Leaf (as test_e3_native_layer_groups.Startup): _enabled reads the overlap composition's settings.
        with mock.patch.object(s5_startup, '_enabled', lambda env: True), \
                self.assertRaises(s5_startup.StartupError) as refused:
            s5_startup.install_engine(types.SimpleNamespace(), env=env)
        self.assertIn(text, str(refused.exception))

    def test_startup_refuses_half_or_without_auto_arm(self):
        self.refused(dict(self.BOTH), 'needs FLASH_S5_AUTO_ARM=1')
        self.refused(dict(self.BOTH, FLASH_S5_AUTO_ARM='0'), 'needs FLASH_S5_AUTO_ARM=1')
        self.refused({'FLASH_S5_AUTO_ARM': '1', 'FLASH_S5_APPEND_YIELD_MAX_NEW': '2048'}, 'all four launch bounds')
        self.refused(dict({k: v for k, v in self.BOTH.items() if 'HANDBACK' not in k}, FLASH_S5_AUTO_ARM='1'),
                     'all four launch bounds')
        self.refused(dict(self.BOTH, FLASH_S5_AUTO_ARM='1', FLASH_S5_APPEND_YIELD_MAX_NEW='0'),
                     'positive')

    def test_startup_installs_the_config_after_auto_arm(self):
        """Source order in install_engine: parse before any construction, attach after auto-arm's install."""
        tree = ast.parse(Path(s5_startup.__file__).read_text())
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'install_engine')
        lines = {}
        for node in ast.walk(function):
            if isinstance(node, ast.Call) and getattr(node.func, 'attr', None) == 'from_env':
                lines['parse'] = node.lineno
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'install'
                    and len(node.args) == 3):
                lines['auto_arm'] = node.lineno
            if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute)
                    and node.targets[0].attr == 'append_yield'):
                lines['attach'] = node.lineno
        self.assertEqual(sorted(lines), ['attach', 'auto_arm', 'parse'])
        self.assertLess(lines['parse'], lines['auto_arm'])
        self.assertLess(lines['auto_arm'], lines['attach'])

    def test_client_and_session_expose_the_control_only_when_installed(self):
        with mock.patch.dict('os.environ', {'FLASH_S5_AUTO_ARM': '1'}, clear=False):
            for name in self.BOTH:
                os_env = dict(self.BOTH)
                del os_env[name]
                with mock.patch.dict('os.environ', os_env), self.subTest(missing=name):
                    import os
                    os.environ.pop(name, None)
                    self.assertFalse(nc._append_yield())
            with mock.patch.dict('os.environ', self.BOTH):
                self.assertTrue(nc._append_yield())
        with mock.patch.dict('os.environ', dict(self.BOTH, FLASH_S5_AUTO_ARM='0')):
            self.assertFalse(nc._append_yield())


class Control(unittest.TestCase):
    def rig(self, configure=True):
        """Paused idle FULL with the auto-arm policy installed and idle (the arm-boundary state)."""
        stack = native.cold_stack()
        core, _ = native.core_for(stack, paused=True)
        driver = stack.driver
        driver.auto_arm = auto.AutoArmPolicy(core, driver)
        if configure:
            stack.controller.append_yield = sh.AppendYieldConfig(2048, 4, 16, 4)
        native.Pump  # the reviewed core; nothing runs
        driver.advance()
        stack.complete_everything()
        return stack, core, driver

    def test_on_off_on_paused_idle_full(self):
        stack, core, driver = self.rig()
        result = driver.submit_control({'action': 'native_append_yield', 'enabled': False,
                                        'engine_epoch': driver.engine_epoch})
        self.assertEqual((result['enabled'], result['previous'], result['max_new'], result['per_bulk'],
                          result['handback_steps'], result['reentry_retries']), (False, True, 2048, 4, 16, 4))
        self.assertFalse(stack.controller.append_yield.enabled)
        self.assertTrue(driver.submit_control({'action': 'native_append_yield', 'enabled': True})['enabled'])
        status = driver.submit_control({'action': 'status'})
        self.assertEqual(status['append_yield']['config'], Config.RECORD)

    def test_refusals(self):
        stack, core, driver = self.rig()
        for command, text in (({'enabled': 1}, 'invalid'), ({'enabled': True, 'x': 1}, 'fields-differ'),
                              ({'enabled': True, 'engine_epoch': driver.engine_epoch + 1}, 'epoch-differs')):
            with self.subTest(command=command), self.assertRaisesRegex(ed.DriverError, text):
                driver.submit_control(dict(command, action='native_append_yield'))
        core.paused = False
        with self.assertRaisesRegex(ed.DriverError, 'paused-idle-full'):
            driver.submit_control({'action': 'native_append_yield', 'enabled': False})
        self.assertTrue(stack.controller.append_yield.enabled)

    def test_not_a_control_and_no_status_row_when_not_installed(self):
        stack, core, driver = self.rig(configure=False)
        with self.assertRaisesRegex(ed.DriverError, 'unknown-action'):
            driver.submit_control({'action': 'native_append_yield', 'enabled': False})
        self.assertNotIn('append_yield', driver.submit_control({'action': 'status'}))


class Candidate(unittest.TestCase):
    """The yield candidate rule over the real controller (auto-arm _plan's refusals minus the bulk floor)."""

    def setUp(self):
        self.rig = Rig(cached={})
        self.c, self.s = self.rig.c, self.rig.s

    def pick(self, decode_counts=None):
        return self.c.append_yield_candidate(self.s, self.rig.bulk, decode_counts)

    def test_fcfs_skipped_first_and_never_the_bulk(self):
        first = self.rig.add_held('ra-1', 6)
        self.rig.add_held('ra-2', 6)
        late = self.rig.add_held('ra-3', 6, queue='waiting')
        self.assertIs(self.pick(), first)
        self.s.skipped_waiting.items.clear()
        self.assertIs(self.pick(), late)
        self.s.waiting.items[:] = [self.rig.bulk]
        self.assertIsNone(self.pick())

    def test_bounds_and_request_refusals(self):
        cases = {
            'over max_new': dict(tokens=10),                 # 9 new > 8
            'computed': dict(computed=2),
            'output': dict(num_output_tokens=1),
            'preempted': dict(num_preemptions=1),
            'resumable': dict(resumable=True),
            'lora': dict(lora_request=object()),
            'pooling': dict(pooling_params=object()),
            'structured': dict(structured_output_request=object()),
            'logprobs': dict(sampling_params=NS(prompt_logprobs=None, logprobs=2)),
            'prompt logprobs': dict(sampling_params=NS(prompt_logprobs=1, logprobs=None)),
            'no sampling': dict(sampling_params=None),
            'not waiting': dict(status=NS(name='RUNNING')),
            'one token': dict(tokens=1),
        }
        for name, fields in cases.items():
            with self.subTest(case=name):
                self.s.skipped_waiting.items.clear()
                tokens = fields.pop('tokens', 6)
                request = self.rig.add_held('ra-x', tokens)
                for key, value in fields.items():
                    if key == 'computed':
                        request.num_computed_tokens = value
                    else:
                        setattr(request, key, value)
                self.assertIsNone(self.pick())
        self.s.skipped_waiting.items.clear()
        request = self.rig.add_held('ra-ok', 9)                # exactly 8 new
        self.assertIs(self.pick(), request)
        self.s.requests.pop('ra-ok')
        self.assertIsNone(self.pick(), 'not the scheduler\'s registered object')

    def test_cached_start_counts_and_shared_prefix_refuses(self):
        request = self.rig.add_held('ra-long', 40, cached=32)  # 7 new beyond the cached 32
        self.assertIs(self.pick(), request)
        manager = self.s.kv_cache_manager
        manager.get_computed_blocks = lambda r: (NS(blocks=()), 32, 1)
        self.assertIsNone(self.pick())

    def test_running_slots_and_proven_decode_counts(self):
        """Review MAJOR-3: the append's slot plus the bulk's kept slot (re-entry cohort <= max - 1), and every
        cohort size 1..running+1 proven (the identity proof's contiguous range)."""
        request = self.rig.add_held('ra-ok', 6)
        self.s.running = [NS(request_id='d%d' % i) for i in range(3)]
        self.assertIs(self.pick(decode_counts=frozenset({1, 2, 3, 4})), request)
        self.assertIsNone(self.pick(decode_counts=frozenset({1, 2, 3})), 'cohort + 1 is not a proven count')
        self.assertIsNone(self.pick(decode_counts=frozenset({1, 2, 4})), 'the proven counts are not contiguous')
        self.s.max_num_running_reqs = 5
        self.assertIs(self.pick(), request, 'running 3 + append + the bulk slot = 5')
        self.s.max_num_running_reqs = 4
        self.assertIsNone(self.pick(), 'the append would take the last running slot')

    def test_free_blocks(self):
        """Review MAJOR-2: auto-arm _plan's free-block check (the whole request must fit now)."""
        request = self.rig.add_held('ra-ok', 6, cached=2)           # 4 blocks needed
        self.rig.free_blocks = 4
        self.assertIs(self.pick(), request)
        self.rig.free_blocks = 3
        self.assertIsNone(self.pick())
        self.rig.free_blocks = 4
        self.s.kv_cache_manager.watermark_blocks = 1
        self.assertIsNone(self.pick())

    def test_disabled_or_absent(self):
        self.rig.add_held('ra-ok', 6)
        self.c.append_yield.enabled = False
        self.assertIsNone(self.pick())
        self.c.append_yield = None
        self.assertIsNone(self.pick())

    def test_no_staged_plan_refuses(self):
        """The plan band: no complete staged plan (here the aligned split leaves nothing) refuses the candidate."""
        self.rig.add_held('ra-ok', 6)
        self.s._mamba_block_aligned_split = lambda request, n, num_new_local_computed_tokens=0: 0
        self.assertIsNone(self.pick())


# --- OffIsFrozen --------------------------------------------------------------------------------------------------

BASE_COMMIT = 'f3951ee02c095386d0d274d735a9c4b57f688c3a'     # the pre-build tree (FIX-DESIGN, no engine change)
ENGINE_FILES = ('engine_driver.py', 'native_auto_arm.py', 'scheduler_hooks.py', 's5_startup.py',
                'native_overlap_control.py', 'native_overlap_session.py')


def frozen_scenario():
    """A bulk with a held append, run to the bulk's FULL return with no yield configured.

    Uses only objects both trees have; the output strips clocks and incarnations."""
    rig = Rig(cached={}, configure=False)
    s, c, d = rig.s, rig.c, rig.d
    append = run_to_yield(rig)
    holds = [sh.defer_stock_prefix_hit(s, NS(blocks=()), request=append)]
    phases = []
    while rig.bulk.num_computed_tokens < rig.bulk.num_tokens - 1:
        txn = rig.admit()
        phases.append(None if txn is None else (txn.chunk_start, txn.chunk_end, txn.txn_seq))
        if txn is None:
            break
        rig.finish_chunk()
        holds.append(sh.defer_stock_prefix_hit(s, NS(blocks=()), request=append))
    to_full(rig)
    holds.append(sh.defer_stock_prefix_hit(s, NS(blocks=()), request=append))
    noisy = {'observed_ns', 'clock_basis', 'incarnations', 'armed_ns', 'clock', 'observed_clock_ns',
             'last_ns', 'since_ns'}

    def scrub(value):                    # wall-clock readings and per-process incarnations only
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items() if k not in noisy}
        if isinstance(value, (list, tuple)):
            return [scrub(v) for v in value]
        return value
    status = d.submit_control({'action': 'status'})
    return {'journal': repr(scrub(list(d._journal))), 'phases': phases, 'holds': holds,
            'record': repr(scrub(c.native_admission_record())),
            'status_keys': sorted(status), 'queues': [r.request_id for r in s.skipped_waiting.items],
            'computed': [rig.bulk.num_computed_tokens, append.num_computed_tokens],
            'route': [d._split_ready, d.route_generation]}


def run_scenario(frozen_dir=None):
    import json
    import subprocess
    code = ('import json, sys\n'
            'frozen, patches, harness = sys.argv[1:4]\n'
            'sys.path[0:0] = ([] if frozen == "-" else [frozen]) + [patches, harness]\n'
            'import %s\n'
            'import test_append_yield as t\n'
            'print(json.dumps(t.frozen_scenario()))\n') % ', '.join(f[:-3] for f in ENGINE_FILES)
    result = subprocess.run([sys.executable, '-c', code, frozen_dir or '-',
                             str(ROOT / 'patches/s5_engine'), str(HERE)],
                            capture_output=True, text=True, timeout=240, cwd=str(HERE))
    if result.returncode != 0:
        raise AssertionError(result.stderr[-3000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


class OffIsFrozen(unittest.TestCase):
    def test_settings_absent_matches_the_pre_build_engine(self):
        """Process-level: the same scenario over this tree and over the six engine files of BASE_COMMIT."""
        import shutil
        import subprocess
        import tempfile
        if shutil.which('git') is None or subprocess.run(
                ['git', '-C', str(ROOT), 'cat-file', '-e', BASE_COMMIT], capture_output=True).returncode:
            self.skipTest('base commit %s not readable here' % BASE_COMMIT[:12])
        with tempfile.TemporaryDirectory(prefix='append-yield-frozen-') as frozen:
            for name in ENGINE_FILES:
                blob = subprocess.run(['git', '-C', str(ROOT), 'show', '%s:patches/s5_engine/%s' % (BASE_COMMIT, name)],
                                      capture_output=True, check=True).stdout
                Path(frozen, name).write_bytes(blob)
                self.assertNotEqual(blob, (ROOT / 'patches/s5_engine' / name).read_bytes(), name)
            before = run_scenario(frozen)
        after = run_scenario()
        self.assertEqual(after, before)
        self.assertEqual(before['holds'], [True, True, True, False])
        self.assertNotIn('append_yield', before['journal'])

    def test_settings_absent_no_yield_in_process(self):
        rig = Rig(cached={}, configure=False)
        run_to_yield(rig)
        self.assertIsNone(rig.d.append_yield_phase())
        self.assertIsNotNone(rig.d._retained)
        self.assertIsNone(rig.c.yield_stock_deferrals_by_kind)
        self.assertNotIn('append_yield', rig.d.submit_control({'action': 'status'}))
        self.assertEqual(rig.admit().request_uid, BULK_UID)

    def test_control_off_equals_settings_absent(self):
        def run(**kwargs):
            rig = Rig(cached={}, **kwargs)
            append = run_to_yield(rig)
            while rig.bulk.num_computed_tokens < rig.bulk.num_tokens - 1:
                self.assertIsNotNone(rig.admit())
                rig.finish_chunk()
            to_full(rig)
            held = sh.defer_stock_prefix_hit(rig.s, NS(blocks=()), request=append)
            return ([e['kind'] for e in rig.d._journal], rig.c.native_admission_record()['split_stock_deferrals_by_kind'],
                    held, rig.c.yield_stock_deferrals_by_kind)
        absent = run(configure=False)
        self.assertEqual(run(enabled=False), absent)
        on = Rig(cached={})
        run_to_yield(on)
        self.assertEqual(on.d.append_yield_phase(), 'closing', 'the same scenario yields when on')


# --- Stops (the yield's cells of the stop matrix) ------------------------------------------------------------------

def abort(rig, request):
    """The engine's abort seam, then stock's own abort of the request (the order EngineCore runs them)."""
    result = eh.before_abort(rig.core, [request.request_id])
    rig.d.pump_stop() if rig.d.stop_kind is not None else None
    return result


def settle_stop(rig):
    rig.drive(lambda: not rig.d.active and rig.d.stop_kind is None and not rig.d._split_ready
              and rig.d._return_pending is None)


def policy_idle(rig):
    rig.tick(lambda: rig.p.phase == 'idle')


class Stops(unittest.TestCase):
    """abort / reset / pause / shutdown at every yield state; client exits are the abort rows.

    Expected order everywhere: no FULL while a row is retained, at most one close per row, no stock prefill of the
    bulk while the yield owns it, and the route FULL once nothing native is owned."""

    def test_bulk_abort_while_its_row_closes(self):
        rig = Rig(cached={})
        run_to_yield(rig)
        generation = rig.d.route_generation
        self.assertIsNotNone(abort(rig, rig.bulk))
        rig.stock_abort(rig.bulk)
        self.assertIsNone(rig.d.append_yield_phase())
        self.assertEqual(rig.journal('append_yield_end')[-1]['reason'], 'stop-abort')
        settle_stop(rig)
        self.assertEqual(len(rig.journal('request_row_closed')), 1, 'the stop took over the same close')
        self.assertEqual(rig.d.route_generation, generation + 1)
        self.assertEqual(rig.d.append_yield_counters['admitted'], 0)

    def test_bulk_abort_during_the_append(self):
        """Client exit: the append completes natively; the yield ends without re-entry; the route stays FULL."""
        rig = Rig(cached={})
        append = run_to_yield(rig)
        to_append(rig)
        self.assertIsNone(abort(rig, rig.bulk), 'the bulk owns nothing native while yielded')
        rig.stock_abort(rig.bulk)
        run_append(rig)
        to_full(rig)
        generation = rig.d.route_generation
        rig.stock_handback(append)
        policy_idle(rig)
        self.assertEqual(rig.journal('append_yield_end')[-1]['reason'], 'bulk-left')
        self.assertEqual(rig.journal('auto_closed')[-1]['reason'], 'yielded-bulk-left')
        self.assertFalse(rig.d._split_ready)
        self.assertEqual(rig.d.route_generation, generation)
        self.assertEqual(rig.d.append_yield_counters['reentries'], 0)
        self.assertEqual(append.num_output_tokens, 1)

    def test_append_abort_mid_chunk(self):
        """Client exit: normal native cancel, FULL, then the bulk re-enters and reopens from its frontier."""
        rig = Rig(cached={}, max_new=10)
        append = run_to_yield(rig, append_tokens=11, append_cached=0)
        frontier = rig.bulk.num_computed_tokens
        to_append(rig)
        self.assertIsNotNone(rig.admit())
        rig.drive(lambda: rig.d.tx is not None)
        self.assertIsNotNone(abort(rig, append))
        rig.stock_abort(append)
        self.assertEqual(rig.d.append_yield_phase(), 'handback')
        settle_stop(rig)
        to_resume(rig)
        txn = rig.admit()
        self.assertEqual((txn.request_uid, txn.chunk_start), (BULK_UID, frontier))
        self.assertEqual(rig.journal('append_yield_append_aborted')[-1]['seq'], 1)
        rig.finish_chunk()
        finish_bulk(rig)
        self.assertEqual(rig.journal('append_yield_end')[-1]['reason'], 'reopened')

    def test_append_abort_after_its_native_commit(self):
        rig = Rig(cached={})
        append = run_to_yield(rig)
        to_append(rig)
        run_append(rig)
        to_full(rig)
        self.assertIsNone(abort(rig, append))
        rig.stock_abort(append)
        to_resume(rig)
        self.assertEqual(rig.p.cohort, ())
        txn = rig.admit()
        self.assertEqual(txn.request_uid, BULK_UID)
        rig.finish_chunk()
        finish_bulk(rig)

    def test_append_abort_while_held(self):
        """The append leaves before its admission: the route is still split and armed, the bulk reopens at once."""
        rig = Rig(cached={})
        append = run_to_yield(rig)
        to_append(rig)
        self.assertIsNone(abort(rig, append))
        rig.stock_abort(append)
        generation = rig.d.route_generation
        txn = rig.admit()
        if txn is None:                  # the lost target is noted first; the bulk is admitted on the next pass
            txn = rig.admit()
        self.assertIn('append_yield_append_left', rig.kinds())
        self.assertEqual(txn.request_uid, BULK_UID)
        rig.drive(lambda: rig.d.tx is not None)
        self.assertEqual(rig.journal('append_yield_end')[-1]['reason'], 'reopened')
        self.assertEqual(rig.d.route_generation, generation)

    def test_bulk_abort_in_the_full_window(self):
        rig = Rig(cached={})
        append = run_to_yield(rig)
        to_append(rig)
        run_append(rig)
        to_full(rig)
        self.assertIsNone(abort(rig, rig.bulk))
        rig.stock_abort(rig.bulk)
        generation = rig.d.route_generation
        rig.stock_handback(append)
        policy_idle(rig)
        self.assertFalse(rig.d._split_ready)
        self.assertEqual(rig.d.route_generation, generation)
        self.assertEqual(rig.journal('append_yield_end')[-1]['reason'], 'bulk-left')

    def test_bulk_abort_during_reentry(self):
        rig = Rig(cached={})
        append = run_to_yield(rig)
        to_append(rig)
        run_append(rig)
        to_full(rig)
        rig.stock_handback(append)
        rig.tick(lambda: rig.p.phase == 'identity')
        self.assertIsNone(abort(rig, rig.bulk))
        rig.stock_abort(rig.bulk)
        policy_idle(rig)
        self.assertFalse(rig.d._split_ready)
        self.assertIsNone(rig.d.append_yield_phase())
        self.assertEqual(rig.d.append_yield_counters['reopened'], 0)

    def test_bulk_abort_after_rearm_before_reopen(self):
        rig = Rig(cached={})
        append = run_to_yield(rig)
        to_append(rig)
        run_append(rig)
        to_full(rig)
        rig.stock_handback(append)
        to_resume(rig)
        self.assertIsNone(abort(rig, rig.bulk))
        rig.stock_abort(rig.bulk)
        self.assertIsNone(rig.admit())
        self.assertIsNone(rig.admit(), 'the armed intent stays bound to the yielded bulk; nothing else is admitted')
        self.assertEqual(rig.d.append_yield_phase(), 'resume')
        policy_idle(rig)
        self.assertEqual(rig.journal('append_yield_end')[-1]['reason'], 'bulk-left')
        self.assertFalse(rig.d._split_ready)

    def test_reset_in_the_full_window_and_during_entry_follows_the_reset_rule(self):
        """Lead resume 5 (review MINOR-2): a reset in any yield state follows the existing reset rule: the yield ends
        ('stop-reset', counted), the route returns or stays FULL, nothing stays held and the bulk continues on stock.
        Covered here: the append's FULL-return settle, the FULL window, the re-entry barrier."""
        for stage in ('settle', 'handback', 'barrier'):
            with self.subTest(stage=stage):
                rig = Rig(cached={})
                append = run_to_yield(rig)
                to_append(rig)
                run_append(rig)
                if stage == 'settle':
                    self.assertTrue(rig.d._split_ready)            # the terminal commit's FULL return is pending
                else:
                    to_full(rig)
                if stage == 'barrier':
                    rig.stock_handback(append)
                    rig.tick(lambda: rig.p.phase == 'barrier')
                try:
                    eh.before_reset_prefix(rig.core)
                except ed.E3Pending:
                    pass
                self.assertIsNone(rig.d.append_yield_phase())
                self.assertEqual(rig.journal('append_yield_end')[-1]['reason'], 'stop-reset')
                self.assertEqual(rig.d.append_yield_record()['end_reasons'], {'stop-reset': 1})
                rig.tick(lambda: rig.p.phase in ('tail', 'idle') and not rig.d._split_ready and not rig.d.active
                         and rig.d._return_pending is None)
                self.assertIsNone(rig.d.admission_intent())
                self.assertIsNone(rig.d.error)
                self.assertFalse(sh.defer_stock_positive_frontier(rig.s, rig.bulk), 'the bulk continues on stock')
                self.assertEqual(rig.d.append_yield_counters['reopened'], 0)

    def test_reset_after_rearm_returns_full_like_any_armed_bulk(self):
        """A reset once the route is split again ends the yield; the bulk then follows the existing stop rule of
        an armed native bulk (FULL return, stock tail).  Flagged for the lead in STATUS, not decided silently."""
        rig = Rig(cached={})
        append = run_to_yield(rig)
        to_append(rig)
        run_append(rig)
        to_full(rig)
        rig.stock_handback(append)
        to_resume(rig)
        with self.assertRaises(ed.E3Pending) if False else contextmanager(lambda: (yield))():
            eh.before_reset_prefix(rig.core)
        self.assertEqual(rig.journal('append_yield_end')[-1]['reason'], 'stop-reset')
        # counted (lead resume 5): on the box any occurrence outside a stop or reset arm is a FAIL
        self.assertEqual(rig.d.append_yield_record()['end_reasons'], {'stop-reset': 1})
        settle_stop(rig)
        self.assertFalse(rig.d._split_ready)

    def test_pause_is_refused_while_the_cycle_owns_the_yield(self):
        rig = Rig(cached={})
        append = run_to_yield(rig)
        stages = [('closing', lambda: None), ('append', lambda: to_append(rig)),
                  ('handback', lambda: (run_append(rig), to_full(rig))),
                  ('resume', lambda: (rig.stock_handback(append), to_resume(rig)))]
        for phase, step in stages:
            step()
            with self.subTest(phase=phase):
                self.assertEqual(rig.d.append_yield_phase(), phase)
                with self.assertRaisesRegex(E3ContractError, 'cycle-owned'):
                    eh.before_pause(rig.core, 'keep')
                self.assertIsNone(rig.d.stop_kind)

    def test_shutdown_mid_append(self):
        rig = Rig(cached={})
        run_to_yield(rig)
        to_append(rig)
        self.assertIsNotNone(rig.admit())
        rig.drive(lambda: rig.d.tx is not None)
        try:
            eh.before_engine_shutdown(rig.core)
        except ed.E3Pending:
            pass
        self.assertIsNone(rig.d.append_yield_phase())
        self.assertEqual(rig.journal('append_yield_end')[-1]['reason'], 'stop-shutdown')
        settle_stop(rig)


# --- Mutants ------------------------------------------------------------------------------------------------------

@contextmanager
def mutant(owner, name, *edits):
    """``owner.name`` recompiled from its own source with each (old, new) edit applied exactly once."""
    import inspect
    import textwrap
    function = owner.__dict__[name] if isinstance(owner, type) else getattr(owner, name)
    source = inspect.getsource(function)
    for old, new in edits:                    # edits are written at the file's own indentation
        assert source.count(old) == 1, (name, old, source.count(old))
        source = source.replace(old, new)
    source = textwrap.dedent(source)
    if isinstance(owner, type):
        # a recompiled method has no __class__ cell: zero-argument super() names its class explicitly
        source = source.replace('super()', 'super(%s, self)' % owner.__name__)
    module = sys.modules[function.__module__]
    namespace = {}
    exec(compile(source, '<mutant %s>' % name, 'exec'), module.__dict__, namespace)
    with mock.patch.object(owner, name, namespace[name]):
        yield


MUTANT_FAILURES = (AssertionError, ed.DriverError, E3ContractError)
MUTANT_LOG = []                          # (test, exception type, first line): which check caught each mutant


class Mutants(unittest.TestCase):
    """Each named engine mutation fails a test of this module (the lead's and Codex's binding mutants)."""

    def fails(self, check, *mutation):
        with mutant(*mutation), self.assertRaises(MUTANT_FAILURES) as caught:
            check()
        MUTANT_LOG.append((self.id().rsplit('.', 1)[-1], type(caught.exception).__name__,
                           str(caught.exception).splitlines()[0][:150] if str(caught.exception) else ''))

    def cycle(self):
        connected_cycle(self)

    # A1 1: the bulk held for the FULL window; the hold keyed on _split_ready only
    def test_hold_on_split_ready_only(self):
        self.fails(self.cycle, sh, '_defer_stock_while_split_armed',
                   ('if not (callable(holds) and holds(request, kind)):', 'if True:'))

    # A1 2: other held requests released in the window
    def test_window_releases_other_requests(self):
        self.fails(self.cycle, ed.NativeSegmentDriver, 'append_yield_holds',
                   ("        return True\n", "        return request is y['bulk']\n"))

    # handback exemption widened to any one-token positive-frontier request
    def test_handback_exemption_for_any_request(self):
        def check():
            rig = Rig(cached={})
            append = run_to_yield(rig)
            to_append(rig)
            run_append(rig)
            to_full(rig)
            third = rig.add_held('ra-third', 6, queue='waiting')
            third.num_computed_tokens = 5
            self.assertFalse(sh.defer_stock_positive_frontier(rig.s, append))
            self.assertTrue(sh.defer_stock_positive_frontier(rig.s, third))
        check()
        self.fails(check, ed.NativeSegmentDriver, 'append_yield_holds', ("request is y['append'] and ", ''))

    # A1 3a: a remaining < 32,768 tokens bulk goes to stock instead of re-entering
    def test_reentry_floor(self):
        self.fails(self.cycle, auto.AutoArmPolicy, '_yield_native',
                   ('        return self._begin_reentry(request)\n',
                    '        if request.num_tokens - request.num_computed_tokens < MIN_NATIVE_ROWS:\n'
                    '            self._restore("stock-tail")\n            return False\n'
                    '        return self._begin_reentry(request)\n'))

    # A1 3b: a different head request selected at re-entry (driver target, then policy binding)
    def head_cycle(self):
        rig = Rig(cached={})
        append = run_to_yield(rig)
        to_append(rig)
        run_append(rig)
        to_full(rig)
        rig.stock_handback(append)
        head = rig.add_held('ra-head', 6, queue='waiting')
        rig.s.waiting.items.remove(head)
        rig.s.waiting.items.insert(0, head)
        rig.s.skipped_waiting.items.clear()
        to_resume(rig)
        txn = rig.admit()
        self.assertEqual(txn.request_uid, BULK_UID)
        rig.drive(lambda: rig.d.tx is not None)
        self.assertIn(head, rig.s.waiting.items)

    def test_reentry_head_target(self):
        self.head_cycle()
        self.fails(self.head_cycle, ed.NativeSegmentDriver, 'append_yield_target',
                   ("return ('resume', y['bulk'])", "return ('resume', self.scheduler.waiting.peek_request())"))

    def test_reentry_head_policy(self):
        self.fails(self.head_cycle, auto.AutoArmPolicy, '_request',
                   ('request = self.request_ref() if self.request_ref is not None else None',
                    'request = self.scheduler.waiting.peek_request() if self.scheduler.waiting else None'))

    # Codex 1: the close frees or preempts the scheduler request
    def test_close_frees_the_bulk(self):
        self.fails(self.cycle, ed.NativeSegmentDriver, '_drive_yield_close',
                   ("            self._retire_retained_cancel(reason='yield')\n",
                    "            self._retire_retained_cancel(reason='yield')\n"
                    "            self.scheduler.kv_cache_manager.free(y['bulk'])\n"))

    def test_close_preempts_the_bulk(self):
        self.fails(self.cycle, ed.NativeSegmentDriver, '_drive_yield_close',
                   ("            self._retire_retained_cancel(reason='yield')\n",
                    "            self._retire_retained_cancel(reason='yield')\n"
                    "            y['bulk'].num_computed_tokens = 0\n"))

    def test_close_sets_stop_kind(self):
        self.fails(self.cycle, ed.NativeSegmentDriver, '_drive_yield_close',
                   ("            self._retire_retained_cancel(reason='yield')\n",
                    "            self._retire_retained_cancel(reason='yield')\n"
                    "            self.stop_kind = 'pause'\n"))

    # Codex 2: the append writes a block the bulk holds
    def reuse_cycle(self):
        rig = Rig(cached={})
        manager = rig.s.kv_cache_manager
        real = manager.allocate_slots

        def allocate(request, num_new, **kwargs):          # an allocator handing out the bulk's blocks
            if request.request_id == APPEND_UID:
                manager.rows[APPEND_UID] = tuple(manager.rows[BULK_UID][:num_new])
                return object()
            return real(request, num_new, **kwargs)
        manager.allocate_slots = allocate
        run_to_yield(rig)
        to_append(rig)
        run_append(rig)
        self.assertFalse(set(rig.rows(APPEND_UID)) & set(rig.rows(BULK_UID)))

    def test_engine_refuses_an_append_on_bulk_blocks(self):
        with self.assertRaisesRegex(ed.DriverError, 'block the yielded bulk holds'):
            self.reuse_cycle()

    def test_block_guard_removed(self):
        self.fails(self.reuse_cycle, ed.NativeSegmentDriver, 'append_yield_check_blocks',
                   ("        if held & {block for block in written if block > 0}:\n"
                    "            raise DriverError('append yield would write a block the yielded bulk holds')\n",
                    "        pass\n"))

    # Codex 3: the reopen from a fresh prefix lookup instead of the committed frontier
    def test_reopen_from_lookup(self):
        self.fails(self.cycle, ed.NativeSegmentDriver, 'append_yield_set_phase',
                   ("        y['phase'] = phase\n", "        y['phase'] = phase\n"
                    "        if phase == 'resume':\n            y['bulk'].num_computed_tokens = 0\n"))

    def test_reopen_from_lookup_without_the_frontier_check(self):
        with mutant(ed.NativeSegmentDriver, '_note_yield_begin',
                    ("            if transaction.chunk_start != y['frontier']:\n"
                     "                raise DriverError('yielded bulk reopened away from its committed frontier')\n",
                     "")):
            self.fails(self.cycle, ed.NativeSegmentDriver, 'append_yield_set_phase',
                       ("        y['phase'] = phase\n", "        y['phase'] = phase\n"
                        "        if phase == 'resume':\n            y['bulk'].num_computed_tokens = 0\n"))

    # Codex 4: the close retires before every rank certified its row closed
    def partial_close(self):
        """Rank 3 answers the first close round with a valid not-closed ack (row still owned): the engine keeps
        the row and the yield in 'closing', retries, and moves on only once every rank certified the close."""
        import wire_v1
        rig = Rig(cached={})
        run_to_yield(rig)
        worker = rig.stack.worker(3)
        real = worker.flash_s5_cancel
        calls = []

        def deferring(message):
            calls.append(message)
            if len(calls) == 1:
                return wire_v1.CancelAck(message.transaction, 3, message.state_epoch, 'accepted', (), (), 0)
            return real(message)
        worker.flash_s5_cancel = deferring
        rig.d.advance()
        rig.d.advance()
        self.assertEqual(rig.d.append_yield_phase(), 'closing')
        self.assertIsNotNone(rig.d._retained)
        self.assertEqual(rig.d.append_yield_counters['rows_closed'], 0)
        to_append(rig)
        self.assertGreaterEqual(len(calls), 2)

    def test_close_without_all_rank_acks(self):
        self.partial_close()
        self.fails(self.partial_close, ed.NativeSegmentDriver, '_drive_yield_close',
                   ("        if set(acks) == set(self.ranks) and all(ack.closed for ack in acks.values()):\n",
                    "        if acks and any(ack.closed for ack in acks.values()):\n"))

    # candidate bounds
    def test_candidate_over_max_new(self):
        def check():
            rig = Rig(cached={})
            rig.add_held('ra-long', 10)                     # 9 new > 8
            self.assertIsNone(rig.c.append_yield_candidate(rig.s, rig.bulk))
        check()
        self.fails(check, sh.PrefillAdmissionController, '_append_yield_plan',
                   ('if end - start < 1 or end - start > config.max_new:', 'if end - start < 1:'))

    def test_candidate_without_a_plan(self):
        def check():
            rig = Rig(cached={})
            rig.add_held('ra-ok', 6)
            rig.s._mamba_block_aligned_split = lambda request, n, num_new_local_computed_tokens=0: 0
            self.assertIsNone(rig.c.append_yield_candidate(rig.s, rig.bulk))
        check()
        self.fails(check, sh.PrefillAdmissionController, '_append_yield_plan',
                   ('        return plan or None\n', '        return plan or [(start, end)]\n'))

    def test_candidate_cohort_not_proven(self):
        def check():
            rig = Rig(cached={})
            rig.add_held('ra-ok', 6)
            rig.s.running = [NS(request_id='d1')]
            self.assertIsNone(rig.c.append_yield_candidate(rig.s, rig.bulk, frozenset({1})))
        check()
        self.fails(check, sh.PrefillAdmissionController, 'append_yield_candidate',
                   ('        if decode_counts is not None and not set(range(1, running + 2)) <= set(decode_counts):\n'
                    '            return None\n', ''))

    # review MAJOR-3: membership instead of the contiguous range; the last running slot taken
    def test_candidate_counts_not_contiguous(self):
        def check():
            rig = Rig(cached={})
            rig.add_held('ra-ok', 6)
            rig.s.running = [NS(request_id='d1'), NS(request_id='d2')]
            self.assertIsNone(rig.c.append_yield_candidate(rig.s, rig.bulk, frozenset({1, 3})))
        check()
        self.fails(check, sh.PrefillAdmissionController, 'append_yield_candidate',
                   ('not set(range(1, running + 2)) <= set(decode_counts)', 'running + 1 not in decode_counts'))

    def test_candidate_takes_the_last_slot(self):
        def check():
            rig = Rig(cached={})
            rig.add_held('ra-ok', 6)
            rig.s.running = [NS(request_id='d%d' % i) for i in range(3)]
            rig.s.max_num_running_reqs = 4
            self.assertIsNone(rig.c.append_yield_candidate(rig.s, rig.bulk))
        check()
        self.fails(check, sh.PrefillAdmissionController, 'append_yield_candidate',
                   ("if running + 2 > getattr(scheduler, 'max_num_running_reqs', 1):",
                    "if running + 1 > getattr(scheduler, 'max_num_running_reqs', 1):"))

    # review MAJOR-2: the candidate without the free-block check
    def test_candidate_without_the_free_block_check(self):
        def check():
            rig = Rig(cached={})
            rig.add_held('ra-ok', 6)
            rig.free_blocks = 2
            self.assertIsNone(rig.c.append_yield_candidate(rig.s, rig.bulk))
        check()
        self.fails(check, sh.PrefillAdmissionController, '_append_yield_plan',
                   ("        if needed + manager.watermark_blocks > manager.block_pool.get_num_free_blocks():\n"
                    "            return None\n", ""))

    def test_second_yield_over_per_bulk(self):
        self.fails(self.cycle, ed.NativeSegmentDriver, '_yield_offer',
                   ('if self._yield_counts.get(key, 0) >= config.per_bulk:',
                    'if self._yield_counts.get(key, 0) > config.per_bulk:'))

    # no idle-hold budget while a yield owns the split route
    def test_idle_hold_clock_during_a_yield(self):
        def check():
            rig = Rig(cached={})
            append = run_to_yield(rig)
            to_append(rig)
            self.assertTrue(sh.defer_stock_positive_frontier(rig.s, rig.bulk))
            run_append(rig)
            to_full(rig)
            rig.stock_handback(append)
            to_resume(rig)
            other = rig.add_held(OTHER_UID, 6)
            self.assertTrue(sh.defer_stock_prefix_hit(rig.s, NS(blocks=()), request=other))
            self.assertIsNone(rig.c._split_hold, 'no 30 s idle-hold clock runs during the yield')
            self.assertIsNone(rig.c.last_split_hold_timeout)
        check()
        self.fails(check, sh.PrefillAdmissionController, '_note_split_hold',
                   ("        if callable(yielding) and yielding():\n", "        if False:\n"))

    # off is frozen: the control's off ignored
    def test_control_off_ignored(self):
        def check():
            rig = Rig(cached={}, enabled=False)
            run_to_yield(rig)
            self.assertIsNone(rig.d.append_yield_phase())
        check()
        with mutant(sh.PrefillAdmissionController, 'append_yield_candidate',
                    ('if config is None or not config.enabled:', 'if config is None:')):
            self.fails(check, ed.NativeSegmentDriver, '_yield_offer',
                       ('if config is None or not config.enabled or self._yield is not None:',
                        'if config is None or self._yield is not None:'))


# --- B2: the FULL window's bound; B3: a refused reopen after the re-arm ------------------------------------------

def handback_window(rig):
    """The yield in its FULL window: the append committed natively, the route FULL, the handback not scheduled."""
    append = run_to_yield(rig)
    to_append(rig)
    run_append(rig)
    to_full(rig)
    return append


def step_admit(rig):
    """The admission seam of one engine step: a PAUSED_ALL core schedules nothing (the policy pauses it before
    every rollback), so the seam is reached only while the core is unpaused."""
    return None if rig.core.paused else rig.admit()


def policy_ticks(rig, count):
    for _ in range(count):
        with auto_arm_workers():
            rig.p.tick()
            rig.d.advance()
            rig.stack.complete_everything()


class HandbackBound(unittest.TestCase):
    """B2 (lead resume 5): the FULL window always ends.  Its ends: the append's first token, the append leaving
    (abort or finish), or FLASH_S5_APPEND_YIELD_HANDBACK_STEPS engine steps on FULL with the handback still
    unscheduled; then append_yield_handback_expired is journaled, the yield ends and the bulk re-enters, and the
    append waits under the split hold as without a yield.  Steps are auto-arm ticks (one per engine step), no clock."""

    def test_end_first_token_counts_no_expiry(self):
        rig = Rig(cached={}, handback_steps=3)
        append = handback_window(rig)
        policy_ticks(rig, 2)                          # two FULL steps without the handback: below the bound
        self.assertEqual(rig.d.append_yield_phase(), 'handback')
        self.assertEqual(rig.d._yield['handback_steps'], 2)
        rig.stock_handback(append)                    # the first token on the third step
        to_resume(rig)
        finish_bulk(rig)
        self.assertEqual(rig.d.append_yield_counters['handback_expired'], 0)
        self.assertEqual([e['reason'] for e in rig.journal('append_yield_end')], ['reopened'])

    def test_end_append_leaves(self):
        for how in ('abort', 'finish'):
            with self.subTest(how=how):
                rig = Rig(cached={}, handback_steps=3)
                append = handback_window(rig)
                policy_ticks(rig, 1)
                if how == 'abort':
                    self.assertIsNone(abort(rig, append))
                    rig.stock_abort(append)
                else:                               # stock finished it (a length-1 stop) after its handback left
                    rig.s.requests.pop(append.request_id)
                    rig.s.skipped_waiting.items.remove(append)
                    append.status = NS(name='FINISHED_STOPPED')
                to_resume(rig)
                finish_bulk(rig)
                self.assertEqual(rig.d.append_yield_counters['handback_expired'], 0)
                self.assertEqual(rig.journal('append_yield_end')[-1]['reason'], 'reopened')

    def expire(self, rig):
        append = handback_window(rig)
        generation = rig.d.route_generation
        frontier = rig.bulk.num_computed_tokens
        policy_ticks(rig, 2)
        self.assertEqual(rig.d.append_yield_phase(), 'handback')
        policy_ticks(rig, 1)                          # the third FULL step spends the bound
        return append, generation, frontier

    def test_end_bound_expires_the_window_and_the_bulk_reenters(self):
        rig = Rig(cached={}, handback_steps=3)
        append, generation, frontier = self.expire(rig)
        expired = rig.journal('append_yield_handback_expired')
        self.assertEqual([(e['seq'], e['steps']) for e in expired], [(1, 3)])
        self.assertEqual(rig.d.append_yield_counters['handback_expired'], 1)
        # the FULL window ended: the bulk re-enters (bound to it, plan from its frontier, no 32,768 floor)
        self.assertEqual(rig.d.append_yield_phase(), 'reenter')
        self.assertNotEqual(rig.p.phase, 'native')
        to_resume(rig)
        self.assertEqual(rig.d.route_generation, generation + 1)
        self.assertEqual(rig.p.plan['start'], frontier)
        self.assertEqual(rig.d.admission_intent()['request_uid'], BULK_UID)
        # the held append (ahead of the bulk in the queues) waits; only the bulk is admitted
        self.assertTrue(sh.defer_stock_positive_frontier(rig.s, append))
        txn = rig.admit()
        self.assertEqual((txn.request_uid, txn.chunk_start), (BULK_UID, frontier))
        rig.drive(lambda: rig.d.tx is not None)
        self.assertIsNone(rig.d.append_yield_phase())
        self.assertEqual(rig.journal('append_yield_end')[-1]['reason'], 'handback-expired')
        entry = rig.d.append_yield_record()['history'][-1]
        self.assertEqual(entry['reason'], 'handback-expired')
        self.assertIn('handback_expired', entry['times'])
        self.assertEqual(rig.d.append_yield_counters['abandoned'], 0)
        # the split hold (as without a yield) keeps the append held until the bulk's FULL return
        self.assertTrue(sh.defer_stock_positive_frontier(rig.s, append))
        rig.finish_chunk()
        finish_bulk(rig)
        self.assertEqual(append.num_output_tokens, 0)
        self.assertIn(append, rig.s.skipped_waiting.items)
        self.assertFalse(sh.defer_stock_positive_frontier(rig.s, append), 'FULL again: stock serves it')

    def test_bulk_aborted_after_the_expiry(self):
        rig = Rig(cached={}, handback_steps=3)
        self.expire(rig)
        self.assertIsNone(abort(rig, rig.bulk))
        rig.stock_abort(rig.bulk)
        policy_idle(rig)
        self.assertFalse(rig.d._split_ready)
        self.assertIsNone(rig.d.admission_intent())

    def test_mutant_without_the_bound(self):
        def check():
            rig = Rig(cached={}, handback_steps=3)
            handback_window(rig)
            rig.tick(lambda: rig.p.phase != 'native' or rig.d.append_yield_phase() != 'handback', limit=40)
        check()
        Mutants.fails(self, check, ed.NativeSegmentDriver, 'append_yield_handback_step',
                      ("        if y['handback_steps'] < self.controller.append_yield.handback_steps:\n",
                       "        if True:\n"))

    def test_mutant_bound_off_by_one(self):
        def check():
            rig = Rig(cached={}, handback_steps=3)
            self.expire(rig)
            self.assertEqual([e['steps'] for e in rig.journal('append_yield_handback_expired')], [3])
        check()
        Mutants.fails(self, check, ed.NativeSegmentDriver, 'append_yield_handback_step',
                      ("if y['handback_steps'] < self.controller", "if y['handback_steps'] <= self.controller"))


def refuse_bulk_blocks(rig):
    """Leaf: the pinned KVCacheManager.allocate_slots returning None (no free blocks) for the bulk only; the
    controller then refuses the admission (scheduler_hooks: allocation refused, nothing parked)."""
    manager = rig.s.kv_cache_manager
    real = manager.allocate_slots

    def allocate(request, num_new, **kwargs):
        if request.request_id == BULK_UID:
            return None
        return real(request, num_new, **kwargs)
    manager.allocate_slots = allocate
    return lambda: setattr(manager, 'allocate_slots', real)


class ReentryRefused(unittest.TestCase):
    """B3 (lead resume 5): a refused reopen after the re-arm never raises.  Past
    FLASH_S5_APPEND_YIELD_REENTRY_RETRIES scheduler passes without the bulk's admission the policy journals
    append_yield_reentry_refused, ends the yield and disarms through the existing refused-admission rule
    (leave_split, FULL, the bulk on stock)."""

    def resumed(self, rig):
        append = handback_window(rig)
        rig.stock_handback(append)
        to_resume(rig)
        return append

    def test_retries_then_disarm_never_raise(self):
        rig = Rig(cached={}, reentry_retries=4)
        self.resumed(rig)
        restore = refuse_bulk_blocks(rig)
        generation = rig.d.route_generation
        for passes in range(1, 5):                    # four refused passes: within the bound
            self.assertIsNone(rig.admit())
            policy_ticks(rig, 1)
            self.assertEqual(rig.d.append_yield_phase(), 'resume', passes)
        self.assertIsNone(rig.admit())                # the fifth refused pass
        policy_ticks(rig, 1)
        refused = rig.journal('append_yield_reentry_refused')
        self.assertEqual([(e['seq'], e['passes']) for e in refused], [(1, 5)])
        self.assertIsNone(rig.d.append_yield_phase())
        self.assertEqual(rig.journal('append_yield_end')[-1]['reason'], 'reentry-refused')
        self.assertEqual(rig.d.append_yield_counters['reentry_refused'], 1)
        policy_idle(rig)
        self.assertEqual(rig.journal('auto_refused')[-1]['reason'], 'request-changed-or-admission-refused')
        self.assertFalse(rig.d._split_ready)
        self.assertIsNone(rig.d.admission_intent())
        self.assertEqual(rig.d.route_generation, generation + 1)
        self.assertIsNone(rig.d.error)
        restore()
        self.assertFalse(sh.defer_stock_positive_frontier(rig.s, rig.bulk), 'the bulk continues on stock')
        self.assertIn(rig.bulk, rig.s.waiting.items)

    def test_a_late_admission_within_the_bound_reopens(self):
        rig = Rig(cached={}, reentry_retries=4)
        self.resumed(rig)
        restore = refuse_bulk_blocks(rig)
        for _ in range(3):
            self.assertIsNone(rig.admit())
            policy_ticks(rig, 1)
        restore()
        txn = rig.admit()
        self.assertEqual(txn.request_uid, BULK_UID)
        policy_ticks(rig, 1)
        rig.drive(lambda: rig.d.tx is not None)
        self.assertEqual(rig.journal('append_yield_end')[-1]['reason'], 'reopened')
        self.assertEqual(rig.d.append_yield_counters['reentry_refused'], 0)

    def test_mutant_raises(self):
        def check():
            rig = Rig(cached={}, reentry_retries=1)
            self.resumed(rig)
            refuse_bulk_blocks(rig)
            for _ in range(3):
                step_admit(rig)
                policy_ticks(rig, 1)
            self.assertEqual(rig.d.append_yield_counters['reentry_refused'], 1)
        check()
        Mutants.fails(self, check, auto.AutoArmPolicy, '_yield_native',
                      ("            if d.append_yield_reentry_wait(c.admitted, c._schedule_seq):\n"
                       "                return self._native_refused()\n",
                       "            if d.append_yield_reentry_wait(c.admitted, c._schedule_seq):\n"
                       "                raise E3ContractError('auto-arm-yielded-bulk-reopen-refused; ownership retained')\n"))

    def test_mutant_never_disarms(self):
        def check():
            rig = Rig(cached={}, reentry_retries=1)
            self.resumed(rig)
            refuse_bulk_blocks(rig)
            for _ in range(6):
                step_admit(rig)
                policy_ticks(rig, 1)
            self.assertFalse(rig.d._split_ready, 'the refused bulk does not keep the split route')
        check()
        Mutants.fails(self, check, ed.NativeSegmentDriver, 'append_yield_reentry_wait',
                      ("        if passes <= self.controller.append_yield.reentry_retries:\n",
                       "        if True:\n"))


class ReentryNeverRaises(unittest.TestCase):
    """The re-entry's other refusals close like the frozen selection-changed rules, never a raise: no plan at the
    window's end (restore on FULL), the frontier moved during the drain (restore on FULL) or before the arm
    (rollback through leave_split)."""

    def window_done(self, rig):
        append = handback_window(rig)
        rig.stock_handback(append)
        return append

    def closes_on_full(self, rig, reason):
        policy_idle(rig)
        self.assertIsNone(rig.d.error)
        self.assertFalse(rig.d._split_ready)
        self.assertIsNone(rig.d.admission_intent())
        self.assertIsNone(rig.d.append_yield_phase())
        self.assertEqual(rig.journal('append_yield_end')[-1]['reason'], reason)
        self.assertEqual(rig.journal('auto_closed')[-1]['reason'], reason)
        self.assertIn(rig.bulk, rig.s.waiting.items)

    def test_no_reentry_plan(self):
        rig = Rig(cached={})
        self.window_done(rig)
        rig.bulk.num_computed_tokens = rig.bulk.num_tokens - 1      # nothing left for a native chunk
        self.closes_on_full(rig, 'yielded-bulk-has-no-reentry-plan')

    def test_frontier_moved_during_the_drain(self):
        rig = Rig(cached={})
        self.window_done(rig)
        rig.tick(lambda: rig.p.phase == 'drain')
        rig.bulk.num_computed_tokens += 1
        self.closes_on_full(rig, 'yielded-bulk-frontier-moved')

    def test_frontier_moved_before_the_arm(self):
        rig = Rig(cached={})
        self.window_done(rig)
        rig.tick(lambda: rig.p.phase == 'split')
        rig.bulk.num_computed_tokens += 1
        # the frozen rollback: refused with the reason, leave_split, closed by the existing return rule
        self.closes_on_full(rig, 'returned-before-native-admission')
        self.assertEqual(rig.journal('auto_refused')[-1]['reason'], 'yielded-bulk-frontier-moved')


class ReentryNeverRaisesStaged(StagedDriver, ReentryNeverRaises):
    """ReentryNeverRaises on StagedSegmentDriver."""


# --- review of f76a31de: MAJOR-1..3, MINOR-1 -----------------------------------------------------------------------

def refuse_profile(rig):
    """The stepped (unfused) QSA path's schedule-profile refusal on the append's in-flight chunk, entered at the
    engine's own _refuse_profile.  Leaf: the active-stage profile query (stage 0, layer 0, stage-0 ranks) the stepped
    path's _drive would have sent; the worker-side profile agreement itself is test_e3_native_schedule_profile's."""
    d = rig.d
    d._profile_query = NS(stage=0, layer=0, ranks=tuple(d.stage_ranks[0]), transaction=d.tx)
    d._profile_observed = {}
    d._refuse_profile('unsupported-native-schedule-profile')


@contextmanager
def profile_failure_leaves(rig, request):
    """Leaves for the driver's own failure publication (_finish_profile_failure, the embedding-core route this rig's
    core takes): the pinned vllm.v1.request.RequestStatus / vllm.v1.engine output types (constructed, never read
    here) and Scheduler.finish_requests(FINISHED_ERROR), which removes the request as stock's abort does."""
    def finish_requests(ids, status):
        for uid in ids:
            target = rig.s.requests.get(uid)
            if target is not None:
                rig.stock_abort(target)
                target.status = NS(name='FINISHED_ERROR')
    request.client_index = 0
    rig.s.finish_requests = finish_requests
    modules = {'vllm': types.ModuleType('vllm'), 'vllm.v1': types.ModuleType('vllm.v1'),
               'vllm.v1.request': types.ModuleType('vllm.v1.request'),
               'vllm.v1.engine': types.ModuleType('vllm.v1.engine')}
    modules['vllm.v1.request'].RequestStatus = NS(FINISHED_ERROR='FINISHED_ERROR')
    modules['vllm.v1.engine'].EngineCoreOutput = lambda **kw: NS(**kw)
    modules['vllm.v1.engine'].EngineCoreOutputs = lambda **kw: NS(**kw)
    modules['vllm.v1.engine'].FinishReason = NS(ERROR='error')
    with mock.patch.dict(sys.modules, modules):
        yield


def refuse_allocation(rig, uid, after=0):
    """Leaf: allocate_slots returning None for ``uid`` after ``after`` granted allocations (a prefix-cache change or
    block pressure between the candidate's check and the admission)."""
    manager = rig.s.kv_cache_manager
    real = manager.allocate_slots
    granted = []

    def allocate(request, num_new, **kwargs):
        if request.request_id == uid:
            if len(granted) >= after:
                return None
            granted.append(num_new)
        return real(request, num_new, **kwargs)
    manager.allocate_slots = allocate
    return lambda: setattr(manager, 'allocate_slots', real)


def engine_steps_until(rig, until, limit):
    """Engine steps (engine_step) until ``until()``; fails when the bound is spent (no clock)."""
    for _ in range(limit):
        if until():
            return
        engine_step(rig)
    raise AssertionError('no exit within %d engine steps: policy=%s yield=%r' % (limit, rig.p.phase,
                                                                                rig.d.append_yield_phase()))


def engine_step(rig):
    """One engine step of the rig: auto-arm's tick, the driver tick, worker completions, then the scheduler's
    admission seam when the core is not paused."""
    with auto_arm_workers():
        rig.p.tick()
        rig.d.advance()
        rig.stack.complete_everything()
    step_admit(rig)


class ReviewMajor1StaleYield(unittest.TestCase):
    """MAJOR-1: a profile refusal (the one stop_kind writer outside stop()) during the append runs the yield's abort
    rule; the stale-yield guard backs up any other bypass.  Neither leaves FULL holding every prefill."""

    def mid_append(self, rig):
        append = run_to_yield(rig, append_tokens=11, append_cached=0)
        frontier = rig.bulk.num_computed_tokens
        to_append(rig)
        self.assertIsNotNone(rig.admit())
        rig.drive(lambda: rig.d.tx is not None)
        return append, frontier

    def recovers(self, rig, append, frontier):
        with profile_failure_leaves(rig, append):
            settle_stop(rig)
            rig.drive(lambda: append.request_id not in rig.s.requests)
        self.assertEqual(append.status.name, 'FINISHED_ERROR')
        to_resume(rig)
        txn = rig.admit()
        self.assertEqual((txn.request_uid, txn.chunk_start), (BULK_UID, frontier))
        rig.finish_chunk()
        finish_bulk(rig)
        self.assertEqual(rig.journal('append_yield_end')[-1]['reason'], 'reopened')

    def test_profile_refusal_runs_the_yield_abort_rule(self):
        rig = Rig(cached={}, max_new=10)
        append, frontier = self.mid_append(rig)
        refuse_profile(rig)
        self.assertEqual(rig.d.append_yield_phase(), 'handback')
        self.assertEqual(rig.journal('append_yield_append_aborted')[-1]['cause'], 'profile')
        self.recovers(rig, append, frontier)
        self.assertNotIn('append_yield_stale', rig.kinds())

    def test_stale_guard_backs_up_a_bypassed_stop(self):
        rig = Rig(cached={}, max_new=10)
        append, frontier = self.mid_append(rig)
        with mutant(ed.NativeSegmentDriver, '_refuse_profile',
                    ("            self._yield_stop('abort', cause='profile')\n", "            pass\n")):
            refuse_profile(rig)
        self.assertEqual(rig.d.append_yield_phase(), 'append')
        self.recovers(rig, append, frontier)
        self.assertEqual(rig.journal('append_yield_stale')[-1]['phase'], 'append')

    def test_mutant_profile_refusal_bypasses_the_yield(self):
        self.mutant_check(("            self._yield_stop('abort', cause='profile')\n", "            pass\n"))

    def mutant_check(self, edit):
        def check():
            rig = Rig(cached={}, max_new=10)
            append, frontier = self.mid_append(rig)
            refuse_profile(rig)
            self.assertEqual(rig.d.append_yield_phase(), 'handback')
            self.recovers(rig, append, frontier)
        check()
        Mutants.fails(self, check, ed.NativeSegmentDriver, '_refuse_profile', edit)

    def test_mutant_no_stale_guard(self):
        def check():
            rig = Rig(cached={}, max_new=10)
            append, frontier = self.mid_append(rig)
            with mutant(ed.NativeSegmentDriver, '_refuse_profile',
                        ("            self._yield_stop('abort', cause='profile')\n", "            pass\n")):
                refuse_profile(rig)
            self.recovers(rig, append, frontier)
        check()
        Mutants.fails(self, check, auto.AutoArmPolicy, '_yield_native',
                      ("        if (phase in ('draining', 'closing', 'append', 'resume') and not d._split_ready and not d.active\n",
                       "        if (False and not d._split_ready and not d.active\n"))


class ReviewMajor2AppendRefused(unittest.TestCase):
    """MAJOR-2: the append's admission refused.  Before any append chunk: the yield goes to resume, the bulk reopens
    from its frontier and the append stays held.  After a partial append: the append's retained row closes through
    the same typed close (append_row_closed), then resume; the bulk reopens from its frontier and the append continues
    on stock from its own frontier after the bulk's FULL return.  A refused append is not reselected for that bulk."""

    def test_refused_before_any_chunk_resumes_the_bulk(self):
        rig = Rig(cached={})
        append = run_to_yield(rig)
        frontier = rig.bulk.num_computed_tokens
        refuse_allocation(rig, APPEND_UID)
        to_append(rig)
        for _ in range(6):
            engine_step(rig)
            if rig.d.append_yield_phase() != 'append':
                break
        self.assertEqual(rig.journal('append_yield_append_refused')[-1]['partial'], False)
        self.assertEqual(rig.d.append_yield_counters['append_refused'], 1)
        rig.drive(lambda: rig.d.tx is not None)
        self.assertEqual((rig.d.tx.request_uid, rig.d.tx.chunk_start), (BULK_UID, frontier))
        self.assertEqual(rig.journal('append_yield_end')[-1]['reason'], 'reopened')
        self.assertIn(append, rig.s.skipped_waiting.items)
        rig.finish_chunk()
        finish_bulk(rig)

    def test_refused_after_a_partial_append_closes_its_row_and_resumes(self):
        rig = Rig(cached={}, max_new=10)
        append = run_to_yield(rig, append_tokens=11, append_cached=0)
        frontier = rig.bulk.num_computed_tokens
        refuse_allocation(rig, APPEND_UID, after=1)
        to_append(rig)
        self.assertIsNotNone(rig.admit())
        rig.finish_chunk()
        self.assertEqual(rig.d._retained['transaction'].request_uid, APPEND_UID)
        for _ in range(12):
            engine_step(rig)
            if rig.d.append_yield_phase() == 'resume':
                break
        self.assertEqual(rig.journal('append_yield_append_refused')[-1]['partial'], True)
        self.assertEqual(len(rig.journal('append_yield_append_row_closed')), 1)
        closed = rig.journal('request_row_closed')
        self.assertEqual([e['reason'] for e in closed], ['yield', 'yield'])
        self.assertIsNone(rig.d._retained)
        self.assertTrue(rig.d._split_ready, 'no FULL return: the bulk reopens natively')
        txn = rig.c.parked.transaction if rig.c.parked is not None else rig.admit()
        self.assertEqual((txn.request_uid, txn.chunk_start), (BULK_UID, frontier))
        rig.drive(lambda: rig.d.tx is not None)
        self.assertEqual(rig.journal('append_yield_end')[-1]['reason'], 'reopened')
        self.assertEqual(append.num_computed_tokens, 4, 'the append keeps its own committed frontier')
        rig.finish_chunk()
        finish_bulk(rig)
        self.assertIn(append, rig.s.skipped_waiting.items)
        self.assertFalse(sh.defer_stock_positive_frontier(rig.s, append), 'FULL again: stock continues it')
        self.assertIsNone(rig.d.error)

    def test_a_refused_append_is_not_selected_again_for_the_bulk(self):
        rig = Rig(cached={})
        run_to_yield(rig)
        refuse_allocation(rig, APPEND_UID)
        to_append(rig)
        for _ in range(6):
            engine_step(rig)
        rig.drive(lambda: rig.d.tx is not None)
        rig.finish_chunk()                               # the reopened bulk's chunk: another boundary
        self.assertIsNone(rig.d.append_yield_phase())
        self.assertEqual(len(rig.journal('append_yield_selected')), 1)

    def test_mutant_no_refusal_exit(self):
        def check():
            rig = Rig(cached={})
            run_to_yield(rig)
            refuse_allocation(rig, APPEND_UID)
            to_append(rig)
            for _ in range(12):
                engine_step(rig)
            self.assertNotEqual(rig.d.append_yield_phase(), 'append', 'the append phase has an exit')
        check()
        Mutants.fails(self, check, ed.NativeSegmentDriver, 'append_yield_append_wait',
                      ("        if schedule_seq <= idle[1]:\n", "        if True:\n"))


class ReviewMajor3ReentryBounded(unittest.TestCase):
    """MAJOR-3 (the candidate side is Candidate / Mutants): a re-entry whose drain or proof keeps refusing closes on
    FULL within REENTRY_RETRIES refusals, unpaused, never a raise."""

    def test_drain_refusal_bounded(self):
        rig = Rig(cached={}, reentry_retries=3)
        append = handback_window(rig)
        rig.stock_handback(append)
        append.num_computed_tokens = append.num_tokens - 1         # a cohort the re-entry cannot prove
        engine_steps_until(rig, lambda: rig.p.phase == 'idle', 60)
        refused = rig.journal('append_yield_reentry_refused')
        self.assertEqual([(e['stage'], e['passes']) for e in refused], [('drain', 4)])
        self.assertEqual(rig.journal('auto_closed')[-1]['reason'], 'yielded-bulk-reentry-refused')
        self.assertFalse(rig.core.paused)
        self.assertFalse(rig.d._split_ready)
        self.assertIsNone(rig.d.error)

    def test_identity_refusal_bounded(self):
        rig = Rig(cached={}, reentry_retries=2)
        append = handback_window(rig)
        rig.stock_handback(append)
        for _ in range(60):
            with auto_arm_workers(counts=(2, 3)):               # a cohort of 1 is not proven
                rig.p.tick()
                rig.d.advance()
                rig.stack.complete_everything()
            step_admit(rig)                                     # the scheduler pass between attempts
            if rig.p.phase == 'idle':
                break
        self.assertEqual(rig.journal('append_yield_reentry_refused')[-1]['stage'], 'identity')
        self.assertEqual(rig.p.phase, 'idle')
        self.assertFalse(rig.core.paused)

    def test_mutant_unbounded_reentry(self):
        def check():
            rig = Rig(cached={}, reentry_retries=3)
            append = handback_window(rig)
            rig.stock_handback(append)
            append.num_computed_tokens = append.num_tokens - 1
            engine_steps_until(rig, lambda: rig.p.phase == 'idle', 60)
        check()
        Mutants.fails(self, check, ed.NativeSegmentDriver, 'append_yield_reentry_refused',
                      ("        if refusals <= self.controller.append_yield.reentry_retries:\n",
                       "        if True:\n"))


class ReviewMinor1NoDeadlineAtResume(unittest.TestCase):
    """MINOR-1: resume reads no clock (B3 replaced the deadline with a pass count): an append that leaves while held
    long after the policy's last deadline still reopens the bulk."""

    def check(self):
        rig = Rig(cached={})
        append = run_to_yield(rig)
        to_append(rig)
        rig.p.deadline = rig.d.clock() - 1.0                     # the selection-time deadline long past
        self.assertIsNone(abort(rig, append))
        rig.stock_abort(append)
        rig.admit()
        policy_ticks(rig, 1)
        txn = rig.admit()
        self.assertEqual(txn.request_uid, BULK_UID)
        rig.drive(lambda: rig.d.tx is not None)
        self.assertEqual(rig.journal('append_yield_end')[-1]['reason'], 'reopened')

    def test_append_left_after_the_old_deadline(self):
        self.check()

    def test_mutant_stale_deadline_raise(self):
        Mutants.fails(self, self.check, auto.AutoArmPolicy, '_yield_native',
                      ("        if phase == 'resume':\n            if d.active:\n",
                       "        if phase == 'resume':\n"
                       "            if self.deadline is not None and d.clock() > self.deadline:\n"
                       "                raise E3ContractError('auto-arm-yielded-bulk-reopen-deadline-expired')\n"
                       "            if d.active:\n"))


class DeltaMajor1ResumeBounded(unittest.TestCase):
    """Delta review of ae865d44, D-MAJOR-1: every path into resume has a bounded exit when the bulk's reopen is refused.
    Three paths enter resume on the existing arm (append_left, the append's refused admission, the partial append's row
    close); the bulk's own chunks moved ``admitted`` since that arm, so a pass count from the last arm's snapshot never
    ran.  The count now starts at resume's entry, on the yield."""
    BOUND = 40

    def append_left(self, rig):
        append = run_to_yield(rig)
        to_append(rig)
        abort(rig, append)
        rig.stock_abort(append)
        return 'append_yield_append_left'

    def append_refused(self, rig):
        run_to_yield(rig)
        refuse_allocation(rig, APPEND_UID)
        to_append(rig)
        return 'append_yield_append_refused'

    def partial_row_closed(self, rig):
        run_to_yield(rig, append_tokens=11, append_cached=0)
        refuse_allocation(rig, APPEND_UID, after=1)
        to_append(rig)
        self.assertIsNotNone(rig.admit())
        rig.finish_chunk()
        return 'append_yield_append_row_closed'

    PATHS = ('append_left', 'append_refused', 'partial_row_closed')

    def run_path(self, name, retries=3):
        rig = Rig(cached={}, max_new=10, reentry_retries=retries)
        admitted_at_arm = rig.p.admitted_before
        mark = getattr(self, name)(rig)
        self.assertGreater(rig.c.admitted, admitted_at_arm, 'the bulk\'s chunks moved admitted since the arm')
        refuse_bulk_blocks(rig)                               # the bulk's reopen is refused on every pass
        engine_steps_until(rig, lambda: rig.d.append_yield_phase() == 'resume', self.BOUND)
        self.assertIn(mark, rig.kinds())
        engine_steps_until(rig, lambda: rig.d.append_yield_phase() is None and rig.p.phase == 'idle', self.BOUND)
        return rig

    def test_every_path_into_resume_exits_within_the_bound(self):
        for name in self.PATHS:
            with self.subTest(path=name):
                rig = self.run_path(name)
                refused = rig.journal('append_yield_reentry_refused')
                self.assertEqual([e['passes'] for e in refused], [4])
                self.assertEqual(rig.journal('append_yield_end')[-1]['reason'], 'reentry-refused')
                self.assertEqual(rig.journal('auto_refused')[-1]['reason'], 'request-changed-or-admission-refused')
                self.assertFalse(rig.core.paused)
                self.assertFalse(rig.d._split_ready)
                self.assertIsNone(rig.d.admission_intent())
                self.assertIsNone(rig.d.error)
                self.assertFalse(sh.defer_stock_positive_frontier(rig.s, rig.bulk), 'the bulk continues on stock')

    def test_mutant_the_last_arms_snapshot(self):
        for name in self.PATHS:
            with self.subTest(path=name):
                self.run_path(name)
                Mutants.fails(self, lambda: self.run_path(name), auto.AutoArmPolicy, '_yield_native',
                              ("            if d.append_yield_reentry_wait(c.admitted, c._schedule_seq):\n",
                               "            if (c.admitted == self.admitted_before\n"
                               "                    and d.append_yield_reentry_wait(c.admitted, c._schedule_seq)):\n"))


class DeltaMinor1HandbackSkipped(unittest.TestCase):
    """D-MINOR-1: an append in the FULL window with more than its one handback token left (stopped by the abort rule:
    a profile refusal, the stale guard; leaf: its frontier set back, the request still in the scheduler) skips the
    window at once instead of spending HANDBACK_STEPS: the bulk re-enters under the yield's binding, the yield ends
    'handback-skipped' at the reopen, and the request continues on stock after the bulk's FULL return."""

    def check(self):
        rig = Rig(cached={}, handback_steps=8)
        append = handback_window(rig)
        append.num_computed_tokens = 4                       # 7 prompt tokens left: no one-token handback
        self.assertTrue(sh.defer_stock_positive_frontier(rig.s, append), 'the window cannot serve it')
        policy_ticks(rig, 1)
        skipped = rig.journal('append_yield_handback_skipped')
        self.assertEqual([e['tokens_left'] for e in skipped], [7])
        self.assertEqual(rig.d.append_yield_phase(), 'reenter')
        self.assertEqual(rig.d.append_yield_counters['handback_expired'], 0)
        engine_steps_until(rig, lambda: rig.d.append_yield_phase() == 'resume', 40)
        txn = rig.c.parked.transaction if rig.c.parked is not None else rig.admit()
        self.assertEqual(txn.request_uid, BULK_UID)
        rig.drive(lambda: rig.d.tx is not None)
        self.assertEqual(rig.journal('append_yield_end')[-1]['reason'], 'handback-skipped')
        self.assertEqual(rig.d.append_yield_end_reasons, {'handback-skipped': 1})
        self.assertEqual(rig.d.append_yield_counters['abandoned'], 0)
        self.assertEqual(rig.d.append_yield_counters['handback_skipped'], 1)
        rig.finish_chunk()
        finish_bulk(rig)
        self.assertFalse(sh.defer_stock_positive_frontier(rig.s, append), 'FULL again: stock continues it')
        self.assertIsNone(rig.d.error)

    def test_an_append_not_at_its_handback_skips_the_window(self):
        self.check()

    def test_the_one_token_handback_still_runs(self):
        rig = Rig(cached={}, handback_steps=8)
        append = handback_window(rig)
        policy_ticks(rig, 1)
        self.assertEqual(rig.journal('append_yield_handback_skipped'), [])
        self.assertEqual(rig.d.append_yield_phase(), 'handback')
        self.assertFalse(sh.defer_stock_positive_frontier(rig.s, append))

    def test_mutant_no_skip(self):
        Mutants.fails(self, self.check, ed.NativeSegmentDriver, 'append_yield_handback_skip',
                      ("        if getattr(append, 'num_output_tokens', 0) > 0 or left <= 1:\n",
                       "        if True:\n"))


class DeltaMinor2DrainRetrySpaced(unittest.TestCase):
    """D-MINOR-2: a re-entry drain refusal waits one scheduler pass with the pause released before the next attempt, and
    the yield's hold lets a decoder's running stock prefill finish its chunk in that pass (leaf: stock scheduling the
    chunk, run only while the core is unpaused and the hold releases it); the retry then succeeds."""

    def check(self):
        rig = Rig(cached={}, reentry_retries=3)
        append = handback_window(rig)
        rig.stock_handback(append)
        append.num_computed_tokens = append.num_tokens - 1   # in the running queue mid-prefill: the cohort refuses
        waits = []
        for _ in range(60):
            if rig.p.phase == 'drain_wait':
                held = sh.defer_stock_running_prefill(rig.s, append)
                waits.append((rig.core.paused, held))
                if not rig.core.paused and not held:
                    append.num_computed_tokens = append.num_tokens      # stock finished the in-flight chunk
            if rig.d.append_yield_phase() == 'resume':
                break
            engine_step(rig)
        self.assertEqual(rig.d.append_yield_phase(), 'resume', 'the retry succeeded')
        self.assertIn((False, False), waits, 'the pause and the running-prefill hold released between attempts')
        self.assertEqual(rig.d.append_yield_counters['reentry_refused'], 0)
        self.assertTrue(sh.defer_stock_running_prefill(rig.s, append) or append.num_computed_tokens == append.num_tokens)
        txn = rig.c.parked.transaction if rig.c.parked is not None else rig.admit()
        self.assertEqual(txn.request_uid, BULK_UID)
        rig.drive(lambda: rig.d.tx is not None)
        self.assertEqual(rig.journal('append_yield_end')[-1]['reason'], 'reopened')
        self.assertIsNone(rig.d.error)

    def test_a_retry_succeeds_after_the_in_flight_work_clears(self):
        self.check()

    def test_mutant_back_to_back_retries(self):
        Mutants.fails(self, self.check, auto.AutoArmPolicy, '_reentry_refused',
                      ("        self.scheduler.set_pause_state(self.old_pause)\n        self.phase = 'drain_wait'\n",
                       "        self.phase = 'drain'\n"))

    def test_mutant_running_prefill_still_held(self):
        Mutants.fails(self, self.check, ed.NativeSegmentDriver, 'append_yield_holds',
                      ("        if y['phase'] == 'reenter' and y.get('drain_wait') and kind == 'running-prefill':\n",
                       "        if False:\n"))


class DeltaMajor1ResumeBoundedStaged(StagedDriver, DeltaMajor1ResumeBounded):
    """D-MAJOR-1 on StagedSegmentDriver."""


class DeltaMinor1HandbackSkippedStaged(StagedDriver, DeltaMinor1HandbackSkipped):
    """D-MINOR-1 on StagedSegmentDriver."""


class DeltaMinor2DrainRetrySpacedStaged(StagedDriver, DeltaMinor2DrainRetrySpaced):
    """D-MINOR-2 on StagedSegmentDriver."""


class ReviewMajor1StaleYieldStaged(StagedDriver, ReviewMajor1StaleYield):
    """MAJOR-1 on StagedSegmentDriver."""


class ReviewMajor2AppendRefusedStaged(StagedDriver, ReviewMajor2AppendRefused):
    """MAJOR-2 on StagedSegmentDriver."""


class ReviewMajor3ReentryBoundedStaged(StagedDriver, ReviewMajor3ReentryBounded):
    """MAJOR-3 re-entry bound on StagedSegmentDriver."""


class ReviewMinor1NoDeadlineAtResumeStaged(StagedDriver, ReviewMinor1NoDeadlineAtResume):
    """MINOR-1 on StagedSegmentDriver."""


# --- binding invariant (lead, after the review): every yield phase has a bounded exit under every event ---------

EXIT_PHASES = ('candidate', 'close', 'append', 'full', 'reentry', 'resume',
               # D-MAJOR-1: resume entered on the existing arm, the bulk's admitted moved since that arm
               'resume:append_left', 'resume:append_refused', 'resume:partial_row_closed')
EXIT_EVENTS = ('stop', 'abort', 'reset', 'refusal', 'client_exit', 'none')
RAISES = 'raises'                        # the existing fail-closed rule: ownership retained, DriverError
# phase -> event -> the yield's end reason (None: no yield was started), or RAISES.  A cell whose run does not reach
# its exit within EXIT_STEPS engine steps fails, so a missing exit fails this table.
EXIT_TABLE = {
    'candidate': dict(stop=None, abort=None, reset=None, refusal=None, client_exit=None, none='reopened'),
    'close': dict(stop='stop-shutdown', abort='stop-abort', reset='stop-reset', refusal=RAISES,
                  client_exit='reopened', none='reopened'),
    'append': dict(stop='stop-shutdown', abort='bulk-left', reset='stop-reset', refusal='reopened',
                   client_exit='reopened', none='reopened'),
    'full': dict(stop='stop-shutdown', abort='bulk-left', reset='stop-reset', refusal='handback-expired',
                 client_exit='reopened', none='reopened'),
    'reentry': dict(stop='stop-shutdown', abort='returned-before-native-admission', reset='stop-reset', refusal='reentry-refused',
                    client_exit='reopened', none='reopened'),
    'resume': dict(stop='stop-shutdown', abort='bulk-left', reset='stop-reset', refusal='reentry-refused',
                   client_exit='reopened', none='reopened'),
    'resume:append_left': dict(stop='stop-shutdown', abort='bulk-left', reset='stop-reset', refusal='reentry-refused',
                               client_exit='reopened', none='reopened'),
    'resume:append_refused': dict(stop='stop-shutdown', abort='bulk-left', reset='stop-reset',
                                  refusal='reentry-refused', client_exit='reopened', none='reopened'),
    'resume:partial_row_closed': dict(stop='stop-shutdown', abort='bulk-left', reset='stop-reset',
                                      refusal='reentry-refused', client_exit='reopened', none='reopened'),
}
EXIT_STEPS = 200


class ExitWorld:
    """One rig driven by engine steps only (no clock): auto-arm's tick, the driver tick, worker completions, the
    admission seam when unpaused, and stock's own step on FULL (the append's one-token handback when the hold
    releases it; leaf: the stock scheduler serving an unheld waiting request)."""

    def __init__(self, tc):
        self.tc = tc
        self.rig = Rig(cached={}, max_new=10, handback_steps=6, reentry_retries=3)
        self.append = None
        self.withhold_handback = False
        self.identity_counts = (1, 2, 3, 4)
        self.phase = None

    def step(self):
        rig = self.rig
        with auto_arm_workers(counts=self.identity_counts):
            rig.p.tick()
            rig.d.advance()
            rig.stack.complete_everything()
        step_admit(rig)
        append = self.append
        if (append is not None and not self.withhold_handback and not rig.d._split_ready
                and rig.d._return_pending is None and append.request_id in rig.s.requests
                and append.num_output_tokens == 0 and append.num_computed_tokens == append.num_tokens - 1
                and not sh.defer_stock_positive_frontier(rig.s, append)):
            rig.stock_handback(append)

    def settled(self):
        rig = self.rig
        if self.phase == 'candidate' and not (
                rig.d.append_yield_counters['selected'] > self.selected_before
                or rig.bulk.request_id not in rig.s.requests
                or len(rig.c._committed) > self.committed_before
                or rig.p.phase != 'native'):
            return False                                      # the candidate boundary has not been decided yet
        return (rig.d.append_yield_phase() is None and rig.p.phase in ('idle', 'tail', 'native')
                and rig.p.pending is None and rig.d.stop_kind is None and rig.d._return_pending is None
                and not (rig.p.phase == 'native' and not rig.d._split_ready and not rig.d.active))

    # -- phases --------------------------------------------------------------
    def to_phase(self, phase):
        rig = self.rig
        self.phase = phase
        if phase == 'candidate':
            self.tc.assertIsNotNone(rig.admit())
            rig.drive(lambda: rig.d.tx is not None)
            self.append = rig.add_held(APPEND_UID, 11, cached=0)
            self.selected_before = rig.d.append_yield_counters['selected']
            self.committed_before = len(rig.c._committed)
            return
        if phase.startswith('resume:'):
            path = phase.split(':', 1)[1]
            self.admitted_at_arm = rig.p.admitted_before
            self.append = run_to_yield(rig, append_tokens=11, append_cached=0)
            to_append(rig)
            if path == 'append_left':
                abort(rig, self.append)
                rig.stock_abort(self.append)
            elif path == 'append_refused':
                self.release_append = refuse_allocation(rig, APPEND_UID)
            else:
                self.release_append = refuse_allocation(rig, APPEND_UID, after=1)
                self.tc.assertIsNotNone(rig.admit())
                rig.finish_chunk()
            self.tc.assertGreater(rig.c.admitted, self.admitted_at_arm)
            self.hold_bulk = refuse_bulk_blocks(rig)          # the reopen is refused until the event says otherwise
            for _ in range(EXIT_STEPS):
                if rig.d.append_yield_phase() == 'resume':
                    break
                self.step()
            self.tc.assertEqual(rig.d.append_yield_phase(), 'resume', phase)
            return
        self.append = run_to_yield(rig, append_tokens=11, append_cached=0)
        if phase == 'close':
            return
        to_append(rig)
        if phase == 'append':
            self.tc.assertIsNotNone(rig.admit())
            rig.drive(lambda: rig.d.tx is not None)
            return
        run_append(rig)
        to_full(rig)
        if phase == 'full':
            return
        rig.stock_handback(self.append)
        if phase == 'reentry':
            rig.tick(lambda: rig.p.phase == 'barrier')
            return
        to_resume(rig)

    # -- events --------------------------------------------------------------
    def inject(self, phase, event):
        rig = self.rig
        if phase.startswith('resume:') and event != 'refusal':
            self.hold_bulk()
        if event == 'none':
            return
        if event == 'stop':
            try:
                eh.before_engine_shutdown(rig.core)
            except ed.E3Pending:
                pass
            return
        if event == 'reset':
            try:
                eh.before_reset_prefix(rig.core)
            except ed.E3Pending:
                pass
            return
        if event in ('abort', 'client_exit'):
            target = rig.bulk if event == 'abort' else self.append
            if target.request_id not in rig.s.requests:
                return                                        # the append already left (append_left)
            abort(rig, target)
            rig.stock_abort(target)
            return
        # refusal, by phase
        if phase == 'candidate':
            rig.free_blocks = 0                               # the candidate rule refuses
        elif phase == 'close':
            worker = rig.stack.worker(3)
            import wire_v1
            worker.flash_s5_cancel = lambda message: wire_v1.CancelAck(
                message.transaction, 3, message.state_epoch, 'accepted', (), (), 0)
        elif phase == 'append':
            refuse_allocation(rig, APPEND_UID, after=0)      # the next append chunk is refused
        elif phase == 'full':
            self.withhold_handback = True                     # stock never schedules the handback
        elif phase == 'reentry':
            self.identity_counts = (2, 3)                     # the cohort is never proven
        elif phase == 'resume':
            refuse_bulk_blocks(rig)
        # resume:<path>: the bulk's reopen is already refused (to_phase)

    def run(self, phase, event):
        rig = self.rig
        reasons_before = dict(rig.d.append_yield_end_reasons)
        self.to_phase(phase)
        self.inject(phase, event)
        if event == 'stop':
            settle_stop(rig)
        else:
            for _ in range(EXIT_STEPS):
                if self.settled():
                    break
                self.step()
            else:
                self.tc.fail('no exit within %d steps: phase=%s yield=%r policy=%s' % (
                    EXIT_STEPS, phase, rig.d.append_yield_phase(), rig.p.phase))
        self.tc.assertIsNone(rig.d.append_yield_phase())
        self.tc.assertIsNone(rig.d.error)
        ended = {k: v - reasons_before.get(k, 0) for k, v in rig.d.append_yield_end_reasons.items()
                 if v != reasons_before.get(k, 0)}
        return ended


def exit_cell(tc, phase, event):
    """One cell of EXIT_TABLE: the run reaches its exit within EXIT_STEPS engine steps, with the table's end reason."""
    expected = EXIT_TABLE[phase][event]
    world = ExitWorld(tc)
    if expected is RAISES:
        with tc.assertRaisesRegex(ed.DriverError, 'ownership retained|all-rank'):
            world.run(phase, event)
        return
    ended = world.run(phase, event)
    tc.assertEqual(ended, {} if expected is None else {expected: 1}, (phase, event))


class YieldExitInvariant(unittest.TestCase):
    """Binding (lead, after the review of f76a31de): every yield phase (candidate, close, append, FULL window,
    re-entry, resume) has a bounded exit under every event (stop, abort, reset, refusal, client exit, none), counted
    in engine steps; one table, EXIT_TABLE, so a missing exit fails it."""

    def test_every_phase_has_a_bounded_exit_under_every_event(self):
        self.assertEqual(set(EXIT_TABLE), set(EXIT_PHASES))
        for phase in EXIT_PHASES:
            self.assertEqual(set(EXIT_TABLE[phase]), set(EXIT_EVENTS), phase)
            for event in EXIT_EVENTS:
                with self.subTest(phase=phase, event=event):
                    exit_cell(self, phase, event)

    # A missing exit fails its cell (each mutant removes one exit the review or 2b added, or the reset rule)
    def cell_fails(self, phase, event, *mutation):
        exit_cell(self, phase, event)
        Mutants.fails(self, lambda: exit_cell(self, phase, event), *mutation)

    def test_mutant_append_phase_without_the_refusal_exit(self):
        self.cell_fails('append', 'refusal', ed.NativeSegmentDriver, 'append_yield_append_wait',
                        ("        if schedule_seq <= idle[1]:\n", "        if True:\n"))

    def test_mutant_full_window_without_the_handback_bound(self):
        self.cell_fails('full', 'refusal', ed.NativeSegmentDriver, 'append_yield_handback_step',
                        ("        if y['handback_steps'] < self.controller.append_yield.handback_steps:\n",
                         "        if True:\n"))

    def test_mutant_reentry_without_the_retry_bound(self):
        self.cell_fails('reentry', 'refusal', ed.NativeSegmentDriver, 'append_yield_reentry_refused',
                        ("        if refusals <= self.controller.append_yield.reentry_retries:\n",
                         "        if True:\n"))

    def test_mutant_resume_without_the_retry_bound(self):
        self.cell_fails('resume', 'refusal', ed.NativeSegmentDriver, 'append_yield_reentry_wait',
                        ("        if passes <= self.controller.append_yield.reentry_retries:\n",
                         "        if True:\n"))

    def test_mutant_resume_counts_from_the_last_arm(self):
        for phase in ('resume:append_left', 'resume:append_refused', 'resume:partial_row_closed'):
            with self.subTest(phase=phase):
                self.cell_fails(phase, 'refusal', auto.AutoArmPolicy, '_yield_native',
                                ("            if d.append_yield_reentry_wait(c.admitted, c._schedule_seq):\n",
                                 "            if (c.admitted == self.admitted_before\n"
                                 "                    and d.append_yield_reentry_wait(c.admitted, c._schedule_seq)):\n"))

    def test_mutant_reset_does_not_end_the_yield(self):
        for phase in ('full', 'resume'):
            with self.subTest(phase=phase):
                self.cell_fails(phase, 'reset', ed.NativeSegmentDriver, '_yield_stop',
                                ("            self._end_yield('stop-' + kind)\n", "            pass\n"))

    def test_mutant_client_exit_keeps_the_append_phase(self):
        # the abort rule's FULL window has a backstop (the MAJOR-1 stale guard), so the mutant removes both
        def check():
            with mutant(auto.AutoArmPolicy, '_yield_native',
                        ("        if (phase in ('draining', 'closing', 'append', 'resume') and not d._split_ready and not d.active\n",
                         "        if (False and not d._split_ready and not d.active\n")):
                exit_cell(self, 'append', 'client_exit')
        check()
        Mutants.fails(self, check, ed.NativeSegmentDriver, '_yield_stop',
                      ("            y['phase'] = 'handback'\n", "            pass\n"))


class YieldExitInvariantStaged(StagedDriver, YieldExitInvariant):
    """The exit table on StagedSegmentDriver."""


class HandbackBoundStaged(StagedDriver, HandbackBound):
    """HandbackBound on StagedSegmentDriver."""


class ReentryRefusedStaged(StagedDriver, ReentryRefused):
    """ReentryRefused on StagedSegmentDriver."""


class StopsStaged(StagedDriver, Stops):
    """Stops (client exits included), every case on StagedSegmentDriver."""


class MutantsStaged(StagedDriver, Mutants):
    """Every mutant above, caught again with StagedSegmentDriver as the driver."""


class StagedMutants(unittest.TestCase):
    """B1: mutants of the staged driver's own overrides the yield crosses fail the staged cycle; the same mutant
    leaves the native cycle green (so it is the staged parametrization that catches it)."""

    def fails(self, check, *mutation):
        with mutant(*mutation), self.assertRaises(MUTANT_FAILURES) as caught:
            check()
        MUTANT_LOG.append((self.id().rsplit('.', 1)[-1], type(caught.exception).__name__,
                           str(caught.exception).splitlines()[0][:150] if str(caught.exception) else ''))

    def staged(self):
        with mock.patch.object(Rig, 'DRIVER', 'staged'):
            connected_cycle(self)

    def native_passes(self, *mutation):
        with mutant(*mutation), mock.patch.object(Rig, 'DRIVER', 'native'):
            connected_cycle(self)

    # the committed frame stays in _frames: outside a scope the driver still shows the bulk's chunk, so the
    # boundary the yield needs (no live frame, tx None) never appears
    def test_reset_keeps_the_committed_frame(self):
        edit = (StagedSegmentDriver, '_reset',
                ("            self._frames = [item for item in self._frames if item is not frame]\n", "            pass\n"))
        self.native_passes(*edit)
        self.fails(self.staged, *edit)

    # a stage's local release never recorded: the bulk's chunk never commits, so no boundary is offered
    def test_local_release_not_recorded(self):
        edit = (StagedSegmentDriver, '_apply_local_release', ("            frame.released.add(stage)\n", ""))
        self.native_passes(*edit)
        self.fails(self.staged, *edit)


# --- RealBuilderState -------------------------------------------------------------------------------------------

MAMBA_BLOCK = 4                          # lc.fake_runner: _get_mamba_group_info -> block_size 4
REFERENCE = ROOT / 'reference/vllm-src/v1/worker'


class AlignWorld:
    """The real LaneChunkFactory / LaneAlign / LaneTables over test_e3_lane_chunk's reviewed fake runner.

    Leaves (each the pinned arithmetic, not a model of it; append_yield_fixture_leaves.json):
      align_preprocess   mamba_utils.preprocess_mamba_align_fused_kernel (:534-548), one request
      run_fused_precopy  mamba_utils.precopy_mamba_align_fused_kernel: copy column src_col -> state_idx when
                         src_col >= 0 and they differ (:604-610)
      postprocess        mamba_utils.postprocess_mamba_fused_kernel decision under PRECOMPUTED_NEW_COMPUTED
      add_request        mamba_hybrid.MambaHybridModelState.add_request (:115-123): accepted = 1,
                         state_idx = (num_computed - 1) // block_size
      req_states         stock RequestState.add_request writes num_computed_tokens; remove_request frees the slot
      forward            the GDN layers read the running state at column state_idx of the lane's table row and
                         write the state at chunk_end there; a block's state is the token count folded into it.
    The physical state store is keyed by block id, so a block shared or rewritten by another request shows up as a
    wrong read.
    """

    def __init__(self):
        lc.FakeMambaContext.instances = []
        self.torch = lc.fake_torch()
        self.calls = []
        runner = lc.fake_runner(self.torch, self.calls)
        states = runner.req_states
        computed = states.num_computed_tokens.gpu

        def add_request(req_id, prompt_len, all_token_ids, num_computed_tokens, max_tokens):
            index = states.free.pop()
            states.req_id_to_index[req_id] = index
            computed.values[index] = num_computed_tokens
        states.add_request = add_request
        states.num_computed_tokens.stage_write_elem = lambda i, v: computed.values.__setitem__(i, v)

        def remove_request(req_id):
            states.free.append(states.req_id_to_index.pop(req_id))
        runner._remove_request = remove_request
        model = runner.model_state

        def model_add(index, data):
            model.num_accepted_tokens_gpu.values[index] = 1
            model._mamba_state_idx_gpu.values[index] = (data.num_computed_tokens - 1) // MAMBA_BLOCK
        model.add_request = model_add
        for table in runner.block_tables.block_tables:      # room for a 20-token bulk (5 blocks, x2 kernel blocks)
            table.gpu.shape = (8, 16)
        ports = lc.fake_ports(self.calls)
        ports.align_preprocess = self._preprocess
        self.runner, self.ports = runner, ports
        self.factory = LaneChunkFactory(runner, ports=ports, torch_module=self.torch)
        self.ctx = lc.FakeMambaContext.instances[-1]
        self.ctx.run_fused_precopy = self._precopy
        self.ctx.run_fused_postprocess_align = self._postprocess
        self.phys = {}                   # block id -> tokens folded into the state stored there
        self.decisions = []              # per chunk: (uid, start, end, src_col, state_idx, src_off, slot)
        self.reads = []                  # per chunk: (uid, start, state read)
        self._last = None

    def _row(self):
        return self.ctx.tables[0].values  # the lane table the align context is bound to (mamba group 0, row 0)

    def _preprocess(self, idx_mapping, state_idx, num_computed, query_start_loc, accepted, src_col, src_off,
                    num_reqs, block_size):
        assert num_reqs == 1
        req = idx_mapping.values[0]
        old, n_acc = state_idx.values[req], accepted.values[req]
        src_off.values[req] = max(n_acc - 1, 0)
        src_col.values[req] = old
        after = num_computed.values[req] + query_start_loc.values[1] - query_start_loc.values[0]
        new = (after + block_size - 1) // block_size - 1
        state_idx.values[req] = new
        if old >= 0 and old != new:
            accepted.values[req] = 1
        self._last = (src_col.values[req], new, src_off.values[req], req)

    def _precopy(self, n, state_idx, src_col, src_off, idx):
        req = idx.values[0]
        src, dst = src_col.values[req], state_idx.values[req]
        if src < 0 or src == dst:
            return
        row = self._row()
        self.phys[row[dst]] = self.phys.get(row[src], 0)

    def _postprocess(self, n, accepted, state_idx, new_computed, idx):
        req = idx.values[0]
        new = new_computed.values[req]
        running = new - accepted.values[req] + 1
        aligned = (new // MAMBA_BLOCK) * MAMBA_BLOCK
        if aligned < running:
            return
        src, dest = state_idx.values[req], aligned // MAMBA_BLOCK - 1
        if src == dest:
            accepted.values[req] = 1
            return
        row = self._row()
        self.phys[row[dest]] = self.phys.get(row[src], 0)

    def run_chunk(self, uid, generation, start, end, prompt_end, blocks, txn_seq):
        """One admitted chunk through the real factory: admit (register/update), forward, finish, complete."""
        width = (end + MAMBA_BLOCK - 1) // MAMBA_BLOCK
        tables = (tuple(blocks[:width]), tuple(b + 100 for b in blocks[:width]), ())
        admission = ChunkAdmission(
            transaction=TransactionId(engine_epoch=1, request_uid=uid, request_generation=generation,
                                      chunk_start=start, chunk_end=end, txn_seq=txn_seq),
            prompt_end=prompt_end, input_ids=tuple(range(start, end)), ngram_context=(1, 2),
            block_tables=tables, new_block_ids_to_zero=(), cow_pairs=(), state_epoch=1)
        chunk = self.factory.admit(admission)
        src, new, off, slot = self._last
        self.decisions.append((uid, start, end, src, new, off))
        block = self._row()[new]
        self.reads.append((uid, start, self.phys.get(block, 0)))
        self.phys[block] = end                                   # the chunk's state writers
        self.slots = getattr(self, 'slots', []) + [(uid, slot)]
        chunk.finish_state()
        self.factory.complete_chunk(chunk, consumers_complete=True)
        return chunk

    def release(self):
        return self.factory.release()


BULK_BLOCKS = (10, 11, 12, 13, 14)
APPEND_BLOCKS = (30, 31, 32)


def bulk_run(plan, *, yield_after=None, append=None, append_blocks=APPEND_BLOCKS, reopen_start=None):
    """The bulk's chunks through one worker's factory; optionally closed after chunk ``yield_after`` (the row
    release the yield's CancelRequest performs), one append request run in between, then reopened as a fresh
    registration of a new generation from ``reopen_start`` (default: the committed frontier)."""
    world = AlignWorld()
    prompt_end = plan[-1][1] + 1
    generation, seq = 1, 0
    for index, (start, end) in enumerate(plan):
        if yield_after is not None and index == yield_after + 1:
            world.release()
            if append is not None:
                a_start, a_end = append
                for chunk_start in range(a_start, a_end, 4):
                    seq += 1
                    world.run_chunk('append', 1, chunk_start, min(chunk_start + 4, a_end), a_end + 1,
                                    append_blocks, seq)
                world.release()
            generation += 1
            if reopen_start is not None:
                start = reopen_start
        seq += 1
        world.run_chunk('bulk', generation, start, end, prompt_end, BULK_BLOCKS, seq)
    return world


PLANS = {
    'crossing': ((0, 6), (6, 13), (13, 19)),     # non-final ends inside Mamba blocks, each chunk crossing one
    'aligned': ((0, 8), (8, 16), (16, 19)),      # non-final ends on Mamba block boundaries
    'mixed': ((0, 4), (4, 7), (7, 12), (12, 19)),
}


class RealBuilderState(unittest.TestCase):
    """Codex 4: the reopened bulk sees exactly the state an uninterrupted bulk sees, through the real builder."""

    def bulk_view(self, world):
        return [d for d in world.decisions if d[0] == 'bulk'], [r for r in world.reads if r[0] == 'bulk']

    def test_every_boundary_of_every_plan_matches_the_uninterrupted_bulk(self):
        for name, plan in PLANS.items():
            reference = bulk_run(plan)
            ref_decisions, ref_reads = self.bulk_view(reference)
            self.assertEqual([r[2] for r in ref_reads], [s for s, _ in plan], name)
            for boundary in range(len(plan) - 1):
                for append in (None, (8, 11), (0, 9)):
                    with self.subTest(plan=name, boundary=boundary, append=append):
                        world = bulk_run(plan, yield_after=boundary, append=append)
                        decisions, reads = self.bulk_view(world)
                        self.assertEqual(decisions, ref_decisions)
                        self.assertEqual(reads, ref_reads)
                        self.assertTrue(all(read == start for _, start, read in reads))

    def test_row_slot_reuse_is_reset_by_the_stock_registration(self):
        world = bulk_run(PLANS['crossing'], yield_after=0, append=(0, 9))
        slots = world.slots
        bulk_first, append_first = slots[0][1], next(s for u, s in slots if u == 'append')
        reopened = [s for u, s in slots if u == 'bulk'][1]
        self.assertEqual(bulk_first, append_first, 'the append reuses the slot the closed bulk row freed')
        self.assertEqual(reopened, bulk_first, 'the reopened bulk reuses it again')
        # the append left its own state index and accepted count in that slot; registration reseeded both
        append_decisions = [d for d in world.decisions if d[0] == 'append']
        self.assertNotEqual(append_decisions[-1][4], (6 - 1) // MAMBA_BLOCK)
        self.assertEqual(self.bulk_view(world), self.bulk_view(bulk_run(PLANS['crossing'])))

    def test_mutant_append_on_a_bulk_block_corrupts_the_reopened_state(self):
        """Codex 2 at the builder: an append writing a block the bulk holds is a wrong read, not a silent pass."""
        reference = self.bulk_view(bulk_run(PLANS['crossing']))
        world = bulk_run(PLANS['crossing'], yield_after=0, append=(0, 9), append_blocks=BULK_BLOCKS)
        self.assertNotEqual(self.bulk_view(world)[1], reference[1])

    def test_mutant_reopen_away_from_the_frontier_reads_a_wrong_state(self):
        """Codex 3 at the builder: a reopen from a fresh lookup (start 4 instead of the frontier 6) is caught."""
        world = bulk_run(PLANS['crossing'], yield_after=0, append=(8, 11), reopen_start=4)
        reads = self.bulk_view(world)[1]
        self.assertFalse(all(read == start for _, start, read in reads))



if __name__ == '__main__':
    unittest.main()
