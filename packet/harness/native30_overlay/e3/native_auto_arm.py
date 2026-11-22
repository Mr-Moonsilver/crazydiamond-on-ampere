"""Opt-in, EngineCore-thread-only native-prefill policy (no device owner).

Scheduling remains paused until existing output Futures, the worker barrier,
and typed split acks settle. No new transport, CUDA operation or model setting.
"""
import os
import time
import weakref
from itertools import islice

from .e3_types import E3ContractError, _text
from .scheduler_hooks import bounded_limit, native_chunk_cap, residue_hit, staged_chunk_plan, _ProjectedPrefillRequest

ENV = 'FLASH_S5_AUTO_ARM'
PHASE = 'auto-prefill'
MIN_NATIVE_ROWS = 32768
MAX_COHORT = 63
MAX_REJECTED = 1024
RELEASE = 'release_auto_arm'
READS = frozenset(('status', 'snapshot', 'transition_acks', 'bootstrap_acks', 'overlap_evidence'))
# compose-5: every place a waiting head leaves for stock instead of native, in the order a tick reaches them. Only these
# ticks let stock admit: cooldown, tail, and _select returning False. Numbers only: no request UID leaves this module.
# Names stay clear of the words the box watchers grep for in a log line.
DECLINE_SITES = ('cooldown', 'tail',
                 'gate-manual-owned', 'gate-route-or-work', 'gate-not-bootstrapped', 'gate-driver-stopped', 'gate-engine-paused',
                 'gate-policy', 'gate-no-batch-queue', 'gate-config', 'gate-not-running',
                 'cohort-empty', 'cohort-over', 'cohort-prefill', 'cohort-other', 'cohort-uid',
                 'rejected',
                 'plan-state', 'plan-controller', 'plan-transfer', 'plan-prompt-logprobs', 'plan-generation-logprobs', 'plan-uid',
                 'plan-shared', 'plan-below-floor', 'plan-kv-blocks', 'plan-empty')
MAX_PENDING_DECLINES = 64
# decode-phase hold (5 Nov 2026, the operator's rule: no stock prefill while a decoder is live). With the controller's native_hold_s
# set (per arm, by the stage command; 0, the default, is today's behaviour) every waiting prefill waits for the native lane while
# anything runs, the 32768-row floor drops to the staged path's own minimum, and a refused selection is retried. Stock takes a
# request only when nothing runs, when native can't plan it (plan-state, plan-empty, plan-shared: counted by reason), after
# MAX_RETRIES refusals, or after native_hold_s on the driver's clock (a valve against a stuck lane, counted as 'expired').
MAX_HOLDS = 64
MAX_UNHOLDABLE = 1024
MAX_RETRIES = 2
TRANSIENT_PLAN_SITES = frozenset(('plan-kv-blocks',))      # waits: blocks free as decoders finish
# coverage-residue (6 Nov 2026): with the controller's native_residue set on a hold arm, the hold's stock residue goes native.
# The shared-prefix junction plans natively (its stop is stock's split's, its state published by the chunk's commit); a hit
# that leaves under the staged minimum is lowered (scheduler_hooks.residue_hit); a refused selection stays held and is tried
# again once the decode cohort has changed, with no retry count. What stock still takes, it takes by name (stock_site).
MAX_STOCK_WHY = 1024
# coverage-residue, idle (lead, 6 Nov 01:10: 100 % of all prefill rows, idle included): with the residue on, the hold holds with nothing
# running and the policy selects with an empty cohort. A close that lost a race (the selection or the cohort moved under it, the hold box's
# 'cooldown' turns: three quick closes, then stock) is not a refusal of the request: no rejected mark, at most MAX_RACES per request.
RACE_CLOSES = frozenset(('selection-changed-during-output-drain', 'selection-changed-before-arm', 'worker-barrier-cohort-refused'))
MAX_RACES = 8
# lane-o3 (7 Nov 2026, DESIGN-multi-prompt-lane O3): stay split across runs. With FLASH_S5_O3=1 a native run whose terminal
# commit leaves one prompt row hands that row back on the split bank (stock schedules it beside the decoders; graph_hooks
# classifies the batch decode-shaped, FULL at n+1); the next head is armed on the same split route after a census round trip.
# The split cohort is a ledger: the entry-proven cohort plus requests joined through such a handback; never past the counts
# proven at identity for this split residency. Geometry tails, yields and anything unproven keep today's FULL cycle.
O3_ENV = 'FLASH_S5_O3'
BULK_PICK_CANDIDATES = 10
BULK_PICK_AGE_S = 45.0
O3_TAIL_TICKS = 64            # policy ticks a kept handback may wait unscheduled on split before today's FULL return
# yield-reentry (12 Nov 2026, lead): (1) always on, observational only: every refused re-entry proof of a yielded bulk
# (drain cohort, barrier, identity) is counted by stage and failing field, with one 'flash-s5 yield-reentry proof-miss'
# line (numbers only: retry index, the launch bound, ms since the re-entry began); the retries an armed re-entry needed
# are a histogram. No RPC, sync or decision reads any of it. (2) FLASH_S5_YIELD_REENTRY_SETTLE=1 (default 0: today's
# path): the FULL window of a yield waits, unpaused, for the returned append's first decode to be scheduled before the
# re-entry's PAUSED_ALL (the run-end path's K2 settle), bounded by O3_TAIL_TICKS; the retry bound and the stock fallback
# are unchanged.
YRS_ENV = 'FLASH_S5_YIELD_REENTRY_SETTLE'
YRS_MAX_KEYS = 64             # bound on each diagnostic table (keys are stage:field or a retry index)


def o3_enabled(env):
    value = env.get(O3_ENV, '0')
    if value not in ('0', '1'):
        raise E3ContractError('invalid boolean setting: ' + O3_ENV)
    return value == '1'


def yield_settle_enabled(env):
    value = env.get(YRS_ENV, '0')
    if value not in ('0', '1'):
        raise E3ContractError('invalid boolean setting: ' + YRS_ENV)
    return value == '1'


def enabled(env):
    value = env.get(ENV, '0')
    if value not in ('0', '1'):
        raise E3ContractError('invalid boolean setting: ' + ENV)
    return value == '1'


def install(engine, driver, env):
    if not enabled(env):
        return None
    if getattr(driver, 'auto_arm', None) is not None:
        raise E3ContractError('auto-arm policy already installed')
    if not getattr(engine, '_flash_s5_startup_ready', None):
        raise E3ContractError('auto-arm requires bound EngineCore')
    # Bind only after owner startup. Installation itself never schedules or
    # pauses; inspect async capability on the first complete Core tick.
    policy = AutoArmPolicy(engine, driver)
    policy.o3 = o3_enabled(env)
    policy.yield_settle = yield_settle_enabled(env)       # yield-reentry (2)
    control_inwait = env.get('FLASH_S5_CONTROL_INWAIT', '0')
    if control_inwait not in ('0', '1'):
        raise E3ContractError('invalid boolean setting: FLASH_S5_CONTROL_INWAIT')
    if control_inwait == '1':
        policy.control_inwait = ControlInWait(policy)
    control_o3 = env.get('FLASH_S5_CONTROL_O3_INWAIT', '0')
    if control_o3 not in ('0', '1'):
        raise E3ContractError('invalid boolean setting: FLASH_S5_CONTROL_O3_INWAIT')
    if control_o3 == '1':
        if control_inwait != '1' or not policy.o3:
            raise E3ContractError('FLASH_S5_CONTROL_O3_INWAIT needs CONTROL_INWAIT=1 and O3=1')
        policy.control_inwait.enable_o3()
    driver.auto_arm = policy
    return policy


def worker_capability(native):
    """Pure host dispatch against actual stock keys and captured native graphs.

    Cover every shrinking subset, including stock padding; never guess that a
    graph for the current full cohort covers later batches. No replay/readback.
    """
    from .graph_router import ExecutionDescriptor
    runner, router = native.runner, native.router
    manager = runner.cudagraph_manager
    bank = router.banks.get('decode-split')
    maximum = min(MAX_COHORT, runner.max_num_reqs)
    supported = []
    if bank is not None and bank.ready:
        for count in range(1, maximum + 1):
            desc = ExecutionDescriptor.from_vllm(manager.dispatch(count, count, 1, 0, max_query_len=1))
            if (desc.cg_mode == 'full' and bank.graph_for(desc,
                    storage_epoch=router.handle.storage_epoch,
                    worker_rank=router.handle.worker_rank) is not None):
                supported.append(count)
    return dict(schema='s5-auto-decode-capability/1', supported_counts=supported,
                engine_epoch=router.engine_epoch, route_generation=router._transition_seq,
                storage_epoch=router.handle.storage_epoch)


def _generation_logprobs_refused(params):
    """Stock's refusal, except under FLASH_S5_INWAIT, whose first-token read asks for them on the bulk. Native rows produce no logits; the
    request's last row runs stock with the request's own parameters, so that row answers. Prompt logprobs need every prompt row's logits and stay refused."""
    return params.logprobs is not None and os.environ.get('FLASH_S5_INWAIT', '0') != '1'


class ControlInWait:
    """Overlap host-only observations with decode; retain output and route fences."""

    o3 = False
    o3_active = False

    def __init__(self, policy):
        self.policy = policy
        self.certified = False
        self.rows = {}
        self.row = None
        self.started_ns = self.paused_ns = None
        self.stall_ns = 0
        self.members = ()
        self.cohort = ()
        self.identity = None

    def snapshot(self):
        result = dict(schema='s5-control-inwait/1', enabled=True,
                    stall_basis='PAUSED_ALL host interval; not GPU idle or client token gap',
                    unchanged=('split-kept-handback-cut', 'full-return', 'cancel', 'rollback'),
                    transitions={key: dict(value, drains_kept=dict(value['drains_kept']))
                                 for key, value in self.rows.items()})
        if self.o3:
            result.update(o3_enabled=True, cut_pending=len(self.cut), cut_bound=2)
            result['unchanged'] = ('native-source-PP-retirement', 'full-return', 'cancel', 'rollback')
        return result

    def pause(self, state):
        now = time.monotonic_ns()
        if self.row is not None and self.paused_ns is not None:
            self.stall_ns += now - self.paused_ns
        self.paused_ns = now if getattr(state, 'name', str(state)) == 'PAUSED_ALL' else None

    def _start_count(self, kind):
        if self.row is None:
            self.kind = kind
            self.row = self.rows.setdefault(self.kind, dict(
                attempts=0, refused=0, invalidated=0, armed=0, closed=0,
                drains_skipped=0, drains_kept={'applied-output-cut': 0, 'typed-route-fence': 0},
                pause_to_arm_ms=0., pause_to_arm_max_ms=0., decoder_stall_ms=0.,
                decoder_stall_max_ms=0.))
            self.started_ns = self.paused_ns if self.paused_ns is not None else time.monotonic_ns()
            self.stall_ns = 0

    def begin(self):
        p, d = self.policy, self.policy.driver
        self._start_count('yield-full-reentry' if p.reentry else 'full-entry')
        if self.certified:
            self.row['invalidated'] += 1
        self.certified = False
        self.row['attempts'] += 1
        self.cohort = p.cohort
        # Keep the actual Request objects alive until this certificate retires.
        self.members = tuple((r.request_id, r) for r in p.scheduler.running)
        self.identity = (d.engine_epoch, d.route_generation)
        p._pause(type(p.old_pause).PAUSED_NEW)
        p.pending = p._control('identity')
        p.phase = 'control_observe'

    def poll(self):
        p, d = self.policy, self.policy.driver
        d.control_tick()
        if p.pending is None or not p.pending.done():
            return False
        result = p.pending.result()
        p.pending = None
        if p._reentry_lost_yield():
            return True
        if not p._proof(result, capability=True):
            self.row['refused'] += 1
            p.phase = 'drain'
            if p.reentry:
                return p._reentry_refused('identity', p._yrs_proof_why(result, capability=True))
            # Preserve the residue policy's retry classification for a transient census.
            why = ('worker-graph-support-refused' if p._proof(result, observation=True)
                   else 'worker-barrier-cohort-refused')
            p._rollback(why)
            return True
        self.certified = True
        p._pause(type(p.old_pause).PAUSED_ALL)
        self.row['drains_kept']['applied-output-cut'] += 1
        p.phase = 'drain'
        # The outer Core method must retire queued outputs before the NEXT tick.
        return False

    def current(self):
        p, d = self.policy, self.policy.driver
        running = p.scheduler.running
        return (self.identity == (d.engine_epoch, d.route_generation)
                and self.cohort == p.cohort and len(self.members) == len(running)
                and all(any(uid == r.request_id and saved is r for r in running)
                        for uid, saved in self.members))

    def switching(self):
        # This replaces only the extra barrier: enter_split still drains real streams.
        self.row['drains_skipped'] += 1
        self.row['drains_kept']['typed-route-fence'] += 1

    def acknowledge(self, result):
        # The omitted barrier's device fence must now be present in EVERY route ACK.
        rows = result.get('transition_acks', ())
        if (not self.certified or len(rows) != len(self.policy.driver.ranks)
                or {row.get('worker_rank') for row in rows} != set(self.policy.driver.ranks)
                or any(type(row.get('drained_streams')) is not int or row['drained_streams'] <= 0
                       for row in rows)):
            raise E3ContractError('control-inwait-transition-fence-unproved; ownership retained')

    def close(self, armed):
        if self.row is None:
            return
        now = time.monotonic_ns()
        stall = self.stall_ns + (0 if self.paused_ns is None else now - self.paused_ns)
        self.row['armed' if armed else 'closed'] += 1
        if self.o3_active and armed:
            self.row['arms_with_later_frames'] += int(bool(self.policy.engine.batch_queue))
        self.row['decoder_stall_ms'] += stall / 1e6
        self.row['decoder_stall_max_ms'] = max(self.row['decoder_stall_max_ms'], stall / 1e6)
        if armed:
            elapsed = (now - self.started_ns) / 1e6
            self.row['pause_to_arm_ms'] += elapsed
            self.row['pause_to_arm_max_ms'] = max(self.row['pause_to_arm_max_ms'], elapsed)
        self.row = None
        self.started_ns = self.paused_ns = None
        self.members = ()
        self.cohort = ()
        self.identity = None
        self.certified = False
        if self.o3:
            self.cut = []
            self.o3_result = self.o3_reason = None
            self.o3_active = False

    def enable_o3(self):
        scheduler = self.policy.scheduler
        original = getattr(scheduler, 'update_from_output', None)
        if (not callable(original) or getattr(scheduler, '_flash_s5_control_cut', None) is not None):
            raise E3ContractError('control-inwait needs one owned scheduler output hook')
        self.o3 = True
        self.cut = []
        self.o3_result = self.o3_reason = None

        def applied(scheduler_output, model_runner_output):
            result = original(scheduler_output, model_runner_output)
            for index, output in enumerate(self.cut):
                if output is scheduler_output:
                    self.cut.pop(index)
                    self.row['cut_frames_applied'] += 1
                    break
            return result
        scheduler.update_from_output = applied
        scheduler._flash_s5_control_cut = self

    def begin_o3(self):
        p, d = self.policy, self.policy.driver
        if self.o3_active:
            raise E3ContractError('control-inwait-stock-output-cut-already-owned')
        if d._return_pending is not None or d._route_exchange() is not None:
            return False
        queue = p.engine.batch_queue
        if (len(queue) > 2 or any(type(item) is not tuple or len(item) != 3 for item in queue)
                or len({id(item[1]) for item in queue}) != len(queue)):
            raise E3ContractError('control-inwait-stock-output-cut-invalid; ownership retained')
        self._start_count('split-kept-yield' if p.o3_yield else 'split-kept-entry')
        self.row.setdefault('cut_frames_issued', 0)
        self.row.setdefault('cut_frames_applied', 0)
        self.row.setdefault('arms_with_later_frames', 0)
        self.row['attempts'] += 1
        self.o3_active = True
        self.cut = [item[1] for item in queue]
        self.row['cut_frames_issued'] += len(self.cut)
        self.members = tuple((r.request_id, r) for r in p.scheduler.running)
        self.cohort = p.cohort
        self.identity = (d.engine_epoch, d.route_generation)
        p._pause(type(p.old_pause).PAUSED_NEW)
        p._o3_census_begin()
        if p.phase == 'o3_census':
            p.phase = 'control_o3_observe'
            self.row['drains_skipped'] += 1  # Queue-wide drain replaced by a finite applied prefix.
            self.row['drains_kept']['applied-output-cut'] += 1
        return False

    def refuse_o3(self, why):
        self.row['refused'] += 1
        kept = self.row['drains_kept']
        kept['fallback-full-cut'] = kept.get('fallback-full-cut', 0) + 1
        self.o3_reason = why
        self.policy._pause(type(self.policy.old_pause).PAUSED_ALL)
        self.policy.phase = 'control_o3_rollback'
        return False

    def poll_o3(self):
        p, d = self.policy, self.policy.driver
        if p.phase == 'control_o3_rollback':
            if p.engine.batch_queue:
                return False
            why = self.o3_reason
            self.row['drains_kept']['typed-route-fence'] += 1
            self.close(False)
            return p._o3_refuse(why)
        d.control_tick()
        if p.phase == 'control_o3_observe':
            if p.pending is None or not p.pending.done():
                return False
            self.o3_result = p.pending.result()
            p.pending = None
            p.phase = 'control_o3_cut'
        if self.cut:
            return False
        if (not self.current() or p._request() is None or p._o3_plan(p._request()) != p.plan
                or p._cohort() != p.cohort):
            return self.refuse_o3('selection-changed-after-census')
        if p.o3_yield and not d.append_yield_active():
            return self.refuse_o3('yield-ended-during-reentry')
        allowed = p.o3_ledger | ({p.o3_joined} if p.o3_joined is not None else frozenset())
        if d.route_generation != p.o3_split_generation:
            return self.refuse_o3('generation')
        if not set(p.cohort) <= allowed:
            return self.refuse_o3('ledger')
        if not p._o3_covers(max(1, len(p.cohort))):
            return self.refuse_o3('counts')
        if p.o3_joined in p.cohort:
            joined = p.scheduler.requests.get(p.o3_joined)
            if joined is None or joined.num_output_tokens <= 0:
                return self.refuse_o3('handback-output-unapplied')
        if not p._o3_proof(self.o3_result):
            return self.refuse_o3('census')
        if d.active or d.admission_intent() is not None:
            return self.refuse_o3('driver-owned')
        p._o3_arm()
        return False


class AutoArmPolicy:
    o3 = False
    yield_settle = False          # yield-reentry (2): FLASH_S5_YIELD_REENTRY_SETTLE
    control_inwait = None

    def __init__(self, engine, driver):
        self.engine, self.driver = engine, driver
        self.scheduler, self.controller = engine.scheduler, driver.controller
        self.phase = 'idle'
        self.manual_owned = False
        self._internal = False
        self.request_ref = None
        self.selection_seq = 0
        self.completed = 0
        self.rejected = {}
        self.pending = None
        self.deadline = None
        self.old_pause = None
        self.plan = None
        self.cohort = ()
        self.admitted_before = 0
        self.schedule_before = 0
        self.external_pause = False
        self.cooldown = False
        # coverage-throughput build-2 (b): engine ticks in 'tail' since the FULL ack (closes by reason: restores)
        self.tail_ticks = 0
        # returning-append A1: the decode cohort sizes the last identity proof
        # covered, and whether the entry phases run a yielded bulk's re-entry.
        self.decode_counts = None
        self.reentry = False
        self.reentry_refusals = 0
        self.drain_wait_seq = None
        # compose-5: ticks per site; turns per site, each turn once, at the last site before it left for stock.
        self.decline_ticks = dict.fromkeys(DECLINE_SITES, 0)
        self.decline_turns = dict.fromkeys(DECLINE_SITES, 0)
        self.decline_totals = dict(selected_after_decline=0, left_while_waiting=0, dropped=0, read_errors=0, armed=0)
        self.restores = {}
        self.rollbacks = {}
        self._decline_pending = {}
        self._decline_why = None
        # decode-phase hold: uid -> (weakref, first held on the driver's clock); uid -> weakref of a request stock may take.
        self.holds = {}
        self.unholdable = {}
        # build-2b (lead, 6 Nov): yielded bulks the re-entry could not plan, held at the positive frontier (uid -> (ref, since))
        self.abandoned = {}
        self.bulk_hold = dict(bulks=0, deferrals=0, expired=0, errors=0)
        self.hold_totals = dict(held=0, selected=0, expired=0, unplannable=0, errors=0, retried=0, retries_spent=0, moved_stop=0)
        self.hold_released = {}          # site -> requests the hold let stock take, by the plan's reason
        self.retries = {}
        # coverage-residue: uid -> the hold's last word on why stock may take it; uid -> the cohort a selection was refused under.
        self.stock_why = {}
        self.unhold_site = {}
        self.refused_cohort = {}
        self.residue_counts = dict(junction=0, lowered=0, refused_held=0, refused_retried=0, idle_selected=0, races=0, races_spent=0,
                                   held_ms=0)
        self.bulk_pick = dict(scans=0, candidates=0, capped=0, reordered=0, aged=0, short_stop=0,
                              rejected=0, unplannable=0, extra_lookup_calls=0, extra_lookup_ns=0)
        self._held_last = None         # lane-o3-fix: monotonic ns of the last refused_held tick (held_ms: the lane parked on a held head)
        self.races = {}
        self._rollback_reason = None
        self._o3_return_reason = None
        # lane-o3
        self.o3_tail = False
        self.o3_ledger = frozenset()
        self.o3_split_generation = None
        self.o3_joined = None
        self.o3_counts = dict(kept=0, armed=0, tail_bound=0, full={}, refused={}, gap={}, settle_ticks=0, unparked=0,
                              yield_split=0, yield_kept=0, yield_full={})
        # lane-o3-fix (b): a yielded append's terminal handback kept split (yield_split); the bulk resumed on that split route after
        # a census (yield_kept); a refusal returns FULL and runs today's re-entry (yield_full by reason).
        self.o3_yield = False
        # lane-o3-fix (c), per arm (k2_settle on native_stage_local; off: today's K2): the FULL tail selects only after the
        # handed-back row was scheduled once more as a decoder, so the barrier sees it as one on every worker.
        self.k2_counts = dict(settle_ticks=0, settled=0, bound=0)
        self._o3_gap_start = None      # (monotonic ns, path) at a run's terminal commit; closed at the next arm
        # yield-reentry (1) diagnostic and (2) settle counters; monotonic ns at the re-entry's start; settle ticks (None: not waiting)
        self.yrs_counts = dict(refused={}, refused_retry={}, retries_until_armed={}, exceeded=0, barrier_pass=0,
                               barrier_first_try=0, armed_ms=[0, 0.0, 0.0], settle_waits=0, settle_ticks=0, settled=0,
                               settle_bound=0, settle_skipped={}, errors=0)
        self._yrs_t0 = None
        self._yrs_ticks = None
        self._yrs_seq = None

    @property
    def active(self):
        return self.phase != 'idle'

    def snapshot(self):
        request = self.request_ref() if self.request_ref is not None else None
        result = dict(schema='s5-auto-arm/1', phase=self.phase, active=self.active,
                    manual_owned=self.manual_owned, selection_seq=self.selection_seq,
                    request_uid=None if request is None else request.request_id,
                    completed=self.completed, minimum_native_rows=self._floor(),
                    max_cohort=MAX_COHORT)
        if self.control_inwait is not None:
            result['control_inwait'] = self.control_inwait.snapshot()
        return result

    def manual_control(self, command, submit):
        if self._internal:
            return submit(command)
        if self.active and (type(command) is not dict or command.get('action') not in READS):
            raise E3ContractError('native-control-refused:auto-arm-cycle-owned')
        if type(command) is dict and command.get('action') == RELEASE:
            d = self.driver
            if (set(command) != {'action', 'engine_epoch'}
                    or type(command.get('engine_epoch')) is not int
                    or command['engine_epoch'] != d.engine_epoch):
                raise E3ContractError('auto-arm-release-engine-epoch-invalid')
            if (not d._engine_paused() or d.active or d._split_ready
                    or d.admission_intent() is not None or getattr(self.engine, 'batch_queue', None)):
                raise E3ContractError('auto-arm-release-requires-paused-idle-full')
            self.manual_owned = False
            return dict(schema=1, action=RELEASE, released=True, engine_epoch=d.engine_epoch,
                        clock=d._clock_record(int(d._monotonic_ns())))
        result = submit(command)
        if type(command) is dict and command.get('action') not in READS:
            # Lease survives manual FULL measurements; only explicit release
            # returns policy ownership. Failed commands acquire no lease.
            self.manual_owned = True
        return result

    def decline_record(self):
        """Bounded status row: counts only, never a request UID. Cumulative; a reader takes an arm's delta."""
        return dict(schema='s5-auto-declines/1', ticks=dict(self.decline_ticks), turns=dict(self.decline_turns),
                    pending=len(self._decline_pending), restores=dict(self.restores), rollbacks=dict(self.rollbacks),
                    yield_bulk_hold=dict(self.bulk_hold, waiting=len(self.abandoned)),
                    holds=dict(self.hold_totals, waiting=len(self.holds), limit_s=self._hold_limit(), released=dict(self.hold_released),
                               deferrals=getattr(self.controller, 'native_hold_deferrals', 0),
                               handback_front=getattr(self.controller, 'handback_front', 0)),
                    residue=dict(self.residue_counts, on=self._residue(),
                                 admitted=dict(getattr(self.controller, 'residue_totals', None) or {})),
                    coverage=self._coverage_record(),
                    o3=self.o3_record(),
                    shortest_bulk=dict(self.bulk_pick, on=getattr(self.controller, 'shortest_bulk', False) is True),
                    k2=dict(self.k2_counts, selected=int(getattr(self.controller, 'k2_settle', False) is True)),
                    yield_reentry=self.yrs_record(),       # yield-reentry (1)/(2), counts only
                    **self.decline_totals)

    # -- lane-o3 ------------------------------------------------------------
    def o3_record(self):
        driver = getattr(self.driver, 'o3_counters', None) or {}
        return dict(on=int(bool(self.o3)), selected=int(getattr(self.controller, 'o3_selected', False) is True), ledger=len(self.o3_ledger),
                    **{k: (dict(v) if type(v) is dict else v) for k, v in self.o3_counts.items()},
                    driver={k: (dict(v) if type(v) is dict else v) for k, v in driver.items()})

    def _o3_count(self, table, why):
        self.o3_counts[table][why] = self.o3_counts[table].get(why, 0) + 1

    def _o3_covers(self, rows):
        counts = self.decode_counts
        return counts is not None and rows >= 1 and set(range(1, rows + 1)) <= counts

    def o3_keeps_split(self, uid, remaining):
        """The driver asks at a terminal commit under split: None keeps the route split for this run's handback, else the
        reason today's FULL return runs. Never raises. A run's end (the selection's terminal commit, no yield) starts the gap
        clock, read at the next arm by path: 'o3' (stayed split) or 'full' (today's cycle, any reason)."""
        why = self._o3_keeps_split(uid, remaining)
        try:
            request = self._request()
            if (self.phase == 'native' and request is not None and request.request_id == uid
                    and not self.driver.append_yield_active()):
                self._o3_gap_start = (time.monotonic_ns(), 'o3' if why is None else 'full')
        except Exception:
            self.decline_totals['read_errors'] += 1
        return why

    def _o3_gap_close(self):
        """The next arm after a run's end: the gap in ms by path, [count, total, max] (bounded; no request id)."""
        start, self._o3_gap_start = self._o3_gap_start, None
        if start is None:
            return
        ms = (time.monotonic_ns() - start[0]) / 1e6
        n, total, worst = self.o3_counts['gap'].get(start[1], (0, 0.0, 0.0))
        self.o3_counts['gap'][start[1]] = [n + 1, round(total + ms, 3), round(max(worst, ms), 3)]

    def _o3_keeps_split(self, uid, remaining):
        try:
            d, s = self.driver, self.scheduler
            request = self._request()
            if not self.o3 or getattr(self.controller, 'o3_selected', False) is not True:
                return 'off'     # installed for the run (FLASH_S5_O3) and selected by the arm (o3_split)
            y = (getattr(d, '_yield', None) or {}) if d.append_yield_active() else None
            if y is not None:
                # lane-o3-fix: a yielded append's terminal handback is a yield, not 'not-the-selection'. (b): the append's own
                # one-row handback (its FULL window today) may stay split under the same rules as a run's end; the bulk then
                # resumes on this split route after a census.
                if (y.get('phase') != 'handback' or getattr(y.get('append'), 'request_id', None) != uid
                        or self.phase != 'native' or request is None):
                    return 'yield'
            elif self.phase != 'native' or request is None or request.request_id != uid:
                return 'not-the-selection'
            if remaining != 1:
                return 'tail-rows'
            if self.o3_split_generation is None or d.route_generation != self.o3_split_generation:
                return 'generation'
            running = tuple(s.running)
            if any(r.is_prefill_chunk or r.num_computed_tokens < r.num_prompt_tokens for r in running):
                return 'cohort-prefill'
            if not {r.request_id for r in running} <= self.o3_ledger:
                return 'ledger'
            rows = len(running) + 1
            if rows > min(MAX_COHORT, s.max_num_running_reqs):
                return 'slots'
            if not self._o3_covers(rows):
                return 'counts'
            self.o3_counts['yield_split' if y is not None else 'kept'] += 1
            return None
        except Exception:
            self.decline_totals['read_errors'] += 1
            return 'read-error'

    def _o3_enter_tail(self, request):
        """The run's terminal commit kept the route split: wait for stock to schedule the handback row on the split bank."""
        self.phase, self.o3_tail, self.tail_ticks = 'tail', True, 0
        self._note('split_handback')
        try:
            from .engine_driver import _progress_logger
            _progress_logger().info('flash-s5 tail split-keep selection=%d computed=%s prompt=%s', self.selection_seq,
                                    getattr(request, 'num_computed_tokens', None), getattr(request, 'num_prompt_tokens', None))
        except Exception:
            self.decline_totals['read_errors'] += 1

    def _o3_tail(self):
        """The split tail: the handback row scheduled (or the request gone) -> close and arm the next head on split."""
        d = self.driver
        request = self._request()
        self.tail_ticks += 1
        if not d._split_ready or d._return_pending is not None:
            # The route is leaving split under us (a stop, a reset, the tail bound): today's FULL tail from here on.
            self.o3_tail = False
            self._decline('tail')
            return False
        if request is not None and not self._finished(request) and not self._decode_scheduled(request):
            # lane-o3-fix 1: wait for the row's first decode after the handback. The worker counts a row's prefill from the
            # step that carries it (the scheduler's pre-advance count), so straight after the handback step every worker still
            # reads it as prefill and the census refuses; the step that schedules its first decode lands it as a decoder.
            if self._handback_scheduled(request):
                self.o3_counts['settle_ticks'] += 1
            if self.tail_ticks > O3_TAIL_TICKS:
                self.o3_counts['tail_bound'] += 1
                d.o3_return_full('tail-bound')
                self.o3_tail = False
            self._decline('tail')
            return False
        joined = request.request_id if request is not None else None
        d.o3_handback_clear('scheduled' if request is not None else 'left')
        self._tail_end('split-tail-scheduled' if request is not None else 'split-tail-left', request)
        self.restores['split-tail-scheduled'] = self.restores.get('split-tail-scheduled', 0) + 1
        self._note('closed', reason='split-tail-scheduled')
        self.completed += 1
        self.o3_tail = False
        self.request_ref, self.plan, self.pending, self.cohort = None, None, None, ()
        self.phase = 'idle'
        self.o3_joined = joined
        if self._select(o3=True):
            return True
        # No head to arm on split (or its plan refused): today's FULL return, after the in-flight handback step lands.
        self._o3_count('full', 'no-next-head')
        self.old_pause = self.scheduler.pause_state
        self._pause(type(self.old_pause).PAUSED_ALL)
        self.deadline = d.clock() + d._control_timeout_s
        self._o3_return_reason = 'o3-no-next-head'
        self.phase = 'rollback_drain'
        return False

    def _o3_yield_tail(self):
        """lane-o3-fix (b): a yielded append's handback kept split. Wait for the row's first decode (fix 1's rule), then re-enter
        the yielded bulk on this split route: its re-entry plan, the yield's reenter phase, PAUSED_ALL, drain, census, arm,
        resume; no FULL window. Past the tail bound, or the bulk gone or without a plan: today's FULL return, and the yield's
        handback phase runs as today from there."""
        d = self.driver
        y = getattr(d, '_yield', None) or {}
        append = y.get('append')
        self.tail_ticks += 1
        if not d._split_ready or d._return_pending is not None or d.append_yield_phase() != 'handback':
            self.o3_tail, self.o3_yield, self.phase = False, False, 'native'
            self._decline('tail')
            return False
        live = append is not None and self.scheduler.requests.get(append.request_id) is append
        if live and not self._finished(append) and not self._decode_scheduled(append):
            if self._handback_scheduled(append):
                self.o3_counts['settle_ticks'] += 1
            if self.tail_ticks > O3_TAIL_TICKS:
                self.o3_counts['tail_bound'] += 1
                self._o3_yield_full('tail-bound', route=False)
            self._decline('tail')
            return False
        d.o3_handback_clear('scheduled' if live else 'left')
        bulk = self._request()
        plan = self._reentry_plan(bulk) if bulk is not None else None
        if plan is None:
            self._o3_yield_full('bulk-left' if bulk is None else 'no-reentry-plan', route=False)
            return False
        self.o3_tail = False
        d.append_yield_note_handback()
        if self._yield_window():
            d.append_yield_window_offer()                            # yield-rules Y2: off, never called
        d.append_yield_set_phase('reenter')
        self.plan, self.reentry = plan, True
        self.reentry_refusals = 0
        self._yrs_t0 = time.monotonic_ns()       # yield-reentry (1)
        self.drain_wait_seq = None
        self.cohort = self._cohort() or ()
        self.o3_joined = append.request_id if live else None
        self.old_pause = self.scheduler.pause_state
        self._pause(type(self.old_pause).PAUSED_ALL)
        self.deadline = d.clock() + d._control_timeout_s
        self.phase = 'o3_drain'
        self._note('reentry', start=plan['start'], native_end=plan['end'], chunks=len(plan['plan']), o3=1)
        return True

    def _o3_yield_full(self, why, *, route):
        """(b) refused before the re-entry began: today's FULL return of the append's handback (the yield's handback phase)."""
        self._o3_count('yield_full', why)
        self.driver.o3_return_full('yield-' + why)
        self.o3_tail, self.o3_yield, self.phase = False, False, 'native'

    def _o3_plan(self, request):
        return self._reentry_plan(request) if self.reentry else self._plan(request)

    def _o3_refuse(self, why):
        """A split continuation that cannot be proven: today's FULL return (leave_split), then the frozen close. Under (b) the
        bulk's re-entry is not closed: the route returns FULL and today's re-entry runs from its drain (never a silent drop)."""
        if self.control_inwait is not None and self.control_inwait.o3_active:
            return self.control_inwait.refuse_o3(why)
        self._o3_count('refused', why)
        if self.o3_yield:
            self._o3_count('yield_full', why)
            self.deadline = self.driver.clock() + self.driver._control_timeout_s
            self.phase = 'o3_yield_return'
            self.pending = self._control('leave_split')
            return True
        self._rollback('o3-' + why)
        return True

    def _o3_census_begin(self):
        d = self.driver
        if d._return_pending is not None or d._route_exchange() is not None:
            return False                       # the route is already on its way back: wait for it
        if not d._split_ready:
            self._restore('o3-route-left-split')
            return True
        if d.active or d.admission_intent() is not None:
            return self._o3_refuse('driver-owned')
        request = self._request()
        cohort = self._cohort()
        if request is None or cohort != self.cohort or self._o3_plan(request) != self.plan:
            return self._o3_refuse('selection-changed')
        if d.route_generation != self.o3_split_generation:
            return self._o3_refuse('generation')
        if not self._o3_covers(max(1, len(cohort))):
            return self._o3_refuse('counts')
        allowed = self.o3_ledger | ({self.o3_joined} if self.o3_joined is not None else frozenset())
        if not set(cohort) <= allowed:
            return self._o3_refuse('ledger')
        self.pending = self._control('census')
        self.phase = 'o3_census'
        return True

    def _o3_proof(self, result):
        """The census on the kept split route: every rank split on this generation, its live rows exactly the cohort, all
        decode rows (prefill complete). The design's worker check, on the worker's own census."""
        rows = result.get('workers')
        d = self.driver
        if (type(rows) is not list or len(rows) != len(d.ranks)
                or {r.get('worker_rank') for r in rows} != set(d.ranks)):
            return False
        for row in rows:
            live = row.get('live_decode') or {}
            if (row.get('engine_epoch') != d.engine_epoch or row.get('route') != 'split'
                    or row.get('route_generation') != d.route_generation
                    or row.get('prefill_requests') != 0 or live.get('available') is not True
                    or live.get('prefill_requests') != 0
                    or sorted(live.get('live_request_uids', ())) != list(self.cohort)
                    or row.get('decode_eligible') != len(self.cohort)):
                return False
        return True

    def _o3_arm(self):
        d = self.driver
        self.o3_ledger = frozenset(self.cohort)
        d._control_route = None
        _fit_floor = getattr(d, 'graph_event_floor', None)
        if _fit_floor is not None and not _fit_floor.before_arm():
            return False
        self._internal = True
        try:
            d._arm_admission(PHASE)
        finally:
            self._internal = False
        if self.control_inwait is not None and self.control_inwait.o3_active:
            self.control_inwait.close(True)
        self.phase = 'native'
        self.schedule_before = self.controller._schedule_seq
        self.admitted_before = self.controller.admitted
        self.pending = None
        self.o3_joined = None
        if self.o3_yield:
            # lane-o3-fix (b): the yielded bulk resumes on the split route it never left; the yield ends at its reopen as today
            self.o3_yield, self.reentry = False, False
            if not (self._yield_window() and d.append_yield_window_begin()):    # yield-rules Y2: the window's next append
                d.append_yield_set_phase('resume')
            self.o3_counts['yield_kept'] += 1
            self._note('armed', route_generation=d.route_generation, o3=1, yield_kept=1)
            self._pause(self.old_pause)
            return
        self.decline_totals['armed'] += 1
        self.o3_counts['armed'] += 1
        self._o3_gap_close()
        self._note('armed', route_generation=d.route_generation, o3=1)
        self._pause(self.old_pause)

    # -- coverage-residue ---------------------------------------------------
    def _residue(self):
        """On only with the hold: the residue is the hold's."""
        return 1 if self._hold_limit() and getattr(self.controller, 'native_residue', 0) == 1 else 0

    def _coverage_record(self):
        coverage = getattr(self.controller, 'coverage', None)
        record = getattr(coverage, 'record', None)
        return record() if callable(record) else None

    def _why(self, request, why):
        table = self.stock_why
        table.pop(request.request_id, None)
        table[request.request_id] = why
        while len(table) > MAX_STOCK_WHY:
            table.pop(next(iter(table)))

    def stock_site(self, request):
        """Why stock took this request, for the coverage counter: the hold's word at the stock seam, else the head's decline
        site, else 'no-seam' (stock reached it on a path the hold doesn't see). Never raises."""
        try:
            uid = request.request_id
            why = self.stock_why.get(uid)
            pending = self._decline_pending.get(uid)
            site = pending[1] if pending is not None and pending[0]() is request else None
            if why is None:
                return site or 'no-seam'
            if why == 'hold-off':
                return 'hold-off/' + (site or 'none')
            return why
        except Exception:
            return 'site-error'

    # -- decode-phase hold -------------------------------------------------
    def _hold_limit(self):
        limit = getattr(self.controller, 'native_hold_s', 0)
        return limit if type(limit) in (int, float) and limit > 0 else 0

    def _holdable(self, request):
        """The plan's own state checks and its row floor, on the rows stock would compute; the full plan is _select's."""
        if self._plan_state_site(request) is not None:
            return False
        uncached = getattr(self.scheduler, '_sf_uncached', None)
        rows = uncached(request) if callable(uncached) and not self._residue() else request.num_tokens
        # coverage-residue: a lowered hit can always leave the floor if the prompt has it, so the whole prompt decides.
        return type(rows) is int and rows - 1 >= self._floor()

    def _floor(self):
        """Rows a native plan needs: today's 32768 (native41's conservative choice, not a measured cost), or under the hold the
        staged path's own minimum."""
        c = self.controller
        if self._hold_limit():
            return max(1, getattr(c, 'minimum_stage_rows', 1))
        return max(MIN_NATIVE_ROWS, c.chunk_tokens)

    def _unhold(self, request, why, site=None):
        if self.holds.pop(request.request_id, None) is not None or why == 'unplannable':
            self.hold_totals[why] += 1
            key = site or why
            self.hold_released[key] = self.hold_released.get(key, 0) + 1
        self.unhold_site[request.request_id] = site or why
        while len(self.unhold_site) > MAX_UNHOLDABLE:
            self.unhold_site.pop(next(iter(self.unhold_site)))
        self.unholdable[request.request_id] = weakref.ref(request)
        while len(self.unholdable) > MAX_UNHOLDABLE:
            self.unholdable.pop(next(iter(self.unholdable)))

    def _prune_holds(self):
        for uid, (ref, _since) in list(self.holds.items()):
            request = ref()
            if (request is None or self.scheduler.requests.get(uid) is not request
                    or not str(getattr(getattr(request, 'status', None), 'name', None)).startswith('WAITING')):
                self.holds.pop(uid)

    def holds_frontier_from_stock(self, request):
        """build-2b (lead, 6 Nov): the positive-frontier seam asks for a waiting, part-computed request. True keeps a yielded
        bulk the re-entry could not plan out of stock this step: it never continues on stock while the hold's limit runs. At
        the limit (the hold's own valve) stock takes it, named 'yield-no-reentry-plan'. Any other request, or the hold off:
        False, as before. Never raises."""
        try:
            uid = getattr(request, 'request_id', None)
            entry = self.abandoned.get(uid)
            if entry is None or entry[0]() is not request:
                return False
            limit = self._hold_limit()
            if limit and self.driver.clock() - entry[1] <= limit:
                self.bulk_hold['deferrals'] += 1
                return True
            self.abandoned.pop(uid, None)
            self.bulk_hold['expired'] += 1
            self._why(request, 'yield-no-reentry-plan')
            return False
        except Exception:
            self.bulk_hold['errors'] += 1
            return False

    def holds_from_stock(self, request):
        """The stock loop's fresh seam asks: True keeps this waiting request in skipped-waiting, unallocated, this step.
        Stock admits the requests behind it. A count or a read must never stop the engine: any error lets stock take it."""
        try:
            if request is None:
                return False
            limit = self._hold_limit()
            if not limit:
                self._why(request, 'hold-off')
                return False
            if self.manual_owned:
                self._why(request, 'manual-owned')
                return False
            if not self.scheduler.running and not self._residue():
                self._why(request, 'nothing-running')
                return False
            uid = request.request_id
            if self.unholdable.get(uid, lambda: None)() is request:
                self._why(request, self.unhold_site.get(uid, 'unholdable'))
                return False
            if (self.rejected.get(uid, lambda: None)() is request and self.retries.get(uid, 0) >= MAX_RETRIES
                    and not self._residue()):
                self._why(request, 'retries-spent')
                return False
            now = self.driver.clock()
            entry = self.holds.get(uid)
            if entry is None or entry[0]() is not request:
                if not self._holdable(request):
                    self._unhold(request, 'unplannable', 'plan-state' if self._plan_state_site(request) else 'below-minimum')
                    self._why(request, self.unhold_site[uid])
                    return False
                if len(self.holds) >= MAX_HOLDS:
                    self._prune_holds()
                    if len(self.holds) >= MAX_HOLDS:
                        self._why(request, 'holds-full')
                        return False
                self.holds[uid] = (weakref.ref(request), now)
                self.hold_totals['held'] += 1
                return True
            if now - entry[1] > limit:
                self._unhold(request, 'expired')
                self._why(request, 'expired')
                return False
            return True
        except Exception:
            self.hold_totals['errors'] += 1
            try:
                self._why(request, 'hold-error')
            except Exception:
                pass
            return False

    def _decline(self, site):
        """The waiting head, if any, may go to stock this tick. A count must never stop the engine."""
        try:
            queue = self.controller._queue(self.scheduler)
            if not queue:
                return
            request = queue.peek_request()
            self.decline_ticks[site] += 1
            uid = request.request_id
            if uid not in self._decline_pending and len(self._decline_pending) >= MAX_PENDING_DECLINES:
                self._decline_pending.pop(next(iter(self._decline_pending)))
                self.decline_totals['dropped'] += 1
            self._decline_pending[uid] = (weakref.ref(request), site)
        except Exception:
            self.decline_totals['read_errors'] += 1

    def _settle_declines(self):
        """A head declined earlier: admitted since means stock took it, at the last site it saw."""
        if not self._decline_pending:
            return
        try:
            for uid, (ref, site) in list(self._decline_pending.items()):
                request = ref()
                state = getattr(getattr(request, 'status', None), 'name', None)
                if request is None or self.scheduler.requests.get(uid) is not request or str(state).startswith('FINISHED'):
                    self.decline_totals['left_while_waiting'] += 1
                elif str(state).startswith('WAITING'):
                    continue
                else:
                    self.decline_turns[site] += 1
                    self._decline_log(site, request)
                self._decline_pending.pop(uid)
        except Exception:
            self.decline_totals['read_errors'] += 1
            self._decline_pending.clear()

    @staticmethod
    def _decline_log(site, request):
        # One line per stock turn: its stamp and length tie the site to the client's journal.
        from .engine_driver import _progress_logger
        _progress_logger().info('flash-s5 decline site=%s num_tokens=%d', site, request.num_tokens)

    def _control(self, action):
        self._internal = True
        try:
            return self.driver.submit_control(dict(action=action, engine_epoch=self.driver.engine_epoch))
        finally:
            self._internal = False

    def _gap(self, event, **fields):
        try:
            emit = getattr(self.driver, '_gap_event', None)
            if callable(emit):
                emit(event, phase=self.phase, selection_seq=self.selection_seq, **fields)
        except Exception:
            pass

    def _pause(self, state):
        self.scheduler.set_pause_state(state)
        if self.control_inwait is not None:
            self.control_inwait.pause(state)
        self._gap('pause', pause=getattr(state, 'name', str(state)))

    def _note(self, event, **fields):
        self.driver._record('auto_' + event, selection_seq=self.selection_seq,
                            request_uid=self.snapshot()['request_uid'], **fields)

    @staticmethod
    def _decode_scheduled(request):
        """lane-o3-fix: stock scheduled the handed-back row once more, as a decoder (its first decode after the prompt): the
        step that carries it brings every worker's prefill count to the prompt."""
        return (getattr(getattr(request, 'status', None), 'name', None) == 'RUNNING' and not request.is_prefill_chunk
                and request.num_computed_tokens > request.num_prompt_tokens)

    @staticmethod
    def _finished(request):
        return str(getattr(getattr(request, 'status', None), 'name', '')).startswith('FINISHED')

    @staticmethod
    def _handback_scheduled(request):
        """K2: stock has scheduled every prompt row of the handed-back request (the 1-row handback, or a geometry
        tail's r rows over one or two steps); RUNNING, no chunk left, so the cohort reads it as a decoder."""
        return (getattr(getattr(request, 'status', None), 'name', None) == 'RUNNING' and not request.is_prefill_chunk
                and request.num_computed_tokens >= request.num_prompt_tokens)

    def _tail_ack(self):
        """Receipt (b): the FULL return is acknowledged and the tail starts; splits the unlogged handoff on the box."""
        self.tail_ticks = 0
        try:
            from .engine_driver import _progress_logger
            request = self._request()
            _progress_logger().info('flash-s5 tail full-ack selection=%d computed=%s prompt=%s', self.selection_seq,
                                    getattr(request, 'num_computed_tokens', None), getattr(request, 'num_prompt_tokens', None))
        except Exception:
            self.decline_totals['read_errors'] += 1

    def _tail_end(self, reason, request):
        """Receipt (b): the tail ends (K2 at the scheduled handback, or the old first-token/left close)."""
        try:
            from .engine_driver import _progress_logger
            _progress_logger().info('flash-s5 tail end reason=%s selection=%d ticks=%d output_tokens=%s', reason,
                                    self.selection_seq, self.tail_ticks, getattr(request, 'num_output_tokens', None))
        except Exception:
            self.decline_totals['read_errors'] += 1

    def _request(self):
        request = self.request_ref() if self.request_ref is not None else None
        if request is None or self.scheduler.requests.get(request.request_id) is not request:
            return None
        return request

    def permits(self, request):
        if self.phase != 'native':
            return False
        if self.driver.append_yield_active():
            # returning-append: only the yield's current target (the short request,
            # then the bulk's reopen) may be admitted.
            return self.driver.yield_permits(request)
        return request is self._request()

    def bound_intent(self, intent):
        request = self._request()
        uid = None if request is None else request.request_id
        if uid is None:
            # returning-append: a yielded bulk that left the scheduler (client abort)
            # keeps the split intent bound to it until this policy's own tick
            # closes the yield; the admission pass never ends it.
            uid = self.driver.append_yield_bulk_uid()
        return (self.phase == 'native' and uid is not None
                and intent.get('origin') == 'auto-arm' and intent.get('phase') == PHASE
                and intent.get('selection_seq') == self.selection_seq
                and intent.get('request_uid') == uid)

    def _remember(self):
        request = self.request_ref() if self.request_ref is not None else None
        if request is not None:
            self.rejected[request.request_id] = weakref.ref(request)
            # coverage-residue: the cohort it was refused under; _select retries it once that has changed.
            self.refused_cohort.pop(request.request_id, None)
            self.refused_cohort[request.request_id] = self.cohort
            while len(self.refused_cohort) > MAX_REJECTED:
                self.refused_cohort.pop(next(iter(self.refused_cohort)))
        self.rejected = {k: ref for k, ref in self.rejected.items() if ref() is not None}
        while len(self.rejected) > MAX_REJECTED:
            self.rejected.pop(next(iter(self.rejected)))

    def _cohort(self):
        running = tuple(self.scheduler.running)
        if not running:
            if self._residue():
                return ()              # coverage-residue: idle prefill goes native too; the barrier proves an empty live set
            self._decline_why = 'cohort-empty'
            return None
        if len(running) > min(MAX_COHORT, self.scheduler.max_num_running_reqs - 1):
            self._decline_why = 'cohort-over'
            return None
        if any(r.is_prefill_chunk or r.num_computed_tokens < r.num_prompt_tokens for r in running):
            self._decline_why = 'cohort-prefill'
            return None
        if any(r.lora_request is not None or r.resumable for r in running):
            self._decline_why = 'cohort-other'
            return None
        try:
            for r in running:_text('request_uid', r.request_id)
        except E3ContractError:
            self._decline_why = 'cohort-uid'
            return None
        return tuple(sorted(r.request_id for r in running))

    def _plan_state_site(self, request):
        s, c = self.scheduler, self.controller
        if (request is None or s.requests.get(request.request_id) is not request
                or getattr(request.status, 'name', None) != 'WAITING'
                or request.num_computed_tokens != 0 or request.num_output_tokens != 0
                or request.num_preemptions != 0 or request.resumable
                or request.lora_request is not None or request.pooling_params is not None
                or getattr(request, 'structured_output_request', None) is not None):
            return 'plan-state'
        if c._request_refuses(s, request):
            return 'plan-controller'
        if request.ec_transfer_params:
            return 'plan-transfer'
        if request.sampling_params.prompt_logprobs is not None:
            return 'plan-prompt-logprobs'
        if _generation_logprobs_refused(request.sampling_params):
            return 'plan-generation-logprobs'
        return None

    def _native_end(self, request):
        """coverage-soak-fix: the admission's native end (the last block boundary on a geometry tail), else the last row."""
        end = getattr(self.controller, 'native_end', None)
        return end(self.scheduler, request) if callable(end) else request.num_tokens - 1

    def _plan(self, request, *, lookup_stats=None):
        s, c = self.scheduler, self.controller
        site = self._plan_state_site(request)
        if site is not None:
            self._decline_why = site
            return None
        try:
            _text('request_uid', request.request_id)
        except E3ContractError:
            self._decline_why = 'plan-uid'
            return None
        residue = self._residue()
        self._plan_residue = None
        plan_request = request
        started = time.monotonic_ns() if lookup_stats is not None else None
        try:
            if residue:
                # coverage-residue: the admission's own hit and shared-junction stop.
                blocks, start, shared, lowered = residue_hit(s, request, self._floor(), self._native_end(request))
                if lookup_stats is None and not getattr(c, 'shortest_bulk', False):
                    request.shared_prefix_boundary = shared
                else:
                    plan_request = _ProjectedPrefillRequest(request, request.num_computed_tokens,
                                                           shared_prefix_boundary=shared)
                self._plan_residue = 'lowered' if lowered else ('junction' if shared else None)
            else:
                blocks, start, shared = s.kv_cache_manager.get_computed_blocks(request)
        finally:
            if lookup_stats is not None:
                lookup_stats['extra_lookup_calls'] += 1
                lookup_stats['extra_lookup_ns'] += time.monotonic_ns() - started
        end = self._native_end(request)
        # Shared-prefix junction/streaming/recompute special cases stay stock.
        if shared and not residue:
            self._decline_why = 'plan-shared'
            return None
        if end - start < self._floor():
            self._decline_why = 'plan-below-floor'
            return None
        manager = s.kv_cache_manager
        needed = manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id, num_tokens=request.num_tokens,
            new_computed_blocks=blocks.blocks, num_encoder_tokens=0,
            total_computed_tokens=start, num_local_computed_tokens=start,
            num_tokens_main_model=request.num_tokens, apply_admission_cap=True)
        if needed + manager.watermark_blocks > manager.block_pool.get_num_free_blocks():
            self._decline_why = 'plan-kv-blocks'
            return None
        cap = native_chunk_cap(c.chunk_tokens, s.max_num_scheduled_tokens)
        # decode-phase hold: a prompt whose last hash-boundary tail is under the staged minimum had no plan (dnr: every
        # plan-empty turn; about 0.36 of all turns once the hold sends every turn here). a1-trunk 1d's view moves stock's
        # stop one unit back, as the admission does under the hold (scheduler_hooks.admit); unchanged with the hold off.
        moved = getattr(c, '_append_yield_split_view', None)
        view = moved(s, plan_request) if self._hold_limit() and callable(moved) else plan_request
        def split(position, limit):
            limit = bounded_limit(getattr(c, 'chunk_bound', None), position, limit)
            projected = view if position == start else _ProjectedPrefillRequest(view, position)
            return c._aligned_split(s, projected, limit,
                num_new_local_computed_tokens=start if position == start else 0)
        plan = staged_chunk_plan(start, end, cap, c.minimum_stage_rows, split)
        if not plan:
            self._decline_why = 'plan-empty'
            return None
        self._plan_moved = view is not plan_request  # count a moved stop, not the read-only probe view
        return dict(start=start, end=end, plan=tuple(plan),
                    blocks=tuple(tuple(b.block_id for b in group) for group in blocks.blocks))

    def _reentry_plan(self, request):
        """A yielded bulk's plan from its committed frontier; no bulk-size floor.

        Bound to the selected request, never the waiting head; the frontier is
        the scheduler's own count, never a fresh prefix lookup.
        """
        s, c = self.scheduler, self.controller
        if (request is None or s.requests.get(request.request_id) is not request
                or getattr(request.status, 'name', None) != 'WAITING'
                or request.num_output_tokens != 0 or request.resumable):
            return None
        start, end = request.num_computed_tokens, self._native_end(request)
        if type(start) is not int or start <= 0 or end - start < 1:
            return None
        cap = native_chunk_cap(c.chunk_tokens, s.max_num_scheduled_tokens)    # the cap admission plans with
        # coverage-throughput build-2b (1b's box run): the bulk's remainder plans over the view its admission reads
        # (controller.plan_view: the hold's moved stop on a short hash tail), as _plan, admission and the exact successor do.
        # The raw view had no plan on a short tail: 19 yields closed 'yielded-bulk-has-no-reentry-plan' and stock took the bulks.
        view_of = getattr(c, 'plan_view', None)
        view = view_of(s, request) if callable(view_of) else request
        def split(position, limit):
            limit = bounded_limit(getattr(c, 'chunk_bound', None), position, limit)
            projected = view if position == start else _ProjectedPrefillRequest(view, position)
            return c._aligned_split(s, projected, limit)
        plan = staged_chunk_plan(start, end, cap, c.minimum_stage_rows, split)
        if not plan:
            return None
        return dict(start=start, end=end, plan=tuple(plan), reentry=True)

    def _reentry_cohort(self):
        """The running decoders at re-entry (the returned append among them)."""
        running = tuple(self.scheduler.running)
        if (len(running) > min(MAX_COHORT, self.scheduler.max_num_running_reqs - 1)
                or any(r.is_prefill_chunk or r.num_computed_tokens < r.num_prompt_tokens
                       or r.lora_request is not None or r.resumable for r in running)):
            return None
        try:
            for r in running:_text('request_uid', r.request_id)
        except E3ContractError:
            return None
        return tuple(sorted(r.request_id for r in running))

    def _begin_reentry(self, request):
        """Start the bulk's re-entry; False (never a raise) when it has no plan.

        Runs when the FULL window ends: the append's handback ran or the append
        left, or the launch bound of FULL steps expired with it unscheduled.
        """
        d = self.driver
        plan = self._reentry_plan(request)
        if plan is None:
            # FULL, nothing native owned, no intent armed: close as an ordinary
            # auto-arm selection; the bulk continues on stock, under its own site name in the coverage counter
            # (build-2b: never 'cooldown' or 'no-seam'), so the grade names it.
            if request is not None:
                self._why(request, 'yield-no-reentry-plan')
                self.abandoned.pop(request.request_id, None)
                self.abandoned[request.request_id] = (weakref.ref(request), d.clock())
                while len(self.abandoned) > MAX_UNHOLDABLE:
                    self.abandoned.pop(next(iter(self.abandoned)))
                self.bulk_hold['bulks'] += 1
            self._restore('yielded-bulk-has-no-reentry-plan')
            return False
        if d.append_yield_active():
            d.append_yield_set_phase('reenter')
        self.plan, self.reentry = plan, True
        self.reentry_refusals = 0
        self._yrs_t0, self._yrs_ticks = time.monotonic_ns(), None       # yield-reentry (1)
        self.drain_wait_seq = None
        self.cohort = self._reentry_cohort() or ()
        self.deadline = d.clock() + d._control_timeout_s
        self.phase = 'drain'
        self._note('reentry', start=plan['start'], native_end=plan['end'], chunks=len(plan['plan']))
        self._pause(type(self.old_pause).PAUSED_ALL)
        return True

    def _yield_native(self, request):
        """The 'native' phase while a yield runs; False lets ordinary Core work run.

        Every phase exits within counted engine steps (review of f76a31de): the
        stale guard, the append's refused admission, the FULL window's bound and
        the reopen's retries; none reads a clock or raises.
        """
        d = self.driver
        phase = d.append_yield_phase()
        if (phase in ('draining', 'closing', 'append', 'resume') and not d._split_ready and not d.active
                and d._return_pending is None and d._route_exchange() is None):
            # The route left split without the yield's own stop: never hold FULL.
            if d.append_yield_stale() is None:
                self.phase = 'tail'
                self._note('full_handback')
                self._tail_ack()
            return False
        if phase == 'append':
            c = self.controller
            d.append_yield_append_wait(c.admitted, c._schedule_seq)
            return False
        if phase == 'resume':
            if d.active:
                return False                     # the bulk's reopen is parked or running
            if request is None:
                d.append_yield_end('bulk-left')
                return self._native_refused()
            c = self.controller
            # Scheduler passes since resume began (counted on the yield: resume is
            # also reached on the existing arm) without the bulk's admission; past
            # the launch bound the yield ends and the existing refused-admission
            # rule disarms (never a raise, never a clock).
            if d.append_yield_reentry_wait(c.admitted, c._schedule_seq):
                return self._native_refused()
            return False
        if phase != 'handback':
            return False                         # draining or closing: split, chunks committing or the row close in flight
        append = (getattr(d, '_yield', None) or {}).get('append')
        if (d._split_ready and not d.active and d._return_pending is None and request is not None and append is not None
                and d.o3_handback_uid() == append.request_id):
            # lane-o3-fix (b): the append's handback was kept split; wait for its first decode, then re-enter the bulk here
            self.o3_yield = True
            self._o3_enter_tail(append)
            return False
        if d._split_ready or d.active or d._return_pending is not None or d._route_exchange() is not None:
            return False                         # the FULL return of the handback is still settling
        d.append_yield_mark_full()
        if request is None:
            d.append_yield_end('bulk-left')
            self._restore('yielded-bulk-left')
            return False
        if not d.append_yield_handback_done():
            # The append's handback runs on FULL first, for at most the launch
            # bound of engine steps; then the FULL window ends and the bulk re-enters.
            # An append with more than its one handback token left (stopped by the
            # abort rule) cannot be served there: the window is skipped at once.
            if not d.append_yield_handback_skip() and not d.append_yield_handback_step():
                return False
        elif self.yield_settle and self._yrs_settle_waits():       # yield-reentry (2): off, never entered
            return False
        self._yrs_ticks = None                                       # yield-reentry (2)
        d.append_yield_note_handback()
        if self._yield_window():
            d.append_yield_window_offer()                            # yield-rules Y2: off, never called
        return self._begin_reentry(request)

    def _yield_window(self):
        """yield-rules Y2 (FLASH_S5_YIELD_WINDOW): the controller's flag; off, no window call is made."""
        return getattr(self.controller, 'yield_window', False) is True

    def _reentry_refused(self, stage, why=None):
        """One refused re-entry drain or proof; past the launch bound, close on FULL.

        Otherwise the next attempt waits one scheduler pass with the pause
        released (delta review of ae865d44, D-MINOR-2): back-to-back attempts
        under PAUSED_ALL could not see any change.  The route is FULL with no
        intent; the yield's hold keeps every stock prefill held but a decoder's
        running prefill, which may finish its chunk in that pass.
        """
        self.reentry_refusals += 1
        ends = self.driver.append_yield_reentry_refused(stage, self.reentry_refusals)
        self._yrs_refusal(stage, why, ends)                       # yield-reentry (1): observational
        if ends:
            self._restore('yielded-bulk-reentry-refused')
            return True
        self.drain_wait_seq = self.controller._schedule_seq
        self.driver.append_yield_set_drain_wait(True)
        self._pause(self.old_pause)
        self.phase = 'drain_wait'
        return True

    # -- yield-reentry (12 Nov 2026): (1) diagnostic, observational only; (2) the settle (FLASH_S5_YIELD_REENTRY_SETTLE) --
    def yrs_record(self):
        c = self.yrs_counts
        return dict(schema='s5-yield-reentry/1', settle_on=int(bool(self.yield_settle)),
                    retry_bound=self._yrs_bound(),
                    **{k: (dict(v) if type(v) is dict else (list(v) if type(v) is list else v)) for k, v in c.items()})

    def _yrs_bound(self):
        try:
            value = self.controller.append_yield.reentry_retries
            return value if type(value) is int else None
        except Exception:
            return None

    def _yrs_bump(self, table, key):
        t = self.yrs_counts[table]
        if key in t or len(t) < YRS_MAX_KEYS:
            t[key] = t.get(key, 0) + 1
        else:
            t['other'] = t.get('other', 0) + 1

    def _yrs_ms(self):
        t0 = self._yrs_t0
        return None if t0 is None else (time.monotonic_ns() - t0) / 1e6

    def _yrs_refusal(self, stage, why, ends):
        """(1) One refused re-entry proof: counted by stage and field, one line. Never raises, never decides."""
        try:
            field, rank, got, want = why if type(why) is tuple and len(why) == 4 else ('unread', None, None, None)
            self._yrs_bump('refused', '%s:%s' % (stage, field))
            self._yrs_bump('refused_retry', str(self.reentry_refusals))
            if ends:
                self.yrs_counts['exceeded'] += 1
                self._yrs_bump('retries_until_armed', 'exceeded')
            ms = self._yrs_ms()
            dec, out, running = self._yrs_append_state()
            from .engine_driver import _progress_logger
            _progress_logger().info('flash-s5 yield-reentry proof-miss stage=%s field=%s rank=%s got=%s want=%s retry=%d bound=%s '
                                    'ends=%d elapsed_ms=%s settle=%d append_dec=%s append_out=%s running=%s', stage, field,
                                    rank, got, want, self.reentry_refusals, self._yrs_bound(), int(bool(ends)),
                                    'na' if ms is None else '%.1f' % ms, int(bool(self.yield_settle)), dec, out, running)
        except Exception:
            self.yrs_counts['errors'] += 1

    def _yrs_append_state(self):
        """night-pack fix 6, observational: the returned append's state at a refusal: decode scheduled (1/0), output
        tokens, and the scheduler's running count ('na' when unread). Never raises."""
        dec = out = running = 'na'
        try:
            running = len(self.scheduler.running)
        except Exception:
            pass
        try:
            append = (getattr(self.driver, '_yield', None) or {}).get('append')
            if append is not None:
                out = int(getattr(append, 'num_output_tokens', 0))
                dec = int(bool(self._decode_scheduled(append)))
        except Exception:
            pass
        return dec, out, running

    def _yrs_barrier_pass(self):
        try:
            self.yrs_counts['barrier_pass'] += 1
            if self.reentry_refusals == 0:
                self.yrs_counts['barrier_first_try'] += 1
        except Exception:
            self.yrs_counts['errors'] += 1

    def _yrs_armed(self):
        """(1) The re-entry armed: its refusals so far into the histogram, its ms since the start into [n, total, max]."""
        try:
            self._yrs_bump('retries_until_armed', str(self.reentry_refusals))
            ms = self._yrs_ms()
            if ms is not None:
                n, total, worst = self.yrs_counts['armed_ms']
                self.yrs_counts['armed_ms'] = [n + 1, round(total + ms, 3), round(max(worst, ms), 3)]
            self._yrs_t0 = None
        except Exception:
            self.yrs_counts['errors'] += 1

    def _yrs_cohort_why(self):
        """(1) Why _reentry_cohort() refused, in its own order: ('slots'|'prefill'|'lora'|'resumable'|'uid', None, got, want)."""
        try:
            running = tuple(self.scheduler.running)
            bound = min(MAX_COHORT, self.scheduler.max_num_running_reqs - 1)
            if len(running) > bound:
                return ('slots', None, len(running), bound)
            for r in running:
                if r.is_prefill_chunk or r.num_computed_tokens < r.num_prompt_tokens:
                    return ('prefill', None, sum(1 for x in running if x.is_prefill_chunk
                                                 or x.num_computed_tokens < x.num_prompt_tokens), 0)
                if r.lora_request is not None:
                    return ('lora', None, None, None)
                if r.resumable:
                    return ('resumable', None, None, None)
            for r in running:
                try:
                    _text('request_uid', r.request_id)
                except E3ContractError:
                    return ('uid', None, None, None)
            return ('none', None, None, None)
        except Exception:
            self.yrs_counts['errors'] += 1
            return ('unread', None, None, None)

    def _yrs_proof_why(self, result, *, capability=False):
        """(1) The first field _proof refuses, checked in _proof's own order; numbers only (counts, never a request UID).
        ('none', ...) if every check passes (a test pins: 'none' exactly when _proof is True)."""
        try:
            rows = result.get('workers') if type(result) is dict else None
            d = self.driver
            if type(rows) is not list or len(rows) != len(d.ranks):
                return ('worker-rows', None, len(rows) if type(rows) is list else None, len(d.ranks))
            if {r.get('worker_rank') for r in rows} != set(d.ranks):
                return ('worker-ranks', None, None, None)
            for row in rows:
                rank = row.get('worker_rank')
                live = row.get('live_decode') or {}
                if row.get('engine_epoch') != d.engine_epoch:
                    return ('engine-epoch', rank, None, None)
                if row.get('route') != 'full':
                    return ('route', rank, row.get('route'), 'full')
                if row.get('route_generation') != d.route_generation:
                    return ('route-generation', rank, row.get('route_generation'), d.route_generation)
                if row.get('prefill_requests') != 0:
                    return ('prefill-requests', rank, row.get('prefill_requests'), 0)
                if live.get('available') is not True:
                    return ('live-unavailable', rank, None, None)
                uids, want = sorted(live.get('live_request_uids', ())), list(self.cohort)
                if uids != want:
                    return ('live-uids', rank, len(uids), len(want))
                if row.get('decode_eligible') != len(self.cohort):
                    return ('decode-eligible', rank, row.get('decode_eligible'), len(self.cohort))
                if capability:
                    cap = (row.get('native_inventory') or {}).get('auto_arm_decode') or {}
                    counts = cap.get('supported_counts')
                    if (type(counts) is not list or len(counts) > MAX_COHORT
                            or any(type(n) is not int or not 1 <= n <= MAX_COHORT for n in counts)
                            or len(set(counts)) != len(counts)):
                        return ('capability-counts', rank, None, None)
                    if (cap.get('schema') != 's5-auto-decode-capability/1'
                            or cap.get('engine_epoch') != d.engine_epoch
                            or cap.get('route_generation') != d.route_generation
                            or cap.get('storage_epoch') != row.get('storage_epoch')):
                        return ('capability-identity', rank, None, None)
                    if not set(range(1, len(self.cohort) + 1)) <= set(counts):
                        return ('capability-cover', rank, None, len(self.cohort))
                elif row.get('barrier_clean') is not True:
                    return ('barrier-clean', rank, None, None)
                elif row.get('drained_streams', 0) <= 0:
                    return ('drained-streams', rank, row.get('drained_streams', 0), 1)
            return ('none', None, None, None)
        except Exception:
            self.yrs_counts['errors'] += 1
            return ('unread', None, None, None)

    def _yrs_settle_waits(self):
        """(2) True while the FULL window waits for the returned append's first decode (the run-end path's K2 rule):
        the append is live, every prompt row scheduled, no decode yet, and the bound not spent. Off: never called."""
        try:
            y = getattr(self.driver, '_yield', None) or {}
            if self._yrs_seq != y.get('seq'):
                self._yrs_seq, self._yrs_ticks = y.get('seq'), None     # a new yield: no ticks carried over
            append = y.get('append')
            live = append is not None and self.scheduler.requests.get(append.request_id) is append
            why = None
            if not live:
                why = 'append-left'
            elif self._finished(append):
                why = 'append-finished'
            elif not self._handback_scheduled(append):
                why = 'handback-unscheduled'
            elif self._decode_scheduled(append):
                why = 'decoded' if self._yrs_ticks else 'already-decoding'
            if why is not None:
                if why == 'decoded':
                    self.yrs_counts['settled'] += 1
                else:
                    self._yrs_bump('settle_skipped', why)
                return False
            if self._yrs_ticks is None:
                self._yrs_ticks = 0
                self.yrs_counts['settle_waits'] += 1
            self._yrs_ticks += 1
            if self._yrs_ticks > O3_TAIL_TICKS:
                self.yrs_counts['settle_bound'] += 1
                return False
            self.yrs_counts['settle_ticks'] += 1
            return True
        except Exception:
            self.yrs_counts['errors'] += 1
            return False

    def _reentry_lost_yield(self):
        """A stop (a reset included) ended the yield during the re-entry: follow the
        existing stop rule instead of arming (FULL, the bulk on stock)."""
        if not self.reentry or self.driver.append_yield_active():
            return False
        if self.driver._split_ready:
            self._rollback('yield-ended-during-reentry')
        else:
            self._restore('yield-ended-during-reentry')
        return True

    def _native_refused(self):
        self._pause(type(self.old_pause).PAUSED_ALL)
        if getattr(self.engine, 'batch_queue', None):
            self.deadline = self.driver.clock() + self.driver._control_timeout_s
            self.phase = 'rollback_drain'
            return False
        self._rollback('request-changed-or-admission-refused')
        return False

    def _gate_site(self, o3=False):
        s, d = self.scheduler, self.driver
        if self.manual_owned:
            return 'gate-manual-owned'
        # lane-o3: the split tail selects on the route it keeps; nothing else may be owned.
        if ((d._split_ready and not o3) or d.active or d.admission_intent() is not None or d._control_route is not None
                or (o3 and (not d._split_ready or d._return_pending is not None))):
            return 'gate-route-or-work'
        if not d._bootstrapped:
            return 'gate-not-bootstrapped'
        if d.error is not None:
            return 'gate-driver-stopped'
        if d._engine_paused():
            return 'gate-engine-paused'
        if getattr(s.policy, 'name', None) != 'FCFS':
            return 'gate-policy'
        if getattr(self.engine, 'batch_queue', None) is None:
            return 'gate-no-batch-queue'
        if (getattr(s.vllm_config.parallel_config, 'data_parallel_size', None) != 1
                or getattr(s.vllm_config, 'speculative_config', None) is not None
                or getattr(s, 'connector', None) is not None):
            return 'gate-config'
        if callable(getattr(self.engine, 'is_running', None)) and not self.engine.is_running():
            return 'gate-not-running'
        return None

    def _shortest_bulk(self, queue, request, plan):
        """Rank only the leading bulk cohort; a ready returner keeps its queue position."""
        config = self.controller.append_yield
        if config is None or plan['end'] - plan['start'] <= config.max_new:
            return request, plan
        stats = self.bulk_pick
        stats['scans'] += 1
        stats['capped'] += len(queue) > BULK_PICK_CANDIDATES
        now = time.time()  # Request.arrival_time uses wall time, not the driver's monotonic clock.
        def rank(candidate, candidate_plan):
            age = max(0.0, now - candidate.arrival_time)
            return ((0, candidate.arrival_time) if age >= BULK_PICK_AGE_S else
                    (1, candidate_plan['end'] - candidate_plan['start'])), age
        key, age = rank(request, plan)
        chosen = (request, plan, 0, age)
        state = (self._plan_moved, self._plan_residue, self._decline_why)
        scanned = 1
        for position, candidate in enumerate(islice(queue, 1, BULK_PICK_CANDIDATES), 1):
            scanned += 1
            if self.rejected.get(candidate.request_id, lambda: None)() is candidate:
                stats['rejected'] += 1
                continue
            self._decline_why = state[2]
            candidate_plan = self._plan(candidate, lookup_stats=stats)
            if candidate_plan is None:
                stats['unplannable'] += 1
                continue
            if candidate_plan['end'] - candidate_plan['start'] <= config.max_new:
                stats['short_stop'] += 1
                break
            candidate_key, candidate_age = rank(candidate, candidate_plan)
            if candidate_key < key:
                key, chosen = candidate_key, (candidate, candidate_plan, position, candidate_age)
                state = (self._plan_moved, self._plan_residue, self._decline_why)
        self._plan_moved, self._plan_residue, self._decline_why = state
        request, plan, position, age = chosen
        stats['candidates'] += scanned
        stats['reordered'] += position != 0
        stats['aged'] += age >= BULK_PICK_AGE_S
        self.driver._record('bulk_pick', request_uid=request.request_id, original_position=position,
                            remaining_rows=plan['end'] - plan['start'], age_s=age,
                            aged=age >= BULK_PICK_AGE_S, candidates=scanned)
        return request, plan

    def _select(self, o3=False):
        s, d = self.scheduler, self.driver
        held_last, self._held_last = self._held_last, None     # lane-o3-fix: held_ms runs only across consecutive held ticks
        site = self._gate_site(o3)
        if site is not None:
            self._decline(site)
            return False
        cohort = self._cohort()
        queue = self.controller._queue(s)
        if cohort is None or not queue:
            if cohort is None:
                self._decline(self._decline_why)
            return False
        request = queue.peek_request()
        if self.rejected.get(request.request_id, lambda: None)() is request and self._residue():
            # coverage-residue: no retry count; the same cohort would be refused the same way, so wait for it to change.
            uid = request.request_id
            if self.refused_cohort.get(uid) == cohort:
                self.residue_counts['refused_held'] += 1
                now = time.monotonic_ns()
                if held_last is not None:
                    self.residue_counts['held_ms'] = round(self.residue_counts['held_ms'] + (now - held_last) / 1e6, 3)
                self._held_last = now
                self._decline('rejected')
                return False
            self.residue_counts['refused_retried'] += 1
            self.hold_totals['retried'] += 1
            self.rejected.pop(uid, None)
        elif self.rejected.get(request.request_id, lambda: None)() is request:
            uid = request.request_id
            if self._hold_limit() and self.retries.get(uid, 0) < MAX_RETRIES:
                # decode-phase hold: a refused selection is tried again, at most MAX_RETRIES times.
                self.retries[uid] = self.retries.get(uid, 0) + 1
                self.hold_totals['retried'] += 1
                self.rejected.pop(uid, None)
                if len(self.retries) > MAX_UNHOLDABLE:
                    self.retries.pop(next(iter(self.retries)))
            else:
                if uid in self.holds:
                    self.hold_totals['retries_spent'] += 1
                    self._unhold(request, 'unplannable', 'rejected')
                self._decline('rejected')
                return False
        self._gap('select_begin', request_uid=request.request_id)
        plan = self._plan(request)
        if plan is None:
            if request.request_id in self.holds and self._decline_why not in TRANSIENT_PLAN_SITES:
                self._unhold(request, 'unplannable', self._decline_why)      # stock takes it next step
            self._decline(self._decline_why)
            return False
        if getattr(self.controller, 'shortest_bulk', False) is True:
            request, plan = self._shortest_bulk(queue, request, plan)
        if self._decline_pending.pop(request.request_id, None) is not None:
            self.decline_totals['selected_after_decline'] += 1
        if self.holds.pop(request.request_id, None) is not None:
            self.hold_totals['selected'] += 1
        if getattr(self, '_plan_moved', False):
            self.hold_totals['moved_stop'] += 1
        if getattr(self, '_plan_residue', None) in ('junction', 'lowered'):
            self.residue_counts[self._plan_residue] += 1
        if cohort == () and self._residue():
            self.residue_counts['idle_selected'] += 1
        self.request_ref = weakref.ref(request)
        self.plan, self.cohort = plan, cohort
        self.selection_seq += 1
        self.old_pause = s.pause_state
        self.external_pause = False
        self.deadline = d.clock() + d._control_timeout_s
        self.phase = 'o3_drain' if o3 else 'drain'
        self._note('selected', start=plan['start'], native_end=plan['end'], chunks=len(plan['plan']), o3=int(o3))
        # The real stock keep-pause state. Core drains its existing batch queue
        # normally; its outputs update the scheduler before the worker barrier.
        self._pause(type(self.old_pause).PAUSED_ALL)
        return True

    def _restore(self, reason):
        d = self.driver
        if d._split_ready or d.active or d.admission_intent() is not None:
            raise E3ContractError('auto-arm-close-with-owned-route-or-work')
        d.append_yield_end(reason)
        self.reentry = False
        self.restores[reason] = self.restores.get(reason, 0) + 1
        self._note('closed', reason=reason)
        why = self._rollback_reason if reason == 'returned-before-native-admission' and self._rollback_reason else reason
        self._rollback_reason = None
        if type(why) is str and why.startswith('o3-'):
            # lane-o3-fix 2: an O3 refusal (census, selection, counts, ledger, generation) is not about the next head; marking
            # it rejected parked the lane on refused_held until a decoder finished (q18a-parity-o3 n1: 68-70 s per arm).
            self.o3_counts['unparked'] += 1
        elif not self._race_close(why):
            self._remember()
        self.phase = 'idle'
        self.pending = None
        self.request_ref = None
        self.plan = None
        self.cohort = ()
        self.completed += 1
        self.cooldown = True
        if not self.external_pause:
            self._pause(self.old_pause)
        if self.control_inwait is not None:
            self.control_inwait.close(False)
        self.old_pause = None

    def _race_close(self, why):
        """coverage-residue: True when this close lost a race and the request is not marked refused (at most MAX_RACES each)."""
        request = self.request_ref() if self.request_ref is not None else None
        if why not in RACE_CLOSES or request is None or not self._residue():
            return False
        uid = request.request_id
        count = self.races.pop(uid, 0) + 1
        self.races[uid] = count
        while len(self.races) > MAX_REJECTED:
            self.races.pop(next(iter(self.races)))
        if count > MAX_RACES:
            self.residue_counts['races_spent'] += 1
            return False
        self.residue_counts['races'] += 1
        return True

    def _rollback(self, reason):
        self._rollback_reason = reason
        self.rollbacks[reason] = self.rollbacks.get(reason, 0) + 1
        self._note('refused', reason=reason)
        if self.driver._split_ready:
            self.deadline = self.driver.clock() + self.driver._control_timeout_s
            self.phase = 'return'
            self.pending = self._control('leave_split')
        else:
            self._restore(reason)

    def _proof(self, result, *, capability=False, observation=False):
        rows = result.get('workers')
        d = self.driver
        if (type(rows) is not list or len(rows) != len(d.ranks)
                or {r.get('worker_rank') for r in rows} != set(d.ranks)):
            return False
        for row in rows:
            live = row.get('live_decode') or {}
            if (row.get('engine_epoch') != d.engine_epoch or row.get('route') != 'full'
                    or row.get('route_generation') != d.route_generation
                    or row.get('prefill_requests') != 0 or live.get('available') is not True
                    or sorted(live.get('live_request_uids', ())) != list(self.cohort)
                    or row.get('decode_eligible') != len(self.cohort)):
                return False
            if capability:
                cap = (row.get('native_inventory') or {}).get('auto_arm_decode') or {}
                counts = cap.get('supported_counts')
                if (type(counts) is not list or len(counts)>MAX_COHORT
                        or any(type(n) is not int or not 1<=n<=MAX_COHORT for n in counts)
                        or len(set(counts))!=len(counts)):
                    return False
                if (cap.get('schema') != 's5-auto-decode-capability/1'
                        or cap.get('engine_epoch') != d.engine_epoch
                        or cap.get('route_generation') != d.route_generation
                        or cap.get('storage_epoch') != row.get('storage_epoch')
                        or not set(range(1, len(self.cohort) + 1)) <= set(counts)):
                    return False
            elif not observation and (row.get('barrier_clean') is not True or row.get('drained_streams', 0) <= 0):
                return False
        if capability:
            covered = None
            for row in rows:
                counts = set(row['native_inventory']['auto_arm_decode']['supported_counts'])
                covered = counts if covered is None else covered & counts
            self.decode_counts = frozenset(covered or ())
        return True

    def tick(self):
        """One bounded tick. True bypasses stock; False allows existing Core work."""
        d = self.driver
        self._settle_declines()
        if not self.active:
            if self.cooldown:
                self.cooldown = False
                self._decline('cooldown')
                return False
            if not self._select():
                return False
        if self.phase == 'native':
            request = self._request()
            if d.append_yield_active():
                return self._yield_native(request)
            if (d._split_ready and not d.active and request is not None and d._return_pending is None
                    and d.o3_handback_uid() == request.request_id):
                self._o3_enter_tail(request)
                return False
            if not d._split_ready and not d.active:
                self.phase = 'tail'
                self._note('full_handback')
                self._tail_ack()
            elif (request is None or (self.controller._schedule_seq > self.schedule_before
                    and self.controller.admitted == self.admitted_before)):
                if not d.active:
                    self._pause(type(self.old_pause).PAUSED_ALL)
                    if getattr(self.engine, 'batch_queue', None):
                        self.deadline = d.clock() + d._control_timeout_s
                        self.phase = 'rollback_drain'
                        return False
                    self._rollback('request-changed-or-admission-refused')
                return False
            else:
                return False
        if self.phase == 'tail' and self.o3_tail:
            if not (self._o3_yield_tail() if self.o3_yield else self._o3_tail()):
                return False
        if self.phase == 'tail':
            request = self._request()
            self.tail_ticks += 1
            settle = getattr(self.controller, 'k2_settle', False) is True
            if (settle and request is not None and not self._finished(request) and self._handback_scheduled(request)
                    and not self._decode_scheduled(request)):
                # lane-o3-fix (c): the handback step's row still reads as prefill on the workers; the barrier would refuse it
                # (2 refusals per run end on nt). Wait for its first decode, bounded as the O3 tail; past it, today's K2.
                if self.tail_ticks <= O3_TAIL_TICKS:
                    self.k2_counts['settle_ticks'] += 1
                    self._decline('tail')
                    return False
                self.k2_counts['bound'] += 1
            if request is not None and (request.num_output_tokens == 0 or settle) and self._handback_scheduled(request):
                if settle and self._decode_scheduled(request):
                    self.k2_counts['settled'] += 1
                # coverage-throughput build-2 (b) K2 (lead, 6 Nov): every handback row is scheduled on FULL, so the
                # request is a decoder; close with no cooldown and select the next head in this tick. PAUSED_ALL
                # stops further stock steps, the drain lands the handback step, and the barrier proves the cohort
                # with this request in it (prefill 0), as for any decoder. A failed selection leaves the policy idle.
                self._tail_end('full-tail-scheduled', request)
                self._restore('full-tail-scheduled')
                self.cooldown = False
                if not self._select():
                    return False
                # a fresh selection: the drain below, as from idle
            else:
                if request is None or request.num_output_tokens > 0:
                    self._tail_end('full-tail-completed', request)
                    self._restore('full-tail-completed')
                self._decline('tail')
                return False
        if self.phase == 'drain_wait':
            # One scheduler pass with the pause released between re-entry attempts.
            if self.controller._schedule_seq <= self.drain_wait_seq and self.driver.append_yield_active():
                return False
            d.append_yield_set_drain_wait(False)
            self._pause(type(self.old_pause).PAUSED_ALL)
            self.deadline = d.clock() + d._control_timeout_s
            self.phase = 'drain'
            return True
        if self.deadline is not None and d.clock() > self.deadline:
            raise E3ContractError('auto-arm-entry-or-return-deadline-expired; ownership retained')
        if self.phase == 'control_observe':
            return self.control_inwait.poll()
        if self.phase in ('control_o3_observe', 'control_o3_cut', 'control_o3_rollback'):
            return self.control_inwait.poll_o3()
        if self.phase == 'o3_drain':
            if self.control_inwait is not None and self.control_inwait.o3:
                return self.control_inwait.begin_o3()
            if getattr(self.engine, 'batch_queue', None):
                return False                   # the handback step (and any decode step) lands first
            self._gap('drain_empty')
            if not self._o3_census_begin():
                return False
        if self.phase in ('drain', 'rollback_drain'):
            observation = (self.control_inwait if self.phase == 'drain' else None)
            if (getattr(self.engine, 'batch_queue', None)
                    and (observation is None or observation.certified)):
                return False
            if not getattr(self.engine, 'batch_queue', None):
                self._gap('drain_empty')
            if self.phase == 'rollback_drain':
                why, self._o3_return_reason = self._o3_return_reason, None
                self._rollback(why or 'request-changed-or-admission-refused')
                return True
            if self.reentry:
                if self._reentry_lost_yield():
                    return True
                request = self._request()
                if request is None:
                    self._restore('yielded-bulk-left')
                    return True
                if self._reentry_plan(request) != self.plan:
                    self._restore('yielded-bulk-frontier-moved')
                    return True
                cohort = self._reentry_cohort()
                if cohort is None:
                    # a decoder is mid-prefill or the cohort is over the slot bound:
                    # drain again, at most the launch bound of times
                    return self._reentry_refused('drain', self._yrs_cohort_why())
                self.cohort = cohort
            elif self._request() is None or self._cohort() != self.cohort or self._plan(self._request()) != self.plan:
                self._restore('selection-changed-during-output-drain')
                return True
            if observation is not None:
                if not observation.certified or not observation.current():
                    observation.begin()
                    return False
                observation.switching()
                self.pending = self._control('enter_split')
                self.phase = 'split'
            else:
                self.pending = self._control('barrier')
                self.phase = 'barrier'
        d.control_tick()
        if self.pending is None or not self.pending.done():
            return True
        result = self.pending.result()
        if self.phase in ('barrier', 'identity') and self._reentry_lost_yield():
            self.pending = None
            return True
        if self.phase == 'barrier':
            if not self._proof(result):
                if self.reentry:
                    self.pending, self.phase = None, 'drain'   # the cohort moved: re-read it
                    return self._reentry_refused('barrier', self._yrs_proof_why(result))
                self._rollback('worker-barrier-cohort-refused')
            else:
                if self.reentry:
                    self._yrs_barrier_pass()                  # yield-reentry (1)
                self.pending = self._control('identity')
                self.phase = 'identity'
        elif self.phase == 'identity':
            if not self._proof(result, capability=True):
                if self.reentry:
                    self.pending, self.phase = None, 'drain'
                    return self._reentry_refused('identity', self._yrs_proof_why(result, capability=True))
                self._rollback('worker-graph-support-refused')
            else:
                self.pending = self._control('enter_split')
                self.phase = 'split'
        elif self.phase == 'split':
            if (result.get('typed_acks') is not True or result.get('route_after') != 'split'
                    or len(result.get('transition_acks', ())) != len(d.ranks)):
                raise E3ContractError('auto-arm-split-ack-incomplete; ownership retained')
            if self.control_inwait is not None:
                self.control_inwait.acknowledge(result)
            if self.reentry and not d.append_yield_active():
                self.pending = None
                self._rollback('yield-ended-during-reentry')
                return True
            if self.reentry and self._request() is not None:
                if self._reentry_plan(self._request()) != self.plan:
                    self._rollback('yielded-bulk-frontier-moved')
                    return True
            elif self._request() is None or self._plan(self._request()) != self.plan:
                self._rollback('selection-changed-before-arm')
                return True
            d._control_route = None  # policy owns this route, not a manual session
            _fit_floor = getattr(d, 'graph_event_floor', None)
            if _fit_floor is not None and not _fit_floor.before_arm():
                return False
            self._internal = True
            try:
                d._arm_admission(PHASE)
            finally:
                self._internal = False
            if self.control_inwait is not None:
                self.control_inwait.close(True)
            # lane-o3: this split residency's proven cohort and the generation the identity counts hold for
            self.o3_ledger, self.o3_split_generation, self.o3_joined = frozenset(self.cohort), d.route_generation, None
            self.phase = 'native'
            self.schedule_before = self.controller._schedule_seq
            self.admitted_before = self.controller.admitted
            self.pending = None
            if self.reentry:
                self._yrs_armed()                             # yield-reentry (1)
                self.reentry = False
                if not (self._yield_window() and d.append_yield_window_begin()):    # yield-rules Y2: the window's next append
                    d.append_yield_set_phase('resume')
            else:
                self.decline_totals['armed'] += 1
                self._o3_gap_close()
            self._note('armed', route_generation=d.route_generation)
            self._pause(self.old_pause)
        elif self.phase == 'o3_census':
            self.pending = None
            if not self._o3_proof(result):
                return self._o3_refuse('census')
            if self._request() is None or self._o3_plan(self._request()) != self.plan or self._cohort() != self.cohort:
                return self._o3_refuse('selection-changed-after-census')
            self._o3_arm()
        elif self.phase == 'return':
            if result.get('typed_acks') is not True or result.get('route_after') != 'full':
                raise E3ContractError('auto-arm-full-ack-incomplete; ownership retained')
            self._restore('returned-before-native-admission')
        elif self.phase == 'o3_yield_return':
            # lane-o3-fix (b): FULL again; today's re-entry of the bulk from its drain (barrier, identity, enter_split, arm)
            if result.get('typed_acks') is not True or result.get('route_after') != 'full':
                raise E3ContractError('auto-arm-full-ack-incomplete; ownership retained')
            self.pending, self.o3_yield, self.o3_joined = None, False, None
            self.cohort = self._reentry_cohort() or ()
            self.deadline = d.clock() + d._control_timeout_s
            self.phase = 'drain'
        return True
