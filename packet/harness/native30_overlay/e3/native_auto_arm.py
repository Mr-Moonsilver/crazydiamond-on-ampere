"""Opt-in, EngineCore-thread-only FCFS native-prefill policy (no device owner).

Scheduling remains paused until existing output Futures, the worker barrier,
and typed split acks settle. No new transport, CUDA operation or model setting.
"""
import os
import weakref

from e3_types import E3ContractError, _text
from scheduler_hooks import bounded_limit, native_chunk_cap, staged_chunk_plan, _ProjectedPrefillRequest

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
    driver.auto_arm = policy
    return policy


def worker_capability(native):
    """Pure host dispatch against actual stock keys and captured native graphs.

    Cover every shrinking subset, including stock padding; never guess that a
    graph for the current full cohort covers later batches. No replay/readback.
    """
    from graph_router import ExecutionDescriptor
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


class AutoArmPolicy:
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

    @property
    def active(self):
        return self.phase != 'idle'

    def snapshot(self):
        request = self.request_ref() if self.request_ref is not None else None
        return dict(schema='s5-auto-arm/1', phase=self.phase, active=self.active,
                    manual_owned=self.manual_owned, selection_seq=self.selection_seq,
                    request_uid=None if request is None else request.request_id,
                    completed=self.completed, minimum_native_rows=max(MIN_NATIVE_ROWS, self.controller.chunk_tokens),
                    max_cohort=MAX_COHORT)

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
                    **self.decline_totals)

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
        from engine_driver import _progress_logger
        _progress_logger().info('flash-s5 decline site=%s num_tokens=%d', site, request.num_tokens)

    def _control(self, action):
        self._internal = True
        try:
            return self.driver.submit_control(dict(action=action, engine_epoch=self.driver.engine_epoch))
        finally:
            self._internal = False

    def _note(self, event, **fields):
        self.driver._record('auto_' + event, selection_seq=self.selection_seq,
                            request_uid=self.snapshot()['request_uid'], **fields)

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
        self.rejected = {k: ref for k, ref in self.rejected.items() if ref() is not None}
        while len(self.rejected) > MAX_REJECTED:
            self.rejected.pop(next(iter(self.rejected)))

    def _cohort(self):
        running = tuple(self.scheduler.running)
        if not running:
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

    def _plan(self, request):
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
        blocks, start, shared = s.kv_cache_manager.get_computed_blocks(request)
        end = request.num_tokens - 1
        # Shared-prefix junction/streaming/recompute special cases stay stock.
        if shared:
            self._decline_why = 'plan-shared'
            return None
        if end - start < max(MIN_NATIVE_ROWS, c.chunk_tokens):
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
        def split(position, limit):
            limit = bounded_limit(getattr(c, 'chunk_bound', None), position, limit)
            projected = request if position == start else _ProjectedPrefillRequest(request, position)
            return c._aligned_split(s, projected, limit,
                num_new_local_computed_tokens=start if position == start else 0)
        plan = staged_chunk_plan(start, end, cap, c.minimum_stage_rows, split)
        if not plan:
            self._decline_why = 'plan-empty'
            return None
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
        start, end = request.num_computed_tokens, request.num_tokens - 1
        if type(start) is not int or start <= 0 or end - start < 1:
            return None
        cap = native_chunk_cap(c.chunk_tokens, s.max_num_scheduled_tokens)    # the cap admission plans with
        def split(position, limit):
            limit = bounded_limit(getattr(c, 'chunk_bound', None), position, limit)
            projected = request if position == start else _ProjectedPrefillRequest(request, position)
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
            # auto-arm selection; the bulk continues on stock.
            self._restore('yielded-bulk-has-no-reentry-plan')
            return False
        if d.append_yield_active():
            d.append_yield_set_phase('reenter')
        self.plan, self.reentry = plan, True
        self.reentry_refusals = 0
        self.drain_wait_seq = None
        self.cohort = self._reentry_cohort() or ()
        self.deadline = d.clock() + d._control_timeout_s
        self.phase = 'drain'
        self._note('reentry', start=plan['start'], native_end=plan['end'], chunks=len(plan['plan']))
        self.scheduler.set_pause_state(type(self.old_pause).PAUSED_ALL)
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
        d.append_yield_note_handback()
        return self._begin_reentry(request)

    def _reentry_refused(self, stage):
        """One refused re-entry drain or proof; past the launch bound, close on FULL.

        Otherwise the next attempt waits one scheduler pass with the pause
        released (delta review of ae865d44, D-MINOR-2): back-to-back attempts
        under PAUSED_ALL could not see any change.  The route is FULL with no
        intent; the yield's hold keeps every stock prefill held but a decoder's
        running prefill, which may finish its chunk in that pass.
        """
        self.reentry_refusals += 1
        if self.driver.append_yield_reentry_refused(stage, self.reentry_refusals):
            self._restore('yielded-bulk-reentry-refused')
            return True
        self.drain_wait_seq = self.controller._schedule_seq
        self.driver.append_yield_set_drain_wait(True)
        self.scheduler.set_pause_state(self.old_pause)
        self.phase = 'drain_wait'
        return True

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
        self.scheduler.set_pause_state(type(self.old_pause).PAUSED_ALL)
        if getattr(self.engine, 'batch_queue', None):
            self.deadline = self.driver.clock() + self.driver._control_timeout_s
            self.phase = 'rollback_drain'
            return False
        self._rollback('request-changed-or-admission-refused')
        return False

    def _gate_site(self):
        s, d = self.scheduler, self.driver
        if self.manual_owned:
            return 'gate-manual-owned'
        if d._split_ready or d.active or d.admission_intent() is not None or d._control_route is not None:
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

    def _select(self):
        s, d = self.scheduler, self.driver
        site = self._gate_site()
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
        if self.rejected.get(request.request_id, lambda: None)() is request:
            self._decline('rejected')
            return False
        plan = self._plan(request)
        if plan is None:
            self._decline(self._decline_why)
            return False
        if self._decline_pending.pop(request.request_id, None) is not None:
            self.decline_totals['selected_after_decline'] += 1
        self.request_ref = weakref.ref(request)
        self.plan, self.cohort = plan, cohort
        self.selection_seq += 1
        self.old_pause = s.pause_state
        self.external_pause = False
        self.deadline = d.clock() + d._control_timeout_s
        self.phase = 'drain'
        self._note('selected', start=plan['start'], native_end=plan['end'], chunks=len(plan['plan']))
        # The real stock keep-pause state. Core drains its existing batch queue
        # normally; its outputs update the scheduler before the worker barrier.
        s.set_pause_state(type(self.old_pause).PAUSED_ALL)
        return True

    def _restore(self, reason):
        d = self.driver
        if d._split_ready or d.active or d.admission_intent() is not None:
            raise E3ContractError('auto-arm-close-with-owned-route-or-work')
        d.append_yield_end(reason)
        self.reentry = False
        self.restores[reason] = self.restores.get(reason, 0) + 1
        self._note('closed', reason=reason)
        self._remember()
        self.phase = 'idle'
        self.pending = None
        self.request_ref = None
        self.plan = None
        self.cohort = ()
        self.completed += 1
        self.cooldown = True
        if not self.external_pause:
            self.scheduler.set_pause_state(self.old_pause)
        self.old_pause = None

    def _rollback(self, reason):
        self.rollbacks[reason] = self.rollbacks.get(reason, 0) + 1
        self._note('refused', reason=reason)
        if self.driver._split_ready:
            self.deadline = self.driver.clock() + self.driver._control_timeout_s
            self.phase = 'return'
            self.pending = self._control('leave_split')
        else:
            self._restore(reason)

    def _proof(self, result, *, capability=False):
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
            elif row.get('barrier_clean') is not True or row.get('drained_streams', 0) <= 0:
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
            if not d._split_ready and not d.active:
                self.phase = 'tail'
                self._note('full_handback')
            elif (request is None or (self.controller._schedule_seq > self.schedule_before
                    and self.controller.admitted == self.admitted_before)):
                if not d.active:
                    self.scheduler.set_pause_state(type(self.old_pause).PAUSED_ALL)
                    if getattr(self.engine, 'batch_queue', None):
                        self.deadline = d.clock() + d._control_timeout_s
                        self.phase = 'rollback_drain'
                        return False
                    self._rollback('request-changed-or-admission-refused')
                return False
            else:
                return False
        if self.phase == 'tail':
            request = self._request()
            if request is None or request.num_output_tokens > 0:
                self._restore('full-tail-completed')
            self._decline('tail')
            return False
        if self.phase == 'drain_wait':
            # One scheduler pass with the pause released between re-entry attempts.
            if self.controller._schedule_seq <= self.drain_wait_seq and self.driver.append_yield_active():
                return False
            d.append_yield_set_drain_wait(False)
            self.scheduler.set_pause_state(type(self.old_pause).PAUSED_ALL)
            self.deadline = d.clock() + d._control_timeout_s
            self.phase = 'drain'
            return True
        if self.deadline is not None and d.clock() > self.deadline:
            raise E3ContractError('auto-arm-entry-or-return-deadline-expired; ownership retained')
        if self.phase in ('drain', 'rollback_drain'):
            if getattr(self.engine, 'batch_queue', None):
                return False
            if self.phase == 'rollback_drain':
                self._rollback('request-changed-or-admission-refused')
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
                    return self._reentry_refused('drain')
                self.cohort = cohort
            elif self._request() is None or self._cohort() != self.cohort or self._plan(self._request()) != self.plan:
                self._restore('selection-changed-during-output-drain')
                return True
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
                    return self._reentry_refused('barrier')
                self._rollback('worker-barrier-cohort-refused')
            else:
                self.pending = self._control('identity')
                self.phase = 'identity'
        elif self.phase == 'identity':
            if not self._proof(result, capability=True):
                if self.reentry:
                    self.pending, self.phase = None, 'drain'
                    return self._reentry_refused('identity')
                self._rollback('worker-graph-support-refused')
            else:
                self.pending = self._control('enter_split')
                self.phase = 'split'
        elif self.phase == 'split':
            if (result.get('typed_acks') is not True or result.get('route_after') != 'split'
                    or len(result.get('transition_acks', ())) != len(d.ranks)):
                raise E3ContractError('auto-arm-split-ack-incomplete; ownership retained')
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
            self._internal = True
            try:
                d._arm_admission(PHASE)
            finally:
                self._internal = False
            self.phase = 'native'
            self.schedule_before = self.controller._schedule_seq
            self.admitted_before = self.controller.admitted
            self.pending = None
            if self.reentry:
                self.reentry = False
                d.append_yield_set_phase('resume')
            else:
                self.decline_totals['armed'] += 1
            self._note('armed', route_generation=d.route_generation)
            self.scheduler.set_pause_state(self.old_pause)
        elif self.phase == 'return':
            if result.get('typed_acks') is not True or result.get('route_after') != 'full':
                raise E3ContractError('auto-arm-full-ack-incomplete; ownership retained')
            self._restore('returned-before-native-admission')
        return True
