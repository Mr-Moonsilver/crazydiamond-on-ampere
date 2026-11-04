"""Opt-in, EngineCore-thread-only FCFS native-prefill policy (no device owner).

Scheduling remains paused until existing output Futures, the worker barrier,
and typed split acks settle. No new transport, CUDA operation or model setting.
"""
import os
import weakref

from e3_types import E3ContractError, _text
from scheduler_hooks import staged_chunk_plan, _ProjectedPrefillRequest

ENV = 'FLASH_S5_AUTO_ARM'
PHASE = 'auto-prefill'
MIN_NATIVE_ROWS = 32768
MAX_COHORT = 63
MAX_REJECTED = 1024
RELEASE = 'release_auto_arm'
READS = frozenset(('status', 'snapshot', 'transition_acks', 'bootstrap_acks', 'overlap_evidence'))


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
        return self.phase == 'native' and request is self._request()

    def bound_intent(self, intent):
        request = self._request()
        return (self.phase == 'native' and request is not None
                and intent.get('origin') == 'auto-arm' and intent.get('phase') == PHASE
                and intent.get('selection_seq') == self.selection_seq
                and intent.get('request_uid') == request.request_id)

    def _remember(self):
        request = self.request_ref() if self.request_ref is not None else None
        if request is not None:
            self.rejected[request.request_id] = weakref.ref(request)
        self.rejected = {k: ref for k, ref in self.rejected.items() if ref() is not None}
        while len(self.rejected) > MAX_REJECTED:
            self.rejected.pop(next(iter(self.rejected)))

    def _cohort(self):
        running = tuple(self.scheduler.running)
        if (not 1 <= len(running) <= min(MAX_COHORT, self.scheduler.max_num_running_reqs - 1)
                or any(r.is_prefill_chunk or r.num_computed_tokens < r.num_prompt_tokens
                       or r.lora_request is not None or r.resumable for r in running)):
            return None
        try:
            for r in running:_text('request_uid', r.request_id)
        except E3ContractError:
            return None
        return tuple(sorted(r.request_id for r in running))

    def _plan(self, request):
        s, c = self.scheduler, self.controller
        if (request is None or s.requests.get(request.request_id) is not request
                or getattr(request.status, 'name', None) != 'WAITING'
                or request.num_computed_tokens != 0 or request.num_output_tokens != 0
                or request.num_preemptions != 0 or request.resumable
                or request.lora_request is not None or request.pooling_params is not None
                or getattr(request, 'structured_output_request', None) is not None
                or c._request_refuses(s, request) or request.ec_transfer_params
                or request.sampling_params.prompt_logprobs is not None
                or _generation_logprobs_refused(request.sampling_params)):
            return None
        try:
            _text('request_uid', request.request_id)
        except E3ContractError:
            return None
        blocks, start, shared = s.kv_cache_manager.get_computed_blocks(request)
        end = request.num_tokens - 1
        # Shared-prefix junction/streaming/recompute special cases stay stock.
        if shared or end - start < max(MIN_NATIVE_ROWS, c.chunk_tokens):
            return None
        manager = s.kv_cache_manager
        needed = manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id, num_tokens=request.num_tokens,
            new_computed_blocks=blocks.blocks, num_encoder_tokens=0,
            total_computed_tokens=start, num_local_computed_tokens=start,
            num_tokens_main_model=request.num_tokens, apply_admission_cap=True)
        if needed + manager.watermark_blocks > manager.block_pool.get_num_free_blocks():
            return None
        cap = min(c.chunk_tokens, s.max_num_scheduled_tokens)
        def split(position, limit):
            projected = request if position == start else _ProjectedPrefillRequest(request, position)
            return c._aligned_split(s, projected, limit,
                num_new_local_computed_tokens=start if position == start else 0)
        plan = staged_chunk_plan(start, end, cap, c.minimum_stage_rows, split)
        if not plan:
            return None
        return dict(start=start, end=end, plan=tuple(plan),
                    blocks=tuple(tuple(b.block_id for b in group) for group in blocks.blocks))

    def _select(self):
        s, d = self.scheduler, self.driver
        if (self.manual_owned or d._split_ready or d.active or d.admission_intent() is not None
                or d._control_route is not None or not d._bootstrapped or d.error is not None
                or d._engine_paused() or getattr(s.policy, 'name', None) != 'FCFS'
                or getattr(self.engine, 'batch_queue', None) is None
                or getattr(s.vllm_config.parallel_config, 'data_parallel_size', None) != 1
                or getattr(s.vllm_config, 'speculative_config', None) is not None
                or getattr(s, 'connector', None) is not None
                or (callable(getattr(self.engine, 'is_running', None)) and not self.engine.is_running())):
            return False
        cohort = self._cohort()
        queue = self.controller._queue(s)
        if cohort is None or not queue:
            return False
        request = queue.peek_request()
        if self.rejected.get(request.request_id, lambda: None)() is request:
            return False
        plan = self._plan(request)
        if plan is None:
            return False
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
        return True

    def tick(self):
        """One bounded tick. True bypasses stock; False allows existing Core work."""
        d = self.driver
        if not self.active:
            if self.cooldown:
                self.cooldown = False
                return False
            if not self._select():
                return False
        if self.phase == 'native':
            request = self._request()
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
            return False
        if self.deadline is not None and d.clock() > self.deadline:
            raise E3ContractError('auto-arm-entry-or-return-deadline-expired; ownership retained')
        if self.phase in ('drain', 'rollback_drain'):
            if getattr(self.engine, 'batch_queue', None):
                return False
            if self.phase == 'rollback_drain':
                self._rollback('request-changed-or-admission-refused')
                return True
            if self._request() is None or self._cohort() != self.cohort or self._plan(self._request()) != self.plan:
                self._restore('selection-changed-during-output-drain')
                return True
            self.pending = self._control('barrier')
            self.phase = 'barrier'
        d.control_tick()
        if self.pending is None or not self.pending.done():
            return True
        result = self.pending.result()
        if self.phase == 'barrier':
            if not self._proof(result):
                self._rollback('worker-barrier-cohort-refused')
            else:
                self.pending = self._control('identity')
                self.phase = 'identity'
        elif self.phase == 'identity':
            if not self._proof(result, capability=True):
                self._rollback('worker-graph-support-refused')
            else:
                self.pending = self._control('enter_split')
                self.phase = 'split'
        elif self.phase == 'split':
            if (result.get('typed_acks') is not True or result.get('route_after') != 'split'
                    or len(result.get('transition_acks', ())) != len(d.ranks)):
                raise E3ContractError('auto-arm-split-ack-incomplete; ownership retained')
            if self._request() is None or self._plan(self._request()) != self.plan:
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
            self._note('armed', route_generation=d.route_generation)
            self.scheduler.set_pause_state(self.old_pause)
        elif self.phase == 'return':
            if result.get('typed_acks') is not True or result.get('route_after') != 'full':
                raise E3ContractError('auto-arm-full-ack-incomplete; ownership retained')
            self._restore('returned-before-native-admission')
        return True
