"""Owner-only decode diagnostic control over the existing AsyncLLM client.

No dev-router switch, new executor, engine lifecycle control or arbitrary RPC.
HTTP registration is lazy and default-off. A timeout may leave remote work in
flight: quarantine blocks resume/transition. A separate, explicitly read-only
status (``read_only: True``) stays available after quarantine without consuming
the mutation sequence or clearing quarantine.
"""
import asyncio
import copy
import hashlib
import ipaddress
import json
import math
import secrets
import time


class ProbeControlError(RuntimeError):
    pass


RANKS = (0, 1, 2, 3)
PATH = '/v1/flash_s5_decode_probe'
NATIVE_PATH = '/v1/flash_s5_native_overlap'
NATIVE_SESSION_ACTIONS = ('prefill_begin', 'prefill_end', 'identity', 'pause', 'prepare', 'pause_enter_split', 'overlap_begin',
                          'overlap_end', 'leave_full', 'resume', 'status',
                          'admission', 'arm_admission', 'disarm_admission')
ACTIONS = ('status', 'pause', 'transition', 'resume', 'observer_begin', 'observer_end')
READ_ONLY_FIELDS = frozenset(('action', 'run_id', 'read_only'))


def _positive(value):
    return type(value) is int and value > 0


def _label(value):
    return (type(value) is str and 1 <= len(value) <= 80
            and all(c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in value))


def _bounded(value, depth=0):
    """Copy only bounded scalar metadata; never exception/payload material."""
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        return value if -(2 ** 63) < value < 2 ** 63 else None
    if type(value) is float:
        return value if math.isfinite(value) else None
    if type(value) is str:
        return value[:200]
    if depth >= 4:
        return None
    if type(value) is list or type(value) is tuple:
        return [_bounded(item, depth + 1) for item in list(value)[:32]]
    if type(value) is dict:
        return {str(key)[:64]: _bounded(item, depth + 1)
                for key, item in list(value.items())[:64]}
    return None


def _rank_map(rows):
    if type(rows) is not list or len(rows) != len(RANKS):
        raise ProbeControlError('incomplete-rank-coverage')
    mapped = {}
    for row in rows:
        if type(row) is not dict or type(row.get('rank')) is not int or row['rank'] not in RANKS or row['rank'] in mapped:
            raise ProbeControlError('invalid-rank-coverage')
        mapped[row['rank']] = row
    return mapped


def _observation(row, rank):
    """Actual worker metadata if present; an explicit unknown otherwise.

    Never substitutes a cached all-rank success: a missing/foreign/failed row is
    reported as unknown with its actual rank, never filled in from another rank.
    """
    if type(row) is not dict or type(row.get('rank')) is not int:
        return {'rank': rank, 'status': 'unknown', 'reason': 'missing-or-invalid-worker'}
    if row['rank'] != rank:
        return {'rank': rank, 'status': 'unknown', 'reason': 'foreign-worker-row'}
    out = {'rank': rank, 'status': 'ok'}
    counters = row.get('counters')
    for key in ('inventory', 'counters', 'memory'):
        if type(row.get(key)) is dict:
            out[key] = _bounded(row[key])
    if type(counters) is not dict:
        out['status'] = 'partial'
        out['reason'] = 'worker-counters-unavailable'
        return out
    if (counters.get('route') not in ('full', 'split')
            or type(counters.get('route_generation')) is not int
            or type(counters.get('storage_epoch')) is not int):
        out['status'] = 'partial'
        out['reason'] = 'worker-identity-unavailable'
    if counters.get('quarantined') is not None:
        out['status'] = 'quarantined'
        out['reason'] = _bounded(counters.get('quarantined'))
    return out


class ProbeController:
    def __init__(self, engine, *, timeout_s=120.0, epoch=None):
        core = getattr(engine, 'engine_core', None)
        if (type(getattr(core, 'client_count', None)) is not int or core.client_count != 1
                or type(getattr(core, 'client_index', None)) is not int or core.client_index != 0):
            raise ProbeControlError('diagnostic-requires-single-api-client')
        parallel = getattr(getattr(engine, 'vllm_config', None), 'parallel_config', None)
        if any(type(getattr(parallel, key, None)) is not int or getattr(parallel, key) != value
               for key, value in (('tensor_parallel_size', 2), ('pipeline_parallel_size', 2),
                                  ('data_parallel_size', 1))):
            raise ProbeControlError('diagnostic-requires-declared-four-worker-topology')
        self.engine = engine
        self.timeout_s = timeout_s
        self.epoch = (secrets.randbits(63) or 1) if epoch is None else epoch
        if not _positive(self.epoch):
            raise ProbeControlError('invalid-epoch')
        self.run_id = None
        self.seq = 0
        self.lock = asyncio.Lock()
        self.paused = False
        self.quarantined = False
        self.quarantine_reason = None
        self.eligibility_failure = None
        self.last = None
        self.ingress = AdmissionSeal(engine)

    async def _rpc(self, command):
        return await self.engine.collective_rpc('flash_s5_decode_probe',
            timeout=self.timeout_s, args=(command,), kwargs={})

    async def configure_teacher(self, plan):
        from teacher_decode import FrontendPlan, ROW_BASIS, ROW_SCHEMA, TeacherDecodeError, validate_plan
        if plan is not None:
            try: validate_plan(plan)
            except TeacherDecodeError as exc: raise ProbeControlError(str(exc)) from None
        owner = getattr(self, '_teacher_plan', None)
        if plan is None and owner is None:
            return None
        await self.ingress.seal()
        if self.engine.get_num_unfinished_requests() != 0:
            raise ProbeControlError('teacher-config-needs-empty-engine')
        await self._barrier(await self._snapshot())
        replies = _rank_map(await self._rpc({'action': 'teacher_configure',
                                           'engine_epoch': self.epoch,
                                           'enabled': plan is not None,
                                           'snapshot_max_bytes':plan.get('snapshot_max_bytes',2*1024**3) if plan else 2*1024**3}))
        for rank, row in replies.items():
            if (row.get('ack') != 'flash_s5_teacher_configure'
                    or row.get('engine_epoch') != self.epoch
                    or row.get('enabled') is not (plan is not None)
                    or row.get('sampler_owner') is not (rank >= 2)
                    or type(row.get('table_bytes')) is not int
                    or row['table_bytes'] < 0
                    or not _positive(row.get('drained_streams'))
                    or row.get('snapshot_max_bytes') != (plan.get('snapshot_max_bytes',2*1024**3) if plan else 2*1024**3)
                    or (plan is not None and rank>=2 and row['table_bytes']==0)
                    or row.get('row_schema') != (ROW_SCHEMA if plan is not None and rank>=2 else None)
                    or row.get('row_basis') != (ROW_BASIS if plan is not None and rank>=2 else None)):
                raise ProbeControlError('teacher-worker-configuration-unproved')
        try:
            if owner is None:
                owner = FrontendPlan(self.engine.input_processor)
                self._teacher_plan = owner
            evidence = owner.configure(plan)
        except TeacherDecodeError as exc:
            raise ProbeControlError(str(exc)) from None
        return dict(evidence, ranks=[replies[r] for r in RANKS])

    async def launch_sample(self,phase):
        if not self.paused:raise ProbeControlError('launch-sample-needs-pause')
        replies=_rank_map(await self._rpc({'action':'launch_'+phase,'engine_epoch':self.epoch}))
        for r in replies.values():
            if r.get('ack')!='flash_s5_launch_'+phase or r.get('engine_epoch')!=self.epoch or not _positive(r.get('drained_streams')):
                raise ProbeControlError('launch-sample-ack-incomplete')
        return True

    async def route_invariant(self, phase):
        owner=getattr(self,'_teacher_plan',None)
        if owner is None or owner.pending is None:
            return None
        if not self.paused: raise ProbeControlError('route-invariant-needs-pause')
        replies=_rank_map(await self._rpc({'action':'route_invariant','engine_epoch':self.epoch,'phase':phase}))
        for row in replies.values():
            if (row.get('ack')!='flash_s5_route_invariant' or row.get('engine_epoch')!=self.epoch
                    or row.get('phase')!=phase or not _positive(row.get('drained_streams'))
                    or row.get('verdict') not in ('PASS','FAIL','INCONCLUSIVE')):
                raise ProbeControlError('route-invariant-evidence-invalid')
        verdicts=[r['verdict'] for r in replies.values()]
        return {'schema':1,'phase':phase,'verdict':'FAIL' if 'FAIL' in verdicts else
                'PASS' if all(v=='PASS' for v in verdicts) else 'INCONCLUSIVE',
                'ranks':[replies[r] for r in RANKS]}

    async def teacher_evidence(self):
        from teacher_decode import ROW_BASIS, ROW_SCHEMA
        owner = getattr(self, '_teacher_plan', None)
        if owner is None or owner.pending is None:
            return None
        if not self.paused:
            raise ProbeControlError('teacher-evidence-needs-pause')
        replies = _rank_map(await self._rpc({'action': 'teacher_evidence', 'engine_epoch': self.epoch}))
        for rank, row in replies.items():
            if (row.get('ack') != 'flash_s5_teacher_evidence'
                    or row.get('engine_epoch') != self.epoch
                    or row.get('sampler_owner') is not (rank >= 2)
                    or not _positive(row.get('drained_streams'))
                    or type(row.get('positions')) is not list or len(row['positions']) > 2048
                    or type(row.get('calls')) is not list or len(row['calls']) > 2048
                    or row.get('row_schema') != (ROW_SCHEMA if rank>=2 else None)
                    or row.get('row_basis') != (ROW_BASIS if rank>=2 else None)
                    or type(row.get('capture_defects')) is not list
                    or type(row.get('call_overflow')) is not bool
                    or type(row.get('overflow')) is not bool):
                raise ProbeControlError('teacher-worker-evidence-unproved')
        return dict(owner.status(), ranks=[replies[r] for r in RANKS])

    async def _snapshot(self):
        rows = await self._rpc({'action': 'status'})
        mapped = _rank_map(rows)
        identities = set()
        for row in mapped.values():
            inv, ctr = row.get('inventory'), row.get('counters')
            if type(inv) is not dict or type(ctr) is not dict:
                raise ProbeControlError('missing-worker-status')
            route, generation, storage = ctr.get('route'), ctr.get('route_generation'), ctr.get('storage_epoch')
            if (route not in ('full', 'split') or type(generation) is not int or generation < 0
                    or type(storage) is not int or storage != 1
                    or ctr.get('mode') != 'decode-only-probe' or ctr.get('state') != 'ready'
                    or inv.get('mode') != 'decode-only-probe'
                    or any(inv.get(k) != ctr.get(k) for k in ('route', 'route_generation', 'storage_epoch'))
                    or ctr.get('quarantined') is not None):
                raise ProbeControlError('invalid-worker-status')
            identities.add((route, generation, storage))
        if len(identities) != 1:
            raise ProbeControlError('mixed-worker-route')
        route, generation, storage = identities.pop()
        return {'route': route, 'route_generation': generation, 'storage_epoch': storage,
                'ranks': [copy.deepcopy(mapped[r]) for r in RANKS]}

    async def _barrier(self, snapshot):
        rows = await self._rpc({'action': 'barrier', 'engine_epoch': self.epoch})
        mapped = _rank_map(rows)
        for row in mapped.values():
            if (row.get('ack') != 'flash_s5_probe_barrier'
                    or row.get('mode') != 'decode-only-probe'
                    or type(row.get('engine_epoch')) is not int or row['engine_epoch'] != self.epoch
                    or not _positive(row.get('drained_streams'))
                    or any(type(row.get(k)) is not int for k in ('route_generation', 'storage_epoch'))
                    or any(row.get(k) != snapshot[k] for k in ('route', 'route_generation', 'storage_epoch'))):
                raise ProbeControlError('invalid-worker-barrier')
        return [copy.deepcopy(mapped[r]) for r in RANKS]

    def _idle(self, snapshot):
        for row in snapshot['ranks']:
            ctr = row['counters']
            if (type(ctr.get('in_flight')) is not int or ctr['in_flight'] != 0
                    or ctr.get('sampling_pending') is not False
                    or ctr.get('transition_pending') is not False):
                raise ProbeControlError('worker-still-active')

    def _eligibility(self, snapshot, cohort_size):
        if not _positive(cohort_size):
            raise ProbeControlError('invalid-cohort-size')
        # Actual live output-processor state, not an owner-supplied boolean.
        unfinished = self.engine.get_num_unfinished_requests()
        queued = self.engine.get_num_queued_tokens()
        if (type(unfinished) is not int or unfinished != cohort_size
                or type(queued) is not int or queued != 0):
            raise ProbeControlError('cohort-not-decode-only')
        observed = []
        for row in snapshot['ranks']:
            batch = row['counters'].get('batch_observations', {}).get('last')
            live = row['counters'].get('live_decode')
            batch_ok = (type(batch) is dict and _positive(batch.get('sequence'))
                and batch.get('decode_only') is True and batch.get('full_graph') is True
                and _positive(batch.get('num_reqs')) and batch['num_reqs'] <= cohort_size
                and type(batch.get('num_tokens')) is int and batch['num_tokens'] == batch['num_reqs'])
            partial = batch_ok and batch['num_reqs'] < cohort_size
            live_ok = (type(live) is dict and live.get('basis') == 'worker-cpu-live-decode-v1'
                and live.get('available') is True and type(live.get('live_requests')) is int
                and live['live_requests'] == cohort_size and type(live.get('prefill_requests')) is int
                and live['prefill_requests'] == 0 and live.get('batch_is_live_subset') is True
                and type(live.get('last_batch_member_count')) is int
                and batch_ok and live['last_batch_member_count'] == batch['num_reqs'])
            if not batch_ok or (partial and not live_ok):
                # Preserve the initiating observation before client disconnect
                # cleanup changes worker rows. No IDs, values or payload reprs.
                self.eligibility_failure = {
                    'rank': row['rank'], 'cohort_size': cohort_size,
                    'unfinished': unfinished, 'queued': queued,
                    'batch': {k: batch.get(k) if type(batch.get(k)) in (int, bool) else None
                              for k in ('sequence','num_reqs','num_tokens','decode_only','full_graph')}
                             if type(batch) is dict else None,
                    'live_decode': {k: live.get(k) if type(live.get(k)) in (int,bool) else None
                                    for k in ('available','live_requests','prefill_requests',
                                              'batch_is_live_subset','last_batch_member_count')}
                                   if type(live) is dict else None}
                raise ProbeControlError('worker-decode-eligibility-unproven')
            observed.append({'rank': row['rank'], 'batch': copy.deepcopy(batch),
                             'live_decode': copy.deepcopy(live) if live_ok else None})
        return {'cohort_size': cohort_size, 'proved': True,
                'basis': 'sealed live cohort with one-token full-graph batch; partial PP/parity batches additionally require fresh CPU live-row decode census',
                'ranks': observed}

    async def _observe(self):
        """Read-only, owner-bound observation that survives quarantine.

        Never raises for a worker/transport failure and never invents a zero or
        an all-rank success. The mutation sequence and prior receipt are not
        read or written.
        """
        started = time.monotonic_ns()
        rows = None
        failure = None
        try:
            rows = await asyncio.wait_for(self._rpc({'action': 'status'}), self.timeout_s)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - observation reports failure
            failure = type(exc).__name__
        # Query the actual awaited pause state.  A partial failure can complete
        # pause_generation() and then fail the barrier before self.paused is
        # updated, so the cached acknowledgement must never stand in for the
        # engine's current state.  Bounded by the existing configured timeout.
        paused = None
        pause_state = 'unknown'
        pause_failure = None
        try:
            actual_paused = await asyncio.wait_for(self.engine.is_paused(), self.timeout_s)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - observation reports failure
            pause_failure = type(exc).__name__
        else:
            if actual_paused is True:
                paused, pause_state = True, 'paused'
            elif actual_paused is False:
                paused, pause_state = False, 'running'
            else:
                pause_failure = 'pause-state-not-boolean'
        by_rank = {}
        foreign = []
        if type(rows) is list:
            for row in rows:
                if type(row) is dict and type(row.get('rank')) is int:
                    rank = row['rank']
                    if rank in RANKS and rank not in by_rank:
                        by_rank[rank] = row
                    elif rank in RANKS:
                        foreign.append(rank)
                    else:
                        foreign.append(rank if abs(rank) <= 63 else 'foreign')
        workers = [_observation(by_rank.get(rank), rank) for rank in RANKS]
        available = sum(1 for row in workers if row.get('status') in ('ok', 'quarantined'))
        return {
            'schema': 1, 'action': 'status', 'read_only': True,
            'run_id': self.run_id, 'sequence_independent': True,
            'observed': rows is not None,
            'failure': failure,
            'workers_available': available,
            'workers_expected': len(RANKS),
            'foreign_ranks': sorted(str(value) for value in foreign),
            'quarantined': self.quarantined,
            'quarantine_reason': self.quarantine_reason,
            'eligibility_failure': copy.deepcopy(self.eligibility_failure),
            # Actual engine observation, not the last acknowledged mutation
            # state.  None means unknown; it is never replaced by self.paused.
            'paused': paused,
            'pause_observation': {
                'state': pause_state,
                'basis': 'awaited AsyncLLM is_paused() under the configured timeout',
                'failure': pause_failure,
            },
            'last_acknowledged_paused': self.paused,
            'command_seq_unchanged': self.seq,
            'engine_epoch': self.epoch,
            'workers': workers,
            'start_monotonic_ns': started,
            'end_monotonic_ns': time.monotonic_ns(),
            'wall_s': time.time(),
        }

    async def execute(self, command):
        if type(command) is not dict or not _label(command.get('run_id')):
            raise ProbeControlError('invalid-command')
        action = command.get('action')
        # Explicitly sequence-independent observation. It is read-only, allowed
        # after quarantine, does not adopt/change ownership and does not touch
        # seq/last/ingress/paused.
        if action == 'status' and command.get('read_only') is True:
            if set(command) != READ_ONLY_FIELDS:
                raise ProbeControlError('command-fields-differ')
            async with self.lock:
                if self.run_id is not None and self.run_id != command['run_id']:
                    raise ProbeControlError('stale-command')
                return await self._observe()
        if action == 'open':
            expected = {'action', 'run_id'}
        else:
            expected = {'action', 'run_id', 'engine_epoch', 'seq'}
            if action == 'transition':
                expected |= {'target', 'expected_route_generation', 'cohort_size'}
            elif action == 'resume':
                expected |= {'cohort_size'}
        if set(command) != expected or (action != 'open' and action not in ACTIONS):
            raise ProbeControlError('command-fields-differ')
        async with self.lock:
            if action == 'open':
                if self.run_id is not None:
                    raise ProbeControlError('controller-already-owned')
                self.run_id = command['run_id']
            else:
                if (self.run_id != command['run_id'] or type(command['engine_epoch']) is not int
                        or command['engine_epoch'] != self.epoch or not _positive(command['seq'])
                        or command['seq'] != self.seq + 1):
                    raise ProbeControlError('stale-command')
                if self.quarantined and action != 'status':
                    raise ProbeControlError('controller-quarantined')
                self.seq = command['seq']  # consume before submission; no ambiguous replay
            started = time.monotonic_ns()
            try:
                result = await asyncio.wait_for(self._execute(action, command), self.timeout_s)
            except BaseException as exc:
                self.quarantined = True
                self.ingress.sealed = True
                if self.quarantine_reason is None:
                    if isinstance(exc, ProbeControlError):
                        self.quarantine_reason = str(exc)[:120]
                    else:
                        self.quarantine_reason = 'control-operation-failed:' + type(exc).__name__
                if isinstance(exc, asyncio.CancelledError):
                    raise
                if isinstance(exc, ProbeControlError):
                    raise
                raise ProbeControlError('control-operation-failed:' + type(exc).__name__) from None
            result.update(schema=1, run_id=self.run_id, engine_epoch=self.epoch,
                          command_seq=self.seq, action=action,
                          start_monotonic_ns=started, end_monotonic_ns=time.monotonic_ns(),
                          wall_s=time.time(), quarantined=self.quarantined)
            self.last = copy.deepcopy(result)
            return copy.deepcopy(result)

    async def _execute(self, action, command):
        if action in ('open', 'status'):
            snapshot = await self._snapshot()
            if action == 'open' and snapshot['route'] != 'full':
                raise ProbeControlError('open-requires-full-route')
            return {'snapshot': snapshot, 'paused': await self.engine.is_paused()}
        if action == 'pause':
            await self.ingress.seal()
            await self.engine.pause_generation(mode='keep', clear_cache=False)
            if await self.engine.is_paused() is not True:
                raise ProbeControlError('pause-not-complete')
            snapshot = await self._snapshot()
            self._idle(snapshot)
            barriers = await self._barrier(snapshot)
            self.paused = True
            return {'snapshot': snapshot, 'barriers': barriers,
                    'pause': {'mode': 'keep', 'clear_cache': False, 'completed': True},
                    'ingress': self.ingress.snapshot()}
        if not self.paused or await self.engine.is_paused() is not True:
            raise ProbeControlError('completed-pause-required')
        snapshot = await self._snapshot()
        self._idle(snapshot)
        if action in ('observer_begin', 'observer_end'):
            if snapshot['route'] != 'split':
                raise ProbeControlError('observer-requires-split-route')
            owners = {}
            for row in snapshot['ranks']:
                owner = row.get('inventory', {}).get('replay_observer')
                if (type(owner) is not dict or owner.get('enabled') is not True
                        or type(owner.get('worker_incarnation')) is not str
                        or not owner['worker_incarnation']):
                    raise ProbeControlError('observer-owner-inventory-missing')
                owners[row['rank']] = owner['worker_incarnation']
            replies = await self._rpc({'action': action, 'engine_epoch': self.epoch,
                                       'run_id': self.run_id, 'window_id': 'short-placement'})
            mapped = _rank_map(replies)
            expected_state = 'window' if action == 'observer_begin' else 'closed'
            for row in mapped.values():
                if (row.get('ack') != 'flash_s5_' + action
                        or type(row.get('engine_epoch')) is not int or row['engine_epoch'] != self.epoch
                        or not _positive(row.get('drained_streams'))
                        or row.get('state') != expected_state
                        or any(type(row.get(k)) is not int or row[k] != snapshot[k]
                               for k in ('route_generation', 'storage_epoch'))
                        or type(row.get('worker_incarnation')) is not str
                        or row['worker_incarnation'] != owners[row['rank']]):
                    raise ProbeControlError('invalid-worker-observer-ack')
                if action == 'observer_end':
                    receipt = row.get('receipt')
                    if (row.get('observer_stopped') is not True or type(receipt) is not dict
                            or receipt.get('schema') != 's5ro-replay-observer/1'
                            or receipt.get('verdict') not in ('PASS', 'FAIL', 'INCONCLUSIVE')):
                        raise ProbeControlError('invalid-worker-observer-receipt')
                    proof = receipt.get('caller_proof')
                    if (type(proof) is not dict or type(proof.get('worker_rank')) is not int
                            or proof['worker_rank'] != row['rank']
                            or proof.get('worker_incarnation') != row['worker_incarnation']
                            or proof.get('engine_incarnation') != 'engine:' + str(self.epoch)
                            or proof.get('route_generation') != snapshot['route_generation']
                            or proof.get('storage_epoch') != snapshot['storage_epoch']
                            or proof.get('relevant_consumers') != row['drained_streams']
                            or proof.get('controller_drained') is not True):
                        raise ProbeControlError('observer-caller-proof-does-not-match-worker')
            after = await self._snapshot()
            for row in after['ranks']:
                owner = row.get('inventory', {}).get('replay_observer')
                if (type(owner) is not dict or owner.get('worker_incarnation') != owners[row['rank']]
                        or owner.get('state') != expected_state
                        or (action == 'observer_end' and owner.get('closed') is not True)):
                    raise ProbeControlError('observer-owner-changed-after-rpc')
            return {'snapshot': after, 'paused': True,
                    'observer': {'schema': 1,
                        'complete': action == 'observer_end' and all(
                            row['receipt']['verdict'] == 'PASS' for row in mapped.values()),
                        'stopped': action == 'observer_end',
                        'ranks': [copy.deepcopy(mapped[r]) for r in RANKS]}}
        if action == 'transition':
            target = command['target']
            if target not in ('full', 'split') or target == snapshot['route']:
                raise ProbeControlError('invalid-transition-target')
            if (type(command['expected_route_generation']) is not int
                    or command['expected_route_generation'] != snapshot['route_generation']):
                raise ProbeControlError('stale-route-generation')
            eligibility = self._eligibility(snapshot, command['cohort_size']) if target == 'split' else None
            transition = {'action': 'transition', 'engine_epoch': self.epoch,
                          'expected_route_generation': snapshot['route_generation'],
                          'route_generation': snapshot['route_generation'] + 1,
                          'storage_epoch': snapshot['storage_epoch'], 'target': target}
            replies = await self._rpc(transition)
            from decode_probe import validate_transition_acks
            validate_transition_acks(transition, replies, RANKS)
            after = await self._snapshot()
            if after['route'] != target or after['route_generation'] != transition['route_generation']:
                raise ProbeControlError('transition-status-differs')
            return {'command': transition, 'replies': copy.deepcopy(replies),
                    'snapshot': after, 'decode_eligibility': eligibility, 'paused': True}
        if action == 'resume':
            eligibility = self._eligibility(snapshot, command['cohort_size']) if snapshot['route'] == 'split' else None
            # A fresh barrier also covers no-op full-route controls and any work
            # enqueued by diagnostic RPCs since the last pause.
            barriers = await self._barrier(snapshot)
            await self.engine.resume_generation()
            if await self.engine.is_paused() is not False:
                raise ProbeControlError('resume-not-complete')
            self.paused = False
            return {'snapshot': snapshot, 'barriers': barriers,
                    'decode_eligibility': eligibility, 'resume': {'completed': True}}
        raise ProbeControlError('unsupported-action')


class AdmissionSeal:
    """One-frontend diagnostic admission boundary, before any async preprocessing.

    Existing accepted requests continue; no scheduler or block ownership changes.
    Streaming-input and multi-output requests are unsupported in this diagnostic.
    """
    def __init__(self, engine):
        original = getattr(engine, 'add_request', None)
        if not callable(original) or getattr(engine, '_flash_s5_admission_seal', None) is not None:
            raise ProbeControlError('diagnostic-admission-seam-unavailable')
        self.sealed, self.pending, self.rejected = False, 0, 0
        self.idle = asyncio.Event()
        self.idle.set()

        async def guarded(*args, **kwargs):
            prompt = args[1] if len(args) > 1 else kwargs.get('prompt')
            params = args[2] if len(args) > 2 else kwargs.get('params')
            if self.sealed or hasattr(prompt, '__aiter__') or getattr(params, 'n', None) != 1:
                self.rejected += 1
                raise ProbeControlError('diagnostic-admission-closed-or-unsupported')
            self.pending += 1
            self.idle.clear()
            try:
                return await original(*args, **kwargs)
            finally:
                self.pending -= 1
                if not self.pending:
                    self.idle.set()
        engine.add_request = guarded
        engine._flash_s5_admission_seal = self

    async def seal(self):
        self.sealed = True
        await self.idle.wait()

    def unseal(self):
        if self.pending:
            raise ProbeControlError('admission-still-pending')
        self.sealed = False

    def snapshot(self):
        return {'sealed': self.sealed, 'pending': self.pending, 'rejected': self.rejected,
                'basis': 'single AsyncLLM add_request entry; existing accepted requests preserved'}


def attach_router(args, app, env=None):
    """Lazy diagnostic-only router, additionally authenticated and loopback-only.

    The generated registration is inert when disabled. Existing configured API
    keys are required; none is minted or printed. General dev endpoints remain off.
    """
    import os
    env = os.environ if env is None else env
    if env.get('FLASH_S5_OVERLAP', '0') == '1':
        state = attach_native_router(args, app, env)
        from native_qualification_frontend import attach_router as qualification_router
        state['qualification'] = qualification_router(args, app, env=env)
        return state
    if env.get('FLASH_S5_DECODE_ONLY_PROBE', '0') != '1':
        return None
    from decode_probe import mode_settings
    mode_settings(env)  # conflict and declared SMS checks, no GPU allocation
    if env.get('VLLM_SERVER_DEV_MODE', '0') not in ('0', ''):
        raise ProbeControlError('diagnostic-control-refuses-broad-dev-router')
    configured = getattr(args, 'api_key', None)
    if configured is not None and type(configured) is not list:
        raise ProbeControlError('invalid-existing-api-auth-shape')
    tokens = [value for value in (configured or [env.get('VLLM_API_KEY')]) if value]
    if not tokens or any(type(value) is not str for value in tokens):
        raise ProbeControlError('diagnostic-control-requires-existing-api-auth')
    hashes = [hashlib.sha256(value.encode()).digest() for value in tokens]
    from fastapi import APIRouter, Request
    from fastapi.responses import JSONResponse
    router = APIRouter()
    state = {'controller': None, 'engine': None, 'session': None}

    async def endpoint(request):
        host = getattr(getattr(request, 'client', None), 'host', '')
        try:
            local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = False
        header = request.headers.get('authorization', '')
        scheme, _, value = header.partition(' ')
        candidate = hashlib.sha256(value.encode()).digest()
        authorized = scheme.lower() == 'bearer' and any(secrets.compare_digest(candidate, h) for h in hashes)
        if not local or not authorized:
            return JSONResponse({'error': 'diagnostic-auth-required'}, status_code=403)
        data = bytearray()
        try:
            async for chunk in request.stream():
                data.extend(chunk)
                if len(data) > 65536:
                    raise ProbeControlError('command-too-large')
            command = json.loads(data)
            engine = request.app.state.engine_client
            if state['controller'] is None:
                state['engine'] = engine
                state['controller'] = ProbeController(engine)
            elif state['engine'] is not engine:
                raise ProbeControlError('engine-client-incarnation-changed')
            if request.url.path == PATH + '/session':
                from probe_session import ProbeSession
                if state['session'] is None:
                    state['session'] = ProbeSession(state['controller'])
                result = await state['session'].execute(command)
            else:
                read_only = (type(command) is dict and command.get('action') == 'status'
                             and command.get('read_only') is True)
                if state['session'] is not None and not read_only:
                    raise ProbeControlError('raw-control-owned-by-session')
                result = await state['controller'].execute(command)
            return JSONResponse(result)
        except (ValueError, TypeError):
            return JSONResponse({'error': 'invalid-command-json'}, status_code=400)
        except ProbeControlError as exc:
            return JSONResponse({'error': str(exc)}, status_code=409)
    endpoint.__annotations__['request'] = Request
    router.add_api_route(PATH, endpoint, methods=['POST'])
    router.add_api_route(PATH + '/session', endpoint, methods=['POST'])
    app.include_router(router)
    return state


def attach_native_router(args, app, env=None):
    """Narrow native-mode endpoint: session actions and exact qualification ones.

    Same guard rails as the diagnostic router (loopback, an existing configured
    API key, no broad dev router, bounded body) but a separate path and a small
    action vocabulary.  It exposes no general RPC: the session's typed actions
    and the four exact qualification actions are the only entries, and the
    qualification payload is forwarded to ``native_qualification.handle_worker``
    through the engine control channel -- the endpoint never supplies a verdict.
    """
    import os
    env = os.environ if env is None else env
    if env.get('FLASH_S5_OVERLAP', '0') != '1':
        return None
    if env.get('VLLM_SERVER_DEV_MODE', '0') not in ('0', ''):
        raise ProbeControlError('native-control-refuses-broad-dev-router')
    configured = getattr(args, 'api_key', None)
    if configured is not None and type(configured) is not list:
        raise ProbeControlError('invalid-existing-api-auth-shape')
    tokens = [value for value in (configured or [env.get('VLLM_API_KEY')]) if value]
    if not tokens or any(type(value) is not str for value in tokens):
        raise ProbeControlError('native-control-requires-existing-api-auth')
    hashes = [hashlib.sha256(value.encode()).digest() for value in tokens]
    from fastapi import APIRouter, Request
    from fastapi.responses import JSONResponse
    from native_overlap_control import (
        QUALIFICATION_ACTIONS,
        NativeControlClient,
        NativeControlError,
    )
    from native_overlap_session import NativeOverlapSession, NativeOverlapSessionError
    router = APIRouter()
    state = {'engine': None, 'control': None, 'session': None,
             'mode': 'native-overlap'}
    install_owner_health(app)

    async def endpoint(request):
        host = getattr(getattr(request, 'client', None), 'host', '')
        try:
            local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = False
        header = request.headers.get('authorization', '')
        scheme, _, value = header.partition(' ')
        candidate = hashlib.sha256(value.encode()).digest()
        authorized = (scheme.lower() == 'bearer'
                      and any(secrets.compare_digest(candidate, h) for h in hashes))
        if not local or not authorized:
            return JSONResponse({'error': 'native-control-auth-required'},
                                status_code=403)
        data = bytearray()
        try:
            async for chunk in request.stream():
                data.extend(chunk)
                if len(data) > 65536:
                    raise ProbeControlError('command-too-large')
            command = json.loads(data)
            if type(command) is not dict:
                raise ProbeControlError('native-command-not-an-object')
            engine = request.app.state.engine_client
            if state['engine'] is None:
                state['engine'] = engine
                state['control'] = NativeControlClient(engine)
                state['session'] = NativeOverlapSession(engine,
                                                        control=state['control'])
            elif state['engine'] is not engine:
                raise ProbeControlError('engine-client-incarnation-changed')
            action = command.get('action')
            if action in QUALIFICATION_ACTIONS:
                result = await state['control'].qualification(
                    action, command.get('payload'))
            elif action in NATIVE_SESSION_ACTIONS:
                result = await state['session'].execute(command)
            else:
                raise ProbeControlError('native-action-not-exposed')
            return JSONResponse(result)
        except (ValueError, TypeError):
            return JSONResponse({'error': 'invalid-command-json'}, status_code=400)
        except (ProbeControlError, NativeControlError,
                NativeOverlapSessionError) as exc:
            return JSONResponse({'error': str(exc)}, status_code=409)
    endpoint.__annotations__['request'] = Request
    router.add_api_route(NATIVE_PATH, endpoint, methods=['POST'])
    app.include_router(router)
    return state


def install_owner_health(app):
    """Read-only startup proof latch; never binds an owner or starts a lane.

    The EngineCore ready handshake is already withheld by synchronous startup.
    Query its latched proof once per EngineClient identity. Stock error health
    still runs on every scrape; a replacement client must prove its own owner.
    """
    from fastapi.responses import JSONResponse
    import asyncio
    proved = None
    lock = asyncio.Lock()

    @app.middleware('http')
    async def owner_health(request, call_next):
        nonlocal proved
        if request.url.path != '/health':
            return await call_next(request)
        engine = getattr(request.app.state, 'engine_client', None)
        if engine is None or getattr(engine, 'errored', True):
            proved = None
            return JSONResponse({'error': 'engine-not-ready'}, status_code=503)
        if proved is not engine:
            async with lock:
                if proved is not engine:
                    try:
                        row = await asyncio.wait_for(engine.engine_core.call_utility_async(
                            'flash_s5_native_control', {'action': 'status'}), timeout=1.0)
                        proof = row.get('startup_owner_ready')
                        if (type(proof) is not dict or proof.get('engine_epoch') != row.get('engine_epoch')
                                or row.get('bootstrapped') is not True
                                or proof.get('full_transition', {}).get('route_after') != 'full'):
                            raise ValueError('startup owner proof absent')
                        proved = engine
                    except Exception:
                        return JSONResponse({'error': 'engine-owner-not-ready'}, status_code=503)
        return await call_next(request)
