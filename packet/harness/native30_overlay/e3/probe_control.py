"""Owner-only decode diagnostic control over the existing AsyncLLM client.

No dev-router switch, new executor, engine lifecycle control or arbitrary RPC.
HTTP registration is lazy and default-off. A timeout may leave remote work in
flight: quarantine blocks resume/transition; status is the only remaining action.
"""
import asyncio
import copy
import hashlib
import ipaddress
import json
import secrets
import time


class ProbeControlError(RuntimeError):
    pass


RANKS = (0, 1, 2, 3)
PATH = '/v1/flash_s5_decode_probe'
ACTIONS = ('status', 'pause', 'transition', 'resume')


def _positive(value):
    return type(value) is int and value > 0


def _label(value):
    return (type(value) is str and 1 <= len(value) <= 80
            and all(c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in value))


def _rank_map(rows):
    if type(rows) is not list or len(rows) != len(RANKS):
        raise ProbeControlError('incomplete-rank-coverage')
    mapped = {}
    for row in rows:
        if type(row) is not dict or type(row.get('rank')) is not int or row['rank'] not in RANKS or row['rank'] in mapped:
            raise ProbeControlError('invalid-rank-coverage')
        mapped[row['rank']] = row
    return mapped


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
        self.last = None
        self.ingress = AdmissionSeal(engine)

    async def _rpc(self, command):
        return await self.engine.collective_rpc('flash_s5_decode_probe',
            timeout=self.timeout_s, args=(command,), kwargs={})

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
            if (type(batch) is not dict or not _positive(batch.get('sequence'))
                    or batch.get('decode_only') is not True or batch.get('full_graph') is not True
                    or type(batch.get('num_reqs')) is not int or batch['num_reqs'] != cohort_size
                    or type(batch.get('num_tokens')) is not int or batch['num_tokens'] != cohort_size):
                raise ProbeControlError('worker-decode-eligibility-unproven')
            observed.append({'rank': row['rank'], 'batch': copy.deepcopy(batch)})
        return {'cohort_size': cohort_size, 'proved': True,
                'basis': 'paused live request/queued-token counts and last actual per-worker batch',
                'ranks': observed}

    async def execute(self, command):
        if type(command) is not dict or not _label(command.get('run_id')):
            raise ProbeControlError('invalid-command')
        action = command.get('action')
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
                if len(data) > 16384:
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
                if state['session'] is not None:
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
