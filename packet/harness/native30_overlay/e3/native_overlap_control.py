"""Owner-only native overlap control over the pinned EngineCore utility channel.

``AsyncLLM.engine_core.call_utility_async`` dispatches an ordinary named method
on the live ``EngineCore`` (pinned ``core_client.py`` and ``core.py``); the
generated ``EngineCore.flash_s5_native_control`` seam is that method.  Nothing
here looks up a driver object by Python attribute, installs a general RPC or
dev endpoint, starts an in-process engine or reads a worker reply queue.

Every call is one bounded typed intent or query:

* reads (``status``/``snapshot``/``transition_acks``/``bootstrap_acks``/
  ``overlap_evidence``/``cancel``) settle synchronously from engine state;
* worker queries and route mutations (``census``/``barrier``/``drain``/
  ``poll_native``/``qualification``/``bootstrap_engine_owner``/``enter_split``/
  ``leave_split``) register one intent that the engine's own tick settles from
  the owned reply transport.  The client only awaits the utility future; it
  never pumps, polls or waits for a decode reply itself.

A client timeout or cancellation never drops engine/worker ownership: the
engine keeps the intent and settles it from the typed reply or the absolute
deadline, and this client refuses to start another intent until it has observed
that settlement.

Pause/resume is the API path and stays awaited: ``pause_generation(mode='keep',
clear_cache=False)`` and ``resume_generation()``.
"""
import asyncio
import math
import time

from native_overlap_metrics import clock_identity, prove_clock_domain

UTILITY = 'flash_s5_native_control'
# The engine's typed phase vocabulary, duplicated here so the client and the
# session name the same strings.  The engine stays the authority: it re-checks
# the owner, epoch, route and phase of every intent it is handed.
ADMISSION_PHASES = ('full-prefill-solo', 'split-prefill-solo',
                    'split-w1-decode', 'row1-native', 'w2-overlap')
NATIVE_ADMISSION_PHASES = ('split-prefill-solo', 'row1-native', 'w2-overlap')
READ_ACTIONS = ('status', 'snapshot', 'transition_acks', 'bootstrap_acks',
                'overlap_evidence', 'cancel')
INTENT_ACTIONS = ('identity', 'census', 'barrier', 'drain', 'poll_native', 'qualification',
                  'bootstrap_engine_owner', 'enter_split', 'leave_split',
                  'arm_admission', 'disarm_admission')
ACTIONS = READ_ACTIONS + INTENT_ACTIONS
QUALIFICATION_ACTIONS = ('teacher_configure', 'teacher_evidence',
                         'observer_begin', 'observer_end',
                         'row1_configure', 'row1_evidence')


class NativeControlError(RuntimeError):
    pass


def _positive(value):
    return type(value) is int and value > 0


def _bounded(value, depth=0):
    """Copy only bounded plain metadata into a command payload."""
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
    if type(value) in (list, tuple):
        return [_bounded(item, depth + 1) for item in list(value)[:32]]
    if type(value) is dict:
        return {str(key)[:64]: _bounded(item, depth + 1)
                for key, item in list(value.items())[:64]}
    return None


class NativeControlClient:
    """One owner-side view of the engine's native-control utility."""

    def __init__(self, engine, *, timeout_s=120.0, clock=time.monotonic_ns,
                 history=256):
        core = getattr(engine, 'engine_core', None)
        if core is None or not callable(getattr(core, 'call_utility_async', None)):
            raise NativeControlError(
                'native-control-requires-the-pinned-engine-utility-channel')
        # This is the API-side AsyncLLM, not EngineCore.  Core exposes
        # is_scheduler_paused(); the pinned frontend exposes async is_paused().
        for name in ('is_paused', 'pause_generation', 'resume_generation'):
            if not callable(getattr(engine, name, None)):
                raise NativeControlError('native-control-frontend-api-missing:' + name)
        if getattr(core, '_flash_s5_driver', None) is not None:
            # The utility channel is a client, not a second engine owner: an
            # object carrying both would be a driver lookup in disguise.
            raise NativeControlError(
                'native-control-client-must-not-own-an-engine-driver')
        if type(getattr(core, 'client_count', None)) is not int or core.client_count != 1:
            raise NativeControlError('native-control-requires-single-api-client')
        if type(getattr(core, 'client_index', None)) is not int or core.client_index != 0:
            raise NativeControlError('native-control-requires-the-owning-api-client')
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise NativeControlError('native-control-timeout-invalid')
        identity = clock_identity()
        if identity.get('proved') is not True:
            raise NativeControlError(
                'native-control-client-clock-identity-unproved:'
                + str(identity.get('reason')))
        self.engine = engine
        self.core = core
        self.timeout_s = float(timeout_s)
        self.clock = clock
        self.identity = identity
        self.engine_epoch = None
        self.last_snapshot = None
        self.pending = None
        self.history = []
        self.history_limit = history
        self.paused = False
        self.routes = []

    # -- transport -------------------------------------------------------
    async def _call(self, command):
        if type(command) is not dict:
            raise NativeControlError('native-control-command-not-an-object')
        action = command.get('action')
        if action not in ACTIONS:
            raise NativeControlError('native-control-unknown-action')
        if self.pending is not None and action not in READ_ACTIONS:
            # Reads stay available while an abandoned intent is registered so
            # this client can observe the engine's settlement; a new intent may
            # not be started until that is seen.
            raise NativeControlError(
                'native-control-intent-still-outstanding:' + str(self.pending))
        payload = dict(command)
        if self.engine_epoch is not None:
            payload.setdefault('engine_epoch', self.engine_epoch)
        send = int(self.clock())
        try:
            reply = await asyncio.wait_for(
                self.core.call_utility_async(UTILITY, payload), self.timeout_s)
        except asyncio.TimeoutError:
            if action not in READ_ACTIONS:
                self.pending = action
            self._remember({'action': action, 'send_ns': send, 'timeout': True,
                            'ownership_retained': action not in READ_ACTIONS})
            raise NativeControlError(
                'native-control-utility-timeout: the engine retains ownership'
            ) from None
        except asyncio.CancelledError:
            if action not in READ_ACTIONS:
                self.pending = action
            self._remember({'action': action, 'send_ns': send,
                            'cancelled': True,
                            'ownership_retained': action not in READ_ACTIONS})
            raise
        recv = int(self.clock())
        if type(reply) is not dict:
            raise NativeControlError('native-control-reply-not-an-object')
        proof = prove_clock_domain(reply.get('clock'), self.identity, send, recv)
        self._remember({'action': action, 'send_ns': send, 'recv_ns': recv,
                        'clock_proof': proof,
                        'engine_epoch': reply.get('engine_epoch'),
                        'intent': reply.get('action')})
        if proof.get('proved') is not True:
            raise NativeControlError(
                'native-control-clock-domain-unproved:'
                + ','.join(proof.get('reasons') or []))
        if action == 'cancel':
            if reply.get('ownership_retained') is not True:
                self.pending = None
        elif self.pending == action:
            self.pending = None
        elif (self.pending is not None and action in ('status', 'snapshot')
              and type(reply.get('snapshot', reply)) is dict
              and (reply.get('snapshot') or reply).get('control_pending') is None):
            # A read that observes the engine holding no intent is the only
            # evidence of settlement an abandoned (timed-out) intent can give;
            # the engine settled it from the typed reply or the absolute
            # deadline, never because this client stopped waiting.  A status
            # record is its own snapshot; a snapshot record carries one.
            observed = self.pending
            self.pending = None
            self._remember({'action': 'settlement-observed',
                            'abandoned_intent': observed,
                            'route': (reply.get('snapshot') or reply).get('route'),
                            'engine_epoch': (reply.get('snapshot') or reply).get('engine_epoch')})
        if type(reply.get('engine_epoch')) is int:
            if (self.engine_epoch is not None
                    and reply['engine_epoch'] != self.engine_epoch):
                raise NativeControlError(
                    'native-control-engine-epoch-changed')
            self.engine_epoch = reply['engine_epoch']
        if type(reply.get('snapshot')) is dict:
            self.last_snapshot = reply['snapshot']
        return reply

    def _remember(self, record):
        self.history.append(record)
        del self.history[:-self.history_limit]

    async def cancel(self):
        return await self._call({'action': 'cancel'})

    # -- reads -----------------------------------------------------------
    async def status(self):
        reply = await self._call({'action': 'status'})
        # The status record *is* the queried engine snapshot.
        self.last_snapshot = reply
        return reply

    async def snapshot(self):
        reply = await self._call({'action': 'snapshot'})
        self.last_snapshot = reply
        return reply

    async def transition_acks(self):
        reply = await self._call({'action': 'transition_acks'})
        if reply.get('complete') is not True:
            raise NativeControlError('native-control-transition-acks-incomplete')
        ranks = {row.get('worker_rank') for row in reply.get('acks') or []
                 if type(row) is dict}
        if len(ranks) != len(reply.get('acks') or []):
            raise NativeControlError('native-control-transition-acks-duplicated')
        for row in reply.get('acks') or []:
            if type(row.get('drained_streams')) is not int or row['drained_streams'] < 0:
                raise NativeControlError(
                    'native-control-transition-ack-drain-unproved')
            if row.get('typed') != 'TransitionAck':
                raise NativeControlError(
                    'native-control-transition-ack-not-typed')
        return reply

    async def bootstrap_acks(self):
        reply = await self._call({'action': 'bootstrap_acks'})
        if reply.get('complete') is not True:
            raise NativeControlError('native-control-bootstrap-acks-incomplete')
        for row in reply.get('acks') or []:
            if row.get('typed') != 'BootstrapAck':
                raise NativeControlError('native-control-bootstrap-ack-not-typed')
        return reply

    async def overlap_evidence(self, start_ns, end_ns):
        if not _positive(start_ns) or not _positive(end_ns) or end_ns <= start_ns:
            raise NativeControlError('native-control-evidence-window-invalid')
        return await self._call({'action': 'overlap_evidence',
                                 'start_ns': int(start_ns), 'end_ns': int(end_ns)})

    # -- pause/resume (awaited API path) ---------------------------------
    async def _is_paused(self):
        value = await self.engine.is_paused()
        if type(value) is not bool:
            raise NativeControlError('native-control-pause-state-unproved')
        return value

    async def ensure_paused(self):
        current = await self._is_paused()
        if current is True:
            self.paused = True
            return {'paused': True, 'called': False, 'mode': 'keep',
                    'clear_cache': False}
        await self.engine.pause_generation(mode='keep', clear_cache=False)
        confirmed = await self._is_paused()
        if confirmed is not True:
            raise NativeControlError('native-control-pause-not-confirmed')
        self.paused = True
        return {'paused': True, 'called': True, 'mode': 'keep',
                'clear_cache': False}

    async def resume(self):
        if self.pending is not None:
            raise NativeControlError(
                'native-control-resume-while-intent-outstanding')
        if self.paused is False and await self._is_paused() is False:
            return {'resumed': False, 'already_running': True}
        await self.engine.resume_generation()
        if await self._is_paused() is not False:
            raise NativeControlError('native-control-resume-not-confirmed')
        self.paused = False
        return {'resumed': True, 'already_running': False}

    # -- intents ---------------------------------------------------------
    async def _intent(self, command, *, pause=True):
        if pause:
            await self.ensure_paused()
        reply = await self._call(command)
        if type(reply.get('ranks')) is list:
            self.last_snapshot = reply.get('snapshot') or self.last_snapshot
        return reply

    async def worker_identity(self):
        return await self._intent({'action': 'identity'})

    async def census(self):
        return await self._intent({'action': 'census'})

    async def barrier(self):
        return await self._intent({'action': 'barrier'})

    async def drain(self, budget=32):
        if type(budget) is not int or not 1 <= budget <= 64:
            raise NativeControlError('native-control-drain-budget-invalid')
        return await self._intent({'action': 'drain', 'budget': budget})

    async def poll_native(self, limit=8):
        if type(limit) is not int or not 1 <= limit <= 64:
            raise NativeControlError('native-control-poll-limit-invalid')
        return await self._intent({'action': 'poll_native', 'limit': limit})

    async def bootstrap_engine_owner(self):
        return await self._intent({'action': 'bootstrap_engine_owner'})

    async def arm_admission(self, phase):
        """Arm one explicit native admission phase (never an env/config flag).

        Valid phases are the engine's native ones: ``split-prefill-solo``,
        ``row1-native`` and ``w2-overlap``.  The engine refuses an unknown,
        non-native, foreign-epoch or already-armed-for-another-phase intent.
        """
        if type(phase) is not str or not phase:
            raise NativeControlError('native-control-admission-phase-invalid')
        if phase not in NATIVE_ADMISSION_PHASES:
            raise NativeControlError('native-control-admission-phase-not-native')
        reply = await self._intent({'action': 'arm_admission', 'phase': phase})
        reply_phase = (reply.get('admission_intent') or {}).get('phase')
        if reply_phase != phase:
            raise NativeControlError('native-control-admission-not-armed')
        return reply

    async def disarm_admission(self):
        """Disable native admission; refused while a native chunk is owned."""
        reply = await self._intent({'action': 'disarm_admission'})
        if reply.get('admission_intent') is not None:
            raise NativeControlError('native-control-admission-not-disarmed')
        return reply

    async def enter_split(self):
        reply = await self._intent({'action': 'enter_split'})
        self.routes.append({'action': 'enter_split',
                            'route': reply.get('route_after'),
                            'route_generation': reply.get('route_generation_after'),
                            'changed': reply.get('changed')})
        return reply

    async def leave_split(self):
        reply = await self._intent({'action': 'leave_split'})
        self.routes.append({'action': 'leave_split',
                            'route': reply.get('route_after'),
                            'route_generation': reply.get('route_generation_after'),
                            'changed': reply.get('changed')})
        return reply

    async def qualification(self, worker_action, payload=None):
        """Dispatch the exact qualification action; never supply its verdict."""
        if worker_action not in QUALIFICATION_ACTIONS:
            raise NativeControlError('native-control-qualification-action-unknown')
        if payload is not None and type(payload) is not dict:
            raise NativeControlError('native-control-qualification-payload-invalid')
        bounded = _bounded(payload or {})
        if type(bounded) is not dict:
            raise NativeControlError('native-control-qualification-payload-invalid')
        return await self._intent({'action': 'qualification',
                                   'worker_action': worker_action,
                                   'payload': bounded})

    async def execute(self, command):
        """Small typed dispatcher for the session/HTTP helper."""
        if type(command) is not dict:
            raise NativeControlError('native-control-command-not-an-object')
        action = command.get('action')
        if action == 'status':
            return await self.status()
        if action == 'snapshot':
            return await self.snapshot()
        if action == 'transition_acks':
            return await self.transition_acks()
        if action == 'bootstrap_acks':
            return await self.bootstrap_acks()
        if action == 'identity':
            return await self.worker_identity()
        if action == 'census':
            return await self.census()
        if action == 'barrier':
            return await self.barrier()
        if action == 'drain':
            return await self.drain(command.get('budget', 32))
        if action == 'poll_native':
            return await self.poll_native(command.get('limit', 8))
        if action == 'bootstrap_engine_owner':
            return await self.bootstrap_engine_owner()
        if action == 'arm_admission':
            return await self.arm_admission(command.get('phase'))
        if action == 'disarm_admission':
            return await self.disarm_admission()
        if action == 'enter_split':
            return await self.enter_split()
        if action == 'leave_split':
            return await self.leave_split()
        if action == 'qualification':
            return await self.qualification(command.get('worker_action'),
                                            command.get('payload'))
        if action == 'overlap_evidence':
            return await self.overlap_evidence(command.get('start_ns'),
                                               command.get('end_ns'))
        if action == 'cancel':
            return await self.cancel()
        if action == 'resume':
            return await self.resume()
        raise NativeControlError('native-control-action-not-exposed')


__all__ = ['ACTIONS', 'ADMISSION_PHASES', 'INTENT_ACTIONS', 'NATIVE_ADMISSION_PHASES',
           'NativeControlClient', 'NativeControlError', 'QUALIFICATION_ACTIONS',
           'READ_ACTIONS', 'UTILITY']
