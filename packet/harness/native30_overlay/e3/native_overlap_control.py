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
import os
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
# attn-meta (FLASH_ATTN_META_GDN_BATCH=1, default off): the per-arm selector's worker actions ride the qualification
# intent (paused idle FULL for the select, enforced by the engine driver); exposed only where the switch is on.
ATTN_META_ACTIONS = ('attn_meta_select', 'attn_meta_read')


class NativeControlError(RuntimeError):
    pass


def _progress_burst():
    """The held-f switch is exposed only where the engine installed burst."""
    return (os.environ.get('FLASH_S5_NATIVE_PROGRESS', '') == 'burst'
            and os.environ.get('FLASH_S5_AUTO_ARM', '0') == '1')


def _qsa_fused():
    """The Step A switch is exposed only where the engine installed it."""
    return (os.environ.get('FLASH_S5_QSA_FUSED', '0') == '1'
            and os.environ.get('FLASH_S5_AUTO_ARM', '0') == '1')


def attn_meta_actions():
    """The attn-meta selector actions, only where the switch installed it (else none)."""
    return ATTN_META_ACTIONS if os.environ.get('FLASH_ATTN_META_GDN_BATCH', '0') == '1' else ()


def _layer_groups():
    """The Step B switch is exposed only where the engine installed it."""
    return os.environ.get('FLASH_S5_LAYER_GROUPS', '0') == '1' and _qsa_fused()


def _stage_local():
    """The C0 switch is exposed only where the engine installed it."""
    return (os.environ.get('FLASH_S5_STAGE_LOCAL', '0') == '1'
            and os.environ.get('FLASH_S5_AUTO_ARM', '0') == '1')


def _stage_overlap():
    """C1's field is exposed only where the engine installed it beside C0 (FLASH_S5_STAGE_OVERLAP)."""
    return _stage_local() and os.environ.get('FLASH_S5_STAGE_OVERLAP', '0') == '1'


def _stage_overlap_defer():
    """alpha-next-c1: the deferred-bind field is exposed only where the engine installed it (FLASH_S5_STAGE_OVERLAP_DEFER_BIND)."""
    return _stage_overlap() and os.environ.get('FLASH_S5_STAGE_OVERLAP_DEFER_BIND', '0') == '1'


def _step_switch(name):
    """stage1-step: a field is exposed only where the engine installed it."""
    return _stage_overlap_defer() and os.environ.get(name, '0') == '1'


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
            if not (action == 'release_auto_arm' and os.environ.get('FLASH_S5_AUTO_ARM', '0') == '1'
                    or action == 'native_progress' and _progress_burst()
                    or action == 'native_qsa_fused' and _qsa_fused()
                    or action == 'native_layer_group' and _layer_groups()
                    or action == 'native_stage_local' and _stage_local()):
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

    async def release_auto_arm(self):
        if self.engine_epoch is None:
            await self.status()
        return await self._call({'action': 'release_auto_arm'})

    async def native_progress(self, mode, idle=None, replies=None):
        """Held-f per-arm switch (installed burst only); the engine re-checks idle FULL.

        Held-g: ``idle`` (``step`` or ``wait``) also selects the arm's idle
        pacing, and ``replies`` (``full`` or ``lean``) its reply format; a
        field left out is not sent and the engine applies ``step``/``full``.
        """
        if mode not in ('current', 'burst'):
            raise NativeControlError('native-control-progress-mode-invalid')
        if idle is not None and idle not in ('step', 'wait'):
            raise NativeControlError('native-control-progress-idle-invalid')
        if replies is not None and replies not in ('full', 'lean'):
            raise NativeControlError('native-control-progress-replies-invalid')
        if self.engine_epoch is None:
            await self.status()
        command = {'action': 'native_progress', 'mode': mode}
        if idle is not None:
            command['idle'] = idle
        if replies is not None:
            command['replies'] = replies
        reply = await self._call(command)
        if reply.get('mode') != mode:
            raise NativeControlError('native-control-progress-mode-not-set')
        if idle is not None and reply.get('idle') != idle:
            raise NativeControlError('native-control-progress-idle-not-set')
        if replies is not None and reply.get('replies') != replies:
            raise NativeControlError('native-control-progress-replies-not-set')
        return reply

    async def native_qsa_fused(self, fused):
        """Step A per-arm switch (installed only); the engine re-checks idle FULL."""
        if type(fused) is not bool:
            raise NativeControlError('native-control-qsa-fused-invalid')
        if self.engine_epoch is None:
            await self.status()
        reply = await self._call({'action': 'native_qsa_fused', 'fused': fused})
        if reply.get('fused') is not fused:
            raise NativeControlError('native-control-qsa-fused-not-set')
        return reply

    async def native_layer_group(self, size):
        """Step B per-arm size (installed only); the engine re-checks idle FULL and fused."""
        if type(size) is not int or size not in (1, 2, 4):
            raise NativeControlError('native-control-layer-group-invalid')
        if self.engine_epoch is None:
            await self.status()
        reply = await self._call({'action': 'native_layer_group', 'size': size})
        if reply.get('size') != size:
            raise NativeControlError('native-control-layer-group-not-set')
        return reply

    async def native_stage_local(self, local):
        """C0 per-arm switch (installed only); the engine re-checks idle FULL."""
        if type(local) is not bool:
            raise NativeControlError('native-control-stage-local-invalid')
        if self.engine_epoch is None:
            await self.status()
        reply = await self._call({'action': 'native_stage_local', 'local': local})
        if reply.get('local') is not local:
            raise NativeControlError('native-control-stage-local-not-set')
        return reply

    async def native_stage_overlap(self, local, overlap, defer_bind=None, lazy_recv=None, decode_merge=None, inwait=None, early_successor=None):
        """C1 per-arm switch (installed only): C0's command with ``overlap``; the engine re-checks idle FULL.
        alpha-next-c1: ``defer_bind`` (installed only) rides on the same command; the reply must acknowledge it."""
        if not _stage_overlap():
            raise NativeControlError('native-control-stage-overlap-not-installed')
        if type(local) is not bool:
            raise NativeControlError('native-control-stage-local-invalid')
        if type(overlap) is not bool:
            raise NativeControlError('native-control-stage-overlap-invalid')
        command = {'action': 'native_stage_local', 'local': local, 'overlap': overlap}
        if defer_bind is not None:
            if not _stage_overlap_defer():
                raise NativeControlError('native-control-stage-overlap-defer-bind-not-installed')
            if type(defer_bind) is not bool:
                raise NativeControlError('native-control-stage-overlap-defer-bind-invalid')
            command['defer_bind'] = defer_bind
        if lazy_recv is not None:
            if not _step_switch('FLASH_S5_LAZY_RECV'):
                raise NativeControlError('native-control-stage-lazy-recv-not-installed')
            if type(lazy_recv) is not bool:
                raise NativeControlError('native-control-stage-lazy-recv-invalid')
            command['lazy_recv'] = lazy_recv
        if decode_merge is not None:
            if not _step_switch('FLASH_S5_DECODE_MERGE'):
                raise NativeControlError('native-control-stage-decode-merge-not-installed')
            if decode_merge not in ('off', 'chunk', 'always'):
                raise NativeControlError('native-control-stage-decode-merge-invalid')
            command['decode_merge'] = decode_merge
        if inwait is not None:
            if not _step_switch('FLASH_S5_INWAIT'):
                raise NativeControlError('native-control-stage-inwait-not-installed')
            if type(inwait) is not bool:
                raise NativeControlError('native-control-stage-inwait-invalid')
            command['inwait'] = inwait
        if early_successor is not None:
            if os.environ.get('FLASH_S5_INWAIT', '0') != '1':
                raise NativeControlError('native-control-stage-inwait-not-installed')
            if type(early_successor) is not bool:
                raise NativeControlError('native-control-stage-early-successor-invalid')
            command['early_successor'] = early_successor
        if self.engine_epoch is None:
            await self.status()
        reply = await self._call(command)
        if reply.get('local') is not local or reply.get('overlap') is not overlap:
            raise NativeControlError('native-control-stage-overlap-not-set')
        if defer_bind is not None and reply.get('defer_bind') is not defer_bind:
            raise NativeControlError('native-control-stage-overlap-defer-bind-not-set')
        if lazy_recv is not None and reply.get('lazy_recv') is not lazy_recv:
            raise NativeControlError('native-control-stage-lazy-recv-not-set')
        if decode_merge is not None and reply.get('decode_merge') != decode_merge:
            raise NativeControlError('native-control-stage-decode-merge-not-set')
        if inwait is not None and reply.get('inwait') is not inwait:
            raise NativeControlError('native-control-stage-inwait-not-set')
        if early_successor is not None and reply.get('early_successor') is not early_successor:
            raise NativeControlError('native-control-stage-early-successor-not-set')
        return reply

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
        if worker_action not in QUALIFICATION_ACTIONS and worker_action not in attn_meta_actions():
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


__all__ = ['ACTIONS', 'ADMISSION_PHASES', 'ATTN_META_ACTIONS', 'INTENT_ACTIONS', 'NATIVE_ADMISSION_PHASES',
           'attn_meta_actions',
           'NativeControlClient', 'NativeControlError', 'QUALIFICATION_ACTIONS',
           'READ_ACTIONS', 'UTILITY']
