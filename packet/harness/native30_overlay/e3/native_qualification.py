"""Native qualification worker actions and the API-owned teacher plan owner.

This is the native counterpart of the four decode-probe worker actions that
collect the teacher rows and the replay-observer window.  The reviewed
instruments are reused unchanged:

* :mod:`teacher_decode` owns the bounded trace state, the sampler/observer
  shadows and the exact ``A1/A2/B/C`` row capture;
* :mod:`observer_runtime` owns the replay/offer window and composes the engine
  placement receipt at ``end``;
* :mod:`split_bank.RouteController`, the native worker bridge and the native
  graph router provide the route generation, retained storage epoch and the
  real queried consumer/replay drains.

What differs from the probe is identity and ownership: there is no
``DecodeProbeRouter`` and no ``_flash_s5_decode_probe``.  The trace state is
bound to a :class:`NativeIdentity` that reads the constructed native lane, and
every action refuses when that construction, the worker bridge, the graph
router or a real drain is missing.  Nothing here invents a stream count, an
epoch, a descriptor or a worker acknowledgement, and nothing here calls
``require_native_ready``: the engine-side control may only construct this
action owner after its own native readiness check and native bootstrap.

The engine-side helper :func:`own_frontend_plan` reuses
``teacher_decode.FrontendPlan`` for the single cohort plan.  It is bound to a
constructed :class:`~native_overlap_control.NativeControlClient`, so a probe
control or a bare processor cannot silently take the native frontend seam.
Importing this module has no side effect: everything is default-off until the
native control asks for it.
"""
MODE = 'native-overlap'
ACTIONS = ('teacher_configure', 'teacher_evidence', 'observer_begin', 'observer_end',
           'row1_configure', 'row1_evidence')
_FIELDS = {
    'teacher_configure': {'action', 'engine_epoch', 'enabled', 'snapshot_max_bytes'},
    'teacher_evidence': {'action', 'engine_epoch'},
    'observer_begin': {'action', 'engine_epoch', 'run_id', 'window_id'},
    'observer_end': {'action', 'engine_epoch', 'run_id', 'window_id'},
    'row1_configure': {'action', 'engine_epoch', 'enabled', 'run_id'},
    'row1_evidence': {'action', 'engine_epoch', 'run_id'},
}
# EngineCore stamps this after copying the caller payload.
_FIELDS = {action: fields | {'route_generation'} for action, fields in _FIELDS.items()}
SNAPSHOT_MIN = 64 * 1024 ** 2
SNAPSHOT_MAX = 2 * 1024 ** 3
STATE_ATTRIBUTE = '_flash_s5_native_qualification'


class NativeQualificationError(RuntimeError):
    """A qualification action refused; the caller must not grade it as PASS."""


class NativeQualificationUnavailable(NativeQualificationError):
    """The real native construction or drain provider is absent."""


def _require(ok, code):
    if not ok:
        raise NativeQualificationUnavailable(code)


def _positive(value):
    return type(value) is int and value >= 1


def _text(value):
    return type(value) is str and bool(value)


def _native(runner):
    native = getattr(runner, '_flash_s5_native', None)
    if native is None:
        raise NativeQualificationUnavailable('native-qualification-worker-native-unavailable')
    # Import lazily so this module stays importable in the harness environment.
    from native_startup import WorkerNative
    if not isinstance(native, WorkerNative):
        raise NativeQualificationUnavailable('native-qualification-worker-native-unavailable')
    return native


def _rank(runner, native):
    rank = getattr(getattr(native, 'state', None), 'worker_rank', None)
    if type(rank) is not int:
        rank = getattr(runner, 'rank', None)
    _require(type(rank) is int and rank >= 0, 'native-qualification-worker-rank-unavailable')
    return rank


def _incarnation(native):
    state = getattr(native, 'state', None)
    incarnation = getattr(state, 'incarnation', None)
    _require(_text(incarnation), 'native-qualification-incarnation-unavailable')
    return incarnation


def _storage_epoch(native):
    owner = getattr(native, 'decode_owner', None)
    _require(owner is not None, 'native-qualification-retained-storage-missing')
    epoch = getattr(owner, 'storage_epoch', None)
    _require(_positive(epoch), 'native-qualification-storage-epoch-unavailable')
    handle = getattr(getattr(getattr(native, 'state', None), 'decode', None), 'handle', None)
    if handle is not None:
        _require(getattr(handle, 'storage_epoch', None) == epoch,
                 'native-qualification-storage-epoch-differs')
    return epoch


def _bridge(runner, native):
    bridge = getattr(runner, '_flash_s5_bridge', None)
    if bridge is None:
        raise NativeQualificationUnavailable('native-qualification-bridge-unavailable')
    from worker_bridge import NativeWorkerBridge
    _require(isinstance(bridge, NativeWorkerBridge), 'native-qualification-bridge-unavailable')
    _require(getattr(bridge, 'rank', None) == _rank(runner, native),
             'native-qualification-bridge-rank-differs')
    return bridge


def _router(runner):
    router = getattr(runner, '_flash_e3_graph_router', None)
    if router is None:
        raise NativeQualificationUnavailable('native-qualification-router-unavailable')
    # Probe ownership is never relabelled as native.
    _require(type(router).__name__ != 'DecodeProbeRouter',
             'native-qualification-refuses-decode-probe-router')
    _require(getattr(runner, '_flash_s5_decode_probe', None) is None,
             'native-qualification-refuses-decode-probe')
    return router


def _bind_epoch(native, epoch):
    _require(_positive(epoch), 'native-qualification-invalid-engine-epoch')
    state = getattr(native, 'state', None)
    _require(getattr(state, 'engine_bound', False) is True,
             'native-qualification-engine-unbound')
    _require(getattr(state, 'engine_epoch', None) == epoch,
             'native-qualification-stale-engine-epoch')
    return epoch


def _replay_census(router):
    poll = getattr(router, 'poll_native_replays', None)
    if not callable(poll):
        raise NativeQualificationUnavailable(
            'native-qualification-replay-census-unavailable')
    poll()
    pending = getattr(router, 'native_pending', None)
    _require(type(pending) is int, 'native-qualification-replay-census-unavailable')
    _require(pending == 0, 'native-qualification-replays-pending')
    census = router.native_census()
    _require(type(census) is dict and census.get('pending') == 0,
             'native-qualification-replays-pending')
    return census


def _drain(runner, native, *, require_quiet):
    """Query every owned consumer stream and prove no row/replay is retained.

    ``drained_streams`` is the number of streams the route controller actually
    recorded and synchronised, plus the owned prefill auxiliaries that were
    synchronised.  ``require_quiet`` additionally refuses an outstanding
    external source hold (used on the empty full route before the teacher
    boundary).
    """
    controller = getattr(native, 'controller', None)
    _require(controller is not None and callable(getattr(controller, 'drain', None)),
             'native-qualification-drain-provider-unavailable')
    _require(getattr(controller, 'admitting', None) is True,
             'native-qualification-controller-not-admitting')
    _require(getattr(runner, 'execute_model_state', None) is None,
             'native-qualification-execute-model-state-open')
    bridge = _bridge(runner, native)
    _require(not getattr(bridge, 'active', False),
             'native-qualification-request-row-retained')
    _require(getattr(bridge, '_retained', None) is None,
             'native-qualification-request-row-retained')
    router = _router(runner)
    census = _replay_census(router)
    status = router.status()
    _require(getattr(status, 'pending_replays', None) == 0,
             'native-qualification-replays-pending')
    _require(getattr(status, 'unclean', None) is False,
             'native-qualification-router-unclean')
    if require_quiet:
        _require(getattr(status, 'external_users', None) == 0,
                 'native-qualification-source-holds-active')
    streams = controller.drain()
    _require(type(streams) is int and streams > 0,
             'native-qualification-drain-unproved')
    owners = getattr(native, 'prefill_streams', None)
    if owners is None:
        raise NativeQualificationUnavailable(
            'native-qualification-prefill-streams-unavailable')
    _require(callable(getattr(owners, 'drain', None)),
             'native-qualification-prefill-streams-unavailable')
    prefill = owners.drain()
    _require(type(prefill) is int and prefill >= 0,
             'native-qualification-prefill-drain-unproved')
    return {'consumer_streams': streams, 'prefill_streams': prefill,
            'drained_streams': streams + prefill,
            'replay_census': census, 'external_users': status.external_users}


class NativeIdentity:
    """Minimal probe-like identity over the constructed native lane.

    The reviewed collector and observer only read these attributes.  They are
    derived from the native route controller, the retained storage owner, the
    native bridge and the authenticated qualification action, never from a
    probe object.
    """

    def __init__(self, runner, native):
        self.runner = runner
        self.native = native
        self._rpc_engine_epoch = None
        self._snapshot_max_bytes = None
        self._last_decode_descriptor = None
        self.quarantined = None
        self._shadow = None
        self.split = getattr(native, 'split', None)
        if self.split is None:
            raise NativeQualificationUnavailable(
                'native-qualification-split-context-unavailable')
        self.bank = getattr(native, 'bank', None)
        if self.bank is None:
            raise NativeQualificationUnavailable(
                'native-qualification-split-bank-unavailable')
        # ``bind_bank`` reads only these two identity fields from the router;
        # the native graph router keeps them on its lane handle/controller.
        identity = self

        class _RouterView:
            @property
            def storage_epoch(inner):
                return identity.storage_epoch

            @property
            def route_generation(inner):
                return identity.route_generation

        self.router = _RouterView()
        self.controller = self

    # -- observer/collector read seams ------------------------------------
    @property
    def route(self):
        return self.native.controller.route

    @property
    def route_generation(self):
        return self.native.controller.route_generation

    @property
    def storage_epoch(self):
        return _storage_epoch(self.native)

    @property
    def admitting(self):
        if self.quarantined is not None:
            return False
        if getattr(self.native.controller, 'admitting', None) is not True:
            return False
        return getattr(_router(self.runner).status(), 'unclean', None) is not True

    @property
    def in_flight(self):
        bridge = _bridge(self.runner, self.native)
        return bool(getattr(bridge, 'active', False))

    def _require_not_down(self):
        if self.quarantined is not None:
            raise NativeQualificationError('native-qualification-quarantined')
        _require(getattr(_router(self.runner).status(), 'unclean', None) is not True,
                 'native-qualification-router-unclean')

    def drain(self):
        record = _drain(self.runner, self.native, require_quiet=False)
        return record['drained_streams']

    # -- exact last FULL decode descriptor --------------------------------
    def watch_decode_descriptor(self):
        """Shadow the router's typed route query to retain the last FULL descriptor.

        This is a read-only counterpart of the probe's descriptor capture: the
        exact object the native driver passed to ``native_decode_route`` is
        retained for the observer's target proof.  The shadow is reversible and
        refuses on drift; no descriptor is ever synthesised.
        """
        if self._shadow is not None:
            return self
        router = _router(self.runner)
        original = getattr(router, 'native_decode_route', None)
        if not callable(original):
            raise NativeQualificationUnavailable(
                'native-qualification-decode-route-unavailable')
        had_instance_attr = 'native_decode_route' in getattr(router, '__dict__', {})
        identity = self

        def native_decode_route(descriptor):
            result = original(descriptor)
            if getattr(descriptor, 'cg_mode', None) == 'full':
                identity._last_decode_descriptor = descriptor
            return result

        router.native_decode_route = native_decode_route
        self._shadow = {'router': router, 'original': original,
                        'wrapper': native_decode_route,
                        'had_instance_attr': had_instance_attr}
        return self

    def unwatch_decode_descriptor(self):
        entry = self._shadow
        if entry is None:
            return
        router = entry['router']
        if getattr(router, 'native_decode_route', None) is not entry['wrapper']:
            raise NativeQualificationUnavailable(
                'native-qualification-descriptor-shadow-drift')
        if entry['had_instance_attr']:
            router.native_decode_route = entry['original']
        else:
            delattr(router, 'native_decode_route')
        self._shadow = None

    def require_descriptor(self):
        descriptor = self._last_decode_descriptor
        if descriptor is None:
            raise NativeQualificationUnavailable(
                'native-qualification-last-decode-descriptor-unproved')
        return descriptor

    def _fail(self, reason):
        self.quarantined = reason
        self.native.qualification_failure = reason


def _identity(runner, native):
    owner = getattr(runner, STATE_ATTRIBUTE, None)
    if owner is not None and owner.runner is runner:
        return owner
    owner = NativeIdentity(runner, native)
    setattr(runner, STATE_ATTRIBUTE, owner)
    return owner


def release(runner):
    """Drop only this module's descriptor shadow; no engine ownership changes."""
    owner = getattr(runner, STATE_ATTRIBUTE, None)
    if owner is None:
        return
    owner.unwatch_decode_descriptor()
    delattr(runner, STATE_ATTRIBUTE)


def _common(runner, native, identity, command, drain, *, ack):
    return {'rank': _rank(runner, native), 'ack': ack, 'mode': MODE,
            'engine_epoch': command['engine_epoch'], 'route': identity.route,
            'route_generation': identity.route_generation,
            'storage_epoch': identity.storage_epoch,
            'worker_incarnation': _incarnation(native),
            'drained_streams': drain['drained_streams'],
            'consumer_streams': drain['consumer_streams'],
            'prefill_streams': drain['prefill_streams'],
            'replay_census': drain['replay_census'],
            'native_route': True}


def handle_worker(runner, command):
    """Exact worker dispatch for the native teacher/observer actions."""
    if type(command) is not dict:
        raise NativeQualificationUnavailable('native-qualification-command-not-object')
    action = command.get('action')
    if action not in ACTIONS:
        raise NativeQualificationUnavailable('native-qualification-unknown-action')
    if set(command) != _FIELDS[action]:
        raise NativeQualificationUnavailable('native-qualification-fields-differ')
    native = _native(runner)
    epoch = _bind_epoch(native, command.get('engine_epoch'))
    generation = command.get('route_generation')
    _require(type(generation) is int and generation >= 0
             and generation == native.controller.route_generation,
             'native-qualification-route-generation-differs')
    if action in ('row1_configure', 'row1_evidence'):
        # Return the actual source-store packet verbatim. The worker envelope
        # carries live route/drain telemetry separately; it must not rename
        # the retained qualification generation after a drained FULL return.
        _drain(runner, native, require_quiet=True)
        server = _bridge(runner, native).server
        from native_resources import WorkerHandoffServer
        _require(isinstance(server, WorkerHandoffServer),
                 'native-qualification-source-server-unavailable')
        if action == 'row1_configure':
            return server.configure_row1(enabled=command['enabled'],
                run_id=command['run_id'], engine_epoch=epoch,
                route_generation=generation)
        return server.row1_evidence(run_id=command['run_id'])
    identity = _identity(runner, native)
    identity._rpc_engine_epoch = epoch
    try:
        if action == 'teacher_configure':
            return _teacher_configure(runner, native, identity, command)
        if action == 'teacher_evidence':
            return _teacher_evidence(runner, native, identity, command)
        if action == 'observer_begin':
            return _observer(runner, native, identity, command, epoch, begin=True)
        return _observer(runner, native, identity, command, epoch, begin=False)
    except BaseException:
        identity._fail('native qualification action failed')
        raise


def _teacher_configure(runner, native, identity, command):
    from teacher_decode import configure_worker

    enabled = command['enabled']
    snapshot = command['snapshot_max_bytes']
    if (type(enabled) is not bool or type(snapshot) is not int
            or not SNAPSHOT_MIN <= snapshot <= SNAPSHOT_MAX):
        raise NativeQualificationUnavailable(
            'native-qualification-teacher-configuration-invalid')
    _require(getattr(native.controller, 'route', None) == 'full',
             'native-qualification-teacher-configuration-needs-full-route')
    _require(getattr(native.controller, 'admitting', None) is True,
             'native-qualification-controller-not-admitting')
    req_states = getattr(runner, 'req_states', None)
    _require(getattr(req_states, 'req_id_to_index', None) == {},
             'native-qualification-teacher-configuration-needs-empty-worker')
    drain = _drain(runner, native, require_quiet=True)
    rank = _rank(runner, native)
    result = configure_worker(runner, enabled, rank=rank, identity=identity)
    if enabled:
        if getattr(runner, 'is_last_pp_rank', False):
            state = getattr(runner, '_flash_s5_teacher_trace', None)
            _require(state is not None,
                     'native-qualification-teacher-state-unavailable')
            # The installer receives the native identity on first creation.
            # Keep the same identity on reset; no probe object is installed.
            state.probe = identity
        # Every rank needs the last FULL descriptor for the observer target;
        # only the last PP rank owns the sampler trace state.
        identity.watch_decode_descriptor()
    else:
        identity.unwatch_decode_descriptor()
    identity._snapshot_max_bytes = snapshot
    result['snapshot_max_bytes'] = snapshot
    return dict(result, **_common(runner, native, identity, command, drain,
                                  ack='flash_s5_teacher_configure'))


def _teacher_evidence(runner, native, identity, command):
    from teacher_decode import worker_evidence

    if getattr(runner, 'is_last_pp_rank', False):
        state = getattr(runner, '_flash_s5_teacher_trace', None)
        _require(state is not None, 'native-qualification-teacher-state-unavailable')
        _require(state.probe is identity, 'native-qualification-teacher-identity-differs')
    drain = _drain(runner, native, require_quiet=True)
    evidence = worker_evidence(runner)
    return dict(evidence, **_common(runner, native, identity, command, drain,
                                    ack='flash_s5_teacher_evidence'))


def _observer(runner, native, identity, command, epoch, *, begin):
    import observer_runtime

    owner = getattr(runner, '_flash_s5_replay_observer', None)
    if owner is None:
        raise NativeQualificationUnavailable(
            'native-qualification-replay-observer-unavailable')
    _require(isinstance(owner, observer_runtime.WorkerObserver),
             'native-qualification-replay-observer-unavailable')
    _require(getattr(owner, 'engine_offers_enabled', None) is True,
             'native-qualification-observer-offers-disabled')
    _require(_text(command['run_id']) and _text(command['window_id']),
             'native-qualification-observer-window-identity-invalid')
    if not begin and owner.closed:
        _require(owner.engine_epoch == epoch,
                 'native-qualification-observer-retained-epoch-differs')
        _require(owner.window_identity == (command['run_id'], command['window_id']),
                 'native-qualification-observer-window-identity-changed')
        _require(type(owner.completed_reply) is dict,
                 'native-qualification-observer-retained-receipt-missing')
        drain = _drain(runner, native, require_quiet=True)
        # Outer worker telemetry carries the live generation. Keep the
        # completed window's own generation unchanged in the retained packet.
        result = dict(_common(runner, native, identity, command, drain,
                              ack='flash_s5_observer_end'))
        result.update(owner.completed_reply)
        result.update(current_route=identity.route,
                      current_route_generation=identity.route_generation)
        return result
    _require(identity.route == 'split',
             'native-qualification-observer-needs-split-route')
    if begin:
        drain = _drain(runner, native, require_quiet=False)
        identity.require_descriptor()
        result = owner.begin(identity, epoch, run_id=command['run_id'],
                             window_id=command['window_id'])
    else:
        drain = _drain(runner, native, require_quiet=False)
        if (owner.window_identity is not None
                and owner.window_identity != (command['run_id'], command['window_id'])):
            raise NativeQualificationUnavailable(
                'native-qualification-observer-window-identity-changed')
        result = owner.end(identity, epoch)
    return dict(result, **_common(runner, native, identity, command, drain,
                                  ack='flash_s5_observer_begin' if begin
                                  else 'flash_s5_observer_end'))


# -- API-owned teacher plan ownership -------------------------------------
def own_frontend_plan(control):
    """Return the single ``FrontendPlan`` for a constructed native control.

    The plan object is created once per input processor and reused, exactly
    like the reviewed probe control.  Requiring a real
    ``NativeControlClient`` blocks a probe controller or a bare processor from
    taking this seam; readiness itself is the control's own gate and is not
    repeated or bypassed here.
    """
    from native_overlap_control import NativeControlClient

    if not isinstance(control, NativeControlClient):
        raise NativeQualificationUnavailable(
            'native-qualification-requires-native-api-control')
    processor = getattr(getattr(control, 'engine', None), 'input_processor', None)
    if processor is None or not callable(getattr(processor, 'process_inputs', None)):
        raise NativeQualificationUnavailable('native-qualification-input-seam-unavailable')
    from teacher_decode import FrontendPlan, TeacherDecodeError

    owner = getattr(processor, '_flash_s5_teacher', None)
    if owner is None:
        try:
            owner = FrontendPlan(processor)
        except TeacherDecodeError as exc:
            raise NativeQualificationUnavailable(str(exc)) from None
    elif not isinstance(owner, FrontendPlan):
        raise NativeQualificationUnavailable(
            'native-qualification-teacher-plan-owner-differs')
    return owner


def configure_frontend(control, plan):
    """Validate/install the one cohort plan, or clear it with ``plan=None``."""
    from teacher_decode import TeacherDecodeError

    owner = own_frontend_plan(control)
    try:
        return owner.configure(plan)
    except TeacherDecodeError as exc:
        raise NativeQualificationError(str(exc)) from None


def frontend_status(control):
    return own_frontend_plan(control).status()


__all__ = ['ACTIONS', 'MODE', 'NativeIdentity', 'NativeQualificationError',
           'NativeQualificationUnavailable', 'configure_frontend',
           'frontend_status', 'handle_worker', 'own_frontend_plan', 'release']
