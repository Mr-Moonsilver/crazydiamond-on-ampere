"""Narrow authenticated frontend adapter for native qualification.

Two allowlisted actions are served, both in the **API / input-processor**
process, both behind the existing loopback + bearer authentication and both
bounded:

``qualification_frontend``
    Own the single teacher cohort plan through
    :func:`native_qualification.configure_frontend` over a constructed
    :class:`native_overlap_control.NativeControlClient`.  The plan is installed
    on the actual ``AsyncLLM.input_processor``; this module never touches the
    sampler, the engine driver or a worker.

``qualification_worker``
    Dispatch one of the exact worker actions through the API owner's
    :meth:`NativeControlClient.qualification` (the pinned named EngineCore
    utility).  The four teacher/observer actions are unchanged; the two
    additive row1 actions (``row1_configure``/``row1_evidence``) carry the
    scope-bound native source store packet.  No worker verdict,
    acknowledgement, drain or row1 record is synthesised here.

The two controls are deliberately **distinct objects**: the plan owner is the
API-side ``NativeControlClient`` that owns the input-processor plan, while
the dispatcher is a separate API client for the named EngineCore utility.
They are constructed and cached separately; the adapter refuses to run if a
single object is supplied for both.

This is deliberately not a general RPC surface: the field sets are exact, the
body is bounded, the only accepted actions are the two above, and the control
objects are constructed once and must stay the same incarnations.  Credentials
are read from the process environment / the existing ``api_key`` argument and
are never accepted in a command, returned, printed or logged.

Registration (the exact branch Codex installs in
``entrypoints/launchers/api_server/routers.py`` immediately after
``register_vllm_serve_api_routers(app)``)::

    from vllm.v1.flash_e3.native_qualification_frontend import attach_router as _flash_s5_native_qualification_frontend
    _flash_s5_native_qualification_frontend(args, app)

The adapter refuses to attach in broad server dev mode, exactly like the
reviewed probe control.  Importing this module has no side effect.
"""
import hashlib
import importlib
import inspect
import ipaddress
import json
import os
import secrets

FRONTEND_PATH = '/v1/flash_s5_native_qualification_frontend'
WORKER_PATH = '/v1/flash_s5_native_qualification_worker'
ACTION_FRONTEND = 'qualification_frontend'
ACTION_WORKER = 'qualification_worker'
WORKER_ACTIONS = ('teacher_configure', 'teacher_evidence',
                  'observer_begin', 'observer_end',
                  'row1_configure', 'row1_evidence')
ROW1_PAYLOAD_FIELDS = {
    'row1_configure': {'enabled', 'run_id'},
    'row1_evidence': {'run_id'},
}
FRONTEND_FIELDS = {'schema', 'action', 'run_id', 'command_seq', 'client_id',
                   'engine_epoch', 'enabled', 'plan'}
WORKER_FIELDS = {'schema', 'action', 'run_id', 'command_seq', 'client_id',
                 'engine_epoch', 'worker_action', 'payload'}
MAX_BODY = 65536
SNAPSHOT_MIN = 64 * 1024 ** 2
SNAPSHOT_MAX = 2 * 1024 ** 3


class NativeQualificationFrontendError(RuntimeError):
    """The adapter refused the command; nothing was installed or dispatched."""


def _text(value):
    return type(value) is str and bool(value.strip())


def _positive(value):
    return type(value) is int and value >= 1


def _require(condition, code):
    if not condition:
        raise NativeQualificationFrontendError(code)


def _peer_module(name):
    """Import a peer module relative to this package, else as a top-level one."""
    if __package__:
        try:
            return importlib.import_module('.' + name, __package__)
        except ImportError:
            pass
    return importlib.import_module(name)


def validate_frontend(command):
    """Exact shape of the one plan-ownership action (no extra fields)."""
    _require(type(command) is dict, 'frontend-command-not-an-object')
    _require(set(command) == FRONTEND_FIELDS, 'frontend-fields-differ')
    _require(command.get('schema') == 1, 'frontend-schema-differs')
    _require(command.get('action') == ACTION_FRONTEND, 'frontend-action-differs')
    _require(_text(command.get('run_id')), 'frontend-run-id-invalid')
    _require(type(command.get('command_seq')) is int
             and command['command_seq'] >= 1, 'frontend-sequence-invalid')
    _require(_text(command.get('client_id')), 'frontend-client-id-invalid')
    _require(_positive(command.get('engine_epoch')), 'frontend-engine-epoch-invalid')
    _require(type(command.get('enabled')) is bool, 'frontend-enabled-not-a-bool')
    plan = command.get('plan')
    if command['enabled']:
        _require(type(plan) is dict, 'frontend-enabled-needs-a-plan')
    else:
        _require(plan is None, 'frontend-disabled-must-clear-the-plan')
    return command


def validate_worker(command):
    """Exact shape of one worker action dispatch."""
    _require(type(command) is dict, 'worker-command-not-an-object')
    _require(set(command) == WORKER_FIELDS, 'worker-fields-differ')
    _require(command.get('schema') == 1, 'worker-schema-differs')
    _require(command.get('action') == ACTION_WORKER, 'worker-action-differs')
    _require(_text(command.get('run_id')), 'worker-run-id-invalid')
    _require(type(command.get('command_seq')) is int
             and command['command_seq'] >= 1, 'worker-sequence-invalid')
    _require(_text(command.get('client_id')), 'worker-client-id-invalid')
    _require(_positive(command.get('engine_epoch')), 'worker-engine-epoch-invalid')
    _require(command.get('worker_action') in WORKER_ACTIONS,
             'worker-action-not-allowlisted')
    _require(type(command.get('payload')) is dict, 'worker-payload-not-an-object')
    payload = command['payload']
    exact = ROW1_PAYLOAD_FIELDS.get(command['worker_action'])
    if exact is not None:
        _require(set(payload) == exact, 'worker-payload-fields-differ')
        _require(_text(payload.get('run_id')), 'worker-payload-run-id-invalid')
        if command['worker_action'] == 'row1_configure':
            _require(type(payload.get('enabled')) is bool,
                     'worker-payload-enabled-not-a-bool')
    return command


def _result(command, action, **fields):
    record = {'schema': 1, 'action': action, 'run_id': command['run_id'],
              'command_seq': command['command_seq'],
              'client_id': command['client_id'],
              'engine_epoch': command['engine_epoch']}
    record.update(fields)
    return record


def handle_frontend(command, *, control, qualification=None, status=None):
    """Install/clear the one plan on the real input processor.

    ``control`` is the constructed native control; ``qualification`` defaults to
    the reviewed module and ``status`` optionally records a receipt after the
    call.  The plan is handed to ``configure_frontend`` unchanged: validation and
    ownership stay in that module, not here.
    """
    validate_frontend(command)
    if control is None:
        raise NativeQualificationFrontendError('frontend-control-unavailable')
    if qualification is None:
        qualification = _peer_module('native_qualification')
    plan = command['plan'] if command['enabled'] else None
    try:
        receipt = qualification.configure_frontend(control, plan)
    except qualification.NativeQualificationError as exc:
        raise NativeQualificationFrontendError(str(exc)) from None
    if type(receipt) is not dict:
        raise NativeQualificationFrontendError('frontend-returned-no-record')
    observed = None
    if status is not None:
        observed = status(control)
    else:
        try:
            observed = qualification.frontend_status(control)
        except qualification.NativeQualificationError as exc:
            raise NativeQualificationFrontendError(str(exc)) from None
    if type(observed) is not dict:
        raise NativeQualificationFrontendError('frontend-status-unavailable')
    if observed.get('enabled') is not command['enabled']:
        raise NativeQualificationFrontendError('frontend-status-enabled-differs')
    return _result(command, ACTION_FRONTEND, enabled=observed.get('enabled'),
                   plan_sha256=observed.get('plan_sha256'),
                   admitted=observed.get('admitted'),
                   pending=observed.get('pending'),
                   steps=observed.get('steps'),
                   receipt={key: receipt.get(key) for key in
                            ('schema', 'enabled', 'plan_sha256', 'admitted',
                             'pending', 'steps') if key in receipt},
                   status=observed)


async def handle_worker(command, *, control):
    """Dispatch exactly one worker action over the owner's named utility.

    ``NativeControlClient.qualification`` is a coroutine on the reviewed peer,
    so the awaited reply is required; a synchronous control is also accepted.
    """
    validate_worker(command)
    if control is None:
        raise NativeQualificationFrontendError('worker-control-unavailable')
    dispatch = getattr(control, 'qualification', None)
    if not callable(dispatch):
        raise NativeQualificationFrontendError('worker-control-unavailable')
    payload = dict(command['payload'])
    # The epoch is carried by the worker command itself, exactly like the
    # reviewed control client; extra identity fields are never injected.
    payload.setdefault('engine_epoch', command['engine_epoch'])
    try:
        reply = dispatch(command['worker_action'], payload)
        if inspect.isawaitable(reply):
            reply = await reply
    except Exception as exc:  # noqa: BLE001 - the control's typed refusal
        raise NativeQualificationFrontendError(
            'worker-dispatch-refused:' + type(exc).__name__) from None
    if type(reply) is not dict:
        raise NativeQualificationFrontendError('worker-returned-no-record')
    if reply.get('engine_epoch') != command['engine_epoch']:
        raise NativeQualificationFrontendError('worker-engine-epoch-differs')
    return _result(command, ACTION_WORKER, worker_action=command['worker_action'],
                   worker_reply=reply)


def _auth_hashes(args, env):
    configured = getattr(args, 'api_key', None)
    if configured is not None and type(configured) is not list:
        raise NativeQualificationFrontendError('invalid-existing-api-auth-shape')
    tokens = [value for value in (configured or [env.get('VLLM_API_KEY')]) if value]
    if not tokens or any(type(value) is not str for value in tokens):
        raise NativeQualificationFrontendError(
            'native-qualification-frontend-requires-existing-api-auth')
    return [hashlib.sha256(value.encode()).digest() for value in tokens]


def attach_router(args, app, *, frontend_control=None, worker_control=None,
                  frontend_factory=None, worker_factory=None,
                  qualification=None, env=None):
    """Attach the two narrow actions to the serve app.

    ``frontend_control`` (API-side plan owner) and ``worker_control``
    (API-side named-utility dispatcher) may be supplied by the composition
    root; otherwise the corresponding factory constructs them from the app's
    ``engine_client``.  The two controls are always distinct objects.  Returns
    the small mutable state the caller may inspect (never containing
    credentials).
    """
    env = os.environ if env is None else env
    if env.get('FLASH_S5_OVERLAP', '0') != '1':
        return None
    if env.get('VLLM_SERVER_DEV_MODE', '0') not in ('0', ''):
        raise NativeQualificationFrontendError(
            'native-qualification-frontend-refuses-broad-dev-router')
    if frontend_control is not None and frontend_control is worker_control:
        raise NativeQualificationFrontendError(
            'api-and-core-control-objects-must-differ')
    hashes = _auth_hashes(args, env)
    if frontend_factory is None:
        def _default_frontend_factory(engine):
            NativeControlClient = _peer_module(
                'native_overlap_control').NativeControlClient
            return NativeControlClient(engine)
        frontend_factory = _default_frontend_factory
    if worker_factory is None:
        def _default_worker_factory(engine):
            # The named-utility dispatch client, not the engine-side owner.
            NativeControlClient = _peer_module(
                'native_overlap_control').NativeControlClient
            return NativeControlClient(engine)
        worker_factory = _default_worker_factory
    from fastapi import APIRouter, Request
    from fastapi.responses import JSONResponse

    router = APIRouter()
    state = {'engine': None, 'frontend': frontend_control,
             'worker': worker_control, 'qualification': qualification}

    async def endpoint(request):
        host = getattr(getattr(request, 'client', None), 'host', '')
        try:
            local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = False
        header = request.headers.get('authorization', '')
        scheme, _, value = header.partition(' ')
        candidate = hashlib.sha256(value.encode()).digest()
        authorized = scheme.lower() == 'bearer' and any(
            secrets.compare_digest(candidate, item) for item in hashes)
        if not local or not authorized:
            return JSONResponse({'error': 'native-qualification-auth-required'},
                                status_code=403)
        data = bytearray()
        try:
            async for chunk in request.stream():
                data.extend(chunk)
                if len(data) > MAX_BODY:
                    raise NativeQualificationFrontendError('command-too-large')
            command = json.loads(data)
        except (ValueError, TypeError):
            return JSONResponse({'error': 'invalid-command-json'}, status_code=400)
        except NativeQualificationFrontendError as exc:
            return JSONResponse({'error': str(exc)}, status_code=409)
        engine = request.app.state.engine_client
        if state['engine'] is None:
            state['engine'] = engine
        elif state['engine'] is not engine:
            return JSONResponse(
                {'error': 'engine-client-incarnation-changed'}, status_code=409)
        if state['frontend'] is None:
            try:
                state['frontend'] = frontend_factory(engine)
            except Exception as exc:  # noqa: BLE001 - construction refused
                return JSONResponse(
                    {'error': 'frontend-control-unavailable:'
                              + type(exc).__name__}, status_code=409)
        if state['worker'] is None:
            try:
                state['worker'] = worker_factory(engine)
            except Exception as exc:  # noqa: BLE001 - construction refused
                return JSONResponse(
                    {'error': 'worker-control-unavailable:'
                              + type(exc).__name__}, status_code=409)
        if state['frontend'] is state['worker']:
            return JSONResponse(
                {'error': 'api-and-core-control-objects-must-differ'},
                status_code=409)
        try:
            if request.url.path == FRONTEND_PATH:
                result = handle_frontend(
                    command, control=state['frontend'],
                    qualification=state['qualification'])
            else:
                result = await handle_worker(command, control=state['worker'])
        except NativeQualificationFrontendError as exc:
            return JSONResponse({'error': str(exc)}, status_code=409)
        return JSONResponse(result)

    endpoint.__annotations__['request'] = Request
    router.add_api_route(FRONTEND_PATH, endpoint, methods=['POST'])
    router.add_api_route(WORKER_PATH, endpoint, methods=['POST'])
    app.include_router(router)
    return state


__all__ = ['ACTION_FRONTEND', 'ACTION_WORKER', 'FRONTEND_FIELDS', 'FRONTEND_PATH',
           'NativeQualificationFrontendError', 'WORKER_ACTIONS', 'WORKER_FIELDS',
           'WORKER_PATH', 'attach_router', 'handle_frontend', 'handle_worker',
           'validate_frontend', 'validate_worker']
