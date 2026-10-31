"""Real generated-package startup/closure/relay binding; CPU filesystem/OS fakes.

No guard/capability/ready flag is patched. The real source/build checker reads
a self-consistent *explicitly synthetic* CPU review/build fixture. Fake ELF
bytes are hashed but never loaded. Only external vLLM types and engine queues /
scheduler stand in for the unavailable running process. No GPU/release claim.
"""
import ast
import dataclasses
import enum
import asyncio
import contextlib
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'patches/s5_engine'), str(ROOT/'harness')]
import make_s5_overlap as builder
import decode_probe
import native_readiness as readiness
import e3_finalize_package as finalizer
from test_e3_native_readiness import source_package, write_json
import test_e3_native_integration as boundary
import test_e3_admission_metadata as allocator_boundary


@contextlib.contextmanager
def generated_package():
    with tempfile.TemporaryDirectory(prefix='native-startup-binding-') as tmp:
        root = Path(tmp)
        source, closure, receipt = source_package(root)
        e3 = source/'e3'
        (e3/'__init__.py').write_text('"CPU generated-package fixture; no GPU."\n')
        for name in decode_probe.GENERATED_SOURCES:
            (e3/(name+'.py')).write_text(builder.package(
                (ROOT/'patches/s5_engine'/(name+'.py')).read_text()))
        closure['generated_source_shas'] = {
            name:readiness.file_sha(e3/(name+'.py')) for name in decode_probe.GENERATED_SOURCES}
        closure['review_sha256'] = readiness.digest(closure['review'])
        write_json(e3/readiness.CLOSURE, closure)
        generation = json.loads((source/finalizer.GENERATION).read_text())
        generation['implementation_review_sha256'] = closure['review_sha256']
        write_json(source/finalizer.GENERATION, generation)
        excluded = {*readiness.LIBS, 'q17-helper-build.json', 'SHA256SUMS'}
        (source/'SHA256SUMS').write_text(''.join(
            finalizer.sha(p)+'  '+p.relative_to(source).as_posix()+'\n'
            for p in sorted(source.rglob('*')) if p.is_file() and p.name not in excluded))
        final = root/'final'
        finalizer.finalize(source, final)
        name = '_native_startup_binding_fixture'
        spec = importlib.util.spec_from_file_location(name, final/'e3/__init__.py',
            submodule_search_locations=[str(final/'e3')])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        try:
            yield name, final, closure
        finally:
            for key in tuple(sys.modules):
                if key == name or key.startswith(name+'.'):
                    del sys.modules[key]


def make_engine(prefix=None):
    success = object()
    if prefix is None:
        workers = [types.SimpleNamespace(rank=i) for i in range(4)]
    else:
        import test_e3_native_overlap as connected
        rig = connected.cold_stack()
        rig.scheduler.waiting.items.clear()
        def translate(value, packaged):
            cls = type(value)
            module = cls.__module__
            if dataclasses.is_dataclass(value) or isinstance(value, enum.Enum):
                name = module.rsplit('.', 1)[-1]
                target_module = prefix+'.'+name if packaged else name
                target = getattr(importlib.import_module(target_module), cls.__name__)
                if isinstance(value, enum.Enum):
                    return target(value.value)
                return target(**{f.name: translate(getattr(value, f.name), packaged)
                                 for f in dataclasses.fields(value) if f.init})
            if type(value) is dict:
                return {k: translate(v, packaged) for k,v in value.items()}
            if type(value) in (tuple,list):
                return type(value)(translate(v, packaged) for v in value)
            return value
        workers = []
        for rank, worker in rig.workers.items():
            def invoke(method, command, worker=worker):
                cmd = translate(command, False)
                fn = (worker.flash_s5_native_overlap if method == 'native_overlap'
                      else getattr(worker._flash_s5_bridge, method))
                return translate(fn(cmd), True)
            workers.append(types.SimpleNamespace(rank=rank,
                flash_s5_bootstrap=lambda c, invoke=invoke: invoke('bootstrap', c),
                flash_s5_transition=lambda c, invoke=invoke: invoke('transition', c),
                flash_s5_native_overlap=lambda c, invoke=invoke: invoke('native_overlap', c)))
    executor = boundary.FakeExecutor(workers, success)
    pool = allocator_boundary.FixturePool()
    manager = allocator_boundary.FixtureManager(pool, [allocator_boundary.FixtureGroup(pool)])
    scheduler = allocator_boundary.FixtureScheduler(manager)
    engine = boundary.FakeEngine(executor, scheduler)
    if prefix is not None:
        class Pause(enum.StrEnum):
            UNPAUSED = 'UNPAUSED'
            PAUSED_ALL = 'PAUSED_ALL'
        scheduler.pause_state = Pause.UNPAUSED
        scheduler.set_pause_state = lambda state: setattr(scheduler, 'pause_state', state)
        engine.is_scheduler_paused = lambda: scheduler.pause_state != Pause.UNPAUSED
        enqueue = executor.enqueue
        def delivered(item):
            enqueue(item)
            executor.deliver()
        executor.enqueue = delivered
        executor.shutdown = lambda: getattr(executor, '_flash_s5_transport', None).close(.1)
    engine.vllm_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(hf_text_config=types.SimpleNamespace(num_hidden_layers=48)),
        parallel_config=types.SimpleNamespace(pipeline_parallel_size=2))
    return engine, success


class ActualStartupBindingTests(unittest.TestCase):
    def test_real_generated_engine_startup_uses_real_closure_transport_backend_and_driver(self):
        with generated_package() as (prefix, final, closure):
            startup = importlib.import_module(prefix+'.s5_startup')
            hooks = importlib.import_module(prefix+'.executor_hooks')
            backend = importlib.import_module(prefix+'.native_backend')
            handoff = importlib.import_module(prefix+'.async_handoff')
            ready = importlib.import_module(prefix+'.native_readiness')
            driver_module = importlib.import_module(prefix+'.engine_driver')
            engine, success = make_engine(prefix)
            vllm = boundary.fake_vllm(boundary.FakeForwardContext(),
                                    boundary.FakeWorkspaceModule(), success)
            env = {'FLASH_S5_OVERLAP':'1', 'VLLM_PP_LAYER_PARTITION':'24,24'}
            with patch.dict(sys.modules, vllm), patch.dict(os.environ, env):
                # Run the actual generated EngineCore hook, no install/guard mocks.
                result = hooks.after_engine_startup(engine)
            self.assertTrue(engine._flash_s5_driver._bootstrapped)
            self.assertIsNotNone(engine._flash_s5_startup_ready)
            self.assertFalse(engine._flash_s5_driver._split_ready)
            self.assertIs(result, engine._flash_s5_composition)
            self.assertIsInstance(result, handoff.Composition)
            self.assertIsInstance(result.dispatcher.backend, backend.RemoteCompletionBackend)
            self.assertTrue(result.dispatcher.backend.native_verified)
            self.assertIs(result.dispatcher.backend._transport, engine.model_executor._flash_s5_transport)
            self.assertIsInstance(engine._flash_s5_driver, driver_module.NativeSegmentDriver)
            self.assertIsNotNone(engine.scheduler._flash_s5_admission_metadata)
            self.assertTrue(ready.require_implementation_closure()['native_ready'])
            self.assertFalse(ready.require_implementation_closure()['hardware_qualified'])
            with self.assertRaises(backend.NativeBackendError):
                result.dispatcher.backend.submit(object())
            self.assertEqual(result.dispatcher.backend.poll(object()), ())
            self.assertTrue(engine.model_executor._flash_s5_transport.close(.2)['clean'])

    def test_no_closure_refuses_before_executor_receiver_claim(self):
        with generated_package() as (prefix, final, _):
            hooks = importlib.import_module(prefix+'.executor_hooks')
            ready = importlib.import_module(prefix+'.native_readiness')
            path = final/'e3'/readiness.CLOSURE
            # Preserve the fixture bytes under another name; make the real
            # package-local lookup fail. No ready result or guard is mocked.
            path.rename(path.with_suffix('.retained'))
            engine, _ = make_engine()
            with patch.dict(os.environ, {'FLASH_S5_OVERLAP':'1','VLLM_PP_LAYER_PARTITION':'24,24'}):
                with self.assertRaises(Exception):
                    hooks.after_engine_startup(engine)
            self.assertFalse(hasattr(engine.model_executor, '_flash_s5_transport'))

    def test_unbound_relay_and_generic_events_never_gain_native_capability(self):
        with generated_package() as (prefix, final, _):
            backend = importlib.import_module(prefix+'.native_backend')
            handoff = importlib.import_module(prefix+'.async_handoff')
            config = handoff.HandoffConfig.from_env({'FLASH_S5_OVERLAP':'1'})
            relay = backend.RemoteCompletionBackend(config)
            self.assertFalse(relay.native_verified)
            with self.assertRaises(handoff.ConfigurationError):
                handoff.require_native_backend(relay)
            with self.assertRaises(AttributeError):
                relay.native_verified = True
            device = backend.NativeDevice(boundary.FakeTorch(), 'cuda:0')
            event = backend.EventCompletionBackend(device)
            self.assertFalse(event.native_verified)
            with self.assertRaises(handoff.ConfigurationError):
                handoff.require_native_backend(event)

    def test_relay_refuses_foreign_transport_lost_claim_and_worker_use(self):
        with generated_package() as (prefix, final, _):
            backend = importlib.import_module(prefix+'.native_backend')
            handoff = importlib.import_module(prefix+'.async_handoff')
            hooks = importlib.import_module(prefix+'.executor_hooks')
            config = handoff.HandoffConfig.from_env({'FLASH_S5_OVERLAP':'1'})
            with self.assertRaises(handoff.ConfigurationError):
                backend.RemoteCompletionBackend(config, transport=object())
            engine, success = make_engine(prefix)
            transport = hooks.install(engine.model_executor, config=config, success_status=success)
            try:
                relay = backend.RemoteCompletionBackend(config, transport=transport)
                self.assertTrue(relay.native_verified)
                with self.assertRaisesRegex(handoff.ConfigurationError, 'engine relay'):
                    handoff.build_composition({'FLASH_S5_OVERLAP':'1'}, transport=transport,
                        backend=relay, lanes={}, side='worker', native_allocations={'fixture':1})
                queue = next(iter(transport._queues.values()))
                queue._flash_s5_receiver_owner = None
                try:
                    with self.assertRaisesRegex(handoff.ConfigurationError, 'ownership'):
                        handoff.require_native_backend(relay)
                finally:
                    queue._flash_s5_receiver_owner = transport
            finally:
                transport.close(.2)

    def test_real_closure_rejects_tampered_source_without_ready_mock(self):
        with generated_package() as (prefix, final, _):
            hooks = importlib.import_module(prefix+'.executor_hooks')
            file = final/'e3/native_backend.py'
            file.write_text(file.read_text()+'\n# changed after closure\n')
            engine, _ = make_engine()
            with patch.dict(os.environ, {'FLASH_S5_OVERLAP':'1','VLLM_PP_LAYER_PARTITION':'24,24'}):
                with self.assertRaisesRegex(Exception, 'generated source differs'):
                    hooks.after_engine_startup(engine)
            self.assertFalse(hasattr(engine.model_executor, '_flash_s5_transport'))

    def test_all_generated_local_imports_keep_package_identity(self):
        for name in builder.E3_MODULES:
            text = builder.package((ROOT/'patches/s5_engine'/(name+'.py')).read_text())
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, ast.Import):
                    self.assertFalse({a.name for a in node.names} & set(builder.E3_MODULES), name)
                if isinstance(node, ast.ImportFrom) and node.module in builder.E3_MODULES:
                    self.assertGreater(node.level, 0, name)


AUTH_PINS = {
    'entrypoints/serve/middleware/authenticate.py':'0633540cb788738dfee5a69075fbdc30eb98a510e26f69a0b43c57c50d9ef5ab',
    'entrypoints/serve/middleware/register.py':'cff2d2c00e16f45a942eb1c1ef591834cb58d34ec9eefb3269f9134c4b1087bf',
}


def pinned_auth():
    """Execute pinned auth code; only Starlette's header/response IO is doubled."""
    reference = ROOT/'reference/vllm-src'
    for name, digest in AUTH_PINS.items():
        if hashlib.sha256((reference/name).read_bytes()).hexdigest() != digest:
            raise AssertionError('pinned auth reference changed: '+name)
    class Headers:
        def __init__(self, scope):
            self.values = {k.decode().lower():v.decode() for k,v in scope['headers']}
        def get(self, name):
            return self.values.get(name.lower())
    class Response:
        def __init__(self, content, status_code):
            self.status_code = status_code
        async def __call__(self, scope, receive, send):
            await send({'type':'http.response.start','status':self.status_code})
    modules = {
        'starlette':types.ModuleType('starlette'),
        'starlette.datastructures':types.SimpleNamespace(Headers=Headers),
        'starlette.responses':types.SimpleNamespace(JSONResponse=Response),
        'starlette.types':types.SimpleNamespace(ASGIApp=object,Receive=object,Scope=object,Send=object),
    }
    module = types.ModuleType('_native_pinned_auth.authenticate')
    source = (reference/'entrypoints/serve/middleware/authenticate.py').read_text()
    with patch.dict(sys.modules, modules):
        exec(compile(source, str(reference/'entrypoints/serve/middleware/authenticate.py'),'exec'), module.__dict__)
    return module


class PinnedAuthBindingTests(unittest.TestCase):
    def test_real_private_client_config_binds_to_pinned_auth_and_wrong_key_is_401(self):
        import canary_hub_candidate_runner as canary
        import e3_private_auth as auth
        middleware = pinned_auth().AuthenticationMiddleware
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);os.chmod(root,0o700)
            auth.prepare(root/'key.env',{'VLLM_API_KEY':'fixture-native-auth-token'})
            original = {'env':{'ANTHROPIC_AUTH_TOKEN':'old-fixture-token'}}
            config = canary.prepare_client_config(original,str(root/'key.env'),root)
            token = json.loads((config/'settings.json').read_text())['env']['ANTHROPIC_AUTH_TOKEN']
            self.assertEqual(original['env']['ANTHROPIC_AUTH_TOKEN'],'old-fixture-token')
            seen = []
            async def app(scope, receive, send):
                await send({'type':'http.response.start','status':204})
            guard = middleware(app,[auth.read_private(root/'key.env')])
            async def run(value, path='/v1/completions'):
                async def send(row):seen.append(row['status'])
                scope={'type':'http','method':'POST','path':path,'headers':[(b'authorization',value.encode())]}
                await guard(scope,None,send)
            asyncio.run(run('Bearer '+token))
            asyncio.run(run('Bearer old-fixture-token'))
            asyncio.run(run('Basic '+token))
            self.assertEqual(seen,[204,401,401])

    def test_pinned_registration_uses_env_key_with_cli_precedence(self):
        auth = pinned_auth()
        reference = ROOT/'reference/vllm-src'
        source = (reference/'entrypoints/serve/middleware/register.py').read_text()
        node = next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef)
                    and x.name=='init_entrypoints_middleware')
        module = ast.Module(body=[ast.ImportFrom(module='__future__',
            names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
        namespace={'__package__':'_native_pinned_auth','envs':types.SimpleNamespace(
            VLLM_API_KEY='fixture-env-token',VLLM_DEBUG_LOG_API_SERVER_RESPONSE=False),
            'CORSMiddleware':object}
        exec(compile(ast.fix_missing_locations(module),'<pinned-auth-register>','exec'),namespace)
        class App:
            def __init__(self):self.calls=[]
            def add_middleware(self, cls, **kwargs):self.calls.append((cls,kwargs))
        for cli, expected in ((None,['fixture-env-token']),(['fixture-cli-token'],['fixture-cli-token'])):
            app=App();args=types.SimpleNamespace(api_key=cli,allowed_origins=[],allow_credentials=False,
                allowed_methods=[],allowed_headers=[],enable_request_id_headers=False,middleware=[])
            package=types.ModuleType('_native_pinned_auth');package.__path__=[]
            with patch.dict(sys.modules,{'_native_pinned_auth':package,'_native_pinned_auth.authenticate':auth}):
                namespace['init_entrypoints_middleware'](args,app,())
            self.assertIs(app.calls[-1][0],auth.AuthenticationMiddleware)
            self.assertEqual(app.calls[-1][1]['tokens'],expected)


if __name__ == '__main__':
    unittest.main()
