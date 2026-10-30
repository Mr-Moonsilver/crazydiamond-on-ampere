#!/usr/bin/env python3
"""Focused CPU tests for the native-qualification frontend adapter.

The tests use the actual composed runtime module for the plan
ownership and exact-field refusals, and fakes only at the network boundary
(FastAPI, HTTP request, transport).  No credentials are printed or stored.
"""
import asyncio
import copy
import hashlib
import importlib.util
import json
import os
import sys
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PATCHES = os.path.join(os.path.dirname(HERE), 'patches', 's5_engine')
PEER_L = os.path.join(PATCHES, 'native_qualification.py')
if PATCHES not in sys.path:
    sys.path.insert(0, PATCHES)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import native_qualification_frontend as adapter  # noqa: E402
import teacher_decode  # noqa: E402


def load_peer():
    spec = importlib.util.spec_from_file_location('peer_native_qualification',
                                                  PEER_L)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeProcessor:
    """The actual API input-processor object the plan must reach."""

    def __init__(self):
        self._flash_s5_teacher = None
        self.model_config = object()
        self.speculative_config = None

    def process_inputs(self, request):
        return request


class FakeParams:
    def __init__(self, **fields):
        self.trace_decode_token_ids = None
        self.n = 1
        self.max_tokens = teacher_decode.STEPS
        self.min_tokens = teacher_decode.STEPS
        self.temperature = 0
        self.ignore_eos = True
        self.logprobs = 20
        self.stop = None
        self.stop_token_ids = None
        self.__dict__.update(fields)

    def _validate_trace_replay(self, model_config, speculative_config):
        assert speculative_config is None
        assert isinstance(self.trace_decode_token_ids, list)


class FakeRequest:
    def __init__(self, request_id, prompt_sha256, trace):
        self.request_id = request_id
        self.prompt_token_ids = [1, 2, 3]
        self.prompt_sha256 = prompt_sha256
        self.sampling_params = FakeParams()
        self.lora_request = None
        self.mm_features = None
        self.trace = trace


def make_plan():
    rows = []
    for stream in range(8):
        trace = [stream * 10000 + index for index in range(teacher_decode.STEPS)]
        rows.append({'request_id': 'tf-0-%d' % stream,
                     'prompt_sha256': teacher_decode.digest([1, 2, 3]),
                     'continuation': trace})
    return {'schema': teacher_decode.SCHEMA, 'requests': rows,
            'snapshot_max_bytes': 2 * 1024 ** 3}


class FakeFrontendControl:
    """Engine-side owner shape; class identity is provided by the test module."""

    def __init__(self, processor):
        self.engine = types.SimpleNamespace(input_processor=processor)


class FakeWorkerControl:
    """API-side named-utility dispatcher shape (``NativeControlClient``)."""

    def __init__(self, epoch=3):
        self.epoch = epoch
        self.calls = []

    async def qualification(self, worker_action, payload=None):
        self.calls.append((worker_action, copy.deepcopy(payload)))
        return {'schema': 1, 'action': 'qualification',
                'engine_epoch': payload.get('engine_epoch'), 'workers': []}


class PeerCompositionTest(unittest.TestCase):
    def setUp(self):
        self.peer = load_peer()
        self.fake_module = types.ModuleType('native_overlap_control')

        class NativeOverlapControl:
            pass

        class NativeControlClient:
            def __init__(self, engine):
                self.engine = engine
                self.calls = []

            async def qualification(self, worker_action, payload=None):
                self.calls.append((worker_action, copy.deepcopy(payload)))
                return {'schema': 1, 'engine_epoch': payload.get('engine_epoch'),
                        'workers': []}

        self.fake_module.NativeOverlapControl = NativeOverlapControl
        self.fake_module.NativeControlClient = NativeControlClient
        self.control_class = NativeControlClient
        self._saved = sys.modules.get('native_overlap_control')
        sys.modules['native_overlap_control'] = self.fake_module
        self.processor = FakeProcessor()

        class Control(NativeControlClient):
            def __init__(self, processor):
                self.engine = types.SimpleNamespace(input_processor=processor)

        self.Control = Control
        self.control = Control(self.processor)
        self.assertTrue(isinstance(self.control, NativeControlClient))

    def tearDown(self):
        if self._saved is None:
            sys.modules.pop('native_overlap_control', None)
        else:
            sys.modules['native_overlap_control'] = self._saved

    def command(self, enabled, plan):
        return {'schema': 1, 'action': adapter.ACTION_FRONTEND,
                'run_id': 'run-1', 'command_seq': 1, 'client_id': 'client-1',
                'engine_epoch': 3, 'enabled': enabled, 'plan': plan}

    def test_real_configure_reaches_the_actual_input_processor(self):
        plan = make_plan()
        result = adapter.handle_frontend(
            self.command(True, plan), control=self.control,
            qualification=self.peer)
        self.assertTrue(result['enabled'])
        self.assertEqual(result['plan_sha256'],
                         teacher_decode.digest(plan))
        owner = getattr(self.processor, '_flash_s5_teacher', None)
        self.assertIsInstance(owner, teacher_decode.FrontendPlan)
        self.assertEqual(owner.plan_sha, teacher_decode.digest(plan))
        # Execute the installed seam: the declared request receives the exact
        # continuation, so the plan really reached the input processor.
        for index, row in enumerate(plan['requests']):
            request = FakeRequest('cmpl-tf-0-%d-0' % index,
                                  row['prompt_sha256'],
                                  row['continuation'])
            out = self.processor.process_inputs(request)
            self.assertIs(out, request)
            self.assertEqual(out.sampling_params.trace_decode_token_ids,
                             row['continuation'])
        self.assertEqual(owner.admitted, 8)
        # Clearing removes ownership and is reported as disabled.
        cleared = adapter.handle_frontend(
            self.command(False, None), control=self.control,
            qualification=self.peer)
        self.assertFalse(cleared['enabled'])
        self.assertEqual(cleared['previous_status']['admitted'], 8)
        self.assertEqual(cleared['previous_status']['pending'], 0)
        self.assertEqual(cleared['previous_status']['plan_sha256'], teacher_decode.digest(plan))
        self.assertEqual(cleared['admitted'], 0)
        self.assertIsNone(self.processor._flash_s5_teacher.pending)

    def test_default_peer_import_reaches_the_actual_input_processor(self):
        saved = sys.modules.get('native_qualification')
        sys.modules['native_qualification'] = self.peer
        plan = make_plan()
        try:
            result = adapter.handle_frontend(
                self.command(True, plan), control=self.control)
            self.assertTrue(result['enabled'])
            self.assertIsInstance(getattr(self.processor, '_flash_s5_teacher', None),
                                  teacher_decode.FrontendPlan)
            for index, row in enumerate(plan['requests']):
                self.processor.process_inputs(
                    FakeRequest('cmpl-tf-0-%d-0' % index,
                                row['prompt_sha256'], row['continuation']))
            adapter.handle_frontend(self.command(False, None),
                                    control=self.control)
            self.assertIsNone(self.processor._flash_s5_teacher.pending)
        finally:
            if saved is None:
                sys.modules.pop('native_qualification', None)
            else:
                sys.modules['native_qualification'] = saved

    def test_status_mismatch_is_refused(self):
        class QualificationError(RuntimeError):
            pass

        lying = types.SimpleNamespace(
            NativeQualificationError=QualificationError,
            configure_frontend=lambda control, plan: {'schema': 1,
                                                     'enabled': True},
            frontend_status=lambda control: {'schema': 1, 'enabled': False})
        with self.assertRaises(adapter.NativeQualificationFrontendError) as caught:
            adapter.handle_frontend(self.command(True, make_plan()),
                                    control=self.control, qualification=lying)
        self.assertEqual(str(caught.exception),
                         'frontend-status-enabled-differs')

    def test_worker_accepts_engine_owned_generation_then_requires_native(self):
        # This is the exact command shape the native engine driver builds for
        # a qualification action: the client payload plus action/engine_epoch
        # and the engine-owned route_generation written last.
        command = {'action': 'teacher_configure', 'engine_epoch': 3,
                   'route_generation': 7, 'enabled': False,
                   'snapshot_max_bytes': 64 * 1024 ** 2}
        with self.assertRaises(self.peer.NativeQualificationUnavailable) as caught:
            self.peer.handle_worker(object(), command)
        self.assertEqual(str(caught.exception),
                         'native-qualification-worker-native-unavailable')

    def test_peer_worker_accepts_its_exact_fields(self):
        # No route_generation: the field set then matches the peer contract and
        # the action proceeds until the (absent) native construction.
        command = {'action': 'teacher_evidence', 'engine_epoch': 3, 'route_generation': 7}
        with self.assertRaises(self.peer.NativeQualificationUnavailable) as caught:
            self.peer.handle_worker(object(), command)
        self.assertEqual(str(caught.exception),
                         'native-qualification-worker-native-unavailable')


class FrontendValidationTest(unittest.TestCase):
    def test_exact_fields_and_clear_semantics(self):
        good = {'schema': 1, 'action': adapter.ACTION_FRONTEND, 'run_id': 'r',
                'command_seq': 1, 'client_id': 'c', 'engine_epoch': 1,
                'enabled': True, 'plan': {'schema': 's5-teacher-decode/1',
                                          'requests': []}}
        adapter.validate_frontend(good)
        for mutation, code in [
                ({'extra': 1}, 'frontend-fields-differ'),
                ({'schema': 2}, 'frontend-schema-differs'),
                ({'enabled': False}, 'frontend-disabled-must-clear-the-plan'),
                ({'enabled': True, 'plan': None}, 'frontend-enabled-needs-a-plan'),
                ({'command_seq': 0}, 'frontend-sequence-invalid'),
                ({'engine_epoch': 0}, 'frontend-engine-epoch-invalid')]:
            bad = dict(good)
            bad.update(mutation)
            with self.assertRaises(adapter.NativeQualificationFrontendError) as c:
                adapter.validate_frontend(bad)
            self.assertEqual(str(c.exception), code, mutation)

    def test_worker_allowlist_and_exact_fields(self):
        good = {'schema': 1, 'action': adapter.ACTION_WORKER, 'run_id': 'r',
                'command_seq': 1, 'client_id': 'c', 'engine_epoch': 1,
                'worker_action': 'observer_begin', 'payload': {}}
        adapter.validate_worker(good)
        for mutation, code in [
                ({'extra': 1}, 'worker-fields-differ'),
                ({'worker_action': 'status'}, 'worker-action-not-allowlisted'),
                ({'payload': []}, 'worker-payload-not-an-object'),
                ({'engine_epoch': 0}, 'worker-engine-epoch-invalid')]:
            bad = dict(good)
            bad.update(mutation)
            with self.assertRaises(adapter.NativeQualificationFrontendError) as c:
                adapter.validate_worker(bad)
            self.assertEqual(str(c.exception), code, mutation)

    def test_worker_dispatch_adds_epoch_and_never_a_credential(self):
        control = FakeWorkerControl(epoch=3)
        command = {'schema': 1, 'action': adapter.ACTION_WORKER, 'run_id': 'r',
                   'command_seq': 1, 'client_id': 'c', 'engine_epoch': 3,
                   'worker_action': 'observer_end',
                   'payload': {'action': 'observer_end', 'run_id': 'r',
                               'window_id': 'w'}}
        result = asyncio.run(adapter.handle_worker(command, control=control))
        self.assertEqual(control.calls[0][0], 'observer_end')
        payload = control.calls[0][1]
        self.assertEqual(payload['engine_epoch'], 3)
        self.assertNotIn('api_key', payload)
        self.assertNotIn('authorization', payload)
        self.assertEqual(result['worker_action'], 'observer_end')

    def test_row1_worker_actions_are_allowlisted_with_exact_payloads(self):
        configure = {'schema': 1, 'action': adapter.ACTION_WORKER,
                     'run_id': 'r', 'command_seq': 1, 'client_id': 'c',
                     'engine_epoch': 1, 'worker_action': 'row1_configure',
                     'payload': {'enabled': True, 'run_id': 'r'}}
        adapter.validate_worker(configure)
        for payload, code in [
                ({'enabled': True}, 'worker-payload-fields-differ'),
                ({'enabled': 1, 'run_id': 'r'},
                 'worker-payload-enabled-not-a-bool'),
                ({'enabled': False, 'run_id': ' '},
                 'worker-payload-run-id-invalid'),
                ({'enabled': True, 'run_id': 'r', 'extra': 1},
                 'worker-payload-fields-differ')]:
            bad = copy.deepcopy(configure)
            bad['payload'] = payload
            with self.assertRaises(adapter.NativeQualificationFrontendError) as c:
                adapter.validate_worker(bad)
            self.assertEqual(str(c.exception), code, payload)
        evidence = copy.deepcopy(configure)
        evidence['worker_action'] = 'row1_evidence'
        evidence['payload'] = {'run_id': 'r'}
        adapter.validate_worker(evidence)
        for payload in ({}, {'run_id': 'r', 'enabled': False},
                        {'run_id': 'r', 'extra': 1}):
            bad = copy.deepcopy(evidence)
            bad['payload'] = payload
            with self.assertRaises(adapter.NativeQualificationFrontendError) as c:
                adapter.validate_worker(bad)
            self.assertEqual(str(c.exception), 'worker-payload-fields-differ')

    def test_row1_evidence_dispatch_carries_only_run_id_and_epoch(self):
        control = FakeWorkerControl(epoch=3)
        command = {'schema': 1, 'action': adapter.ACTION_WORKER, 'run_id': 'r',
                   'command_seq': 1, 'client_id': 'c', 'engine_epoch': 3,
                   'worker_action': 'row1_evidence',
                   'payload': {'run_id': 'r'}}
        result = asyncio.run(adapter.handle_worker(command, control=control))
        self.assertEqual(control.calls[0],
                         ('row1_evidence',
                          {'run_id': 'r', 'engine_epoch': 3}))
        self.assertEqual(result['worker_reply']['engine_epoch'], 3)
        self.assertNotIn('api_key', repr(control.calls))
        self.assertNotIn('authorization', repr(control.calls))

    def test_worker_epoch_difference_is_refused(self):
        class WrongEpoch(FakeWorkerControl):
            async def qualification(self, worker_action, payload=None):
                return {'schema': 1, 'engine_epoch': 9}
        command = {'schema': 1, 'action': adapter.ACTION_WORKER, 'run_id': 'r',
                   'command_seq': 1, 'client_id': 'c', 'engine_epoch': 3,
                   'worker_action': 'teacher_evidence',
                   'payload': {'action': 'teacher_evidence'}}
        with self.assertRaises(adapter.NativeQualificationFrontendError) as c:
            asyncio.run(adapter.handle_worker(command, control=WrongEpoch()))
        self.assertEqual(str(c.exception), 'worker-engine-epoch-differs')


class FakeJSONResponse:
    def __init__(self, content, status_code=200):
        self.content = content
        self.status_code = status_code


class FakeAPIRouter:
    def __init__(self):
        self.routes = []

    def add_api_route(self, path, endpoint, methods):
        self.routes.append((path, endpoint, tuple(methods)))


class FakeApp:
    def __init__(self):
        self.routers = []
        self.state = types.SimpleNamespace(engine_client=object())

    def include_router(self, router):
        self.routers.append(router)


class FakeHTTPRequest:
    def __init__(self, *, body=b'', path=adapter.FRONTEND_PATH,
                 host='127.0.0.1', auth='Bearer test-key', app=None,
                 chunks=None):
        self._body = body
        self._chunks = chunks
        self.url = types.SimpleNamespace(path=path)
        self.client = types.SimpleNamespace(host=host)
        self.headers = {'authorization': auth}
        self.app = app

    async def stream(self):
        if self._chunks is not None:
            for chunk in self._chunks:
                yield chunk
            return
        yield self._body


class RegistrationTest(unittest.TestCase):
    def setUp(self):
        self.peer = load_peer()
        self.fake_module = types.ModuleType('native_overlap_control')

        class NativeOverlapControl:
            def __init__(self, engine=None):
                if engine is not None:
                    self.engine = engine

        class NativeControlClient:
            def __init__(self, engine):
                self.engine = engine
                self.calls = []

            async def qualification(self, worker_action, payload=None):
                self.calls.append((worker_action, copy.deepcopy(payload)))
                return {'schema': 1, 'engine_epoch': payload.get('engine_epoch'),
                        'workers': []}

        self.fake_module.NativeOverlapControl = NativeOverlapControl
        self.fake_module.NativeControlClient = NativeControlClient
        self._saved = sys.modules.get('native_overlap_control')
        sys.modules['native_overlap_control'] = self.fake_module
        self.fastapi = types.ModuleType('fastapi')
        self.fastapi.APIRouter = FakeAPIRouter
        self.fastapi.Request = object
        self.responses = types.ModuleType('fastapi.responses')
        self.responses.JSONResponse = FakeJSONResponse
        self._saved_fastapi = sys.modules.get('fastapi')
        self._saved_responses = sys.modules.get('fastapi.responses')
        sys.modules['fastapi'] = self.fastapi
        sys.modules['fastapi.responses'] = self.responses
        self.args = types.SimpleNamespace(api_key=['test-key'])
        self.processor = FakeProcessor()

        class Control(NativeControlClient):
            def __init__(self, processor):
                self.engine = types.SimpleNamespace(input_processor=processor)

        self.frontend = Control(self.processor)
        self.worker = FakeWorkerControl()

    def tearDown(self):
        for name, saved in [('native_overlap_control', self._saved),
                            ('fastapi', self._saved_fastapi),
                            ('fastapi.responses', self._saved_responses)]:
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved

    def test_attach_is_off_by_default_and_refuses_broad_dev_mode(self):
        app = FakeApp()
        self.assertIsNone(adapter.attach_router(
            self.args, app, frontend_control=self.frontend,
            worker_control=self.worker, qualification=self.peer,
            env={}))
        with self.assertRaises(adapter.NativeQualificationFrontendError):
            adapter.attach_router(
                self.args, app, frontend_control=self.frontend,
                worker_control=self.worker, qualification=self.peer,
                env={'FLASH_S5_OVERLAP': '1', 'VLLM_SERVER_DEV_MODE': '1'})

    def test_attach_keeps_api_and_core_controls_distinct(self):
        app = FakeApp()
        state = adapter.attach_router(
            self.args, app, frontend_control=self.frontend,
            worker_control=self.worker, qualification=self.peer,
            env={'FLASH_S5_OVERLAP': '1'})
        self.assertIs(state['frontend'], self.frontend)
        self.assertIs(state['worker'], self.worker)
        self.assertIsNot(state['frontend'], state['worker'])
        routes = app.routers[0].routes
        self.assertEqual([row[0] for row in routes],
                         [adapter.FRONTEND_PATH, adapter.WORKER_PATH])
        with self.assertRaises(adapter.NativeQualificationFrontendError) as c:
            adapter.attach_router(
                self.args, FakeApp(), frontend_control=self.frontend,
                worker_control=self.frontend, qualification=self.peer,
                env={'FLASH_S5_OVERLAP': '1'})
        self.assertEqual(str(c.exception),
                         'api-and-core-control-objects-must-differ')

    def test_auth_requires_existing_key_without_leaking_it(self):
        with self.assertRaises(adapter.NativeQualificationFrontendError):
            adapter.attach_router(
                types.SimpleNamespace(api_key=None), FakeApp(),
                frontend_control=self.frontend, worker_control=self.worker,
                qualification=self.peer,
                env={'FLASH_S5_OVERLAP': '1'})
        hashes = adapter._auth_hashes(self.args, {})
        self.assertEqual(hashes, [hashlib.sha256(b'test-key').digest()])
        self.assertNotIn('test-key', repr(hashes))

    def endpoint(self, app):
        return app.routers[0].routes[0][1]

    def test_lazy_default_factories_construct_distinct_controls(self):
        app = FakeApp()
        app.state.engine_client = types.SimpleNamespace(
            input_processor=self.processor)
        state = adapter.attach_router(
            self.args, app, qualification=self.peer,
            env={'FLASH_S5_OVERLAP': '1'})
        endpoint = self.endpoint(app)
        command = {'schema': 1, 'action': adapter.ACTION_FRONTEND,
                   'run_id': 'run-1', 'command_seq': 1, 'client_id': 'c',
                   'engine_epoch': 3, 'enabled': True, 'plan': make_plan()}
        ok = asyncio.run(endpoint(FakeHTTPRequest(
            body=json.dumps(command).encode(), app=app)))
        self.assertEqual(ok.status_code, 200)
        self.assertIsInstance(state['frontend'],
                              self.fake_module.NativeControlClient)
        self.assertIsInstance(state['worker'],
                              self.fake_module.NativeControlClient)
        self.assertIsNot(state['frontend'], state['worker'])
        self.assertIs(state['frontend'].engine, app.state.engine_client)

    def test_endpoint_enforces_loopback_bearer_and_body_bound(self):
        app = FakeApp()
        adapter.attach_router(
            self.args, app, frontend_control=self.frontend,
            worker_control=self.worker, qualification=self.peer,
            env={'FLASH_S5_OVERLAP': '1'})
        endpoint = self.endpoint(app)
        command = {'schema': 1, 'action': adapter.ACTION_FRONTEND,
                   'run_id': 'run-1', 'command_seq': 1, 'client_id': 'c',
                   'engine_epoch': 3, 'enabled': True,
                   'plan': make_plan()}
        body = json.dumps(command).encode()
        ok = asyncio.run(endpoint(FakeHTTPRequest(body=body, app=app)))
        self.assertEqual(ok.status_code, 200)
        self.assertTrue(ok.content['enabled'])

        remote = asyncio.run(endpoint(FakeHTTPRequest(
            body=body, app=app, host='<internal-host>')))
        self.assertEqual(remote.status_code, 403)
        wrong = asyncio.run(endpoint(FakeHTTPRequest(
            body=body, app=app, auth='Bearer other')))
        self.assertEqual(wrong.status_code, 403)
        oversized = asyncio.run(endpoint(FakeHTTPRequest(
            app=app, chunks=[b'x' * (adapter.MAX_BODY + 1)])))
        self.assertEqual(oversized.status_code, 409)
        self.assertEqual(oversized.content['error'], 'command-too-large')
        invalid = asyncio.run(endpoint(FakeHTTPRequest(
            body=b'not-json', app=app)))
        self.assertEqual(invalid.status_code, 400)

    def test_worker_endpoint_dispatches_through_the_distinct_control(self):
        app = FakeApp()
        adapter.attach_router(
            self.args, app, frontend_control=self.frontend,
            worker_control=self.worker, qualification=self.peer,
            env={'FLASH_S5_OVERLAP': '1'})
        route = app.routers[0].routes[1]
        self.assertEqual(route[0], adapter.WORKER_PATH)
        command = {'schema': 1, 'action': adapter.ACTION_WORKER,
                   'run_id': 'run-1', 'command_seq': 2, 'client_id': 'c',
                   'engine_epoch': 3, 'worker_action': 'teacher_evidence',
                   'payload': {'action': 'teacher_evidence'}}
        reply = asyncio.run(route[1](FakeHTTPRequest(
            body=json.dumps(command).encode(), path=adapter.WORKER_PATH,
            app=app)))
        self.assertEqual(reply.status_code, 200)
        self.assertEqual(self.worker.calls[0][0], 'teacher_evidence')
        self.assertEqual(self.worker.calls[0][1]['engine_epoch'], 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
