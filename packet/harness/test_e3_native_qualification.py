"""CPU-only native qualification worker actions and teacher plan ownership.

The chain below is built from the shipped modules: the real ``WorkerNative``
composition, the real ``NativeWorkerBridge`` (over a typed dispatcher seam), the
real ``RouteController`` drain contract, the reviewed ``teacher_decode``
configure/evidence functions, the real ``WorkerObserver`` type gate and the real
``NativeOverlapControl``.  Only the GPU/engine leaves are stand-ins; no CUDA,
torch, vLLM or network is touched.

The negative cases are the ones a metadata-only shim would let through: a
flags-only runner, a probe router relabelled as native, a retained request row,
a pending replay, an unclean router, an int incarnation, a missing descriptor,
a probe/bare frontend owner.
"""
import sys
from pathlib import Path
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
sys.path.insert(0, str(ROOT / 'harness'))

import native_qualification as qualification                 # noqa: E402
import observer_runtime                                       # noqa: E402
import replay_observer                                        # noqa: E402
import teacher_decode                                         # noqa: E402
from native_overlap_control import NativeControlClient        # noqa: E402
from native_startup import WorkerNative                       # noqa: E402
from worker_bridge import NativeWorkerBridge                  # noqa: E402
from split_bank import SplitDecodeBank, SplitGraphHandle       # noqa: E402
EPOCH = 3
STORAGE = 1

RUN = 'q18a-run'


class Dispatcher:
    """The narrow typed dispatcher the bridge requires, nothing more."""

    def __init__(self, rank):
        self.rank = rank

    def submit(self, command):
        return None

    def open(self, request):
        return None


class Status:
    pending_replays = 0
    unclean = False
    external_users = 0


class Router:
    """A native graph router stand-in: no probe identity and no invented count."""

    def __init__(self):
        self.native_pending = 0
        self.descriptors = []
        self.polled = 0

    def poll_native_replays(self):
        self.polled += 1

    def native_census(self):
        return {'pending': 0, 'polled': self.polled}

    def status(self):
        return Status()

    def native_decode_route(self, descriptor):
        self.descriptors.append(descriptor)
        return descriptor


class Controller:
    def __init__(self):
        self.route = 'full'
        self.route_generation = 1
        self.admitting = True
        self.drains = 0

    def drain(self):
        self.drains += 1
        return 3


class Streams:
    def __init__(self):
        self.drains = 0

    def drain(self):
        self.drains += 1
        return 1


class TraceState:
    """The fields the reviewed collector resets and reads."""

    def __init__(self):
        self.traces = {}
        self.calls = []
        self.positions = []
        self.sampler_calls = 0
        self.call_overflow = False
        self.proof_overflow = False
        self.list_overflow = False
        self.capture_overflow = False
        self.capture_defects = set()
        self.last_graph = None
        self.dispatch_seq = 0
        self.consumed_dispatch_seq = -1
        self.rank = None
        self.probe = None


class Runner:
    def __init__(self, rank, *, last=False):
        self.rank = rank
        self.is_last_pp_rank = last
        self.req_states = types.SimpleNamespace(req_id_to_index={},
                                                max_num_reqs=8)
        self.sampler = None
        self.sample = lambda *args, **kwargs: None
        self.execute_model_state = None
        self._flash_e3_graph_router = Router()

    def make_native(self, *, route='full', state=None):
        self.native = WorkerNative(
            self,
            types.SimpleNamespace(prefill_stream=object(),
                                  describe=lambda: {'lane': 'decode'}),
            types.SimpleNamespace(measured_bytes=4096),
            backend=object(), controller=self.controller,
            comms=types.SimpleNamespace(pp=None),
            ple_layer=0, ple_scratch=None, chunk_tokens=1024,
            tp_group=None, torch=object(), state=state,
            decode_owner=types.SimpleNamespace(storage_epoch=STORAGE),
            prefill_streams=Streams())
        self._flash_s5_native = self.native
        self._flash_s5_bridge = NativeWorkerBridge(Dispatcher(self.rank),
                                                   rank=self.rank)
        return self.native

    @property
    def controller(self):
        return self._controller


def make_runner(rank=0, *, last=False):
    runner = Runner(rank, last=last)
    runner._controller = Controller()
    state = types.SimpleNamespace(
        worker_rank=rank, incarnation='worker-%d' % rank,
        engine_bound=True, engine_epoch=EPOCH,
        decode=types.SimpleNamespace(
            handle=types.SimpleNamespace(storage_epoch=STORAGE)))
    runner.make_native(state=state)
    return runner


def configure_command(**overrides):
    command = {'action': 'teacher_configure', 'engine_epoch': EPOCH, 'route_generation': 1,
               'enabled': True, 'snapshot_max_bytes': 64 * 1024 ** 2}
    command.update(overrides)
    return command


def evidence_command(**overrides):
    command = {'action': 'teacher_evidence', 'engine_epoch': EPOCH, 'route_generation': 1}
    command.update(overrides)
    return command


def observer_command(action, **overrides):
    command = {'action': action, 'engine_epoch': EPOCH, 'route_generation': 2, 'run_id': RUN,
               'window_id': 'win-0'}
    command.update(overrides)
    return command


class WorkerChainTests(unittest.TestCase):
    def setUp(self):
        self.runner = make_runner()
        self.identity = None

    def tearDown(self):
        qualification.release(self.runner)

    def configure(self, **overrides):
        result = qualification.handle_worker(
            self.runner, configure_command(**overrides))
        self.identity = getattr(self.runner, qualification.STATE_ATTRIBUTE)
        return result

    def test_teacher_configure_binds_the_constructed_native_lane(self):
        result = self.configure()
        self.assertEqual(result['ack'], 'flash_s5_teacher_configure')
        self.assertEqual(result['mode'], 'native-overlap')
        self.assertEqual(result['route'], 'full')
        self.assertEqual(result['rank'], 0)
        self.assertEqual(result['worker_incarnation'], 'worker-0')
        self.assertEqual(result['storage_epoch'], STORAGE)
        self.assertEqual(result['consumer_streams'], 3)
        self.assertEqual(result['prefill_streams'], 1)
        self.assertEqual(result['drained_streams'], 4)
        self.assertFalse(result['sampler_owner'])
        self.assertTrue(result['native_route'])
        self.assertEqual(self.runner.controller.drains, 1)
        self.assertEqual(self.runner.native.prefill_streams.drains, 1)

    def test_observer_census_uses_real_split_bank_replays_and_refuses_missing_counts(self):
        # Native24 reached ObserverSession.bind_bank on the box, then its
        # native router view lacked the counters read by replay_census().
        # Exercise the actual SplitDecodeBank.replay counter and the unchanged
        # observer reader, including a zero-replay descriptor.
        bank = object.__new__(SplitDecodeBank)
        first, second = object(), object()
        local_first, local_second = object(), object()
        bank.pairs = [(first, local_first), (second, local_second)]
        bank.handles = {
            local_first: SplitGraphHandle(first, local_first, bank),
            local_second: SplitGraphHandle(second, local_second, bank),
        }
        bank.manager = types.SimpleNamespace(run_fullgraph=lambda descriptor: descriptor)
        self.runner.native.bank = bank
        identity = qualification.NativeIdentity(self.runner, self.runner.native)
        session = object.__new__(replay_observer.ObserverSession)
        session._router = identity.router
        session._local_handles = (local_first, local_second)
        self.assertEqual(session.replay_census(),
                         {'total': 0, 'graphs': {'graph:0': 0, 'graph:1': 0}})
        self.assertIs(bank.replay(bank.handles[local_first]), first)
        self.assertEqual(session.replay_census(),
                         {'total': 1, 'graphs': {'graph:0': 1, 'graph:1': 0}})
        bank.handles[local_first].replays = -1
        self.assertIsNone(session.replay_census())
        bank.handles[local_first].replays = 1
        bank.handles.pop(local_second)
        self.assertIsNone(session.replay_census())

    def test_descriptor_shadow_retains_only_the_full_route(self):
        self.configure()
        router = self.runner._flash_e3_graph_router
        partial = types.SimpleNamespace(cg_mode='partial')
        full = types.SimpleNamespace(cg_mode='full')
        router.native_decode_route(partial)
        router.native_decode_route(full)
        self.assertIs(self.identity.require_descriptor(), full)
        self.assertEqual(router.descriptors, [partial, full])

    def test_release_restores_the_router_method(self):
        router = self.runner._flash_e3_graph_router
        self.assertNotIn('native_decode_route', router.__dict__)
        self.configure()
        self.assertIn('native_decode_route', router.__dict__)
        qualification.release(self.runner)
        self.assertNotIn('native_decode_route', router.__dict__)
        self.assertFalse(hasattr(self.runner, qualification.STATE_ATTRIBUTE))

    def test_disabled_configuration_drops_the_shadow(self):
        router = self.runner._flash_e3_graph_router
        self.configure()
        self.assertIn('native_decode_route', router.__dict__)
        result = qualification.handle_worker(
            self.runner, configure_command(enabled=False))
        self.assertFalse(result['enabled'])
        self.assertNotIn('native_decode_route', router.__dict__)

    def test_teacher_evidence_uses_the_reviewed_collector(self):
        self.configure()
        result = qualification.handle_worker(self.runner, evidence_command())
        self.assertEqual(result['ack'], 'flash_s5_teacher_evidence')
        self.assertFalse(result['sampler_owner'])
        self.assertEqual(result['positions'], [])
        self.assertEqual(result['row_schema'], None)
        self.assertFalse(result['overflow'])
        self.assertEqual(result['native_route'], True)

    def test_last_pp_rank_owns_the_sampler_trace_and_its_probe_seam(self):
        runner = make_runner(0, last=True)
        state = TraceState()
        runner._flash_s5_teacher_trace = state
        runner.sampler = types.SimpleNamespace(
            logprobs_mode='raw_logprobs', trace_replay_state=state)
        try:
            result = qualification.handle_worker(runner,
                                                 configure_command())
            identity = getattr(runner, qualification.STATE_ATTRIBUTE)
            self.assertTrue(result['sampler_owner'])
            self.assertIs(state.probe, identity)
            self.assertIs(runner.sampler.trace_replay_state, state)
            self.assertEqual(state.rank, 0)
            evidence = qualification.handle_worker(runner, evidence_command())
            self.assertTrue(evidence['sampler_owner'])
            self.assertEqual(evidence['row_schema'], teacher_decode.ROW_SCHEMA)
        finally:
            qualification.release(runner)

    def test_a_flags_only_runner_is_never_native(self):
        runner = types.SimpleNamespace(
            rank=0, _flash_s5_native={'installed': True, 'native_route': True},
            _flash_s5_bridge=NativeWorkerBridge(Dispatcher(0), rank=0),
            _flash_e3_graph_router=Router(), req_states=types.SimpleNamespace(
                req_id_to_index={}))
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'worker-native-unavailable'):
            qualification.handle_worker(runner, configure_command())

    def test_missing_construction_refuses(self):
        runner = make_runner()
        del runner._flash_s5_native
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'worker-native-unavailable'):
            qualification.handle_worker(runner, configure_command())

    def test_unknown_or_inexact_commands_refuse(self):
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'unknown-action'):
            qualification.handle_worker(self.runner, {'action': 'status'})
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'command-not-object'):
            qualification.handle_worker(self.runner, ['teacher_configure'])
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'fields-differ'):
            qualification.handle_worker(
                self.runner, configure_command(extra=True))
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'fields-differ'):
            qualification.handle_worker(
                self.runner,
                {'action': 'teacher_evidence', 'engine_epoch': EPOCH, 'route_generation': 1,
                 'enabled': True})

    def test_stale_or_absent_epoch_refuses(self):
        self.runner.native.state.engine_epoch = EPOCH - 1
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'stale-engine-epoch'):
            qualification.handle_worker(self.runner, configure_command())
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'invalid-engine-epoch'):
            qualification.handle_worker(self.runner,
                                        configure_command(engine_epoch=0))

    def test_snapshot_must_stay_inside_the_measured_range(self):
        for value in (1, 3 * 1024 ** 3):
            with self.assertRaisesRegex(
                    qualification.NativeQualificationUnavailable,
                    'teacher-configuration-invalid'):
                qualification.handle_worker(
                    self.runner, configure_command(snapshot_max_bytes=value))

    def test_teacher_boundary_needs_full_split_empty_and_drained(self):
        self.runner.controller.route = 'split'
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'needs-full-route'):
            qualification.handle_worker(self.runner, configure_command())
        self.runner.controller.route = 'full'
        self.runner.req_states.req_id_to_index[0] = 0
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'needs-empty-worker'):
            qualification.handle_worker(self.runner, configure_command())
        self.runner.req_states.req_id_to_index.clear()
        self.runner.controller.admitting = False
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'controller-not-admitting'):
            qualification.handle_worker(self.runner, configure_command())
        self.runner.controller.admitting = True
        self.runner.controller.drain = lambda: 0
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'drain-unproved'):
            qualification.handle_worker(self.runner, configure_command())

    def test_retained_request_row_and_bridge_refuse_the_boundary(self):
        self.runner._flash_s5_bridge.transaction = object()
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'request-row-retained'):
            qualification.handle_worker(self.runner, configure_command())
        self.runner._flash_s5_bridge.transaction = None
        self.runner._flash_s5_bridge._retained = {'transaction': object()}
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'request-row-retained'):
            qualification.handle_worker(self.runner, configure_command())

    def test_pending_replays_and_unclean_router_refuse_the_boundary(self):
        self.runner._flash_e3_graph_router.native_pending = 1
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'replays-pending'):
            qualification.handle_worker(self.runner, configure_command())
        self.runner._flash_e3_graph_router.native_pending = 0
        self.runner._flash_e3_graph_router.status = lambda: types.SimpleNamespace(
            pending_replays=0, unclean=True, external_users=0)
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'router-unclean'):
            qualification.handle_worker(self.runner, configure_command())

    def test_external_source_hold_refuses_the_quiet_boundary(self):
        self.runner._flash_e3_graph_router.status = lambda: types.SimpleNamespace(
            pending_replays=0, unclean=False, external_users=1)
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'source-holds-active'):
            qualification.handle_worker(self.runner, configure_command())

    def test_a_probe_router_is_never_relabelled_native(self):
        class DecodeProbeRouter:
            pass

        self.runner._flash_e3_graph_router = DecodeProbeRouter()
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'refuses-decode-probe-router'):
            qualification.handle_worker(self.runner, configure_command())
        self.runner._flash_e3_graph_router = Router()
        self.runner._flash_s5_decode_probe = object()
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'refuses-decode-probe'):
            qualification.handle_worker(self.runner, configure_command())

    def test_integer_incarnation_is_unavailable_not_coerced(self):
        self.runner.native.state.incarnation = 7
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'incarnation-unavailable'):
            qualification.handle_worker(self.runner, configure_command())
        identity = getattr(self.runner, qualification.STATE_ATTRIBUTE)
        self.assertIsNotNone(identity.quarantined)

    def test_missing_retained_storage_refuses(self):
        self.runner.native.decode_owner = None
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'retained-storage-missing'):
            qualification.handle_worker(self.runner, configure_command())
        self.runner.native.decode_owner = types.SimpleNamespace(
            storage_epoch=STORAGE)
        self.runner.native.state.decode.handle.storage_epoch = STORAGE + 1
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'storage-epoch-differs'):
            qualification.handle_worker(self.runner, configure_command())

    def test_prefill_streams_are_required(self):
        self.runner.native.prefill_streams = None
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'prefill-streams-unavailable'):
            qualification.handle_worker(self.runner, configure_command())


class ObserverGateTests(unittest.TestCase):
    def setUp(self):
        self.runner = make_runner()
        qualification.handle_worker(self.runner, configure_command())
        self.identity = getattr(self.runner, qualification.STATE_ATTRIBUTE)
        self.runner.controller.route = 'split'
        self.runner.controller.route_generation = 2
        self.identity.watch_decode_descriptor()

    def tearDown(self):
        qualification.release(self.runner)

    def owner(self, *, offers=True):
        return observer_runtime.WorkerObserver(
            types.SimpleNamespace(state='started'), 0, engine_offers=offers)

    def test_observer_requires_the_real_worker_observer(self):
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'replay-observer-unavailable'):
            qualification.handle_worker(self.runner,
                                        observer_command('observer_begin'))
        self.runner._flash_s5_replay_observer = types.SimpleNamespace(
            begin=lambda *a, **k: None, end=lambda *a, **k: None,
            window_identity=None)
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'replay-observer-unavailable'):
            qualification.handle_worker(self.runner,
                                        observer_command('observer_begin'))

    def test_observer_requires_engine_offer_accounting(self):
        self.runner._flash_s5_replay_observer = self.owner(offers=False)
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'observer-offers-disabled'):
            qualification.handle_worker(self.runner,
                                        observer_command('observer_begin'))

    def test_observer_requires_identity_and_split_route(self):
        owner = self.owner()
        self.runner._flash_s5_replay_observer = owner
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'window-identity-invalid'):
            qualification.handle_worker(
                self.runner, observer_command('observer_begin', run_id=''))
        self.runner.controller.route = 'full'
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'needs-split-route'):
            qualification.handle_worker(self.runner,
                                        observer_command('observer_begin'))

    def test_observer_begin_needs_the_retained_full_descriptor(self):
        self.runner._flash_s5_replay_observer = self.owner()
        self.identity._last_decode_descriptor = None
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'descriptor-unproved'):
            qualification.handle_worker(self.runner,
                                        observer_command('observer_begin'))

    def test_observer_end_refuses_a_changed_window_identity(self):
        owner = self.owner()
        owner.window_identity = (RUN, 'other-window')
        self.runner._flash_s5_replay_observer = owner
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'window-identity-changed'):
            qualification.handle_worker(self.runner,
                                        observer_command('observer_end'))

    def test_observer_actions_refuse_a_retained_row_or_pending_replay(self):
        self.runner._flash_s5_replay_observer = self.owner()
        self.runner._flash_s5_bridge.transaction = object()
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'request-row-retained'):
            qualification.handle_worker(self.runner,
                                        observer_command('observer_begin'))
        self.runner._flash_s5_bridge.transaction = None
        self.runner._flash_e3_graph_router.native_pending = 1
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'replays-pending'):
            qualification.handle_worker(self.runner,
                                        observer_command('observer_end'))


class FrontendOwnershipTests(unittest.TestCase):
    def setUp(self):
        async def scheduling_forbidden(*args, **kwargs):
            raise AssertionError('plan ownership must never pause, resume or query scheduling')
        self.engine = types.SimpleNamespace(engine_core=types.SimpleNamespace(
            client_count=1, client_index=0, call_utility_async=lambda *args: None),
            is_paused=scheduling_forbidden, pause_generation=scheduling_forbidden,
            resume_generation=scheduling_forbidden)
        self.processor = types.SimpleNamespace(
            process_inputs=lambda request: request)
        self.engine.input_processor = self.processor
        self.control = NativeControlClient(self.engine, timeout_s=5.0)

    def test_native_control_owns_one_reused_plan(self):
        plan = qualification.own_frontend_plan(self.control)
        self.assertIsInstance(plan, teacher_decode.FrontendPlan)
        self.assertIs(self.processor._flash_s5_teacher, plan)
        self.assertIs(qualification.own_frontend_plan(self.control), plan)
        status = qualification.configure_frontend(self.control, None)
        self.assertFalse(status['enabled'])
        self.assertEqual(qualification.frontend_status(self.control),
                         status)

    def test_probe_or_bare_owner_is_refused(self):
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'requires-native-api-control'):
            qualification.own_frontend_plan(types.SimpleNamespace(
                engine=self.engine))
        self.engine.input_processor = types.SimpleNamespace()
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'input-seam-unavailable'):
            qualification.own_frontend_plan(self.control)

    def test_another_owner_is_not_overwritten(self):
        self.processor._flash_s5_teacher = object()
        with self.assertRaisesRegex(
                qualification.NativeQualificationUnavailable,
                'teacher-plan-owner-differs'):
            qualification.own_frontend_plan(self.control)

    def test_frontend_configuration_refuses_the_probe_seam(self):
        with self.assertRaises(
                qualification.NativeQualificationUnavailable):
            qualification.configure_frontend(object(), None)


if __name__ == '__main__':
    unittest.main()
