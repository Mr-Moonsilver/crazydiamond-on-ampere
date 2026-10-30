"""Native teacher receipt joins and action grammar; CPU only, no engine."""
import copy
from pathlib import Path
import sys
import types
import unittest
import asyncio
import tempfile
import json
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'harness'), str(ROOT/'patches/s5_engine')]
import e3_teacher_decode_client as client
import teacher_decode
import test_e3_teacher_decode as fixtures


def native_runs(instrument):
    fixture = fixtures.ClientTests('test_api_original_logprobs_are_bound_to_fixed_token_positions')
    runs = fixture.fixture_runs(instrument)
    for repeat, run in enumerate(runs):
        old = run.pop('calls')[0]['receipt_core']
        proof = old['teacher_forced']
        previous = {key: value for key, value in proof.items() if key != 'ranks'}
        previous.update(enabled=True, steps=client.TOTAL)
        rows = []
        for row in proof['ranks']:
            row.update(native_route=True, mode='native-overlap',
                       ack='flash_s5_teacher_evidence')
            rows.append({'worker_rank': row['rank'], 'qualification': row})
        run['native_teacher_window'] = {
            'schema': 's5-native-teacher-window/1', 'repeat': repeat,
            'worker_reply': {'engine_epoch': old['engine_epoch'], 'workers': rows},
            'frontend_clear': {'engine_epoch': old['engine_epoch'],
                               'enabled': False, 'previous_status': previous}}
    return runs


class NativeTeacherBindings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = fixtures.ClientTests(
            'test_api_original_logprobs_are_bound_to_fixed_token_positions')
        cls.instrument = cls.fixture.instrument()
        cls.runs = native_runs(cls.instrument)

    def setUp(self):
        self.run = copy.deepcopy(self.runs[0])

    def bind(self):
        return client._native_teacher_binding(self.run, self.instrument.plan_shas)

    def test_native_receipts_keep_their_own_names_and_all_rank_identity(self):
        proof, core = self.bind()
        self.assertEqual({row['rank'] for row in proof['ranks']}, set(range(4)))
        self.assertEqual(core['engine_epoch'], 71)
        self.assertEqual(proof['admitted'], 8)
        self.assertNotIn('calls', self.run)
        self.assertNotIn('teacher_forced', self.run['native_teacher_window'])

    def test_full_admission_status_must_be_observed_before_clear(self):
        for field, bad in [('pending',1), ('admitted',7), ('steps',767),
                           ('enabled',False), ('plan_sha256','f'*64)]:
            with self.subTest(field=field):
                self.run = copy.deepcopy(self.runs[0])
                self.run['native_teacher_window']['frontend_clear']['previous_status'][field] = bad
                with self.assertRaises(ValueError): self.bind()

    def test_native_cannot_be_a_probe_or_missing_drain(self):
        for field, bad in [('mode','decode-only-probe'), ('native_route',False),
                           ('drained_streams',0), ('engine_epoch',72), ('rank',1)]:
            with self.subTest(field=field):
                self.run = copy.deepcopy(self.runs[0])
                self.run['native_teacher_window']['worker_reply']['workers'][0]['qualification'][field] = bad
                with self.assertRaises(ValueError): self.bind()

    def test_no_missing_worker_or_mixed_window_generation(self):
        self.run['native_teacher_window']['worker_reply']['workers'].pop()
        with self.assertRaises(ValueError): self.bind()
        self.run = copy.deepcopy(self.runs[0])
        self.run['native_teacher_window']['worker_reply']['workers'][0]['qualification']['route_generation'] += 1
        with self.assertRaises(ValueError): self.bind()

    def test_same_sampler_rows_and_sse_give_identical_numerical_inputs(self):
        i = self.fixture.instrument()
        legacy = i.finish(self.fixture.fixture_runs(i))
        native = i.finish(native_runs(i))
        for name, run in native['evidence']['runs'].items():
            retained = run.pop('native_teacher_window')
            self.assertEqual(retained['schema'], 's5-native-teacher-window/1')
            self.assertEqual(run, legacy['evidence']['runs'][name])
        self.assertEqual(native['grade'], legacy['grade'])

    def test_bad_native_binding_is_inconclusive_before_numerics(self):
        i = self.fixture.instrument()
        runs = native_runs(i)
        runs[0]['native_teacher_window']['frontend_clear'].pop('previous_status')
        out = i.finish(runs)
        self.assertEqual(out['grade']['verdict'], 'INCONCLUSIVE')
        self.assertTrue(out['evidence']['runs']['full-A1']['row_mapping']['defects'])

    def test_first_native_sampler_configuration_does_not_require_a_probe_attribute(self):
        runner = fixtures.worker_runner()
        del runner._flash_s5_decode_probe
        identity = types.SimpleNamespace(route='full', route_generation=3)
        teacher_decode.configure_worker(runner, True, rank=2,
            factory=lambda _states: types.SimpleNamespace(), identity=identity)
        self.assertIs(runner._flash_s5_teacher_trace.probe, identity)
        self.assertFalse(hasattr(runner, '_flash_s5_decode_probe'))
        teacher_decode.configure_worker(runner, False, identity=identity)
        self.assertIsNone(runner.sampler.trace_replay_state)


class NativeSequenceGrammar(unittest.TestCase):
    """Execute the real runner; only HTTP/engine/GPU leaves are stand-ins."""
    def run_sequence(self, *, bad_entry=False):
        import e3_decode_correctness as correctness
        import e3_native_qualification_driver as orchestration
        from e3_native_teacher_sequence import NativeTeacherSequence
        from wire_v1 import ROUTE_BANK_FULL, ROUTE_BANK_SPLIT
        import json
        manifest = json.loads((ROOT/'harness/workloads/q17b-teacher-decode-v3.json').read_text())
        workload = correctness.validate_manifest(manifest)[1]
        outer = self
        calls = []

        class Instrument:
            snapshot_max_bytes = 2*1024**3
            plan_shas = {}
            def cohort(self, url, model):
                return [dict(stream=n,prompt=[n,1,2],prompt_sha256=client.digest([n,1,2]),
                             prompt_chars=0) for n in range(8)]
            def requests(self, bodies, cohort, repeat):
                self.plan = {'schema':'test-native-plan', 'repeat':repeat}
                self.plan_shas[repeat] = client.digest(self.plan)
            def finish(self, runs):
                outer.assertEqual(len(runs), 4)
                outer.assertTrue(all(run['complete'] for run in runs))
                return {'evidence': 'CPU-transport-only', 'runs': runs}

        class Driver:
            url, model, run_id, engine_epoch = 'http://fixture', 'fixture', 'run-1', 3
            route, generation, plan = 'full', 2, None
            def route_banks(self): return ROUTE_BANK_FULL, ROUTE_BANK_SPLIT
            def _session(self, phase, action, **fields):
                calls.append(('session',action))
                if action == 'resume': return {'resumed':True}
                target = 'split' if action == 'pause_enter_split' else 'full'
                if self.route != target:self.generation += 1
                self.route = target
                bank = ROUTE_BANK_SPLIT if target == 'split' else ROUTE_BANK_FULL
                return dict(route=target, engine_epoch=3, typed_acks=True,
                    admission={'phase':'full-prefill-solo','armed':False},
                    worker_drain=[{'worker_rank':n,'drained_streams':2} for n in range(4)],
                    transition_acks=[dict(worker_rank=n,typed='TransitionAck',engine_epoch=3,
                        applied_bank=bank,drained_streams=2,applied_route_generation=self.generation)
                        for n in range(4)])
            def _worker(self, phase, action, payload):
                calls.append(('worker',action))
                rows=[]
                for n in range(4):
                    row=dict(rank=n,engine_epoch=3,route=self.route,
                        route_generation=self.generation,storage_epoch=1,
                        worker_incarnation='cpu-rank-%d'%n,drained_streams=2,
                        enabled=payload.get('enabled'),native_route=True,mode='native-overlap',
                        ack='flash_s5_teacher_evidence')
                    if bad_entry and action == 'teacher_evidence' and n == 0:
                        row['route'] = 'foreign'
                    rows.append(dict(worker_rank=n,qualification=row))
                return dict(engine_epoch=3,workers=rows)
            def _frontend(self, phase, enabled, plan):
                calls.append(('frontend',enabled))
                if enabled:
                    self.plan = plan
                    return {'enabled':True,'engine_epoch':3}
                return dict(enabled=False,engine_epoch=3,previous_status=dict(
                    enabled=True,plan_sha256=client.digest(self.plan),admitted=8,pending=0,
                    steps=client.TOTAL))

        streams = [types.SimpleNamespace(index=n,count=lambda:1) for n in range(8)]
        def trajectories(states, window, workload):
            return [dict(stream=n,complete=True,_tokens=[n]*client.TOTAL,teacher_top=[])
                    for n in range(8)]
        with tempfile.TemporaryDirectory() as root:
            driver = Driver(); driver.store = orchestration.ImmutableStore(root)
            with patch.object(correctness,'_start_streams',return_value=streams) as start, \
                 patch.object(correctness,'_wait_open',return_value={'complete':True}), \
                 patch.object(correctness,'_wait_window',return_value={'complete':True}), \
                 patch.object(correctness,'_wait_finish',return_value=[]), \
                 patch.object(correctness,'_stop_streams',return_value=[]), \
                 patch.object(correctness,'_trajectory_rows',side_effect=trajectories):
                sequence=NativeTeacherSequence(driver,Instrument(),workload)
                if bad_entry:
                    with self.assertRaises(orchestration.QualificationRefused) as refused:
                        sequence.run()
                    self.assertEqual(start.call_count, 1)
                    self.assertEqual(refused.exception.cleanup['restored'], True)
                    self.assertEqual(driver.store.names(), ['teacher-native-full-A1.json'])
                else:
                    result=sequence.run()
                    self.assertEqual(start.call_count,4)
                    self.assertEqual(len(driver.store.names()),4)
                    self.assertEqual([r['route'] for r in result['native_runs']],
                                     ['full','full','split','full'])
        return calls

    def test_one_fixed_native_sequence_never_calls_probe_controller_actions(self):
        calls=self.run_sequence()
        self.assertNotIn(('session','pause_end'),calls)
        self.assertNotIn(('session','pause_full'),calls)
        self.assertEqual(calls.count(('worker','teacher_evidence')),8)
        self.assertEqual(calls.count(('session','pause_enter_split')),1)
        self.assertEqual(calls.count(('frontend',True)),4)
        self.assertEqual(calls.count(('frontend',False)),4)

    def test_first_refusal_retains_native_proof_and_never_selects_another_run(self):
        self.run_sequence(bad_entry=True)


class ColdFullPreparation(unittest.IsolatedAsyncioTestCase):
    async def test_real_driver_and_four_bridges_produce_initial_full_transition_ack(self):
        import test_e3_native_overlap as rig
        from native_overlap_session import NativeOverlapSession
        stack = rig.cold_stack()
        core, frontend = rig.core_for(stack)
        pump = rig.Pump(core).start()
        try:
            session = NativeOverlapSession(frontend, timeout_s=5)
            result = await session.execute(dict(schema=1,action='prepare',
                run_id='cold-full-native',target_route='full'))
            self.assertEqual(result['route'], 'full')
            self.assertEqual(len(result['transition_acks']), 4)
            self.assertTrue(all(r['typed']=='TransitionAck' and r['drained_streams']>0
                                for r in result['transition_acks']))
            proof=result['initial_full_fence']
            self.assertEqual(proof['initial']['transition_acks'], [])
            self.assertEqual(proof['entered']['route_after'], 'split')
            self.assertEqual(proof['returned']['route_after'], 'full')
            self.assertIsNone(stack.driver.admission_intent())
            # Subsequent FULL preparation uses the existing typed identity;
            # the initial empty route warmup is not repeated.
            again = await session.execute(dict(schema=1,action='prepare',
                run_id='cold-full-native',target_route='full'))
            self.assertIsNone(again['initial_full_fence'])
            self.assertEqual(again['transition_acks'], result['transition_acks'])
        finally:
            await pump.stop()
        if pump.failure is not None: raise pump.failure

    async def test_initial_full_warmup_refuses_live_workers_before_route_change(self):
        from native_overlap_session import NativeOverlapSession, NativeOverlapSessionError
        class Control:
            changed=False
            async def bootstrap_engine_owner(self):return {'snapshot':{'ranks':[0]}}
            async def barrier(self):
                return {'workers':[dict(worker_rank=0,engine_epoch=3,route='full',
                    route_generation=0,barrier_clean=True,live_decode={'available':True},
                    live_decode_requests=1,running_requests=1,prefill_requests=0,
                    retained_rows=0,pending_native_replays=0,pending_prefill_streams=0,
                    pending_output_copies=0)]}
            async def enter_split(self):self.changed=True;raise AssertionError('live warmup')
        control=Control();session=NativeOverlapSession(object(),control=control)
        with self.assertRaisesRegex(NativeOverlapSessionError,'needs-empty-drained-workers'):
            await session._initial_full_fence(dict(route_after='full',route_generation_after=0,
                engine_epoch=3,transition_acks=[],bootstrap_acks=[]))
        self.assertFalse(control.changed)


class ActualWorkerReplyBound(unittest.TestCase):
    def test_collector_error_message_and_frames_are_bounded_and_redacted(self):
        from e3_native_qualification_driver import exception_details
        secret='cpu-only-error-redaction-fixture'
        try:
            raise TypeError('invalid field '+secret+' '+('x'*1000))
        except TypeError as exc:
            detail=exception_details(exc,types.SimpleNamespace(key=secret))
        self.assertEqual(detail['error_type'],'TypeError')
        self.assertIn('***',detail['error_message'])
        self.assertNotIn(secret,detail['error_message'])
        self.assertLessEqual(len(detail['error_message']),512)
        self.assertTrue(detail['error_stack'])
        self.assertLessEqual(len(detail['error_stack']),16)

    def test_generation_zero_is_not_relabelled_as_a_typed_transition_ack(self):
        from e3_native_qualification_driver import NativePlacementRunner, QualificationRefused
        from wire_v1 import ROUTE_BANK_FULL
        rows=[dict(worker_rank=n,typed='TransitionAck',engine_epoch=3,
            applied_bank=ROUTE_BANK_FULL,applied_route_generation=0,drained_streams=2)
            for n in range(4)]
        with self.assertRaises(QualificationRefused):
            NativePlacementRunner._require_acks(dict(typed_acks=True,transition_acks=rows),
                bank=ROUTE_BANK_FULL,label='cold-full',epoch=3)

    def test_worker_uses_bounded_rich_reply_while_frontend_stays_small(self):
        import http.client
        from e3_native_qualification_driver import LoopbackControl, QualificationRefused
        class Connection:
            def __init__(self, *args, **kwargs):pass
            def request(self, method, path, body, headers):self.command=json.loads(body)
            def getresponse(self):
                command=self.command
                raw=json.dumps(dict(schema=1,action=command['action'],
                    run_id=command['run_id'],client_id=command['client_id'],
                    command_seq=command['command_seq'],payload='x'*70000)).encode()
                return types.SimpleNamespace(status=200,read=lambda size:raw[:size])
            def close(self):pass
        command=dict(schema=1,action='qualification_worker',run_id='run-1',
                     client_id='fixture',command_seq=1)
        with patch.object(http.client,'HTTPConnection',Connection):
            control=LoopbackControl('http://127.0.0.1:9',key='cpu-fixture-key')
            self.assertEqual(len(control.worker(command)['payload']),70000)
            command['action']='qualification_frontend'
            with self.assertRaisesRegex(QualificationRefused,'response-too-large'):
                control.frontend(command)


if __name__ == '__main__': unittest.main()
