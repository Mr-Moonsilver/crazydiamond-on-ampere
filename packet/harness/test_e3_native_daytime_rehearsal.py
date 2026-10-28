"""CPU composition rehearsal, never engine/hardware qualification.

Real Row1Qualifier/store/WorkerHandoffServer scope, NativeOverlapSession,
NativeRow1Producer, loopback HTTP client, immutable artifact store and raw
row1 grader. Byte tensors, source ownership, EngineCore utility replies and SSE
are explicit test fixtures. Routes alternate deliberately; natural model route
coverage is NOT established. No production module or readiness flag is patched.
"""
from dataclasses import replace
import http.server
import json
import math
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
import e3_native_qualification_driver as qd
import e3_native_row1_grade as grade
import e3_probe_rpc as rpc
import row1_oracle as ro
from test_e3_native_integration import make_worker_stack
import test_e3_native_source_oracle as source
import test_e3_native_qualification_driver as composition

RUN = 'daytime-synthetic-row1'
TEMPLATE = json.loads((ROOT / 'harness/workloads/q18a-native-row1-template.json').read_text())
LAYER_TYPES = json.loads((ROOT / 'reference/model-config.json').read_text())['text_config']['layer_types']
# Prompt's final token returns to stock; no other requests compete in this fixture.
CHUNKS = math.ceil((TEMPLATE['workload']['prompt']['tokens'] - 1) / 1024)


class SourceControl(composition.EngineSessionControl):
    """Actual worker scope/record production inside the existing CPU transport rig."""
    def __init__(self, engine, *, terminal_full=False, omit_last=False):
        super().__init__(engine, [], terminal_full=terminal_full)
        self.stacks = {rank: make_worker_stack(rank) for rank in range(4)}
        self.rigs = {}
        self.produced = False
        self.omit_last = omit_last
        self.packet_bytes = 0
        for rank, stack in self.stacks.items():
            rig = source.Row1QualifierTests('test_a_matching_bucket_records_the_actual_bytes')
            rig.setUp()
            rig.lane = replace(rig.lane, worker_rank=rank)
            self.rigs[rank] = rig
            begin = (rank // 2) * 24
            layers = tuple('qsa.%d' % n for n in range(begin, begin+24)
                           if LAYER_TYPES[n] == 'full_attention')
            def provider(store, epoch, generation, *, rig=rig, stack=stack, layers=layers):
                def factory(_operation):
                    return ro.Row1Qualifier(store, transaction=rig.transaction,
                        lane=rig.lane, backend=rig.backend,
                        epochs=lambda: dict(worker_incarnation=stack.state.incarnation,
                            engine_epoch=epoch, route_generation=generation))
                return (None if store is None else factory), layers
            stack.server.row1_provider = provider

    def produce(self):
        assert self.engine.oracle and self.engine.intent['phase'] == 'row1-native'
        assert not self.produced
        for rank, stack in self.stacks.items():
            rig = self.rigs[rank]
            scope = stack.server.row1_evidence(run_id=RUN)
            assert scope['enabled'] is True
            for chunk in range(CHUNKS):
                route = 'dense_dma' if chunk % 2 == 0 else 'sparse_dma'
                for layer in scope['local_layers']:
                    rig.qualifier = stack.server.row1_factory(None)
                    hold = types.SimpleNamespace(released=False)
                    run = rig.make_run(layer_name=layer, state_epoch=chunk+1,
                                       sparse_hold=hold, row1_reference=lambda: None)
                    buckets = []
                    for b in range(2):
                        bucket = replace(rig.bucket(index=b), route=route,
                                         compact_start=b*8 if route == 'sparse_dma' else None)
                        rig.fill_actual(bucket)
                        rig.qualifier.bucket(run, bucket, rig.arena)
                        buckets.append(bucket)
                    def reference(session):
                        session.route('dense_dma' if route == 'dense_dma' else 'sparse_uva', 2)
                        if route == 'dense_dma':
                            for _ in range(2):
                                session.fill(rig.arena, 0, 1, source.T((1,), values=[0]))
                    rig.output(run, reference, rig.out_tensor([0.0, 0.0]),
                               expected_route='dense_dma' if route == 'dense_dma' else 'sparse_uva',
                               expected_fills=2 if route == 'dense_dma' else 0)
                    rig.qualifier.check_releasable(run)
                    hold.released = True
                    count = 2 if route == 'sparse_dma' else 0
                    rig.qualifier.finish_layer(run, hold_stats=dict(
                        buckets=count, copied=count, published=count, consumed=count,
                        abandoned=0, rows=8*count, descriptors=2*count,
                        nbytes=sum(b.nbytes for b in buckets) if count else 0, released=1))
        self.produced = True

    def worker(self, command, timeout_s=None):
        action = command['worker_action']; payload = command['payload']
        self.worker_actions.append(action)
        self.events.append(action)
        inner = {}
        if action == 'row1_configure':
            wanted = payload['enabled']
            if wanted:
                assert self.engine.route == 'split' and self.engine.intent is not None
            if not wanted and self.terminal_full:
                self.engine.route = 'full'; self.engine.generation += 1
            for rank, stack in self.stacks.items():
                inner[rank] = stack.server.configure_row1(enabled=wanted,
                    run_id=RUN, engine_epoch=self.engine.epoch,
                    route_generation=self.engine.generation)
            self.engine.oracle = wanted
        elif action == 'row1_evidence':
            self.evidence_when_enabled = self.engine.oracle
            assert not self.engine.oracle
            inner = {rank: stack.server.row1_evidence(run_id=RUN)
                     for rank, stack in self.stacks.items()}
        else:
            raise AssertionError(action)
        result = dict(schema=1, action='qualification_worker', run_id=command['run_id'],
            command_seq=command['command_seq'], client_id=command['client_id'],
            engine_epoch=self.engine.epoch, worker_reply=self._rows(inner))
        if action == 'row1_evidence' and self.omit_last:
            result['worker_reply']['workers'].pop()
        self.packet_bytes = max(self.packet_bytes, len(json.dumps(result).encode()))
        return result


class Handler(composition._CompositionHandler):
    def _completions(self, body):
        self.server.control.produce()
        return super()._completions(body)


class DaytimeRehearsal(unittest.TestCase):
    def collect(self, root, *, terminal_full=False, omit_last=False):
        engine = composition.FakeEngine(epoch=1, terminal_full=terminal_full)
        control = SourceControl(engine, terminal_full=terminal_full, omit_last=omit_last)
        httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        httpd.control = control
        httpd.prompt_tokens = 8  # HTTP/SSE fixture only; record count is CHUNKS.
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        url = 'http://127.0.0.1:%d' % httpd.server_address[1]
        plan = composition.row1_plan()
        # Only this CPU fixture needs time to generate thousands of byte records.
        # The deployed workload and its 900-second budget are unchanged.
        plan['workload']['timeouts'] = {k: 120.0 for k in plan['workload']['timeouts']}
        frozen = qd.freeze_row1_plan(plan, run_id=RUN, model='fixture')
        producer = qd.NativeRow1Producer(plan=frozen, url=url, model='fixture')
        driver = qd.QualificationDriver(root, run_id=RUN, model='fixture', engine_epoch=1,
            url=url, key='fixture-key', row1_producer=producer, timeout_s=30)
        try:
            result = driver.collect_phase('row1')
        except qd.QualificationRefused as exc:
            if not omit_last:
                raise
            result = exc
        return result, control, engine, driver

    def test_full_prompt_record_volume_survives_scope_http_and_raw_grade(self):
        with tempfile.TemporaryDirectory() as root:
            result, control, engine, driver = self.collect(root)
            self.assertTrue(result['complete'], result['missing'])
            evidence = result['evidence']
            verdict = grade.grade(evidence, run_id=RUN, engine_epoch=1)
            self.assertEqual(verdict['verdict'], 'PASS', verdict)
            self.assertEqual(CHUNKS, 97)
            for packet in evidence['workers']:
                self.assertEqual(len(packet['local_layers']), 6)
                self.assertEqual(packet['raw']['layers'], CHUNKS*6)
                self.assertEqual(packet['raw']['comparisons'], CHUNKS*6*3)
                self.assertLess(packet['raw']['comparisons'], packet['raw']['capacity'])
                self.assertFalse(packet['enabled'])
            self.assertLess(control.packet_bytes, rpc.MAX_RESPONSE)
            self.assertEqual(engine.route, 'full')
            self.assertIsNone(engine.intent)
            self.assertFalse(control.evidence_when_enabled)
            self.assertEqual(result['row1_sha256'], qd.canonical_sha(evidence))
            # The store persists and a repeat collection cannot overwrite/select.
            for rank, stack in control.stacks.items():
                self.assertEqual(len(stack.server.row1_store.records()), CHUNKS*18)
            with self.assertRaises(qd.QualificationError):
                driver.collect_phase('row1')
            print('daytime-row1: chunks=%d ranks=4 layers_per_rank=6 comparisons_per_rank=%d response_bytes=%d limit=%d synthetic_cpu_only=true'
                  % (CHUNKS, CHUNKS*18, control.packet_bytes, rpc.MAX_RESPONSE))

    def test_terminal_full_keeps_all_records_and_both_generations(self):
        with tempfile.TemporaryDirectory() as root:
            result, control, engine, _ = self.collect(root, terminal_full=True)
            self.assertTrue(result['complete'], result['missing'])
            self.assertEqual(grade.grade(result['evidence'], run_id=RUN, engine_epoch=1)['verdict'], 'PASS')
            for packet in result['evidence']['workers']:
                self.assertEqual(packet['disabled_route_generation'], packet['route_generation']+1)
            self.assertTrue(result['evidence']['cleanup']['terminal_full'])
            self.assertEqual(engine.route, 'full')
            self.assertNotIn('leave_full', control.session_actions)

    def test_missing_worker_cannot_turn_a_large_partial_receipt_into_pass(self):
        with tempfile.TemporaryDirectory() as root:
            result, control, engine, _ = self.collect(root, omit_last=True)
            self.assertIsInstance(result, qd.QualificationRefused)
            self.assertEqual(result.code, 'row1-worker-ranks-incomplete')
            self.assertTrue(result.cleanup['restored'])
            self.assertTrue(control.produced)
            self.assertEqual(engine.route, 'full')
            self.assertTrue(all(not s.server.row1_evidence(run_id=RUN)['enabled']
                                for s in control.stacks.values()))


import test_e3_native_performance_queue as native_queue


class HeldNative12QueueTests(native_queue.old.OverlapQueue, unittest.TestCase):
    """Exact held queue bytes, fake OS and phase executables only.

    Tests shell order/cleanup, not phase calculations or engine behavior.
    The actual source/client/grader composition is exercised above.
    """
    step = native_queue.NativeQueueTests.step
    read_json = native_queue.NativeQueueTests.read_json
    mark = native_queue.NativeQueueTests.mark
    rehash_requests = native_queue.NativeQueueTests.rehash_requests
    withheld = native_queue.NativeQueueTests.withheld

    def fixture_helper(self):
        helper = native_queue.NativeQueueTests('test_every_native_phase_and_unchanged_hygiene_then_exact_restore')
        # Preserve the reviewed fixture's super() chain over the same temporary
        # filesystem. This object performs no additional setUp/tearDown.
        helper.__dict__ = self.__dict__
        return helper

    def make_harness(self):
        self.fixture_helper().make_harness()
        for suffix in ('.draft.sh', '_settings.sh'):
            native_queue.shutil.copyfile(ROOT/'harness/runs'/('q18a_native12'+suffix),
                                         self.h/'runs'/('q18a_native12'+suffix))
        native_queue.write_manifest(self.h)

    def make_requests(self):
        self.fixture_helper().make_requests()
        pins = self.r/'SHA256SUMS'
        pins.write_text(pins.read_text().replace('q18a_native1.', 'q18a_native12.')
                        .replace('q18a_native1_', 'q18a_native12_'))
        self.rehash_requests()

    def run_queue(self, env=None):
        settings = self.install_fakes()
        if env:
            settings.update(env)
        return native_queue.subprocess.run(
            ['bash', str(self.h/'runs/q18a_native12.draft.sh'), str(self.q)],
            capture_output=True, text=True, timeout=180, env=settings, cwd=self.root)

    test_success_preserves_all_gates_and_measured_shortfall = native_queue.NativeQueueTests.test_every_native_phase_and_unchanged_hygiene_then_exact_restore
    test_row1_refusal_withholds_measurement = native_queue.NativeQueueTests.test_row1_refusal_withholds_w1_w2_and_hygiene
    test_failed_stop_withholds_restoration = native_queue.NativeQueueTests.test_failed_stop_withholds_both_restores


if __name__ == '__main__':
    unittest.main()
