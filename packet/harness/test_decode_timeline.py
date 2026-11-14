"""CPU timing/lifetime proof; CUDA cross-green elapsed support remains a device gate."""
from contextlib import ExitStack
import ast
import gc
import inspect
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace as NS, MethodType, ModuleType
import unittest
import weakref
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'patches/s5_engine'), str(ROOT/'harness')]
import decode_timeline as dt
import native_measurement as nm
import executor_hooks as eh
from graph_router import ExecutionDescriptor
import test_stage1_handler_anatomy as ha
import test_n3_fused_phases as fp
import test_e3_sparse_history_dma as sparse
import test_alpha_next_receipts as atomic
import receipt_lines as rl


class Port:
    compute = 'compute'
    copy = 'copy'
    native_fused_phase_every = 16

    def __init__(self, torch):
        self.torch = torch
        self.streams = {s: ha.Stream(torch.device, s) for s in ('compute', 'copy')}
        for i, stream in enumerate(self.streams.values(), 201):
            stream.cuda_stream = i

    def timing_event(self, stream):
        event = ha.Event(self.torch.device, enable_timing=True)
        event.record(self.streams[stream])
        return event

    def query(self, event):
        return event.query()

    def stream_token(self, stream):
        return self.streams[stream].cuda_stream


class Case(unittest.TestCase):
    def rig(self, *, selected=True, rank=2):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.dict(os.environ, {dt.ENV: '1' if selected else '0'}))
        r = ha.Rig(stack, rank, runner='v2split')
        port = Port(r.torch)
        r.torch.stream.cuda_stream = 301
        r.runner._flash_s5_worker = NS(operation_device=port)
        r.runner._flash_s5_native = NS(controller=NS(route='split', route_generation=7))
        r.runner._flash_s5_split = NS(decode_sms=24, total_sms=82)
        # The existing pinned-order runner leaf passes a string descriptor. Supply the real mirror shape instead.
        src = textwrap.dedent(inspect.getsource(ha.RunnerV2.execute_model)).replace(
            "native_replay('descriptor',", 'native_replay(scheduler_output.descriptor,')
        scope = dict(vars(ha))
        exec(compile(src, '<declared-runner-descriptor-leaf>', 'exec'), scope)
        r.runner.execute_model = MethodType(scope['execute_model'], r.runner)
        recorder = eh._anatomy(r.proc)
        return r, port, recorder

    def native(self, r, port, seq=1):
        run = NS(layer_name='qsa.0', valid_tokens=201261, pages=tuple(range(50)), buckets=(0,))
        op = NS(device=port, transaction=sparse.make_transaction(chunk_end=201261, txn_seq=seq),
                lane=sparse.make_lane(worker_rank=r.rank), _layers={'qsa.0': run}, _cancelled=None, stats={})
        op._current_run = lambda: run
        trace = nm.FusedPhaseSpans(op)
        trace.group = (0, 2)
        trace.stage, trace.route_generation = r.rank//2, 7
        trace.fused_layers.add('qsa.0')
        return trace

    def step(self, r, names=('not-a-payload-a', 'not-a-payload-b')):
        output = NS(num_scheduled_tokens={k: 1 for k in names},
                    scheduled_cached_reqs=NS(req_ids=list(names), num_computed_tokens=[48000+i for i in range(len(names))]),
                    descriptor=ExecutionDescriptor('full', 4, 4, 1, 1))
        seq = r.rpc('execute_model', (output,))
        r.rpc('sample_tokens', (None,))
        r.torch.device.advance()
        return seq

    def finish(self, trace):
        lines = []
        with patch.object(nm, 'emit_receipt_lines', side_effect=lambda texts, logger: lines.extend(texts)), \
                patch.object(dt, 'emit_receipt_lines', side_effect=lambda texts, logger: lines.extend(texts)):
            trace.emit()
        return [json.loads(s[len(dt.TimelineTrace.LINE):]) for s in lines if s.startswith(dt.TimelineTrace.LINE)]

    def decoded(self, lines):
        return {tag: [row for line in lines for row in line['rows'].get(tag, [])] for tag in ('d', 'p', 's')}

    def test_real_generated_rpc_recorder_and_fused_join_all_ranks(self):
        for rank in range(4):
            r, port, recorder = self.rig(rank=rank)
            trace = self.native(r, port)
            fp.all_phases(trace)
            seq = self.step(r)
            got = self.finish(trace)
            rows = self.decoded(got)
            self.assertEqual(rows['d'][0][0], seq)
            meta = rows['d'][0][1]
            self.assertEqual((meta['actual_rows'], meta['actual_tokens'], meta['num_tokens'], meta['num_reqs']), (2, 2, 4, 4))
            self.assertEqual((meta['bank'], meta['route_generation'], meta['decode_sms'], meta['stream']), ('split', 7, 24, 301))
            self.assertEqual(meta['bank_sms'], 24)
            self.assertEqual(len(meta['uid_digest']), 64)
            self.assertEqual(meta['history_tokens'], [48000, 48001])
            self.assertNotIn('not-a-payload', json.dumps(got))
            self.assertEqual(rows['s'][-1][0], True)
            self.assertTrue(all(line['rank'] == rank for line in got))
            self.assertAlmostEqual((rows['d'][0][3]-rows['d'][0][2])*1e6, rows['d'][0][4], delta=2)
            self.assertEqual({row[0] for row in rows['p']}, dt.PHASES - {'prefetch_copy'})
            self.assertTrue(all(line['stage'] == rank//2 for line in got))
            self.assertEqual(r.torch.device.violations, [])
            self.assertIsNone(recorder.timeline.active)

    def test_no_added_event_records_or_dependencies_and_same_rpc_results(self):
        calls, replies = [], []
        for selected in (False, True):
            r, port, recorder = self.rig(selected=selected)
            trace = self.native(r, port)
            fp.all_phases(trace)
            self.step(r)
            lines = self.finish(trace)
            self.assertEqual(recorder.timeline is not None, selected)
            self.assertEqual(bool(lines), selected)
            calls.append([c for c in r.torch.device.calls if c == 'Event.record'])
            replies.append([getattr(x[1], 'tokens', x[1]) for x in r.replies(2)])
            self.assertEqual(r.torch.device.violations, [])
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(replies[0], replies[1])

    def test_recycled_forward_pair_cannot_change_recorded_offsets(self):
        r, port, recorder = self.rig()
        trace = self.native(r, port)
        fp.all_phases(trace)
        self.step(r)
        recorder._resolve()
        before = list(recorder.timeline.active['trace'].rows)
        for first, last in recorder.pool:
            first.stamp = last.stamp = 99*10**15
        got = self.finish(trace)
        self.assertEqual(self.decoded(got)['d'][0][2:4], list(before[0][3:5]))

    def test_full_bank_metadata_distinguishes_partition_from_provisioned_sms(self):
        r, port, recorder = self.rig()
        trace = self.native(r, port)
        fp.all_phases(trace)
        r.runner._flash_s5_native.controller.route = 'full'
        self.step(r)
        meta = self.decoded(self.finish(trace))['d'][0][1]
        self.assertEqual((meta['bank'], meta['decode_sms'], meta['bank_sms']), ('full', 24, 82))

    def test_prior_completed_event_survives_trace_but_not_rotation(self):
        r, port, recorder = self.rig()
        first = self.native(r, port)
        fp.all_phases(first)
        self.step(r)
        self.finish(first)
        prior_event = weakref.ref(recorder.timeline.previous[2])
        prior_trace = weakref.ref(first)
        del first
        gc.collect()
        self.assertIsNone(prior_trace())
        self.assertIsNotNone(prior_event())
        second = self.native(r, port, seq=2)
        self.step(r)
        recorder._resolve()  # Before this op has any native phase event.
        self.assertTrue(all(x is not None for x in recorder.timeline.active['trace'].rows[0][3:5]))
        fp.all_phases(second)
        got = self.finish(second)
        self.assertTrue(all((x['sample'], x['anchor_sample']) == (2, 1) for x in got))
        gc.collect()
        self.assertIsNone(prior_event())
        self.assertEqual(recorder.timeline.previous[1], 2)

    def test_previous_anchor_never_crosses_epoch_or_rotates_on_cancel(self):
        from dataclasses import replace
        r, port, recorder = self.rig()
        first = self.native(r, port)
        fp.all_phases(first)
        self.step(r)
        self.finish(first)
        original = recorder.timeline.previous
        cancelled = self.native(r, port, seq=2)
        fp.all_phases(cancelled)
        self.step(r)
        cancelled.operation._cancelled = 'fixture cancel'
        self.finish(cancelled)
        self.assertIs(recorder.timeline.previous, original)
        next_trace = self.native(r, port, seq=3)
        next_trace.operation.transaction = replace(next_trace.operation.transaction,
                                                  engine_epoch=next_trace.operation.transaction.engine_epoch + 1)
        recorder.timeline.begin(next_trace)
        self.assertIsNone(recorder.timeline.active['anchor'])
        self.assertIsNone(recorder.timeline.previous)

    def test_copy_and_decode_share_one_relative_clock_without_dependency(self):
        r, port, recorder = self.rig()
        trace = self.native(r, port)
        fp.all_phases(trace)
        token = trace.begin('copy', 'copy', bucket=0)
        port.streams['copy'].launch(7_000_000)
        trace.end(token)
        self.step(r)
        rows = self.decoded(self.finish(trace))
        copy = [p for p in rows['p'] if p[0] == 'copy'][-1]
        self.assertEqual(copy[-2:], [0., 7.])
        self.assertEqual(copy[4], 202)
        self.assertEqual(r.torch.device.violations, [])

    def test_anchor_pending_and_cross_green_error_never_wait_or_fail_rpc(self):
        for pending in (True, False):
            r, port, recorder = self.rig()
            trace = self.native(r, port)
            fp.all_phases(trace)
            self.step(r)
            anchor = recorder.timeline.active['anchor']
            with patch.object(anchor, 'query', return_value=not pending), \
                    patch.object(anchor, 'elapsed_time', side_effect=RuntimeError('fixture cross-context refusal')):
                recorder._resolve()
                got = self.finish(trace)
            rows = self.decoded(got)
            self.assertFalse(rows['s'][-1][0])
            self.assertTrue(rows['s'][-1][1].get('pending' if pending else 'timing_errors'))
            self.assertTrue(rows['d'])  # Shape metadata survives unavailable timing.
            self.assertTrue(all(row[2:4] == [None, None] for row in rows['d']))
            self.assertEqual(r.torch.device.violations, [])

    def test_old_or_empty_rpc_not_mislabelled_as_current_sample(self):
        r, port, recorder = self.rig()
        self.step(r)
        trace = self.native(r, port)
        fp.all_phases(trace)
        recorder.timeline.inputs(999, NS(num_scheduled_tokens={}))
        seq = self.step(r)
        rows = self.decoded(self.finish(trace))
        self.assertEqual([d[0] for d in rows['d']], [seq])
        self.assertTrue(rows['s'][-1][0])

    def test_cancel_and_incomplete_fused_source_are_not_complete(self):
        r, port, recorder = self.rig()
        trace = self.native(r, port)
        fp.all_phases(trace)
        self.step(r)
        trace.operation._cancelled = object()
        rows = self.decoded(self.finish(trace))
        self.assertFalse(rows['s'][-1][0])

    def test_incomplete_source_without_cancel_stays_incomplete(self):
        r, port, recorder = self.rig()
        trace = self.native(r, port)
        fp.all_phases(trace)
        trace.pairs = [p for p in trace.pairs if p[1] != 'score']
        self.step(r)
        rows = self.decoded(self.finish(trace))
        self.assertTrue(rows['p'])
        self.assertFalse(rows['s'][-1][0])

    def test_foreign_old_finish_does_not_close_new_anchor(self):
        r, port, recorder = self.rig()
        old = self.native(r, port)
        fp.all_phases(old)
        newer = self.native(r, port, seq=2)
        fp.all_phases(newer)
        self.assertEqual(self.finish(old), [])
        seq = self.step(r, names=('next-request',))
        got = self.finish(newer)
        self.assertTrue(all(line['txn'] == 2 for line in got))
        self.assertEqual(self.decoded(got)['d'][0][0], seq)

    def test_metadata_and_phase_caps(self):
        r, port, recorder = self.rig()
        trace = self.native(r, port)
        fp.all_phases(trace)
        self.step(r)
        recorder._resolve()
        with patch.object(dt, 'MAX_METADATA', 1):
            for seq in (1000, 1001):
                recorder.timeline.inputs(seq, NS(num_scheduled_tokens={'extra-request': 1}))
            self.assertEqual(len(recorder.timeline.active['metadata']), 1)
        with patch.object(dt, 'MAX_PHASES', 1):
            rows = self.decoded(self.finish(trace))
        self.assertEqual(len(rows['p']), 1)
        self.assertEqual(rows['s'][-1][1]['metadata_dropped'], 1)
        self.assertGreater(rows['s'][-1][1]['phases_dropped'], 0)

    def test_limits_and_log_budget_drop_visibly(self):
        r, port, recorder = self.rig()
        trace = self.native(r, port)
        fp.all_phases(trace)
        with patch.object(dt, 'MAX_FORWARDS', 1):
            self.step(r); recorder._resolve()
            self.step(r); recorder._resolve()
            got = self.finish(trace)
        rows = self.decoded(got)
        self.assertEqual(len(rows['d']), 1)
        self.assertEqual(rows['s'][-1][1]['forwards_dropped'], 1)
        trace = self.native(r, port, seq=2)
        fp.all_phases(trace)
        self.step(r)
        recorder.timeline.bytes = dt.MAX_LOG_BYTES - 4096
        rows = self.decoded(self.finish(trace))
        self.assertEqual(rows['s'], [[False, {'log_budget_exhausted': 1}]])
        self.assertTrue(recorder.timeline.blocked)
        self.assertLessEqual(recorder.timeline.bytes, dt.MAX_LOG_BYTES)

    def test_startup_off_and_incomplete_config(self):
        dt.check_startup({})
        env = {dt.ENV: '1', 'FLASH_S5_HANDLER_ANATOMY':'1', 'FLASH_S5_AUTO_ARM':'1',
               'FLASH_S5_NATIVE_PROGRESS':'burst', 'FLASH_S5_FUSED_PHASE_EVERY':'16'}
        dt.check_startup(env)
        for k in env:
            wrong = dict(env, **{k:'bad'})
            with self.assertRaises(ValueError): dt.check_startup(wrong)

    def test_atomic_writer_accepts_new_prefix_and_no_raw_uid(self):
        r, port, recorder = self.rig()
        trace = self.native(r, port)
        fp.all_phases(trace)
        self.step(r)
        writer, cap = atomic.capture_writer()
        try:
            with patch.object(nm, '_RECEIPT_WRITER', writer): trace.emit()
            lines = atomic.lines_from(cap.writes)
            self.assertEqual(rl.verdict(rl.check(lines))['verdict'], 'OK')
            self.assertTrue(any(dt.TimelineTrace.LINE in s for s in lines))
            self.assertEqual((writer.oversize, writer.faults), (0, 0))
            self.assertTrue(all(len(data) <= 3968 for _, data in cap.writes))
        finally:
            atomic.close_pipe(writer)

    def test_failure_in_diagnostic_sink_leaves_rpc_and_native_source_unchanged(self):
        r, port, recorder = self.rig()
        trace = self.native(r, port)
        fp.all_phases(trace)
        self.step(r)
        with patch.object(nm, 'emit_receipt_lines', return_value=True), \
                patch.object(dt, 'emit_receipt_lines', side_effect=OSError('fixture sink failure')):
            trace.emit()
        self.assertIsNone(recorder.timeline.active)
        self.assertTrue(recorder.timeline.blocked)
        self.assertEqual((trace.errors, trace.dropped), (0, 0))
        self.assertEqual(len(r.replies(2)), 2)

    def test_refused_sink_stops_observer_and_releases_retained_anchor(self):
        r, port, recorder = self.rig()
        trace = self.native(r, port)
        fp.all_phases(trace)
        self.step(r)
        with patch.object(nm, 'emit_receipt_lines', return_value=True), \
                patch.object(dt, 'emit_receipt_lines', return_value=False):
            trace.emit()
        self.assertTrue(recorder.timeline.blocked)
        self.assertIsNone(recorder.timeline.previous)
        self.assertEqual((trace.errors, trace.dropped), (0, 0))

    def test_supersession_sink_refusal_stays_blocked(self):
        r, port, recorder = self.rig()
        first = self.native(r, port)
        fp.all_phases(first)
        self.step(r)
        with patch.object(dt, 'emit_receipt_lines', return_value=False):
            self.native(r, port, seq=2)
        self.assertTrue(recorder.timeline.blocked)
        self.assertIsNone(recorder.timeline.active)
        self.assertEqual(recorder.timeline.sessions, 1)

    def test_manual_split_requires_real_current_route_lease(self):
        driver = NS(engine_epoch=7, route_generation=9, _split_ready=True,
                    auto_arm=NS(manual_owned=True))
        engine = NS(model_executor=NS(), _flash_s5_driver=driver)
        good = dict(owner='native-control', intent='enter_split', engine_epoch=7,
                    route_generation=9, since_ns=123)
        for route in (None, 'split', {}, dict(good, owner='other'), dict(good, intent='leave_split'),
                      dict(good, engine_epoch=6), dict(good, route_generation=8)):
            driver._control_route = route
            dt.engine_step(engine)
            self.assertFalse(engine.model_executor._flash_s5_decode_timeline.enabled)
        driver._control_route = good
        dt.engine_step(engine)
        self.assertTrue(engine.model_executor._flash_s5_decode_timeline.enabled)
        driver._split_ready = False
        dt.engine_step(engine)
        self.assertFalse(engine.model_executor._flash_s5_decode_timeline.enabled)

    def outside_rpc(self, r, sampler, method, args=()):
        r.seq += 1
        tag = sampler.wire(method, r.seq, args, True)
        kwargs = {dt.OUTSIDE_KEY: tag} if tag is not None else {}
        r.proc._execute_worker_rpc((method, args, kwargs, ha.OUTPUT_RANK))
        self.assertNotIn(dt.OUTSIDE_KEY, kwargs)
        return r.seq

    def test_outside_native_chunks_all_ranks_have_own_local_completion_proof(self):
        for rank in range(4):
            r, port, recorder = self.rig(rank=rank)
            sampler = dt.OutsideSampler()
            sampler.enabled, sampler.epoch = True, 1
            output = NS(num_scheduled_tokens={'outside-member': 1},
                        scheduled_cached_reqs=NS(req_ids=['outside-member'], num_computed_tokens=[48000]),
                        descriptor=ExecutionDescriptor('full', 4, 4, 1, 1))
            records_before = r.torch.device.calls.count('Event.record')
            lines = []
            with patch.object(dt, 'emit_receipt_lines', side_effect=lambda texts, logger: lines.extend(texts)):
                sequence = self.outside_rpc(r, sampler, 'execute_model', (output,))
                self.outside_rpc(r, sampler, 'sample_tokens', (None,))
                r.torch.device.advance()
                # Untagged RPC polls existing endpoints without creating a handler row.
                self.outside_rpc(r, sampler, 'execute_model', (output,))
            got = [json.loads(x[len(dt.TimelineTrace.LINE):]) for x in lines]
            rows = self.decoded(got)
            self.assertEqual(len(rows['d']), 1)
            self.assertEqual(rows['d'][0][0], sequence)
            self.assertEqual(rows['d'][0][1]['history_tokens'], [48000])
            self.assertIsNotNone(rows['d'][0][-1])
            complete_ns = got[0]['base_ns'] + rows['d'][0][-1]*1000
            enter_ns = got[0]['base_ns'] + rows['d'][0][-3]*1000
            self.assertGreaterEqual(complete_ns, enter_ns)
            self.assertTrue(all(x['scope'] == 'outside-native-progress' and x['stage'] == rank//2 for x in got))
            self.assertTrue(rows['s'][-1][0])
            self.assertEqual(rows['p'], [])
            self.assertIsNone(recorder.timeline.active)
            self.assertEqual(r.torch.device.violations, [])
            self.assertNotIn('outside-member', json.dumps(got))
            self.assertIn(r.torch.device.calls.count('Event.record') - records_before, (2, 4))

    def test_outside_sampler_skips_manual_native_recording_empty_and_caps(self):
        sampler = dt.OutsideSampler()
        output = NS(num_scheduled_tokens={'member': 1})
        self.assertIsNone(sampler.wire('execute_model', 1, (output,), True))
        sampler.enabled, sampler.epoch = True, 1
        self.assertIsNone(sampler.wire('execute_model', 2, (output,), False))
        self.assertIsNone(sampler.wire('execute_model', 3, (NS(num_scheduled_tokens={}),), True))
        with patch.object(dt, 'MAX_OUTSIDE', 1):
            tag = sampler.wire('execute_model', 4, (output,), True)
            self.assertEqual(tag[:2], [4, 1])
            self.assertEqual(sampler.wire('sample_tokens', 5, (), False)[-1], 4)
            self.assertIsNone(sampler.wire('execute_model', 6, (output,), True))

    def test_outside_sample_is_not_bound_across_an_intervening_execute(self):
        sampler = dt.OutsideSampler()
        sampler.enabled, sampler.epoch = True, 1
        output = NS(num_scheduled_tokens={'member': 1})
        self.assertIsNotNone(sampler.wire('execute_model', 1, (output,), True))
        self.assertIsNone(sampler.wire('execute_model', 2, (output,), False))
        self.assertIsNone(sampler.wire('sample_tokens', 3, (), False))

    def test_outside_pending_forward_is_not_proven_complete_or_waited(self):
        r, port, recorder = self.rig()
        sampler = dt.OutsideSampler()
        sampler.enabled, sampler.epoch = True, 1
        output = NS(num_scheduled_tokens={'member': 1}, descriptor=ExecutionDescriptor('full', 4, 4, 1, 1))
        self.outside_rpc(r, sampler, 'execute_model', (output,))
        recorder.timeline.poll_outside()
        self.assertEqual(recorder.timeline.outside['forwards'], 0)
        self.assertEqual(r.torch.device.violations, [])

    def test_outside_local_proof_survives_unsupported_cross_green_timing(self):
        r, port, recorder = self.rig()
        recorder.timeline.timing_failed = True
        sampler = dt.OutsideSampler()
        sampler.enabled, sampler.epoch = True, 1
        output = NS(num_scheduled_tokens={'member': 1}, descriptor=ExecutionDescriptor('full', 4, 4, 1, 1))
        lines = []
        with patch.object(dt, 'emit_receipt_lines', side_effect=lambda texts, logger: lines.extend(texts)):
            self.outside_rpc(r, sampler, 'execute_model', (output,))
            r.torch.device.advance()
            self.outside_rpc(r, sampler, 'sample_tokens', (None,))
        got = [json.loads(x[len(dt.TimelineTrace.LINE):]) for x in lines]
        row = self.decoded(got)['d'][0]
        self.assertEqual(row[2], 0.)
        self.assertAlmostEqual(row[3]*1e6, row[4], delta=2)
        self.assertTrue(row[1]['completion'].startswith('own-forward'))

    def test_outside_malformed_tag_is_stripped_without_changing_dispatch(self):
        r, port, recorder = self.rig()
        output = NS(num_scheduled_tokens={'member': 1}, descriptor=ExecutionDescriptor('full', 4, 4, 1, 1))
        kwargs = {dt.OUTSIDE_KEY: [-1]}
        r.proc._execute_worker_rpc(('execute_model', (output,), kwargs, ha.OUTPUT_RANK))
        self.assertEqual(kwargs, {})
        self.assertEqual(recorder.counters['observer_errors'], 1)
        self.assertIsNone(recorder.timeline.outside)

    def test_disabled_observer_cannot_emit_a_second_budget_notice(self):
        r, port, recorder = self.rig()
        observer = recorder.timeline
        trace = self.native(r, port)
        fp.all_phases(trace)
        sampler = dt.OutsideSampler()
        sampler.enabled, sampler.epoch = True, 1
        output = NS(num_scheduled_tokens={'member': 1}, descriptor=ExecutionDescriptor('full', 4, 4, 1, 1))
        self.outside_rpc(r, sampler, 'execute_model', (output,))
        r.torch.device.advance()
        observer.bytes = dt.MAX_LOG_BYTES - 4096
        self.finish(trace)
        before = observer.bytes
        with patch.object(dt, 'emit_receipt_lines') as sink:
            observer.poll_outside()
        self.assertFalse(sink.called)
        self.assertEqual(observer.bytes, before)


class OutsideWire(unittest.TestCase):
    def test_real_generated_collective_rpc_keeps_fifo_arguments_and_default_off_bytes(self):
        import test_e3_native_reply_hol as hol
        for selected in (False, True):
            with ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ, {dt.ENV: '1' if selected else '0'}))
                stack.enter_context(patch.object(eh, '_anatomy', return_value=None))
                stack.enter_context(patch.object(eh, '_ANATOMY', False))
                stack.enter_context(patch.object(eh, '_WORKER_PROGRESS', None))
                stack.enter_context(patch.object(eh, '_DECODE_STEP', False))
                net = hol.Network(self)
                for proc in net.procs:
                    proc.worker.execute_model = MethodType(lambda worker, output: ('executed', worker.rank), proc.worker)
                engine = NS(model_executor=net.executor,
                    _flash_s5_driver=NS(engine_epoch=7, auto_arm=NS(manual_owned=False)))
                dt.engine_step(engine)
                output = NS(num_scheduled_tokens={'member': 1})
                execute = net.executor.collective_rpc('execute_model', args=(output,), non_block=True,
                    unique_reply_rank=hol.OUTPUT_RANK, timeout=5)
                self.assertEqual(execute.result(5), ('executed', 2))
                sample = net.executor.collective_rpc('sample_tokens', args=('outside',), non_block=True,
                    unique_reply_rank=hol.OUTPUT_RANK, timeout=5)
                net.procs[2].worker.decode_done.setdefault('outside', __import__('threading').Event()).set()
                self.assertEqual(sample.result(5), ('decode', 'outside'))
                sent = net.executor.rpc_broadcast_mq.sent
                self.assertEqual([x[0] for x in sent], ['execute_model', 'sample_tokens'])
                self.assertEqual([list(x[2]) for x in sent], [[dt.OUTSIDE_KEY], [dt.OUTSIDE_KEY]] if selected else [[], []])
                if selected:
                    self.assertEqual(sent[1][2][dt.OUTSIDE_KEY][-1], execute.sequence)
                    self.assertTrue(all(x[2][dt.OUTSIDE_KEY][1] == 7 for x in sent))
                self.assertIsNone(execute._progress)
                self.assertIsNone(sample._progress)
                engine._flash_s5_driver.auto_arm.manual_owned = True
                dt.engine_step(engine)
                self.assertFalse(net.executor._flash_s5_decode_timeline.enabled)
                engine._flash_s5_driver._control_route = 'split'
                dt.engine_step(engine)
                self.assertFalse(net.executor._flash_s5_decode_timeline.enabled)
                engine._flash_s5_driver._split_ready = True
                engine._flash_s5_driver.route_generation = 9
                engine._flash_s5_driver._control_route = dict(owner='native-control', intent='enter_split',
                    engine_epoch=7, route_generation=9, since_ns=1)
                dt.engine_step(engine)
                self.assertTrue(net.executor._flash_s5_decode_timeline.enabled)
                net.close()


class RealManualSplit(unittest.IsolatedAsyncioTestCase):
    async def test_actual_route_control_dict_enables_and_full_return_disables_sampling(self):
        import asyncio
        import test_e3_native_overlap as native
        from native_overlap_control import NativeControlClient
        with patch.dict(os.environ, {dt.ENV: '1'}):
            stack = native.idle_stack()
            core, frontend = native.core_for(stack)
            control = NativeControlClient(frontend, timeout_s=5)
            task = asyncio.create_task(control.enter_split())
            await native.pump_until(lambda: task.done(), core)
            self.assertEqual(task.result()['route_after'], 'split')
            self.assertIs(type(stack.driver._control_route), dict)
            stack.driver.auto_arm = NS(manual_owned=True)
            dt.engine_step(core)
            self.assertTrue(core.model_executor._flash_s5_decode_timeline.enabled)
            del stack.driver.auto_arm
            task = asyncio.create_task(control.leave_split())
            await native.pump_until(lambda: task.done(), core)
            self.assertEqual(task.result()['route_after'], 'full')
            stack.driver.auto_arm = NS(manual_owned=True)
            dt.engine_step(core)
            self.assertFalse(core.model_executor._flash_s5_decode_timeline.enabled)



class StripDiagnostic(ast.NodeTransformer):
    def visit_If(self, node):
        test = ast.dump(node.test)
        if 'FLASH_S5_DECODE_TIMELINE' in test or "attr='timeline'" in test:
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if any(isinstance(t, ast.Attribute) and t.attr == 'timeline' for t in node.targets):
            return None
        return self.generic_visit(node)

    def visit_Tuple(self, node):
        # 'decode_overlap': lever 2's module-list entry (claude/n3-lever2-split, 14 Nov); its own default-off proof is
        # test_decode_overlap (OffPath), so the frozen parent stays the timeline's only other difference here
        node.elts = [x for x in node.elts if not isinstance(x, ast.Constant)
                     or x.value not in (dt.TimelineTrace.LINE, 'decode_timeline', 'decode_overlap')]
        return self.generic_visit(node)


class Freeze(unittest.TestCase):
    def test_default_off_ast_is_the_frozen_compose_parent(self):
        pins = json.loads((ROOT/'harness/decode_timeline_parent_ast.json').read_text())
        for name, expected in pins['ast_sha256'].items():
            tree = StripDiagnostic().visit(ast.parse((ROOT/name).read_text()))
            self.assertEqual(hashlib.sha256(ast.dump(tree).encode()).hexdigest(), expected, name)

    def test_default_off_imports_no_timeline_in_fresh_process(self):
        code = "import sys,handler_anatomy,native_measurement,s5_startup; assert 'decode_timeline' not in sys.modules"
        env = dict(os.environ, PYTHONPATH=str(ROOT/'patches/s5_engine'), PYTHONDONTWRITEBYTECODE='1', **{dt.ENV:'0'})
        p = subprocess.run([sys.executable, '-B', '-c', code], env=env, text=True, capture_output=True)
        self.assertEqual(p.returncode, 0, p.stderr)


class Mutants(unittest.TestCase):
    def killed(self, old, new, case):
        source = (ROOT/'patches/s5_engine/decode_timeline.py').read_text()
        self.assertEqual(source.count(old), 1)
        module = ModuleType('decode_timeline_mutant')
        exec(compile(source.replace(old, new), '<timeline-mutant>', 'exec'), module.__dict__)
        original = Case(case)
        with patch.dict(sys.modules, {'decode_timeline': module}), patch.object(sys.modules[__name__], 'dt', module):
            result = unittest.TestResult()
            original.run(result)
        self.assertTrue(result.failures or result.errors, case)

    def test_reversed_gpu_offsets_caught(self):
        self.killed('float(anchor.elapsed_time(first)), float(anchor.elapsed_time(last))',
                    'float(first.elapsed_time(anchor)), float(last.elapsed_time(anchor))',
                    'test_real_generated_rpc_recorder_and_fused_join_all_ranks')

    def test_unqueried_anchor_caught(self):
        self.killed('anchor is None or not anchor.query() or', 'anchor is None or',
                    'test_anchor_pending_and_cross_green_error_never_wait_or_fail_rpc')

    def test_no_forward_bound_caught(self):
        self.killed("session['forwards'] >= MAX_FORWARDS", 'False', 'test_limits_and_log_budget_drop_visibly')

    def test_log_budget_guard_caught(self):
        self.killed('self.bytes + size > MAX_LOG_BYTES - 4096', 'False', 'test_limits_and_log_budget_drop_visibly')

    def test_no_source_completeness_guard_caught(self):
        self.killed('not source_complete or not phases', 'not phases',
                    'test_incomplete_source_without_cancel_stays_incomplete')

    def test_added_event_wait_caught(self):
        self.killed("        anchor = session['anchor']", "        anchor = session['anchor']\n        anchor.synchronize()",
                    'test_real_generated_rpc_recorder_and_fused_join_all_ranks')

    def test_missing_prior_anchor_caught(self):
        self.killed('anchor=previous[2] if previous else None', 'anchor=None',
                    'test_prior_completed_event_survives_trace_but_not_rotation')

    def test_restarting_after_refused_flush_caught(self):
        self.killed("            self._emit(False)\n            if self.blocked:\n                return",
                    "            self._emit(False)", 'test_supersession_sink_refusal_stays_blocked')

    def test_manual_split_dict_gate_removed_caught(self):
        self.killed("type(route) is dict", 'False', 'test_manual_split_requires_real_current_route_lease')

    def test_outside_scope_removed_caught(self):
        self.killed('if recorder is None or recorder.timeline is None:', 'if True:',
                    'test_outside_native_chunks_all_ranks_have_own_local_completion_proof')

    def test_local_completion_proof_removed_caught(self):
        self.killed('observer_ns, t_enter, t_exit, completed_ns)', 'observer_ns, t_enter, t_exit, None)',
                    'test_outside_native_chunks_all_ranks_have_own_local_completion_proof')


if __name__ == '__main__':
    unittest.main()
