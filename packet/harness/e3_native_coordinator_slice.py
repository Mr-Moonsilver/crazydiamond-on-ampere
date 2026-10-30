#!/usr/bin/env python3
"""Connected native19 CPU coordinator proof; device/model/PP leaves declared.

Actual mounted runtime classes and real OS-process pickle/FIFO replies. This
does not construct CUDA owners or claim shmem overflow/NCCL equivalence.
"""
import argparse
import contextlib
import dataclasses
import hashlib
import importlib
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import queue
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace

NETWORKS = []


def bootstrap(graph_path):
    sys.path[:0] = ['/proof', '/proof/harness', '/proof/builders',
                    '/usr/local/lib/python3.12/dist-packages/vllm/v1/flash_e3']
    graph = json.loads(Path(graph_path).read_text())
    for row in graph['modules']:
        name = row['module']
        if name.startswith(('vllm.v1.flash_e3.', 'vllm.v1.flash_e2.')) and row['active_mount']:
            mod = importlib.import_module(name)
            assert sys.modules.setdefault(name.rsplit('.', 1)[-1], mod) is mod
            assert hashlib.sha256(Path(inspect.getfile(mod)).read_bytes()).hexdigest() == row['sha256']
    return graph


def exception_graph(exc):
    result, seen = [], set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        result.append({'type': type(exc).__module__ + '.' + type(exc).__name__,
                       'message': str(exc), 'frames': [dict(file=f.filename, line=f.lineno,
                         function=f.name) for f in traceback.extract_tb(exc.__traceback__)]})
        exc = exc.__cause__ if exc.__cause__ is not None else exc.__context__
    return result


def write(path, data):
    # Proof-progress snapshots only; raw step journals below are append-only.
    path.write_text(json.dumps(data, indent=2, default=str) + '\n')


class ReplyReader:
    def __init__(self, connection):
        self.connection = connection

    def dequeue(self, timeout):
        if not self.connection.poll(timeout):
            raise TimeoutError('proof endpoint did not reply')
        return pickle.loads(self.connection.recv_bytes())


class ProcessBroadcast:
    """Bounded CPU FIFO leaf replacing the shmem MQ, no silent queue growth."""
    def __init__(self, connections, counts):
        self.connections, self.counts = connections, counts
        self.current_idx = 0
        self.n_remote_reader = 0
        self.buffer = SimpleNamespace(max_chunks=10, max_chunk_bytes=1024 * 1024,
                                      n_reader=4, get_metadata=self.metadata)
        self.envelopes = []

    @contextlib.contextmanager
    def metadata(self, index):
        # Actual unread command counts, not an always-ready ring flag.
        full = any(x.value >= self.buffer.max_chunks for x in self.counts)
        yield [int(full), 0, 0, 0, 0]

    def enqueue(self, envelope):
        payload = pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
        if any(x.value >= 10 for x in self.counts):
            raise RuntimeError('proof broadcast FIFO full')
        self.envelopes.append(envelope)
        for connection, count in zip(self.connections, self.counts):
            with count.get_lock():
                count.value += 1
            connection.send_bytes(payload)


def make_case(selection_count=4097):
    import test_e3_native_schedule_profile as proof
    class Mounted(proof.ScheduleProfile):
        generated_ops = Path('/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ops/qsa.py').read_text()
        generated_qsa = Path('/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/qsa.py').read_text()
    case = Mounted('runTest')
    case.selection_counts = (selection_count,) * 4
    case.setUp()
    return case


def child_worker(rank, stage, admission, recv, reply, count, controls, pp_slots, boot, graph, out, variant):
    try:
        bootstrap(graph)
        import test_e3_native_schedule_profile as proof
        import test_e3_native_submission_credit as credit
        import test_e3_r4_worker_bridge as r4
        import executor_hooks as eh
        import wire_v1 as w
        import e3_types as et
        from vllm.v1.executor.multiproc_executor import WorkerProc
        case = make_case(variant.get('selection_count', 4097))
        selection = list(case.selection_counts)
        if variant.get('mismatch_stage') == stage and rank % 2:
            selection[rank] = 97153
        case.selection_counts = tuple(selection)
        shell = r4.Stack.__new__(r4.Stack)
        shell.workers = {}
        shell._add_worker(rank, stage)
        worker = shell.workers[rank]
        stack = proof.HistoryStack(case, rank=rank, route='sparse',
                                   transaction=admission.transaction, ranks=(2 * stage, 2 * stage + 1))
        # PP submission metadata crosses actual process boundaries. Device
        # events remain CPU leaves, but cannot complete without a matching
        # transaction/stage/transfer at the other OS-process endpoint.
        class ProcessPP(r4.FakePPChannel):
            def publish(self, ticket):
                raw = pickle.dumps((ticket.transaction, ticket.stage, ticket.seq), protocol=5)
                assert len(raw) + 4 <= len(pp_slots[rank]), 'PP identity leaf capacity'
                with pp_slots[rank].get_lock():
                    pp_slots[rank][:len(raw) + 4] = len(raw).to_bytes(4, 'big') + raw
                return ticket

            def matched(self, ticket):
                slot = pp_slots[rank ^ 2]
                with slot.get_lock():
                    length = int.from_bytes(bytes(slot[:4]), 'big')
                    raw = bytes(slot[4:4 + length])
                return raw == pickle.dumps((ticket.transaction, ticket.stage, ticket.seq), protocol=5)

            def send(self, *args):
                return self.publish(super().send(*args))

            def post_receive(self, *args):
                return self.publish(super().post_receive(*args))

        stack.bus = worker.bus
        stack.torch = r4.bus_torch(stack.bus)
        worker.channel = ProcessPP(stack.bus, is_sender=stage == 0)
        stack.stage = stage
        class StageModel(credit.FakeModel):
            def _flash_e3_is_last_stage(self):
                return stage == 1
        # Existing single-stage fixture always answered last-stage=True. The
        # two-stage CPU model leaf must expose its actual synthetic stage.
        prior_layers = stack.model.layers
        stack.model = StageModel(stack.tokens)
        stack.model.layers = prior_layers
        if stage:
            stack.model.layers = [None, None] + stack.model.layers
            stack.model.start_layer, stack.model.end_layer = 2, 4
        stack.native.split = stack.split = worker.split
        stack.native.controller = worker.controller
        stack.native.router = worker.router
        stack.native.comms.pp = worker.channel
        stack.server = proof.ProfileServer(stack, worker.state)
        stack.dispatcher = credit.LaneDispatcher('coordinator-cpu-' + str(rank), stack.native,
            stack.factory, server=stack.server, driver_factory=stack._driver_factory,
            torch_module=stack.torch, clock=time.monotonic_ns)
        stack.bridge = credit.worker_bridge.NativeWorkerBridge(stack.dispatcher,
            server=stack.server, source_binder=stack._source_binder)
        worker.dispatcher = stack.dispatcher
        worker._flash_s5_bridge = stack.bridge
        worker.server, worker.factory = stack.server, stack.factory
        stack.bridge.attach(worker)
        worker.channel.ready = True
        boot.send({'rank': rank, 'pid': os.getpid(), 'ready': True,
                   'runtime': type(stack.bridge).__module__})
        rows, repeat, cached, pregrant = [], {}, 0, None
        selection_pending = False
        telemetry = {'rank': rank, 'pid': os.getpid(), 'profile_polls': 0,
                     'cached_retries': 0, 'pregrant_retries': 0, 'reply_count': 0}
        with (Path(out) / f'worker-{rank}.jsonl').open('a') as journal:
            while True:
                if not recv.poll(120):
                    raise TimeoutError('CPU coordinator stopped sending')
                envelope = pickle.loads(recv.recv_bytes())
                with count.get_lock():
                    count.value -= 1
                method, args, kwargs, output_rank = envelope
                if method == '__proof_exit__':
                    break
                # Device completion control is a declared leaf; driver and
                # bridge methods/flags are never patched.
                worker.bus.complete_all(True)
                stack.bus.complete_all(True)
                message = args[0] if args else None
                is_profile = isinstance(message, w.QsaProfileQuery)
                active = getattr(message, 'stage', None) == stage
                should_hold = (selection_pending and active and
                               controls['hold_stage'].value == stage and
                               controls['hold_rank'].value == rank)
                if not should_hold:
                    stack.device.ready_all()
                if is_profile:
                    telemetry['profile_polls'] += 1
                if method == '__proof_stats__':
                    value = dict(telemetry, operation_stats=dict(stack.operation.stats) if stack.operation else {},
                        accepted=len(stack.dispatcher.accepted), owned=stack.dispatcher.transaction is not None,
                        pp_sends=worker.channel.sends, pp_tickets=[x.state for x in worker.channel.tickets])
                else:
                    key = (getattr(getattr(message, 'descriptor', None), 'transaction', None),
                           getattr(message, 'op_seq', None), method)
                    before_stats = dict(stack.operation.stats) if stack.operation else {}
                    accepted_before = len(stack.dispatcher.accepted)
                    value = getattr(worker, method)(*args, **kwargs)
                    if isinstance(value, w.SegmentAcceptance) and value.profile_boundary:
                        selection_pending = True
                        if active and controls['retry_rank'].value == rank and pregrant is None:
                            pregrant = stack.operation.stats.copy()
                            for _ in range(1100):
                                try:
                                    stack.operation.advance()
                                except et.E3Pending:
                                    pass
                                else:
                                    raise AssertionError('bucket submitted before profile grant')
                            assert pregrant == stack.operation.stats
                            telemetry['pregrant_retries'] = 1100
                        if active and controls['retry_rank'].value == rank:
                            # Exact worker-side retry of the accepted boundary
                            # while its selection copy is still outstanding.
                            # The real bridge/dispatcher handles the duplicate.
                            before_retry = dict(stack.operation.stats)
                            accepted_before_retry = len(stack.dispatcher.accepted)
                            cached = worker.flash_s5_submit(message)
                            assert dataclasses.replace(cached, report=None) == dataclasses.replace(value, report=None)
                            assert len(stack.dispatcher.accepted) == accepted_before_retry
                            assert dict(stack.operation.stats) == before_retry
                            telemetry['cached_retries'] += 1
                    if isinstance(value, w.SegmentAcceptance) and value.accepted:
                        if key in repeat:
                            # The bridge intentionally refreshes its queried
                            # completion report; only cached acceptance fields
                            # are identical. This is not a frozen report claim.
                            assert dataclasses.replace(value, report=None) == repeat[key], 'cached acceptance changed'
                            assert accepted_before == len(stack.dispatcher.accepted)
                            assert before_stats == (dict(stack.operation.stats) if stack.operation else {})
                            telemetry['cached_retries'] += 1
                        repeat[key] = dataclasses.replace(value, report=None)
                    if getattr(message, 'profile', None) is not None:
                        selection_pending = False
                telemetry['reply_count'] += 1
                event = {'method': method, 'type': type(message).__name__, 'stage': getattr(message, 'stage', None),
                         'op_seq': getattr(message, 'op_seq', None), 'reply_type': type(value).__name__,
                         'accepted': getattr(value, 'accepted', None), 'pending': getattr(value, 'pending', None),
                         'profile_boundary': getattr(value, 'profile_boundary', None),
                         'tail_boundary': getattr(value, 'tail_boundary', None),
                         'cursor': repr(getattr(value, 'cursor', None)), 'stats': dict(telemetry)}
                journal.write(json.dumps(event) + '\n'); journal.flush()
                if controls['drop_rank'].value == rank and is_profile:
                    continue
                if controls['delay_rank'].value == rank:
                    time.sleep(0.003)
                if output_rank is None or output_rank == rank:
                    reply.send_bytes(pickle.dumps((WorkerProc.ResponseStatus.SUCCESS, value),
                                                   protocol=pickle.HIGHEST_PROTOCOL))
        write(Path(out) / f'worker-{rank}-terminal.json', telemetry)
        case.doCleanups()
    except BaseException as exc:
        write(Path(out) / f'worker-{rank}-exception.json', {'exception_graph': exception_graph(exc)})
        try:
            boot.send({'rank': rank, 'ready': False, 'exception_graph': exception_graph(exc)})
        except Exception:
            pass
        raise


class Network:
    def __init__(self, scheduler, controller, admission, graph, out, variant):
        import executor_hooks as eh
        import engine_driver as ed
        from vllm.v1.executor.multiproc_executor import WorkerProc
        self.ctx = mp.get_context('spawn')
        self.stopped = False
        self.processes, self.senders, self.counts = [], [], []
        NETWORKS.append(self)
        self.controls = {n: self.ctx.Value('i', -1) for n in
                         ('hold_stage', 'hold_rank', 'retry_rank', 'delay_rank', 'drop_rank')}
        self.controls['retry_rank'].value = variant.get('ready_rank', 0)
        self.controls['hold_rank'].value = variant.get('pending_rank', 1)
        self.controls['hold_stage'].value = variant.get('hold_stage', 0)
        self.controls['delay_rank'].value = variant.get('delay_rank', 3)
        self.controls['drop_rank'].value = variant.get('drop_rank', -1)
        pp_slots = [self.ctx.Array('B', 1024) for _ in range(4)]
        senders, counts = self.senders, self.counts
        readers, boots = [], []
        for rank in range(4):
            child_recv, parent_send = self.ctx.Pipe(duplex=False)
            parent_recv, child_send = self.ctx.Pipe(duplex=False)
            boot_recv, boot_send = self.ctx.Pipe(duplex=False)
            count = self.ctx.Value('i', 0)
            process = self.ctx.Process(target=child_worker, args=(rank, rank // 2, admission,
                child_recv, child_send, count, self.controls, pp_slots, boot_send, graph, str(out), variant))
            process.start()
            self.processes.append(process); senders.append(parent_send)
            readers.append(ReplyReader(parent_recv)); counts.append(count); boots.append(boot_recv)
        self.boots = []
        for boot in boots:
            assert boot.poll(180), 'worker bootstrap timeout'
            row = boot.recv(); self.boots.append(row)
            assert row['ready'], row
        assert len({row['pid'] for row in self.boots}) == 4
        self.broadcast = ProcessBroadcast(senders, counts)
        self.executor = SimpleNamespace(response_mqs=readers, rpc_broadcast_mq=self.broadcast,
            futures_queue=[], is_failed=False)
        self.transport = eh.install(self.executor, config=SimpleNamespace(enabled=True, command_slots=10),
                                    success_status=WorkerProc.ResponseStatus.SUCCESS)
        self.engine = SimpleNamespace(scheduler=scheduler)
        self.clock_offset = [0.]
        def proof_clock():
            return time.monotonic() + self.clock_offset[0]
        self.driver = ed.NativeSegmentDriver(controller, executor=self.executor, scheduler=scheduler,
            engine=self.engine, layer_count=4, layer_split=2,
            stage_of=((0, 0), (1, 0), (2, 1), (3, 1)),
            reply_timeout_s=variant.get('reply_timeout_s', 5.), stop_budget=64,
            clock=proof_clock).install()
        self.high_water = 0

    def stats(self):
        import executor_hooks as eh
        f = eh.collective_rpc(self.executor, None, '__proof_stats__', non_block=True, timeout=5.)
        assert isinstance(f, eh.LocalFuture)
        return f.result(timeout=5.)

    def stop(self, failed=False):
        # Exit proof processes only; no engine/service termination or deletion.
        if self.stopped:
            return
        self.stopped = True
        payload = pickle.dumps(('__proof_exit__', (), {}, None))
        for process, sender, count in zip(self.processes, self.senders, self.counts):
            if process.is_alive():
                try:
                    with count.get_lock():count.value += 1
                    sender.send_bytes(payload)
                except (OSError, EOFError):
                    pass
        for process in self.processes:
            process.join(20)
            if not failed:
                assert process.exitcode == 0, ('proof endpoint exit', process.pid, process.exitcode)


def run_scenario(cfg, scheduler, leases, owner, cache_record, *, graph, out, variant):
    from e3_native_cow_lease_probe import engine_request, host_preprocess
    import scheduler_hooks as sh
    import admission_metadata as am
    import async_handoff as ah
    import executor_hooks as eh
    from vllm.v1.outputs import ModelRunnerOutput
    request = host_preprocess(owner, engine_request('coordinator-first', [1] * 65, 'first', 4))
    scheduler.add_request(request)
    controller = sh.PrefillAdmissionController(scheduler, leases=leases, frontier=ah.CommittedFrontier(),
        engine_epoch=7, chunk_tokens=64, ranks=(0, 1, 2, 3), minimum_stage_rows=64).attach()
    am.install_admission_metadata(scheduler)
    admission = controller.admit(scheduler)
    assert admission is not None
    network = Network(scheduler, controller, admission, graph, out, variant)
    driver = network.driver
    outputs, polls, last, stop_sent = [], 0, None, False
    deadline = time.monotonic() + (360 if variant.get('paged_cancel_refill') else 120)
    with (out / 'coordinator.jsonl').open('a') as journal:
        while time.monotonic() < deadline:
            if driver.phase == 'profile':
                polls += 1
                if (variant.get('virtual_profile_deadline')
                        and sum(row['profile_polls'] for row in
                                driver.qsa_control.snapshot()['layers']) >= 3):
                    # Injectable clock leaf advances beyond the *same* absolute
                    # driver deadline after real typed pending observations.
                    # RPC reply time stays sufficient for the OS processes.
                    network.clock_offset[0] = driver.reply_timeout_s + 1.
                if polls > 40 and not variant.get('hold_until_timeout'):
                    network.controls['hold_rank'].value = -1
            stop_stage = variant.get('stop_at_profile_stage')
            if (not stop_sent and stop_stage is not None and driver._profile_query is not None
                    and driver._profile_query.stage == stop_stage):
                if stop_stage == 1:
                    obligation = driver._pp_submission
                    assert obligation is not None
                    assert obligation.send_accepted == {0, 1}
                    assert obligation.receive_accepted == {2, 3}
                driver.stop('abort')
                stop_sent = True
            if (not stop_sent and variant.get('disconnect_at_report_poll')
                    and driver.phase == 'finalize'
                    and 'poll' in driver._inflight.values()):
                driver.stop('disconnect')
                stop_sent = True
            if (variant.get('shutdown_at_profile') and driver.phase == 'profile'
                    and 'profile' in driver._inflight.values()):
                result_box = []
                thread = threading.Thread(target=lambda: result_box.append(
                    eh.before_shutdown(network.executor)), name='stock-monitor-shutdown')
                thread.start(); thread.join(5.)
                assert not thread.is_alive() and len(result_box) == 1
                clean, close = result_box[0]
                transport = network.transport
                assert not clean and close['outstanding_calls'] >= 1
                assert all(q._flash_s5_receiver_owner is transport
                           for q in transport._queues.values())
                network.stop()
                return dict(verdict='PASS', kind='monitor_thread_shutdown',
                    monitor_thread=thread.name, actual_profile_rpc_outstanding=True,
                    transport_close=close, ownership_retained=True,
                    next_scheduler_request='not-attempted:executor-is-shutting-down',
                    control_path=driver.control_path.snapshot(),
                    qsa_control=driver.qsa_control.snapshot(), box_evidence=False)
            try:
                result = eh.before_step_with_batch_queue(network.engine)
            except eh.TransportError as exc:
                if not variant.get('missing_reply') or 'reply_timeout' not in str(exc):
                    raise
                clean, close = eh.before_shutdown(network.executor)
                transport = network.transport
                assert not clean and close['failure'] == 'reply_timeout'
                assert close['outstanding_calls'] >= 1
                assert all(q._flash_s5_receiver_owner is transport
                           for q in transport._queues.values())
                network.stop()
                return dict(verdict='PASS', kind='missing_reply',
                    transport_failure=str(exc), transport_close=close,
                    ownership_retained=True, fabricated_completion=False,
                    next_scheduler_request='not-attempted:transport-poisoned',
                    control_path=driver.control_path.snapshot(),
                    qsa_control=driver.qsa_control.snapshot(), box_evidence=False)
            if result is not None and result[0]:
                outputs.append(result)
            network.high_water = max(network.high_water, len(network.transport._calls))
            state = (driver.phase, driver.index, tuple(driver._inflight), driver.stop_kind)
            if state != last:
                journal.write(json.dumps({'state': state, 'qsa_control': driver.qsa_control.snapshot(),
                    'control_path': driver.control_path.snapshot()}, default=str) + '\n');journal.flush()
                last = state
            if not driver.active and not driver._inflight:
                break
            time.sleep(0.001)
        else:
            raise TimeoutError('connected coordinator deadline')
    workers = network.stats()
    assert driver.error is None
    assert stop_sent == bool(variant.get('stop_at_profile_stage') is not None
                             or variant.get('disconnect_at_report_poll'))
    expected_refusal = variant.get('expected_refusal')
    errors = [item.stop_reason for result in outputs for batch in result[0].values()
              for item in batch.outputs if item.finish_reason is not None]
    if stop_sent:
        from vllm.v1.request import RequestStatus
        assert not errors, ('explicit abort emitted a profile refusal', errors)
        assert driver._last_native_closure['complete']
        assert not driver._split_ready
        scheduler.finish_requests([request.request_id], RequestStatus.FINISHED_ABORTED)
        assert request.is_finished()
        if variant.get('paged_cancel_refill'):
            from native_measurement import JOURNAL_VECTOR_SCHEMA
            cancel_closure = driver._last_native_closure
            paged = [entry for entry in driver._journal if entry['kind'] == 'report'
                     and any(type(entry.get(key)) is dict and
                             entry[key].get('schema') == JOURNAL_VECTOR_SCHEMA
                             for key in ('completed_op_seqs', 'pending_op_seqs', 'proofs'))]
            assert paged, 'cancel did not consume an actual paged worker report'
            refill = host_preprocess(owner, engine_request(
                'coordinator-refill', [7] * 65, 'refill', 4))
            scheduler.add_request(refill)
            next_admission = controller.admit(scheduler)
            assert next_admission is not None and next_admission.transaction != admission.transaction
            refill_deadline = time.monotonic() + 360
            while time.monotonic() < refill_deadline:
                result = eh.before_step_with_batch_queue(network.engine)
                if result is not None and result[0]:
                    outputs.append(result)
                if not driver.active and not driver._inflight:
                    break
                time.sleep(0.001)
            else:
                raise TimeoutError('native refill did not close')
            commits = [entry for entry in driver._journal if entry['kind'] == 'chunk_commit'
                       and entry['transaction'] == next_admission.transaction.txn_seq]
            assert driver.error is None and len(commits) == 1 and commits[0]['terminal']
            assert not driver._split_ready and leases.shutdown()['clean']
            variant['paged_cancel_proof'] = dict(paged_reports=len(paged),
                max_report_length=max(entry[key]['length'] for entry in paged
                    for key in ('completed_op_seqs', 'pending_op_seqs', 'proofs')
                    if type(entry.get(key)) is dict),
                cancel_closure=cancel_closure, refill_commit=commits[0],
                refill_transaction=next_admission.transaction.txn_seq)
    elif expected_refusal:
        assert errors == [expected_refusal], ('client errors', errors, expected_refusal)
        assert driver._last_native_closure['complete']
        assert not driver._split_ready
        assert driver._profile_mismatches == int(expected_refusal == 'unsupported-native-schedule-profile')
        assert driver._profile_timeouts == int(expected_refusal == 'native-schedule-profile-timeout')
        assert request.is_finished(), 'refused request was not retired'
    else:
        assert not errors, ('unexpected client error', errors)
        layers = driver.qsa_control.snapshot()['layers']
        assert sum(row['tail_boundaries'] for row in layers) == 2
        assert sum(row['tail_accepted_commands'] for row in layers) == 2
    assert leases.shutdown()['clean'] and not network.transport._calls, {
        'leases': leases.shutdown(), 'calls': list(network.transport._calls),
        'driver': driver.status()}
    # The actual ordinary scheduler advances the final prompt token and then
    # four tokens through an explicitly synthetic sampling-output leaf.
    if not expected_refusal and not stop_sent:
        for tick in range(40):
            result = scheduler.schedule(False)
            ids = list(result.num_scheduled_tokens)
            tokens = [[17] if scheduler.requests[k].num_computed_tokens >= scheduler.requests[k].num_prompt_tokens else [] for k in ids]
            scheduler.update_from_output(result, ModelRunnerOutput(req_ids=ids,
                req_id_to_index={k: i for i, k in enumerate(ids)}, sampled_token_ids=tokens))
            if request.is_finished():break
        assert request.is_finished()
    successor = host_preprocess(owner, engine_request('coordinator-next', [2, 3, 4], 'next', 4))
    scheduler.add_request(successor)
    for tick in range(40):
        result = scheduler.schedule(False); ids = list(result.num_scheduled_tokens)
        tokens = [[17] if scheduler.requests[k].num_computed_tokens >= scheduler.requests[k].num_prompt_tokens else [] for k in ids]
        scheduler.update_from_output(result, ModelRunnerOutput(req_ids=ids,
            req_id_to_index={k: i for i, k in enumerate(ids)}, sampled_token_ids=tokens))
        if successor.is_finished():break
    assert successor.is_finished() and successor.num_output_tokens == 4
    clean, close = eh.before_shutdown(network.executor)
    assert clean, close
    network.stop()
    control = driver.control_path.snapshot()
    per_operation = {key: (control[key] / control['ops_count'] if control['ops_count'] else None)
                     for key in ('prepare_ns', 'poll_ns', 'validate_ns', 'serialize_ns')}
    return {'verdict': 'PASS', 'pids': network.boots, 'workers': workers,
        'qsa_control': driver.qsa_control.snapshot(), 'control_path': control,
        'control_per_operation_ns': per_operation,
        'credit_high_water': network.high_water, 'credits_returned': True,
        'client_errors': errors,
        'native_closure': driver._last_native_closure,
        'profile_mismatches': driver._profile_mismatches,
        'profile_timeouts': driver._profile_timeouts,
        'virtual_clock_advanced': bool(network.clock_offset[0]),
        'explicit_abort_stage': variant.get('stop_at_profile_stage') if stop_sent else None,
        'report_phase_disconnect': bool(stop_sent and variant.get('disconnect_at_report_poll')),
        'paged_cancel_refill': variant.get('paged_cancel_proof'),
        'next_scheduler_request_completed': True, 'next_tokens': successor.num_output_tokens,
        'transport_close': close, 'cache': cache_record,
        'actual_scheduler': type(scheduler).__module__ + '.' + type(scheduler).__name__}


def prefix_hit_under_lease(cfg, scheduler, leases, owner, cache_record, *, partial,
                           graph, out):
    """Real stock prefix lookup against a parked native request's held pages."""
    from e3_native_cow_lease_probe import engine_request, host_preprocess
    import scheduler_hooks as sh
    import admission_metadata as am
    import async_handoff as ah
    import executor_hooks as eh
    from vllm.v1.outputs import ModelRunnerOutput
    # The pinned hybrid manager's Mamba groups require a complete 4048-row
    # page before the shared prefix is cacheable in every group.
    seed_tokens = [1 + i % 100 for i in range(4048)]
    salt = 'coordinator-shared-prefix'
    seed = host_preprocess(owner, engine_request('prefix-seed', seed_tokens + [151], salt, 4))
    scheduler.add_request(seed)
    for _ in range(40):
        output = scheduler.schedule(False)
        ids = list(output.num_scheduled_tokens)
        sampled = [[17] if scheduler.requests[key].num_computed_tokens >=
                   scheduler.requests[key].num_prompt_tokens else [] for key in ids]
        scheduler.update_from_output(output, ModelRunnerOutput(
            req_ids=ids, req_id_to_index={key: i for i, key in enumerate(ids)},
            sampled_token_ids=sampled))
        if seed.is_finished():
            break
    assert seed.is_finished() and seed.num_output_tokens == 4
    native = host_preprocess(owner, engine_request(
        'prefix-native', seed_tokens + [152] * 65, salt, 4))
    scheduler.add_request(native)
    controller = sh.PrefillAdmissionController(scheduler, leases=leases,
        frontier=ah.CommittedFrontier(), engine_epoch=7, chunk_tokens=64,
        ranks=(0, 1, 2, 3), minimum_stage_rows=64).attach()
    am.install_admission_metadata(scheduler)
    admission = controller.admit(scheduler)
    assert admission is not None and admission.committed_end >= 176
    assert leases._held and not leases.shutdown()['clean']
    probe_tokens = seed_tokens + [153] * 25 if partial else seed_tokens + [153]
    probe = host_preprocess(owner, engine_request(
        'prefix-partial-hit' if partial else 'prefix-full-hit', probe_tokens, salt, 4))
    cached, computed, _ = scheduler.kv_cache_manager.get_computed_blocks(probe)
    cached_ids = sorted({getattr(block, 'block_id', None)
                         for group in cached.blocks for block in group})
    overlap = sorted(set(cached_ids) & set(leases.held_blocks()))
    assert computed > 0 and overlap, ('stock prefix lookup did not touch leased pages',
                                       computed, cached_ids, leases.held_blocks())
    scheduler.add_request(probe)
    unrelated = host_preprocess(owner, engine_request(
        'prefix-unrelated', [27] * 65, 'unrelated', 4))
    scheduler.add_request(unrelated)
    before = dict(held=leases.held_blocks(), parked=controller.parked.transaction,
                  seed_hashes=len(seed.block_hashes), native_start=admission.committed_end,
                  probe_hashes=len(probe.block_hashes), cache_hit_tokens=computed,
                  cached_ids=cached_ids, held_overlap=overlap)
    try:
        output = scheduler.schedule(False)
    except BaseException as exc:
        # A stock cache touch that raises out of schedule is a real engine
        # finding. Keep the exact exception chain and held owner state.
        return dict(verdict='PASS', finding='ENGINE', hit='partial' if partial else 'full',
                    before=before, exception_graph=exception_graph(exc),
                    held_after=leases.held_blocks(), owner_retained=controller.parked is not None,
                    actual_scheduler=type(scheduler).__module__ + '.' + type(scheduler).__name__,
                    next_request='not-attempted:real-scheduler-raised')
    assert probe.request_id not in output.num_scheduled_tokens
    assert unrelated.request_id in output.num_scheduled_tokens
    assert leases.stock_prefix_deferrals == 1
    assert probe.num_computed_tokens == 0 and not scheduler.kv_cache_manager.get_block_ids(
        probe.request_id)[0]
    network = Network(scheduler, controller, admission, graph, out,
                      dict(ready_rank=0, pending_rank=1, delay_rank=3))
    # This scenario targets stock prefix conflict and native commit, not a
    # long selection-copy pending interval. The worker still queries its real
    # profile; the declared CPU event leaf is ready on its first poll.
    network.controls['hold_rank'].value = -1
    driver = network.driver
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        eh.before_step_with_batch_queue(network.engine)
        if not driver.active and not driver._inflight:
            break
        time.sleep(.001)
    else:
        raise TimeoutError('native prefix-holder did not complete')
    commits = [e for e in driver._journal if e['kind'] == 'chunk_commit'
               and e['transaction'] == admission.transaction.txn_seq]
    assert len(commits) == 1 and commits[0]['terminal']
    assert not driver._split_ready and leases.shutdown()['clean']
    after_cached, after_count, _ = scheduler.kv_cache_manager.get_computed_blocks(probe)
    assert after_count == computed and after_cached.get_block_ids() == cached.get_block_ids()
    first_probe_schedule = None
    pending = output
    for _ in range(80):
        ids = list(pending.num_scheduled_tokens)
        if probe.request_id in pending.num_scheduled_tokens and first_probe_schedule is None:
            first_probe_schedule = pending.num_scheduled_tokens[probe.request_id]
        sampled = [[17] if scheduler.requests[key].num_computed_tokens >=
                   scheduler.requests[key].num_prompt_tokens else [] for key in ids]
        scheduler.update_from_output(pending, ModelRunnerOutput(
            req_ids=ids, req_id_to_index={key: i for i, key in enumerate(ids)},
            sampled_token_ids=sampled))
        if all(r.is_finished() for r in (probe, unrelated, native)):
            break
        pending = scheduler.schedule(False)
    assert all(r.is_finished() and r.num_output_tokens == 4
               for r in (probe, unrelated, native))
    assert first_probe_schedule == len(probe_tokens) - computed
    clean, close = eh.before_shutdown(network.executor)
    assert clean and close['outstanding_calls'] == 0
    network.stop()
    assert driver._status_record()['stock_prefix_deferrals'] == leases.stock_prefix_deferrals
    return dict(verdict='PASS', finding='NONE', hit='partial' if partial else 'full',
                before=before, first_scheduled=dict(output.num_scheduled_tokens),
                native_commit=commits[0], stock_prefix_deferrals=leases.stock_prefix_deferrals,
                cached_hit_after_release=after_count,
                first_probe_scheduled_tokens=first_probe_schedule,
                all_requests_completed=True, next_scheduler_tokens=probe.num_output_tokens,
                transport_close=close, actual_scheduler=type(scheduler).__module__ + '.' + type(scheduler).__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--graph', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--mode', default='cpu-full')
    p.add_argument('--scenario', default=None)
    args = p.parse_args()
    result = {'verdict': 'INCONCLUSIVE', 'rc': 2, 'failure_class': 'OPEN', 'sections': {},
              'box_evidence': False, 'gpu_execution': False}
    try:
        bootstrap(args.graph)
        from native18_scheduler_fixture import with_actual_scheduler
        variants = (
            ('success_left_ready', {'ready_rank': 0, 'pending_rank': 1, 'delay_rank': 3}),
            ('success_right_ready', {'ready_rank': 1, 'pending_rank': 0, 'delay_rank': 2}),
            ('stage0_mismatch', {'ready_rank': 0, 'pending_rank': 1, 'delay_rank': 3,
                 'mismatch_stage': 0, 'expected_refusal': 'unsupported-native-schedule-profile'}),
            ('stage0_timeout', {'ready_rank': 0, 'pending_rank': 1, 'delay_rank': 3,
                 'reply_timeout_s': 5., 'hold_until_timeout': True,
                 'virtual_profile_deadline': True,
                 'expected_refusal': 'native-schedule-profile-timeout'}),
            ('stage0_abort', {'ready_rank': 0, 'pending_rank': 1, 'delay_rank': 3,
                 'stop_at_profile_stage': 0}),
            ('stage1_abort_after_sender', {'ready_rank': 2, 'pending_rank': 3,
                 'hold_stage': 1, 'delay_rank': 0, 'stop_at_profile_stage': 1}),
            ('report_phase_disconnect', {'ready_rank': 0, 'pending_rank': 1,
                 'delay_rank': 3, 'disconnect_at_report_poll': True}),
            ('paged_report_cancel_refill', {'ready_rank': 0, 'pending_rank': 1,
                 'delay_rank': 3, 'disconnect_at_report_poll': True,
                 'paged_cancel_refill': True, 'selection_count': 16385}),
            ('monitor_shutdown_profile', {'ready_rank': 0, 'pending_rank': 1,
                 'delay_rank': 3, 'shutdown_at_profile': True}),
            ('missing_profile_reply', {'ready_rank': 0, 'pending_rank': 1,
                 'delay_rank': 3, 'missing_reply': True, 'drop_rank': 1}),
            ('prefix_full_hit_held', {'prefix_hit': 'full'}),
            ('prefix_partial_hit_held', {'prefix_hit': 'partial'}),
        )
        if args.scenario is not None:
            matches = [(name, variant) for name, variant in variants if name == args.scenario]
            assert len(matches) == 1, 'unknown connected scenario'
            name, variant = matches[0]
            if variant.get('prefix_hit'):
                result['sections'][name] = with_actual_scheduler(lambda *a:
                    prefix_hit_under_lease(*a, partial=variant['prefix_hit'] == 'partial',
                                           graph=str(args.graph), out=args.out.parent),
                    max_model_len=8192)
            else:
                result['sections'][name] = with_actual_scheduler(lambda *a:
                    run_scenario(*a, graph=str(args.graph), out=args.out.parent, variant=variant))
            import torch
            assert not torch.cuda.is_initialized()
            finding = result['sections'][name].get('finding')
            result.update(verdict='FAIL' if finding == 'ENGINE' else 'PASS',
                          rc=1 if finding == 'ENGINE' else 0,
                          failure_class='ENGINE' if finding == 'ENGINE' else 'NONE',
                          cuda_initialized=False)
        else:
            # The stock Gloo/model-parallel owner is process-global. Each
            # scenario gets its own real initialization in a fresh interpreter,
            # not a manually reset global or a patched fixture guard.
            for name, _ in variants:
                scenario_out = args.out.parent / name
                scenario_out.mkdir(exist_ok=False)
                with (scenario_out / 'process.log').open('xb') as log:
                    completed = subprocess.run([sys.executable, '-B', str(Path(__file__)),
                        '--graph', str(args.graph), '--out', str(scenario_out / 'probe.json'),
                        '--mode', args.mode, '--scenario', name],
                        stdout=log, stderr=subprocess.STDOUT, timeout=900)
                scenario = json.loads((scenario_out / 'probe.json').read_text())
                result['sections'][name] = scenario
                write(args.out, result)
                if scenario['failure_class'] == 'ENGINE':
                    result.update(verdict='FAIL', rc=1, failure_class='ENGINE',
                                  engine_finding_scenarios=result.get(
                                      'engine_finding_scenarios', []) + [name])
                    continue
                assert completed.returncode == 0 and scenario['verdict'] == 'PASS', (
                    'connected scenario failed', name, completed.returncode)
            if result['failure_class'] != 'ENGINE':
                result.update(verdict='PASS', rc=0, failure_class='NONE', cuda_initialized=False)
    except BaseException as exc:
        result.update(verdict='FAIL', rc=1, failure_class='OPEN', exception_graph=exception_graph(exc))
    finally:
        for network in NETWORKS:
            try:network.stop(failed=result['rc'] != 0)
            except BaseException as exc:
                result.setdefault('cleanup_exceptions', []).append(exception_graph(exc))
                result.update(verdict='FAIL', rc=1)
    write(args.out, result)
    print(json.dumps({k: result[k] for k in ('verdict', 'rc', 'failure_class')}), flush=True)
    return result['rc']


if __name__ == '__main__':
    raise SystemExit(main())
