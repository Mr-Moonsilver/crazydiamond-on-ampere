#!/usr/bin/env python3
"""Connected native18 CPU coordinator proof; device/model/PP leaves declared.

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


def make_case():
    import test_e3_native_schedule_profile as proof
    class Mounted(proof.ScheduleProfile):
        generated_ops = Path('/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ops/qsa.py').read_text()
        generated_qsa = Path('/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/qsa.py').read_text()
    case = Mounted('runTest')
    case.selection_counts = (4097,) * 4
    case.setUp()
    return case


def child_worker(rank, stage, admission, recv, reply, count, controls, boot, graph, out, variant):
    try:
        bootstrap(graph)
        import test_e3_native_schedule_profile as proof
        import test_e3_native_submission_credit as credit
        import test_e3_r4_worker_bridge as r4
        import executor_hooks as eh
        import wire_v1 as w
        import e3_types as et
        from vllm.v1.executor.multiproc_executor import WorkerProc
        case = make_case()
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
                stack.bus.complete_all()
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
                    key = (getattr(message, 'op_seq', None), method)
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
        senders, counts = self.senders, self.counts
        readers, boots = [], []
        for rank in range(4):
            child_recv, parent_send = self.ctx.Pipe(duplex=False)
            parent_recv, child_send = self.ctx.Pipe(duplex=False)
            boot_recv, boot_send = self.ctx.Pipe(duplex=False)
            count = self.ctx.Value('i', 0)
            process = self.ctx.Process(target=child_worker, args=(rank, rank // 2, admission,
                child_recv, child_send, count, self.controls, boot_send, graph, str(out), variant))
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
        self.driver = ed.NativeSegmentDriver(controller, executor=self.executor, scheduler=scheduler,
            engine=self.engine, layer_count=4, layer_split=2,
            stage_of=((0, 0), (1, 0), (2, 1), (3, 1)), reply_timeout_s=5., stop_budget=64).install()
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
    outputs, polls, last = [], 0, None
    deadline = time.monotonic() + 120
    with (out / 'coordinator.jsonl').open('a') as journal:
        while time.monotonic() < deadline:
            if driver.phase == 'profile':
                polls += 1
                if polls > 40:
                    network.controls['hold_rank'].value = -1
            result = eh.before_step_with_batch_queue(network.engine)
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
    assert leases.shutdown()['clean'] and not network.transport._calls, {
        'leases': leases.shutdown(), 'calls': list(network.transport._calls),
        'driver': driver.status()}
    # The actual ordinary scheduler advances the final prompt token and then
    # four tokens through an explicitly synthetic sampling-output leaf.
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
    return {'verdict': 'PASS', 'pids': network.boots, 'workers': workers,
        'qsa_control': driver.qsa_control.snapshot(), 'control_path': driver.control_path.snapshot(),
        'credit_high_water': network.high_water, 'credits_returned': True,
        'next_scheduler_request_completed': True, 'next_tokens': successor.num_output_tokens,
        'transport_close': close, 'cache': cache_record,
        'actual_scheduler': type(scheduler).__module__ + '.' + type(scheduler).__name__}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--graph', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--mode', default='cpu-full')
    args = p.parse_args()
    result = {'verdict': 'INCONCLUSIVE', 'rc': 2, 'failure_class': 'OPEN', 'sections': {},
              'box_evidence': False, 'gpu_execution': False}
    try:
        bootstrap(args.graph)
        from native18_scheduler_fixture import with_actual_scheduler
        variant = {'ready_rank': 0, 'pending_rank': 1, 'delay_rank': 3}
        result['sections']['success_left_ready'] = with_actual_scheduler(lambda *a:
            run_scenario(*a, graph=str(args.graph), out=args.out.parent, variant=variant))
        import torch
        assert not torch.cuda.is_initialized()
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
