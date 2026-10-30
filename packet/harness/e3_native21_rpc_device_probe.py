#!/usr/bin/env python3
"""One-card check of the QSA worker RPC thread against a real CUDA device.

The accepted layer boundary is seeded because this probe has no model forward.
The operation, CUDA stream guard, layer driver, dispatcher type and worker
bridge poll are the mounted production classes. This is a stream-contract
proof, not a row1, scheduler, TP/PP or release qualification.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import threading
import time
import traceback
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def modules():
    from vllm.v1.flash_e3 import e3_types, lane_dispatch, qsa_driver, qsa_operation, wire_v1, worker_bridge
    return e3_types, lane_dispatch, qsa_driver, qsa_operation, wire_v1, worker_bridge


def cpu_shape():
    et, ld, qd, qo, wire, wb = modules()
    expected = (qo.NativeQsaDevice, qo.QsaPrefillOperation, qd.QsaLayerDriver,
                ld.LaneDispatcher, wb.NativeWorkerBridge, wire.QsaProfileQuery)
    assert all(inspect.isclass(item) for item in expected)
    assert callable(qo.NativeQsaDevice.validate_compute)
    assert callable(qo.QsaPrefillOperation.schedule_profile)
    assert callable(qo.QsaPrefillOperation.submission_credit)
    assert callable(wb.NativeWorkerBridge.poll)
    # Access the exact attributes consumed in the GPU predicate. The pinned
    # image's CUDA constructor and thread-local stream check run on the VM.
    source = inspect.getsource(qo.QsaPrefillOperation.schedule_profile)
    assert 'compute_required=False' in source
    assert 'compute_required=False' in inspect.getsource(qo.QsaPrefillOperation.submission_credit)
    assert 'validate_compute' in inspect.getsource(qo.QsaPrefillOperation._check_identity)
    assert 'self.dispatcher.driver.schedule_profile' in inspect.getsource(wb.NativeWorkerBridge.poll)
    assert 'self.operation.schedule_profile()' in inspect.getsource(qd.QsaLayerDriver.schedule_profile)
    return dict(verdict='PASS', actual_mounted_classes=True,
                constructors_accessed=len(expected), stream_guard_accessed=True,
                bridge_poll_accessed=True, cuda_execution=False)


def gpu_case():
    import torch
    et, ld, qd, qo, wire, wb = modules()
    assert torch.cuda.device_count() == 1
    device_name = 'cuda:0'
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., 2 * 1024**3 / total), 0)
    compute = torch.cuda.Stream(device=device_name)
    copy = torch.cuda.Stream(device=device_name)
    device = qo.NativeQsaDevice(device_name)
    device.bind_streams(compute, copy)
    assert compute.cuda_stream != copy.cuda_stream
    transaction = et.TransactionId(1, 'native21-rpc-stream', 1, 0, 64, 1)
    lane = et.LaneId(0, et.LaneRole.PREFILL, 1, 1)
    operation = qo.QsaPrefillOperation(transaction, lane, device=device,
                                      plan_provider=lambda _name, count: tuple(range(count)))
    source = torch.empty((1, 1, 4048, 512), dtype=torch.bfloat16, pin_memory=True)
    arena = torch.empty((24, 1, 4048, 512), dtype=torch.bfloat16, device=device_name)
    table = torch.zeros((1,), dtype=torch.int32, device=device_name)
    with torch.cuda.stream(compute):
        operation.register_source('qsa.0', source)
        operation.begin_forward(layers=('qsa.0',))
        operation.begin_layer('qsa.0')
        operation.producer_submitted('qsa.0', source)
        operation.bind_plan('qsa.0', table, 1, 4048, arena)
        boundary = operation.schedule_boundary()
        device.validate_compute()
    phase = qd.LayerPhase(0, None, None, None)
    phase.boundary = boundary
    driver = qd.QsaLayerDriver(operation, staged_operation=lambda *_: None,
                              forward_context=lambda: None)
    driver.phases[0] = phase
    native = SimpleNamespace(split=SimpleNamespace(prefill_stream=compute))
    dispatcher = ld.LaneDispatcher('native21-vm-probe', native, None, torch_module=torch)
    dispatcher.transaction = transaction
    dispatcher.driver = driver
    command = ld.SegmentCommand(transaction, 0, 'layer', 1, layer_start=0, layer_end=1)
    # This is the declared fixture leaf: the boundary was submitted by the
    # real operation but the model-free probe supplies its dispatcher ACK.
    dispatcher.accepted[1] = (command, {'result': {'profile_boundary': True}})
    bridge = wb.NativeWorkerBridge(dispatcher, rank=0)
    bridge.transaction = transaction
    bridge.state_epoch = 1
    bridge.stage = 0
    query = wire.QsaProfileQuery(transaction, 1, 0, 0, 1, (0,))

    answer = {}
    def rpc_thread():
        try:
            answer['thread_id'] = threading.get_ident()
            answer['current_stream'] = torch.cuda.current_stream(device_name).cuda_stream
            answer['observation'] = bridge.poll(query)
            answer['credit'] = operation.submission_credit()
            try:
                operation.advance()
            except qo.OperationError as exc:
                answer['advance_refusal'] = str(exc)
            try:
                operation.register_source('qsa.0', source)
            except qo.OperationError as exc:
                answer['bind_refusal'] = str(exc)
        except BaseException as exc:
            answer['error'] = repr(exc)
            answer['trace'] = traceback.format_exc()
    thread = threading.Thread(target=rpc_thread, name='native21-worker-rpc')
    thread.start(); thread.join(timeout=20)
    assert not thread.is_alive(), 'RPC poll thread did not finish'
    if answer.get('error'):
        raise RuntimeError(answer['trace'])
    observation = answer['observation']
    assert answer['thread_id'] != threading.get_ident()
    assert answer['current_stream'] != compute.cuda_stream
    assert observation.status == 'ready'
    assert observation.basis == 'bound-dense-host-plan'
    assert observation.shape == wire.QsaScheduleShape('dense_dma', (1,), (1,))
    assert answer['credit'].can_submit is True
    refusal = 'operation called outside its owning compute stream'
    assert answer.get('advance_refusal') == refusal
    assert answer.get('bind_refusal') == refusal
    assert operation._failed is False and operation._released is False
    assert operation.stats['copy_events'] == 0 and operation.stats['buckets'] == 0
    with torch.cuda.stream(compute):
        assert operation.schedule_profile() == observation.shape
        device.validate_compute()
    compute.synchronize(); copy.synchronize()
    return dict(verdict='PASS', actual_native_qsa_device=True,
                actual_operation=True, actual_layer_driver=True,
                actual_dispatcher_type=True, actual_bridge_poll=True,
                bound_compute_stream=compute.cuda_stream,
                rpc_current_stream=answer['current_stream'],
                rpc_thread_distinct=True, observation_status=observation.status,
                observation_basis=observation.basis,
                shape_route=observation.shape.route,
                host_credit=True, compute_advance_refused=True,
                compute_source_bind_refused=True, kernel_or_dma_submitted=False,
                fixture_seeded_boundary=True, engine_constructed=False,
                gpu_execution=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('cpu-shape', 'gpu'), required=True)
    p.add_argument('--graph', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    start = time.monotonic()
    result = dict(schema='native21-rpc-device/1', verdict='FAIL',
                  mode=args.mode, failures=[], missing=[],
                  gpu_execution=False, engine_constructed=False)
    try:
        graph = json.loads(args.graph.read_text())
        assert graph['verdict'] == 'PASS'
        expected = graph['source_hashes']['e3/qsa_operation.py']
        assert sha(Path(modules()[3].__file__)) == expected, 'mounted QSA operation differs from import graph'
        section = cpu_shape() if args.mode == 'cpu-shape' else gpu_case()
        result['section'] = section
        result['gpu_execution'] = bool(section.get('gpu_execution', False))
        result['verdict'] = section['verdict']
    except BaseException as exc:
        result['failures'] = ['HARNESS:' + type(exc).__name__]
        result['exception'] = dict(type=type(exc).__name__, message=str(exc),
                                   trace=traceback.format_exc())
    result['elapsed_s'] = time.monotonic() - start
    result['rc'] = {'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 2}[result['verdict']]
    with args.out.open('x') as file:
        json.dump(result, file, indent=2, sort_keys=True)
        file.write('\n')
    print('NATIVE21_RPC_DEVICE', result['verdict'], flush=True)
    return result['rc']


if __name__ == '__main__':
    raise SystemExit(main())
