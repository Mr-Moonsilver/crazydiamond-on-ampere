#!/usr/bin/env python3
"""One-card proof of native-observer graph retention and real graph binding.

The CUDA case captures through the generated worker hook and binds the real
PyTorch CUDAGraph to the real replay observer. It does not construct a model,
multi-rank engine, or placement workload.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_check(graph):
    from vllm.v1.flash_e3 import worker_hooks
    actual = sha(worker_hooks.__file__)
    assert actual == graph['source_hashes']['e3/worker_hooks.py']
    return worker_hooks, actual


def cpu_shape(worker_hooks):
    calls = []
    class Graph:
        def __init__(self, keep_graph=False):
            self.keep_graph = keep_graph
        def instantiate(self):
            calls.append('instantiate')
        def raw_cuda_graph(self):
            if not self.keep_graph:
                raise RuntimeError('keep_graph required')
            return 19
    torch = NS(cuda=NS(CUDAGraph=lambda **kw: Graph(**kw)))
    env = {'FLASH_S5_DECODE_ONLY_PROBE':'0', 'FLASH_S5_OVERLAP':'1',
           'FLASH_S5_REPLAY_OBSERVER':'1'}
    prior = {key:os.environ.get(key) for key in env}
    try:
        os.environ.update(env)
        split = NS(_flash_s5_capture_context=lambda: None)
        retained = worker_hooks.capture_graph(split, torch_module=torch)
        worker_hooks.finish_graph(split, retained)
        assert retained.raw_cuda_graph() == 19
        assert split._flash_s5_keep_graphs is True
        ordinary = NS()
        unretained = worker_hooks.capture_graph(ordinary, torch_module=torch)
        worker_hooks.finish_graph(ordinary, unretained)
        assert not unretained.keep_graph
        try:
            unretained.raw_cuda_graph()
        except RuntimeError:
            pass
        else:
            raise AssertionError('ordinary graph unexpectedly exposes raw handle')
        os.environ['FLASH_S5_REPLAY_OBSERVER'] = '0'
        observer_off = worker_hooks.capture_graph(split, torch_module=torch)
        assert not observer_off.keep_graph
        assert calls == ['instantiate']
        return {'verdict':'PASS', 'retained_split_graph':True,
                'ordinary_graph_unchanged':True, 'observer_off_unchanged':True,
                'instantiate_once':True}
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def gpu_capture_and_bind(worker_hooks):
    import torch
    from vllm.v1.flash_e3 import green_context as gc
    from vllm.v1.flash_e3 import replay_observer as ro
    native = Path(worker_hooks.__file__).resolve().parent / 'native'
    session = split = None
    prior = {key:os.environ.get(key) for key in
             ('FLASH_S5_DECODE_ONLY_PROBE', 'FLASH_S5_OVERLAP',
              'FLASH_S5_REPLAY_OBSERVER')}
    try:
        os.environ.update({'FLASH_S5_DECODE_ONLY_PROBE':'0',
                           'FLASH_S5_OVERLAP':'1',
                           'FLASH_S5_REPLAY_OBSERVER':'1'})
        lib = native / 'flash_s5_replay_observer.so'
        source = native / 'flash_s5_replay_observer.cpp'
        binding = ro.ReplayBinding(
            str(lib), sha(lib), sha(source),
            'e2f9ed861fe27c492b8bb52b5e3220ef5120f3edcda36312e96b7fd8a186be3e',
            cupti_library_path='/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcupti.so.13')
        session = ro.ObserverSession(binding, ro.ObserverLimits())
        session.start()
        split = gc.open_split(0, env={'FLASH_S5_DECODE_SMS':'44'},
                              torch_module=torch,
                              binding=gc.DriverBinding(str(native / 'flash_s5_green.so')))
        split.retain_allocator_context()
        stream = split.decode_stream
        manager = NS(device=torch.device('cuda:0'),
                     _flash_s5_capture_context=lambda: None,
                     _flash_s5_capture_owner=split, graphs={})
        capture_desc = NS(cg_mode='FULL', num_tokens=4, num_reqs=1)
        x = torch.ones((1024,), device='cuda', dtype=torch.float32)
        y = torch.zeros_like(x)
        with torch.cuda.stream(stream):
            y.copy_(x)
        stream.synchronize()
        checked = worker_hooks.checked_capture_stream(
            manager, stream, capture_desc, torch_module=torch)
        assert checked is stream
        graph = worker_hooks.capture_graph(manager, torch_module=torch)
        with torch.cuda.graph(graph, stream=stream):
            y.add_(x)
        worker_hooks.finish_graph(manager, graph)
        raw = graph.raw_cuda_graph()
        assert type(raw) is int and raw > 0
        descriptor, local = object(), object()
        bank = NS(captured=True, pairs=[(descriptor, local)], handles={},
                  manager=manager)
        bank.handles[local] = NS(local=local, bank=bank, destroyed=False)
        manager.graphs[descriptor] = graph
        probe = NS(bank=bank, router=NS(storage_epoch=1, route_generation=1),
                   split=split, storage_epoch=1, route_generation=1,
                   route='split', interval='active')
        session.bind_bank(probe)
        assert session.state == 'bound' and graph.raw_cuda_graph() == raw
        checks = manager._flash_s5_capture_stream_checks
        assert len(checks) == 1 and manager._flash_s5_keep_graphs is True
        return {'verdict':'PASS', 'real_cuda_graph':True,
                'keep_graph':True, 'instantiated':True,
                'raw_handle_stable':True, 'real_observer_bound':True,
                'capture_stream_checks':len(checks),
                'gpu_execution':True}
    finally:
        if session is not None:
            session.close()
        if split is not None:
            split.close()
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--graph', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=('gpu', 'cpu-shape'), required=True)
    args = p.parse_args()
    result = {'schema':'native24-graph-capture-probe/1',
              'verdict':'FAIL', 'gpu_execution':False,
              'engine_constructed':False, 'sections':{}, 'failures':[]}
    try:
        import torch
        graph = json.loads(args.graph.read_text())
        hooks, digest = source_check(graph)
        result['worker_hooks_sha256'] = digest
        result['sections']['cpu_shape'] = cpu_shape(hooks)
        if args.mode == 'gpu':
            result['sections']['gpu_graph_bind'] = gpu_capture_and_bind(hooks)
            result['gpu_execution'] = True
        result['verdict'] = 'PASS'
    except Exception as exc:
        result['failures'].append(type(exc).__name__ + ': ' + str(exc))
        result['traceback'] = traceback.format_exc()
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print('NATIVE24_GRAPH ' + result['verdict'], flush=True)
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
