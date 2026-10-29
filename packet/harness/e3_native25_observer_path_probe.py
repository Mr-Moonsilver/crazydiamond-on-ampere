#!/usr/bin/env python3
"""One-card proof of the native observer's bound window and engine ledger.

The CUDA case captures through the generated worker hook and binds the real
PyTorch CUDAGraph to the real replay observer, then reads the native bank's
actual replay counters through NativeIdentity and ReplaySession, then drives
WorkerObserver.begin/end with its real CorrelationBridge and EngineOfferLedger.
It does not construct a model, multi-rank engine, or placement workload.
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
    from vllm.v1.flash_e3 import (worker_hooks, native_qualification,
                                 replay_observer, observer_runtime,
                                 engine_offers, activity_accounting)
    modules = dict(worker_hooks=worker_hooks,
                   native_qualification=native_qualification,
                   replay_observer=replay_observer,
                   observer_runtime=observer_runtime,
                   engine_offers=engine_offers,
                   activity_accounting=activity_accounting)
    digests = {name:sha(module.__file__) for name, module in modules.items()}
    for name, digest in digests.items():
        assert digest == graph['source_hashes']['e3/' + name + '.py']
    return worker_hooks, digests


def census_cpu_shape():
    from vllm.v1.flash_e3 import native_qualification as nq
    from vllm.v1.flash_e3 import replay_observer as ro
    from vllm.v1.flash_e3 import split_bank as sb
    bank = object.__new__(sb.SplitDecodeBank)
    descriptor, local = object(), object()
    bank.pairs = [(descriptor, local)]
    handle = sb.SplitGraphHandle(descriptor, local, bank)
    bank.handles = {local: handle}
    bank.manager = NS(run_fullgraph=lambda item: item)
    native = NS(split=object(), bank=bank,
                controller=NS(route_generation=1),
                decode_owner=NS(storage_epoch=1))
    identity = nq.NativeIdentity(NS(), native)
    session = object.__new__(ro.ObserverSession)
    session._router = identity.router
    session._local_handles = (local,)
    assert session.replay_census() == {'total':0, 'graphs':{'graph:0':0}}
    assert bank.replay(handle) is descriptor
    assert session.replay_census() == {'total':1, 'graphs':{'graph:0':1}}
    handle.replays = -1
    assert session.replay_census() is None
    return {'verdict':'PASS', 'actual_split_bank_counter':True,
            'zero_preserved':True, 'malformed_refused':True}


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
    stream = NS(cuda_stream=91)
    current = [stream]
    torch = NS(cuda=NS(CUDAGraph=lambda **kw: Graph(**kw),
                      current_stream=lambda device: current[0]))
    env = {'FLASH_S5_DECODE_ONLY_PROBE':'0', 'FLASH_S5_OVERLAP':'1',
           'FLASH_S5_REPLAY_OBSERVER':'1'}
    prior = {key:os.environ.get(key) for key in env}
    try:
        os.environ.update(env)
        owner = NS(state='open', decode_stream=stream,
                   decode_stream_token=stream.cuda_stream)
        split = NS(device='cuda:0', _flash_s5_capture_context=lambda: None,
                   _flash_s5_capture_owner=owner)
        desc = NS(cg_mode='FULL', num_tokens=4, num_reqs=1)
        assert worker_hooks.checked_capture_stream(
            split, stream, desc, torch_module=torch) is stream
        current[0] = NS(cuda_stream=92)
        try:
            worker_hooks.checked_capture_stream(split, stream, desc, torch_module=torch)
        except Exception as exc:
            assert 'split capture stream agreement failed' in str(exc)
        else:
            raise AssertionError('off-stream capture check accepted')
        current[0] = stream
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
                'instantiate_once':True, 'on_stream_check':True,
                'off_stream_refused':True}
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
    from vllm.v1.flash_e3 import native_qualification as nq
    from vllm.v1.flash_e3 import split_bank as sb
    from vllm.v1.flash_e3 import observer_runtime as runtime
    from vllm.v1.flash_e3 import engine_offers as offers
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
        owner = runtime.WorkerObserver(session, 0, engine_offers=True)
        owner.start()
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
        with torch.cuda.stream(stream):
            checked = worker_hooks.checked_capture_stream(
                manager, stream, capture_desc, torch_module=torch)
            assert checked is stream
            graph = worker_hooks.capture_graph(manager, torch_module=torch)
            with torch.cuda.graph(graph, stream=stream):
                y.add_(x)
        worker_hooks.finish_graph(manager, graph)
        raw = graph.raw_cuda_graph()
        assert type(raw) is int and raw > 0
        class Descriptor:
            cg_mode = 'FULL'
            num_tokens = 4
            num_reqs = 1
            uniform_token_count = 1
            max_query_len = 4
            num_active_loras = 0
            num_ubatches = 1
        descriptor, local = Descriptor(), object()
        bank = object.__new__(sb.SplitDecodeBank)
        bank.captured = True
        bank.pairs = [(descriptor, local)]
        bank.handles = {local: sb.SplitGraphHandle(descriptor, local, bank)}
        bank.manager = manager
        manager.run_fullgraph = lambda item: graph.replay() if item is descriptor else None
        manager.graphs[descriptor] = graph
        native_lane = NS(split=split, bank=bank,
                         controller=NS(route_generation=1),
                         decode_owner=NS(storage_epoch=1))
        identity = nq.NativeIdentity(NS(), native_lane)
        runner = NS(execute_model_state=None, main_stream=stream,
                    parallel_config=NS(pipeline_parallel_size=1),
                    is_first_pp_rank=True, is_last_pp_rank=True)
        probe = NS(bank=bank, router=identity.router,
                   split=split, storage_epoch=1, route_generation=1,
                   route='split', interval='active', in_flight=0,
                   runner=runner, controller=NS(admitting=True, drain=lambda:1),
                   _last_decode_descriptor=descriptor,
                   _require_not_down=lambda:None)
        begin = owner.begin(probe, 71, run_id='one-card-observer',
                            window_id='short-window')
        assert session.state == 'window' and graph.raw_cuda_graph() == raw
        assert isinstance(owner.ledger, offers.EngineOfferLedger)
        assert offers.current_ledger(runner) is owner.ledger
        assert getattr(runner, '_flash_s5_engine_census', None)
        start = begin['replay_census']
        assert start == {'total':0, 'graphs':{'graph:0':0}}
        with torch.cuda.stream(stream):
            bank.replay(bank.handles[local])
        stream.synchronize()
        end = session.replay_census()
        assert end == {'total':1, 'graphs':{'graph:0':1}}
        completed = owner.end(probe, 71)
        assert completed['state'] == 'closed' and completed['observer_stopped']
        assert completed['replay_census'] == end
        assert owner.ledger.binding.router_census == {'total':1,
                                                       'graphs':{'graph:0':1}}
        assert offers.current_ledger(runner) is None
        assert not getattr(runner, '_flash_s5_engine_census', None)
        try:
            owner.begin(probe, 71, run_id='one-card-observer',
                        window_id='second-window')
        except ro.ReplayObserverError as exc:
            assert exc.code == 'bad_state'
        else:
            raise AssertionError('closed observer reopened')
        checks = manager._flash_s5_capture_stream_checks
        assert len(checks) == 1 and manager._flash_s5_keep_graphs is True
        return {'verdict':'PASS', 'real_cuda_graph':True,
                'keep_graph':True, 'instantiated':True,
                'raw_handle_stable':True, 'real_observer_bound':True,
                'census_start':start, 'census_after_replay':end,
                'window_finished':True, 'engine_ledger_actual':True,
                'census_installed_then_removed':True,
                'closed_observer_refused':True,
                'window_receipt_verdict':completed['receipt']['verdict'],
                'engine_accounting_verdict':completed['engine_accounting']['verdict'],
                'real_bank_replay':True,
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
    result = {'schema':'native25-replay-census-probe/1',
              'verdict':'FAIL', 'gpu_execution':False,
              'engine_constructed':False, 'sections':{}, 'failures':[]}
    try:
        import torch
        graph = json.loads(args.graph.read_text())
        hooks, digests = source_check(graph)
        result['source_shas'] = digests
        result['sections']['cpu_shape'] = cpu_shape(hooks)
        result['sections']['census_cpu_shape'] = census_cpu_shape()
        if args.mode == 'gpu':
            result['sections']['gpu_graph_bind'] = gpu_capture_and_bind(hooks)
            result['gpu_execution'] = True
        result['verdict'] = 'PASS'
    except Exception as exc:
        result['failures'].append(type(exc).__name__ + ': ' + str(exc))
        result['traceback'] = traceback.format_exc()
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print('NATIVE25_CENSUS ' + result['verdict'], flush=True)
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
