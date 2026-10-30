#!/usr/bin/env python3
"""VM<vm-id> one-card stock keep-pause / resume proof for the row 4 harness gate.

The real reduced Qwen EngineCore and MultiprocExecutor process a real Request.
The fixture is model size/topology, not scheduler or pause behavior. This does
not qualify TP2PP2 placement, overlap speed or the four-rank observer.
"""
import argparse
import ast
import hashlib
import inspect
import json
import time
import traceback
from pathlib import Path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_order(source):
    """Bind the tested harness source to both required explicit resumes."""
    tree = ast.parse(source)
    owner = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef)
                 and node.name == 'NativePlacementRunner')
    run = next(node for node in owner.body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and node.name == 'run')
    calls = []
    for node in ast.walk(run):
        if not isinstance(node, ast.Call):
            continue
        fn = ast.unparse(node.func)
        if fn not in ('driver._worker', 'driver._session',
                      'correctness._start_streams', 'correctness._wait_open'):
            continue
        label = None
        if node.args and isinstance(node.args[0], ast.Constant):
            label = node.args[0].value
        calls.append((node.lineno, fn, label))
    calls.sort()
    worker = [line for line, fn, label in calls
              if fn == 'driver._worker' and label == 'row4']
    stream = [line for line, fn, _ in calls
              if fn == 'correctness._start_streams']
    wait = [line for line, fn, _ in calls
            if fn == 'correctness._wait_open']
    resume = [line for line, fn, label in calls
              if fn == 'driver._session' and label == 'row4'
              and any(keyword.arg == 'require_epoch'
                      and isinstance(keyword.value, ast.Constant)
                      and keyword.value.value is False
                      for call in ast.walk(run) if isinstance(call, ast.Call)
                      and call.lineno == line for keyword in call.keywords)]
    # The first worker call configures the teacher, the last clears it. The
    # first stream launch is the stock cohort; a later one is native prefill.
    assert len(worker) >= 2 and len(stream) >= 2 and wait
    assert any(worker[0] < stream[0] < row < wait[0] for row in resume)
    assert any(worker[-1] < row for row in resume)
    return {'teacher_worker_lines': worker, 'stream_lines': stream,
            'open_wait_lines': wait, 'explicit_resume_lines': resume,
            'stock_open_order_verified': True, 'full_return_order_verified': True}


def cpu_shape():
    import torch
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor
    from vllm.v1.request import Request
    from vllm.sampling_params import SamplingParams
    from e3_native_stock_rpc_probe import gpu_config, cache_capacity_shape

    source = Path(__file__).with_name('e3_native_qualification_driver.py')
    order = candidate_order(source.read_text())
    cfg = gpu_config('cpu')
    request = Request(request_id='native23-pause-shape',
                      prompt_token_ids=[1, 2, 3, 4],
                      sampling_params=SamplingParams(
                          max_tokens=3, temperature=0.0, ignore_eos=True),
                      pooling_params=None)
    inspect.signature(EngineCore).bind(cfg, MultiprocExecutor, False)
    inspect.signature(EngineCore.pause_scheduler).bind(
        None, mode='keep', clear_cache=False)
    for name in ('is_scheduler_paused', 'resume_scheduler',
                 'add_request', 'step_with_batch_queue', 'shutdown'):
        assert callable(getattr(EngineCore, name))
    # Stock VllmConfig resolution may initialize CUDA on a GPU venue even
    # before EngineCore construction. This is a venue observation, not a
    # precondition for the paused-scheduler proof.
    cuda_after_config = torch.cuda.is_initialized()
    return {'verdict': 'PASS', 'candidate_driver_sha256': _sha(source),
            'candidate_order': order,
            'actual_core_class': EngineCore.__module__ + '.' + EngineCore.__name__,
            'actual_executor_class': (MultiprocExecutor.__module__ + '.'
                                      + MultiprocExecutor.__name__),
            'actual_request_class': type(request).__module__ + '.'
                                    + type(request).__name__,
            'cache_capacity': cache_capacity_shape(cfg),
            'cuda_initialized_after_config': cuda_after_config,
            'gpu_owner_constructed': False}


def gpu_pause_resume():
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor
    from vllm.v1.request import Request
    from vllm.sampling_params import SamplingParams
    from e3_native_stock_rpc_probe import gpu_config

    cfg = gpu_config('cuda')
    core = None
    try:
        core = EngineCore(cfg, MultiprocExecutor, False)
        assert type(core.model_executor) is MultiprocExecutor
        assert not core.is_scheduler_paused()
        core.pause_scheduler(mode='keep', clear_cache=False)
        assert core.is_scheduler_paused()
        request = Request(request_id='native23-paused-stock-request',
                          prompt_token_ids=[1, 2, 3, 4],
                          sampling_params=SamplingParams(
                              max_tokens=3, temperature=0.0, ignore_eos=True),
                          pooling_params=None)
        core.add_request(request)
        # has_requests() deliberately reports false under PAUSED_ALL. The
        # registered request dictionary is the stock source of truth for a
        # submitted cohort while the scheduler is stopped.
        queued = core.scheduler.requests.get(request.request_id) is request
        assert queued
        for _ in range(8):
            core.step_with_batch_queue()
            assert request.num_output_tokens == 0
        assert (core.is_scheduler_paused()
                and core.scheduler.requests.get(request.request_id) is request)
        core.resume_scheduler()
        assert not core.is_scheduler_paused()
        first = None
        iterations = 0
        deadline = time.monotonic() + 120
        while core.scheduler.has_requests() or core.batch_queue:
            assert time.monotonic() < deadline, 'resumed-request-deadline'
            core.step_with_batch_queue()
            iterations += 1
            if first is None and request.num_output_tokens:
                first = iterations
        assert request.num_output_tokens == 3 and first is not None
        return {'verdict': 'PASS', 'actual_core_constructed': True,
                'actual_multiproc_executor': True, 'actual_request': True,
                'paused_steps': 8, 'queued_while_paused': queued,
                'tokens_while_paused': 0, 'first_token_after_resume_step': first,
                'completion_tokens': request.num_output_tokens,
                'resumed_iterations': iterations, 'gpu_execution': True,
                'tp': 1, 'pp': 1, 'model_weights': 'dummy',
                'four_rank_placement_qualified': False}
    finally:
        if core is not None:
            core.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=('cpu-shape', 'gpu'), required=True)
    args = parser.parse_args()
    started = time.monotonic()
    record = {'schema': 'native23-row4-pause-probe/1', 'verdict': 'FAIL',
              'failure_class': 'OPEN', 'sections': {},
              'engine_constructed': False, 'gpu_execution': False,
              'failures': [], 'missing': []}
    try:
        graph = json.loads(args.graph.read_text())
        assert graph['verdict'] == 'PASS'
        record['sections']['cpu_shape'] = cpu_shape()
        print('END cpu_shape PASS', flush=True)
        if args.mode == 'gpu':
            record['sections']['gpu_pause_resume'] = gpu_pause_resume()
            record['engine_constructed'] = True
            record['gpu_execution'] = True
            print('END gpu_pause_resume PASS', flush=True)
        record['verdict'] = 'PASS'
        record['failure_class'] = 'NONE'
    except Exception as exc:
        record['failures'].append(type(exc).__name__ + ': ' + str(exc))
        record['traceback'] = traceback.format_exc()
    record['elapsed_s'] = time.monotonic() - started
    with args.out.open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print('NATIVE23_ROW4_PAUSE ' + record['verdict'], flush=True)
    return 0 if record['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
