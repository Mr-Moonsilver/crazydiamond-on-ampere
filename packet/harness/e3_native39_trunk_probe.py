#!/usr/bin/env python3
"""Native39 R/W in the pinned image: positive-frontier stock prefills on an armed split route.

Runs against whichever overlay is mounted (the native39 candidate, or the
native38 held-b overlay as the negative control) and reports PASS/FAIL per
scenario without retrying.  Construction is native37's T1/T2 stack, reused
unchanged (e3_native37_admission_probe.Stack): the pinned AsyncScheduler built
by its own __init__ from the mounted generated core/scheduler.py, the real
KV-cache configuration path, SchedulerLeases, PrefillAdmissionController,
admission metadata and NativeSegmentDriver.  Requests come from the real
EngineCore.preprocess_add_request.  Declared leaves: native37's
(native37_fixture_leaves.json t2-*) plus the one-token sampler output below
(native39_fixture_leaves.json).

R  running prefill (pinned scheduler.py:719 running loop).  A 10000-token
   request gets its first stock 1024-token chunk on FULL; the harness then
   enters split and arms (paused engine, as prepare does).  On the next
   schedule() the request is RUNNING mid-prefill: it must not be scheduled on
   the armed split route.  After disarm + FULL it continues with 1024 tokens.
W  positive-frontier waiting request (pinned scheduler.py:1145 else branch).
   A resumable streaming-input session prefills 2048 tokens on FULL and stops
   after one sampled token; its next input chunk arrives (74 tokens) and the
   real _update_request_as_session puts it back to WAITING with
   num_computed_tokens 2048.  On the armed split route native admission
   refuses its plan (typed no-staged-plan: a 9-token tail after the 176
   stop), and the stock loop must hold it, not schedule its chunk on split.
   After disarm + FULL stock schedules the chunk.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import traceback

import e3_native37_admission_probe as base

R_TOKENS = 10000
W_FIRST, W_NEXT = 2048, 74
BUDGET = base.BUDGET


class _Stop(Exception):
    pass


def kinds(controller):
    record = base.typed(controller) or {}
    return dict(record.get('split_stock_deferrals_by_kind') or {})


def sampled_step(stack, request, token):
    """One EngineCore step with the declared one-token sampler leaf.

    The worker samples a token for a request exactly when this step scheduled
    the end of its prompt (stock sets is_prefill_chunk False then); a
    mid-prefill chunk samples nothing.
    """
    from vllm.v1.outputs import ModelRunnerOutput
    out = stack.scheduler.schedule()
    ids = list(out.num_scheduled_tokens)
    sampled = [[token] if k == request.request_id and not request.is_prefill_chunk else [] for k in ids]
    stack.scheduler.update_from_output(out, ModelRunnerOutput(
        req_ids=ids, req_id_to_index={k: i for i, k in enumerate(ids)}, sampled_token_ids=sampled))
    return out


def streaming_request(stack, name, tokens):
    from vllm.v1.engine import EngineCoreRequest
    from vllm.sampling_params import SamplingParams
    from e3_native_cow_lease_probe import host_preprocess
    stack.salt += 1
    request = EngineCoreRequest(request_id=name, prompt_token_ids=tokens, mm_features=None,
                                sampling_params=SamplingParams(max_tokens=1, temperature=0., ignore_eos=True),
                                pooling_params=None, arrival_time=time.time(), lora_request=None,
                                cache_salt='native39-w', data_parallel_rank=None, resumable=True)
    actual = host_preprocess(stack.core, request)
    stack.scheduler.add_request(actual)
    return actual


def scenario(stack, body):
    checks = []

    def check(name, ok, **detail):
        checks.append(dict(name=name, verdict='PASS' if ok else 'FAIL', **detail))
        if not ok:
            raise _Stop(name)

    record = dict(checks=checks)
    free0 = stack.free_blocks()
    try:
        body(stack, check)
        check('clean', not stack.scheduler.requests and stack.free_blocks() == free0
              and not stack.leases.held_blocks() and stack.controller.parked is None
              and stack.driver.admission_intent() is None)
        record['verdict'] = 'PASS'
    except _Stop as stop:
        record.update(verdict='FAIL', first_failure=str(stop))
    finally:
        base._retire_all(stack)
    return record


def r_body(stack, check):
    S, controller = stack.scheduler, stack.controller
    b = stack.request('r-running', R_TOKENS)
    out, _ = stack.step()
    check('stock-chunk-on-full', out.num_scheduled_tokens.get(b.request_id) == BUDGET
          and b in S.running and b.is_prefill_chunk and b.num_computed_tokens == BUDGET,
          scheduled=dict(out.num_scheduled_tokens), computed=b.num_computed_tokens)
    stack.arm()
    before = kinds(controller)
    deferrals = controller.split_stock_deferrals
    for step in range(2):
        out, _ = stack.step()
        check('running-prefill-held-on-armed-split-%d' % step, b.request_id not in out.num_scheduled_tokens
              and b.num_computed_tokens == BUDGET and b in S.running and controller.parked is None,
              scheduled=dict(out.num_scheduled_tokens), computed=b.num_computed_tokens)
    after = kinds(controller)
    check('held-typed-as-running-prefill', after.get('running-prefill', 0) - before.get('running-prefill', 0) == 2
          and controller.split_stock_deferrals - deferrals == 2, before=before, after=after)
    stack.disarm()
    out, _ = stack.step()
    check('stock-continues-on-full', out.num_scheduled_tokens.get(b.request_id) == BUDGET
          and b.num_computed_tokens == 2 * BUDGET, scheduled=dict(out.num_scheduled_tokens))
    stack.abort(b)


def w_body(stack, check):
    from vllm.v1.request import RequestStatus
    S, controller = stack.scheduler, stack.controller
    first = [(i * 11 + 3) % 251 for i in range(W_FIRST)]
    s = streaming_request(stack, 'w-stream', first)
    scheduled = []
    for _ in range(8):
        out = sampled_step(stack, s, 7)
        scheduled.append(out.num_scheduled_tokens.get(s.request_id))
        if s.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            break
    check('first-input-prefilled-on-full', s.status == RequestStatus.WAITING_FOR_STREAMING_REQ
          and s.num_computed_tokens == W_FIRST and sum(n or 0 for n in scheduled) == W_FIRST,
          status=str(s.status), computed=s.num_computed_tokens, scheduled=scheduled)
    # The next streaming input: the real add_request -> _update_request_as_session.
    nxt = [(i * 13 + 5) % 251 for i in range(W_NEXT)]
    streaming_request(stack, 'w-stream', nxt)
    check('resumed-with-positive-frontier', s.status == RequestStatus.WAITING
          and s.num_computed_tokens == W_FIRST and s.num_tokens == W_FIRST + W_NEXT
          and s.num_output_tokens == 0,
          status=str(s.status), computed=s.num_computed_tokens, tokens=s.num_tokens)
    stack.arm()
    before = kinds(controller)
    for step in range(2):
        out, _ = stack.step()
        check('positive-frontier-held-on-armed-split-%d' % step, s.request_id not in out.num_scheduled_tokens
              and s.num_computed_tokens == W_FIRST and controller.parked is None,
              scheduled=dict(out.num_scheduled_tokens), computed=s.num_computed_tokens)
    after = kinds(controller)
    last = (base.typed(controller) or {}).get('last_refusal') or {}
    check('native-refusal-typed', last.get('reason') == 'no-staged-plan' and last.get('start') == W_FIRST
          and last.get('native_end') == W_FIRST + W_NEXT - 1,
          last_refusal={k: last.get(k) for k in ('reason', 'start', 'native_end', 'furthest_boundary', 'gap')})
    check('held-typed-as-positive-frontier',
          after.get('positive-frontier', 0) - before.get('positive-frontier', 0) == 2, before=before, after=after)
    stack.disarm()
    out, _ = stack.step()
    check('stock-schedules-the-chunk-on-full', (out.num_scheduled_tokens.get(s.request_id) or 0) > 0,
          scheduled=dict(out.num_scheduled_tokens))
    stack.abort(s)


def mounted():
    rows = base.mounted()
    from vllm.v1.flash_e3 import scheduler_hooks
    rows['e3/scheduler_hooks.py']['native39_seams'] = all(
        callable(getattr(scheduler_hooks, name, None))
        for name in ('defer_stock_positive_frontier', 'defer_stock_running_prefill'))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--label', required=True)
    a = p.parse_args()
    record = dict(schema='native39-trunk-probe/1', label=a.label, verdict='INCONCLUSIVE', rc=2,
                  gpu_execution=False, box_evidence=False)
    try:
        import torch
        record['mounted'] = mounted()
        stack = base.Stack()
        record['geometry'] = stack.geometry
        record['r'] = scenario(stack, r_body)
        record['w'] = scenario(stack, w_body)
        assert not torch.cuda.is_initialized()
        record['cuda_initialized'] = False
        failed = [name for name in ('r', 'w') if record[name]['verdict'] != 'PASS']
        record.update(verdict='FAIL' if failed else 'PASS', rc=1 if failed else 0, failed=failed)
    except Exception as exc:  # noqa: BLE001 - classified, never retried
        record.update(verdict='FAIL', rc=1, failure_class='HARNESS-OR-ENGINE-UNCLASSIFIED',
                      error_type=type(exc).__name__, error=str(exc)[:2000],
                      traceback=traceback.format_exc()[-6000:])
    record['probe_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with a.out.open('x') as f:
        json.dump(record, f, indent=2, sort_keys=True, default=str)
        f.write('\n')
    summary = {k: record.get(k) for k in ('label', 'verdict', 'rc', 'failed', 'error_type', 'error')}
    for name in ('r', 'w'):
        if name in record:
            summary[name] = {k: record[name].get(k) for k in ('verdict', 'first_failure')}
    print(json.dumps(summary, sort_keys=True))
    raise SystemExit(record['rc'])


if __name__ == '__main__':
    main()
