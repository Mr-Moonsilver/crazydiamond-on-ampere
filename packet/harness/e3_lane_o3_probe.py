#!/usr/bin/env python3
"""lane-o3: three back-to-back native requests beside live decoders, through the real stack, O3 on and off. CPU only; no pass here is
box evidence.

Real: startup (install_engine, the policy's install reading FLASH_S5_O3), Core step, scheduler with its generated seams (the
positive-frontier seam the kept handback passes), KV managers, block pool, admission, driver, policy, bridge workers, graph router
(native_decode_route: split refuses a non-FULL descriptor) and graph_hooks.handback_uniform (e3_native41_auto_probe.Connected).
Leaves (Connected's, plus):
  * dispatch classification: Connected's execute counts any one-token step as decode; that is the premise dispatch_probe.out showed
    false. Here each step is classified as the pinned gather_batch_req_state does (is_prefilling = computed-at-step-start < prompt,
    has_prefill = any; uniform 1 only without prefill and with every row one token), then handback_uniform runs on the worker's real
    router; uniform 1 -> a FULL descriptor, else NONE. check_route then routes it on the real router.
  * the split bank and dispatch candidates are widened to 1..8 decoders (the template's declared leaf; no graph exists on CPU).
Cases: on, off, mutant-no-classify (handback_uniform returns stock's value), mutant-no-seam (the driver never permits the kept row).
A mutant case passes when the invariant it breaks is reported violated.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace as NS
from unittest import mock

DECODERS = 2
NATIVE = ('o3-a', 'o3-b', 'o3-c')
PROMPT = 32801            # over the 32768-row floor (no hold set): every row but the last native
LIMIT_STEPS = 4000
CASES = ('on', 'off', 'mutant-no-classify', 'mutant-no-seam')


def run(case):
    import e3_native41_auto_probe as auto_probe
    root, failures = auto_probe.overlay_aliases()
    import s5_startup as startup
    import test_e3_native_storage_replay as replay
    import graph_hooks
    import native_overlap_worker as nw
    on = case != 'off'
    os.environ['FLASH_S5_O3'] = '1' if on else '0'            # the workers' switch (graph_hooks reads it per step)
    real_install, real_manager, build = startup.install_engine, auto_probe.Connected.dispatch_manager, replay.build_native
    plain = replay._Vllm

    class Shaped(plain):
        def __init__(self, tokens=1, reqs=1, uniform=None):
            super().__init__(tokens, reqs, uniform)
            if uniform == 1:
                self.max_query_len = 1
    wider = [Shaped(n, n, 1) for n in range(1, 9)]

    def wider_build(rank, state, bus, **kwargs):
        return build(rank, state, bus, **dict(kwargs, descriptors=wider))

    def wider_manager(self, worker):
        manager = real_manager(self, worker)
        descriptors = manager._candidates[(1, 0)]
        manager._candidates = {(n, 0): [d for d in descriptors if d.num_tokens >= n] for n in range(1, 9)}
        return manager
    env_extra = {'FLASH_S5_O3': '1' if on else '0'}
    with mock.patch.object(startup, 'install_engine', lambda engine, env: real_install(engine, env=dict(env, **env_extra))), \
            mock.patch.object(auto_probe.Connected, 'dispatch_manager', wider_manager), \
            mock.patch.object(replay, 'build_native', wider_build):
        conn = auto_probe.Connected('1')
    conn.dispatched = set()
    policy, driver, S = conn.driver.auto_arm, conn.driver, conn.scheduler
    steps = []
    refused = []

    def execute(out, **kwargs):
        from concurrent.futures import Future
        from vllm.v1.outputs import ModelRunnerOutput
        ids = list(out.num_scheduled_tokens)
        rows = []
        for rid in ids:
            request = S.requests[rid]
            scheduled = out.num_scheduled_tokens[rid]
            start = request.num_computed_tokens - scheduled          # the scheduler advanced it at schedule time
            rows.append((scheduled, min(start, request.num_prompt_tokens), request.num_prompt_tokens))
        conn.dispatched.update(ids)
        conn.sync_rows()
        record = dict(step=conn.ticks, split=driver._split_ready, generation=driver.route_generation, rows=len(ids),
                      scheduled=dict(out.num_scheduled_tokens), handback=None, cg_mode=None)
        if ids:
            import numpy as np
            sched = np.array([r[0] for r in rows]); done = np.array([r[1] for r in rows]); need = np.array([r[2] for r in rows])
            prefilling = done < need
            state = NS(num_scheduled_tokens=sched, num_tokens=int(sched.sum()), num_computed_prefill_tokens_np=done,
                       prefill_len_np=need, is_prefilling_np=prefilling, has_prefill=bool(prefilling.any()))
            stock = 1 if (not state.has_prefill and int(sched.max()) == 1 and int(sched.sum()) == len(ids)) else None
            handbacks = [rid for rid, r in zip(ids, rows) if r[0] == 1 and r[1] + 1 == r[2]]
            record.update(has_prefill=state.has_prefill, prefill_rows=int(prefilling.sum()),
                          handback=handbacks[0] if handbacks else None)
            for w in conn.workers.values():
                runner = NS(_flash_e3_graph_router=w.router, decode_query_len=1,
                            _flash_s5_native_census=getattr(w.model_runner, '_flash_s5_native_census', None))
                uniform = graph_hooks.handback_uniform(runner, out, state, stock)
                desc = Shaped(int(sched.sum()), len(ids), uniform)
                if uniform != 1:
                    desc.cg_mode = replay._Mode('NONE')
                record['cg_mode'], record['uniform'] = desc.cg_mode.name if hasattr(desc.cg_mode, 'name') else str(desc.cg_mode), uniform
                nw.install_census(w.model_runner).observe_scheduler(out)
                try:
                    graph_hooks.check_route(runner, out, desc, len(ids), desc.num_tokens, uniform)
                except Exception as exc:
                    refused.append(dict(step=conn.ticks, rank=w.router.handle.worker_rank, error=type(exc).__name__,
                                        message=str(exc)[:200], record=record))
                    raise
        steps.append(record)
        conn.traces.append(record)
        conn.future = Future()
        conn.future.set_result(ModelRunnerOutput(req_ids=ids, req_id_to_index={k: i for i, k in enumerate(ids)},
            sampled_token_ids=[[7] if not S.requests[k].is_prefill_chunk else [] for k in ids]))
        return conn.future
    conn.executor.execute_model = execute
    if case == 'mutant-no-classify':
        graph_hooks.handback_uniform = lambda runner, output, state, uniform: uniform
    if case == 'mutant-no-seam':
        driver.o3_handback_permits = lambda request, kind: False
    checks, violations = [], []

    def check(name, ok, **details):
        checks.append(dict(name=name, ok=bool(ok), **details))
        if not ok:
            violations.append(name)

    decoders = [conn.request('decoder-%d' % i, 40, outputs=60000) for i in range(DECODERS)]
    for _ in range(4):
        conn.step()
    natives = [conn.request(uid, PROMPT, outputs=400) for uid in NATIVE]
    error = None
    started = time.monotonic()
    try:
        for _ in range(LIMIT_STEPS):
            if all(r.num_output_tokens > 0 for r in natives) and not driver.active and policy.phase == 'idle':
                break
            conn.step()
            if driver.error is not None:
                raise AssertionError('driver-error: %r' % (driver.error,))
    except Exception as exc:
        error = dict(type=type(exc).__name__, message=str(exc)[:400])
    transitions = [e for e in driver._journal if e['kind'] == 'transition']
    handback_steps = [s for s in steps if s['handback'] in NATIVE]
    record = dict(case=case, error=error, refused=refused[:4], wall_s=round(time.monotonic() - started, 1), ticks=conn.ticks,
                  native=[dict(uid=r['uid'], start=r['start'], end=r['end'], terminal=r['terminal'], split=r['split']) for r in conn.native],
                  handback_steps=handback_steps, transitions=[dict(target=e.get('target_bank'), generation=e.get('route_generation'))
                                                              for e in transitions],
                  o3=policy.o3_record(), restores=dict(policy.restores), rollbacks=dict(policy.rollbacks),
                  first_tokens={r.request_id: list(r.output_token_ids[:1]) for r in natives},
                  coverage=policy._coverage_record(), alias_failures=failures)
    if case in ('on', 'off'):
        check('no-error', error is None, error=error)
        check('all-three-first-tokens', all(list(r.output_token_ids[:1]) == [7] for r in natives))
        for uid in NATIVE:
            runs = [r for r in conn.native if r['uid'] == uid]
            check('native-covers-0-to-n-1-' + uid, bool(runs) and runs[0]['start'] == 0 and runs[-1]['end'] == PROMPT - 1
                  and all(a['end'] == b['start'] for a, b in zip(runs, runs[1:])) and sum(r['terminal'] for r in runs) == 1)
        check('one-handback-step-each', sorted(s['handback'] for s in handback_steps) == sorted(NATIVE), got=[s['handback'] for s in handback_steps])
        check('decoders-continued', all(d.num_output_tokens > 8 for d in decoders))
        # stock prefill on split: only a kept one-row handback, alone among the prefilling rows
        check('no-other-stock-prefill-on-split', all(not s['split'] or not s.get('has_prefill') or (s['prefill_rows'] == 1 and s['handback'] in NATIVE)
                                                     for s in steps))
    if case == 'on':
        check('handbacks-on-split-full-n-plus-1', all(s['split'] and s['cg_mode'] == 'FULL' and s['uniform'] == 1 for s in handback_steps),
              got=[(s['split'], s['cg_mode'], s['rows']) for s in handback_steps])
        check('handback-rows-grow', [s['rows'] for s in handback_steps] == [DECODERS + 1, DECODERS + 2, DECODERS + 3],
              got=[s['rows'] for s in handback_steps])
        check('one-split-entry-one-return', [t['target'] for t in record['transitions']] == ['split', 'full'], got=record['transitions'])
        check('o3-kept-and-armed', policy.o3_counts['kept'] == 3 and policy.o3_counts['armed'] == 2
              and policy.o3_counts['full'] == {'no-next-head': 1}, o3=record['o3'])
    if case == 'off':
        check('handbacks-on-full-eager', all(not s['split'] and s['cg_mode'] == 'NONE' for s in handback_steps),
              got=[(s['split'], s['cg_mode'], s['rows']) for s in handback_steps])
        check('a-split-cycle-per-run', [t['target'] for t in record['transitions']] == ['split', 'full'] * 3, got=record['transitions'])
        check('o3-off-nothing-kept', policy.o3_counts['kept'] == 0 and driver.o3_counters['kept'] == 0)
    if case == 'mutant-no-classify':
        check('split-handback-refused-by-the-router', bool(refused) and refused[0]['error'] == 'UnsupportedRoute', refused=refused[:1])
    if case == 'mutant-no-seam':
        check('stays-split-across-runs', policy.o3_counts['armed'] >= 1 and policy.o3_counts['tail_bound'] == 0, o3=record['o3'])
    expect = case.startswith('mutant-')
    record.update(checks=checks, violations=violations, expect_violation=expect,
                  verdict='PASS' if (bool(violations) == expect) else 'FAIL')
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', choices=CASES, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    try:
        r = run(a.case)
    except Exception as exc:
        r = dict(verdict='FAIL', case=a.case, error_type=type(exc).__name__, error=str(exc)[:600], traceback=traceback.format_exc())
    r.update(schema='lane-o3-connected-cpu/1', gpu_execution=False, box_evidence=False,
             probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with a.out.open('x') as f:
        json.dump(r, f, sort_keys=True, indent=2, default=str)
        f.write('\n')
    print(json.dumps({k: r.get(k) for k in ('verdict', 'case', 'violations', 'error_type', 'error')}, default=str))
    return 0 if r.get('verdict') == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
