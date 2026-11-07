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

lane-o3-fix (q18a-parity-o3 n1: O3 armed on split 0 times on the box; READING-o3-gap-20261107): the lag-* cases replace
Connected.sync_rows with the real worker's lag, and give every scheduled row the box's V2+PP+async decode cadence
(next_decode_eligible_step = current_step + pp_size, the async scheduler's update; the stack's scheduler honours it). vLLM's runner sets a row's computed-prefill from the scheduler output that carries
it, i.e. the count before that step, and only when the row is in a step; Connected's leaf read the scheduler's post-advance count, so
the handed-back row read complete at once and the census passed on CPU while the box refused it every time. The lag-* cases run with
the box arms' hold and residue (native_residue 1) so a parked head shows.
  * lag-on: O3 with the fix: the tail waits for the row's first decode; armed on split twice, no census refusal, no parked head.
  * lag-off: today's FULL path under the lag: the K2 barrier refuses the stale row (the box's 2 refusals per run end).
  * lag-k2: lag-off with k2_settle (fix c): K2 selects after the row's first decode; no barrier refusal.
  * mutant-lag-no-settle: fix 1 reverted (the tail proceeds on the handback step): the census refuses (the box's refusal on CPU).
  * mutant-lag-park: fixes 1 and 2 reverted (q18a-parity-o3's engine): the refusal also parks the next head on refused_held.
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
CASES = ('on', 'off', 'uninstalled', 'mutant-no-classify', 'mutant-no-seam',
         'lag-on', 'lag-off', 'lag-k2', 'mutant-lag-no-settle', 'mutant-lag-park')
OFF_CASES = ('off', 'uninstalled', 'lag-off', 'lag-k2')   # off: the box's off arms (FLASH_S5_O3=1, o3_split False); uninstalled: today's engine
PP_SIZE = 2                  # the box's pipeline-parallel size (TP2 x PP2)
LAG_CASES = ('lag-on', 'lag-off', 'lag-k2', 'mutant-lag-no-settle', 'mutant-lag-park')
ON_CASES = ('on', 'lag-on')


def run(case):
    import e3_native41_auto_probe as auto_probe
    root, failures = auto_probe.overlay_aliases()
    import s5_startup as startup
    import test_e3_native_storage_replay as replay
    import graph_hooks
    import native_overlap_worker as nw
    on = case not in OFF_CASES
    installed = case != 'uninstalled'
    os.environ['FLASH_S5_O3'] = '1' if installed else '0'     # the workers' switch (graph_hooks reads it per step), installed for the run
    os.environ['FLASH_S5_AUTO_ARM'] = '1'                     # worker_capability reads the process env (recipe trap 2)
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
    env_extra = {'FLASH_S5_O3': '1' if installed else '0'}
    with mock.patch.object(startup, 'install_engine', lambda engine, env: real_install(engine, env=dict(env, **env_extra))), \
            mock.patch.object(auto_probe.Connected, 'dispatch_manager', wider_manager), \
            mock.patch.object(replay, 'build_native', wider_build):
        conn = auto_probe.Connected('1')
    conn.dispatched = set()
    policy, driver, S = conn.driver.auto_arm, conn.driver, conn.scheduler
    lag = case in LAG_CASES
    worker_done = {}             # rid -> the pre-step count of the last step that carried it (the worker's own view)

    def lagged_sync_rows():
        # lane-o3-fix: the real worker's host mirror. update_requests takes num_computed_tokens from the scheduler output, the
        # count before the step, for the rows in that step only; a row no step carried since keeps its last value.
        for w in conn.workers.values():
            r = w.model_runner
            live = [req for req in S.running if req.request_id in conn.dispatched]
            r.req_states = NS(req_id_to_index={req.request_id: i for i, req in enumerate(live)},
                              prefill_len=NS(np=[req.num_prompt_tokens for req in live]),
                              num_computed_prefill_tokens=[worker_done.get(req.request_id, 0) for req in live])
            assert nw.live_census(r, r._flash_s5_native_census)['available']
    if lag:
        conn.sync_rows = lagged_sync_rows
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
            worker_done[rid] = min(start, request.num_prompt_tokens)
            if lag:
                # the box's V2+PP+async cadence (async_scheduler.py:49, scheduler.py:634): a scheduled row is eligible for its next
                # decode pp_size steps later, so the step after the handback does not carry the row
                request.next_decode_eligible_step = S.current_step + PP_SIZE
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
    if case in ('mutant-lag-no-settle', 'mutant-lag-park'):
        # fix 1 reverted: the tail proceeds once the handback row is scheduled (q18a-parity-o3's _o3_tail)
        policy._decode_scheduled = policy._handback_scheduled
    if case == 'mutant-lag-park':
        # fix 2 reverted: an o3- refusal marks the next head rejected under its cohort (q18a-parity-o3's _restore)
        import inspect
        import sys
        import textwrap
        import types
        restore = type(policy)._restore
        source = textwrap.dedent(inspect.getsource(restore))
        old = "if type(why) is str and why.startswith('o3-'):"
        assert source.count(old) == 1, 'the fix-2 line moved'
        namespace = {}
        exec(compile(source.replace(old, 'if False:'), '<mutant _restore>', 'exec'), sys.modules[restore.__module__].__dict__, namespace)
        policy._restore = types.MethodType(namespace['_restore'], policy)
    checks, violations = [], []

    def check(name, ok, **details):
        checks.append(dict(name=name, ok=bool(ok), **details))
        if not ok:
            violations.append(name)

    conn.controller.native_hold_s = 30      # the box arms' decode-phase hold (set per arm by the stage command); both cases
    conn.controller.o3_selected = on        # the arm's o3_split (set per arm by the stage command, stage_overlap._stage_control)
    conn.controller.k2_settle = case == 'lag-k2'     # the arm's k2_settle (fix c), set the same way
    if lag:
        conn.controller.native_residue = 1  # the box replay arms' residue (nt's row): a refused head is held until its cohort changes
    decoders = [conn.request('decoder-%d' % i, 40, outputs=60000) for i in range(DECODERS)]
    for _ in range(4):
        conn.step()
    natives = [conn.request(uid, PROMPT, outputs=60000) for uid in NATIVE]     # each stays a decoder: the cohort grows by one per run
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
    bank_name = {1: 'split', 0: 'full', 'split': 'split', 'full': 'full'}
    handback_steps = [s for s in steps if s['handback'] in NATIVE]
    first_native = next((i for i, s in enumerate(steps) if s['split']), len(steps))
    route_runs = []                         # (split, generation) runs over the steps from the first split step on
    for s_ in steps[first_native:]:
        key = (s_['split'], s_['generation'])
        if not route_runs or route_runs[-1] != key:
            route_runs.append(key)
    record = dict(case=case, error=error, refused=refused[:4], wall_s=round(time.monotonic() - started, 1), ticks=conn.ticks,
                  native=[dict(uid=r['uid'], start=r['start'], end=r['end'], terminal=r['terminal'], split=r['split']) for r in conn.native],
                  handback_steps=handback_steps, transitions=[dict(target=bank_name.get(e.get('target_bank'), e.get('target_bank')), generation=e.get('route_generation'))
                                                              for e in transitions],
                  o3=policy.o3_record(), route_runs=route_runs, restores=dict(policy.restores), rollbacks=dict(policy.rollbacks),
                  first_tokens={r.request_id: list(r.output_token_ids[:1]) for r in natives},
                  residue=dict(policy.residue_counts), k2=dict(policy.k2_counts), lag=lag,
                  coverage=policy._coverage_record(), alias_failures=failures)
    if case in ('on', 'off') + LAG_CASES:
        check('no-error', error is None, error=error)
        check('all-three-first-tokens', all(list(r.output_token_ids[:1]) == [7] for r in natives))
        for uid in NATIVE:
            runs = [r for r in conn.native if r['uid'] == uid]
            check('native-covers-0-to-n-1-' + uid, bool(runs) and runs[0]['start'] == 0 and runs[-1]['end'] == PROMPT - 1
                  and all(a['end'] == b['start'] for a, b in zip(runs, runs[1:])) and sum(r['terminal'] for r in runs) == 1)
        check('one-handback-step-each', sorted(s['handback'] for s in handback_steps) == sorted(NATIVE), got=[s['handback'] for s in handback_steps])
        check('decoders-continued', all(d.num_output_tokens > 8 for d in decoders))
        # stock prefill on split: only a kept one-row handback, alone among the prefilling rows
        check('no-route-refusal', not refused, refused=refused[:1])
        cov = record['coverage'] or {}
        check('coverage-handback-counted-once-at-site-handback', cov.get('native_turns') == 3
              and (cov.get('rows') or {}).get('live', {}).get('stock') == {'handback': 3}
              and (cov.get('requests') or {}).get('live', {}).get('stock') == {'handback': 3}, coverage=cov)
        check('no-other-stock-prefill-on-split', all(not s['split'] or not s.get('has_prefill') or (s['prefill_rows'] == 1 and s['handback'] in NATIVE)
                                                     for s in steps))
    if case in ON_CASES + ('mutant-lag-no-settle', 'mutant-lag-park'):
        check('handbacks-on-split-full-n-plus-1', all(s['split'] and s['cg_mode'] == 'FULL' and s['uniform'] == 1 for s in handback_steps),
              got=[(s['split'], s['cg_mode'], s['rows']) for s in handback_steps])
        if not lag:
            check('handback-rows-grow', [s['rows'] for s in handback_steps] == [DECODERS + 1, DECODERS + 2, DECODERS + 3],
                  got=[s['rows'] for s in handback_steps])
        else:
            # under the PP cadence a decoder rides every pp_size-th step, so a handback step carries its row and at most the
            # cohort proven for that run (DECODERS + the runs before it), never more
            check('handback-rows-within-the-proven-cohort', len(handback_steps) == 3 and all(
                s_['handback'] in s_['scheduled'] and 1 <= s_['rows'] <= DECODERS + i + 1 for i, s_ in enumerate(handback_steps)),
                got=[s_['rows'] for s_ in handback_steps])
        last = max(i for i, s_ in enumerate(steps) if s_['handback'] in NATIVE) if handback_steps else -1
        window = steps[first_native:last + 1]
        check('one-split-residency-across-the-three-runs', bool(window) and all(s_['split'] for s_ in window)
              and len({s_['generation'] for s_ in window}) == 1 and record['transitions'][-1:] == [dict(target='full', generation=window[0]['generation'] + 1)],
              got=route_runs, transitions=record['transitions'])
        check('o3-kept-and-armed', policy.o3_counts['kept'] == 3 and policy.o3_counts['armed'] == 2
              and policy.o3_counts['full'] == {'no-next-head': 1}, o3=record['o3'])
    if lag:
        # lane-o3-fix: the box's failure modes, under the worker's real lag
        check('no-census-refusal', policy.o3_counts['refused'] == {}, refused=policy.o3_counts['refused'])
        check('no-parked-head', policy.residue_counts['refused_held'] == 0, residue=record['residue'])
    if case == 'lag-on':
        check('tail-waited-for-the-first-decode', policy.o3_counts['settle_ticks'] >= 1, o3=record['o3'])
    if case == 'lag-off':
        # the control: today's K2 under the lag refuses the stale row at the barrier (a race close, retried); counted, not failed
        record['k2_barrier_refusals'] = policy.rollbacks.get('worker-barrier-cohort-refused', 0)
        check('k2-barrier-refuses-the-stale-row', record['k2_barrier_refusals'] >= 1, rollbacks=record['rollbacks'])
    if case == 'lag-k2':
        record['k2_barrier_refusals'] = policy.rollbacks.get('worker-barrier-cohort-refused', 0)
        check('k2-settle-no-barrier-refusal', record['k2_barrier_refusals'] == 0, rollbacks=record['rollbacks'], k2=record['k2'])
        check('k2-settled-each-run-end', policy.k2_counts['settled'] == 3 and policy.k2_counts['bound'] == 0, k2=record['k2'])
    if case in OFF_CASES:
        check('k2-off-unless-selected', (policy.k2_counts['settle_ticks'] == 0) is (case != 'lag-k2'), k2=record['k2'])
        check('handbacks-on-full-eager', all(not s['split'] and s['cg_mode'] == 'NONE' for s in handback_steps),
              got=[(s['split'], s['cg_mode'], s['rows']) for s in handback_steps])
        check('a-split-cycle-per-run', [split for split, _ in route_runs] == [True, False] * 3, got=route_runs)
        check('o3-off-nothing-kept', policy.o3_counts['kept'] == 0 and driver.o3_counters['kept'] == 0)
    # a mutant case counts only where its scenario was reached: the first run's native chunks ran to the terminal commit
    reached = sum(r['terminal'] for r in conn.native) >= 1
    if case == 'mutant-no-classify':
        check('no-route-refusal', not refused, refused=refused[:1])      # the mutant: UnsupportedRoute at the first split handback
    if case == 'mutant-no-seam':
        check('stays-split-across-runs', policy.o3_counts['armed'] >= 1 and policy.o3_counts['tail_bound'] == 0, o3=record['o3'])
    expect = case.startswith('mutant-')
    record.update(checks=checks, violations=violations, expect_violation=expect, scenario_reached=reached,
                  verdict='PASS' if (reached and bool(violations) == expect) else 'FAIL')
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
