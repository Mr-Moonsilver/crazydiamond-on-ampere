#!/usr/bin/env python3
"""Native41 status receipts over a real automatic cycle, CPU only.

Composition: Codex's connected Core (e3_native41_auto_probe.Connected: actual
startup, EngineCore step, scheduler, driver, policy and CPU workers) plus the
actual NativeOverlapSession and NativeControlClient over the generated
EngineCore.flash_s5_native_control utility. Status is read through the session
after every Core step (the box poller reads the same records at 0.25 s / 0.02 s)
and graded by the box arm's own reader, e3_native41_autoarm_box.receipt_proof.
No window or other native action exists on this engine; the probe sends status
only. Leaves are listed in native41_fixture_leaves.json (receipts-probe entries).
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import traceback

import e3_native41_auto_probe as probe

MODES = ('auto', 'latched', 'below-threshold', 'off')


class Front:
    """API-side AsyncLLM leaf: the pinned single utility client over this Core.

    Status reads never pause or resume; those two calls fail the probe.
    """

    def __init__(self, engine):
        import test_e3_native_overlap as native
        self.engine_core = native.UtilityClient(engine)
        self._engine = engine

    async def is_paused(self):
        return self._engine.is_scheduler_paused()

    async def pause_generation(self, mode, clear_cache):
        raise AssertionError('receipt reads must not pause')

    async def resume_generation(self):
        raise AssertionError('receipt reads must not resume')


def run(mode):
    _, failures = probe.overlay_aliases()
    # After the aliases: the box reader imports harness modules that import e3_types.
    import e3_native41_autoarm_box as box
    conn = probe.Connected('0' if mode == 'off' else '1')
    conn.dispatched = set()
    import executor_hooks as eh
    from native_overlap_control import NativeControlClient
    from native_overlap_session import NativeOverlapSession, NativeOverlapSessionError
    # The generated EngineCore utility method, bound to this Core.
    conn.engine.flash_s5_native_control = lambda command: eh.native_control(conn.engine, command)
    front = Front(conn.engine)
    session = NativeOverlapSession(front, control=NativeControlClient(front, timeout_s=5))
    actions = []

    def call(action):
        actions.append(action)
        return asyncio.run(session.execute(dict(schema=1, action=action, run_id='native41-receipts')))

    checks = []

    def check(name, ok, **details):
        checks.append(dict(name=name, ok=bool(ok), **details))
        assert ok, name

    cohort = conn.request('cohort', 40, outputs=2048)
    for _ in range(4):
        conn.step()
    policy = getattr(conn.driver, 'auto_arm', None)
    result = dict(mode=mode)
    if mode == 'off':
        check('default-off-no-policy', policy is None)
        check('default-off-status-has-no-auto-arm', 'auto_arm' not in call('status'))
        try:
            call('release_auto_arm')
            refused = None
        except NativeOverlapSessionError as exc:
            refused = str(exc)
        check('default-off-release-not-exposed', refused == 'session-action-not-exposed', refused=refused)
    else:
        if mode == 'latched':
            # The lease as a manual prepare leaves it (the stock arm's state).
            policy.manual_owned = True
        before = call('status')
        tokens = 32721 if mode == 'below-threshold' else 32801
        req = conn.request('long', tokens, outputs=8)
        poller = box.Poller(None)
        poller.baseline = box.compact(before)
        for _ in range(2500):
            conn.step()
            view = box.compact(call('status'))
            poller.count += 1
            poller.keep(view)
            if req.num_output_tokens and not conn.driver.active and not (policy and policy.active):
                break
        poller.actions = ['status'] * poller.count
        polled = {'samples_read': poller.count, 'samples_kept': len(poller.samples), 'samples': poller.samples,
                  'actions': ['status'], 'errors': poller.errors}
        after = call('status')
        check('first-token-produced', req.num_output_tokens > 0)
        auto_proof = box.receipt_proof('auto', before, polled, after, req.num_prompt_tokens, actions)
        stock_proof = box.receipt_proof('stock', before, polled, after, req.num_prompt_tokens, actions)
        result.update(auto_proof={k: v for k, v in auto_proof.items()}, stock_proof=stock_proof,
                      native_chunks=len(conn.native), prompt_tokens=req.num_prompt_tokens,
                      samples_read=poller.count, samples_kept=len(poller.samples))
        check('status-only', set(actions) == {'status'}, actions=sorted(set(actions)))
        if mode == 'auto':
            check('auto-receipts-complete', auto_proof['complete'], reasons=auto_proof['reasons'])
            check('engaged-request-is-the-plain-request',
                  auto_proof['engaged']['request_uid_sha256'] == hashlib.sha256(req.request_id.encode()).hexdigest())
            check('every-native-chunk-observed', auto_proof['native_chunks_observed'] == len(conn.native) > 1,
                  observed=auto_proof['native_chunks_observed'], native=len(conn.native))
            check('last-native-row-is-prompt-minus-one', auto_proof['max_native_row_observed'] == tokens - 1)
            check('admitted-delta-is-the-native-chunks', auto_proof['admitted_delta'] == len(conn.native))
            check('return-bounded', type(auto_proof['return']['upper_bound_ms']) in (int, float)
                  and auto_proof['return']['upper_bound_ms'] >= 0, ret=auto_proof['return'])
            check('stock-reader-rejects-an-auto-cycle', stock_proof['complete'] is False
                  and 'stock-lease-not-held-at-every-sample' in stock_proof['reasons']
                  and 'stock-arm-auto-selection-or-native-work-observed' in stock_proof['reasons'],
                  reasons=stock_proof['reasons'])
        elif mode == 'latched':
            check('latched-not-native', not conn.native and policy.selection_seq == 0)
            check('stock-receipts-complete', stock_proof['complete'], reasons=stock_proof['reasons'])
            check('auto-reader-rejects-the-latched-lease', auto_proof['complete'] is False
                  and {'manual-lease-at-some-sample', 'auto-lane-engagement-not-observed'} <= set(auto_proof['reasons']),
                  reasons=auto_proof['reasons'])
            policy.manual_owned = False
        else:
            check('below-threshold-not-native', not conn.native and policy.selection_seq == 0)
            check('below-threshold-auto-not-proven', auto_proof['complete'] is False
                  and 'auto-lane-engagement-not-observed' in auto_proof['reasons']
                  and 'no-native-admission-counted' in auto_proof['reasons'], reasons=auto_proof['reasons'])
        if req.request_id in conn.scheduler.requests:
            conn.stack.abort(conn.scheduler.requests[req.request_id])
    conn.stack.abort(cohort)
    for _ in range(8):
        conn.step()
    check('clean-owned-state', not conn.scheduler.requests and not conn.controller.active
          and not conn.stack.leases.held_blocks() and not conn.driver.active)
    import torch
    check('no-cuda-initialized', not torch.cuda.is_initialized())
    return dict(result, verdict='PASS', checks=checks, alias_failures=failures, cuda_initialized=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=MODES, default='auto')
    a = p.parse_args()
    try:
        r, rc = run(a.mode), 0
    except Exception as exc:  # noqa: BLE001 - reported, never retried
        r, rc = dict(verdict='FAIL', mode=a.mode, error_type=type(exc).__name__, error=str(exc),
                     traceback=traceback.format_exc()), 1
    r.update(schema='native41-receipts-cpu/1', gpu_execution=False, box_evidence=False,
             probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with a.out.open('x') as f:
        json.dump(r, f, sort_keys=True, indent=2, default=str)
        f.write('\n')
    print(json.dumps({k: r.get(k) for k in ('verdict', 'mode', 'error_type', 'error')}))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
