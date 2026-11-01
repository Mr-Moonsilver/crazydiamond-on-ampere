#!/usr/bin/env python3
"""Native41 read-only auto window over a real automatic cycle, CPU only.

Composition: Codex's connected Core (e3_native41_auto_probe.Connected: actual
startup, EngineCore step, scheduler, driver, policy and CPU workers) plus the
actual NativeOverlapSession and NativeControlClient over the generated
EngineCore.flash_s5_native_control utility. The box arm's own proof and return
readers (e3_native41_autoarm_box.auto_proof / returns_ms) grade the window.
Leaves are listed in native41_fixture_leaves.json (window-probe entries).
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
import traceback

import e3_native41_auto_probe as probe

MODES = ('auto', 'latched', 'below-threshold', 'off')


class Front:
    """API-side AsyncLLM leaf: the pinned single utility client over this Core.

    A window never pauses or resumes; those two calls fail the probe.
    """

    def __init__(self, engine):
        import test_e3_native_overlap as native
        self.engine_core = native.UtilityClient(engine)
        self._engine = engine

    async def is_paused(self):
        return self._engine.is_scheduler_paused()

    async def pause_generation(self, mode, clear_cache):
        raise AssertionError('auto window must not pause')

    async def resume_generation(self):
        raise AssertionError('auto window must not resume')


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

    def call(action):
        return asyncio.run(session.execute(dict(schema=1, action=action, run_id='native41-window')))

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
        try:
            call('auto_window_begin')
            refused = None
        except NativeOverlapSessionError as exc:
            refused = str(exc)
        check('default-off-window-not-exposed', refused == 'session-action-not-exposed', refused=refused)
        check('default-off-no-policy', policy is None)
    else:
        if mode == 'latched':
            policy.manual_owned = True
            try:
                call('auto_window_begin')
                refused = None
            except NativeOverlapSessionError as exc:
                refused = str(exc)
            check('latched-window-refused', refused == 'auto-window-requires-released-idle-full', refused=refused)
            policy.manual_owned = False
        start = time.monotonic_ns()
        begin = call('auto_window_begin')
        check('begin-on-released-full', begin.get('route') == 'full' and begin.get('phase') == 'auto-prefill'
              and begin['snapshot']['auto_arm']['manual_owned'] is False)
        tokens = 32721 if mode == 'below-threshold' else 32801
        req = conn.request('long', tokens, outputs=8)
        for _ in range(2500):
            conn.step()
            if req.num_output_tokens and not conn.driver.active and not policy.active:
                break
        try:
            end = call('auto_window_end')
            end_error = None
        except Exception as exc:  # noqa: BLE001 - the no-cycle window may journal nothing
            end, end_error = None, str(exc)
        stop = time.monotonic_ns()
        snap = policy.snapshot()
        result.update(end_error=end_error, snapshot={k: snap[k] for k in ('selection_seq', 'completed', 'last_close',
                                                                            'last_unselected', 'last_refusal')})
        if mode in ('auto', 'latched'):
            check('window-end-returned', end is not None, error=end_error)
            proof = box.auto_proof(end, {'start_ns': start, 'end_ns': stop}, conn.driver.engine_epoch,
                                   req.num_prompt_tokens)
            ret = box.returns_ms(end)
            result.update(proof={k: v for k, v in proof.items() if k != 'binding'}, returns=ret,
                          window_complete=end.get('complete'), auto_reasons=end.get('auto_reasons'))
            check('one-armed-closed-cycle', end.get('auto_reasons') == [] and end.get('complete') is True,
                  reasons=end.get('auto_reasons'), binding=(end.get('measurement_binding') or {}).get('reasons'))
            check('box-proof-complete', proof['complete'], reasons=proof['reasons'])
            check('armed-request-is-the-plain-request',
                  proof['request_uid_sha256'] == hashlib.sha256(req.request_id.encode()).hexdigest()
                  and proof['origin'] == 'auto-arm')
            check('segments-are-the-native-chunks', proof['segments'] == len(conn.native) > 1)
            check('group-is-the-whole-prompt', (proof.get('group') or {}).get('prompt_end') == tokens)
            check('native-window-rate', type(proof.get('prefill_tokens_per_s')) in (int, float)
                  and proof['prefill_tokens_per_s'] > 0)
            span = proof.get('group') or {}
            # The arm's engine-basis decode window reads exactly these fields.
            check('engine-basis-span', type(span.get('start_ns')) is int and type(span.get('end_ns')) is int
                  and start <= span['start_ns'] < span['end_ns'] <= stop
                  and type(proof.get('headline_basis')) is str, group_keys=sorted(span))
            check('return-ms-measured', type(ret.get('final_commit_to_all_rank_full_ack_ms')) in (int, float)
                  and ret['final_commit_to_all_rank_full_ack_ms'] >= 0, returns=ret)
            check('closed-full-tail', snap['last_close']['reason'] == 'full-tail-completed')
        else:
            proof = (box.auto_proof(end, {'start_ns': start, 'end_ns': stop}, conn.driver.engine_epoch,
                                    req.num_prompt_tokens) if end is not None else {'complete': False})
            check('below-threshold-not-native', not conn.native and proof['complete'] is False)
            check('below-threshold-window-not-proven', end is None or end.get('complete') is False,
                  reasons=None if end is None else end.get('auto_reasons'), error=end_error)
            check('below-threshold-reason-recorded', (snap['last_unselected'] or {}).get('reason') == 'plan'
                  and snap['last_unselected'].get('num_tokens') == tokens, last=snap['last_unselected'])
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
    r.update(schema='native41-window-cpu/1', gpu_execution=False, box_evidence=False,
             probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with a.out.open('x') as f:
        json.dump(r, f, sort_keys=True, indent=2, default=str)
        f.write('\n')
    print(json.dumps({k: r.get(k) for k in ('verdict', 'mode', 'error_type', 'error')}))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
