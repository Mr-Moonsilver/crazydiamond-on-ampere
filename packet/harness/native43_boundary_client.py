#!/usr/bin/env python3
"""One fresh prompt, no cohort: engine-owned four-chunk boundary cancellation."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import e3_overlap_performance as shared
import e3_decode_correctness as streams
from e3_native_runtime_client import NativeRuntimeClient
from e3_native39_w2_length import guard
from native43_boundary_clock import prove

FRONTIER = 101200  # 24*4048, followed by 1024+1024+1024+976.
PHASE = 'split-prefill-solo'


def status(client):
    send = time.monotonic_ns()
    packet = client.call('status')
    recv = time.monotonic_ns()
    bracket = prove(packet, send, recv)
    if not bracket['proved']:
        raise ValueError('native43-engine-sample-outside-client-bracket')
    return packet, bracket


def run(a):
    raw = json.loads(Path(a.workload).read_text())
    problems, cfg = shared.validate_manifest(raw)
    if problems:
        raise ValueError('native43-invalid-workload')
    prompts = shared._prefill_prompts(cfg)
    length = guard(a.url, a.model, a.workload)
    if length.get('verdict') != 'PASS' or length['observed']['fresh']['tokens'] != 197270:
        raise ValueError('native43-prompt-not-plannable-197270')
    client = NativeRuntimeClient(a.url, a.run_id, epoch=a.epoch)
    record = dict(schema='native43-boundary-client/1', verdict='INCONCLUSIVE',
                  policy=a.policy, fresh=True, cohort_streams=0, speed_claim=False,
                  frontier=FRONTIER, length_guard=length,
                  prompt_sha256=prompts['fresh_sha256'], paired_salt_id=a.pair_id)
    state, workload = None, None
    try:
        before, record['before_clock'] = status(client)
        if before.get('route') != 'full' or before.get('scheduler_requests') != 0:
            raise ValueError('native43-needs-empty-full-engine')
        record['prepare'] = client.prepare('split', PHASE)
        record['begin'] = client.call('prefill_begin', phase=PHASE)
        salt = hashlib.sha256((a.pair_id + '|native43-fresh').encode()).hexdigest()
        spec = cfg['prefill']['fresh']
        state, workload = shared._send_prefill(a.url, a.model, cfg, 'prefill_split', 0,
            prompts['fresh'], spec['generation_budget'], spec['seed'], salt)
        deadline = time.monotonic() + a.timeout_s
        polls = 0
        while time.monotonic() < deadline:
            packet, bracket = status(client)
            polls += 1
            diagnostic = packet.get('native43_boundary') or {}
            checkpoint = diagnostic.get('checkpoint') or {}
            closure = packet.get('last_native_closure') or {}
            if checkpoint and (checkpoint.get('chunk_end') != FRONTIER or checkpoint.get('exact') is not True):
                raise ValueError('native43-cutoff-not-exact')
            if (checkpoint and diagnostic.get('cancel_started') is True
                    and closure.get('complete') is True and packet.get('route') == 'full'
                    and not packet.get('transition_pending') and not packet.get('return_pending')):
                record.update(checkpoint=checkpoint, closure=closure, final_status=packet,
                              final_clock=bracket, status_polls=polls)
                rows = closure.get('rows') or []
                if (len(rows) != 4 or {r.get('worker_rank') for r in rows} != {0,1,2,3}
                        or any(r.get('request_closed') is not True for r in rows)
                        or closure.get('chunk_end') != FRONTIER
                        or (closure.get('source_leases') or {}).get('clean') is not True
                        or closure.get('pp_receipts_pending')):
                    raise ValueError('native43-certified-four-rank-close-unproved')
                if state.first_advance_ns is not None:
                    raise ValueError('native43-request-finished-before-cutoff')
                record['verdict'] = 'PASS'
                break
            if state.error is not None or state.malformed is not None:
                raise ValueError('native43-request-error-before-checkpoint')
            if packet.get('error'):
                raise ValueError('native43-engine-error')
            time.sleep(0.1)
        else:
            raise TimeoutError('native43-cutoff-deadline')
    except Exception as error:
        record.update(verdict='INCONCLUSIVE', failure_class='HARNESS',
                      error_type=type(error).__name__, cause=str(error)[:160])
    finally:
        if state is not None:
            # Close the original HTTP stream, so the stock request cannot be
            # resumed; engine-side diagnostic hold independently blocks it.
            try:
                alive = streams._stop_streams(record, [state], workload)
                if alive:
                    record.update(verdict='INCONCLUSIVE', cause='native43-reader-still-alive')
            except Exception as error:
                record.update(verdict='INCONCLUSIVE', cause='native43-client-cancel-unproved',
                              cleanup_error_type=type(error).__name__)
        if record['verdict'] != 'PASS':
            try:
                # Error path uses the same real typed stop; its evidence is
                # retained, never a substitute for the exact cutoff.
                record['error_cleanup'] = client.leave()
            except Exception as error:
                record['error_cleanup_error_type'] = type(error).__name__
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('url', 'model', 'run-id', 'pair-id', 'workload', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--epoch', type=int, required=True)
    p.add_argument('--policy', choices=('current', 'dense'), required=True)
    p.add_argument('--timeout-s', type=float, default=300)
    a = p.parse_args(argv)
    if Path(a.out).exists():
        return 2
    if not 0 < a.timeout_s <= 300:
        p.error('timeout-s must be in (0,300]')
    try:
        record = run(a)
    except Exception as error:
        record = dict(verdict='INCONCLUSIVE', failure_class='HARNESS',
                      error_type=type(error).__name__, cause=str(error)[:160])
    with open(a.out, 'x') as f:
        json.dump(record, f, sort_keys=True, indent=2)
        f.write('\n')
    return 0 if record['verdict'] == 'PASS' else 2

if __name__ == '__main__':
    raise SystemExit(main())
