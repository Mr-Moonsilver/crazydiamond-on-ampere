#!/usr/bin/env python3
"""Offline replay of native_window (old vs new) on a retained box reuse record.

The record is the native39 lifecycle held-a run 1 reuse (the box's e3_q18a_lifecycle output).  It
predates the per-call stamps, so two client stamps are reconstructed from the record's own client
stamps, each a sound bound on the client's monotonic clock:
  begin_recv_ns := request.send_at_ns.  send_at_ns is time.monotonic_ns() taken in the request
      thread after prefill_begin returned (e3_decode_correctness.py:775), so it is >= the true
      receive stamp; the reconstructed begin bracket contains the true one.
  end_send_ns := request.send_at_ns + first_advance_read_timeout_s.  prefill_end is called only
      after _wait_first_advance returned.  It returns on its monotonic deadline (set after
      send_at, +read_s) or on state.error; here error is 'read_timeout', which the urlopen socket
      timeout (read_s, monotonic, started after send_at) raises.  Either way the call is sent
      >= send_at + read_s, so the reconstructed end bracket contains the true one.
Nothing else in the record is changed.  The old module is graded in this process's proc view.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def clean(result):
    return {k: v for k, v in result.items() if k not in ('view', 'binding', 'group')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--harness', type=Path, required=True)
    p.add_argument('--old', type=Path, required=True, help='pre-fix e3_native_performance.py')
    p.add_argument('--record', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    sys.path[:0] = [str(a.harness), str(a.harness.parent / 'patches/s5_engine')]
    new = load('e3_native_performance', a.harness / 'e3_native_performance.py')
    old = load('e3_native_performance_old', a.old)
    data = a.record.read_bytes()
    lifecycle = json.loads(data)
    reuse = lifecycle['reuse']
    epoch = reuse['end']['engine_epoch']
    request = reuse['request']
    read_ns = int(reuse['first_advance_read_timeout_s'] * 1_000_000_000)
    rebuilt = copy.deepcopy(reuse)
    rebuilt['client_bracket'].update(begin_recv_ns=request['send_at_ns'],
                                     end_send_ns=request['send_at_ns'] + read_ns)
    out = dict(
        schema='reuse-clock-replay/1',
        record=str(a.record), record_sha256=hashlib.sha256(data).hexdigest(),
        new_module_sha256=hashlib.sha256((a.harness / 'e3_native_performance.py').read_bytes()).hexdigest(),
        old_module_sha256=hashlib.sha256(a.old.read_bytes()).hexdigest(),
        engine_epoch=epoch, family=reuse['family'],
        reconstructed=dict(begin_recv_ns=rebuilt['client_bracket']['begin_recv_ns'],
                           end_send_ns=rebuilt['client_bracket']['end_send_ns'],
                           client_bracket=rebuilt['client_bracket'],
                           begin_bracket_ns=rebuilt['client_bracket']['begin_recv_ns'] - rebuilt['client_bracket']['start_ns'],
                           end_bracket_ns=rebuilt['client_bracket']['end_ns'] - rebuilt['client_bracket']['end_send_ns']),
        engine_samples=dict(begin=reuse['begin']['snapshot'].get('observed_clock_ns'),
                            end=reuse['end']['raw_journal_evidence'].get('observed_clock_ns'),
                            engine_view={k: reuse['begin']['snapshot']['clock'].get(k) for k in ('boot_id', 'time_namespace')}),
        replay_view=new.prove({}, 1, 1)['client_clock'],
        new_reconstructed=clean(new.native_window(rebuilt, reuse['family'], epoch)),
        new_as_retained=clean(new.native_window(reuse, reuse['family'], epoch)),
        old_reconstructed=clean(old.native_window(rebuilt, reuse['family'], epoch)),
        old_as_retained=clean(old.native_window(reuse, reuse['family'], epoch)),
        completion=dict(reuse_complete=reuse.get('complete'), error=reuse.get('error'),
                        request_error=request.get('error'), ttft_s=request.get('ttft_s'),
                        first_at_ns=request.get('first_at_ns'), finish_at_ns=request.get('finish_at_ns'),
                        lifecycle_verdict=lifecycle.get('verdict'), lifecycle_missing=lifecycle.get('missing')))
    nr = out['new_reconstructed']
    clock_new = all(nr['client_clock'][n]['proved'] for n in ('begin', 'end')) and not any(
        r.startswith('client-engine-clock') for r in nr['reasons'])
    clock_old = not any(r.startswith('client-engine-clock') for r in out['old_reconstructed']['reasons'])
    out['summary'] = dict(new_clock_proved=clock_new, old_clock_proved=clock_old,
                          new_as_retained_clock_proved=not any(r.startswith('client-engine-clock') for r in out['new_as_retained']['reasons']),
                          completion_failed=reuse.get('complete') is not True,
                          new_non_clock_reasons=[r for r in nr['reasons'] if not r.startswith('client-engine-clock')])
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
    print(json.dumps(out['summary'], sort_keys=True))


if __name__ == '__main__':
    main()
