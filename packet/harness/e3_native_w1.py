#!/usr/bin/env python3
"""Unchanged standing W1 requests/windows over actual native session receipts.

No pause_full/pause_end aliases or probe counters. The source dispatch census
is sampled at queried, drained native barriers; every nonempty dispatched batch
inside the client window must be actual one-token FULL graph decode. All fixed
repeats and any first refusal remain in the artifact; no survivor summary.
"""
import json
import os
from pathlib import Path
import sys
import threading
import time

import e3_w1_decode_driver as w1
from e3_native_runtime_client import NativeRuntimeClient
from e3_native_qualification_driver import exception_details


def interval_proof(begin, end, client, route):
    missing = []
    try:
        first = client.workers(begin.get('barrier'))
        last = client.workers(end.get('barrier'))
    except ValueError as exc:
        return {'complete': False, 'reasons': [str(exc)], 'ranks': {}}
    ranks = {}
    for rank in first:
        a, b = first[rank], last[rank]
        reasons = []
        for key in ('engine_epoch', 'storage_epoch', 'worker_incarnation', 'route_generation'):
            if a.get(key) is None or a.get(key) != b.get(key):
                reasons.append('identity:' + key)
        if a.get('route') != route or b.get('route') != route:
            reasons.append('route')
        for row in (a, b):
            if (row.get('prefill_requests') != 0 or row.get('decode_eligible') != 8
                    or row.get('live_decode_requests') != 8
                    or row.get('barrier_clean') is not True
                    or (row.get('live_decode') or {}).get('available') is not True):
                reasons.append('live-cohort-or-drain')
        ca, cb = [(r.get('live_decode') or {}).get('census') or {} for r in (a,b)]
        delta = {}
        for key in ('dispatches', 'decode_dispatches', 'non_decode_dispatches',
                    'full_graph_dispatches', 'dispatch_errors'):
            x,y = ca.get(key),cb.get(key)
            if type(x) is not int or type(y) is not int or y < x:
                reasons.append('counter:' + key)
            else:
                delta[key] = y-x
        if (delta.get('dispatches',0) < 1
                or delta.get('decode_dispatches') != delta.get('dispatches')
                or delta.get('full_graph_dispatches') != delta.get('dispatches')
                or delta.get('non_decode_dispatches') != 0 or delta.get('dispatch_errors') != 0):
            reasons.append('dispatch-not-full-decode')
        if ca.get('rejected') or cb.get('rejected'):
            reasons.append('source-census-rejected')
        ranks[rank] = {'delta': delta, 'reasons': sorted(set(reasons))}
        missing.extend('rank%d:%s' % (rank,r) for r in reasons)
    return {'complete': not missing, 'reasons': sorted(set(missing)), 'ranks': ranks,
            'basis': 'actual unpadded CPU batch at graph_hooks.check_route, monotonic counters; queried native barrier'}


def run(cfg, url, model, client=None):
    client = client or NativeRuntimeClient(url, cfg.run_id, timeout_s=cfg.controller_timeout_s)
    requests, meta = w1.build_requests(cfg.workload)
    _, hashes, digest = w1.build_request_set(model, requests, cfg.workload)
    repeats, abort = [], None
    for repeat, route in enumerate(cfg.workload.route_plan):
        row = {'repeat': repeat, 'route': route, 'valid': False, 'reasons': [],
               'streams': [], 'native_receipts': {}, 'metrics': {}}
        repeats.append(row)
        if abort:
            row.update(not_run=True, reasons=['earlier-refusal:' + abort])
            continue
        states, stop, window = [], threading.Event(), None
        try:
            row['native_receipts']['prepare'] = client.prepare('full', 'full-prefill-solo')
            for i,messages in enumerate(requests):
                state = w1.StreamState(i,messages,meta[i],w1.request_body(model,messages,cfg.workload))
                states.append(state)
                state.thread = threading.Thread(target=w1.stream_reader,args=(state,cfg,url,stop),daemon=True)
                state.thread.start()
            deadline = time.monotonic()+cfg.workload.open_timeout_s
            for state in states:
                state.opened.wait(max(0,deadline-time.monotonic()))
            if any(s.first_at_ns is None or s.finish_at_ns is not None or s.error is not None for s in states):
                raise ValueError('native-w1-cohort-not-live')
            if route == 'split':
                row['native_receipts']['enter'] = client.enter('split-w1-decode')
            before = client.pause()
            row['native_receipts']['begin'] = before
            row['native_receipts']['resume'] = client.resume()
            start = time.monotonic_ns()
            deadline_ns = start+int(cfg.workload.window_s*1e9)
            while time.monotonic_ns() < deadline_ns:
                if any(s.error is not None or s.finish_at_ns is not None for s in states):
                    raise ValueError('native-w1-stream-ended-in-window')
                time.sleep(min(.02,max(0,(deadline_ns-time.monotonic_ns())/1e9)))
            end = time.monotonic_ns()
            window = {'measured': True,'start_ns': start,'end_ns': end,'wall_s':(end-start)/1e9}
            after = client.pause()
            row['native_receipts']['end'] = after
            row['interval'] = interval_proof(before,after,client,route)
            row['reasons'].extend(row['interval']['reasons'])
            row['native_receipts']['leave_full'] = client.leave()
            row['native_receipts']['cleanup_resume'] = client.resume()
        except Exception as exc:
            row['error'] = exception_details(exc,client.control)
            row['reasons'].append('native-w1-refusal:' + row['error']['error_message'])
            abort = row['error']['error_message']
        finally:
            row['streams'], _ = w1._finalize_streams(cfg,states,stop,window,time.monotonic_ns())
            row['window'] = window
            row['metrics'] = w1._repeat_metrics(row['streams'],window)
            row['exposure'] = w1._exposure(row['streams'],cfg.workload)
            row['reasons'].extend(row['exposure']['reasons'])
            if any(r.get('drain_timeout') for r in row['streams']):
                row['reasons'].append('client-reader-drain-incomplete')
            if [s.get('request_sha256') for s in row['streams']] != hashes:
                row['reasons'].append('request-bytes-changed')
            row['valid'] = not row['reasons'] and row.get('interval',{}).get('complete') is True
    complete = len(repeats)==cfg.workload.repeats and all(r['valid'] for r in repeats)
    routes = {}
    for route in ('full','split'):
        rows = [r for r in repeats if r['route']==route]
        valid = bool(rows) and all(r['valid'] for r in rows)
        values = [r['metrics'].get('aggregate_tokens_per_s') for r in rows]
        gaps = [g for r in rows for s in r['streams'] for g in s.get('window',{}).get('gaps_ms',[])]
        routes[route] = {'valid_repeats':[r['repeat'] for r in rows if r['valid']],
            'aggregate_tokens_per_s_median': w1._pct(values,.5) if valid and all(type(v) in (int,float) for v in values) else None,
            'gap_ms_p99':w1._pct(gaps,.99) if valid else None}
    return {'schema':2,'mode':'native-overlap','config':{'WORKLOAD_SHA256':cfg.workload.manifest_sha256,
        'REPEATS':cfg.workload.repeats,'request_set_sha256':digest},'repeats':repeats,
        'summary':{'verdict':'COMPLETE' if complete else 'INCONCLUSIVE','routes':routes,
            'totals':{'required':cfg.workload.repeats,'attempted':len(repeats),
                      'valid':sum(r['valid'] for r in repeats),'invalid':sum(not r['valid'] for r in repeats),'contract':0},
            'all_declared_repeats_retained':True,'filtered_invalid':False,'retry_selected':False,'picked':False}}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv)!=3:
        return 2
    url,model,out = argv
    if Path(out).exists():
        return 2
    try:
        cfg = w1.config_from_env(os.environ,out)
        result = run(cfg,url,model)
    except Exception as exc:
        result = {'schema':2,'verdict':'INCONCLUSIVE','error':exception_details(exc)}
    with open(out,'x') as handle:
        json.dump(result,handle,sort_keys=True)
    complete = result.get('summary',{}).get('verdict')=='COMPLETE'
    print('NATIVE_W1 '+('COMPLETE' if complete else 'INCONCLUSIVE'))
    return 0 if complete else 2


if __name__=='__main__':
    raise SystemExit(main())
