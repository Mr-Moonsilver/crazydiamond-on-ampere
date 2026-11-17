#!/usr/bin/env python3
"""Timing/route/close diagnostic; all numerical comparisons informational."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path
from native43_boundary_clock import prove

PREFIX = 'flash-s5 native43_qsa '
FRONTIER = 101200
MAX_LAYER_RECORDS = 3000
EXPECTED_LAYER_RECORDS = 2400  # 100 chunks * 6 sparse layers * 4 ranks, per arm


def read_layers(paths):
    rows = []
    for path in paths:
        with open(path) as handle:
            for line in handle:
                if PREFIX in line:
                    if len(line) > 32768:
                        raise ValueError('native43-layer-line-over-cap')
                    row = json.loads(line.split(PREFIX, 1)[1])
                    rows.append(row)
                    if len(rows) > MAX_LAYER_RECORDS:
                        raise ValueError('native43-layer-records-over-cap')
    return rows



def certified_client(client):
    """Recheck source packets; a cached PASS flag is not the stop proof."""
    checkpoint = client.get('checkpoint') or {}
    closure = client.get('closure') or {}
    status = client.get('final_status') or {}
    bracket = client.get('final_clock') or {}
    diagnostic = status.get('native43_boundary') or {}
    releases, closed = checkpoint.get('worker_releases'), closure.get('rows')
    if not (checkpoint.get('exact') is True and checkpoint.get('chunk_end') == FRONTIER
            and closure.get('complete') is True and closure.get('chunk_end') == FRONTIER
            and closure.get('transaction') == checkpoint.get('transaction')
            and (closure.get('source_leases') or {}).get('clean') is True
            and not closure.get('pp_receipts_pending')
            and diagnostic.get('stop_at') == FRONTIER and diagnostic.get('cancel_started') is True
            and diagnostic.get('checkpoint') == checkpoint and status.get('last_native_closure') == closure
            and status.get('route') == 'full' and not status.get('transition_pending')
            and not status.get('return_pending')
            and type(releases) is list and len(releases) == 4
            and type(closed) is list and len(closed) == 4):
        return False
    if (any(type(r) is not dict for r in releases + closed)
            or {r.get('worker_rank') for r in releases} != {0,1,2,3}
            or {r.get('worker_rank') for r in closed} != {0,1,2,3}
            or any(r.get('released') is not True for r in releases)
            or any(r.get('typed') != 'CancelAck' or r.get('closed') is not True
                   or r.get('request_closed') is not True or r.get('disposition') != 'released'
                   or type(r.get('report')) is not dict or r['report'].get('pending_op_seqs') != []
                   for r in closed)):
        return False
    return prove(status, bracket.get('send_ns'), bracket.get('recv_ns'))['proved']



def bounded_output_difference(left, right):
    a, b = left.get('sample_values'), right.get('sample_values')
    result = dict(role='INFORMATIONAL', observed=False,
        scope='first min(256,numel) flattened output elements; not full-tensor maximum',
        relative_reference='max(abs(current),1e-12)', full_tensor_coverage=False)
    if (type(a) is not list or type(b) is not list or not a or len(a) != len(b)
            or len(a) > 256 or left.get('shape') != right.get('shape')):
        return result
    if any(type(x) not in (int,float) for x in a+b):
        return result
    pairs = [(x,y) for x,y in zip(a,b) if math.isfinite(x) and math.isfinite(y)]
    result.update(observed=True, elements=len(a), nonfinite_pairs=len(a)-len(pairs),
        full_tensor_coverage=left.get('full_tensor_coverage') is True
                             and right.get('full_tensor_coverage') is True,
        max_abs=max((abs(x-y) for x,y in pairs),default=None),
        max_rel=max((abs(x-y)/max(abs(x),1e-12) for x,y in pairs),default=None))
    return result


def grade(current, dense, current_layers, dense_layers):
    missing, failures, timings = [], [], {}
    for policy, client, layers in [('current', current, current_layers), ('dense', dense, dense_layers)]:
        if client.get('verdict') != 'PASS' or not certified_client(client):
            missing.append(policy + ':client-cutoff-or-cleanup-unproved')
        if client.get('cohort_streams') != 0 or client.get('fresh') is not True:
            failures.append(policy + ':not-fresh-no-cohort')
        if not layers:
            missing.append(policy + ':layer-receipts')
        if len(layers) != EXPECTED_LAYER_RECORDS:
            missing.append(policy + ':expected-2400-layer-records')
        keys = [(r.get('rank'),r.get('transaction'),r.get('layer')) for r in layers]
        if len(set(keys)) != len(keys):
            failures.append(policy + ':duplicate-layer-record')
        boundary = {}
        by_chunk = {}
        for row in layers:
            if row.get('policy') != policy:
                failures.append(policy + ':startup-policy-not-executed')
            if row.get('gpu_timing_complete') is not True:
                missing.append(policy + ':gpu-timing-coverage')
            if row.get('chunk_end', FRONTIER+1) > FRONTIER:
                failures.append(policy + ':cutoff-overrun')
            if policy == 'dense' and row.get('route') != 'dense_dma':
                failures.append('dense:sparse-route-executed')
            if row.get('chunk_end') == FRONTIER:
                key = (row.get('rank'), row.get('layer'))
                if key in boundary:
                    failures.append(policy + ':duplicate-boundary-layer')
                boundary[key] = row
            if row.get('timing_excluded') is None:
                key = (row.get('rank'), row.get('transaction'))
                by_chunk.setdefault(key, dict(chunk_start=row['chunk_start'], chunk_end=row['chunk_end'],
                    operation_elapsed_s=(row['operation_released_ns']-row['operation_open_ns'])/1e9))
        for rank in range(4):
            ends = {row['chunk_end'] for key,row in by_chunk.items() if key[0] == rank}
            if not {97152,98176,99200,100224} <= ends:
                missing.append(policy + ':rank%d:timed-boundary-chunks' % rank)
        if len(boundary) != 24 or any(sum(k[0] == rank for k in boundary) != 6 for rank in range(4)):
            missing.append(policy + ':six-sparse-layers-per-rank-boundary')
        timings[policy] = dict(boundary=boundary, timed_chunks=by_chunk,
            route_counts=dict(Counter(row.get('route', 'NOT_OBSERVED') for row in layers)),
            layer_records=len(layers))
    for name in ('prompt_sha256', 'paired_salt_id'):
        if not current.get(name) or current.get(name) != dense.get(name):
            failures.append('pair:' + name)
    left, right = timings['current']['boundary'], timings['dense']['boundary']
    comparisons, first_stage0 = [], []
    for key in sorted(set(left) | set(right)):
        a, b = left.get(key) or {}, right.get(key) or {}
        entry = dict(rank=key[0],layer=key[1],role='INFORMATIONAL')
        for name in ('selection_evidence','output_evidence'):
            av, bv = a.get(name) or {}, b.get(name) or {}
            observed = bool(av.get('sha256') and bv.get('sha256'))
            entry[name] = dict(observed=observed,bitwise_equal=(all(av.get(k)==bv.get(k)
                for k in ('sha256','shape','dtype','bytes')) if observed else None))
        entry['bounded_output_difference'] = bounded_output_difference(
            a.get('output_evidence') or {}, b.get('output_evidence') or {})
        comparisons.append(entry)
        if key[0] in (0,1) and a.get('first_qsa_in_stage') is True and b.get('first_qsa_in_stage') is True:
            first_stage0.append(dict(rank=key[0],layer=key[1],role='INFORMATIONAL',
                **entry['selection_evidence']))
    return dict(schema='native43-dense-boundary-grade/1',
        verdict='FAIL' if failures else 'INCONCLUSIVE' if missing else 'PASS',
        failure_class='HARNESS' if failures or missing else None,
        failures=sorted(set(failures)), missing=sorted(set(missing)), speed_claim=False,
        numerical_evidence_role='INFORMATIONAL', numerical_comparisons=comparisons,
        stage0_first_qsa_selection=first_stage0,
        record_bound=dict(per_arm_cap=MAX_LAYER_RECORDS,expected_per_arm=EXPECTED_LAYER_RECORDS,
                          expected_headroom=MAX_LAYER_RECORDS-EXPECTED_LAYER_RECORDS,
                          headroom_over_expected_pct=25),
        performance_basis='operation-open-to-release per rank; final readback chunk excluded; no client/engine clock subtraction',
        arms={policy: dict(route_counts=t['route_counts'],layer_records=t['layer_records'],
                record_headroom=MAX_LAYER_RECORDS-t['layer_records'],timed_chunks=[dict(rank=k[0],transaction=k[1],**v)
                for k,v in sorted(t['timed_chunks'].items())],
                boundary=[dict(rank=k[0],layer=k[1],selection_evidence=v.get('selection_evidence'),
                               output_evidence=v.get('output_evidence')) for k,v in sorted(t['boundary'].items())])
              for policy,t in timings.items()})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for policy in ('current', 'dense'):
        p.add_argument('--' + policy, type=Path, required=True)
        p.add_argument('--' + policy + '-log', type=Path, action='append', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    record = grade(json.loads(a.current.read_text()), json.loads(a.dense.read_text()),
                   read_layers(a.current_log), read_layers(a.dense_log))
    with a.out.open('x') as f:
        json.dump(record, f, sort_keys=True, indent=2);f.write('\n')
    return {'PASS':0,'FAIL':1,'INCONCLUSIVE':2}[record['verdict']]

if __name__ == '__main__':
    raise SystemExit(main())
