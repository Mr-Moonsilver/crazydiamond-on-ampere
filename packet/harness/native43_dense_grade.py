#!/usr/bin/env python3
"""Strict diagnostic pairing; exact boundary hashes, no throughput PASS gate."""
import argparse
import json
from pathlib import Path

PREFIX = 'flash-s5 native43_qsa '
FRONTIER = 101200


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
                    if len(rows) > 3000:
                        raise ValueError('native43-layer-records-over-cap')
    return rows


def grade(current, dense, current_layers, dense_layers):
    missing, failures, timings = [], [], {}
    for policy, client, layers in [('current', current, current_layers), ('dense', dense, dense_layers)]:
        if client.get('verdict') != 'PASS':
            missing.append(policy + ':client-cutoff-or-cleanup-unproved')
        if client.get('cohort_streams') != 0 or client.get('fresh') is not True:
            failures.append(policy + ':not-fresh-no-cohort')
        if not layers:
            missing.append(policy + ':layer-receipts')
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
        if len(boundary) != 24 or any(sum(k[0] == rank for k in boundary) != 6 for rank in range(4)):
            missing.append(policy + ':six-sparse-layers-per-rank-boundary')
        for row in boundary.values():
            for name in ('selection_evidence', 'output_evidence'):
                if not isinstance(row.get(name), dict) or not row[name].get('sha256'):
                    missing.append(policy + ':' + name)
        timings[policy] = dict(boundary=boundary, timed_chunks=by_chunk)
    for name in ('prompt_sha256', 'paired_salt_id'):
        if not current.get(name) or current.get(name) != dense.get(name):
            failures.append('pair:' + name)
    left, right = timings['current']['boundary'], timings['dense']['boundary']
    if set(left) != set(right):
        missing.append('pair:boundary-layer-keys')
    for key in set(left) & set(right):
        for name in ('selection_evidence', 'output_evidence'):
            a, b = left[key].get(name) or {}, right[key].get(name) or {}
            if a.get('sha256') and b.get('sha256') and any(a.get(k) != b.get(k)
                    for k in ('sha256', 'shape', 'dtype', 'bytes')):
                failures.append('pair:%s:rank%d:%s' % (name, key[0], key[1]))
    return dict(schema='native43-dense-boundary-grade/1',
        verdict='FAIL' if failures else 'INCONCLUSIVE' if missing else 'PASS',
        failure_class='ENGINE' if failures else 'HARNESS' if missing else None,
        failures=sorted(set(failures)), missing=sorted(set(missing)), speed_claim=False,
        exact_boundary_evidence=not failures and not missing,
        performance_basis='operation-open-to-release per rank; final readback chunk excluded; no client/engine clock subtraction',
        arms={policy: dict(timed_chunks=[dict(rank=k[0],transaction=k[1],**v)
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
