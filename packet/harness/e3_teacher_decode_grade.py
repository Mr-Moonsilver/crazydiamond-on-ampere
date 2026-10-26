#!/usr/bin/env python3
"""Frozen rule v3 at fixed decode positions; never selects/retries a control.

Input must include worker-observed route/position proof, not client arrival
estimates. Launch-config diagnostics and free-running equality are separate.
This pure grader supplies no fabricated worker evidence or baseline results.
"""
import hashlib
import json
import math
from pathlib import Path

from teacher_forced import RULE, compare

BASELINE_SHA256 = '1f1d7c23c66bf6d71aba3baef5e9f89f9fd12c297a70641f7cb397b100a2eae3'
FROZEN_RULE = {'version':3, 'TOL_WIN':0.375, 'TOL_TOP5':0.5, 'NEAR_TIE':0.375}
RUNS = ('full-A1','full-A2','split-B','full-C')
STEPS, STREAMS = 256, 8


def baseline_record(path):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != BASELINE_SHA256:
        raise ValueError('teacher baseline source differs')
    return {'sha256': BASELINE_SHA256, 'rule': dict(FROZEN_RULE),
            'scope':'q15 prefill baseline; decode-path controls are measured again, not substituted'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class ForcedTokenMismatch(ValueError):
    pass


def _rows(run, route, positions, prefix_digests, ordered=False):
    _require(type(run) is dict and run.get('route') == route, 'route differs')
    _require(run.get('lifecycle_verdict') == 'PASS', 'lifecycle unproved')
    _require(run.get('prefix_digests') == prefix_digests, 'forced prefixes differ')
    rows=run.get('positions')
    _require(type(rows) is list and len(rows)==STREAMS*STEPS, 'position coverage incomplete')
    observed={}
    for row in rows:
        _require(type(row) is dict, 'position row invalid')
        stream,position=row.get('stream'),row.get('position')
        _require(type(stream) is int and 0<=stream<STREAMS and type(position) is int
                 and position in positions, 'position outside declared set')
        key=(stream,position)
        _require(key not in observed, 'duplicate position')
        if row.get('forced_token_exact') is False:
            raise ForcedTokenMismatch('worker observed forced token mismatch')
        _require(row.get('worker_route')==route and row.get('full_graph') is True
                 and row.get('decode_only') is True and row.get('forced_token_exact') is True,
                 'decode route or forced token unproved')
        _require(type(row.get('route_generation')) is int and row['route_generation']>=0,
                 'route generation missing')
        top=row.get('top')
        _require(type(top) is dict and len(top)>=20 and len(top)<=21
                 and all(type(k) is str and k.startswith('token_id:') and k[9:].isdigit()
                         and type(v) in (int,float) and math.isfinite(v) for k,v in top.items()),
                 'original top20 logprobs incomplete')
        if ordered:
            order=row.get('top_order')
            _require(type(order) is list and len(order)==len(top)
                     and all(type(k) is str for k in order)
                     and len(set(order))==len(order) and set(order)==set(top),
                     'original top20 order incomplete')
            # Frozen compare resolves equal values by insertion order. Restore
            # the response order after sort_keys JSON, without changing rule v3.
            top={k:top[k] for k in order}
        observed[key]={'top':top}
    _require(set(observed)=={(s,p) for s in range(STREAMS) for p in positions},
             'position set differs')
    return observed


def grade(evidence):
    out={'schema':1,'rule':dict(FROZEN_RULE),'verdict':'INCONCLUSIVE',
         'baseline_sha256':BASELINE_SHA256,'coverage_per_stream':0,
         'comparisons':{},'missing':[], 'failures':[]}
    try:
        _require(RULE==FROZEN_RULE, 'teacher rule drift')
        _require(type(evidence) is dict and evidence.get('schema')=='s5-teacher-decode/1',
                 'teacher evidence schema differs')
        _require(evidence.get('baseline_sha256')==BASELINE_SHA256, 'baseline identity missing')
        _require(evidence.get('rule')==FROZEN_RULE, 'declared rule differs')
        positions=evidence.get('fixed_positions')
        _require(type(positions) is list and len(positions)==STEPS
                 and all(type(p) is int and p>=1 for p in positions)
                 and positions==list(range(positions[0],positions[0]+STEPS)),
                 'fixed contiguous decode positions missing')
        prefixes=evidence.get('prefix_digests')
        _require(type(prefixes) is list and len(prefixes)==STREAMS
                 and all(type(p) is str and len(p)==64 and all(c in '0123456789abcdef' for c in p)
                         for p in prefixes), 'fixed prefix identity missing')
        runs=evidence.get('runs')
        _require(type(runs) is dict and set(runs)==set(RUNS), 'fixed run set differs')
        order_basis=evidence.get('top_order_basis')
        _require(order_basis in (None,'response-insertion-order/1'), 'top order basis unsupported')
        out['top_order_basis']=order_basis or 'legacy-object-order; original tie order unproved'
        mapped={name:_rows(runs[name], 'split' if name=='split-B' else 'full',positions,prefixes,order_basis is not None)
                for name in RUNS}
    except ForcedTokenMismatch as exc:
        out['verdict']='FAIL'; out['failures']=[str(exc)]
        return out
    except (ValueError, TypeError, KeyError) as exc:
        out['missing']=[str(exc) if isinstance(exc,ValueError) else 'invalid evidence shape']
        return out
    # Compare expects arbitrary sortable probe keys; tuple coordinates bind
    # the unchanged rule to every stream/position, with no tolerances added.
    results=out['comparisons']
    for label,right in (('control_pre','full-A2'),('control_post','full-C'),('split','split-B')):
        results[label]=compare(mapped['full-A1'],mapped[right])
    out['coverage_per_stream']=STEPS
    out['comparisons']['split']['comparable']=not any(results[k]['fail_probes'] for k in ('control_pre','control_post'))
    if not results['split']['comparable']:
        out['missing']=['same-instrument control instability under frozen rule v3']
    elif results['split']['fail_probes']:
        out['verdict']='FAIL';out['failures']=['split outside frozen rule v3']
    else:
        out['verdict']='PASS'
    return out
