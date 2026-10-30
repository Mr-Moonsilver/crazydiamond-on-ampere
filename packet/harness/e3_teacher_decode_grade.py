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
LEGACY_EVIDENCE_SCHEMA='s5-teacher-decode/1'
EVIDENCE_SCHEMA='s5-teacher-decode/2'
ROW_SCHEMA='s5-teacher-rows/1'
ROW_MAPPING_SCHEMA='s5-teacher-row-mapping/1'
ROW_BASIS='sampler-invocation/1'
MAX_ROWS=2048
MAX_CALLS=2048 * 2  # two replicated sampler ranks; each worker remains capped at 2048
MAX_DEFECTS=32
# PP1 TP ranks 2 and 3 both own a sampler. Without batch-sharded sampling
# (the pinned default; full11 receipts carry 2048 rows on each) both report
# every coordinate, so a run holds up to one MAX_ROWS block per sampler rank.
SAMPLER_RANKS=2


def logprob_digest(trace_sha, position, top, order):
    _require(type(top) is dict and type(order) is list
             and len(order) == len(top) and len(set(order)) == len(order)
             and set(order) == set(top), 'logprob order missing for row binding')
    payload = [trace_sha, position, [[key, float(top[key])] for key in order]]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


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


class RowMappingDefect(ValueError):
    """Collector defect detected before the frozen numerical comparison."""
    def __init__(self, defects):
        self.defects=list(defects)
        super().__init__('; '.join(self.defects[:3]) or 'row mapping defect')


def _int_list(value):
    return type(value) is list and all(type(v) is int for v in value)


def _bool_list(value):
    return type(value) is list and all(type(v) is bool for v in value)


def _decode_only(snapshot):
    """A one-token, no-prefill, no-draft decode proven from the captured batch."""
    if type(snapshot) is not dict:
        return False
    scheduled=snapshot.get('num_scheduled_tokens');prefilling=snapshot.get('is_prefilling_np')
    return (snapshot.get('num_draft_tokens')==0
            and snapshot.get('num_tokens')==snapshot.get('num_reqs')
            and _int_list(scheduled) and len(scheduled)>0 and all(v==1 for v in scheduled)
            and _bool_list(prefilling) and len(prefilling)==len(scheduled)
            and not any(prefilling))


def _snapshot_ok(snapshot):
    if type(snapshot) is not dict:
        return False
    return (all(type(snapshot.get(k)) is int for k in ('num_reqs','num_reqs_after_padding',
                'num_tokens','num_tokens_after_padding','num_draft_tokens'))
            and all(_int_list(snapshot.get(k)) for k in ('idx_mapping','expanded_idx_mapping',
                'logits_indices','cu_num_logits','expanded_local_pos','logit_input_positions',
                'num_scheduled_tokens'))
            and _bool_list(snapshot.get('is_prefilling_np')))


def _canonical_decode(snapshot, local=False):
    """Pinned no-draft relation: cu is arange and every request has one logit.

    The global batch also selects logits_indices == arange; a TP-sharded local
    batch keeps the same logits but selects them by global query row.
    """
    reqs=snapshot.get('num_reqs')
    if type(reqs) is not int or reqs<1:
        return False
    return (snapshot['cu_num_logits']==list(range(reqs+1))
            and len(snapshot['logits_indices'])==reqs
            and (local or snapshot['logits_indices']==list(range(reqs)))
            and snapshot['expanded_idx_mapping']==snapshot['idx_mapping']
            and len(snapshot['expanded_local_pos'])==reqs
            and all(v==0 for v in snapshot['expanded_local_pos'])
            and len(snapshot['logit_input_positions'])==reqs
            and snapshot['num_draft_tokens']==0)


def _row_mapping(evidence, runs):
    """Validate every SSE coordinate's physical sampler record before comparison."""
    cohort=evidence.get('cohort_trace_digests')
    if not (type(cohort) is list and len(cohort)==STREAMS
            and all(type(d) is str and len(d)==64 and all(c in '0123456789abcdef' for c in d)
                    for d in cohort) and len(set(cohort))==STREAMS):
        raise RowMappingDefect(['cohort trace digests missing'])
    row_schema=evidence.get('row_schema')
    if row_schema!=ROW_SCHEMA or evidence.get('row_mapping_basis')!=ROW_BASIS:
        raise RowMappingDefect(['row mapping basis differs'])
    measured=set(evidence['fixed_positions'])
    summary={'schema':ROW_MAPPING_SCHEMA,'basis':ROW_BASIS,'records':0,'calls':0,
             'physical_layout_differs':False,'ranks':{},'defects':[]}
    defects=summary['defects']
    for name in RUNS:
        run=runs[name]
        route='split' if name=='split-B' else 'full'
        mapping=run.get('row_mapping')
        if type(mapping) is not dict:
            defects.append(name+': row mapping absent')
            continue
        if mapping.get('schema')!=ROW_MAPPING_SCHEMA or mapping.get('basis')!=ROW_BASIS:
            defects.append(name+': row mapping schema differs')
        if mapping.get('overflow') is not False:
            defects.append(name+': row mapping overflow')
        reported=mapping.get('defects')
        if type(reported) is not list:
            defects.append(name+': row mapping defects invalid')
        else:
            # The client already refused these before comparison; keep its
            # explicit reason instead of reporting only downstream coverage.
            defects.extend(name+': '+str(reason) for reason in reported)
        calls=mapping.get('calls');records=mapping.get('records')
        if type(calls) is not list or len(calls)>MAX_CALLS:
            defects.append(name+': sampler calls missing or unbounded');continue
        if type(records) is not list or len(records)>MAX_ROWS*SAMPLER_RANKS:
            defects.append(name+': physical records missing or unbounded');continue
        repeat=run.get('repeat');epoch=run.get('engine_epoch')
        generation=run.get('route_generation');storage=run.get('storage_epoch')
        if not all(type(v) is int for v in (repeat,epoch,generation,storage)):
            defects.append(name+': run identity incomplete');continue
        callmap={}
        for call in calls:
            if type(call) is not dict:
                defects.append(name+': sampler call invalid');continue
            rank,seq=call.get('rank'),call.get('sampler_call_seq')
            if not (type(rank) is int and rank in (2,3) and type(seq) is int and seq>=1):
                defects.append(name+': sampler call identity invalid');continue
            if (rank,seq) in callmap:
                defects.append(name+': duplicate sampler call');continue
            if (call.get('engine_epoch')!=epoch or call.get('route')!=route
                    or call.get('route_generation')!=generation
                    or call.get('storage_epoch')!=storage):
                defects.append(name+': sampler call generation differs');continue
            graph=call.get('graph')
            if not (type(graph) is dict and graph.get('cg_mode') in ('none','piecewise','full')
                    and type(graph.get('num_tokens')) is int
                    and (graph.get('num_reqs') is None or type(graph.get('num_reqs')) is int)):
                defects.append(name+': graph descriptor invalid');continue
            if call.get('descriptor_stale') is not False or call.get('descriptor_matches_batch') is not True:
                defects.append(name+': graph descriptor unproved');continue
            glob=call.get('global_snapshot');local=call.get('local_snapshot')
            if not (_snapshot_ok(glob) and _snapshot_ok(local)):
                defects.append(name+': batch snapshot incomplete');continue
            if glob != local or call.get('global_cohort_order') != call.get('local_cohort_order'):
                defects.append(name+': batch-sharded sampling is not qualified');continue
            if not (_canonical_decode(glob) and _canonical_decode(local, local=True)):
                defects.append(name+': sampler batch is not a no-draft decode');continue
            if (graph['num_tokens']!=glob['num_tokens_after_padding']
                    or (graph['num_reqs'] is not None
                        and graph['num_reqs']!=glob['num_reqs_after_padding'])):
                defects.append(name+': descriptor/batch padding differs');continue
            order=call.get('global_cohort_order')
            if not (type(order) is list and len(order)==len(glob['idx_mapping'])
                    and all(value is None or value in cohort for value in order)):
                defects.append(name+': global cohort order invalid');continue
            present=[value for value in order if value is not None]
            if len(set(present))!=len(present):
                defects.append(name+': cohort member duplicated in a batch');continue
            local_order=call.get('local_cohort_order')
            if not (type(local_order) is list and len(local_order)==len(local['idx_mapping'])
                    and all(value is None or value in cohort for value in local_order)):
                defects.append(name+': local cohort order invalid');continue
            callmap[(rank,seq)]=call
        summary['calls']+=len(callmap)
        numeric={}
        for row in run.get('positions',[]):
            key=(row.get('stream'),row.get('position')) if type(row) is dict else None
            if key is None or key in numeric:
                defects.append(name+': numerical coordinate invalid');continue
            numeric[key]=row
        joined={};logical={};rank_stream={}
        for record in records:
            if type(record) is not dict:
                defects.append(name+': physical record invalid');continue
            rank,seq=record.get('rank'),record.get('sampler_call_seq')
            call=callmap.get((rank,seq))
            if call is None:
                defects.append(name+': record references an absent sampler call');continue
            stream,position=record.get('stream'),record.get('position')
            if not (type(stream) is int and 0<=stream<STREAMS
                    and type(position) is int and position in measured):
                defects.append(name+': record coordinate invalid');continue
            if record.get('run')!=name or record.get('repeat')!=repeat:
                defects.append(name+': record run identity differs');continue
            if record.get('trace_sha256')!=cohort[stream]:
                defects.append(name+': foreign row');continue
            key=(stream,position)
            if (rank,stream,position) in joined:
                defects.append(name+': duplicate physical coordinate');continue
            local=call['local_snapshot'];glob=call['global_snapshot']
            batch_row,slot=record.get('batch_row'),record.get('slot')
            idx_mapping=local['idx_mapping']
            if not (type(batch_row) is int and 0<=batch_row<len(idx_mapping)
                    and type(slot) is int and idx_mapping[batch_row]==slot):
                defects.append(name+': batch row/slot differs');continue
            if call['local_cohort_order'][batch_row]!=cohort[stream]:
                defects.append(name+': local cohort row differs');continue
            # Pinned positions[] is num_computed_tokens: at a decode sampler
            # call post_update already counted the fed token, so it is total_len-1.
            prompt_len,total_len=record.get('prompt_len'),record.get('total_len')
            if not (type(prompt_len) is int and type(total_len) is int
                    and total_len-prompt_len==position
                    and record.get('output_position')==position
                    and record.get('input_position')==total_len-1):
                defects.append(name+': derived position differs');continue
            logits_row=record.get('logits_row');cu=local['cu_num_logits']
            if not (type(logits_row) is int and batch_row+1<len(cu)
                    and cu[batch_row]==logits_row
                    and 0<=logits_row<len(local['logit_input_positions'])
                    and local['logit_input_positions'][logits_row]==total_len-1):
                defects.append(name+': logit row/input position differs');continue
            if slot not in glob['idx_mapping']:
                defects.append(name+': request slot absent from the global batch');continue
            global_row=glob['idx_mapping'].index(slot)
            if not (global_row+1<len(glob['cu_num_logits'])
                    and local['logits_indices'][logits_row]
                        ==glob['logits_indices'][glob['cu_num_logits'][global_row]]):
                defects.append(name+': logits selection differs across ranks');continue
            numeric_row=numeric.get(key)
            if numeric_row is None:
                defects.append(name+': numerical row absent for a physical coordinate');continue
            full_graph=call['graph']['cg_mode']=='full'
            decode_only=_decode_only(glob)
            if (numeric_row.get('worker_route')!=call.get('route')
                    or numeric_row.get('route_generation')!=generation
                    or numeric_row.get('full_graph') is not full_graph
                    or numeric_row.get('decode_only') is not decode_only):
                defects.append(name+': numerical row differs from its physical record');continue
            if type(record.get('forced_token_exact')) is not bool or record['forced_token_exact'] is not numeric_row.get('forced_token_exact'):
                defects.append(name+': forced-token proof differs from physical row');continue
            sha = record.get('logprob_sha256')
            if type(sha) is not str or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
                defects.append(name+': sampler logprob binding absent');continue
            if rank == 2 and sha != logprob_digest(cohort[stream], position, numeric_row.get('top'), numeric_row.get('top_order')):
                defects.append(name+': sampler/SSE logprob binding differs');continue
            fields=tuple(record.get(field) for field in ('slot','trace_sha256','prompt_len',
                         'total_len','input_position','forced_token_exact'))
            if logical.setdefault(key,fields)!=fields:
                defects.append(name+': pp ranks disagree on a coordinate');continue
            joined[(rank,stream,position)]=record
            rank_stream.setdefault((rank,stream),[]).append(record)
        expected={(stream,position) for stream in range(STREAMS) for position in measured}
        if set(logical)!=expected:
            defects.append(name+': physical coverage incomplete')
        for rank in (2,3):
            if {(stream,position) for r,stream,position in joined if r==rank} != expected:
                defects.append(name+': rank%d physical coverage incomplete'%rank)
        for (rank,stream),rows in rank_stream.items():
            ordered=sorted(rows,key=lambda row:row['position'])
            for left,right in zip(ordered,ordered[1:]):
                if right['sampler_call_seq'] <= left['sampler_call_seq']:
                    defects.append(name+': one-position shift on rank%d stream%d'%(rank,stream))
                    break
        ranks=sorted({record['rank'] for record in joined.values()})
        for seq in sorted({seq for (_,seq) in callmap}):
            present=[rank for rank in ranks if (rank,seq) in callmap]
            if len(present)!=len(ranks) or len(present)<2:
                defects.append(name+': pp rank sampler-call gap');break
            first=callmap[(present[0],seq)]
            for rank in present[1:]:
                other=callmap[(rank,seq)]
                if (other['global_snapshot']!=first['global_snapshot']
                        or other['global_cohort_order']!=first['global_cohort_order']
                        or other['graph']!=first['graph']
                        or other['route']!=first['route']
                        or other['route_generation']!=first['route_generation']
                        or other['engine_epoch']!=first['engine_epoch']
                        or other['storage_epoch']!=first['storage_epoch']):
                    defects.append(name+': pp ranks disagree on the logical row');break
                if (other['local_snapshot']!=first['local_snapshot']
                        or other['local_cohort_order']!=first['local_cohort_order']):
                    summary['physical_layout_differs']=True
        summary['records']+=len(joined)
        summary['ranks'][name]=ranks
    if len(defects)>MAX_DEFECTS:
        summary['defects']=defects[:MAX_DEFECTS-1]+['defects-overflow']
    summary['defect_count']=len(defects)
    return summary


def grade(evidence):
    out={'schema':1,'rule':dict(FROZEN_RULE),'verdict':'INCONCLUSIVE',
         'baseline_sha256':BASELINE_SHA256,'coverage_per_stream':0,
         'comparisons':{},'missing':[], 'failures':[]}
    try:
        _require(RULE==FROZEN_RULE, 'teacher rule drift')
        _require(type(evidence) is dict
                 and evidence.get('schema') in (LEGACY_EVIDENCE_SCHEMA, EVIDENCE_SCHEMA),
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
        if evidence['schema']==EVIDENCE_SCHEMA:
            # Row/batch/join defects are collector defects: detect them before
            # the frozen comparison and never fold them into its tolerance.
            out['row_mapping']=_row_mapping(evidence,runs)
            if out['row_mapping']['defects']:
                out['verdict']='INCONCLUSIVE'
                out['missing']=['row mapping collector defect: '+reason
                                for reason in out['row_mapping']['defects']]
                return out
        else:
            out['row_mapping']={'basis':'legacy: absent',
                                'note':'original row/batch geometry unproved'}
            out['missing']=['row mapping schema /2 required for qualification']
            return out
        mapped={name:_rows(runs[name], 'split' if name=='split-B' else 'full',positions,prefixes,order_basis is not None)
                for name in RUNS}
    except RowMappingDefect as exc:
        out['verdict']='INCONCLUSIVE'
        out['missing']=['row mapping collector defect: '+reason for reason in exc.defects]
        out['row_mapping']={'defects':exc.defects}
        return out
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
