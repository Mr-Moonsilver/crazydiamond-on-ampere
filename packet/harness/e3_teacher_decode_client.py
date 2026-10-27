"""Fixed teacher-forced instrument for the existing owned cohort lifecycle.

This is not a controller, engine mode, sampler configuration, or W1 workload.
It supplies a fixed 256-entry / 256-measured / 256-tail continuation, consumes
original top-20 logprobs, and joins worker sampler-position proofs to those
responses. No client arrival timestamp is used to claim split execution.
"""
import hashlib
import json
import math
import os
import random
from pathlib import Path
import urllib.request

import e3_teacher_decode_grade as grade_module

ENTRY, MEASURE, TAIL, TOTAL = 256, 256, 256, 768
SCHEMA='s5-teacher-decode/1'
EVIDENCE_SCHEMA='s5-teacher-decode/2'
ROW_SCHEMA='s5-teacher-rows/1'
ROW_BASIS='sampler-invocation/1'
ROW_MAPPING_SCHEMA='s5-teacher-row-mapping/1'
MAX_ROWS=2048
MAX_CALLS=2048


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _decode_only(snapshot):
    """Prove a one-token, no-prefill, no-draft batch from the captured global batch."""
    if type(snapshot) is not dict:
        return False
    scheduled = snapshot.get('num_scheduled_tokens')
    prefilling = snapshot.get('is_prefilling_np')
    return (snapshot.get('num_draft_tokens') == 0
            and snapshot.get('num_tokens') == snapshot.get('num_reqs')
            and type(scheduled) is list and len(scheduled) > 0
            and all(type(value) is int and value == 1 for value in scheduled)
            and type(prefilling) is list and len(prefilling) == len(scheduled)
            and all(type(value) is bool and value is False for value in prefilling))


def _native_teacher_binding(run, plans):
    """Project actual native worker/frontend receipts without probe relabelling."""
    native = run.get('native_teacher_window')
    if type(native) is not dict or native.get('schema') != 's5-native-teacher-window/1':
        raise ValueError('native teacher window absent')
    repeat = native.get('repeat')
    reply = native.get('worker_reply')
    cleared = native.get('frontend_clear')
    if type(reply) is not dict or type(cleared) is not dict:
        raise ValueError('native teacher receipts absent')
    if type(reply.get('engine_epoch')) is not int or reply['engine_epoch'] <= 0:
        raise ValueError('native teacher engine epoch absent')
    previous = cleared.get('previous_status')
    if (type(repeat) is not int or type(previous) is not dict
            or previous.get('enabled') is not True
            or type(previous.get('plan_sha256')) is not str
            or len(previous['plan_sha256']) != 64
            or previous.get('plan_sha256') != plans.get(repeat)
            or previous.get('pending') != 0 or previous.get('admitted') != 8
            or previous.get('steps') != TOTAL or cleared.get('enabled') is not False
            or cleared.get('engine_epoch') != reply.get('engine_epoch')):
        raise ValueError('native frontend admission binding differs')
    workers = reply.get('workers')
    if type(workers) is not list or len(workers) != 4:
        raise ValueError('native teacher worker coverage differs')
    ranks, rows, generations = set(), [], set()
    for worker in workers:
        if type(worker) is not dict:
            raise ValueError('native teacher worker invalid')
        row = worker.get('qualification')
        rank = worker.get('worker_rank')
        if (type(rank) is not int or rank not in (0,1,2,3) or rank in ranks or type(row) is not dict
                or row.get('rank') != rank or row.get('native_route') is not True
                or row.get('mode') != 'native-overlap'
                or row.get('ack') != 'flash_s5_teacher_evidence'
                or row.get('engine_epoch') != reply.get('engine_epoch')
                or row.get('route') != run.get('route')
                or type(row.get('route_generation')) is not int
                or row['route_generation'] < 0
                or type(row.get('drained_streams')) is not int
                or row['drained_streams'] <= 0):
            raise ValueError('native teacher worker identity differs')
        ranks.add(rank); rows.append(row); generations.add(row['route_generation'])
    if len(generations) != 1:
        raise ValueError('native teacher window spans generations')
    proof = dict(previous, ranks=rows)
    core = dict(repeat=repeat, engine_epoch=reply['engine_epoch'],
                route_generation_after=next(iter(generations)))
    return proof, core


class Instrument:
    def __init__(self, baseline, post=None, snapshot_max_bytes=2*1024**3,
                 numerical_rule=3, calibration_path=None):
        if type(snapshot_max_bytes) is not int or not 64*1024**2<=snapshot_max_bytes<=2*1024**3:
            raise ValueError("snapshot budget invalid")
        if type(numerical_rule) is not int or numerical_rule not in (3,4):
            raise ValueError("numerical rule selection invalid")
        if calibration_path is not None and not isinstance(calibration_path,(str,Path)):
            raise ValueError("calibration path invalid")
        self.snapshot_max_bytes=snapshot_max_bytes
        self.numerical_rule=numerical_rule
        self.calibration_path=calibration_path
        self.baseline=grade_module.baseline_record(baseline)
        self.post=post or self._post
        self.prompts=None;self.traces=None;self.plan=None;self.plan_shas={}

    @staticmethod
    def _post(url,path,body):
        headers={'Content-Type':'application/json'}
        if os.environ.get('VLLM_API_KEY'):headers['Authorization']='Bearer '+os.environ['VLLM_API_KEY']
        req=urllib.request.Request(url+path,json.dumps(body).encode(),headers)
        with urllib.request.urlopen(req,timeout=120) as response:return json.loads(response.read())

    def token_ids(self,url,model,rng,n):
        # Same seeded-word construction as frozen teacher_forced.token_ids.
        vocabulary='alpha beta gamma delta epsilon kernel page block cache stream layer token commit merge diff patch queue batch thread'.split()
        words=[rng.choice(vocabulary)+str(rng.randint(0,999)) for _ in range(int(n/2.)+64)]
        ids=self.post(url,'/tokenize',{'model':model,'prompt':' '.join(words),'add_special_tokens':False})['tokens']
        if type(ids) is not list or len(ids)<n or any(type(t) is not int or t<0 for t in ids):
            raise ValueError('teacher tokenization incomplete')
        return ids[:n]

    def cohort(self,url,model):
        # Preserve the existing q17b short correctness prompt workload. Only
        # this new instrument's continuation is longer; q15 supplies the frozen
        # comparison rule, not a claim that its prefill logits are decode data.
        import e3_decode_correctness as c
        workload=c.validate_manifest(json.loads((Path(__file__).resolve().parent/
                     'workloads/q17b-correctness-precontrols.json').read_text()))[1]
        original=c.build_cohort(workload)
        self.prompts=[]
        for row in original:
            ids=self.post(url,'/tokenize',{'model':model,'prompt':row['prompt'],'add_special_tokens':True})['tokens']
            if type(ids) is not list or not ids or any(type(t) is not int or t<0 for t in ids):
                raise ValueError('teacher prompt tokenization incomplete')
            self.prompts.append(ids)
        rng=random.Random(7)
        self.traces=[self.token_ids(url,model,rng,TOTAL) for _ in range(8)]
        if len({digest(t) for t in self.traces})!=8:raise ValueError('teacher trace identities collide')
        return [dict(stream=n,prompt=p,prompt_sha256=digest(p),prompt_chars=0)
                for n,p in enumerate(self.prompts)]

    def requests(self,bodies,cohort,repeat):
        rows=[]
        for i,(body,trace) in enumerate(zip(bodies,self.traces)):
            if body['max_tokens']!=TOTAL or body['min_tokens']!=TOTAL:
                raise ValueError('teacher workload budget differs')
            name=f'tf-{repeat}-{i}'
            body.update(request_id=name,logprobs=20,return_tokens_as_token_ids=True)
            rows.append(dict(request_id=name,prompt_sha256=digest(cohort[i]['prompt']),continuation=trace))
        self.plan={'schema':SCHEMA,'requests':rows,'snapshot_max_bytes':self.snapshot_max_bytes}
        self.plan_shas[repeat]=digest(self.plan)

    def controller(self,controller,action):
        if action!='prepare':return controller
        plan=self.plan
        class WithPlan:
            def execute(self,command,timeout_s):
                reply,transport=controller.execute(dict(command,teacher_forced=plan),timeout_s)
                proof=reply.get('teacher_forced',{})
                if (proof.get('plan_sha256')!=digest(plan) or proof.get('enabled') is not True
                        or proof.get('pending')!=8 or proof.get('steps')!=TOTAL):
                    raise ValueError('teacher configuration receipt differs')
                return reply,transport
        return WithPlan()

    def consume(self,state,event):
        choices=event.get('choices',[])
        if not choices:return
        if len(choices)!=1:raise ValueError('teacher completion choice count')
        row=choices[0];tokens=row.get('token_ids') or []
        if not tokens:return
        lp=row.get('logprobs') or {};top=lp.get('top_logprobs');encoded=lp.get('tokens')
        if (type(top) is not list or type(encoded) is not list
                or len(top)!=len(tokens) or len(encoded)!=len(tokens)):
            raise ValueError('teacher logprobs missing')
        for i,(token,values,name) in enumerate(zip(tokens,top,encoded)):
            position=state.token_count+i
            if type(token) is not int or position>=TOTAL or token!=self.traces[state.index][position]:
                raise ValueError('teacher forced continuation differs')
            if name!='token_id:'+str(token):raise ValueError('teacher token encoding differs')
            if not ENTRY<=position<ENTRY+MEASURE:continue
            if (type(values) is not dict or not 20<=len(values)<=21
                    or any(type(k) is not str or not k.startswith('token_id:') or not k[9:].isdigit()
                           or type(v) not in (int,float) or not math.isfinite(v) for k,v in values.items())):
                raise ValueError('teacher original top20 incomplete')
            state.teacher_top.append(dict(position=position,top=dict(values),top_order=list(values)))

    def finish(self,runs):
        prefixes=[digest(p+t[:ENTRY]) for p,t in zip(self.prompts,self.traces)]
        cohort=[digest(t) for t in self.traces]
        measured=set(range(ENTRY,ENTRY+MEASURE))
        evidence=dict(schema=EVIDENCE_SCHEMA,top_order_basis='response-insertion-order/1',
                      rule=dict(grade_module.FROZEN_RULE),baseline_sha256=self.baseline['sha256'],
                      fixed_positions=list(range(ENTRY,ENTRY+MEASURE)),prefix_digests=prefixes,
                      cohort_trace_digests=cohort,row_schema=ROW_SCHEMA,
                      row_mapping_basis=ROW_BASIS,runs={})
        known_failure=[]
        for run in runs:
            name=run['label'];route=run['route']
            defects=[]
            mapping=dict(schema=ROW_MAPPING_SCHEMA,basis=ROW_BASIS,calls=[],records=[],
                         defects=defects,overflow=False,physical_layout_differs=False,
                         rank_layout={})
            result=dict(route=route,lifecycle_verdict='PASS' if run.get('complete') else 'INCONCLUSIVE',
                        prefix_digests=prefixes,positions=[],repeat=None,engine_epoch=None,
                        route_generation=None,storage_epoch=None,row_mapping=mapping)
            evidence['runs'][name]=result
            if run.get('contract'):known_failure.append(name+': lifecycle/data contract failure')
            if 'native_teacher_window' in run:
                try:
                    proof, core = _native_teacher_binding(run, self.plan_shas)
                except (KeyError, TypeError, ValueError) as exc:
                    defects.append(str(exc)); continue
                # Keep the actual native receipts separately. No native action
                # or acknowledgement is renamed to the legacy pause_end.
                result['native_teacher_window'] = run['native_teacher_window']
            else:
                ends=[c for c in run.get('calls',[]) if c.get('action')=='pause_end' and c.get('lifecycle_ok')]
                if len(ends)!=1:
                    defects.append('pause_end receipt absent');continue
                call=ends[0];proof=(call.get('receipt') or {}).get('teacher_forced')
                core=call.get('receipt_core') or {}
                if (type(proof) is not dict or core.get('teacher_forced')!=proof
                        or call.get('receipt_sha256')!=digest(core)
                        or proof.get('pending')!=0 or proof.get('admitted')!=8
                        or proof.get('plan_sha256')!=self.plan_shas.get(core.get('repeat'))):
                    defects.append('teacher receipt binding differs');continue
            result['repeat']=core.get('repeat')
            result['engine_epoch']=core.get('engine_epoch')
            result['route_generation']=core.get('route_generation_after')
            workers=proof.get('ranks')
            if (type(workers) is not list or len(workers)!=4
                    or {r.get('rank') for r in workers}!={0,1,2,3}):
                defects.append('four worker evidence blocks required');continue
            by_trace={cohort[i]:i for i in range(len(cohort))}
            actual={};calls={}
            owner_ranks=sorted(worker['rank'] for worker in workers if worker.get('sampler_owner'))
            layouts={}
            for worker in workers:
                rank=worker['rank'];owner=rank>=2
                if (worker.get('overflow') is not False or worker.get('call_overflow') is not False
                        or type(worker.get('capture_defects')) is not list or worker['capture_defects']
                        or worker.get('engine_epoch')!=core.get('engine_epoch')
                        or worker.get('route_generation')!=core.get('route_generation_after')
                        or worker.get('sampler_owner') is not owner
                        or type(worker.get('drained_streams')) is not int or worker['drained_streams']<=0
                        or (owner and (worker.get('row_schema')!=ROW_SCHEMA
                                       or worker.get('row_basis')!=ROW_BASIS))
                        or (not owner and (worker.get('positions')!=[] or worker.get('calls')!=[]))):
                    if (worker.get('overflow') is not False
                            or worker.get('call_overflow') is not False):
                        mapping['overflow']=True
                    defects.append('rank%d: worker evidence unproved'%rank);continue
                if not owner:continue
                if result['storage_epoch'] is None:
                    result['storage_epoch']=worker.get('storage_epoch')
                elif result['storage_epoch']!=worker.get('storage_epoch'):
                    defects.append('rank%d: storage generation differs'%rank)
                raw_calls=worker.get('calls');raw_positions=worker.get('positions')
                if (type(raw_calls) is not list or len(raw_calls)>MAX_CALLS
                        or type(raw_positions) is not list or len(raw_positions)>MAX_ROWS):
                    mapping['overflow']=True
                    defects.append('rank%d: worker evidence exceeds bounds'%rank);continue
                worker_calls={};seen=set()
                for row_call in raw_calls:
                    seq=row_call.get('sampler_call_seq') if type(row_call) is dict else None
                    if (type(row_call) is not dict or row_call.get('rank')!=rank or type(seq) is not int):
                        defects.append('rank%d: sampler call identity differs'%rank);continue
                    if seq in worker_calls:
                        defects.append('rank%d: duplicate sampler call'%rank);continue
                    if (row_call.get('engine_epoch')!=core.get('engine_epoch')
                            or row_call.get('route')!=worker.get('route')
                            or row_call.get('route_generation')!=core.get('route_generation_after')
                            or row_call.get('storage_epoch')!=worker.get('storage_epoch')
                            or type(row_call.get('graph')) is not dict
                            or row_call.get('descriptor_stale') is not False
                            or row_call.get('descriptor_matches_batch') is not True
                            or type(row_call.get('global_snapshot')) is not dict
                            or type(row_call.get('local_snapshot')) is not dict):
                        defects.append('rank%d: sampler call proof differs'%rank);continue
                    worker_calls[seq]=row_call;calls[(rank,seq)]=row_call
                    mapping['calls'].append(row_call)
                if rank not in (2,3):
                    defects.append('sampler rank is not qualified');continue
                layout=layouts.setdefault(str(rank),{'streams':set(),'num_reqs':set(),'slots':set()})
                for position in raw_positions:
                    if type(position) is not dict:
                        defects.append('rank%d: position row invalid'%rank);continue
                    seq=position.get('sampler_call_seq');row_call=worker_calls.get(seq)
                    if row_call is None:
                        defects.append('rank%d: row references an absent sampler call'%rank);continue
                    stream=by_trace.get(position.get('trace_sha256'))
                    position_index=position.get('position')
                    if stream is None:
                        defects.append('rank%d: foreign row'%rank);continue
                    if type(position_index) is not int or position_index not in measured:
                        defects.append('rank%d: row outside measured set'%rank);continue
                    key=(stream,position_index)
                    if key in seen:
                        defects.append('rank%d: duplicate measured row'%rank);continue
                    local=row_call['local_snapshot'];global_snapshot=row_call['global_snapshot']
                    batch_row=position.get('batch_row');slot=position.get('slot')
                    idx_mapping=local.get('idx_mapping')
                    if (type(idx_mapping) is not list or type(batch_row) is not int
                            or not 0<=batch_row<len(idx_mapping) or idx_mapping[batch_row]!=slot):
                        defects.append('rank%d: batch row/slot differs'%rank);continue
                    order=row_call.get('local_cohort_order')
                    if (type(order) is not list or batch_row>=len(order)
                            or order[batch_row]!=position.get('trace_sha256')):
                        defects.append('rank%d: local cohort row differs'%rank);continue
                    prompt_len,total_len=position.get('prompt_len'),position.get('total_len')
                    if (type(prompt_len) is not int or type(total_len) is not int
                            or total_len-prompt_len!=position_index):
                        defects.append('rank%d: derived output position differs'%rank);continue
                    # Pinned positions[] = num_computed_tokens = total_len-1 here.
                    if position.get('input_position')!=total_len-1:
                        defects.append('rank%d: input position differs'%rank);continue
                    logits_row=position.get('logits_row');cu_num_logits=local.get('cu_num_logits')
                    if (type(logits_row) is not int or type(cu_num_logits) is not list
                            or batch_row+1>=len(cu_num_logits) or cu_num_logits[batch_row]!=logits_row):
                        defects.append('rank%d: logits row differs'%rank);continue
                    input_positions=local.get('logit_input_positions')
                    if (type(input_positions) is not list or not 0<=logits_row<len(input_positions)
                            or input_positions[logits_row]!=position.get('input_position')):
                        defects.append('rank%d: logit input position differs'%rank);continue
                    if local != global_snapshot or row_call.get('local_cohort_order') != row_call.get('global_cohort_order'):
                        defects.append('rank%d: batch-sharded sampling is not qualified'%rank);continue
                    seen.add(key)
                    mapping['records'].append(dict(
                        rank=rank,run=name,repeat=core.get('repeat'),stream=stream,
                        position=position_index,sampler_call_seq=seq,
                        trace_sha256=position['trace_sha256'],slot=slot,batch_row=batch_row,
                        logits_row=logits_row,prompt_len=prompt_len,total_len=total_len,
                        output_position=position_index,input_position=position['input_position'],
                        forced_token_exact=position.get('forced_token_exact'),
                        logprob_sha256=position.get('logprob_sha256')))
                    layout['streams'].add(stream);layout['num_reqs'].add(local.get('num_reqs'))
                    layout['slots'].add(slot)
                    full_graph=row_call['graph'].get('cg_mode')=='full'
                    decode_only=_decode_only(global_snapshot)
                    p={'position':position_index,'worker_route':row_call['route'],
                       'route_generation':row_call['route_generation'],'full_graph':full_graph,
                       'decode_only':decode_only,
                       'forced_token_exact':position.get('forced_token_exact')}
                    if key in actual and actual[key]!=p:
                        defects.append('rank%d: sampler ranks disagree on a coordinate'%rank);continue
                    actual[key]=p
                if seen != {(stream,position) for stream in range(len(cohort)) for position in measured}:
                    defects.append('rank%d: physical coverage incomplete'%rank)
            for seq in sorted({seq for (_,seq) in calls}):
                present=[rank for rank in owner_ranks if (rank,seq) in calls]
                if len(present)!=len(owner_ranks):
                    defects.append('pp rank sampler-call gap');continue
                first=calls[(present[0],seq)]
                for rank in present[1:]:
                    other=calls[(rank,seq)]
                    if (other.get('global_snapshot')!=first.get('global_snapshot')
                            or other.get('global_cohort_order')!=first.get('global_cohort_order')
                            or other.get('graph')!=first.get('graph')
                            or other.get('route')!=first.get('route')
                            or other.get('route_generation')!=first.get('route_generation')
                            or other.get('engine_epoch')!=first.get('engine_epoch')
                            or other.get('storage_epoch')!=first.get('storage_epoch')):
                        defects.append('pp ranks disagree on the logical row')
                    if (other.get('local_snapshot')!=first.get('local_snapshot')
                            or other.get('local_cohort_order')!=first.get('local_cohort_order')):
                        mapping['physical_layout_differs']=True
            expected={(stream,position) for stream in range(len(cohort)) for position in measured}
            if set(actual)!=expected:defects.append('row mapping coverage incomplete')
            mapping['rank_layout']={rank:{'streams':sorted(layout['streams']),
                'num_reqs':sorted(value for value in layout['num_reqs'] if type(value) is int),
                'slots':sorted(layout['slots'])} for rank,layout in layouts.items()}
            for stream in run.get('streams',[]):
                i=stream['stream']
                if stream.get('_tokens')!=self.traces[i]:
                    known_failure.append(name+': forced-token bitwise mismatch');continue
                for row in stream.get('teacher_top',[]):
                    p=actual.get((i,row['position']))
                    if p is not None:result['positions'].append(dict(p,stream=i,top=row['top'],top_order=row.get('top_order')))
        for result in evidence['runs'].values():
            mapping=result['row_mapping']
            mapping['defect_count']=len(mapping['defects'])
            if len(mapping['defects'])>32:
                mapping['defects']=mapping['defects'][:31]+['defects-overflow']
        rule=getattr(self,'numerical_rule',3)
        if rule not in (3,4):
            raise ValueError("numerical rule selection invalid")
        if rule==4:
            import e3_teacher_decode_rule4 as rule4
            verdict=rule4.grade(evidence,calibration_path=getattr(self,'calibration_path',None))
        else:
            verdict=grade_module.grade(evidence)
        if known_failure:
            verdict['verdict']='FAIL';verdict['failures'].extend(known_failure)
        result={'schema':1,'grade':verdict,'evidence':evidence,'baseline':self.baseline,
                'geometry':dict(entry=ENTRY,measured=MEASURE,tail=TAIL,total=TOTAL),
                'scope':'fixed decode positions and original raw top20; not a replacement for bitwise storage gates'}
        if rule==4:
            result['numerical_rule']=4
            result['gate_policy']={
                'name':'control-envelope-v4','version':4,
                'comparison_policy':'control-envelope-v4',
                'collector_schema':evidence.get('schema'),
                'collector_rule_version':(evidence.get('rule') or {}).get('version'),
                'calibration_sha256':verdict.get('calibration_sha256'),
                'note':('v4 comparison policy applied separately to schema/2 collection '
                        'evidence; the retained artifact still declares rule v3')}
        return result
