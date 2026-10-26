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


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class Instrument:
    def __init__(self, baseline, post=None, snapshot_max_bytes=2*1024**3):
        if type(snapshot_max_bytes) is not int or not 64*1024**2<=snapshot_max_bytes<=2*1024**3:
            raise ValueError("snapshot budget invalid")
        self.snapshot_max_bytes=snapshot_max_bytes
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
        evidence=dict(schema=SCHEMA,top_order_basis='response-insertion-order/1',rule=dict(grade_module.FROZEN_RULE),
                      baseline_sha256=self.baseline['sha256'],fixed_positions=list(range(ENTRY,ENTRY+MEASURE)),
                      prefix_digests=prefixes,runs={})
        known_failure=[]
        for run in runs:
            name=run['label'];route=run['route']
            result=dict(route=route,lifecycle_verdict='PASS' if run.get('complete') else 'INCONCLUSIVE',
                        prefix_digests=prefixes,positions=[])
            evidence['runs'][name]=result
            if run.get('contract'):known_failure.append(name+': lifecycle/data contract failure')
            ends=[c for c in run.get('calls',[]) if c.get('action')=='pause_end' and c.get('lifecycle_ok')]
            if len(ends)!=1:continue
            call=ends[0];proof=(call.get('receipt') or {}).get('teacher_forced')
            core=call.get('receipt_core') or {}
            if (type(proof) is not dict or core.get('teacher_forced')!=proof
                    or call.get('receipt_sha256')!=digest(core)
                    or proof.get('pending')!=0 or proof.get('admitted')!=8
                    or proof.get('plan_sha256')!=self.plan_shas.get(core.get('repeat'))):continue
            workers=proof.get('ranks')
            if (type(workers) is not list or len(workers)!=4
                    or {r.get('rank') for r in workers}!={0,1,2,3}):continue
            by_trace={digest(t):i for i,t in enumerate(self.traces)}
            actual={};valid=True
            for worker in workers:
                rank=worker['rank']
                if (worker.get('overflow') is not False
                        or worker.get('engine_epoch')!=core.get('engine_epoch')
                        or worker.get('route_generation')!=core.get('route_generation_after')
                        or worker.get('sampler_owner') is not (rank>=2)
                        or type(worker.get('drained_streams')) is not int or worker['drained_streams']<=0
                        or rank<2 and worker.get('positions')!=[]):
                    valid=False;break
                seen=set()
                for position in worker.get('positions',[]):
                    stream=by_trace.get(position.get('trace_sha256'))
                    key=(stream,position.get('position'))
                    if (stream is None or key in seen or position.get('route_generation')!=worker['route_generation']
                            or position.get('worker_route')!=worker.get('route')):valid=False;break
                    seen.add(key)
                    p={k:position.get(k) for k in ('position','worker_route','route_generation',
                                                  'full_graph','decode_only','forced_token_exact')}
                    if key in actual and actual[key]!=p:valid=False;break
                    actual[key]=p
            if not valid:continue
            for stream in run.get('streams',[]):
                i=stream['stream']
                if stream.get('_tokens')!=self.traces[i]:
                    known_failure.append(name+': forced-token bitwise mismatch');continue
                for row in stream.get('teacher_top',[]):
                    p=actual.get((i,row['position']))
                    if p is not None:result['positions'].append(dict(p,stream=i,top=row['top'],top_order=row.get('top_order')))
        verdict=grade_module.grade(evidence)
        if known_failure:
            verdict['verdict']='FAIL';verdict['failures'].extend(known_failure)
        return {'schema':1,'grade':verdict,'evidence':evidence,'baseline':self.baseline,
                'geometry':dict(entry=ENTRY,measured=MEASURE,tail=TAIL,total=TOTAL),
                'scope':'fixed decode positions and original raw top20; not a replacement for bitwise storage gates'}
