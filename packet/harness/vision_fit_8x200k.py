#!/usr/bin/env python3
"""Gate M: bound offline image features, actual-token geometry, ordered fit-D observer."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import signal
import threading
import time
import sys
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'e3_builder'),str(HERE.parent/'patches/s5_engine'),str(HERE.parent/'tools')]

import fit_8x200k as base
from packet_client_contract import report_exception
from vision_answer_probe import validate_embedding, embedding_content
from vision_answer_read import read as read_answers

NAME='q18a-vision-fit-8x200k-b'
PREFIXES=(60000,120000,160000,190320)
OUTPUT_CAPS={60000:2048,120000:2048,160000:2048,190320:8192,190000:8192}
OUTPUT=8192
IMAGE_TOKEN=248056
BINDING_SHA='d8e776dee8f302e4961db69e36c13fc29c8a8006f50cf809154b307d5bae3373'
CACHE_SHAS={'semantic':'7f3a7448a8c32f3bc39ce201e99272e0cd826b46a0f8f8678d063f53c57731d7',
            'eight-unique':'8f03bfa42322d6ecf26a96ce7e1587f59d45c60f6fa454ff7c6fa66bee17dd65'}
Hold=base.Hold


def plan():
    return dict(schema='vision-fit-plan/1',name=NAME,agents=8,max_model_len=200000,
        prefixes=list(PREFIXES),staged_output_caps={str(k):v for k,v in OUTPUT_CAPS.items()},
        image_tail_cohort=dict(prompt_tokens=190320,output_tokens=8192,members=8),
        cold_beside_seven=dict(prompt_tokens=190000,output_tokens=8192,companion_prime_output=1,required_hit=190080),
        native_workload_cap_s=2700,startup_cap_s=900,start_identity_cap_s=900,end_identity_cap_s=1200,stop_cap_s=300,
        outer_cap_s=6000,owned_cleanup_grace_s=330,floor_mib=40,inputs_binding_sha256=BINDING_SHA,
        growth_protocol='fit-D exact token-prefix continuation in trusted text; no normal chat-history rerender',
        cache_shas=CACHE_SHAS,scope='memory/ownership fit; not speed/adoption or visual-reference qualification')


def post(url,path,body,timeout=60):
    conn,root=base.connection(url,timeout)
    try:
        conn.request('POST',root+path,json.dumps(body).encode(),base.headers())
        response=conn.getresponse()
        if response.status!=200:
            response.read(1024);raise base.Hold(path+'-HTTP-'+str(response.status))
        return json.load(response)
    finally:conn.close()


def load_inputs(args):
    binding=Path(args.binding);raw=binding.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=BINDING_SHA:raise base.Hold('input-binding-SHA')
    directory=Path(args.private_cache_dir)
    if directory.is_symlink() or directory.stat().st_mode & 0o777!=0o700:raise base.Hold('private-cache-directory-mode')
    declared=json.loads(raw);result={}
    for name in CACHE_SHAS:
        path=Path(args.private_cache_dir)/(name+'.json')
        if path.is_symlink() or path.stat().st_mode & 0o777!=0o600:raise base.Hold('private-cache-mode')
        if path.stat().st_size!=declared['inputs'][name]['cache_bytes']:raise base.Hold('private-cache-size-'+name)
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=CACHE_SHAS[name]:raise base.Hold('private-cache-SHA-'+name)
        cache=json.loads(raw);entry=declared['inputs'][name]
        cases=Path(args.packet)/('vision-assets' if name=='semantic' else 'vision-fit-assets')/'cases.json'
        if hashlib.sha256(cases.read_bytes()).hexdigest()!=entry['cases_sha256']:raise base.Hold('case-manifest-SHA')
        result[name]=[]
        for case in json.loads(cases.read_text()):
            image=cases.parent/case['image'];digest=hashlib.sha256(image.read_bytes()).hexdigest()
            if entry['image_sha256_by_case'].get(case['id'])!=digest:raise base.Hold('image-SHA')
            feature=validate_embedding(cache['features'][digest])
            result[name].append(dict(case,image_sha256=digest,content=embedding_content(feature),rows=feature['shape'][0],
                feature_sha256=feature['embedding_sha256']))
        if len(result[name])!=entry['feature_count']:raise base.Hold('feature-count')
    return result



def tokenize(args,messages):
    value=post(args.url,'/tokenize',dict(model=args.model,messages=messages,
        chat_template_kwargs=dict(enable_thinking=False),add_generation_prompt=True))
    ids=value['tokens']
    if (value['count']!=len(ids) or value['max_model_len']!=200000
            or any(type(t) is not int or t<0 or t>=248320 for t in ids)):
        raise base.Hold('tokenizer-schema-or-model')
    return ids


def prepare(args,feature,target,uid,previous=None,tail=False):
    # Fixed-work fit-D token-prefix protocol: do not let chat rerendering rewrite an old generation header.
    retained_text=''
    if previous is not None:
        image_end=max(i for i,t in enumerate(previous.prompt) if t==IMAGE_TOKEN)+1
        if previous.prompt[image_end]!=248054:raise base.Hold('missing-vision-end')
        retained_text=post(args.url,'/detokenize',dict(model=args.model,tokens=previous.prompt[image_end+1:]+previous.tokens))['prompt']
        old_text=next(part['text'] for part in previous.messages[-1]['content'] if part['type']=='text')
        if retained_text.startswith(old_text):pass
        elif retained_text.startswith('\n'+old_text):retained_text=retained_text[1:]
        else:raise base.Hold('unmodelled-chat-image-text-separator')
        retained_text+='<|im_end|>\n<|im_start|>user\n'
    messages=[]
    filler=0
    for attempt in range(8):
        text=dict(type='text',text=retained_text+'Agent memory-fit '+uid+'.'+' x'*filler+' Keep responding until the output cap.')
        content=[text,feature['content']] if tail else [feature['content'],text]
        candidate=messages+[dict(role='user',content=content)]
        ids=tokenize(args,candidate)
        if len(ids)==target:break
        filler+=target-len(ids)
        if filler<0:raise base.Hold('growth-would-truncate-prompt-or-decoded-reply')
    else:raise base.Hold('exact-chat-quantum-not-reached')
    positions=[i for i,t in enumerate(ids) if t==IMAGE_TOKEN]
    if len(positions)!=feature['rows'] or positions!=list(range(positions[0],positions[0]+len(positions))):
        raise base.Hold('actual-image-placeholder-geometry')
    if previous is not None:
        retained=previous.prompt+previous.tokens
        if ids[:len(retained)]!=retained:raise base.Hold('growth-does-not-continue-exact-decoded-tokens')
    if tail and not any(i>=target-64 for i in positions):raise base.Hold('image-does-not-overlap-64-row-tail')
    return Stream(uid,ids,candidate,feature,OUTPUT_CAPS.get(target,128),tail)


def companion_prime(args,stream,index):
    # Preserve the previous chat prefix, but checkpoint one hash unit before its final tail.
    messages=copy.deepcopy(stream.messages)
    part=next(p for p in messages[-1]['content'] if p['type']=='text')
    suffix=' x'*176+' Keep responding until the output cap.'
    if not part['text'].endswith(suffix):raise base.Hold('companion-prime-filler-geometry')
    part['text']=part['text'][:-len(suffix)]+' Keep responding until the output cap.'
    ids=tokenize(args,messages)
    if len(ids)!=190144 or ids[:190080]!=stream.prompt[:190080]:
        raise base.Hold('companion-prime-checkpoint-prefix-differs')
    return Stream('vf-cold-prime-'+str(index),ids,messages,stream.feature,1)


class Stream:
    def __init__(self,uid,prompt,messages,feature,cap,tail=False):
        self.uid,self.prompt,self.messages,self.feature,self.output_cap=uid,prompt,messages,feature,cap
        self.tail=tail;self.first=threading.Event();self.done=threading.Event();self.tokens=[];self.text=''
        self.cancel_requested=False;self.outcome='pending'
        self.error=None;self.open_ns=self.first_ns=self.end_ns=None;self.socket=None
        self.cached_tokens=self.usage=self.finish_reason=self.http_status=None;self.prompt_verified=False
    def record(self):
        positions=[i for i,t in enumerate(self.prompt) if t==IMAGE_TOKEN]
        return dict(uid=self.uid,prompt_tokens=len(self.prompt),prompt_sha256=base.sha(self.prompt),
            image_sha256=self.feature['image_sha256'],feature_sha256=self.feature['feature_sha256'],
            image_rows=len(positions),image_start=positions[0],image_end=positions[-1]+1,
            image_tail_overlap=sum(i>=len(self.prompt)-64 for i in positions) if self.tail else 0,
            output_tokens=len(self.tokens),output_cap=self.output_cap,output_sha256=base.sha(self.tokens),
            resume_ns=self.open_ns,first_ns=self.first_ns,end_ns=self.end_ns,cached_tokens=self.cached_tokens,
            finish_reason=self.finish_reason,http_status=self.http_status,error=self.error,prompt_verified=self.prompt_verified,
            outcome=self.outcome,cancel_requested=self.cancel_requested)


def generate(args,stream):
    conn,root=base.connection(args.url,args.timeout_s)
    try:
        conn.connect();stream.socket=conn.sock;stream.open_ns=time.monotonic_ns()
        body=dict(model=args.model,messages=stream.messages,max_tokens=stream.output_cap,temperature=0,ignore_eos=True,
            stream=True,stream_options={'include_usage':True},return_token_ids=True,request_id=stream.uid,
            skip_special_tokens=False,chat_template_kwargs=dict(enable_thinking=False))
        conn.request('POST',root+'/v1/chat/completions',json.dumps(body).encode(),base.headers())
        response=conn.getresponse();stream.http_status=response.status
        if response.status!=200:response.read(1024);raise base.Hold('chat-HTTP-'+str(response.status))
        ended=False
        for line in response:
            if not line.startswith(b'data: '):continue
            if line.strip()==b'data: [DONE]':ended=True;break
            row=json.loads(line[6:])
            if row.get('error'):raise base.Hold('chat-stream-server-error')
            if row.get('prompt_token_ids') is not None:
                if row['prompt_token_ids']!=stream.prompt:raise base.Hold('actual-chat-prompt-differs-from-tokenize')
                stream.prompt_verified=True
            if row.get('usage'):
                stream.usage=row['usage'];stream.cached_tokens=(row['usage'].get('prompt_tokens_details') or {}).get('cached_tokens')
                if row['usage'].get('prompt_tokens')!=len(stream.prompt):raise base.Hold('usage-prompt-differs')
            for choice in row.get('choices',[]):
                tokens=choice.get('token_ids')
                if tokens:
                    if type(tokens) is not list or any(type(t) is not int or t<0 for t in tokens):raise base.Hold('output-token-schema')
                    if stream.first_ns is None:stream.first_ns=time.monotonic_ns();stream.first.set()
                    stream.tokens.extend(tokens)
                text=(choice.get('delta') or {}).get('content')
                if text:stream.text+=text
                if choice.get('finish_reason') is not None:stream.finish_reason=choice['finish_reason']
        if (not ended or len(stream.tokens)!=stream.output_cap or stream.finish_reason!='length'
                or not stream.prompt_verified):
            if stream.cancel_requested and stream.first_ns is None and stream.http_status==200:
                stream.outcome='expected-cancel'
            else:raise base.Hold('incomplete-fixed-chat-or-unverified-prompt')
        else:stream.outcome='complete'
    except Exception as exc:
        if (stream.cancel_requested and stream.first_ns is None and stream.http_status==200
                and isinstance(exc,(OSError,base.http.client.IncompleteRead))):stream.outcome='expected-cancel'
        else:stream.outcome='error';stream.error=report_exception(exc)
    finally:
        stream.end_ns=time.monotonic_ns();stream.first.set();stream.done.set();conn.close()


def assess(record):
    # Exact fit-D coverage/exceptions/stop/startup contract; only the registered workload differs.
    old=(base.PREFIXES,base.COLD_HISTORY,base.OUTPUT,base.OUTPUT_CAPS)
    try:
        base.PREFIXES,base.COLD_HISTORY,base.OUTPUT,base.OUTPUT_CAPS=PREFIXES,190000,8192,OUTPUT_CAPS
        result=base.assess(record)
    finally:base.PREFIXES,base.COLD_HISTORY,base.OUTPUT,base.OUTPUT_CAPS=old
    reasons=result['reasons']
    if not stop_proven(record.get('memory',{})):reasons.append('all-rank-FULL-retirement-proof')
    if record.get('input_binding_sha256')!=BINDING_SHA or record.get('cache_shas')!=CACHE_SHAS:reasons.append('bound-inputs-unproven')
    tail=record.get('tail') or {}
    if tail.get('complete') is not True or len(tail.get('conversations',[]))!=8:reasons.append('tail-cache-workload-incomplete')
    if type(tail.get('native_admissions')) is not int or tail['native_admissions']<=0:reasons.append('no-native-tail-execution')
    for stage in record.get('stages',[])+[tail,record.get('cold') or {}]:
        for member in stage.get('conversations',[]):
            if (member.get('prompt_verified') is not True or member.get('image_rows')!=1024
                    or member.get('error') or member.get('output_tokens')!=member.get('output_cap')):reasons.append('MM-stream-unproven')
            if stage is tail and (member.get('image_tail_overlap',0)<=0 or member.get('prompt_tokens')!=190320):reasons.append('tail-image-geometry-unproven')
    cancel=record.get('cancel',{})
    if (cancel.get('retired') is not True or cancel.get('cancelled_stream',{}).get('outcome')!='expected-cancel'
            or cancel.get('cancelled_stream',{}).get('output_tokens')!=0):reasons.append('cancel-retirement-unproven')
    if record.get('calibration_lease',{}).get('manual_owned') is not True:reasons.append('FULL-calibration-lease-unproven')
    cold=record.get('cold') or {}
    hits=cold.get('cache_readiness',[])
    if (len(hits)!=7 or len({h.get('uid') for h in hits})!=7
            or {h.get('uid') for h in hits}!={m.get('uid') for m in cold.get('conversations',[])[:-1]}
            or any(h.get('start')!=190080 or h.get('prompt_tokens')!=190320
                   or h.get('engine_epoch')!=record.get('memory',{}).get('engine_epoch') for h in hits)):
        reasons.append('cached-companions-unproven')
    if record.get('reuse',{}).get('same_image_cached',0)<=0 or record.get('reuse',{}).get('different_image_verified') is not True:reasons.append('MM-prefix-reuse-unproven')
    mm=record.get('memory',{}).get('mm_peaks',{})
    if set(mm)!=set(map(str,range(4))):reasons.append('MM-peer-inventory-missing')
    else:
        for rank,row in mm.items():
            if row.get('tower_registered_cuda_bytes')!=0 or row.get('cache_storage_bytes',-1)>40*2**20:reasons.append('MM-tower-or-cache-budget')
        # The single native queue may retire one feature before the next tail; report actual occupancy, never fabricate a full cache.
    if not record.get('host',{}).get('samples') or record.get('host',{}).get('errors'):reasons.append('CT-host-memory-coverage')
    rss=record.get('memory',{}).get('host_rss_peaks',{})
    if set(rss)!=set(map(str,range(4))) or any(any(type(row.get(k)) is not int or row[k]<0 for k in ('VmRSS','VmHWM','RssAnon','RssShmem')) for row in rss.values()):reasons.append('worker-host-RSS-coverage')
    if record.get('semantic',{}).get('verdict')!='MATCH':reasons.append('semantic-labels-not-matched')
    resident=record.get('memory',{}).get('simultaneous_resident_token_peak',{})
    if resident.get('eight_at_190k_seen') is not True or resident.get('eight_minimum_peak',0)<190000:reasons.append('eight-concurrent-190k-frontiers-unproven')
    if record.get('host',{}).get('oom_kill_delta',0) or record.get('host',{}).get('oom_delta',0):return dict(result,verdict='FAIL',reasons=['host-cgroup-OOM'])
    if result['verdict']!='FAIL':result['verdict']='HOLD' if reasons else 'FIT'
    result['reasons']=sorted(set(reasons));result['scope']='memory/ownership only; pending-prefix CUDA race and27B visual reference remain separate qualification'
    from e3_compose5_box import coverage_delta
    wrap=lambda value:dict(append_yield=dict(declines=dict(coverage=value)))
    result['whole_protocol_native_coverage']=coverage_delta(wrap(record.get('launch_native_coverage')),wrap(record.get('memory',{}).get('native_coverage')))
    result['semantic']=record.get('semantic');result['tail']=tail
    return result


def stop_proven(value):
    return value.get('schema')=='fit-memory/1' and value.get('role')=='e6' and value.get('settled') is True and value.get('stop_proof')==dict(
        route='full',all_rank_stop=True,admission_disarmed=True,output_queue_empty=True)

class HostMemory:
    def __init__(self):
        self.samples=0;self.errors=[];self.peaks={};self.first_events=None;self.last_events=None
    def sample(self):
        try:
            root=Path('/sys/fs/cgroup')
            stats=dict(line.split() for line in (root/'memory.stat').read_text().splitlines())
            events={k:int(v) for k,v in (line.split() for line in (root/'memory.events').read_text().splitlines())}
            values=dict(current=int((root/'memory.current').read_text()),**{k:int(stats[k]) for k in ('anon','shmem','file')})
            if (root/'memory.peak').exists():values['kernel_cgroup_peak']=int((root/'memory.peak').read_text())
            meminfo=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line)
            available=int(meminfo['MemAvailable'].split()[0])*1024
            self.minimum_host_available=min(getattr(self,'minimum_host_available',available),available)
            for k,v in values.items():self.peaks[k]=max(self.peaks.get(k,0),v)
            self.first_events=events if self.first_events is None else self.first_events;self.last_events=events;self.samples+=1
        except Exception as exc:
            if len(self.errors)<4:self.errors.append(report_exception(exc))
    def record(self):
        before=self.first_events or {};after=self.last_events or {}
        return dict(samples=self.samples,peak_bytes=self.peaks,errors=self.errors,
            memory_events_first=before,memory_events_last=after,
            oom_delta=after.get('oom',0)-before.get('oom',0),oom_kill_delta=after.get('oom_kill',0)-before.get('oom_kill',0),
            minimum_host_available_bytes=getattr(self,'minimum_host_available',None),
            basis='CT collector sampled memory.current/stat/events, lifetime memory.peak when available, worker VmHWM; no GPU or CUDA call')


def run(args):
    record=dict(schema='vision-fit-8x200k-client/1',role='e6',plan=plan(),complete=False,stages=[],cold=None,
        verdict='HOLD',reason='protocol-incomplete',input_binding_sha256=BINDING_SHA,cache_shas=CACHE_SHAS)
    pool=ThreadPoolExecutor(max_workers=8);all_streams=[];current=[];host=HostMemory();started=time.monotonic()
    def save():
        record['elapsed_s']=time.monotonic()-started;record['host']=host.record()
        for stage in record.get('stages',[])+[record.get('tail'),record.get('cold')]:
            if stage and '_streams' in stage:
                stage['conversations']=[stream.record() for stream in stage['_streams']]
        record['streams']=[stream.record() for stream in all_streams]
        saved=dict(record,stages=[{k:v for k,v in stage.items() if k!='_streams'} for stage in record['stages']])
        for name in ('tail','cold'):
            if record.get(name):saved[name]={k:v for k,v in record[name].items() if k!='_streams'}
        pending=args.out.with_suffix('.next.json');pending.write_text(json.dumps(saved,indent=2)+'\n');pending.replace(args.out)
    def status():
        host.sample();value=post(args.url,base.CONTROL,dict(action='status'))
        record['memory']=value;base.check_status(value,'e6');return value
    def wait(streams,first=False):
        while True:
            if time.monotonic()-started>=args.timeout_s:raise base.Hold('declared-native-workload-cap')
            status();save()
            if any(s.error for s in streams):raise base.Hold('chat-stream-error')
            if all((s.first if first else s.done).is_set() for s in streams):return
            time.sleep(.1)
    def start(stream):
        all_streams.append(stream);pool.submit(generate,args,stream);return stream
    def cohort(name,features,target,previous=None,tail=False):
        nonlocal current
        before=status()['native_admissions'];stage=dict(name=name,prompt_tokens=target,complete=False,conversations=[])
        current=[]
        for i,feature in enumerate(features):
            old=None if previous is None else previous[i]
            current.append(start(prepare(args,feature,target,'vf-'+name+'-'+str(i),old,tail)))
        wait(current)
        stage.update(complete=True,conversations=[s.record() for s in current],native_admissions=status()['native_admissions']-before)
        if stage['native_admissions']<=0:raise base.Hold('no-native-execution')
        return stage,list(current)
    try:
        host.sample();inputs=load_inputs(args);record['auth']=base.keyed_serve(args.url);status()
        record['launch_native_coverage']=record['memory']['native_coverage']
        # Calibration is already allowed on FULL; keep its manual lease until stress.
        from e3_native_runtime_client import NativeRuntimeClient
        import e3_compose5_box as box
        lease=dict(statuses=[])
        session=post(args.url,'/v1/flash_s5_native_overlap',dict(schema=1,action='status'))
        if (session.get('schema')!=1 or session.get('action')!='status' or not session.get('run_id')
                or session.get('engine_epoch')!=record['memory']['engine_epoch']):
            raise base.Hold('existing-native-session-missing-or-foreign')
        client=NativeRuntimeClient(args.url,session['run_id'],epoch=record['memory']['engine_epoch'],timeout_s=60)
        try:
            record['calibration_prepare']=client.prepare('full')
            box.keep_lease(client,lease)
        finally:record['calibration_control']=list(client.history)
        record['calibration_lease']=lease['native_off']
        answers=dict(verdict='COMPLETE',cases=[])
        for case in inputs['semantic']:
            answer=post(args.url,'/v1/chat/completions',dict(model=args.model,temperature=0,max_tokens=128,
                messages=[dict(role='user',content=[case['content'],dict(type='text',text=case['question'])])],
                chat_template_kwargs=dict(enable_thinking=False)),timeout=300)
            text=answer['choices'][0]['message']['content']
            if not isinstance(text,str) or not text.strip():raise base.Hold('semantic-answer-empty')
            answers['cases'].append(dict(id=case['id'],image_sha256=case['image_sha256'],question=case['question'],expected=case['expected'],answer=text))
            status()
        record['semantic']=read_answers(answers)
        record['reference_comparison']='NOT_RUN: separate27B reference/quality evidence; not a memory FIT claim'
        features=inputs['eight-unique'];feature=features[0]
        first=prepare(args,feature,4096,'vf-reuse-layout');first.uid='vf-reuse-1';start(first);wait([first])
        again=Stream('vf-reuse-2',first.prompt,first.messages,feature,128);start(again);wait([again])
        different=prepare(args,features[1],4096,'vf-reuse-layout');different.uid='vf-reuse-3';start(different);wait([different])
        image_start=first.record()['image_start']
        record['reuse']=dict(same_image_cached=again.cached_tokens,different_image_cached=different.cached_tokens,
            different_image_verified=different.prompt==first.prompt and type(different.cached_tokens) is int and different.cached_tokens<=image_start,
            images_differ=first.feature['image_sha256']!=different.feature['image_sha256'],
            conversations=[s.record() for s in (first,again,different)])
        import n3j2_packet
        table=n3j2_packet.configure();row=next(r for r in table.arms() if r['label']=='n3.g2-live-0')
        selection=record['native_selection']=dict(role='e6',selection=row,statuses=[])
        with box._bound():box.select_and_release(client,row,selection)
        selection['control']=client.history
        record['initial_native_coverage']=status()['native_coverage']
        record['coverage_scope']='native stress after semantic/cache calibration; whole protocol also reported'
        cancel=prepare(args,feature,60000,'vf-cancel-reuse');before=status()['native_admissions'];start(cancel)
        while status()['native_admissions']==before:
            if cancel.done.is_set() or time.monotonic()-started>=args.timeout_s:raise base.Hold('cancel-native-admission-not-observed')
            time.sleep(.05)
        if cancel.first_ns is not None:raise base.Hold('cancel-first-output-already-served')
        if cancel.socket is None:raise base.Hold('cancel-owned-socket-missing')
        cancel.cancel_requested=True;cancel.socket.shutdown(socket.SHUT_RDWR)
        deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            current_status=status()
            if cancel.done.is_set() and not any(cancel.uid in uid for uid in current_status.get('active_request_ids',[])):break
            time.sleep(.1)
        else:raise base.Hold('cancel-request-not-retired')
        replacement=Stream(cancel.uid,cancel.prompt,cancel.messages,feature,128);start(replacement);wait([replacement])
        record['cancel']=dict(retired=True,before_first_output=True,reused_uid_success=True,
            pending_prefix_event='not forced; CPU lease proof remains separate',cancelled_stream=cancel.record(),replacement=replacement.record())
        previous=None
        for target in PREFIXES:
            stage,previous=cohort('grow-'+str(target),features,target,previous)
            record['stages'].append(stage);save()
        record['tail'],tail_streams=cohort('tail-images',features,190320,tail=True);save()
        # Tokenize cold before any timed decoder consumes its 8192-token runway.
        cold=prepare(args,features[7],190000,'vf-cold-probe')
        record['cold']=dict(live_at_arrival=0,complete=False,conversations=[],cache_readiness=[],priming=[],admission_baseline=None)
        for i in range(7):
            old=previous[i]
            prime=start(companion_prime(args,old,i));wait([prime])
            record['cold']['priming'].append(prime.record());save()
        before=status()['native_admissions'];current=[]
        record['cold']['admission_baseline']=before
        record['cold']['_streams']=current
        for i in range(7):
            old=previous[i];current.append(start(Stream('vf-cold-companion-'+str(i),old.prompt,old.messages,features[i],8192)))
        wait(current,first=True)
        snapshot=status()
        for stream in current:
            hits=[v for uid,v in snapshot.get('first_admissions',{}).items() if uid.startswith('chatcmpl-'+stream.uid+'-')]
            if (len(hits)!=1 or hits[0].get('start')!=190080 or hits[0].get('prompt_tokens')!=190320
                    or hits[0].get('engine_epoch')!=snapshot['engine_epoch']
                    or type(hits[0].get('transaction')) is not int or hits[0]['transaction']<1):
                raise base.Hold('cached-companion-checkpoint-not-ready')
            record['cold']['cache_readiness'].append(dict(uid=stream.uid,**hits[0]))
        if any(s.done.is_set() for s in current):raise base.Hold('seven-decoders-not-live-at-cold-arrival')
        record['cold']['live_at_arrival']=7;current.append(start(cold));wait(current)
        record['cold']['native_admissions']=status()['native_admissions']-before;save()
        if any(s.end_ns<=cold.first_ns for s in current[:-1]):raise base.Hold('seven-decoders-did-not-span-cold-prefill')
        record['cold']['complete']=True
        if record['cold']['native_admissions']<=0:raise base.Hold('no-native-execution')
        record.update(complete=True,verdict='FIT',reason='fixed-image-growth-tail-and-cold-under-seven-complete')
    except Exception as exc:record.update(verdict='HOLD',reason=report_exception(exc))
    finally:
        if record.get('cold') is not None:
            baseline=record['cold'].get('admission_baseline')
            record['cold']['native_admissions']=(record.get('memory',{}).get('native_admissions',0)-baseline if type(baseline) is int else None)
        save()
        try:
            record['cleanup']=dict(action='finish' if record['complete'] else 'stop',workload_reason=record['reason'])
            settled=post(args.url,base.CONTROL,dict(action=record['cleanup']['action']))
            end=time.monotonic()+120
            while not settled.get('settled') and time.monotonic()<end:
                host.sample();time.sleep(.1);settled=post(args.url,base.CONTROL,dict(action='status'))
            record['memory']=settled
            if not settled.get('settled'):record.update(verdict='HOLD',reason='all-rank-stop-unsettled')
        except Exception as exc:record.update(verdict='HOLD',reason=report_exception(exc))
        for stream in all_streams:
            if not stream.done.is_set() and stream.socket is not None:
                try:stream.socket.shutdown(socket.SHUT_RDWR)
                except OSError:pass
        pool.shutdown(wait=True,cancel_futures=False)
        for stage in record.get('stages',[])+[record.get('tail'),record.get('cold')]:
            if stage and '_streams' in stage:
                stage['conversations']=[stream.record() for stream in stage.pop('_streams')]
        record['errors400']=sum(s.http_status==400 for s in all_streams);host.sample();save()
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    c=sub.add_parser('collect');c.add_argument('--url',required=True);c.add_argument('--model',required=True)
    c.add_argument('--packet',type=Path,required=True);c.add_argument('--binding',type=Path,required=True)
    c.add_argument('--private-cache-dir',type=Path,required=True);c.add_argument('--timeout-s',type=float,default=2700)
    c.add_argument('--out',type=Path,required=True)
    r=sub.add_parser('read');r.add_argument('--record',type=Path,required=True);r.add_argument('--out',type=Path,required=True)
    f=sub.add_parser('preflight');f.add_argument('--packet',type=Path,required=True);f.add_argument('--binding',type=Path,required=True)
    f.add_argument('--private-cache-dir',type=Path,required=True);f.add_argument('--out',type=Path,required=True)
    stop=sub.add_parser('stop');stop.add_argument('--url',required=True);stop.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('fresh-output-required')
    if a.action=='stop':
        value=post(a.url,base.CONTROL,dict(action='stop'));deadline=time.monotonic()+120
        while value.get('settled') is not True and time.monotonic()<deadline:
            time.sleep(.1);value=post(a.url,base.CONTROL,dict(action='status'))
        if not stop_proven(value):raise base.Hold('owned-stop-all-rank-FULL-proof-missing')
        a.out.write_text(json.dumps(value,indent=2)+'\n')
        intent=a.out.parent/'host-stop-intent.json'
        if not intent.exists():intent.write_text(json.dumps(dict(schema='host-stop-intent/v1',wall_ns=time.time_ns(),reason='queue-owned cleanup requested systemctl stop'))+'\n')
        print('VISION_FIT_STOP SETTLED all-rank-FULL');return 0
    if a.action=='preflight':
        inputs=load_inputs(a);host=HostMemory();host.sample()
        if host.errors:raise base.Hold('CT-cgroup-preflight-unavailable')
        value=dict(verdict='PASS',cache_shas=CACHE_SHAS,counts={k:len(v) for k,v in inputs.items()},host=host.record())
        a.out.write_text(json.dumps(value,indent=2)+'\n');print('VISION_FIT_PREFLIGHT PASS bound-caches-and-CT-memory');return 0
    if a.action=='collect':
        if a.timeout_s>2700 or a.timeout_s<=0:raise ValueError('native-cap-must-be-positive-at-most2700')
        def stop_signal(*_):raise base.Hold('owner-workload-stop-or-45min-cap')
        signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
        value=run(a)
        summary=assess(value);value['verdict']=summary['verdict'];value['collector_reasons']=summary['reasons']
        a.out.write_text(json.dumps(value,indent=2)+'\n')
    else:value=assess(json.loads(a.record.read_text()));a.out.write_text(json.dumps(value,indent=2)+'\n')
    detail=summary if a.action=='collect' else value
    native=detail.get('native_coverage')
    if native is not None:
        print('VISION_FIT_NATIVE '+json.dumps({k:native.get(k) for k in ('live_native_share','all_native_share',
            'live_native_share_strict','all_native_share_strict','handback','geometry_tail','rows')},sort_keys=True))
    print('VISION_FIT_WHOLE_PROTOCOL_NATIVE '+json.dumps(detail.get('whole_protocol_native_coverage'),sort_keys=True))
    print('VISION_FIT_MEMORY '+json.dumps(dict(minimum_mib=detail.get('minimum_mib'),
        host=value.get('host',detail.get('raw',{}).get('host')),worker_host_rss_peaks=detail.get('raw',value).get('memory',{}).get('host_rss_peaks')),sort_keys=True))
    print('VISION_FIT_8X200K '+value['verdict']+' '+str(value.get('reason',value.get('reasons',[]))))
    return {'FIT':0,'HOLD':2,'FAIL':1}[value['verdict']]

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:report_exception(exc);print('VISION_FIT_8X200K HOLD '+type(exc).__name__);raise SystemExit(2)
