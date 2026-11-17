"""Actual generated client argv through COMPLETE protocol, never VM/box/GPU."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import asyncio,base64,hashlib,json,os,re
from pathlib import Path
import subprocess,sys,tempfile,threading,time,unittest
from unittest.mock import patch

import vision_fit_8x200k as vf
import test_fit_8x200k as old

ROOT=Path(__file__).resolve().parents[1]
PACKET=Path(os.environ.get('VISION_FIT_PACKET','<hub-dir>/cargo-target/flash-next/vision-native-mm-20261116/serve-vision-fe-20261116a-build-7'))
BINDING=Path('<hub-dir>/cargo-target/flash-next/vision-native-mm-20261116/vision-fit-inputs-v<vm-id>-b-20261116.json')
CACHES=Path('<hub-dir>/cargo-target/flash-next/vision-encoder-3090-20261116/trial-b/private')

class Protocol(unittest.TestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix='vision-fit-protocol-'));self.seen=[];self.admitted=0;self.active=set();self.prompts=set();self.finished=False;self.low=False;self.layouts=[];self.cache_starts={};self.manual_owned=False;self.bad_checkpoint=False;self.expire_companion=False;self.fail_span=False;self.cold_arrived=threading.Event();self.cold_prepared=False;self.late_prepare=False
        owner=self
        from native_overlap_session import NativeOverlapSession
        from wire_v1 import ROUTE_BANK_FULL
        epoch=old.memory('e6')['engine_epoch']
        workers=[dict(worker_rank=r,engine_epoch=epoch,engine_epoch_ok=True,barrier_clean=True,drained_streams=1,
                      pending_native_replays=0,retained_rows=0,pending_prefill_streams=0,pending_output_copies=0) for r in range(4)]
        class Control:
            engine_epoch=epoch
            last_snapshot=dict(engine_epoch=epoch,route='full',route_generation=2,workers=workers)
            async def ensure_paused(self):return dict(paused=True,mode='keep',clear_cache=False)
            async def barrier(self):return dict(workers=workers)
            async def status(self):return self.last_snapshot
            async def disarm_admission(self):return dict(admission_intent=None)
            async def resume(self):return dict(resumed=True)
            async def leave_split(self):return dict(bootstrapped=True,route_after='full',engine_epoch=epoch,changed=False,typed_acks=True,
                transition_acks=[dict(worker_rank=r,typed='TransitionAck',engine_epoch=epoch,applied_bank=ROUTE_BANK_FULL,
                                     drained_streams=1,applied_route_generation=2) for r in range(4)])
        session=NativeOverlapSession(None,control=Control());session.run_id='serve-existing-session'
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*_):pass
            def json(self,value,status=200):
                raw=json.dumps(value).encode();self.send_response(status);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
            def do_GET(self):self.json(dict(data=[dict(id='Qwen3.8-Flash-Next')]),200 if self.headers.get('Authorization')=='Bearer '+'a'*64 else 401)
            def ids(self,messages):
                ids=[50,51]
                for message in messages:
                    if message['role']=='assistant':raise AssertionError('the fit uses exact token-prefix continuation')
                    ids.append(101)
                    for part in message['content']:
                        if part['type']=='image_embeds':ids.extend([vf.IMAGE_TOKEN]*1024+[248054])
                        else:
                            marker=r'@@GEN:(\d+)@@<\|im_end\|>\n<\|im_start\|>user\n'
                            parts=re.split(marker,part['text'])
                            for index,text in enumerate(parts):
                                if index%2:
                                    ids.extend([102,200,201]+[123]*int(text)+[222,101]);continue
                                unique=int(hashlib.sha256(text.split(' x')[0].encode()).hexdigest()[:6],16)%200000
                                ids.extend([unique]+[111]*(text.count(' x')+30))
                    ids.append(102)
                result=ids+[200,201]
                end=max(i for i,t in enumerate(result) if t==vf.IMAGE_TOKEN)+2
                text=next(p['text'] for p in messages[-1]['content'] if p['type']=='text')
                owner.layouts.append((result[end:],text))
                return result
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));owner.seen.append(self.path)
                if self.headers.get('Authorization')!='Bearer '+'a'*64:self.json({},401);return
                if self.path in ('/v1/flash_s5_native_overlap','/v1/flash_s5_native_qualification_worker'):
                    action=body['action']
                    payload=dict(body,run_id=session.run_id,engine_epoch=epoch,route='full',route_generation=2,observed_clock_ns=time.monotonic_ns(),
                        mode=body.get('mode'),installed='burst',fused=True,released=True,resumed=True,previous=False,
                        size=body.get('size'),local=True,max_new=8192,per_bulk=8,handback_steps=16,reentry_retries=4,
                        auto_arm=dict(manual_owned=owner.manual_owned,active=False,phase='idle'),scheduler_requests=0)
                    if action in ('prepare','pause'):
                        with patch('native_readiness.implementation_status',return_value=dict(native_ready=True,armable=True,scope='synthetic construction leaf')):
                            payload=asyncio.run(session.execute(body))
                        owner.manual_owned=True
                    if action=='release_auto_arm':owner.manual_owned=False
                    if action=='qualification_worker':
                        requested=body['payload'];after=dict(selected=requested.get('on'))
                        after.update({k:dict(selected=v) for k,v in requested.items() if k!='on'})
                        payload['worker_reply']=dict(qualification=[dict(worker_rank=r,qualification=dict(worker_action=body['worker_action'],
                            engine_supplied_verdict=False,before={},after=after,counters={},order=dict(ok=True))) for r in range(4)])
                    self.json(payload);return
                if self.path==vf.base.CONTROL:
                    if body['action'] in ('finish','stop'):owner.finished=True;owner.cold_arrived.set()
                    value=old.memory('e6',owner.finished);value['native_admissions']=owner.admitted
                    value['native_coverage']=old.native_counter(owner.admitted>0)
                    if owner.admitted:
                        value['native_coverage']['native_turns']=owner.admitted
                        value['native_coverage']['rows']['live']['native']['fixture']=owner.admitted*1000000
                        value['native_coverage']['requests']['live']['native']['fixture']=owner.admitted
                    value['first_admissions']=dict(owner.cache_starts)
                    value['active_request_ids']=sorted(owner.active)
                    value['simultaneous_resident_token_peak'].update(eight_at_190k_seen=True,eight_minimum_peak=190320)
                    value['host_rss_peaks']={str(r):dict(VmRSS=123456,VmHWM=130000,RssAnon=111111,RssShmem=12345) for r in range(4)}
                    value['mm_peaks']={str(r):dict(tower_registered_cuda_bytes=0,cache_storage_bytes=40*2**20 if r<2 else 0) for r in range(4)}
                    if owner.low:value['min_free_bytes']['1']=vf.base.FLOOR-1;value['hold']='physical-free-below-40MiB'
                    self.json(value);return
                if self.path=='/tokenize':
                    if any('vf-cold-probe' in str(part.get('text','')) for m in body['messages'] for part in m['content']):
                        if any(uid.startswith('vf-cold-companion-') for uid in owner.active):
                            owner.late_prepare=True;self.json({},400);return
                        owner.cold_prepared=True
                    ids=self.ids(body['messages']);self.json(dict(count=len(ids),tokens=ids,max_model_len=200000));return
                if self.path=='/detokenize':
                    tokens=body['tokens'];matches=[(prefix,text) for prefix,text in owner.layouts if tokens[:len(prefix)]==prefix]
                    if not matches:raise AssertionError('unknown decoder-prefix fixture')
                    prefix,text=max(matches,key=lambda item:len(item[0]));reply=tokens[len(prefix):]
                    assert all(t==123 for t in reply)
                    self.json(dict(prompt=text+'@@GEN:'+str(len(reply))+'@@'));return
                if self.path!='/v1/chat/completions':self.json({},404);return
                owner.admitted+=1
                if not body.get('stream'):
                    question=body['messages'][0]['content'][-1]['text'];cases=json.loads((PACKET/'vision-assets/cases.json').read_text())
                    answer=next(c['expected'] for c in cases if c['question']==question)
                    self.json(dict(choices=[dict(message=dict(content=answer))]));return
                uid=body['request_id'];ids=self.ids(body['messages']);owner.active.add(uid)
                feature=next(p['image_embeds'] for m in body['messages'] if m['role']=='user' for p in m['content'] if p['type']=='image_embeds')
                key=(vf.base.sha(ids),hashlib.sha256(json.dumps(feature).encode()).hexdigest())
                warm=uid.startswith('vf-cold-companion-') and len([v for k,v in owner.cache_starts.items() if k.startswith('chatcmpl-vf-cold-prime-')])==7
                cached=190080 if warm else (len(ids)//4048*4048 if key in owner.prompts else 0)
                if not owner.manual_owned and len(ids)==4096:cached=0  # production's native lowering
                owner.prompts.add(key)
                hit=0 if owner.bad_checkpoint else cached
                owner.cache_starts['chatcmpl-'+uid+'-frozen']=dict(start=hit,prompt_tokens=len(ids),transaction=owner.admitted,engine_epoch=epoch)
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                def emit(value):self.wfile.write(b'data: '+json.dumps(value).encode()+b'\n\n');self.wfile.flush()
                if uid=='vf-cold-probe':
                    owner.cold_arrived.set()
                    if owner.fail_span:time.sleep(.3)
                try:
                    emit(dict(id='chatcmpl-'+uid,object='chat.completion.chunk',created=1,model='Qwen3.8-Flash-Next',prompt_token_ids=ids,choices=[]))
                    time.sleep(.4 if uid=='vf-cancel-reuse' and 'again' not in uid else .02)
                    emit(dict(choices=[dict(index=0,delta=dict(content='fixture'),token_ids=[123],finish_reason=None)]))
                    if 'cold-companion' in uid and not owner.expire_companion:
                        assert owner.cold_arrived.wait(30)
                        time.sleep(.01 if owner.fail_span else .4)
                    else:time.sleep(.02)
                    cap=body['max_tokens'];emit(dict(choices=[dict(index=0,delta=dict(content='fixture'),token_ids=[123]*(cap-1),finish_reason='length')],usage=dict(prompt_tokens=len(ids),completion_tokens=cap,prompt_tokens_details=dict(cached_tokens=cached))))
                    self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
                except (BrokenPipeError,ConnectionResetError):pass
                finally:owner.active.discard(uid)
        self.http=ThreadingHTTPServer(('127.0.0.1',0),Handler);self.http.daemon_threads=True
        threading.Thread(target=self.http.serve_forever,daemon=True).start();self.addCleanup(self.http.server_close);self.addCleanup(self.http.shutdown)
        self.url='http://127.0.0.1:'+str(self.http.server_port)
        # Root hub cgroup has no memory.events; model the CT cgroup filesystem leaf only.
        shim=self.root/'shim';shim.mkdir()
        (shim/'sitecustomize.py').write_text("from pathlib import Path\n_old=Path.read_text\ndef read(self,*a,**k):\n s=str(self)\n if s=='/sys/fs/cgroup/memory.stat':return 'anon 100\\nshmem 50\\nfile 80\\n'\n if s=='/sys/fs/cgroup/memory.events':return 'oom 0\\noom_kill 0\\n'\n if s=='/sys/fs/cgroup/memory.current':return '200'\n return _old(self,*a,**k)\nPath.read_text=read\n")
        self.shim=shim
    def argv(self,name):
        return [sys.executable,'-B',str(Path(vf.__file__)),'collect','--url',self.url,'--model','Qwen3.8-Flash-Next',
            '--packet',str(PACKET),'--binding',str(BINDING),'--private-cache-dir',str(CACHES),'--timeout-s','120','--out',str(self.root/name)]
    def run_cli(self,argv,name):
        (self.root/(name+'.argv.json')).write_text(json.dumps(argv))
        result=subprocess.run(argv,env=dict(os.environ,VLLM_API_KEY='a'*64,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(self.shim)+os.pathsep+os.environ.get('PYTHONPATH','')),capture_output=True,text=True,timeout=150)
        (self.root/(name+'.stdout')).write_text(result.stdout);(self.root/(name+'.stderr')).write_text(result.stderr)
        return result
    def test_actual_argv_complete_then_reader(self):
        result=self.run_cli(self.argv('complete.json'),'collect')
        self.assertEqual(result.returncode,0,result.stderr)
        value=json.loads((self.root/'complete.json').read_text());self.assertTrue(value['complete'],value.get('reason'))
        argv=[sys.executable,'-B',str(Path(vf.__file__)),'read','--record',str(self.root/'complete.json'),'--out',str(self.root/'read.json')]
        read=self.run_cli(argv,'read');self.assertEqual(read.returncode,0,read.stdout+read.stderr)
        self.assertIn('VISION_FIT_8X200K FIT',read.stdout)
        self.assertTrue(self.finished);self.assertIn('/detokenize',self.seen)
        self.assertEqual([s['prompt_tokens'] for s in value['stages']],list(vf.PREFIXES));self.assertTrue(value['cancel']['retired'])
    def test_low_peer_actual_argv_holds_and_settles(self):
        self.low=True;result=self.run_cli(self.argv('low.json'),'low');self.assertEqual(result.returncode,2)
        self.assertTrue(self.finished);self.assertIn('physical-free-below-40MiB',result.stderr)
    def test_reader_missing_complete_work_holds(self):
        record=self.root/'missing.json';record.write_text('{}')
        result=self.run_cli([sys.executable,'-B',str(Path(vf.__file__)),'read','--record',str(record),'--out',str(self.root/'missing-read.json')],'missing')
        self.assertEqual(result.returncode,2);self.assertIn('VISION_FIT_8X200K HOLD',result.stdout)

    def test_actual_bad_checkpoint_holds_and_retains_companion_records(self):
        self.bad_checkpoint=True
        result=self.run_cli(self.argv('bad-checkpoint.json'),'bad-checkpoint')
        self.assertEqual(result.returncode,2,result.stderr)
        row=json.loads((self.root/'bad-checkpoint.json').read_text())
        self.assertIn('cached-companion-checkpoint-not-ready',row['reason'])
        self.assertEqual(len(row['cold']['conversations']),7)
        self.assertEqual(row['memory']['hold'],None)  # stub cleanup; real observer records client-stop

    def test_actual_failed_span_retains_eight_records_and_real_admissions(self):
        self.fail_span=True
        result=self.run_cli(self.argv('failed-span.json'),'failed-span')
        self.assertEqual(result.returncode,2,result.stderr)
        row=json.loads((self.root/'failed-span.json').read_text())
        self.assertIn('seven-decoders-did-not-span-cold-prefill',row['reason'])
        self.assertEqual(len(row['cold']['conversations']),8)
        self.assertGreater(row['cold']['native_admissions'],0)
        self.assertEqual(len(row['cold']['cache_readiness']),7)
        self.assertTrue(all(s['end_ns'] for s in row['cold']['conversations']))

    def test_full_lease_revert_mutant_cannot_pass_reuse_calibration(self):
        source=Path(vf.__file__).read_text().replace('box.keep_lease(client,lease)',
            "client.call('release_auto_arm');lease['native_off']=dict(manual_owned=False)")
        script=self.root/'native-calibration-mutant.py';script.write_text(source)
        argv=self.argv('native-calibration.json');argv[2]=str(script)
        result=self.run_cli(argv,'native-calibration-mutant')
        self.assertEqual(result.returncode,2,result.stderr)
        row=json.loads((self.root/'native-calibration.json').read_text())
        self.assertEqual(row['reuse']['same_image_cached'],0)
        self.assertIn('MM-prefix-reuse-unproven',row['collector_reasons'])

    def test_cold_prepare_after_first_outputs_revert_mutant_is_exposed(self):
        source=Path(vf.__file__).read_text()
        source=source.replace("        cold=prepare(args,features[7],190000,'vf-cold-probe')\n",'',1)
        source=source.replace("record['cold']['live_at_arrival']=7;current.append(start(cold));wait(current)",
            "record['cold']['live_at_arrival']=7;cold=prepare(args,features[7],190000,'vf-cold-probe');current.append(start(cold));wait(current)")
        script=self.root/'late-prepare-mutant.py';script.write_text(source)
        argv=self.argv('late-prepare.json');argv[2]=str(script)
        result=self.run_cli(argv,'late-prepare-mutant')
        self.assertEqual(result.returncode,2,result.stderr);self.assertTrue(self.late_prepare)
        row=json.loads((self.root/'late-prepare.json').read_text())
        self.assertEqual(len(row['cold']['conversations']),7)

    def test_actual_preflight_bad_binding_refuses_before_any_http(self):
        binding=self.root/'bad-binding.json';binding.write_text('{}')
        argv=[sys.executable,'-B',str(Path(vf.__file__)),'preflight','--packet',str(PACKET),
            '--binding',str(binding),'--private-cache-dir',str(CACHES),'--out',str(self.root/'preflight.json')]
        result=self.run_cli(argv,'preflight-bad')
        self.assertEqual(result.returncode,2);self.assertEqual(self.seen,[])
        self.assertIn('input-binding-SHA',result.stderr)

    def test_actual_owned_stop_to_all_rank_settled_verdict(self):
        argv=[sys.executable,'-B',str(Path(vf.__file__)),'stop','--url',self.url,'--out',str(self.root/'manual-stop.json')]
        result=self.run_cli(argv,'stop')
        self.assertEqual(result.returncode,0,result.stderr);self.assertTrue(self.finished)
        self.assertIn('VISION_FIT_STOP SETTLED all-rank-FULL',result.stdout)
        self.assertTrue((self.root/'host-stop-intent.json').exists())

if __name__=='__main__':unittest.main()
