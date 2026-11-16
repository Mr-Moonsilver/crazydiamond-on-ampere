"""Actual CLI through a schema-valid full protocol; no server/engine/GPU outside loopback."""
import copy
import asyncio
from contextlib import nullcontext
import inspect
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import fit_8x200k as fit
from packet_client_contract import full_protocol_result

HERE = Path(__file__).resolve().parent
W20 = json.loads((HERE/'fit_8x200k_w20.json').read_text())
sys.path.insert(0,str(HERE.parent/'patches/s5_engine'))


def native_counter(done=False):
    from scheduler_hooks import PrefillCoverage
    row=PrefillCoverage().record()
    if done:
        row['rows']['live']['native']['fixture']=1000000
        row['requests']['live']['native']['fixture']=40
        row['native_turns']=40
    return row


def startup():
    peers={}
    for rank in range(4):
        before=dict(device_index=rank,total_bytes=24*2**30,free_bytes=2**30,
                    allocated_bytes=22*2**30,reserved_bytes=23*2**30,observed_ns=1)
        after=dict(before,observed_ns=2)
        peers[str(rank)]=dict(verdict='PASS',limits=dict(total_delta_bytes=4*2**20,outside_delta_bytes=4*2**20,
            outside_absolute_bytes=(1259,1371,1197,1309)[rank]*2**20),receipts=[dict(rank=rank,phase='before',snapshot=before),dict(rank=rank,phase='after',before=before,after=after)])
    return dict(verdict='PASS',ranks=list(range(4)),peers=peers)


def memory(role,done=False):
    return dict(schema='fit-memory/1', role=role, settled=done, hold=None,engine_epoch=W20['native'].get(role,{}).get('engine_epoch'),
                configuration=dict(max_model_len=200000,max_num_seqs=8,kv_cache_memory_bytes=2150000000,
                    tensor_parallel_size=2,pipeline_parallel_size=2,rowcache_reqs='8',
                    rowcache_slots={'e6':'6144','e4096':'4096','stock':'10240'}[role]),
                startup=startup(),
                min_free_bytes={str(r):128*2**20 for r in range(4)},
                preemptions=0, ooms=0, fatal=None, errors400=0,
                native_admissions=128 if done else 0,
                stop_proof=dict(route='full' if done else None,all_rank_stop=done,admission_disarmed=done,output_queue_empty=done),
                permit_latency={'chunk':dict(count=128 if done else 0,p50_ms=0.5 if done else None,max_ms=2.0 if done else None)},
                simultaneous_resident_token_peak=dict(tokens=1590000 if done else 0,samples=100,unknown=0,basis='synthetic scheduler frontiers'),
                coverage=dict(decode_rounds=32000, decode_checked=32000, chunks=128,
                              chunk_checked=128, checks=32128), native_coverage=native_counter(done))


def complete(role='e6'):
    member = dict(output_tokens=fit.OUTPUT, error=None)
    return dict(schema='fit-8x200k-client/1', role=role, complete=True, memory=memory(role,True),
                initial_native_coverage=native_counter(),errors400=0,
                auth=dict(auth='key',unauthenticated_status=401,authenticated_status=200),
                stages=[dict(prompt_tokens=n, complete=True, native_admissions=32, conversations=[dict(member,prompt_tokens=n,output_tokens=fit.OUTPUT_CAPS[n]) for _ in range(8)])
                        for n in fit.PREFIXES],
                cold=dict(live_at_arrival=7,native_admissions=8,conversations=[dict(member,prompt_tokens=fit.PREFIXES[-1],first_ns=10,end_ns=40) for _ in range(7)]+[dict(member,prompt_tokens=fit.COLD_HISTORY,resume_ns=20,first_ns=30)]))


class Rule(unittest.TestCase):
    def test_complete_and_each_peer_floor(self):
        self.assertEqual(fit.assess(complete())['verdict'],'FIT')
        for rank in range(4):
            row=complete();row['memory']['min_free_bytes'][str(rank)]=fit.FLOOR-1
            self.assertEqual(fit.assess(row)['verdict'],'HOLD')

    def test_no_fit_from_unknown_or_missing_work(self):
        mutations=(lambda x:x['memory'].update(native_coverage=native_counter()),
                   lambda x:x['memory']['startup']['peers'].pop('1'),
                   lambda x:x['memory']['min_free_bytes'].pop('3'),
                   lambda x:x['memory'].update(preemptions=None),
                   lambda x:x['memory']['coverage'].update(decode_checked=31999),
                   lambda x:x['stages'].pop(),
                   lambda x:x['stages'][2]['conversations'].pop(),
                   lambda x:x['cold'].update(live_at_arrival=6))
        for mutate in mutations:
            row=complete();mutate(row);self.assertEqual(fit.assess(row)['verdict'],'HOLD')
        row=complete();row['memory']['ooms']=1;self.assertEqual(fit.assess(row)['verdict'],'FAIL')

    def test_zero_native_stage_never_fits_even_with_perfect_row_fractions(self):
        for role in ('e6','e4096'):
            for section in ('stages','cold'):
                row=complete(role)
                target=row['stages'][0] if section=='stages' else row['cold']
                target['native_admissions']=0
                result=fit.assess(row)
                self.assertEqual(result['verdict'],'HOLD')
                self.assertIn('no-native-execution',result['reasons'])
        original=inspect.getsource(fit.assess)
        mutant=original.replace("reasons.append('no-native-execution')","pass # revert native admission guard")
        namespace=dict(fit.__dict__);exec(mutant,namespace)
        row=complete();row['stages'][0]['native_admissions']=0
        self.assertEqual(namespace['assess'](row)['verdict'],'FIT')

    def test_keyed_serve_proof_all_roles_and_keyless_mutant(self):
        for role in fit.ROLES:
            row=complete(role);row.pop('auth')
            self.assertIn('keyless-test-serve',fit.assess(row)['reasons'])
        class Response:
            def __init__(self,status):self.status=status
            def read(self):return b'{}'
        class Connection:
            def __init__(self,*args,**kwargs):pass
            def request(self,*args,**kwargs):pass
            def getresponse(self):return Response(codes.pop(0))
            def close(self):pass
        with mock.patch.dict('os.environ',{'VLLM_API_KEY':'1'*64}),mock.patch.object(fit.http.client,'HTTPConnection',Connection):
            codes=[401,200];self.assertEqual(fit.keyed_serve('http://127.0.0.1:8000')['auth'],'key')
            codes=[200,200]
            with self.assertRaisesRegex(fit.Hold,'keyless-test-serve'):fit.keyed_serve('http://127.0.0.1:8000')
        with mock.patch.dict('os.environ',{},clear=True):
            with self.assertRaisesRegex(fit.Hold,'keyless-test-serve'):fit.keyed_serve('http://127.0.0.1:8000')

    def test_growth_keeps_generated_reply_exactly(self):
        old=fit.initial(1,60000);reply=list(range(8000))
        new=fit.grow(old,reply,1,120000)
        self.assertEqual(new[:68000],old+reply);self.assertEqual(len(new),120000)
        self.assertEqual(old[:64],fit.initial(2,60000)[:64])
        self.assertNotEqual(old[64:],fit.initial(2,60000)[64:])

    def test_rule_and_coverage_mutants(self):
        source=inspect.getsource(fit.assess)
        for old,new in (("x < FLOOR","x < 0"),
                        ("coverage.get('decode_checked') != coverage.get('decode_rounds')","False")):
            self.assertEqual(source.count(old),1)
            namespace=dict(fit.__dict__);exec(source.replace(old,new),namespace)
            row=complete()
            if 'FLOOR' in old:row['memory']['min_free_bytes']['1']=fit.FLOOR-1
            else:row['memory']['coverage']['decode_checked']-=1
            self.assertEqual(namespace['assess'](row)['verdict'],'FIT')
            self.assertEqual(fit.assess(row)['verdict'],'HOLD')


class FullProtocol(unittest.TestCase):
    def test_actual_collect_argv_to_final_verdict_then_actual_reader(self):
        for role in fit.ROLES:
            with self.subTest(role=role):self.protocol(role)

    def protocol(self,role):
        from native_overlap_session import NativeOverlapSession
        from wire_v1 import ROUTE_BANK_FULL
        seen=[];cold=threading.Event();settling=[False]
        epoch=W20['native'].get(role,{}).get('engine_epoch')
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
            async def leave_split(self):
                return dict(bootstrapped=True,route_after='full',engine_epoch=epoch,changed=False,typed_acks=True,
                    transition_acks=[dict(worker_rank=r,typed='TransitionAck',engine_epoch=epoch,applied_bank=ROUTE_BANK_FULL,
                                          drained_streams=1,applied_route_generation=2) for r in range(4)])
        session=NativeOverlapSession(None,control=Control())
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                assert self.path=='/v1/models'
                keyed=self.headers.get('Authorization')=='Bearer '+'1'*64
                seen.append('keyed-models' if keyed else 'unauth-models')
                self.send_response(200 if keyed else 401);self.end_headers();self.wfile.write(b'{}')
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path in ('/v1/flash_s5_native_overlap','/v1/flash_s5_native_qualification_worker'):
                    action=body['action'];seen.append('native-'+action)
                    payload=dict(body,run_id=session.run_id,engine_epoch=epoch,route='full',route_generation=2,observed_clock_ns=time.monotonic_ns(),
                                 mode=body.get('mode'),installed='burst',fused=True,released=True,resumed=True,
                                 previous=False,size=body.get('size'),local=True,
                                 max_new=8192,per_bulk=8,handback_steps=16,reentry_retries=4)
                    if action in ('prepare','pause'):
                        # Execute the real identity state machine: unprepared pause
                        # responds run_id:null, never an echo of the command ID.
                        import native_readiness
                        with mock.patch.object(native_readiness,'implementation_status',return_value=dict(native_ready=True,armable=True,scope='synthetic construction leaf')):
                            payload=asyncio.run(session.execute(body))
                    if action=='qualification_worker':
                        requested=body['payload'];after=dict(selected=requested.get('on'))
                        after.update({k:dict(selected=v) for k,v in requested.items() if k!='on'})
                        payload['worker_reply']=dict(qualification=[dict(worker_rank=r,qualification=dict(worker_action=body['worker_action'],
                            engine_supplied_verdict=False,before={},after=after,counters={},order=dict(ok=True))) for r in range(4)])
                    data=json.dumps(payload).encode();self.send_response(200);self.send_header('Content-Length',str(len(data)))
                    self.end_headers();self.wfile.write(data);return
                if self.path==fit.CONTROL:
                    seen.append(body['action'])
                    time.sleep(.002) # utility reply follows an independent server turn
                    value=memory(role,settling[0])
                    if body['action'] in ('finish','stop'):
                        settling[0]=True;value=memory(role,False)
                    if seen.count('status')==1 and body['action']=='status':
                        value['startup']=dict(verdict='HOLD',ranks=[],peers={})
                    value['native_admissions']=sum(x.startswith('growth-') or x.startswith('cold-') for x in seen)
                    payload=json.dumps(value).encode()
                    self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(payload)))
                    self.end_headers();self.wfile.write(payload);return
                if self.path!='/v1/completions':raise AssertionError(self.path)
                uid=body['request_id'];n=len(body['prompt'])
                cap=fit.OUTPUT_CAPS[n]
                assert body['return_token_ids'] is True and body['max_tokens']==cap
                step='cold-companion' if '-cold-companion-' in uid else 'cold-probe' if '-cold-probe' in uid else 'growth-'+str(n)
                seen.append(step)
                if step=='cold-probe':cold.set()
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                import uuid
                server_id='cmpl-'+uuid.uuid4().hex
                assert server_id!=uid
                first=dict(id=server_id,text='',choices=[dict(token_ids=[123],finish_reason=None)])
                self.wfile.write(b'data: '+json.dumps(first).encode()+b'\n\n');self.wfile.flush()
                if step=='cold-companion':
                    assert cold.wait(20)
                    time.sleep(0.2)
                last=dict(id=server_id,text='',choices=[dict(token_ids=[123]*(cap-1),finish_reason='length')],
                          usage=dict(prompt_tokens=n,completion_tokens=cap,prompt_tokens_details=dict(cached_tokens=0 if n==60000 else 60000)))
                self.wfile.write(b'data: '+json.dumps(last).encode()+b'\n\ndata: [DONE]\n\n')
                self.wfile.flush()
            def log_message(self,*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=True
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with nullcontext(tempfile.mkdtemp(prefix='fit-protocol-')) as root:
                root=Path(root)
                # Same argv shape that is frozen in the owner phase; only roots/loopback port vary.
                argv=[sys.executable,'-B',str(HERE/'fit_8x200k.py'),'collect','--url',f'http://127.0.0.1:{server.server_port}',
                      '--model','fixture-model','--role',role,'--timeout-s','30','--out',str(root/'collect.json')]
                binding=HERE/'packet-client-commands.json'
                if binding.exists():
                    commands=json.loads(binding.read_text())['commands']
                    generated=next(c['python_argv'] for c in commands if c['label']=='fit-collect' and '--role' in c['python_argv'] and c['python_argv'][c['python_argv'].index('--role')+1]==role)
                    argv=[sys.executable,'-B']+generated
                    for option,value in (('--url',f'http://127.0.0.1:{server.server_port}'),('--out',str(root/'collect.json')),('--timeout-s','30')):
                        argv[argv.index(option)+1]=value
                import os
                parent=next(Path('<hub-dir>/cargo-target/flash-next/cx-20261114i/cohort-fx42/build-b1').glob('*bundle*')) if not (HERE/'n3j2_packet.py').exists() else HERE
                childenv=dict(os.environ,PYTHONPATH=os.pathsep.join((str(HERE/'e3_builder'),str(parent),os.environ.get('PYTHONPATH',''))),VLLM_API_KEY='1'*64)
                done=subprocess.run(argv,capture_output=True,text=True,timeout=45,env=childenv)
                expected=['unauth-models','keyed-models']+(['native-prepare'] if role!='stock' else [])+['status','growth-60000','growth-120000','growth-160000','growth-192000','cold-companion','cold-probe','finish']
                full_protocol_result(argv,stdout=done.stdout,stderr=done.stderr,returncode=done.returncode,
                                     verdict_prefix='FIT_8X200K ',expected_verdict='FIT',expected_steps=expected,observed_steps=seen)
                row=json.loads((root/'collect.json').read_text())
                self.assertTrue(row['complete']);self.assertEqual(fit.assess(row)['verdict'],'FIT')
                reader=[sys.executable,'-B',str(HERE/'fit_8x200k.py'),'read','--record',str(root/'collect.json'),'--out',str(root/'read.json')]
                if binding.exists():
                    generated=next(c['python_argv'] for c in commands if c['label']=='fit-read' and fit.NAME+'-'+role in c['python_argv'][c['python_argv'].index('--record')+1])
                    reader=[sys.executable,'-B']+generated
                    for option,value in (('--record',str(root/'collect.json')),('--out',str(root/'read.json'))):reader[reader.index(option)+1]=value
                read=subprocess.run(reader,capture_output=True,text=True,timeout=10,env=childenv)
                full_protocol_result(reader,stdout=read.stdout,stderr=read.stderr,returncode=read.returncode,
                                     verdict_prefix='FIT_8X200K ',expected_verdict='FIT',expected_steps=['complete-record'],observed_steps=['complete-record'])
                if role!='stock':
                    original=(HERE/'fit_8x200k.py').read_text();anchor="record['prepare']=client.prepare('full')"
                    self.assertEqual(original.count(anchor),1)
                    mutant_source=root/'fit_without_prepare.py';mutant_source.write_text(original.replace(anchor,'pass # revert omitted prepare'))
                    bad=list(argv);bad[2]=str(mutant_source);bad[bad.index('--out')+1]=str(root/'without-prepare.json')
                    session.run_id=None;before=len(seen)
                    failure=subprocess.run(bad,capture_output=True,text=True,timeout=15,env=childenv)
                    self.assertEqual(failure.returncode,2,failure.stderr)
                    self.assertIn('response-identity-differs',failure.stderr)
                    self.assertFalse(any(x.startswith('growth-') for x in seen[before:]))
                    self.assertIsNone(session.run_id)
                # Real parser mutant: the very same command must refuse before any HTTP request.
                mutant=subprocess.run(argv+['--unregistered-option'],capture_output=True,text=True,timeout=10)
                self.assertEqual(mutant.returncode,2);self.assertIn('unrecognized arguments',mutant.stderr)
                # A late reader failure cannot pass a first-I/O check as full protocol.
                with self.assertRaises(ValueError):
                    full_protocol_result(reader,stdout='',stderr='ValueError at reader.py:1',returncode=2,
                                         verdict_prefix='FIT_8X200K ',expected_verdict='FIT',expected_steps=['complete-record'],observed_steps=['complete-record'])
        finally:
            server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
