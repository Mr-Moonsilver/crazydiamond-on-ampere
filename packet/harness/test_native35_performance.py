"""Native35 P: exact queue argv for W1 and both W2 rounds through real validation, the real reviewed
package and the real session/SSE clients; the aggregate grade over actual driver source packets. CPU only.

Fixture leaves (declared in native35_p_fixture_leaves.json): LoopbackAPIServer stands in for the
engine APIServer's two routes (the native session POST and /v1/completions SSE) on 127.0.0.1;
http.client.HTTPConnection.request is replaced only where a test stops at the first network send;
the decode SSE counts inside an otherwise real source packet window are declared numbers.
"""
import copy
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import shlex
from string import Template
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path[:0]=[str(HERE),str(ROOT/'patches/s5_engine')]
import e3_native35_performance as rounds
import native35_performance_grade as aggregate
import e3_native_performance as perf
import e3_native_w1 as native_w1
import e3_overlap_performance as shared
from wire_v1 import ROUTE_BANK_FULL,ROUTE_BANK_SPLIT

QUEUE=HERE/'runs/q18a_native35_p.draft.sh'
SETTINGS=HERE/'runs/q18a_native35_p_settings.sh'
PERF_RECEIPTS=[]
KEY='native35-p-cpu-transport-fixture'
RUN_ID='q18a-native35-p-20261031T120000Z-4242'
EPOCH=7
# The reviewed package: inside the frozen bundle it is the bundle's own overlay;
# on the hub tree it is the byte-identical frozen O overlay P inherits.
OVERLAY=HERE/'native30_overlay'
if not OVERLAY.is_dir():
    OVERLAY=Path('<hub-dir>/agents/flash-next/tmp/native35-o-held-a/q18a-native35-o-bundle-188f4d5ca79a/native30_overlay')


def settings_read_timeout():
    text=SETTINGS.read_text()
    assert 'export Q18A_FIRST_ADVANCE_READ_TIMEOUT_S\n' in text
    values=re.findall(r'(?m)^Q18A_FIRST_ADVANCE_READ_TIMEOUT_S=(\S+)$',text)
    assert len(values)==1;return values[0]


def invocation(start,stop):
    """One exact queue invocation: env assignments then argv, continuation lines joined."""
    text=QUEUE.read_text();assert text.count(start)==1
    body=text[text.index(start):];body=body[:body.index(stop)]
    return shlex.split(body.replace('\\\n',' '))


def substitute(tokens,values):
    env,argv={},[]
    for token in tokens:
        token=Template(token).substitute(values)
        if not argv and re.fullmatch(r'[A-Z_][A-Z0-9_]*=.*',token):
            name,_,value=token.partition('=');env[name]=value
        else:argv.append(token)
    return env,argv


def queue_values(directory,url,round_index=0):
    return dict(H=str(HERE),Q=str(directory),PROBE_URL=url,MODEL='qwen38-w8a8-tp4',RUN_ID=RUN_ID,EPOCH=str(EPOCH),
                _round=str(round_index),Q18A_READINESS=str(OVERLAY/'e3/native-overlap-readiness.json'),
                S5_OVERLAY=str(OVERLAY),S5_LAUNCHER_NAME='run-flash-s5-test.sh',
                Q18A_FIRST_ADVANCE_READ_TIMEOUT_S=settings_read_timeout())


def w2_argv(directory,url,round_index):
    tokens=invocation('    READINESS=$Q18A_READINESS GEN=','> "$Q/overlap-round-$_round.log"')
    env,argv=substitute(tokens,queue_values(directory,url,round_index))
    assert argv[:2]==['python3',str(HERE/'e3_native35_performance.py')],argv[:2]
    return env,argv[2:]


def w1_argv(directory,url):
    tokens=invocation('  WORKLOAD=$H/workloads/q17b-w1.json','> "$Q/w1.log"')
    env,argv=substitute(tokens,queue_values(directory,url))
    assert argv[:2]==['python3',str(HERE/'e3_native_w1.py')],argv[:2]
    return env,argv[2:]


def write_gates(directory,verdicts=None):
    """The queue's own PY_GATES producer, executed exactly as the P draft holds it."""
    text=QUEUE.read_text();begin=text.index("python3 - \"$Q/gates.json\" <<'PY_GATES'\n")
    code=text[text.index('\n',begin)+1:text.index('\nPY_GATES\n',begin)]
    names=('startup','row1','rule4_controls','rule4_candidate','row4','w1_controls')
    verdicts=verdicts or {name:'PASS' for name in names}
    env=dict(os.environ,Q18A_PHASE_VERDICTS=json.dumps(verdicts,separators=(',',':')),RULE4_CONTINUATION='0')
    import subprocess
    done=subprocess.run([sys.executable,'-B','-',str(directory/'gates.json')],input=code,text=True,env=env,capture_output=True)
    return done.returncode


def round_env(extra):
    env=dict(os.environ,VLLM_API_KEY=KEY,Q18A_FIRST_ADVANCE_READ_TIMEOUT_S=settings_read_timeout());env.update(extra)
    for name in ('S5_LAUNCHER_SHA',):env.pop(name,None)
    return env


class LoopbackAPIServer:
    """Declared leaf for the engine APIServer: native session route plus /v1/completions SSE.

    Session replies carry the exact typed shapes NativeRuntimeClient validates
    (four TransitionAcks, awaited keep pause with a clean four-rank barrier).
    ``prefill_begin`` is refused with 409 as the real session refuses it without
    armed native admission, so each round stops after its FULL-solo lane.
    """

    def __init__(self,prompt_tokens=4096):
        self.commands,self.completions=[],[];self.lock=threading.Lock();self.generation=0;self.prompt_tokens=prompt_tokens
        outer=self

        class Handler(BaseHTTPRequestHandler):
            protocol_version='HTTP/1.1'

            def log_message(self,*args):pass

            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.headers.get('Authorization')!='Bearer '+KEY:
                    return self.reply(401,{'error':'auth'})
                if self.path=='/v1/flash_s5_native_overlap':return outer.session(self,body)
                if self.path=='/v1/completions':return outer.completion(self,body)
                return self.reply(404,{'error':'path'})

            def reply(self,status,obj):
                data=json.dumps(obj).encode()
                self.send_response(status);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(data)));self.send_header('Connection','close');self.end_headers()
                self.wfile.write(data);self.close_connection=True

        self.httpd=ThreadingHTTPServer(('127.0.0.1',0),Handler);self.httpd.daemon_threads=True
        threading.Thread(target=self.httpd.serve_forever,daemon=True).start()
        self.url='http://127.0.0.1:%d'%self.httpd.server_address[1]

    def close(self):
        self.httpd.shutdown();self.httpd.server_close()

    def session(self,handler,command):
        with self.lock:self.commands.append(command)
        action=command['action'];reply={'schema':1,'action':action,'run_id':command['run_id'],'engine_epoch':EPOCH}
        if action in ('prepare','leave_full'):
            route=command['target_route'] if action=='prepare' else 'full'
            with self.lock:self.generation+=1;generation=self.generation
            reply.update(route=route,typed_acks=True,transition_acks=[{'typed':'TransitionAck','worker_rank':rank,'engine_epoch':EPOCH,
                'applied_bank':ROUTE_BANK_FULL if route=='full' else ROUTE_BANK_SPLIT,'drained_streams':1,'applied_route_generation':generation} for rank in range(4)])
        elif action=='pause':
            reply.update(pause={'paused':True,'mode':'keep','clear_cache':False},barrier={'workers':[{'worker_rank':rank,'engine_epoch':EPOCH,
                'engine_epoch_ok':True,'barrier_clean':True,'drained_streams':1,'pending_native_replays':0,'retained_rows':0,
                'pending_prefill_streams':0,'pending_output_copies':0} for rank in range(4)]})
        else:
            return handler.reply(409,{'error':'admission-not-armed'})
        return handler.reply(200,reply)

    def completion(self,handler,body):
        with self.lock:self.completions.append({'cache_salt':body['cache_salt'],'max_tokens':body['max_tokens'],
            'prompt_sha256':hashlib.sha256(body['prompt'].encode()).hexdigest()})
        n=self.prompt_tokens
        handler.send_response(200);handler.send_header('Content-Type','text/event-stream');handler.send_header('Connection','close');handler.end_headers()
        handler.close_connection=True
        for event in ({'choices':[{'index':0,'token_ids':[11],'prompt_token_ids':[1]*n,'finish_reason':None}],'usage':{'prompt_tokens':n,'completion_tokens':1}},
                      {'choices':[{'index':0,'token_ids':[],'finish_reason':'length'}],'usage':{'prompt_tokens':n,'completion_tokens':1}}):
            handler.wfile.write(('data: '+json.dumps(event)+'\n\n').encode());handler.wfile.flush()
        handler.wfile.write(b'data: [DONE]\n\n');handler.wfile.flush()


class QueueArgvBoundary(unittest.TestCase):
    """Real main(argv) from the P draft text, refused at the first network send."""

    def refuse(self,calls,timeout):
        def request(connection,method,url,body=None,headers=None,**kwargs):
            command=json.loads(body)
            self.assertEqual((connection.host,connection.port,connection.timeout),('127.0.0.1',1,timeout))
            self.assertEqual((method,url),('POST','/v1/flash_s5_native_overlap'))
            self.assertEqual(headers['Authorization'],'Bearer '+KEY)
            calls.append(command)
            raise ConnectionRefusedError('declared CPU boundary')
        return request

    def test_w1_exact_queue_argv_reaches_prepare_full_and_retains_all_six_repeats(self):
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d);extra,argv=w1_argv(directory,'http://127.0.0.1:1')
            self.assertEqual(extra,{'WORKLOAD':str(HERE/'workloads/q17b-w1.json'),'READ_TIMEOUT_S':'1200'})
            calls=[]
            with patch.dict(os.environ,dict(extra,VLLM_API_KEY=KEY,RUN_ID=RUN_ID)),patch('http.client.HTTPConnection.request',self.refuse(calls,900.0)):
                self.assertEqual(native_w1.main(argv),2)
            record=json.loads((directory/'w1.json').read_text())
            self.assertEqual([(c['action'],c['target_route'],c['run_id'],c['command_seq']) for c in calls],[('prepare','full',RUN_ID,1)])
            self.assertEqual([r['route'] for r in record['repeats']],['full','split','full','full','split','full'])
            self.assertIn('native-w1-refusal:',record['repeats'][0]['reasons'][0])
            self.assertTrue(all(r.get('not_run') for r in record['repeats'][1:]))
            self.assertEqual(record['summary']['verdict'],'INCONCLUSIVE')
            self.assertEqual(record['summary']['totals'],{'required':6,'attempted':6,'valid':0,'invalid':6,'contract':0})
            self.assertTrue(record['summary']['all_declared_repeats_retained'])
            self.assertNotIn(KEY,json.dumps(record))
            PERF_RECEIPTS.append(dict(case='w1-exact-queue-argv',verdict='PASS',first_network_call='prepare-full',network_calls=len(calls),
                session_run_id=RUN_ID,read_timeout_s=1200,repeats_retained=6,network_contact=False))

    def test_both_w2_round_argvs_reach_prepare_full_with_the_real_session_identity(self):
        namespaces=[]
        for round_index in (0,1):
            with self.subTest(round=round_index),tempfile.TemporaryDirectory() as d:
                directory=Path(d);self.assertEqual(write_gates(directory),0)
                (directory/'w1.json').write_text(json.dumps({'schema':2,'summary':{'verdict':'COMPLETE'}}))
                extra,argv=w2_argv(directory,'http://127.0.0.1:1',round_index)
                self.assertEqual(argv[argv.index('--round')+1],str(round_index))
                calls=[]
                with patch.dict(os.environ,round_env(extra)),patch('http.client.HTTPConnection.request',self.refuse(calls,120)):
                    self.assertEqual(rounds.main(argv),2)
                record=json.loads((directory/('overlap-round-%d.json'%round_index)).read_text())
                self.assertEqual([(c['action'],c['target_route'],c['phase'],c['run_id'],c['engine_epoch'],c['command_seq']) for c in calls],
                                 [('prepare','full','full-prefill-solo',RUN_ID,EPOCH,1)])
                meta=record['native35_round']
                self.assertEqual((record['run_id'],meta['session_run_id'],meta['session_client']['run_id']),(RUN_ID,RUN_ID,RUN_ID))
                self.assertEqual(meta['salt_namespace'],RUN_ID+'|native35-p-w2-round-%d'%round_index)
                self.assertEqual(record['first_advance_read_timeout_s'],1200.0)
                self.assertEqual(record['first_error']['error_message'],'native-lane-refused:prefill_full:fresh')
                self.assertEqual(record['grade']['verdict'],'INCONCLUSIVE')
                self.assertTrue(meta['salts']['distinct_prefill_lane_salts']);self.assertEqual(len(set(meta['salts']['overlap_standing_salts'])),8)
                self.assertTrue(meta['salts']['inherited_overlap_index0_coincidence'])
                self.assertEqual(meta['prompts']['fresh']['prompt_tokens_declared'],200000)
                self.assertNotIn(KEY,json.dumps(record))
                namespaces.append(meta)
        a,b=namespaces
        self.assertFalse(set(a['salts']['prefill_salt_by_lane'].values())&set(b['salts']['prefill_salt_by_lane'].values()))
        self.assertFalse(set(a['salts']['overlap_standing_salts'])&set(b['salts']['overlap_standing_salts']))
        self.assertEqual(a['prompts'],b['prompts'])
        PERF_RECEIPTS.append(dict(case='w2-both-round-exact-queue-argv',verdict='PASS',first_network_call='prepare-full',
            session_run_id=RUN_ID,salt_namespaces=[a['salt_namespace'],b['salt_namespace']],package_gate='real reviewed overlay',
            gates_producer='queue PY_GATES',read_timeout_s=1200,network_contact=False))

    def test_refusals_before_any_network_send(self):
        def no_network(*args,**kwargs):raise AssertionError('network reached')
        cases={}
        with tempfile.TemporaryDirectory() as d:
            for name in ('gates','timeout','workload','package'):
                directory=Path(d)/name;directory.mkdir()
                self.assertEqual(write_gates(directory),0)
                (directory/'w1.json').write_text('{}')
                extra,argv=w2_argv(directory,'http://127.0.0.1:1',0);env=round_env(extra)
                if name=='gates':
                    (directory/'gates.json').unlink()
                    self.assertEqual(write_gates(directory,{'startup':'PASS','row1':'PASS','rule4_controls':'PASS','rule4_candidate':'PASS','row4':'PASS','w1_controls':'INCONCLUSIVE'}),2)
                if name=='timeout':env['Q18A_FIRST_ADVANCE_READ_TIMEOUT_S']='480'  # O's inherited value
                if name=='workload':
                    data=json.loads((HERE/'workloads/q18a-w2-overlap.json').read_text());data['repeats']=2
                    (directory/'w2.json').write_text(json.dumps(data));argv[argv.index('--workload')+1]=str(directory/'w2.json')
                if name=='package':
                    ready=json.loads(Path(env['READINESS']).read_text());ready['reviewed']=False
                    (directory/'ready.json').write_text(json.dumps(ready));env['READINESS']=str(directory/'ready.json')
                with patch.dict(os.environ,env,clear=True),patch('http.client.HTTPConnection.request',no_network):
                    self.assertEqual(rounds.main(argv),2)
                record=json.loads((directory/'overlap-round-0.json').read_text())
                cases[name]=record['error']['error_message']
                self.assertEqual(record['grade'],{'verdict':'INCONCLUSIVE','missing':['run-refused'],'failures':[]})
                self.assertEqual(record['native35_round']['round'],0)
                self.assertNotIn('session_client',record['native35_round'])
                with patch.dict(os.environ,env,clear=True),patch('http.client.HTTPConnection.request',no_network):
                    self.assertEqual(rounds.main(argv),2)  # existing artifact is never replaced
                self.assertEqual(json.loads((directory/'overlap-round-0.json').read_text()),record)
            directory=Path(d)/'gates'
            for bad in ('2','-1'):
                with self.assertRaises(SystemExit):
                    rounds.main(['--url','http://127.0.0.1:1','--model','m','--run-id',RUN_ID,'--epoch','7','--round',bad,
                                 '--workload','w','--w1','w','--gates','g','--out',str(directory/'never.json')])
            self.assertFalse((directory/'never.json').exists())
        self.assertEqual(cases,{'gates':'same-queue-qualification-gates-not-pass','timeout':'first-advance-read-timeout-declaration-invalid',
                                'workload':'frozen-native-workload-invalid','package':'reviewed-package-readiness-not-pass'})
        PERF_RECEIPTS.append(dict(case='w2-refusals-before-network',verdict='PASS',refusals=cases,round_index_bounded=True,network_calls=0))


class TwoRoundComposition(unittest.TestCase):
    """Both declared rounds through real main(argv), real clients and the loopback APIServer leaf."""

    def test_rounds_keep_session_identity_differ_only_by_salt_and_are_both_retained(self):
        server=LoopbackAPIServer();self.addCleanup(server.close)
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d);self.assertEqual(write_gates(directory),0)
            w1={'schema':2,'mode':'native-overlap','summary':{'verdict':'INCONCLUSIVE','routes':{}},'repeats':[]}
            (directory/'w1.json').write_text(json.dumps(w1))
            seen=[]
            for round_index in (0,1):
                extra,argv=w2_argv(directory,server.url,round_index)
                before=len(server.commands),len(server.completions)
                with patch.dict(os.environ,round_env(extra)):
                    self.assertEqual(rounds.main(argv),2)
                seen.append((server.commands[before[0]:],server.completions[before[1]:]))
            records=[json.loads((directory/('overlap-round-%d.json'%i)).read_text()) for i in (0,1)]
            for index,(record,(commands,completions)) in enumerate(zip(records,seen)):
                meta=record['native35_round']
                self.assertEqual([c['action'] for c in commands],['prepare','pause','leave_full','prepare','pause','leave_full','prepare','prefill_begin'])
                self.assertEqual({c['run_id'] for c in commands},{RUN_ID})
                self.assertEqual([c['command_seq'] for c in commands],list(range(1,9)))
                self.assertEqual(len({c['client_id'] for c in commands}),1)
                self.assertEqual(meta['session_client'],{'run_id':RUN_ID,'client_id':commands[0]['client_id'],'commands_sent':8,'replies_retained':7})
                lane=meta['salts']['prefill_salt_by_lane']['prefill_full']
                self.assertEqual([c['cache_salt'] for c in completions],[lane,lane])  # fresh -> append share the lane salt
                self.assertEqual([c['prompt_sha256'] for c in completions],[meta['prompts']['fresh']['prompt_sha256'],meta['prompts']['append']['prompt_sha256']])
                full=record['lanes']['prefill_full']
                self.assertTrue(full['fresh']['complete'] and full['append']['complete'])
                self.assertEqual(full['fresh']['request']['prompt_tokens_observed'],4096)
                self.assertGreater(full['fresh']['serving_prefill_tokens_per_s'],0)
                self.assertEqual(record['first_error']['error_message'],'native-lane-refused:prefill_split:fresh')
                self.assertIn('controller-refusal',record['lanes']['prefill_split']['fresh']['error']['error_message'])
                self.assertEqual(record['grade']['verdict'],'INCONCLUSIVE');self.assertEqual(record['native35_round']['round'],index)
            self.assertNotEqual(seen[0][0][0]['client_id'],seen[1][0][0]['client_id'])
            self.assertNotEqual(seen[0][1][0]['cache_salt'],seen[1][1][0]['cache_salt'])
            self.assertEqual([c['prompt_sha256'] for c in seen[0][1]],[c['prompt_sha256'] for c in seen[1][1]])
            # Aggregate over the exact queue invocation; memory grades were not produced.
            text=QUEUE.read_text();start=text.index('python3 "$H/native35_performance_grade.py"')
            tokens=shlex.split(text[start:text.index('> "$Q/native35-performance-grade.log"',start)].replace('\\\n',' '))
            _,argv=substitute(tokens,queue_values(directory,server.url))
            self.assertEqual(aggregate.main(argv[2:]),2)
            graded=json.loads((directory/'native35-performance-grade.json').read_text())
            self.assertEqual(graded['verdict'],'INCONCLUSIVE');self.assertEqual([r['round'] for r in graded['rounds']],[0,1])
            self.assertEqual(graded['headline_by_round'],[None,None]);self.assertFalse(graded['target_is_gate'])
            for view in graded['rounds']:
                self.assertIn('round-%d:declared-run-error-retained'%view['round'],view['missing'])
                self.assertEqual(view['memory_grade']['present'],False)
                self.assertEqual(view['raw']['full_serving_solo']['fresh']['prompt_tokens_observed'],4096)
            self.assertIn('w1-fixed-repeats-incomplete',graded['missing'])
            PERF_RECEIPTS.append(dict(case='two-round-composition',verdict='PASS',rounds=2,session_commands=[len(c) for c,_ in seen],
                completions=[len(s) for _,s in seen],salts_by_round=[s[0]['cache_salt'] for _,s in seen],
                session_run_id=RUN_ID,aggregate_verdict=graded['verdict'],both_rounds_retained=True))


def packet(stack,entries,generation_index=0):
    """The actual driver's queried evidence over ``entries`` (NativeWindowSource shape)."""
    import native_overlap_metrics as metrics
    import test_e3_native_measurement as source
    start,stop=source.window_of(entries)
    raw=stack.driver.overlap_evidence(start,stop);raw['start_ns'],raw['end_ns']=start,stop
    generation=raw['measurement']['route_generations'][generation_index]
    begin={'schema':1,'engine_epoch':stack.driver.engine_epoch,'route':'split','route_generation':generation,'ranks':list(stack.driver.ranks),
           'storage_epochs_by_rank':{r:stack.worker(r).state.decode_owner.storage_epoch for r in stack.driver.ranks}}
    proof=dict(metrics.clock_identity(),proved=True)
    return {'begin':{'engine_epoch':stack.driver.engine_epoch,'snapshot':begin,'measure_start_ns':start},
            'end':{'engine_epoch':stack.driver.engine_epoch,'measure_end_ns':stop,'raw_journal_evidence':raw,'clock':{'begin':proof,'end':proof}},
            'client_bracket':{'start_ns':start-1,'end_ns':stop+1}}


def source_packets():
    """Fresh: a fresh-only real journal. Append: the uid-append request after its own route transition."""
    import test_e3_native_measurement as source
    fresh_stack=source.make_long_stack(tokens=2050,layers=4)
    fresh=packet(fresh_stack,source.run_journal(fresh_stack))
    stack=source.make_long_stack(tokens=2050,layers=4)
    append=source.bridge_suite.Request(uid='uid-append',tokens=2350,computed=2050,generation=1)
    entries=source.run_journal(stack,extra_requests=(append,))
    first=min(i for i,e in enumerate(entries) if e.get('kind')=='chunk_open' and e.get('request')=='uid-append')
    transition=max(i for i,e in enumerate(entries[:first]) if e.get('kind')=='transition')
    appended=packet(stack,entries[transition:])
    assert fresh_stack.driver.engine_epoch==stack.driver.engine_epoch
    return {'fresh':fresh,'append':appended},stack.driver.engine_epoch


def w1_record(full=320.0,split=300.0):
    repeats=[]
    for index,route in enumerate(('full','split','full','full','split','full')):
        rate=full if route=='full' else split
        repeats.append({'repeat':index,'route':route,'valid':True,'reasons':[],'streams':[],
            'metrics':{'aggregate_tokens_per_s':rate,'total_completion_delta':int(rate*20),'wall_s':20.0,'stream_rates_tokens_per_s':[rate/8]*8}})
    return {'schema':2,'mode':'native-overlap','config':{'WORKLOAD_SHA256':'w'*64,'REPEATS':6},'repeats':repeats,
            'summary':{'verdict':'COMPLETE','routes':{'full':{'valid_repeats':[0,2,3,5],'aggregate_tokens_per_s_median':full,'gap_ms_p99':30.0},
                                                  'split':{'valid_repeats':[1,4],'aggregate_tokens_per_s_median':split,'gap_ms_p99':30.0}}}}


def decode_window(start,end,aggregate_rate,streams=8):
    """Declared client-visible SSE counts for the real packet window: exact declared rates, >=1 token per stream."""
    wall=(end-start)/1e9;rate=aggregate_rate/streams;delta=max(1,round(rate*wall))
    rows=[{'stream':i,'start_completion':100,'end_completion':100+delta,'completion_delta':delta,'rate_tokens_per_s':rate,
           'in_window_arrivals':delta,'gaps_ms':[1.0],'gap_ms_p99':1.0,'open_gap_ms':0.5,'error_before_window_end':False} for i in range(streams)]
    return {'start_ns':start,'end_ns':end,'wall_s':round(wall,6),'aggregate_tokens_per_s':aggregate_rate,
            'stream_rates_tokens_per_s':[rate]*streams,'gap_ms_p99':1.0,'gap_ms_max':1.0,'streams':rows,'client_visible':True}


class AggregateGrade(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packets,cls.epoch=source_packets();cls.w1=w1_record()

    def round_record(self,index,decode_share=0.5,mutate=None):
        lanes={'prefill_full':{f:{'lane':'prefill_full','family':f,'complete':True,'native':False,
            'request':{'prompt_tokens_observed':4096,'prompt_tokens_declared':200000 if f=='fresh' else 204096,'prompt_sha256':f[0]*64,'ttft_s':2.0},
            'serving_prefill_tokens_per_s':2048.0 if f=='fresh' else None,'serving_rate_basis':'observed fresh prompt tokens / client TTFT'} for f in aggregate.FAMILIES}}
        baseline=perf.legacy_grade._w1_route(self.w1,'full')['decode_tokens_per_s']
        for lane in ('prefill_split','overlap'):
            lanes[lane]={}
            for family in aggregate.FAMILIES:
                row=dict(copy.deepcopy(self.packets[family]),lane=lane,family=family,native=True,complete=True,
                         request={'prompt_tokens_observed':4096,'prompt_sha256':family[0]*64,'ttft_s':2.0})
                if lane=='overlap':
                    group=perf.native_window(row,family,self.epoch)['group']
                    row['decode_window']=decode_window(group['start_ns'],group['end_ns'],baseline*decode_share)
                lanes[lane][family]=row
        record={'schema':'s5-native-performance/1','run_id':RUN_ID,'engine_epoch':self.epoch,'lanes':lanes,'w1':copy.deepcopy(self.w1),
                'native35_round':{'schema':rounds.SCHEMA,'round':index,'session_run_id':RUN_ID,'salt_namespace':rounds.salt_namespace(RUN_ID,index)}}
        if mutate:mutate(record)
        record['grade']=perf.grade(record)
        return record

    def grade(self,records,memories=(None,None)):
        meta=lambda r:({'path':'round','present':r is not None,'sha256':'0'*64,'readable':True} if r is not None else {'path':'round','present':False})
        return aggregate.grade(self.w1,{'present':True},[(r,meta(r)) for r in records],[(m,meta(m)) for m in memories],{'outcomes':{'row4':'PASS'}})

    def test_actual_packets_shortfall_reports_raw_rates_and_never_fails(self):
        memory={'verdict':'PASS','failures':[],'missing':[],'reserve_mib':512,'min_free_mib_under_pressure':{'0':1576}}
        result=self.grade([self.round_record(0),self.round_record(1,decode_share=0.6)],(memory,memory))
        self.assertEqual(result['verdict'],'PASS',result['missing']);self.assertFalse(result['target_is_gate'])
        self.assertEqual(result['target_status_by_round'],['SHORTFALL','SHORTFALL'])
        for view,share in zip(result['rounds'],(0.5,0.6)):
            raw=view['raw'];self.assertTrue(view['coverage_proved'])
            self.assertAlmostEqual(view['headline']['decode_share'],share,places=9);self.assertEqual(view['headline']['prefill_share'],1.0)
            for lane in ('prefill_split','overlap'):
                for family in aggregate.FAMILIES:
                    native=raw['native_prefill'][lane][family]
                    self.assertTrue(native['complete'],native['source_reasons']);self.assertGreater(native['useful_prefill_tokens'],0)
                    self.assertGreater(native['source_joined_wall_s'],0);self.assertGreater(native['prefill_tokens_per_s'],0)
            self.assertTrue(raw['decode_during_prefill']['fresh']['eight_live_streams_advanced'])
            self.assertEqual(len(raw['decode_during_prefill']['fresh']['completion_delta_by_stream']),8)
            self.assertEqual(raw['w1_full_median_tokens_per_s'],320.0)
            self.assertEqual(view['memory']['verdict'],'PASS')
        self.assertEqual(result['w1']['routes']['full']['median'],320.0);self.assertEqual(result['w1']['routes']['split']['aggregate_tokens_per_s_by_repeat'],[300.0,300.0])
        PERF_RECEIPTS.append(dict(case='aggregate-actual-packets-shortfall',verdict='PASS',aggregate=result['verdict'],
            target_status_by_round=result['target_status_by_round'],source_packets='actual CPU driver overlap_evidence fresh+append',target_is_gate=False))

    def test_incomplete_or_mismatched_round_is_retained_without_headline(self):
        def seven_streams(record):
            window=record['lanes']['overlap']['fresh']['decode_window'];window['streams'].pop();window['stream_rates_tokens_per_s'].pop()
        def window_shift(record):record['lanes']['overlap']['fresh']['decode_window']['start_ns']+=1
        def stalled_stream(record):
            row=record['lanes']['overlap']['fresh']['decode_window']['streams'][3];row['completion_delta']=0
        def foreign_w1(record):record['w1']['summary']['routes']['full']['aggregate_tokens_per_s_median']=999.0
        cases={'eight-stream':(seven_streams,'round-1:eight-stream-decode-share-unproved'),
               'window':(window_shift,'round-1:decode-window-does-not-match-native-headline:fresh'),
               'stalled':(stalled_stream,'round-1:eight-live-streams-unproved'),
               'w1':(foreign_w1,'round-1:embedded-w1-differs-from-queue-w1')}
        observed={}
        for name,(mutate,reason) in cases.items():
            with self.subTest(name):
                result=self.grade([self.round_record(0),self.round_record(1,mutate=mutate)])
                self.assertEqual(result['verdict'],'INCONCLUSIVE');self.assertIn(reason,result['missing'])
                self.assertEqual([v['round'] for v in result['rounds']],[0,1])
                self.assertIsNotNone(result['headline_by_round'][0]);self.assertIsNone(result['headline_by_round'][1])
                self.assertIsNotNone(result['rounds'][1]['raw']['decode_during_prefill']['fresh']['aggregate_tokens_per_s'])
                self.assertFalse(result['filtered_invalid']);self.assertFalse(result['picked']);observed[name]=reason
        result=self.grade([self.round_record(0),None])
        self.assertEqual(result['verdict'],'INCONCLUSIVE');self.assertEqual(result['rounds'][1]['verdict'],'NOT_RUN')
        self.assertIn('round-1:artifact-absent-not-run',result['missing'])
        tampered=self.round_record(1);tampered['grade']=dict(tampered['grade'],verdict='PASS',missing=[])
        tampered['lanes']['overlap']['append']['complete']=False
        result=self.grade([self.round_record(0),tampered])
        self.assertIn('round-1:recheck-differs-from-written-grade',result['missing']);self.assertEqual(result['verdict'],'INCONCLUSIVE')
        PERF_RECEIPTS.append(dict(case='aggregate-refusals-retained',verdict='PASS',refusals=sorted(observed.values()),
            not_run_round_retained=True,written_grade_rechecked=True))

    def test_cli_writes_once_and_exit_codes_follow_the_verdict(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);paths=[]
            for index in (0,1):
                path=d/('overlap-round-%d.json'%index);path.write_text(json.dumps(self.round_record(index)));paths.append(str(path))
            (d/'w1.json').write_text(json.dumps(self.w1))
            argv=['--w1',str(d/'w1.json'),'--round',paths[0],'--round',paths[1],'--memory',str(d/'m0.json'),'--memory',str(d/'m1.json'),'--out',str(d/'grade.json')]
            self.assertEqual(aggregate.main(argv),0)
            first=(d/'grade.json').read_bytes();self.assertEqual(aggregate.main(argv),2);self.assertEqual((d/'grade.json').read_bytes(),first)
            self.assertEqual(aggregate.main(argv[:4]+['--memory','m','--memory','m','--out',str(d/'one.json')]),2)
            self.assertEqual(json.loads((d/'one.json').read_text())['missing'],['aggregate-grade-refused:ValueError'])


if __name__=='__main__':unittest.main()
