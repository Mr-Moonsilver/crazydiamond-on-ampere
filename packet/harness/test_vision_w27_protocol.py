"""Real client argv through COMPLETE synthetic G5/A3/G6 protocols; explicit scorer/GPU leaves."""
import contextlib,shutil,copy,datetime as dt,hashlib,http.client,importlib.util,io,json,os,subprocess,sys,tempfile,threading,time,unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from unittest.mock import patch
import vision_g5_probe as probe,vision_prod_a3 as a3,vision_w27_read as read,vision_service_owner as owner
ROOT=Path(__file__).resolve().parents[1]
Q=Path('<hub-dir>/cargo-target/flash-next/quality-bench-20261116')
SEL=ROOT/'A3-main100-selection-20261117.json'
if not SEL.is_file():SEL=Path('<hub-dir>/agents/flash-next/tmp/final-engine/A3-main100-selection-20261117.json')
CASES=ROOT/'vision-assets/cases.json'
if not CASES.is_file():CASES=Path('<hub-dir>/cargo-target/flash-next/vision-native-mm-20261116/serve-vision-fe-20261116a-build-7/vision-assets/cases.json')
class WholeProtocol(unittest.TestCase):
    def setUp(self):
        self.tmp=Path(tempfile.mkdtemp(prefix='w27-full-protocol-'));self.seen=[]
        self.key=self.tmp/'key.env';self.key.write_text('VLLM_API_KEY='+'a'*64+'\n');self.key.chmod(0o600)
        owner_test=self
        class API(BaseHTTPRequestHandler):
            def log_message(self,*_):pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));owner_test.seen.append(body)
                if self.headers.get('Authorization')!='Bearer '+'a'*64:self.send_response(401);self.end_headers();return
                if self.path=='/tokenize':
                    self.send_response(200);self.end_headers();self.wfile.write(json.dumps(dict(tokens=[1]*1536+[248056]*2609+[2]*2000)).encode());return
                self.send_response(200);self.send_header('Content-Type','text/event-stream' if body.get('stream') else 'application/json');self.end_headers()
                if body.get('stream'):
                    self.wfile.write(b'data: {"id":"cpu-stub","choices":[{"delta":{"content":"CPU_SCORER_LEAF"},"finish_reason":"stop"}],"usage":{"completion_tokens":1}}\n\ndata: [DONE]\n\n')
                else:
                    case=next(c for c in owner_test.cases if body['request_id']=='w27-g5-'+c['id'])
                    answer=json.dumps(case['expected']) if isinstance(case['expected'],list) else case['expected']
                    self.wfile.write(json.dumps(dict(id='chatcmpl-'+body['request_id'],choices=[dict(message=dict(content=answer))],usage=dict(completion_tokens=1),model=body['model'])).encode())
        self.api=ThreadingHTTPServer(('127.0.0.1',0),API);self.api.daemon_threads=True
        threading.Thread(target=self.api.serve_forever,daemon=True).start();self.addCleanup(self.api.server_close);self.addCleanup(self.api.shutdown)
        self.url='http://127.0.0.1:'+str(self.api.server_port)
        from vision_g5_plan import cases
        from build_vision_encoder_packet import red_png
        self.cases=cases(json.loads(CASES.read_text()),1500)
        for image in CASES.parent.glob('*.png'):shutil.copy2(image,self.tmp/image.name)
        (self.tmp/'red1024.png').write_bytes(red_png(1024))
        self.cases_path=self.tmp/'cases.json';self.cases_path.write_text(json.dumps(self.cases))
    def cli(self,args):
        r=subprocess.run([sys.executable,'-B',str(ROOT/'harness/vision_w27_read.py'),*args],capture_output=True,text=True,timeout=10)
        self.assertIn('W27_',r.stdout);return r
    def test_g5_actual_argv_all_seven_cases_final_reader_and_reference_mutant(self):
        cases=self.cases_path
        for phase in ('reference','production'):
            args=['--base-url',self.url,'--key-file',str(self.key),'--cases',str(cases),'--phase',phase,'--out',str(self.tmp/(phase+'.json'))]
            r=subprocess.run([sys.executable,'-B',str(ROOT/'harness/vision_g5_probe.py'),*args],capture_output=True,text=True,timeout=15)
            self.assertEqual(r.returncode,0,r.stderr);self.assertIn('COMPLETE',r.stdout)
        production=json.loads((self.tmp/'production.json').read_text());start=min(c['started_wall_ns'] for c in production['cases']);end=max(c['finished_wall_ns'] for c in production['cases'])
        speech=dict(verdict='RUNNING',physical_peak_mib=1468,speech_overlap_images=[dict(id='real-encoder-load-leaf')],speech=dict(pre_asr=dict(correct=True,elapsed_s=1.14,observed_wall_ns=start-100),post_load_asr=dict(correct=True,elapsed_s=1.16),asr_history=[dict(correct=True,elapsed_s=1.16,observed_wall_ns=end+100)],last_health=dict(observed_wall_ns=end+101)))
        (self.tmp/'speech.json').write_text(json.dumps(speech));(self.tmp/'boundary.json').write_text(json.dumps(dict(verdict='PASS')));args=['g5','--record',str(self.tmp/'production.json'),'--reference',str(self.tmp/'reference.json'),'--speech',str(self.tmp/'speech.json'),'--boundary',str(self.tmp/'boundary.json')]
        result=self.cli(args);self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        reference=json.loads((self.tmp/'reference.json').read_text());reference['cases'][0]['answer']='wrong';(self.tmp/'reference.json').write_text(json.dumps(reference))
        self.assertIn('FAIL',self.cli(args).stdout)
        production['cases'][0]['image_sha256s']=['0'*64];(self.tmp/'production.json').write_text(json.dumps(production));self.assertIn('HOLD',self.cli(args).stdout)
    def test_g5_client_missing_case_mutant_is_incomplete_at_final_reader(self):
        source=(ROOT/'harness/vision_g5_probe.py').read_text();mutant=self.tmp/'mutant.py'
        mutant.write_text(source.replace('for case in cases:', 'for case in cases[:-1]:').replace("sys.path.insert(0,str(Path(__file__).resolve().parent))", "sys.path.insert(0,"+repr(str(ROOT/'harness'))+");sys.path.insert(0,"+repr(str(ROOT/'tools'))+")"))
        cases=self.cases_path
        argv=['--base-url',self.url,'--key-file',str(self.key),'--cases',str(cases),'--phase','production','--out',str(self.tmp/'mutant.json')]
        result=subprocess.run([sys.executable,'-I','-B',str(mutant),*argv],capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)
        value=json.loads((self.tmp/'mutant.json').read_text());self.assertEqual(len(value['cases']),6)
        self.assertEqual(read.g5_read(value,value,{})['verdict'],'HOLD')

    def test_g5_boundary_actual_argv_complete_and_missing_peer_mutant(self):
        path=ROOT/'tools/vision_g5_boundary_read.py';record=self.tmp/'production.json';log=self.tmp/'engine.log';out=self.tmp/'boundary.json'
        record.write_text(json.dumps(dict(cases=[dict(id='four-native-boundary',kind='native-boundary',request_id='request',response_id='chatcmpl-request')])))
        rows=[dict(schema='native-mm-boundary/1',worker_rank=r,stage=r//2,request_sha256=hashlib.sha256(b'chatcmpl-request').hexdigest(),free_bytes=100*2**20,rows=2048 if r<2 else 0,native_mm=dict(tower_registered_cuda_bytes=0,cache_storage_bytes=40*2**20)) for r in range(4)]
        def run():
            log.write_text(''.join('flash-s5 native-mm-boundary '+json.dumps(x)+'\n' for x in rows))
            return subprocess.run([sys.executable,'-I','-B',str(path),'--engine-log',str(log),'--record',str(record),'--out',str(out)],capture_output=True,text=True)
        self.assertEqual(run().returncode,0);self.assertEqual(json.loads(out.read_text())['verdict'],'PASS')
        rows.pop();self.assertEqual(run().returncode,2);self.assertEqual(json.loads(out.read_text())['verdict'],'HOLD')
        rows.append(dict(rows[-1],worker_rank=3,free_bytes=39*2**20));self.assertEqual(run().returncode,2);self.assertEqual(json.loads(out.read_text())['verdict'],'FAIL')

    def test_g6_complete_actual_reader_argv_missing_caller_and_probe_only_negative(self):
        calls={name:dict(status='PASS',application_path=True,wall_s=1,output_tokens=5,fixture_sha256='c'*64,receipt_path='/private/owner/call.json',caller_timeout_contract='owner own config; no timeout observed',timed_out=False) for name in read.CALLERS}
        pre=dict(phase='27B',calls=copy.deepcopy(calls));post=dict(phase='Flash-Next',calls=copy.deepcopy(calls))
        def run():
            (self.tmp/'pre.json').write_text(json.dumps(pre));(self.tmp/'post.json').write_text(json.dumps(post));return self.cli(['g6','--record',str(self.tmp/'post.json'),'--reference',str(self.tmp/'pre.json')])
        self.assertEqual(run().returncode,0)
        self.assertEqual(read.g6_read(dict(pre,calls={k:v for k,v in calls.items() if k in read.REQUIRED_CALLERS}),dict(post,calls={k:v for k,v in calls.items() if k in read.REQUIRED_CALLERS}))['verdict'],'PASS')
        argv=[sys.executable,'-I','-B',str(ROOT/'harness/vision_g6_capture.py'),'--phase','27B','--owner-record',str(self.tmp/'pre.json'),'--out',str(self.tmp/'capture.json')]
        captured=subprocess.run(argv,capture_output=True,text=True,timeout=10);self.assertEqual(captured.returncode,0,captured.stderr);self.assertIn('W27_G6_CAPTURE COMPLETE',captured.stdout)
        self.assertEqual(json.loads((self.tmp/'capture.json').read_text())['calls'],pre['calls'])
        # The collector refuses duplicate owners rather than silently choosing a passing receipt.
        argv[-1]=str(self.tmp/'duplicate.json');argv+=['--owner-record',str(self.tmp/'pre.json')]
        duplicate=subprocess.run(argv,capture_output=True,text=True,timeout=10);self.assertEqual(duplicate.returncode,2);self.assertIn('duplicate',duplicate.stderr)
        post['calls']['<caller-a>.text']['application_path']=False;self.assertIn('HOLD',run().stdout)
        post['calls']['<caller-a>.text']['application_path']=True;post['calls']['<caller-d>']['timed_out']=True;self.assertIn('FAIL',run().stdout)
        del post['calls']['<caller-d>'];self.assertIn('HOLD',run().stdout)
    def test_a3_frozen_runner_actual100_requests_and_scorer_leaf_to_final_verdict(self):
        out=self.tmp/'a3';deadline=(dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=3)).isoformat()
        argv=['--quality-packet',str(Q),'--selection',str(SEL),'--base-url',self.url,'--key-file',str(self.key),'--out',str(out),'--deadline',deadline]
        expected=json.loads(SEL.read_text())['ids'];baseline={r['id']:r for r in map(json.loads,(Q/'run/B/scores/main.jsonl').read_text().splitlines())}
        real_run=subprocess.run
        def score_leaf(command,**kw):
            if command[0]=='sha256sum':return real_run(command,**kw)
            self.assertEqual(command,[sys.executable,'-I',str(Q/'code/score.py'),'--packet',str(Q),'--out',str(out)])
            (out/'scores').mkdir();(out/'scores/main.jsonl').write_text(''.join(json.dumps(baseline[i])+'\n' for i in expected))
            return subprocess.CompletedProcess(command,0)
        # Load the real frozen runner (httpx) from Q's CPU environment in the self-run interpreter.
        with patch.object(a3.subprocess,'run',side_effect=score_leaf),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(a3.main(argv),0)
        self.assertEqual(len(self.seen),100);self.assertTrue(all(b['chat_template_kwargs']==dict(enable_thinking=False) for b in self.seen))
        args=['a3','--record',str(out),'--quality-packet',str(Q),'--selection',str(SEL)]
        result=self.cli(args);self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        values=[json.loads(x) for x in (out/'scores/main.jsonl').read_text().splitlines()];values=values[:-1];(out/'scores/main.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in values));self.assertIn('HOLD',self.cli(args).stdout)
    def test_service_owner_exact_argv_owned_stop_and_admission_refusals(self):
        packet=self.tmp/'packet';packet.mkdir();clear=self.tmp/'clearance.json';clear.write_text('{}')
        argv=['--packet',str(packet),'--run-id','cpu-owner','--clearance',str(clear),'--state',str(self.tmp/'state/admissions.json'),'--key-file',str(self.key)]
        commands=[]
        class Child:
            returncode=0
            def __init__(self,command):commands.append(command)
            def wait(self,timeout=None):return 0
            def poll(self):return 0
        with patch.object(owner,'validate_clearance',return_value=dict(service_cleared=True)),patch.object(owner.subprocess,'run',return_value=subprocess.CompletedProcess([],0)),patch.object(owner.subprocess,'Popen',side_effect=Child),patch.object(owner.signal,'signal'),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(owner.main(argv),0)
        self.assertIn('--service',commands[0]);self.assertIn('--model',commands[0]);self.assertEqual(commands[0][-1],'--service')
        refused_argv=argv[:];refused_argv[refused_argv.index('--state')+1]=str(self.tmp/'fresh-state/admissions.json')
        with patch.object(owner,'validate_clearance',return_value=dict(service_cleared=False)),patch.object(owner.subprocess,'run',return_value=subprocess.CompletedProcess([],0)),patch.object(owner.subprocess,'Popen',side_effect=Child),patch.object(owner.signal,'signal'),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(owner.Refused):owner.main(refused_argv)

        with self.assertRaises(Exception):owner.admission(dict(attempts=[time.time()]*3),time.time())
        with self.assertRaises(Exception):owner.admission(dict(yielded_at=time.time()-599),time.time())
        self.assertEqual(owner.admission(dict(attempts=[]),time.time()),[])
if __name__=='__main__':unittest.main()
