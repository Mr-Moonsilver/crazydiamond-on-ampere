"""Frozen actual A3-L shell/client argv to final verdict; HTTP and scorer are explicit CPU leaves."""
import copy,datetime as dt,hashlib,importlib.util,json,os,shutil,subprocess,sys,tempfile,threading,time,unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import vision_prod_a3l as load,vision_w28_g6_read as g6
P=Path(os.environ.get('W28_PAIR','<hub-dir>/cargo-target/flash-next/vision-headroom-w28-20261118/build-6'))/'production'
Q=Path('<hub-dir>/cargo-target/flash-next/quality-bench-20261116')
BG=Path('<hub-dir>/cargo-target/flash-next/quality-bench-20261117')

class Protocol(unittest.TestCase):
 def setUp(self):
  self.root=Path(tempfile.mkdtemp(prefix='a3l-full-'));self.solo=self.root/'solo';self.solo.mkdir();self.seen=[]
  self.key=self.root/'key.env';self.key.write_text('VLLM_API_KEY='+'a'*64+'\n');self.key.chmod(0o600)
  runner=load.module(Q/'code/runner.py','a3l_test_w26');self.ids=json.loads((P/'A3-main100-selection-20261117.json').read_text())['ids'];items=runner.load_items(str(Q))
  model=type('Body',(),dict(model='qwen38-w8a8-tp4'))();bodies={i:runner.Client.body(model,items[i]) for i in self.ids}
  self.bodyshas={i:hashlib.sha256(json.dumps(b,sort_keys=True).encode()).hexdigest() for i,b in bodies.items()}
  for name in ('results','scores'):
   (self.solo/name).mkdir();(self.solo/name/'main.jsonl').write_text(''.join(json.dumps(dict(id=i,score=1.0,phase='main',suite=items[i]['suite']))+'\n' for i in self.ids))
  (self.solo/'a3-collection.json').write_text(json.dumps(dict(selection_sha256=load.SELECTION_SHA,quality_sha256=load.QUALITY_SHA,stats=dict(done=100,skipped_cap=0,errors=0),request_body_shas=self.bodyshas)))
  owner=self
  class API(BaseHTTPRequestHandler):
   def log_message(self,*_):pass
   def do_GET(self):
    self.send_response(200);self.end_headers();self.wfile.write(b'vllm:num_requests_running{model_name="cpu"} 8\n')
   def do_POST(self):
    body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
    if self.headers.get('Authorization')!='Bearer '+'a'*64:self.send_response(401);self.end_headers();return
    if self.path=='/v1/flash_s5_native_overlap':
     self.send_response(200);self.end_headers();self.wfile.write(json.dumps(dict(schema=1,action='status',native_ready=True,run_id='cpu-native',engine_epoch=7)).encode());return
    owner.seen.append(body);isbg=sum(len(str(m.get('content',''))) for m in body['messages'])>100000
    time.sleep(.06 if isbg else .02)
    self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
    usage=dict(prompt_tokens=150000 if isbg else 100,completion_tokens=1)
    try:self.wfile.write(('data: '+json.dumps(dict(id='chatcmpl-cpu',choices=[dict(delta=dict(content='CPU_SCORER_LEAF'),finish_reason='stop')],usage=usage))+'\n\ndata: [DONE]\n\n').encode())
    except (BrokenPipeError,ConnectionResetError):pass  # owned background cancellation at end
  self.api=ThreadingHTTPServer(('127.0.0.1',0),API);self.api.daemon_threads=True;threading.Thread(target=self.api.serve_forever,daemon=True).start();self.addCleanup(self.api.server_close);self.addCleanup(self.api.shutdown)
  self.env=dict(os.environ,A3L_BASE_URL='http://127.0.0.1:'+str(self.api.server_port),A3L_KEY_FILE=str(self.key),PYTHONDONTWRITEBYTECODE='1')
  # The actual frozen scorer argv is checked; synthetic scores are the only subprocess leaf.
  shim=self.root/'shim';shim.mkdir();(shim/'sitecustomize.py').write_text('''import subprocess,json,os\nfrom pathlib import Path\noriginal=subprocess.run\ndef run(argv,**kw):\n if isinstance(argv,list) and len(argv)>2 and argv[2].endswith('/code/score.py'):\n  expected=[os.environ['A3L_TEST_PYTHON'],'-I',os.environ['A3L_TEST_QUALITY']+'/code/score.py','--packet',os.environ['A3L_TEST_QUALITY'],'--out',argv[-1]]\n  assert argv==expected,argv\n  out=Path(argv[-1]);(out/'scores').mkdir(exist_ok=True);(out/'scores/main.jsonl').write_bytes((Path(os.environ['A3L_TEST_SOLO'])/'scores/main.jsonl').read_bytes())\n  return subprocess.CompletedProcess(argv,0)\n return original(argv,**kw)\nsubprocess.run=run\n''')
  # -I intentionally disables sitecustomize; the real-argv launcher uses a Python IO shim just for the scorer.
  self.shim=shim
 def invoke(self,out,packet=P):
  argv=['bash',str(packet/'a3l.sh'),str(self.solo),str(out),(dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=90)).isoformat()]
  # Replace only the interpreter executable with a shim that retains the exact argv, and adds scorer IO.
  launcher=self.root/('python-'+out.name);launcher.mkdir()
  py=launcher/'python';py.write_text('#!/bin/bash\nset -eu\n[[ $1 == -I && $2 == -B ]]\nshift\nexec '+str(sys.executable)+' "$@"\n');py.chmod(0o700)
  env=dict(self.env,A3L_PYTHON=str(py),PYTHONPATH=str(self.shim),A3L_TEST_PYTHON=str(sys.executable),A3L_TEST_QUALITY=str(Q),A3L_TEST_SOLO=str(self.solo))
  (self.root/(out.name+'.argv.json')).write_text(json.dumps(argv))
  result=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=120);(self.root/(out.name+'.owner.stdout')).write_text(result.stdout);(self.root/(out.name+'.owner.stderr')).write_text(result.stderr)
  return result
 def test_actual_shell_whole_protocol_final_reader_and_mutants(self):
  out=self.root/'loaded';r=self.invoke(out);self.assertEqual(r.returncode,0,r.stdout+r.stderr+(out.with_suffix('.stderr.log')).read_text());self.assertIn('W28_A3L PASS',r.stdout)
  args=[sys.executable,'-I','-B',str(P/'harness/vision_prod_a3l.py'),'read','--solo',str(self.solo),'--out',str(out),'--quality-packet',str(Q),'--selection',str(P/'A3-main100-selection-20261117.json')]
  rr=subprocess.run(args,capture_output=True,text=True,timeout=15);self.assertEqual(rr.returncode,0,rr.stderr);self.assertIn('W28_A3L PASS',rr.stdout)
  self.assertEqual({hashlib.sha256(json.dumps(b,sort_keys=True).encode()).hexdigest() for b in self.seen if sum(len(str(m.get('content',''))) for m in b['messages'])<100000},set(self.bodyshas.values()))
  meta=json.loads((out/'a3l-collection.json').read_text());self.assertEqual(meta['warm']['streams_warm'],5)
  original=(out/'scores/main.jsonl').read_text();rows=[json.loads(x) for x in original.splitlines()];rows[0]['score']=0
  for row in rows[:5]:row['score']=0
  (out/'scores/main.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));self.assertEqual(load.read(out,self.solo,Q,P/'A3-main100-selection-20261117.json')['verdict'],'PASS')
  rows[5]['score']=0
  (out/'scores/main.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));self.assertEqual(load.read(out,self.solo,Q,P/'A3-main100-selection-20261117.json')['verdict'],'FAIL')
  (out/'scores/main.jsonl').write_text(original);meta['request_body_shas'][self.ids[0]]='0'*64;(out/'a3l-collection.json').write_text(json.dumps(meta));self.assertEqual(load.read(out,self.solo,Q,P/'A3-main100-selection-20261117.json')['verdict'],'HOLD')
  meta['request_body_shas']=self.bodyshas;(out/'a3l-collection.json').write_text(json.dumps(meta))
  (out/'attempt-errors.jsonl').write_text(json.dumps(dict(error='http_500'))+'\n');self.assertEqual(load.read(out,self.solo,Q,P/'A3-main100-selection-20261117.json')['verdict'],'FAIL')
  print('A3L_FULL_PROTOCOL_PASS '+str(self.root))
 def test_actual_concurrency_mutant_is_not_PASS(self):
  packet=self.root/'mut-packet';shutil.copytree(P,packet)
  source=packet/'harness/vision_prod_a3l.py';text=source.read_text();self.assertIn("'main',todo,3,deadline",text);source.write_text(text.replace("'main',todo,3,deadline","'main',todo,8,deadline"))
  r=self.invoke(self.root/'wrong-conc',packet);self.assertEqual(r.returncode,2,r.stdout+r.stderr);self.assertIn('actual-probe-concurrency-not-three',r.stdout)

 def test_actual_missing_background_mutant_is_not_PASS(self):
  packet=self.root/'mut-background';shutil.copytree(P,packet)
  source=packet/'harness/vision_prod_a3l.py';source.write_text(source.read_text().replace("driver.BG_N['n']=5","driver.BG_N['n']=4"))
  r=self.invoke(self.root/'wrong-bg',packet);self.assertEqual(r.returncode,2,r.stdout+r.stderr);self.assertIn('W28_A3L HOLD',r.stdout)

class Readers(unittest.TestCase):
 def calls(self):return {n:dict(status='PASS',application_path=True,wall_s=1,output_tokens=5,fixture_sha256='c'*64,receipt_path='/private/owner.json',caller_timeout_contract='owner timeout',timed_out=False) for n in g6.REQUIRED_CALLERS}
 def test_g6_actual_cli_complete_missing_three_and_timeout_negative(self):
  root=Path(tempfile.mkdtemp(prefix='w28-g6-'));calls=self.calls();pre=dict(phase='27B',calls={n:c for n,c in calls.items() if n not in g6.UNPAIRED});post=dict(phase='Flash-Next',calls=calls)
  for name,value in [('pre',pre),('post',post)]:(root/(name+'.json')).write_text(json.dumps(value))
  args=[sys.executable,'-I','-B',str(P/'harness/vision_w28_g6_read.py'),'--reference',str(root/'pre.json'),'--record',str(root/'post.json'),'--out',str(root/'read.json')]
  r=subprocess.run(args,capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr);self.assertIn('W28_G6 PASS',r.stdout);self.assertEqual(sum(c['reference']=='27B not captured' for c in json.loads((root/'read.json').read_text())['calls']),3)
  post['calls']['<caller-d>']['timed_out']=True;(root/'post.json').write_text(json.dumps(post));r=subprocess.run(args,capture_output=True,text=True);self.assertEqual(r.returncode,1);self.assertIn('W28_G6 FAIL',r.stdout)
  post['calls']['<caller-d>']['timed_out']=False;del pre['calls']['<caller-b>.dev1'];self.assertEqual(g6.read(pre,post)['verdict'],'HOLD')
  pre['calls']['<caller-b>.dev1']=self.calls()['<caller-b>.dev1'];post['calls']['<caller-a>.text']['application_path']=False;self.assertEqual(g6.read(pre,post)['verdict'],'HOLD')
  (root/'post.json').write_text('{broken');r=subprocess.run(args,capture_output=True,text=True);self.assertEqual(r.returncode,2);self.assertIn('PACKET_CLIENT_ERROR',r.stderr);self.assertIn('W28_G6 HOLD',r.stdout)
 def test_g5r_exact_copy_and_wrong_reference_report_only(self):
  original=Path('<hub-dir>/cargo-target/flash-next/vision-production-w27-20261117/g5-reader-ruling-20261117/vision_w27_read_g5r.py')
  self.assertEqual(hashlib.sha256((P/'harness/vision_w27_read_g5r.py').read_bytes()).hexdigest(),'a612d40af065f90a755a3afe44013022789e9a8be97be87f5400e279ab7a167c');self.assertEqual((P/'harness/vision_w27_read_g5r.py').read_bytes(),original.read_bytes())
  # Exercise the actual approved reader's COMPLETE CLI fixture, replacing only which frozen reader is invoked.
  import test_vision_w27_protocol as existing
  class Approved(existing.WholeProtocol):
   def cli(self,args):
    return subprocess.run([sys.executable,'-B',str(P/'harness/vision_w27_read_g5r.py'),*args],capture_output=True,text=True,timeout=10)
  proof=Approved();proof.setUp()
  try:
   args=['--base-url',proof.url,'--key-file',str(proof.key),'--cases',str(proof.cases_path)]
   for phase in ('reference','production'):
    r=subprocess.run([sys.executable,'-B',str(P/'harness/vision_g5_probe.py'),*args,'--phase',phase,'--out',str(proof.tmp/(phase+'.json'))],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
   values=json.loads((proof.tmp/'production.json').read_text());start=min(c['started_wall_ns'] for c in values['cases']);end=max(c['finished_wall_ns'] for c in values['cases']);ref=json.loads((proof.tmp/'reference.json').read_text());ref['cases'][0]['answer']='wrong';(proof.tmp/'reference.json').write_text(json.dumps(ref))
   speech=dict(verdict='RUNNING',physical_peak_mib=1408,speech_overlap_images=[{}],speech=dict(pre_asr=dict(correct=True,elapsed_s=1,observed_wall_ns=start-1),post_load_asr=dict(correct=True,elapsed_s=1),asr_history=[dict(correct=True,elapsed_s=1,observed_wall_ns=end+1)],last_health=dict(observed_wall_ns=end+1)))
   (proof.tmp/'speech.json').write_text(json.dumps(speech));(proof.tmp/'boundary.json').write_text(json.dumps(dict(verdict='PASS')))
   cli=['g5','--record',str(proof.tmp/'production.json'),'--reference',str(proof.tmp/'reference.json'),'--speech',str(proof.tmp/'speech.json'),'--boundary',str(proof.tmp/'boundary.json')]
   r=proof.cli(cli);self.assertEqual(r.returncode,0,r.stdout+r.stderr);self.assertIn('W27_G5 PASS',r.stdout)
   values['cases'][0]['answer']='wrong';(proof.tmp/'production.json').write_text(json.dumps(values));self.assertIn('FAIL',proof.cli(cli).stdout)
  finally:
   proof.doCleanups()

if __name__=='__main__':unittest.main()
