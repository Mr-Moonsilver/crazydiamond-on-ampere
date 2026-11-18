#!/usr/bin/env python3
"""Run generated unit's gateway argv through HTTP/SSE to shutdown, without Docker/GPU/service actions."""
import concurrent.futures as futures
import hashlib,http.client,json,os,shlex,socket,subprocess,sys,tempfile,threading,time,traceback
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

def run(packet,parent,out):
 out.mkdir(parents=True,exist_ok=False)
 line=next(x for x in (packet/'flashnext-vision-gateway.service').read_text().splitlines() if x.startswith('ExecStart='))
 command=shlex.split(line.split('=',1)[1]);defaults=dict(x.split('=',1) for x in (packet/'gateway-defaults.env').read_text().splitlines())
 command=[defaults.get(x[2:-1],x) if x.startswith('${') else x for x in command]
 assert command[:2]==['/usr/bin/docker','run'] and '--gpus' not in command
 assert 'NVIDIA_VISIBLE_DEVICES=void' in command and command[command.index('--runtime')+1]=='runc'
 image='sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b'
 assert image in command
 argv=command[command.index(image)+1:]
 assert argv[argv.index('--max-inflight')+1]=='32' and argv[argv.index('--queue-timeout-s')+1]=='3000'
 assert argv[argv.index('--max-images')+1]=='4'
 (out/'generated-docker-argv.json').write_text(json.dumps(command,indent=2)+'\n')
 release=threading.Event();condition=threading.Condition();arrivals=[]
 class Backend(BaseHTTPRequestHandler):
  def log_message(self,*_):pass
  def do_GET(self):self.send_response(200);self.end_headers();self.wfile.write(b'{"data":[]}')
  def do_POST(self):
   value=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
   with condition:arrivals.append((self.path,value));condition.notify_all()
   release.wait(15);self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers();self.wfile.write(b'data: [DONE]\n\n')
 backend=ThreadingHTTPServer(('127.0.0.1',0),Backend);backend.daemon_threads=True;threading.Thread(target=backend.serve_forever,daemon=True).start()
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 for name,var,char in (('key.env','VLLM_API_KEY','a'),('encoder.env','VISION_ENCODER_API_KEY','b')):
  p=out/name;p.write_text(var+'='+char*64+'\n');p.chmod(0o600)
 ready=out/'readiness.json';parent_sha=argv[argv.index('--packet-sha256')+1];ready.write_text(json.dumps(dict(ready=True,packet_sha256=parent_sha)))
 # Only network bindings and fixture paths are translated; every generated option is retained.
 argv[1]=str(packet/'vision_public_gateway.py')
 for flag,value in (('--bind','127.0.0.1'),('--port',str(port)),('--backend','http://127.0.0.1:'+str(backend.server_port)),('--key-file',str(out/'key.env')),('--encoder-key-file',str(out/'encoder.env')),('--ready-file',str(ready))):argv[argv.index(flag)+1]=value
 actual=[sys.executable]+argv
 (out/'actual-argv.json').write_text(json.dumps(actual,indent=2)+'\n')
 env=dict(os.environ,PYTHONPATH=str(parent/'harness'),PYTHONDONTWRITEBYTECODE='1')
 def call(path='/v1/chat/completions',post=True):
  c=http.client.HTTPConnection('127.0.0.1',port,timeout=20)
  body=json.dumps(dict(model='qwen38-w8a8-tp4',messages=[dict(role='user',content='hello')],stream=True)) if post else None
  c.request('POST' if post else 'GET',path,body,{'Content-Type':'application/json','Authorization':'Bearer '+'a'*64} if post else {})
  r=c.getresponse();code=r.status;raw=r.read();c.close();return code,raw
 stdout=(out/'stdout.log').open('x');stderr=(out/'stderr.log').open('x');proc=subprocess.Popen(actual,stdout=stdout,stderr=stderr,env=env)
 try:
  until=time.monotonic()+10
  while 'VISION_GATEWAY_READY' not in (out/'stdout.log').read_text():
   if proc.poll()is not None or time.monotonic()>until:raise AssertionError((out/'stderr.log').read_text())
   time.sleep(.02)
  with futures.ThreadPoolExecutor(max_workers=32) as pool:
   jobs=[pool.submit(call,'/v1/messages' if i%2 else '/v1/chat/completions') for i in range(32)]
   with condition:assert condition.wait_for(lambda:len(arrivals)==32,timeout=8),len(arrivals)
   assert not any(job.done() for job in jobs)
   for _ in range(3):
    for path in ('/health','/metrics','/v1/models'):assert call(path,False)[0]==200
   release.set();assert all(job.result()==(200,b'data: [DONE]\n\n') for job in jobs)
  assert len(arrivals)==32 and all('chat_template_kwargs' not in value for _,value in arrivals)
 finally:
  release.set();proc.terminate();proc.wait(timeout=10);stdout.close();stderr.close();backend.shutdown();backend.server_close()
 assert proc.returncode==0 and 'VISION_GATEWAY_STOPPED' in (out/'stdout.log').read_text()
 bad=actual.copy();bad[bad.index('--encoder')+1]='http://wrong-endpoint:9041'
 mutant=subprocess.run(bad,capture_output=True,text=True,env=env,timeout=10)
 (out/'mutant.stderr.log').write_text(mutant.stderr)
 assert mutant.returncode==2 and 'ValueError' in mutant.stderr and 'vision_public_gateway.py:' in mutant.stderr
 result=dict(verdict='PASS',protocol='generated unit argv -> READY -> 32 simultaneous POSTs (chat/messages) -> nine keyless metadata probes -> SSE complete -> SIGTERM -> STOPPED',model_arrivals=32,metadata_probes=9,mutant='wrong encoder endpoint refuses with type/message/file:line',stub='CPU HTTP backend; no model or GPU')
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('GATEWAY_QUEUE_SELFRUN PASS 32 callers +9 probes +SSE +owned process stop')
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 try:run(a.packet,a.parent,a.out)
 except Exception as exc:
  f=traceback.extract_tb(exc.__traceback__)[-1];print(f'PACKET_CLIENT_ERROR {type(exc).__name__}: {exc} at {f.filename}:{f.lineno}',file=sys.stderr);raise SystemExit(2)
