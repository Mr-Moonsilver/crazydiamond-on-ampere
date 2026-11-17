#!/usr/bin/env python3
"""CPU container only: exact frozen gateway shell argv through sockets to owned stop. Never real Docker/GPU."""
import argparse,base64,hashlib,http.client,json,os,subprocess,sys,time,threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from packet_client_contract import report_exception

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',type=Path,required=True);p.add_argument('--proof',type=Path,required=True);a=p.parse_args(argv)
 if os.getuid()!=0 or os.environ.get('NVIDIA_VISIBLE_DEVICES')!='void' or Path('/dev/nvidia0').exists():raise ValueError('isolated CPU root container required')
 a.proof.mkdir(parents=True,exist_ok=False);os.umask(0o077)
 kdir=Path('/root/.flashnext-serve-vision-prod-w27-20261117a');kdir.mkdir(parents=True,exist_ok=False)
 for name,var,c in (('vllm-api.env','VLLM_API_KEY','a'),('vision-encoder.env','VISION_ENCODER_API_KEY','b')):(kdir/name).write_text(var+'='+c*64+'\n')
 packet_sha=hashlib.sha256((a.packet/'PACKET-SHA256SUMS').read_bytes()).hexdigest();state=Path('/data/flash-next-control/vision-prod-w27/readiness.json');state.parent.mkdir(parents=True)
 seen=[]
 class Model(BaseHTTPRequestHandler):
  def log_message(self,*_):pass
  def do_GET(self):
   self.send_response(200);self.end_headers();self.wfile.write(b'{"data":[{"id":"qwen38-w8a8-tp4"}]}')
  def do_POST(self):
   body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));seen.append((self.path,body))
   self.send_response(200);self.end_headers()
   if self.path=='/v1/flash_s5_native_overlap':self.wfile.write(json.dumps(dict(schema=1,action='status',run_id='cpu-existing',engine_epoch=1,native_ready=True,auto_arm=dict(manual_owned=False,selection_seq=1))).encode())
   elif body.get('stream'):self.wfile.write(b'event: message_start\ndata: {"type":"message_start"}\n\ndata: [DONE]\n\n')
   else:self.wfile.write(b'{"choices":[{"message":{"content":"red"}}],"usage":{"completion_tokens":1}}')
 model=ThreadingHTTPServer(('127.0.0.1',8001),Model);model.daemon_threads=True;threading.Thread(target=model.serve_forever,daemon=True).start()
 child=None
 try:
  # Actual generated ExecStartPre args, existing-lane path: no pause or reselection.
  ready=[sys.executable,'-I','-B',str(a.packet/'harness/vision_production_ready.py'),'--packet',str(a.packet),'--state-file',str(state)]
  result=subprocess.run(ready,capture_output=True,text=True,timeout=10)
  (a.proof/'ready.stdout.log').write_text(result.stdout);(a.proof/'ready.stderr.log').write_text(result.stderr)
  if result.returncode or 'VISION_PRODUCTION_READY PASS' not in result.stdout:raise ValueError('generated readiness argv did not finish')
  driver=a.proof/'driver.py';driver.write_text('import sys\nsys.path.insert(0,'+repr(str(a.packet/'harness'))+')\nimport vision_public_gateway as g\nfrom test_vision_encoder_protocol import FakeBackend\ng.Gateway.encode=lambda self,data:FakeBackend().encode(data)\nraise SystemExit(g.main(sys.argv[1:]))\n')
  bindir=a.proof/'io-bin';bindir.mkdir();docker=bindir/'docker'
  docker.write_text('#!'+sys.executable+'\nimport json,os,sys\nfrom pathlib import Path\na=sys.argv[1:]\nif a[0]=="inspect":raise SystemExit(1)\nassert a[0]=="run" and "--gpus" not in a and a[a.index("--runtime")+1]=="runc"\nimage="sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b"\nPath('+repr(str(a.proof/'generated-docker-argv.json'))+').write_text(json.dumps(a))\nclient=a[a.index(image)+1:]\nassert client[:2]==["-B",'+repr(str(a.packet/'harness/vision_public_gateway.py'))+']\nos.execv(sys.executable,[sys.executable,"-I","-B",'+repr(str(driver))+']+client[2:])\n');docker.chmod(0o755)
  stdout=(a.proof/'gateway.stdout.log').open('x');stderr=(a.proof/'gateway.stderr.log').open('x')
  child=subprocess.Popen(['bash',str(a.packet/'gateway-start.sh')],env=dict(os.environ,PATH=str(bindir)+':'+os.environ['PATH']),stdout=stdout,stderr=stderr)
  end=time.monotonic()+15
  while time.monotonic()<end:
   if child.poll() is not None:raise ValueError('generated gateway shell exited early')
   if 'VISION_GATEWAY_READY' in (a.proof/'gateway.stdout.log').read_text():break
   time.sleep(.05)
  def call(path,body=None,auth=True):
   c=http.client.HTTPConnection('127.0.0.1',8000,timeout=5);headers={'Content-Type':'application/json'}
   if auth:headers['Authorization']='Bearer '+'a'*64
   c.request('GET' if body is None else 'POST',path,None if body is None else json.dumps(body),headers);r=c.getresponse();raw=r.read();code=r.status;c.close();return code,raw
  assert call('/v1/models',auth=False)[0]==200
  text=dict(model='qwen38-w8a8-tp4',messages=[dict(role='user',content='hello')]);assert call('/v1/chat/completions',text,False)[0]==401
  assert call('/v1/chat/completions',text)[0]==200
  image=dict(model='qwen38-w8a8-tp4',messages=[dict(role='user',content=[dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(b'\x89PNG\r\n\x1a\nCPU_IMAGE_LEAF').decode()))])])
  assert call('/v1/chat/completions',image)[0]==200;assert seen[-1][1]['messages'][0]['content'][0]['type']=='image_embeds'
  anth=dict(model='qwen38-w8a8-tp4',messages=[dict(role='user',content='text')],max_tokens=16,stream=True);code,raw=call('/v1/messages',anth);assert code==200 and b'message_start' in raw
  state.write_text(json.dumps(dict(ready=False,packet_sha256=packet_sha)));assert call('/v1/chat/completions',text)[0]==503;assert call('/health',auth=False)[0]==503
 finally:
  if child is not None:
   child.terminate();child.wait(timeout=10);stdout.close();stderr.close()
  model.shutdown();model.server_close()
 if child.returncode or 'VISION_GATEWAY_STOPPED' not in (a.proof/'gateway.stdout.log').read_text():raise ValueError('owned gateway stop not proven')
 (a.proof/'result.json').write_text(json.dumps(dict(verdict='PASS',checks=7,readiness_argv=ready,wrapper_argv=['bash',str(a.packet/'gateway-start.sh')],scope='exact frozen shell/argv, real HTTP/SSE, synthetic encoder/model/driver; no GPU'),indent=2)+'\n')
 print('VISION_GATEWAY_LAUNCH_SELFRUN PASS 7/7');return 0
if __name__=='__main__':
 try:raise SystemExit(main())
 except Exception as exc:report_exception(exc);print('VISION_GATEWAY_LAUNCH_SELFRUN FAIL');raise SystemExit(1)
