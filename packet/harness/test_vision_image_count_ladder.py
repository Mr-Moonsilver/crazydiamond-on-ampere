import hashlib,json,os,subprocess,sys,tempfile,threading,unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import vision_image_count_ladder as ladder
class Ladder(unittest.TestCase):
 def test_generated_actual_argv_full_protocol_stops_on_frozen_limit(self):
  tmp=Path(tempfile.mkdtemp());log=tmp/'engine.log';log.write_text('');seen=[]
  class Backend(BaseHTTPRequestHandler):
   def log_message(self,*_):pass
   def do_POST(self):
    body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));n=sum(x['type']=='image_url' for x in body['messages'][0]['content']);seen.append(n)
    if n>8:
     self.send_response(400);self.end_headers();self.wfile.write(b'{"error":{"code":"frozen_engine_image_count_limit_8"}}');return
    uid=body['request_id'];rid='chatcmpl-'+uid+'-01234567'
    with log.open('a') as f:
     for rank in range(4):f.write('flash-s5 native-mm-boundary '+json.dumps(dict(request_sha256=hashlib.sha256(rid.encode()).hexdigest(),worker_rank=rank,free_bytes=300*2**20,native_mm=dict(cache_storage_bytes=40*2**20)))+'\n')
    raw=json.dumps(dict(id=rid,choices=[dict(message=dict(content=json.dumps(list(range(1,n+1)))))] )).encode()
    self.send_response(200)
    for name,value in [('wall-ms','5'),('peak-allocated-bytes',str(900*2**20)),('peak-reserved-bytes',str(1100*2**20))]:self.send_header('x-flashnext-encoder-'+name,value)
    self.end_headers();self.wfile.write(raw)
  server=ThreadingHTTPServer(('127.0.0.1',0),Backend);threading.Thread(target=server.serve_forever,daemon=True).start();self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
  key=tmp/'key.env';key.write_text('VLLM_API_KEY='+'a'*64);key.chmod(0o600)
  argv=[sys.executable,'-B',str(Path(os.environ.get('PACKET_ROOT',str(Path(ladder.__file__).parent)))/Path(ladder.__file__).name),'--url','http://127.0.0.1:'+str(server.server_port),'--key-file',str(key),'--engine-log',str(log),'--out',str(tmp/'result')]
  proc=subprocess.run(argv,capture_output=True,text=True,timeout=20,env=dict(os.environ,PYTHONPATH=str(Path(ladder.__file__).parent)));self.assertEqual(proc.returncode,0,proc.stderr)
  self.assertIn('VISION_IMAGE_LADDER largest_clean=8 limit=http-400:frozen_engine_image_count_limit_8',proc.stdout);self.assertEqual(seen,[4,8,12])
  rows=json.loads((tmp/'result/result.json').read_text());self.assertEqual(rows['largest_clean'],8);self.assertEqual(len(rows['steps'][0]['ranks']),4)
  bad=argv.copy();bad[bad.index('--engine-log')+1]=str(tmp/'missing');bad[bad.index('--out')+1]=str(tmp/'bad')
  mutated=subprocess.run(bad,capture_output=True,text=True,timeout=20,env=dict(os.environ,PYTHONPATH=str(Path(ladder.__file__).parent)));self.assertNotIn('largest_clean=8',mutated.stdout)
 def test_memory_floor_cache_and_missing_rank_are_visible(self):
  uid='one';digest=hashlib.sha256(uid.encode()).hexdigest()
  rows=[dict(worker_rank=r,request_sha256=digest,free_bytes=(39 if r==1 else 250)*2**20,native_mm=dict(cache_storage_bytes=41*2**20)) for r in range(4)]
  lines='\n'.join('flash-s5 native-mm-boundary '+json.dumps(r) for r in rows)
  result=ladder.memory(lines,uid,uid);self.assertEqual(result['1']['minimum_free_bytes'],39*2**20);self.assertEqual(result['0']['max_cache_bytes'],41*2**20)
  with self.assertRaises(ValueError):ladder.memory('\n'.join(lines.splitlines()[:3]),uid,uid)
 def test_images_distinct_real_PNG(self):
  images=[ladder.png(i) for i in range(1,33)];self.assertEqual(len(set(images)),32);self.assertTrue(all(i.startswith(b'\x89PNG') for i in images))
if __name__=='__main__':unittest.main()
