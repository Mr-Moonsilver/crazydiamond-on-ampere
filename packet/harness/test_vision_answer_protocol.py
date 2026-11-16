"""Whole actual-argv image client/reader against schema-valid HTTP stubs, no GPU/VM."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import hashlib,json,os
from pathlib import Path
import subprocess,sys,tempfile,threading,unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from build_vision_encoder_packet import red_png

class Protocol(unittest.TestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix='vision-answer-'))
        self.calls=[];self.bad_manifest=False;self.empty_answer=False
        self.race_out=None
        for name,key in [('engine','a'),('encoder','b')]:
            p=self.root/(name+'.env');p.write_text(('VLLM_API_KEY' if name=='engine' else 'VISION_ENCODER_API_KEY')+'='+key*64);p.chmod(0o600)
        (self.root/'red.png').write_bytes(red_png(32))
        self.cases=[dict(id='red',image='red.png',question='What color? One word.',expected='red')]
        (self.root/'cases.json').write_text(json.dumps(self.cases))
        owner=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*_):pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                owner.calls.append((self.path,body))
                expected=('b' if self.path=='/v1/encode' else 'a')*64
                if self.headers.get('Authorization')!='Bearer '+expected:
                    self.send_response(401);self.end_headers();self.wfile.write(b'{}');return
                if self.path=='/v1/encode':
                    if owner.race_out is not None:
                        owner.race_out.write_bytes(b'other-owner-artifact')
                    value=dict(schema='flash-next-image-embeds/1',manifest_sha256='x'*64 if owner.bad_manifest else '8b333e2380aac6ed98c6ed5700474972cefb919cf126949fc31f70b47a39ca93',
                        dtype='bfloat16',shape=[1,2560],image_grid_thw=[[1,2,2]],embedding_sha256='c'*64,
                        content=dict(type='image_embeds',image_embeds=dict(image_embeds='CPU_STUB',image_grid_thw='CPU_STUB')))
                elif self.path=='/v1/chat/completions':
                    content=body['messages'][0]['content']
                    assert content[0]['type'] in ('image_embeds','image_url') and content[1]['text']==owner.cases[0]['question']
                    value=dict(choices=[dict(message=dict(content='' if owner.empty_answer else 'red'))],usage=dict(prompt_tokens=32,completion_tokens=1))
                else:raise AssertionError('unexpected protocol step')
                self.send_response(200);self.end_headers();self.wfile.write(json.dumps(value).encode())
        self.http=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        threading.Thread(target=self.http.serve_forever,daemon=True).start()
        self.addCleanup(self.http.server_close);self.addCleanup(self.http.shutdown)
        self.url='http://127.0.0.1:'+str(self.http.server_port)
    def drive(self,mode='embeddings',name='result.json',extra=()):
        argv=[sys.executable,'-B',str(ROOT/'harness/vision_answer_probe.py'),'--mode',mode,
              '--engine-url',self.url,'--engine-key-file',str(self.root/'engine.env'),'--model','CPU-STUB',
              '--encoder-url',self.url,'--encoder-key-file',str(self.root/'encoder.env'),
              '--cases',str(self.root/'cases.json'),'--out',str(self.root/name),*extra]
        (self.root/(name+'.argv.json')).write_text(json.dumps(argv))
        result=subprocess.run(argv,capture_output=True,text=True,timeout=20)
        (self.root/(name+'.stdout')).write_text(result.stdout);(self.root/(name+'.stderr')).write_text(result.stderr)
        return result
    def test_whole_embeddings_and_reference_cli_then_reader(self):
        for mode,name in [('embeddings','engine.json'),('reference','reference.json')]:
            result=self.drive(mode,name);self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('VISION_ANSWER_PROBE COMPLETE',result.stdout)
        read=subprocess.run([sys.executable,'-B',str(ROOT/'tools/vision_answer_read.py'),'--record',str(self.root/'engine.json'),
            '--reference-record',str(self.root/'reference.json')],capture_output=True,text=True)
        self.assertEqual(read.returncode,0,read.stderr);self.assertIn('VISION_ANSWER_READ MATCH',read.stdout)
        self.assertEqual([p for p,_ in self.calls],['/v1/encode','/v1/chat/completions','/v1/chat/completions'])
        self.assertNotIn('CPU_STUB', (self.root/'engine.json').read_text())
    def test_encoder_identity_refuses_before_engine_call(self):
        self.bad_manifest=True;result=self.drive()
        self.assertEqual(result.returncode,2)
        self.assertEqual([p for p,_ in self.calls],['/v1/encode'])
        self.assertIn('ValueError',result.stderr);self.assertRegex(result.stderr,r'"file": "[^"]+", "line": [0-9]+')
    def test_missing_actual_encoder_key_argument_refuses(self):
        (self.root/'encoder.env').chmod(0o644)
        result=self.drive();self.assertEqual(result.returncode,2);self.assertEqual(self.calls,[])
        self.assertIn('private0600',result.stderr)
    def test_per_image_grid_is_canonical_and_does_not_change_the_cache(self):
        from vision_answer_probe import embedding_content
        response=dict(schema='flash-next-image-embeds/1',manifest_sha256='8b333e2380aac6ed98c6ed5700474972cefb919cf126949fc31f70b47a39ca93',
            dtype='bfloat16',shape=[1,2560],embedding_sha256='c'*64,image_grid_thw=[[1,2,2]],
            content=dict(type='image_embeds',image_embeds=dict(image_embeds='original-payload',image_grid_thw='original-grid-tensor')))
        before=json.dumps(response,sort_keys=True);content=embedding_content(response)
        self.assertEqual(content['image_embeds']['image_grid_thw'],[1,2,2])
        self.assertEqual(content['image_embeds']['image_embeds'],'original-payload')
        self.assertEqual(json.dumps(response,sort_keys=True),before)
        response['image_grid_thw']=[[1,4,4]]
        with self.assertRaisesRegex(ValueError,'grid differs'):embedding_content(response)

    def test_empty_engine_reply_refuses_with_exception_location(self):
        self.empty_answer=True;result=self.drive()
        self.assertEqual(result.returncode,2);self.assertIn('answer text missing',result.stderr)
        self.assertRegex(result.stderr,r'"file": "[^"]+", "line": [0-9]+')
    def test_reader_incomplete_record_never_matches(self):
        p=self.root/'incomplete.json';p.write_text(json.dumps(dict(verdict='HOLD',cases=[])))
        result=subprocess.run([sys.executable,'-B',str(ROOT/'tools/vision_answer_read.py'),'--record',str(p)],capture_output=True,text=True)
        self.assertEqual(result.returncode,2);self.assertIn('VISION_ANSWER_READ HOLD',result.stdout)

    def test_export_actual_argv_then_offline_client_uses_no_encoder(self):
        cache=self.root/'features.json'
        argv=[sys.executable,'-B',str(ROOT/'harness/vision_embeddings_export.py'),'--encoder-url',self.url,
            '--encoder-key-file',str(self.root/'encoder.env'),'--cases',str(self.root/'cases.json'),'--out',str(cache)]
        (self.root/'export.argv.json').write_text(json.dumps(argv))
        exported=subprocess.run(argv,capture_output=True,text=True)
        self.assertEqual(exported.returncode,0,exported.stderr);self.assertIn('VISION_EMBEDDINGS_EXPORT PASS',exported.stdout)
        self.assertEqual(cache.stat().st_mode & 0o777,0o600)
        digest=hashlib.sha256(cache.read_bytes()).hexdigest()
        second=subprocess.run(argv,capture_output=True,text=True)
        self.assertEqual(second.returncode,2)
        self.assertIn('FileExistsError',second.stderr)
        self.assertEqual(hashlib.sha256(cache.read_bytes()).hexdigest(),digest)
        result=self.drive(extra=['--embeddings-cache',str(cache),'--embeddings-cache-sha256',digest,
            '--encoder-url','http://127.0.0.1:1'])
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual([p for p,_ in self.calls],['/v1/encode','/v1/chat/completions'])
        self.calls.clear()
        refused=self.drive(name='corrupt.json',extra=['--embeddings-cache',str(cache),'--embeddings-cache-sha256','0'*64])
        self.assertEqual(refused.returncode,2);self.assertEqual(self.calls,[])
        self.assertIn('cache SHA differs',refused.stderr)

    def test_export_refuses_bad_model_without_writing_cache(self):
        self.bad_manifest=True;cache=self.root/'bad-export.json'
        result=subprocess.run([sys.executable,'-B',str(ROOT/'harness/vision_embeddings_export.py'),
            '--encoder-url',self.url,'--encoder-key-file',str(self.root/'encoder.env'),
            '--cases',str(self.root/'cases.json'),'--out',str(cache)],capture_output=True,text=True)
        self.assertEqual(result.returncode,2);self.assertFalse(cache.exists())
        self.assertIn('VISION_EMBEDDINGS_EXPORT HOLD',result.stdout)
        self.assertIn('ValueError',result.stderr)

    def test_export_race_never_overwrites_another_artifact(self):
        cache=self.root/'racing-export.json';self.race_out=cache
        result=subprocess.run([sys.executable,'-B',str(ROOT/'harness/vision_embeddings_export.py'),
            '--encoder-url',self.url,'--encoder-key-file',str(self.root/'encoder.env'),
            '--cases',str(self.root/'cases.json'),'--out',str(cache)],capture_output=True,text=True)
        self.assertEqual(result.returncode,2)
        self.assertEqual(cache.read_bytes(),b'other-owner-artifact')
if __name__=='__main__':unittest.main()
