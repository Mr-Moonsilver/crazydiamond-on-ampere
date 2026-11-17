"""Whole HTTP proxy protocol with real sockets/SSE and bounded schema-valid encoder/model leaves."""
import base64,copy,http.client,json,os,subprocess,sys,tempfile,threading,time,unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from unittest.mock import patch
import vision_public_gateway as g
from test_vision_encoder_protocol import FakeBackend

class Protocol(unittest.TestCase):
    def setUp(self):
        self.seen=[];self.encodes=[];self.encoder_down=False
        owner=self
        class Model(BaseHTTPRequestHandler):
            def log_message(self,*_):pass
            def do_GET(self):
                self.send_response(200);self.end_headers();self.wfile.write(b'{"data":[{"id":"qwen38-w8a8-tp4"}]}')
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));owner.seen.append((self.path,body,dict(self.headers)))
                self.send_response(200)
                self.send_header('Content-Type','text/event-stream' if body.get('stream') else 'application/json');self.end_headers()
                if body.get('stream'):
                    self.wfile.write(b'event: message_start\ndata: {"type":"message_start"}\n\n' if self.path=='/v1/messages' else b'data: {"choices":[{"delta":{"content":"red"}}]}\n\n')
                    self.wfile.flush();time.sleep(.01);self.wfile.write(b'data: [DONE]\n\n')
                else:self.wfile.write(b'{"choices":[{"message":{"content":"red"}}],"usage":{"completion_tokens":1}}')
        self.model=ThreadingHTTPServer(('127.0.0.1',0),Model)
        self.gateway=g.Gateway('http://127.0.0.1:'+str(self.model.server_port),'http://<encoder-host>:9041','a'*64,'b'*64)
        def encode(raw):
            if self.encoder_down:raise g.GatewayError(503,'encoder_unavailable')
            self.encodes.append(raw);return FakeBackend().encode(raw)
        self.gateway.encode=encode
        self.http=ThreadingHTTPServer(('127.0.0.1',0),g.handler(self.gateway))
        for server in (self.model,self.http):
            server.daemon_threads=True;threading.Thread(target=server.serve_forever,daemon=True).start()
            self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        self.image='data:image/png;base64,'+base64.b64encode(b'\x89PNG\r\n\x1a\nCPU_SCHEMA_LEAF').decode()
    def call(self,path='/v1/chat/completions',body=None,key='a'*64,headers=None):
        conn=http.client.HTTPConnection('127.0.0.1',self.http.server_port,timeout=5)
        hdr={'Content-Type':'application/json'}
        if key is not None:hdr['Authorization']='Bearer '+key
        hdr.update(headers or {})
        conn.request('GET' if body is None else 'POST',path,None if body is None else json.dumps(body).encode(),hdr)
        response=conn.getresponse();data=response.read();status=response.status;conn.close();return status,data
    def body(self):return dict(model='qwen38-w8a8-tp4',messages=[dict(role='user',content=[dict(type='image_url',image_url=dict(url=self.image)),dict(type='text',text='Colour?')])],max_tokens=128)
    def test_full_text_image_tool_messages_sse_and_keyless_metadata(self):
        for path in ('/health','/metrics','/v1/models'):self.assertEqual(self.call(path,key=None)[0],200)
        self.assertEqual(self.call(body=self.body(),key=None)[0],401);self.assertEqual(self.encodes,[])
        self.assertEqual(self.call(body=self.body())[0],200)
        forwarded=self.seen[-1][1];self.assertEqual(forwarded['messages'][0]['content'][0]['type'],'image_embeds')
        self.assertNotIn('chat_template_kwargs',forwarded);self.assertNotIn('reasoning_effort',forwarded)
        text=dict(model='qwen38-w8a8-tp4',messages=[dict(role='user',content='Text')],tools=[dict(type='function',function=dict(name='f',parameters=dict(type='object')))],stream=True)
        self.assertEqual(self.call(body=text)[0],200);self.assertEqual(self.seen[-1][1],text)
        anth=dict(model='qwen38-w8a8-tp4',max_tokens=128,messages=[dict(role='user',content=[dict(type='text',text='hello')])],stream=True)
        code,raw=self.call('/v1/messages',anth,key=None,headers={'x-api-key':'a'*64,'anthropic-version':'2023-06-01'})
        self.assertEqual(code,200);self.assertIn(b'message_start',raw);self.assertEqual(self.seen[-1][1],anth)
        self.assertEqual(self.seen[-1][2]['Authorization'],'Bearer '+'a'*64)
        self.assertEqual(self.seen[-1][2]['anthropic-version'],'2023-06-01')
    def test_anthropic_image_and_tool_result_embedding_scope(self):
        image=dict(type='image',source=dict(type='base64',media_type='image/png',data=base64.b64encode(b'\x89PNG\r\n\x1a\nCPU_SCHEMA_LEAF').decode()))
        for part in (image,dict(type='tool_result',tool_use_id='one',content=[image])):
            body=dict(model='qwen38-w8a8-tp4',max_tokens=128,messages=[dict(role='user',content=[part])])
            self.assertEqual(self.call('/v1/messages',body)[0],200)
            result=self.seen[-1][1]['messages'][0]['content'][0]
            if result['type']=='tool_result':result=result['content'][0]
            self.assertEqual(result['source']['type'],'flash_next_embeds')
    def test_video_multiple_untrusted_embeddings_and_downstream_unavailability(self):
        bad=self.body();bad['messages'][0]['content'].append(copy.deepcopy(bad['messages'][0]['content'][0]))
        self.assertEqual(self.call(body=bad)[0],400)
        for kind in ('image_embeds','video_url','input_audio'):
            body=self.body();body['messages'][0]['content'][0]=dict(type=kind)
            self.assertEqual(self.call(body=body)[0],400)
        self.assertEqual(self.seen,[])
        self.encoder_down=True;code,raw=self.call(body=self.body());self.assertEqual(code,503);self.assertIn(b'encoder_unavailable',raw)
        self.assertEqual(self.seen,[])
    def test_native_readiness_refuses_stale_or_missing_epoch_receipt(self):
        self.gateway.ready_file=Path(tempfile.mkdtemp(prefix='gateway-ready-'))/'readiness.json';self.gateway.packet_sha='c'*64
        self.assertEqual(self.call(body=self.body())[0],503);self.assertEqual(self.seen,[])
        self.gateway.ready_file.write_text(json.dumps(dict(ready=True,packet_sha256='d'*64)))
        self.assertEqual(self.call(body=self.body())[0],503)
        self.gateway.ready_file.write_text(json.dumps(dict(ready=True,packet_sha256='c'*64)))
        self.assertEqual(self.call(body=self.body())[0],200)
        self.gateway.ready_file.write_text(json.dumps(dict(ready=False,packet_sha256='c'*64)))
        self.assertEqual(self.call('/health',key=None)[0],503)

    def test_public_url_pins_dns_and_refuses_private_or_redirect(self):
        for url in ('file:///etc/passwd','http://127.0.0.1/a','http://***@example.com/a'):
            with self.assertRaises(g.GatewayError):g.raw_image(url)
        with patch.object(g.socket,'getaddrinfo',return_value=[(2,1,6,'',('<encoder-host>',80))]):
            with self.assertRaises(g.GatewayError):g.fetch_image('http://example.com/a')
        with patch.object(g,'fetch_image',return_value=b'\x89PNG\r\n\x1a\nCPU_SCHEMA_LEAF') as fetch:
            self.assertEqual(g.raw_image('https://example.com/image.png'),b'\x89PNG\r\n\x1a\nCPU_SCHEMA_LEAF');fetch.assert_called_once()
    def test_gates_revert_mutants(self):
        import inspect
        code=inspect.getsource(g.transform).replace("if len(images)>1:","if False:")
        namespace=dict(g.__dict__);exec(code,namespace)
        bad=self.body();bad['messages'][0]['content'].append(copy.deepcopy(bad['messages'][0]['content'][0]))
        with self.assertRaises(g.GatewayError):g.transform(bad,self.gateway.encode)
        # Removing the cardinality guard would forward a second raw image to the private engine.
        forwarded,count=namespace['transform'](bad,self.gateway.encode)
        self.assertEqual(count,1);self.assertEqual(forwarded['messages'][0]['content'][2]['type'],'image_url')


class Process(unittest.TestCase):
    def test_generated_main_argv_reaches_stop_receipt_with_real_http_and_encoder_leaf(self):
        root=Path(__file__).resolve().parent;tmp=Path(tempfile.mkdtemp(prefix='gateway-process-'))
        key=tmp/'key.env';key.write_text('VLLM_API_KEY='+'a'*64);key.chmod(0o600)
        enc=tmp/'encoder.env';enc.write_text('VISION_ENCODER_API_KEY='+'b'*64);enc.chmod(0o600)
        seen=[]
        class Model(BaseHTTPRequestHandler):
            def log_message(self,*_):pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));seen.append(body)
                self.send_response(200);self.end_headers();self.wfile.write(b'{"choices":[{"message":{"content":"red"}}],"usage":{"completion_tokens":1}}')
        model=ThreadingHTTPServer(('127.0.0.1',0),Model);threading.Thread(target=model.serve_forever,daemon=True).start();self.addCleanup(model.server_close);self.addCleanup(model.shutdown)
        import socket
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        driver=tmp/'driver.py';driver.write_text('import sys\nsys.path.insert(0,'+repr(str(root))+')\nimport vision_public_gateway as g\nfrom test_vision_encoder_protocol import FakeBackend\ng.Gateway.encode=lambda self,data:FakeBackend().encode(data)\nraise SystemExit(g.main())\n')
        argv=[sys.executable,'-I','-B',str(driver),'--bind','127.0.0.1','--port',str(port),'--backend','http://127.0.0.1:'+str(model.server_port),'--encoder','http://<encoder-host>:9041','--key-file',str(key),'--encoder-key-file',str(enc)]
        (tmp/'actual-argv.json').write_text(json.dumps(argv))
        stdout=(tmp/'stdout.log').open('x');stderr=(tmp/'stderr.log').open('x');proc=subprocess.Popen(argv,stdout=stdout,stderr=stderr)
        try:
            until=time.monotonic()+10
            while 'VISION_GATEWAY_READY' not in (tmp/'stdout.log').read_text():
                if proc.poll() is not None or time.monotonic()>until:self.fail((tmp/'stderr.log').read_text())
                time.sleep(.05)
            conn=http.client.HTTPConnection('127.0.0.1',port,timeout=5)
            body=dict(model='qwen38-w8a8-tp4',messages=[dict(role='user',content=[dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(b'\x89PNG\r\n\x1a\nCPU_LEAF').decode()))])])
            conn.request('POST','/v1/chat/completions',json.dumps(body),{'Content-Type':'application/json','Authorization':'Bearer '+'a'*64});response=conn.getresponse();self.assertEqual(response.status,200);response.read();conn.close()
            self.assertEqual(seen[0]['messages'][0]['content'][0]['type'],'image_embeds')
        finally:
            proc.terminate();proc.wait(timeout=10);stdout.close();stderr.close()
        self.assertEqual(proc.returncode,0,(tmp/'stderr.log').read_text());self.assertIn('VISION_GATEWAY_STOPPED', (tmp/'stdout.log').read_text())

if __name__=='__main__':unittest.main()
