"""Admission tests use real HTTP sockets; backend completion is deliberately held."""
import concurrent.futures as futures
import http.client, inspect, json, tempfile, threading, time, unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler
import vision_public_gateway as g

class Queue(unittest.TestCase):
    def setUp(self):
        self.release=threading.Event();self.condition=threading.Condition();self.arrived=0;self.peak=0;self.active=0
        owner=self
        class Model(BaseHTTPRequestHandler):
            def log_message(self,*_):pass
            def do_GET(self):
                self.send_response(200);self.end_headers();self.wfile.write(b'{"data":[]}')
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                with owner.condition:
                    owner.arrived+=1;owner.active+=1;owner.peak=max(owner.peak,owner.active);owner.condition.notify_all()
                owner.release.wait(12)
                with owner.condition:owner.active-=1
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers();self.wfile.write(b'data: [DONE]\n\n')
        self.model=g.GatewayServer(('127.0.0.1',0),Model)
        self.gateway=g.Gateway('http://127.0.0.1:'+str(self.model.server_port),'http://<encoder-host>:9041','a'*64,'b'*64)
        self.http=g.GatewayServer(('127.0.0.1',0),g.handler(self.gateway))
        for server in (self.model,self.http):
            threading.Thread(target=server.serve_forever,daemon=True).start()
            self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        self.addCleanup(self.release.set)
    def call(self,path='/v1/chat/completions',body=True):
        conn=http.client.HTTPConnection('127.0.0.1',self.http.server_port,timeout=15)
        data=json.dumps(getattr(self,'post_body',dict(model='qwen38-w8a8-tp4',messages=[dict(role='user',content='hello')],stream=True))) if body else None
        conn.request('POST' if body else 'GET',path,data,{'Content-Type':'application/json','Authorization':'Bearer '+'a'*64} if body else {})
        reply=conn.getresponse();result=reply.status,reply.read();conn.close();return result
    def wait_arrivals(self,count):
        with self.condition:self.assertTrue(self.condition.wait_for(lambda:self.arrived>=count,timeout=6),f'only {self.arrived}/{count} reached backend')
    def probes(self):
        started=time.monotonic()
        for _ in range(3):
            for path in ('/health','/metrics','/v1/models'):self.assertEqual(self.call(path,False)[0],200)
        self.assertLess(time.monotonic()-started,2)
    def wave(self,callers):
        with futures.ThreadPoolExecutor(max_workers=callers) as pool:
            jobs=[pool.submit(self.call) for _ in range(callers)]
            self.wait_arrivals(min(callers,32));self.probes()
            self.assertFalse(any(j.done() for j in jobs),'waiting caller was rejected')
            self.release.set()
            self.assertTrue(all(j.result()[0]==200 for j in jobs))
        self.assertLessEqual(self.peak,32)
    def test_16_callers_and_metadata(self):self.wave(16)
    def test_32_callers_and_metadata_at_full_admission(self):self.wave(32)
    def test_64_callers_queue_then_complete_without_429(self):self.wave(64)
    def test_32_image_callers_keep_encoder_preparation_at_eight(self):
        import base64
        from test_vision_encoder_protocol import FakeBackend
        image='data:image/png;base64,'+base64.b64encode(b'\x89PNG\r\n\x1a\nCPU_LEAF').decode()
        self.post_body=dict(model='qwen38-w8a8-tp4',messages=[dict(role='user',content=[dict(type='image_url',image_url=dict(url=image))])],stream=True)
        active=peak=0;lock=threading.Lock()
        def encode(data):
            nonlocal active,peak
            with lock:active+=1;peak=max(peak,active)
            try:time.sleep(.03);return FakeBackend().encode(data)
            finally:
                with lock:active-=1
        self.gateway.encode=encode;self.wave(32);self.assertLessEqual(peak,8);self.assertGreater(peak,1)
    def test_timeout_waits_then_504_and_metadata_bypasses(self):
        self.gateway.slots=threading.BoundedSemaphore(1);self.gateway.queue_timeout_s=.2
        with futures.ThreadPoolExecutor(max_workers=2) as pool:
            first=pool.submit(self.call);self.wait_arrivals(1)
            started=time.monotonic();code,body=self.call()
            self.assertEqual(code,504);self.assertIn(b'gateway_queue_timeout',body);self.assertGreaterEqual(time.monotonic()-started,.17)
            self.probes();self.release.set();self.assertEqual(first.result()[0],200)
        self.assertEqual(self.gateway.counts['admission_timeouts'],1)
    def test_queued_request_rechecks_native_readiness(self):
        self.gateway.slots=threading.BoundedSemaphore(1)
        self.gateway.ready_file=Path(tempfile.mkdtemp())/'ready.json';self.gateway.packet_sha='c'*64
        self.gateway.ready_file.write_text(json.dumps(dict(ready=True,packet_sha256='c'*64)))
        with futures.ThreadPoolExecutor(max_workers=2) as pool:
            first=pool.submit(self.call);self.wait_arrivals(1);second=pool.submit(self.call)
            time.sleep(.1);self.gateway.ready_file.write_text(json.dumps(dict(ready=False,packet_sha256='c'*64)))
            self.release.set();self.assertEqual(first.result()[0],200);self.assertEqual(second.result()[0],503)
        self.assertEqual(self.arrived,1)
    def test_invalid_body_releases_both_permits(self):
        conn=http.client.HTTPConnection('127.0.0.1',self.http.server_port,timeout=3)
        conn.request('POST','/v1/chat/completions',b'!',{'Content-Type':'application/json','Authorization':'Bearer '+'a'*64})
        reply=conn.getresponse();self.assertEqual(reply.status,400);reply.read();conn.close()
        for semaphore,count in ((self.gateway.slots,32),(self.gateway.prepare_slots,8)):
            self.assertTrue(all(semaphore.acquire(False) for _ in range(count)));self.assertFalse(semaphore.acquire(False))
            for _ in range(count):semaphore.release()
    def test_nonblocking_revert_mutant_is_detected(self):
        source=inspect.getsource(g.handler)
        old='gateway.slots.acquire(timeout=max(0,deadline-time.monotonic()))'
        self.assertIn(old,source)
        namespace=dict(g.__dict__);exec(source.replace(old,'gateway.slots.acquire(blocking=False)'),namespace)
        self.http.RequestHandlerClass=namespace['handler'](self.gateway)
        self.gateway.slots=threading.BoundedSemaphore(1)
        with futures.ThreadPoolExecutor(max_workers=2) as pool:
            first=pool.submit(self.call);self.wait_arrivals(1)
            started=time.monotonic();self.assertEqual(self.call()[0],504)
            self.assertLess(time.monotonic()-started,.15,'mutant no longer reproduces immediate refusal')
            self.release.set();self.assertEqual(first.result()[0],200)
    def test_metadata_admission_revert_mutant_is_detected(self):
        source=inspect.getsource(g.handler);old="if post or self.path not in ('/health','/metrics','/v1/models'):"
        self.assertIn(old,source);namespace=dict(g.__dict__);exec(source.replace(old,'if True:'),namespace)
        self.http.RequestHandlerClass=namespace['handler'](self.gateway)
        self.gateway.slots=threading.BoundedSemaphore(1);self.gateway.queue_timeout_s=.1
        with futures.ThreadPoolExecutor(max_workers=2) as pool:
            first=pool.submit(self.call);self.wait_arrivals(1)
            self.assertEqual(self.call('/health',False)[0],504)
            self.release.set();self.assertEqual(first.result()[0],200)
    def test_ladder_setting32_keeps_frozen_engine_eight_image_cap(self):
        self.gateway.max_images=32;called=[]
        body=dict(model='qwen38-w8a8-tp4',messages=[dict(role='user',content=[dict(type='image_url',image_url=dict(url='data:image/png;base64,AAAA')) for _ in range(12)])])
        with self.assertRaisesRegex(g.GatewayError,'frozen_engine_image_count_limit_8'):g.transform(body,called.append,max_images=32)
        self.assertEqual(called,[])
    def test_invalid_config_refuses(self):
        for kwargs in (dict(max_inflight=0),dict(max_inflight=33),dict(max_inflight=True),dict(queue_timeout_s=float('nan')),dict(queue_timeout_s=0)):
            with self.assertRaises(ValueError):g.Gateway('http://127.0.0.1:8001','','','',**kwargs)

if __name__=='__main__':unittest.main()
