"""Real loopback HTTP tests with a fake upstream; never calls the model/gateway."""
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest
from canary_relay import Relay, SSEEnd

HERE=Path(__file__).resolve().parent


class RelayTests(unittest.TestCase):
    def test_sse_end_requires_terminal_event(self):
        p=SSEEnd();p.feed(b'data: {"type":"message_delta"}\n\n')
        self.assertFalse(p.stopped)
        p.feed(b'data: {"type":"message_');p.feed(b'stop"}\n\n')
        self.assertTrue(p.stopped);self.assertFalse(p.error)
        p.feed(b'data: {"type":"error"}\n\n');self.assertTrue(p.error)

    def exchange(self, mode):
        seen={};first_read=threading.Event()
        class Upstream(BaseHTTPRequestHandler):
            protocol_version='HTTP/1.1'
            def log_message(self,*_): pass
            def do_POST(self):
                seen.update(path=self.path,auth=self.headers.get('Authorization'),
                            body=self.rfile.read(int(self.headers['Content-Length'])))
                if mode=='nonstream':
                    b=b'{"type":"message","id":"background"}'
                    self.send_response(200);self.send_header('Content-Length',str(len(b)));self.end_headers()
                    self.wfile.write(b);return
                if mode=='5xx':
                    self.send_response(503);self.send_header('Content-Length','0');self.end_headers();return
                self.send_response(200);self.send_header('Content-Type','text/event-stream')
                self.send_header('Transfer-Encoding','chunked');self.end_headers()
                parts=[b'data: {"type":"message_start","message":{"id":"m1"}}\n\n']
                if mode!='incomplete': parts.append(b'data: {"type":"message_stop"}\n\n')
                for i,part in enumerate(parts):
                    if i and not first_read.wait(2): seen['buffered']=True
                    self.wfile.write(f'{len(part):x}\r\n'.encode()+part+b'\r\n');self.wfile.flush()
                self.wfile.write(b'0\r\n\r\n');self.wfile.flush()
        upstream=ThreadingHTTPServer(('127.0.0.1',0),Upstream)
        thread=threading.Thread(target=upstream.serve_forever,daemon=True);thread.start()
        root=Path(tempfile.mkdtemp(prefix='relay-',dir=HERE.parents[1]/'tmp'))
        self.addCleanup(shutil.rmtree,root,ignore_errors=True)
        relay=Relay(f'http://127.0.0.1:{upstream.server_port}',{'test-r1-s1'},root/'journal.jsonl');relay.start()
        body=b'{"model":"test","messages":[],"private":"fixture-body"}'
        conn=http.client.HTTPConnection('127.0.0.1',relay.server.server_port,timeout=5)
        try:
            conn.request('POST','/s/test-r1-s1/v1/messages?beta=true',body,{'Authorization':'fixture-auth','Content-Type':'application/json'})
            response=conn.getresponse()
            chunk=response.read1(65536);first_read.set();rest=response.read()
            for _ in range(100):
                if relay.snapshot()['active']==0: break
                time.sleep(.01)
            snap=relay.snapshot()
            self.assertEqual(relay.server.server_address[0],'127.0.0.1')
            self.assertEqual(seen['path'],'/v1/messages?beta=true')
            self.assertEqual(seen['body'],body);self.assertEqual(seen['auth'],'fixture-auth')
            self.assertNotIn('buffered',seen)
            journal=(root/'journal.jsonl').read_text()
            self.assertNotIn('fixture-body',journal);self.assertNotIn('fixture-auth',journal)
            return snap['requests'][0],chunk+rest
        finally:
            first_read.set();conn.close();relay.close();upstream.shutdown();upstream.server_close();thread.join()

    def test_sse_is_forwarded_before_upstream_finishes(self):
        row,body=self.exchange('sse')
        self.assertEqual(row['outcome'],'completed');self.assertEqual(row['message_ids'],['m1'])
        self.assertIn(b'message_stop',body)

    def test_nonstream_background_is_counted(self):
        row,_=self.exchange('nonstream');self.assertEqual(row['outcome'],'completed')

    def test_incomplete_stream_is_not_completed(self):
        row,_=self.exchange('incomplete');self.assertEqual(row['outcome'],'incomplete_sse')

    def test_5xx_is_not_completed(self):
        row,_=self.exchange('5xx');self.assertEqual(row['outcome'],'upstream_5xx')

if __name__=='__main__': unittest.main()
