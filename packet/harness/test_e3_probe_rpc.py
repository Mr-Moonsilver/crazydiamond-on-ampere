import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import unittest
from unittest.mock import patch

from e3_probe_rpc import exchange, target, TransportError, PATH


class RpcTests(unittest.TestCase):
    def setUp(self):
        owner = self
        self.requests = []
        self.status = 200
        self.result = dict(schema=1,run_id='fixture',action='prepare',command_seq=1)
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_POST(self):
                data = self.rfile.read(int(self.headers['Content-Length']))
                owner.requests.append((self.path,self.headers.get('Authorization'),json.loads(data)))
                self.send_response(owner.status)
                self.send_header('Location','http://192.0.2.1/should-not-be-read')
                self.end_headers()
                self.wfile.write(json.dumps(owner.result).encode())
        self.server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread = threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.url = 'http://127.0.0.1:' + str(self.server.server_port)
        self.command = dict(run_id='fixture',action='prepare',command_seq=1)

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(2)

    def call(self,**kw):
        return exchange(self.command,url=self.url,key='fixture-auth',timeout_s=2,**kw)

    def test_authenticated_fixed_endpoint_single_request(self):
        self.assertEqual(self.call(),self.result)
        self.assertEqual(self.requests,[(PATH+'/session','Bearer fixture-auth',self.command)])

    def test_raw_control_has_fixed_separate_path(self):
        self.command = dict(run_id='fixture',action='open')
        self.result.update(action='open',command_seq=0)
        self.call(raw=True)
        self.assertEqual(self.requests[0][0],PATH)

    def test_redirect_and_remote_error_body_never_followed_or_echoed(self):
        self.status=307; self.result={'error':'fixture-auth private-payload'}
        with self.assertRaisesRegex(TransportError,'^http-status-307$'): self.call()
        self.assertEqual(len(self.requests),1)

    def test_changed_reply_identity_refused(self):
        for change in (dict(run_id='other'),dict(command_seq=True),dict(command_seq=2),dict(schema=True)):
            original=self.result.copy(); self.result.update(change)
            with self.assertRaisesRegex(TransportError,'identity'): self.call()
            self.result=original

    def test_no_proxy_or_hostnames_credentials_query_arbitrary_path(self):
        for url in ('http://example.com','http://127.0.0.1/a','http://***@127.0.0.1',
                    'http://127.0.0.1?key=x','file:///tmp/file','http://192.0.2.1'):
            with self.assertRaises(TransportError): target(url)
        with patch.dict('os.environ',{'http_proxy':'http://192.0.2.1:1','HTTP_PROXY':'http://192.0.2.1:1'}):
            self.call()
        self.assertEqual(len(self.requests),1)

    def test_response_limit_is_real(self):
        self.result['extra']='x'*200
        with patch('e3_probe_rpc.MAX_RESPONSE',64):
            with self.assertRaisesRegex(TransportError,'too-large'): self.call()


if __name__ == '__main__': unittest.main()
