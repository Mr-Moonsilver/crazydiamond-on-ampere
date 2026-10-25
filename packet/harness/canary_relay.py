"""Canary-only loopback relay: unchanged bodies/auth, immediate SSE forwarding.

Records request metadata and terminal outcome, never headers, bodies or response
text. Session path markers are stripped before forwarding to the real gateway.
No cache_salt, model, prompt or sampling settings are modified.
"""
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlsplit
import uuid

HOP={'connection','proxy-connection','keep-alive','transfer-encoding','te','trailer','upgrade','host'}


class SSEEnd:
    def __init__(self):
        self.pending=b''; self.stopped=False; self.error=False; self.ids=set()

    def feed(self, chunk):
        self.pending += chunk
        while b'\n' in self.pending:
            line,self.pending=self.pending.split(b'\n',1)
            if not line.startswith(b'data:'): continue
            try: d=json.loads(line[5:].strip())
            except ValueError:
                self.error=True; continue
            if d.get('type')=='message_stop': self.stopped=True
            if d.get('type')=='error': self.error=True
            if d.get('type')=='message_start' and isinstance(d.get('message',{}).get('id'),str):
                self.ids.add(d['message']['id'])
        if len(self.pending)>2**20:
            self.error=True; self.pending=b''


class Relay:
    def __init__(self, upstream, markers, journal):
        u=urlsplit(upstream)
        if u.scheme!='http' or u.username or u.password or u.path not in ('','/'):
            raise ValueError('reviewed HTTP gateway required')
        self.upstream=u
        self.markers=set(markers)
        self.journal=Path(journal)
        if self.journal.exists(): raise ValueError('existing relay journal')
        self.lock=threading.Lock(); self.rows=[]; self.active=0
        owner=self

        class Handler(BaseHTTPRequestHandler):
            protocol_version='HTTP/1.1'
            def log_message(self,*_): pass  # never log auth/path queries

            def do_POST(self):
                match=re.fullmatch(r'/s/([A-Za-z0-9_-]+)(/.*)',self.path)
                if not match or match[1] not in owner.markers:
                    self.send_error(403); return
                session,path=match[1],match[2]
                kind='messages' if path.split('?',1)[0]=='/v1/messages' else (
                    'count_tokens' if path.split('?',1)[0]=='/v1/messages/count_tokens' else 'other')
                row=dict(id=uuid.uuid4().hex,session=session,kind=kind,started_wall=time.time(),
                         outcome='pending',status=None,message_ids=[])
                conn=None; headers_sent=False; parser=SSEEnd(); phase='client-read'
                with owner.lock: owner.active+=1
                try:
                    if self.headers.get('Transfer-Encoding'):
                        # The Anthropic JSON client sends Content-Length. An
                        # unreviewed request framing fails closed, never truncated.
                        row['outcome']='unsupported_request_framing'; self.send_error(411); return
                    length=int(self.headers.get('Content-Length','-1'))
                    if length<0 or length>64*2**20: raise ValueError('invalid request size')
                    body=self.rfile.read(length)
                    if len(body)!=length: raise ConnectionResetError('short request')
                    headers={k:v for k,v in self.headers.items() if k.lower() not in HOP}
                    # Authorization and x-api-key remain unchanged, in memory only.
                    conn=http.client.HTTPConnection(owner.upstream.hostname,owner.upstream.port or 80,timeout=910)
                    phase='upstream-request'
                    conn.request('POST',path,body=body,headers=headers)
                    response=conn.getresponse(); row['status']=response.status
                    sse='text/event-stream' in response.getheader('Content-Type','')
                    phase='client-write'
                    self.send_response_only(response.status)
                    for k,v in response.getheaders():
                        if k.lower() not in HOP|{'content-length'}: self.send_header(k,v)
                    self.send_header('Transfer-Encoding','chunked'); self.send_header('Connection','close')
                    self.end_headers();headers_sent=True;self.close_connection=True
                    while True:
                        phase='upstream-read'
                        chunk=response.read1(65536)
                        if not chunk: break
                        # Forward and flush before parsing; do not hold a complete
                        # event or message while waiting for the next upstream byte.
                        phase='client-write'
                        self.wfile.write(f'{len(chunk):x}\r\n'.encode()+chunk+b'\r\n'); self.wfile.flush()
                        if sse: parser.feed(chunk)
                    phase='client-write'
                    self.wfile.write(b'0\r\n\r\n');self.wfile.flush()
                    if response.status>=500: row['outcome']='upstream_5xx'
                    elif response.status!=200: row['outcome']='upstream_non200'
                    elif sse:
                        if parser.pending: parser.feed(b'\n')
                        row['outcome']='completed' if parser.stopped and not parser.error else 'incomplete_sse'
                    else: row['outcome']='completed'
                    row['message_ids']=sorted(parser.ids)
                except (BrokenPipeError,ConnectionResetError):
                    row['outcome']='client_disconnect' if phase.startswith('client') else 'upstream_disconnect'
                except Exception as e:
                    row['outcome']='relay_error';row['error_type']=type(e).__name__
                    if not headers_sent:
                        try: self.send_error(502)
                        except (BrokenPipeError,ConnectionResetError): pass
                finally:
                    if conn: conn.close()
                    row['ended_wall']=time.time()
                    with owner.lock:
                        owner.rows.append(row);owner.active-=1
                        with owner.journal.open('a') as f: f.write(json.dumps(row)+'\n')

        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.server.daemon_threads=True
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)

    @property
    def url(self): return f'http://127.0.0.1:{self.server.server_port}'

    def start(self): self.thread.start()

    def snapshot(self):
        with self.lock: return dict(active=self.active,requests=list(self.rows))

    def close(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=3)
