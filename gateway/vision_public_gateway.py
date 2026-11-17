#!/usr/bin/env python3
"""Keyed OpenAI chat proxy: one bounded still image -> trusted encoder -> loopback engine."""
import argparse,base64,copy,hmac,http.client,ipaddress,json,re,socket,ssl,sys,threading,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0,str(Path(__file__).resolve().parent))
from vision_answer_probe import embedding_content,key as read_key,validate_embedding
from vision_encoder_backend import MODEL_MANIFEST_SHA
sys.path.insert(0,str(Path(__file__).resolve().parent))
from packet_client_contract import report_exception

MAX_BODY=16*2**20
MAX_IMAGE=8*2**20
GET_PATHS={'/v1/models','/metrics','/version','/health'}
POST_PATHS={'/v1/chat/completions','/v1/completions','/v1/messages','/v1/messages/count_tokens','/tokenize','/detokenize'}
FORBIDDEN={'image_embeds','prompt_embeds','multi_modal_data','video','video_url','input_video','input_audio','audio','audio_url'}

class GatewayError(ValueError):
    def __init__(self,status,code):self.status,self.code=status,code;super().__init__(code)


def raw_image(url):
    if not isinstance(url,str):raise GatewayError(400,'image_url_must_be_string')
    if url.startswith('data:'):
        match=re.fullmatch(r'data:image/(png|jpeg|jpg|webp);base64,([A-Za-z0-9+/=\r\n]+)',url)
        if match is None or len(match[2])>4*((MAX_IMAGE+2)//3)+1024:raise GatewayError(400,'invalid_or_oversized_image_data_url')
        try:data=base64.b64decode(match[2],validate=True)
        except ValueError:raise GatewayError(400,'invalid_image_base64') from None
    else:data=fetch_image(url)
    if not 0<len(data)<=MAX_IMAGE:raise GatewayError(413,'image_exceeds_8MiB')
    if not (data.startswith(b'\x89PNG\r\n\x1a\n') or data.startswith(b'\xff\xd8\xff')
            or (data.startswith(b'RIFF') and data[8:12]==b'WEBP')):
        raise GatewayError(400,'only_still_png_jpeg_webp_supported')
    return data


def fetch_image(url):
    # Pin the validated public IP for this connection; never resolve a second time or follow a redirect.
    if len(url)>4096 or any(ord(c)<32 or ord(c)==127 for c in url):raise GatewayError(400,'invalid_image_url')
    try:
        u=urlsplit(url);port=u.port or (443 if u.scheme=='https' else 80)
        if u.scheme not in ('http','https') or not u.hostname or u.username or u.password or u.fragment:
            raise ValueError()
        if port not in (80,443):raise ValueError()
        host=u.hostname.encode('idna').decode('ascii')
        addresses=socket.getaddrinfo(host,port,type=socket.SOCK_STREAM)
        if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
            raise GatewayError(400,'image_url_private_or_reserved_address_refused')
    except GatewayError:raise
    except (ValueError,UnicodeError,socket.gaierror):raise GatewayError(400,'invalid_or_unresolvable_image_url') from None
    conn=http.client.HTTPConnection(addresses[0][4][0],port,timeout=10)
    try:
        conn.connect()
        if u.scheme=='https':conn.sock=ssl.create_default_context().wrap_socket(conn.sock,server_hostname=host)
        authority=('['+host+']') if ':' in host else host
        if port!=(443 if u.scheme=='https' else 80):authority+=':'+str(port)
        conn.request('GET',(u.path or '/')+('?' + u.query if u.query else ''),headers={'Host':authority,'Accept':'image/png,image/jpeg,image/webp','Accept-Encoding':'identity'})
        response=conn.getresponse()
        if response.status!=200:raise GatewayError(400,'image_fetch_status_'+str(response.status))
        if response.getheader('Content-Encoding','identity')!='identity':raise GatewayError(400,'image_content_encoding_refused')
        length=response.getheader('Content-Length')
        if length is not None and (not length.isdigit() or int(length)>MAX_IMAGE):raise GatewayError(413,'image_exceeds_8MiB')
        data=response.read(MAX_IMAGE+1)
        if len(data)>MAX_IMAGE:raise GatewayError(413,'image_exceeds_8MiB')
        return data
    except GatewayError:raise
    except (OSError,http.client.HTTPException):raise GatewayError(504,'image_fetch_failed') from None
    finally:conn.close()


def transform(body,encode,anthropic=False):
    if type(body) is not dict:raise GatewayError(400,'JSON_object_required')
    if FORBIDDEN.intersection(body):raise GatewayError(400,'untrusted_embeddings_audio_or_video_refused')
    messages=body.get('messages',[])
    if not isinstance(messages,list):raise GatewayError(400,'messages_must_be_array')
    changed=copy.deepcopy(body)
    images=[]
    def parts(content):
        if not isinstance(content,list):return
        for part in content:
            if type(part) is not dict:raise GatewayError(400,'invalid_content_part')
            kind=part.get('type')
            if kind=='image_url' and not anthropic:
                image=part.get('image_url')
                if type(image) is not dict or set(image)-{'url','detail'} or 'url' not in image:
                    raise GatewayError(400,'invalid_image_url_part')
                images.append((part,image['url']))
            elif kind=='image' and anthropic:
                source=part.get('source',{})
                if type(source) is not dict:raise GatewayError(400,'invalid_image_source')
                if source.get('type')=='base64':
                    url='data:'+str(source.get('media_type',''))+';base64,'+str(source.get('data',''))
                elif source.get('type')=='url':url=source.get('url')
                else:raise GatewayError(400,'untrusted_image_source')
                images.append((part,url))
            elif kind=='tool_result' and anthropic:parts(part.get('content'))
            elif FORBIDDEN.intersection(part) or kind in FORBIDDEN:
                raise GatewayError(400,'unsupported_audio_video_or_embedding_part')
            elif not anthropic and kind!='text':raise GatewayError(400,'unsupported_content_part')
    for message in changed.get('messages',[]):
        if type(message) is not dict:raise GatewayError(400,'invalid_message')
        parts(message.get('content'))
    if len(images)>1:raise GatewayError(400,'one_image_per_request_limit')
    if not images:return body,0
    target,url=images[0]
    result=encode(raw_image(url))
    try:
        validate_embedding(result)
        encoded=result['content']['image_embeds']['image_embeds']
        if not isinstance(encoded,str) or len(encoded)>9*2**20:raise ValueError()
        part=embedding_content(result)
    except (KeyError,TypeError,ValueError):raise GatewayError(502,'encoder_identity_or_shape_invalid') from None
    target.clear()
    target.update(dict(type='image',source=dict(type='flash_next_embeds',content=part)) if anthropic else part)
    return changed,1


class Gateway:
    def __init__(self,backend,encoder,api_key,encoder_key):
        u=urlsplit(backend)
        if u.scheme!='http' or u.hostname!='127.0.0.1' or u.path not in ('','/') or not u.port:
            raise ValueError('backend must be a declared loopback HTTP port')
        self.backend=u;self.encoder=encoder.rstrip('/');self.api_key=api_key;self.encoder_key=encoder_key
        self.slots=threading.BoundedSemaphore(16);self.lock=threading.Lock()
        self.counts=dict(requests=0,images=0,encoder_errors=0,client_disconnects=0)
    def count(self,name,n=1):
        with self.lock:self.counts[name]+=n
    def encode(self,data):
        u=urlsplit(self.encoder);conn=http.client.HTTPConnection(u.hostname,u.port or 80,timeout=35)
        try:
            conn.request('POST',u.path+'/v1/encode',json.dumps(dict(image_base64=base64.b64encode(data).decode())).encode(),
                {'Content-Type':'application/json','Authorization':'Bearer '+self.encoder_key})
            reply=conn.getresponse();raw=reply.read(10*2**20+1)
            if reply.status!=200:
                if reply.status in (400,413):raise GatewayError(reply.status,'encoder_image_refused')
                raise GatewayError(503 if reply.status in (429,503,504) else 502,'encoder_unavailable_'+str(reply.status))
            if len(raw)>10*2**20:raise GatewayError(502,'encoder_response_too_large')
            return json.loads(raw)
        except GatewayError:self.count('encoder_errors');raise
        except (OSError,ValueError,http.client.HTTPException):
            self.count('encoder_errors');raise GatewayError(503,'encoder_unavailable') from None
        finally:conn.close()


def handler(gateway):
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.0'
        def log_message(self,*_):pass
        def setup(self):super().setup();self.connection.settimeout(60)
        def error(self,status,code):
            raw=json.dumps(dict(error=dict(message=code,type='vision_gateway_error',code=code))).encode()
            self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        def authenticated(self):
            supplied=self.headers.get('Authorization')
            if supplied is not None:return supplied.isascii() and hmac.compare_digest(supplied,'Bearer '+gateway.api_key)
            supplied=self.headers.get('x-api-key','')
            return supplied.isascii() and hmac.compare_digest(supplied,gateway.api_key)
        def do_GET(self):self.proxy(False)
        def do_POST(self):self.proxy(True)
        def proxy(self,post):
            if (post or self.path not in ('/health','/metrics','/v1/models')) and not self.authenticated():return self.error(401,'Unauthorized')
            if self.path not in (POST_PATHS if post else GET_PATHS):return self.error(404,'unsupported_endpoint')
            if not gateway.slots.acquire(blocking=False):return self.error(429,'gateway_queue_full')
            conn=None;started=False
            try:
                body=None;image_count=0
                if post:
                    if self.headers.get('Transfer-Encoding'):raise GatewayError(400,'chunked_request_body_refused')
                    try:size=int(self.headers.get('Content-Length','-1'))
                    except ValueError:raise GatewayError(400,'invalid_content_length') from None
                    if not 0<size<=MAX_BODY:raise GatewayError(413,'request_body_bound_16MiB')
                    if self.headers.get_content_type()!='application/json':raise GatewayError(415,'JSON_content_type_required')
                    raw=self.rfile.read(size)
                    if len(raw)!=size:raise GatewayError(400,'incomplete_request_body')
                    try:value=json.loads(raw)
                    except (ValueError,UnicodeError):raise GatewayError(400,'invalid_JSON') from None
                    value,image_count=transform(value,gateway.encode,anthropic=self.path.startswith('/v1/messages'))
                    body=json.dumps(value,separators=(',',':'),allow_nan=False).encode()
                conn=http.client.HTTPConnection(gateway.backend.hostname,gateway.backend.port,timeout=3000)
                headers={'Authorization':'Bearer '+gateway.api_key,'Content-Type':'application/json'}
                for name in ('anthropic-version','anthropic-beta','x-request-id'):
                    if self.headers.get(name):headers[name]=self.headers[name]
                conn.request('POST' if post else 'GET',self.path,body,headers)
                reply=conn.getresponse()
                if image_count and reply.status>=400:
                    reply.read(65536);raise GatewayError(reply.status,'image_engine_http_'+str(reply.status))
                gateway.count('requests');gateway.count('images',image_count)
                self.send_response(reply.status)
                for name in ('Content-Type','Content-Length','x-request-id'):
                    if reply.getheader(name):self.send_header(name,reply.getheader(name))
                self.send_header('Connection','close');self.end_headers();started=True
                while True:
                    chunk=reply.read1(65536)
                    if not chunk:break
                    self.wfile.write(chunk);self.wfile.flush()
            except GatewayError as exc:
                if not started:self.error(exc.status,exc.code)
            except (BrokenPipeError,ConnectionResetError):gateway.count('client_disconnects')
            except (OSError,http.client.HTTPException):
                if not started:self.error(502,'engine_unavailable')
            except (ValueError,TypeError):
                if not started:self.error(400,'invalid_request')
            finally:
                if conn is not None:conn.close()
                gateway.slots.release();self.close_connection=True
    return Handler


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bind',default='0.0.0.0');p.add_argument('--port',type=int,default=8000)
    p.add_argument('--backend',default='http://127.0.0.1:8001');p.add_argument('--encoder',default='http://<encoder-host>:9041')
    p.add_argument('--key-file',type=Path,required=True);p.add_argument('--encoder-key-file',type=Path,required=True)
    args=p.parse_args(argv)
    if args.encoder!='http://<encoder-host>:9041':raise ValueError('encoder endpoint differs from cleared VM<vm-id> binding')
    gateway=Gateway(args.backend,args.encoder,read_key(args.key_file,'VLLM_API_KEY'),read_key(args.encoder_key_file,'VISION_ENCODER_API_KEY'))
    server=ThreadingHTTPServer((args.bind,args.port),handler(gateway));server.daemon_threads=True
    import signal
    def stop(*_):threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        print('VISION_GATEWAY_READY '+json.dumps(dict(bind=args.bind,port=args.port,backend=args.backend,encoder=args.encoder,
            default_thinking='unchanged; no request defaults injected',images_per_request=1,encoder_manifest=MODEL_MANIFEST_SHA)),flush=True)
        server.serve_forever(poll_interval=.2)
    finally:
        server.server_close()
        print('VISION_GATEWAY_STOPPED '+json.dumps(gateway.counts,sort_keys=True),flush=True)
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:report_exception(exc);raise SystemExit(2)
