#!/usr/bin/env python3
"""Keyed OpenAI chat proxy: bounded still images -> trusted encoder -> loopback engine."""
import argparse,base64,copy,uuid,hmac,http.client,ipaddress,json,math,re,socket,ssl,sys,threading,time
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
MAX_IMAGES=32
ENGINE_MAX_IMAGES=8
MAX_REQUEST_IMAGE_ROWS=8192
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


def transform(body,encode,anthropic=False,max_images=4,receipt=None):
    if type(max_images)is not int or not 1<=max_images<=MAX_IMAGES:raise ValueError('image limit outside code bound')
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
    if receipt is not None:receipt['image_count']=len(images)
    if len(images)>max_images:raise GatewayError(400,'configured_image_count_limit')
    if len(images)>ENGINE_MAX_IMAGES:raise GatewayError(400,'frozen_engine_image_count_limit_8')
    if not images:return body,0
    total_bytes=total_rows=0
    for target,url in images:
        raw=raw_image(url);total_bytes+=len(raw)
        if total_bytes>MAX_IMAGE:raise GatewayError(413,'aggregate_images_exceed_8MiB')
        result=encode(raw)
        try:
            validate_embedding(result)
            encoded=result['content']['image_embeds']['image_embeds']
            if not isinstance(encoded,str) or len(encoded)>9*2**20:raise ValueError()
            total_rows+=result['shape'][0]
            if total_rows>MAX_REQUEST_IMAGE_ROWS:raise ValueError()
            part=embedding_content(result)
        except (KeyError,TypeError,ValueError):raise GatewayError(502,'encoder_identity_or_shape_invalid') from None
        if receipt is not None:receipt['total_image_rows']=total_rows
        target.clear()
        target.update(dict(type='image',source=dict(type='flash_next_embeds',content=part)) if anthropic else part)
    return changed,len(images)



class Gateway:
    def __init__(self,backend,encoder,api_key,encoder_key,max_images=4,max_inflight=32,queue_timeout_s=3000):
        u=urlsplit(backend)
        if u.scheme!='http' or u.hostname!='127.0.0.1' or u.path not in ('','/') or not u.port:
            raise ValueError('backend must be a declared loopback HTTP port')
        self.backend=u;self.encoder=encoder.rstrip('/');self.api_key=api_key;self.encoder_key=encoder_key
        if type(max_inflight)is not int or not 1<=max_inflight<=32:raise ValueError('inflight limit must be 1..32')
        if type(queue_timeout_s)not in (int,float) or not math.isfinite(queue_timeout_s) or not 0<queue_timeout_s<=3000:raise ValueError('queue timeout must be finite and >0, <=3000 seconds')
        self.max_inflight=max_inflight;self.queue_timeout_s=queue_timeout_s
        self.slots=threading.BoundedSemaphore(max_inflight)
        # Retain the original body/encoder concurrency bound; release it before the model stream waits.
        self.prepare_slots=threading.BoundedSemaphore(min(8,max_inflight));self.lock=threading.Lock()
        self.counts=dict(requests=0,images=0,encoder_errors=0,client_disconnects=0,admission_timeouts=0,prepare_timeouts=0)
        self.receipt_lock=threading.Lock()
        self.ready_file=None;self.packet_sha=None
        if type(max_images)is not int or not 1<=max_images<=MAX_IMAGES:raise ValueError('configured image limit exceeds code bound')
        self.max_images=max_images
    def ready(self):
        if self.ready_file is None:return True
        try:
            value=json.loads(self.ready_file.read_text())
            return value.get('ready') is True and value.get('packet_sha256')==self.packet_sha
        except (OSError,ValueError):return False
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
            self.receipt.update(status='refused',http_status=status)
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
            self.receipt=dict(schema='vision-gateway-count/1',request_id='gw-'+uuid.uuid4().hex,
                caller_alias='unknown',image_count=None,total_image_rows=0,status='incomplete',http_status=None,started_wall_ns=time.time_ns())
            try:return self.forward(post)
            finally:
                if post:
                    self.receipt['finished_wall_ns']=time.time_ns()
                    with gateway.receipt_lock:print('flash-s5 gateway-request '+json.dumps(self.receipt,sort_keys=True,separators=(',',':')),flush=True)
        def forward(self,post):
            route=urlsplit(self.path).path
            if (post or route not in ('/health','/metrics','/v1/models')) and not self.authenticated():return self.error(401,'Unauthorized')
            if route not in (POST_PATHS if post else GET_PATHS):return self.error(404,'unsupported_endpoint')
            if (post or route=='/health') and not gateway.ready():return self.error(503,'native_lane_not_ready')
            conn=None;started=False;admitted=False;preparing=False
            deadline=time.monotonic()+gateway.queue_timeout_s
            try:
                if post or route not in ('/health','/metrics','/v1/models'):
                    queued=time.monotonic_ns()
                    admitted=gateway.slots.acquire(timeout=max(0,deadline-time.monotonic()))
                    self.receipt['admission_wait_ns']=time.monotonic_ns()-queued
                    if not admitted:
                        gateway.count('admission_timeouts');raise GatewayError(504,'gateway_queue_timeout')
                body=None;image_count=0
                if post:
                    preparing=gateway.prepare_slots.acquire(timeout=max(0,deadline-time.monotonic()))
                    if not preparing:
                        gateway.count('prepare_timeouts');raise GatewayError(504,'gateway_prepare_timeout')
                    # A queued request must not cross an engine/readiness invalidation.
                    if not gateway.ready():raise GatewayError(503,'native_lane_not_ready')
                    if self.headers.get('Transfer-Encoding'):raise GatewayError(400,'chunked_request_body_refused')
                    try:size=int(self.headers.get('Content-Length','-1'))
                    except ValueError:raise GatewayError(400,'invalid_content_length') from None
                    if not 0<size<=MAX_BODY:raise GatewayError(413,'request_body_bound_16MiB')
                    if self.headers.get_content_type()!='application/json':raise GatewayError(415,'JSON_content_type_required')
                    raw=self.rfile.read(size)
                    if len(raw)!=size:raise GatewayError(400,'incomplete_request_body')
                    try:value=json.loads(raw)
                    except (ValueError,UnicodeError):raise GatewayError(400,'invalid_JSON') from None
                    alias=value.get('model') if type(value)is dict else None
                    allowed={'qwen38-w8a8-tp4','cyankiwi/Qwen3.8-27B-AWQ-INT4','cyankiwi/Qwen3.5-122B-A10B-AWQ-4bit','Qwen3.8-Flash-Next','default','qwen3.8-27b'}
                    self.receipt['caller_alias']=alias if isinstance(alias,str) and alias in allowed else 'other'
                    encoder_metrics=dict(wall_ms=0,peak_allocated_bytes=0,peak_reserved_bytes=0)
                    def encode(data):
                        result=gateway.encode(data);metrics=result.get('metrics',{})
                        encoder_metrics['wall_ms']+=metrics.get('wall_ms',0)
                        for field in ('peak_allocated_bytes','peak_reserved_bytes'):encoder_metrics[field]=max(encoder_metrics[field],metrics.get(field,0))
                        return result
                    value,image_count=transform(value,encode,anthropic=route.startswith('/v1/messages'),max_images=gateway.max_images,receipt=self.receipt)
                    body=json.dumps(value,separators=(',',':'),allow_nan=False,ensure_ascii=False).encode()
                    del value,raw
                conn=http.client.HTTPConnection(gateway.backend.hostname,gateway.backend.port,timeout=3000)
                headers={'Authorization':'Bearer '+gateway.api_key,'Content-Type':'application/json'}
                for name in ('anthropic-version','anthropic-beta','x-request-id'):
                    if self.headers.get(name):headers[name]=self.headers[name]
                conn.request('POST' if post else 'GET',self.path,body,headers)
                body=None
                if preparing:gateway.prepare_slots.release();preparing=False
                reply=conn.getresponse()
                self.receipt['http_status']=reply.status
                if image_count and reply.status>=400:
                    reply.read(65536);raise GatewayError(reply.status,'image_engine_http_'+str(reply.status))
                gateway.count('requests');gateway.count('images',image_count)
                self.send_response(reply.status)
                for name in ('Content-Type','Content-Length','x-request-id'):
                    if reply.getheader(name):self.send_header(name,reply.getheader(name))
                if image_count:
                    self.receipt['encoder_metrics']=encoder_metrics
                    for name,value in encoder_metrics.items():self.send_header('x-flashnext-encoder-'+name.replace('_','-'),str(value))
                self.send_header('x-flashnext-request-id',self.receipt['request_id']);self.send_header('Connection','close');self.end_headers();started=True
                while True:
                    chunk=reply.read1(65536)
                    if not chunk:break
                    self.wfile.write(chunk);self.wfile.flush()
                self.receipt['status']='transport-complete'
            except GatewayError as exc:
                if not started:self.error(exc.status,exc.code)
            except (BrokenPipeError,ConnectionResetError):
                self.receipt['status']='client-disconnected';gateway.count('client_disconnects')
            except (OSError,http.client.HTTPException):
                self.receipt['status']='backend-interrupted'
                if not started:self.error(502,'engine_unavailable')
            except (ValueError,TypeError):
                if not started:self.error(400,'invalid_request')
            finally:
                if conn is not None:conn.close()
                if preparing:gateway.prepare_slots.release()
                if admitted:gateway.slots.release()
                self.close_connection=True
    return Handler


class GatewayServer(ThreadingHTTPServer):
    request_queue_size=128
    daemon_threads=True


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bind',default='0.0.0.0');p.add_argument('--port',type=int,default=8000)
    p.add_argument('--backend',default='http://127.0.0.1:8001');p.add_argument('--encoder',default='http://<encoder-host>:9041')
    p.add_argument('--max-images',type=int,choices=range(1,MAX_IMAGES+1),default=4)
    p.add_argument('--max-inflight',type=int,default=32);p.add_argument('--queue-timeout-s',type=float,default=3000)
    p.add_argument('--ready-file',type=Path);p.add_argument('--packet-sha256')
    p.add_argument('--key-file',type=Path,required=True);p.add_argument('--encoder-key-file',type=Path,required=True)
    args=p.parse_args(argv)
    if args.encoder!='http://<encoder-host>:9041':raise ValueError('encoder endpoint differs from cleared VM<vm-id> binding')
    gateway=Gateway(args.backend,args.encoder,read_key(args.key_file,'VLLM_API_KEY'),read_key(args.encoder_key_file,'VISION_ENCODER_API_KEY'),max_images=args.max_images,max_inflight=args.max_inflight,queue_timeout_s=args.queue_timeout_s)
    if (args.ready_file is None)!=(args.packet_sha256 is None):raise ValueError('readiness path and packet SHA required together')
    gateway.ready_file=args.ready_file;gateway.packet_sha=args.packet_sha256
    server=GatewayServer((args.bind,args.port),handler(gateway))
    import signal
    def stop(*_):threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        print('VISION_GATEWAY_READY '+json.dumps(dict(bind=args.bind,port=args.port,backend=args.backend,encoder=args.encoder,
            default_thinking='unchanged; no request defaults injected',images_per_request=args.max_images,max_inflight=args.max_inflight,queue_timeout_s=args.queue_timeout_s,encoder_manifest=MODEL_MANIFEST_SHA)),flush=True)
        server.serve_forever(poll_interval=.2)
    finally:
        server.server_close()
        print('VISION_GATEWAY_STOPPED '+json.dumps(gateway.counts,sort_keys=True),flush=True)
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:report_exception(exc);raise SystemExit(2)
