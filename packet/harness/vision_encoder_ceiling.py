"""One-shot serial N-image gateway sweep; called only by the infra-cleared GPU owner."""
import base64,concurrent.futures,ipaddress,json,os,time,threading
from packet_client_contract import report_exception
from pathlib import Path
from vision_speech_guard import Refused,SpeechLatency
from vision_public_gateway import Gateway,transform
CANCEL=threading.Event()
LATEST={};PENDING=None;CARD=dict(used_peak_mib=0,free_minimum_mib=24576,outside_peak_mib=0)

def preflight(clear):
 global LATEST,PENDING,CARD
 CANCEL.clear();LATEST=dict(steps=[],complete=False);PENDING=None;CARD=dict(used_peak_mib=0,free_minimum_mib=24576,outside_peak_mib=0)
 pids=clear.get('speech_gpu_pids')
 if not isinstance(pids,list) or not pids or any(type(p)is not int or p<2 for p in pids):raise Refused('ceiling-clearance-needs-current-speech-GPU-PIDs')
 if os.path.lexists('/opt/<gpu-tenant>/local/gpu-wanted'):raise Refused('<gpu-tenant>-gate-exists; no flag contents read or written')

def snapshot():return dict(LATEST,card=dict(CARD))
def cancel():CANCEL.set()
def check_cancel():
 if CANCEL.is_set():raise Refused('ceiling-owned-stop-requested')
def remote_speech(command):
 text=command(['ss','-tnH','state','established'])
 peers=[]
 for line in text.splitlines():
  words=line.split()
  for i,word in enumerate(words[:-1]):
   if not word.endswith(':9030'):continue
   host=words[i+1].rsplit(':',1)[0].strip('[]')
   try:
    ip=ipaddress.ip_address(host);ip=getattr(ip,'ipv4_mapped',None) or ip
   except ValueError:raise Refused('unreadable-speech-connection-peer') from None
   if str(ip) not in ('127.0.0.1','::1','<encoder-host>'):peers.append(str(ip))
 return len(peers)

def card(clear,pid,command):
 text=command(['nvidia-smi','-i',clear['gpu_uuid'],'--query-gpu=memory.used,memory.free','--format=csv,noheader,nounits'],timeout=3)
 used,free=[int(x.strip()) for x in text.split(',')]
 text=command(['nvidia-smi','--query-compute-apps=pid,used_memory,gpu_uuid','--format=csv,noheader,nounits'],timeout=3)
 apps={}
 for line in text.splitlines():
  fields=[x.strip() for x in line.split(',')]
  if len(fields)!=3:raise Refused('ceiling-GPU-process-census-unreadable')
  if fields[2]==clear['gpu_uuid']:apps[int(fields[0])]=int(fields[1])
 speech=set(clear['speech_gpu_pids'])
 if not speech<=set(apps):raise Refused('speech-GPU-PID-disappeared; stop for new clearance')
 process_ids={int(x) for x in command(['sudo','-n','docker','top','speech-stack','-eo','pid']).split() if x.isdigit()}
 if (set(apps)&process_ids)!=speech:raise Refused('speech-GPU-PID-set-changed; stop for new clearance')
 outside=sum(mib for p,mib in apps.items() if p not in speech and p!=pid)
 CARD.update(used_peak_mib=max(CARD['used_peak_mib'],used),free_minimum_mib=min(CARD['free_minimum_mib'],free),outside_peak_mib=max(CARD['outside_peak_mib'],outside))
 if used>22528 or outside>500:raise Refused('ceiling-card-total-or-outside-speech-gate')
 return dict(used_mib=used,free_mib=free,outside_mib=outside)

def sweep(url,key,images,baseline,asr,command,*,clock=time.monotonic,sleep=time.sleep,measure=None):
 global LATEST,PENDING
 bound=min(10,max(2*baseline['elapsed_s'],3.0));steps=[];previous=None
 pool=concurrent.futures.ThreadPoolExecutor(max_workers=2)
 try:
  for n in (1,2,4,6,8):
   check_cancel()
   if previous is not None:
    while clock()-previous<120:
     check_cancel();sleep(min(.25,120-(clock()-previous)))
   waited=clock();busy=remote_speech(command)
   while busy and clock()-waited<300:
    check_cancel();sleep(.25);busy=remote_speech(command)
   if busy:
    steps.append(dict(n=n,status='SKIPPED',reason='live-<caller-c>-after-five-minutes',wait_s=clock()-waited));LATEST=dict(steps=list(steps),complete=False,bound_s=bound);continue
   body=dict(model='qwen38-w8a8-tp4',messages=[dict(role='user',content=[dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(Path(p).read_bytes()).decode())) for p in images[:n]])])
   size=len(json.dumps(body,separators=(',',':')).encode())
   if size>8*2**20:raise Refused('ceiling-logical-request-body-exceeds8MiB')
   gateway=Gateway('http://127.0.0.1:8001',url,'a'*64,key,max_images=8)
   details=[];encode=gateway.encode
   def one(raw):
    before=measure() if measure else None
    result=encode(raw)
    after=measure() if measure else None
    details.append(dict(shape=result['shape'],metrics=result['metrics'],memory_before=before,memory_after=after));return result
   previous=clock();started=time.time_ns();PENDING=pool.submit(asr);future=pool.submit(transform,body,one,False,8)
   try:
    while not future.done() or not PENDING.done():
     check_cancel()
     if not PENDING.done() and clock()-previous>bound:raise SpeechLatency('ceiling-ASR-latency-bound-while-pending')
     if future.done():future.result()
     if PENDING.done():PENDING.result()
     sleep(.05)
    result,count=future.result();speech=PENDING.result()
    if not speech.get('correct') or speech['elapsed_s']>bound:raise SpeechLatency('ceiling-ASR-MATCH-or-latency-bound')
    if count!=n or len(details)!=n or any(not 0<r['shape'][0]<=1024 for r in details):raise Refused('ceiling-image-count-or-row-bound')
    steps.append(dict(n=n,status='PASS',request_bytes=size,rows=[r['shape'][0] for r in details],per_image=details,wall_s=clock()-previous,started_wall_ns=started,started_monotonic=previous,finished_monotonic=clock(),asr=speech,wait_s=previous-waited,card=dict(CARD)))
    LATEST=dict(steps=list(steps),complete=False,bound_s=bound)
   except Exception as exc:
    steps.append(dict(n=n,status='FAIL',request_bytes=size,wall_s=clock()-previous,error_type=type(exc).__name__,reason=str(exc)[:200],card=dict(CARD)))
    LATEST=dict(steps=list(steps),complete=False,bound_s=bound);raise
  LATEST=dict(steps=list(steps),complete=len(steps)==5 and all(x['status']=='PASS' for x in steps),bound_s=bound,ceiling=max((x['n'] for x in steps if x['status']=='PASS'),default=0))
  return LATEST
 finally:pool.shutdown(wait=False,cancel_futures=True)

def post_stop(asr,baseline):
 if PENDING is not None and not PENDING.done():
  try:PENDING.result(timeout=65)
  except Exception as exc:report_exception(exc)  # Original step failure remains recorded; GPU is already OFF.
 result=asr();bound=min(10,max(2*baseline['elapsed_s'],3.0))
 if not result.get('correct') or result['elapsed_s']>bound:raise SpeechLatency('ceiling-post-stop-ASR-bound')
 return result
