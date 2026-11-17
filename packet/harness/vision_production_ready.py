#!/usr/bin/env python3
"""Guard public traffic until the frozen native lane is selected after every engine start."""
import argparse,hashlib,json,os,subprocess,sys,time,urllib.request
from pathlib import Path
from urllib.parse import urlsplit
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vision_answer_probe import key
from packet_client_contract import report_exception

def write_state(path,value):
 path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 value=dict(value,wall_ns=time.time_ns())
 with (path.parent/'transitions.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
 pending=path.with_suffix('.next.json');pending.write_text(json.dumps(value)+'\n');pending.replace(path)

def status(url,k):
 # Unbound discovery is read-only and intentionally omits run_id; Session.execute permits it.
 request=urllib.request.Request(url+'/v1/flash_s5_native_overlap',json.dumps(dict(schema=1,action='status')).encode(),{'Content-Type':'application/json','Authorization':'Bearer '+k})
 with urllib.request.urlopen(request,timeout=60) as response:value=json.loads(response.read(16*2**20))
 if value.get('schema')!=1 or value.get('action')!='status':raise ValueError('native status shape differs')
 return value

def released(value):
 auto=value.get('auto_arm') or {}
 return (value.get('native_ready') is True and isinstance(value.get('run_id'),str) and type(value.get('engine_epoch')) is int and value['engine_epoch']>0
         and auto.get('manual_owned') is False and type(auto.get('selection_seq')) is int and auto['selection_seq']>0)

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',type=Path,required=True);p.add_argument('--state-file',type=Path,required=True)
 p.add_argument('--key-file',type=Path,default=Path('/root/.flashnext-serve-vision-prod-w27-20261117a/vllm-api.env'));p.add_argument('--url',default='http://127.0.0.1:8001')
 p.add_argument('--invalidate-only',action='store_true');a=p.parse_args(argv);os.umask(0o077)
 u=urlsplit(a.url)
 if u.scheme!='http' or u.hostname!='127.0.0.1' or not u.port or u.path or u.query or u.fragment or u.username:raise ValueError('private loopback backend URL required')
 packet_sha=hashlib.sha256((a.packet/'PACKET-SHA256SUMS').read_bytes()).hexdigest()
 if a.invalidate_only:write_state(a.state_file,dict(ready=False,packet_sha256=packet_sha,reason='owned-engine-start-or-stop'));print('VISION_PRODUCTION_READY INVALIDATED');return 0
 for folder,name in ((a.packet,'PACKET-SHA256SUMS'),(a.packet/'harness','SHA256SUMS'),(a.packet/'harness/native30_overlay','SHA256SUMS')):
  subprocess.run(['sha256sum','--strict','--quiet','-c',name],cwd=folder,check=True)
 k=key(a.key_file,'VLLM_API_KEY');env=dict(os.environ,VLLM_API_KEY=k,PYTHONDONTWRITEBYTECODE='1')
 end=time.monotonic()+900
 while True:
  try:
   with urllib.request.urlopen(a.url+'/health',timeout=3) as response:
    if response.status==200:break
  except OSError:
   if time.monotonic()>=end:raise TimeoutError('private engine health not ready')
   time.sleep(2)
 value=status(a.url,k)
 if not released(value):
  write_state(a.state_file,dict(ready=False,packet_sha256=packet_sha,reason='native-lane-not-selected'))
  if value.get('run_id') is not None:raise ValueError('bound native lane not released; refuse to reconfigure a live cohort')
  run_id='vision-w27-boot-'+str(time.time_ns())+'-'+str(os.getpid());out=a.state_file.parent/'runs'/run_id;out.mkdir(parents=True,mode=0o700)
  def run(name,args,limit):
   command=[sys.executable,'-B',str(a.packet/'serve_lane.py'),*args]
   (out/(name+'.argv.json')).write_text(json.dumps(command)+'\n')
   with (out/(name+'.stdout.log')).open('x') as stdout,(out/(name+'.stderr.log')).open('x') as stderr:
    result=subprocess.run(command,env=env,stdout=stdout,stderr=stderr,timeout=limit)
   if result.returncode:raise ValueError('native bootstrap '+name+' refused; see retained private client streams')
  h=str(a.packet/'harness');overlay=str(Path('/data/vllm/flash-next-tp4/prod/overlay-v16s5-vision-prod-w27-20261117a'))
  run('startup',['cx-step','--h',h,'--','e3_split_startup_grade.py','--url',a.url,'--run-id',run_id,'--package',overlay,'--out',str(out/'startup-grade.json')],600)
  grade=json.loads((out/'startup-grade.json').read_text());epoch=grade.get('engine_epoch')
  if grade.get('verdict')!='PASS' or type(epoch)is not int or epoch<=0:raise ValueError('native restart startup grade not PASS')
  run('full-idle',['cx-step','--h',h,'--','e3_native39_trunk_checks.py','full-idle','--url',a.url,'--run-id',run_id,'--epoch',str(epoch),'--control-timeout-s','60','--timeout-s','120','--out',str(out/'full-idle.json')],450)
  run('select',['select','--h',h,'--url',a.url,'--run-id',run_id,'--epoch',str(epoch),'--out',str(out/'lane-select.json')],300)
  value=status(a.url,k)
  if not released(value) or value['run_id']!=run_id or value['engine_epoch']!=epoch:raise ValueError('native restart selection/release not proven')
 write_state(a.state_file,dict(ready=True,packet_sha256=packet_sha,engine_epoch=value['engine_epoch'],run_id=value['run_id']))
 print('VISION_PRODUCTION_READY PASS selected-and-released');return 0
if __name__=='__main__':
 try:raise SystemExit(main())
 except Exception as exc:report_exception(exc);print('VISION_PRODUCTION_READY HOLD unsafe-or-incomplete-lane');raise SystemExit(2)
