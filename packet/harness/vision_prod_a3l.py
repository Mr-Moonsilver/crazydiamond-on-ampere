#!/usr/bin/env python3
"""w28 A3-L: frozen main100 at c3 beside w27's five150k agent streams."""
import argparse,asyncio,datetime as dt,hashlib,importlib.util,json,math,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from packet_client_contract import report_exception
from vision_prod_a3 import SELECTION_SHA,QUALITY_SHA
from vision_w27_read import a3_read
BACKGROUND_SHA='cc38173c04458c4bb840431695a625f7a5adc4f219cb486ca6beffc4120c9e13'
APPROVAL_SHA='3b2f66d8'  # Full approval file is pinned by the packet manifest.

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def read(out,solo,quality,selection):
 ids=json.loads(selection.read_text())['ids'];meta=json.loads((out/'a3l-collection.json').read_text());prior=json.loads((solo/'a3-collection.json').read_text())
 def answer(v,r,**kw):return dict(schema='w28-a3l/1',verdict=v,reason=r,approval=APPROVAL_SHA,**kw)
 if sha(selection)!=SELECTION_SHA or len(ids)!=100 or len(set(ids))!=100:return answer('HOLD','frozen-selection-differs')
 if a3_read(solo,quality,selection)['verdict']!='PASS':return answer('HOLD','same-build-solo-A3-not-PASS')
 if any(m.get('selection_sha256')!=SELECTION_SHA or m.get('quality_sha256')!=QUALITY_SHA for m in (meta,prior)):return answer('HOLD','quality-or-selection-binding-differs')
 if (meta.get('request_body_shas')!=prior.get('request_body_shas') or set(meta.get('request_body_shas',{}))!=set(ids)):return answer('HOLD','probe-request-bodies-differ')
 bg=load(out/'bg-load.jsonl') if (out/'bg-load.jsonl').exists() else [];attempts=load(out/'attempt-errors.jsonl') if (out/'attempt-errors.jsonl').exists() else []
 raw=load(out/'results/main.jsonl');scores=load(out/'scores/main.jsonl');before=load(solo/'scores/main.jsonl')
 if attempts or meta.get('background_task_errors') or any(x.get('error') for x in raw+scores+bg):return answer('FAIL','engine-or-stream-errors',errors=len(attempts)+sum(bool(x.get('error')) for x in raw+scores+bg),background_task_errors=meta.get('background_task_errors'))
 if (not meta.get('complete') or meta.get('stats')!=dict(done=100,skipped_cap=0,errors=0) or any(len(rows)!=100 or {r['id'] for r in rows}!=set(ids) for rows in (raw,scores,before))):return answer('HOLD','incomplete-or-duplicate-main100')
 if any(x.get('error') or x.get('skipped') for x in before+raw+scores):return answer('FAIL','request-or-score-errors')
 if any(type(x.get('score')) not in (int,float) or not math.isfinite(x['score']) or not 0<=x['score']<=1 for x in before+scores):return answer('HOLD','invalid-score')
 if (meta.get('background_sha256')!=BACKGROUND_SHA or meta.get('probe_concurrency')!=3 or meta.get('background_streams')!=5 or meta.get('background_cap')!=150000 or meta.get('warm',{}).get('streams_warm')!=5 or meta.get('warm',{}).get('warmup')!='all_first_turns_done'):return answer('HOLD','background-protocol-not-shown')
 complete=[r for r in bg if not r.get('cancelled_at_stop')]
 if {r.get('stream') for r in complete}!=set(range(5)) or any(r.get('status')!=200 or type(r.get('prompt_tokens')) is not int for r in complete):return answer('HOLD','five-background-streams-not-proven')
 begin,end=meta.get('native_before',{}),meta.get('native_after',{})
 if type(begin.get('engine_epoch')) is not int or begin['engine_epoch']<=0:return answer('HOLD','native-epoch-missing')
 if any(x.get('schema')!=1 or x.get('action')!='status' or x.get('native_ready') is not True for x in (begin,end)) or not begin.get('run_id') or (begin.get('run_id'),begin.get('engine_epoch'))!=(end.get('run_id'),end.get('engine_epoch')):return answer('HOLD','native-engine-identity-or-readiness-differs')
 left=sum(x['score'] for x in before)/100;right=sum(x['score'] for x in scores)/100;delta=100*(right-left)
 submits=load(out/'probe-submit.jsonl')
 if len(submits)!=100 or {r['id'] for r in submits}!=set(ids) or any(type(r.get('probe_inflight')) is not int or not 1<=r['probe_inflight']<=3 for r in submits) or max(r['probe_inflight'] for r in submits)!=3:return answer('HOLD','actual-probe-concurrency-not-three')
 wait=[r['ttft_s'] for r in raw if isinstance(r.get('ttft_s'),(int,float))]
 return answer('PASS' if delta>=-5-1e-10 else 'FAIL','paired-loaded-minus-solo-main100',n=100,errors=0,solo=left,loaded=right,delta_pp=delta,
  background_turns=len(complete),background_prompt_tokens={str(i):[r['prompt_tokens'] for r in complete if r['stream']==i] for i in range(5)},
  queueing_report_only=dict(ttft_s=wait,background_inflight_at_probe_submit=[r['background_inflight'] for r in submits]),native_identity=dict(run_id=begin['run_id'],engine_epoch=begin['engine_epoch']))

def collect(a):
 os.umask(0o077)
 if a.out.exists():raise ValueError('fresh A3-L output required')
 for folder,pin in ((a.quality_packet,QUALITY_SHA),(a.background_packet,BACKGROUND_SHA)):
  if sha(folder/'PACKET-SHA256SUMS')!=pin:raise ValueError('frozen-quality-packet-differs')
  subprocess.run(['sha256sum','--strict','--quiet','-c','PACKET-SHA256SUMS'],cwd=folder,check=True)
 if sha(a.selection)!=SELECTION_SHA or a.key_file.is_symlink() or a.key_file.stat().st_mode&0o777!=0o600:raise ValueError('selection-or-private-key-mode-differs')
 if a3_read(a.solo,a.quality_packet,a.selection)['verdict']!='PASS':raise ValueError('same-build-solo-A3-not-PASS')
 old=module(a.quality_packet/'code/runner.py','a3l_w26');driver=module(a.background_packet/'code/runner.py','a3l_w27')
 ids=json.loads(a.selection.read_text())['ids'];items=old.load_items(str(a.quality_packet));todo=[items[i] for i in ids]
 if len(ids)!=100 or len(set(ids))!=100 or any(it.get('thinking') for it in todo):raise ValueError('frozen100-thinking-off-required')
 rec=driver.Rec(str(a.out));meta=dict(selection_sha256=SELECTION_SHA,quality_sha256=QUALITY_SHA,background_sha256=BACKGROUND_SHA,
  probe_concurrency=3,background_streams=5,background_cap=150000,complete=False,stats={},background_task_errors=[],started_wall_s=time.time())
 driver.BG_N['n']=5;deadline=dt.datetime.fromisoformat(a.deadline).timestamp()
 def save():(a.out/'a3l-collection.json').write_text(json.dumps(meta,indent=2)+'\n')
 class Tracked(driver.Client):
  async def chat_once(self,body,timeout_s):
   value=await super().chat_once(body,timeout_s)
   if value.get('error'):
    rec.append('attempt-errors.jsonl',dict(error=value['error'],status=value.get('status'),wall_s=time.time()));report_exception(RuntimeError('A3-L attempt '+str(value['error'])))
   return value
  async def chat(self,it,phase,max_tokens=None):
   if phase=='bg-load':active.add(it['id'])
   else:
    probes.add(it['id']);rec.append('probe-submit.jsonl',dict(id=it['id'],wall_s=time.time(),probe_inflight=len(probes),background_inflight=len(active),gauges=dict(self.last_metrics)))
   try:return await super().chat(it,phase,max_tokens)
   finally:active.discard(it['id']);probes.discard(it['id'])
 active=set();probes=set()
 async def run():
  client=Tracked(a.base_url,a.model,driver.read_key(str(a.key_file)),rec);client.hard_end=deadline
  bg=driver.Background(client,rec,str(a.background_packet),'load',150000);stop=asyncio.Event();metrics=None
  async def native():
   response=await client.http.post(client.base+'/v1/flash_s5_native_overlap',json=dict(schema=1,action='status'),headers={'Authorization':'Bearer '+client.key},timeout=60);response.raise_for_status();value=response.json()
   if value.get('schema')!=1 or value.get('action')!='status' or value.get('native_ready') is not True:raise ValueError('production-native-ready-not-proven')
   return {k:value.get(k) for k in ('schema','action','native_ready','run_id','engine_epoch')}
  try:
   meta['request_body_shas']={it['id']:hashlib.sha256(json.dumps(client.body(it),sort_keys=True).encode()).hexdigest() for it in todo}
   for it in todo:
    if client.body(it)!=old.Client.body(client,it):raise ValueError('w26-request-shape-differs')
   prior=json.loads((a.solo/'a3-collection.json').read_text())
   if meta['request_body_shas']!=prior.get('request_body_shas'):raise ValueError('loaded-vs-solo-request-bodies-differ')
   meta['native_before']=await native();save();metrics=asyncio.create_task(driver.metrics_loop(client,rec,stop,every=1))
   async with asyncio.timeout(max(.001,deadline-time.time())):
    meta['warm']=await bg.start();save()
    if meta['warm']['streams_warm']!=5 or meta['warm']['warmup']!='all_first_turns_done':raise ValueError('five-stream-warmup-incomplete')
    meta['stats']=await driver.run_items(client,rec,'main',todo,3,deadline,200000)
   meta['native_after']=await native();meta['complete']=meta['stats']==dict(done=100,skipped_cap=0,errors=0)
  except Exception as exc:meta['exception']=report_exception(exc)
  finally:
   await bg.finish()
   for task in bg.tasks:
    if not task.cancelled() and task.done() and task.exception():meta['background_task_errors'].append(report_exception(task.exception()))
   stop.set()
   if metrics:await metrics
   await client.http.aclose();meta['finished_wall_s']=time.time();save();rec.logf.close()
 asyncio.run(run())
 if meta['stats'].get('done')==100:
  with (a.out/'score.stdout.log').open('x') as stdout,(a.out/'score.stderr.log').open('x') as stderr:
   score=subprocess.run([sys.executable,'-I',str(a.quality_packet/'code/score.py'),'--packet',str(a.quality_packet),'--out',str(a.out)],stdout=stdout,stderr=stderr,timeout=600)
  if score.returncode:raise RuntimeError('frozen scorer nonzero; retained score.stderr.log')
 if not (a.out/'results/main.jsonl').exists():(a.out/'results/main.jsonl').touch()
 (a.out/'scores').mkdir(exist_ok=True)
 if not (a.out/'scores/main.jsonl').exists():(a.out/'scores/main.jsonl').touch()
 return read(a.out,a.solo,a.quality_packet,a.selection)

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('collect','read'));p.add_argument('--out',type=Path,required=True);p.add_argument('--solo',type=Path,required=True);p.add_argument('--quality-packet',type=Path,required=True);p.add_argument('--selection',type=Path,required=True)
 p.add_argument('--background-packet',type=Path);p.add_argument('--base-url');p.add_argument('--key-file',type=Path);p.add_argument('--model',default='qwen38-w8a8-tp4');p.add_argument('--deadline');a=p.parse_args(argv)
 if a.action=='collect' and any(getattr(a,k) is None for k in ('background_packet','base_url','key_file','deadline')):raise ValueError('collect argv incomplete')
 value=collect(a) if a.action=='collect' else read(a.out,a.solo,a.quality_packet,a.selection)
 (a.out/'a3l-read.json').write_text(json.dumps(value,indent=2)+'\n');print('W28_A3L '+value['verdict']+' '+value['reason']);return dict(PASS=0,HOLD=2,FAIL=1)[value['verdict']]
if __name__=='__main__':
 try:raise SystemExit(main())
 except Exception as exc:report_exception(exc);print('W28_A3L HOLD incomplete-or-exception');raise SystemExit(2)
