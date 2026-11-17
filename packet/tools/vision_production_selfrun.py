#!/usr/bin/env python3
"""Replay frozen production owner argv against existing CPU-only systemd/Docker/NVML/HTTP leaves."""
import argparse,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
import vision_serve_selfrun as old
ROOT=Path(__file__).resolve().parents[1]
TAG='vision-prod-w27-20261117a'
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--g0',type=Path,required=True);p.add_argument('--fetch-b1',type=Path,required=True);p.add_argument('--port',type=int,default=18871);a=p.parse_args(argv)
 a.bundle=a.packet/'harness';out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
 old.TAG=TAG;old.OVERLAY_NAME=json.loads((a.packet/'vision-delta.json').read_text())['overlay_name']
 # Retained, explicit IO leaves. No systemctl, Docker, GPU, SSH or VM call can escape these shims.
 leaves=out/'io-leaves';leaves.mkdir();shutil.copytree(ROOT/'deploy/final-engine-serve/selfrun/shims',leaves/'shims')
 stub=(ROOT/'deploy/final-engine-serve/selfrun/stub_engine.py').read_text()
 stub=stub.replace("ThreadingHTTPServer(('0.0.0.0', a.port), H)","ThreadingHTTPServer((host, a.port), H)")
 (leaves/'stub_engine.py').write_text(stub);old.SELF=leaves
 PK=out/'packet';shutil.copytree(a.packet,PK);SR=out/'shimstate';SR.mkdir();R,SR=old.layout(out,a)
 port=old.free_port(a.port);hub=out/'hub'/('serve-'+TAG)
 env=dict(os.environ,PATH=str(leaves/'shims')+':'+os.environ['PATH'],R=str(R),SR=str(SR),STUB_ENGINE=str(leaves/'stub_engine.py'),STUB_IMAGE=old.IMAGE,
  STUB_RECORDED=str(a.fetch_b1/'startup-grade.json'),STUB_RECORDED_ARM=str(a.fetch_b1/'arm-n3.g2-live-0.json'),STUB_OVERLAY=str(a.bundle/'native30_overlay'),SERVE_PORT=str(port),
  SERVE_OWNER_UID=str(os.getuid()),SERVE_OWNER=os.environ.get('USER','<service-user>'),HEALTH_DEADLINE_S='90',PVE=str(leaves/'shims/pve'),HUBD=str(hub),STUB_IMAGE_ENV=str(SR/'image-env.json'),PYTHONDONTWRITEBYTECODE='1')
 recorded=json.loads((a.fetch_b1/'startup-grade.json').read_text())
 def mm(value):
  if isinstance(value,dict):
   inv=value.get('native_inventory')
   if isinstance(inv,dict):
    rank=inv['worker_rank'];inv['native_mm']=dict(schema='native-mm-inventory/1',tower_registered_cuda_bytes=0,gpu_encoder_cache_present=rank<2,cache_budget_tokens=8192,cache_budget_bytes=40*2**20,cache_feature_bytes=0,cache_storage_bytes=0,encoder_inputs_buffer_bytes=5*2**20 if rank<2 else 0,active_splice_device_bytes=0,basis='CPU schema-valid leaf, not measured memory')
   for v in value.values():mm(v)
  elif isinstance(value,list):
   for v in value:mm(v)
 mm(recorded);fixture=SR/'recorded-mm.json';fixture.write_text(json.dumps(recorded));env['STUB_RECORDED']=str(fixture)
 run=old.Run(out,env,SR);eng=R/('data/serve-'+TAG+'-engine')
 run.step('frozen-packet',['bash','-c','cd "$1" && sha256sum --strict -c PACKET-SHA256SUMS','verify',str(PK)],r'production-delta.json: OK',timeout=90)
 run.step('mint-test-key',['bash',str(PK/'serve-mint-key.sh')],r'^SERVE_KEY_MINTED len=64 ')
 (SR/'systemd/vllm-qwen38.active').write_text('active\n')
 run.step('production-up-refusal',['bash',str(PK/'serve-launch.sh'),'engine'],r'^SERVE_LAUNCH FAILED at units$',rc_ok=(1,),extra_env=dict(OUT=str(R/'data/refused-production-up')))
 (SR/'systemd/vllm-qwen38.active').write_text('inactive\n')
 run.step('private-key-refusal',['bash',str(PK/'serve-launch.sh'),'engine'],r'^SERVE_LAUNCH FAILED at auth$',rc_ok=(1,),extra_env=dict(OUT=str(R/'data/refused-missing-key'),AUTH_ENV=str(R/'root/absent.env')))
 try:
  run.step('launch-to-SERVE_UP',['bash',str(PK/'serve-launch.sh'),'engine'],r'^SERVE_UP .*arm=engine .*auth key .*SERVE_SMOKE PASS$',timeout=300)
  if (eng/'readback.json').exists():
   d=json.loads((eng/'readback.json').read_text());assert d['auth']=='key' and d['noauth_probe']['code']==401
  else:raise ValueError('no frozen launch readback')
  key=(hub/'vllm-api.env').read_text().split('=',1)[1].strip();url='http://127.0.0.1:'+str(port)
  assert old.http(url+'/v1/models')[0]==401
  assert old.http(url+'/v1/chat/completions',key,dict(model='qwen38-w8a8-tp4',messages=[dict(role='user',content='hi')],max_tokens=4))[0]==200
 finally:
  if (eng/'pids.env').exists():run.step('owned-stop',['bash',str(PK/'serve-stop.sh'),str(eng)],r'^SERVE_STOP_STATE READY ',timeout=180)
 verdict='PASS' if all(s['ok'] for s in run.steps) else 'FAIL'
 (out/'selfrun.json').write_text(json.dumps(dict(verdict=verdict,scope='actual frozen owner argv; CPU-only fake engine/GPU memory, not device qualification',steps=run.steps),indent=2)+'\n')
 print('VISION_PRODUCTION_SELFRUN '+verdict+' '+str(sum(s['ok'] for s in run.steps))+'/'+str(len(run.steps)));return 0 if verdict=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
