#!/usr/bin/env python3
"""Freeze one-shotVM serial ceiling trial; no VM action or GPU use by this builder."""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
from build_vision_encoder_packet import build,red_png
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.out)
 for name in ('vision_encoder_ceiling.py','vision_public_gateway.py','vision_answer_probe.py','packet_client_contract.py','test_vision_encoder_ceiling.py','vision_encoder_ceiling_read.py'):
  shutil.copy2(ROOT/'harness'/name,a.out/'harness'/name)
 shutil.copy2(ROOT/'tools/build_vision_ceiling.py',a.out/'tools/build_vision_ceiling.py')
 from PIL import Image,ImageDraw
 pages=a.out/'pages';pages.mkdir()
 for i in range(1,9):
  image=Image.new('RGB',(1536,2176),'white');draw=ImageDraw.Draw(image)
  for row in range(24):
   y=120+row*72;draw.rectangle((100,y,1436,y+32),fill=(20+i*10,40+row*3,80));draw.text((110,y+8),'Synthetic <caller-b>-size page '+str(i)+' row '+str(row),fill='white')
  image.save(pages/('page-'+str(i)+'.png'))
 terms=Path('<hub-dir>/agents/infra/q3090/w27-readonly/CLEARANCE-TERMS-encoder-ceiling-v<vm-id>-20261117-c.md')
 if sha(terms)!='1d70ca6a513e2dd8691b483a20ff69cd753b21a79283cea4a1e104d9e3d4573a':raise ValueError('ceiling-clearance-terms-differ')
 shutil.copy2(terms,a.out/'CLEARANCE-TERMS.md')
 (a.out/'ceiling-plan.json').write_text(json.dumps(dict(schema='vision-encoder-ceiling/1',run_id='v<vm-id>-20261117-c',n=[1,2,4,6,8],gpu_batching=False,raw_body_limit_bytes=8*2**20,gateway_body_limit_bytes=16*2**20,max_rows_per_image=1024,page_pixels=[1536,2176],card_total_limit_mib=22528,gpu_owned_limit_mib=3072,allocator_limit_mib=2048,asr_calls=7,asr_step_spacing_s=120,asr_bound='min(10,max(2*baseline,3))',remote_speech_wait_s=300,code_bound_images=8,consumer='not run; gateway transform ends at trusted embedding construction',source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()),indent=2)+'\n')
 vm='/opt/flashnext/vision-ceiling-20261117c'
 argv=['python3','-B',vm+'/harness/vision_encoder_job.py','--model','/opt/flashnext/models/flash-next-vision-8b333e23','--run-dir','/opt/flashnext/runs/vision-v<vm-id>-20261117-c','--run-id','v<vm-id>-20261117-c','--clearance','/opt/flashnext/clearances/v<vm-id>-20261117-c.json','--key-file','/opt/flashnext/keys/vision-encoder.env','--expected-asr',vm+'/asr-expected.txt','--asr-url','http://<encoder-host>:9030/asr?language=en&output_format=json','--asr-max-seconds','10','--speech-health','http://<encoder-host>:9030/health','--fixture','<hub-dir>/scratch/infra/asr-fixture/asr10.wav','--images',*[vm+'/pages/page-'+str(i)+'.png' for i in range(1,9)],'--ceiling']
 (a.out/'ceiling-owner-argv.json').write_text(json.dumps(argv,indent=2)+'\n')
 import shlex
 script='#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(vm)+'\nsha256sum --strict --quiet -c PACKET-SHA256SUMS\n'
 script+='exec timeout --foreground --signal=TERM --kill-after=60s 2520s '+shlex.join(argv)+'\n'
 (a.out/'owner-run.sh').write_text(script);(a.out/'owner-run.sh').chmod(0o755)
 names=sorted(p.relative_to(a.out).as_posix() for p in a.out.rglob('*') if p.is_file() and p.name!='PACKET-SHA256SUMS')
 (a.out/'PACKET-SHA256SUMS').write_text(''.join(sha(a.out/n)+'  '+n+'\n' for n in names));print('VISION_CEILING_PACKET '+str(a.out)+' '+sha(a.out/'PACKET-SHA256SUMS'))
if __name__=='__main__':main()
