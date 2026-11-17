#!/usr/bin/env python3
"""Rebind the frozen fit-b workload to the byte-identical production overlay; no device action."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
from build_vision_serve import sums,repin_params,once
FIT=Path('<hub-dir>/cargo-target/flash-next/vision-fit-8x200k-b-20261117/build-3')
FIT_SHA='fba0c197afd643b94028fcfdd171c03aae5106c3d02595d3e17c1a88bd583c0c'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--production',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if sha(FIT/'PACKET-SHA256SUMS')!=FIT_SHA:raise ValueError('fit-b-parent-differs')
 for root in (FIT,a.production):subprocess.run(['sha256sum','--strict','--quiet','-c','PACKET-SHA256SUMS'],cwd=root,check=True)
 a.out.mkdir(exist_ok=False)
 for source in FIT.iterdir():
  if source.name=='harness':continue
  target=a.out/source.name
  if source.is_dir():shutil.copytree(source,target)
  else:shutil.copy2(source,target)
 shutil.copytree(a.production/'harness',a.out/'harness')
 for path in a.out.rglob('*'):
  if path.is_file():path.chmod(path.stat().st_mode|0o200)
 for name in ('serve-launch.sh','serve_readback.py','serve-params.env'):
  shutil.copy2(a.production/name,a.out/name)
 launch=a.out/'serve-launch.sh';text=launch.read_text();launch.write_text(once(text,'SERVE_SETTINGS+=("$kv FLASH_S5_NATIVE_MM=1")','SERVE_SETTINGS+=("$kv FLASH_S5_NATIVE_MM=1 FLASH_S5_FIT_MEMORY=e6")'))
 readback=a.out/'serve_readback.py';readback.write_text(once(readback.read_text(),"'FLASH_S5_NATIVE_MM=1', 'FLASH_S5_OVERLAP=1'","'FLASH_S5_FIT_MEMORY=e6', 'FLASH_S5_NATIVE_MM=1', 'FLASH_S5_OVERLAP=1'"))
 env=json.loads((a.out/'vision-container-env.json').read_text());env['FLASH_S5_FIT_MEMORY']='e6';(a.out/'vision-container-env.json').write_text(json.dumps(env,indent=2)+'\n')
 repin_params(a.out/'serve-params.env',dict(SERVE_TAG='vision-fit-prod-w27-20261117b'))
 for name in ('owner.sh','seq.sh'):
  path=a.out/name;path.write_text(once(path.read_text(),'set -euo pipefail','set -euo pipefail\nexport SERVE_PORT=${SERVE_PORT:-8001}\nexport PROBE_URL=${PROBE_URL:-http://127.0.0.1:$SERVE_PORT}'))
 overlay=json.loads((a.production/'vision-delta.json').read_text())['overlay_name']
 delta=json.loads((a.out/'vision-delta.json').read_text());delta['overlay_name']=overlay;(a.out/'vision-delta.json').write_text(json.dumps(delta,indent=2)+'\n')
 preserved=('fit-plan.json','protocol-b.md','vision-fit-rule.md','vision-fit-inputs.json')
 for name in preserved:
  if sha(a.out/name)!=sha(FIT/name):raise ValueError('registered-fit-input-moved:'+name)
 for name in ('vision_fit_8x200k.py','fit_8x200k.py'):
  if sha(a.out/'harness'/name)!=sha(FIT/'harness'/name):raise ValueError('registered-fit-client-moved:'+name)
 if sha(a.out/'harness/native30_overlay/SHA256SUMS')!=sha(a.production/'harness/native30_overlay/SHA256SUMS'):raise ValueError('not-the-same-production-overlay')
 declaration=dict(schema='w27-fit-production-overlay/1',frozen_fit_parent_sha256=FIT_SHA,production_packet_sha256=sha(a.production/'PACKET-SHA256SUMS'),overlay_sha256=sha(a.production/'harness/native30_overlay/SHA256SUMS'),harness_sha256=sha(a.production/'harness/SHA256SUMS'),fit_workload_unchanged=True,preserved={n:sha(a.out/n) for n in preserved},observer='e6 on fit only; production default off',private_transport='127.0.0.1:8001',production_gateway='not started in fit-b; actual raw-image route tested by G5')
 (a.out/'production-overlay-binding.json').write_text(json.dumps(declaration,indent=2)+'\n')
 names=sorted(p.relative_to(a.out).as_posix() for p in a.out.iterdir() if p.is_file() and p.name!='PACKET-SHA256SUMS')
 names+=['harness/SHA256SUMS']+[p.relative_to(a.out).as_posix() for sub in ('tools','vision-assets','vision-fit-assets','patches') for p in (a.out/sub).rglob('*') if p.is_file()]
 (a.out/'PACKET-SHA256SUMS').write_text(''.join(sha(a.out/n)+'  '+n+'\n' for n in sorted(names)))
 print('VISION_FIT_PRODUCTION_PACKET '+str(a.out)+' '+sha(a.out/'PACKET-SHA256SUMS'))
if __name__=='__main__':main()
