#!/usr/bin/env python3
"""Freeze the manually admitted VM<vm-id> encoder service; no install or GPU action."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
from build_vision_encoder_packet import build,FILES
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 build(a.out)
 for source in ('harness/vision_service_owner.py','harness/packet_client_contract.py','tools/build_vision_service.py'):
  dest=a.out/source;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/source,dest)
 units=a.out/'units';units.mkdir()
 for name in ('flashnext-vision-encoder.service','journald-vision.conf','ENCODER-ALERT.txt'):shutil.copy2(ROOT/'deploy/vision-production'/name,units/name)
 value=json.loads((a.out/'delivery.json').read_text());value.update(source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),scope='manual-service candidate; tower/backend already measured1468MiB in trial-a/b; new long-lived supervisor CPU-only',gpu_cleared=False,service_cleared=False,service_admission='fresh infra service_cleared:true clearance, manual/no automatic restart',alert_hook='/usr/local/sbin/flashnext-encoder-alert',total_log_budget_mib=832,measured_trial_result_sha256='510acd86c00aa8db7524b1df5588bfc827109201088637f490b0badf65ade343')
 (a.out/'delivery.json').write_text(json.dumps(value,indent=2)+'\n')
 files=sorted(p.relative_to(a.out).as_posix() for p in a.out.rglob('*') if p.is_file() and p.name!='PACKET-SHA256SUMS')
 (a.out/'PACKET-SHA256SUMS').write_text(''.join(sha(a.out/n)+'  '+n+'\n' for n in files))
 print('VISION_SERVICE_PACKET '+str(a.out)+' '+sha(a.out/'PACKET-SHA256SUMS'))
if __name__=='__main__':main()
