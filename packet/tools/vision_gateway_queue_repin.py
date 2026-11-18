#!/usr/bin/env python3
"""Owner-only pin metadata update after all gates; preserve every previous blob/manifest."""
import argparse,hashlib,json,os,shutil,stat,sys,time,traceback
from pathlib import Path
PATCH='/data/flash-next-packets/gateway-queue-w28-20261118-build-7'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def update(root):
 root=Path(root);pin=root/'root/flashnext-prod-pins/vision-w28';manifest=pin/'manifest.json'
 value=json.loads(manifest.read_text());assert value['schema']=='vision-production-pins/1'
 mappings={'/etc/systemd/system/flashnext-vision-gateway.service':'flashnext-vision-gateway.service','/etc/default/flashnext-vision-gateway':'gateway-defaults.env',PATCH+'/vision_public_gateway.py':'vision_public_gateway.py'}
 stamp=str(time.time_ns());shutil.copy2(manifest,pin/('manifest-before-queue7-'+stamp+'.json'))
 for path,name in mappings.items():
  target=root/path.lstrip('/');source=root/PATCH.lstrip('/')/name
  assert target.is_file() and not target.is_symlink() and sha(target)==sha(source), 'installed gateway differs: '+path
  blob='queue7-'+stamp+'-'+name;shutil.copyfile(target,pin/blob);(pin/blob).chmod(0o600)
  row=dict(path=path,blob=blob,sha256=sha(target),mode=stat.S_IMODE(target.stat().st_mode))
  found=[i for i,r in enumerate(value['files']) if r['path']==path];assert len(found)<=1
  if found:value['files'][found[0]]=row
  else:value['files'].append(row)
 stage=pin/('manifest-queue7-'+stamp+'.next');stage.write_text(json.dumps(value,indent=2)+'\n');stage.chmod(0o600);os.replace(stage,manifest)
 print('GATEWAY_QUEUE_PIN PASS three gateway targets; all engine rows preserved')
def main():
 p=argparse.ArgumentParser();p.add_argument('--all-gates-pass',action='store_true',required=True);p.add_argument('--root',default='/');a=p.parse_args();update(a.root)
if __name__=='__main__':
 try:main()
 except Exception as exc:
  f=traceback.extract_tb(exc.__traceback__)[-1];print(f'PACKET_CLIENT_ERROR {type(exc).__name__}: {exc} at {f.filename}:{f.lineno}',file=sys.stderr);raise SystemExit(2)
