#!/usr/bin/env python3
"""Extend existing boot pins to the production model/gateway units, launchers and config; retain found files."""
import argparse,hashlib,json,os,shutil,stat,subprocess,sys,time
from pathlib import Path
PK='/data/flash-next-packets/serve-vision-prod-w27-20261117a'
TARGETS=('/etc/systemd/system/vllm-flash.service','/etc/systemd/system/vllm-flash.service.d/90-vision-native-ready.conf',
 '/etc/systemd/system/flashnext-vision-gateway.service',
 PK+'/gateway-start.sh',PK+'/gateway-stop.sh',PK+'/serve-params.env',PK+'/PACKET-SHA256SUMS',
 '/usr/local/sbin/flashnext-boot-guard.sh','/usr/local/sbin/flashnext-vision-pin-guard.py')
DIRS=('/etc/systemd/system/vllm-flash.service.d','/etc/systemd/system/flashnext-vision-gateway.service.d')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rooted(root,p):return Path(root)/p.lstrip('/')
def register(root):
 pin=rooted(root,'/root/flashnext-prod-pins/vision-w27');pin.mkdir(parents=True,exist_ok=False,mode=0o700)
 names=list(TARGETS)
 for directory in DIRS:
  path=rooted(root,directory)
  if any(str(Path(directory)/p.name) not in names for p in path.glob('*.conf')):raise ValueError('unreviewed pre-promotion drop-in:'+directory)
 rows=[]
 for index,name in enumerate(names):
  path=rooted(root,name)
  if not path.is_file() or path.is_symlink():raise ValueError('production pin source missing/symlink:'+name)
  blob=pin/str(index);shutil.copy2(path,blob);blob.chmod(0o600)
  rows.append(dict(path=name,blob=str(index),sha256=sha(blob),mode=stat.S_IMODE(path.stat().st_mode)))
 (pin/'manifest.json').write_text(json.dumps(dict(schema='vision-production-pins/1',files=rows),indent=2)+'\n')
 print('VISION_PIN_REGISTER PASS '+str(len(rows))+' files');return rows

def guard(root,dry=False):
 pin=rooted(root,'/root/flashnext-prod-pins/vision-w27');value=json.loads((pin/'manifest.json').read_text())
 if value.get('schema')!='vision-production-pins/1':raise ValueError('vision pin manifest differs')
 rows=value['files'];allowed={r['path'] for r in rows};changes=[]
 aside=rooted(root,'/root/flashnext-vision-boot-aside')/str(time.time_ns())
 def retain(path):
  aside.mkdir(parents=True,exist_ok=True,mode=0o700)
  dest=aside/(str(len(changes))+'-'+path.name);shutil.move(path,dest)
 for directory in DIRS:
  for path in rooted(root,directory).glob('*.conf'):
   name=str(Path(directory)/path.name)
   if name not in allowed:
    if path.is_symlink() or not path.is_file():raise ValueError('unexpected nonregular unit drop-in:'+name)
    changes.append(name)
    if not dry:retain(path)
 for row in rows:
  blob=pin/row['blob'];path=rooted(root,row['path'])
  if sha(blob)!=row['sha256']:raise ValueError('vision pin blob differs:'+row['path'])
  if path.is_symlink():raise ValueError('symlink in production pin target:'+row['path'])
  if not path.is_file() or sha(path)!=row['sha256'] or stat.S_IMODE(path.stat().st_mode)!=row['mode']:
   changes.append(row['path'])
   if not dry:
    if path.exists():retain(path)
    path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(blob,path);path.chmod(row['mode'])
 if changes and not dry and not root:subprocess.run(['systemctl','daemon-reload'],check=True)
 print('VISION_PIN_GUARD '+('DRY' if dry else 'PASS')+' '+json.dumps(dict(changes=changes,aside=str(aside) if changes else None)))
 return changes

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args(argv)
 root=os.environ.get('FNG_ROOT','');os.umask(0o077)
 if a.register:register(root)
 else:guard(root,a.dry_run)
 return 0
if __name__=='__main__':
 try:raise SystemExit(main())
 except Exception as exc:
  import traceback
  frame=traceback.extract_tb(exc.__traceback__)[-1];print(f'PACKET_CLIENT_ERROR {type(exc).__name__}: {exc} at {frame.filename}:{frame.lineno}',file=sys.stderr);raise SystemExit(2)
