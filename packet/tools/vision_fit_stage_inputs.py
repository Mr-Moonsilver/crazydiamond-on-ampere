#!/usr/bin/env python3
"""OWNER-ONLY private cache copy on the existing PVE route; CPU self-run uses a local PVE stub."""
import argparse,hashlib,json,os,shlex,stat,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'harness'))
from vision_fit_8x200k import CACHE_SHAS
from packet_client_contract import report_exception
PVE='ssh -F /dev/null -i /home/<service-user>/.ssh/<ssh-key> -o IdentitiesOnly=yes -o BatchMode=yes root@<pve-host>'
REMOTE=r'''
import hashlib,json,os,stat,sys
from pathlib import Path
operation,run,*args=sys.argv[1:]
if not run.startswith('/data/') or any(p in ('','..','.') for p in run[1:].split('/')):raise ValueError('private-run-must-be-under-data')
root=Path(os.environ.get('R','')+run)
if operation=='mkdir':
 root.mkdir(mode=0o700);(root/'private').mkdir(mode=0o700)
 print('VISION_FIT_CT_PRIVATE PASS 0700');raise SystemExit(0)
for p in (root,root/'private'):
 if p.is_symlink() or not p.is_dir() or p.stat().st_mode & 0o777!=0o700 or p.stat().st_uid!=os.geteuid():raise ValueError('private-directory-owner-or-mode')
if operation=='copy':
 name,expected,size=args;size=int(size)
 if name not in ('semantic.json','eight-unique.json') or not 0<size<=64*2**20:raise ValueError('cache-name-or-size')
 raw=sys.stdin.buffer.read(size+1)
 if len(raw)!=size or hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('CT-cache-SHA-or-size-mismatch')
 path=root/'private'/name
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as f:f.write(raw)
 print('VISION_FIT_CT_COPY PASS '+name+' '+expected);raise SystemExit(0)
if operation!='verify':raise ValueError('unknown-copy-operation')
rows=[]
for name,expected in zip(('semantic.json','eight-unique.json'),args):
 p=root/'private'/name;s=p.lstat()
 if not stat.S_ISREG(s.st_mode) or s.st_mode & 0o777!=0o600 or s.st_uid!=os.geteuid():raise ValueError('cache-owner-mode-or-type')
 actual=hashlib.sha256(p.read_bytes()).hexdigest()
 if actual!=expected:raise ValueError('CT-final-cache-SHA-mismatch')
 rows.append(dict(name=name,sha256=actual,bytes=s.st_size,mode='0600'))
print('VISION_FIT_CT_INPUTS PASS '+json.dumps(rows,sort_keys=True))
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--private-cache-dir',type=Path,required=True);p.add_argument('--ct-run',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--pve-cmd',default=os.environ.get('PVE',PVE));a=p.parse_args()
    if a.out.exists():raise ValueError('fresh-stage-receipt-required')
    source=a.private_cache_dir
    if source.is_symlink() or source.stat().st_mode & 0o777!=0o700:raise ValueError('hub-private-directory')
    files=[]
    for name,expected in CACHE_SHAS.items():
        path=source/(name+'.json');s=path.lstat()
        if not stat.S_ISREG(s.st_mode) or s.st_mode & 0o777!=0o600 or not 0<s.st_size<=64*2**20:raise ValueError('hub-private-file')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('hub-cache-SHA-'+name)
        files.append((path,expected,s.st_size))
    steps=[]
    def call(operation,*words,stdin=None):
        remote=shlex.join(['pct','exec','<ct-id>','--','python3','-B','-c',REMOTE,operation,a.ct_run,*map(str,words)])
        argv=shlex.split(a.pve_cmd)+[remote]
        result=subprocess.run(argv,stdin=stdin,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
        # This fixed remote program prints metadata only; cache bytes travel on stdin.
        label=operation+'-'+str(len(steps))
        a.out.with_name(a.out.stem+'-'+label+'.stdout.log').write_bytes(result.stdout)
        a.out.with_name(a.out.stem+'-'+label+'.stderr.log').write_bytes(result.stderr)
        if result.returncode:raise RuntimeError('private-CT-stage-'+operation+'-rc='+str(result.returncode)+' '+result.stderr.decode('utf-8','replace')[-1600:])
        text=result.stdout.decode()
        if not text.startswith('VISION_FIT_CT_') or ' PASS ' not in text:raise ValueError('CT-stage-receipt-schema')
        print(text.strip());steps.append(dict(operation=operation,receipt=text.strip()))
    call('mkdir')
    for path,expected,size in files:
        with path.open('rb') as f:call('copy',path.name,expected,size,stdin=f)
    call('verify',*CACHE_SHAS.values())
    a.out.write_text(json.dumps(dict(verdict='PASS',ct_run=a.ct_run,hub_authoritative=True,retained=True,steps=steps),indent=2)+'\n')
    print('VISION_FIT_STAGE_INPUTS PASS')
if __name__=='__main__':
    try:main()
    except Exception as exc:report_exception(exc);print('VISION_FIT_STAGE_INPUTS HOLD');raise SystemExit(2)
