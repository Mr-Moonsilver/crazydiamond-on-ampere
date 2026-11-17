#!/usr/bin/env python3
"""Real declaration renderer/probe over the retained owner CPU filesystem fixture; no engine control."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'harness'))
from packet_client_contract import report_exception

def main():
    p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--fixture-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(mode=0o700);root=a.out/'root';shutil.copytree(a.fixture_root,root)
    run=next((a.fixture_root/'data').glob('serve-*-engine'))
    defaults=(run/'private/defaults.serve').read_text().replace(str(a.fixture_root),'')
    defaults=defaults.replace('/root/.flashnext-serve-',str(root)+'/root/.flashnext-serve-')
    (root/'etc/default/vllm-flash').write_text(defaults)
    launcher=root/'data/vllm/flash-next-tp4/prod/run-flash-prod.sh'
    shutil.copy2(a.packet/'harness/native30_overlay/run-flash-s5-test.sh',launcher)
    image=a.out/'image-inspect.json'
    with image.open('x') as f:subprocess.run(['docker','image','inspect',(a.packet/'image-id.txt').read_text().strip()],stdout=f,check=True)
    image.chmod(0o600)
    argv=[sys.executable,'-B',str(a.packet/'harness/engine_identity.py'),'declare',str(a.out/'request.json'),
        '--defaults',str(root/'etc/default/vllm-flash'),'--launcher',str(launcher),'--setcfg',str(root/'tmp/setcfg.py'),
        '--image-file',str(a.packet/'image-id.txt'),'--image-inspect',str(image),'--source-mirror',str(root/'data/vllm/flash-next-tp4'),
        '--mount-policy',str(a.packet/'harness/e3_startup_mount_policy.json'),'--timeout','60']
    (a.out/'argv.json').write_text(json.dumps(argv))
    result=subprocess.run(argv,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    (a.out/'stdout').write_text(result.stdout);(a.out/'stderr').write_text(result.stderr)
    if result.returncode:raise RuntimeError('real-declaration-rc='+str(result.returncode)+'; retained '+str(a.out))
    request=json.loads((a.out/'request.json').read_text());args=request['identity']['command']['args']
    assert '--enable-mm-embeds' in args and '--language-model-only' not in args
    assert '--skip-mm-profiling' in args and '--mm-processor-cache-gb' in args
    # Root policy excludes inactive duplicate models only; actual /model content remains fully hashed.
    model=next(m for m in request['identity']['mounts'] if m['destination']=='/model')
    assert model['content']['exclusions']==[]
    (a.out/'result.json').write_text(json.dumps(dict(verdict='PASS',mounts=len(request['identity']['mounts']),
        scope='real declaration/launcher/private-env/mount hashing over CPU owner fixture; model files are synthetic, not hardware/weight evidence'),indent=2)+'\n')
    print('VISION_FIT_REAL_DECLARATION PASS')
if __name__=='__main__':
    try:main()
    except Exception as exc:report_exception(exc);raise SystemExit(2)
