#!/usr/bin/env python3
"""Run only the pre-registered100 ids through the frozen w26 runner and scorer."""
import argparse,asyncio,datetime as dt,hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from packet_client_contract import report_exception
SELECTION_SHA='281f87162263e61d46f75e8436ff0f20102c9a12a1e86e6ce2075547546372ce'
QUALITY_SHA='f0b8479fd998bf45ab55246a5152f41c1434d127a356eea7e3d83bfa2fcc6a25'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--quality-packet',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--base-url',required=True);p.add_argument('--key-file',type=Path,required=True)
    p.add_argument('--model',default='qwen38-w8a8-tp4');p.add_argument('--out',type=Path,required=True);p.add_argument('--deadline',required=True)
    a=p.parse_args(argv);os.umask(0o077)
    if a.out.exists():raise ValueError('fresh A3 output required')
    if sha(a.selection)!=SELECTION_SHA or sha(a.quality_packet/'PACKET-SHA256SUMS')!=QUALITY_SHA:raise ValueError('frozen selection/quality identity differs')
    subprocess.run(['sha256sum','--strict','--quiet','-c','PACKET-SHA256SUMS'],cwd=a.quality_packet,check=True)
    selected=json.loads(a.selection.read_text())['ids']
    if len(selected)!=100 or len(set(selected))!=100:raise ValueError('100 unique ids required')
    spec=importlib.util.spec_from_file_location('frozen_w26_runner',a.quality_packet/'code/runner.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    items=runner.load_items(str(a.quality_packet));todo=[items[i] for i in selected]
    if any(it.get('thinking') for it in todo):raise ValueError('A3 requests must keep frozen w26 thinking-off shape')
    if a.key_file.is_symlink() or a.key_file.stat().st_mode&0o777!=0o600:raise ValueError('0600 key file required')
    rec=runner.Rec(str(a.out));deadline=dt.datetime.fromisoformat(a.deadline).timestamp()
    async def run():
        client=runner.Client(a.base_url,a.model,runner.read_key(str(a.key_file)),rec);client.hard_end=deadline
        try:
            return await runner.run_items(client,rec,'main',todo,8,deadline,200000)
        finally:await client.http.aclose();rec.logf.close()
    stats=asyncio.run(run())
    (a.out/'a3-collection.json').write_text(json.dumps(dict(selection_sha256=SELECTION_SHA,quality_sha256=QUALITY_SHA,stats=stats,server_defaults='unchanged; A3 alone explicitly thinking off',request_body_shas={i:hashlib.sha256(json.dumps(runner.Client.body(type('Body',(),dict(model=a.model))(),items[i]),sort_keys=True).encode()).hexdigest() for i in selected}),indent=2)+'\n')
    if stats!=dict(done=100,skipped_cap=0,errors=0):print('W27_A3 HOLD incomplete-or-request-errors');return 2
    with (a.out/'score.stdout.log').open('x') as stdout,(a.out/'score.stderr.log').open('x') as stderr:
        result=subprocess.run([sys.executable,'-I',str(a.quality_packet/'code/score.py'),'--packet',str(a.quality_packet),'--out',str(a.out)],stdout=stdout,stderr=stderr,timeout=600)
    if result.returncode:raise ValueError('frozen scorer nonzero; see private score.stderr.log')
    from vision_w27_read import a3_read
    result=a3_read(a.out,a.quality_packet,a.selection);(a.out/'a3-read.json').write_text(json.dumps(result,indent=2)+'\n')
    print('W27_A3 '+result['verdict']+' '+result['reason']);return 0 if result['verdict']=='PASS' else 2
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:report_exception(exc);print('W27_A3 HOLD exception');raise SystemExit(2)
