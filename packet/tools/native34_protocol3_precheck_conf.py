#!/usr/bin/env python3
"""Window-only native34 I2/S2/O2 precheck, PREV the corresponding frozen arm."""
import argparse,difflib,hashlib,json,re,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('<hub-dir>/agents/flash-next/tmp')


def generate(build,out):
    b=json.loads(build.read_text());arm=b['arm'].lower();stem=b['name'].replace('-','_')
    previous=json.loads((BASE/('native34-'+arm+'2-held-a')/'build.json').read_text())
    text=(ROOT/'tools/owner-precheck'/('native-precheck.conf.native34-'+arm+'2')).read_text()
    fields=dict(BUNDLE=b['bundle'],PREV_BUNDLE=previous['bundle'],RUNNER=b['runner'],PREV_RUNNER=previous['runner'],CODEX_WT=str(ROOT),CODEX_COMMIT=b['source'],GENERATION=b['overlay'],EXPECTED_MANIFEST=b['manifest'],EXPECTED_RUNNER_SHA=b['runner_sha256'],EXPECTED_RUNNER_SIZE=str(b['runner_size']),EXPECTED_REVIEW_PIN=b['review_sha256'],SELFRUN_CONTAINER='native34-'+arm+'3-hub-cpu-'+b['manifest'][:12],WORK_ROOT=str(BASE/'native34-protocol3-owner-precheck-work'),QUEUE_NAME_GLOB=stem+'[._]*')
    for key,value in fields.items():
        text,n=re.subn(r'(?m)^'+key+r'=.*$',lambda m:key+'='+shlex.quote(value),text);assert n==1,key
    changed=difflib.ndiff(Path(previous['runner']).read_text().splitlines(),Path(b['runner']).read_text().splitlines())
    allowed=sorted({'^'+re.escape(line[2:])+'$' for line in changed if line.startswith(('- ','+ '))})
    block="RUNNER_DIFF_ALLOW=$(cat <<'PROTOCOL3_EXACT_DIFF'\n"+'\n'.join(allowed)+"\nPROTOCOL3_EXACT_DIFF\n)"
    text,n=re.subn(r"RUNNER_DIFF_ALLOW=\$\(cat <<'[A-Z0-9_]+'\n.*?\n[A-Z0-9_]+\n\)",lambda m:block,text,flags=re.S);assert n==1
    names=[stem+s for s in ('.draft.sh','_declare.sh','_settings.sh')]
    pins='EXPECTED_QUEUE_SHAS=(\n'+''.join('  '+name+':'+hashlib.sha256((Path(b['bundle'])/'runs'/name).read_bytes()).hexdigest()+'\n' for name in names)+')'
    text,n=re.subn(r'EXPECTED_QUEUE_SHAS=\(\n.*?\n\)',lambda m:pins,text,flags=re.S);assert n==1
    out.write_text('# Native34 protocol repair '+arm.upper()+'3; PREV corresponding frozen native34 role2.\n'+text)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();generate(a.build,a.out)
