#!/usr/bin/env python3
"""Native35 P owner precheck conf; PREV is the frozen native35 O held-a bundle and runner."""
import argparse,difflib,hashlib,json,re,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('<hub-dir>/agents/flash-next/tmp')
PARENT_BUILD=BASE/'native35-o-held-a/build.json'


def generate(build,out):
    b=json.loads(build.read_text());assert b['arm']=='P';stem=b['name'].replace('-','_')
    previous=json.loads(PARENT_BUILD.read_text());assert previous['manifest']==b['parent_manifest'] and previous['runner_sha256']==b['parent_runner_sha256']
    text=(ROOT/'tools/owner-precheck/native-precheck.conf.native35-o').read_text()
    fields=dict(BUNDLE=b['bundle'],PREV_BUNDLE=previous['bundle'],RUNNER=b['runner'],PREV_RUNNER=previous['runner'],CODEX_WT=str(ROOT),CODEX_COMMIT=b['source'],
                GENERATION=b['overlay'],EXPECTED_MANIFEST=b['manifest'],EXPECTED_RUNNER_SHA=b['runner_sha256'],EXPECTED_RUNNER_SIZE=str(b['runner_size']),
                EXPECTED_REVIEW_PIN=b['review_sha256'],SELFRUN_CONTAINER='native35-p-hub-cpu-'+b['manifest'][:12],WORK_ROOT=str(BASE/'native35-p-owner-precheck-work'),
                QUEUE_NAME_GLOB=stem+'[._]*')
    for key,value in fields.items():
        text,n=re.subn(r'(?m)^'+key+r'=.*$',lambda m:key+'='+shlex.quote(value),text);assert n==1,key
    changed=difflib.ndiff(Path(previous['runner']).read_text().splitlines(),Path(b['runner']).read_text().splitlines())
    allowed=sorted({'^'+re.escape(line[2:])+'$' for line in changed if line.startswith(('- ','+ '))})
    block="RUNNER_DIFF_ALLOW=$(cat <<'NATIVE35_P_EXACT_DIFF'\n"+'\n'.join(allowed)+"\nNATIVE35_P_EXACT_DIFF\n)"
    text,n=re.subn(r"RUNNER_DIFF_ALLOW=\$\(cat <<'[A-Z0-9_]+'\n.*?\n[A-Z0-9_]+\n\)",lambda m:block,text,flags=re.S);assert n==1
    names=[stem+s for s in ('.draft.sh','_declare.sh','_settings.sh')]
    pins='EXPECTED_QUEUE_SHAS=(\n'+''.join('  '+name+':'+hashlib.sha256((Path(b['bundle'])/'runs'/name).read_bytes()).hexdigest()+'\n' for name in names)+')'
    text,n=re.subn(r'EXPECTED_QUEUE_SHAS=\(\n.*?\n\)',lambda m:pins,text,flags=re.S);assert n==1
    text,n=re.subn(r'SEAM_TESTS=\(',r'SEAM_TESTS=(test_native35_performance.py test_native35_performance_queue.py ',text);assert n==1
    out.write_text('# Native35 P (speed arm, fresh process after O); PREV is the frozen native35 O held-a bundle and runner.\n'
        '# Engine delta vs PREV: none (overlay, reviews and compiled libraries byte-identical); harness adds the P queue, round CLI, aggregate grade, window and suites.\n'+text)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();generate(a.build,a.out)
