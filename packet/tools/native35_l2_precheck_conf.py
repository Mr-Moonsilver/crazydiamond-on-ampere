#!/usr/bin/env python3
"""Native35 L2 owner precheck conf; PREV is the frozen native35 L held-a bundle and runner.

Derived from the native35 L conf: only the bundle/runner/source pins, the self-run
container, the work root, the queue glob/pins, the exact L -> L2 runner diff and
the added L2 seam test change. VM overlay exclusions, source maps, the bundle-only
allowance, credential patterns, key files and fixture files are L's.
"""
import argparse,difflib,hashlib,json,re,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('<hub-dir>/agents/flash-next/tmp')
PARENT_BUILD=BASE/'native35-l-held-a/build.json'


def generate(build,out):
    b=json.loads(build.read_text());assert b['arm']=='L2' and b['name']=='q18a-native35-l2';stem=b['name'].replace('-','_')
    previous=json.loads(PARENT_BUILD.read_text());assert previous['manifest']==b['parent_manifest']
    text=(ROOT/'tools/owner-precheck/native-precheck.conf.native35-l').read_text()
    fields=dict(BUNDLE=b['bundle'],PREV_BUNDLE=previous['bundle'],RUNNER=b['runner'],PREV_RUNNER=previous['runner'],CODEX_WT=str(ROOT),CODEX_COMMIT=b['source'],GENERATION=b['overlay'],EXPECTED_MANIFEST=b['manifest'],EXPECTED_RUNNER_SHA=b['runner_sha256'],EXPECTED_RUNNER_SIZE=str(b['runner_size']),EXPECTED_REVIEW_PIN=b['review_sha256'],SELFRUN_CONTAINER='native35-l2-hub-cpu-'+b['manifest'][:12],WORK_ROOT=str(BASE/'native35-l2-owner-precheck-work'),QUEUE_NAME_GLOB=stem+'[._]*')
    assert b['review_sha256']==previous['review_sha256'] and b['overlay_manifest']==previous['overlay_manifest']
    for key,value in fields.items():
        text,n=re.subn(r'(?m)^'+key+r'=.*$',lambda m:key+'='+shlex.quote(value),text);assert n==1,key
    changed=difflib.ndiff(Path(previous['runner']).read_text().splitlines(),Path(b['runner']).read_text().splitlines())
    allowed=sorted({'^'+re.escape(line[2:])+'$' for line in changed if line.startswith(('- ','+ '))})
    block="RUNNER_DIFF_ALLOW=$(cat <<'NATIVE35_L2_EXACT_DIFF'\n"+'\n'.join(allowed)+"\nNATIVE35_L2_EXACT_DIFF\n)"
    text,n=re.subn(r"RUNNER_DIFF_ALLOW=\$\(cat <<'[A-Z0-9_]+'\n.*?\n[A-Z0-9_]+\n\)",lambda m:block,text,flags=re.S);assert n==1
    names=[stem+s for s in ('.draft.sh','_declare.sh','_settings.sh')]
    pins='EXPECTED_QUEUE_SHAS=(\n'+''.join('  '+name+':'+hashlib.sha256((Path(b['bundle'])/'runs'/name).read_bytes()).hexdigest()+'\n' for name in names)+')'
    text,n=re.subn(r'EXPECTED_QUEUE_SHAS=\(\n.*?\n\)',lambda m:pins,text,flags=re.S);assert n==1
    text,n=re.subn(r'SEAM_TESTS=\(',r'SEAM_TESTS=(test_native35_l2.py ',text);assert n==1
    out.write_text('# Native35 L2; PREV is the frozen native35 L held-a bundle and runner. Overlay byte-identical to L (no engine delta);\n'
        '# L2 adds the row4 resume-receipt window collector, its queue under fresh names, tests, fixture leaves and self-run tools.\n'+text)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();generate(a.build,a.out)
