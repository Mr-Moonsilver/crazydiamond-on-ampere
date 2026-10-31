#!/usr/bin/env python3
"""Native35 R/O/L owner precheck conf; PREV is the frozen native34 I4 held-a bundle for every arm."""
import argparse,difflib,hashlib,json,re,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('<hub-dir>/agents/flash-next/tmp')
PARENT_BUILD=BASE/'native34-i4-held-a/build.json'


def generate(build,out):
    b=json.loads(build.read_text());arm=b['arm'].lower();stem=b['name'].replace('-','_')
    previous=json.loads(PARENT_BUILD.read_text())
    text=(ROOT/'tools/owner-precheck/native-precheck.conf.native34-i4').read_text()
    fields=dict(BUNDLE=b['bundle'],PREV_BUNDLE=previous['bundle'],RUNNER=b['runner'],PREV_RUNNER=previous['runner'],CODEX_WT=str(ROOT),CODEX_COMMIT=b['source'],GENERATION=b['overlay'],EXPECTED_MANIFEST=b['manifest'],EXPECTED_RUNNER_SHA=b['runner_sha256'],EXPECTED_RUNNER_SIZE=str(b['runner_size']),EXPECTED_REVIEW_PIN=b['review_sha256'],SELFRUN_CONTAINER='native35-'+arm+'-hub-cpu-'+b['manifest'][:12],WORK_ROOT=str(BASE/'native35-owner-precheck-work'),QUEUE_NAME_GLOB=stem+'[._]*')
    for key,value in fields.items():
        text,n=re.subn(r'(?m)^'+key+r'=.*$',lambda m:key+'='+shlex.quote(value),text);assert n==1,key
    changed=difflib.ndiff(Path(previous['runner']).read_text().splitlines(),Path(b['runner']).read_text().splitlines())
    allowed=sorted({'^'+re.escape(line[2:])+'$' for line in changed if line.startswith(('- ','+ '))})
    block="RUNNER_DIFF_ALLOW=$(cat <<'NATIVE35_EXACT_DIFF'\n"+'\n'.join(allowed)+"\nNATIVE35_EXACT_DIFF\n)"
    text,n=re.subn(r"RUNNER_DIFF_ALLOW=\$\(cat <<'[A-Z0-9_]+'\n.*?\n[A-Z0-9_]+\n\)",lambda m:block,text,flags=re.S);assert n==1
    names=[stem+s for s in ('.draft.sh','_declare.sh','_settings.sh')]
    pins='EXPECTED_QUEUE_SHAS=(\n'+''.join('  '+name+':'+hashlib.sha256((Path(b['bundle'])/'runs'/name).read_bytes()).hexdigest()+'\n' for name in names)+')'
    text,n=re.subn(r'EXPECTED_QUEUE_SHAS=\(\n.*?\n\)',lambda m:pins,text,flags=re.S);assert n==1
    text,n=re.subn(r'"native34_tools:tools:\*\.py"\)',r'"native34_tools:tools:*.py" "native35_tools:tools:*.py")',text);assert n==1
    text,n=re.subn(r'\|native34_tools\)\$',r'|native34_tools|native35_tools)$',text);assert n==1
    text,n=re.subn(r'SEAM_TESTS=\(',r'SEAM_TESTS=(test_native35_queue.py test_native35_readback.py test_native35_grade.py test_native35_dispatch_lifetime.py native35_tools/native35_builder_view_cpu_check.py ',text);assert n==1
    text,n=re.subn(r'FIXTURE_FILES=\(\n',r'FIXTURE_FILES=(\n  test_native35_grade.py # synthetic CPU auth fixture only\n',text);assert n==1
    # The VM comparison overlay predates the fix; its four engine files differ by design.
    text,n=re.subn(r'^VM_OVERLAY_EXCLUSIONS=\((.*)\)$',lambda m:'VM_OVERLAY_EXCLUSIONS=('+m.group(1)+' e3/lane_chunk.py e3/lane_dispatch.py e3/native_startup.py e3/s5_startup.py)',text,flags=re.M);assert n==1
    out.write_text('# Native35 '+arm.upper()+' (R then O then L, fresh processes); PREV is the frozen native34 I4 held-a bundle for every arm.\n'
        '# Engine delta vs PREV: lane_chunk/lane_dispatch/native_startup/s5_startup (Codex native35 fix) plus native34_worker/native34_readback plugins (arm-O evidence readback).\n'+text)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();generate(a.build,a.out)
