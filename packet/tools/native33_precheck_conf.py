#!/usr/bin/env python3
"""Generate native33 exact owner precheck against frozen native32."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shlex
ROOT=Path(__file__).resolve().parents[1]
PARENT=Path('<hub-dir>/agents/flash-next/tmp/native32-held-c/q18a-native32-bundle-b50aea5737fb')
PREV_RUNNER=PARENT.parent/'q18a-native32-owner-run-b50aea5737fb.py'


def generate(build,vm,out):
    b=json.loads(build.read_text())
    text=(ROOT/'tools/owner-precheck/native-precheck.conf.native32').read_text()
    fields=dict(BUNDLE=b['bundle'],PREV_BUNDLE=str(PARENT),RUNNER=b['runner'],
                PREV_RUNNER=str(PREV_RUNNER),CODEX_WT=str(ROOT),CODEX_COMMIT=b['source'],
                GENERATION=b['overlay'],VM_OVERLAY=str(vm),EXPECTED_MANIFEST=b['manifest'],
                EXPECTED_RUNNER_SHA=b['runner_sha256'],EXPECTED_RUNNER_SIZE=str(b['runner_size']),
                EXPECTED_REVIEW_PIN=b['review_sha256'],SELFRUN_CONTAINER='native33-hub-cpu-'+b['manifest'][:12],
                WORK_ROOT='<hub-dir>/agents/flash-next/tmp/native33-owner-precheck-work',
                QUEUE_NAME_GLOB='q18a_native33[._]*')
    for key,value in fields.items():
        text,n=re.subn(r'(?m)^'+key+r'=.*$',lambda m:key+'='+shlex.quote(value),text);assert n==1,key
    changed=difflib.ndiff(PREV_RUNNER.read_text().splitlines(),Path(b['runner']).read_text().splitlines())
    allowed=sorted({'^'+re.escape(line[2:])+'$' for line in changed if line.startswith(('- ','+ '))})
    block="RUNNER_DIFF_ALLOW=$(cat <<'NATIVE33_EXACT_DIFF'\n"+'\n'.join(allowed)+"\nNATIVE33_EXACT_DIFF\n)"
    text,n=re.subn(r"RUNNER_DIFF_ALLOW=\$\(cat <<'[A-Z0-9_]+'\n.*?\n[A-Z0-9_]+\n\)",lambda m:block,text,flags=re.S);assert n==1
    names=['q18a_native33'+suffix for suffix in ('.draft.sh','_declare.sh','_settings.sh')]
    pins='EXPECTED_QUEUE_SHAS=(\n'+''.join('  '+name+':'+hashlib.sha256((Path(b['bundle'])/'runs'/name).read_bytes()).hexdigest()+'\n' for name in names)+')'
    text,n=re.subn(r'EXPECTED_QUEUE_SHAS=\(\n.*?\n\)',lambda m:pins,text,flags=re.S);assert n==1
    text=text.replace('"native32_tools:tools:*.py")','"native32_tools:tools:*.py" "native33_tools:tools:*.py")')
    text=text.replace('native32_tools)$','native32_tools|native33_tools)$')
    text=text.replace('test_native32.py)','test_native32.py test_native33.py)')
    text=text.replace('SOURCE_FILE_MAP=(', 'SOURCE_FILE_MAP=("native30_overlay/e3/native30_plugins/native33_diag.py:harness/native33_diag.py" ')
    out.write_text('# Native33 PREV=native32. Retained old comments below are historical.\n'+text)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--vm-input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();generate(a.build,a.vm_input,a.out)
