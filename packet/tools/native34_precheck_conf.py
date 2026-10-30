#!/usr/bin/env python3
"""Generate native34 role-specific owner precheck against frozen native33."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shlex
ROOT=Path(__file__).resolve().parents[1]
PARENT=Path('<hub-dir>/agents/flash-next/tmp/native33-held-a/q18a-native33-bundle-8b5fe8528dc9')
PREV_RUNNER=PARENT.parent/'q18a-native33-owner-run-8b5fe8528dc9.py'


def generate(build,vm,out):
    b=json.loads(build.read_text());arm=b['arm'];stem='q18a_native34_'+arm.lower()
    text=(ROOT/'tools/owner-precheck/native-precheck.conf.native33').read_text()
    fields=dict(BUNDLE=b['bundle'],PREV_BUNDLE=str(PARENT),RUNNER=b['runner'],
                PREV_RUNNER=str(PREV_RUNNER),CODEX_WT=str(ROOT),CODEX_COMMIT=b['source'],
                GENERATION=b['overlay'],VM_OVERLAY=str(vm),EXPECTED_MANIFEST=b['manifest'],
                EXPECTED_RUNNER_SHA=b['runner_sha256'],EXPECTED_RUNNER_SIZE=str(b['runner_size']),
                EXPECTED_REVIEW_PIN=b['review_sha256'],SELFRUN_CONTAINER='native34-'+arm.lower()+'-hub-cpu-'+b['manifest'][:12],
                WORK_ROOT='<hub-dir>/agents/flash-next/tmp/native34-owner-precheck-work',
                QUEUE_NAME_GLOB=stem+'[._]*')
    for key,value in fields.items():
        text,n=re.subn(r'(?m)^'+key+r'=.*$',lambda m:key+'='+shlex.quote(value),text);assert n==1,key
    changed=difflib.ndiff(PREV_RUNNER.read_text().splitlines(),Path(b['runner']).read_text().splitlines())
    allowed=sorted({'^'+re.escape(line[2:])+'$' for line in changed if line.startswith(('- ','+ '))})
    block="RUNNER_DIFF_ALLOW=$(cat <<'NATIVE33_EXACT_DIFF'\n"+'\n'.join(allowed)+"\nNATIVE33_EXACT_DIFF\n)"
    text,n=re.subn(r"RUNNER_DIFF_ALLOW=\$\(cat <<'[A-Z0-9_]+'\n.*?\n[A-Z0-9_]+\n\)",lambda m:block,text,flags=re.S);assert n==1
    names=[stem+suffix for suffix in ('.draft.sh','_declare.sh','_settings.sh')]
    pins='EXPECTED_QUEUE_SHAS=(\n'+''.join('  '+name+':'+hashlib.sha256((Path(b['bundle'])/'runs'/name).read_bytes()).hexdigest()+'\n' for name in names)+')'
    text,n=re.subn(r'EXPECTED_QUEUE_SHAS=\(\n.*?\n\)',lambda m:pins,text,flags=re.S);assert n==1
    text=text.replace('"native33_tools:tools:*.py")','"native33_tools:tools:*.py" "native34_tools:tools:*.py")')
    text=text.replace('native33_tools)$','native33_tools|native34_tools)$')
    text=text.replace('test_native33.py)','test_native33.py test_native34_storage.py test_native34_control.py test_native34_indexer.py test_native34_runtime.py test_native34_frontend.py test_native34_queue.py)')
    additions=' '.join('"native30_overlay/e3/native30_plugins/'+n+':harness/'+n+'"' for n in ('native34_diag.py','native34_worker.py','native34_control.py','native34_storage.py','native34_indexer.py'))
    text=text.replace('SOURCE_FILE_MAP=(', 'SOURCE_FILE_MAP=('+additions+' ')
    out.write_text('# Native34 '+arm+' PREV=native33. I then S then O, fresh processes.\n'+text)



if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--vm-input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();generate(a.build,a.vm_input,a.out)
