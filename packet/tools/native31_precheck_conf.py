#!/usr/bin/env python3
"""Generate exact native31 owner precheck pins from a completed frozen build."""
import argparse
import difflib
import json
from pathlib import Path
import re
import shlex
ROOT=Path(__file__).resolve().parents[1]
PARENT=Path('<hub-dir>/agents/flash-next/tmp/native30-localize-e-held-b/q18a-native30-localize-e-bundle-9f94ea1a0142')
PREV_RUNNER=PARENT.parent/'q18a-native30-localize-e-owner-run-9f94ea1a0142.py'

def generate(build,vm,out):
    b=json.loads(build.read_text());text=(ROOT/'tools/owner-precheck/native-precheck.conf.native30-localize-e').read_text()
    fields=dict(BUNDLE=b['bundle'],PREV_BUNDLE=str(PARENT),RUNNER=b['runner'],PREV_RUNNER=str(PREV_RUNNER),CODEX_WT=str(ROOT),CODEX_COMMIT=b['source'],GENERATION=b['overlay'],VM_OVERLAY=str(vm),EXPECTED_MANIFEST=b['manifest'],EXPECTED_RUNNER_SHA=b['runner_sha256'],EXPECTED_RUNNER_SIZE=str(b['runner_size']),EXPECTED_REVIEW_PIN=b['review_sha256'],SELFRUN_CONTAINER='native31-hub-cpu-c',WORK_ROOT='<hub-dir>/agents/flash-next/tmp/native31-owner-precheck-work',QUEUE_NAME_GLOB='q18a_native31[._]*')
    for key,value in fields.items():
        text,n=re.subn(r'(?m)^'+key+r'=.*$',lambda m:key+'='+shlex.quote(value),text);assert n==1,key
    diff=difflib.ndiff(PREV_RUNNER.read_text().splitlines(),Path(b['runner']).read_text().splitlines())
    allowed=sorted({'^'+re.escape(line[2:])+'$' for line in diff if line.startswith(('- ','+ '))})
    text,n=re.subn(r"RUNNER_DIFF_ALLOW=\$\(cat <<'NATIVE30_EXACT_DIFF'\n.*?\nNATIVE30_EXACT_DIFF\n\)",lambda m:"RUNNER_DIFF_ALLOW=$(cat <<'NATIVE31_EXACT_DIFF'\n"+'\n'.join(allowed)+"\nNATIVE31_EXACT_DIFF\n)",text,flags=re.S);assert n==1
    queue=Path(b['bundle'])/'runs'
    import hashlib
    names=['q18a_native31'+suffix for suffix in ('.draft.sh','_declare.sh','_settings.sh')]
    pins='EXPECTED_QUEUE_SHAS=(\n'+''.join('  '+name+':'+hashlib.sha256((queue/name).read_bytes()).hexdigest()+'\n' for name in names)+')'
    text,n=re.subn(r'EXPECTED_QUEUE_SHAS=\(\n.*?\n \)',lambda m:pins,text,flags=re.S);assert n==1
    text=text.replace(' "vmid_sources/s5_engine:patches/s5_engine:*"','')
    text=text.replace('native30-provenance\\.json)$','native30-provenance\\.json|native31-fixtures)$')
    text=text.replace('test_native30_e.py)','test_native30_e.py test_native31.py native31_cpu_leaves.py native30_tools/native31_cpu_check.py)')
    text='# Native31: current executable builder/tools compared to committed source; PREV=e.\n# vmid_sources is a retained byte-identical historical prerequisite snapshot, not native31 device proof.\n'+text
    out.write_text(text)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--vm-input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();generate(a.build,a.vm_input,a.out)
