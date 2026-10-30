#!/usr/bin/env python3
"""Read-only hub gate for the native30 held delivery; no remote/engine access."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root):
    rows={}
    for line in (root/'SHA256SUMS').read_text().splitlines():
        pin,name=line.split(None,1);assert sha(root/name)==pin,name;rows[name]=pin
    assert set(rows)=={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p!=root/'SHA256SUMS'}
    return rows

def check(build_path,result_path):
    build=json.loads(build_path.read_text());bundle=Path(build['bundle']);overlay=Path(build['overlay']);owner=Path(build['runner'])
    rows=verify(bundle);verify(overlay)
    assert sha(bundle/'SHA256SUMS')==build['manifest'] and sha(overlay/'SHA256SUMS')==build['overlay_manifest']
    assert sha(owner)==build['runner_sha256']
    source=owner.read_text();tree=ast.parse(source);compile(source,str(owner),'exec')
    functions={n.name:ast.get_source_segment(source,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
    assert 'systemctl' not in functions['preflight'] and 'nvidia-smi' not in functions['preflight']
    assert 'stopped_unit(' in functions['live_guards'] and 'nvidia-smi' in functions['live_guards']
    assert '--staging' in functions['main'] and '--arm' in functions['main']
    assert functions['main'].index('if a.staging:')<functions['main'].index('live_guards(')
    for pin in ('45cc5d813dca6610dd871b99b095956f307f29ab5f8de4993400018c1ab80c88','f9fb82486ef3132ac3e6fa9c9ed12fb33225b08e1727727376d3e5ac1a678311'):assert pin in functions['preflight']
    assert sha(overlay/'run-flash-s5-test.sh')=='22473a909dad1a4cb8e9c22de88349e65ff1eb46b0fa05b281504afb21775f3c'
    proof=json.loads((bundle/'native30-provenance.json').read_text());reg=proof['selftest']
    assert reg['cases']==reg['required'] and reg['exit_code']==reg['skips']==0
    baseline=Path('<hub-dir>/agents/flash-next/tmp/native28-held-b/final')
    for name in proof['runtime_byte_equal']:assert sha(overlay/name)==sha(baseline/name),name
    assert sha(overlay/'e3/native30_plugins/native30_localize.py')==sha(bundle/'native30_localize.py')
    leaves=json.loads((bundle/'native30_fixture_leaves.json').read_text())['leaves']
    assert len(leaves)==6 and all(x['replaces'] and x['equivalence'] and x['cannot_prove'] for x in leaves)
    result=json.loads(result_path.read_text());assert result['verdict']=='PASS' and result['rc']==result['skips']==result['gpu_device_requests']==0
    assert result['bundle_manifest']==build['manifest'] and result['overlay_manifest']==build['overlay_manifest']
    assert result['identity_capture_check']=='PASS' and result['identity_timeout_negative_rejected'] is True
    assert result['plugin_loader']=='PASS' and result['tests_run']==13 and result['cuda_initialized'] is False
    assert result['image']=='sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b'
    sys.path.insert(0,str(bundle/'e3_builder'))
    from native_readiness import verify_finalized_package
    closure=json.loads((overlay/'e3/native-implementation-closure.json').read_text());verify_finalized_package(overlay,closure['generated_source_shas'])
    return dict(verdict='PASS',classification=None,precheck_ready=True,bundle_manifest=build['manifest'],overlay_manifest=build['overlay_manifest'],runner_sha256=sha(owner),checksummed_files=len(rows),runtime_files_unchanged=len(proof['runtime_byte_equal']),fixture_leaves=len(leaves),selfrun_sha256=sha(result_path),selftest=reg,box_action=False,scope='hub gates; owner --staging checks live production file hashes, then --arm requires stopped engine and four idle GPUs')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--selfrun',type=Path,required=True);a=p.parse_args();print(json.dumps(check(a.build,a.selfrun),sort_keys=True,indent=2))
