#!/usr/bin/env python3
"""Independent read-only final-byte proof for a native34 held arm."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'harness')]
from build_native30_held import sha,verify,replace,dump
from test_native34_queue import normalize_queue

PARENT=Path('<hub-dir>/agents/flash-next/tmp/native33-held-a/q18a-native33-bundle-8b5fe8528dc9')
PREV_RUNNER=PARENT.parent/'q18a-native33-owner-run-8b5fe8528dc9.py'


def check(build_path,out):
    b=json.loads(build_path.read_text());root=Path(b['bundle']);arm=b['arm']
    assert arm in ('I','S','O')
    rows=verify(root);old=verify(PARENT);ov=verify(root/'native30_overlay');prev=verify(PARENT/'native30_overlay')
    assert sha(root/'SHA256SUMS')==b['manifest'] and sha(Path(b['runner']))==b['runner_sha256']
    assert not set(prev)-set(ov)
    changed=[n for n in prev if prev[n]!=ov[n]]
    assert not [n for n in changed if n.endswith(('.py','.so'))]
    additions={'e3/native30_plugins/'+n for n in ('native34_diag.py','native34_worker.py','native34_control.py','native34_indexer.py','native34_storage.py')}
    additions|={'e3/native30_plugins/flashnext_native34_diagnostic-0.1.dist-info/'+n for n in ('METADATA','entry_points.txt')}
    assert set(ov)-set(prev)==additions
    assert all(ov[n]==pin for n,pin in prev.items() if n.startswith('e3/native30_plugins/'))
    stem='q18a_native34_'+arm.lower();name=stem.replace('_','-')
    assert normalize_queue((root/'runs'/(stem+'.draft.sh')).read_text(),arm)==(PARENT/'runs/q18a_native33.draft.sh').read_text()
    settings=(root/'runs'/(stem+'_settings.sh')).read_text().replace(stem,'q18a_native33').replace(name,'q18a-native33')
    settings=settings.replace('FLASH_NATIVE33_DIAG=0 FLASH_NATIVE34_DIAG=1 FLASH_NATIVE34_ARM='+arm,'FLASH_NATIVE33_DIAG=1')
    assert settings==(PARENT/'runs/q18a_native33_settings.sh').read_text()
    declaration=(root/'runs'/(stem+'_declare.sh')).read_text().replace(stem,'q18a_native33').replace(name,'q18a-native33')
    assert declaration==(PARENT/'runs/q18a_native33_declare.sh').read_text()
    original=PREV_RUNNER.read_text()
    expected=original.replace('q18a_native33',stem).replace('q18a-native33',name)
    expected=expected.replace('from native31_gate import window_ok','from native34_gate import window_ok')
    expected=expected.replace("result.get('tests_run',0)==61","result.get('tests_run',0)==90 and result.get('final_input_proof') is True")
    expected=replace(expected,dict(H="Path('/home/<service-user>/harness/"+b['manifest'][:12]+"')",BUNDLE=repr(b['manifest']),REVIEW=repr(b['review_sha256']),EXPECTED_SOURCES=repr(ov),LAUNCHER=repr(ov['run-flash-s5-test.sh'])))
    assert expected==Path(b['runner']).read_text()
    result=dict(schema='native34-independent-freeze-proof/1',verdict='PASS',classification=None,
        arm=arm,operational_order=['I','S','O'],manifest=b['manifest'],runner_sha256=b['runner_sha256'],
        parent_manifest=sha(PARENT/'SHA256SUMS'),checksummed_files=len(rows),
        engine_py_so_changed=[],all_parent_plugins_byte_identical=True,
        overlay_metadata_changed=changed,overlay_added=sorted(additions),
        runner_only_registered_deltas=True,
        runner_deltas=['arm names','identity pins','90-test exact-container requirement','final_input_proof guard','fresh native34_gate import'],
        queue_preconditioning_and_restore_byte_equal=True,settings_only_names_and_diagnostic_role=True,declaration_only_names=True,
        queue_registered_deltas=['arm names','causal row4 collector','causal diagnostic grade','fresh gate import'],
        prior_builder_proof=str(build_path.parent/'freeze-proof.json'),
        prior_builder_proof_note='its inherited runner summary label omits the window-import/final-proof deltas; exact runner assertion above names and verifies both',
        cpu_only_fixtures_outside_worker_overlay=True,gpu_execution=False,box_action=False)
    assert not out.exists();dump(out,result);print(json.dumps(dict(arm=arm,verdict=result['verdict'],out=str(out),sha256=sha(out))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();check(a.build,a.out)
