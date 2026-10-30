#!/usr/bin/env python3
"""Row4 CLI JSON repair refreeze; exact parent engine/plugin/queue behavior."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'patches/s5_engine'))
from native_readiness import digest,payload_identity,verify_finalized_package
from build_native30_held import sha,dump,manifest,verify,replace

def build(parent,parent_runner,out,variant):
    oldname='q18a-native30-localize-'+variant+'2';name='q18a-native30-localize-'+variant+'3'
    oldcount,newcount=(18,20) if variant=='c' else (27,29)
    oldscript=oldname.replace('-','_');script=name.replace('-','_')
    previous=verify(parent);assert not out.exists()
    out.mkdir();bundle=out/'bundle';shutil.copytree(parent,bundle)
    for p in [bundle,*bundle.rglob('*')]:p.chmod(0o755 if p.is_dir() else 0o644)
    tracked=subprocess.check_output(['git','ls-files','harness'],cwd=ROOT,text=True).splitlines()
    changed=[];added=[]
    for item in tracked:
        source=ROOT/item;rel=str(Path(item).relative_to('harness'));target=bundle/rel
        if not source.is_file():continue
        if target.exists() and sha(source)==sha(target):continue
        (changed if target.exists() else added).append(rel)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    assert set(changed)=={'e3_native30_row4.py','native30_fixture_leaves.json','test_native30_localize.py','test_native30_c.py','test_native30_identity.py'},changed
    assert set(added)==({'runs/'+script+s for s in ('.draft.sh','_declare.sh','_settings.sh')}|{'test_native30_cli.py'}),added
    gate=(bundle/'native30_gate.py').read_text()
    assert gate==(parent/'native30_gate.py').read_text()
    queue_proof={}
    for suffix in ('.draft.sh','_declare.sh','_settings.sh'):
        old=(parent/'runs'/(oldscript+suffix)).read_text();fresh=(bundle/'runs'/(script+suffix)).read_text()
        normalized=fresh.replace(script,oldscript).replace(name,oldname)
        assert normalized==old,suffix
        queue_proof[script+suffix]=dict(sha256=sha(bundle/'runs'/(script+suffix)),normalized_parent_equal=True)
        if suffix=='.draft.sh':
            for fn in ('stop_engine','restore_launcher','restore_defaults','write_cleanup'):
                body=lambda s:s.split(fn+'() {',1)[1].split('\n}\n',1)[0]
                assert body(fresh)==body(old),fn
    tool_changed=[]
    for tool in ('native30_owner_functions.py','native30_cpu_probe.py','native30_selfrun.py'):
        target=bundle/'native30_tools'/tool
        if sha(ROOT/'tools'/tool)!=sha(target):tool_changed.append('native30_tools/'+tool)
        shutil.copyfile(ROOT/'tools'/tool,target)
    assert set(tool_changed)=={'native30_tools/native30_owner_functions.py','native30_tools/native30_cpu_probe.py'}
    old_driver=(parent/'e3_native30_row4.py').read_text()
    assert (bundle/'e3_native30_row4.py').read_text()==old_driver.replace('_workload_object(args.workload)','_workload_object(json.loads(Path(args.workload).read_text()))')
    # Preserve all inherited qualification receipts, explicitly label the waived rerun.
    overlay=bundle/'native30_overlay';closure=json.loads((overlay/'e3/native-implementation-closure.json').read_text())
    review=closure['review']
    window=dict(parent_manifest=sha(parent/'SHA256SUMS'),candidate=name,not_before='2026-10-30T15:00:00Z',hard_end='2026-10-30T23:00:00Z',scope='row4 CLI workload path JSON-loaded before unchanged validator; queue/engine/plugin/gate unchanged; fresh exact-argv CPU boundary tests',full_regression='NOT_RUN: owner explicitly waived full regression for urgent c3/d3 CLI repair; inherited regression is historical',fresh_validation='exact-container CPU focused suite, window boundaries and owner precheck')
    review['row4_cli_rebuild']=window
    for key in ('night_window','native30_localization'):
        review[key]['not_before']='2026-10-30T15:00:00Z'
    review['harness_source_shas']={str(Path(n).relative_to('harness')):sha(ROOT/n) for n in tracked if (ROOT/n).is_file()}
    closure['review_sha256']=digest(review);dump(overlay/'e3/native-implementation-closure.json',closure);dump(bundle/'native30-review.json',review)
    gen=json.loads((overlay/'S5-OVERLAP-GENERATION.json').read_text());ready=json.loads((overlay/'e3/native-overlap-readiness.json').read_text())
    gen['implementation_review_sha256']=ready['review_sha256']=closure['review_sha256']
    gen['package_sha256']=ready['package_sha256']=payload_identity(overlay)
    ready['receipt_sha256']=digest({k:v for k,v in ready.items() if k!='receipt_sha256'})
    dump(overlay/'S5-OVERLAP-GENERATION.json',gen);dump(overlay/'e3/native-overlap-readiness.json',ready);manifest(overlay)
    verify_finalized_package(overlay,closure['generated_source_shas']);new_overlay=verify(overlay);old_overlay=verify(parent/'native30_overlay')
    overlay_diff=[n for n in old_overlay if new_overlay[n]!=old_overlay[n]]
    assert set(new_overlay)==set(old_overlay)
    assert set(overlay_diff)=={'S5-OVERLAP-GENERATION.json','e3/native-overlap-readiness.json','e3/native-implementation-closure.json'},overlay_diff
    provenance=json.loads((bundle/'native30-provenance.json').read_text())
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    provenance.update(source=source,row4_cli_rebuild=window,overlay_manifest=sha(overlay/'SHA256SUMS'),review_sha256=closure['review_sha256'])
    dump(bundle/'native30-provenance.json',provenance);manifest(bundle);pin=sha(bundle/'SHA256SUMS')
    final=out/(name+'-bundle-'+pin[:12]);bundle.rename(final);overlay=final/'native30_overlay'
    original=parent_runner.read_text();owner=original.replace(oldscript,script).replace(oldname,name).replace("result.get('tests_run',0)=="+str(oldcount),"result.get('tests_run',0)=="+str(newcount))
    owner=replace(owner,dict(H="Path('/home/<service-user>/harness/"+pin[:12]+"')",BUNDLE=repr(pin),REVIEW=repr(closure['review_sha256']),EXPECTED_SOURCES=repr(new_overlay)))
    normalized=replace(owner.replace(script,oldscript).replace(name,oldname).replace("result.get('tests_run',0)=="+str(newcount),"result.get('tests_run',0)=="+str(oldcount)),dict(H=ast.get_source_segment(original,next(n.value for n in ast.parse(original).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='H')),BUNDLE=repr(sha(parent/'SHA256SUMS')),REVIEW=repr(json.loads((parent/'native30-review.json').read_text()) and json.loads((parent/'native30_overlay/e3/native-implementation-closure.json').read_text())['review_sha256']),EXPECTED_SOURCES=repr(old_overlay)))
    assert normalized==original,'runner has a nonidentity change'
    runner=out/(name+'-owner-run-'+pin[:12]+'.py');runner.write_text(owner);compile(owner,str(runner),'exec')
    rows=verify(final);diff=[n for n in previous if rows[n]!=previous[n]]
    expected=set(changed)|set(tool_changed)|{'native30-review.json','native30-provenance.json','native30_overlay/SHA256SUMS'}|{'native30_overlay/'+n for n in overlay_diff}
    assert set(diff)==expected,diff
    assert set(rows)-set(previous)==set(added) and not set(previous)-set(rows)
    proof=dict(verdict='PASS',classification=None,source=source,parent_bundle=str(parent),parent_manifest=sha(parent/'SHA256SUMS'),bundle=str(final),manifest=pin,changed=sorted(diff),added=sorted(added),removed=[],queue_normalized_parent_equal=queue_proof,runner_normalized_parent_equal=True,runner_test_count_update=[oldcount,newcount],driver_change='JSON-load workload PATH before unchanged validator',overlay_changed=sorted(overlay_diff),engine_and_plugin_byte_equal=True,fixture_leaves_byte_equal=sha(final/'native30_fixture_leaves.json')==sha(parent/'native30_fixture_leaves.json'),window=window,gpu_execution=False,box_action=False)
    dump(out/'freeze-proof.json',proof)
    result=dict(bundle=str(final),manifest=pin,overlay=str(overlay),overlay_manifest=sha(overlay/'SHA256SUMS'),runner=str(runner),runner_sha256=sha(runner),runner_size=runner.stat().st_size,review_sha256=closure['review_sha256'],checksummed_files=len(rows),held=True,name=name,source=source,identity_timeout_s=360)
    dump(out/'build.json',result)
    for p in [final,*final.rglob('*'),runner]:p.chmod(0o555 if p.is_dir() else 0o444)
    print(json.dumps(result,sort_keys=True));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--parent-runner',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--variant',choices=('c','d'),required=True);a=p.parse_args();build(a.parent,a.parent_runner,a.out,a.variant)
