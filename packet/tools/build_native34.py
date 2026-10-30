#!/usr/bin/env python3
"""Native34: native33 engine unchanged; independently frozen I/S/O causal arms."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
PARENT=Path('<hub-dir>/agents/flash-next/tmp/native33-held-a/q18a-native33-bundle-8b5fe8528dc9')
PARENT_RUNNER=PARENT.parent/'q18a-native33-owner-run-8b5fe8528dc9.py'
BASE=Path('<hub-dir>/agents/flash-next/tmp/e2f-native-offline-base-20261027T211405Z')
BASE_SHA='da561411fadbb82922eeb8eafbe9424bc9703aaf8c2030bd813743c692b641a5'
sys.path[:0]=[str(ROOT/'patches/s5_engine'),str(ROOT/'tools'),str(ROOT/'harness')]
from build_native30_held import sha,dump,manifest,verify,replace
from native_readiness import digest,payload_identity,verify_finalized_package


def prepare_review(log):
    text=log.read_text();summary=re.findall(r'^selftest: (\d+)/(\d+) cases as required$',text,re.M)
    suites=[json.JSONDecoder().raw_decode(text[m.end():])[0] for m in re.finditer('selftest suite: ',text)]
    assert len(summary)==1 and summary[0][0]==summary[0][1] and suites and all(s['returncode']==0 and not s['skipped'] for s in suites)
    regression=dict(cases=int(summary[0][0]),required=int(summary[0][1]),exit_code=0,skips=0,suites=len(suites),reported_tests=sum(s['tests_run'] or 0 for s in suites),log_sha256=sha(log),basis='fresh native34 strict regression, real review fixture supplied')
    review=json.loads((ROOT/'patches/s5_engine/native_candidate_review.json').read_text())
    tracked=subprocess.check_output(['git','ls-files','harness'],cwd=ROOT,text=True).splitlines()
    review['source_shas']={n:sha(ROOT/'patches/s5_engine'/n) for n in review['source_shas']}
    review['harness_source_shas']={str(Path(n).relative_to('harness')):sha(ROOT/n) for n in tracked if (ROOT/n).is_file()}
    details=dict(parent_manifest=sha(PARENT/'SHA256SUMS'),native34=True,engine_byte_identical_to_native33=True,arms=['I','S','O'],diagnostic='I rank1 prior-work/indexer discriminator; S serialized prefix; O original overlap; bounded storage and phase receipts',owner_prelaunch_cpu_executed=True,vm_input='NONE',sync='none',not_before='2026-10-30T22:01:00Z',hard_end='2026-11-02T06:00:00Z',active_s=2100,cleanup_margin_s=420,phase_sequence=['startup','row1','rule4_controls','rule4_candidate','row4-control-A','bounded-native-prefix'],ple_repair=False,device_run='NOT_RUN',tp_score_identity='UNPROVEN',gpu_fault_mechanism='NOT_ESTABLISHED',rule4_causality='UNPROVEN',fixture_leaves='native31_fixture_leaves.json',numerical_fix=False)
    focused=ROOT.parent/'tmp/native34-focused-g.log';focused_text=focused.read_text()
    assert 'Ran 29 tests' in focused_text and focused_text.rstrip().endswith('OK')
    review['post_review_cpu']=dict(tests=29,skips=0,verdict='PASS',log_sha256=sha(focused),basis='rank-specific grading negative and all native34 focused tests rerun after review')
    review['native34']=details;review['cpu_regression']=regression
    review['night_window']=dict(not_before=details['not_before'],hard_end=details['hard_end'],queue_timeout_s=2100,cleanup_margin_s=420,placeholder=False)
    return review


def build(out,log,arm):
    assert arm in ('I','S','O');candidate='q18a-native34-'+arm.lower();stem=candidate.replace('-','_')
    assert not out.exists();assert sha(PARENT/'SHA256SUMS')=='8b5fe8528dc9cd0eae2b784edf97763b08eb317d9b166a7401277545e6504a50'
    previous=verify(PARENT)
    text=log.read_text();summary=re.findall(r'^selftest: (\d+)/(\d+) cases as required$',text,re.M)
    suites=[json.JSONDecoder().raw_decode(text[m.end():])[0] for m in re.finditer('selftest suite: ',text)]
    assert len(summary)==1 and summary[0][0]==summary[0][1] and suites and all(s['returncode']==0 and not s['skipped'] for s in suites)
    regression=dict(cases=int(summary[0][0]),required=int(summary[0][1]),exit_code=0,skips=0,suites=len(suites),reported_tests=sum(s['tests_run'] or 0 for s in suites),log_sha256=sha(log),basis='fresh native34 strict regression, real review fixture supplied')
    out.mkdir();bundle=out/'bundle';shutil.copytree(PARENT,bundle)
    for path in [bundle,*bundle.rglob('*')]:path.chmod(0o755 if path.is_dir() else 0o644)
    tracked=subprocess.check_output(['git','ls-files','harness','patches/s5_engine'],cwd=ROOT,text=True).splitlines()
    for name in tracked:
        source=ROOT/name
        if not source.is_file():continue
        if name.startswith('harness/'):target=bundle/Path(name).relative_to('harness')
        else:target=bundle/'e3_builder'/Path(name).relative_to('patches/s5_engine')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    tools=bundle/'native34_tools';shutil.copytree(PARENT/'native33_tools',tools)
    for path in [tools,*tools.rglob('*')]:path.chmod(0o755 if path.is_dir() else 0o644)
    for name in ('native34_selfrun.py','native34_cpu_probe.py','native31_cpu_check.py'):
        shutil.copyfile(ROOT/'tools'/name,tools/name)
    # Frozen oracle only; never mounted over a worker model module.
    fixtures=bundle/'native31-fixtures';fixtures.mkdir(exist_ok=True);shutil.copyfile(PARENT/'native31-fixtures/qsa-before.py',fixtures/'qsa-before.py')
    for name in ('qsa.py','ops/qsa.py'):
        source=ROOT/'reference/vllm-src/models/qwen4_exp/nvidia'/name
        expected=__import__('resident_attention').BASE_SHA if name.startswith('ops/') else sha(bundle/'e3_reference/models/qwen4_exp/nvidia/qsa.py')
        assert sha(source)==expected,'CPU generator reference pin differs'
        target=fixtures/'reference/models/qwen4_exp/nvidia'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    core_fixture=fixtures/'reference/v1/engine/core.py';core_fixture.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(PARENT/'e3_reference/v1/engine/core.py',core_fixture)
    review_path=bundle/'e3_builder/native_candidate_review.json';review=prepare_review(log)
    assert json.loads(review_path.read_text())==review,'prepare and commit the native31 source review before freeze'
    review['source_shas']={n:sha(bundle/'e3_builder'/n) for n in review['source_shas']}
    review['harness_source_shas']={str(Path(n).relative_to('harness')):sha(ROOT/n) for n in tracked if n.startswith('harness/') and (ROOT/n).is_file()}
    details=dict(parent_manifest=sha(PARENT/'SHA256SUMS'),native34=True,engine_byte_identical_to_native33=True,arms=['I','S','O'],diagnostic='I rank1 prior-work/indexer discriminator; S serialized prefix; O original overlap; bounded storage and phase receipts',owner_prelaunch_cpu_executed=True,vm_input='NONE',sync='none',not_before='2026-10-30T22:01:00Z',hard_end='2026-11-02T06:00:00Z',active_s=2100,cleanup_margin_s=420,phase_sequence=['startup','row1','rule4_controls','rule4_candidate','row4-control-A','bounded-native-prefix'],ple_repair=False,device_run='NOT_RUN',tp_score_identity='UNPROVEN',gpu_fault_mechanism='NOT_ESTABLISHED',rule4_causality='UNPROVEN',fixture_leaves='native31_fixture_leaves.json',numerical_fix=False)
    focused=ROOT.parent/'tmp/native34-focused-g.log';focused_text=focused.read_text()
    assert 'Ran 29 tests' in focused_text and focused_text.rstrip().endswith('OK')
    review['post_review_cpu']=dict(tests=29,skips=0,verdict='PASS',log_sha256=sha(focused),basis='rank-specific grading negative and all native34 focused tests rerun after review')
    review['native34']=details;review['cpu_regression']=regression
    review['night_window']=dict(not_before=details['not_before'],hard_end=details['hard_end'],queue_timeout_s=2100,cleanup_margin_s=420,placeholder=False)
    dump(review_path,review);dump(bundle/'native30-review.json',review)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.pathsep.join(map(str,(bundle/'e3_builder',bundle/'e2_builder',bundle))))
    raw=out/'raw';final=out/'final'
    commands=[['python3','-B',str(bundle/'e3_builder/make_s5_overlap.py'),str(BASE),str(raw),'--reference',str(bundle/'e3_reference'),'--base-manifest-sha256',BASE_SHA,'--native-candidate']]
    with (out/'generation.log').open('x') as f:subprocess.run(commands[0],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    # No native source/ABI change: retain exact owner-qualified CPU-build libs.
    for name in ('flash_s5_green.so','flash_s5_replay_observer.so','q17-helper-build.json'):
        shutil.copyfile(PARENT/'native30_overlay/e3/native'/name,raw/'e3/native'/name)
    with (out/'finalize.log').open('x') as f:subprocess.run(['python3','-B',str(bundle/'e3_finalize_package.py'),str(raw),str(final)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    overlay=bundle/'native30_overlay'
    for path in final.rglob('*'):
        if path.is_file():
            target=overlay/path.relative_to(final);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    for name in ('native30_host.py','native30_policy.py'):
        shutil.copyfile(ROOT/'harness'/name,overlay/'e3/native30_plugins'/name)
    plugins=('native34_diag.py','native34_worker.py','native34_control.py','native34_storage.py','native34_indexer.py')
    for plugin in plugins:shutil.copyfile(ROOT/'harness'/plugin,overlay/'e3/native30_plugins'/plugin)
    dist=overlay/'e3/native30_plugins/flashnext_native34_diagnostic-0.1.dist-info';dist.mkdir()
    (dist/'METADATA').write_text('Metadata-Version: 2.1\nName: flashnext_native34_diagnostic\nVersion: 0.1\n')
    (dist/'entry_points.txt').write_text('[vllm.general_plugins]\nnative34_diagnostic = native34_diag:install\n')
    gen=json.loads((overlay/'S5-OVERLAP-GENERATION.json').read_text());ready=json.loads((overlay/'e3/native-overlap-readiness.json').read_text())
    gen['package_sha256']=ready['package_sha256']=payload_identity(overlay)
    ready['receipt_sha256']=digest({k:v for k,v in ready.items() if k!='receipt_sha256'})
    dump(overlay/'S5-OVERLAP-GENERATION.json',gen);dump(overlay/'e3/native-overlap-readiness.json',ready);manifest(overlay)
    closure=json.loads((overlay/'e3/native-implementation-closure.json').read_text())
    verify_finalized_package(overlay,closure['generated_source_shas'])
    oldov=verify(PARENT/'native30_overlay');newov=verify(overlay)
    changed=[n for n in oldov if newov[n]!=oldov[n]]
    engine_changed=[n for n in changed if n.endswith(('.py','.so'))]
    assert not engine_changed,engine_changed
    assert set(newov)-set(oldov)=={'e3/native30_plugins/'+n for n in plugins}|{
        'e3/native30_plugins/flashnext_native34_diagnostic-0.1.dist-info/METADATA',
        'e3/native30_plugins/flashnext_native34_diagnostic-0.1.dist-info/entry_points.txt'}
    assert not set(oldov)-set(newov)
    for path in (PARENT/'native30_overlay/e3/native30_plugins').rglob('*'):
        if path.is_file():assert sha(path)==sha(overlay/'e3/native30_plugins'/path.relative_to(PARENT/'native30_overlay/e3/native30_plugins'))
    from test_native34_queue import normalize_queue
    assert normalize_queue((bundle/'runs'/(stem+'.draft.sh')).read_text(),arm)==(PARENT/'runs/q18a_native33.draft.sh').read_text()
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    provenance=json.loads((bundle/'native30-provenance.json').read_text());provenance.update(source=source,selftest=regression,native34=details,overlay_manifest=sha(overlay/'SHA256SUMS'),review_sha256=closure['review_sha256'])
    dump(bundle/'native30-provenance.json',provenance);manifest(bundle);pin=sha(bundle/'SHA256SUMS');held=out/(candidate+'-bundle-'+pin[:12]);bundle.rename(held)
    owner=PARENT_RUNNER.read_text().replace('q18a_native33',stem).replace('q18a-native33',candidate)
    owner=owner.replace('from native31_gate import window_ok','from native34_gate import window_ok')
    owner=owner.replace("result.get('tests_run',0)==61","result.get('tests_run',0)==90 and result.get('final_input_proof') is True")
    owner=replace(owner,dict(H="Path('/home/<service-user>/harness/"+pin[:12]+"')",BUNDLE=repr(pin),REVIEW=repr(closure['review_sha256']),EXPECTED_SOURCES=repr(newov),LAUNCHER=repr(newov['run-flash-s5-test.sh'])))
    runner=out/(candidate+'-owner-run-'+pin[:12]+'.py');runner.write_text(owner);compile(owner,str(runner),'exec')
    original=PARENT_RUNNER.read_text()
    normalized_owner=owner.replace(stem,'q18a_native33').replace(candidate,'q18a-native33').replace('from native34_gate import window_ok','from native31_gate import window_ok').replace("result.get('tests_run',0)==90 and result.get('final_input_proof') is True","result.get('tests_run',0)==61")
    assignments={node.targets[0].id:ast.get_source_segment(original,node.value) for node in ast.parse(original).body if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)}
    assert replace(normalized_owner,{key:assignments[key] for key in ('H','BUNDLE','REVIEW','EXPECTED_SOURCES','LAUNCHER')})==original,'unexpected runner body change'
    assert all(sha(path)==sha(bundle_path) for path in (PARENT/'vmid_sources').rglob('*') if path.is_file() for bundle_path in [held/'vmid_sources'/path.relative_to(PARENT/'vmid_sources')]),'historical VM source snapshot changed'
    rows=verify(held)
    proof=dict(verdict='PASS',classification=None,source=source,parent_bundle=str(PARENT),parent_manifest=sha(PARENT/'SHA256SUMS'),manifest=pin,engine_changed=engine_changed,overlay_changed=changed,changed=sorted(n for n in previous if rows[n]!=previous[n]),added=sorted(set(rows)-set(previous)),removed=sorted(set(previous)-set(rows)),engine_byte_identical_to_native33=True,all_parent_plugins_byte_identical=True,changed_plugins=[],added_plugins=list(plugins),historical_vm_snapshots_byte_identical=True,runner_only_names_pins_test_count_and_sync_none=True,queue_preconditioning_and_restore_unchanged=True,fresh_weekend_gate=True,bounded_row4_causal_driver=True,restore_bodies_equal=True,gpu_execution=False,box_action=False,details=details)
    dump(out/'freeze-proof.json',proof)
    result=dict(name=candidate,arm=arm,bundle=str(held),manifest=pin,overlay=str(held/'native30_overlay'),overlay_manifest=sha(held/'native30_overlay/SHA256SUMS'),runner=str(runner),runner_sha256=sha(runner),runner_size=runner.stat().st_size,review_sha256=closure['review_sha256'],checksummed_files=len(rows),source=source,held=True,regression=regression)
    dump(out/'build.json',result)
    for path in [held,*held.rglob('*'),runner]:path.chmod(0o555 if path.is_dir() else 0o444)
    print(json.dumps(result,sort_keys=True));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.add_argument('--arm',choices=('I','S','O'),default='I');p.add_argument('--selftest-log',type=Path,required=True);p.add_argument('--prepare-review',action='store_true');a=p.parse_args()
    if a.prepare_review:dump(ROOT/'patches/s5_engine/native_candidate_review.json',prepare_review(a.selftest_log))
    else:
        assert a.out is not None;build(a.out,a.selftest_log,a.arm)
