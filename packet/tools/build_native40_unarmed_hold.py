#!/usr/bin/env python3
"""Regenerate native40 and its exact held-b reference from the pinned BASE.
Native39 held-b supplies the unchanged compiled libraries/packaging shell.
Adds the short box queue and owner runner; no model settings or device action.
"""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'patches/s5_engine'),str(ROOT/'harness')]
from build_native30_held import sha,dump,manifest,verify
from build_native38_plain_bind import regression_from,focused_from
from native_readiness import digest,payload_identity,verify_finalized_package
PARENT='1ffc394783d3d67e140056868b50ca2f74c88b3b'
SHELL=Path('<hub-dir>/agents/flash-next/tmp/native39-trunk-held-b/q18a-native39-trunk-bundle-756f826c221c')
SHELL_SHA='756f826c221c026440d7955a9f834fe72db83581439fed9bb69d07d314ea4264'
BASE=Path('<hub-dir>/agents/flash-next/tmp/e2f-native-offline-base-20261027T211405Z')
BASE_SHA='da561411fadbb82922eeb8eafbe9424bc9703aaf8c2030bd813743c692b641a5'
TOOLS=('build_native40_unarmed_hold.py','native40_unarmed_freeze.py','native40_unarmed_declarations.json','native40_unarmed_selfrun.py',
       'native40_box_queue.py','native40_box_cpu_probe.py','native40_box_selfrun.py')

def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True)
def paths(commit=None):return git('ls-tree','-r','--name-only',commit,'harness','tools','patches/s5_engine').splitlines() if commit else git('ls-files','harness','tools','patches/s5_engine').splitlines()
def source(commit,name):return subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT) if commit else (ROOT/name).read_bytes()
def review_for(bundle,regression,focused,label):
    p=bundle/'e3_builder/native_candidate_review.json'
    r=json.loads(p.read_text())
    r['source_shas']={n:sha(bundle/'e3_builder'/n) for n in r['source_shas']}
    r['harness_source_shas']={str(Path(n).relative_to('harness')):sha(bundle/Path(n).relative_to('harness'))
                             for n in paths(PARENT if label=='parent' else None) if n.startswith('harness/')}
    if label!='parent':
        r['cpu_regression']=regression;r['post_review_cpu']=focused
        r['native40_unarmed_hold']=dict(base=PARENT,scope='Physical split-bank stock-prefill exclusion, no automatic native admission; existing 30s idle FULL return covers unarmed holds',
            fixture_leaves='native40_fixture_leaves.json',device_execution='NOT_RUN',model_settings_changed=False,tail_handback=False,
            targeted_queue=['startup','bind','unarmed-stock-prefill','row1','one-fresh-W2'],speed_claim=False)
    dump(p,r);dump(bundle/'native30-review.json',r)
    return r

def prepare_review(strict,focused):
    review=json.loads((ROOT/'patches/s5_engine/native_candidate_review.json').read_text())
    review['source_shas']={n:sha(ROOT/'patches/s5_engine'/n) for n in review['source_shas']}
    review['harness_source_shas']={str(Path(n).relative_to('harness')):sha(ROOT/n) for n in paths() if n.startswith('harness/')}
    reg,focus=regression_from(strict),focused_from(focused)
    reg['basis']='fresh full strict native40 CPU regression';focus['basis']='native40 and affected native39/native38/admission/driver/bridge/storage suites'
    review['cpu_regression']=reg;review['post_review_cpu']=focus
    review['native40_unarmed_hold']=dict(base=PARENT,scope='Physical split-bank stock-prefill exclusion, no automatic native admission; existing 30s idle FULL return covers unarmed holds',
        fixture_leaves='native40_fixture_leaves.json',device_execution='NOT_RUN',model_settings_changed=False,tail_handback=False,
            targeted_queue=['startup','bind','unarmed-stock-prefill','row1','one-fresh-W2'],speed_claim=False)
    return review

def stage(out,label,regression,focused):
    parent=label=='parent';bundle=out/label
    shutil.copytree(SHELL,bundle)
    for p in [bundle,*bundle.rglob('*')]:p.chmod(0o755 if p.is_dir() else 0o644)
    copied=[]
    for n in paths(PARENT if parent else None):
        if n.startswith('harness/'):target=bundle/Path(n).relative_to('harness')
        elif n.startswith('patches/s5_engine/'):target=bundle/'e3_builder'/Path(n).relative_to('patches/s5_engine')
        else:target=bundle/'native35_tools'/Path(n).relative_to('tools')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source(PARENT if parent else None,n));copied.append(n)
    review=review_for(bundle,regression,focused,label)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.pathsep.join(map(str,(bundle/'e3_builder',bundle/'e2_builder',bundle))))
    raw,final=out/(label+'-raw'),out/(label+'-final')
    with (out/(label+'-generation.log')).open('x') as f:
        subprocess.run(['python3','-B',str(bundle/'e3_builder/make_s5_overlap.py'),str(BASE),str(raw),'--reference',str(bundle/'e3_reference'),'--base-manifest-sha256',BASE_SHA,'--native-candidate'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    for n in ('flash_s5_green.so','flash_s5_replay_observer.so','q17-helper-build.json'):shutil.copyfile(SHELL/'native30_overlay/e3/native'/n,raw/'e3/native'/n)
    with (out/(label+'-finalize.log')).open('x') as f:
        subprocess.run(['python3','-B',str(bundle/'e3_finalize_package.py'),str(raw),str(final)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    ov=bundle/'native30_overlay'
    for p in final.rglob('*'):
        if p.is_file():
            t=ov/p.relative_to(final);t.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,t)
    gen=json.loads((ov/'S5-OVERLAP-GENERATION.json').read_text());ready=json.loads((ov/'e3/native-overlap-readiness.json').read_text())
    gen['package_sha256']=ready['package_sha256']=payload_identity(ov)
    ready['receipt_sha256']=digest({k:v for k,v in ready.items() if k!='receipt_sha256'})
    dump(ov/'S5-OVERLAP-GENERATION.json',gen);dump(ov/'e3/native-overlap-readiness.json',ready);manifest(ov)
    closure=json.loads((ov/'e3/native-implementation-closure.json').read_text())
    verify_finalized_package(ov,closure['generated_source_shas'])
    assert closure['review']==review
    # No archived engine/model/plugin files may change except the two native39
    # sources plus native40's two sources, generated core scheduler and metadata.
    allowed={'e3/engine_driver.py','e3/scheduler_hooks.py','core/scheduler.py',
             'core/SHA256SUMS','S5-OVERLAP-GENERATION.json','e3/native-implementation-closure.json',
             'e3/native-overlap-readiness.json','e3/decode-probe-evidence.json'}
    before,after=verify(SHELL/'native30_overlay'),verify(ov)
    assert set(before)==set(after)
    changed={n for n in before if before[n]!=after[n]}
    assert changed<=allowed,sorted(changed-allowed)
    if parent:
        assert not changed & {'e3/engine_driver.py','e3/scheduler_hooks.py','core/scheduler.py'}, \
            'regenerated reference must have the exact held-b engine bytes'
    ancestry=json.loads((bundle/'native30-provenance.json').read_text())
    provenance=dict(schema='native40-provenance/1',
        candidate='q18a-native40-unarmed-hold' if not parent else 'native39-1c334bc-reference',
        source=git('rev-parse',PARENT if parent else 'HEAD').strip(),overlay_manifest=sha(ov/'SHA256SUMS'),
        review_sha256=closure['review_sha256'],parent_source=PARENT,
        owner_runner='q18a-native40-unarmed-hold-owner-run-<SUMS12>.py' if not parent else None,box_ready=not parent,
        native35_arm='TRUNK',
        parent_bundle_manifest=SHELL_SHA,ancestral_parent_provenance=ancestry,
        overlay_engine_changed_vs_parent=sorted(changed-{'core/SHA256SUMS','S5-OVERLAP-GENERATION.json',
            'e3/native-implementation-closure.json','e3/native-overlap-readiness.json','e3/decode-probe-evidence.json'}),
        engine_changed_vs_parent=[] if parent else ['core/scheduler.py','e3/engine_driver.py','e3/scheduler_hooks.py'],
        selftest=None if parent else regression,post_review_cpu=None if parent else focused,
        model_settings_changed=False,tail_handback=False,
        queue='runs/q18a_native40_unarmed_hold.draft.sh' if not parent else None,gpu_work=False,box_action=False)
    dump(bundle/'native30-provenance.json',provenance);manifest(bundle)
    return bundle

def build(out,strict,focused):
    import native40_unarmed_freeze as freeze
    assert not out.exists()
    assert not git('status','--porcelain','--','harness','tools','patches/s5_engine').strip(),'commit source before freeze'
    assert sha(SHELL/'SHA256SUMS')==SHELL_SHA and sha(BASE/'SHA256SUMS')==BASE_SHA
    verify(SHELL);verify(BASE)
    prepared=prepare_review(strict,focused)
    assert json.loads((ROOT/'patches/s5_engine/native_candidate_review.json').read_text())==prepared,'prepare and commit review before freeze'
    reg,focus=prepared['cpu_regression'],prepared['post_review_cpu']
    import native40_box_queue
    for n,text in native40_box_queue.derive().items():assert (ROOT/n).read_text()==text,n
    out.mkdir()
    parent=stage(out,'parent',reg,focus)
    bundle=stage(out,'candidate',reg,focus)
    source_proof=freeze.source_level()
    overlay_proof=freeze.overlay_level(parent/'native30_overlay',bundle/'native30_overlay')
    proof=dict(verdict='PASS',source=source_proof,overlay=overlay_proof,negative_controls=freeze.negative_controls(),
        branch_base=git('rev-parse',PARENT).strip(),parent_bundle_sha256=SHELL_SHA,compiled_libraries_and_plugins_byte_identical=True,
        parent_overlay_regenerated_from_exact_base=True,model_settings_changed=False,tail_handback=False)
    pin=sha(bundle/'SHA256SUMS');held=out/('q18a-native40-unarmed-hold-bundle-'+pin[:12]);bundle.rename(held)
    import build_native39_trunk as parent_build
    runner_text,_=parent_build.derive_runner(pin,
        json.loads((held/'native30_overlay/e3/native-implementation-closure.json').read_text())['review_sha256'],
        dict(sorted(verify(held/'native30_overlay').items())))
    runner_text=runner_text.replace('q18a-native39-trunk','q18a-native40-unarmed-hold').replace('q18a_native39_trunk','q18a_native40_unarmed_hold')
    runner_text=runner_text.replace('from native39_trunk_gate import trunk_window_ok as window_ok',
                                    'from native40_box_gate import window_ok')
    runner_text=runner_text.replace('==%d and result.get'%parent_build.expected_tests(),
                                    '==%d and result.get'%(parent_build.expected_tests()+native40_box_queue.TEST_CASES),1)
    begin=runner_text.index('        # Native39: the engine self-run')
    end=runner_text.index("'exact-engine-selfrun-required')",begin)+len("'exact-engine-selfrun-required')")
    runner_text=runner_text[:begin]+'''        # Native40: exact frozen Core/hold/typed-ACK proof, distinct from queue-leaf tests.
        engine=json.loads(a.engine_selfrun_record.read_text())
        require(engine.get('schema')=='native40-selfrun/1' and engine.get('manifest')==BUNDLE
                and engine.get('verdict')=='PASS' and engine.get('negative_control') is False
                and engine.get('container_rc')==0 and (engine.get('suites') or {}).get('ok') is True
                and (engine.get('suites') or {}).get('skips')==0
                and (engine.get('connected') or {}).get('verdict')=='PASS'
                and engine.get('gpu_device_requests')==0 and engine.get('network')=='none'
                and engine.get('image')==IMAGE and engine.get('box_action') is False,'exact-engine-selfrun-required')'''+runner_text[end:]
    compile(runner_text,'native40-owner','exec')
    runner=out/('q18a-native40-unarmed-hold-owner-run-'+pin[:12]+'.py');runner.write_text(runner_text)
    parent_runner=SHELL.parent/('q18a-native39-trunk-owner-run-'+SHELL_SHA[:12]+'.py')
    assert parent_runner.is_file(),'held-b owner runner required'
    old_runner,_=parent_build.derive_runner(SHELL_SHA,
        json.loads((SHELL/'native30_overlay/e3/native-implementation-closure.json').read_text())['review_sha256'],
        dict(sorted(verify(SHELL/'native30_overlay').items())))
    assert old_runner==parent_runner.read_text(),'held-b runner derivation differs'
    before,after=verify(SHELL),verify(held)
    assert set(before)<=set(after),'no parent files may be removed'
    proof.update(manifest=pin,parent_manifest=SHELL_SHA,parent_bundle=str(SHELL),
        bundle_changed=sorted(n for n in before if before[n]!=after[n]),bundle_added=sorted(set(after)-set(before)),
        bundle_removed=[],runner=dict(parent=str(parent_runner),parent_sha256=sha(parent_runner),sha256=sha(runner),
            differences=[line for line in difflib.ndiff(old_runner.splitlines(),runner_text.splitlines())
                         if line.startswith(('- ','+ ')) and not line[2:].startswith('EXPECTED_SOURCES=')]),
        queue_derivation='native40_box_queue.derive(): targeted startup/bind/unarmed/row1/single-fresh-W2, shared phase budget',
        active_s=__import__('native40_box_gate').ACTIVE_S,cleanup_margin_s=__import__('native40_box_gate').CLEANUP_S)
    dump(out/'freeze-proof.json',proof)
    result=dict(schema='native40-build/1',name='q18a-native40-unarmed-hold',arm='TRUNK',verdict='PASS',source=git('rev-parse','HEAD').strip(),branch_base=git('rev-parse',PARENT).strip(),
        bundle=str(held),manifest=pin,overlay=str(held/'native30_overlay'),overlay_manifest=sha(held/'native30_overlay/SHA256SUMS'),
        reference_bundle=str(parent),reference_overlay_manifest=sha(parent/'native30_overlay/SHA256SUMS'),
        regression=reg,focused=focus,files=len(verify(held)),freeze_proof_sha256=sha(out/'freeze-proof.json'),
        held=True,box_ready=True,runner=str(runner),runner_sha256=sha(runner),runner_size=runner.stat().st_size,
        expected_selfrun_tests=parent_build.expected_tests()+native40_box_queue.TEST_CASES,
        review_sha256=json.loads((held/'native30_overlay/e3/native-implementation-closure.json').read_text())['review_sha256'],
        image=parent_build.IMAGE,checksummed_files=len(verify(held)),
        queue=str(held/'runs/q18a_native40_unarmed_hold.draft.sh'),gpu_execution=False)
    dump(out/'build.json',result)
    for d in (held,parent):
        for p in [*d.rglob('*'),d]:p.chmod(0o555 if p.is_dir() else 0o444)
    runner.chmod(0o444)
    print(json.dumps(result,sort_keys=True));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--selftest-log',type=Path,required=True);p.add_argument('--focused-log',type=Path,required=True);a=p.parse_args();build(a.out.resolve(),a.selftest_log.resolve(),a.focused_log.resolve())
