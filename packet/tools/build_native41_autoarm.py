#!/usr/bin/env python3
"""Regenerate native41 and its exact native40 (f6d1db9) reference from the pinned BASE.

Native40's delivered bundle supplies the unchanged compiled libraries and the
packaging shell. Adds the native41 auto-arm queue and its owner runner: the
runner is native40's runner (re-derived here and proven byte-identical to the
delivered one) with the named native41 swaps. No model setting, launcher
change or device action.
"""
import argparse
import difflib
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
PARENT='f6d1db9c90e7bffa5423945a7148a5bb800e247c'
SHELL=Path('<hub-dir>/agents/flash-next/tmp/native40-unarmed-hold-held-b/rebuild-b/q18a-native40-unarmed-hold-bundle-d2ae6c8b07f1')
SHELL_SHA='d2ae6c8b07f131b51c3d112b9fae1082080199a64159085d64cabac51693b685'
SHELL_RUNNER_SHA='baeca063c937431b3dcdcf2a9d84849a1c78deab95026c4bddb1abe10880ad10'
BASE=Path('<hub-dir>/agents/flash-next/tmp/e2f-native-offline-base-20261027T211405Z')
BASE_SHA='da561411fadbb82922eeb8eafbe9424bc9703aaf8c2030bd813743c692b641a5'
CANDIDATE='q18a-native41-autoarm'
STEM='q18a_native41_autoarm'
TOOLS=('build_native41_autoarm.py','native41_autoarm_freeze.py','native41_autoarm_declarations.json','native41_autoarm_selfrun.py',
       'native41_box_queue.py','native41_box_cpu_probe.py','native41_box_selfrun.py')
SCOPE=('Opt-in engine-owned FCFS automatic native cycles for plain requests (request binding, real drain, typed route '
       'ACKs, one-request FULL handback, manual lease exclusion), the auto-arm actions on the native router only with '
       'FLASH_S5_AUTO_ARM=1, and a read-only auto window; default-off unchanged')
TARGETED=['startup','bind','unarmed-stock-prefill','plain-W2 stock-0','auto-0','stock-1','auto-1','typed-FULL']
META={'core/SHA256SUMS','S5-OVERLAP-GENERATION.json','e3/native-implementation-closure.json',
      'e3/native-overlap-readiness.json','e3/decode-probe-evidence.json'}
# Native40's runner engine-proof block, verbatim from build_native40_unarmed_hold.build.
NATIVE40_ENGINE_BLOCK='''        # Native40: exact frozen Core/hold/typed-ACK proof, distinct from queue-leaf tests.
        engine=json.loads(a.engine_selfrun_record.read_text())
        require(engine.get('schema')=='native40-selfrun/1' and engine.get('manifest')==BUNDLE
                and engine.get('verdict')=='PASS' and engine.get('negative_control') is False
                and engine.get('container_rc')==0 and (engine.get('suites') or {}).get('ok') is True
                and (engine.get('suites') or {}).get('skips')==0
                and (engine.get('connected') or {}).get('verdict')=='PASS'
                and engine.get('gpu_device_requests')==0 and engine.get('network')=='none'
                and engine.get('image')==IMAGE and engine.get('box_action') is False,'exact-engine-selfrun-required')'''

def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True)
def paths(commit=None):return git('ls-tree','-r','--name-only',commit,'harness','tools','patches/s5_engine').splitlines() if commit else git('ls-files','harness','tools','patches/s5_engine').splitlines()
def source(commit,name):return subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT) if commit else (ROOT/name).read_bytes()

def runtime():
    from native41_autoarm_freeze import RUNTIME,ADDED
    return RUNTIME,ADDED

def details():
    return dict(base=PARENT,scope=SCOPE,fixture_leaves='native41_fixture_leaves.json',device_execution='NOT_RUN',
                model_settings_changed=False,tail_handback=False,launcher_changed=False,targeted_queue=TARGETED,speed_claim=False)

def review_for(bundle,regression,focused,label):
    p=bundle/'e3_builder/native_candidate_review.json'
    r=json.loads(p.read_text())
    r['source_shas']={n:sha(bundle/'e3_builder'/n) for n in r['source_shas']}
    r['harness_source_shas']={str(Path(n).relative_to('harness')):sha(bundle/Path(n).relative_to('harness'))
                             for n in paths(PARENT if label=='parent' else None) if n.startswith('harness/')}
    if label!='parent':
        r['cpu_regression']=regression;r['post_review_cpu']=focused
        r['native41_autoarm']=details()
    dump(p,r);dump(bundle/'native30-review.json',r)
    return r

def prepare_review(strict,focused):
    review=json.loads((ROOT/'patches/s5_engine/native_candidate_review.json').read_text())
    review['source_shas']={n:sha(ROOT/'patches/s5_engine'/n) for n in review['source_shas']}
    review['source_shas']['native_auto_arm.py']=sha(ROOT/'patches/s5_engine/native_auto_arm.py')
    review['harness_source_shas']={str(Path(n).relative_to('harness')):sha(ROOT/n) for n in paths() if n.startswith('harness/')}
    reg,focus=regression_from(strict),focused_from(focused)
    reg['basis']='fresh full strict native41 CPU regression';focus['basis']='native41 and affected native40/native39/native38/admission/driver/bridge/storage/router suites'
    review['cpu_regression']=reg;review['post_review_cpu']=focus
    review['native41_autoarm']=details()
    return review

def native40_runner(pin,review_sha,sources):
    """Native40's owner runner for these pins, as build_native40_unarmed_hold.build derives it."""
    import build_native39_trunk as parent_build
    import native40_box_queue
    text,_=parent_build.derive_runner(pin,review_sha,sources)
    text=text.replace('q18a-native39-trunk','q18a-native40-unarmed-hold').replace('q18a_native39_trunk','q18a_native40_unarmed_hold')
    text=text.replace('from native39_trunk_gate import trunk_window_ok as window_ok','from native40_box_gate import window_ok')
    text=text.replace('==%d and result.get'%parent_build.expected_tests(),
                      '==%d and result.get'%(parent_build.expected_tests()+native40_box_queue.TEST_CASES),1)
    begin=text.index('        # Native39: the engine self-run')
    end=text.index("'exact-engine-selfrun-required')",begin)+len("'exact-engine-selfrun-required')")
    return text[:begin]+NATIVE40_ENGINE_BLOCK+text[end:]

def expected_tests():
    import build_native39_trunk as parent_build
    import native40_box_queue
    import native41_box_queue
    return parent_build.expected_tests()+native40_box_queue.TEST_CASES+native41_box_queue.NATIVE41_QUEUE_TESTS

def engine_cases():
    from native41_autoarm_selfrun import CASES
    return sorted(CASES)

def runner_swaps():
    import build_native39_trunk as parent_build
    import native40_box_queue
    n40=parent_build.expected_tests()+native40_box_queue.TEST_CASES
    return (('q18a-native40-unarmed-hold',CANDIDATE),('q18a_native40_unarmed_hold',STEM),
            ('from native40_box_gate import window_ok','from native41_box_gate import window_ok'),
            ("tests_run',0)==%d and"%n40,"tests_run',0)==%d and"%expected_tests()),
            ('        # Native40: exact frozen Core/hold/typed-ACK proof, distinct from queue-leaf tests.',
             '        # Native41: exact frozen connected auto-arm and read-only window cases, distinct from queue-leaf tests.'),
            ("engine.get('schema')=='native40-selfrun/1'","engine.get('schema')=='native41-selfrun/1'"),
            ("(engine.get('connected') or {}).get('verdict')=='PASS'",
             "sorted(engine.get('cases') or {})==%r\n                and all((c or {}).get('verdict')=='PASS' for c in engine['cases'].values())"%engine_cases()))

def derive_runner(pin,review_sha,sources):
    old=native40_runner(pin,review_sha,sources);text=old
    for a,b in runner_swaps():
        assert text.count(a)>=1,a
        if a.startswith('q18a'):text=text.replace(a,b)
        else:
            assert text.count(a)==1,a
            text=text.replace(a,b)
    compile(text,CANDIDATE+'-owner-run','exec')
    back=text
    for a,b in reversed(runner_swaps()):back=back.replace(b,a)
    assert back==old,'unexpected native41 runner change'
    return text,old

def stage(out,label,regression,focused):
    parent=label=='parent';bundle=out/label
    shutil.copytree(SHELL,bundle)
    for p in [bundle,*bundle.rglob('*')]:p.chmod(0o755 if p.is_dir() else 0o644)
    for n in paths(PARENT if parent else None):
        if n.startswith('harness/'):target=bundle/Path(n).relative_to('harness')
        elif n.startswith('patches/s5_engine/'):target=bundle/'e3_builder'/Path(n).relative_to('patches/s5_engine')
        else:target=bundle/'native35_tools'/Path(n).relative_to('tools')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source(PARENT if parent else None,n))
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
    # Only native41's runtime sources, the added policy module and metadata may differ from native40's
    # delivered overlay; the regenerated reference must carry native40's exact engine bytes.
    RUNTIME,ADDED=runtime()
    before,after=verify(SHELL/'native30_overlay'),verify(ov)
    added=set(after)-set(before)
    assert not set(before)-set(after),'no native40 overlay file may be removed'
    assert added==(set() if parent else ADDED),sorted(added)
    changed={n for n in before if before[n]!=after[n]}
    assert changed<=(META if parent else RUNTIME|META),sorted(changed-(META if parent else RUNTIME|META))
    assert after['run-flash-s5-test.sh']==before['run-flash-s5-test.sh'],'launcher must stay byte-identical'
    ancestry=json.loads((bundle/'native30-provenance.json').read_text())
    provenance=dict(schema='native41-provenance/1',
        candidate=CANDIDATE if not parent else 'native40-f6d1db9-reference',
        source=git('rev-parse',PARENT if parent else 'HEAD').strip(),overlay_manifest=sha(ov/'SHA256SUMS'),
        review_sha256=closure['review_sha256'],parent_source=PARENT,
        owner_runner=CANDIDATE+'-owner-run-<SUMS12>.py' if not parent else None,box_ready=not parent,
        native35_arm='TRUNK',parent_bundle_manifest=SHELL_SHA,ancestral_parent_provenance=ancestry,
        overlay_engine_changed_vs_parent=sorted((changed|added)-META-{'SHA256SUMS'}),
        engine_changed_vs_parent=[] if parent else sorted(RUNTIME|ADDED),
        selftest=None if parent else regression,post_review_cpu=None if parent else focused,
        model_settings_changed=False,tail_handback=False,launcher_changed=False,
        queue='runs/'+STEM+'.draft.sh' if not parent else None,gpu_work=False,box_action=False)
    dump(bundle/'native30-provenance.json',provenance);manifest(bundle)
    return bundle

def build(out,strict,focused):
    import native41_autoarm_freeze as freeze
    import native41_box_queue
    import native41_box_gate
    assert not out.exists()
    assert not git('status','--porcelain','--','harness','tools','patches/s5_engine').strip(),'commit source before freeze'
    assert sha(SHELL/'SHA256SUMS')==SHELL_SHA and sha(BASE/'SHA256SUMS')==BASE_SHA
    verify(SHELL);verify(BASE)
    prepared=prepare_review(strict,focused)
    assert json.loads((ROOT/'patches/s5_engine/native_candidate_review.json').read_text())==prepared,'prepare and commit review before freeze'
    reg,focus=prepared['cpu_regression'],prepared['post_review_cpu']
    for n,text in (native41_box_queue.derive()|native41_box_queue.derive_tools()).items():assert (ROOT/n).read_text()==text,n
    # The native40 runner derivation reproduces native40's delivered runner byte for byte.
    shell_runner=SHELL.parent/('q18a-native40-unarmed-hold-owner-run-'+SHELL_SHA[:12]+'.py')
    assert sha(shell_runner)==SHELL_RUNNER_SHA
    shell_review=json.loads((SHELL/'native30_overlay/e3/native-implementation-closure.json').read_text())['review_sha256']
    assert native40_runner(SHELL_SHA,shell_review,dict(sorted(verify(SHELL/'native30_overlay').items())))==shell_runner.read_text(),\
        'native40 runner derivation differs from the delivered runner'
    out.mkdir()
    parent=stage(out,'parent',reg,focus)
    bundle=stage(out,'candidate',reg,focus)
    source_proof=freeze.source_level()
    overlay_proof=freeze.overlay_level(parent/'native30_overlay',bundle/'native30_overlay')
    proof=dict(verdict='PASS',source=source_proof,overlay=overlay_proof,negative_controls=freeze.negative_controls(),
        branch_base=git('rev-parse',PARENT).strip(),parent_bundle_sha256=SHELL_SHA,compiled_libraries_and_plugins_byte_identical=True,
        parent_overlay_regenerated_from_exact_base=True,model_settings_changed=False,tail_handback=False,launcher_changed=False)
    pin=sha(bundle/'SHA256SUMS');held=out/(CANDIDATE+'-bundle-'+pin[:12]);bundle.rename(held)
    review_sha=json.loads((held/'native30_overlay/e3/native-implementation-closure.json').read_text())['review_sha256']
    runner_text,native40_text=derive_runner(pin,review_sha,dict(sorted(verify(held/'native30_overlay').items())))
    runner=out/(CANDIDATE+'-owner-run-'+pin[:12]+'.py');runner.write_text(runner_text)
    before,after=verify(SHELL),verify(held)
    assert set(before)<=set(after),'no parent files may be removed'
    proof.update(manifest=pin,parent_manifest=SHELL_SHA,parent_bundle=str(SHELL),
        bundle_changed=sorted(n for n in before if before[n]!=after[n]),bundle_added=sorted(set(after)-set(before)),
        bundle_removed=[],runner=dict(parent=str(shell_runner),parent_sha256=SHELL_RUNNER_SHA,sha256=sha(runner),
            native40_derivation_reproduces_delivered_runner=True,
            differences=[line for line in difflib.ndiff(native40_text.splitlines(),runner_text.splitlines())
                         if line.startswith(('- ','+ '))]),
        queue_derivation='native41_box_queue.derive(): native40 startup/bind/unarmed, four plain W2 arms, typed FULL, one phase budget',
        active_s=native41_box_gate.ACTIVE_S,cleanup_margin_s=native41_box_gate.CLEANUP_S)
    dump(out/'freeze-proof.json',proof)
    result=dict(schema='native41-build/1',name=CANDIDATE,arm='TRUNK',verdict='PASS',source=git('rev-parse','HEAD').strip(),branch_base=git('rev-parse',PARENT).strip(),
        bundle=str(held),manifest=pin,overlay=str(held/'native30_overlay'),overlay_manifest=sha(held/'native30_overlay/SHA256SUMS'),
        reference_bundle=str(parent),reference_overlay_manifest=sha(parent/'native30_overlay/SHA256SUMS'),
        regression=reg,focused=focus,files=len(verify(held)),freeze_proof_sha256=sha(out/'freeze-proof.json'),
        held=True,box_ready=True,runner=str(runner),runner_sha256=sha(runner),runner_size=runner.stat().st_size,
        expected_selfrun_tests=expected_tests(),engine_cases=engine_cases(),review_sha256=review_sha,
        image=__import__('build_native39_trunk').IMAGE,checksummed_files=len(verify(held)),
        queue=str(held/'runs'/(STEM+'.draft.sh')),gpu_execution=False,automatic_native_admission=True,
        auto_arm_setting='FLASH_S5_AUTO_ARM=1 in the queue EXTRA_ENV only')
    dump(out/'build.json',result)
    for d in (held,parent):
        for p in [*d.rglob('*'),d]:p.chmod(0o555 if p.is_dir() else 0o444)
    runner.chmod(0o444)
    print(json.dumps(result,sort_keys=True));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--selftest-log',type=Path,required=True);p.add_argument('--focused-log',type=Path,required=True)
    p.add_argument('--prepare-review',action='store_true',help='write the prepared review into patches/s5_engine and stop')
    a=p.parse_args()
    if a.prepare_review:
        dump(ROOT/'patches/s5_engine/native_candidate_review.json',prepare_review(a.selftest_log.resolve(),a.focused_log.resolve()))
    else:build(a.out.resolve(),a.selftest_log.resolve(),a.focused_log.resolve())
