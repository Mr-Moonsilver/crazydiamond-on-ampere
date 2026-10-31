#!/usr/bin/env python3
"""Native35 R/O/L held bundles: parent is the frozen native34 I4 held-a bundle.

Unlike native31-34, the engine source changes: exactly the native35 fix files
(private QSA metadata ownership) and the two diagnostic plugins that carry the
evidence-only arm-O readback. Everything else in the overlay, every compiled
library and every parent plugin stays byte-identical to I4 and is asserted so.
"""
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
PARENT=Path('<hub-dir>/agents/flash-next/tmp/native34-i4-held-a/q18a-native34-i4-bundle-c2fdad02b7b1')
PARENT_MANIFEST='c2fdad02b7b1c4444138ea267f07948cbd586136600434dedb647462e28cef66'
PARENT_RUNNER=PARENT.parent/'q18a-native34-i4-owner-run-c2fdad02b7b1.py'
BASE=Path('<hub-dir>/agents/flash-next/tmp/e2f-native-offline-base-20261027T211405Z')
BASE_SHA='da561411fadbb82922eeb8eafbe9424bc9703aaf8c2030bd813743c692b641a5'
ARMS={'R':'q18a-native35-r','O':'q18a-native35-o','L':'q18a-native35-l'}
ORDER=['R','O','L']
# The native35 fix (Codex, codex/native35) and the evidence-only readback plugins (Fable).
EXPECTED_ENGINE_CHANGED={'e3/lane_chunk.py','e3/lane_dispatch.py','e3/native_startup.py','e3/s5_startup.py'}
EXPECTED_PLUGIN_CHANGED={'e3/native30_plugins/native34_worker.py','e3/native30_plugins/native34_readback.py'}
PLUGINS=('native34_diag.py','native34_worker.py','native34_control.py','native34_storage.py','native34_indexer.py','native34_readback.py')
NATIVE35_TOOLS=('native35_selfrun.py','native35_cpu_probe.py','native35_builder_view_cpu_check.py')
sys.path[:0]=[str(ROOT/'patches/s5_engine'),str(ROOT/'tools'),str(ROOT/'harness')]
from build_native30_held import sha,dump,manifest,verify,replace
from native_readiness import digest,payload_identity,verify_finalized_package

OLD_GATE="result.get('tests_run',0)==98 and result.get('final_input_proof') is True and result.get('draft_window_slots')=='PASS' and result.get('native34_protocol')=='PASS' and result.get('native34_readback')=='PASS'"


def expected_tests():
    tree=ast.parse((ROOT/'tools/native35_cpu_probe.py').read_text())
    values=[node.value.value for node in tree.body if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='EXPECTED_TESTS']
    assert len(values)==1 and type(values[0]) is int;return values[0]


def new_gate(arm):
    return OLD_GATE.replace("==98","==%d"%expected_tests())+" and result.get('native35_builder')=='PASS' and result.get('native35_arm')=="+repr(arm)


def regression_from(log):
    text=log.read_text();summary=re.findall(r'^selftest: (\d+)/(\d+) cases as required$',text,re.M)
    suites=[json.JSONDecoder().raw_decode(text[m.end():])[0] for m in re.finditer('selftest suite: ',text)]
    assert len(summary)==1 and summary[0][0]==summary[0][1] and suites and all(s['returncode']==0 and not s['skipped'] for s in suites),'strict regression log must be complete, all-pass, zero skips'
    return dict(cases=int(summary[0][0]),required=int(summary[0][1]),exit_code=0,skips=0,suites=len(suites),reported_tests=sum(s['tests_run'] or 0 for s in suites),log_sha256=sha(log),basis='fresh full strict registered CPU regression on the native35 final tree')


def focused_from(log):
    text=log.read_text();ran=re.findall(r'^Ran (\d+) tests? in ',text,re.M)
    assert ran and text.rstrip().endswith('OK') and 'skipped' not in text.rsplit('OK',1)[-1]
    return dict(tests=sum(int(n) for n in ran),skips=0,verdict='PASS',log_sha256=sha(log),basis='native35 suites plus the native34 readback/runtime and e3 lane chunk/dispatch suites on the final tree')


def source_head():
    dirty=subprocess.check_output(['git','status','--porcelain','--','harness','tools','patches/s5_engine'],cwd=ROOT,text=True)
    assert not dirty,'commit harness/tools/patches before building:\n'+dirty
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def details(arm):
    return dict(parent_manifest=PARENT_MANIFEST,native35=True,engine_changed=sorted(EXPECTED_ENGINE_CHANGED),plugins_changed=sorted(EXPECTED_PLUGIN_CHANGED),
        arms=ORDER,arm=arm,candidate=ARMS[arm],diagnostic='R: unfenced O-style arm with evidence-only readback and native-private writer catalogue; O: plain O-style deciding arm; L: native lifetime, hygiene and leak gates (FLASH_NATIVE34_DIAG=0)',
        fix='native-owned QSA metadata-builder storage via a native-only group view; ordinary builder path unchanged; one metadata generation at a time; private buffers allocated once at startup and ledgered per rank',
        owner_prelaunch_cpu_executed=True,vm_input='NONE',sync='none',not_before='2026-10-30T22:01:00Z',hard_end='2026-11-02T06:00:00Z',
        active_s=7200 if arm=='L' else 2100,cleanup_margin_s=420,
        phase_sequence=['startup','native35-ledger','row1','rule4_controls','rule4_candidate','row4-placement','lifecycle','hygiene-full','cross-session-leak','pressure/partial-leak','pressure-lifetime'] if arm=='L'
                       else ['startup','native35-ledger','row1','rule4_controls','rule4_candidate','row4-control-A','bounded-native-prefix'],
        ple_repair=False,device_run='NOT_RUN',tp_score_identity='UNPROVEN',gpu_fault_mechanism='CANDIDATE_REPAIR_UNPROVEN_ON_DEVICE',rule4_causality='UNPROVEN',
        fixture_leaves=['native31_fixture_leaves.json','native35_fixture_leaves.json'],numerical_fix=False,causal_repair_candidate=True)


def prepare_review(strict_log,focused_log):
    review=json.loads((ROOT/'patches/s5_engine/native_candidate_review.json').read_text())
    tracked=subprocess.check_output(['git','ls-files','harness'],cwd=ROOT,text=True).splitlines()
    review['source_shas']={n:sha(ROOT/'patches/s5_engine'/n) for n in review['source_shas']}
    review['harness_source_shas']={str(Path(n).relative_to('harness')):sha(ROOT/n) for n in tracked if (ROOT/n).is_file()}
    review['cpu_regression']=regression_from(strict_log)
    review['post_review_cpu']=focused_from(focused_log)
    review['native35']=dict(details('R'),arm=None,candidate=None,active_s={'R':2100,'O':2100,'L':7200},
        fresh_names=[ARMS[a] for a in ORDER],window_source='native34_gate.window_ok(time.time()) for R/O; native35_gate.lifetime_window_ok(time.time()) for L',
        inline_dates=False,base_manifest=PARENT_MANIFEST,readback='arm O evidence-only copies without drain or fence; never repairs or gates kernel inputs',
        builder_repro='pinned CPU builder twice: old shared storage refuses (request ids 1-3 out of bounds); native-private view passes with ordinary bytes equal to ordinary-alone; real pinned builder class cloned through the shipped private_attention_groups')
    review['night_window']=dict(not_before='2026-10-30T22:01:00Z',hard_end='2026-11-02T06:00:00Z',queue_timeout_s=2100,lifetime_queue_timeout_s=7200,cleanup_margin_s=420,placeholder=False)
    return review


def refresh_tools(bundle):
    """Every bundled tool copy equals the committed tools/ tree the precheck compares against."""
    for directory in sorted(bundle.glob('native3*_tools')):
        for path in sorted(directory.glob('*.py')):
            source=ROOT/'tools'/path.name
            if source.is_file():shutil.copyfile(source,path)


def build(out,strict_log,focused_log,arm):
    assert arm in ARMS;candidate=ARMS[arm];stem=candidate.replace('-','_')
    assert not out.exists();assert sha(PARENT/'SHA256SUMS')==PARENT_MANIFEST
    previous=verify(PARENT);source=source_head()
    subprocess.run([sys.executable,'-B',str(ROOT/'tools/native35_queue_drafts.py'),'--check'],check=True,cwd=ROOT)
    regression=regression_from(strict_log)
    out.mkdir();bundle=out/'bundle';shutil.copytree(PARENT,bundle)
    for path in [bundle,*bundle.rglob('*')]:path.chmod(0o755 if path.is_dir() else 0o644)
    tracked=subprocess.check_output(['git','ls-files','harness','patches/s5_engine'],cwd=ROOT,text=True).splitlines()
    for name in tracked:
        src=ROOT/name
        if not src.is_file():continue
        target=bundle/Path(name).relative_to('harness') if name.startswith('harness/') else bundle/'e3_builder'/Path(name).relative_to('patches/s5_engine')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)
    tools=bundle/'native35_tools';shutil.copytree(PARENT/'native34_tools',tools)
    for path in [tools,*tools.rglob('*')]:path.chmod(0o755 if path.is_dir() else 0o644)
    for name in NATIVE35_TOOLS:shutil.copyfile(ROOT/'tools'/name,tools/name)
    refresh_tools(bundle)
    # Frozen oracle fixtures are inherited from the parent and re-pinned to the reference tree.
    fixtures=bundle/'native31-fixtures'
    for name in ('qsa.py','ops/qsa.py'):
        assert sha(fixtures/'reference/models/qwen4_exp/nvidia'/name)==sha(ROOT/'reference/vllm-src/models/qwen4_exp/nvidia'/name),'CPU generator reference pin differs'
    review_path=bundle/'e3_builder/native_candidate_review.json';review=prepare_review(strict_log,focused_log)
    assert json.loads(review_path.read_text())==review,'prepare and commit the native35 source review before freeze'
    review['source_shas']={n:sha(bundle/'e3_builder'/n) for n in review['source_shas']}
    review['harness_source_shas']={str(Path(n).relative_to('harness')):sha(ROOT/n) for n in tracked if n.startswith('harness/') and (ROOT/n).is_file()}
    dump(review_path,review);dump(bundle/'native30-review.json',review)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.pathsep.join(map(str,(bundle/'e3_builder',bundle/'e2_builder',bundle))))
    raw=out/'raw';final=out/'final'
    command=['python3','-B',str(bundle/'e3_builder/make_s5_overlap.py'),str(BASE),str(raw),'--reference',str(bundle/'e3_reference'),'--base-manifest-sha256',BASE_SHA,'--native-candidate']
    with (out/'generation.log').open('x') as f:subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    # Python-only fix: retain the exact owner-qualified compiled libraries.
    for name in ('flash_s5_green.so','flash_s5_replay_observer.so','q17-helper-build.json'):
        shutil.copyfile(PARENT/'native30_overlay/e3/native'/name,raw/'e3/native'/name)
    with (out/'finalize.log').open('x') as f:subprocess.run(['python3','-B',str(bundle/'e3_finalize_package.py'),str(raw),str(final)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    overlay=bundle/'native30_overlay'
    for path in final.rglob('*'):
        if path.is_file():
            target=overlay/path.relative_to(final);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    for name in ('native30_host.py','native30_policy.py',*PLUGINS):
        shutil.copyfile(ROOT/'harness'/name,overlay/'e3/native30_plugins'/name)
    gen=json.loads((overlay/'S5-OVERLAP-GENERATION.json').read_text());ready=json.loads((overlay/'e3/native-overlap-readiness.json').read_text())
    gen['package_sha256']=ready['package_sha256']=payload_identity(overlay)
    ready['receipt_sha256']=digest({k:v for k,v in ready.items() if k!='receipt_sha256'})
    dump(overlay/'S5-OVERLAP-GENERATION.json',gen);dump(overlay/'e3/native-overlap-readiness.json',ready);manifest(overlay)
    closure=json.loads((overlay/'e3/native-implementation-closure.json').read_text())
    verify_finalized_package(overlay,closure['generated_source_shas'])
    oldov=verify(PARENT/'native30_overlay');newov=verify(overlay)
    assert set(newov)==set(oldov),(set(newov)^set(oldov))
    changed=[n for n in oldov if newov[n]!=oldov[n]]
    engine_changed=sorted(n for n in changed if n.endswith(('.py','.so')))
    assert set(engine_changed)==EXPECTED_ENGINE_CHANGED|EXPECTED_PLUGIN_CHANGED,engine_changed
    assert not any(n.endswith('.so') for n in changed)
    metadata_changed=sorted(set(changed)-set(engine_changed))
    assert set(metadata_changed)<={'S5-OVERLAP-GENERATION.json','e3/native-implementation-closure.json','e3/native-overlap-readiness.json','e3/decode-probe-evidence.json'},metadata_changed
    # The decode-probe evidence lists generated module shas; only the four fix modules may move.
    old_probe=json.loads((PARENT/'native30_overlay/e3/decode-probe-evidence.json').read_text());new_probe=json.loads((overlay/'e3/decode-probe-evidence.json').read_text())
    assert {k for k in set(old_probe)|set(new_probe) if old_probe.get(k)!=new_probe.get(k)}<={'generated_source_shas'},'decode-probe evidence changed outside generated_source_shas'
    probe_moved={k for k in set(old_probe['generated_source_shas'])|set(new_probe['generated_source_shas']) if old_probe['generated_source_shas'].get(k)!=new_probe['generated_source_shas'].get(k)}
    assert probe_moved=={Path(n).stem for n in EXPECTED_ENGINE_CHANGED},probe_moved
    fix_files=subprocess.check_output(['git','diff','--name-only',parent_source(),'HEAD','--','patches/s5_engine'],cwd=ROOT,text=True).split()
    assert {'e3/'+Path(n).name for n in fix_files if n.endswith('.py') and Path(n).name!='native_candidate_review.json'}==EXPECTED_ENGINE_CHANGED,fix_files
    for name in ('e3/lane_chunk.py','e3/lane_dispatch.py','e3/native_startup.py','e3/s5_startup.py'):
        assert closure['generated_source_shas'][name[3:-3]]==hashlib.sha256((overlay/name).read_text().encode()).hexdigest()
    provenance=json.loads((bundle/'native30-provenance.json').read_text())
    provenance.update(source=source,candidate=candidate,native35_arm=arm,operational_order=ORDER,selftest=regression,native35=details(arm),
        overlay_manifest=sha(overlay/'SHA256SUMS'),review_sha256=closure['review_sha256'],parent_bundle_manifest=PARENT_MANIFEST,engine_changed=engine_changed)
    dump(bundle/'native30-provenance.json',provenance);manifest(bundle);pin=sha(bundle/'SHA256SUMS');held=out/(candidate+'-bundle-'+pin[:12]);bundle.rename(held)
    owner=PARENT_RUNNER.read_text().replace('q18a_native34_i4',stem).replace('q18a-native34-i4',candidate)
    if arm=='L':owner=owner.replace('from native34_gate import window_ok','from native35_gate import lifetime_window_ok as window_ok')
    assert owner.count(OLD_GATE)==1;owner=owner.replace(OLD_GATE,new_gate(arm))
    owner=replace(owner,dict(H="Path('/home/<service-user>/harness/"+pin[:12]+"')",BUNDLE=repr(pin),REVIEW=repr(closure['review_sha256']),EXPECTED_SOURCES=repr(newov),LAUNCHER=repr(newov['run-flash-s5-test.sh'])))
    runner=out/(candidate+'-owner-run-'+pin[:12]+'.py');runner.write_text(owner);compile(owner,str(runner),'exec')
    original=PARENT_RUNNER.read_text()
    normalized=owner.replace(stem,'q18a_native34_i4').replace(candidate,'q18a-native34-i4').replace('from native35_gate import lifetime_window_ok as window_ok','from native34_gate import window_ok').replace(new_gate(arm),OLD_GATE)
    assignments={node.targets[0].id:ast.get_source_segment(original,node.value) for node in ast.parse(original).body if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)}
    assert replace(normalized,{key:assignments[key] for key in ('H','BUNDLE','REVIEW','EXPECTED_SOURCES','LAUNCHER')})==original,'unexpected runner body change'
    assert all(sha(path)==sha(held/'vmid_sources'/path.relative_to(PARENT/'vmid_sources')) for path in (PARENT/'vmid_sources').rglob('*') if path.is_file()),'historical VM source snapshot changed'
    rows=verify(held)
    proof=dict(verdict='PASS',classification=None,source=source,parent_bundle=str(PARENT),parent_manifest=PARENT_MANIFEST,manifest=pin,engine_changed=engine_changed,overlay_changed=sorted(changed),
        changed=sorted(n for n in previous if n in rows and rows[n]!=previous[n]),added=sorted(set(rows)-set(previous)),removed=sorted(set(previous)-set(rows)),
        compiled_libraries_byte_identical=True,parent_plugins_byte_identical_except=sorted(EXPECTED_PLUGIN_CHANGED),historical_vm_snapshots_byte_identical=True,
        runner_only_names_pins_count_gate_and_l_window=True,queue_preconditioning_and_restore_unchanged=True,fresh_weekend_gate=True,gpu_execution=False,box_action=False,details=details(arm))
    dump(out/'freeze-proof.json',proof)
    result=dict(name=candidate,arm=arm,bundle=str(held),manifest=pin,overlay=str(held/'native30_overlay'),overlay_manifest=sha(held/'native30_overlay/SHA256SUMS'),runner=str(runner),runner_sha256=sha(runner),runner_size=runner.stat().st_size,review_sha256=closure['review_sha256'],checksummed_files=len(rows),source=source,held=True,regression=regression,expected_selfrun_tests=expected_tests())
    dump(out/'build.json',result)
    for path in [held,*held.rglob('*'),runner]:path.chmod(0o555 if path.is_dir() else 0o444)
    print(json.dumps(result,sort_keys=True));return result


def parent_source():
    return json.loads((PARENT/'native30-provenance.json').read_text())['source']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.add_argument('--arm',choices=tuple(ARMS));p.add_argument('--selftest-log',type=Path,required=True);p.add_argument('--focused-log',type=Path,required=True);p.add_argument('--prepare-review',action='store_true');a=p.parse_args()
    if a.prepare_review:dump(ROOT/'patches/s5_engine/native_candidate_review.json',prepare_review(a.selftest_log,a.focused_log))
    else:
        assert a.out is not None and a.arm is not None;build(a.out,a.selftest_log,a.focused_log,a.arm)
