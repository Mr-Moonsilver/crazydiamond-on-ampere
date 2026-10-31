#!/usr/bin/env python3
"""Native35 L2 held bundle: parent is the frozen native35 L held-a bundle.

L2 = L plus the row4 resume-receipt window collector (e3_native35_l2_placement.py),
its queue under fresh names, its tests/fixture leaves and its self-run tools.
The overlay (engine, plugins, compiled libraries, generation/readiness/closure
metadata), the review JSON, e3_builder and every parent harness file stay
byte-identical to L and are asserted so; the overlay is not regenerated.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
PARENT=Path('<hub-dir>/agents/flash-next/tmp/native35-l-held-a/q18a-native35-l-bundle-1cdf24641e15')
PARENT_MANIFEST='1cdf24641e1542d545e8f9b1c97c3703b46a4baac6e077b135dbdd74a31de328'
PARENT_OVERLAY_MANIFEST='32673c374f8b258df54d229eac8bcd456cd633ca51715c939a55f8615194591f'
PARENT_REVIEW='3f8f7201bd71ffecdd13df9c85f429efa4850a53cf4fb44701ba985a9a8c58a9'
PARENT_RUNNER=PARENT.parent/'q18a-native35-l-owner-run-1cdf24641e15.py'
PARENT_RUNNER_SHA='fdcc028225b84b6d8b70c706ea6af038d2e9d353a4e84cbfb42cd681fc09ed40'
PARENT_SOURCE='7bc7bf0846aa8dcef6020baedf6970d800f3b5cf'
CANDIDATE='q18a-native35-l2';STEM='q18a_native35_l2';ARM='L2'
PARENT_CANDIDATE='q18a-native35-l';PARENT_STEM='q18a_native35_l'
HARNESS_ADDED={'e3_native35_l2_placement.py','test_native35_l2.py','native35_l2_fixture_leaves.json',
               'runs/q18a_native35_l2.draft.sh','runs/q18a_native35_l2_declare.sh','runs/q18a_native35_l2_settings.sh'}
L2_TOOLS=('native35_l2_cpu_probe.py','native35_l2_selfrun.py','native35_l2_tools_derive.py')
BUNDLE_CHANGED={'native30-provenance.json'}
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'harness')]
from build_native30_held import sha,dump,manifest,verify,replace

OLD_GATE=("result.get('tests_run',0)==125 and result.get('final_input_proof') is True and result.get('draft_window_slots')=='PASS' and result.get('native34_protocol')=='PASS' and result.get('native34_readback')=='PASS' and result.get('native35_builder')=='PASS' and result.get('native35_arm')=='L'")


def expected_tests():
    tree=ast.parse((ROOT/'tools/native35_l2_cpu_probe.py').read_text())
    values=[node.value.value for node in tree.body if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='EXPECTED_TESTS']
    assert len(values)==1 and type(values[0]) is int;return values[0]


def new_gate():
    return OLD_GATE.replace("==125","==%d"%expected_tests()).replace("result.get('native35_arm')=='L'","result.get('native35_arm')=='L2' and result.get('native35_l2')=='PASS'")


def regression_from(log):
    text=log.read_text();summary=re.findall(r'^selftest: (\d+)/(\d+) cases as required$',text,re.M)
    suites=[json.JSONDecoder().raw_decode(text[m.end():])[0] for m in re.finditer('selftest suite: ',text)]
    assert len(summary)==1 and summary[0][0]==summary[0][1] and suites and all(s['returncode']==0 and not s['skipped'] for s in suites),'strict regression log must be complete, all-pass, zero skips'
    return dict(cases=int(summary[0][0]),required=int(summary[0][1]),exit_code=0,skips=0,suites=len(suites),reported_tests=sum(s['tests_run'] or 0 for s in suites),log_sha256=sha(log),basis='fresh full strict registered CPU regression on the native35 L2 final tree')


def focused_from(log):
    # One unittest run: its 'Ran N tests' / 'OK' block, no skip/failure lines, then the shell rc line.
    text=log.read_text();ran=re.findall(r'^Ran (\d+) tests? in [0-9.]+s\n\nOK$',text,re.M)
    assert len(ran)==1 and not re.search(r'\.\.\. skipped|^FAILED|^FAIL:|^ERROR:',text,re.M) and text.rstrip().endswith('rc=0'),'focused log must be one all-pass run with zero skips'
    return dict(tests=int(ran[0]),skips=0,verdict='PASS',log_sha256=sha(log),basis='test_native35_l2 plus the placement-driver, decode-correctness, native35 queue and native30 CLI suites on the final tree')


def source_head():
    dirty=subprocess.check_output(['git','status','--porcelain','--','harness','tools','patches/s5_engine'],cwd=ROOT,text=True)
    assert not dirty,'commit harness/tools/patches before building:\n'+dirty
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def details():
    return dict(parent_manifest=PARENT_MANIFEST,parent_candidate=PARENT_CANDIDATE,parent_source=PARENT_SOURCE,arm=ARM,candidate=CANDIDATE,
        change='row4 client window opens at the accepted resume receipt, then the native prefill open-wait (ResumeWindowPlacementRunner, e3_native35_l2_placement.py); every other phase, workload, bound, grader and the overlay byte-identical to L',
        diagnosis='reviews/claude-20261031-native35-l-row4-window.md: L row4 INCONCLUSIVE placement-window-incomplete was a HARNESS ordering defect',
        overlay_byte_identical=True,overlay_manifest=PARENT_OVERLAY_MANIFEST,engine_changed=[],review_sha256=PARENT_REVIEW,review_unchanged=True,
        review_harness_source_shas='L source tree (overlay generation input); the overlay is not regenerated, so the L2 harness additions are bound by the bundle SHA256SUMS and the precheck harness check',
        settings='identical to L except QNAME (FLASH_NATIVE33_DIAG=0 FLASH_NATIVE34_DIAG=0)',not_before='2026-10-30T22:01:00Z',hard_end='2026-11-02T06:00:00Z',active_s=7200,cleanup_margin_s=420,
        phase_sequence=['startup','native35-ledger','row1','rule4_controls','rule4_candidate','row4-placement (resume-receipt window)','lifecycle','hygiene-full','cross-session-leak','pressure/partial-leak','pressure-lifetime'],
        selftest_registration='test_native35_l2.py runs in the focused log and in the pinned-image self-run probe; harness/selftest.py is unchanged (a queue-run script)',
        fixture_leaves=['native31_fixture_leaves.json','native35_fixture_leaves.json','native35_l2_fixture_leaves.json'],device_run='NOT_RUN',vm_input='NONE')


def build(out,strict_log,focused_log):
    assert not out.exists();assert sha(PARENT/'SHA256SUMS')==PARENT_MANIFEST and sha(PARENT_RUNNER)==PARENT_RUNNER_SHA
    previous=verify(PARENT);parent_overlay=verify(PARENT/'native30_overlay');assert sha(PARENT/'native30_overlay/SHA256SUMS')==PARENT_OVERLAY_MANIFEST
    source=source_head()
    for tool in ('native35_l2_queue_draft.py','native35_l2_tools_derive.py','native35_queue_drafts.py'):
        subprocess.run([sys.executable,'-B',str(ROOT/'tools'/tool),'--check'],check=True,cwd=ROOT)
    # L's harness and engine sources are the L2 tree minus the declared additions.
    for spec in ('harness','patches/s5_engine'):
        moved=subprocess.check_output(['git','diff','--name-status',PARENT_SOURCE,'HEAD','--',spec],cwd=ROOT,text=True).split('\n')
        moved=[line for line in moved if line]
        expect=sorted('A\tharness/'+n for n in HARNESS_ADDED) if spec=='harness' else []
        assert sorted(moved)==expect,(spec,moved)
    regression=regression_from(strict_log);focused=focused_from(focused_log)
    out.mkdir();bundle=out/'bundle';shutil.copytree(PARENT,bundle)
    for path in [bundle,*bundle.rglob('*')]:path.chmod(0o755 if path.is_dir() else 0o644)
    tracked=subprocess.check_output(['git','ls-files','harness','patches/s5_engine'],cwd=ROOT,text=True).splitlines()
    added=[]
    for name in tracked:
        src=ROOT/name
        if not src.is_file():continue
        target=bundle/Path(name).relative_to('harness') if name.startswith('harness/') else bundle/'e3_builder'/Path(name).relative_to('patches/s5_engine')
        if target.exists():assert sha(target)==sha(src),'parent copy differs from the committed tree: '+name
        else:target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target);added.append(str(target.relative_to(bundle)))
    assert set(added)==HARNESS_ADDED,sorted(set(added)^HARNESS_ADDED)
    tools=bundle/'native35_tools'
    for directory in sorted(bundle.glob('native3*_tools')):
        for path in sorted(directory.glob('*.py')):
            if (ROOT/'tools'/path.name).is_file():assert sha(path)==sha(ROOT/'tools'/path.name),'bundled tool differs from committed tools: '+path.name
    for name in L2_TOOLS:
        assert not (tools/name).exists();shutil.copyfile(ROOT/'tools'/name,tools/name);added.append('native35_tools/'+name)
    provenance=json.loads((bundle/'native30-provenance.json').read_text())
    assert provenance['candidate']==PARENT_CANDIDATE and provenance['native35_arm']=='L' and provenance['overlay_manifest']==PARENT_OVERLAY_MANIFEST and provenance['review_sha256']==PARENT_REVIEW
    provenance.update(source=source,candidate=CANDIDATE,native35_arm=ARM,operational_order=['L2'],selftest=regression,post_review_cpu=focused,native35_l2=details(),
        parent_bundle_manifest=PARENT_MANIFEST,engine_changed=[],engine_changed_vs_native34_i4=provenance['engine_changed'])
    dump(bundle/'native30-provenance.json',provenance);manifest(bundle);pin=sha(bundle/'SHA256SUMS')
    held=out/(CANDIDATE+'-bundle-'+pin[:12]);bundle.rename(held)
    overlay=verify(held/'native30_overlay')
    assert overlay==parent_overlay and sha(held/'native30_overlay/SHA256SUMS')==PARENT_OVERLAY_MANIFEST,'overlay must stay byte-identical to L'
    for name in ('native30-review.json','e3_builder/native_candidate_review.json'):assert sha(held/name)==sha(PARENT/name),name
    # Runner: names, H, BUNDLE and the self-run gate line only; REVIEW/EXPECTED_SOURCES/LAUNCHER stay L's.
    original=PARENT_RUNNER.read_text()
    assert CANDIDATE not in original and STEM not in original and original.count(OLD_GATE)==1
    owner=original.replace(PARENT_STEM,STEM).replace(PARENT_CANDIDATE,CANDIDATE).replace(OLD_GATE,new_gate())
    owner=replace(owner,dict(H="Path('/home/<service-user>/harness/"+pin[:12]+"')",BUNDLE=repr(pin)))
    assignments=lambda text:{node.targets[0].id:ast.get_source_segment(text,node.value) for node in ast.parse(text).body if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)}
    old_a,new_a=assignments(original),assignments(owner)
    for key in ('REVIEW','EXPECTED_SOURCES','LAUNCHER','IMAGE','GREEN','OBSERVER'):
        if key in old_a:assert old_a[key]==new_a[key],key
    assert ast.literal_eval(new_a['REVIEW'])==PARENT_REVIEW
    runner=out/(CANDIDATE+'-owner-run-'+pin[:12]+'.py');runner.write_text(owner);compile(owner,str(runner),'exec')
    normalized=replace(owner.replace(new_gate(),OLD_GATE).replace(STEM,PARENT_STEM).replace(CANDIDATE,PARENT_CANDIDATE),{key:old_a[key] for key in ('H','BUNDLE')})
    assert normalized==original,'unexpected runner body change'
    import difflib
    runner_diff=[line for line in difflib.ndiff(original.splitlines(),owner.splitlines()) if line.startswith(('- ','+ '))]
    rows=verify(held)
    changed=sorted(n for n in previous if n in rows and rows[n]!=previous[n]);new=sorted(set(rows)-set(previous));removed=sorted(set(previous)-set(rows))
    assert set(changed)==BUNDLE_CHANGED and set(new)==set(added) and not removed,(changed,new,removed)
    proof=dict(verdict='PASS',classification=None,source=source,parent_bundle=str(PARENT),parent_manifest=PARENT_MANIFEST,parent_runner=str(PARENT_RUNNER),parent_runner_sha256=PARENT_RUNNER_SHA,manifest=pin,
        changed=changed,added=new,removed=removed,overlay_byte_identical=True,overlay_manifest=PARENT_OVERLAY_MANIFEST,overlay_files=len(overlay),engine_changed=[],
        review_files_byte_identical=['native30-review.json','e3_builder/native_candidate_review.json'],review_sha256=PARENT_REVIEW,e3_builder_byte_identical=True,parent_harness_files_byte_identical=True,
        parent_tools_byte_identical=True,historical_vm_snapshots_byte_identical=True,
        runner_changes=dict(names=[PARENT_CANDIDATE+' -> '+CANDIDATE,PARENT_STEM+' -> '+STEM],pins=['H','BUNDLE'],gate=dict(old=OLD_GATE,new=new_gate()),unchanged=['REVIEW','EXPECTED_SOURCES','LAUNCHER','IMAGE'],diff_lines=runner_diff),
        queue_changes='q18a_native35_l2 drafts = L drafts with fresh names plus the row4 CLI branch (tools/native35_l2_queue_draft.py --check)',
        gpu_execution=False,box_action=False,details=details())
    dump(out/'freeze-proof.json',proof)
    result=dict(name=CANDIDATE,arm=ARM,bundle=str(held),manifest=pin,overlay=str(held/'native30_overlay'),overlay_manifest=sha(held/'native30_overlay/SHA256SUMS'),runner=str(runner),runner_sha256=sha(runner),runner_size=runner.stat().st_size,
        review_sha256=PARENT_REVIEW,checksummed_files=len(rows),source=source,held=True,regression=regression,focused=focused,expected_selfrun_tests=expected_tests(),parent_manifest=PARENT_MANIFEST)
    dump(out/'build.json',result)
    for path in [held,*held.rglob('*'),runner]:path.chmod(0o555 if path.is_dir() else 0o444)
    print(json.dumps(result,sort_keys=True));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--selftest-log',type=Path,required=True);p.add_argument('--focused-log',type=Path,required=True);a=p.parse_args()
    build(a.out,a.selftest_log,a.focused_log)
