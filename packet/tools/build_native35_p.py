#!/usr/bin/env python3
"""Native35 P held bundle: the exact frozen native35 O held-a bundle plus the declared speed-arm harness files.

Nothing in the engine changes: the overlay (SHA256SUMS 32673c37...), both review
JSONs, every compiled library and every parent plugin stay byte-identical to O
and are asserted so; the overlay is copied, never regenerated. The bundle adds
exactly the files git adds under harness/ between O's source and this tree
(plus the P probe/self-run tools) and changes exactly selftest.py and
native30-provenance.json. The runner is O's with names, the bundle pins, the
self-run gate and the P window import changed; everything else is asserted equal.
"""
import argparse
import ast
import difflib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
PARENT=Path('<hub-dir>/agents/flash-next/tmp/native35-o-held-a/q18a-native35-o-bundle-188f4d5ca79a')
PARENT_MANIFEST='188f4d5ca79acfb933929eae938d751f70c3fd88637519503bbe4741bc5c039a'
PARENT_OVERLAY_MANIFEST='32673c374f8b258df54d229eac8bcd456cd633ca51715c939a55f8615194591f'
PARENT_RUNNER=PARENT.parent/'q18a-native35-o-owner-run-188f4d5ca79a.py'
PARENT_RUNNER_SHA='760886da3c0ec1994a04bceed6722020ecd903205f44b3d42c7a880ef64f45d8'
PARENT_SOURCE='7bc7bf0846aa8dcef6020baedf6970d800f3b5cf'
IMAGE='sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b'
CANDIDATE='q18a-native35-p'
# Declared bundle delta vs O (bundle-relative paths).
ADDED_HARNESS={'e3_native35_performance.py','native35_performance_grade.py','native35p_gate.py',
               'runs/q18a_native35_p.draft.sh','runs/q18a_native35_p_settings.sh','runs/q18a_native35_p_declare.sh',
               'test_native35_performance.py','test_native35_performance_queue.py','native35_p_fixture_leaves.json'}
CHANGED_HARNESS={'selftest.py'}
SHIPPED_TOOLS=('native35_p_cpu_probe.py','native35_p_selfrun.py')
REGENERATED={'native30-provenance.json'}
O_GATE=("result.get('tests_run',0)==125 and result.get('final_input_proof') is True and result.get('draft_window_slots')=='PASS' "
        "and result.get('native34_protocol')=='PASS' and result.get('native34_readback')=='PASS' and result.get('native35_builder')=='PASS' "
        "and result.get('native35_arm')=='O'")
O_WINDOW='from native34_gate import window_ok'
P_WINDOW='from native35p_gate import speed_window_ok as window_ok'
sys.path[:0]=[str(ROOT/'patches/s5_engine'),str(ROOT/'tools'),str(ROOT/'harness')]
from build_native30_held import sha,dump,manifest,verify,replace
import native35p_gate as window


def expected_tests():
    tree=ast.parse((ROOT/'tools/native35_p_cpu_probe.py').read_text())
    values=[n.value.value for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='EXPECTED_TESTS']
    assert len(values)==1 and type(values[0]) is int;return values[0]


def p_gate():
    return O_GATE.replace('==125','==%d'%expected_tests()).replace("result.get('native35_arm')=='O'","result.get('native35_arm')=='P' and result.get('native35_performance')=='PASS'")


def regression_from(log):
    text=log.read_text();summary=re.findall(r'^selftest: (\d+)/(\d+) cases as required$',text,re.M)
    suites=[json.JSONDecoder().raw_decode(text[m.end():])[0] for m in re.finditer('selftest suite: ',text)]
    assert len(summary)==1 and summary[0][0]==summary[0][1] and suites and all(s['returncode']==0 and not s['skipped'] for s in suites),'strict regression log must be complete, all-pass, zero skips'
    return dict(cases=int(summary[0][0]),required=int(summary[0][1]),exit_code=0,skips=0,suites=len(suites),reported_tests=sum(s['tests_run'] or 0 for s in suites),
                log_sha256=sha(log),basis='fresh full strict registered CPU regression on the native35 P final tree')


def focused_from(log):
    text=log.read_text();ran=re.findall(r'^Ran (\d+) tests? in ',text,re.M)
    assert ran and text.rstrip().endswith('OK') and 'skipped' not in text.rsplit('OK',1)[-1]
    return dict(tests=sum(int(n) for n in ran),skips=0,verdict='PASS',log_sha256=sha(log),
                basis='native35 P suites plus the native35 R/O/L queue, native28 host guard, native measurement client/source and overlap performance suites on the final tree')


def source_head():
    dirty=subprocess.check_output(['git','status','--porcelain','--','harness','tools','patches/s5_engine'],cwd=ROOT,text=True)
    assert not dirty,'commit harness/tools/patches before building:\n'+dirty
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def source_delta(source):
    rows=subprocess.check_output(['git','diff','--name-status',PARENT_SOURCE,source,'--','harness','patches/s5_engine'],cwd=ROOT,text=True).splitlines()
    status={line.split('\t',1)[1]:line.split('\t',1)[0] for line in rows}
    assert all(s in ('A','M') for s in status.values()),status
    added={n[len('harness/'):] for n,s in status.items() if s=='A'};changed={n[len('harness/'):] for n,s in status.items() if s=='M'}
    assert all(n.startswith('harness/') for n in status),status
    assert added==ADDED_HARNESS and changed==CHANGED_HARNESS,(added^ADDED_HARNESS,changed^CHANGED_HARNESS)
    return added,changed


def details(regression,focused):
    return dict(parent='q18a-native35-o',parent_manifest=PARENT_MANIFEST,parent_runner_sha256=PARENT_RUNNER_SHA,parent_source=PARENT_SOURCE,
        arm='P',candidate=CANDIDATE,purpose='speed: decode tok/s during a long native prefill vs W1 FULL solo, and native prefill speed with overlap vs matched split solo',
        engine_changed_vs_parent=[],overlay_manifest=PARENT_OVERLAY_MANIFEST,overlay_byte_identical=True,review_byte_identical=True,
        settings='O settings with FLASH_NATIVE34_DIAG=0 (no FLASH_NATIVE34_ARM, as L) and Q18A_FIRST_ADVANCE_READ_TIMEOUT_S=1200 (native28 declaration; the only value e3_native_performance accepts)',
        localize='FLASH_NATIVE30_LOCALIZE=1 inherited unchanged (per-step receipt lines stay in the measured path)',sync='none',readback=False,
        row4='native33-form real placement (O causal --arm O branch removed, as L)',
        phase_sequence=['startup','native35-ledger','row1','rule4_controls','rule4_candidate','row4-placement','native30-receipt-tail','w1_controls (six fixed repeats)',
                        'overlap-gates','overlap-round-0','memory-round-0','overlap-round-1','memory-round-1','overlap-engine','native35-performance'],
        w1='workloads/q17b-w1.json unchanged: 8x153537 prompts, 8192 gen, 20 s windows, FULL/split/FULL/FULL/split/FULL',
        w2='workloads/q18a-w2-overlap.json unchanged (repeats=1) run as two fixed rounds; cache-salt namespace RUN_ID|native35-p-w2-round-<r>; session commands keep the real RUN_ID',
        target='80/80 reported marker; target_is_gate=False; SHORTFALL never fails',
        not_before='2026-10-30T22:01:00Z',hard_end='2026-11-02T06:00:00Z',active_s=window.SPEED_ACTIVE_S,cleanup_margin_s=window.SPEED_CLEANUP_MARGIN_S,
        window_source='native35p_gate.speed_window_ok(time.time())',active_derivation=[list(r) for r in window.ACTIVE_DERIVATION],
        cleanup_derivation=[list(r) for r in window.CLEANUP_DERIVATION],expected_runtime_s=window.EXPECTED_RUNTIME_S,
        fixture_leaves=['native35_p_fixture_leaves.json'],regression=regression,focused=focused,device_run='NOT_RUN',owner_prelaunch_cpu_executed=True,vm_input='NONE')


def queue_differences():
    """Every line the P queue texts differ from O's (names normalized to P)."""
    rows={}
    for suffix in ('.draft.sh','_settings.sh','_declare.sh'):
        o=(PARENT/'runs'/('q18a_native35_o'+suffix)).read_text().replace('q18a-native35-o',CANDIDATE).replace('q18a_native35_o','q18a_native35_p')
        p=(ROOT/'harness/runs'/('q18a_native35_p'+suffix)).read_text()
        rows['q18a_native35_p'+suffix]=list(difflib.unified_diff(o.splitlines(),p.splitlines(),'O (names as P)','P',n=0,lineterm=''))
    return rows


def build(out,strict_log,focused_log):
    assert not out.exists();assert sha(PARENT/'SHA256SUMS')==PARENT_MANIFEST and sha(PARENT_RUNNER)==PARENT_RUNNER_SHA
    assert sha(PARENT/'native30_overlay/SHA256SUMS')==PARENT_OVERLAY_MANIFEST
    previous=verify(PARENT);previous_overlay=verify(PARENT/'native30_overlay');source=source_head()
    assert json.loads((PARENT/'native30-provenance.json').read_text())['source']==PARENT_SOURCE
    subprocess.run([sys.executable,'-B',str(ROOT/'tools/native35_p_queue_drafts.py'),'--check'],check=True,cwd=ROOT)
    added,changed=source_delta(source);regression=regression_from(strict_log);focused=focused_from(focused_log)
    # Every tracked harness/patches file not in the declared delta is already byte-identical in O.
    tracked=subprocess.check_output(['git','ls-files','harness','patches/s5_engine'],cwd=ROOT,text=True).splitlines()
    for name in tracked:
        rel=name[len('harness/'):] if name.startswith('harness/') else 'e3_builder/'+name[len('patches/s5_engine/'):]
        if name.startswith('harness/') and rel in added|changed:continue
        assert previous.get(rel)==sha(ROOT/name),'undeclared difference vs O: '+rel
    for directory in sorted(PARENT.glob('native3*_tools')):
        for path in directory.glob('*.py'):
            if (ROOT/'tools'/path.name).is_file():assert sha(path)==sha(ROOT/'tools'/path.name),'bundled tool differs from tools/: '+path.name
    out.mkdir();bundle=out/'bundle';shutil.copytree(PARENT,bundle)
    for path in [bundle,*bundle.rglob('*')]:path.chmod(0o755 if path.is_dir() else 0o644)
    for rel in sorted(added|changed):
        target=bundle/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/'harness'/rel,target)
    for name in SHIPPED_TOOLS:shutil.copyfile(ROOT/'tools'/name,bundle/'native35_tools'/name)
    provenance=json.loads((bundle/'native30-provenance.json').read_text())
    provenance.update(source=source,candidate=CANDIDATE,native35_arm='P',selftest=regression,parent_bundle_manifest=PARENT_MANIFEST,
                      native35_p=details(regression,focused))
    assert provenance['overlay_manifest']==PARENT_OVERLAY_MANIFEST
    dump(bundle/'native30-provenance.json',provenance);manifest(bundle)
    pin=sha(bundle/'SHA256SUMS');held=out/(CANDIDATE+'-bundle-'+pin[:12]);bundle.rename(held)
    rows=verify(held);overlay=verify(held/'native30_overlay')
    assert overlay==previous_overlay and sha(held/'native30_overlay/SHA256SUMS')==PARENT_OVERLAY_MANIFEST
    bundle_changed=sorted(n for n in previous if n in rows and rows[n]!=previous[n])
    bundle_added=sorted(set(rows)-set(previous));bundle_removed=sorted(set(previous)-set(rows))
    assert set(bundle_changed)==CHANGED_HARNESS|REGENERATED,bundle_changed
    assert set(bundle_added)==ADDED_HARNESS|{'native35_tools/'+n for n in SHIPPED_TOOLS},bundle_added
    assert not bundle_removed
    for name in ('native30-review.json','e3_builder/native_candidate_review.json'):assert rows[name]==previous[name],name
    # Runner: O's runner, names/pins/gate/window only.
    original=PARENT_RUNNER.read_text()
    owner=original.replace('q18a_native35_o','q18a_native35_p').replace('q18a-native35-o',CANDIDATE)
    assert owner.count(O_WINDOW)==1 and owner.count(O_GATE)==1
    owner=owner.replace(O_WINDOW,P_WINDOW).replace(O_GATE,p_gate())
    owner=replace(owner,dict(H="Path('/home/<service-user>/harness/"+pin[:12]+"')",BUNDLE=repr(pin)))
    runner=out/(CANDIDATE+'-owner-run-'+pin[:12]+'.py');runner.write_text(owner);compile(owner,str(runner),'exec')
    normalized=owner.replace(p_gate(),O_GATE).replace(P_WINDOW,O_WINDOW).replace('q18a_native35_p','q18a_native35_o').replace(CANDIDATE,'q18a-native35-o')
    assignments={n.targets[0].id:ast.get_source_segment(original,n.value) for n in ast.parse(original).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    assert replace(normalized,{k:assignments[k] for k in ('H','BUNDLE')})==original,'unexpected runner body change'
    runner_diff=[line for line in difflib.ndiff(original.splitlines(),owner.splitlines()) if line.startswith(('- ','+ '))]
    proof=dict(verdict='PASS',classification=None,source=source,parent_source=PARENT_SOURCE,parent_bundle=str(PARENT),parent_manifest=PARENT_MANIFEST,manifest=pin,
        bundle_changed={n:dict(parent_sha256=previous[n],sha256=rows[n]) for n in bundle_changed},bundle_added={n:rows[n] for n in bundle_added},bundle_removed=bundle_removed,
        overlay_manifest=PARENT_OVERLAY_MANIFEST,overlay_byte_identical=True,overlay_files=len(overlay),review_byte_identical=True,compiled_libraries_byte_identical=True,
        engine_changed_vs_parent=[],queue_differences=queue_differences(),runner_differences=runner_diff,parent_runner_sha256=PARENT_RUNNER_SHA,
        runner_only_names_bundle_pins_selfrun_gate_and_window=True,gpu_execution=False,box_action=False,details=details(regression,focused))
    dump(out/'freeze-proof.json',proof)
    result=dict(name=CANDIDATE,arm='P',bundle=str(held),manifest=pin,overlay=str(held/'native30_overlay'),overlay_manifest=PARENT_OVERLAY_MANIFEST,runner=str(runner),
        runner_sha256=sha(runner),runner_size=runner.stat().st_size,review_sha256=assignments['REVIEW'].strip("'"),checksummed_files=len(rows),source=source,held=True,
        regression=regression,focused=focused,expected_selfrun_tests=expected_tests(),image=IMAGE,parent_manifest=PARENT_MANIFEST,parent_runner_sha256=PARENT_RUNNER_SHA)
    dump(out/'build.json',result)
    for path in [held,*held.rglob('*'),runner]:path.chmod(0o555 if path.is_dir() else 0o444)
    print(json.dumps({k:result[k] for k in ('name','manifest','runner_sha256','runner_size','overlay_manifest','expected_selfrun_tests')},sort_keys=True));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--selftest-log',type=Path,required=True);p.add_argument('--focused-log',type=Path,required=True)
    a=p.parse_args();build(a.out,a.selftest_log,a.focused_log)
