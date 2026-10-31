#!/usr/bin/env python3
"""Native35 P-c held bundle: the exact frozen native35 O held-a bundle plus the declared P speed-arm,
P-b, P-c and L2 row4 (claude/native35-l2, taken unchanged) harness files.

P-c is P-b with FLASH_NATIVE30_LOCALIZE=0, the second declared diagnostic switch
vs O. Its queue deltas from P-b are that setting, native35-diagnostic as an
explicit NOT_RUN record, the post-row4 step note, the row4 runner
(e3_native35_p_c_placement.py: the L2 runner with the decode cohort stopped at
the 20-token window boundary) and the header comment.  Lead decision 31 Oct
2026 (option A + spool), after P-b's row4 refused collect:response-too-large:
the shared driver's PLACEMENT_WINDOW_TOKENS is 20 (2 cohorts x 20 = the declared
40 replays), observed replays are checked against [40, 56], the observer_end
reply is read through a bounded 512 MiB spool with bytes/sha256 always kept,
and ring/list overflows are explicit row4 missing reasons (e3_observer_census).

Nothing in the engine changes: the overlay (SHA256SUMS 32673c37...), both review
JSONs, every compiled library and every parent plugin stay byte-identical to O
and are asserted so; the overlay is copied, never regenerated. The bundle adds
exactly the files git adds under harness/ between O's source and this tree
(plus the P-c probe/self-run/inert check and the L2 tools its L2 suite reads) and changes
exactly selftest.py and native30-provenance.json. The runner is O's with names,
the bundle pins, the self-run gate and the P window import changed; everything
else is asserted equal. P-a (held-a) and P-b (held-b) are left untouched.
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
CANDIDATE='q18a-native35-p-c'
ARM='P-C'
STEM='q18a_native35_p_c'
L2_RUNNER_SHA='9327e01cd9f923e31b101a53208743990e4f877c68d9f6de6225c26ee30fd406'  # claude/native35-l2 29f8265
# Declared bundle delta vs O (bundle-relative paths).
ADDED_HARNESS={'e3_native35_performance.py','native35_performance_grade.py','native35p_gate.py',
               'runs/q18a_native35_p.draft.sh','runs/q18a_native35_p_settings.sh','runs/q18a_native35_p_declare.sh',
               'test_native35_performance.py','test_native35_performance_queue.py','native35_p_fixture_leaves.json',
               'runs/q18a_native35_p_b.draft.sh','runs/q18a_native35_p_b_settings.sh','runs/q18a_native35_p_b_declare.sh','test_native35_p_b.py',
               'runs/q18a_native35_p_c.draft.sh','runs/q18a_native35_p_c_settings.sh','runs/q18a_native35_p_c_declare.sh','test_native35_p_c.py',
               # claude/native35-l2 (merged unchanged): the row4 runner, its suite and leaves, and the L2 queue texts that suite derives
               'e3_native35_l2_placement.py','test_native35_l2.py','native35_l2_fixture_leaves.json',
               'runs/q18a_native35_l2.draft.sh','runs/q18a_native35_l2_settings.sh','runs/q18a_native35_l2_declare.sh',
               # P-c row4 (option A + spool): the stop-at-window runner, the observer census, their suite and leaves
               'e3_native35_p_c_placement.py','e3_observer_census.py','test_native35_p_c_window.py','native35_p_c_fixture_leaves.json'}
# The shared driver (window tokens 20, replay bound, observer_end spool, census in the row4 evidence) and the
# three suites whose fixtures pinned the 40-token window or the exact row4 missing list.
CHANGED_HARNESS={'selftest.py','e3_native_qualification_driver.py','test_e3_native_qualification_driver.py','test_native30_e.py'}
# The P-c self-run tools (probe, self-run, inert check), and the L2 tools that test_native35_l2.Tools reads from /source/tools.
SHIPPED_TOOLS=('native35_p_c_cpu_probe.py','native35_p_c_selfrun.py','native35_p_c_inert_check.py','native35_l2_tools_derive.py','native35_l2_cpu_probe.py','native35_l2_selfrun.py')
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
    tree=ast.parse((ROOT/'tools/native35_p_c_cpu_probe.py').read_text())
    values=[n.value.value for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='EXPECTED_TESTS']
    assert len(values)==1 and type(values[0]) is int;return values[0]


def p_gate():
    return O_GATE.replace('==125','==%d'%expected_tests()).replace("result.get('native35_arm')=='O'","result.get('native35_arm')=='P-C' and result.get('native35_performance')=='PASS' and result.get('native35_row4_l2')=='PASS' and result.get('native35_localize_off')=='PASS' and result.get('native35_p_c_row4')=='PASS'")


def regression_from(log):
    text=log.read_text();summary=re.findall(r'^selftest: (\d+)/(\d+) cases as required$',text,re.M)
    suites=[json.JSONDecoder().raw_decode(text[m.end():])[0] for m in re.finditer('selftest suite: ',text)]
    assert len(summary)==1 and summary[0][0]==summary[0][1] and suites and all(s['returncode']==0 and not s['skipped'] for s in suites),'strict regression log must be complete, all-pass, zero skips'
    return dict(cases=int(summary[0][0]),required=int(summary[0][1]),exit_code=0,skips=0,suites=len(suites),reported_tests=sum(s['tests_run'] or 0 for s in suites),
                log_sha256=sha(log),basis='fresh full strict registered CPU regression on the native35 P-c final tree')


def focused_from(log):
    text=log.read_text();ran=re.findall(r'^Ran (\d+) tests? in ',text,re.M)
    assert ran and text.rstrip().endswith('OK') and 'skipped' not in text.rsplit('OK',1)[-1]
    return dict(tests=sum(int(n) for n in ran),skips=0,verdict='PASS',log_sha256=sha(log),
                basis='native35 P-c, P-b, L2 and P suites plus the native35 R/O/L queue/grade, native28 host guard and rule4 continuation, native qualification driver, native measurement client/source and overlap performance suites on the final tree')


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
        arm=ARM,candidate=CANDIDATE,predecessor='q18a-native35-p-b (held-b af520f825a1f, LOCALIZE=1, armed on the box 31 Oct 15:11:47Z); before it q18a-native35-p (held-a f1d1a16aa07a, never armed: native33-form row4 window ordering defect)',purpose='speed: decode tok/s during a long native prefill vs W1 FULL solo, and native prefill speed with overlap vs matched split solo',
        engine_changed_vs_parent=[],overlay_manifest=PARENT_OVERLAY_MANIFEST,overlay_byte_identical=True,review_byte_identical=True,
        settings='O settings with FLASH_NATIVE34_DIAG=0 (no FLASH_NATIVE34_ARM, as L) and Q18A_FIRST_ADVANCE_READ_TIMEOUT_S=1200 (native28 declaration; the only value e3_native_performance accepts)',
        localize=('FLASH_NATIVE30_LOCALIZE=0: second declared diagnostic switch vs O (lead decision 31 Oct). The four plugins stay in the overlay and on '
                  'PYTHONPATH and return before any hook (native35_p_c_inert_check, positive control under the P-b settings)'),sync='none (unused without localize)',readback=False,
        native35_diagnostic='explicit NOT_RUN record, rc 2: LOCALIZE=0 leaves no receipts; native35_grade.py would read zero rows and print a vacuous PASS',
        native30_localize='kept unchanged: per_rank begins 0 on all four ranks is the box-side proof the switch was off',
        receipt_tail='the 30 s post-row4 gap kept, so W1 starts at the same offset as in P-b; only the step note changed',
        row4=('native33-form real placement through e3_native35_p_c_placement.py (StoppedWindowPlacementRunner = the L2 ResumeWindowPlacementRunner, '
              'claude/native35-l2 29f8265 sha256 '+L2_RUNNER_SHA+', plus the decode-cohort stop directly after the client window); same argv, workload, '
              'budgets and grader; the qualification phase is the L2 queue phase with only the runner name swapped'),
        row4_window=dict(window_tokens=20,replays_declared=40,replays_low=40,replays_high=56,
                         derivation='lead decision 31 Oct 2026, option A: the observer is declared for 40 replays (PLACEMENT_REPLAYS, the ~60k-node window); 8 streams decode as 2 parity cohorts, so 2 x 20 tokens restores it; P-b let ~125 replays in'),
        observer_end_spool=dict(ceiling_bytes=512*1024**2,drain_max_bytes=4*1024**3,default_cap_bytes=112*1024**2,
                                derivation='client peak ~8x body (parse 3.2x, deep copies 2.1x, dump 2.2x) -> 4 GiB inside the 5 GiB observer transient margin (native30_gate TRANSIENT_MARGIN); caveat: workers and API server hold completed_reply and its encoded body at the same time, outside that margin'),
        phase_sequence=['startup','native35-ledger','row1','rule4_controls','rule4_candidate','row4-placement','native30-receipt-tail','w1_controls (six fixed repeats)',
                        'overlap-gates','overlap-round-0','memory-round-0','overlap-round-1','memory-round-1','overlap-engine','native35-performance'],
        w1='workloads/q17b-w1.json unchanged: 8x153537 prompts, 8192 gen, 20 s windows, FULL/split/FULL/FULL/split/FULL',
        w2='workloads/q18a-w2-overlap.json unchanged (repeats=1) run as two fixed rounds; cache-salt namespace RUN_ID|native35-p-w2-round-<r>; session commands keep the real RUN_ID',
        target='80/80 reported marker; target_is_gate=False; SHORTFALL never fails',
        not_before='2026-10-30T22:01:00Z',hard_end='2026-11-02T06:00:00Z',active_s=window.SPEED_ACTIVE_S,cleanup_margin_s=window.SPEED_CLEANUP_MARGIN_S,
        window_source='native35p_gate.speed_window_ok(time.time())',active_derivation=[list(r) for r in window.ACTIVE_DERIVATION],
        cleanup_derivation=[list(r) for r in window.CLEANUP_DERIVATION],expected_runtime_s=window.EXPECTED_RUNTIME_S,
        fixture_leaves=['native35_p_fixture_leaves.json','native35_l2_fixture_leaves.json','native35_p_c_fixture_leaves.json'],regression=regression,focused=focused,device_run='NOT_RUN',owner_prelaunch_cpu_executed=True,vm_input='NONE')


def queue_differences():
    """Every line the P-c queue texts differ from O's and from P-b's (names normalized to P-c)."""
    rows={}
    for suffix in ('.draft.sh','_settings.sh','_declare.sh'):
        p=(ROOT/'harness/runs'/(STEM+suffix)).read_text()
        for label,stem,qname in (('vs_o','q18a_native35_o','q18a-native35-o'),('vs_p_b','q18a_native35_p_b','q18a-native35-p-b')):
            base=(PARENT/'runs'/(stem+suffix)).read_text() if label=='vs_o' else (ROOT/'harness/runs'/(stem+suffix)).read_text()
            base=base.replace(qname,CANDIDATE).replace(stem,STEM)
            rows.setdefault(label,{})[STEM+suffix]=list(difflib.unified_diff(base.splitlines(),p.splitlines(),label+' (names as P-c)','P-c',n=0,lineterm=''))
    return rows


def build(out,strict_log,focused_log):
    assert not out.exists();assert sha(PARENT/'SHA256SUMS')==PARENT_MANIFEST and sha(PARENT_RUNNER)==PARENT_RUNNER_SHA
    assert sha(PARENT/'native30_overlay/SHA256SUMS')==PARENT_OVERLAY_MANIFEST
    previous=verify(PARENT);previous_overlay=verify(PARENT/'native30_overlay');source=source_head()
    assert json.loads((PARENT/'native30-provenance.json').read_text())['source']==PARENT_SOURCE
    for tool in ('native35_p_queue_drafts.py','native35_p_b_queue_drafts.py','native35_p_c_queue_drafts.py','native35_l2_queue_draft.py','native35_l2_tools_derive.py'):
        subprocess.run([sys.executable,'-B',str(ROOT/'tools'/tool),'--check'],check=True,cwd=ROOT)
    assert sha(ROOT/'harness/e3_native35_l2_placement.py')==L2_RUNNER_SHA
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
    provenance.update(source=source,candidate=CANDIDATE,native35_arm=ARM,selftest=regression,parent_bundle_manifest=PARENT_MANIFEST,
                      native35_p_c=details(regression,focused))
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
    owner=original.replace('q18a_native35_o',STEM).replace('q18a-native35-o',CANDIDATE)
    assert owner.count(O_WINDOW)==1 and owner.count(O_GATE)==1
    owner=owner.replace(O_WINDOW,P_WINDOW).replace(O_GATE,p_gate())
    owner=replace(owner,dict(H="Path('/home/<service-user>/harness/"+pin[:12]+"')",BUNDLE=repr(pin)))
    runner=out/(CANDIDATE+'-owner-run-'+pin[:12]+'.py');runner.write_text(owner);compile(owner,str(runner),'exec')
    normalized=owner.replace(p_gate(),O_GATE).replace(P_WINDOW,O_WINDOW).replace(STEM,'q18a_native35_o').replace(CANDIDATE,'q18a-native35-o')
    assignments={n.targets[0].id:ast.get_source_segment(original,n.value) for n in ast.parse(original).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    assert replace(normalized,{k:assignments[k] for k in ('H','BUNDLE')})==original,'unexpected runner body change'
    runner_diff=[line for line in difflib.ndiff(original.splitlines(),owner.splitlines()) if line.startswith(('- ','+ '))]
    proof=dict(verdict='PASS',classification=None,source=source,parent_source=PARENT_SOURCE,parent_bundle=str(PARENT),parent_manifest=PARENT_MANIFEST,manifest=pin,
        bundle_changed={n:dict(parent_sha256=previous[n],sha256=rows[n]) for n in bundle_changed},bundle_added={n:rows[n] for n in bundle_added},bundle_removed=bundle_removed,
        overlay_manifest=PARENT_OVERLAY_MANIFEST,overlay_byte_identical=True,overlay_files=len(overlay),review_byte_identical=True,compiled_libraries_byte_identical=True,
        engine_changed_vs_parent=[],queue_differences=queue_differences(),runner_differences=runner_diff,parent_runner_sha256=PARENT_RUNNER_SHA,
        runner_only_names_bundle_pins_selfrun_gate_and_window=True,gpu_execution=False,box_action=False,details=details(regression,focused))
    dump(out/'freeze-proof.json',proof)
    result=dict(name=CANDIDATE,arm=ARM,bundle=str(held),manifest=pin,overlay=str(held/'native30_overlay'),overlay_manifest=PARENT_OVERLAY_MANIFEST,runner=str(runner),
        runner_sha256=sha(runner),runner_size=runner.stat().st_size,review_sha256=assignments['REVIEW'].strip("'"),checksummed_files=len(rows),source=source,held=True,
        regression=regression,focused=focused,expected_selfrun_tests=expected_tests(),image=IMAGE,parent_manifest=PARENT_MANIFEST,parent_runner_sha256=PARENT_RUNNER_SHA,l2_runner_sha256=L2_RUNNER_SHA)
    dump(out/'build.json',result)
    for path in [held,*held.rglob('*'),runner]:path.chmod(0o555 if path.is_dir() else 0o444)
    print(json.dumps({k:result[k] for k in ('name','manifest','runner_sha256','runner_size','overlay_manifest','expected_selfrun_tests')},sort_keys=True));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--selftest-log',type=Path,required=True);p.add_argument('--focused-log',type=Path,required=True)
    a=p.parse_args();build(a.out,a.selftest_log,a.focused_log)
