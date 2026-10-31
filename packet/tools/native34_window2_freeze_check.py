#!/usr/bin/env python3
"""Prove a window-only native34 rebuild against the corresponding frozen arm."""
import argparse,ast,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=Path('<hub-dir>/agents/flash-next/tmp')
sys.path.insert(0,str(ROOT/'tools'))
from build_native30_held import sha,verify,replace,dump


def fix_slot(text):
    a=text.index('\n',text.index("python3 - <<'PY_SLOT'"))+1;b=text.index('\nPY_SLOT',a)
    body="import os,sys,time\nsys.path.insert(0,os.environ.get('HARNESS_DIR') or os.environ.get('H') or '')\nfrom native34_gate import window_ok\nassert window_ok(time.time()), 'queue outside owner window or cleanup margin'"
    return text[:a]+body+text[b:]


def check(build,out):
    b=json.loads(build.read_text());arm=b['arm'].lower();candidate='q18a-native34-'+arm+'2';stem=candidate.replace('-','_')
    parent=json.loads((BASE/('native34-'+arm+'-held-b')/'build.json').read_text());old=Path(parent['bundle']);new=Path(b['bundle'])
    before=verify(old);after=verify(new);ov=verify(new/'native30_overlay');prev=verify(old/'native30_overlay')
    assert sha(new/'SHA256SUMS')==b['manifest'];assert ov.keys()==prev.keys()
    metadata={'S5-OVERLAP-GENERATION.json','e3/native-implementation-closure.json','e3/native-overlap-readiness.json'}
    changed_overlay={p for p in ov if ov[p]!=prev[p]};assert changed_overlay<=metadata,changed_overlay
    for role in ('i','s','o'):
        base='q18a_native34_'+role
        assert (new/'runs'/(base+'.draft.sh')).read_text()==fix_slot((old/'runs'/(base+'.draft.sh')).read_text())
        for suffix in ('.draft.sh','_declare.sh','_settings.sh'):
            text=(old/'runs'/(base+suffix)).read_text()
            if suffix=='.draft.sh':text=fix_slot(text)
            text=text.replace(base,base+'2').replace('q18a-native34-'+role,'q18a-native34-'+role+'2')
            assert text==(new/'runs'/(base+'2'+suffix)).read_text()
    for p in ('native34_gate.py','native30_gate.py','e3_native34_causal.py','native34_worker.py','native34_diag.py','native34_control.py','native34_indexer.py','native34_storage.py','native34_grade.py'):
        assert sha(old/p)==sha(new/p),p
    prior=Path(parent['runner']).read_text();actual=Path(b['runner']).read_text()
    normalized=actual.replace(stem,'q18a_native34_'+arm).replace(candidate,'q18a-native34-'+arm)
    normalized=normalized.replace("result.get('tests_run',0)==91 and result.get('final_input_proof') is True and result.get('draft_window_slots')=='PASS'","result.get('tests_run',0)==90 and result.get('final_input_proof') is True")
    assignments={n.targets[0].id:ast.get_source_segment(prior,n.value) for n in ast.parse(prior).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    assert replace(normalized,{k:assignments[k] for k in ('H','BUNDLE','REVIEW','EXPECTED_SOURCES','LAUNCHER')})==prior
    allowed={'native30-provenance.json','native30-review.json','e3_builder/native_candidate_review.json','native31_fixture_leaves.json','test_native34_queue.py','native34_tools/native34_cpu_probe.py','native30_overlay/SHA256SUMS'}
    allowed|={'native30_overlay/'+p for p in metadata}|{'runs/q18a_native34_'+r+'.draft.sh' for r in ('i','s','o')}
    changed={p for p in before if p in after and before[p]!=after[p]};assert changed<=allowed,changed-allowed
    added=set(after)-set(before)
    expected_added={'native34_window_before.txt'}|{'runs/q18a_native34_'+r+'2'+s for r in ('i','s','o') for s in ('.draft.sh','_declare.sh','_settings.sh')}
    assert added==expected_added,added^expected_added
    assert not set(before)-set(after)
    result=dict(verdict='PASS',classification=None,arm=b['arm'],parent_manifest=parent['manifest'],manifest=b['manifest'],window_source='native34_gate.window_ok(time.time())',engine_plugin_scenario_gate_modules_byte_identical=True,ordinary_bytes_unchanged=True,queue_only_names_and_entry_window=True,settings_declaration_only_names=True,runner_only_names_pins_test_count_and_slot_receipt_requirement=True,changed=sorted(changed),added=sorted(added),removed=[],overlay_changed=sorted(changed_overlay),gpu_execution=False,box_action=False)
    assert not out.exists();dump(out,result);print(json.dumps(dict(verdict='PASS',arm=b['arm'],out=str(out),sha256=sha(out))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();check(a.build,a.out)
