#!/usr/bin/env python3
"""Bound the protocol repair against the corresponding frozen role2 arm."""
import argparse,ast,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent/'tmp'
sys.path.insert(0,str(ROOT/'tools'))
from build_native30_held import sha,verify,replace,dump

def check(build,out):
    b=json.loads(build.read_text());arm=b['arm'].lower();parent=json.loads((BASE/('native34-'+arm+'2-held-a/build.json')).read_text())
    old=Path(parent['bundle']);new=Path(b['bundle']);before=verify(old);after=verify(new)
    prev=verify(old/'native30_overlay');ov=verify(new/'native30_overlay');assert prev.keys()==ov.keys()
    metadata={'S5-OVERLAP-GENERATION.json','e3/native-implementation-closure.json','e3/native-overlap-readiness.json'}
    plugin='e3/native30_plugins/native34_worker.py'
    changed_overlay={p for p in prev if prev[p]!=ov[p]};assert changed_overlay==metadata|{plugin},changed_overlay
    source=(ROOT/'harness/native34_worker.py').read_text();prior=(old/'native34_worker.py').read_text()
    expected=prior.replace("    if marker is None:return original(*args,**kwargs)","    # Qualification markers belong to qualification(), reached through the\n    # real native_overlap_worker handler; only census owns CHECKPOINT.\n    if marker is None or command.get('action')=='teacher_evidence':\n        return original(*args,**kwargs)")
    assert source==expected,'unexpected diagnostic worker delta'
    assert (new/plugin.replace('e3/','native30_overlay/e3/',1)).read_text()==source
    for role in ('i','s','o'):
        base='q18a_native34_'+role
        for suffix in ('.draft.sh','_declare.sh','_settings.sh'):
            assert (old/'runs'/(base+'2'+suffix)).read_text().replace(base+'2',base+'3').replace('q18a-native34-'+role+'2','q18a-native34-'+role+'3')==(new/'runs'/(base+'3'+suffix)).read_text()
    for p in ('native34_gate.py','native30_gate.py','e3_native34_causal.py','native34_diag.py','native34_control.py','native34_indexer.py','native34_storage.py','native34_grade.py'):
        assert sha(old/p)==sha(new/p),p
    prior_runner=Path(parent['runner']).read_text();actual=Path(b['runner']).read_text()
    norm=actual.replace('q18a_native34_'+arm+'3','q18a_native34_'+arm+'2').replace('q18a-native34-'+arm+'3','q18a-native34-'+arm+'2')
    norm=norm.replace(" and result.get('native34_protocol')=='PASS'",'')
    assignments={n.targets[0].id:ast.get_source_segment(prior_runner,n.value) for n in ast.parse(prior_runner).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    assert replace(norm,{k:assignments[k] for k in ('H','BUNDLE','REVIEW','EXPECTED_SOURCES','LAUNCHER')})==prior_runner
    allowed={'native30-provenance.json','native30-review.json','e3_builder/native_candidate_review.json','native31_fixture_leaves.json','test_native34_queue.py','test_native34_runtime.py','native34_worker.py','native34_tools/native34_cpu_probe.py','native30_overlay/SHA256SUMS'}|{'native30_overlay/'+p for p in metadata|{plugin}}
    changed={p for p in before if p in after and before[p]!=after[p]};assert changed<=allowed,changed-allowed
    added=set(after)-set(before);expected_added={'native34_protocol_before.txt'}|{'runs/q18a_native34_'+r+'3'+s for r in ('i','s','o') for s in ('.draft.sh','_declare.sh','_settings.sh')}
    assert added==expected_added,added^expected_added;assert not set(before)-set(after)
    assert sha(new/'SHA256SUMS')==b['manifest']
    result=dict(verdict='PASS',classification=None,arm=b['arm'],parent_manifest=parent['manifest'],manifest=b['manifest'],engine_ordinary_gate_driver_byte_identical=True,plugin_only_qualification_marker_routing=True,queue_only_names=True,runner_only_names_pins_protocol_gate=True,changed=sorted(changed),added=sorted(added),removed=[],overlay_changed=sorted(changed_overlay),gpu_execution=False,box_action=False)
    assert not out.exists();dump(out,result);print(json.dumps(dict(verdict='PASS',arm=b['arm'],out=str(out),sha256=sha(out))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();check(a.build,a.out)
