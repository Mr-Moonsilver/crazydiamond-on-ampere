#!/usr/bin/env python3
"""Native35 P-c freeze proof check: re-derive every difference of the held P-c bundle and runner from frozen O.

Independent of the build: both bundles are re-verified against their SHA256SUMS,
the file-level delta is recomputed and must equal the declared delta and the
build's freeze-proof.json; every added/changed file must equal its git blob at
the build source; the overlay, both review JSONs and every other file must be
byte-identical to O; provenance may differ only in the declared keys; the
runner may differ from O's only by names, H/BUNDLE pins, the self-run gate and
the P window import.
"""
import argparse,ast,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from build_native30_held import sha,verify,replace,dump
import build_native35_p_c as bp

PROVENANCE_KEYS={'source','candidate','native35_arm','selftest','parent_bundle_manifest','native35_p_c'}


def blob(source,path):
    return hashlib.sha256(subprocess.check_output(['git','show',source+':'+path],cwd=ROOT)).hexdigest()


def check(build,proof_path,out):
    b=json.loads(build.read_text());proof=json.loads(proof_path.read_text());new=Path(b['bundle']);old=bp.PARENT
    assert sha(old/'SHA256SUMS')==bp.PARENT_MANIFEST and sha(new/'SHA256SUMS')==b['manifest']==proof['manifest']
    before=verify(old);after=verify(new)
    assert verify(old/'native30_overlay')==verify(new/'native30_overlay') and sha(new/'native30_overlay/SHA256SUMS')==bp.PARENT_OVERLAY_MANIFEST
    changed=sorted(n for n in before if n in after and before[n]!=after[n]);added=sorted(set(after)-set(before));removed=sorted(set(before)-set(after))
    assert set(changed)==bp.CHANGED_HARNESS|bp.REGENERATED and set(added)==bp.ADDED_HARNESS|{'native35_tools/'+n for n in bp.SHIPPED_TOOLS} and not removed
    assert {n:dict(parent_sha256=before[n],sha256=after[n]) for n in changed}==proof['bundle_changed']
    assert {n:after[n] for n in added}==proof['bundle_added'] and proof['bundle_removed']==[]
    for rel in sorted(bp.ADDED_HARNESS|bp.CHANGED_HARNESS):assert after[rel]==blob(b['source'],'harness/'+rel),rel
    for name in bp.SHIPPED_TOOLS:assert after['native35_tools/'+name]==blob(b['source'],'tools/'+name),name
    for name in ('native30-review.json','e3_builder/native_candidate_review.json'):assert after[name]==before[name],name
    po=json.loads((old/'native30-provenance.json').read_text());pn=json.loads((new/'native30-provenance.json').read_text())
    differ={k for k in set(po)|set(pn) if po.get(k)!=pn.get(k)};assert differ<=PROVENANCE_KEYS,differ-PROVENANCE_KEYS
    assert after['e3_native35_l2_placement.py']==bp.L2_RUNNER_SHA and b['l2_runner_sha256']==bp.L2_RUNNER_SHA
    assert pn['native35_arm']==bp.ARM and pn['candidate']==bp.CANDIDATE and pn['source']==b['source'] and pn['parent_bundle_manifest']==bp.PARENT_MANIFEST
    queue={}
    for suffix in ('.draft.sh','_settings.sh','_declare.sh'):
        name='runs/'+bp.STEM+suffix;assert after[name]==blob(b['source'],'harness/'+name),name;queue[name]=after[name]
    subprocess.run(['git','diff','--quiet',b['source'],'--','harness','patches/s5_engine'],cwd=ROOT,check=True)
    for tool in ('native35_p_queue_drafts.py','native35_p_b_queue_drafts.py','native35_p_c_queue_drafts.py','native35_l2_queue_draft.py','native35_l2_tools_derive.py'):
        subprocess.run([sys.executable,'-B',str(ROOT/'tools'/tool),'--check'],cwd=ROOT,check=True)
    original=bp.PARENT_RUNNER.read_text();runner=Path(b['runner']).read_text()
    assert sha(bp.PARENT_RUNNER)==bp.PARENT_RUNNER_SHA and sha(b['runner'])==b['runner_sha256']
    normalized=runner.replace(bp.p_gate(),bp.O_GATE).replace(bp.P_WINDOW,bp.O_WINDOW).replace(bp.STEM,'q18a_native35_o').replace(bp.CANDIDATE,'q18a-native35-o')
    values={n.targets[0].id:ast.get_source_segment(original,n.value) for n in ast.parse(original).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    assert replace(normalized,{k:values[k] for k in ('H','BUNDLE')})==original,'runner differs from O beyond the declared lines'
    mine={n.targets[0].id:ast.get_source_segment(runner,n.value) for n in ast.parse(runner).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    assert mine['H']=="Path('/home/<service-user>/harness/%s')"%b['manifest'][:12] and mine['BUNDLE']==repr(b['manifest'])
    for key in ('REVIEW','LAUNCHER','EXPECTED_SOURCES','IMAGE','BASE_MANIFEST','OBSERVER'):assert mine[key]==values[key],key
    assert mine['QNAME']=="'q18a-native35-p-c'" and mine['OVERLAY']=="Path('/data/vllm/flash-next-tp4/prod/overlay-v16s5-q18a-native35-p-c')"
    assert "result.get('tests_run',0)==%d"%b['expected_selfrun_tests'] in runner and "result.get('native35_arm')=='P-C' and result.get('native35_performance')=='PASS' and result.get('native35_row4_l2')=='PASS' and result.get('native35_localize_off')=='PASS'" in runner
    result=dict(verdict='PASS',classification=None,manifest=b['manifest'],parent_manifest=bp.PARENT_MANIFEST,source=b['source'],parent_source=bp.PARENT_SOURCE,
        bundle_changed=changed,bundle_added=added,bundle_removed=removed,overlay_byte_identical=True,overlay_manifest=bp.PARENT_OVERLAY_MANIFEST,
        reviews_byte_identical=True,provenance_keys_changed=sorted(differ),queue_scripts=queue,runner_sha256=b['runner_sha256'],
        runner_declared_differences=['names q18a-native35-o -> q18a-native35-p-c (QNAME, OVERLAY, proof schema, declare/draft phase scripts)','H and BUNDLE pins',
            'self-run gate: tests_run 125 -> %d, native35_arm O -> P-C, plus native35_performance PASS, native35_row4_l2 PASS and native35_localize_off PASS'%b['expected_selfrun_tests'],'check_slot window: native34_gate.window_ok -> native35p_gate.speed_window_ok'],
        runner_unchanged=['REVIEW','LAUNCHER','EXPECTED_SOURCES','IMAGE','BASE_MANIFEST','OBSERVER','preflight/staging/arm/cleanup bodies'],
        l2_runner_sha256=bp.L2_RUNNER_SHA,freeze_proof_sha256=sha(proof_path),gpu_execution=False,box_action=False)
    with out.open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps(dict(verdict='PASS',changed=changed,added=len(added)),sort_keys=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--freeze-proof',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();check(a.build,a.freeze_proof,a.out)
