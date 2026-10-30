#!/usr/bin/env python3
"""Freeze e: native28 phase block, d3 runtime/plugin bytes, separate receipt policy."""
import argparse
import ast
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
PARENT=Path('<hub-dir>/agents/flash-next/tmp/native30-localize-d3-held-a/q18a-native30-localize-d3-bundle-705007c6a1ee')
PARENT_RUNNER=PARENT.parent/'q18a-native30-localize-d3-owner-run-705007c6a1ee.py'
N28=Path('<hub-dir>/agents/flash-next/tmp/native28-held-b/q18a-native28-bundle-71571d7efbca')
sys.path.insert(0,str(ROOT/'patches/s5_engine'))
from native_readiness import digest,payload_identity,verify_finalized_package
from build_native30_held import sha,dump,manifest,verify,replace

def build(out,log):
    assert sha(PARENT/'SHA256SUMS')=='705007c6a1ee01d34fc3585de2b796b1bed088cd6bffcd817edb803f48012c8a'
    old=verify(PARENT);assert not out.exists()
    text=log.read_text();summ=re.findall(r'^selftest: (\d+)/(\d+) cases as required$',text,re.M)
    suites=[json.JSONDecoder().raw_decode(text[m.end():])[0] for m in re.finditer('selftest suite: ',text)]
    assert len(summ)==1 and summ[0][0]==summ[0][1] and all(s['returncode']==0 and not s['skipped'] for s in suites)
    reg=dict(cases=int(summ[0][0]),required=int(summ[0][1]),exit_code=0,skips=0,suites=len(suites),reported_tests=sum(s['tests_run'] or 0 for s in suites),log_sha256=sha(log),basis='fresh e strict regression')
    out.mkdir();bundle=out/'bundle';shutil.copytree(PARENT,bundle)
    for p in [bundle,*bundle.rglob('*')]:p.chmod(0o755 if p.is_dir() else 0o644)
    tracked=subprocess.check_output(['git','ls-files','harness'],cwd=ROOT,text=True).splitlines()
    changes=[];added=[]
    for name in tracked:
        source=ROOT/name
        if not source.is_file():continue
        rel=str(Path(name).relative_to('harness'));target=bundle/rel
        if target.exists() and sha(target)==sha(source):continue
        (changes if target.exists() else added).append(rel)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    assert set(changes)=={'native30_fixture_leaves.json','native30_gate.py','selftest.py','test_native30_c.py','test_native30_host.py','test_native30_identity.py'},changes
    assert set(added)=={'native30_policy.py','e3_native30_e_grade.py','test_native30_e.py'}|{'runs/q18a_native30_localize_e'+s for s in ('.draft.sh','_declare.sh','_settings.sh')},added
    tools=[]
    for name in ('native30_owner_functions.py','native30_cpu_probe.py','native30_selfrun.py'):
        target=bundle/'native30_tools'/name
        if sha(target)!=sha(ROOT/'tools'/name):tools.append(str(target.relative_to(bundle)))
        shutil.copyfile(ROOT/'tools'/name,target)
    oldqueue=(PARENT/'runs/q18a_native30_localize_d3.draft.sh').read_text();queue=(bundle/'runs/q18a_native30_localize_e.draft.sh').read_text()
    phase_start='python3 "$H/e3_native_request_plans.py" --url "$PROBE_URL"'
    block=phase_start+(N28/'runs/q18a_native28.draft.sh').read_text().split(phase_start,1)[1].split('\nRULE4_CONTINUATION=0',1)[0]
    assert block in queue
    for fn in ('stop_engine','restore_launcher','restore_defaults','write_cleanup'):
        body=lambda s:s.split(fn+'() {',1)[1].split('\n}\n',1)[0]
        assert body(queue)==body(oldqueue),fn
    inputs=['e3_native_qualification_driver.py','e3_q18a_qualification.py','e3_native_request_plans.py','workloads/q17b-correctness-precontrols.json','workloads/q17b-teacher-decode-v3.json','workloads/q18a-native-placement-prefill.json','workloads/q18a-native-row1-template.json','baselines/q15-teacher-forced-rule3.json']
    assert all(sha(bundle/n)==sha(N28/n) for n in inputs)
    overlay=bundle/'native30_overlay';plugins=overlay/'e3/native30_plugins'
    for name in ('native30_localize.py','native30_host.py'):
        assert sha(plugins/name)==sha(PARENT/'native30_overlay/e3/native30_plugins'/name)==sha(bundle/name)
    shutil.copyfile(bundle/'native30_policy.py',plugins/'native30_policy.py')
    dist=plugins/'flashnext_native30_receipt_policy-0.1.dist-info';dist.mkdir()
    (dist/'METADATA').write_text('Metadata-Version: 2.1\nName: flashnext-native30-receipt-policy\nVersion: 0.1\n')
    (dist/'entry_points.txt').write_text('[vllm.general_plugins]\nnative30_receipt_policy = native30_policy:install\n')
    closure=json.loads((overlay/'e3/native-implementation-closure.json').read_text());review=closure['review']
    details=dict(parent_manifest=sha(PARENT/'SHA256SUMS'),engine_and_d3_plugin_bytes_unchanged=True,phase_sequence=['startup','row1','rule4_controls','rule4_candidate','row4'],native28_phase_block_sha256=__import__('hashlib').sha256(block.encode()).hexdigest(),input_shas={n:sha(bundle/n) for n in inputs},sync='none',not_before='2026-10-30T16:00:00Z',hard_end='2026-10-30T23:00:00Z',active_s=2100,cleanup_margin_s=420,receipt_bytes_per_rank=1024**3,receipt_records_per_rank=2*1024**2,receipt_tail_s=30,expected_box_minutes=[23,28],scope='full native28 preconditioning; receipt-only companion policy; no engine or d3 plugin edits; no hardware/performance claim')
    review['native30_e']=details;review['cpu_regression']=reg
    review['night_window']=dict(not_before=details['not_before'],hard_end=details['hard_end'],queue_timeout_s=2100,cleanup_margin_s=420,placeholder=False)
    review['harness_source_shas']={str(Path(n).relative_to('harness')):sha(ROOT/n) for n in tracked if (ROOT/n).is_file()}
    closure['review_sha256']=digest(review);dump(overlay/'e3/native-implementation-closure.json',closure);dump(bundle/'native30-review.json',review)
    gen=json.loads((overlay/'S5-OVERLAP-GENERATION.json').read_text());ready=json.loads((overlay/'e3/native-overlap-readiness.json').read_text())
    gen['implementation_review_sha256']=ready['review_sha256']=closure['review_sha256']
    gen['package_sha256']=ready['package_sha256']=payload_identity(overlay)
    ready['receipt_sha256']=digest({k:v for k,v in ready.items() if k!='receipt_sha256'})
    dump(overlay/'S5-OVERLAP-GENERATION.json',gen);dump(overlay/'e3/native-overlap-readiness.json',ready);manifest(overlay)
    verify_finalized_package(overlay,closure['generated_source_shas']);newov=verify(overlay);oldov=verify(PARENT/'native30_overlay')
    ovchanged=[n for n in oldov if newov[n]!=oldov[n]];ovadded=sorted(set(newov)-set(oldov))
    assert set(ovchanged)=={'S5-OVERLAP-GENERATION.json','e3/native-overlap-readiness.json','e3/native-implementation-closure.json'},ovchanged
    assert set(ovadded)=={'e3/native30_plugins/native30_policy.py','e3/native30_plugins/flashnext_native30_receipt_policy-0.1.dist-info/METADATA','e3/native30_plugins/flashnext_native30_receipt_policy-0.1.dist-info/entry_points.txt'} and not set(oldov)-set(newov)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    provenance=json.loads((bundle/'native30-provenance.json').read_text());provenance.update(source=source,selftest=reg,native30_e=details,overlay_manifest=sha(overlay/'SHA256SUMS'),review_sha256=closure['review_sha256'])
    dump(bundle/'native30-provenance.json',provenance);manifest(bundle);pin=sha(bundle/'SHA256SUMS');final=out/('q18a-native30-localize-e-bundle-'+pin[:12]);bundle.rename(final)
    original=PARENT_RUNNER.read_text();owner=original.replace('q18a_native30_localize_d3','q18a_native30_localize_e').replace('q18a-native30-localize-d3','q18a-native30-localize-e').replace("result.get('tests_run',0)==29","result.get('tests_run',0)==35")
    owner=replace(owner,dict(H="Path('/home/<service-user>/harness/"+pin[:12]+"')",BUNDLE=repr(pin),REVIEW=repr(closure['review_sha256']),EXPECTED_SOURCES=repr(newov)))
    runner=out/('q18a-native30-localize-e-owner-run-'+pin[:12]+'.py');runner.write_text(owner);compile(owner,str(runner),'exec')
    normalized=owner.replace('q18a_native30_localize_e','q18a_native30_localize_d3').replace('q18a-native30-localize-e','q18a-native30-localize-d3').replace("result.get('tests_run',0)==35","result.get('tests_run',0)==29")
    assignments={n.targets[0].id:ast.get_source_segment(original,n.value) for n in ast.parse(original).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    assert replace(normalized,{k:assignments[k] for k in ('H','BUNDLE','REVIEW','EXPECTED_SOURCES')})==original
    rows=verify(final);changed=[n for n in old if rows[n]!=old[n]];new=sorted(set(rows)-set(old));assert not set(old)-set(rows)
    expected=set(changes)|set(tools)|{'native30-review.json','native30-provenance.json','native30_overlay/SHA256SUMS'}|{'native30_overlay/'+n for n in ovchanged}
    assert set(changed)==expected,changed
    assert set(new)==set(added)|{'native30_overlay/'+n for n in ovadded},new
    proof=dict(verdict='PASS',classification=None,source=source,parent_bundle=str(PARENT),parent_manifest=sha(PARENT/'SHA256SUMS'),manifest=pin,changed=sorted(changed),added=new,removed=[],overlay_changed=ovchanged,overlay_added=ovadded,engine_and_d3_plugin_bytes_unchanged=True,native28_phase_block_equal=True,restore_bodies_equal=True,runner_only_names_pins_and_test_count=True,details=details,gpu_execution=False,box_action=False)
    proof['d3_plugin_shas']={n:sha(final/'native30_overlay/e3/native30_plugins'/n) for n in ('native30_localize.py','native30_host.py')}
    dump(out/'freeze-proof.json',proof)
    result=dict(bundle=str(final),manifest=pin,overlay=str(final/'native30_overlay'),overlay_manifest=sha(final/'native30_overlay/SHA256SUMS'),runner=str(runner),runner_sha256=sha(runner),runner_size=runner.stat().st_size,review_sha256=closure['review_sha256'],checksummed_files=len(rows),held=True,name='q18a-native30-localize-e',source=source,identity_timeout_s=360)
    dump(out/'build.json',result)
    for p in [final,*final.rglob('*'),runner]:p.chmod(0o555 if p.is_dir() else 0o444)
    print(json.dumps(result,sort_keys=True));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--selftest-log',type=Path,required=True);a=p.parse_args();build(a.out,a.selftest_log)
