#!/usr/bin/env python3
"""Compose the two sealed packets; no box commands, source deletion or device work."""
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
NAME = 'q18a-parity-n3s24q2024p-j2'
PARENT_BUILD = Path('<hub-dir>/cargo-target/flash-next/parity-n3s/builds/n3s24q2024p-build-1')
PARENT = PARENT_BUILD/'q18a-parity-n3s24q2024p-bundle-3b784537290a'
PARENT_SHA = '3b784537290abf21da7f8adc3e71ba140e441ff447263706646808e793aeb999'
J2_BUILD = Path('<hub-dir>/cargo-target/flash-next/n3-junction-j2-package-20261110/build-1')
J2 = J2_BUILD/'q18a-parity-n3j2-bundle-321e4cd7ffd8'
J2_SHA = '321e4cd7ffd85cb7a729b47983188b5cdd92baa86c71b047ec70b169943f38c2'
RUNTIME = ('engine_driver','native_auto_arm','scheduler_hooks')
SUITES = ('test_n3_junction_yield','test_n3_shortest_bulk','test_n3_yield_quota',
          'test_compose4_request_ids','test_append_yield','test_coverage_residue',
          'test_compose5_declines','test_lane_o3','test_native40_unarmed_hold',
          'test_n3j2_junction_rule','test_n3j2_junction_rule_mutants','test_q2024p_j2',
          'test_n3_next_layer_prefetch','test_n3_producer_witness','test_n3_fused_phases',
          'test_n3_gap_receipts','test_n3_indexer_lifetime')

def helper():
    spec=importlib.util.spec_from_file_location('sealed_j2_build',J2/'tools/build_n3j2.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.verify_manifest(J2,J2_SHA);m.verify_manifest(PARENT,PARENT_SHA)
    return m

def once(text,old,new):
    assert text.count(old)==1,(old[:80],text.count(old))
    return text.replace(old,new,1)

def queues():
    result={}
    for suffix in ('.draft.sh','_declare.sh','_settings.sh'):
        text=(J2/'runs'/('q18a_parity_n3j2'+suffix)).read_text()
        text=text.replace('q18a_parity_n3j2',NAME.replace('-','_')).replace('q18a-parity-n3j2',NAME)
        if suffix=='_settings.sh':
            text=once(text,'FLASH_S5_AHEAD_NOSYNC=1 ','FLASH_S5_AHEAD_NOSYNC=1 FLASH_S5_FUSED_PHASE_EVERY=16 FLASH_S5_NEXT_LAYER_PREFETCH=1 ')
        else:
            anchor="assert env.get('FLASH_S5_AHEAD_NOSYNC')"
            lines=text.splitlines(True);found=[i for i,l in enumerate(lines) if anchor in l]
            assert len(found)==1
            lines.insert(found[0]+1,"assert env.get('FLASH_S5_FUSED_PHASE_EVERY') == '16', 'phase sampling differs'\nassert env.get('FLASH_S5_NEXT_LAYER_PREFETCH') == '1', 'prefetch differs'\n")
            text=''.join(lines)
        result['runs/'+NAME.replace('-','_')+suffix]=text
    return result

def prepare(commit,out):
    b=helper();assert not out.exists();out.mkdir(parents=True)
    bundle=out/(NAME+'-bundle-pending');shutil.copytree(PARENT,bundle)
    for p in (bundle,*bundle.rglob('*')):p.chmod(0o755 if p.is_dir() else 0o644)
    # Keep all prefetch code and its launcher/identity declarations; add the sealed j2 harness.
    for p in J2.iterdir():
        if p.is_file() and (p.name.startswith(('n3j1_','n3j2_','test_n3j','n3_junction','n3_shortest','n3_yield','test_n3_')) or p.name in ('test_compose4_request_ids.py','test_append_yield.py')):
            shutil.copyfile(p,bundle/p.name)
    for p in (J2/'tools').iterdir():
        if p.is_file():
            (bundle/'tools').mkdir(exist_ok=True);shutil.copyfile(p,bundle/'tools'/p.name)
    shutil.copytree(J2/'n3j1_reference',bundle/'n3j1_reference')
    (bundle/'deploy_reference').mkdir()
    (bundle/'deploy_reference/run-flash-prod.sh').write_bytes(b.git(ROOT,'show',commit+':deploy/run-flash-prod.sh'))
    for suffix in ('.draft.sh','_declare.sh','_settings.sh'):
        shutil.copyfile(J2/'runs'/('q18a_parity_n3j1'+suffix),bundle/'runs'/('q18a_parity_n3j1'+suffix))
    # Replay identity adds no RNG draw, prompt, salt, target or execution parameter.
    shutil.copyfile(J2/'compose4_replay.py',bundle/'compose4_replay.py')
    for name in RUNTIME:
        (bundle/'e3_builder'/(name+'.py')).write_bytes(b.git(ROOT,'show',commit+':patches/s5_engine/'+name+'.py'))
    (bundle/'e3_builder/native_candidate_review.json').write_bytes(b.git(ROOT,'show',commit+':patches/s5_engine/native_candidate_review.json'))
    for name in ('test_q2024p_j2.py','test_append_yield.py','q2024p_j2_fixture_leaves.json'):
        (bundle/name).write_bytes(b.git(ROOT,'show',commit+':harness/'+name))
    shutil.copyfile(Path(__file__),bundle/'tools'/Path(__file__).name)
    for name,text in queues().items():
        (bundle/name).write_text(text)
        subprocess.run(['bash','-n',str(bundle/name)],check=True)
    # The accepted generator is reused verbatim except for its three-file delta allowlist.
    source=(J2/'tools/build_n3j2.py').read_text()
    source=once(source,"engine = {'e3/engine_driver.py','e3/scheduler_hooks.py'}","engine = {'e3/engine_driver.py','e3/scheduler_hooks.py','e3/native_auto_arm.py'}")
    module=type(sys)('q2024p_j2_regenerator');module.__file__=str(J2/'tools/build_n3j2.py')
    exec(compile(source,module.__file__,'exec'),module.__dict__)
    module.PARENT=PARENT;proof=module.regenerate(bundle,out)
    assert set(proof['overlay_changed']) <= {'e3/'+n+'.py' for n in RUNTIME}|{'S5-OVERLAP-GENERATION.json','e3/native-implementation-closure.json','e3/native-overlap-readiness.json','e3/decode-probe-evidence.json'}
    # Own queue adapter exercises this packet's draft through j2's real shell/capture proof.
    adapter='''from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import n3j2_queue as old
READER=old.READER
def derive():
    return {'harness/runs/q18a_parity_n3j2'+s:(ROOT/'harness/runs'/('''+repr(NAME.replace('-','_'))+'''+s)).read_text() for s in ('.draft.sh','_declare.sh','_settings.sh')}
def capture_body():
    text=derive()['harness/runs/q18a_parity_n3j2.draft.sh'];at=text.index('capture_logs || true\\n'+READER);return text[at:at+len('capture_logs || true\\n'+READER)]
def final_body():
    text=derive()['harness/runs/q18a_parity_n3j2.draft.sh'];at=text.index('alpha_next_arm() {');return text[at:text.index('capture_end "$(native41_budget end_identity)" || true',at)]
'''
    (bundle/'tools/q2024p_j2_queue.py').write_text(adapter)
    probe=(J2/'tools/n3j2_queue_probe.py').read_text().replace('import n3j2_queue as queue','import q2024p_j2_queue as queue')
    (bundle/'tools/q2024p_j2_queue_probe.py').write_text(probe)
    cpu=(J2/'tools/n3j2_frozen_cpu.py').read_text().replace('from n3j2_queue_probe import run as queue_boundary','from q2024p_j2_queue_probe import run as queue_boundary')
    cpu=once(cpu,'    mounts = []',"    mounts = ['--mount','type=bind,src='+str(bundle/'sparse_stage')+',dst=/src/patches/sparse_stage,readonly','--mount','type=bind,src='+str(bundle/'deploy_reference')+',dst=/src/deploy,readonly']")
    (bundle/'tools/n3j2_frozen_cpu.py').write_text(cpu)
    for p in bundle.rglob('*.py'):compile(p.read_bytes(),str(p),'exec')
    manifest=b.manifest(bundle);final=out/(NAME+'-bundle-'+manifest[:12]);bundle.rename(final)
    owner=J2_BUILD/'q18a-parity-n3j2-owner-run-321e4cd7ffd8.py'
    text=owner.read_text().replace('q18a-parity-n3j2',NAME).replace('q18a_parity_n3j2',NAME.replace('-','_'))
    replacements=dict(H="Path('/home/<service-user>/harness/%s')"%manifest[:12],BUNDLE=repr(manifest),LAUNCHER=repr(b.sha(final/'native30_overlay/run-flash-s5-test.sh')),REVIEW=repr(proof['review_sha256']),EXPECTED_SOURCES=repr(b.verify_manifest(final/'native30_overlay')),SOURCE_COMMIT=repr(commit),CPU_SUITES=repr(SUITES))
    lines=text.splitlines(True);seen=set()
    for node in ast.parse(text).body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in replacements:
            n=node.targets[0].id;assert node.lineno==node.end_lineno;lines[node.lineno-1]=n+'='+replacements[n]+'\n';seen.add(n)
    assert seen==set(replacements);runner=out/(NAME+'-owner-run-'+manifest[:12]+'.py');runner.write_text(''.join(lines))
    record=dict(proof,schema='q2024p-j2-build/1',name=NAME,source_commit=commit,bundle=str(final),manifest=manifest,runner=str(runner),runner_sha256=b.sha(runner),runner_size=runner.stat().st_size,held=True,hardware_qualified=False,parent_manifest=PARENT_SHA,j2_manifest=J2_SHA,cpu_suites=SUITES)
    b.dump(out/'build.json',record);return record

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-commit',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(a.source_commit,a.out),sort_keys=True))
