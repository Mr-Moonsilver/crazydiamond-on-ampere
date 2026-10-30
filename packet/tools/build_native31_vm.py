#!/usr/bin/env python3
"""Fresh one-card input, actual native31 engine + B5 passing scenario."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'patches/s5_engine')]
from build_native30_held import sha,dump,manifest,verify
from native_readiness import digest,payload_identity,verify_finalized_package
from audit_native_runtime_imports import audit
BASE=Path('<hub-dir>/agents/flash-next/tmp/native29-overlap-codex-b5-input')

def build(bundle,out):
    assert not out.exists();verify(BASE);verify(bundle);shutil.copytree(BASE,out)
    candidate=out.name.removesuffix('-input')
    for p in [out,*out.rglob('*')]:p.chmod(0o755 if p.is_dir() else 0o644)
    previous=verify(BASE/'overlay');overlay=out/'overlay'
    for name in previous:
        shutil.copyfile(bundle/'native30_overlay'/name,overlay/name)
    # Plugin files were never in this VM overlay. Adjust identity, engine same.
    gen=json.loads((overlay/'S5-OVERLAP-GENERATION.json').read_text());ready=json.loads((overlay/'e3/native-overlap-readiness.json').read_text())
    gen['package_sha256']=ready['package_sha256']=payload_identity(overlay)
    ready['receipt_sha256']=digest({k:v for k,v in ready.items() if k!='receipt_sha256'})
    dump(overlay/'S5-OVERLAP-GENERATION.json',gen);dump(overlay/'e3/native-overlap-readiness.json',ready);manifest(overlay)
    closure=json.loads((overlay/'e3/native-implementation-closure.json').read_text());verify_finalized_package(overlay,closure['generated_source_shas'])
    for name in ('run_gpu.py','run_gpu_async.py','run_gpu_pressure.py'):
        p=out/name;s=p.read_text().replace('native29-overlap-codex-b5',candidate);p.write_text(s)
    # Existing proof sources, scenario and all constructor leaves remain B5's.
    shutil.copyfile(ROOT/'tools/native31_probe_archive.py',out/'proof/native31_probe_archive.py')
    p=out/'proof/e3_native29_overlap_probe.py';s=p.read_text();anchor='            args.out.write_text(json.dumps(record, indent=2, sort_keys=True, default=str)'
    assert s.count(anchor)==1;s=s.replace(anchor,"            from native31_probe_archive import archive_observer\n            archive_observer(record,args.out.parent)\n"+anchor);p.write_text(s)
    launcher=sha(overlay/'run-flash-s5-test.sh');assert launcher==sha(BASE/'overlay/run-flash-s5-test.sh')=='22473a909dad1a4cb8e9c22de88349e65ff1eb46b0fa05b281504afb21775f3c'
    graph=audit(overlay,ROOT,launcher_sha256=launcher);assert graph['verdict']=='PASS',graph['runtime_ownership_errors'];dump(out/'proof/import-graph.json',graph)
    provenance=json.loads((out/'provenance.json').read_text());provenance.update(name=candidate,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),base_input=str(BASE),base_manifest_sha256=sha(BASE/'SHA256SUMS'),overlay_source=str(bundle/'native30_overlay'),engine_scope='native31 exact box engine Python/native bytes; plugins not mounted in VM; same B5 scenario',gpu_run=False,box_action=False)
    dump(out/'provenance.json',provenance)
    leaf=json.loads((out/'fixture-leaves.json').read_text())
    leaf['leaves'].append({'name':'native31-observer-archive','replaces':'primary probe embedding of full observer_end','equivalence':'lossless XZ archive plus original verdict and first 32 failure summaries, checksum and archive path','cannot_prove':'no placement/performance promotion; existing observer overflow remains visible','box_evidence':False})
    dump(out/'fixture-leaves.json',leaf);dump(out/'proof/native29_overlap_fixture_leaves.json',leaf)
    manifest(out);verify(out)
    result=dict(verdict='PASS',path=str(out),manifest=sha(out/'SHA256SUMS'),box_overlay=str(bundle/'native30_overlay'),engine_byte_identical=True,scenario_changes='none; output serialization only',cpu_fixture_manifest=provenance['cpu_fixture_manifest'],cpu_fixtures=provenance['cpu_fixtures'],gpu_execution=False)
    dump(out.parent/(out.name+'-build.json'),result)
    for p in [out,*out.rglob('*')]:p.chmod(0o555 if p.is_dir() else 0o444)
    print(json.dumps(result,sort_keys=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.bundle,a.out)
