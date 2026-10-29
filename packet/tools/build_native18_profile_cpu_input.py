#!/usr/bin/env python3
"""CPU input assembly only; not a candidate/readiness bundle or VM GO.

Base is an owner-retained immutable proof input. Package transforms and attention
source generator are the real builder functions. Inherited graph edges are not
a new import-audit result; module file hashes are recomputed and verified at run.
"""
import sys,json,shutil,hashlib,ast
from pathlib import Path
import argparse
p=argparse.ArgumentParser(description='Build a fresh CPU-only native18 host-control development input')
p.add_argument('--base',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--container',required=True)
a=p.parse_args()
repo=Path(__file__).resolve().parents[1]
base=a.base.resolve();root=a.out.absolute();name=a.container
assert base.is_dir() and not root.exists() and not root.is_symlink()
assert name.startswith('flashnext-native18-') and all(c.isalnum() or c in '-_' for c in name)
root.mkdir();shutil.copytree(base/'overlay',root/'overlay');shutil.copytree(base/'proof',root/'proof')
for p in root.rglob('*'):p.chmod(0o700 if p.is_dir() else 0o600)
sys.path.insert(0,str(repo/'patches/s5_engine'))
import make_s5_overlap
for name0 in ['engine_driver','executor_hooks','wire_v1','qsa_operation','qsa_driver','lane_dispatch','worker_bridge','operation_source']:
 (root/'overlay/e3'/f'{name0}.py').write_text(make_s5_overlap.package((repo/'patches/s5_engine'/f'{name0}.py').read_text()))
sys.path.insert(0,str(repo/'patches/s5_engine'))
import arena_attention,operation_source
src=(repo/'reference/vllm-src/models/qwen4_exp/nvidia/ops/qsa.py').read_text()
(root/'overlay/ops_qsa.py').write_text(operation_source.generate_ops(arena_attention.generate(src)))
for file in ['e3_native_profile_cpu_probe.py','native18_scheduler_fixture.py']:
 shutil.copyfile(repo/'harness'/file,root/'proof'/file)
(root/'proof/harness').mkdir();(root/'proof/builders').mkdir()
for name0 in ['test_e3_native_schedule_profile','test_e3_native_submission_credit','test_e3_native_selection_asymmetry','test_e3_r4_worker_bridge','test_e3_admission_native','test_e3_lane_dispatch','test_e3_sparse_history_dma','vmid_sparse_plan']:
 shutil.copyfile(repo/'harness'/f'{name0}.py',root/'proof/harness'/f'{name0}.py')
# Builder-only imports used by test-module collection, never engine modules.
for name0 in ['resident_attention','mirror_hooks','telemetry_hooks']:
 assert not (root/'overlay/e3'/f'{name0}.py').exists()
 shutil.copyfile(repo/'patches/s5_engine'/f'{name0}.py',root/'proof/builders'/f'{name0}.py')
pending=list((root/'proof/harness').glob('*.py'))+list((root/'proof/builders').glob('*.py'));seen=set()
while pending:
 p=pending.pop()
 if p in seen:continue
 seen.add(p)
 for n in ast.walk(ast.parse(p.read_text())):
  names=([n.module.split('.')[0]] if isinstance(n,ast.ImportFrom) and n.module and not n.level else [a.name.split('.')[0] for a in n.names] if isinstance(n,ast.Import) else [])
  for nm in names:
   if (root/'overlay/e3'/f'{nm}.py').exists() or (root/'overlay/e2'/f'{nm}.py').exists():continue
   for category,dst in [('harness','harness'),('patches/s5_engine','builders')]:
    src=repo/category/f'{nm}.py';target=root/'proof'/dst/f'{nm}.py'
    if src.is_file() and not target.exists():shutil.copyfile(src,target);pending.append(target)
g=json.loads((root/'proof/import-graph.json').read_text())
for row in g['modules']:
 p=root/'overlay'/row['source']
 if p.is_file():row['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
g['proof_scope']='host profile-refusal composition; import index inherited, module bytes re-pinned; no new full import-audit claim'
(root/'proof/import-graph.json').write_text(json.dumps(g,indent=2)+'\n')
s=(base/'run_cpu.py').read_text().replace("NAME='flashnext-native17-cow-executor-b-hub-cpu'",f'NAME={name!r}').replace('e3_native_cow_lease_probe.py','e3_native_profile_cpu_probe.py')
(root/'run_cpu.py').write_text(s)
files=sorted(p for p in root.rglob('*') if p.is_file())
(root/'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root)}\n' for p in files))
for p in root.rglob('*'):p.chmod(0o500 if p.is_dir() else 0o400)
root.chmod(0o500)
print(root,len(files))
