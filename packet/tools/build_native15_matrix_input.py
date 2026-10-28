#!/usr/bin/env python3
"""Build fresh immutable matrix proof input; CPU only, no engine/box access."""
import argparse,ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def build(overlay,graph_path,out):
    graph=json.loads(graph_path.read_text());assert graph['verdict']=='PASS'
    assert not out.exists();out.mkdir(parents=True)
    def copy(source,target):
        assert source.is_file() and not source.is_symlink()
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert target.read_bytes()==source.read_bytes()
        else:target.write_bytes(source.read_bytes())
    for mount in graph['mounts']:
        if not mount['active']:continue
        source=overlay/mount['source'];assert source.exists() and not source.is_symlink()
        files=[source] if source.is_file() else [p for p in source.rglob('*') if p.is_file()]
        for p in files:
            assert p.suffix!='.pyc' and '__pycache__' not in p.parts
            copy(p,out/'overlay'/p.relative_to(overlay))
    pending=['e3_native_transitions_matrix_probe'];seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name);source=ROOT/'harness'/(name+'.py');assert source.is_file()
        copy(source,out/'proof'/(name+'.py'))
        for node in ast.walk(ast.parse(source.read_text())):
            imports=([node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else
                     [alias.name.split('.')[0] for alias in node.names] if isinstance(node,ast.Import) else [])
            for other in imports:
                if (ROOT/'harness'/(other+'.py')).is_file():pending.append(other)
    copy(ROOT/'harness/e3_teacher_decode_rule4_calibration.json',out/'proof/e3_teacher_decode_rule4_calibration.json')
    copy(ROOT/'harness/native15_transitions_fixture_leaves.json',out/'proof/fixture-leaves.json')
    copy(graph_path,out/'proof/import-graph.json')
    gpu=(ROOT/'tools/native15_staged_device_owner.py').read_text()
    gpu=gpu.replace("NAME='flashnext-native15-staged-boundaries'","NAME='flashnext-native15-matrix1b'")
    gpu=gpu.replace("'--env','VLLM_CACHE_ROOT=/tmp/vllm'",
                    "'--env','HOME=/tmp/flashnext-home','--env','XDG_CACHE_HOME=/tmp/flashnext-cache',"
                    "'--env','FLASHINFER_WORKSPACE_BASE=/tmp/flashnext-home',"
                    "'--env','VLLM_CACHE_ROOT=/tmp/vllm'")
    gpu=gpu.replace('e3_native_staged_plan_device_probe.py','e3_native_transitions_matrix_probe.py')
    gpu=gpu.replace("'--graph','/proof/import-graph.json','--out','/out/probe.json']",
                    "'--graph','/proof/import-graph.json','--out','/out/probe.json','--mode','gpu-all']")
    gpu=gpu.replace("schema='native15-device-owner/1'","schema='native15-matrix-owner/1'")
    (out/'run_gpu.py').write_text(gpu)
    copy(ROOT/'tools/native15_matrix_cpu_owner.py',out/'run_cpu.py')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    files=sorted(p for p in out.rglob('*') if p.is_file())
    (out/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.relative_to(out).as_posix()+'\n' for p in files))
    result=dict(input=str(out),manifest_sha256=sha(out/'SHA256SUMS'),listed=len(files),total=len(files)+1,
        probe_sha256=sha(out/'proof/e3_native_transitions_matrix_probe.py'),graph_sha256=sha(out/'proof/import-graph.json'),
        cpu_owner_sha256=sha(out/'run_cpu.py'),gpu_owner_sha256=sha(out/'run_gpu.py'),proof_modules=sorted(seen))
    for p in [out,*out.rglob('*')]:p.chmod(p.stat().st_mode&~0o222)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--overlay',type=Path,required=True);p.add_argument('--graph',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.overlay.resolve(),a.graph.resolve(),a.out.absolute()),indent=2))
