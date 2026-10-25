#!/usr/bin/env python3
"""Build a NEW serial E2 overlay from independently declared baseline sources.

Offline generation only. Does not install, start or arm anything. The caller
must supply the approved baseline manifest SHA256, never learn-and-trust a
running engine. Every baseline file must be covered; all changed sources have
additional full-file pins. The output still needs the owner's GPU gates.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

import arena_attention
import mirror_hooks
import runner_hooks
import telemetry_hooks

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUNTIME=('retention_contract','arena_contract','block_mirror','serial_dma','arena_dma',
         'runner_lifecycle','startup','exactness')
PINS={
 'model_runner_v2.py':'a49588ad1ab173b1b5b1695d28fa64da7bbfc5a13d7d32d7aaaf9781879069a2',
 'qsa.py':'4ad6e74fe6a6d2678f3944ca4780c61c32e56680da2456869a789d12ef860c69',
 'ops_qsa.py':'5a0d962022ec4e47809fa072d2b13745c84413de894048e67f10c3c10a45d51b',
 'core/block_pool.py':'7f781378710e616624f87d95ccc7b245e1e741c85f3751268e672c2a262038e8',
 'core/scheduler.py':'4d797e5ab6c5abaa06229dd88035ceede7562f1dbe054bdf7ad59be659c04c87',
}
REFERENCE_PINS={
 'v1/worker/gpu/block_table.py':'61c004315d5af7e7eae4e2a9e6be92ea82c520327690a7f55a73bb9ce95f520a',
 # The output schema must match the pinned scheduler; filled from the same image.
 'v1/core/sched/output.py':'3a66c8efaefe10c572f6f33cf1f8aa4610758c945130cad62eb9f7331005677c',
 'config/__init__.py':'56fb0cb1598d297e4f95a97e5d86dc624145d1532f5faa432b7d2834d7155bd5',
 'config/compilation.py':'c9cec5c7200e8e559810ec8c30113ad61dab6780f9f7fb3c116bd0d9b5a43065',
 'config/model.py':'f4168b67cc93ce72ff2a6d4c66974223e32fd0f1fab4c05e44bd68a275e18d69',
}
LAUNCHER_SHA='98d6d2a21927b8943f1ae9b0b43616757e80adac0c628e65f482e4332e3cf126'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(folder):
    folder=Path(folder);result={}
    for line in (folder/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split(maxsplit=1);path=folder/name
        if (len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest) or
                name in result or Path(name).is_absolute() or '..' in Path(name).parts or
                path.is_symlink() or not path.is_file() or sha(path)!=digest):
            raise ValueError('invalid baseline manifest entry')
        result[name]=digest
    if not result:raise ValueError('empty baseline manifest')
    return result


def package(source):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node,ast.ImportFrom) and node.level==0 and node.module in RUNTIME:
            source=source.replace('from '+node.module+' import ','from .'+node.module+' import ')
    ast.parse(source)
    return source


def launcher(source):
    if hashlib.sha256(source.encode()).hexdigest()!=LAUNCHER_SHA:
        raise ValueError('unreviewed launcher source')
    addition='''# E2 ordinary-engine test payload. Always checksummed; switches remain opt-in.
e2=()
if [[ -d $OVERLAY/e2 ]]; then
  [[ ${CORE_OVERLAY:-0} == 1 ]] || { echo "E2 needs the declared core overlay" >&2; exit 2; }
  for f in e2/__init__.py e2_sources/scheduler_output.py e2_sources/block_table.py; do
    grep -q "  $f$" "$OVERLAY/SHA256SUMS" || { echo "E2 mount not checksummed" >&2; exit 2; }
  done
  V=/usr/local/lib/python3.12/dist-packages/vllm/v1
  e2=(-v "$OVERLAY/e2:$V/flash_e2:ro"
      -v "$OVERLAY/e2_sources/scheduler_output.py:$V/core/sched/output.py:ro"
      -v "$OVERLAY/e2_sources/block_table.py:$V/worker/gpu/block_table.py:ro")
fi
'''
    addition=addition.replace('for f in e2/__init__.py ', 'for f in e2/__init__.py '+
                              ' '.join('e2/'+name+'.py' for name in RUNTIME)+' ')
    source=mirror_hooks.once(source,'# VISION=1: image input',addition+'# VISION=1: image input')
    return mirror_hooks.once(source,'  "${runner[@]}" \\\n','  "${runner[@]}" "${e2[@]}" \\\n')


def build(base,out,reference,base_manifest_sha256,launcher_path,core_manifest_sha256):
    base,out,reference=map(Path,(base,out,reference))
    if out.exists():raise ValueError('new output directory required; nothing overwritten')
    if out.resolve().is_relative_to(base.resolve()):raise ValueError('output cannot be inside baseline')
    if sha(base/'SHA256SUMS')!=base_manifest_sha256:raise ValueError('baseline manifest differs from declaration')
    if sha(base/'core/SHA256SUMS')!=core_manifest_sha256:raise ValueError('core manifest differs from declaration')
    root=manifest(base);core=manifest(base/'core')
    covered=set(root)|{'core/'+name for name in core}|{'SHA256SUMS','core/SHA256SUMS'}
    for path in base.rglob('*'):
        if path.is_symlink():raise ValueError('baseline symlinks not supported')
        if path.is_file() and path.relative_to(base).as_posix() not in covered:
            raise ValueError('unmanifested baseline file')
    for rel,digest in PINS.items():
        if rel not in covered or sha(base/rel)!=digest:raise ValueError('unreviewed changed source: '+rel)
    for rel,digest in REFERENCE_PINS.items():
        if sha(reference/rel)!=digest:raise ValueError('unreviewed image source: '+rel)
    changes={
        'ops_qsa.py':arena_attention.generate((base/'ops_qsa.py').read_text()),
        'qsa.py':runner_hooks.qsa_staged((base/'qsa.py').read_text()),
        'model_runner_v2.py':runner_hooks.runner_startup(runner_hooks.runner_execution((base/'model_runner_v2.py').read_text())),
        'core/block_pool.py':mirror_hooks.block_pool(telemetry_hooks.block_pool((base/'core/block_pool.py').read_text())),
        'core/scheduler.py':mirror_hooks.scheduler(telemetry_hooks.scheduler((base/'core/scheduler.py').read_text())),
        'e2_sources/scheduler_output.py':mirror_hooks.scheduler_output((reference/'v1/core/sched/output.py').read_text()),
        'e2_sources/block_table.py':mirror_hooks.block_tables((reference/'v1/worker/gpu/block_table.py').read_text()),
        'e2/__init__.py':'"""Pinned serial E2 arena payload; opt-in, GPU-unqualified until gated."""\n',
    }
    for name in RUNTIME:changes['e2/'+name+'.py']=package((HERE/(name+'.py')).read_text())
    for name,text in changes.items():ast.parse(text,filename=name)
    changes['run-flash-e2-test.sh']=launcher(Path(launcher_path).read_text())
    record=dict(schema=1,base_manifest_sha256=base_manifest_sha256,core_manifest_sha256=core_manifest_sha256,
                base_files={**root,**{'core/'+k:v for k,v in core.items()}},
                changed_source_pins=PINS,reference_source_pins=REFERENCE_PINS,
                telemetry_generator_sha256=telemetry_hooks.SOURCE_SHA,
                generated_files=sorted(changes),status='offline-generated; owner gates required',
                incremental_kv_bytes=0,default_feature='FLASH_E2_ARENA=0',
                experiment='serial dense arena DMA; sparse arithmetic unchanged; no S5 overlap')
    # Finish all validation/transforms before making an output directory.
    shutil.copytree(base,out)
    for name,text in changes.items():
        path=out/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    (out/'run-flash-e2-test.sh').chmod(0o755)
    (out/'E2-GENERATION.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    (out/'core/SHA256SUMS').write_text(''.join(sha(out/'core'/name)+'  '+name+'\n' for name in sorted(core)))
    entries=[]
    for path in sorted(out.rglob('*')):
        if path.is_file() and path!=out/'SHA256SUMS':
            entries.append(sha(path)+'  '+path.relative_to(out).as_posix()+'\n')
    (out/'SHA256SUMS').write_text(''.join(entries))
    # Generated overlays contain code/provenance only. The queue may run as
    # root with umask 077; the <service-user> service must still be able to checksum
    # every source and traverse newly created package directories (q12c).
    # Normalize ONLY the new output, never the approved baseline.
    out.chmod(0o755)
    for path in out.rglob('*'):
        path.chmod(0o755 if path.is_dir() else 0o644 | (path.stat().st_mode & 0o111))
    return dict(manifest_sha256=sha(out/'SHA256SUMS'),generated_files=record['generated_files'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('base',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--reference',required=True,type=Path)
    p.add_argument('--base-manifest-sha256',required=True)
    p.add_argument('--core-manifest-sha256',required=True)
    p.add_argument('--launcher',type=Path,default=ROOT/'deploy/run-flash-prod.sh')
    a=p.parse_args()
    print(json.dumps(build(a.base,a.out,a.reference,a.base_manifest_sha256,a.launcher,a.core_manifest_sha256),sort_keys=True))
