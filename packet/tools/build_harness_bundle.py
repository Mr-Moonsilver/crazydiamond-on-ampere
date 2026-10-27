#!/usr/bin/env python3
"""Build a new, self-contained harness bundle from tracked repo sources.

Includes the generated PP grouping payload and reviewed test launcher. Never
modifies a deployed bundle. OUT must not exist. No engine/box access.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def build(out):
    out=out.absolute()
    if out.exists(): raise ValueError('new bundle directory required')
    files=subprocess.check_output(['git','ls-files','-z','harness'],cwd=ROOT).decode().split('\0')
    out.mkdir(mode=0o700)
    for name in filter(None,files):
        source=ROOT/name;dest=out/Path(name).relative_to('harness')
        if source.is_symlink() or not source.is_file(): raise ValueError('unsupported harness entry')
        dest.parent.mkdir(exist_ok=True,parents=True)
        shutil.copy2(source,dest)
    # Immutable public QSA source for VM-only synthetic attention qualification.
    qsa=ROOT/'patches/retain/ops_qsa.py.v8.1'
    if sha(qsa)!='5a0d962022ec4e47809fa072d2b13745c84413de894048e67f10c3c10a45d51b':
        raise ValueError('VM QSA source pin')
    (out/'vmid_sources').mkdir()
    shutil.copy2(qsa,out/'vmid_sources/ops_qsa.py')
    source=ROOT/'patches/ppgrouping/overlay/core'
    manifest={name:digest for digest,name in (line.split(maxsplit=1) for line in (source/'SHA256SUMS').read_text().splitlines())}
    (out/'ppg').mkdir()
    for name in ('block_pool.py','kv_cache_manager.py','kv_cache_utils.py'):
        if sha(source/name)!=manifest[name]: raise ValueError('PP grouping source checksum mismatch')
        shutil.copy2(source/name,out/'ppg'/name)
    shutil.copy2(ROOT/'deploy/run-flash-prod.sh',out/'run-flash-prod.sh.q9')
    shutil.copy2(ROOT/'deploy/setcfg.py',out/'setcfg.reference.py')
    # E2 generator/runtime and the exact pinned image inputs it needs. These
    # are CPU-only build inputs, never generated from a running container.
    sys.path.insert(0,str(ROOT/'patches/s5_engine'))
    import make_e2_arena
    builder=out/'e2_builder';builder.mkdir()
    for name in (*make_e2_arena.RUNTIME,'make_e2_arena','arena_attention',
                 'resident_attention','mirror_hooks','runner_hooks','telemetry_hooks'):
        shutil.copy2(ROOT/'patches/s5_engine'/(name+'.py'),builder/(name+'.py'))
    source=ROOT/'patches/ppgrouping/make_ppgrouping.py'
    if sha(source)!=make_e2_arena.telemetry_hooks.SOURCE_SHA:raise ValueError('E2 pool telemetry source checksum mismatch')
    shutil.copy2(source,builder/'pool_stats_source.py')
    reference=ROOT/'reference/vllm-src'
    for name,digest in make_e2_arena.REFERENCE_PINS.items():
        source=reference/name
        if sha(source)!=digest:raise ValueError('E2 image reference checksum mismatch: '+name)
        dest=out/'e2_reference'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
    # The fixed-capacity sparse union is generated from these exact reviewed
    # kernels. Both the standalone overlay generator and the VM operation
    # probe resolve the input relative to their own s5_engine directory.
    import operation_source
    sparse_source=ROOT/'patches/sparse_stage/kernels.py'
    if sha(sparse_source)!=operation_source.SPARSE_STAGE_SHA256:
        raise ValueError('sparse union input checksum mismatch')
    sparse_dest=out/'sparse_stage/kernels.py'
    sparse_dest.parent.mkdir()
    shutil.copy2(sparse_source,sparse_dest)
    vm_sources=out/'vmid_sources'
    (vm_sources/'sparse_stage').mkdir()
    shutil.copy2(sparse_source,vm_sources/'sparse_stage/kernels.py')
    (vm_sources/'s5_engine').mkdir()
    for name in ('e3_types','sparse_history_dma','qsa_operation','operation_source',
                 'arena_attention','resident_attention','native_source',
                 'row1_oracle'):
        shutil.copy2(ROOT/'patches/s5_engine'/(name+'.py'),
                     vm_sources/'s5_engine'/(name+'.py'))
    # E3 build closure: every tracked CPU-only generator/source module and the
    # complete pinned image reference set make_s5_overlap compiles against.
    # The runtime overlay is generated later from these bytes; the bundle never
    # fetches a checkout or imports vLLM while building.
    import executor_hooks, graph_hooks, model_hooks, probe_control_hooks, worker_hooks
    import capture_streams, engine_offers
    pins=dict(executor_hooks.PINS)
    pins.update(engine_offers.PINS)
    pins.update(capture_streams.PINS)
    pins.update(worker_hooks.PINS)
    pins.update(worker_hooks.CAPTURE_REFERENCE_PINS)
    pins.update(model_hooks.PINS)
    pins.update(probe_control_hooks.PINS)
    pins['v1/worker/gpu/model_runner.py']=graph_hooks.PINNED_SHA256
    tracked=subprocess.check_output(
        ['git','ls-files','-z','patches/s5_engine'],cwd=ROOT).decode().split('\0')
    tracked=[name for name in tracked if name and Path(name).suffix in ('.py','.cu','.cpp','.sh','.cuh','.h')]
    if not tracked:raise ValueError('E3 builder sources are not tracked')
    e3_builder=out/'e3_builder';e3_builder.mkdir()
    builder_sources={}
    for name in tracked:
        source=ROOT/name
        if source.is_symlink() or not source.is_file():raise ValueError('unsupported E3 builder entry: '+name)
        rel=Path(name).relative_to('patches/s5_engine').as_posix()
        dest=e3_builder/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
        builder_sources[rel]=sha(source)
    pp_source=ROOT/'patches/ppgrouping/make_ppgrouping.py'
    shutil.copy2(pp_source,e3_builder/'pool_stats_source.py')
    builder_sources['pool_stats_source.py']=sha(pp_source)
    qualification_pins={
        'models/qwen4_exp/nvidia/qsa.py':'4ad6e74fe6a6d2678f3944ca4780c61c32e56680da2456869a789d12ef860c69',
        'v1/worker/gpu/input_batch.py':'abfbb9075a80354bd4bb59d2729fcdea3c62eed827067a823761c98356670ad1',
        'entrypoints/openai/completion/protocol.py':'0be20e3d7f21ab6cbe0ba77cb50068aa39a35fc0369ef38de068ac38ca9800aa',
        'entrypoints/openai/completion/serving.py':'75d05ddddf303734a078a3e42fcb173fef58ea56da09bc0b5c7525cb9bbbf527',
    }
    for name,digest in {**pins,**qualification_pins}.items():
        source=reference/name
        if sha(source)!=digest:raise ValueError('E3 image reference checksum mismatch: '+name)
        dest=out/'e3_reference'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
    closure={'schema':1,'builder_sources':dict(sorted(builder_sources.items())),
             'external_source_pins':{'sparse_stage/kernels.py':sha(sparse_source)},
             'reference_pins':dict(sorted(pins.items())),
             'qualification_reference_pins':qualification_pins,
             'basis':'tracked CPU-only generator/native/source bytes; reference tree verified against every transform pin',
             'runtime_overlay':'generated separately by make_s5_overlap; not fetched on the box'}
    (out/'e3_build_closure.json').write_text(json.dumps(closure,indent=2,sort_keys=True)+'\n')
    needed=('identity_declare.py','identity_declare_stub.py','pp_release_mount_policy.json','pp_canary_bridge.py',
            'e3_startup_mount_policy.json','e3_startup_grade.py','runs/q17_decode_startup.draft.sh',
            'runs/q17_startup_settings.sh','runs/q17s_declare.sh')
    if not all((out/name).is_file() for name in needed):
        raise ValueError('required helper missing: commit new sources before building')
    with (out/'SHA256SUMS').open('x') as f:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='SHA256SUMS':
                f.write(sha(p)+'  '+p.relative_to(out).as_posix()+'\n')
    return sha(out/'SHA256SUMS')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);a=p.parse_args()
    print('bundle SHA256SUMS sha256='+build(a.out))
