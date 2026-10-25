#!/usr/bin/env python3
"""Build a new, self-contained harness bundle from tracked repo sources.

Includes the generated PP grouping payload and reviewed test launcher. Never
modifies a deployed bundle. OUT must not exist. No engine/box access.
"""
import argparse
import hashlib
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
    needed=('identity_declare.py','identity_declare_stub.py','pp_release_mount_policy.json','pp_canary_bridge.py')
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
