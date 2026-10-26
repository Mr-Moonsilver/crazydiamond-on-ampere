#!/usr/bin/env python3
"""Create a new final q17b overlay after the owned CPU-only native build.

Never mutates the generated/build directory. Only the three declared build
artifacts may be outside its original manifest. This is packaging, not an ABI,
GPU, startup or replay qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

IMAGE = 'sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b'
CUPTI = 'e2f9ed861fe27c492b8bb52b5e3220ef5120f3edcda36312e96b7fd8a186be3e'
BUILD = 'e3/native/q17-helper-build.json'
LIBS = ('flash_s5_green.so', 'flash_s5_replay_observer.so')
SOURCES = ('flash_s5_green.cu', 'flash_s5_replay_observer.cpp', 'flash_s5_graph_bytes.cpp')
EXTRA = {BUILD, *(f'e3/native/{name}' for name in LIBS)}
GENERATION = 'S5-OVERLAP-GENERATION.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(root):
    result = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        p = Path(name)
        if (len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)
                or name in result or p.is_absolute() or '..' in p.parts
                or not (root/p).is_file() or (root/p).is_symlink()
                or sha(root/p) != digest):
            raise ValueError('source manifest is invalid')
        result[name] = digest
    if not result:
        raise ValueError('source manifest is empty')
    return result


def finalize(source, out):
    source, out = Path(source).absolute(), Path(out).absolute()
    if out.exists() or out.is_symlink() or source.is_symlink():
        raise ValueError('fresh output and regular source required')
    if out.resolve().is_relative_to(source.resolve()):
        raise ValueError('output must be outside the immutable source')
    declared = manifest(source)
    entries = list(source.rglob('*'))
    if any(p.is_symlink() or (not p.is_file() and not p.is_dir()) for p in entries):
        raise ValueError('unsupported package entry')
    actual = {p.relative_to(source).as_posix() for p in entries if p.is_file()}
    if actual != set(declared) | EXTRA | {'SHA256SUMS'} or set(declared) & EXTRA:
        raise ValueError('unexpected or missing native build artifacts')
    receipt = json.loads((source/BUILD).read_text())
    if (receipt.get('schema') != 1 or receipt.get('image') != IMAGE
            or receipt.get('gpu_work') is not False
            or receipt.get('cupti_library_sha256') != CUPTI
            or receipt.get('declared_abi') != {'green': 1, 'observer': 1}):
        raise ValueError('native build identity differs')
    if set(receipt.get('files', {})) != set(LIBS + SOURCES):
        raise ValueError('native build file inventory differs')
    for name in LIBS + SOURCES:
        if receipt['files'][name] != sha(source/'e3/native'/name):
            raise ValueError('native build hash mismatch')
    for name in LIBS:
        if not (source/'e3/native'/name).read_bytes().startswith(b'\x7fELF'):
            raise ValueError('native output is not ELF')
    needed = receipt.get('needed', {})
    if (set(needed) != set(LIBS)
            or any(type(needed[n]) is not list or 'libcuda.so.1' not in needed[n] for n in LIBS)
            or 'libcupti.so.13' not in needed[LIBS[1]]
            or any('libcupti' in x and x != 'libcupti.so.13' for xs in needed.values() for x in xs)):
        raise ValueError('native dependency identity differs')
    generation = json.loads((source/GENERATION).read_text())
    if generation.get('armable') is not False or 'native_build' in generation:
        raise ValueError('package is not a fresh unarmed diagnostic')
    generation['native_build'] = dict(receipt=receipt, receipt_sha256=sha(source/BUILD),
                                    source_manifest_sha256=sha(source/'SHA256SUMS'),
                                    source_generation_sha256=sha(source/GENERATION))
    generation['generated_files'] = sorted(set(generation['generated_files']) | EXTRA)
    out.mkdir(mode=0o755)
    for path in entries:
        rel = path.relative_to(source)
        if path.is_file() and rel.as_posix() not in ('SHA256SUMS', GENERATION):
            target = out/rel
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            shutil.copy2(path, target)
    (out/GENERATION).write_text(json.dumps(generation, indent=2, sort_keys=True)+'\n')
    lines = [sha(p)+'  '+p.relative_to(out).as_posix()+'\n'
             for p in sorted(out.rglob('*')) if p.is_file()]
    with (out/'SHA256SUMS').open('x') as f:
        f.writelines(lines)
    # Owner queues use umask 077; public overlay bytes must still be readable
    # by the service user. No private credential file belongs to this package.
    out.chmod(0o755)
    for path in out.rglob('*'):
        path.chmod(0o755 if path.is_dir() else 0o644 | (path.stat().st_mode & 0o111))
    return {'manifest_sha256': sha(out/'SHA256SUMS'), 'armable': False,
            'green_library_sha256': receipt['files'][LIBS[0]],
            'observer_library_sha256': receipt['files'][LIBS[1]],
            'cupti_library_sha256': CUPTI}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('out', type=Path)
    a = p.parse_args()
    print(json.dumps(finalize(a.source, a.out), sort_keys=True))


if __name__ == '__main__':
    main()
