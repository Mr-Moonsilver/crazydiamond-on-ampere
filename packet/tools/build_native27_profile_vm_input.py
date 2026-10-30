#!/usr/bin/env python3
"""Package a fresh one-card native27 refusal-receipt proof; never runs a GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BASE = Path('<hub-dir>/agents/flash-next/tmp/native25-census-vm-a-input')
BASE_SHA = '1d06581ff544882397a4036cdce66b7018c844935755acb40587e38947c5df2a'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(root):
    rows = dict(line.split(maxsplit=1)[::-1]
                for line in (root / 'SHA256SUMS').read_text().splitlines())
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    assert actual == set(rows) | {'SHA256SUMS'}
    assert all(sha(root / name) == digest for name, digest in rows.items())
    assert not any(p.is_symlink() for p in root.rglob('*'))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--name', default='native27-profile-vm-a')
    args = parser.parse_args()
    out = args.out.absolute()
    assert not out.exists() and not out.is_symlink()
    assert sha(BASE / 'SHA256SUMS') == BASE_SHA
    inventory(BASE)
    out.mkdir(mode=0o700)
    shutil.copytree(BASE / 'overlay', out / 'overlay')
    proof = out / 'proof'
    proof.mkdir()
    sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
    import make_s5_overlap
    source = ROOT / 'patches/s5_engine/engine_driver.py'
    generated = out / 'overlay/e3/engine_driver.py'
    old_sha = sha(generated)
    packaged = make_s5_overlap.package(source.read_text())
    assert hashlib.sha256(packaged.encode()).hexdigest() != old_sha
    generated.chmod(0o600)
    generated.write_text(packaged)
    new_sha = sha(generated)
    graph = json.loads((BASE / 'proof/import-graph.json').read_text())
    assert graph['source_hashes']['e3/engine_driver.py'] == old_sha
    graph['source_hashes']['e3/engine_driver.py'] = new_sha
    row = [r for r in graph['modules'] if r['module'] == 'vllm.v1.flash_e3.engine_driver']
    assert len(row) == 1 and row[0]['sha256'] == old_sha
    row[0]['sha256'] = new_sha
    graph['overlay'] = str(out / 'overlay')
    (proof / 'import-graph.json').write_text(json.dumps(graph, sort_keys=True, indent=2) + '\n')
    # The connected CPU fixture suite runs from the exact pinned source tree,
    # while the mounted engine_driver method bodies are AST-compared in-probe.
    for prefix in ('harness', 'patches/s5_engine'):
        names = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '-z', '--', prefix])
        for raw in names.split(b'\0'):
            if not raw:
                continue
            name = Path(raw.decode())
            src = ROOT / name
            assert src.is_file() and not src.is_symlink()
            dst = proof / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    reference = ROOT / 'reference/vllm-src'
    assert reference.is_dir()
    for src in reference.rglob('*'):
        if not src.is_file() or '__pycache__' in src.parts or src.suffix == '.pyc':
            continue
        assert not src.is_symlink()
        dst = proof / 'reference/vllm-src' / src.relative_to(reference)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    for name in ('e3_native27_profile_vm_probe.py', 'native27_profile_fixture_leaves.json',
                 'e3_native27_profile_receipt.py'):
        shutil.copyfile(ROOT / 'harness' / name, proof / name)
    shutil.copyfile(ROOT / 'harness/native27_profile_cpu_selfrun.py', out / 'run_cpu.py')
    wrapper = (BASE / 'run_gpu.py').read_text()
    wrapper = wrapper.replace("NAME='flashnext-native25-census-vm-a'",
                              "NAME='flashnext-" + args.name + "'")
    wrapper = wrapper.replace('e3_native25_census_probe.py', 'e3_native27_profile_vm_probe.py')
    wrapper = wrapper.replace('native25-replay-census-owner/1', 'native27-profile-vm-owner/1')
    (out / 'run_gpu.py').write_text(wrapper)
    rows = {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*') if p.is_file()}
    (out / 'SHA256SUMS').write_text(''.join(
        digest + '  ' + name + '\n' for name, digest in sorted(rows.items())))
    assert inventory(out) == rows
    for path in out.rglob('*'):
        path.chmod(0o500 if path.is_dir() else 0o400)
    out.chmod(0o500)
    print(json.dumps({'input': str(out), 'manifest_sha256': sha(out / 'SHA256SUMS'),
                      'files': len(rows), 'wrapper_sha256': sha(out / 'run_gpu.py'),
                      'probe_sha256': sha(proof / 'e3_native27_profile_vm_probe.py'),
                      'leaves_sha256': sha(proof / 'native27_profile_fixture_leaves.json'),
                      'generated_driver_sha256': new_sha}, sort_keys=True))


if __name__ == '__main__':
    main()
