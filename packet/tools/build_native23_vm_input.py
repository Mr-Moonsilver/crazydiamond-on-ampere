#!/usr/bin/env python3
"""Build a fresh immutable native23 one-card pause/resume proof input."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
PROOF_FILES = (
    'e3_native23_row4_pause_probe.py',
    'e3_native_stock_rpc_probe.py',
    'e3_native_transitions_lifecycle_probe.py',
    'e3_native_qualification_driver.py',
    'native23_row4_pause_fixture_leaves.json',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_manifest(root):
    entries = dict(line.split(maxsplit=1)[::-1]
                   for line in (root / 'SHA256SUMS').read_text().splitlines())
    files = {item.relative_to(root).as_posix() for item in root.rglob('*')
             if item.is_file()}
    assert files == set(entries) | {'SHA256SUMS'}
    assert not any(item.is_symlink() for item in root.rglob('*'))
    assert all(sha(root / name) == digest for name, digest in entries.items())
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--name', default='native23-row4-pause-a')
    args = parser.parse_args()
    base = args.base_input.resolve()
    out = args.out.absolute()
    assert base.is_dir() and not out.exists() and not out.is_symlink()
    check_manifest(base)
    out.mkdir(mode=0o700)
    shutil.copytree(base / 'overlay', out / 'overlay')
    proof = out / 'proof'
    proof.mkdir()
    graph = json.loads((base / 'proof/import-graph.json').read_text())
    assert graph['verdict'] == 'PASS'
    graph['overlay'] = str(out / 'overlay')
    (proof / 'import-graph.json').write_text(json.dumps(graph, indent=2, sort_keys=True) + '\n')
    for name in PROOF_FILES:
        shutil.copyfile(ROOT / 'harness' / name, proof / name)
    shutil.copyfile(ROOT / 'harness/native23_vm_cpu_selfrun.py', out / 'run_cpu.py')
    wrapper = (base / 'run_gpu.py').read_text()
    wrapper = wrapper.replace('flashnext-native21-rpc-device-f', 'flashnext-' + args.name)
    wrapper = wrapper.replace('e3_native21_rpc_device_probe.py',
                              'e3_native23_row4_pause_probe.py')
    wrapper = wrapper.replace('native21-rpc-device-owner/1',
                              'native23-row4-pause-owner/1')
    (out / 'run_gpu.py').write_text(wrapper)
    entries = []
    for item in sorted(out.rglob('*')):
        if item.is_file():
            entries.append(f'{sha(item)}  {item.relative_to(out).as_posix()}')
    (out / 'SHA256SUMS').write_text('\n'.join(entries) + '\n')
    check_manifest(out)
    for item in out.rglob('*'):
        item.chmod(0o500 if item.is_dir() else 0o400)
    out.chmod(0o500)
    print(json.dumps({'input': str(out), 'name': args.name,
                      'files': len(entries), 'manifest_sha256': sha(out / 'SHA256SUMS'),
                      'wrapper_sha256': sha(out / 'run_gpu.py'),
                      'probe_sha256': sha(proof / 'e3_native23_row4_pause_probe.py'),
                      'candidate_driver_sha256': sha(proof / 'e3_native_qualification_driver.py')},
                     sort_keys=True))


if __name__ == '__main__':
    main()
