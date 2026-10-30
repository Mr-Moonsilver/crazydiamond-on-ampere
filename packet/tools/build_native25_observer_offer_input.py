#!/usr/bin/env python3
"""Package a fresh one-card observer-offer proof from the path B input."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(root):
    entries = dict(line.split(maxsplit=1)[::-1]
                   for line in (root / 'SHA256SUMS').read_text().splitlines())
    files = {path.relative_to(root).as_posix() for path in root.rglob('*')
             if path.is_file()}
    assert files == set(entries) | {'SHA256SUMS'}
    assert not any(path.is_symlink() for path in root.rglob('*'))
    assert all(sha(root / name) == digest for name, digest in entries.items())
    return entries


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--name', default='native25-observer-offer-c')
    args = p.parse_args()
    base, out = args.base_input.resolve(), args.out.absolute()
    assert base.is_dir() and not out.exists() and not out.is_symlink()
    manifest(base)
    out.mkdir(mode=0o700)
    shutil.copytree(base / 'overlay', out / 'overlay')
    proof = out / 'proof'
    proof.mkdir()
    shutil.copyfile(base / 'proof/import-graph.json', proof / 'import-graph.json')
    shutil.copyfile(ROOT / 'harness/e3_native25_observer_offer_probe.py',
                    proof / 'e3_native25_observer_offer_probe.py')
    shutil.copyfile(ROOT / 'harness/native25_observer_offer_fixture_leaves.json',
                    proof / 'native25_observer_offer_fixture_leaves.json')
    shutil.copyfile(ROOT / 'harness/native25_observer_offer_cpu_selfrun.py',
                    out / 'run_cpu.py')
    wrapper = (base / 'run_gpu.py').read_text()
    assert "NAME='flashnext-native25-observer-path-b'" in wrapper
    wrapper = wrapper.replace("NAME='flashnext-native25-observer-path-b'",
                              "NAME='flashnext-" + args.name + "'")
    wrapper = wrapper.replace('e3_native25_observer_path_probe.py',
                              'e3_native25_observer_offer_probe.py')
    wrapper = wrapper.replace('native25-observer-path-owner/1',
                              'native25-observer-offer-owner/1')
    (out / 'run_gpu.py').write_text(wrapper)
    entries = [f'{sha(path)}  {path.relative_to(out).as_posix()}'
               for path in sorted(out.rglob('*')) if path.is_file()]
    (out / 'SHA256SUMS').write_text('\n'.join(entries) + '\n')
    manifest(out)
    for path in out.rglob('*'):
        path.chmod(0o500 if path.is_dir() else 0o400)
    out.chmod(0o500)
    print(json.dumps({'input':str(out), 'name':args.name,
                      'files':len(entries), 'manifest_sha256':sha(out / 'SHA256SUMS'),
                      'wrapper_sha256':sha(out / 'run_gpu.py'),
                      'probe_sha256':sha(proof / 'e3_native25_observer_offer_probe.py')},
                     sort_keys=True))


if __name__ == '__main__':
    main()
