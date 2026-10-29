#!/usr/bin/env python3
"""Fresh-name one-card diagnostic input; keep the native19 overlay byte-identical."""
import argparse
import hashlib
from pathlib import Path
import shutil


BASE_MANIFEST = 'fccef44eee128f8662bbacf90d8dd261990faa492dccc99613fec0263f6f9d83'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu-container', required=True)
    parser.add_argument('--cpu-container', required=True)
    args = parser.parse_args()
    base, out = args.base.resolve(), args.out.absolute()
    assert sha(base / 'SHA256SUMS') == BASE_MANIFEST
    listed = {name: digest for digest, name in
              (line.split(maxsplit=1) for line in (base / 'SHA256SUMS').read_text().splitlines())}
    assert all(sha(base / name) == digest for name, digest in listed.items())
    assert {path.relative_to(base).as_posix() for path in base.rglob('*') if path.is_file()} == (
        set(listed) | {'SHA256SUMS'})
    assert not any(path.is_symlink() for path in base.rglob('*'))
    assert not out.exists() and not out.is_symlink()
    for name in (args.gpu_container, args.cpu_container):
        assert name.startswith('flashnext-native19-indexer-diag-')
        assert all(c.isalnum() or c in '-_' for c in name)
    shutil.copytree(base, out, copy_function=shutil.copyfile)
    for path in (out, *out.rglob('*')):
        path.chmod(0o700 if path.is_dir() else 0o600)
    repo = Path(__file__).resolve().parents[1]
    target = out / 'proof/e3_native19_indexer_diag_probe.py'
    shutil.copyfile(repo / 'harness/e3_native19_indexer_diag_probe.py', target)
    for name, container in (('run_cpu.py', args.cpu_container),
                            ('run_gpu.py', args.gpu_container)):
        path = out / name
        source = path.read_text()
        old_name = ('flashnext-native19-vm-suite-b-hub-cpu' if name == 'run_cpu.py'
                    else 'flashnext-native19-vm-suite-b')
        assert source.count("NAME='" + old_name + "'") == 1
        assert source.count('/proof/e3_native19_vm_suite.py') == 1
        source = source.replace("NAME='" + old_name + "'", "NAME='" + container + "'")
        source = source.replace('/proof/e3_native19_vm_suite.py',
                                '/proof/e3_native19_indexer_diag_probe.py')
        path.write_text(source)
    assert all(sha(out / 'overlay' / path.relative_to(base / 'overlay')) == sha(path)
               for path in (base / 'overlay').rglob('*') if path.is_file())
    files = sorted(path for path in out.rglob('*')
                   if path.is_file() and path.name != 'SHA256SUMS')
    (out / 'SHA256SUMS').write_text(''.join(
        sha(path) + '  ' + path.relative_to(out).as_posix() + '\n' for path in files))
    for path in (out, *out.rglob('*')):
        path.chmod(0o500 if path.is_dir() else 0o400)
    print('INDEXER_DIAG_INPUT', str(out), len(files), sha(out / 'SHA256SUMS'))


if __name__ == '__main__':
    main()
