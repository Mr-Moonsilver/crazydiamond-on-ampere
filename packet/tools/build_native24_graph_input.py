#!/usr/bin/env python3
"""Package a fresh one-card proof of the native observer's retained graphs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


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
    p.add_argument('--name', default='native24-graph-capture-a')
    args = p.parse_args()
    base, out = args.base_input.resolve(), args.out.absolute()
    assert base.is_dir() and not out.exists() and not out.is_symlink()
    manifest(base)
    out.mkdir(mode=0o700)
    shutil.copytree(base / 'overlay', out / 'overlay')
    proof = out / 'proof'
    proof.mkdir()
    sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
    import make_s5_overlap
    source = ROOT / 'patches/s5_engine/worker_hooks.py'
    packaged = make_s5_overlap.package(source.read_text())
    generated = out / 'overlay/e3/worker_hooks.py'
    old = generated.read_text()
    assert old != packaged
    generated.chmod(0o600)
    generated.write_text(packaged)
    graph = json.loads((base / 'proof/import-graph.json').read_text())
    assert graph['verdict'] == 'PASS'
    assert graph['source_hashes']['e3/worker_hooks.py'] == hashlib.sha256(old.encode()).hexdigest()
    digest = sha(generated)
    graph['source_hashes']['e3/worker_hooks.py'] = digest
    module = [item for item in graph['modules']
              if item['module'] == 'vllm.v1.flash_e3.worker_hooks']
    assert len(module) == 1 and module[0]['sha256'] == hashlib.sha256(old.encode()).hexdigest()
    module[0]['sha256'] = digest
    graph['overlay'] = str(out / 'overlay')
    (proof / 'import-graph.json').write_text(json.dumps(graph, indent=2, sort_keys=True) + '\n')
    for name in ('e3_native24_graph_probe.py', 'native24_graph_fixture_leaves.json'):
        shutil.copyfile(ROOT / 'harness' / name, proof / name)
    shutil.copyfile(ROOT / 'harness/native24_graph_cpu_selfrun.py', out / 'run_cpu.py')
    wrapper = (base / 'run_gpu.py').read_text()
    assert "NAME='flashnext-native23-row4-pause-e'" in wrapper
    wrapper = wrapper.replace("NAME='flashnext-native23-row4-pause-e'",
                              "NAME='flashnext-" + args.name + "'")
    wrapper = wrapper.replace('e3_native23_row4_pause_probe.py',
                              'e3_native24_graph_probe.py')
    wrapper = wrapper.replace('native23-row4-pause-owner/1',
                              'native24-graph-capture-owner/1')
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
                      'probe_sha256':sha(proof / 'e3_native24_graph_probe.py'),
                      'worker_hooks_generated_sha256':digest}, sort_keys=True))


if __name__ == '__main__':
    main()
