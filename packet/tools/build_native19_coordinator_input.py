#!/usr/bin/env python3
"""Assemble one immutable CPU coordinator self-run from a pinned prior input.

The inherited import graph is used only for bootstrap shape. Every mounted
module hash is recomputed from the assembled overlay before execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--container', required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    base, root = args.base.resolve(), args.out.absolute()
    assert base.is_dir() and not root.exists() and not root.is_symlink()
    assert args.container.startswith('flashnext-native19-coordinator-')
    assert all(c.isalnum() or c in '-_' for c in args.container)
    root.mkdir()
    shutil.copytree(base / 'overlay', root / 'overlay')
    shutil.copytree(base / 'proof', root / 'proof')
    for path in root.rglob('*'):
        path.chmod(0o700 if path.is_dir() else 0o600)
    sys.path.insert(0, str(repo / 'patches/s5_engine'))
    import make_s5_overlap
    for source in (repo / 'patches/s5_engine').glob('*.py'):
        target = root / 'overlay/e3' / source.name
        if target.is_file():
            target.write_text(make_s5_overlap.package(source.read_text()))
    for name in ('test_e3_native_schedule_profile', 'test_e3_native_submission_credit',
                 'test_e3_r4_worker_bridge', 'test_e3_lane_dispatch',
                 'test_e3_native_pp_stop_order'):
        shutil.copyfile(repo / 'harness' / (name + '.py'),
                        root / 'proof/harness' / (name + '.py'))
    for name in ('e3_native_cow_lease_probe', 'native18_scheduler_fixture'):
        shutil.copyfile(repo / 'harness' / (name + '.py'),
                        root / 'proof' / (name + '.py'))
    shutil.copyfile(repo / 'harness/e3_native_coordinator_slice.py',
                    root / 'proof/e3_native_coordinator_slice.py')
    shutil.copyfile(repo / 'harness/native19_coordinator_fixture_leaves.json',
                    root / 'proof/native19_coordinator_fixture_leaves.json')
    graph_path = root / 'proof/import-graph.json'
    graph = json.loads(graph_path.read_text())
    for row in graph['modules']:
        path = root / 'overlay' / row['source']
        if path.is_file():
            row['sha256'] = sha(path)
    graph['proof_scope'] = ('native19 CPU coordinator composition; inherited import edges, '
                            'current mounted module bytes re-pinned; no CUDA or box claim')
    graph_path.write_text(json.dumps(graph, indent=2) + '\n')
    wrapper = (base / 'run_cpu.py').read_text()
    old = "NAME='flashnext-native19-coordinator-dev-c'"
    assert old in wrapper
    wrapper = wrapper.replace(old, f'NAME={args.container!r}')
    wrapper = wrapper.replace("'900s','docker'", "'1800s','docker'")
    wrapper = wrapper.replace("a=p.parse_args()", "p.add_argument('--scenario',default=None);a=p.parse_args()")
    wrapper = wrapper.replace("'--mode',a.mode]", "'--mode',a.mode]+(['--scenario',a.scenario] if a.scenario else [])")
    (root / 'run_cpu.py').write_text(wrapper)
    files = sorted(path for path in root.rglob('*') if path.is_file())
    (root / 'SHA256SUMS').write_text(''.join(
        f'{sha(path)}  {path.relative_to(root).as_posix()}\n' for path in files))
    for path in root.rglob('*'):
        path.chmod(0o500 if path.is_dir() else 0o400)
    root.chmod(0o500)
    print(json.dumps({'input': str(root), 'listed_files': len(files),
                      'manifest_sha256': sha(root / 'SHA256SUMS')}))


if __name__ == '__main__':
    main()
