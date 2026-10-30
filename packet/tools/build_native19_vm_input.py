#!/usr/bin/env python3
"""Build one immutable native19 VM proof input; no GPU or box action."""
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
    parser.add_argument('--gpu-container', required=True)
    parser.add_argument('--cpu-container', required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    base, root = args.base.resolve(), args.out.absolute()
    assert base.is_dir() and not root.exists() and not root.is_symlink()
    listed = {name: digest for digest, name in
              (line.split(maxsplit=1) for line in (base / 'SHA256SUMS').read_text().splitlines())}
    assert {path.relative_to(base).as_posix() for path in base.rglob('*') if path.is_file()} == (
        set(listed) | {'SHA256SUMS'})
    assert not any(path.is_symlink() for path in base.rglob('*'))
    assert all(sha(base / name) == digest for name, digest in listed.items())
    for name in (args.gpu_container, args.cpu_container):
        assert name.startswith('flashnext-native19-')
        assert all(c.isalnum() or c in '-_' for c in name)
    root.mkdir()
    shutil.copytree(base / 'overlay', root / 'overlay')
    shutil.copytree(base / 'proof', root / 'proof')
    for path in root.rglob('*'):
        path.chmod(0o700 if path.is_dir() else 0o600)
    sys.path.insert(0, str(repo / 'patches/s5_engine'))
    import make_s5_overlap
    import scheduler_hooks
    for source in (repo / 'patches/s5_engine').glob('*.py'):
        target = root / 'overlay/e3' / source.name
        if target.is_file():
            target.write_text(make_s5_overlap.package(source.read_text()))
    core = root / 'overlay/core/scheduler.py'
    old, new = scheduler_hooks._SCHEDULER_FORWARD[1]
    core.write_text(scheduler_hooks._replace(core.read_text(), old, new))
    for name in ('test_e3_native_schedule_profile', 'test_e3_native_submission_credit',
                 'test_e3_r4_worker_bridge', 'test_e3_lane_dispatch',
                 'test_e3_native_pp_stop_order', 'test_e3_admission_native'):
        target = root / 'proof/harness' / (name + '.py')
        shutil.copyfile(repo / 'harness' / (name + '.py'), target)
    for name in ('e3_native19_vm_suite', 'e3_native19_pp_event_probe',
                 'e3_native19_indexer_ties_probe', 'e3_native_selection_skew_probe',
                 'e3_native_sparse_waves_device_probe', 'native18_scheduler_fixture',
                 'e3_native_cow_lease_probe'):
        shutil.copyfile(repo / 'harness' / (name + '.py'), root / 'proof' / (name + '.py'))
    for path in (root / 'proof').rglob('*.py'):
        source = repo / 'harness' / path.name
        if source.is_file() and source.read_bytes() != path.read_bytes():
            raise ValueError('proof source differs from committed harness: ' +
                             str(path.relative_to(root)))
    shutil.copyfile(repo / 'harness/native19_vm_fixture_leaves.json',
                    root / 'proof/native19_vm_fixture_leaves.json')
    graph_path = root / 'proof/import-graph.json'
    graph = json.loads(graph_path.read_text())
    for row in graph['modules']:
        path = root / 'overlay' / row['source']
        if path.is_file():
            row['sha256'] = sha(path)
    graph['proof_scope'] = ('native19 one-card PP-event/TP-skew/indexer-tie proof; '
                            'inherited import edges, current mounted module bytes re-pinned')
    graph_path.write_text(json.dumps(graph, indent=2) + '\n')
    cpu = (base / 'run_cpu.py').read_text()
    assert "NAME='flashnext-native18-selection-skew-b-hub-cpu'" in cpu
    cpu = cpu.replace("NAME='flashnext-native18-selection-skew-b-hub-cpu'",
                      f'NAME={args.cpu_container!r}')
    cpu = cpu.replace('/proof/e3_native_selection_skew_probe.py',
                      '/proof/e3_native19_vm_suite.py')
    assert '/proof/e3_native19_vm_suite.py' in cpu
    (root / 'run_cpu.py').write_text(cpu)
    gpu = (base / 'run_gpu.py').read_text()
    assert "NAME='flashnext-native18-selection-skew-b'" in gpu
    gpu = gpu.replace("NAME='flashnext-native18-selection-skew-b'",
                      f'NAME={args.gpu_container!r}')
    gpu = gpu.replace('/proof/e3_native_selection_skew_probe.py',
                      '/proof/e3_native19_vm_suite.py')
    gpu = gpu.replace("schema='native18-selection-skew-owner/1'",
                      "schema='native19-vm-owner/1'")
    assert '/proof/e3_native19_vm_suite.py' in gpu
    (root / 'run_gpu.py').write_text(gpu)
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
