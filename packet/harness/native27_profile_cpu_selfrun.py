#!/usr/bin/env python3
"""Run the native27 VM input's exact CPU-executable proof in the pinned image."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import run_gpu


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.input.resolve(), args.out.absolute()
    assert root.is_dir() and not out.exists() and not out.is_symlink()
    rows = dict(line.split(maxsplit=1)[::-1]
                for line in (root / 'SHA256SUMS').read_text().splitlines())
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    assert actual == set(rows) | {'SHA256SUMS'}
    assert not any(p.is_symlink() for p in root.rglob('*'))
    assert all(sha(root / name) == digest for name, digest in rows.items())
    graph = json.loads((root / 'proof/import-graph.json').read_text())
    out.mkdir(mode=0o700)
    command = run_gpu.docker_command(root, out, graph)
    command[command.index('--name') + 1] += '-hub-cpu'
    command[command.index('--runtime') + 1] = 'runc'
    command[command.index('NVIDIA_VISIBLE_DEVICES=0')] = 'NVIDIA_VISIBLE_DEVICES=none'
    command[command.index('CUDA_VISIBLE_DEVICES=0')] = 'CUDA_VISIBLE_DEVICES='
    command[-1] = 'cpu-shape'
    with (out / 'container.log').open('xb') as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                 timeout=180)
    probe = json.loads((out / 'probe.json').read_text()) if (out / 'probe.json').is_file() else None
    passed = (process.returncode == 0 and probe is not None
              and probe.get('verdict') == 'PASS'
              and probe['sections']['receipt_path']['verdict'] == 'PASS'
              and probe['sections']['receipt_path']['journal_refusal_evicted']
              and probe['sections']['receipt_path']['phase_receipt_exact']
              and probe['sections']['receipt_path']['queue_grade_exact']
              and not probe.get('gpu_execution'))
    record = {'schema': 'native27-profile-hub-selfrun/1',
              'verdict': 'PASS' if passed else 'FAIL', 'rc': process.returncode,
              'mode': 'cpu-shape', 'manifest_sha256': sha(root / 'SHA256SUMS'),
              'probe_sha256': sha(out / 'probe.json') if probe else None,
              'container': command[command.index('--name') + 1],
              'image': run_gpu.IMAGE, 'gpu_device_requests': 0,
              'command': command, 'box_action': False}
    (out / 'result.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print(json.dumps(record, sort_keys=True))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
