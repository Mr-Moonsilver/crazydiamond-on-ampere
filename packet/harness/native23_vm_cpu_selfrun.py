#!/usr/bin/env python3
"""Run the exact native23 VM input's CPU-shape path in the pinned hub image."""
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
    root = args.input.resolve()
    out = args.out.absolute()
    assert root.is_dir() and not out.exists() and not out.is_symlink()
    rows = dict(line.split(maxsplit=1)[::-1]
                for line in (root / 'SHA256SUMS').read_text().splitlines())
    files = {item.relative_to(root).as_posix() for item in root.rglob('*')
             if item.is_file()}
    assert files == set(rows) | {'SHA256SUMS'}
    assert not any(item.is_symlink() for item in root.rglob('*'))
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
    probe = json.loads((out / 'probe.json').read_text()) if (out / 'probe.json').exists() else None
    verdict = ('PASS' if process.returncode == 0 and probe is not None
               and probe.get('verdict') == 'PASS' and not probe.get('gpu_execution')
               else 'FAIL')
    result = {'schema': 'native23-hub-cpu-selfrun/1', 'verdict': verdict,
              'rc': process.returncode, 'manifest_sha256': sha(root / 'SHA256SUMS'),
              'probe_sha256': sha(out / 'probe.json') if probe else None,
              'container': command[command.index('--name') + 1],
              'mode': 'cpu-shape', 'image': run_gpu.IMAGE,
              'gpu_device_requests': 0}
    with (out / 'result.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result, sort_keys=True))
    return 0 if verdict == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
