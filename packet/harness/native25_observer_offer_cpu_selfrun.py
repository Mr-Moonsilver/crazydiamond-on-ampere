#!/usr/bin/env python3
"""Run the exact native25 observer-path input's CPU-executable shape."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

import run_gpu


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_gpu_stream_scope(probe_path):
    tree = ast.parse(probe_path.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == 'gpu_capture_and_bind')
    stream_scopes = [node for node in ast.walk(function)
                     if isinstance(node, ast.With)
                     and any(ast.unparse(item.context_expr) == 'torch.cuda.stream(stream)'
                             for item in node.items)]
    checked = [node for scope in stream_scopes for node in ast.walk(scope)
               if isinstance(node, ast.Call)
               and ast.unparse(node.func) == 'worker_hooks.checked_capture_stream']
    assert len(stream_scopes) >= 1 and len(checked) == 1, (
        'capture stream check must execute under the decode stream',
        len(stream_scopes), len(checked))
    return {'verdict':'PASS', 'checked_capture_stream_calls_in_scope':len(checked),
            'stream_scope_line':stream_scopes[0].lineno}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    root, out = args.input.resolve(), args.out.absolute()
    assert root.is_dir() and not out.exists() and not out.is_symlink()
    entries = dict(line.split(maxsplit=1)[::-1]
                   for line in (root / 'SHA256SUMS').read_text().splitlines())
    files = {path.relative_to(root).as_posix() for path in root.rglob('*')
             if path.is_file()}
    assert files == set(entries) | {'SHA256SUMS'}
    assert not any(path.is_symlink() for path in root.rglob('*'))
    assert all(sha(root / name) == digest for name, digest in entries.items())
    graph = json.loads((root / 'proof/import-graph.json').read_text())
    stream_audit = audit_gpu_stream_scope(
        root / 'proof/e3_native25_observer_offer_probe.py')
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
    probe_path = out / 'probe.json'
    probe = json.loads(probe_path.read_text()) if probe_path.is_file() else None
    verdict = ('PASS' if process.returncode == 0 and probe is not None
               and probe.get('verdict') == 'PASS'
               and probe['sections'].get('cpu_shape', {}).get('verdict') == 'PASS'
               and probe['sections'].get('census_cpu_shape', {}).get('verdict') == 'PASS'
               and not probe.get('gpu_execution') else 'FAIL')
    result = {'schema':'native25-hub-cpu-selfrun/1', 'verdict':verdict,
              'rc':process.returncode, 'manifest_sha256':sha(root / 'SHA256SUMS'),
              'probe_sha256':sha(probe_path) if probe else None,
              'container':command[command.index('--name') + 1],
              'image':run_gpu.IMAGE, 'gpu_device_requests':0,
              'mode':'cpu-shape', 'gpu_stream_scope_audit':stream_audit}
    (out / 'result.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, sort_keys=True))
    return 0 if verdict == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
