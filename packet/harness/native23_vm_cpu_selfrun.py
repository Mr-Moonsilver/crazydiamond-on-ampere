#!/usr/bin/env python3
"""Run the exact native23 VM input's CPU-shape path in the pinned hub image."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

import run_gpu


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_paused_predicates(probe_path):
    """Refuse scheduler runnable-count predicates before explicit resume."""
    tree = ast.parse(probe_path.read_text())
    method = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == 'gpu_pause_resume')
    resumes = [node.lineno for node in ast.walk(method)
               if isinstance(node, ast.Call)
               and ast.unparse(node.func) == 'core.resume_scheduler']
    assert len(resumes) == 1, 'one explicit resume required'
    resume = resumes[0]
    forbidden = ('core.scheduler.has_requests',
                 'core.scheduler.get_num_unfinished_requests')
    checks = [(node.lineno, ast.unparse(node.func)) for node in ast.walk(method)
              if isinstance(node, ast.Call)
              and ast.unparse(node.func) in forbidden]
    assert all(line > resume for line, _ in checks), (
        'paused runnable-count predicate', resume, checks)
    registered = [node.lineno for node in ast.walk(method)
                  if isinstance(node, ast.Call)
                  and ast.unparse(node.func) == 'core.scheduler.requests.get']
    assert len(registered) >= 2 and all(line < resume for line in registered), (
        'paused request registration not checked', registered)
    return {'verdict': 'PASS', 'resume_line': resume,
            'post_resume_runnable_checks': checks,
            'paused_registration_checks': registered}


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
    paused_audit = audit_paused_predicates(
        root / 'proof/e3_native23_row4_pause_probe.py')
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
              'gpu_device_requests': 0,
              'paused_predicate_audit': paused_audit}
    with (out / 'result.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result, sort_keys=True))
    return 0 if verdict == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
