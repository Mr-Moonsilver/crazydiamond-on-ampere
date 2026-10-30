#!/usr/bin/env python3
"""One immutable VM<vm-id> run for PP events, TP skew and real indexer ties."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=('cpu-shape', 'gpu-all'), required=True)
    args = parser.parse_args()
    result = dict(schema='native19-vm-suite/1', verdict='INCONCLUSIVE', rc=2,
                  failure_class='OPEN', sections={}, gpu_execution=False,
                  box_evidence=False, source_graph_sha256=sha(args.graph))
    work = args.out.parent
    cases = (
        ('pp_event', 'e3_native19_pp_event_probe.py'),
        ('selection_skew', 'e3_native_selection_skew_probe.py'),
        ('indexer_ties', 'e3_native19_indexer_ties_probe.py'),
    )
    try:
        graph = json.loads(args.graph.read_text())
        assert graph['verdict'] == 'PASS'
        for name, filename in cases:
            output = work / (name + '.json')
            log = work / (name + '.log')
            mode = ('gpu-all' if args.mode == 'gpu-all' else 'cpu-shape') if name == 'selection_skew' else (
                'gpu' if args.mode == 'gpu-all' else 'cpu-shape')
            command = [sys.executable, '-B', '/proof/' + filename,
                       '--out', str(output), '--mode', mode]
            if name == 'selection_skew':
                command += ['--graph', str(args.graph)]
            started = time.monotonic()
            with log.open('xb') as stream:
                completed = subprocess.run(command, stdout=stream,
                                           stderr=subprocess.STDOUT, timeout=600)
            assert output.is_file(), name + ':missing-result'
            row = json.loads(output.read_text())
            result['sections'][name] = dict(verdict=row.get('verdict'),
                process_rc=completed.returncode, elapsed_s=time.monotonic()-started,
                receipt_sha256=sha(output), log_sha256=sha(log),
                result=row)
            args.out.write_text(json.dumps(result, indent=2) + '\n')
            assert (completed.returncode == 0 and row.get('verdict') == 'PASS'), (
                name, completed.returncode, row.get('verdict'))
        result.update(verdict='PASS', rc=0, failure_class='NONE',
                      gpu_execution=args.mode == 'gpu-all')
    except BaseException as exc:
        result.update(verdict='FAIL', rc=1, failure_class='OPEN',
                      error_type=type(exc).__name__, error_message=str(exc),
                      stack=traceback.format_exc())
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(result['verdict'], flush=True)
    return result['rc']


if __name__ == '__main__':
    raise SystemExit(main())
