#!/usr/bin/env python3
"""Native43 engine self-run: the trunk engine self-run plus the CT-vs-container clock split.

1. tools/native39_trunk_selfrun.py run() unchanged (pinned image, CPU only, network none):
   py_compile of the spec's declared files, the native39 R/W/P probe, the native37 T1/T2
   probe and the spec's suites against the bundle's own overlay.
2. tools/native43_clock_split.py: the bundle's client clock checks across a physical
   CT-vs-container identity split (LXC-style boot id, separate time namespaces, zero
   timens offsets), with a shared-view positive control; outcomes as the spec declares.
The record keeps the engine self-run fields the owner runner gates on and adds
clock_split; schema and label come from the spec.  No GPU, no box action.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import native39_trunk_selfrun as base  # noqa: E402
import native43_clock_split as clock_split  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(spec_path, tree, bundle, out, name):
    spec = json.loads(Path(spec_path).read_text())
    selfrun, split = spec['engine_selfrun'], spec['clock_split']
    tree, bundle, out = Path(tree).resolve(), Path(bundle).resolve(), Path(out).resolve()
    out.mkdir()
    base.SUITES = tuple(selfrun['suites'])
    base.run(tree, bundle / 'native30_overlay', out / 'engine', name, spec['candidate'], True, bundle,
             tuple(selfrun['compiled']))
    engine = json.loads((out / 'engine/result.json').read_text())
    split_rc = clock_split.run(bundle, tree, out / 'clock-split', name + '-clock', split['side'], split['expect'],
                               split['modules'])
    clock = json.loads((out / 'clock-split/result.json').read_text())
    engine_ok = (engine['container_rc'] == 0 and engine['probe39']['verdict'] == 'PASS'
                 and engine['probe37']['verdict'] == 'PASS' and engine['py_compile']['ok'] is True
                 and engine['suites']['ok'] is True and engine['suites']['skipped'] == 0)
    record = dict(engine, schema=selfrun['schema'], spec=str(spec_path), spec_sha256=sha(spec_path),
                  engine_record_sha256=sha(out / 'engine/result.json'),
                  clock_split=dict(verdict=clock['verdict'], reasons=clock['reasons'], outcomes=clock['outcomes'],
                                   expected=clock['expected'], control_outcomes=clock.get('control_outcomes'),
                                   split=clock['split'], rc=split_rc, record_sha256=sha(out / 'clock-split/result.json'), side=clock['side'],
                                   gpu_device_requests=clock.get('gpu_device_requests')),
                  verdict='PASS' if engine_ok and clock['verdict'] == 'PASS' else 'FAIL',
                  classification=None if engine_ok and clock['verdict'] == 'PASS' else 'HARNESS')
    (out / 'result.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(verdict=record['verdict'], engine_ok=engine_ok, clock_split=record['clock_split']['verdict'],
                          suites=record['suites']['ran'], result_sha256=sha(out / 'result.json')), sort_keys=True))
    return 0 if record['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--tree', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--name', required=True)
    a = p.parse_args()
    raise SystemExit(run(a.spec, a.tree, a.bundle, a.out, a.name))
