#!/usr/bin/env python3
"""Native43 engine self-run: the trunk engine self-run plus the CT-vs-container clock split.

1. tools/native39_trunk_selfrun.py run() unchanged (pinned image, CPU only, network none):
   py_compile of the spec's declared files, the native39 R/W/P probe, the native37 T1/T2
   probe and the spec's suites against the bundle's own overlay.
2. tools/native43_clock_split.py: the bundle's client clock checks across a physical
   CT-vs-container identity split (LXC-style boot id, separate time namespaces, zero
   timens offsets), with a shared-view positive control; outcomes as the spec declares.
3. The spec's extra harness probes (engine_selfrun.probes, e.g. native43_scheduler_tick_probe.py) in the
   same container after probe39/probe37; each must report PASS, exit 0 and the spec's required fields.
The record keeps the engine self-run fields the owner runner gates on and adds
clock_split and one field per extra probe; schema and label come from the spec.  No GPU, no box action.
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
    probes = selfrun.get('probes', [])
    original = base.inner_script

    def inner_script(label, suites, compiled):
        # Extra harness probes declared by the spec run in the same container, after probe39/probe37.
        return original(label, suites, compiled) + ''.join(
            'python3 -B %s --out /out/%s > /out/%s.log 2>&1; echo "%s rc=$?" >> /out/%s.log\n'
            % (row['script'], row['out'], row['out'], row['field'], row['out']) for row in probes)
    base.inner_script = inner_script
    base.run(tree, bundle / 'native30_overlay', out / 'engine', name, spec['candidate'], True, bundle,
             tuple(selfrun['compiled']))
    engine = json.loads((out / 'engine/result.json').read_text())
    extra = {}
    for row in probes:
        path, log = out / 'engine' / row['out'], out / 'engine' / (row['out'] + '.log')
        view = json.loads(path.read_text()) if path.is_file() else {}
        text = log.read_text() if log.is_file() else ''
        extra[row['field']] = dict({k: view.get(k) for k in row['keep']}, script=row['script'],
                                   verdict=view.get('verdict'), rc_line=text.rstrip().splitlines()[-1] if text.strip() else None,
                                   sha256=sha(path) if path.is_file() else None,
                                   required=row.get('require', {}),
                                   required_ok=all(view.get(k) == v for k, v in row.get('require', {}).items()))
    probes_ok = all(v['verdict'] == 'PASS' and v['rc_line'] == '%s rc=0' % k and v['required_ok'] for k, v in extra.items())
    split_rc = clock_split.run(bundle, tree, out / 'clock-split', name + '-clock', split['side'], split['expect'],
                               split['modules'])
    clock = json.loads((out / 'clock-split/result.json').read_text())
    engine_ok = (engine['container_rc'] == 0 and engine['probe39']['verdict'] == 'PASS'
                 and engine['probe37']['verdict'] == 'PASS' and engine['py_compile']['ok'] is True
                 and engine['suites']['ok'] is True and engine['suites']['skipped'] == 0 and probes_ok)
    record = dict(engine, **extra, schema=selfrun['schema'], spec=str(spec_path), spec_sha256=sha(spec_path),
                  engine_record_sha256=sha(out / 'engine/result.json'),
                  clock_split=dict(verdict=clock['verdict'], reasons=clock['reasons'], outcomes=clock['outcomes'],
                                   expected=clock['expected'], control_outcomes=clock.get('control_outcomes'),
                                   split=clock['split'], method=clock.get('method'), rc=split_rc, record_sha256=sha(out / 'clock-split/result.json'), side=clock['side'],
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
