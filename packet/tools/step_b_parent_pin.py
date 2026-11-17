#!/usr/bin/env python3
"""Write tools/step_b_parent.json: Step A's frozen build, its engine self-run record and their shas, for Step B's tools to pin.

Reads Step A's OUT directory (the one tools/step_a_gates.sh wrote) and its delivery JSON, and refuses unless that delivery is PASS and
names the same build. Nothing else is read or written. --check verifies the committed file against the same inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'tools/step_b_parent.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pins(out, delivery):
    out, delivery = Path(out).resolve(), Path(delivery).resolve()
    build = json.loads((out / 'build.json').read_text())
    record = json.loads(delivery.read_text())
    assert build['schema'] == 'native43-build/1' and build['name'] == 'q18a-step-a', build['name']
    assert record['schema'] == 'step-a-delivery/1' and record['verdict'] == 'PASS', record.get('verdict')
    assert record['manifest'] == build['manifest'] and record['runner_sha256'] == build['runner_sha256']
    engine = out / 'engine-selfrun'
    assert json.loads((engine / 'result.json').read_text())['verdict'] == 'PASS'
    assert record['engine_selfrun']['sha256'] == sha(engine / 'result.json')
    return dict(schema='step-b-parent/1', build=str(out / 'build.json'), build_sha256=sha(out / 'build.json'),
                manifest=build['manifest'], runner_sha256=build['runner_sha256'], source=build['source'],
                engine_selfrun=str(engine), engine_selfrun_sha256=sha(engine / 'result.json'),
                delivery=str(delivery), delivery_sha256=sha(delivery))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', type=Path, required=True, help="Step A's gate-chain OUT directory")
    p.add_argument('--delivery', type=Path, required=True, help="Step A's delivery JSON")
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    text = json.dumps(pins(a.out, a.delivery), indent=1, sort_keys=True) + '\n'
    if a.check:
        assert TARGET.read_text() == text, 'tools/step_b_parent.json differs from Step A\'s build'
        print('step-b parent verified')
    else:
        with TARGET.open('x') as handle:
            handle.write(text)
        print('step-b parent written ' + sha(TARGET))
