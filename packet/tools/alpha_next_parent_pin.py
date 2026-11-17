#!/usr/bin/env python3
"""Write tools/alpha_next_parent.json: Step B's frozen build (chain e), its engine self-run record and their shas, for alpha-next's tools to pin.

Reads Step B's OUT directory (the one tools/step_b_gates.sh wrote, chain e) and its delivery JSON, and refuses unless that delivery is PASS and
names the same build. Nothing else is read or written. --check verifies the committed file against the same inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'tools/alpha_next_parent.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pins(out, delivery):
    out, delivery = Path(out).resolve(), Path(delivery).resolve()
    build = json.loads((out / 'build.json').read_text())
    record = json.loads(delivery.read_text())
    assert build['schema'] == 'native43-build/1' and build['name'] == 'q18a-step-b', build['name']
    assert record['schema'] == 'step-b-delivery/1' and record['verdict'] == 'PASS', record.get('verdict')
    assert record['manifest'] == build['manifest'] and record['runner_sha256'] == build['runner_sha256']
    engine = out / 'engine-selfrun'
    assert json.loads((engine / 'result.json').read_text())['verdict'] == 'PASS'
    assert record['engine_selfrun']['sha256'] == sha(engine / 'result.json')
    return dict(schema='alpha-next-parent/1', build=str(out / 'build.json'), build_sha256=sha(out / 'build.json'),
                manifest=build['manifest'], runner_sha256=build['runner_sha256'], source=build['source'],
                engine_selfrun=str(engine), engine_selfrun_sha256=sha(engine / 'result.json'),
                delivery=str(delivery), delivery_sha256=sha(delivery))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', type=Path, required=True, help="Step B's gate-chain OUT directory")
    p.add_argument('--delivery', type=Path, required=True, help="Step B's delivery JSON")
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    text = json.dumps(pins(a.out, a.delivery), indent=1, sort_keys=True) + '\n'
    if a.check:
        assert TARGET.read_text() == text, 'tools/alpha_next_parent.json differs from Step B\'s build'
        print('alpha-next parent verified')
    else:
        with TARGET.open('x') as handle:
            handle.write(text)
        print('alpha-next parent written ' + sha(TARGET))
