#!/usr/bin/env python3
"""Write tools/alpha_next_c1b_parent.json: alpha-next-c1's delivered build (build-b), for alpha-next-c1b's tools to pin (CPU, no box action).

alpha-next-c1b is alpha-next-c1 with harness-only changes: the engine bytes do not move. Its parent is alpha-next-c1's gate-chain OUT
directory (tmp/alpha-next-c1/build-b), delivered (results/hub/alpha-next-c1/delivery.json, verdict PASS, committed fbb9123e). The pin
requires, from alpha-next-c1's own records:
  * build.json (native43-build/1, name q18a-alpha-next-c1) and the bundle, runner and spec it names (the spec file in this tree,
    tools/native43_spec.alpha-next-c1.json, by sha: the runner swaps of c1b's spec are taken from its rows);
  * alpha-next-c1-freeze.json: verdict PASS, naming this bundle;
  * the engine self-run it delivered (engine-selfrun, or engine-selfrun-2 after a declared flaky rerun): verdict PASS; its
    result.json is pinned by sha (c1b's engine comparison reads it: no engine case may differ);
  * engine-comparison.json and staging-check.json: verdict PASS;
  * the chain's steps file: every required step rc=0 (tools/alpha_next_c1_parent_pin.steps_ok, unchanged, plus
    rebind-default-off) and the line 'stopped-before-delivery: VM run pending';
  * the delivery record: verdict PASS, same manifest, runner sha and review sha as build.json;
  * the two VM probe records the delivery carries (results/vm<vm-id>/alpha-next-c1-b-plain|stage/probe.json), by sha: c1b's VM
    carry-over gate (tools/alpha_next_c1b_vm_carry.py) reads their mounted_origins;
  * the READY commit (first commit id in the READY note's first lines) and build.source are ancestors of HEAD.

The committed file is never a placeholder for long: the parent is a delivered build, so the pin is written once, reviewed and
committed (a pinned file is never overwritten; pin a new parent by a new commit after review). load() refuses a placeholder (rc 3,
'parent not pinned') and a development pin outside ALPHA_NEXT_C1B_PARENT, as tools/alpha_next_c1_parent_pin.load does.

    python3 tools/alpha_next_c1b_parent_pin.py --out C1_OUT --steps GATES_STEPS --ready READY_NOTE --delivery DELIVERY [--check]
    python3 tools/alpha_next_c1b_parent_pin.py --pinned        # exit 0 when pinned, 3 'parent not pinned' otherwise
    python3 tools/alpha_next_c1b_parent_pin.py --out C1_OUT --dev   # development pins (no chain records), printed only
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import alpha_next_c1_parent_pin as c1_pin  # noqa: E402  (steps_ok, ready_commit: unchanged)

TARGET = ROOT / 'tools/alpha_next_c1b_parent.json'
SCHEMA = 'alpha-next-c1b-parent/1'
PLACEHOLDER = 'NOT-PINNED'
NOT_PINNED_RC = 3
DEV_ENV = 'ALPHA_NEXT_C1B_PARENT'
PARENT_NAME = 'q18a-alpha-next-c1'
SPEC = ROOT / 'tools/native43_spec.alpha-next-c1.json'
VM_PROBES = ('results/vm<vm-id>/alpha-next-c1-b-plain/probe.json', 'results/vm<vm-id>/alpha-next-c1-b-stage/probe.json')
EXTRA_STEPS = ('parent-pin', 'rebind-default-off')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True)


def ancestor(commit):
    return git('merge-base', '--is-ancestor', commit, 'HEAD').returncode == 0


def load(path=None):
    """The pinned parent; exit 3 'parent not pinned' on a placeholder or a misplaced development pin."""
    dev = os.environ.get(DEV_ENV)
    path = Path(path) if path is not None else Path(dev) if dev else TARGET
    pins = json.loads(path.read_text())
    if pins.get('status', '').startswith(PLACEHOLDER) or pins.get('schema') != SCHEMA:
        print('parent not pinned: %s is the placeholder (%s)' % (path.name, pins.get('status')), file=sys.stderr)
        raise SystemExit(NOT_PINNED_RC)
    if bool(pins.get('dev')) != (path != TARGET and bool(dev)):
        print('parent not pinned: a development pin is only read through %s, never committed' % DEV_ENV, file=sys.stderr)
        raise SystemExit(NOT_PINNED_RC)
    return pins


def steps_ok(text):
    missing = c1_pin.steps_ok(text)
    rows = dict(re.findall(r'^\d\d:\d\d:\d\dZ (\S+) rc=(\d+)$', text, re.M))
    return missing + [s for s in EXTRA_STEPS if rows.get(s) != '0']


def pins(out, steps=None, ready=None, delivery=None, dev=False):
    out = Path(out).resolve()
    build = json.loads((out / 'build.json').read_text())
    assert build['schema'] == 'native43-build/1' and build['name'] == PARENT_NAME, build.get('name')
    assert sha(SPEC) == build['spec_sha256'] and Path(build['spec']).name == SPEC.name, 'c1 spec in this tree is not the build\'s'
    assert sha(build['runner']) == build['runner_sha256'], 'c1 runner changed'
    assert sha(Path(build['bundle']) / 'SHA256SUMS') == build['manifest'], 'c1 bundle changed'
    assert sha(Path(build['overlay']) / 'SHA256SUMS') == build['overlay_manifest'], 'c1 overlay changed'
    freeze = json.loads((out / 'alpha-next-c1-freeze.json').read_text())
    assert freeze['verdict'] == 'PASS' and freeze['candidate_manifest'] == build['manifest'], 'c1 freeze is not this build\'s'
    assert ancestor(build['source']), 'c1 build source %s is not merged here' % build['source'][:12]
    record = dict(schema=SCHEMA, build=str(out / 'build.json'), build_sha256=sha(out / 'build.json'), manifest=build['manifest'],
                  bundle=build['bundle'], overlay_manifest=build['overlay_manifest'], runner=build['runner'],
                  runner_sha256=build['runner_sha256'], review_sha256=build['review_sha256'], source=build['source'],
                  source_commit=build['source_commit'], expected_selfrun_tests=build['expected_selfrun_tests'],
                  freeze=str(out / 'alpha-next-c1-freeze.json'), freeze_sha256=sha(out / 'alpha-next-c1-freeze.json'),
                  spec=str(SPEC.relative_to(ROOT)), spec_sha256=build['spec_sha256'])
    if dev:
        record.update(dev=True, status='DEV: development parent; no chain, engine, delivery or VM records pinned')
        return record
    text = Path(steps).read_text()
    missing = steps_ok(text)
    assert not missing, ('alpha-next-c1 chain not complete through the staging check', missing)
    engine = out / ('engine-selfrun-2' if re.search(r' engine-selfrun-2 rc=0$', text, re.M) else 'engine-selfrun')
    result = json.loads((engine / 'result.json').read_text())
    assert result['verdict'] == 'PASS' and result['manifest'] == build['manifest'], 'c1 engine self-run not PASS on this bundle'
    for name in ('engine-comparison.json', 'staging-check.json'):
        assert json.loads((out / name).read_text())['verdict'] == 'PASS', name + ' not PASS'
    commit = c1_pin.ready_commit(ready)
    assert ancestor(commit), 'alpha-next-c1 READY commit %s is not merged here' % commit[:12]
    d = json.loads(Path(delivery).read_text())
    assert d['verdict'] == 'PASS' and d['name'] == PARENT_NAME, 'c1 delivery not PASS'
    assert (d['manifest'], d['runner_sha256'], d['review_sha256']) == \
        (build['manifest'], build['runner_sha256'], build['review_sha256']), 'c1 delivery names another build'
    probes = {}
    for rel in VM_PROBES:
        probe = ROOT / rel
        assert git('ls-files', '--error-unmatch', rel).returncode == 0, 'VM probe not committed: ' + rel
        probes[rel] = sha(probe)
    record.update(engine_selfrun=str(engine), engine_selfrun_sha256=sha(engine / 'result.json'),
                  engine_cases=len(result['cases']), engine_tests=result['tests'],
                  comparison_sha256=sha(out / 'engine-comparison.json'), staging_check_sha256=sha(out / 'staging-check.json'),
                  steps=str(Path(steps).resolve()), steps_sha256=sha(steps), ready=str(Path(ready).resolve()),
                  ready_sha256=sha(ready), ready_commit=commit, delivery=str(Path(delivery).resolve().relative_to(ROOT)),
                  delivery_sha256=sha(delivery), vm_probes=probes)
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', type=Path, help="alpha-next-c1's gate-chain OUT directory (build-b)")
    p.add_argument('--steps', type=Path, help="alpha-next-c1's gates-TAG.steps")
    p.add_argument('--ready', type=Path, help="alpha-next-c1's READY-pre-vm note")
    p.add_argument('--delivery', type=Path, help='results/hub/alpha-next-c1/delivery.json')
    p.add_argument('--check', action='store_true')
    p.add_argument('--dev', action='store_true', help='print the pins of a development parent; writes nothing')
    p.add_argument('--pinned', action='store_true', help='exit 0 when pinned, 3 otherwise')
    a = p.parse_args(argv)
    if a.pinned:
        pinned = load()
        print('alpha-next-c1b parent pinned: %s manifest %s' % (pinned['source'][:12], pinned['manifest'][:12]))
        return 0
    if a.dev:
        print(json.dumps(pins(a.out, dev=True), indent=1, sort_keys=True))
        return 0
    assert a.out and a.steps and a.ready and a.delivery, '--out, --steps, --ready and --delivery are required'
    text = json.dumps(pins(a.out, a.steps, a.ready, a.delivery), indent=1, sort_keys=True) + '\n'
    if a.check:
        assert TARGET.read_text() == text, 'tools/alpha_next_c1b_parent.json differs from alpha-next-c1\'s build'
        print('alpha-next-c1b parent verified')
        return 0
    if TARGET.exists():
        current = json.loads(TARGET.read_text())
        assert current.get('status', '').startswith(PLACEHOLDER), 'the parent is pinned already; never overwritten'
    TARGET.write_text(text)
    print('alpha-next-c1b parent written ' + sha(TARGET))
    return 0


if __name__ == '__main__':
    sys.exit(main())
