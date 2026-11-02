#!/usr/bin/env python3
"""Write tools/alpha_next_c1_parent.json: alpha-next's delivered pre-VM build, for alpha-next-c1's tools to pin (CPU, no box action).

The parent is alpha-next's gate-chain OUT directory that its READY-pre-vm note names (tmp/alpha-next/READY-pre-vm.md, the READY
commit at its top). Unlike tools/alpha_next_parent_pin.py (Step B had a PASS delivery), alpha-next is pinned before its VM run, so
there is no delivery record; instead the pin requires, from alpha-next's own chain:
  * build.json (native43-build/1, name q18a-alpha-next) and the bundle, runner and overlay it names;
  * the engine self-run it delivered (engine-selfrun or, after a declared flaky rerun, engine-selfrun-2): verdict PASS;
  * alpha-next-freeze.json: verdict PASS and source.head == build.source (the engine commit the freeze proved is the bundle's);
  * engine-comparison.json and staging-check.json: verdict PASS;
  * the steps file: every step rc=0 through staging-check and the line 'stopped-before-delivery: VM run pending';
  * the READY commit (first 40-hex or 7..40-hex commit id in the READY note's first lines) is an ancestor of HEAD, and so is
    build.source: the alpha-next-c1 branch must have merged alpha-next's READY commit before the pin is written.

Until it is written the committed file is a placeholder (status NOT-PINNED). Every alpha-next-c1 tool reads the parent through
load(), which exits 3 with 'parent not pinned' on the placeholder: the gate chain refuses (rc 3) at its first step.
Writing replaces only the placeholder; a pinned file is never overwritten (pin a new parent by a new commit after review).
--check verifies the committed file against the same inputs.  --dev writes nothing and prints the pins for a development parent
(no engine self-run, comparison, staging check or steps needed; marked dev) so the tests can exercise the tools.

    python3 tools/alpha_next_c1_parent_pin.py --out ALPHA_NEXT_OUT --steps GATES_STEPS --ready READY_NOTE [--check]
    python3 tools/alpha_next_c1_parent_pin.py --pinned        # exit 0 when pinned, 3 'parent not pinned' otherwise
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
TARGET = ROOT / 'tools/alpha_next_c1_parent.json'
SCHEMA = 'alpha-next-c1-parent/1'
PLACEHOLDER = 'NOT-PINNED'
NOT_PINNED_RC = 3
DEV_ENV = 'ALPHA_NEXT_C1_PARENT'
STEPS_REQUIRED = ('composition', 'review-pins', 'prepare-review', 'review-commit', 'build', 'bundle-selfrun', 'queue-pins-write',
                  'queue-pins-check', 'engine-comparison', 'engine-negative', 'overlay-tamper', 'clock-scan', 'compile-proof',
                  'precheck-conf', 'precheck', 'staging-check')
STOPPED = 'stopped-before-delivery: VM run pending'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True)


def ancestor(commit):
    return git('merge-base', '--is-ancestor', commit, 'HEAD').returncode == 0


def load(path=None):
    """The pinned parent; exit 3 with 'parent not pinned' while the file is the placeholder.

    With ALPHA_NEXT_C1_PARENT set (tests and dry runs only; the gate chain unsets it) a development pin file outside the tree is
    read instead, and only a dev pin is accepted there; a dev pin in the committed file is refused."""
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
    """Every required step rc=0 (a flaky rerun's '-2' attempt counts for strict, focused and the engine self-run), and the stop line."""
    rows = dict(re.findall(r'^\d\d:\d\d:\d\dZ (\S+) rc=(\d+)$', text, re.M))
    missing = [s for s in STEPS_REQUIRED if rows.get(s) != '0']
    for name in ('strict', 'focused', 'engine-selfrun'):
        if rows.get(name) != '0' and rows.get(name + '-2') != '0':
            missing.append(name)
    if STOPPED not in text.splitlines():
        missing.append('stop-line')
    return missing


def ready_commit(note):
    head = '\n'.join(Path(note).read_text().splitlines()[:8])
    found = re.findall(r'\b([0-9a-f]{40}|[0-9a-f]{7,12})\b', head)
    assert found, 'no READY commit in the first lines of ' + str(note)
    full = git('rev-parse', '--verify', found[0] + '^{commit}')
    assert full.returncode == 0, 'READY commit unknown here: ' + found[0]
    return full.stdout.strip()


def pins(out, steps=None, ready=None, dev=False):
    out = Path(out).resolve()
    build = json.loads((out / 'build.json').read_text())
    assert build['schema'] == 'native43-build/1' and build['name'] == 'q18a-alpha-next', build.get('name')
    freeze = json.loads((out / 'alpha-next-freeze.json').read_text())
    assert freeze['verdict'] == 'PASS' and freeze['source']['head'] == build['source'], 'alpha-next freeze is not this build\'s'
    assert freeze['candidate_manifest'] == build['manifest'], 'alpha-next freeze names another bundle'
    assert ancestor(build['source']), 'alpha-next build source %s is not merged here' % build['source'][:12]
    record = dict(schema=SCHEMA, build=str(out / 'build.json'), build_sha256=sha(out / 'build.json'), manifest=build['manifest'],
                  overlay_manifest=build['overlay_manifest'], runner_sha256=build['runner_sha256'], source=build['source'],
                  source_commit=build['source_commit'], expected_selfrun_tests=build['expected_selfrun_tests'],
                  freeze=str(out / 'alpha-next-freeze.json'), freeze_sha256=sha(out / 'alpha-next-freeze.json'),
                  freeze_head=freeze['source']['head'], spec=build['spec'], spec_sha256=build['spec_sha256'])
    if dev:
        record.update(dev=True, status='DEV: development parent; no engine self-run, comparison, staging check or steps pinned')
        return record
    text = Path(steps).read_text()
    missing = steps_ok(text)
    assert not missing, ('alpha-next chain not complete through the staging check', missing)
    engine = out / ('engine-selfrun-2' if re.search(r' engine-selfrun-2 rc=0$', text, re.M) else 'engine-selfrun')
    assert json.loads((engine / 'result.json').read_text())['verdict'] == 'PASS', 'alpha-next engine self-run not PASS'
    for name in ('engine-comparison.json', 'staging-check.json'):
        assert json.loads((out / name).read_text())['verdict'] == 'PASS', name + ' not PASS'
    commit = ready_commit(ready)
    assert ancestor(commit), 'alpha-next READY commit %s is not merged here' % commit[:12]
    record.update(engine_selfrun=str(engine), engine_selfrun_sha256=sha(engine / 'result.json'),
                  comparison_sha256=sha(out / 'engine-comparison.json'), staging_check_sha256=sha(out / 'staging-check.json'),
                  steps=str(Path(steps).resolve()), steps_sha256=sha(steps), ready=str(Path(ready).resolve()),
                  ready_sha256=sha(ready), ready_commit=commit)
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', type=Path, help="alpha-next's gate-chain OUT directory")
    p.add_argument('--steps', type=Path, help="alpha-next's gates-TAG.steps")
    p.add_argument('--ready', type=Path, help="alpha-next's READY-pre-vm note")
    p.add_argument('--check', action='store_true')
    p.add_argument('--dev', action='store_true', help='print the pins of a development parent; writes nothing')
    p.add_argument('--pinned', action='store_true', help='exit 0 when pinned, 3 otherwise')
    a = p.parse_args(argv)
    if a.pinned:
        pinned = load()
        print('alpha-next-c1 parent pinned: %s manifest %s' % (pinned['source'][:12], pinned['manifest'][:12]))
        return 0
    if a.dev:
        print(json.dumps(pins(a.out, dev=True), indent=1, sort_keys=True))
        return 0
    assert a.out and a.steps and a.ready, '--out, --steps and --ready are required'
    text = json.dumps(pins(a.out, a.steps, a.ready), indent=1, sort_keys=True) + '\n'
    if a.check:
        assert TARGET.read_text() == text, 'tools/alpha_next_c1_parent.json differs from alpha-next\'s build'
        print('alpha-next-c1 parent verified')
        return 0
    current = json.loads(TARGET.read_text())
    assert current.get('status', '').startswith(PLACEHOLDER), 'the parent is pinned already; never overwritten'
    TARGET.write_text(text)
    print('alpha-next-c1 parent written ' + sha(TARGET))
    return 0


if __name__ == '__main__':
    sys.exit(main())
