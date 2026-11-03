#!/usr/bin/env python3
"""Write tools/stage1_step_parent.json: alpha-next-c1b's READY build (chain b2, manifest 6595d6692a0d), the parent of q18a-stage1-step.
tools/attn_meta_parent_pin_b.py's rule by name (claude/attn-meta-c1b; never edited there). CPU, no box action.

c1b is pinned before its VM step, so there is no delivery record; the pin requires, from c1b's own chain:
  * build.json (native43-build/1, name q18a-alpha-next-c1b) and its alpha-next-c1b-freeze.json (verdict PASS, dev_parent false, naming this
    bundle);
  * the engine self-run it delivered (engine-selfrun, or engine-selfrun-2 after a declared flaky rerun): verdict PASS, sha-pinned;
  * engine-comparison.json and staging-check.json (or a continuation's staging-check-TAG.json): verdict PASS;
  * the steps file: every step of STEPS_REQUIRED rc=0 (c1b's chain names: composition, rebind-default-off and vm-carry included; strict,
    focused and engine-selfrun may pass on their '-2' rerun) and the line 'stopped-before-delivery: VM run pending'; or, through a
    continuation (--continuation), alpha_next_c1_parent_pin's continuation rule unchanged, applied with this STEPS_REQUIRED;
  * the READY commit is the commit that added results/hub/alpha-next-c1b/READY-c1b.md; the committed note must equal --ready byte for byte;
    it and build.source are ancestors of HEAD.
load() refuses a placeholder (rc 3, 'parent not pinned') and a development pin outside STAGE1_STEP_PARENT. Never overwritten.
    python3 tools/stage1_step_parent_pin.py --out C1B_OUT --steps STEPS --ready READY_NOTE [--continuation CONT] [--check]
    python3 tools/stage1_step_parent_pin.py --pinned
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import alpha_next_c1_parent_pin as c1pin  # noqa: E402  (its sha, ancestor, ready_commit and continuation_ok, unchanged)

ROOT = TOOLS.parent
TARGET = ROOT / 'tools/stage1_step_parent.json'
SCHEMA = 'stage1-step-parent/1'
PLACEHOLDER = 'NOT-PINNED'
NOT_PINNED_RC = 3
DEV_ENV = 'STAGE1_STEP_PARENT'
PARENT_NAME = 'q18a-alpha-next-c1b'
FREEZE = 'alpha-next-c1b-freeze.json'
STEPS_REQUIRED = ('parent-pin', 'composition', 'rebind-default-off', 'review-pins', 'composition-test', 'prepare-review', 'review-commit',
                  'build', 'vm-carry', 'bundle-selfrun', 'queue-pins-write', 'queue-pins-check', 'engine-comparison',
                  'engine-negative', 'overlay-tamper', 'clock-scan', 'compile-proof', 'precheck-conf', 'precheck', 'staging-check')
RERUN_STEPS = ('strict', 'focused', 'engine-selfrun')  # as c1pin.steps_ok: the '-2' attempt counts
STOPPED = c1pin.STOPPED
READY_PATH = 'results/hub/alpha-next-c1b/READY-c1b.md'


def ready_commit(note):
    found = c1pin.git('log', '--diff-filter=A', '-1', '--format=%H', '--', READY_PATH).stdout.strip()
    assert found, 'no commit adds ' + READY_PATH
    committed = c1pin.git('show', found + ':' + READY_PATH).stdout
    assert committed == Path(note).read_text(), 'the READY note differs from the committed one'
    return found


def load(path=None):
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
    rows = dict(re.findall(r'^\d\d:\d\d:\d\dZ (\S+) rc=(\d+)$', text, re.M))
    missing = [s for s in STEPS_REQUIRED if rows.get(s) != '0']
    missing += [s for s in RERUN_STEPS if rows.get(s) != '0' and rows.get(s + '-2') != '0']
    if STOPPED not in text.splitlines():
        missing.append('stop-line')
    return missing


def pins(out, steps=None, ready=None, continuation=None, dev=False):
    out = Path(out).resolve()
    build = json.loads((out / 'build.json').read_text())
    assert build['schema'] == 'native43-build/1' and build['name'] == PARENT_NAME, build.get('name')
    freeze = json.loads((out / FREEZE).read_text())
    assert freeze['schema'] == 'alpha-next-c1b-freeze/1' and freeze['dev_parent'] is False, 'c1b freeze ran on a development parent'
    assert freeze['verdict'] == 'PASS' and freeze['candidate_manifest'] == build['manifest'], 'c1b freeze is not this build\'s'
    assert c1pin.ancestor(build['source']), 'c1b build source %s is not merged here' % build['source'][:12]
    record = dict(schema=SCHEMA, build=str(out / 'build.json'), build_sha256=c1pin.sha(out / 'build.json'),
                  manifest=build['manifest'], overlay_manifest=build['overlay_manifest'], runner_sha256=build['runner_sha256'],
                  source=build['source'], source_commit=build['source_commit'],
                  expected_selfrun_tests=build['expected_selfrun_tests'], freeze=str(out / FREEZE),
                  freeze_sha256=c1pin.sha(out / FREEZE), spec=build['spec'], spec_sha256=build['spec_sha256'])
    if dev:
        record.update(dev=True, status='DEV: development parent; no engine self-run, comparison, staging check or steps pinned')
        return record
    text = Path(steps).read_text()
    saved = c1pin.STEPS_REQUIRED
    if continuation is None:
        missing = steps_ok(text)
        assert not missing, ('c1b chain not complete through the staging check', missing)
    else:
        c1pin.STEPS_REQUIRED = STEPS_REQUIRED  # c1pin.steps_ok adds the three rerun steps itself
        try:
            missing = c1pin.continuation_ok(text, Path(continuation).read_text(), ready_commit(ready))
        finally:
            c1pin.STEPS_REQUIRED = saved
        assert not missing, ('c1b chain and continuation not complete through the staging check', missing)
    engine = out / ('engine-selfrun-2' if re.search(r' engine-selfrun-2 rc=0$', text, re.M) else 'engine-selfrun')
    assert json.loads((engine / 'result.json').read_text())['verdict'] == 'PASS', 'c1b engine self-run not PASS'
    tag = None
    if continuation is not None:
        head = re.match(r'head \S+ continuation (\S+) ', Path(continuation).read_text())
        tag = head.group(1) if head else None
    staging = 'staging-check-%s.json' % tag if tag else 'staging-check.json'
    for name in ('engine-comparison.json', staging):
        assert json.loads((out / name).read_text())['verdict'] == 'PASS', name + ' not PASS'
    commit = ready_commit(ready)
    assert c1pin.ancestor(commit), 'c1b READY commit %s is not merged here' % commit[:12]
    record.update(engine_selfrun=str(engine), engine_selfrun_sha256=c1pin.sha(engine / 'result.json'),
                  comparison_sha256=c1pin.sha(out / 'engine-comparison.json'), staging_check=staging,
                  staging_check_sha256=c1pin.sha(out / staging), steps=str(Path(steps).resolve()),
                  steps_sha256=c1pin.sha(steps), ready=str(Path(ready).resolve()), ready_sha256=c1pin.sha(ready),
                  ready_commit=commit)
    if continuation is not None:
        record.update(continuation=str(Path(continuation).resolve()), continuation_sha256=c1pin.sha(continuation))
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', type=Path)
    p.add_argument('--steps', type=Path)
    p.add_argument('--ready', type=Path)
    p.add_argument('--continuation', type=Path)
    p.add_argument('--check', action='store_true')
    p.add_argument('--dev', action='store_true')
    p.add_argument('--pinned', action='store_true')
    a = p.parse_args(argv)
    if a.pinned:
        pinned = load()
        print('stage1-step parent (c1b) pinned: %s manifest %s' % (pinned['source'][:12], pinned['manifest'][:12]))
        return 0
    assert a.out, '--out is required'
    if a.dev:
        print(json.dumps(pins(a.out, dev=True), indent=1, sort_keys=True))
        return 0
    assert a.steps and a.ready, '--steps and --ready are required'
    text = json.dumps(pins(a.out, a.steps, a.ready, a.continuation), indent=1, sort_keys=True) + '\n'
    if a.check:
        assert TARGET.read_text() == text, 'tools/stage1_step_parent.json differs from c1b\'s READY build'
        print('stage1-step parent (c1b) verified')
        return 0
    current = json.loads(TARGET.read_text())
    assert current.get('status', '').startswith(PLACEHOLDER), 'the parent is pinned already; never overwritten'
    TARGET.write_text(text)
    print('stage1-step parent (c1b) written ' + c1pin.sha(TARGET))
    return 0


if __name__ == '__main__':
    sys.exit(main())
