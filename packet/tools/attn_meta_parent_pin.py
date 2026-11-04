#!/usr/bin/env python3
"""Write tools/attn_meta_parent.json: alpha-next-c1's delivered build (chain b), for the q18a-attn-meta tools to pin (CPU, no box).

The parent is c1's delivery: results/hub/alpha-next-c1/delivery.json at c1's delivery commit (fbb9123e, the commit this branch is
off), schema alpha-next-c1-delivery/1, verdict PASS. As tools/alpha_next_parent_pin.py did for Step B (a PASS delivery), the pin
refuses unless that delivery names the same build: the build OUT's build.json (native43-build/1, name q18a-alpha-next-c1, its
manifest, overlay manifest and runner sha the delivery's), its engine self-run (verdict PASS, sha the delivery's), its
alpha-next-c1-freeze.json (verdict PASS, sha the delivery's, candidate manifest the build's), and the delivery commit and the
build source are ancestors of HEAD. Nothing else is read or written.

Until written the committed file is a placeholder (status NOT-PINNED): every attn-meta tool reads the parent through load(), which
exits 3 'parent not pinned'. Writing replaces only the placeholder; a pinned file is never overwritten (on the rebind to c1b's
READY a new parent is pinned by a new commit, after review). --check verifies the committed file. --dev prints a development pin
(no delivery; marked dev; read only through ATTN_META_PARENT, never committed) so tests can exercise the tools.

    python3 tools/attn_meta_parent_pin.py --out C1_OUT --delivery DELIVERY_JSON [--check]
    python3 tools/attn_meta_parent_pin.py --pinned        # exit 0 when pinned, 3 'parent not pinned' otherwise
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'tools/attn_meta_parent.json'
SCHEMA = 'attn-meta-parent/1'
PLACEHOLDER = 'NOT-PINNED'
NOT_PINNED_RC = 3
DEV_ENV = 'ATTN_META_PARENT'
PARENT_NAME = 'q18a-alpha-next-c1'
DELIVERY_SCHEMA = 'alpha-next-c1-delivery/1'
DELIVERY_PATH = 'results/hub/alpha-next-c1/delivery.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True)


def ancestor(commit):
    return git('merge-base', '--is-ancestor', commit, 'HEAD').returncode == 0


def load(path=None):
    """The pinned parent; exit 3 with 'parent not pinned' while the file is the placeholder (alpha_next_c1_parent_pin's rule)."""
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


def delivery_commit(delivery):
    """The commit that added the delivery record (its last change in history)."""
    rel = Path(delivery).resolve().relative_to(ROOT)
    out = git('log', '-1', '--format=%H', '--', str(rel))
    assert out.returncode == 0 and out.stdout.strip(), 'delivery record not committed: ' + str(rel)
    return out.stdout.strip()


def pins(out, delivery=None, dev=False):
    out = Path(out).resolve()
    build = json.loads((out / 'build.json').read_text())
    assert build['schema'] == 'native43-build/1' and build['name'] == PARENT_NAME, build.get('name')
    freeze_path = out / 'alpha-next-c1-freeze.json'
    freeze = json.loads(freeze_path.read_text())
    assert freeze['verdict'] == 'PASS' and freeze['candidate_manifest'] == build['manifest'], 'c1 freeze is not this build\'s'
    assert ancestor(build['source']), 'c1 build source %s is not merged here' % build['source'][:12]
    record = dict(schema=SCHEMA, build=str(out / 'build.json'), build_sha256=sha(out / 'build.json'), manifest=build['manifest'],
                  overlay_manifest=build['overlay_manifest'], runner_sha256=build['runner_sha256'], source=build['source'],
                  source_commit=build['source_commit'], expected_selfrun_tests=build['expected_selfrun_tests'],
                  freeze=str(freeze_path), freeze_sha256=sha(freeze_path), spec=build['spec'], spec_sha256=build['spec_sha256'])
    if dev:
        record.update(dev=True, status='DEV: development parent; no delivery pinned')
        return record
    d = json.loads(Path(delivery).read_text())
    assert d['schema'] == DELIVERY_SCHEMA and d['verdict'] == 'PASS', (d.get('schema'), d.get('verdict'))
    assert d['name'] == PARENT_NAME and d['manifest'] == build['manifest'], 'delivery names another bundle'
    assert d['overlay_manifest'] == build['overlay_manifest'] and d['runner_sha256'] == build['runner_sha256']
    assert d['build_commit'] == build['source'], 'delivery names another build commit'
    assert d['freeze_proof_sha256'] == build['freeze_proof_sha256']
    assert d['alpha_next_c1_freeze']['verdict'] == 'PASS' and d['alpha_next_c1_freeze']['sha256'] == sha(freeze_path)
    engine = Path(d['engine_selfrun']['path'])
    assert engine.parent.parent == out, 'delivered engine self-run is not this OUT\'s'
    assert d['engine_selfrun']['verdict'] == 'PASS' and json.loads(engine.read_text())['verdict'] == 'PASS'
    assert d['engine_selfrun']['sha256'] == sha(engine), 'engine self-run record changed'
    commit = delivery_commit(delivery)
    assert ancestor(commit), 'c1 delivery commit %s is not merged here' % commit[:12]
    record.update(engine_selfrun=str(engine.parent), engine_selfrun_sha256=sha(engine),
                  comparison_sha256=d['comparison']['sha256'], staging_check_sha256=d['staging_check']['sha256'],
                  vm_sha256=d['vm']['sha256'], delivery=str(Path(delivery).resolve().relative_to(ROOT)),
                  delivery_sha256=sha(delivery), delivery_commit=commit)
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', type=Path, help="c1's gate-chain OUT directory (its build-b)")
    p.add_argument('--delivery', type=Path, default=ROOT / DELIVERY_PATH)
    p.add_argument('--check', action='store_true')
    p.add_argument('--dev', action='store_true', help='print the pins of a development parent; writes nothing')
    p.add_argument('--pinned', action='store_true', help='exit 0 when pinned, 3 otherwise')
    a = p.parse_args(argv)
    if a.pinned:
        pinned = load()
        print('attn-meta parent pinned: %s manifest %s' % (pinned['source'][:12], pinned['manifest'][:12]))
        return 0
    assert a.out, '--out is required'
    if a.dev:
        print(json.dumps(pins(a.out, dev=True), indent=1, sort_keys=True))
        return 0
    text = json.dumps(pins(a.out, a.delivery), indent=1, sort_keys=True) + '\n'
    if a.check:
        assert TARGET.read_text() == text, 'tools/attn_meta_parent.json differs from c1\'s delivered build'
        print('attn-meta parent verified')
        return 0
    current = json.loads(TARGET.read_text())
    assert current.get('status', '').startswith(PLACEHOLDER), 'the parent is pinned already; never overwritten'
    TARGET.write_text(text)
    print('attn-meta parent written ' + sha(TARGET))
    return 0


if __name__ == '__main__':
    sys.exit(main())
