#!/usr/bin/env python3
"""Write tools/alpha_next_c1_queue_pins.json from the gate chain's build output (OUT); --check recomputes from OUT and compares.

tools/alpha_next_queue_pins_write.py for alpha-next-c1 (file, schema and freeze record names only).

The queue's bundle and overlay pins are held hub-side (lead's decision, 2 Nov): the overlay manifest and package identity depend on
the declaration text through the review the build prepares (build_native43.prepare_review pins every tracked harness file,
harness/runs/ included; make_s5_overlap copies that review into e3/native-implementation-closure.json, which the package identity
and the overlay SHA256SUMS cover), so no harness text can hold them. Step B's and C0's chains held theirs outside the harness
bundle too (the derived owner runner's BUNDLE/EXPECTED_SOURCES/REVIEW and the precheck conf). This file is the hub-side record the
waiter and tools/alpha_next_c1_delivery.py compare against.

Values, all from OUT and every one cross-checked before anything is written:
  harness_bundle_sha256    sha256 of the held bundle's SHA256SUMS (build.json 'manifest'; the bundle directory name carries its
                           first 12 characters); the bundle verifies completely against its SHA256SUMS;
  overlay_manifest_sha256  sha256 of the bundle's native30_overlay/SHA256SUMS (build.json 'overlay_manifest'); the overlay
                           verifies completely against it;
  package_sha256           S5-OVERLAP-GENERATION.json package_sha256, equal to e3/native-overlap-readiness.json's and to the
                           payload identity recomputed by the bundle's own e3_builder/native_readiness.payload_identity.
OUT must be a finished build: build.json (schema native43-build/1), freeze-proof.json at build.json's sha, alpha-next-c1-freeze.json
(schema alpha-next-c1-freeze/1, verdict PASS, naming the same bundle and overlay manifests), the owner runner at build.json's sha, the
held bundle and its overlay; anything missing or partial refuses.

Outside the build (lead's item c), proved on every write and check: the file is under tools/ and not under harness/ or
patches/s5_engine/ (the only trees build_native43.stage copies); it is not in the build spec's tools (the only tools the build
copies into the bundle and hashes; tools/alpha_next_c1_spec.py leaves it out as HUB_SIDE, though it takes every other added
tools/*.py and *.json); no file of that name and no SHA256SUMS row naming it is in the held bundle; and the build's recorded
hashes (bundle and overlay SHA256SUMS, runner) are recomputed after the write in --check, so the write changed none of them.

Writing: placeholder values are filled; equal values are left as they are (nothing written); different non-placeholder values
refuse unless --supersede REASON is given, and then the previous record (values, status, build OUT) is appended to 'history',
never deleted. The commit of the pins file is the lead's; this tool never runs git commit. No GPU, no box action.
usage: alpha_next_c1_queue_pins_write.py --out OUT [--check] [--supersede REASON]
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from build_native30_held import sha, verify  # noqa: E402

PINS = ROOT / 'tools/alpha_next_c1_queue_pins.json'
REL = 'tools/alpha_next_c1_queue_pins.json'
SCHEMA = 'alpha-next-c1-queue-pins/1'
PLACEHOLDER = 'PLACEHOLDER-filled-by-the-packaging-step'
KEYS = ('overlay_manifest_sha256', 'package_sha256', 'harness_bundle_sha256')
STAGED_TREES = ('harness/', 'patches/s5_engine/')
NOTE = ('Hub-side queue pins, written by tools/alpha_next_c1_queue_pins_write.py from the gate chain build OUT and committed by the '
        'lead. overlay_manifest_sha256 = sha256 of the held bundle native30_overlay/SHA256SUMS; package_sha256 = its '
        'S5-OVERLAP-GENERATION.json package_sha256; harness_bundle_sha256 = sha256 of the held bundle SHA256SUMS. No harness text '
        'holds them (the review the build prepares pins the declaration text, and the overlay covers the review); the waiter and '
        'tools/alpha_next_c1_delivery.py compare against this file. A placeholder means not packaged yet.')
HEX = re.compile(r'^[0-9a-f]{64}$')


def _payload_identity(overlay, e3_builder):
    """The bundle's own native_readiness.payload_identity over the overlay (the candidate's rule, not a copy of it)."""
    spec = importlib.util.spec_from_file_location('alpha_next_c1_pins_native_readiness', e3_builder / 'native_readiness.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.payload_identity(overlay)


def _verified(root, what):
    try:
        return verify(root)
    except AssertionError as exc:
        raise AssertionError('%s does not verify against its SHA256SUMS: %s' % (what, str(exc) or 'file set differs')) from None


def from_out(out):
    """The three values from a finished build OUT, every one cross-checked; refuses a missing or partial OUT."""
    out = Path(out).resolve()
    assert out.is_dir(), 'build OUT missing: %s' % out
    for name in ('build.json', 'freeze-proof.json', 'alpha-next-c1-freeze.json'):
        assert (out / name).is_file(), 'partial build OUT, no %s' % name
    build = json.loads((out / 'build.json').read_text())
    assert build.get('schema') == 'native43-build/1' and build.get('held') is True, 'not a finished build record'
    assert sha(out / 'freeze-proof.json') == build['freeze_proof_sha256'], 'freeze-proof.json differs from build.json'
    freeze = json.loads((out / 'alpha-next-c1-freeze.json').read_text())
    assert freeze.get('schema') == 'alpha-next-c1-freeze/1' and freeze.get('verdict') == 'PASS', 'alpha-next-c1 freeze not PASS'
    bundle, runner = Path(build['bundle']), Path(build['runner'])
    assert bundle.parent == out and bundle.is_dir(), 'held bundle not under OUT: %s' % bundle
    assert runner.parent == out and runner.is_file() and sha(runner) == build['runner_sha256'], 'owner runner missing or moved'
    manifest = sha(bundle / 'SHA256SUMS')
    assert manifest == build['manifest'] == freeze['candidate_manifest'] and bundle.name.endswith('-bundle-' + manifest[:12]), \
        'bundle manifest differs from the build record'
    _verified(bundle, 'held bundle')
    overlay = bundle / 'native30_overlay'
    assert Path(build['overlay']) == overlay, 'build overlay is not the held bundle overlay'
    overlay_manifest = sha(overlay / 'SHA256SUMS')
    assert overlay_manifest == build['overlay_manifest'] == freeze['candidate_overlay_manifest'], \
        'overlay manifest differs from the build record'
    _verified(overlay, 'overlay')
    generation = json.loads((overlay / 'S5-OVERLAP-GENERATION.json').read_text())
    ready = json.loads((overlay / 'e3/native-overlap-readiness.json').read_text())
    package = generation.get('package_sha256')
    assert package and HEX.match(package) and ready.get('package_sha256') == package, 'package identity missing or inconsistent'
    assert _payload_identity(overlay, bundle / 'e3_builder') == package, 'package identity differs from the overlay payload'
    values = dict(overlay_manifest_sha256=overlay_manifest, package_sha256=package, harness_bundle_sha256=manifest)
    assert all(HEX.match(v) for v in values.values()), values
    return values, build


def outside_build(build):
    """Item c: the pins file cannot be part of the bundle or of the tools the build copies and hashes."""
    assert not REL.startswith(STAGED_TREES), REL
    spec = json.loads(Path(build['spec']).read_text())
    assert sha(build['spec']) == build['spec_sha256'], 'build spec changed since the build'
    assert Path(REL).name not in spec['tools'], 'pins file in the build tools'
    tracked = subprocess.check_output(['git', 'ls-files', '--', *STAGED_TREES], cwd=ROOT, text=True).split()
    assert REL not in tracked, 'pins file tracked under a staged tree'
    bundle = Path(build['bundle'])
    named = [str(p.relative_to(bundle)) for p in bundle.rglob(Path(REL).name)]
    rows = [line for line in (bundle / 'SHA256SUMS').read_text().splitlines() if line.endswith(Path(REL).name)]
    assert not named and not rows, ('pins file inside the held bundle', named, rows)
    return dict(path=REL, staged_trees=list(STAGED_TREES), in_spec_tools=False, spec_tools=len(spec['tools']), in_bundle=False,
                in_bundle_manifest=False, tracked_under_staged_tree=False)


def record(values, build, out, current):
    rec = dict(schema=SCHEMA, status='FILLED: from the gate chain build at build_out', **values, build_out=str(Path(out).resolve()),
               build_json_sha256=sha(Path(out).resolve() / 'build.json'), source_commit=build['source_commit'],
               review_sha256=build['review_sha256'], runner_sha256=build['runner_sha256'], written_by=REL.replace('.json', '_write.py'),
               note=NOTE, history=list(current.get('history') or []))
    return rec


def placeholders(current):
    return all(current.get(k) == PLACEHOLDER for k in KEYS)


def check(out):
    """The pins file holds exactly OUT's values, taken from OUT; for --check and tools/alpha_next_c1_delivery.py."""
    current = json.loads(PINS.read_text())
    assert current.get('schema') == SCHEMA, current.get('schema')
    values, build = from_out(out)
    proof = outside_build(build)
    assert not placeholders(current), 'queue pins are placeholders: not written yet'
    got = {k: current.get(k) for k in KEYS}
    assert got == values, ('queue pins differ from OUT', {k: (got[k], values[k]) for k in KEYS if got[k] != values[k]})
    assert current.get('build_out') == str(Path(out).resolve()), ('queue pins taken from another OUT', current.get('build_out'))
    return dict(verdict='PASS', path=REL, sha256=sha(PINS), build_out=current['build_out'], outside_build=proof,
                history=len(current.get('history') or []), **values)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', type=Path, required=True, help='the gate chain build OUT (holds build.json)')
    p.add_argument('--check', action='store_true', help='recompute from OUT and compare with the pins file; write nothing')
    p.add_argument('--supersede', metavar='REASON', help='replace different non-placeholder values; the old record goes to history')
    a = p.parse_args(argv)
    if a.check:
        assert not a.supersede, '--check takes no --supersede'
        print(json.dumps(dict(check(a.out), check=True), sort_keys=True))
        return 0
    current = json.loads(PINS.read_text())
    assert current.get('schema') == SCHEMA, current.get('schema')
    values, build = from_out(a.out)
    proof = outside_build(build)
    if placeholders(current):
        rec = record(values, build, a.out, current)
    else:
        assert all(HEX.match(str(current.get(k))) for k in KEYS), 'pins neither placeholders nor filled: refused'
        if {k: current[k] for k in KEYS} == values:
            assert not a.supersede, 'values equal: nothing to supersede'
            print(json.dumps(dict(verdict='UNCHANGED', outside_build=proof, **values), sort_keys=True))
            return 0
        assert a.supersede and a.supersede.strip(), 'pins hold different values: refused without --supersede REASON'
        rec = record(values, build, a.out, current)
        rec['history'].append(dict({k: v for k, v in current.items() if k not in ('history', 'note')},
                                   superseded_because=a.supersede.strip(), superseded_by_build_out=rec['build_out']))
    PINS.write_text(json.dumps(rec, indent=2) + '\n')
    print(json.dumps(dict(verdict='WRITTEN', outside_build=proof, **values), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
