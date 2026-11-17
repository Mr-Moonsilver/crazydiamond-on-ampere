#!/usr/bin/env python3
"""Fixed source declaration, compile proof and exact trunk-bundle identity for native39 lifecycle.

Source level: every path changed since the trunk held-b bundle source (75f7189)
is declared; patches/s5_engine is untouched; the only modified Python file is
harness/selftest.py and its only changed definition is the registry loop
(module:For#12, native37_robust_freeze.prove); every added or modified Python
file compiles.  Bundle level: against the trunk held-b bundle, the engine overlay
(native30_overlay), the engine sources (e3_builder) and the review
(native30-review.json) are byte-identical file by file, and the only changed
bundle files are selftest.py and native30-provenance.json.  Negative controls
show an undeclared function, an undeclared path and a moved overlay file are
refused.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from native37_robust_freeze import prove, manifest

ROOT = Path(__file__).resolve().parents[1]
PARENT_SOURCE = '75f71897aac4bf0764596c83d6ddbbba4b343301'
PARENT_BUNDLE = Path('<hub-dir>/agents/flash-next/tmp/native39-trunk-held-b/q18a-native39-trunk-bundle-756f826c221c')
PARENT_MANIFEST = '756f826c221c026440d7955a9f834fe72db83581439fed9bb69d07d314ea4264'
DECLARATION = ROOT / 'tools/native39_lifecycle_declarations.json'
IDENTICAL_TREES = ('native30_overlay/', 'e3_builder/')
IDENTICAL_FILES = ('native30-review.json',)
BUNDLE_CHANGED = {'selftest.py', 'native30-provenance.json'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_show(commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=ROOT, text=True)


def declaration():
    return json.loads(DECLARATION.read_text())


def source_paths():
    return subprocess.check_output(['git', 'diff', '--name-status', PARENT_SOURCE, '--', 'patches/s5_engine', 'harness',
                                    'tools'], cwd=ROOT, text=True).splitlines()


def compile_proof(paths):
    files = sorted(row.split('\t', 1)[1] for row in paths if row.endswith('.py') and row[0] in 'AM')
    for name in files:
        compile((ROOT / name).read_text(), name, 'exec')
    return dict(files=files, count=len(files), ok=True)


def check_paths(paths, declared):
    assert set(paths) == set(declared['paths']), ('undeclared source paths', sorted(set(paths) ^ set(declared['paths'])))
    assert not any(row.split('\t', 1)[1].startswith('patches/') for row in paths), 'engine sources changed'


def source_level():
    declared = declaration()
    assert declared['parent'] == PARENT_SOURCE
    paths = source_paths()
    check_paths(paths, declared)
    modified = {row.split('\t', 1)[1] for row in paths if row.startswith('M\t') and row.endswith('.py')}
    assert modified == set(declared['definitions']), sorted(modified ^ set(declared['definitions']))
    result = {}
    for path, names in sorted(declared['definitions'].items()):
        result[path] = prove(git_show(PARENT_SOURCE, path), (ROOT / path).read_text(), path, names)
    return dict(parent=PARENT_SOURCE, paths=paths, definitions=result, compile=compile_proof(paths),
                declaration_sha256=sha(DECLARATION))


def identical(before, after):
    """The engine trees and the review: same file set, same bytes."""
    for prefix in IDENTICAL_TREES:
        a = {k: v for k, v in before.items() if k.startswith(prefix)}
        b = {k: v for k, v in after.items() if k.startswith(prefix)}
        assert a and a == b, ('engine tree differs from the trunk bundle', prefix,
                              sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k)))
    for name in IDENTICAL_FILES:
        assert before[name] == after[name], ('review differs from the trunk bundle', name)


def bundle_level(bundle):
    assert sha(PARENT_BUNDLE / 'SHA256SUMS') == PARENT_MANIFEST
    before, after = manifest(PARENT_BUNDLE), manifest(bundle)
    identical(before, after)
    overlay_before, overlay_after = manifest(PARENT_BUNDLE / 'native30_overlay'), manifest(bundle / 'native30_overlay')
    assert overlay_before == overlay_after and sha(PARENT_BUNDLE / 'native30_overlay/SHA256SUMS') == \
        sha(bundle / 'native30_overlay/SHA256SUMS')
    changed = sorted(n for n in before if n in after and before[n] != after[n])
    assert set(changed) == BUNDLE_CHANGED, changed
    assert not set(before) - set(after), 'files removed from the trunk bundle'
    so = sorted(n for n in after if n.endswith('.so'))
    return dict(parent_bundle=str(PARENT_BUNDLE), parent_manifest=PARENT_MANIFEST, changed=changed,
                added=sorted(set(after) - set(before)), removed=[],
                identical_trees=list(IDENTICAL_TREES), identical_files=list(IDENTICAL_FILES),
                overlay_manifest=sha(bundle / 'native30_overlay/SHA256SUMS'), overlay_files=len(overlay_after),
                compiled_libraries=so, compiled_libraries_byte_identical=True, engine_changed=[])


def refused(check):
    try:
        check()
    except AssertionError:
        return 'REFUSED'
    raise AssertionError('undeclared change accepted')


def negative_controls():
    declared = declaration()
    path = 'harness/selftest.py'
    old, current = git_show(PARENT_SOURCE, path), (ROOT / path).read_text()
    before = manifest(PARENT_BUNDLE)
    moved = dict(before)
    overlay_file = next(n for n in sorted(moved) if n.startswith('native30_overlay/e3/') and n.endswith('.py'))
    moved[overlay_file] = '0' * 64
    return {
        'undeclared-function': refused(lambda: prove(old, current + '\ndef native39_undeclared():\n    pass\n', path,
                                                     declared['definitions'][path])),
        'undeclared-path': refused(lambda: check_paths(source_paths() + ['A\tharness/native39_undeclared.py'], declared)),
        'engine-source-path': refused(lambda: check_paths(source_paths() + ['M\tpatches/s5_engine/engine_driver.py'],
                                                          dict(declared, paths=declared['paths']
                                                               + ['M\tpatches/s5_engine/engine_driver.py']))),
        'moved-overlay-file': refused(lambda: identical(before, moved)),
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    record = dict(verdict='PASS', source=source_level(), negative_controls=negative_controls())
    if a.bundle:
        record['bundle'] = bundle_level(a.bundle)
    with a.out.open('x') as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps(dict(verdict=record['verdict'], out=str(a.out))))
