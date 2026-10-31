#!/usr/bin/env python3
"""Fixed source/AST declaration, compile proof and exact parent-overlay diff for native39 trunk.

Source level: every path changed since the native38 landing commit (72ea0ba) is
declared, and every changed module statement, class method or function of each
modified Python file is declared by name (native37_robust_freeze.prove).  Every
added or modified Python file compiles.  Overlay level: against the native38
held-b overlay, exactly the four engine files and the generation metadata move,
each engine file's changed definitions are declared, and the compiled libraries
are byte-identical.  Negative controls show an undeclared function, an
undeclared overlay definition and an undeclared path are refused.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from native37_robust_freeze import prove, manifest

ROOT = Path(__file__).resolve().parents[1]
PARENT_SOURCE = '72ea0baa208e1cfc04446898bbd58a0f3c4f84b8'
PARENT = Path('<hub-dir>/agents/flash-next/tmp/native38-plain-bind-held-b/'
              'q18a-native38-plain-bind-bundle-d7a0943b90ab/native30_overlay')
DECLARATION = ROOT / 'tools/native39_trunk_declarations.json'
RUNTIME = {'core/scheduler.py', 'e3/engine_driver.py', 'e3/native_overlap_session.py', 'e3/scheduler_hooks.py'}
META = {'S5-OVERLAP-GENERATION.json', 'e3/native-implementation-closure.json',
        'e3/native-overlap-readiness.json', 'e3/decode-probe-evidence.json', 'core/SHA256SUMS', 'SHA256SUMS'}


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


def source_level():
    declared = declaration()
    paths = source_paths()
    check_paths(paths, declared)
    modified = {row.split('\t', 1)[1] for row in paths if row.startswith('M\t') and row.endswith('.py')}
    assert modified == set(declared['definitions']), sorted(modified ^ set(declared['definitions']))
    result = {}
    for path, names in sorted(declared['definitions'].items()):
        result[path] = prove(git_show(PARENT_SOURCE, path), (ROOT / path).read_text(), path, names)
    return dict(parent=PARENT_SOURCE, paths=paths, definitions=result, compile=compile_proof(paths),
                declaration_sha256=hashlib.sha256(DECLARATION.read_bytes()).hexdigest())


def overlay_level(overlay):
    before, after = manifest(PARENT), manifest(overlay)
    assert set(before) == set(after), 'overlay file set changed'
    moved = {n for n in before if before[n] != after[n]} | {'SHA256SUMS'}
    assert moved == RUNTIME | META, ('unexpected overlay changes', sorted(moved ^ (RUNTIME | META)))
    declared = declaration()['overlay_definitions']
    assert set(declared) == RUNTIME
    result = {}
    for path in sorted(RUNTIME):
        result[path] = prove((PARENT / path).read_text(), (overlay / path).read_text(), path, declared[path])
    so = [n for n in before if n.endswith('.so')]
    assert so and all(before[n] == after[n] for n in so)
    return dict(parent_overlay=str(PARENT), changed=sorted(moved), definitions=result,
                compiled_libraries_byte_identical=True, compiled_libraries=sorted(so))


def refused(check):
    try:
        check()
    except AssertionError:
        return 'REFUSED'
    raise AssertionError('undeclared change accepted')


def negative_controls():
    declared = declaration()
    path = 'patches/s5_engine/scheduler_hooks.py'
    old, current = git_show(PARENT_SOURCE, path), (ROOT / path).read_text()
    overlay_path = 'core/scheduler.py'
    parent_text = (PARENT / overlay_path).read_text()
    return {
        'undeclared-function': refused(lambda: prove(old, current + '\ndef native39_undeclared():\n    pass\n', path,
                                                     declared['definitions'][path])),
        'undeclared-overlay-definition': refused(lambda: prove(parent_text, parent_text + '\nNATIVE39_UNDECLARED = 1\n',
                                                               overlay_path, declared['overlay_definitions'][overlay_path])),
        'undeclared-path': refused(lambda: check_paths(source_paths() + ['A\tharness/native39_undeclared.py'], declared)),
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--overlay', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    record = dict(verdict='PASS', source=source_level(), negative_controls=negative_controls())
    if a.overlay:
        record['overlay'] = overlay_level(a.overlay)
    with a.out.open('x') as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps(dict(verdict=record['verdict'], out=str(a.out))))
