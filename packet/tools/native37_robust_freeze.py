#!/usr/bin/env python3
"""Native37 freeze/AST diff: only the declared files and the declared definitions changed.

Source level: the committed tree against the held-c build source (720b123) for
patches/s5_engine, harness and tools; every changed path must be declared, and
in every changed Python file the set of changed/added/removed definitions
(module functions, classes and their methods, module assignments and imports)
must equal the declared set.  Overlay level: the regenerated overlay against
held-c's overlay; the changed files must be exactly the three engine files plus
the generation/closure metadata, and each engine file's definition diff must
equal the source-level declaration.  Negative controls mutate one undeclared
definition and must be refused.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PARENT_SOURCE = '720b1238310de0ba61cc9a2b9f8a7dd8d9c1abb4'
HELD_C_OVERLAY = Path('<hub-dir>/agents/flash-next/tmp/native35-p-held-c/'
                      'q18a-native35-p-c-bundle-fe194172a6c8/native30_overlay')
# The declared change, definition by definition.  'module:<name>' is a module-level
# assignment or import line; 'Class.method' a method; bare names are module defs.
ENGINE = {
    'scheduler_hooks.py': {
        'module:import time', 'module:MAX_PLAN_REFUSAL_MEMO', 'staged_chunk_plan', '_plan_result',
        'PrefillAdmissionController.__init__', 'PrefillAdmissionController._refuse',
        'PrefillAdmissionController._now_ns', 'PrefillAdmissionController._plan_key',
        'PrefillAdmissionController._memoised_plan_refusal', 'PrefillAdmissionController._remember_plan_refusal',
        'PrefillAdmissionController.native_admission_record', 'PrefillAdmissionController._admit_observed',
        'defer_stock_prefix_hit', '_defer_stock_while_split_armed'},
    'engine_driver.py': {
        'NativeSegmentDriver._native_admission_record', 'NativeSegmentDriver._status_record',
        'NativeSegmentDriver.overlap_evidence'},
    'native_overlap_session.py': {
        'NativeOverlapSession._action_prefill_end', 'NativeOverlapSession._refused_prefill_end',
        'native_admission_refusal'},
}
HARNESS_PY = {
    'e3_native_performance.py': {'prefill', 'return_full_after_error'},
    'test_native35_performance.py': {
        'TwoRoundComposition.test_rounds_keep_session_identity_differ_only_by_salt_and_are_both_retained'},
    'selftest.py': {'module:For#12'},  # the registration loop's suite list grows by one row (checked below)
}
SOURCE_PATHS = ({'M\tpatches/s5_engine/' + n for n in ENGINE} | {'M\tpatches/s5_engine/native_candidate_review.json'}
                | {'M\tharness/' + n for n in HARNESS_PY}
                | {'A\tharness/' + n for n in ('e3_native37_admission_probe.py', 'test_native37_robust.py',
                                               'native37_fixture_leaves.json')}
                | {'A\ttools/' + n for n in ('native37_robust_selfrun.py', 'native37_robust_freeze.py',
                                             'build_native37_robust.py')})
OVERLAY_METADATA = {'S5-OVERLAP-GENERATION.json', 'e3/native-implementation-closure.json',
                    'e3/native-overlap-readiness.json', 'e3/decode-probe-evidence.json', 'SHA256SUMS'}


def definitions(text):
    """Name -> ast.dump for every module statement, class method and module def."""
    tree = ast.parse(text)
    rows, kinds = {}, {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            rows[node.name] = ast.dump(node)
        elif isinstance(node, ast.ClassDef):
            body = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    rows[node.name + '.' + item.name] = ast.dump(item)
                else:
                    body.append(ast.dump(item))
            rows['class:' + node.name] = ast.dump(ast.ClassDef(
                name=node.name, bases=node.bases, keywords=node.keywords, body=[],
                decorator_list=node.decorator_list, type_params=getattr(node, 'type_params', []))) + '|' + '|'.join(body)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            rows['module:' + ast.unparse(node)] = ast.dump(node)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            rows['module:' + node.targets[0].id] = ast.dump(node)
        else:
            # Unnamed statements are keyed by kind and per-kind order, so an added
            # definition elsewhere never renames them.
            kind = type(node).__name__
            kinds[kind] = kinds.get(kind, 0) + 1
            rows['module:%s#%d' % (kind, kinds[kind])] = ast.dump(node)
    return rows


def changed(old, new):
    a, b = definitions(old), definitions(new)
    return sorted(name for name in set(a) | set(b) if a.get(name) != b.get(name))


def prove(old, new, label, declared):
    got = set(changed(old, new))
    if got != set(declared):
        raise AssertionError('%s: changed definitions differ from the declaration: undeclared %s, unchanged %s'
                             % (label, sorted(got - set(declared)), sorted(set(declared) - got)))
    return dict(changed=sorted(got), old_sha256=hashlib.sha256(old.encode()).hexdigest(),
                new_sha256=hashlib.sha256(new.encode()).hexdigest())


def selftest_rows(old, new):
    """selftest.py: the only change is one added registration row for the native37 suite."""
    a, b = old.splitlines(), new.splitlines()
    added = [line for line in b if line not in a]
    assert len(b) == len(a) + 1 and all(line in b for line in a), 'selftest.py changed beyond one added line'
    assert len(added) == 1 and added[0].lstrip().startswith("('test_native37_robust.py',"), added
    return added[0].strip()[:80]


def git_show(rev, path):
    return subprocess.check_output(['git', 'show', '%s:%s' % (rev, path)], cwd=ROOT, text=True)


def source_level():
    rows = [line for line in subprocess.check_output(
        ['git', 'diff', '--name-status', PARENT_SOURCE, 'HEAD', '--', 'patches/s5_engine', 'harness', 'tools'],
        cwd=ROOT, text=True).splitlines() if line]
    assert set(rows) == SOURCE_PATHS, ('undeclared paths', sorted(set(rows) ^ SOURCE_PATHS))
    proofs = {}
    for name, declared in ENGINE.items():
        path = 'patches/s5_engine/' + name
        proofs[path] = prove(git_show(PARENT_SOURCE, path), (ROOT / path).read_text(), path, declared)
    for name, declared in HARNESS_PY.items():
        path = 'harness/' + name
        old, new = git_show(PARENT_SOURCE, path), (ROOT / path).read_text()
        proofs[path] = prove(old, new, path, declared)
        if name == 'selftest.py':
            proofs[path]['added_row'] = selftest_rows(old, new)
    return dict(paths=sorted(rows), definitions=proofs)


def manifest(root):
    rows = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        pin, name = line.split(None, 1)
        rows[name] = pin
    return rows


def overlay_level(overlay):
    old, new = manifest(HELD_C_OVERLAY), manifest(overlay)
    assert set(old) == set(new), sorted(set(old) ^ set(new))
    moved = sorted(n for n in old if old[n] != new[n]) + ['SHA256SUMS']
    engine = {'e3/' + n for n in ENGINE}
    assert set(moved) == engine | OVERLAY_METADATA, ('overlay changed outside the declaration', moved)
    proofs = {name: prove((HELD_C_OVERLAY / name).read_text(), (overlay / name).read_text(), name,
                          ENGINE[Path(name).name]) for name in sorted(engine)}
    return dict(changed=moved, definitions=proofs,
                compiled_libraries_byte_identical=all(old[n] == new[n] for n in old if n.endswith('.so')))


def negative_controls():
    """The proof refuses one undeclared change next to the declared ones."""
    out = {}
    path = ROOT / 'patches/s5_engine/scheduler_hooks.py'
    text, old = path.read_text(), git_show(PARENT_SOURCE, 'patches/s5_engine/scheduler_hooks.py')
    cases = {
        'undeclared-method-body': text.replace('        return 0 if self.parked is None else 1\n',
                                               '        return 0 if self.parked is None else 2\n', 1),
        'undeclared-module-def': text.replace('def block_held(block_pool, block):',
                                              'def block_held(block_pool, block, extra=None):', 1),
        'declared-change-reverted': old.replace('def staged_chunk_plan(start, end, cap, minimum, split, *, query_limit=65536):',
                                                'def staged_chunk_plan(start, end, cap, minimum, split, *, query_limit=65537):', 1),
    }
    for name, mutated in cases.items():
        assert mutated not in (text, old), name
        try:
            prove(old, mutated, 'scheduler_hooks.py', ENGINE['scheduler_hooks.py'])
        except AssertionError as exc:
            out[name] = 'REFUSED: ' + str(exc)[:160]
        else:
            raise AssertionError('freeze proof accepted a mutation: ' + name)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--overlay', type=Path)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    record = dict(schema='native37-robust-freeze/1', parent_source=PARENT_SOURCE,
                  head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  source=source_level(), negative_controls=negative_controls())
    if a.overlay is not None:
        record['overlay'] = overlay_level(a.overlay.resolve())
        record['held_c_overlay'] = str(HELD_C_OVERLAY)
    record['verdict'] = 'PASS'
    text = json.dumps(record, indent=2, sort_keys=True) + '\n'
    if a.out is not None:
        with a.out.open('x') as f:
            f.write(text)
    print(json.dumps({'verdict': 'PASS', 'paths': len(record['source']['paths']),
                      'overlay_changed': (record.get('overlay') or {}).get('changed'),
                      'negative_controls': sorted(record['negative_controls'])}))
    print('freeze_sha256', hashlib.sha256(text.encode()).hexdigest())


if __name__ == '__main__':
    sys.exit(main())
