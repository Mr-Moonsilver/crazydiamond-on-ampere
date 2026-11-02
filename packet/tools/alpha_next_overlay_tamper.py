#!/usr/bin/env python3
"""Negative control for the alpha-next freeze's overlay check: a one-byte change in an overlay file must fail overlay_level.

tools/alpha_c0_overlay_tamper.py's cases (C0 was the first candidate to add overlay modules), over tools/alpha_next_freeze.py
against Step B's frozen overlay. One change: the added set and the moved e3 modules are computed from the two overlays themselves
and asserted to agree with the freeze's NEW and CHANGED/RUNTIME where the freeze declares them (lead, 2 Nov: NEW = the three added
modules, CHANGED = RUNTIME = every e3 module whose engine source differs from Step B's source); the tampered set is added | moved |
the declared CHANGED modules present in the overlay, as C0's NEW | CHANGED. The check compares each engine module with
make_s5_overlap.package(committed source). It works on a temporary copy of the built overlay and leaves the build untouched; frozen bundles are read-only, so the copy is made writable here (--candidate can be the frozen overlay itself).

Real objects: alpha_next_freeze.overlay_level, the built parent and candidate overlays, the committed sources, and
build_native30_held.manifest for the regenerated checksum list. Nothing is faked. The tamper is one changed byte.

Each case names the refusal it expects, so a refusal for the wrong reason fails the control. The checksum list is
regenerated except in the stale-manifest case: a tamperer who can edit a module can also rewrite the list, so the byte
comparison must hold without it.

Usage: alpha_next_overlay_tamper.py --parent PARENT/native30_overlay --candidate BUNDLE/native30_overlay --out JSON
"""
import argparse
import ast
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import tokenize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_native30_held import manifest, sha, verify
import alpha_next_freeze as freeze

DIFFERS = 'overlay module differs from the packaged committed source: '
META = {'SHA256SUMS', 'S5-OVERLAP-GENERATION.json', 'e3/native-implementation-closure.json',
        'e3/native-overlap-readiness.json', 'e3/decode-probe-evidence.json', 'core/SHA256SUMS'}


def change_sets(parent, candidate):
    """(added, runtime) from the two overlays: added = e3 files the candidate has and the parent lacks; runtime = added, the e3
    python modules whose bytes moved, and the freeze's declared CHANGED modules present in the overlay. Added must equal the
    freeze's NEW and every moved module must be declared, where the freeze declares them."""
    before, after = verify(parent), verify(candidate)
    added = {n for n in set(after) - set(before)}
    assert added and all(n.startswith('e3/') and n.endswith('.py') for n in added), ('added overlay files', sorted(added))
    moved = {n for n in before if n in after and before[n] != after[n] and n not in META}
    assert all(n.startswith('e3/') and n.endswith('.py') for n in moved), ('moved overlay files outside e3 modules', sorted(moved))
    if hasattr(freeze, 'NEW'):
        assert added == set(freeze.NEW), ('added overlay files differ from the freeze', sorted(added ^ set(freeze.NEW)))
    declared = set()
    for name in ('CHANGED', 'RUNTIME'):
        if hasattr(freeze, name):
            present = {n for n in getattr(freeze, name) if n in after} - added
            assert moved <= present, ('moved overlay modules the freeze does not declare (%s)' % name, sorted(moved - present))
            declared |= present
    return added, moved | added | declared


def flip(data, offset):
    """One byte changed to a different printable byte of the same width."""
    old = data[offset]
    return data[:offset] + bytes([old + 1 if old != 0x7e and old != 0x39 else old - 1]) + data[offset + 1:]


def comment_offset(data):
    """Byte offset of a letter inside the module's first comment: the AST does not see this byte."""
    lines = data.decode().splitlines(True)
    for token in tokenize.generate_tokens(io.StringIO(data.decode()).readline):
        if token.type == tokenize.COMMENT and not token.string.startswith('#!') and token.string[2:3].isalpha():
            row, col = token.start
            return len(''.join(lines[:row - 1]).encode()) + len(lines[row - 1][:col + 2].encode())
    raise AssertionError('no comment to tamper')


def declared_constant_offset(data):
    """Byte offset of a digit in an integer constant inside a definition of a new or declared module.
    The source proof reads committed sources, not this file, so only the byte comparison can catch this one."""
    text = data.decode()
    lines = text.splitlines(True)
    tree = ast.parse(text)
    for node in tree.body:
        scopes = [(node.name, node)] if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else []
        if isinstance(node, ast.ClassDef):
            scopes = [('%s.%s' % (node.name, f.name), f) for f in node.body
                      if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for name, scope in scopes:
            for item in ast.walk(scope):
                if isinstance(item, ast.Constant) and type(item.value) is int and item.value > 0:
                    return (name, item.value, len(''.join(lines[:item.lineno - 1]).encode())
                            + len(lines[item.lineno - 1].encode()[:item.col_offset]))
    return None


def attempt(parent, root):
    try:
        freeze.overlay_level(parent, root)
        return None
    except AssertionError as exc:
        return str(exc) or repr(exc)


def run(parent, candidate, log):
    rows = []
    added, runtime = change_sets(parent, candidate)
    with tempfile.TemporaryDirectory(prefix='alpha-next-tamper-') as held:
        root = Path(held) / 'native30_overlay'
        shutil.copytree(candidate, root)
        for path in [root, *root.rglob('*')]:
            if not path.is_symlink():
                path.chmod(path.stat().st_mode | 0o700 if path.is_dir() else path.stat().st_mode | 0o200)
        sums = (root / 'SHA256SUMS').read_bytes()

        def case(name, path, offset, expect, regenerate=True, note=None):
            target = root / path
            original = target.read_bytes()
            if offset is None:
                changed = None
            else:
                changed = flip(original, offset)
                assert len(changed) == len(original) and sum(a != b for a, b in zip(changed, original)) == 1
                target.write_bytes(changed)
                if regenerate:
                    manifest(root)
            refusal = attempt(parent, root)
            target.write_bytes(original)
            (root / 'SHA256SUMS').write_bytes(sums)
            ok = refusal is None if expect is None else refusal is not None and expect in refusal
            row = dict(case=name, path=path, offset=offset, one_byte=changed is not None, manifest_regenerated=regenerate and changed is not None,
                       expected='accepted' if expect is None else 'refused: ' + expect,
                       got='accepted' if refusal is None else 'refused: ' + refusal[:300], ok=ok)
            if note:
                row['note'] = note
            rows.append(row)
            print('%s %s %s offset=%s manifest_regenerated=%s got=%s' % ('OK  ' if ok else 'FAIL', name, path, offset,
                  row['manifest_regenerated'], row['got']), file=log, flush=True)

        case('T0-untampered-copy-accepted', 'SHA256SUMS', None, None)
        for rel in sorted(runtime):
            data = (root / rel).read_bytes()
            case('T1-comment-byte', rel, comment_offset(data), DIFFERS + rel,
                 note='a comment byte, invisible to the AST checks')
        rel, name, value, offset = next((rel,) + found for rel, found in
                                        ((rel, declared_constant_offset((root / rel).read_bytes()))
                                         for rel in sorted(added)) if found)
        case('T2-constant-in-a-new-module', rel, offset, DIFFERS + rel,
             note='%s, integer %d; this rests on the byte check' % (name, value))
        rel = 'e3/engine_driver.py'
        data = (root / rel).read_bytes()
        case('T3-stale-manifest', rel, comment_offset(data), rel, regenerate=False,
             note='checksum list left as built; verify() refuses first')
        others = sorted(n for n in verify(root) if n.endswith('.py') and n.startswith('e3/') and n not in runtime)
        generated = sorted(n for n in verify(root) if n.startswith('e3_sources/'))
        native = sorted(n for n in verify(root) if n.endswith('.so'))
        for label, rel in (('T4-other-e3-module', others[0]), ('T5-generated-vllm-source', generated[0]),
                           ('T6-launcher', 'run-flash-s5-test.sh'), ('T7-native-library', native[0])):
            size = (root / rel).stat().st_size
            case(label, rel, size // 2, rel, note='outside the declared change set')
        case('T8-restored-copy-accepted', 'SHA256SUMS', None, None)
        assert verify(root) == verify(candidate), 'the temporary copy did not restore'
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    rows = run(a.parent, a.candidate, sys.stdout)
    ok = all(r['ok'] for r in rows)
    result = dict(schema='alpha-next-overlay-tamper/1', verdict='PASS' if ok else 'FAIL', cases=rows,
                  derived_from='tools/alpha_c0_overlay_tamper.py',
                  candidate_manifest=sha(a.candidate / 'SHA256SUMS'), parent_manifest=sha(a.parent / 'SHA256SUMS'))
    with a.out.open('x') as handle:
        json.dump(result, handle, indent=1, sort_keys=True)
        handle.write('\n')
    print('OVERLAY_TAMPER %s %d/%d cases as expected' % (result['verdict'], sum(r['ok'] for r in rows), len(rows)))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
