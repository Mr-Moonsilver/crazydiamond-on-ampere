#!/usr/bin/env python3
"""Negative control for the q18a-stage1-step freeze's overlay check: a one-byte change in an overlay file must fail overlay_level.

c1's control (tools/alpha_c0_overlay_tamper.py imported unchanged with its freeze module set), attn-meta's wrapper by name, with the freeze set to tools/stage1_step_freeze.py: the
same cases (T0 untampered, T1 a comment byte in every changed runtime module, T2 an integer constant, T3 a stale checksum list, T4-T7 a byte outside the declared change set, T8
restored). c0's T2 reads ``freeze.NEW``; the view below gives it the added module and the changed runtime modules (T2's name is c0's; the row's note says which module);
overlay_level itself is the real one. Works on a temporary copy; the build is untouched.
Usage: stage1_step_overlay_tamper.py --parent PARENT/native30_overlay --candidate BUNDLE/native30_overlay --out JSON
"""
import argparse
import ast
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alpha_c0_overlay_tamper as c0  # noqa: E402
import stage1_step_freeze as freeze  # noqa: E402



class FreezeView:
    """tools/stage1_step_freeze.py as c0's tamper reads it: NEW is the added module and the changed runtime modules."""
    NEW = set(freeze.NEW) | {n for n in freeze.CHANGED if n.startswith('e3/')}

    def __getattr__(self, name):
        return getattr(freeze, name)


c0.freeze = FreezeView()
_comment_offset = c0.comment_offset


def comment_offset(data):
    """c0's comment byte; a module with no comment at all: a letter inside its module docstring (attn-meta's rule). T1 still requires the
    byte check to refuse; every other module keeps the comment byte the AST does not see."""
    try:
        return _comment_offset(data)
    except AssertionError:
        text = data.decode()
        start = text.index('"""') + 3
        assert ast.get_docstring(ast.parse(text)) and text[start].isalpha(), 'no comment and no docstring to tamper'
        return len(text[:start].encode())


c0.comment_offset = comment_offset


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    rows = c0.run(a.parent, a.candidate, sys.stdout)
    names = {r['case'] for r in rows}
    ok = all(r['ok'] for r in rows) and any(r['case'] == 'T2-constant-in-a-new-module' and r['path'] in c0.freeze.NEW
                                            for r in rows) and {'T0-untampered-copy-accepted', 'T8-restored-copy-accepted'} <= names
    result = dict(schema='stage1-step-overlay-tamper/1', verdict='PASS' if ok else 'FAIL', cases=rows,
                  derived_from='tools/alpha_c0_overlay_tamper.py over tools/stage1_step_freeze.py',
                  candidate_manifest=c0.sha(a.candidate / 'SHA256SUMS'), parent_manifest=c0.sha(a.parent / 'SHA256SUMS'))
    with a.out.open('x') as handle:
        json.dump(result, handle, indent=1, sort_keys=True)
        handle.write('\n')
    print('OVERLAY_TAMPER %s %d/%d cases as expected' % (result['verdict'], sum(r['ok'] for r in rows), len(rows)))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
