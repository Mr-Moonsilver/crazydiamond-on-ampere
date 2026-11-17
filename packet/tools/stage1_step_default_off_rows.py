#!/usr/bin/env python3
"""Write tools/stage1_step_default_off_rows.json: the rows (current text -> base text) that project this tree's engine onto c1's delivery fbb9123e.

The rows come from the committed files, not from a hand: for each changed engine file, one row per changed hunk, its context widened until its current
text is found exactly once in the file at its turn, so tools/stage1_step_default_off.py can apply them in order. Each row carries ``why`` (why the hunk holds with
FLASH_S5_LAZY_RECV and FLASH_S5_DECODE_MERGE unset), taken from tools/stage1_step_engine_suites.json 'why' by file: the builder of the engine change states it; a changed file without a why
refuses. Exactly the declared added file (native_fifo.py) may be added; any other added or removed engine file refuses. The review record is exempt (each chain's prepare-review step rewrites it).
--check verifies the committed file. No GPU, no box action.
usage: stage1_step_default_off_rows.py [--check] [--base REV]
"""
import argparse
import difflib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ENGINE = 'patches/s5_engine'
TARGET = ROOT / 'tools/stage1_step_default_off_rows.json'
SUITES = ROOT / 'tools/stage1_step_engine_suites.json'
BASE = 'fbb9123e'
EXEMPT = ('native_candidate_review.json',)
ADDED = ()       # the engine merge adds it; before the merge the file is absent and ``added`` is empty
SCHEMA = 'stage1-step-default-off-rows/1'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def hunks(old, new):
    """Line hunks of old -> new as (new_start, new_end, old_lines): the changed ranges, in file order."""
    a, b = old.splitlines(True), new.splitlines(True)
    return a, b, [(j1, j2, i1, i2) for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if tag != 'equal']


def rows_of(name, old, new, why):
    """One row per hunk: applied in order over the partly projected text, each current text found exactly once at its turn."""
    a, b, changed = hunks(old, new)
    rows, text = [], new
    # Applied from the top: the lines above a hunk are already the base's, the lines below still the current ones.
    for j1, j2, i1, i2 in changed:
        before, after = 0, 0
        while True:
            current = ''.join(a[i1 - before:i1] + b[j1:j2] + b[j2:j2 + after])
            base = ''.join(a[i1 - before:i2] + b[j2:j2 + after])
            if current and text.count(current) == 1:
                break
            if j2 + after < len(b):
                after += 1
            elif before < i1:
                before += 1
            else:
                raise ValueError('%s: no unique text for the hunk at new line %d' % (name, j1 + 1))
        rows.append(dict(file=name, current=current, base=base, why=why))
        text = text.replace(current, base, 1)
    assert text == old, (name, 'rows do not give the base back')
    return rows


def generate(base=BASE):
    suites = json.loads(SUITES.read_text())
    full = git('rev-parse', base).decode().strip()
    base_files = set(git('ls-tree', '-r', '--name-only', full, ENGINE).decode().split())
    head_files = set(git('ls-tree', '-r', '--name-only', 'HEAD', ENGINE).decode().split())
    added = sorted(path.split('/', 2)[2] for path in head_files - base_files)
    assert set(added) <= set(ADDED) and not base_files - head_files, ('an engine file added or removed', sorted(base_files ^ head_files))
    rows, files = [], []
    for path in sorted(base_files & head_files):
        name = path.split('/', 2)[2]
        if name in EXEMPT:
            continue
        old, new = git('show', '%s:%s' % (full, path)), git('show', 'HEAD:' + path)
        if old == new:
            continue
        why = (suites.get('why') or {}).get(name)
        assert why and why.strip(), 'changed engine file without a why in the suites file: ' + name
        files.append(name)
        rows += rows_of(name, old.decode(), new.decode(), why)
    return dict(schema=SCHEMA, base=base, engine=ENGINE, added=added, exempt=list(EXEMPT), files=files, rows=rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--check', action='store_true')
    p.add_argument('--base', default=BASE)
    a = p.parse_args(argv)
    text = json.dumps(generate(a.base), indent=1, sort_keys=True) + '\n'
    if a.check:
        assert TARGET.read_text() == text, 'rows differ from the committed engine: regenerate'
        print('default-off rows verified')
    else:
        TARGET.write_text(text)
        print('default-off rows written: %d' % len(json.loads(text)['rows']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
