#!/usr/bin/env python3
"""attn-meta default-off proof (FLASH_ATTN_META_GDN_BATCH unset or 0): the engine is c1's (alpha-next-c1 delivery fbb9123e).

1. Under patches/s5_engine every file is byte-identical to the base (``--base``, default fbb9123e), except:
   - the two added modules (attn_meta_batch.py, attn_meta_select.py), absent at the base and imported only under the switch
     (s5_startup's block and native_overlap_worker's branch, both below; the subprocess test in
     harness/test_attn_meta_select.py proves the off startup imports neither);
   - native_candidate_review.json (the review record each chain's prepare-review step rewrites; not engine code);
   - the files named in tools/attn_meta_default_off_rows.json, whose rows (current text -> base text, each found exactly
     once, applied in order) give the base file byte for byte. Every row carries why it holds with the switch unset.
2. Switch unset, actions unknown: REFUSAL_TESTS (harness/test_attn_meta_control.py, test_attn_meta_select.py) run in a child with the switch removed from
   the environment and pass: the client, the driver's first check and the worker know no attn-meta action, and startup with
   the switch unset or profiling calls nothing.
Prints a JSON receipt (``--out`` written once); exit 0 only when both hold.
usage: attn_meta_default_off.py [--base REV] [--rows FILE] [--out FILE]
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ENGINE = 'patches/s5_engine'
SWITCH = 'FLASH_ATTN_META_GDN_BATCH'
REFUSAL_TESTS = ('test_attn_meta_control.Exposure.test_switch_unset_knows_no_action',
                 'test_attn_meta_control.Exposure.test_qualification_tuples_unchanged',
                 'test_attn_meta_control.Startup.test_off_and_profiling_call_nothing',
                 'test_attn_meta_select.Hub.test_off_startup_sets_nothing_imports_nothing',
                 'test_attn_meta_control.WorkerPath.test_switch_unset_refuses')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, check=True).stdout


def project(text, rows):
    for row in rows:
        if text.count(row['current']) != 1:
            raise ValueError('row not found exactly once: %r' % row['current'][:80])
        text = text.replace(row['current'], row['base'], 1)
    return text


def files_check(base, doc):
    base_files = set(git('ls-tree', '-r', '--name-only', base, ENGINE).decode().split())
    cur_files = set(str(p.relative_to(ROOT)) for p in (ROOT / ENGINE).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    added = {ENGINE + '/' + n for n in doc['added']}
    exempt = {ENGINE + '/' + n for n in doc['exempt']}
    by_file = {}
    for row in doc['rows']:
        by_file.setdefault(ENGINE + '/' + row['file'], []).append(row)
    problems, rows_out = [], []
    if cur_files - base_files != added:
        problems.append('added files differ: %s' % sorted(cur_files - base_files))
    if base_files - cur_files:
        problems.append('removed files: %s' % sorted(base_files - cur_files))
    for path in sorted(cur_files & base_files):
        if path in exempt:
            continue
        cur = (ROOT / path).read_bytes()
        old = git('show', '%s:%s' % (base, path))
        if path in by_file:
            try:
                projected = project(cur.decode(), by_file[path]).encode()
            except ValueError as error:
                problems.append('%s: %s' % (path, error))
                continue
            same = projected == old
            rows_out.append(dict(file=path, rows=len(by_file[path]), projected_equals_base=same))
            if not same:
                problems.append('%s: projection differs from the base' % path)
        elif cur != old:
            problems.append('%s: undeclared change' % path)
    for path in by_file:
        if path not in cur_files & base_files:
            problems.append('%s: rows for a file that is not a changed base file' % path)
    return problems, rows_out


def refusals():
    env = {k: v for k, v in os.environ.items() if k != SWITCH}
    run = subprocess.run([sys.executable, '-B', '-m', 'unittest', *REFUSAL_TESTS], cwd=ROOT / 'harness', env=env,
                         capture_output=True, text=True, timeout=600)
    return run.returncode, run.stderr[-1500:]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--base', default='fbb9123e')
    p.add_argument('--rows', type=Path, default=ROOT / 'tools/attn_meta_default_off_rows.json')
    p.add_argument('--out', type=Path)
    a = p.parse_args(argv)
    doc = json.loads(a.rows.read_text())
    base = git('rev-parse', a.base).decode().strip()
    problems, rows = files_check(base, doc)
    rc, tail = refusals()
    if rc:
        problems.append('refusal tests rc %d' % rc)
    receipt = dict(schema='attn-meta-default-off/1', base=base, head=git('rev-parse', 'HEAD').decode().strip(),
                   rows_file=str(a.rows.relative_to(ROOT)) if a.rows.is_relative_to(ROOT) else str(a.rows),
                   rows_sha256=sha(a.rows.read_bytes()), tool_sha256=sha(Path(__file__).read_bytes()),
                   files=rows, added=doc['added'], exempt=doc['exempt'], refusal_tests=list(REFUSAL_TESTS),
                   refusal_rc=rc, refusal_tail=tail if rc else '', problems=problems,
                   verdict='PASS' if not problems else 'FAIL', passed=not problems)
    text = json.dumps(receipt, indent=1, sort_keys=True) + '\n'
    if a.out is not None:
        with a.out.open('x') as f:
            f.write(text)
    print(text, end='')
    return 0 if not problems else 1


if __name__ == '__main__':
    sys.exit(main())
