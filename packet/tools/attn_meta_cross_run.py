#!/usr/bin/env python3
"""The pinned q18a-alpha-next-c1 cross-run numbers for the q18a-attn-meta read (``e3_attn_meta_box.py read --cross-run FILE``).

Lead 3 Nov (frozen reference): absolute decode tok/s are read, labelled cross-run, next to the q18a-alpha-next-c1 report's C1-off
band-1 decode-only rate; never a verdict input; no --frozen-reference file. The report is pinned by sha256 from fn-main
(results/box/q18a-alpha-next-c1/box/run/q18a-alpha-next-c1/alpha-next-c1-report.md, with the commit that added it); the numbers are
taken from the structured read record beside it (alpha-next-c1-read.json.xz, also sha-pinned): every core arm of band 1 with C1
off, its own decode-only window rate (``decode_only_tok_s``), and their median. Numbers only. Written once (--out); --check
recomputes and compares.
usage: attn_meta_cross_run.py --fn-main DIR --out FILE [--check]
"""
import argparse
import hashlib
import json
import lzma
from pathlib import Path
import statistics
import subprocess
import sys

REL = 'results/box/q18a-alpha-next-c1/box/run/q18a-alpha-next-c1'
REPORT = 'alpha-next-c1-report.md'
READ = 'alpha-next-c1-read.json.xz'
SCHEMA = 'attn-meta-cross-run/1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numbers(fn_main):
    fn_main = Path(fn_main).resolve()
    place = fn_main / REL
    commit = subprocess.run(['git', '-C', str(fn_main), 'log', '-1', '--format=%H', '--', REL + '/' + REPORT],
                            capture_output=True, text=True, check=True).stdout.strip()
    assert commit, 'the c1 report is not committed in fn-main'
    dirty = subprocess.run(['git', '-C', str(fn_main), 'status', '--porcelain', '--', REL], capture_output=True, text=True,
                           check=True).stdout.strip()
    assert not dirty, 'the c1 box record differs from its commit in fn-main'
    read = json.loads(lzma.open(place / READ).read())
    rows = {label: row['decode_only_tok_s'] for label, row in sorted(read['table'].items())
            if row.get('band') == 1 and row.get('c1') == 'off' and row.get('part') == 'core'}
    assert rows and all(isinstance(v, (int, float)) for v in rows.values()), rows
    return dict(schema=SCHEMA, supplied=True,
                label='cross-run read (another queue, another day): q18a-alpha-next-c1 C1-off band-1 decode-only tok/s; never a verdict input',
                source=dict(repo='fn-main', commit=commit, report=REL + '/' + REPORT, report_sha256=sha(place / REPORT),
                            read=REL + '/' + READ, read_sha256=sha(place / READ)),
                rule="every core arm of band 1 with c1 'off' in the read record's table, its decode_only_tok_s",
                c1_off_band1_decode_only_tok_s=rows, median_tok_s=statistics.median(rows.values()), arms=len(rows))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--fn-main', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--check', action='store_true')
    a = p.parse_args(argv)
    text = json.dumps(numbers(a.fn_main), indent=1, sort_keys=True) + '\n'
    if a.check:
        assert a.out.read_text() == text, 'cross-run numbers differ from the pinned record'
        print('cross-run numbers verified')
        return 0
    with a.out.open('x') as handle:
        handle.write(text)
    print(text, end='')
    return 0


if __name__ == '__main__':
    sys.exit(main())
