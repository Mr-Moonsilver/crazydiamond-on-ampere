#!/usr/bin/env python3
"""Run the unchanged stress workload/oracle with a metadata-only output sink.

The only AST substitutions are print -> quiet and the sole terminal json.dump
-> metadata_dump. Requests, random choices, retries, cancellation and leakgrade
calls are unchanged. No raw prompts, identifiers, responses or exception text
reach stdout or the output file. The harness manifest pins the source script.
"""
import ast
import json
from pathlib import Path
import sys


def metadata(d):
    run = d['run']
    if run['verdict'] not in ('PASS', 'FAIL', 'INCONCLUSIVE'):
        raise ValueError('unknown verdict')
    allowed = {'OK', 'MISS', 'NOISE', 'LEAK', 'COPYFAIL', 'INCONCLUSIVE'}
    if set(run['counts']) - allowed:
        raise ValueError('unknown count')
    notes = []
    for note in run['inconclusive']:
        # Only allow known oracle-generated messages, never exception/payload text.
        import re
        if not re.fullmatch(r'(partial-page hits on \d+ of \d+ answers|\d+ unsettled suspect answers|MISS \d+ > \d+% of answers|no request was cancelled)', note):
            raise ValueError('unexpected oracle note')
        notes.append(note)
    rows = []
    for r in d['results']:
        row = {k: r.get(k) for k in ('round', 'session', 'cached', 'cached3', 'p2_len',
                                    'answer_first', 'answer_settled', 'answer3_first', 'answer3_settled')}
        row.update(p1_len=len(r['p1_ids']), gen1_len=len(r['gen1_ids']), p3_len=len(r['p3']))
        rows.append(row)
    return dict(tool='legacy_partial_tail_metadata', source='leak_partial_tail.py',
                MODE=d['MODE'], B=d['B'], N=d['N'], ROUNDS=d['ROUNDS'], GEN=d['GEN'],
                identity_start=d['identity_start'], identity_end=d['identity_end'],
                run=dict(verdict=run['verdict'], counts=run['counts'], cancelled=run['cancelled'],
                         fails=['legacy oracle failure (detail redacted)'] * len(run['fails']), inconclusive=notes),
                error_count=len(d['errors']), results=rows,
                scope='Original stress workload/oracle; only persistence and console sinks replaced')


def transformed(source):
    tree = ast.parse(source)
    dumps = []
    class Sinks(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == 'print':
                node.func.id = '_quiet'
            if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == 'json' and node.func.attr == 'dump'):
                dumps.append(node)
                node.func = ast.Name(id='_metadata_dump', ctx=ast.Load())
            return node
    tree = Sinks().visit(tree)
    if len(dumps) != 1:
        raise ValueError('stress source output shape changed')
    return ast.fix_missing_locations(tree)


def main(argv):
    source = Path(__file__).with_name('leak_partial_tail.py')
    emitted = []
    def dump(d, f, *args, **kw):
        value = metadata(d)
        json.dump(value, f, *args, **kw)
        f.close()
        emitted.append(value)
    old = sys.argv
    sys.argv = [str(source), *argv[1:]]
    try:
        ns = dict(__name__='__main__', __file__=str(source),
                  _quiet=lambda *a, **k: None, _metadata_dump=dump)
        try:
            exec(compile(transformed(source.read_text()), str(source), 'exec'), ns)
        except SystemExit as e:
            rc = e.code
        else:
            rc = 1
        if len(emitted) != 1 or rc not in (0, 1, 2):
            raise ValueError('missing terminal stress result')
        r = emitted[0]['run']
        if rc != {'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 2}[r['verdict']]:
            raise ValueError('stress exit/result mismatch')
        print('legacy-stress', r['verdict'], 'counts', r['counts'], 'inconclusive', r['inconclusive'])
        return rc
    except Exception as e:
        print('legacy-stress FAIL', type(e).__name__)
        return 1
    finally:
        sys.argv = old


if __name__ == '__main__':
    sys.exit(main(sys.argv))
