#!/usr/bin/env python3
"""Hub-side staging check: the frozen owner runner's own staging `require` blocks over the exact records a delivery names.

A box staging refused a package whose runner pinned an engine self-run case list that no self-run had been held against (the
lead's note, 2 Nov 2026: cancel build-c, `exact-engine-selfrun-required`). This evaluates the runner's text, not a copy of it:
  1. parse the frozen runner; inside main's `if a.staging:` block take every `require(<expr>, '<reason>')` whose expression reads
     a record (`result` or `engine`, the runner's names for the two records it loads);
  2. bind `result` and `engine` to the exact selfrun and engine-selfrun records (the runner's own json.loads), the runner's
     module-level literal constants (BUNDLE, IMAGE, ...), its `sha` over sha256, `Path`, and `__file__` = the runner path, so
     `sha(Path(__file__))` is the frozen runner's sha as on the box;
  3. evaluate each expression, and each clause of a top-level `and`, reporting the clause source of every false clause and,
     for a fixed case list, the missing and extra case names.
A name the expression reads that is not bound here is a HARNESS failure (the runner's shape moved; this check must follow it),
never a silent pass. The expected reasons must all be found. Reads files, writes one receipt (open 'x'). No box, no GPU.

usage: runner_staging_check.py --runner RUNNER --selfrun SELFRUN_RESULT --engine ENGINE_SELFRUN_RESULT --out RECEIPT
"""
import argparse
import ast
import builtins
import hashlib
import json
from pathlib import Path
import sys

SCHEMA = 'runner-staging-check/1'
RECORD_NAMES = ('result', 'engine')
EXPECTED_REASONS = ('exact-frozen-selfrun-required', 'exact-engine-selfrun-required')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _is_require(node):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'require'
            and len(node.args) == 2 and isinstance(node.args[1], ast.Constant) and type(node.args[1].value) is str)


def _staging_block(tree):
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and fn.name == 'main':
            for node in ast.walk(fn):
                if isinstance(node, ast.If) and ast.unparse(node.test) == 'a.staging':
                    return node
    raise ValueError('runner has no main with an `if a.staging:` block')


def _names(expr):
    return {n.id for n in ast.walk(expr) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _free_names(expr):
    """Names read and not bound inside the expression (comprehension targets, lambda arguments, walrus)."""
    bound = {n.id for n in ast.walk(expr) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    bound |= {arg.arg for n in ast.walk(expr) if isinstance(n, ast.Lambda) for arg in n.args.args}
    return _names(expr) - bound


def staging_requires(text):
    """[(reason, expression node)] for the staging requires that read a record."""
    tree = ast.parse(text)
    out = []
    for stmt in _staging_block(tree).body:
        for node in ast.walk(stmt):
            if _is_require(node) and _names(node.args[0]) & set(RECORD_NAMES):
                out.append((node.args[1].value, node.args[0]))
    return out


def constants(text):
    """The runner's module-level NAME = <literal> assignments."""
    values = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return values


def _case_lists(expr):
    """Fixed sorted(<rec>.get('cases') ...) == [literal names] comparisons inside the expression."""
    rows = []
    for node in ast.walk(expr):
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
            left, right = node.left, node.comparators[0]
            if 'cases' in ast.unparse(left) and isinstance(right, ast.List):
                try:
                    rows.append((left, ast.literal_eval(right)))
                except ValueError:
                    continue
    return rows


def evaluate(text, runner_path, records):
    space = dict(constants(text), sha=sha, Path=Path, __file__=str(runner_path), **records)
    blocks, verdict = [], 'PASS'
    found = []
    for reason, expr in staging_requires(text):
        found.append(reason)
        unbound = sorted(_free_names(expr) - set(space) - set(dir(builtins)))
        block = dict(reason=reason, unbound=unbound, failed_clauses=[], case_lists=[])
        if unbound:
            block.update(verdict='FAIL', classification='HARNESS', error='names the check does not bind: %s' % unbound)
            verdict = 'FAIL'
            blocks.append(block)
            continue
        clauses = expr.values if isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.And) else [expr]
        for clause in clauses:
            try:
                ok = bool(eval(compile(ast.Expression(clause), '<runner>', 'eval'), {'__builtins__': builtins}, space))
            except Exception as exc:  # noqa: BLE001 - a clause the record cannot satisfy is a failed clause
                ok = False
                block.setdefault('clause_errors', []).append('%s: %s' % (type(exc).__name__, exc))
            if not ok:
                block['failed_clauses'].append(ast.unparse(clause)[:400])
        for left, expected in _case_lists(expr):
            got = eval(compile(ast.Expression(left), '<runner>', 'eval'), {'__builtins__': builtins}, space)
            got = list(got) if isinstance(got, (list, tuple)) else got
            block['case_lists'].append(dict(expected=len(expected), got=len(got) if isinstance(got, list) else None,
                                            missing=sorted(set(expected) - set(got or [])),
                                            extra=sorted(set(got or []) - set(expected))))
        whole = bool(eval(compile(ast.Expression(expr), '<runner>', 'eval'), {'__builtins__': builtins}, space)) \
            if not block['failed_clauses'] else False
        block['verdict'] = 'PASS' if whole else 'FAIL'
        if not whole:
            block['classification'] = 'HARNESS'
            verdict = 'FAIL'
        blocks.append(block)
    missing = [r for r in EXPECTED_REASONS if r not in found]
    if missing:
        verdict = 'FAIL'
    return dict(verdict=verdict, blocks=blocks, reasons_found=found, reasons_missing=missing)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--runner', type=Path, required=True)
    p.add_argument('--selfrun', type=Path, required=True)
    p.add_argument('--engine', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    text = a.runner.read_text()
    records = dict(result=json.loads(a.selfrun.read_text()), engine=json.loads(a.engine.read_text()))
    try:
        body = evaluate(text, a.runner.resolve(), records)
    except Exception as exc:  # noqa: BLE001 - the check itself failed: HARNESS, never a pass
        body = dict(verdict='FAIL', classification='HARNESS', error='%s: %s' % (type(exc).__name__, exc))
    receipt = dict(schema=SCHEMA, runner=str(a.runner), runner_sha256=sha(a.runner), selfrun=str(a.selfrun),
                   selfrun_sha256=sha(a.selfrun), engine_selfrun=str(a.engine), engine_selfrun_sha256=sha(a.engine),
                   gpu_work=False, box_action=False, **body)
    receipt['rc'] = 0 if receipt['verdict'] == 'PASS' else 1
    with a.out.open('x') as handle:
        json.dump(receipt, handle, sort_keys=True, indent=2)
        handle.write('\n')
    print(json.dumps({k: receipt[k] for k in ('verdict', 'rc', 'reasons_found', 'reasons_missing') if k in receipt}))
    for block in receipt.get('blocks', []):
        print(block['reason'], block['verdict'], block['failed_clauses'][:6], block['case_lists'])
    return receipt['rc']


if __name__ == '__main__':
    sys.exit(main())
