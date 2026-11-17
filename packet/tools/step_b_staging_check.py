#!/usr/bin/env python3
"""Hub-side staging check: the frozen runner's own staging require blocks, evaluated against the exact records to be delivered.

The runner refuses staging on the box unless two require() calls in main()'s `if a.staging:` branch hold: the one with reason
'exact-frozen-selfrun-required' over the bundle self-run record (`result`) and the one with reason 'exact-engine-selfrun-required'
over the engine self-run record (`engine`). This tool takes those two condition expressions out of the frozen runner's text by AST,
unchanged, and evaluates them here with the runner's own module constants (BUNDLE, IMAGE, ...) and a sha() with the runner's rule, so a
case list, a count or a field the box would refuse is refused on the hub first. Each conjunct is evaluated on its own too, so a refusal
names the failing term.

Negative controls: the same expressions over a copy of each record with one case removed and with tests_run moved by one must refuse.
No GPU, no box action, nothing is written but --out.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

REASONS = {'exact-frozen-selfrun-required': 'result', 'exact-engine-selfrun-required': 'engine'}


def staging_requires(tree):
    """The require(condition, reason) calls of main()'s `if a.staging:` branch, by reason."""
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    branch = next(n for n in ast.walk(main) if isinstance(n, ast.If) and ast.unparse(n.test) == 'a.staging')
    found = {}
    for node in ast.walk(ast.Module(body=branch.body, type_ignores=[])):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'require' and len(node.args) == 2
                and isinstance(node.args[1], ast.Constant) and node.args[1].value in REASONS):
            assert node.args[1].value not in found, 'two require blocks with one reason: ' + node.args[1].value
            found[node.args[1].value] = node.args[0]
    assert set(found) == set(REASONS), sorted(set(REASONS) ^ set(found))
    return found


def constants(tree):
    """The runner's module-level literal assignments (BUNDLE, IMAGE, QNAME, ...)."""
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def runner_sha(path):
    """The runner's sha(): a regular, non-symlink file's sha256, else refused."""
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError('regular-public-file-required')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(expr, runner, names, record_name, record):
    scope = dict(names, sha=runner_sha, Path=lambda p: Path(runner) if str(p) == str(runner) else Path(p),
                 __file__=str(runner), **{record_name: record})
    whole = bool(eval(compile(ast.Expression(expr), str(runner), 'eval'), {'__builtins__': __builtins__}, scope))
    terms = expr.values if isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.And) else [expr]
    failing = []
    for term in terms:
        try:
            ok = bool(eval(compile(ast.Expression(term), str(runner), 'eval'), {'__builtins__': __builtins__}, scope))
        except Exception as exc:
            ok, term_error = False, '%s: %s' % (type(exc).__name__, exc)
        else:
            term_error = None
        if not ok:
            failing.append(dict(term=ast.unparse(term)[:300], error=term_error))
    return dict(holds=whole, terms=len(terms), failing=failing)


def check(runner, selfrun, engine):
    runner = Path(runner).resolve()
    tree = ast.parse(runner.read_text())
    blocks, names = staging_requires(tree), constants(tree)
    records = {'result': json.loads(Path(selfrun).read_text()), 'engine': json.loads(Path(engine).read_text())}
    rows = {reason: evaluate(blocks[reason], runner, names, name, records[name]) for reason, name in REASONS.items()}
    controls = {}
    engine_short = copy.deepcopy(records['engine'])
    engine_short['cases'].pop(sorted(engine_short['cases'])[-1])
    controls['engine-record-one-case-short'] = not evaluate(blocks['exact-engine-selfrun-required'], runner, names, 'engine',
                                                            engine_short)['holds']
    selfrun_moved = dict(records['result'], tests_run=(records['result'].get('tests_run') or 0) + 1)
    controls['selfrun-tests-run-moved'] = not evaluate(blocks['exact-frozen-selfrun-required'], runner, names, 'result',
                                                       selfrun_moved)['holds']
    ok = all(r['holds'] for r in rows.values()) and all(controls.values())
    return dict(schema='step-b-staging-check/1', verdict='PASS' if ok else 'FAIL', classification=None if ok else 'HARNESS',
                runner=str(runner), runner_sha256=runner_sha(runner), selfrun=str(selfrun), selfrun_sha256=runner_sha(selfrun),
                engine_selfrun=str(engine), engine_selfrun_sha256=runner_sha(engine), requires=rows,
                negative_controls_refused=controls, gpu_execution=False, box_action=False,
                meaning='the frozen runner\'s staging require blocks, taken from its text unchanged, hold for these exact records')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--runner', type=Path, required=True)
    p.add_argument('--selfrun', type=Path, required=True)
    p.add_argument('--engine', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = check(a.runner, a.selfrun, a.engine)
    with a.out.open('x') as handle:
        handle.write(json.dumps(result, sort_keys=True, indent=1) + '\n')
    print(json.dumps(dict(verdict=result['verdict'], failing={k: v['failing'] for k, v in result['requires'].items()},
                          controls=result['negative_controls_refused']), sort_keys=True))
    raise SystemExit(0 if result['verdict'] == 'PASS' else 1)
