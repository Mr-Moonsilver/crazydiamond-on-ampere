#!/usr/bin/env python3
"""Native43 l-f owner runner lineage: the box-run l-e runner with the l-f names, pins and two declared lines.

The l-f runner is built by build_native43.py from the frozen trunk held-b runner (names, H/BUNDLE, REVIEW/
EXPECTED_SOURCES, the self-run test count and the spec's declared swaps).  This check proves, over the written
runner file, that
  1. normalised to the l-e names and pins, it equals the l-e owner runner that ran on the box (q18a-native43-lifecycle
     run 1, bundle 296adc0886b6, runner 8644b753...) except the declared lines: the self-run test count and the
     l-f queue receipt beside the inherited ones;
  2. REVIEW and EXPECTED_SOURCES equal the build's own review and overlay rows, and the EXPECTED_SOURCES rows that
     differ from the l-e runner's are generation metadata plus the declared engine delta (--engine-delta ROW, each of
     which must move): l-f's e3/executor_hooks.py, e3/qsa_operation.py, e3/row1_oracle.py and e3_sources/executor.py.
So the staging/declare/arm code paths, the window import and the engine self-run gate are the bytes that ran on the
box for l-e (and, by its own lineage record, for the native39 lifecycle run before it).  No box action.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

BOX_RUNNER = Path('<hub-dir>/agents/flash-next/tmp/native43-lifecycle-e/q18a-native43-lifecycle-owner-run-296adc0886b6.py')
BOX_RUNNER_SHA = '8644b7539307a74e149cfa1d6dde379ee60174282f331d059a15895babccc508'
BOX_LINEAGE = Path('<hub-dir>/agents/flash-next/tmp/native43-lifecycle-e/runner-lineage.json')
METADATA = {'S5-OVERLAP-GENERATION.json', 'e3/native-implementation-closure.json', 'e3/native-overlap-readiness.json',
            'e3/decode-probe-evidence.json', 'core/SHA256SUMS'}
PINS = ('H', 'BUNDLE', 'REVIEW', 'EXPECTED_SOURCES')
NAMES = (('q18a_native43_lifecycle_f', 'q18a_native43_lifecycle'), ('q18a-native43-lifecycle-f', 'q18a-native43-lifecycle'))
# (l-f runner line, l-e runner line): the declared differences after normalisation.
DECLARED = (
    ("result.get('tests_run',0)==258", "result.get('tests_run',0)==247"),
    ("result.get('native39_trunk_queue')=='PASS' and result.get('native43_queue')=='PASS' and result.get('native43_lifecycle_queue')=='PASS' and result.get('native43_lf_queue')=='PASS'",
     "result.get('native39_trunk_queue')=='PASS' and result.get('native43_queue')=='PASS' and result.get('native43_lifecycle_queue')=='PASS'"),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assignments(text):
    """PIN name -> (source text, value); the value is the literal, or the source text for a call (H=Path(...))."""
    def value(node):
        try:
            return ast.literal_eval(node)
        except ValueError:
            return ast.get_source_segment(text, node)
    return {n.targets[0].id: (ast.get_source_segment(text, n.value), value(n.value))
            for n in ast.parse(text).body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id in PINS}


def check(runner, build, engine_delta=()):
    assert sha(BOX_RUNNER) == BOX_RUNNER_SHA
    assert json.loads(BOX_LINEAGE.read_text())['verdict'] == 'PASS'
    text, box = Path(runner).read_text(), BOX_RUNNER.read_text()
    built = json.loads(Path(build).read_text())
    mine, theirs = assignments(text), assignments(box)
    # 2. pins: the build's own review and overlay rows; rows differing from l-e are metadata or the declared delta.
    assert mine['REVIEW'][1] == built['review_sha256'] and mine['BUNDLE'][1] == built['manifest']
    overlay = Path(built['overlay'])
    rows = {line.split(None, 1)[1]: line.split(None, 1)[0] for line in (overlay / 'SHA256SUMS').read_text().splitlines()}
    assert mine['EXPECTED_SOURCES'][1] == {k: rows[k] for k in mine['EXPECTED_SOURCES'][1]}
    a, b = mine['EXPECTED_SOURCES'][1], theirs['EXPECTED_SOURCES'][1]
    assert set(a) == set(b)
    moved = sorted(k for k in a if a[k] != b[k])
    delta = set(engine_delta)
    assert not delta & METADATA and set(moved) <= METADATA | delta, moved
    assert delta <= set(moved), ('declared engine delta did not move', sorted(delta - set(moved)))
    # 1. normalised to the l-e names and pins, the box-run l-e runner but for the declared lines.
    norm = text
    for new, old in NAMES:
        norm = norm.replace(new, old)
    for name in PINS:
        assert norm.count(mine[name][0]) == 1, name
        norm = norm.replace(mine[name][0], theirs[name][0])
    for new, old in DECLARED:
        assert norm.count(new) == 1, new[:60]
        norm = norm.replace(new, old)
    assert norm == box, 'l-f runner differs from the box-run l-e runner beyond the declared lines'
    return dict(schema='native43-lf-runner-lineage/1', verdict='PASS', runner=str(runner), runner_sha256=sha(runner),
                box_runner=str(BOX_RUNNER), box_runner_sha256=BOX_RUNNER_SHA, box_runner_lineage=str(BOX_LINEAGE),
                box_runner_lineage_sha256=sha(BOX_LINEAGE), declared_differences=[list(r) for r in DECLARED],
                expected_sources_rows=len(a), metadata_rows_moved_vs_le=sorted(set(moved) - delta),
                declared_engine_delta_vs_le={k: a[k] for k in sorted(delta)},
                engine_rows_equal_le=sorted(k for k in a if k not in set(moved)),
                review_sha256=mine['REVIEW'][1], le_review_sha256=theirs['REVIEW'][1], box_action=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--runner', required=True)
    p.add_argument('--build', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--engine-delta', action='append', default=[],
                   help='EXPECTED_SOURCES row declared to differ from the l-e runner (engine delta)')
    a = p.parse_args()
    record = check(a.runner, a.build, a.engine_delta)
    with a.out.open('x') as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps(dict(verdict=record['verdict'], moved=record['metadata_rows_moved_vs_le'],
                          delta=sorted(record['declared_engine_delta_vs_le']))))
