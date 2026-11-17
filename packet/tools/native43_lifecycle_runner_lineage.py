#!/usr/bin/env python3
"""Native43 lifecycle owner runner lineage: the box-run single-queue runner shape with the native43 engine pins.

The native43 lifecycle runner is built by build_native43.py from the frozen trunk held-b runner (names, H/BUNDLE,
REVIEW/EXPECTED_SOURCES, the self-run test count and five declared swaps).  This check proves, over the written
runner file, that
  1. normalised to the native39 lifecycle names and pins, it equals the native39 lifecycle owner runner that ran on
     the box (held-a run 1, 74f2ee4a...) except the declared lines: the window import, the self-run arm/test count/
     queue receipts and the native43 engine self-run gate (schema, clock split, scheduler tick probe);
  2. those engine self-run gate lines are byte-identical to the frozen n43-f runner's (9130d18d...);
  3. its EXPECTED_SOURCES engine rows equal the n43-f runner's, and the rows that differ are generation metadata
     only (the prepared review moves them), while REVIEW and EXPECTED_SOURCES equal the build's own review and
     overlay rows.  A declared engine delta (--engine-delta ROW, e.g. l-e's e3/engine_driver.py = n43-f + the
     cfffffc TP finalize-skew hunks) may also differ and must differ; its bytes are bound by the delivery identity.
So the staging/declare/arm code paths are the single-queue runner's that ran on the box twice (native39 trunk run 1,
native39 lifecycle run 1); the native43 owner seam (pair wrapper, one key across two arms) does not apply.
No box action.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

BOX_RUNNER = Path('<hub-dir>/agents/flash-next/tmp/native39-lifecycle-held-a/q18a-native39-lifecycle-owner-run-b90f1e292956.py')
BOX_RUNNER_SHA = '74f2ee4a9c54c68ef97b975a5b42f73d379e4e10cf77651803d85c2c304ae2f0'
ENGINE_RUNNER = Path('<hub-dir>/agents/flash-next/tmp/native43-held-f/q18a-native43-dense-boundary-owner-run-4f5f49c98e82.py')
ENGINE_RUNNER_SHA = '9130d18d12530a0986ca55acda0115f2783e5d8cb7efab43bd652e01a9afc141'
METADATA = {'S5-OVERLAP-GENERATION.json', 'e3/native-implementation-closure.json', 'e3/native-overlap-readiness.json',
            'e3/decode-probe-evidence.json', 'core/SHA256SUMS'}
PINS = ('H', 'BUNDLE', 'REVIEW', 'EXPECTED_SOURCES')
# (native43 lifecycle runner line, native39 lifecycle runner line): the declared differences after normalisation.
DECLARED = (
    ('    from native43_lifecycle_gate import lifecycle_window_ok as window_ok\n',
     '    from native39_lifecycle_gate import lifecycle_window_ok as window_ok\n'),
    ("result.get('tests_run',0)==247", "result.get('tests_run',0)==235"),
    ("result.get('native35_arm')=='NATIVE43L'", "result.get('native35_arm')=='LIFECYCLE'"),
    ("result.get('native39_trunk_queue')=='PASS' and result.get('native43_queue')=='PASS' and result.get('native43_lifecycle_queue')=='PASS'",
     "result.get('native39_trunk_queue')=='PASS' and result.get('native39_lifecycle_queue')=='PASS'"),
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


def gate_block(text):
    """The engine self-run requirement: from the schema line through 'exact-engine-selfrun-required')."""
    start = text.index("        require(engine.get('schema')==")
    end = text.index("'exact-engine-selfrun-required')", start) + len("'exact-engine-selfrun-required')")
    return text[start:end]


def check(runner, build, engine_delta=()):
    assert sha(BOX_RUNNER) == BOX_RUNNER_SHA and sha(ENGINE_RUNNER) == ENGINE_RUNNER_SHA
    text, box, engine = Path(runner).read_text(), BOX_RUNNER.read_text(), ENGINE_RUNNER.read_text()
    built = json.loads(Path(build).read_text())
    mine, theirs, n43f = assignments(text), assignments(box), assignments(engine)
    # 3. engine pins: the build's own review and overlay rows; engine rows as n43-f; differing rows are metadata only.
    assert mine['REVIEW'][1] == built['review_sha256'] and mine['BUNDLE'][1] == built['manifest']
    overlay = Path(built['overlay'])
    rows = {line.split(None, 1)[1]: line.split(None, 1)[0] for line in (overlay / 'SHA256SUMS').read_text().splitlines()}
    assert mine['EXPECTED_SOURCES'][1] == {k: rows[k] for k in mine['EXPECTED_SOURCES'][1]}
    a, b = mine['EXPECTED_SOURCES'][1], n43f['EXPECTED_SOURCES'][1]
    assert set(a) == set(b)
    moved = sorted(k for k in a if a[k] != b[k])
    delta = set(engine_delta)
    assert not delta & METADATA and set(moved) <= METADATA | delta, moved
    assert delta <= set(moved), ('declared engine delta did not move', sorted(delta - set(moved)))
    engine_rows = sorted(k for k in a if k.startswith('e3/') and k.endswith('.py') and k not in METADATA | delta)
    # 2. the engine self-run gate lines are the n43-f runner's, byte for byte.
    assert gate_block(text) == gate_block(engine)
    # 1. normalised to the native39 lifecycle names and pins, the box-run runner but for the declared lines.
    norm = text.replace('q18a_native43_lifecycle', 'q18a_native39_lifecycle').replace('q18a-native43-lifecycle',
                                                                                     'q18a-native39-lifecycle')
    for name in PINS:
        assert norm.count(mine[name][0]) == 1, name
        norm = norm.replace(mine[name][0], theirs[name][0])
    norm = norm.replace(gate_block(engine), gate_block(box))
    for new, old in DECLARED:
        assert norm.count(new) == 1, new[:60]
        norm = norm.replace(new, old)
    assert norm == box, 'lifecycle runner differs from the box-run native39 lifecycle runner beyond the declared lines'
    return dict(schema='native43-lifecycle-runner-lineage/1', verdict='PASS', runner=str(runner), runner_sha256=sha(runner),
                box_runner=str(BOX_RUNNER), box_runner_sha256=BOX_RUNNER_SHA, engine_runner=str(ENGINE_RUNNER),
                engine_runner_sha256=ENGINE_RUNNER_SHA, declared_differences=[list(r) for r in DECLARED],
                engine_gate_from_n43f=True, expected_sources_rows=len(a),
                metadata_rows_moved_vs_n43f=sorted(set(moved) - delta),
                declared_engine_delta_vs_n43f={k: a[k] for k in sorted(delta)},
                engine_rows_equal_n43f=sorted(set(engine_rows)), review_sha256=mine['REVIEW'][1],
                n43f_review_sha256=n43f['REVIEW'][1], owner_seam='not applicable: single-queue runner; staging, '
                'declare and arm bytes are the box-run native39 lifecycle runner\'s', box_action=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--runner', required=True)
    p.add_argument('--build', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--engine-delta', action='append', default=[],
                   help='EXPECTED_SOURCES row declared to differ from the n43-f runner (engine delta)')
    a = p.parse_args()
    record = check(a.runner, a.build, a.engine_delta)
    with a.out.open('x') as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps(dict(verdict=record['verdict'], moved=record['metadata_rows_moved_vs_n43f'])))
